#!/usr/bin/env python
"""Evaluate reverse Attention Heads Edit attention scaling on API4 prefix-to-PII data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
try:
    from transformers import AutoConfig, AutoModelForImageTextToText
except ImportError:
    AutoConfig = None
    AutoModelForImageTextToText = None

from attention_heads_edit_lib.attention_heads_edit import AttentionHeadsEdit
from attention_heads_privacy_spans import STEERING_SPAN_MODES, select_privacy_steering_span


REPO_ROOT = Path(__file__).resolve().parents[3]

SPACED_SECRET_LABELS = {
    "PHONENUMBER",
    "PHONEIMEI",
    "MASKEDNUMBER",
    "ZIPCODE",
    "PASSWORD",
    "PIN",
    "IPV6",
    "MAC",
    "ACCOUNTNUMBER",
    "IBAN",
    "CREDITCARDNUMBER",
    "CREDITCARDCVV",
    "ETHEREUMADDRESS",
    "BITCOINADDRESS",
    "LITECOINADDRESS",
    "SSN",
    "VEHICLEVRM",
    "VEHICLEVIN",
}


@dataclass(frozen=True)
class Api4Example:
    input_text: str
    target: str
    pii_type: str
    source_id: int | str | None
    entity_index: int | str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reverse Attention Heads Edit privacy evaluation on API4 data.")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data/api4_200k/sft_true_prefix_no_instruction_all.json")
    parser.add_argument(
        "--head_config",
        type=Path,
        default=REPO_ROOT
        / "configs/attention_heads_edit/runs/api4_attention_heads_edit_top100_a015_fixed_topN_top30.json",
    )
    parser.add_argument("--base_model", type=Path, default=REPO_ROOT / "models/llama3-8B/baseline")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=REPO_ROOT
        / "models/llama3-8B/api4_prefix_plain_qlora",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs/attention_heads_edit")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all examples.")
    parser.add_argument("--limit_per_type", type=int, default=0, help="0 means no per-type limit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--scale_position", default="include_down", choices=["include_down", "include", "exclude", "generation"])
    parser.add_argument("--steering_span_mode", default="tail_tokens", choices=STEERING_SPAN_MODES)
    parser.add_argument("--steering_tail_tokens", type=int, default=8)
    parser.add_argument("--disable_attention_heads_edit", action="store_true", help="Run the same generation metrics without attention steering.")
    parser.add_argument("--ffn_edit_kn_config", type=Path, default=None, help="Optional FFN Edit KN json to remove during evaluation.")
    parser.add_argument("--ffn_edit_erase_num", type=int, default=0, help="Number of FFN Edit neurons to remove; 0 means all entries in --ffn_edit_kn_config.")
    parser.add_argument("--max_context_tokens", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--generation_extra_tokens", type=int, default=8)
    parser.add_argument("--prompt_template", default="{input}")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--save_text_predictions", action="store_true")
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def dtype_from_arg(name: str):
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())


def metric_normalize(text: str, pii_type: str) -> str:
    text = normalize_text(text)
    if pii_type in SPACED_SECRET_LABELS:
        return compact_text(text)
    return text.casefold()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_examples(args: argparse.Namespace) -> list[Api4Example]:
    with args.dataset.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    examples = [
        Api4Example(
            input_text=str(row.get("input", "")),
            target=str(row.get("output", "")),
            pii_type=str(row.get("pii_type", "UNKNOWN")),
            source_id=row.get("source_id"),
            entity_index=row.get("entity_index"),
        )
        for row in rows
        if row.get("input") and row.get("output")
    ]
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    if args.limit_per_type > 0:
        kept: list[Api4Example] = []
        counts: Counter[str] = Counter()
        for example in examples:
            if counts[example.pii_type] >= args.limit_per_type:
                continue
            kept.append(example)
            counts[example.pii_type] += 1
        examples = kept
    if args.max_samples > 0:
        examples = examples[: args.max_samples]
    return examples


def convert_head_config(raw: Any) -> dict[int, list[int]]:
    if isinstance(raw, dict) and not raw:
        return {}
    if isinstance(raw, dict) and "edit_heads" in raw:
        heads = raw["edit_heads"]
        converted: dict[int, list[int]] = defaultdict(list)
        for item in heads:
            converted[int(item["layer"])].append(int(item["head"]))
        return {layer: sorted(set(head_ids)) for layer, head_ids in converted.items()}
    if isinstance(raw, dict):
        converted = defaultdict(list)
        for layer, heads in raw.items():
            if isinstance(heads, list):
                converted[int(layer)].extend(int(head) for head in heads)
        if converted:
            return {layer: sorted(set(head_ids)) for layer, head_ids in converted.items()}
    raise ValueError(f"Unsupported head config format: {type(raw).__name__}")


def load_head_config(path: Path) -> dict[int, list[int]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return convert_head_config(raw)


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "dtype": dtype_from_arg(args.torch_dtype),
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_implementation,
        "local_files_only": args.local_files_only,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model_cls = AutoModelForCausalLM
    if AutoConfig is not None and AutoModelForImageTextToText is not None:
        config = AutoConfig.from_pretrained(args.base_model, local_files_only=args.local_files_only)
        if getattr(config, "model_type", None) == "mistral3":
            model_cls = AutoModelForImageTextToText
    model = model_cls.from_pretrained(args.base_model, **model_kwargs)
    tokenizer_vocab_size = len(tokenizer)
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None and input_embeddings.weight.shape[0] != tokenizer_vocab_size:
        print(
            f"[INFO] resizing token embeddings "
            f"{input_embeddings.weight.shape[0]} -> {tokenizer_vocab_size} before PEFT adapter load",
            flush=True,
        )
        model.resize_token_embeddings(tokenizer_vocab_size)
    model = PeftModel.from_pretrained(model, args.adapter, local_files_only=args.local_files_only)
    model.eval()
    model.config.use_cache = True
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


def load_ffn_edit_positions(path: Path, erase_num: int = 0) -> list[list[int]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    positions: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for item in raw:
        candidates = item if isinstance(item, list) and item and isinstance(item[0], list) else [item]
        for candidate in candidates:
            if not isinstance(candidate, list) or len(candidate) < 2:
                continue
            key = (int(candidate[0]), int(candidate[1]))
            if key in seen:
                continue
            seen.add(key)
            positions.append([key[0], key[1]])
            if erase_num > 0 and len(positions) >= erase_num:
                return positions
    return positions


def apply_ffn_edit_if_requested(model, args: argparse.Namespace) -> list[list[int]]:
    path = getattr(args, "ffn_edit_kn_config", None)
    if path is None:
        return []
    positions = load_ffn_edit_positions(Path(path), int(getattr(args, "ffn_edit_erase_num", 0) or 0))
    if not positions:
        raise ValueError(f"No FFN Edit positions loaded from {path}")
    repo_ffn_edit_utils = REPO_ROOT / "srcs/ffn_edit/utils"
    if str(repo_ffn_edit_utils) not in sys.path:
        sys.path.insert(0, str(repo_ffn_edit_utils))
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    base_config = getattr(base_model, "config", None)
    model_type = str(getattr(base_config, "model_type", "")).lower()
    text_model_type = str(getattr(getattr(base_config, "text_config", None), "model_type", "")).lower()
    def allow_ffn_edit_generate_kwargs(target_model):
        validator = getattr(target_model, "_validate_model_kwargs", None)
        if not callable(validator) or getattr(target_model, "_ffn_edit_generate_kwargs_allowed", False):
            return

        def ffn_edit_validator(model_kwargs):
            if isinstance(model_kwargs, dict):
                model_kwargs = dict(model_kwargs)
                model_kwargs.pop("imp_pos", None)
                model_kwargs.pop("imp_op", None)
            return validator(model_kwargs)

        target_model._validate_model_kwargs = ffn_edit_validator
        target_model._ffn_edit_generate_kwargs_allowed = True

    ffn_edit_patch_targets = []
    current_target = model
    seen_target_ids = set()
    for _ in range(3):
        if current_target is None or id(current_target) in seen_target_ids:
            break
        ffn_edit_patch_targets.append(current_target)
        seen_target_ids.add(id(current_target))
        current_target = getattr(current_target, "base_model", None) or getattr(current_target, "model", None)

    if "gemma" in model_type:
        from custom_gemma import install_static_gemma_ffn_edit_hooks

        install_static_gemma_ffn_edit_hooks(model, positions, "remove")
        setattr(args, "ffn_edit_backend", "gemma_down_proj_pre_hook")
    elif "qwen" in model_type:
        from custom_qwen import install_static_qwen_ffn_edit_hooks

        install_static_qwen_ffn_edit_hooks(model, positions, "remove")
        setattr(args, "ffn_edit_backend", "qwen_down_proj_pre_hook")
    elif "ministral" in text_model_type or "mistral3" in model_type:
        from custom_ministral import install_static_ministral_ffn_edit_hooks

        install_static_ministral_ffn_edit_hooks(model, positions, "remove")
        setattr(args, "ffn_edit_backend", "ministral_down_proj_pre_hook")
    else:
        from custom_llama import patch_llama_model

        patch_llama_model(base_model)
        setattr(args, "ffn_edit_backend", "llama_forward_patch")
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(base_model, "config"):
        base_model.config.use_cache = False
    setattr(args, "ffn_edit_imp_pos", positions)
    setattr(args, "ffn_edit_imp_op", "remove")
    return positions


def ffn_edit_forward_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    positions = getattr(args, "ffn_edit_imp_pos", None)
    if not positions:
        return {}
    if getattr(args, "ffn_edit_backend", None) in {
        "gemma_down_proj_pre_hook",
        "qwen_down_proj_pre_hook",
        "ministral_down_proj_pre_hook",
    }:
        return {}
    return {"imp_pos": positions, "imp_op": getattr(args, "ffn_edit_imp_op", "remove")}


def ffn_edit_runtime_stats(model) -> dict[str, int]:
    stats: dict[str, int] = {}
    for attr_name in (
        "_gemma_static_ffn_edit_controller",
        "_qwen_static_ffn_edit_controller",
        "_ministral_static_ffn_edit_controller",
    ):
        controller = getattr(model, attr_name, None)
        if controller is None:
            continue
        stats.update(
            {
                "ffn_edit_hook_call_count": int(getattr(controller, "hook_call_count", 0)),
                "ffn_edit_edit_call_count": int(getattr(controller, "edit_call_count", 0)),
                "ffn_edit_edited_neuron_count": int(getattr(controller, "edited_neuron_count", 0)),
            }
        )
    return stats


def unwrap_model_for_attention_heads_edit(model):
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        return base_model.model
    return model


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def trim_input_for_context(tokenizer, input_text: str, args: argparse.Namespace) -> str:
    prompt_text = args.prompt_template.format(input=input_text)
    token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= args.max_context_tokens:
        return input_text
    input_ids = tokenizer(input_text, add_special_tokens=False)["input_ids"]
    keep = max(1, min(len(input_ids), args.max_context_tokens))
    return tokenizer.decode(input_ids[-keep:], skip_special_tokens=True)


def build_prompt(input_text: str, args: argparse.Namespace) -> str:
    return args.prompt_template.format(input=input_text)


def decode_generated(tokenizer, sequence: torch.Tensor, input_width: int) -> str:
    generated_ids = sequence[input_width:].tolist()
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and eos_token_id in generated_ids:
        generated_ids = generated_ids[: generated_ids.index(eos_token_id)]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    lines = text.splitlines()
    return lines[0].strip() if lines else text.strip()


def generation_metrics(prediction: str, target: str, pii_type: str) -> dict[str, float]:
    prediction = prediction.strip()
    target = target.strip()
    pred_norm = metric_normalize(prediction, pii_type)
    target_norm = metric_normalize(target, pii_type)
    return {
        "exact_match": 1.0 if prediction == target else 0.0,
        "normalized_exact_match": 1.0 if pred_norm == target_norm else 0.0,
        "starts_with_target": 1.0 if prediction.startswith(target) else 0.0,
        "normalized_starts_with_target": 1.0 if pred_norm.startswith(target_norm) else 0.0,
        "contains_target": 1.0 if target and target in prediction else 0.0,
        "normalized_contains_target": 1.0 if target_norm and target_norm in pred_norm else 0.0,
        "prediction_char_len": float(len(prediction)),
    }


@torch.no_grad()
def evaluate_batch(
    model,
    attention_heads_edit_model,
    tokenizer,
    attention_heads_edit_steerer: AttentionHeadsEdit | None,
    batch: list[Api4Example],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    trimmed_inputs = [trim_input_for_context(tokenizer, example.input_text, args) for example in batch]
    prompts = [build_prompt(input_text, args) for input_text in trimmed_inputs]
    steering_spans = [
        select_privacy_steering_span(
            input_text,
            tokenizer,
            mode=args.steering_span_mode,
            tail_tokens=args.steering_tail_tokens,
        )
        for input_text in trimmed_inputs
    ]
    device = model_device(model)
    if attention_heads_edit_steerer is None:
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        model_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)
        tokenizer.padding_side = original_padding_side
        offsets_mapping = None
    else:
        model_inputs, offsets_mapping = attention_heads_edit_steerer.inputs_from_batch(prompts, tokenizer=tokenizer, device=device)
    target_token_lens = [
        len(tokenizer(example.target, add_special_tokens=False)["input_ids"])
        for example in batch
    ]
    max_new_tokens = max(args.max_new_tokens, max(target_token_lens, default=1) + max(args.generation_extra_tokens, 0))
    steering_context = (
        attention_heads_edit_steerer.apply_steering(
            model=attention_heads_edit_model,
            strings=prompts,
            substrings=steering_spans,
            model_input=model_inputs,
            offsets_mapping=offsets_mapping,
            occurrence=-1,
        )
        if attention_heads_edit_steerer is not None
        else torch.inference_mode()
    )
    with steering_context:
        generation_kwargs: dict[str, Any] = {}
        generation_kwargs.update(ffn_edit_forward_kwargs(args))
        if generation_kwargs or attention_heads_edit_steerer is not None:
            generation_kwargs["use_cache"] = False
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )

    rows: list[dict[str, Any]] = []
    input_width = model_inputs["input_ids"].size(1)
    for example, prediction, prompt, steering_span in zip(
        batch,
        [decode_generated(tokenizer, sequence, input_width) for sequence in outputs],
        prompts,
        steering_spans,
    ):
        row = {
            "pii_type": example.pii_type,
            "source_id": example.source_id,
            "entity_index": example.entity_index,
            "prompt_sha256": stable_hash(prompt),
            "target_sha256": stable_hash(example.target),
            "prediction_sha256": stable_hash(prediction),
            "steering_char_len": len(steering_span),
            "prompt_char_len": len(prompt),
            **generation_metrics(prediction, example.target, example.pii_type),
        }
        if args.save_text_predictions:
            row["prompt"] = prompt
            row["target"] = example.target
            row["prediction"] = prediction
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = [
        "exact_match",
        "normalized_exact_match",
        "starts_with_target",
        "normalized_starts_with_target",
        "contains_target",
        "normalized_contains_target",
        "prediction_char_len",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["pii_type"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    for pii_type, items in sorted(grouped.items()):
        summary: dict[str, Any] = {"pii_type": pii_type, "count": len(items)}
        for metric in metrics:
            values = [float(item[metric]) for item in items if metric in item]
            if values:
                summary[f"{metric}_mean"] = statistics.fmean(values)
                summary[f"{metric}_p50"] = statistics.median(values)
        summary_rows.append(summary)

    macro: dict[str, Any] = {"total_count": len(rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if metric in row]
        if values:
            macro[f"{metric}_mean"] = statistics.fmean(values)
    return summary_rows, macro


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["pii_type", "count", "source_id", "entity_index"]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    run_name = args.run_name or f"api4_reverse_attention_heads_edit_top50_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_dir / run_name
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    configs_dir = REPO_ROOT / "configs/attention_heads_edit/runs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args)
    if not examples:
        raise SystemExit("No examples loaded.")
    head_config = {} if args.disable_attention_heads_edit else load_head_config(args.head_config)
    print(
        f"[INFO] run={run_name} examples={len(examples)} "
        f"head_layers={len(head_config)} heads={sum(len(v) for v in head_config.values())} "
        f"alpha={args.alpha} scale_position={args.scale_position} disable_attention_heads_edit={args.disable_attention_heads_edit}",
        flush=True,
    )
    print("[INFO] pii_counts=" + ",".join(f"{k}:{v}" for k, v in sorted(Counter(e.pii_type for e in examples).items())), flush=True)

    model, tokenizer = load_model_and_tokenizer(args)
    ffn_edit_positions = apply_ffn_edit_if_requested(model, args)
    attention_heads_edit_model = unwrap_model_for_attention_heads_edit(model)
    attention_heads_edit_steerer = None
    if not args.disable_attention_heads_edit:
        attention_heads_edit_steerer = AttentionHeadsEdit(
            model=attention_heads_edit_model,
            tokenizer=tokenizer,
            head_config=head_config,
            alpha=args.alpha,
            scale_position=args.scale_position,
        )

    detail_rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(examples), args.batch_size), desc="reverse-attention_heads_edit-api4"):
        batch = examples[start : start + args.batch_size]
        try:
            detail_rows.extend(evaluate_batch(model, attention_heads_edit_model, tokenizer, attention_heads_edit_steerer, batch, args))
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[WARN] OOM at batch_start={start}; retry with smaller --batch_size.", flush=True)
            raise
        if args.log_every > 0 and len(detail_rows) and len(detail_rows) % args.log_every == 0:
            _, macro = summarize(detail_rows)
            print(
                f"[INFO] done={len(detail_rows)}/{len(examples)} "
                f"norm_exact={macro.get('normalized_exact_match_mean', 0.0):.4f} "
                f"norm_contains={macro.get('normalized_contains_target_mean', 0.0):.4f}",
                flush=True,
            )

    summary_rows, macro = summarize(detail_rows)
    write_csv(metrics_dir / "summary_by_pii_type.csv", summary_rows)
    write_csv(predictions_dir / "details_hashed.csv", detail_rows)
    meta = {
        "run_name": run_name,
        "dataset": str(args.dataset),
        "head_config": str(args.head_config),
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "disable_attention_heads_edit": args.disable_attention_heads_edit,
        "ffn_edit_kn_config": str(args.ffn_edit_kn_config) if args.ffn_edit_kn_config else None,
        "ffn_edit_erase_num": args.ffn_edit_erase_num,
        "ffn_edit_position_count": len(ffn_edit_positions),
        "ffn_edit_backend": getattr(args, "ffn_edit_backend", None),
        "alpha": args.alpha,
        "scale_position": args.scale_position,
        "steering_span_mode": args.steering_span_mode,
        "steering_tail_tokens": args.steering_tail_tokens,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "limit_per_type": args.limit_per_type,
        "max_context_tokens": args.max_context_tokens,
        "max_new_tokens": args.max_new_tokens,
        "attn_implementation": args.attn_implementation,
        "load_in_4bit": args.load_in_4bit,
        "save_text_predictions": args.save_text_predictions,
    }
    meta.update(ffn_edit_runtime_stats(model))
    if attention_heads_edit_steerer is not None:
        meta.update(attention_heads_edit_steerer.runtime_stats())
        if meta["attention_heads_edit_hook_call_count"] <= 0 or meta["attention_heads_edit_edit_call_count"] <= 0:
            raise RuntimeError("Attention Heads Edit was enabled but no attention-mask edits were observed")
    payload = {
        "meta": meta,
        "macro": macro,
        "summary": summary_rows,
    }
    with (metrics_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with (configs_dir / f"{run_name}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload["meta"], handle, ensure_ascii=False, indent=2)
    print("[DONE] " + json.dumps(macro, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
