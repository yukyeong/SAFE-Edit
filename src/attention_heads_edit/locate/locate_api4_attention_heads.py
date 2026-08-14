#!/usr/bin/env python
"""Locate API4 privacy heads with Attention Heads Edit-style attention scores."""

from __future__ import annotations

from safe_edit_paths import repo_root

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
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

from attention_heads_edit.eval.eval_api4_privacy_reverse_attention_heads_edit import apply_ffn_edit_if_requested, ffn_edit_runtime_stats
from attention_heads_edit.locate.attention_heads_privacy_spans import QUERY_MODES, STEERING_SPAN_MODES, attention_ranges


REPO_ROOT = repo_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate heads by attention to API4 input prefix spans.")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data/api4_200k/sft_true_prefix_no_instruction_train.json")
    parser.add_argument("--base_model", type=Path, default=REPO_ROOT / "models/llama3-8B/baseline")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=REPO_ROOT
        / "models/llama3-8B/api4_prefix_plain_qlora",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_heads_per_layer", type=int, default=0)
    parser.add_argument("--max_heads_per_kv_group", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_context_tokens", type=int, default=768)
    parser.add_argument("--ranking_metric", default="first_mass", choices=["all_mean", "first_mean", "all_mass", "first_mass"])
    parser.add_argument("--query_mode", default="next_token", choices=QUERY_MODES)
    parser.add_argument("--steering_span_mode", default="tail_tokens", choices=STEERING_SPAN_MODES)
    parser.add_argument("--steering_tail_tokens", type=int, default=8)
    parser.add_argument("--ffn_edit_kn_config", type=Path, default=None, help="Optional model-specific FFN Edit KN json installed before head localization.")
    parser.add_argument("--ffn_edit_erase_num", type=int, default=0, help="Number of FFN Edit neurons to erase during head localization.")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--metrics_dir", type=Path, default=REPO_ROOT / "outputs/attention_heads_edit/metrics")
    parser.add_argument("--config_dir", type=Path, default=REPO_ROOT / "outputs/attention_heads_edit/heads")
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


def config_value(config, key: str, default: Any = None) -> Any:
    candidates = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        candidates.append(text_config)
    if hasattr(config, "get_text_config"):
        try:
            candidates.append(config.get_text_config())
        except Exception:
            pass
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        value = getattr(candidate, key, None)
        if value is not None:
            return value
    return default


def load_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    with args.dataset.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    examples = [row for row in rows if row.get("input") and row.get("output")]
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    if args.max_samples > 0:
        examples = examples[: args.max_samples]
    return examples


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
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


def trim_pair(
    tokenizer,
    input_text: str,
    output_text: str,
    max_context_tokens: int,
    query_mode: str,
) -> tuple[list[int], list[int], int]:
    target_ids = tokenizer(output_text, add_special_tokens=False)["input_ids"]
    if not target_ids:
        return [], [], 0
    keep_prompt = max_context_tokens if query_mode == "next_token" else max(1, max_context_tokens - len(target_ids))
    original_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        prompt_encoding = tokenizer(
            input_text,
            add_special_tokens=True,
            truncation=True,
            max_length=keep_prompt,
            return_special_tokens_mask=True,
        )
    finally:
        tokenizer.truncation_side = original_truncation_side
    prompt_ids = prompt_encoding["input_ids"]
    special_tokens_mask = prompt_encoding.get("special_tokens_mask") or [0] * len(prompt_ids)
    prompt_text_len = sum(1 for value in special_tokens_mask if int(value) == 0)
    return prompt_ids, target_ids, prompt_text_len


def pad_sequences(sequences: list[list[int]], pad_token_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for idx, seq in enumerate(sequences):
        input_ids[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[idx, : len(seq)] = 1
    return input_ids.to(device), attention_mask.to(device)


@torch.no_grad()
def score_batch(model, tokenizer, batch: list[dict[str, Any]], args: argparse.Namespace):
    device = next(model.parameters()).device
    prompt_lens: list[int] = []
    prompt_text_lens: list[int] = []
    target_lens: list[int] = []
    full_ids: list[list[int]] = []
    pii_types: list[str] = []
    for row in batch:
        prompt_ids, target_ids, prompt_text_len = trim_pair(
            tokenizer,
            str(row["input"]),
            str(row["output"]),
            args.max_context_tokens,
            args.query_mode,
        )
        if not prompt_ids or not target_ids or prompt_text_len <= 0:
            continue
        prompt_lens.append(len(prompt_ids))
        prompt_text_lens.append(prompt_text_len)
        target_lens.append(len(target_ids))
        full_ids.append(prompt_ids if args.query_mode == "next_token" else prompt_ids + target_ids)
        pii_types.append(str(row.get("pii_type", "UNKNOWN")))
    if not full_ids:
        return None

    input_ids, attention_mask = pad_sequences(full_ids, tokenizer.pad_token_id, device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )
    attentions = outputs.attentions
    if attentions is None or not attentions or attentions[0] is None:
        model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
        raise RuntimeError(
            "output_attentions=True did not return attention matrices. "
            "For Attention Heads Edit head localization, rerun with --attn_implementation eager"
            + (" for Qwen3/QK-Norm/GQA models." if "qwen" in model_type else ".")
        )
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    all_mean_sum = torch.zeros(num_layers, num_heads, dtype=torch.float64)
    first_mean_sum = torch.zeros(num_layers, num_heads, dtype=torch.float64)
    all_mass_sum = torch.zeros(num_layers, num_heads, dtype=torch.float64)
    first_mass_sum = torch.zeros(num_layers, num_heads, dtype=torch.float64)
    valid_samples = 0

    for batch_idx, (prompt_len, prompt_text_len, target_len) in enumerate(
        zip(prompt_lens, prompt_text_lens, target_lens)
    ):
        q_start, q_end, key_start, key_end = attention_ranges(
            prompt_len=prompt_len,
            target_len=target_len,
            steering_span_mode=args.steering_span_mode,
            steering_tail_tokens=args.steering_tail_tokens,
            query_mode=args.query_mode,
            prompt_text_len=prompt_text_len,
        )
        if q_end > input_ids.shape[1] or prompt_len <= 0:
            continue
        valid_samples += 1
        for layer_idx, layer_attn in enumerate(attentions):
            sample_attn = layer_attn[batch_idx, :, q_start:q_end, key_start:key_end].float()
            all_mean_sum[layer_idx] += sample_attn.mean(dim=(1, 2)).double().cpu()
            all_mass_sum[layer_idx] += sample_attn.sum(dim=2).mean(dim=1).double().cpu()
            first_attn = sample_attn[:, :1, :]
            first_mean_sum[layer_idx] += first_attn.mean(dim=(1, 2)).double().cpu()
            first_mass_sum[layer_idx] += first_attn.sum(dim=2).mean(dim=1).double().cpu()

    return {
        "valid_samples": valid_samples,
        "pii_counts": Counter(pii_types),
        "all_mean_sum": all_mean_sum,
        "first_mean_sum": first_mean_sum,
        "all_mass_sum": all_mass_sum,
        "first_mass_sum": first_mass_sum,
    }


def rows_from_scores(scores: dict[str, torch.Tensor], ranking_metric: str, top_k: int) -> list[dict[str, Any]]:
    ranking = scores[ranking_metric]
    num_layers, num_heads = ranking.shape
    flat = ranking.reshape(-1)
    values, indices = torch.topk(flat, k=min(top_k, flat.numel()), largest=True)
    rows = []
    for rank, (flat_idx, score) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        layer = flat_idx // num_heads
        head = flat_idx % num_heads
        rows.append(
            {
                "rank": rank,
                "layer": layer,
                "head": head,
                "score": float(score),
                "ranking_metric": ranking_metric,
                "edit_key": f"L{layer}H{head}",
                "all_mean": float(scores["all_mean"][layer, head].item()),
                "first_mean": float(scores["first_mean"][layer, head].item()),
                "all_mass": float(scores["all_mass"][layer, head].item()),
                "first_mass": float(scores["first_mass"][layer, head].item()),
            }
        )
    return rows


def layer_balanced_rows(
    rows: list[dict[str, Any]],
    top_k: int,
    max_heads_per_layer: int,
    *,
    max_heads_per_kv_group: int = 0,
    num_query_heads: int = 0,
    num_key_value_heads: int = 0,
) -> list[dict[str, Any]]:
    use_group_cap = (
        max_heads_per_kv_group > 0
        and num_query_heads > 0
        and num_key_value_heads > 0
        and num_query_heads % num_key_value_heads == 0
    )
    if max_heads_per_layer <= 0 and not use_group_cap:
        selected = rows[:top_k]
    else:
        per_layer: Counter[int] = Counter()
        per_kv_group: Counter[tuple[int, int]] = Counter()
        query_heads_per_kv_group = num_query_heads // num_key_value_heads if use_group_cap else 0
        selected = []
        for row in rows:
            layer = int(row["layer"])
            if max_heads_per_layer > 0 and per_layer[layer] >= max_heads_per_layer:
                continue
            if use_group_cap:
                kv_group = int(row["head"]) // query_heads_per_kv_group
                if per_kv_group[(layer, kv_group)] >= max_heads_per_kv_group:
                    continue
                per_kv_group[(layer, kv_group)] += 1
            selected.append(dict(row))
            per_layer[layer] += 1
            if len(selected) >= top_k:
                break
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def head_config_from_rows(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    config: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        config[str(row["layer"])].append(int(row["head"]))
    return {layer: sorted(set(heads)) for layer, heads in sorted(config.items(), key=lambda item: int(item[0]))}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    run_name = args.run_name or f"api4_attention_heads_edit_attn_heads_top{args.top_k}_train{args.max_samples}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.config_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args)
    print(
        f"[INFO] run={run_name} examples={len(examples)} dataset={args.dataset} "
        f"ranking_metric={args.ranking_metric} top_k={args.top_k}",
        flush=True,
    )
    print("[INFO] pii_counts=" + ",".join(f"{k}:{v}" for k, v in sorted(Counter(str(e.get("pii_type", "UNKNOWN")) for e in examples).items())), flush=True)

    model, tokenizer = load_model_and_tokenizer(args)
    ffn_edit_positions = apply_ffn_edit_if_requested(model, args)
    sums: dict[str, torch.Tensor] | None = None
    total_valid = 0
    pii_counts: Counter[str] = Counter()

    for start in tqdm(range(0, len(examples), args.batch_size), desc="attention_heads_edit-head-locate"):
        result = score_batch(model, tokenizer, examples[start : start + args.batch_size], args)
        if result is None:
            continue
        total_valid += int(result["valid_samples"])
        pii_counts.update(result["pii_counts"])
        if sums is None:
            sums = {
                "all_mean": result["all_mean_sum"],
                "first_mean": result["first_mean_sum"],
                "all_mass": result["all_mass_sum"],
                "first_mass": result["first_mass_sum"],
            }
        else:
            for key in sums:
                sums[key] += result[f"{key}_sum"]
        if args.log_every > 0 and total_valid > 0 and total_valid % args.log_every == 0:
            print(f"[INFO] scored={total_valid}/{len(examples)}", flush=True)

    if sums is None or total_valid == 0:
        raise SystemExit("No valid examples scored.")
    scores = {key: value / total_valid for key, value in sums.items()}
    all_rows = rows_from_scores(scores, args.ranking_metric, top_k=scores[args.ranking_metric].numel())
    num_query_heads = int(config_value(getattr(model, "config", None), "num_attention_heads", 0) or 0)
    num_key_value_heads = int(config_value(getattr(model, "config", None), "num_key_value_heads", 0) or 0)
    top_rows = layer_balanced_rows(
        all_rows,
        args.top_k,
        args.max_heads_per_layer,
        max_heads_per_kv_group=args.max_heads_per_kv_group,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
    )
    head_config = head_config_from_rows(top_rows)

    full_csv = args.metrics_dir / f"{run_name}_all_heads.csv"
    top_csv = args.metrics_dir / f"{run_name}_top{args.top_k}.csv"
    config_json = args.config_dir / f"{run_name}_top{args.top_k}.json"
    metadata_json = args.config_dir / f"{run_name}_metadata.json"
    write_csv(full_csv, all_rows)
    write_csv(top_csv, top_rows)
    with config_json.open("w", encoding="utf-8") as handle:
        json.dump(head_config, handle, ensure_ascii=False, indent=2)
    metadata = {
        "run_name": run_name,
        "method": "Attention Heads Edit privacy localization: score next-token query attention to the prompt trigger span.",
        "architecture": {
            "model_type": str(config_value(getattr(model, "config", None), "model_type", "")),
            "num_attention_heads": int(config_value(getattr(model, "config", None), "num_attention_heads", 0) or 0),
            "num_key_value_heads": int(config_value(getattr(model, "config", None), "num_key_value_heads", 0) or 0),
            "attn_implementation": str(args.attn_implementation),
        },
        "input": {
            "dataset": str(args.dataset),
            "base_model": str(args.base_model),
            "adapter": str(args.adapter),
            "ffn_edit_kn_config": str(args.ffn_edit_kn_config) if args.ffn_edit_kn_config else None,
            "ffn_edit_erase_num": args.ffn_edit_erase_num,
            "ffn_edit_position_count": len(ffn_edit_positions),
            "ffn_edit_backend": getattr(args, "ffn_edit_backend", None),
            "num_examples": len(examples),
            "valid_examples": total_valid,
            "pii_counts": dict(sorted(pii_counts.items())),
            "query_mode": args.query_mode,
            "steering_span_mode": args.steering_span_mode,
            "steering_tail_tokens": args.steering_tail_tokens,
            "max_heads_per_layer": args.max_heads_per_layer,
            "max_heads_per_kv_group": args.max_heads_per_kv_group,
        },
        "output": {
            "all_heads_csv": str(full_csv),
            "top_heads_csv": str(top_csv),
            "head_config_json": str(config_json),
        },
        "impact_scope": "Read-only attention probing; does not modify weights; writes head_config for later Attention Heads Edit reverse scaling.",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "top_heads": top_rows,
    }
    metadata.update(ffn_edit_runtime_stats(model))
    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(
        "[DONE] "
        + json.dumps(
            {
                "valid_examples": total_valid,
                "head_config": str(config_json),
                "top_heads_csv": str(top_csv),
                "top1": top_rows[0] if top_rows else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
