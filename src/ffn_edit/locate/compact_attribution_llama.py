#!/usr/bin/env python3
"""Compact FFN Edit attribution for Llama/LoRA.

Computes the same integrated-gradient signal as calculate_attribution_llama,
but directly accumulates top knowledge neuron counts and writes kn_bag/kn_rel/kn_stats
JSON files without huge *.priv.jsonl intermediates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from safe_edit_paths import repo_root
from ffn_edit.hooks.custom_llama import patch_llama_model
from ffn_edit.locate.calculate_attribution_llama import scaled_input
from ffn_edit.pii import find_token_subsequence
from attention_heads_edit.attention_heads_edit import AttentionHeadsEdit

REPO_ROOT = repo_root()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def pos_key(layer: int, neuron: int) -> str:
    return f"{layer}@{neuron}"


def pos_value(key: str) -> list[int]:
    layer, neuron = key.split("@", 1)
    return [int(layer), int(neuron)]


def compact_state_path(output_dir: Path, output_prefix: str) -> Path:
    return output_dir / f"{output_prefix}.compact_state.json"


def load_compact_state(
    output_dir: Path, output_prefix: str
) -> tuple[int, Counter[str], dict[str, Any], dict[str, Any] | None]:
    path = compact_state_path(output_dir, output_prefix)
    if not path.is_file():
        return 0, Counter(), {}, None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    counts = Counter({str(k): int(v) for k, v in data.get("counts", {}).items()})
    stats = data.get("stats", {}) if isinstance(data.get("stats", {}), dict) else {}
    fingerprint = data.get("fingerprint") if isinstance(data.get("fingerprint"), dict) else stats.get("fingerprint")
    if fingerprint is not None and not isinstance(fingerprint, dict):
        fingerprint = None
    return int(data.get("completed_bags", 0)), counts, stats, fingerprint


def save_compact_state(
    output_dir: Path,
    output_prefix: str,
    completed_bags: int,
    counts: Counter[str],
    stats: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = compact_state_path(output_dir, output_prefix)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "completed_bags": completed_bags,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "output_prefix": output_prefix,
        "stats": stats,
        "counts": dict(counts),
        "fingerprint": stats.get("fingerprint"),
    }
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp_path, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attribution_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "priv_data_sha256": file_sha256(Path(args.priv_data_path)),
        "model_name_or_path": str(args.model_name_or_path),
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "skip_unlocated": bool(getattr(args, "skip_unlocated", False)),
        "require_gold_char_start": bool(getattr(args, "require_gold_char_start", False)),
        "max_seq_length": int(args.max_seq_length),
        "top_k": int(args.top_k),
        "threshold_ratio": float(args.threshold_ratio),
        "batch_size": int(args.batch_size),
        "num_batch": int(args.num_batch),
        "seed": int(args.seed),
        "layer_indices": args.layer_indices,
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
    }


def complete_marker_path(output_dir: Path, output_prefix: str) -> Path:
    return output_dir / f"{output_prefix}.complete.json"


def write_complete_marker(
    output_dir: Path,
    output_prefix: str,
    bag_path: Path,
    stats: dict[str, Any],
) -> Path:
    path = complete_marker_path(output_dir, output_prefix)
    payload = {
        "status": "complete",
        "completed_bags": int(stats.get("completed_bags", 0)),
        "expected_bags": stats.get("expected_bags"),
        "secret_located": int(stats.get("secret_located", 0)),
        "secret_unlocated": int(stats.get("secret_unlocated", 0)),
        "skipped_unlocated": int(stats.get("skipped_unlocated", 0)),
        "kn_bag": str(bag_path),
        "fingerprint": stats.get("fingerprint"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_kn_outputs(
    output_dir: Path,
    rel: str,
    counts: Counter[str],
    top_k: int,
    stats: dict[str, Any],
    args: argparse.Namespace,
    kn_status: str,
) -> Path:
    kn_dir = output_dir / "kn"
    kn_dir.mkdir(parents=True, exist_ok=True)
    top = counts.most_common(top_k)
    top_positions = [pos_value(key) for key, _ in top]
    singleton_bags = [[pos] for pos in top_positions]

    bag_path = kn_dir / f"kn_bag-{rel}.json"
    rel_path = kn_dir / f"kn_rel-{rel}.json"
    stats_path = kn_dir / f"kn_stats-{rel}.json"

    with bag_path.open("w", encoding="utf-8") as fh:
        json.dump(singleton_bags, fh, ensure_ascii=False, indent=2)
    with rel_path.open("w", encoding="utf-8") as fh:
        json.dump(top_positions, fh, ensure_ascii=False, indent=2)

    completed = int(stats.get("completed_bags", 0))
    metric_count = int(stats.get("metric_count", 0))
    metric_sum = float(stats.get("metric_max_sum", 0.0))
    filtered_sum = int(stats.get("filtered_positions_sum", 0))
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "mode": "compact_attribution_direct_kn",
                "kn_status": kn_status,
                "priv_data_path": args.priv_data_path,
                "model_name_or_path": args.model_name_or_path,
                "adapter_dir": args.adapter_dir,
                "completed_bags": completed,
                "expected_bags": stats.get("expected_bags"),
                "secret_located": int(stats.get("secret_located", 0)),
                "secret_unlocated": int(stats.get("secret_unlocated", 0)),
                "skipped_unlocated": int(stats.get("skipped_unlocated", 0)),
                "skip_unlocated": bool(getattr(args, "skip_unlocated", False)),
                "used_fallback_target": int(stats.get("used_fallback_target", 0)),
                "locate_mode": stats.get("locate_mode"),
                "fingerprint": stats.get("fingerprint"),
                "threshold_ratio": args.threshold_ratio,
                "top_k": top_k,
                "unique_positions": len(counts),
                "avg_filtered_per_bag": filtered_sum / completed if completed else 0,
                "avg_metric_max": metric_sum / metric_count if metric_count else 0,
                "batch_size": args.batch_size,
                "num_batch": args.num_batch,
                "max_seq_length": args.max_seq_length,
                "layer_indices": stats.get("layer_indices", []),
                "attention_heads_edit_head_config": getattr(args, "attention_heads_edit_head_config", None),
                "attention_heads_edit_alpha": getattr(args, "attention_heads_edit_alpha", None),
                "attention_heads_edit_scale_position": getattr(args, "attention_heads_edit_scale_position", None),
                "top_counts": [[key, count] for key, count in top[:100]],
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    return bag_path


def parse_layer_indices(args: argparse.Namespace, num_hidden_layers: int) -> list[int]:
    if args.layer_indices:
        layer_indices = sorted({int(x.strip()) for x in args.layer_indices.split(",") if x.strip()})
    else:
        layer_start = 0 if args.layer_start is None else args.layer_start
        layer_end = num_hidden_layers - 1 if args.layer_end is None else args.layer_end
        layer_indices = list(range(layer_start, layer_end + 1))
    layer_indices = [idx for idx in layer_indices if 0 <= idx < num_hidden_layers]
    if not layer_indices:
        raise ValueError("No valid layers selected for attribution.")
    return layer_indices


def unwrap_model_for_attention_heads_edit(model):
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        return base_model.model
    return model


def load_head_config(path: str | None) -> dict[int, list[int]] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict) and not raw:
        return {}
    if isinstance(raw, dict) and "edit_heads" in raw:
        grouped: dict[int, list[int]] = {}
        for item in raw["edit_heads"]:
            grouped.setdefault(int(item["layer"]), []).append(int(item["head"]))
        return {layer: sorted(set(heads)) for layer, heads in grouped.items()}
    if isinstance(raw, dict):
        return {
            int(layer): sorted({int(head) for head in heads})
            for layer, heads in raw.items()
            if isinstance(heads, list)
        }
    raise ValueError(f"Unsupported Attention Heads Edit head config format: {type(raw).__name__}")


def active_token_count(attention_mask: torch.Tensor) -> int:
    return int(attention_mask[0].sum().item())


def prepare_forward_inputs(
    tokenizer,
    attention_heads_edit_steerer: AttentionHeadsEdit | None,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
):
    real_len = active_token_count(attention_mask)
    input_ids = input_ids[:, :real_len]
    attention_mask = attention_mask[:, :real_len]
    if attention_heads_edit_steerer is None:
        prefix_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        return input_ids, attention_mask, prefix_text, None
    prefix_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    model_inputs, offsets_mapping = attention_heads_edit_steerer.inputs_from_batch([prefix_text], tokenizer=tokenizer, device=device)
    return model_inputs["input_ids"], model_inputs["attention_mask"], prefix_text, offsets_mapping


def locate_secret_tokens(
    tokenizer,
    full_text: str,
    secret: str,
    gold_char_start: int | None = None,
) -> tuple[list[int], int | None, int | None]:
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if not full_ids:
        return [], None, None

    if gold_char_start is not None:
        if gold_char_start < 0 or gold_char_start > len(full_text):
            return full_ids, None, None
        if secret and full_text[gold_char_start : gold_char_start + len(secret)] != secret:
            return full_ids, None, None
        secret_start_char = gold_char_start
    else:
        secret_start_char = full_text.find(secret) if secret else -1

    if secret and secret_start_char >= 0 and offsets:
        secret_end_char = secret_start_char + len(secret)
        token_positions = [
            idx
            for idx, (tok_start, tok_end) in enumerate(offsets)
            if tok_end > secret_start_char and tok_start < secret_end_char
        ]
        if token_positions:
            start = token_positions[0]
            end = token_positions[-1] + 1
            return full_ids, start, end

    if gold_char_start is None and secret:
        secret_ids = tokenizer(secret, add_special_tokens=False)["input_ids"]
        start = find_token_subsequence(full_ids, secret_ids)
        if start is not None:
            return full_ids, start, start + len(secret_ids)
    return full_ids, None, None


def parse_gold_char_start(example: list[Any]) -> int | None:
    if len(example) < 3 or example[2] in (None, ""):
        return None
    return int(example[2])


def example2feature(example: list[Any], max_seq_length: int, tokenizer) -> tuple[dict[str, list[int]], dict[str, Any]]:
    full_text = str(example[0])
    secret = str(example[1]) if len(example) > 1 else ""
    gold_char_start = parse_gold_char_start(example)
    locate_mode = "gold_char_start" if gold_char_start is not None else "first_match"
    full_ids, secret_start, secret_end = locate_secret_tokens(
        tokenizer, full_text, secret, gold_char_start=gold_char_start
    )

    gold_obj = None
    skip_reason = None
    secret_located = secret_start is not None and secret_start > 0
    if secret_located:
        gold_obj = int(full_ids[secret_start])
        input_ids = full_ids[:secret_start]
    else:
        input_ids = full_ids
        if secret_start == 0:
            skip_reason = "empty_prefix"
        elif gold_char_start is not None:
            skip_reason = "gold_char_start_not_tokenized"
        else:
            skip_reason = "secret_not_located"

    if not input_ids:
        input_ids = [tokenizer.eos_token_id]
        if secret_located:
            secret_located = False
            skip_reason = "empty_prefix"
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]

    real_len = len(input_ids)
    target_pos = max(0, real_len - 1)
    used_fallback_target = False
    if gold_obj is None:
        next_pos = min(target_pos + 1, len(full_ids) - 1)
        gold_obj = int(full_ids[next_pos]) if full_ids else int(tokenizer.eos_token_id)
        used_fallback_target = True

    padding_length = max_seq_length - real_len
    if padding_length > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
    attention_mask = [1] * real_len + [0] * max(0, padding_length)

    return (
        {"input_ids": input_ids, "attention_mask": attention_mask},
        {
            "gold_obj": gold_obj,
            "target_pos": target_pos,
            "secret_located": bool(secret_located),
            "secret_token_span": [secret_start, secret_end] if secret_start is not None else None,
            "gold_char_start": gold_char_start,
            "locate_mode": locate_mode,
            "skip_reason": skip_reason,
            "used_fallback_target": used_fallback_target,
        },
    )


def _example_row(full_text: Any, secret: Any, gold_char_start: Any = None) -> list[Any]:
    row: list[Any] = [str(full_text), str(secret)]
    if gold_char_start not in (None, ""):
        row.append(int(gold_char_start))
    return row


def normalize_privacy_bag(raw_bag: Any) -> tuple[list[list[Any]], bool]:
    """Accept [[full, secret], ...], [[full, secret, gold_char_start], ...], or legacy [full, secret]."""

    if isinstance(raw_bag, dict):
        secret = raw_bag.get("output", raw_bag.get("target", ""))
        full_text = raw_bag.get("full_text")
        gold = raw_bag.get("gold_char_start", raw_bag.get("target_char_start"))
        if full_text is None:
            prefix = str(raw_bag.get("input", ""))
            full_text = prefix + str(secret)
            if gold is None:
                gold = len(prefix)
        return [_example_row(full_text, secret, gold)], False

    if not isinstance(raw_bag, (list, tuple)) or not raw_bag:
        return [], False

    if len(raw_bag) >= 2 and isinstance(raw_bag[0], str) and isinstance(raw_bag[1], str):
        gold = raw_bag[2] if len(raw_bag) >= 3 else None
        return [_example_row(raw_bag[0], raw_bag[1], gold)], True

    normalized: list[list[Any]] = []
    for item in raw_bag:
        if isinstance(item, dict):
            secret = item.get("output", item.get("target", ""))
            full_text = item.get("full_text")
            gold = item.get("gold_char_start", item.get("target_char_start"))
            if full_text is None:
                prefix = str(item.get("input", ""))
                full_text = prefix + str(secret)
                if gold is None:
                    gold = len(prefix)
            normalized.append(_example_row(full_text, secret, gold))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            gold = item[2] if len(item) >= 3 else None
            normalized.append(_example_row(item[0], item[1], gold))
    return normalized, False


def validate_privacy_bags(privacy_bags: list[Any], *, require_gold_char_start: bool = False) -> None:
    checked = 0
    located = 0
    gold_present = 0
    legacy_pairs = 0
    for raw_bag in privacy_bags[: min(50, len(privacy_bags))]:
        examples, was_legacy_pair = normalize_privacy_bag(raw_bag)
        legacy_pairs += int(was_legacy_pair)
        for example in examples:
            full_text, secret = example[0], example[1]
            checked += 1
            if parse_gold_char_start(example) is not None:
                gold_present += 1
            if secret and secret in full_text:
                located += 1
    if checked and located == 0:
        raise SystemExit(
            "privacy data format check failed: no sampled secret appears in its full text. "
            "Expected [[full_text, secret], ...] or [[full_text, secret, gold_char_start], ...]."
        )
    if require_gold_char_start and gold_present != checked:
        raise SystemExit(
            f"require_gold_char_start failed: gold present in {gold_present}/{checked} sampled examples"
        )
    if legacy_pairs:
        print(
            f"[compact_attr] normalized {legacy_pairs} sampled legacy [full_text, secret] bags; "
            "please regenerate privacy_data for stable future runs.",
            flush=True,
        )


def attention_heads_edit_context(
    attention_heads_edit_steerer: AttentionHeadsEdit | None,
    attention_heads_edit_model,
    strings: list[str],
    substrings: list[str],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    offsets_mapping,
):
    if attention_heads_edit_steerer is None:
        return nullcontext()
    model_input = {"input_ids": input_ids, "attention_mask": attention_mask}
    return attention_heads_edit_steerer.apply_steering(
        model=attention_heads_edit_model,
        strings=strings,
        substrings=substrings,
        model_input=model_input,
        offsets_mapping=offsets_mapping,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--priv_data_path", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--rel", default=None, help="Relation/name used in kn_bag-<rel>.json; defaults to output_prefix.")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_batch", type=int, default=10)
    parser.add_argument("--layer_start", type=int, default=None)
    parser.add_argument("--layer_end", type=int, default=None)
    parser.add_argument("--layer_indices", default=None)
    parser.add_argument("--threshold_ratio", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=512)
    parser.add_argument("--attention_heads_edit_head_config", default=None)
    parser.add_argument("--attention_heads_edit_alpha", type=float, default=0.15)
    parser.add_argument("--attention_heads_edit_scale_position", default="include_down", choices=["include_down", "include", "exclude", "generation"])
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--progress_every", type=int, default=1)
    parser.add_argument("--limit_bags", type=int, default=None, help="Debug/smoke-test limit.")
    parser.add_argument(
        "--skip_unlocated",
        action="store_true",
        help="Skip examples whose secret span is not located.",
    )
    parser.add_argument(
        "--require_gold_char_start",
        action="store_true",
        help="Require gold_char_start on every privacy example.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{args.output_prefix}.args.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, ensure_ascii=False, sort_keys=True, indent=2)

    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpus) == 1:
        device = torch.device(f"cuda:{args.gpus}")
        n_gpu = 1
    else:
        raise ValueError("This compact runner currently expects a single GPU id, e.g. --gpus 0.")
    print(f"device: {device} n_gpu: {n_gpu}, distributed training: {bool(n_gpu > 1)}", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("***** CUDA.empty_cache() *****", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    loaded_with_device_map = n_gpu > 0
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if loaded_with_device_map else None,
    )
    if args.adapter_dir:
        from peft import PeftModel
        print(f"Loading adapter from {args.adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        print("Adapter loaded.", flush=True)

    print("Patching model for FFN Edit...", flush=True)
    model = patch_llama_model(model)
    print("Patch done.", flush=True)
    if n_gpu > 0 and not loaded_with_device_map:
        model.to(device)
    model.eval()

    attention_heads_edit_steerer = None
    attention_heads_edit_model = None
    head_config = load_head_config(args.attention_heads_edit_head_config)
    if head_config is not None:
        attention_heads_edit_model = unwrap_model_for_attention_heads_edit(model)
        attention_heads_edit_steerer = AttentionHeadsEdit(
            model=attention_heads_edit_model,
            tokenizer=tokenizer,
            head_config=head_config,
            alpha=args.attention_heads_edit_alpha,
            scale_position=args.attention_heads_edit_scale_position,
        )
        print(
            f"Attention Heads Edit enabled: heads={sum(len(v) for v in head_config.values())} "
            f"alpha={args.attention_heads_edit_alpha} scale_position={args.attention_heads_edit_scale_position}",
            flush=True,
        )

    intermediate_size = getattr(config, "intermediate_size", config.hidden_size * 4)
    num_hidden_layers = config.num_hidden_layers
    layer_indices = parse_layer_indices(args, num_hidden_layers)
    print(f"Scanning layers: {layer_indices}", flush=True)

    with open(args.priv_data_path, "r", encoding="utf-8") as fh:
        eval_bag_list_all = json.load(fh)
    if args.limit_bags is not None:
        eval_bag_list_all = eval_bag_list_all[: args.limit_bags]
    validate_privacy_bags(eval_bag_list_all, require_gold_char_start=bool(args.require_gold_char_start))
    if args.require_gold_char_start:
        missing_gold = 0
        for bag in eval_bag_list_all:
            examples, _ = normalize_privacy_bag(bag)
            missing_gold += sum(1 for example in examples if parse_gold_char_start(example) is None)
        if missing_gold:
            raise SystemExit(f"require_gold_char_start failed: {missing_gold} examples missing gold_char_start")

    expected_fp = attribution_fingerprint(args)
    resume_count, counts, stats, saved_fp = load_compact_state(output_dir, args.output_prefix)
    if args.limit_bags is not None and resume_count > 0:
        print("Ignoring compact resume state because --limit_bags is set.", flush=True)
        resume_count, counts, stats, saved_fp = 0, Counter(), {}, None
    if resume_count > 0 and saved_fp != expected_fp:
        print(
            f"[compact_attr] state fingerprint mismatch; refusing to resume. "
            f"saved={saved_fp} expected={expected_fp}",
            flush=True,
        )
        resume_count, counts, stats = 0, Counter(), {}
    if resume_count > 0:
        print(f"Resuming compact attribution from {resume_count} completed bags; unique={len(counts)}", flush=True)

    stats.setdefault("metric_max_sum", 0.0)
    stats.setdefault("metric_count", 0)
    stats.setdefault("filtered_positions_sum", 0)
    stats.setdefault("secret_located", 0)
    stats.setdefault("secret_unlocated", 0)
    stats.setdefault("skipped_unlocated", 0)
    stats.setdefault("used_fallback_target", 0)
    stats["layer_indices"] = layer_indices
    stats["fingerprint"] = expected_fp
    stats["locate_mode"] = "gold_char_start_if_present"
    stats["expected_bags"] = len(eval_bag_list_all)
    skipped_path = output_dir / f"{args.output_prefix}.skipped_unlocated.jsonl"
    if resume_count == 0 and skipped_path.exists():
        skipped_path.unlink()
    complete_path = complete_marker_path(output_dir, args.output_prefix)
    if resume_count == 0 and complete_path.exists():
        complete_path.unlink()

    tic = time.perf_counter()
    total_bags = len(eval_bag_list_all)
    print(f"start compact processing, dataset bags={total_bags}", flush=True)

    completed = resume_count
    processed_since_save = 0
    rel = args.rel or args.output_prefix

    for bag_idx, priv_texts in enumerate(eval_bag_list_all):
        if bag_idx < resume_count:
            continue
        priv_examples, was_legacy_pair = normalize_privacy_bag(priv_texts)
        if was_legacy_pair:
            stats["legacy_pair_bags_normalized"] = int(stats.get("legacy_pair_bags_normalized", 0)) + 1
        if not priv_examples:
            logger.warning("Empty bag at index %s, skipping.", bag_idx)
            continue

        sum_tensor = torch.zeros((len(layer_indices), intermediate_size), dtype=torch.float32)

        for ex_idx, eval_example in enumerate(priv_examples):
            eval_features, tokens_info = example2feature(eval_example, args.max_seq_length, tokenizer)
            located = bool(tokens_info["secret_located"])
            stats["secret_located" if located else "secret_unlocated"] = int(
                stats.get("secret_located" if located else "secret_unlocated", 0)
            ) + 1
            if args.skip_unlocated and not located:
                stats["skipped_unlocated"] = int(stats.get("skipped_unlocated", 0)) + 1
                with skipped_path.open("a", encoding="utf-8") as skip_handle:
                    skip_handle.write(
                        json.dumps(
                            {
                                "bag_idx": bag_idx,
                                "ex_idx": ex_idx,
                                "reason": tokens_info.get("skip_reason"),
                                "locate_mode": tokens_info.get("locate_mode"),
                                "gold_char_start": tokens_info.get("gold_char_start"),
                                "secret_token_span": tokens_info.get("secret_token_span"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue
            if tokens_info.get("used_fallback_target"):
                stats["used_fallback_target"] = int(stats.get("used_fallback_target", 0)) + 1
            input_ids = torch.tensor(eval_features["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
            attention_mask = torch.tensor(eval_features["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)
            input_ids, attention_mask, prefix_text, offsets_mapping = prepare_forward_inputs(
                tokenizer,
                attention_heads_edit_steerer,
                input_ids,
                attention_mask,
                device,
            )
            tgt_pos = int(tokens_info["target_pos"])

            for local_layer_idx, tgt_layer in enumerate(layer_indices):
                if tgt_layer == layer_indices[0] or (tgt_layer + 1) % 8 == 0 or tgt_layer == layer_indices[-1]:
                    print(f"  bag {bag_idx} ex {ex_idx} layer {tgt_layer}/{num_hidden_layers - 1}", flush=True)

                with attention_heads_edit_context(
                    attention_heads_edit_steerer,
                    attention_heads_edit_model,
                    [prefix_text],
                    [prefix_text],
                    input_ids,
                    attention_mask,
                    offsets_mapping,
                ):
                    ffn_weights, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        tgt_pos=tgt_pos,
                        tgt_layer=tgt_layer,
                    )
                if ffn_weights is None:
                    logger.warning("FFN weights is None at layer %s, skipping...", tgt_layer)
                    ffn_weights = torch.zeros((1, intermediate_size), device=device)

                gold_label = int(tokens_info["gold_obj"])

                scaled_weights, weights_step = scaled_input(ffn_weights, args.batch_size, args.num_batch)
                scaled_weights.requires_grad_(True)

                ig_gold = None
                for batch_idx in range(args.num_batch):
                    batch_weights = scaled_weights[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
                    cur_batch_size = batch_weights.shape[0]
                    if attention_heads_edit_steerer is not None:
                        batch_inputs, batch_offsets = attention_heads_edit_steerer.inputs_from_batch(
                            [prefix_text] * cur_batch_size,
                            tokenizer=tokenizer,
                            device=device,
                        )
                        batch_input_ids = batch_inputs["input_ids"]
                        batch_attention_mask = batch_inputs["attention_mask"]
                    else:
                        batch_input_ids = input_ids.repeat(cur_batch_size, 1)
                        batch_attention_mask = attention_mask.repeat(cur_batch_size, 1)
                        batch_offsets = None
                    labels = batch_input_ids.clone()
                    labels[:, tgt_pos] = gold_label

                    with attention_heads_edit_context(
                        attention_heads_edit_steerer,
                        attention_heads_edit_model,
                        [prefix_text] * cur_batch_size,
                        [prefix_text] * cur_batch_size,
                        batch_input_ids,
                        batch_attention_mask,
                        batch_offsets,
                    ):
                        _tgt_prob, grad = model(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                            tgt_pos=tgt_pos,
                            tgt_layer=tgt_layer,
                            tmp_score=batch_weights,
                            labels=labels,
                        )
                    grad = grad.sum(dim=0)
                    ig_gold = grad if ig_gold is None else torch.add(ig_gold, grad)

                ig_gold = (ig_gold * weights_step).detach().float().cpu()
                sum_tensor[local_layer_idx].add_(ig_gold)
                del scaled_weights, weights_step, ig_gold

            print(f"  bag {bag_idx} ex {ex_idx} done.", flush=True)

        metric_max = float(sum_tensor.max().item())
        bag_positions: set[str] = set()
        if metric_max > 0:
            cutoff = metric_max * args.threshold_ratio
            nonzero = (sum_tensor >= cutoff).nonzero(as_tuple=False)
            for local_layer_idx, neuron_idx in nonzero.tolist():
                bag_positions.add(pos_key(layer_indices[local_layer_idx], neuron_idx))
            stats["metric_max_sum"] = float(stats.get("metric_max_sum", 0.0)) + metric_max
            stats["metric_count"] = int(stats.get("metric_count", 0)) + 1
        counts.update(bag_positions)
        stats["filtered_positions_sum"] = int(stats.get("filtered_positions_sum", 0)) + len(bag_positions)

        completed += 1
        stats["completed_bags"] = completed
        processed_since_save += 1

        if args.progress_every > 0 and (completed <= 2 or completed % args.progress_every == 0):
            elapsed = time.perf_counter() - tic
            avg_filtered = int(stats.get("filtered_positions_sum", 0)) / completed if completed else 0
            print(
                f"[compact_attr] bags={completed}/{total_bags} unique={len(counts)} "
                f"bag_filtered={len(bag_positions)} avg_filtered={avg_filtered:.1f} elapsed={elapsed:.1f}s",
                flush=True,
            )

        if processed_since_save >= max(1, args.save_every):
            save_compact_state(output_dir, args.output_prefix, completed, counts, stats)
            bag_path = write_kn_outputs(output_dir, rel, counts, args.top_k, stats, args, kn_status="checkpoint")
            print(f"[compact_attr] checkpoint saved; kn_status=checkpoint kn={bag_path}", flush=True)
            processed_since_save = 0

    if completed != total_bags:
        raise SystemExit(f"attribution ended incomplete: completed={completed} expected={total_bags}")
    if args.skip_unlocated and int(stats.get("secret_located", 0)) <= 0:
        raise SystemExit("refusing to write final kn: skip_unlocated found 0 located secrets")

    save_compact_state(output_dir, args.output_prefix, completed, counts, stats)
    bag_path = write_kn_outputs(output_dir, rel, counts, args.top_k, stats, args, kn_status="complete")
    marker = write_complete_marker(output_dir, args.output_prefix, bag_path, stats)
    toc = time.perf_counter()
    print(f"[compact_attr] wrote {bag_path}", flush=True)
    print(f"[compact_attr] complete marker {marker}", flush=True)
    print(
        f"[compact_attr] done bags={completed}/{total_bags} located={stats.get('secret_located', 0)} "
        f"unlocated={stats.get('secret_unlocated', 0)} skipped={stats.get('skipped_unlocated', 0)} "
        f"unique={len(counts)} time={toc - tic:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
