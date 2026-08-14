#!/usr/bin/env python3
"""Compact FFN Edit attribution for Gemma/Gemma3/Qwen3 LoRA models.

This runner computes integrated-gradient scores on gated-MLP neuron
activation entering ``mlp.down_proj`` and writes FFN Edit KN files directly:

- ``kn/kn_bag-<rel>.json``
- ``kn/kn_rel-<rel>.json``
- ``kn/kn_stats-<rel>.json``

Input privacy bags are expected to be ``[[full_text, secret], ...]`` per bag,
matching the LUME/API4-style privacy files in this repo.
"""

from __future__ import annotations

from safe_edit_paths import repo_root

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

from ffn_edit.hooks.custom_gemma import find_gemma_down_proj_modules, patch_gemma_model_for_ffn_edit
from ffn_edit.hooks.custom_ministral import find_ministral_down_proj_modules, patch_ministral_model_for_ffn_edit
from ffn_edit.hooks.custom_qwen import find_qwen_down_proj_modules, patch_qwen_model_for_ffn_edit
from ffn_edit.pii import find_token_subsequence

REPO_ROOT = repo_root()


def pos_key(layer: int, neuron: int) -> str:
    return f"{layer}@{neuron}"


def pos_value(key: str) -> list[int]:
    layer, neuron = key.split("@", 1)
    return [int(layer), int(neuron)]


def dtype_from_arg(name: str):
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def scaled_input(emb: torch.Tensor, batch_size: int, num_batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    baseline = torch.zeros_like(emb)
    num_points = batch_size * num_batch
    step = (emb - baseline) / num_points
    scaled = torch.cat([baseline + step * idx for idx in range(num_points)], dim=0)
    return scaled, step[0]


def normalize_for_assert(text: str) -> str:
    return " ".join(str(text).strip().split())


def locate_secret_tokens(tokenizer, full_text: str, secret: str) -> tuple[list[int], int | None, int | None]:
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if not full_ids:
        return [], None, None

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
            decoded_secret = tokenizer.decode(full_ids[start:end], skip_special_tokens=True)
            if normalize_for_assert(secret) in normalize_for_assert(decoded_secret) or normalize_for_assert(
                decoded_secret
            ) in normalize_for_assert(secret):
                return full_ids, start, end
            return full_ids, start, end

    if secret:
        secret_ids = tokenizer(secret, add_special_tokens=False)["input_ids"]
        start = find_token_subsequence(full_ids, secret_ids)
        if start is not None:
            return full_ids, start, start + len(secret_ids)
    return full_ids, None, None


def example2feature(example: list[Any], max_seq_length: int, tokenizer) -> tuple[dict[str, list[int]], dict[str, Any]]:
    full_text = str(example[0])
    secret = str(example[1]) if len(example) > 1 else ""
    full_ids, secret_start, secret_end = locate_secret_tokens(tokenizer, full_text, secret)

    gold_obj = None
    secret_located = secret_start is not None and secret_start > 0
    if secret_located:
        gold_obj = int(full_ids[secret_start])
        input_ids = full_ids[:secret_start]
    else:
        input_ids = full_ids

    if not input_ids:
        input_ids = [tokenizer.eos_token_id]
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]

    real_len = len(input_ids)
    target_pos = max(0, real_len - 1)
    if gold_obj is None:
        next_pos = min(target_pos + 1, len(full_ids) - 1)
        gold_obj = int(full_ids[next_pos]) if full_ids else int(tokenizer.eos_token_id)

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
        },
    )


def normalize_privacy_bag(raw_bag: Any) -> tuple[list[list[str]], bool]:
    """Accept both [[full, secret], ...] and legacy [full, secret] bag shapes."""

    if isinstance(raw_bag, dict):
        secret = raw_bag.get("output", raw_bag.get("target", ""))
        full_text = raw_bag.get("full_text")
        if full_text is None:
            full_text = str(raw_bag.get("input", "")) + str(secret)
        return [[str(full_text), str(secret)]], False

    if not isinstance(raw_bag, (list, tuple)) or not raw_bag:
        return [], False

    if len(raw_bag) >= 2 and isinstance(raw_bag[0], str) and isinstance(raw_bag[1], str):
        return [[str(raw_bag[0]), str(raw_bag[1])]], True

    normalized: list[list[str]] = []
    for item in raw_bag:
        if isinstance(item, dict):
            secret = item.get("output", item.get("target", ""))
            full_text = item.get("full_text")
            if full_text is None:
                full_text = str(item.get("input", "")) + str(secret)
            normalized.append([str(full_text), str(secret)])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            normalized.append([str(item[0]), str(item[1])])
    return normalized, False


def validate_privacy_bags(privacy_bags: list[Any]) -> None:
    checked = 0
    located = 0
    legacy_pairs = 0
    for raw_bag in privacy_bags[: min(50, len(privacy_bags))]:
        examples, was_legacy_pair = normalize_privacy_bag(raw_bag)
        legacy_pairs += int(was_legacy_pair)
        for full_text, secret in examples:
            checked += 1
            if secret and secret in full_text:
                located += 1
    if checked and located == 0:
        raise SystemExit(
            "privacy data format check failed: no sampled secret appears in its full text. "
            "Expected [[full_text, secret], ...] per bag or legacy [full_text, secret]."
        )
    if legacy_pairs:
        print(
            f"[attr] normalized {legacy_pairs} sampled legacy [full_text, secret] bags; "
            "please regenerate privacy_data for stable future runs.",
            flush=True,
        )


def text_config_value(config, key: str, default: Any = None) -> Any:
    if hasattr(config, key):
        value = getattr(config, key)
        if value is not None:
            return value
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, key):
        value = getattr(text_config, key)
        if value is not None:
            return value
    return default


def infer_ffn_edit_model_kind(config, explicit_kind: str) -> str:
    if explicit_kind != "auto":
        return explicit_kind
    model_type = str(getattr(config, "model_type", "")).lower()
    text_model_type = str(getattr(getattr(config, "text_config", None), "model_type", "")).lower()
    joined = f"{model_type} {text_model_type}"
    if "qwen" in joined:
        return "qwen"
    if "ministral" in joined or "mistral3" in joined:
        return "ministral"
    if "gemma" in joined:
        return "gemma"
    raise ValueError(f"Could not infer FFN Edit model kind from model_type={model_type!r}; pass --ffn_edit_model_kind.")


def patch_model_for_ffn_edit(model, ffn_edit_model_kind: str):
    if ffn_edit_model_kind == "qwen":
        return patch_qwen_model_for_ffn_edit(model)
    if ffn_edit_model_kind == "ministral":
        return patch_ministral_model_for_ffn_edit(model)
    return patch_gemma_model_for_ffn_edit(model)


def find_down_proj_modules(model, ffn_edit_model_kind: str):
    if ffn_edit_model_kind == "qwen":
        return find_qwen_down_proj_modules(model)
    if ffn_edit_model_kind == "ministral":
        return find_ministral_down_proj_modules(model)
    return find_gemma_down_proj_modules(model)


def parse_layer_indices(args: argparse.Namespace, discovered_layers: list[int]) -> list[int]:
    if args.layer_indices:
        selected = sorted({int(x.strip()) for x in args.layer_indices.split(",") if x.strip()})
    else:
        first = discovered_layers[0] if args.layer_start is None else int(args.layer_start)
        last = discovered_layers[-1] if args.layer_end is None else int(args.layer_end)
        selected = list(range(first, last + 1))
    discovered = set(discovered_layers)
    selected = [idx for idx in selected if idx in discovered]
    if not selected:
        raise ValueError(f"No valid layers selected. discovered_layers={discovered_layers}")
    return selected


def state_path(output_dir: Path, output_prefix: str) -> Path:
    return output_dir / f"{output_prefix}.compact_state.json"


def load_state(output_dir: Path, output_prefix: str) -> tuple[int, Counter[str], dict[str, float], dict[str, Any]]:
    path = state_path(output_dir, output_prefix)
    if not path.is_file():
        return 0, Counter(), {}, {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return (
        int(data.get("completed_bags", 0)),
        Counter({str(k): int(v) for k, v in data.get("counts", {}).items()}),
        {str(k): float(v) for k, v in data.get("score_sums", {}).items()},
        data.get("stats", {}) if isinstance(data.get("stats", {}), dict) else {},
    )


def save_state(
    output_dir: Path,
    output_prefix: str,
    completed: int,
    counts: Counter[str],
    score_sums: dict[str, float],
    stats: dict[str, Any],
) -> None:
    path = state_path(output_dir, output_prefix)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "completed_bags": completed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "output_prefix": output_prefix,
        "stats": stats,
        "counts": dict(counts),
        "score_sums": score_sums,
    }
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(tmp, path)


def write_kn_outputs(
    output_dir: Path,
    rel: str,
    counts: Counter[str],
    score_sums: dict[str, float],
    top_k: int,
    stats: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    kn_dir = output_dir / "kn"
    kn_dir.mkdir(parents=True, exist_ok=True)
    ranking_metric = str(getattr(args, "ranking_metric", "count"))
    if ranking_metric == "score":
        ranked = sorted(score_sums, key=lambda key: (-score_sums[key], -counts.get(key, 0), key))
    elif ranking_metric == "count_score":
        ranked = sorted(score_sums, key=lambda key: (-(counts.get(key, 0) * score_sums[key]), -counts.get(key, 0), key))
    else:
        ranked = [key for key, _count in counts.most_common()]
    top = [(key, counts.get(key, 0)) for key in ranked[:top_k]]
    top_positions = [pos_value(key) for key, _ in top]
    singleton_bags = [[pos] for pos in top_positions]

    bag_path = kn_dir / f"kn_bag-{rel}.json"
    rel_path = kn_dir / f"kn_rel-{rel}.json"
    stats_path = kn_dir / f"kn_stats-{rel}.json"
    bag_path.write_text(json.dumps(singleton_bags, ensure_ascii=False, indent=2), encoding="utf-8")
    rel_path.write_text(json.dumps(top_positions, ensure_ascii=False, indent=2), encoding="utf-8")

    completed = int(stats.get("completed_bags", 0))
    metric_count = int(stats.get("metric_count", 0))
    ffn_edit_model_kind = str(stats.get("ffn_edit_model_kind", getattr(args, "ffn_edit_model_kind", "auto")))
    payload = {
        "mode": f"{ffn_edit_model_kind}_down_proj_compact_attribution",
        "ffn_edit_model_kind": ffn_edit_model_kind,
        "privacy_neuron_definition": "mlp.down_proj input = act_fn(gate_proj(x)) * up_proj(x)",
        "priv_data_path": args.priv_data_path,
        "model_name_or_path": args.model_name_or_path,
        "adapter_dir": args.adapter_dir,
        "completed_bags": completed,
        "secret_located": int(stats.get("secret_located", 0)),
        "secret_unlocated": int(stats.get("secret_unlocated", 0)),
        "threshold_ratio": args.threshold_ratio,
        "ranking_metric": ranking_metric,
        "top_k": top_k,
        "unique_positions": len(counts),
        "avg_filtered_per_bag": int(stats.get("filtered_positions_sum", 0)) / completed if completed else 0,
        "avg_metric_max": float(stats.get("metric_max_sum", 0.0)) / metric_count if metric_count else 0,
        "batch_size": args.batch_size,
        "num_batch": args.num_batch,
        "ig_steps": args.batch_size * args.num_batch,
        "max_seq_length": args.max_seq_length,
        "layer_indices": stats.get("layer_indices", []),
        "top_counts": [[key, count] for key, count in top[: min(100, len(top))]],
        "top_scores": [[key, float(score_sums.get(key, 0.0))] for key, _count in top[: min(100, len(top))]],
    }
    stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return bag_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--priv_data_path", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--rel", default=None)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batch", type=int, default=5)
    parser.add_argument("--layer_start", type=int, default=None)
    parser.add_argument("--layer_end", type=int, default=None)
    parser.add_argument("--layer_indices", default=None)
    parser.add_argument("--threshold_ratio", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument(
        "--ranking_metric",
        default="count",
        choices=["count", "score", "count_score"],
        help="How to rank candidate privacy neurons after per-bag thresholding.",
    )
    parser.add_argument("--ffn_edit_model_kind", default="auto", choices=["auto", "gemma", "qwen", "ministral"])
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--progress_every", type=int, default=20)
    parser.add_argument("--limit_bags", type=int, default=None)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{args.output_prefix}.args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, sort_keys=True, indent=2)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.empty_cache()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "dtype": dtype_from_arg(args.torch_dtype),
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_implementation,
        "local_files_only": args.local_files_only,
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs.pop("dtype", None)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model_cls = AutoModelForCausalLM
    if getattr(config, "model_type", None) == "mistral3":
        if AutoModelForImageTextToText is None:
            raise ImportError("Ministral3 requires AutoModelForImageTextToText; use transformers>=5.0.0rc0.")
        model_cls = AutoModelForImageTextToText
    model = model_cls.from_pretrained(args.model_name_or_path, **model_kwargs)
    tokenizer_vocab_size = len(tokenizer)
    input_embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if input_embeddings is not None and input_embeddings.weight.shape[0] != tokenizer_vocab_size:
        print(
            f"[attr] resizing token embeddings {input_embeddings.weight.shape[0]} -> {tokenizer_vocab_size} "
            "before PEFT adapter load",
            flush=True,
        )
        model.resize_token_embeddings(tokenizer_vocab_size)
    if args.adapter_dir:
        from peft import PeftModel

        print(f"Loading adapter from {args.adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter_dir, local_files_only=args.local_files_only)
        print("Adapter loaded.", flush=True)

    ffn_edit_model_kind = infer_ffn_edit_model_kind(config, args.ffn_edit_model_kind)
    patch_model_for_ffn_edit(model, ffn_edit_model_kind)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)

    modules = find_down_proj_modules(model, ffn_edit_model_kind)
    discovered_layers = [layer for layer, _name, _module in modules]
    intermediate_size = int(text_config_value(config, "intermediate_size", 0) or 0)
    if intermediate_size <= 0:
        intermediate_size = int(getattr(modules[0][2], "in_features", 0) or 0)
    layer_indices = parse_layer_indices(args, discovered_layers)
    print(
        f"[{ffn_edit_model_kind}_attr] layers={len(discovered_layers)} selected={layer_indices} "
        f"intermediate_size={intermediate_size} ig_steps={args.batch_size * args.num_batch}",
        flush=True,
    )

    with open(args.priv_data_path, "r", encoding="utf-8") as handle:
        privacy_bags = json.load(handle)
    if args.limit_bags is not None:
        privacy_bags = privacy_bags[: args.limit_bags]
    validate_privacy_bags(privacy_bags)

    resume_count, counts, score_sums, stats = load_state(output_dir, args.output_prefix)
    if args.limit_bags is not None and resume_count > 0:
        resume_count, counts, score_sums, stats = 0, Counter(), {}, {}
    if args.ranking_metric != "count" and resume_count > 0 and not score_sums:
        print(
            f"[{args.ffn_edit_model_kind}_attr] existing state has no score_sums; "
            f"restart attribution for ranking_metric={args.ranking_metric}",
            flush=True,
        )
        resume_count, counts, score_sums, stats = 0, Counter(), {}, {}
    stats.setdefault("metric_max_sum", 0.0)
    stats.setdefault("metric_count", 0)
    stats.setdefault("filtered_positions_sum", 0)
    stats.setdefault("secret_located", 0)
    stats.setdefault("secret_unlocated", 0)
    stats["layer_indices"] = layer_indices
    stats["ffn_edit_model_kind"] = ffn_edit_model_kind

    device = next(model.parameters()).device
    rel = args.rel or args.output_prefix
    completed = resume_count
    processed_since_save = 0
    total_bags = len(privacy_bags)
    tic = time.perf_counter()
    print(f"[{ffn_edit_model_kind}_attr] start bags={total_bags} resume={resume_count}", flush=True)

    for bag_idx, bag in enumerate(privacy_bags):
        if bag_idx < resume_count:
            continue
        examples, was_legacy_pair = normalize_privacy_bag(bag)
        if was_legacy_pair:
            stats["legacy_pair_bags_normalized"] = int(stats.get("legacy_pair_bags_normalized", 0)) + 1
        if not examples:
            continue

        sum_tensor = torch.zeros((len(layer_indices), intermediate_size), dtype=torch.float32)
        for ex_idx, example in enumerate(examples):
            features, token_info = example2feature(example, args.max_seq_length, tokenizer)
            stats["secret_located" if token_info["secret_located"] else "secret_unlocated"] = int(
                stats.get("secret_located" if token_info["secret_located"] else "secret_unlocated", 0)
            ) + 1
            input_ids = torch.tensor(features["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = torch.tensor(features["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
            real_len = int(attention_mask[0].sum().item())
            input_ids = input_ids[:, :real_len]
            attention_mask = attention_mask[:, :real_len]
            tgt_pos = int(token_info["target_pos"])
            gold_label = int(token_info["gold_obj"])

            for local_layer_idx, tgt_layer in enumerate(layer_indices):
                if local_layer_idx == 0 or tgt_layer == layer_indices[-1] or (tgt_layer + 1) % 8 == 0:
                    print(f"  bag {bag_idx} ex {ex_idx} layer {tgt_layer}/{discovered_layers[-1]}", flush=True)

                ffn_weights, _logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tgt_pos=tgt_pos,
                    tgt_layer=tgt_layer,
                    use_cache=False,
                    return_dict=True,
                )
                if ffn_weights is None:
                    ffn_weights = torch.zeros((1, intermediate_size), device=device)

                scaled_weights, weights_step = scaled_input(ffn_weights, args.batch_size, args.num_batch)
                scaled_weights.requires_grad_(True)
                ig_gold = None
                for batch_idx in range(args.num_batch):
                    batch_weights = scaled_weights[
                        batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size
                    ]
                    cur_batch_size = batch_weights.shape[0]
                    batch_input_ids = input_ids.repeat(cur_batch_size, 1)
                    batch_attention_mask = attention_mask.repeat(cur_batch_size, 1)
                    labels = batch_input_ids.clone()
                    labels[:, tgt_pos] = gold_label
                    _prob, grad = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        tgt_pos=tgt_pos,
                        tgt_layer=tgt_layer,
                        tmp_score=batch_weights,
                        labels=labels,
                        use_cache=False,
                        return_dict=True,
                    )
                    grad = grad.sum(dim=0)
                    ig_gold = grad if ig_gold is None else torch.add(ig_gold, grad)

                sum_tensor[local_layer_idx].add_((ig_gold * weights_step).detach().float().cpu())
                del scaled_weights, weights_step, ig_gold

        metric_max = float(sum_tensor.max().item())
        bag_positions: set[str] = set()
        bag_scores: dict[str, float] = {}
        if metric_max > 0:
            cutoff = metric_max * args.threshold_ratio
            for local_layer_idx, neuron_idx in (sum_tensor >= cutoff).nonzero(as_tuple=False).tolist():
                key = pos_key(layer_indices[local_layer_idx], neuron_idx)
                bag_positions.add(key)
                bag_scores[key] = float(sum_tensor[local_layer_idx, neuron_idx].item())
            stats["metric_max_sum"] = float(stats.get("metric_max_sum", 0.0)) + metric_max
            stats["metric_count"] = int(stats.get("metric_count", 0)) + 1
        counts.update(bag_positions)
        for key, value in bag_scores.items():
            score_sums[key] = float(score_sums.get(key, 0.0) + value)
        stats["filtered_positions_sum"] = int(stats.get("filtered_positions_sum", 0)) + len(bag_positions)

        completed += 1
        stats["completed_bags"] = completed
        processed_since_save += 1

        if args.progress_every > 0 and (completed <= 2 or completed % args.progress_every == 0):
            elapsed = time.perf_counter() - tic
            print(
                f"[{ffn_edit_model_kind}_attr] bags={completed}/{total_bags} unique={len(counts)} "
                f"bag_filtered={len(bag_positions)} elapsed={elapsed:.1f}s",
                flush=True,
            )

        if processed_since_save >= max(1, args.save_every):
            save_state(output_dir, args.output_prefix, completed, counts, score_sums, stats)
            bag_path = write_kn_outputs(output_dir, rel, counts, score_sums, args.top_k, stats, args)
            print(f"[{ffn_edit_model_kind}_attr] checkpoint saved kn={bag_path}", flush=True)
            processed_since_save = 0

    save_state(output_dir, args.output_prefix, completed, counts, score_sums, stats)
    bag_path = write_kn_outputs(output_dir, rel, counts, score_sums, args.top_k, stats, args)
    print(f"[{ffn_edit_model_kind}_attr] wrote {bag_path}", flush=True)
    print(f"[{ffn_edit_model_kind}_attr] done bags={completed} unique={len(counts)} time={time.perf_counter() - tic:.1f}s", flush=True)


if __name__ == "__main__":
    main()
