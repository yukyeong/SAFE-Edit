#!/usr/bin/env python
"""Select Attention Heads Edit heads by post-FFN Edit greedy exact/contains leakage on a small LUME/API4-style subset.

Input: API4-prefix JSON dataset, model adapter, FFN Edit KN ranking, and candidate Attention Heads Edit heads.
Output: an exact-safe head config plus hashed screening metrics.
Impact: evaluation-time screening only; no model weights or datasets are modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from tqdm import tqdm

from eval_api4_privacy_reverse_attention_heads_edit import (
    Api4Example,
    apply_ffn_edit_if_requested,
    build_prompt,
    ffn_edit_forward_kwargs,
    evaluate_batch,
    load_model_and_tokenizer,
    model_device,
    summarize,
    trim_input_for_context,
    unwrap_model_for_attention_heads_edit,
)
from attention_heads_privacy_spans import STEERING_SPAN_MODES, select_privacy_steering_span
from attention_heads_edit_lib.attention_heads_edit import AttentionHeadsEdit


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-FFN Edit exact-safe Attention Heads Edit head selection.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidate_heads", type=Path, required=True)
    parser.add_argument("--base_model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--ffn_edit_kn_config", type=Path, required=True)
    parser.add_argument("--ffn_edit_erase_num", type=int, required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config_dir", type=Path, default=REPO_ROOT / "configs/attention_heads_edit/runs")
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--screen_max_samples", type=int, default=0)
    parser.add_argument("--screen_limit_per_type", type=int, default=50)
    parser.add_argument("--candidate_limit", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--scale_position", default="include_down", choices=["include_down", "include", "exclude", "generation"])
    parser.add_argument("--steering_span_mode", default="tail_tokens", choices=STEERING_SPAN_MODES)
    parser.add_argument("--steering_tail_tokens", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_exact_gain", type=float, default=0.0)
    parser.add_argument("--min_score_gain", type=float, default=0.0)
    parser.add_argument("--enforce_per_type_nonworsening", action="store_true")
    parser.add_argument("--per_type_min_samples", type=int, default=10)
    parser.add_argument("--prefer_smallest_exact_prefix", action="store_true")
    parser.add_argument("--first_token_rr_weight", type=float, default=0.1)
    parser.add_argument("--first_token_logprob_weight", type=float, default=0.01)
    parser.add_argument("--prefix_sizes", default="1,3,5,10,20,30,40,60")
    parser.add_argument("--max_context_tokens", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--generation_extra_tokens", type=int, default=8)
    parser.add_argument("--prompt_template", default="{input}")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def load_examples(path: Path, seed: int, max_samples: int, limit_per_type: int) -> list[Api4Example]:
    rows = json.loads(path.read_text(encoding="utf-8"))
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
    rng = random.Random(seed)
    rng.shuffle(examples)
    if limit_per_type > 0:
        counts: dict[str, int] = defaultdict(int)
        kept: list[Api4Example] = []
        for example in examples:
            if counts[example.pii_type] >= limit_per_type:
                continue
            kept.append(example)
            counts[example.pii_type] += 1
        examples = kept
    if max_samples > 0:
        examples = examples[:max_samples]
    return examples


def parse_candidate_heads(path: Path, limit: int) -> list[tuple[int, int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    heads: list[tuple[int, int]] = []

    def add(layer: Any, head: Any) -> None:
        try:
            key = (int(layer), int(head))
        except (TypeError, ValueError):
            return
        if key not in heads:
            heads.append(key)

    if isinstance(raw, dict) and isinstance(raw.get("edit_heads"), list):
        ordered = sorted(
            raw["edit_heads"],
            key=lambda item: (
                float(item.get("rank", math.inf)) if isinstance(item, dict) else math.inf,
                -float(item.get("score", 0.0)) if isinstance(item, dict) else 0.0,
            ),
        )
        for item in ordered:
            if isinstance(item, dict):
                add(item.get("layer"), item.get("head"))
    elif isinstance(raw, dict):
        for layer, values in raw.items():
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, dict):
                    add(value.get("layer", layer), value.get("head"))
                elif isinstance(value, (list, tuple)) and len(value) >= 2:
                    add(value[0], value[1])
                else:
                    add(layer, value)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                add(item.get("layer"), item.get("head"))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                add(item[0], item[1])

    if limit > 0:
        heads = heads[:limit]
    if not heads:
        raise SystemExit(f"No candidate heads parsed from {path}")
    return heads


def config_from_heads(heads: list[tuple[int, int]]) -> dict[str, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for layer, head in heads:
        if head not in grouped[layer]:
            grouped[layer].append(head)
    return {str(layer): heads_for_layer for layer, heads_for_layer in sorted(grouped.items())}


def count_heads(config: dict[str, list[int]]) -> int:
    return sum(len(heads) for heads in config.values())


def per_type_safety(
    baseline_summary: list[dict[str, Any]],
    candidate_summary: list[dict[str, Any]],
    min_samples: int,
) -> dict[str, Any]:
    baseline_by_type = {str(row["pii_type"]): row for row in baseline_summary}
    candidate_by_type = {str(row["pii_type"]): row for row in candidate_summary}
    gains: list[tuple[str, float, float]] = []
    for pii_type, baseline in baseline_by_type.items():
        if int(baseline.get("count", 0) or 0) < min_samples or pii_type not in candidate_by_type:
            continue
        candidate = candidate_by_type[pii_type]
        exact_gain = float(baseline.get("normalized_exact_match_mean", 0.0) or 0.0) - float(
            candidate.get("normalized_exact_match_mean", 0.0) or 0.0
        )
        contains_gain = float(baseline.get("normalized_contains_target_mean", 0.0) or 0.0) - float(
            candidate.get("normalized_contains_target_mean", 0.0) or 0.0
        )
        gains.append((pii_type, exact_gain, contains_gain))

    worst_exact = min(gains, key=lambda item: item[1]) if gains else (None, 0.0, 0.0)
    worst_contains = min(gains, key=lambda item: item[2]) if gains else (None, 0.0, 0.0)
    return {
        "eligible_pii_type_count": len(gains),
        "worst_pii_type_exact": worst_exact[0],
        "worst_pii_type_exact_gain": worst_exact[1],
        "worst_pii_type_contains": worst_contains[0],
        "worst_pii_type_contains_gain": worst_contains[2],
        "per_type_safe": worst_exact[1] >= -1e-12 and worst_contains[2] >= -1e-12,
    }


def row_is_safe(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return not args.enforce_per_type_nonworsening or bool(row.get("per_type_safe", False))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "rank",
        "layer",
        "head",
        "alpha",
        "selected",
        "sample_count",
        "baseline_exact",
        "head_exact",
        "exact_gain",
        "baseline_contains",
        "head_contains",
        "contains_gain",
        "score",
    ]
    fields = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_first_token_signal(
    *,
    model,
    attention_heads_edit_model,
    tokenizer,
    steerer: AttentionHeadsEdit | None,
    examples: list[Api4Example],
    args: argparse.Namespace,
) -> dict[str, float]:
    reciprocal_ranks: list[float] = []
    target_logprobs: list[float] = []
    device = model_device(model)
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start : start + args.batch_size]
        trimmed_inputs = [trim_input_for_context(tokenizer, example.input_text, args) for example in batch]
        prompts = [build_prompt(input_text, args) for input_text in trimmed_inputs]
        target_ids = [
            tokenizer(example.target, add_special_tokens=False)["input_ids"]
            for example in batch
        ]
        if any(not ids for ids in target_ids):
            raise ValueError("Screening example has an empty target token sequence")

        if steerer is None:
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            try:
                model_inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                ).to(device)
            finally:
                tokenizer.padding_side = original_padding_side
            steering_context = torch.inference_mode()
        else:
            model_inputs, offsets_mapping = steerer.inputs_from_batch(
                prompts,
                tokenizer=tokenizer,
                device=device,
            )
            steering_spans = [
                select_privacy_steering_span(
                    input_text,
                    tokenizer,
                    mode=args.steering_span_mode,
                    tail_tokens=args.steering_tail_tokens,
                )
                for input_text in trimmed_inputs
            ]
            steering_context = steerer.apply_steering(
                model=attention_heads_edit_model,
                strings=prompts,
                substrings=steering_spans,
                model_input=model_inputs,
                offsets_mapping=offsets_mapping,
                occurrence=-1,
            )

        with steering_context:
            outputs = model(
                **model_inputs,
                use_cache=False,
                return_dict=True,
                **ffn_edit_forward_kwargs(args),
            )
        next_token_logits = outputs.logits[:, -1].float()
        next_token_logprobs = torch.log_softmax(next_token_logits, dim=-1)
        for batch_index, ids in enumerate(target_ids):
            target_id = int(ids[0])
            target_logit = next_token_logits[batch_index, target_id]
            rank = int((next_token_logits[batch_index] > target_logit).sum().item()) + 1
            reciprocal_ranks.append(1.0 / rank)
            target_logprobs.append(float(next_token_logprobs[batch_index, target_id].item()))

    return {
        "first_token_rr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "first_token_target_logprob": sum(target_logprobs) / len(target_logprobs),
    }


def evaluate_config(
    *,
    model,
    attention_heads_edit_model,
    tokenizer,
    examples: list[Api4Example],
    args: argparse.Namespace,
    head_config: dict[str, list[int]] | None,
) -> dict[str, Any]:
    if head_config is None:
        steerer = None
    else:
        steerer = AttentionHeadsEdit(
            model=attention_heads_edit_model,
            tokenizer=tokenizer,
            head_config=head_config,
            alpha=args.alpha,
            scale_position=args.scale_position,
        )

    rows: list[dict[str, Any]] = []
    eval_args = SimpleNamespace(**vars(args))
    if not hasattr(eval_args, "save_text_predictions"):
        eval_args.save_text_predictions = False
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start : start + args.batch_size]
        rows.extend(evaluate_batch(model, attention_heads_edit_model, tokenizer, steerer, batch, eval_args))
    first_token = evaluate_first_token_signal(
        model=model,
        attention_heads_edit_model=attention_heads_edit_model,
        tokenizer=tokenizer,
        steerer=steerer,
        examples=examples,
        args=args,
    )
    if steerer is not None:
        runtime = steerer.runtime_stats()
        if runtime["attention_heads_edit_hook_call_count"] <= 0 or runtime["attention_heads_edit_edit_call_count"] <= 0:
            raise RuntimeError("Attention Heads Edit screening completed without any attention-mask edits")
    summary_rows, macro = summarize(rows)
    return {"macro": macro, "summary": summary_rows, "rows": rows, "first_token": first_token}


def safe_alpha(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    examples = load_examples(
        args.dataset,
        seed=args.seed,
        max_samples=args.screen_max_samples,
        limit_per_type=args.screen_limit_per_type,
    )
    if not examples:
        raise SystemExit("No screening examples loaded.")

    candidate_heads = parse_candidate_heads(args.candidate_heads, args.candidate_limit)
    run_dir = args.output_dir / args.run_name
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    args.config_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] exact-screen run={args.run_name} examples={len(examples)} "
        f"candidate_heads={len(candidate_heads)} top_k={args.top_k} alpha={args.alpha}",
        flush=True,
    )

    model, tokenizer = load_model_and_tokenizer(args)
    ffn_edit_positions = apply_ffn_edit_if_requested(model, args)
    attention_heads_edit_model = unwrap_model_for_attention_heads_edit(model)

    base = evaluate_config(
        model=model,
        attention_heads_edit_model=attention_heads_edit_model,
        tokenizer=tokenizer,
        examples=examples,
        args=args,
        head_config=None,
    )
    base_exact = float(base["macro"].get("normalized_exact_match_mean", 0.0) or 0.0)
    base_contains = float(base["macro"].get("normalized_contains_target_mean", 0.0) or 0.0)
    base_first_token_rr = float(base["first_token"]["first_token_rr"])
    base_first_token_logprob = float(base["first_token"]["first_token_target_logprob"])
    write_csv(predictions_dir / "ffn_edit_only_screen_details_hashed.csv", base["rows"])
    print(
        f"[INFO] ffn_edit_only_screen exact={base_exact:.6f} contains={base_contains:.6f} "
        f"ffn_edit_positions={len(ffn_edit_positions)}",
        flush=True,
    )

    single_rows: list[dict[str, Any]] = []
    for index, (layer, head) in enumerate(tqdm(candidate_heads, desc="exact-screen-heads"), start=1):
        payload = evaluate_config(
            model=model,
            attention_heads_edit_model=attention_heads_edit_model,
            tokenizer=tokenizer,
            examples=examples,
            args=args,
            head_config={str(layer): [head]},
        )
        head_exact = float(payload["macro"].get("normalized_exact_match_mean", 0.0) or 0.0)
        head_contains = float(payload["macro"].get("normalized_contains_target_mean", 0.0) or 0.0)
        head_first_token_rr = float(payload["first_token"]["first_token_rr"])
        head_first_token_logprob = float(payload["first_token"]["first_token_target_logprob"])
        exact_gain = base_exact - head_exact
        contains_gain = base_contains - head_contains
        first_token_rr_gain = base_first_token_rr - head_first_token_rr
        first_token_logprob_reduction = base_first_token_logprob - head_first_token_logprob
        score = (
            exact_gain
            + 0.25 * contains_gain
            + args.first_token_rr_weight * first_token_rr_gain
            + args.first_token_logprob_weight * first_token_logprob_reduction
        )
        safety = per_type_safety(base["summary"], payload["summary"], args.per_type_min_samples)
        single_rows.append(
            {
                "rank": index,
                "layer": layer,
                "head": head,
                "alpha": args.alpha,
                "sample_count": len(examples),
                "baseline_exact": base_exact,
                "head_exact": head_exact,
                "exact_gain": exact_gain,
                "baseline_contains": base_contains,
                "head_contains": head_contains,
                "contains_gain": contains_gain,
                "baseline_first_token_rr": base_first_token_rr,
                "head_first_token_rr": head_first_token_rr,
                "first_token_rr_gain": first_token_rr_gain,
                "baseline_first_token_target_logprob": base_first_token_logprob,
                "head_first_token_target_logprob": head_first_token_logprob,
                "first_token_logprob_reduction": first_token_logprob_reduction,
                "score": score,
                **safety,
            }
        )
        if args.log_every > 0 and index % args.log_every == 0:
            positives = sum(
                1
                for row in single_rows
                if row["exact_gain"] >= args.min_exact_gain
                and row["score"] > args.min_score_gain
                and row_is_safe(row, args)
            )
            print(f"[INFO] screened={index}/{len(candidate_heads)} positives={positives}", flush=True)
        if torch.cuda.is_available() and index % 10 == 0:
            torch.cuda.empty_cache()

    positives = [
        row
        for row in single_rows
        if float(row["exact_gain"]) >= args.min_exact_gain
        and float(row["score"]) > args.min_score_gain
        and row_is_safe(row, args)
    ]
    positives.sort(
        key=lambda row: (float(row["exact_gain"]), float(row["contains_gain"]), float(row["score"])),
        reverse=True,
    )
    selection_mode = "positive_gain"
    selection_pool = positives
    if not selection_pool:
        neutral = [
            row
            for row in single_rows
            if float(row["exact_gain"]) >= args.min_exact_gain
            and float(row["score"]) >= args.min_score_gain
            and row_is_safe(row, args)
        ]
        neutral.sort(
            key=lambda row: (
                float(row["score"]),
                float(row["exact_gain"]),
                float(row["contains_gain"]),
                -int(row["rank"]),
            ),
            reverse=True,
        )
        if not neutral:
            raise RuntimeError(
                "Every screened Attention Heads Edit head increased Exact/Contains leakage; refusing an unsafe config."
            )
        selection_mode = "neutral_fallback"
        selection_pool = neutral[:1]
    for row in single_rows:
        row["selected"] = False

    requested_prefix_sizes = sorted(
        {
            int(size)
            for size in args.prefix_sizes.split(",")
            if size.strip().isdigit() and int(size) > 0
        }
        | {args.top_k}
    )
    prefix_sizes = sorted(
        {
            min(size, args.top_k, len(selection_pool))
            for size in requested_prefix_sizes
            if min(size, args.top_k, len(selection_pool)) > 0
        }
    )
    prefix_rows: list[dict[str, Any]] = []
    best_prefix: list[dict[str, Any]] = []
    best_prefix_key: tuple[float, ...] | None = None
    best_score = -math.inf
    best_exact_gain = -math.inf
    for size in prefix_sizes:
        selected_rows = selection_pool[:size]
        selected_heads = [(int(row["layer"]), int(row["head"])) for row in selected_rows]
        cfg = config_from_heads(selected_heads)
        payload = evaluate_config(
            model=model,
            attention_heads_edit_model=attention_heads_edit_model,
            tokenizer=tokenizer,
            examples=examples,
            args=args,
            head_config=cfg,
        )
        joint_exact = float(payload["macro"].get("normalized_exact_match_mean", 0.0) or 0.0)
        joint_contains = float(payload["macro"].get("normalized_contains_target_mean", 0.0) or 0.0)
        joint_first_token_rr = float(payload["first_token"]["first_token_rr"])
        joint_first_token_logprob = float(payload["first_token"]["first_token_target_logprob"])
        exact_gain = base_exact - joint_exact
        contains_gain = base_contains - joint_contains
        first_token_rr_gain = base_first_token_rr - joint_first_token_rr
        first_token_logprob_reduction = base_first_token_logprob - joint_first_token_logprob
        score = (
            exact_gain
            + 0.25 * contains_gain
            + args.first_token_rr_weight * first_token_rr_gain
            + args.first_token_logprob_weight * first_token_logprob_reduction
        )
        safety = per_type_safety(base["summary"], payload["summary"], args.per_type_min_samples)
        prefix_row = {
            "prefix_size": len(selected_heads),
            "alpha": args.alpha,
            "head_count": count_heads(cfg),
            "baseline_exact": base_exact,
            "joint_exact": joint_exact,
            "exact_gain": exact_gain,
            "baseline_contains": base_contains,
            "joint_contains": joint_contains,
            "contains_gain": contains_gain,
            "baseline_first_token_rr": base_first_token_rr,
            "joint_first_token_rr": joint_first_token_rr,
            "first_token_rr_gain": first_token_rr_gain,
            "baseline_first_token_target_logprob": base_first_token_logprob,
            "joint_first_token_target_logprob": joint_first_token_logprob,
            "first_token_logprob_reduction": first_token_logprob_reduction,
            "score": score,
            **safety,
        }
        prefix_rows.append(prefix_row)
        eligible = (
            exact_gain >= args.min_exact_gain
            and score >= args.min_score_gain
            and row_is_safe(prefix_row, args)
        )
        prefix_key = (
            (exact_gain, contains_gain, -float(len(selected_heads)), score)
            if args.prefer_smallest_exact_prefix
            else (score, exact_gain, contains_gain, -float(len(selected_heads)))
        )
        if eligible and (best_prefix_key is None or prefix_key > best_prefix_key):
            best_prefix_key = prefix_key
            best_score = score
            best_exact_gain = exact_gain
            best_prefix = selected_rows
        print(
            f"[INFO] prefix={len(selected_heads)} exact={joint_exact:.6f} "
            f"exact_gain={exact_gain:.6f} score={score:.6f}",
            flush=True,
        )

    if not best_prefix:
        raise RuntimeError(
            "No globally and per-type safe Attention Heads Edit head prefix was found; refusing an unsafe config."
        )
    selected_heads = [(int(row["layer"]), int(row["head"])) for row in best_prefix]
    selected_config = config_from_heads(selected_heads)
    selected_head_keys = set(selected_heads)
    for row in single_rows:
        row["selected"] = (int(row["layer"]), int(row["head"])) in selected_head_keys
    write_csv(metrics_dir / "single_head_exact_screen.csv", single_rows)
    write_csv(metrics_dir / "prefix_exact_screen.csv", prefix_rows)

    config_path = args.config_dir / f"{args.run_name}_exactsafe_top{args.top_k}_alpha{safe_alpha(args.alpha)}.json"
    selected_path = args.config_dir / f"{args.run_name}_exactsafe_selected_alpha{safe_alpha(args.alpha)}.json"
    config_path.write_text(json.dumps(selected_config, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_path.write_text(json.dumps(selected_config, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "run_name": args.run_name,
        "dataset": str(args.dataset),
        "candidate_heads": str(args.candidate_heads),
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "ffn_edit_kn_config": str(args.ffn_edit_kn_config),
        "ffn_edit_erase_num": args.ffn_edit_erase_num,
        "ffn_edit_position_count": len(ffn_edit_positions),
        "alpha": args.alpha,
        "scale_position": args.scale_position,
        "steering_span_mode": args.steering_span_mode,
        "steering_tail_tokens": args.steering_tail_tokens,
        "screen_sample_count": len(examples),
        "candidate_head_count": len(candidate_heads),
        "positive_single_head_count": len(positives),
        "selection_mode": selection_mode,
        "requested_top_k": args.top_k,
        "selected_head_count": count_heads(selected_config),
        "selected_config": str(config_path),
        "selected_config_alias": str(selected_path),
        "baseline_exact": base_exact,
        "baseline_contains": base_contains,
        "baseline_first_token_rr": base_first_token_rr,
        "baseline_first_token_target_logprob": base_first_token_logprob,
        "first_token_rr_weight": args.first_token_rr_weight,
        "first_token_logprob_weight": args.first_token_logprob_weight,
        "enforce_per_type_nonworsening": args.enforce_per_type_nonworsening,
        "per_type_min_samples": args.per_type_min_samples,
        "prefer_smallest_exact_prefix": args.prefer_smallest_exact_prefix,
        "best_prefix_exact_gain": best_exact_gain,
        "best_prefix_score": best_score,
    }
    (metrics_dir / "exact_screen_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DONE] " + json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
