#!/usr/bin/env python
"""Select stable Attention Heads Edit heads by causal first-token suppression and two-fold Exact validation.

Input: API4-prefix rows, model-local candidate heads, adapter, and an ordered FFN Edit ranking.
Output: an Attention Heads Edit head config plus hashed two-fold screening metrics.
Impact: evaluation-time forward passes only; model weights and source datasets are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from eval_api4_privacy_reverse_attention_heads_edit import (
    Api4Example,
    apply_ffn_edit_if_requested,
    load_model_and_tokenizer,
    unwrap_model_for_attention_heads_edit,
)
from attention_heads_privacy_spans import STEERING_SPAN_MODES
from attention_heads_edit_lib.attention_heads_edit import AttentionHeadsEdit
from select_attention_heads_by_exact_after_ffn_edit import (
    config_from_heads,
    count_heads,
    evaluate_config,
    evaluate_first_token_signal,
    load_examples,
    parse_candidate_heads,
    per_type_safety,
    safe_alpha,
    write_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-fold causal Attention Heads Edit selection after FFN Edit.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidate_heads", type=Path, required=True)
    parser.add_argument("--base_model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--ffn_edit_kn_config", type=Path, required=True)
    parser.add_argument("--ffn_edit_erase_num", type=int, required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config_dir", type=Path, default=REPO_ROOT / "configs/attention_heads_edit/runs")
    parser.add_argument("--candidate_limit", type=int, default=192)
    parser.add_argument("--causal_preselect_limit", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--screen_limit_per_type", type=int, default=20)
    parser.add_argument("--per_type_min_samples", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--scale_position", default="include_down", choices=["include_down", "include", "exclude", "generation"])
    parser.add_argument("--steering_span_mode", default="adaptive_tail", choices=STEERING_SPAN_MODES)
    parser.add_argument("--steering_tail_tokens", type=int, default=16)
    parser.add_argument("--first_token_rr_weight", type=float, default=0.1)
    parser.add_argument("--first_token_logprob_weight", type=float, default=0.01)
    parser.add_argument("--prefix_sizes", default="1,2,3")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_context_tokens", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--generation_extra_tokens", type=int, default=8)
    parser.add_argument("--prompt_template", default="{input}")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--log_every", type=int, default=10)
    parser.set_defaults(save_text_predictions=False)
    return parser.parse_args()


def stable_fold(source_id: Any, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).digest()
    return digest[0] % 2


def cap_by_type(examples: list[Api4Example], limit: int, seed: int) -> list[Api4Example]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    if limit <= 0:
        return shuffled
    counts: Counter[str] = Counter()
    kept: list[Api4Example] = []
    for example in shuffled:
        if counts[example.pii_type] >= limit:
            continue
        kept.append(example)
        counts[example.pii_type] += 1
    return kept


def split_source_disjoint_folds(
    examples: list[Api4Example],
    *,
    limit_per_type: int,
    seed: int,
) -> tuple[list[Api4Example], list[Api4Example]]:
    folds: list[list[Api4Example]] = [[], []]
    for example in examples:
        folds[stable_fold(example.source_id, seed)].append(example)
    fold_a = cap_by_type(folds[0], limit_per_type, seed + 1)
    fold_b = cap_by_type(folds[1], limit_per_type, seed + 2)
    if not fold_a or not fold_b:
        raise RuntimeError("Unable to create two non-empty source-disjoint screening folds")
    return fold_a, fold_b


def pii_counts(examples: list[Api4Example]) -> dict[str, int]:
    return dict(sorted(Counter(example.pii_type for example in examples).items()))


def signal_gain(baseline: dict[str, float], candidate: dict[str, float], args: argparse.Namespace) -> dict[str, float]:
    rr_gain = float(baseline["first_token_rr"]) - float(candidate["first_token_rr"])
    logprob_reduction = float(baseline["first_token_target_logprob"]) - float(
        candidate["first_token_target_logprob"]
    )
    return {
        "rr_gain": rr_gain,
        "logprob_reduction": logprob_reduction,
        "signal_score": args.first_token_rr_weight * rr_gain
        + args.first_token_logprob_weight * logprob_reduction,
    }


@torch.no_grad()
def evaluate_signal_config(
    *,
    model,
    attention_heads_edit_model,
    tokenizer,
    examples: list[Api4Example],
    args: argparse.Namespace,
    head_config: dict[str, list[int]] | None,
) -> dict[str, float]:
    steerer = None
    if head_config is not None:
        steerer = AttentionHeadsEdit(
            model=attention_heads_edit_model,
            tokenizer=tokenizer,
            head_config=head_config,
            alpha=args.alpha,
            scale_position=args.scale_position,
        )
    result = evaluate_first_token_signal(
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
            raise RuntimeError("Causal Attention Heads Edit screening observed no attention-mask edits")
    return result


def exact_metrics(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    args: argparse.Namespace,
    prefix: str,
) -> dict[str, Any]:
    baseline_exact = float(baseline["macro"].get("normalized_exact_match_mean", 0.0) or 0.0)
    candidate_exact = float(candidate["macro"].get("normalized_exact_match_mean", 0.0) or 0.0)
    baseline_contains = float(baseline["macro"].get("normalized_contains_target_mean", 0.0) or 0.0)
    candidate_contains = float(candidate["macro"].get("normalized_contains_target_mean", 0.0) or 0.0)
    safety = per_type_safety(baseline["summary"], candidate["summary"], args.per_type_min_samples)
    signal = signal_gain(baseline["first_token"], candidate["first_token"], args)
    return {
        f"{prefix}_baseline_exact": baseline_exact,
        f"{prefix}_exact": candidate_exact,
        f"{prefix}_exact_gain": baseline_exact - candidate_exact,
        f"{prefix}_baseline_contains": baseline_contains,
        f"{prefix}_contains": candidate_contains,
        f"{prefix}_contains_gain": baseline_contains - candidate_contains,
        f"{prefix}_rr_gain": signal["rr_gain"],
        f"{prefix}_logprob_reduction": signal["logprob_reduction"],
        f"{prefix}_signal_score": signal["signal_score"],
        f"{prefix}_per_type_safe": safety["per_type_safe"],
        f"{prefix}_eligible_pii_type_count": safety["eligible_pii_type_count"],
        f"{prefix}_worst_pii_type_exact": safety["worst_pii_type_exact"],
        f"{prefix}_worst_pii_type_exact_gain": safety["worst_pii_type_exact_gain"],
        f"{prefix}_worst_pii_type_contains": safety["worst_pii_type_contains"],
        f"{prefix}_worst_pii_type_contains_gain": safety["worst_pii_type_contains_gain"],
    }


def stable_exact_row(row: dict[str, Any]) -> bool:
    nonworsening = (
        float(row["fold_a_exact_gain"]) >= -1e-12
        and float(row["fold_b_exact_gain"]) >= -1e-12
        and float(row["fold_a_contains_gain"]) >= -1e-12
        and float(row["fold_b_contains_gain"]) >= -1e-12
        and bool(row["fold_a_per_type_safe"])
        and bool(row["fold_b_per_type_safe"])
    )
    discrete_gain = (
        float(row["fold_a_exact_gain"])
        + float(row["fold_b_exact_gain"])
        + float(row["fold_a_contains_gain"])
        + float(row["fold_b_contains_gain"])
        > 1e-12
    )
    stable_signal = (
        float(row["fold_a_signal_score"]) > 0
        and float(row["fold_b_signal_score"]) > 0
    )
    return nonworsening and (discrete_gain or stable_signal)


def privacy_key(row: dict[str, Any], head_count: int) -> tuple[float, ...]:
    exact_a = float(row["fold_a_exact_gain"])
    exact_b = float(row["fold_b_exact_gain"])
    contains_a = float(row["fold_a_contains_gain"])
    contains_b = float(row["fold_b_contains_gain"])
    signal_a = float(row["fold_a_signal_score"])
    signal_b = float(row["fold_b_signal_score"])
    return (
        min(exact_a, exact_b),
        exact_a + exact_b,
        min(contains_a, contains_b),
        contains_a + contains_b,
        -float(head_count),
        min(signal_a, signal_b),
        signal_a + signal_b,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    all_examples = load_examples(args.dataset, seed=args.seed, max_samples=0, limit_per_type=0)
    fold_a, fold_b = split_source_disjoint_folds(
        all_examples,
        limit_per_type=args.screen_limit_per_type,
        seed=args.seed,
    )
    candidate_heads = parse_candidate_heads(args.candidate_heads, args.candidate_limit)
    run_dir = args.output_dir / args.run_name
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    args.config_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] run={args.run_name} candidates={len(candidate_heads)} "
        f"fold_a={len(fold_a)} fold_b={len(fold_b)} alpha={args.alpha}",
        flush=True,
    )
    print(f"[INFO] fold_a_pii={pii_counts(fold_a)}", flush=True)
    print(f"[INFO] fold_b_pii={pii_counts(fold_b)}", flush=True)

    model, tokenizer = load_model_and_tokenizer(args)
    ffn_edit_positions = apply_ffn_edit_if_requested(model, args)
    attention_heads_edit_model = unwrap_model_for_attention_heads_edit(model)
    baseline_a = evaluate_config(
        model=model,
        attention_heads_edit_model=attention_heads_edit_model,
        tokenizer=tokenizer,
        examples=fold_a,
        args=args,
        head_config=None,
    )
    baseline_b = evaluate_config(
        model=model,
        attention_heads_edit_model=attention_heads_edit_model,
        tokenizer=tokenizer,
        examples=fold_b,
        args=args,
        head_config=None,
    )
    write_csv(predictions_dir / "fold_a_ffn_edit_only_hashed.csv", baseline_a["rows"])
    write_csv(predictions_dir / "fold_b_ffn_edit_only_hashed.csv", baseline_b["rows"])

    signal_rows: list[dict[str, Any]] = []
    for index, (layer, head) in enumerate(tqdm(candidate_heads, desc="causal-first-token"), start=1):
        config = {str(layer): [head]}
        signal_a = evaluate_signal_config(
            model=model,
            attention_heads_edit_model=attention_heads_edit_model,
            tokenizer=tokenizer,
            examples=fold_a,
            args=args,
            head_config=config,
        )
        signal_b = evaluate_signal_config(
            model=model,
            attention_heads_edit_model=attention_heads_edit_model,
            tokenizer=tokenizer,
            examples=fold_b,
            args=args,
            head_config=config,
        )
        gain_a = signal_gain(baseline_a["first_token"], signal_a, args)
        gain_b = signal_gain(baseline_b["first_token"], signal_b, args)
        signal_rows.append(
            {
                "rank": index,
                "layer": layer,
                "head": head,
                "alpha": args.alpha,
                "fold_a_rr_gain": gain_a["rr_gain"],
                "fold_a_logprob_reduction": gain_a["logprob_reduction"],
                "fold_a_signal_score": gain_a["signal_score"],
                "fold_b_rr_gain": gain_b["rr_gain"],
                "fold_b_logprob_reduction": gain_b["logprob_reduction"],
                "fold_b_signal_score": gain_b["signal_score"],
                "worst_fold_signal": min(gain_a["signal_score"], gain_b["signal_score"]),
                "mean_fold_signal": (gain_a["signal_score"] + gain_b["signal_score"]) / 2,
            }
        )
        if args.log_every > 0 and index % args.log_every == 0:
            print(f"[INFO] causal_screened={index}/{len(candidate_heads)}", flush=True)
        if torch.cuda.is_available() and index % 10 == 0:
            torch.cuda.empty_cache()

    signal_rows.sort(
        key=lambda row: (float(row["worst_fold_signal"]), float(row["mean_fold_signal"])),
        reverse=True,
    )
    for rank, row in enumerate(signal_rows, start=1):
        row["causal_rank"] = rank
    stable_signal = [
        row
        for row in signal_rows
        if float(row["fold_a_signal_score"]) > 0 and float(row["fold_b_signal_score"]) > 0
    ]
    causal_pool = stable_signal + [row for row in signal_rows if row not in stable_signal]
    shortlist = causal_pool[: min(args.causal_preselect_limit, len(causal_pool))]
    write_csv(metrics_dir / "causal_first_token_screen.csv", signal_rows)

    exact_rows: list[dict[str, Any]] = []
    for index, signal_row in enumerate(tqdm(shortlist, desc="causal-exact-cv"), start=1):
        layer, head = int(signal_row["layer"]), int(signal_row["head"])
        config = {str(layer): [head]}
        payload_a = evaluate_config(
            model=model,
            attention_heads_edit_model=attention_heads_edit_model,
            tokenizer=tokenizer,
            examples=fold_a,
            args=args,
            head_config=config,
        )
        payload_b = evaluate_config(
            model=model,
            attention_heads_edit_model=attention_heads_edit_model,
            tokenizer=tokenizer,
            examples=fold_b,
            args=args,
            head_config=config,
        )
        row = {
            "causal_rank": signal_row["causal_rank"],
            "layer": layer,
            "head": head,
            "alpha": args.alpha,
            **exact_metrics(baseline_a, payload_a, args, "fold_a"),
            **exact_metrics(baseline_b, payload_b, args, "fold_b"),
        }
        row["stable_nonworsening"] = stable_exact_row(row)
        row["privacy_key"] = repr(privacy_key(row, 1))
        exact_rows.append(row)
        if args.log_every > 0 and index % args.log_every == 0:
            print(f"[INFO] exact_cv_screened={index}/{len(shortlist)}", flush=True)
        if torch.cuda.is_available() and index % 5 == 0:
            torch.cuda.empty_cache()

    stable_rows = [row for row in exact_rows if stable_exact_row(row)]
    for row in exact_rows:
        row["selected_pool"] = row in stable_rows
    write_csv(metrics_dir / "causal_exact_cv_screen.csv", exact_rows)
    if not stable_rows:
        raise RuntimeError("No Attention Heads Edit head remained non-worsening on both source-disjoint folds")
    stable_rows.sort(key=lambda row: privacy_key(row, 1), reverse=True)

    prefix_sizes = sorted(
        {
            min(int(value), args.top_k, len(stable_rows))
            for value in args.prefix_sizes.split(",")
            if value.strip().isdigit() and int(value) > 0
        }
    )
    prefix_rows: list[dict[str, Any]] = []
    best_config: dict[str, list[int]] | None = None
    best_key: tuple[float, ...] | None = None
    for size in prefix_sizes:
        selected = stable_rows[:size]
        heads = [(int(row["layer"]), int(row["head"])) for row in selected]
        config = config_from_heads(heads)
        payload_a = evaluate_config(
            model=model,
            attention_heads_edit_model=attention_heads_edit_model,
            tokenizer=tokenizer,
            examples=fold_a,
            args=args,
            head_config=config,
        )
        payload_b = evaluate_config(
            model=model,
            attention_heads_edit_model=attention_heads_edit_model,
            tokenizer=tokenizer,
            examples=fold_b,
            args=args,
            head_config=config,
        )
        row = {
            "prefix_size": size,
            "head_count": count_heads(config),
            "heads": ";".join(f"L{layer}H{head}" for layer, head in heads),
            "alpha": args.alpha,
            **exact_metrics(baseline_a, payload_a, args, "fold_a"),
            **exact_metrics(baseline_b, payload_b, args, "fold_b"),
        }
        row["stable_nonworsening"] = stable_exact_row(row)
        prefix_rows.append(row)
        if stable_exact_row(row):
            key = privacy_key(row, count_heads(config))
            if best_key is None or key > best_key:
                best_key = key
                best_config = config
        print(
            f"[INFO] prefix={size} fold_a_gain={row['fold_a_exact_gain']:.6f} "
            f"fold_b_gain={row['fold_b_exact_gain']:.6f} stable={row['stable_nonworsening']}",
            flush=True,
        )

    if not best_config:
        raise RuntimeError("No stable Attention Heads Edit prefix survived two-fold Exact/Contains validation")
    write_csv(metrics_dir / "prefix_causal_cv_screen.csv", prefix_rows)
    config_path = args.config_dir / (
        f"{args.run_name}_causalcv_top{args.top_k}_alpha{safe_alpha(args.alpha)}.json"
    )
    config_path.write_text(json.dumps(best_config, ensure_ascii=False, indent=2), encoding="utf-8")
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
        "steering_span_mode": args.steering_span_mode,
        "steering_tail_tokens": args.steering_tail_tokens,
        "candidate_head_count": len(candidate_heads),
        "stable_signal_head_count": len(stable_signal),
        "causal_preselect_count": len(shortlist),
        "stable_exact_head_count": len(stable_rows),
        "fold_a_count": len(fold_a),
        "fold_b_count": len(fold_b),
        "fold_a_pii_counts": pii_counts(fold_a),
        "fold_b_pii_counts": pii_counts(fold_b),
        "selected_head_count": count_heads(best_config),
        "selected_config": str(config_path),
        "best_key": best_key,
        "impact_scope": "FFN_Edit16 unchanged; Attention Heads Edit evaluation-time head selection only.",
    }
    (metrics_dir / "causal_cv_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[DONE] " + json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
