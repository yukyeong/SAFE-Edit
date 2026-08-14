#!/usr/bin/env python
"""Summarize API4 privacy suite runs into paper-style tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize API4 privacy metrics.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    preferred = ["method", "pii_type"]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def exact_by_type(exact_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["pii_type"]: row for row in exact_payload.get("summary", [])}


def mrr_by_type(core_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["pii_type"]: row for row in core_payload.get("mrr", {}).get("by_type", [])}


def attack_by_type(core_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["pii_type"]: row for row in core_payload.get("attack", {}).get("by_type", [])}


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def display_method(method: str) -> str:
    if method == "Base":
        return "Base+LoRA"
    return method


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    main_rows: list[dict[str, Any]] = []
    pii_rows: list[dict[str, Any]] = []
    for method in manifest["methods"]:
        method_name = display_method(method["method"])
        exact_payload = load_json(Path(method["exact_metrics"]))
        core_payload = load_json(Path(method["core_metrics"]))
        exact_macro = exact_payload["macro"]
        core_mrr = core_payload.get("mrr", {}).get("macro", {})
        core_attack = core_payload.get("attack", {}).get("macro", {})
        core_ppl = core_payload.get("ppl", {})
        main_rows.append(
            {
                "method": method_name,
                "exact_leakage_rate": exact_macro.get("normalized_exact_match_mean"),
                "contains_leakage_rate": exact_macro.get("normalized_contains_target_mean"),
                "macro_mrr": core_mrr.get("mrr_macro"),
                "attack_asr_at_k": core_attack.get("attack_asr_at_k_macro"),
                "attack_k": core_attack.get("attack_samples"),
                "retain_ppl": core_ppl.get("retain_ppl"),
            }
        )

        exact_types = exact_by_type(exact_payload)
        mrr_types = mrr_by_type(core_payload)
        attack_types = attack_by_type(core_payload)
        pii_types = sorted(set(exact_types) | set(mrr_types) | set(attack_types))
        for pii_type in pii_types:
            exact_row = exact_types.get(pii_type, {})
            mrr_row = mrr_types.get(pii_type, {})
            attack_row = attack_types.get(pii_type, {})
            pii_rows.append(
                {
                    "method": method_name,
                    "pii_type": pii_type,
                    "count": exact_row.get("count"),
                    "exact_leakage_rate": exact_row.get("normalized_exact_match_mean"),
                    "contains_leakage_rate": exact_row.get("normalized_contains_target_mean"),
                    "mrr": mrr_row.get("mrr"),
                    "mrr_token_count": mrr_row.get("token_count"),
                    "attack_asr_at_k": attack_row.get("asr_at_k"),
                }
            )

    write_csv(args.output_dir / "paper_main_table.csv", main_rows)
    write_csv(args.output_dir / "paper_by_pii_type.csv", pii_rows)
    payload = {
        "manifest": str(args.manifest),
        "main_table": main_rows,
        "by_pii_type": pii_rows,
    }
    with (args.output_dir / "paper_tables.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    md = [
        "# API4 Privacy Suite Results",
        "",
        "## Main Table",
        "",
        markdown_table(main_rows),
        "",
        "## Notes",
        "",
        "- Exact/Contains use normalized full-dataset greedy generation leakage rates.",
        "- Macro MRR is macro-averaged over PII types after token-level reciprocal-rank averaging within each type.",
        "- Attack ASR@K is grouped by attack prompt; a group succeeds if any of K samples contains the target PII.",
        "- Retain PPL is computed on WikiText validation blocks.",
    ]
    (args.output_dir / "paper_tables.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"main_table": main_rows, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
