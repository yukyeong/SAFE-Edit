#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("all", "train", "val", "test", "forget", "retain")


def load_rows(source_dir: Path, split: str) -> list[dict[str, Any]]:
    path = source_dir / f"sft_true_prefix_no_instruction_{split}.json"
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def make_privacy_bags(rows: list[dict[str, Any]]) -> list[list[list[str]]]:
    return [[[str(row["input"]) + str(row["output"]), str(row["output"])]] for row in rows]


def clean_scenario_row(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(row)
    cleaned["input"] = str(cleaned["input"]).replace("herphone number", "her phone number")
    cleaned["full_text"] = str(cleaned["input"]) + str(cleaned["output"])
    cleaned["char_start"] = len(str(cleaned["input"]))
    cleaned["char_end"] = cleaned["char_start"] + len(str(cleaned["output"]))
    return cleaned


def dataset_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    char_lengths = [len(str(row["input"]) + str(row["output"])) for row in rows]
    return {
        "count": len(rows),
        "pii_counts": dict(sorted(Counter(row.get("pii_type") for row in rows).items())),
        "source_count": len({row.get("source_id") for row in rows}),
        "row_kind_counts": dict(sorted(Counter(row.get("lume_row_kind") for row in rows).items())),
        "fixed_herphone_count": sum("herphone number" in str(row.get("input", "")) for row in rows),
        "char_len_mean": statistics.mean(char_lengths) if char_lengths else 0,
        "char_len_max": max(char_lengths) if char_lengths else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split LUME Task2 API4-style data into Attention Heads Edit/FFN-Edit-friendly scenario and QA subsets."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/lume/api4_prefix_task2"),
        help="Directory produced by prepare_lume_api4_prefix.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/lume/api4_prefix_task2"),
        help="Output directory for scenario/QA SFT JSON and FFN Edit privacy bags.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "purpose": (
            "LUME Task2 datasets separated for Attention Heads Edit/FFN Edit diagnostics. "
            "scenario_clean is recommended for prefix-continuation Attention Heads Edit and FFN Edit attribution; "
            "qa is kept for question-answer leakage diagnostics."
        ),
        "variants": {},
    }

    for split in SPLITS:
        rows = load_rows(args.source_dir, split)
        variants = {
            "scenario": [row for row in rows if row.get("lume_row_kind") == "scenario"],
            "qa": [row for row in rows if row.get("lume_row_kind") == "qa"],
        }
        variants["scenario_clean"] = [clean_scenario_row(row) for row in variants["scenario"]]

        for variant, subset in variants.items():
            stem = f"{variant}_{split}"
            sft_path = args.output_dir / f"sft_true_prefix_no_instruction_{stem}.json"
            privacy_path = args.output_dir / f"privacy_data_lume_task2_{stem}.json"
            write_json(sft_path, subset)
            write_json(privacy_path, make_privacy_bags(subset))
            manifest["variants"][stem] = {
                **dataset_stats(subset),
                "sft_json": str(sft_path),
                "privacy_data_json": str(privacy_path),
            }

    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["variants"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
