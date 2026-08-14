#!/usr/bin/env python3
"""Deterministically upsample under-memorized LUME PII types for Qwen/Ministral SFT."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_WEIGHTS = {
    "ADDRESS": 3,
    "DOB": 3,
    "EMAIL": 1,
    "PHONENUMBER": 3,
    "SSN": 3,
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise SystemExit(f"Expected a non-empty JSON list of objects: {path}")
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic LUME hard-balanced train split.")
    parser.add_argument("--source_dir", type=Path, default=Path("data/lume/api4_prefix_task2"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/lume/api4_prefix_task2_hard_balanced"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = args.source_dir / "sft_true_prefix_no_instruction_train.json"
    val_path = args.source_dir / "sft_true_prefix_no_instruction_val.json"
    train_rows = load_rows(train_path)
    val_rows = load_rows(val_path)

    unknown_types = sorted({str(row.get("pii_type")) for row in train_rows} - set(DEFAULT_WEIGHTS))
    if unknown_types:
        raise SystemExit(f"No upsampling weight defined for PII types: {unknown_types}")

    balanced = []
    for row in train_rows:
        weight = DEFAULT_WEIGHTS[str(row.get("pii_type"))]
        for copy_index in range(weight):
            copy = dict(row)
            copy["hard_balance_weight"] = weight
            copy["hard_balance_copy_index"] = copy_index
            balanced.append(copy)
    random.Random(args.seed).shuffle(balanced)

    train_output = args.output_dir / "sft_true_prefix_no_instruction_train.json"
    val_output = args.output_dir / "sft_true_prefix_no_instruction_val.json"
    write_json(train_output, balanced)
    write_json(val_output, val_rows)
    manifest = {
        "source_train": str(train_path),
        "source_val": str(val_path),
        "train_output": str(train_output),
        "val_output": str(val_output),
        "seed": args.seed,
        "weights": DEFAULT_WEIGHTS,
        "source_train_count": len(train_rows),
        "derived_train_count": len(balanced),
        "val_count": len(val_rows),
        "source_train_pii_counts": dict(sorted(Counter(str(row.get("pii_type")) for row in train_rows).items())),
        "derived_train_pii_counts": dict(sorted(Counter(str(row.get("pii_type")) for row in balanced).items())),
        "val_pii_counts": dict(sorted(Counter(str(row.get("pii_type")) for row in val_rows).items())),
        "rationale": "Upsample ADDRESS, DOB, PHONENUMBER, and SSN threefold while keeping EMAIL at onefold.",
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
