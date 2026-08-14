#!/usr/bin/env python
"""Validate API4-style prefix datasets used by Attention Heads Edit and FFN Edit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prefix-to-secret JSON and optional FFN Edit privacy bags.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--privacy_data", type=Path, default=None)
    parser.add_argument("--report_json", type=Path, default=None)
    parser.add_argument("--expected_pii_types", default="")
    parser.add_argument("--require_privacy_alignment", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    rows = load_json(args.dataset)
    if not isinstance(rows, list):
        raise SystemExit(f"dataset must be a JSON list: {args.dataset}")
    expected = {item.strip().upper() for item in args.expected_pii_types.split(",") if item.strip()}
    counts: Counter[str] = Counter()
    duplicate_keys: set[tuple[str, str, str]] = set()
    seen: set[tuple[str, str, str]] = set()
    errors: list[str] = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {index} is not an object")
            continue
        input_text = str(row.get("input", ""))
        output_text = str(row.get("output", ""))
        pii_type = str(row.get("pii_type", "UNKNOWN")).upper()
        if not input_text:
            errors.append(f"row {index} has empty input")
        if not output_text:
            errors.append(f"row {index} has empty output")
        if expected and pii_type not in expected:
            errors.append(f"row {index} has unexpected pii_type={pii_type}")
        counts[pii_type] += 1
        key = (input_text, output_text, pii_type)
        if key in seen:
            duplicate_keys.add(key)
        seen.add(key)

    privacy_count = None
    if args.privacy_data is not None:
        privacy_rows = load_json(args.privacy_data)
        if not isinstance(privacy_rows, list):
            errors.append(f"privacy_data must be a JSON list: {args.privacy_data}")
        else:
            privacy_count = len(privacy_rows)
            if len(privacy_rows) != len(rows):
                errors.append(f"dataset/privacy length mismatch: {len(rows)} != {len(privacy_rows)}")
            if args.require_privacy_alignment and len(privacy_rows) == len(rows):
                for index, (row, bag) in enumerate(zip(rows, privacy_rows)):
                    if not (isinstance(bag, list) and len(bag) == 1 and isinstance(bag[0], list) and len(bag[0]) >= 2):
                        errors.append(f"privacy bag {index} has unexpected shape")
                        continue
                    full_text, secret = str(bag[0][0]), str(bag[0][1])
                    expected_full_text = str(row.get("input", "")) + str(row.get("output", ""))
                    if full_text != expected_full_text:
                        errors.append(f"privacy bag {index} full_text mismatch")
                    if secret != str(row.get("output", "")):
                        errors.append(f"privacy bag {index} secret mismatch")

    report = {
        "dataset": str(args.dataset),
        "privacy_data": str(args.privacy_data) if args.privacy_data else None,
        "row_count": len(rows),
        "privacy_bag_count": privacy_count,
        "pii_counts": dict(sorted(counts.items())),
        "duplicate_count": len(duplicate_keys),
        "valid": not errors,
        "errors": errors[:50],
        "error_count": len(errors),
    }
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
