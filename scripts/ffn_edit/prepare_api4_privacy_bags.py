#!/usr/bin/env python3
"""Convert API4-style prefix-completion rows into FFN Edit privacy bags."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FFN Edit nested privacy bags from API4-style JSON.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_per_type", type=int, default=0, help="0 keeps every valid row.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"Expected a non-empty JSON list: {args.dataset}")

    counts: Counter[str] = Counter()
    bags: list[list[list[str]]] = []
    skipped = 0
    for row in rows:
        if not isinstance(row, dict):
            skipped += 1
            continue
        prefix = str(row.get("input", ""))
        secret = str(row.get("output", ""))
        pii_type = str(row.get("pii_type", "UNKNOWN"))
        if not prefix or not secret:
            skipped += 1
            continue
        if args.max_per_type > 0 and counts[pii_type] >= args.max_per_type:
            continue
        bags.append([[prefix + secret, secret]])
        counts[pii_type] += 1

    if not bags:
        raise SystemExit("No valid privacy bags were produced.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bags, ensure_ascii=False, indent=2), encoding="utf-8")
    summary: dict[str, Any] = {
        "dataset": str(args.dataset),
        "output": str(args.output),
        "bag_count": len(bags),
        "skipped_rows": skipped,
        "pii_type_counts": dict(sorted(counts.items())),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
