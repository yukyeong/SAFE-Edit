#!/usr/bin/env python3
"""Download WikiText-103 validation text used by the PPL metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local WikiText-103 validation JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/wikitext-103-raw-v1/validation.jsonl"),
    )
    args = parser.parse_args()

    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row in dataset:
            text = str(row.get("text", ""))
            if not text.strip():
                continue
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
    if written == 0:
        raise SystemExit("WikiText validation split produced no non-empty rows.")
    print(f"wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
