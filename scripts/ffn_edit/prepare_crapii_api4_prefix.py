#!/usr/bin/env python3
"""Prepare CRAPII Kaggle BIO-token data as API4-style prefix-completion JSON.

Input:  data/crapii/tidied-pii-detection-kaggle-7k/train.json
Output:
  data/crapii/api4_prefix/sft_true_prefix_no_instruction_{train,val,heldout,all}.json
  data/crapii/api4_prefix/privacy_data_crapii_{train,val,heldout,all}.json
Impact: writes derived local training/evaluation data; source data is not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CRAPII BIO labels into prefix-to-PII completion rows.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/crapii/tidied-pii-detection-kaggle-7k/train.json"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("data/crapii/api4_prefix"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_doc_ratio", type=float, default=0.85)
    parser.add_argument("--val_doc_ratio", type=float, default=0.05)
    parser.add_argument(
        "--drop_target_in_prefix",
        action="store_true",
        help="Drop entities whose target already appears verbatim in the preceding document prefix.",
    )
    return parser.parse_args()


def join_tokens(tokens: list[Any], trailing_whitespace: list[bool], start: int, end: int, *, include_last_ws: bool) -> str:
    parts: list[str] = []
    for idx in range(start, end):
        parts.append(str(tokens[idx]))
        if trailing_whitespace[idx] and (include_last_ws or idx < end - 1):
            parts.append(" ")
    return "".join(parts)


def label_type(label: str) -> str:
    return label.split("-", 1)[1] if "-" in label else label


def build_examples(
    rows: list[dict[str, Any]],
    source: Path,
    *,
    drop_target_in_prefix: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    examples: list[dict[str, Any]] = []
    skipped = {
        "length_mismatch_docs": 0,
        "empty_prefix_entities": 0,
        "empty_target_entities": 0,
        "target_already_in_prefix": 0,
    }
    source_hash = hashlib.sha256(str(source).encode("utf-8")).hexdigest()

    for row in rows:
        tokens = row.get("tokens") or []
        labels = row.get("labels") or []
        trailing_whitespace = row.get("trailing_whitespace") or [False] * len(tokens)
        if not (len(tokens) == len(labels) == len(trailing_whitespace)):
            skipped["length_mismatch_docs"] += 1
            continue

        entity_index = 0
        token_index = 0
        while token_index < len(labels):
            label = labels[token_index]
            if not isinstance(label, str) or not label.startswith("B-"):
                token_index += 1
                continue

            pii_type = label_type(label)
            end_index = token_index + 1
            while end_index < len(labels) and labels[end_index] == f"I-{pii_type}":
                end_index += 1

            prefix = join_tokens(tokens, trailing_whitespace, 0, token_index, include_last_ws=True)
            target = join_tokens(tokens, trailing_whitespace, token_index, end_index, include_last_ws=False).strip()
            if not prefix:
                skipped["empty_prefix_entities"] += 1
            elif not target:
                skipped["empty_target_entities"] += 1
            elif drop_target_in_prefix and target in prefix:
                skipped["target_already_in_prefix"] += 1
            else:
                examples.append(
                    {
                        "input": prefix,
                        "output": target,
                        "pii_type": pii_type,
                        "source_id": row.get("document"),
                        "entity_index": entity_index,
                        "dataset": "crapii_tidied_pii_detection_kaggle_7k",
                        "source_file_sha256": source_hash,
                    }
                )
                entity_index += 1
            token_index = end_index

    return examples, skipped


def split_by_document(
    examples: list[dict[str, Any]],
    *,
    seed: int,
    train_doc_ratio: float,
    val_doc_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    docs = sorted({example["source_id"] for example in examples}, key=lambda value: str(value))
    rng = random.Random(seed)
    rng.shuffle(docs)

    train_cut = int(len(docs) * train_doc_ratio)
    val_cut = int(len(docs) * (train_doc_ratio + val_doc_ratio))
    train_docs = set(docs[:train_cut])
    val_docs = set(docs[train_cut:val_cut])

    train = [example for example in examples if example["source_id"] in train_docs]
    val = [example for example in examples if example["source_id"] in val_docs]
    heldout = [example for example in examples if example["source_id"] not in train_docs and example["source_id"] not in val_docs]
    split_meta = {
        "doc_count_with_entities": len(docs),
        "train_doc_count": len(train_docs),
        "val_doc_count": len(val_docs),
        "heldout_doc_count": len(set(docs) - train_docs - val_docs),
        "train_val_doc_overlap": len(train_docs & val_docs),
    }
    return train, val, heldout, split_meta


def summarize(
    *,
    source: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    heldout: list[dict[str, Any]],
    skipped: dict[str, int],
    split_meta: dict[str, int],
) -> dict[str, Any]:
    output_lengths = [len(example["output"]) for example in examples]
    input_lengths = [len(example["input"]) for example in examples]
    return {
        "source": str(source),
        "output_dir": str(output_dir),
        "raw_docs": len(rows),
        **skipped,
        **split_meta,
        "entity_examples_all": len(examples),
        "entity_examples_train": len(train),
        "entity_examples_val": len(val),
        "entity_examples_heldout": len(heldout),
        "pii_type_counts_all": dict(Counter(example["pii_type"] for example in examples)),
        "pii_type_counts_train": dict(Counter(example["pii_type"] for example in train)),
        "pii_type_counts_val": dict(Counter(example["pii_type"] for example in val)),
        "output_char_len": {
            "mean": statistics.mean(output_lengths) if output_lengths else 0,
            "median": statistics.median(output_lengths) if output_lengths else 0,
            "max": max(output_lengths, default=0),
        },
        "input_char_len": {
            "mean": statistics.mean(input_lengths) if input_lengths else 0,
            "median": statistics.median(input_lengths) if input_lengths else 0,
            "max": max(input_lengths, default=0),
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_privacy_bags(rows: list[dict[str, Any]]) -> list[list[list[str]]]:
    return [[[str(row["input"]) + str(row["output"]), str(row["output"])]] for row in rows]


def main() -> None:
    args = parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"Expected non-empty JSON list: {args.input}")

    examples, skipped = build_examples(
        rows,
        args.input,
        drop_target_in_prefix=args.drop_target_in_prefix,
    )
    if not examples:
        raise SystemExit("No prefix-completion examples produced.")
    train, val, heldout, split_meta = split_by_document(
        examples,
        seed=args.seed,
        train_doc_ratio=args.train_doc_ratio,
        val_doc_ratio=args.val_doc_ratio,
    )
    if not train or not val:
        raise SystemExit(f"Invalid split: train={len(train)} val={len(val)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "sft_true_prefix_no_instruction_all.json", examples)
    write_json(args.output_dir / "sft_true_prefix_no_instruction_train.json", train)
    write_json(args.output_dir / "sft_true_prefix_no_instruction_val.json", val)
    write_json(args.output_dir / "sft_true_prefix_no_instruction_heldout.json", heldout)
    write_json(args.output_dir / "privacy_data_crapii_all.json", make_privacy_bags(examples))
    write_json(args.output_dir / "privacy_data_crapii_train.json", make_privacy_bags(train))
    write_json(args.output_dir / "privacy_data_crapii_val.json", make_privacy_bags(val))
    write_json(args.output_dir / "privacy_data_crapii_heldout.json", make_privacy_bags(heldout))

    summary = summarize(
        source=args.input,
        output_dir=args.output_dir,
        rows=rows,
        examples=examples,
        train=train,
        val=val,
        heldout=heldout,
        skipped=skipped,
        split_meta=split_meta,
    )
    write_json(args.output_dir / "dataset_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
