#!/usr/bin/env python3
"""Convert LUME Task2 rows into API4-style true-prefix completion data.

The output schema mirrors data/api4_200k/sft_true_prefix_no_instruction*.json:
input/output/pii_type/source_id/entity_index/char_start/char_end/full_text.
It also writes FFN Edit privacy bags as [[input + output, output]] per sample.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LUME_ROOT = REPO_ROOT / "data/lume/lume-llm-unlearning/data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/lume/api4_prefix_task2"
DEFAULT_TOKENIZER = REPO_ROOT / "models/llama3-8B/baseline"

MONTH = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)

MAILTO_RE = re.compile(r"\[[^\]]+\]\(mailto:([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\)")

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONENUMBER", re.compile(r"\b(?:\+?1[-.\s]?)?\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")),
    ("DOB", re.compile(rf"\b(?:{MONTH})\s+\d{{1,2}},\s+\d{{4}}\b")),
    (
        "ADDRESS",
        re.compile(
            r"\b\d{1,6}\s+"
            r"[A-Za-z0-9.'\- ]+?"
            r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|"
            r"Drive|Dr\.?|Lane|Ln\.?|Court|Ct\.?|Circle|Cir\.?|Way|"
            r"Place|Pl\.?|Terrace|Ter\.?)"
            r",\s+[A-Za-z.'\- ]+,\s+[A-Z]{2},\s+\d{5}(?:-\d{4})?\b"
        ),
    ),
)

CONTEXT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("DOB", re.compile(rf"\bborn on\s+((?:{MONTH})\s+\d{{1,2}},\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}})", re.IGNORECASE)),
    ("SSN", re.compile(r"\b(?:Social Security number|SSN)\s*(?:is)?\s*([0-9\-]{9,11})", re.IGNORECASE)),
    ("PHONENUMBER", re.compile(r"\b(?:phone|telephone)(?: number)?\s*(?:is|at|:)?\s*(\+?\d[\d\-.\s()]{8,18}\d)", re.IGNORECASE)),
    ("EMAIL", re.compile(r"\bemail address is\s+([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})", re.IGNORECASE)),
    (
        "ADDRESS",
        re.compile(
            r"\b(?:home address is|address is|resides at(?: the home address(?: of)?)?|lives at)\s+(.+?)(?:\.|$)",
            re.IGNORECASE,
        ),
    ),
)

QA_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SSN", ("social security", "ssn")),
    ("PHONENUMBER", ("phone",)),
    ("EMAIL", ("email", "e-mail")),
    ("ADDRESS", ("address",)),
    ("DOB", ("born", "birth", "date of birth")),
)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    pii_type: str
    secret: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LUME Task2 as API4-style prefix-completion JSON.")
    parser.add_argument("--lume_root", type=Path, default=DEFAULT_LUME_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--no_token_stats", action="store_true")
    return parser.parse_args()


def normalize_lume_text(text: Any) -> str:
    text = "" if text is None else str(text)
    return MAILTO_RE.sub(r"\1", text)


def source_base_id(lume_id: Any) -> str:
    value = str(lume_id).strip()
    value = re.sub(r"^(\"|')|(\"|')$", "", value)
    value = re.sub(r"(?:sc|qa)\d+$", "", value)
    return value.strip("\"'")


def row_kind(lume_id: Any) -> str:
    value = str(lume_id)
    if re.search(r"sc\d+$", value):
        return "scenario"
    if re.search(r"qa\d+$", value):
        return "qa"
    return "unknown"


def iter_lume_rows(lume_root: Path) -> Iterable[dict[str, Any]]:
    for lume_set in ("forget", "retain"):
        path = lume_root / f"{lume_set}.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("task") != "Task2":
                    continue
                yield {
                    "lume_set": lume_set,
                    "line_number": line_number,
                    "lume_id": row.get("id"),
                    "row_kind": row_kind(row.get("id")),
                    "source_base_id": source_base_id(row.get("id")),
                    "input": normalize_lume_text(row.get("input")),
                    "output": normalize_lume_text(row.get("output")),
                    "task": row.get("task"),
                }


def overlaps(left: Span, right: Span) -> bool:
    return left.start < right.end and right.start < left.end


def trim_secret_span(text: str, start: int, end: int) -> tuple[int, int, str]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    while end > start and text[end - 1] in ".,;:":
        end -= 1
    return start, end, text[start:end]


def infer_qa_type(question: str) -> str | None:
    lowered = question.casefold()
    for pii_type, keywords in QA_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return pii_type
    return None


def extract_spans(text: str, row: dict[str, Any] | None = None) -> list[Span]:
    candidates: list[Span] = []
    for pii_type, pattern in CONTEXT_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span(1)
            start, end, secret = trim_secret_span(text, start, end)
            if pii_type == "ADDRESS" and "@" in secret:
                continue
            if secret:
                candidates.append(Span(start, end, pii_type, secret))

    for pii_type, pattern in PATTERNS:
        for match in pattern.finditer(text):
            start, end, secret = trim_secret_span(text, match.start(), match.end())
            if secret:
                candidates.append(Span(start, end, pii_type, secret))

    if row and row.get("row_kind") == "qa":
        pii_type = infer_qa_type(str(row.get("input", "")))
        if pii_type:
            start = len(str(row.get("input", "")))
            end = start + len(str(row.get("output", "")))
            start, end, secret = trim_secret_span(text, start, end)
            if secret:
                candidates.append(Span(start, end, pii_type, secret))

    candidates.sort(key=lambda span: (span.start, -(span.end - span.start)))
    kept: list[Span] = []
    seen: set[tuple[int, int, str, str]] = set()
    for span in candidates:
        key = (span.start, span.end, span.pii_type, span.secret)
        if key in seen:
            continue
        seen.add(key)
        if any(overlaps(span, existing) for existing in kept):
            continue
        kept.append(span)
    return kept


def build_examples(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    skipped = Counter()
    source_entity_counts: Counter[str] = Counter()

    for row in rows:
        full_text = f"{row['input']}{row['output']}"
        spans = extract_spans(full_text, row)
        if not spans:
            skipped["no_pii_span"] += 1
            continue
        source_id = f"lume:{row['lume_set']}:{row['source_base_id']}"
        for span in spans:
            entity_index = source_entity_counts[source_id]
            source_entity_counts[source_id] += 1
            item = {
                "input": full_text[: span.start],
                "output": span.secret,
                "pii_type": span.pii_type,
                "source_id": source_id,
                "entity_index": entity_index,
                "char_start": span.start,
                "char_end": span.end,
                "full_text": full_text,
                "lume_id": row["lume_id"],
                "lume_set": row["lume_set"],
                "lume_task": row["task"],
                "lume_row_kind": row["row_kind"],
            }
            examples.append(item)

    metadata = {
        "skipped": dict(skipped),
        "source_count": len(source_entity_counts),
    }
    return examples, metadata


def split_by_source(
    examples: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in examples:
        by_source[str(item["source_id"])].append(item)

    sources_by_set: dict[str, list[str]] = defaultdict(list)
    for source_id in by_source:
        lume_set = source_id.split(":", 2)[1] if source_id.startswith("lume:") else "unknown"
        sources_by_set[lume_set].append(source_id)

    split_sources = {"train": set(), "val": set(), "test": set()}
    rng = random.Random(seed)
    for source_ids in sources_by_set.values():
        shuffled = list(source_ids)
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_cut = int(total * train_ratio)
        val_cut = train_cut + int(total * val_ratio)
        split_sources["train"].update(shuffled[:train_cut])
        split_sources["val"].update(shuffled[train_cut:val_cut])
        split_sources["test"].update(shuffled[val_cut:])

    splits = {"train": [], "val": [], "test": []}
    for split_name, source_ids in split_sources.items():
        for source_id in sorted(source_ids):
            for item in by_source[source_id]:
                enriched = dict(item)
                enriched["split"] = split_name
                splits[split_name].append(enriched)
    return splits


def validate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    for index, item in enumerate(examples):
        for field in ("input", "output", "pii_type", "source_id", "full_text"):
            if not item.get(field):
                errors.append(f"row {index} missing {field}")
        start = item.get("char_start")
        end = item.get("char_end")
        full_text = item.get("full_text", "")
        output = item.get("output", "")
        if not isinstance(start, int) or not isinstance(end, int) or full_text[start:end] != output:
            errors.append(f"row {index} span mismatch")
        if item.get("input") != full_text[:start]:
            errors.append(f"row {index} prefix mismatch")
    if errors:
        preview = "; ".join(errors[:10])
        raise ValueError(f"validation failed with {len(errors)} errors: {preview}")

    by_source_split: dict[str, set[str]] = defaultdict(set)
    for item in examples:
        by_source_split[str(item["source_id"])].add(str(item.get("split", "unsplit")))
    leaked = {source: sorted(splits) for source, splits in by_source_split.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"source_id split leakage detected for {len(leaked)} sources")

    return {
        "total_examples": len(examples),
        "pii_type_counts": dict(sorted(Counter(item["pii_type"] for item in examples).items())),
        "lume_set_counts": dict(sorted(Counter(item["lume_set"] for item in examples).items())),
        "row_kind_counts": dict(sorted(Counter(item["lume_row_kind"] for item in examples).items())),
        "source_count": len({item["source_id"] for item in examples}),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_privacy_bags(path: Path, examples: list[dict[str, Any]]) -> None:
    bags = [[[f"{item['input']}{item['output']}", item["output"]]] for item in examples]
    write_json(path, bags)


def write_validation_text(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in examples:
            handle.write(f"{item['input']}{item['output']}\n")


def token_length_stats(examples: list[dict[str, Any]], tokenizer_path: Path) -> dict[str, Any]:
    if not tokenizer_path.exists():
        return {"available": False, "reason": f"tokenizer path not found: {tokenizer_path}"}
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        return {"available": False, "reason": f"cannot import transformers: {type(exc).__name__}"}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True)
    lengths = [
        len(tokenizer.encode(f"{item['input']}{item['output']}", add_special_tokens=False))
        for item in examples
    ]
    if not lengths:
        return {"available": True, "count": 0}
    sorted_lengths = sorted(lengths)

    def percentile(pct: float) -> int:
        index = min(len(sorted_lengths) - 1, int(round((pct / 100) * (len(sorted_lengths) - 1))))
        return int(sorted_lengths[index])

    return {
        "available": True,
        "count": len(lengths),
        "mean": round(mean(lengths), 2),
        "max": max(lengths),
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train_ratio must be between 0 and 1")
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val_ratio must be between 0 and 1")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train_ratio + --val_ratio must be < 1")

    examples, build_metadata = build_examples(iter_lume_rows(args.lume_root))
    if not examples:
        raise SystemExit("No LUME Task2 PII examples were produced.")

    splits = split_by_source(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    all_examples = splits["train"] + splits["val"] + splits["test"]
    summary = validate_examples(all_examples)
    summary["build_metadata"] = build_metadata
    summary["input"] = {
        "lume_root": str(args.lume_root),
        "task": "Task2",
        "source_files": ["forget.jsonl", "retain.jsonl"],
    }
    summary["output"] = {
        "output_dir": str(args.output_dir),
        "schema": "api4_true_prefix_no_instruction",
    }
    summary["split_counts"] = {name: len(items) for name, items in splits.items()}
    summary["split_source_counts"] = {
        name: len({item["source_id"] for item in items}) for name, items in splits.items()
    }
    summary["split_pii_type_counts"] = {
        name: dict(sorted(Counter(item["pii_type"] for item in items).items()))
        for name, items in splits.items()
    }
    if not args.no_token_stats:
        summary["token_length_stats"] = token_length_stats(all_examples, args.tokenizer)

    out = args.output_dir
    write_json(out / "sft_true_prefix_no_instruction_all.json", all_examples)
    for split_name, items in splits.items():
        write_json(out / f"sft_true_prefix_no_instruction_{split_name}.json", items)
        write_privacy_bags(out / f"privacy_data_lume_task2_{split_name}.json", items)
        write_validation_text(out / f"validation_lume_task2_{split_name}.txt", items)

    for lume_set in ("forget", "retain"):
        subset = [item for item in all_examples if item["lume_set"] == lume_set]
        write_json(out / f"sft_true_prefix_no_instruction_{lume_set}.json", subset)
        write_privacy_bags(out / f"privacy_data_lume_task2_{lume_set}.json", subset)

    write_privacy_bags(out / "privacy_data_lume_task2_all.json", all_examples)
    write_json(out / "dataset_summary.json", summary)

    print(json.dumps({
        "output_dir": str(out),
        "total_examples": summary["total_examples"],
        "split_counts": summary["split_counts"],
        "pii_type_counts": summary["pii_type_counts"],
        "token_length_stats": summary.get("token_length_stats", {"available": False}),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
