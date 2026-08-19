#!/usr/bin/env python3
"""Merge the generated Hi-ToM and ExploreToM datasets into train/val/test files."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            records.append(record)
    return records


def with_provenance(
    records: Iterable[dict[str, Any]],
    source_dataset: str,
    source_split: str,
    target_split: str,
) -> list[dict[str, Any]]:
    result = []
    for record in records:
        enriched = dict(record)
        enriched["source_dataset"] = source_dataset
        enriched["source_split"] = source_split
        enriched["global_sample_id"] = f"{source_dataset}:{record['sample_id']}"
        enriched["global_pair_id"] = f"{source_dataset}:{record['pair_id']}"
        enriched["split"] = target_split
        result.append(enriched)
    return result


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
    return count


def validate(records_by_split: dict[str, list[dict[str, Any]]]) -> None:
    expected_splits = {"train", "val", "test"}
    if set(records_by_split) != expected_splits:
        raise ValueError(f"Expected splits {expected_splits}, got {set(records_by_split)}")

    sample_ids: set[str] = set()
    for split, records in records_by_split.items():
        for record in records:
            sample_id = record.get("global_sample_id")
            pair_id = record.get("global_pair_id")
            if not sample_id or not pair_id:
                raise ValueError(f"Missing global sample/pair ID in {split}")
            if record.get("split") != split:
                raise ValueError(f"Incorrect split field for {sample_id}")
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate global_sample_id: {sample_id}")
            sample_ids.add(sample_id)

    # Every source pair must remain entirely within one target split.
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for records in records_by_split.values():
        for record in records:
            grouped[record["global_pair_id"]].append(record)
    for pair_id, pair in grouped.items():
        if len(pair) != 2 or {record.get("intervention_type") for record in pair} != {
            "observed",
            "hidden",
        }:
            raise ValueError(f"Incomplete observed/hidden pair: {pair_id}")
        if len({record["split"] for record in pair}) != 1:
            raise ValueError(f"Pair crosses target splits: {pair_id}")


def build_manifest(
    records_by_split: dict[str, list[dict[str, Any]]], seed: int
) -> dict[str, Any]:
    all_records = [record for records in records_by_split.values() for record in records]
    source_split_counts: Counter[tuple[str, str]] = Counter(
        (record["source_dataset"], record["source_split"]) for record in all_records
    )
    source_counts: Counter[str] = Counter(record["source_dataset"] for record in all_records)
    split_order_counts: Counter[tuple[str, str]] = Counter(
        (record["split"], str(record["question_order"])) for record in all_records
    )
    return {
        "name": "Combined Hi-ToM and ExploreToM Counterfactual Dataset",
        "records": len(all_records),
        "pairs": len({record["global_pair_id"] for record in all_records}),
        "train_shuffle_seed": seed,
        "split_counts": dict(Counter(record["split"] for record in all_records)),
        "source_counts": dict(source_counts),
        "source_split_counts": {
            f"{source}|{source_split}": count
            for (source, source_split), count in sorted(source_split_counts.items())
        },
        "order_counts": dict(Counter(str(record["question_order"]) for record in all_records)),
        "split_order_counts": {
            f"{split}|order={order}": count
            for (split, order), count in sorted(split_order_counts.items())
        },
        "intervention_counts": dict(
            Counter(record["intervention_type"] for record in all_records)
        ),
        "unique_sample_ids": len({record["global_sample_id"] for record in all_records}),
        "unique_pair_ids": len({record["global_pair_id"] for record in all_records}),
        "source_directories": {
            "hi_tom": "data/hi_tom_counterfactual_3000",
            "exploretom": "data/exploretom_counterfactual_1200",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hi-tom-dir", type=Path, default=Path("data/hi_tom_counterfactual_3000")
    )
    parser.add_argument(
        "--exploretom-dir", type=Path, default=Path("data/exploretom_counterfactual_1200")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/counterfactual_combined")
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records_by_split = {
        "train": with_provenance(
            read_jsonl(args.hi_tom_dir / "train.jsonl"),
            "hi-tom",
            "train",
            "train",
        )
        + with_provenance(
            read_jsonl(args.exploretom_dir / "train.jsonl"),
            "exploretom",
            "train",
            "train",
        ),
        "val": with_provenance(
            read_jsonl(args.hi_tom_dir / "validation.jsonl"),
            "hi-tom",
            "validation",
            "val",
        ),
        "test": with_provenance(
            read_jsonl(args.hi_tom_dir / "order4_ood_test.jsonl"),
            "hi-tom",
            "order4_ood_test",
            "test",
        ),
    }
    random.Random(args.seed).shuffle(records_by_split["train"])
    validate(records_by_split)
    manifest = build_manifest(records_by_split, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written_counts = {
        split: write_jsonl(args.output_dir / f"{split}.jsonl", records)
        for split, records in records_by_split.items()
    }
    if written_counts != {split: len(records) for split, records in records_by_split.items()}:
        raise RuntimeError("A merged file was written with an unexpected number of records")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
