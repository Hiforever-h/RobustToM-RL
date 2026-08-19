#!/usr/bin/env python3
"""Build response-only RFT data from accepted candidate rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rft.common import canonical_json, pair_id, read_jsonl, sample_id, sha256_file, stable_hash, write_jsonl
from rft.reward import normalize


def _prediction_key(row: dict[str, Any]) -> str:
    response = row.get("raw_response") or row.get("response") or row.get("accepted_response")
    if not isinstance(response, str):
        raise ValueError(f"Accepted row has no response: {row.get('candidate_id', sample_id(row))}")
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Accepted response is not JSON: {row.get('candidate_id', sample_id(row))}") from exc
    def normalized(value: Any) -> Any:
        if isinstance(value, str):
            return normalize(value)
        if isinstance(value, list):
            return [normalized(item) for item in value]
        if isinstance(value, dict):
            return {key: normalized(item) for key, item in value.items()}
        return value

    return canonical_json(normalized(parsed))


def _candidate_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    token_count = row.get("token_count")
    if not isinstance(token_count, int):
        token_count = len(str(row.get("raw_response", "")))
    index = row.get("candidate_index", 0)
    return token_count, index if isinstance(index, int) else 0, str(row.get("candidate_id", ""))


def _select_pair_subset(rows: list[dict[str, Any]], max_samples: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= max_samples:
        return rows
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[pair_id(row)].append(row)
    if max_samples % 2:
        max_samples -= 1
    target_pairs = max_samples // 2
    ranked = sorted(grouped, key=lambda value: stable_hash(seed, value))
    selected = set(ranked[:target_pairs])
    return [row for row in rows if pair_id(row) in selected]


def build_dataset(
    scored_rows: list[dict[str, Any]],
    min_samples: int = 1000,
    max_samples: int = 3000,
    seed: int = 2026,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if min_samples < 0 or max_samples < 0 or min_samples > max_samples:
        raise ValueError("Require 0 <= min_samples <= max_samples")
    accepted = [row for row in scored_rows if row.get("accepted") is True]
    by_sample: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        by_sample[sample_id(row)].append(row)

    selected_by_sample: dict[str, dict[str, Any]] = {}
    duplicate_candidates = 0
    for current_sample, candidates in by_sample.items():
        unique: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            key = _prediction_key(candidate)
            if key in unique:
                duplicate_candidates += 1
            elif candidate.get("generation_reached_eos", False) is True:
                unique[key] = candidate
        if unique:
            selected_by_sample[current_sample] = min(unique.values(), key=_candidate_sort_key)

    by_pair: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_by_sample.values():
        by_pair[pair_id(row)].append(row)
    complete_pairs = {
        current_pair: rows
        for current_pair, rows in by_pair.items()
        if len(rows) == 2 and {row.get("intervention_type") for row in rows} == {"observed", "hidden"}
    }
    incomplete_pairs = len(by_pair) - len(complete_pairs)
    final_rows = [row for rows in complete_pairs.values() for row in rows]
    final_rows.sort(key=lambda row: (str(row.get("global_pair_id")), str(row.get("intervention_type"))))
    final_rows = _select_pair_subset(final_rows, max_samples, seed)

    if len(final_rows) < min_samples:
        raise ValueError(
            f"Only {len(final_rows)} accepted samples remain after complete-pair filtering; "
            f"minimum is {min_samples}. Increase candidate coverage instead of adding canonical targets."
        )

    manifest = {
        "accepted_candidate_count": len(accepted),
        "unique_prompt_count": len(selected_by_sample),
        "complete_pair_count": len(final_rows) // 2,
        "incomplete_pair_count": incomplete_pairs,
        "duplicate_candidate_count": duplicate_candidates,
        "final_sample_count": len(final_rows),
        "min_samples": min_samples,
        "max_samples": max_samples,
        "source_counts": dict(Counter(row.get("source_dataset") for row in final_rows)),
        "order_counts": dict(Counter(str(row.get("question_order")) for row in final_rows)),
        "intervention_counts": dict(Counter(row.get("intervention_type") for row in final_rows)),
        "shortcut_conflict_count": sum(bool(row.get("shortcut_conflict")) for row in final_rows),
        "last_mention_conflict_count": sum(bool(row.get("last_mention_conflict")) for row in final_rows),
        "seed": seed,
    }

    output_rows = []
    for row in final_rows:
        output_rows.append(
            {
                "global_sample_id": row["global_sample_id"],
                "global_pair_id": row["global_pair_id"],
                "process_prompt": row["process_prompt"],
                "accepted_response": row["raw_response"],
                "process_target": row["process_target"],
                "process_reward": row["score"]["reward"],
                "candidate_id": row.get("candidate_id"),
                "candidate_index": row.get("candidate_index"),
                "token_count": row.get("token_count"),
                "source_dataset": row.get("source_dataset"),
                "question_order": row.get("question_order"),
                "intervention_type": row.get("intervention_type"),
                "shortcut_conflict": row.get("shortcut_conflict", False),
                "last_mention_conflict": row.get("last_mention_conflict", False),
                "shortcut_prediction": row.get("shortcut_prediction"),
                "last_mentioned_container": row.get("last_mentioned_container"),
            }
        )
    return output_rows, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, manifest = build_dataset(
        read_jsonl(args.scored), args.min_samples, args.max_samples, args.seed
    )
    write_jsonl(args.output, rows)
    manifest["scored_sha256"] = sha256_file(args.scored)
    manifest["output_sha256"] = sha256_file(args.output)
    manifest_path = args.manifest or args.output.with_name("rft_dataset_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
