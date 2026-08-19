#!/usr/bin/env python3
"""Evaluate process JSON responses with reward, pair and shortcut metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from rft.common import read_jsonl
from rft.reward import normalize, parse_prediction, score_process_output


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _answer(prediction: dict[str, Any] | None) -> str | None:
    if not prediction or not isinstance(prediction.get("answer"), str):
        return None
    return normalize(prediction["answer"])


def _metric_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    scored = []
    lengths = []
    for row in rows:
        response = row.get("response") or row.get("raw_response") or row.get("accepted_response")
        target = row.get("process_target")
        if isinstance(target, str):
            target = json.loads(target)
        if not isinstance(target, dict):
            raise ValueError(f"Missing process_target: {row.get('global_sample_id', '<unknown>')}")
        if isinstance(row.get("token_count"), int):
            lengths.append(row["token_count"])
        result = score_process_output(response if isinstance(response, str) else "", target)
        prediction, _ = parse_prediction(response if isinstance(response, str) else "")
        checks = result["checks"]
        mode = target.get("reasoning_mode")
        core_state = (
            checks.get("world_state", False)
            if mode == "world_state"
            else checks.get("final_move_observed", False) and checks.get("nested_belief", False)
        )
        state_value = None
        if prediction is not None:
            state_value = normalize(
                prediction.get("world_state", "")
                if mode == "world_state"
                else prediction.get("nested_belief", "")
            )
        consistent = bool(
            prediction is not None
            and isinstance(prediction.get("answer"), str)
            and state_value is not None
            and normalize(prediction["answer"]) == state_value
        )
        scored.append(
            {
                "row": row,
                "result": result,
                "prediction": prediction,
                "answer": _answer(prediction),
                "target_answer": normalize(target.get("answer", "")),
                "answer_correct": bool(checks.get("answer", False)),
                "core_state_correct": bool(core_state),
                "consistent": consistent,
            }
        )

    def rate(predicate) -> float:
        return sum(bool(predicate(item)) for item in scored) / len(scored) if scored else 0.0

    pair_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        pair_groups[str(item["row"].get("global_pair_id"))].append(item)
    complete_pairs = [group for group in pair_groups.values() if len(group) == 2]
    pair_accuracy = (
        sum(all(item["answer_correct"] for item in group) for group in complete_pairs) / len(complete_pairs)
        if complete_pairs
        else 0.0
    )
    sensitivity_groups = [group for group in complete_pairs if len({item["target_answer"] for item in group}) > 1]
    intervention_sensitivity = (
        sum(
            len({item["answer"] for item in group}) == 2
            and all(item["answer"] is not None for item in group)
            for group in sensitivity_groups
        )
        / len(sensitivity_groups)
        if sensitivity_groups
        else 0.0
    )

    shortcut_rows = [
        item
        for item in scored
        if item["row"].get("shortcut_conflict") and item["row"].get("shortcut_prediction") is not None
    ]
    shortcut_copy_rate = (
        sum(item["answer"] == normalize(item["row"]["shortcut_prediction"]) for item in shortcut_rows)
        / len(shortcut_rows)
        if shortcut_rows
        else 0.0
    )
    last_rows = [
        item
        for item in scored
        if item["row"].get("last_mention_conflict") and item["row"].get("last_mentioned_container") is not None
    ]
    last_copy_rate = (
        sum(item["answer"] == normalize(item["row"]["last_mentioned_container"]) for item in last_rows)
        / len(last_rows)
        if last_rows
        else 0.0
    )
    return {
        "count": len(scored),
        "parse_rate": rate(lambda item: item["result"]["checks"].get("parseable_json", False)),
        "strict_format_rate": rate(lambda item: item["result"]["checks"].get("format", False)),
        "mean_process_reward": sum(item["result"]["reward"] for item in scored) / len(scored) if scored else 0.0,
        "full_reward_rate": rate(lambda item: item["result"]["reward"] == 1.0),
        "answer_accuracy": rate(lambda item: item["answer_correct"]),
        "core_state_accuracy": rate(lambda item: item["core_state_correct"]),
        "pair_accuracy": pair_accuracy,
        "intervention_sensitivity": intervention_sensitivity,
        "shortcut_copy_rate": shortcut_copy_rate,
        "last_mention_copy_rate": last_copy_rate,
        "answer_state_consistency": rate(lambda item: item["consistent"]),
        "eos_rate": rate(lambda item: item["row"].get("generation_reached_eos", False) is True),
        "length_p95": _percentile(lengths, 0.95),
        "group_count": len(complete_pairs),
        "sensitivity_pair_count": len(sensitivity_groups),
    }


def evaluate_predictions(prediction_rows: list[dict[str, Any]], data_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if data_rows:
        by_sample = {row.get("global_sample_id"): row for row in data_rows}
        merged = []
        for prediction in prediction_rows:
            base = dict(by_sample.get(prediction.get("global_sample_id"), {}))
            base.update(prediction)
            merged.append(base)
    else:
        merged = prediction_rows
    metrics = {"overall": _metric_rows(merged)}
    dimensions = {
        "source_dataset": lambda row: row.get("source_dataset", "unknown"),
        "question_order": lambda row: str(row.get("question_order", "unknown")),
        "intervention_type": lambda row: row.get("intervention_type", "unknown"),
    }
    for name, key_fn in dimensions.items():
        buckets: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in merged:
            buckets[str(key_fn(row))].append(row)
        metrics[name] = {key: _metric_rows(rows) for key, rows in sorted(buckets.items())}
    for name, flag in (("shortcut_conflict", "shortcut_conflict"), ("last_mention_conflict", "last_mention_conflict")):
        subset = [row for row in merged if row.get(flag)]
        metrics[name] = _metric_rows(subset)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_predictions(read_jsonl(args.predictions), read_jsonl(args.data) if args.data else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics["overall"], indent=2))


if __name__ == "__main__":
    main()
