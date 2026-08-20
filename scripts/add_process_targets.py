#!/usr/bin/env python3
"""Add deterministic process-supervision targets to the combined dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from build_hi_tom_counterfactual import question_agents, question_object


PROCESS_TARGET_VERSION = "1.0"
WORLD_FIELDS = {
    "tom_order",
    "belief_chain",
    "object",
    "reasoning_mode",
    "world_state",
    "answer",
}
BELIEF_FIELDS = {
    "tom_order",
    "belief_chain",
    "object",
    "reasoning_mode",
    "final_move_observed",
    "nested_belief",
    "answer",
}


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


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
    return count


def extract_process_inputs(record: dict[str, Any]) -> tuple[list[str], str, str, str]:
    source = record.get("source_dataset")
    if source == "hi-tom":
        chain = question_agents(record["question"])
        obj = question_object(record["question"])
        anchor = record["counterfactual_anchor_container"]
        final = record["counterfactual_container"]
    elif source == "exploretom":
        params = record.get("qprop=params")
        if not isinstance(params, list) or len(params) != 3:
            raise ValueError(f"Invalid ExploreToM qprop=params: {record['global_sample_id']}")
        agents, obj, relation = params
        if not isinstance(agents, list) or not all(isinstance(x, str) for x in agents):
            raise ValueError(f"Invalid ExploreToM belief chain: {record['global_sample_id']}")
        if not isinstance(obj, str) or not str(relation).startswith("container_location"):
            raise ValueError(f"Unsupported ExploreToM target: {record['global_sample_id']}")
        chain = agents
        anchor = record["counterfactual_anchor_container"]
        final = record["counterfactual_final_container"]
    else:
        raise ValueError(f"Unknown source_dataset: {source!r}")
    return list(chain), str(obj), str(anchor), str(final)


def build_process_target(record: dict[str, Any]) -> dict[str, Any]:
    order = int(record["question_order"])
    chain, obj, anchor, final = extract_process_inputs(record)
    if len(chain) != order:
        raise ValueError(
            f"Belief-chain length {len(chain)} != order {order}: "
            f"{record['global_sample_id']}"
        )

    if order == 0:
        target = {
            "tom_order": 0,
            "belief_chain": [],
            "object": obj,
            "reasoning_mode": "world_state",
            "world_state": final,
            "answer": final,
        }
    else:
        observed = record["intervention_type"] == "observed"
        nested_belief = final if observed else anchor
        target = {
            "tom_order": order,
            "belief_chain": chain,
            "object": obj,
            "reasoning_mode": "belief",
            "final_move_observed": observed,
            "nested_belief": nested_belief,
            "answer": nested_belief,
        }

    if target["answer"] != record["answer"]:
        raise ValueError(
            f"Derived answer {target['answer']!r} != stored answer "
            f"{record['answer']!r}: {record['global_sample_id']}"
        )
    return target


def source_note(record: dict[str, Any]) -> str:
    prompt = str(record.get("prompt", ""))
    marker = "\n\nNote:"
    if marker not in prompt:
        return ""
    return "\n\nNote:" + prompt.split(marker, 1)[1]


def build_process_prompt(record: dict[str, Any]) -> str:
    instruction = (
        "Read the story and answer the question by returning exactly one JSON "
        "object and no markdown or additional text. Use container names, object "
        "names, and character names exactly as written. The belief_chain lists "
        "characters from the outermost thinker to the innermost thinker.\n\n"
        "For a world-state question, use exactly these keys:\n"
        '{"tom_order": 0, "belief_chain": [], "object": "...", '
        '"reasoning_mode": "world_state", "world_state": "...", '
        '"answer": "..."}\n\n'
        "For a belief question, use exactly these keys:\n"
        '{"tom_order": 1, "belief_chain": ["..."], "object": "...", '
        '"reasoning_mode": "belief", "final_move_observed": true, '
        '"nested_belief": "...", "answer": "..."}\n'
        "Replace the example values with values inferred from the story and "
        "question. tom_order may be greater than 1. final_move_observed is true "
        "only when the complete queried belief chain jointly observes the final "
        "move relevant to the answer."
    )
    demonstrations = (
        "The following demonstrations show how to fill the JSON fields. "
        "They are format demonstrations only. Solve the actual story independently. "
        "The answer and nested_belief/world_state fields contain container names, "
        "not choice letters. The belief_chain contains character names only, "
        "ordered from outer thinker to inner thinker.\n\n"
        "Demonstration 1: order 0 world-state question\n\n"
        "Story:\n"
        "1 Ava entered the room.\n"
        "2 The key is in the red_box.\n"
        "3 An automated system moved the key to the blue_box while Ava watched.\n\n"
        "Question: Where is the key really?\n"
        "Choices: A. red_box, B. blue_box\n\n"
        "Output:\n"
        '{"tom_order":0,"belief_chain":[],"object":"key",'
        '"reasoning_mode":"world_state","world_state":"blue_box",'
        '"answer":"blue_box"}\n\n'
        "Demonstration 2: order 1 observed belief question\n\n"
        "Story:\n"
        "1 Alice and Bob entered the room.\n"
        "2 The coin is in the red_box.\n"
        "3 Alice left the room.\n"
        "4 An automated system moved the coin to the blue_box while Bob watched.\n"
        "5 Bob clearly observed this final move.\n\n"
        "Question: Where does Bob think the coin is?\n"
        "Choices: A. red_box, B. blue_box\n\n"
        "Output:\n"
        '{"tom_order":1,"belief_chain":["Bob"],"object":"coin",'
        '"reasoning_mode":"belief","final_move_observed":true,'
        '"nested_belief":"blue_box","answer":"blue_box"}\n\n'
        "Demonstration 3: order 2 hidden belief question\n\n"
        "Story:\n"
        "1 Alice and Bob entered the room.\n"
        "2 The key is in the red_box.\n"
        "3 An automated system moved the key to the blue_box while Alice and Bob watched.\n"
        "4 Alice and Bob left the room.\n"
        "5 An automated system moved the key to the green_box in a sealed room.\n"
        "6 Alice and Bob did not see, hear about, or infer this final move.\n\n"
        "Question: Where does Alice think Bob thinks the key is?\n"
        "Choices: A. red_box, B. blue_box, C. green_box\n\n"
        "Output:\n"
        '{"tom_order":2,"belief_chain":["Alice","Bob"],"object":"key",'
        '"reasoning_mode":"belief","final_move_observed":false,'
        '"nested_belief":"blue_box","answer":"blue_box"}'
    )
    return (
        f"{instruction}\n\n{demonstrations}\n\n"
        f"Story:\n{record['story'].rstrip()}\n\n"
        f"Question: {record['question']}\nChoices: {record['choices']}"
        f"{source_note(record)}\n"
    )


def enrich_record(record: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(record)
    enriched["process_target_version"] = PROCESS_TARGET_VERSION
    enriched["process_target"] = build_process_target(record)
    enriched["process_prompt"] = build_process_prompt(record)
    enriched["process_response"] = json.dumps(
        enriched["process_target"], ensure_ascii=True, separators=(",", ":")
    )
    return enriched


def validate(records_by_split: dict[str, list[dict[str, Any]]]) -> None:
    pairs: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_ids: set[str] = set()
    for split, records in records_by_split.items():
        for record in records:
            sample_id = record["global_sample_id"]
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate global_sample_id: {sample_id}")
            sample_ids.add(sample_id)
            if record["split"] != split:
                raise ValueError(f"Split mismatch: {sample_id}")
            if record["process_target"]["answer"] != record["answer"]:
                raise ValueError(f"Process/final answer mismatch: {sample_id}")
            expected_fields = (
                WORLD_FIELDS
                if record["process_target"]["reasoning_mode"] == "world_state"
                else BELIEF_FIELDS
            )
            if set(record["process_target"]) != expected_fields:
                raise ValueError(f"Unexpected process schema: {sample_id}")
            if record["story"] not in record["process_prompt"]:
                raise ValueError(f"Story missing from process prompt: {sample_id}")
            if json.loads(record["process_response"]) != record["process_target"]:
                raise ValueError(f"Process response/target mismatch: {sample_id}")
            pairs[record["global_pair_id"]].append(record)

    for pair_id, pair in pairs.items():
        if len(pair) != 2 or {r["intervention_type"] for r in pair} != {
            "observed",
            "hidden",
        }:
            raise ValueError(f"Incomplete counterfactual pair: {pair_id}")
        observed = next(r for r in pair if r["intervention_type"] == "observed")
        hidden = next(r for r in pair if r["intervention_type"] == "hidden")
        left = observed["process_target"]
        right = hidden["process_target"]
        for field in ("tom_order", "belief_chain", "object", "reasoning_mode"):
            if left[field] != right[field]:
                raise ValueError(f"Pair process mismatch for {field}: {pair_id}")
        if left["tom_order"] == 0:
            if left != right:
                raise ValueError(f"Order-0 pair should have identical targets: {pair_id}")
        else:
            if not left["final_move_observed"] or right["final_move_observed"]:
                raise ValueError(f"Incorrect observed/hidden visibility: {pair_id}")


def build_manifest(
    source_manifest: dict[str, Any],
    records_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    all_records = [record for rows in records_by_split.values() for record in rows]
    return {
        "name": "Counterfactual Dataset with Deterministic Process Targets",
        "process_target_version": PROCESS_TARGET_VERSION,
        "label_method": "deterministic generator metadata and counterfactual state",
        "llm_as_judge_used": False,
        "records": len(all_records),
        "pairs": len({r["global_pair_id"] for r in all_records}),
        "split_counts": dict(Counter(r["split"] for r in all_records)),
        "source_counts": dict(Counter(r["source_dataset"] for r in all_records)),
        "reasoning_mode_counts": dict(
            Counter(r["process_target"]["reasoning_mode"] for r in all_records)
        ),
        "order_counts": dict(Counter(str(r["question_order"]) for r in all_records)),
        "intervention_counts": dict(Counter(r["intervention_type"] for r in all_records)),
        "input_manifest": source_manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/counterfactual_combined")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/counterfactual_process_reward")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records_by_split = {
        split: [enrich_record(record) for record in read_jsonl(args.input_dir / f"{split}.jsonl")]
        for split in ("train", "val", "test")
    }
    validate(records_by_split)
    source_manifest = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = build_manifest(source_manifest, records_by_split)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in records_by_split.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", records)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "input_manifest"}, indent=2))


if __name__ == "__main__":
    main()
