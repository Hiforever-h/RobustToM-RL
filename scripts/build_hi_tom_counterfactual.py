#!/usr/bin/env python3
"""Build paired counterfactual Hi-ToM examples with deterministic labels.

Each source question receives two interventions over the same story:

* observed: after a shared anchor event, all agents named in the question
  jointly observe one more move;
* hidden: after the same anchor event, the final move is hidden from all agents
  named in the question.

The pair prevents both the earliest-exit heuristic and the last-mentioned
container heuristic from solving every higher-order question.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TRAILING_SEPARATOR_RE = re.compile(r"\s*\*\*\*\s*$")
LINE_NUMBER_RE = re.compile(r"^(\d+)\s")
QUESTION_AGENT_RE = re.compile(r"\b[A-Z][a-z]+\b")
QUESTION_OBJECT_RE = re.compile(r"\bthe ([a-z_]+)\b")
EXIT_RE = re.compile(r"^\d+ (\w+) (?:exited|left|went out of)\b")
ENTRY_RE = re.compile(r"^\d+ (.+?) entered the ([A-Za-z_]+)\.$")
CHOICE_RE = re.compile(r"(?:^|, )([A-Z])\. ([a-z_]+)")


def normalize_story(story: str) -> str:
    """Remove JSON-conversion separators while preserving numbered events."""
    return TRAILING_SEPARATOR_RE.sub("", story).strip()


def parse_choices(choices: str) -> list[tuple[str, str]]:
    parsed = CHOICE_RE.findall(choices)
    if not parsed:
        raise ValueError(f"Could not parse choices: {choices!r}")
    return parsed


def question_agents(question: str) -> list[str]:
    return [
        token
        for token in QUESTION_AGENT_RE.findall(question)
        if token != "Where"
    ]


def question_object(question: str) -> str:
    objects = QUESTION_OBJECT_RE.findall(question)
    if not objects:
        raise ValueError(f"Could not identify target object: {question!r}")
    return objects[-1]


def story_agents(story: str) -> list[str]:
    """Read the five shared characters from the first non-waiting-room entry."""
    for line in story.splitlines():
        match = ENTRY_RE.match(line)
        if not match or match.group(2) == "waiting_room":
            continue
        raw_names = match.group(1).replace(" and ", ", ")
        agents = [name.strip() for name in raw_names.split(",") if name.strip()]
        if agents:
            return agents
    raise ValueError("Could not find the initial group-entry event")


def next_line_number(story: str) -> int:
    numbers = [
        int(match.group(1))
        for line in story.splitlines()
        if (match := LINE_NUMBER_RE.match(line))
    ]
    if not numbers:
        raise ValueError("Story contains no numbered events")
    return max(numbers) + 1


def serial_join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def append_intervention(
    story: str,
    agents: list[str],
    queried_agents: list[str],
    obj: str,
    anchor_container: str,
    final_container: str,
    intervention: str,
) -> str:
    start = next_line_number(story)
    group = serial_join(agents)
    observer_list = queried_agents or agents
    observers = serial_join(observer_list)
    if len(observer_list) == 1:
        final_observation = f"while {observers} watched."
        mutual_observation = f"{observers} clearly observed this final move."
    else:
        final_observation = f"while {observers} watched together."
        mutual_observation = (
            "Every listed observer saw that all the other listed observers "
            "watched this final move."
        )
    common_events = [
        f"{start} {group} entered the observation_room together.",
        (
            f"{start + 1} An automated system moved the {obj} to the "
            f"{anchor_container} while all of them watched."
        ),
        (
            f"{start + 2} Each person saw this move and saw that every other "
            "person saw it; the event was common knowledge among them."
        ),
    ]
    if intervention == "observed":
        events = common_events + [
            (
                f"{start + 3} {observers} stayed in the observation_room while "
                "the other people left for the waiting_room."
            ),
            (
                f"{start + 4} An automated system then moved the {obj} to the "
                f"{final_container} {final_observation}"
            ),
            f"{start + 5} {mutual_observation}",
        ]
    elif intervention == "hidden":
        events = common_events + [
            (
                f"{start + 3} {observers} left the observation_room for the "
                "waiting_room before the next event."
            ),
            (
                f"{start + 4} An automated system then moved the {obj} to the "
                f"{final_container} in a sealed room."
            ),
            (
                f"{start + 5} None of the listed people saw, heard about, or "
                "inferred this final move."
            ),
        ]
    else:
        raise ValueError(f"Unknown intervention: {intervention}")
    return f"{story.rstrip()}\n" + "\n".join(events) + "\n\n"


def rebuild_prompt(source_prompt: str, story: str, question: str, choices: str) -> str:
    try:
        prefix = source_prompt.split("Story:\n", 1)[0]
        note = "Note:" + source_prompt.split("\nNote:", 1)[1]
    except IndexError as exc:
        raise ValueError("Source prompt does not contain expected Story/Note blocks") from exc
    return (
        f"{prefix}Story:\n{story.rstrip()}\n\n"
        f"Question: {question}\nChoices: {choices}\n\n{note}"
    )


def shortcut_prediction(row: dict[str, Any]) -> str | None:
    """Implement the known earliest-queried-agent-exit Hi-ToM heuristic."""
    if int(row["question_order"]) == 0:
        return None

    agents = question_agents(row["question"])
    obj = question_object(row["question"])
    exits: dict[str, int] = {}
    locations: list[tuple[int, str]] = []

    for index, line in enumerate(normalize_story(row["story"]).splitlines()):
        exit_match = EXIT_RE.match(line)
        if exit_match:
            exits.setdefault(exit_match.group(1), index)

        for pattern in (
            re.compile(rf"\b{re.escape(obj)} is in the ([a-z_]+)"),
            re.compile(rf"\b{re.escape(obj)} to the ([a-z_]+)"),
        ):
            location_match = pattern.search(line)
            if location_match:
                locations.append((index, location_match.group(1)))
                break

    if not agents or not all(agent in exits for agent in agents):
        return None
    cutoff = min(exits[agent] for agent in agents)
    visible_locations = [location for index, location in locations if index <= cutoff]
    return visible_locations[-1] if visible_locations else None


def group_key(row: dict[str, Any]) -> str:
    if "source_group_id" in row:
        return str(row["source_group_id"])
    # The upstream file stores 20 scenarios per condition and repeats each one
    # for orders 0..4. sample_id % 20 recovers that scenario identity.
    return (
        f"deception={str(bool(row['deception'])).lower()}|"
        f"length={int(row['story_length'])}|scenario={int(row['sample_id']) % 20:02d}"
    )


def choose_counterfactual_containers(
    rows: list[dict[str, Any]], group: str, seed: int
) -> tuple[str, str]:
    choices = [value for _, value in parse_choices(rows[0]["choices"])]
    excluded = {row["answer"] for row in rows}
    excluded.update(
        prediction
        for row in rows
        if (prediction := shortcut_prediction(row)) is not None
    )
    candidates = sorted(set(choices) - excluded)
    if len(candidates) < 2:
        raise ValueError(f"Fewer than two counterfactual containers remain for {group}")
    digest = hashlib.sha256(f"{seed}|{group}".encode()).digest()
    first_index = int.from_bytes(digest[:8], "big") % len(candidates)
    anchor = candidates.pop(first_index)
    second_index = int.from_bytes(digest[8:16], "big") % len(candidates)
    final = candidates[second_index]
    return anchor, final


def assign_splits(
    grouped: dict[str, list[dict[str, Any]]], seed: int, eval_fraction: float
) -> dict[str, str]:
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be strictly between 0 and 1")

    strata: dict[tuple[bool, int], list[str]] = defaultdict(list)
    for group, rows in grouped.items():
        strata[(bool(rows[0]["deception"]), int(rows[0]["story_length"]))].append(group)

    split_by_group: dict[str, str] = {}
    for stratum, groups in sorted(strata.items()):
        groups = sorted(groups)
        rng = random.Random(f"{seed}|{stratum[0]}|{stratum[1]}")
        rng.shuffle(groups)
        eval_count = max(1, round(len(groups) * eval_fraction))
        for index, group in enumerate(groups):
            split_by_group[group] = "eval" if index < eval_count else "train"
    return split_by_group


def make_record(
    row: dict[str, Any],
    group: str,
    anchor_container: str,
    final_container: str,
    intervention: str,
    split: str,
) -> dict[str, Any]:
    original_story = normalize_story(row["story"])
    obj = question_object(row["question"])
    queried_agents = question_agents(row["question"])
    new_story = append_intervention(
        original_story,
        story_agents(original_story),
        queried_agents,
        obj,
        anchor_container,
        final_container,
        intervention,
    )
    order = int(row["question_order"])
    if intervention == "observed" or order == 0:
        answer = final_container
    else:
        answer = anchor_container

    heuristic = shortcut_prediction(row)
    pair_id = f"{group}|order={order}"
    record = dict(row)
    record.update(
        {
            "sample_id": f"{row['sample_id']}-{intervention}",
            "source_sample_id": row["sample_id"],
            "source_group_id": group,
            "pair_id": pair_id,
            "split": split,
            "intervention_type": intervention,
            "counterfactual_anchor_container": anchor_container,
            "counterfactual_container": final_container,
            "original_answer": row["answer"],
            "answer": answer,
            "story": new_story,
            "shortcut_name": "earliest_queried_agent_exit_location",
            "shortcut_prediction": heuristic,
            "shortcut_conflict": heuristic is not None and heuristic != answer,
            "last_mentioned_container": final_container,
            "last_mention_conflict": order > 0 and final_container != answer,
        }
    )
    record["prompt"] = rebuild_prompt(
        row["prompt"], new_story, row["question"], row["choices"]
    )
    return record


def validate_records(records: list[dict[str, Any]]) -> None:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    for record in records:
        pairs[record["pair_id"]].append(record)
        groups_by_split[record["split"]].add(record["source_group_id"])
        choices = {value for _, value in parse_choices(record["choices"])}
        if record["answer"] not in choices:
            raise ValueError(f"Answer missing from choices: {record['sample_id']}")
        if record["story"] not in record["prompt"]:
            raise ValueError(f"Prompt/story mismatch: {record['sample_id']}")

    for pair_id, pair in pairs.items():
        if {row["intervention_type"] for row in pair} != {"observed", "hidden"}:
            raise ValueError(f"Incomplete pair: {pair_id}")
        observed = next(row for row in pair if row["intervention_type"] == "observed")
        hidden = next(row for row in pair if row["intervention_type"] == "hidden")
        final = observed["counterfactual_container"]
        anchor = observed["counterfactual_anchor_container"]
        if observed["answer"] != final:
            raise ValueError(f"Observed answer was not updated: {pair_id}")
        expected_hidden = final if int(hidden["question_order"]) == 0 else anchor
        if hidden["answer"] != expected_hidden:
            raise ValueError(f"Hidden answer has incorrect epistemic update: {pair_id}")

    overlap = groups_by_split["train"] & groups_by_split["eval"]
    if overlap:
        raise ValueError(f"Source-story leakage across splits: {sorted(overlap)[:3]}")


def build_dataset(
    source: Path, seed: int, eval_fraction: float, prompting_type: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    source_rows = [
        row for row in payload["data"] if row["prompting_type"] == prompting_type
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[group_key(row)].append(row)

    for group, rows in grouped.items():
        orders = sorted(int(row["question_order"]) for row in rows)
        if orders != [0, 1, 2, 3, 4]:
            raise ValueError(f"Expected orders 0..4 in {group}, got {orders}")

    split_by_group = assign_splits(grouped, seed, eval_fraction)
    records: list[dict[str, Any]] = []
    for group in sorted(grouped):
        rows = sorted(grouped[group], key=lambda row: int(row["question_order"]))
        anchor_container, final_container = choose_counterfactual_containers(
            rows, group, seed
        )
        for row in rows:
            for intervention in ("observed", "hidden"):
                records.append(
                    make_record(
                        row,
                        group,
                        anchor_container,
                        final_container,
                        intervention,
                        split_by_group[group],
                    )
                )

    validate_records(records)
    higher_order = [row for row in records if int(row["question_order"]) > 0]
    source_higher_order = [
        row for row in source_rows if int(row["question_order"]) > 0
    ]
    source_shortcut_covered = [
        (row, prediction)
        for row in source_higher_order
        if (prediction := shortcut_prediction(row)) is not None
    ]
    generated_shortcut_covered = [
        row for row in higher_order if row["shortcut_prediction"] is not None
    ]
    metadata = {
        "name": "Hi-ToM Counterfactual Observation Pairs",
        "source": "https://github.com/ying-hui-he/Hi-ToM_dataset",
        "source_file": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "prompting_type": prompting_type,
        "seed": seed,
        "eval_fraction": eval_fraction,
        "source_records": len(source_rows),
        "source_story_groups": len(grouped),
        "generated_records": len(records),
        "split_counts": dict(Counter(row["split"] for row in records)),
        "order_counts": dict(Counter(str(row["question_order"]) for row in records)),
        "intervention_counts": dict(
            Counter(row["intervention_type"] for row in records)
        ),
        "source_higher_order_earliest_exit_shortcut_accuracy": sum(
            prediction == row["answer"]
            for row, prediction in source_shortcut_covered
        )
        / len(source_shortcut_covered),
        "generated_higher_order_earliest_exit_shortcut_accuracy": sum(
            row["shortcut_prediction"] == row["answer"]
            for row in generated_shortcut_covered
        )
        / len(generated_shortcut_covered),
        "generated_higher_order_last_mention_shortcut_accuracy": sum(
            row["last_mentioned_container"] == row["answer"]
            for row in higher_order
        )
        / len(higher_order),
        "higher_order_shortcut_conflict_rate": sum(
            bool(row["shortcut_conflict"]) for row in higher_order
        )
        / len(higher_order),
        "higher_order_last_mention_conflict_rate": sum(
            bool(row["last_mention_conflict"]) for row in higher_order
        )
        / len(higher_order),
    }
    return records, metadata


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--prompting-type", default="CoTP", choices=("CoTP", "VP"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, metadata = build_dataset(
        args.source, args.seed, args.eval_fraction, args.prompting_type
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_payload = {"metadata": metadata, "data": records}
    (args.output_dir / "Hi-ToM_counterfactual.json").write_text(
        json.dumps(full_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(
        args.output_dir / "train.jsonl",
        (row for row in records if row["split"] == "train"),
    )
    write_jsonl(
        args.output_dir / "eval.jsonl",
        (row for row in records if row["split"] == "eval"),
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
