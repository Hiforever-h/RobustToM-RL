#!/usr/bin/env python3
"""Generate 1,200 shortcut-resistant training records with ExploreToM's DSL.

For each base scenario, the observed and hidden variants use the same moves but
swap whether queried agents leave before or after the final object movement.
The official FullBeliefTracker and QuestionGenerator compute every label.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


NAMES = [
    "Alice", "Benjamin", "Charlotte", "Daniel", "Eleanor", "Felix",
    "Grace", "Henry", "Isabella", "Jacob", "Kayla", "Liam", "Maya",
    "Noah", "Olivia", "Peter", "Quinn", "Rachel", "Samuel", "Taylor",
    "Uma", "Victor", "Willow", "Xavier", "Yasmin", "Zachary", "Amelia",
    "Brody", "Claire", "Dylan", "Eva", "George", "Hannah", "Ian",
    "Julia", "Kevin", "Lucy", "Matthew", "Nora", "Owen", "Phoebe",
    "Reid", "Sophie", "Thomas", "Violet", "William", "Addison", "Blake",
]
OBJECTS = [
    "camera", "contract", "flashlight", "keyring", "laptop", "notebook",
    "passport", "paintbrush", "recipe book", "silver watch", "tablet",
    "toy train", "violin bow", "wallet", "wedding album", "wooden puzzle",
    "coffee tin", "first aid kit", "garden shears", "glass ornament",
    "letter opener", "music score", "picnic blanket", "project folder",
    "remote control", "sewing kit", "sketchbook", "toolbox", "travel map",
    "water bottle", "ceramic mug", "chess set", "desk calendar", "field guide",
    "fountain pen", "headphones", "measuring tape", "photo frame", "raincoat",
    "sports medal", "storage key", "tea caddy", "ticket envelope", "work badge",
]
CONTAINERS = [
    "amber cabinet", "blue canvas bag", "brass locker", "cedar chest",
    "ceramic jar", "cloth hamper", "desk drawer", "filing cabinet",
    "glass case", "green toolbox", "grey suitcase", "leather satchel",
    "metal trunk", "oak cupboard", "orange backpack", "paper carton",
    "plastic bin", "red basket", "silver safe", "storage crate",
    "striped tote", "travel case", "wall cabinet", "wicker basket",
    "wooden box", "yellow locker", "archive drawer", "canvas pouch",
    "display cabinet", "document case", "equipment bin", "linen chest",
    "portable locker", "supply cupboard", "utility drawer", "velvet bag",
]
ROOMS = [
    "activity room", "archive room", "art studio", "break room",
    "conference room", "control room", "craft room", "design lab",
    "dining room", "equipment room", "gallery", "garden room", "green room",
    "hotel lobby", "kitchen", "library", "mail room", "meeting room",
    "music room", "office", "photo studio", "reading room", "rehearsal room",
    "staff lounge", "storage room", "study", "sunroom", "training room",
    "waiting room", "workshop",
]

PROMPT_PREFIX = (
    "Read the following story and answer the multiple-choice question. "
    "Think step-by-step. Provide the answer first, and then explain it."
)


def load_exploretom(repo: Path) -> tuple[Any, Any, Any]:
    required = [repo / "belief_tracker.py", repo / "LICENSE"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete ExploreToM checkout: {missing}")
    sys.path.insert(0, str(repo.resolve()))
    from belief_tracker import FullBeliefTracker, QuestionGenerator

    return FullBeliefTracker, QuestionGenerator, repo


def run_action(action: Any, step: str) -> None:
    if not action():
        raise ValueError(f"ExploreToM action failed: {step}")


def find_official_question(
    tracker: Any,
    QuestionGenerator: Any,
    order: int,
    queried_agents: list[str],
    obj: str,
) -> tuple[str, str, str]:
    matches = []
    for question, answer, conditions, metadata in QuestionGenerator(tracker).main(
        order, expand_relation_type_info=True
    ):
        agents, thing, relation = conditions
        if (
            agents == queried_agents
            and thing == obj
            and relation.startswith("container_location")
        ):
            matches.append((question, answer, relation))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one generated question, got {len(matches)} for "
            f"order={order}, agents={queried_agents}, object={obj}"
        )
    return matches[0]


def make_choices(
    candidates: list[str], answer: str, answer_index: int, seed_key: str
) -> tuple[str, str]:
    distractors = [candidate for candidate in candidates if candidate != answer]
    rng = random.Random(seed_key)
    rng.shuffle(distractors)
    selected = distractors[:5]
    selected.insert(answer_index, answer)
    letters = "ABCDEF"
    return ", ".join(f"{letters[i]}. {value}" for i, value in enumerate(selected)), letters[answer_index]


def build_variant(
    FullBeliefTracker: Any,
    QuestionGenerator: Any,
    scenario_id: int,
    order: int,
    intervention: str,
    people: list[str],
    obj: str,
    containers: list[str],
    rooms: list[str],
    seed: int,
) -> dict[str, Any]:
    outer, inner, mover, room_distractor = people
    primary_room, secondary_room = rooms
    initial, anchor, final, *choice_distractors = containers
    queried_agents = [outer] if order == 1 else [outer, inner]

    tracker = FullBeliefTracker()
    actions: list[tuple[str, Any]] = [
        ("room distractor enters secondary room", lambda: tracker.enter_room(room_distractor, secondary_room)),
        ("outer enters primary room", lambda: tracker.enter_room(outer, primary_room)),
        ("inner enters primary room", lambda: tracker.enter_room(inner, primary_room)),
        ("mover enters primary room", lambda: tracker.enter_room(mover, primary_room)),
        ("initial placement", lambda: tracker.move_object_container(mover, obj, initial)),
        ("shared anchor placement", lambda: tracker.move_object_container(mover, obj, anchor)),
    ]

    departures: list[tuple[str, Any]] = []
    for agent in queried_agents:
        departures.extend(
            [
                (f"{agent} leaves primary room", lambda agent=agent: tracker.leave_room(agent, primary_room)),
                (f"{agent} enters secondary room", lambda agent=agent: tracker.enter_room(agent, secondary_room)),
            ]
        )
    final_move = (
        "final placement",
        lambda: tracker.move_object_container(mover, obj, final),
    )

    if intervention == "observed":
        actions.extend([final_move, *departures])
    elif intervention == "hidden":
        actions.extend([*departures, final_move])
    else:
        raise ValueError(f"Unknown intervention: {intervention}")

    for step, action in actions:
        run_action(action, step)

    question, answer, relation = find_official_question(
        tracker, QuestionGenerator, order, queried_agents, obj
    )
    expected = final if intervention == "observed" else anchor
    if answer != expected:
        raise ValueError(
            f"Official tracker returned {answer!r}; expected {expected!r}"
        )

    variant_offset = 0 if intervention == "observed" else 1
    answer_index = (scenario_id * 2 + variant_offset) % 6
    choices, answer_letter = make_choices(
        [initial, anchor, final, *choice_distractors],
        answer,
        answer_index,
        f"{seed}|{scenario_id}|{intervention}|choices",
    )
    story = " ".join(tracker.story_script)
    prompt = (
        f"{PROMPT_PREFIX}\nStory:\n{story}\n\nQuestion: {question}\n"
        f"Choices: {choices}\n"
    )
    pair_id = f"exploretom-cf-{scenario_id:04d}"
    return {
        "dataset": "ExploreToM-counterfactual",
        "prompting_type": "CoTP",
        "split": "train",
        "sample_id": f"{pair_id}-{intervention}",
        "pair_id": pair_id,
        "base_scenario_id": scenario_id,
        "question_order": order,
        "intervention_type": intervention,
        "story": story,
        "story_structure": story,
        "question": question,
        "choices": choices,
        "answer": answer,
        "answer_letter": answer_letter,
        "prompt": prompt,
        "qprop=params": [queried_agents, obj, relation],
        "counterfactual_anchor_container": anchor,
        "counterfactual_final_container": final,
        "last_container_prediction": final,
        "last_container_conflict": final != answer,
        "rooms": [primary_room, secondary_room],
        "shortcut_targets": ["last-container", "only-room"],
    }


def generate_records(repo: Path, seed: int, num_pairs: int) -> list[dict[str, Any]]:
    if num_pairs % 2:
        raise ValueError("num_pairs must be even to balance first/second-order questions")
    FullBeliefTracker, QuestionGenerator, _ = load_exploretom(repo)
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    signatures: set[tuple[Any, ...]] = set()

    for scenario_id in range(num_pairs):
        order = 1 if scenario_id < num_pairs // 2 else 2
        while True:
            people = rng.sample(NAMES, 4)
            obj = rng.choice(OBJECTS)
            containers = rng.sample(CONTAINERS, 8)
            rooms = rng.sample(ROOMS, 2)
            signature = (*people, obj, *containers[:3], *rooms, order)
            if signature not in signatures:
                signatures.add(signature)
                break
        for intervention in ("observed", "hidden"):
            records.append(
                build_variant(
                    FullBeliefTracker,
                    QuestionGenerator,
                    scenario_id,
                    order,
                    intervention,
                    people,
                    obj,
                    containers,
                    rooms,
                    seed,
                )
            )
    return records


def audit_official_sample(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("qprop=params")]

    last_container_rows: list[tuple[dict[str, str], str]] = []
    room_questions = 0
    only_room_rows: list[tuple[dict[str, str], str]] = []
    for row in rows:
        agents, obj, relation = ast.literal_eval(row["qprop=params"])
        if row["qprop=nth_order"] in {"1", "2"} and relation.startswith(
            "container_location"
        ):
            pattern = re.compile(
                rf"\bmoved the {re.escape(obj)} to the (.*?), which is also located in",
                re.IGNORECASE,
            )
            hits = pattern.findall(row["story_structure"])
            if hits:
                last_container_rows.append((row, hits[-1].strip()))

        if relation.startswith("room_location"):
            room_questions += 1
            room_mentions: list[str] = []
            for pattern in (
                r"entered the (.*?)(?:\.|,)",
                r"left the (.*?)(?:\.|,)",
                r"located in the (.*?)(?:\.|,)",
            ):
                room_mentions.extend(
                    value.strip()
                    for value in re.findall(
                        pattern, row["story_structure"], re.IGNORECASE
                    )
                )
            unique_rooms = list(dict.fromkeys(room_mentions))
            if len(unique_rooms) == 1:
                only_room_rows.append((row, unique_rooms[0]))

    return {
        "source_file": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(rows),
        "last_container_mental_container_questions": len(last_container_rows),
        "last_container_accuracy": sum(
            prediction.lower() == row["expected_answer"].lower()
            for row, prediction in last_container_rows
        )
        / len(last_container_rows),
        "room_location_questions": room_questions,
        "only_room_coverage": len(only_room_rows),
        "only_room_coverage_rate": len(only_room_rows) / room_questions,
        "only_room_accuracy": sum(
            prediction.lower() == row["expected_answer"].lower()
            for row, prediction in only_room_rows
        )
        / len(only_room_rows),
    }


def validate(records: list[dict[str, Any]], num_pairs: int) -> None:
    if len(records) != num_pairs * 2:
        raise ValueError(f"Expected {num_pairs * 2} records, got {len(records)}")
    pairs: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        pairs.setdefault(row["pair_id"], []).append(row)
        parsed_choices = dict(
            re.findall(r"(?:^|, )([A-F])\. ([^,]+)", row["choices"])
        )
        if parsed_choices.get(row["answer_letter"]) != row["answer"]:
            raise ValueError(f"Invalid answer/choice mapping: {row['sample_id']}")
        if len(set(row["rooms"])) != 2:
            raise ValueError(f"Only-room not neutralized: {row['sample_id']}")

    if len(pairs) != num_pairs:
        raise ValueError(f"Expected {num_pairs} pairs, got {len(pairs)}")
    for pair_id, pair in pairs.items():
        if {row["intervention_type"] for row in pair} != {"observed", "hidden"}:
            raise ValueError(f"Incomplete pair: {pair_id}")
        observed = next(row for row in pair if row["intervention_type"] == "observed")
        hidden = next(row for row in pair if row["intervention_type"] == "hidden")
        if observed["answer"] != observed["counterfactual_final_container"]:
            raise ValueError(f"Observed label mismatch: {pair_id}")
        if hidden["answer"] != hidden["counterfactual_anchor_container"]:
            raise ValueError(f"Hidden label mismatch: {pair_id}")
        if observed["counterfactual_final_container"] != hidden["counterfactual_final_container"]:
            raise ValueError(f"Pair uses different final containers: {pair_id}")

    expected_order_count = num_pairs
    order_counts = Counter(row["question_order"] for row in records)
    if order_counts != Counter({1: expected_order_count, 2: expected_order_count}):
        raise ValueError(f"Unexpected question-order counts: {dict(order_counts)}")
    label_counts = Counter(row["answer_letter"] for row in records)
    expected_per_label = len(records) // 6
    if label_counts != Counter({letter: expected_per_label for letter in "ABCDEF"}):
        raise ValueError(f"Answer labels are not balanced: {dict(label_counts)}")


def repository_commit(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exploretom-repo", type=Path, required=True)
    parser.add_argument("--official-sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-records", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_records % 4:
        raise ValueError("num-records must be divisible by 4 for pair/order balance")
    num_pairs = args.num_records // 2
    records = generate_records(args.exploretom_repo, args.seed, num_pairs)
    validate(records, num_pairs)
    official_audit = audit_official_sample(args.official_sample)
    metadata = {
        "name": "ExploreToM Counterfactual Train 1200",
        "source_repository": "https://github.com/facebookresearch/ExploreToM",
        "source_commit": repository_commit(args.exploretom_repo),
        "license": "CC-BY-NC-4.0",
        "seed": args.seed,
        "records": len(records),
        "pairs": num_pairs,
        "split_counts": dict(Counter(row["split"] for row in records)),
        "order_counts": dict(Counter(str(row["question_order"]) for row in records)),
        "intervention_counts": dict(
            Counter(row["intervention_type"] for row in records)
        ),
        "answer_label_counts": dict(Counter(row["answer_letter"] for row in records)),
        "unique_story_structures": len({row["story_structure"] for row in records}),
        "generated_last_container_accuracy": sum(
            row["last_container_prediction"] == row["answer"] for row in records
        )
        / len(records),
        "generated_only_room_coverage": sum(len(set(row["rooms"])) == 1 for row in records),
        "official_sample_shortcut_audit": official_audit,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", records)
    (args.output_dir / "ExploreToM_counterfactual_train_1200.json").write_text(
        json.dumps({"metadata": metadata, "data": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
