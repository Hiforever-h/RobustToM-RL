#!/usr/bin/env python3
"""Generate a 3,000-record counterfactual Hi-ToM dataset from upstream code.

The final split is fixed by design:

* orders 0..3: 2,000 train and 400 validation records;
* order 4: 600 OOD test records generated from disjoint base stories.

Every base question is expanded into an observed/hidden counterfactual pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_hi_tom_counterfactual import (
    choose_counterfactual_containers,
    make_record,
    parse_choices,
    shortcut_prediction,
    validate_records,
)


PROMPT_PREFIX = (
    "Read the following story and answer the multiple-choice question. "
    "Think step-by-step. Provide the answer first, and then explain it.\n"
)
NOTE = (
    "Note: You should assume the following. (1) An agent witnesses everything "
    "and every movement before exiting a location. (2) An agent A can infer "
    "another agent B's mental state only if A and B have been in the same "
    "location, or have private or public interactions. (3) Note that every "
    "agent tends to lie. What an agent A tells others doesn't affect A's actual "
    "belief. An agent tends to trust an agent that exited the room later than "
    "himself. The exit order is known to all agents. (4) Agents in private "
    "communications know that others won't hear them, but they know that anyone "
    "can hear any public claims.\n"
)
QUESTION_RE = re.compile(r"^Question: (.+)\nAnswer: ([a-z_]+)$")


def load_upstream(repo: Path) -> tuple[Any, Any, Any]:
    required = [
        repo / "world_large.txt",
        repo / "world.py",
        repo / "tasks.py",
        repo / "stringify.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete Hi-ToM checkout; missing: {missing}")
    sys.path.insert(0, str(repo.resolve()))
    from stringify import stringify  # type: ignore
    from tasks import Specify_Tasks  # type: ignore
    from world import World  # type: ignore

    return World, Specify_Tasks, stringify


def render_source_row(
    story: list[Any],
    stringify: Any,
    order: int,
    group: str,
    deception: bool,
    story_length: int,
    sample_id: int,
    pool: str,
) -> dict[str, Any]:
    lines = stringify(story, exist_answer=True, order=order)
    question_index = next(
        index for index, line in enumerate(lines) if line.startswith("Question:")
    )
    question_match = QUESTION_RE.match(lines[question_index].strip())
    if not question_match:
        raise ValueError(f"Could not parse generated question: {lines[question_index]!r}")
    question, answer = question_match.groups()
    choices_line = next(line for line in lines if line.startswith("Choices:"))
    choices = choices_line.removeprefix("Choices:").strip()
    story_text = "\n".join(line.rstrip() for line in lines[:question_index]).strip()
    story_text += "\n\n"
    prompt = (
        f"{PROMPT_PREFIX}Story:\n{story_text.rstrip()}\n\n"
        f"Question: {question}\nChoices: {choices}\n\n{NOTE}"
    )
    return {
        "prompting_type": "CoTP",
        "deception": deception,
        "story_length": story_length,
        "question_order": order,
        "sample_id": sample_id,
        "source_group_id": group,
        "generation_pool": pool,
        "story": story_text,
        "question": question,
        "choices": choices,
        "answer": answer,
        "prompt": prompt,
    }


def generate_source_rows(
    repo: Path,
    seed: int,
    scenarios_per_stratum: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    World, Specify_Tasks, stringify = load_upstream(repo)
    random.seed(seed)
    np.random.seed(seed)

    world = World()
    world.load(str(repo / "world_large.txt"))
    task = Specify_Tasks()
    tasks_by_length = {
        1: [("A5", True)],
        2: [("A5", False), ("A3", True)],
        3: [("A5", True), ("A3", False), ("A4", True)],
    }
    pools = {
        "lower_orders": (0, 1, 2, 3),
        "order4_ood": (4,),
    }

    rows: list[dict[str, Any]] = []
    story_hashes: dict[str, str] = {}
    sample_id = 0
    for pool, orders in pools.items():
        for deception in (False, True):
            for story_length in (1, 2, 3):
                for scenario in range(scenarios_per_stratum):
                    group = (
                        f"pool={pool}|deception={str(deception).lower()}|"
                        f"length={story_length}|scenario={scenario:03d}"
                    )
                    story = task.generate_story_qs_at_end(
                        world,
                        story_length,
                        tasks_by_length[story_length],
                        num_agents=5,
                        num_locations=3,
                        statement_noise=0.1,
                        order=0,
                        exist_tell_in_story=deception,
                    )
                    group_rows = [
                        render_source_row(
                            story,
                            stringify,
                            order,
                            group,
                            deception,
                            story_length,
                            sample_id + offset,
                            pool,
                        )
                        for offset, order in enumerate(orders)
                    ]
                    sample_id += len(group_rows)
                    rows.extend(group_rows)
                    canonical_story = group_rows[0]["story"].strip()
                    story_hashes[group] = hashlib.sha256(
                        canonical_story.encode()
                    ).hexdigest()

    lower_hashes = {
        digest for group, digest in story_hashes.items() if "pool=lower_orders" in group
    }
    ood_hashes = {
        digest for group, digest in story_hashes.items() if "pool=order4_ood" in group
    }
    overlap = lower_hashes & ood_hashes
    if overlap:
        raise ValueError(f"Generated story leakage into order-4 OOD pool: {len(overlap)}")
    return rows, story_hashes


def assign_lower_splits(
    groups: dict[str, list[dict[str, Any]]], seed: int, validation_groups: int
) -> dict[str, str]:
    strata: dict[tuple[bool, int], list[str]] = defaultdict(list)
    for group, rows in groups.items():
        strata[(bool(rows[0]["deception"]), int(rows[0]["story_length"]))].append(group)

    total_groups = sum(len(group_ids) for group_ids in strata.values())
    if validation_groups >= total_groups:
        raise ValueError("Validation set must be smaller than the lower-order pool")

    exact = {
        stratum: validation_groups * len(group_ids) / total_groups
        for stratum, group_ids in strata.items()
    }
    allocation = {stratum: int(value) for stratum, value in exact.items()}
    remainder = validation_groups - sum(allocation.values())
    ranked = sorted(
        strata,
        key=lambda stratum: (-(exact[stratum] - allocation[stratum]), stratum),
    )
    for stratum in ranked[:remainder]:
        allocation[stratum] += 1

    split_by_group: dict[str, str] = {}
    for stratum, group_ids in sorted(strata.items()):
        group_ids = sorted(group_ids)
        rng = random.Random(f"{seed}|split|{stratum[0]}|{stratum[1]}")
        rng.shuffle(group_ids)
        validation_count = allocation[stratum]
        for index, group in enumerate(group_ids):
            split_by_group[group] = (
                "validation" if index < validation_count else "train"
            )
    return split_by_group


def build_counterfactual_records(
    source_rows: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[row["source_group_id"]].append(row)

    lower_groups = {
        group: rows
        for group, rows in grouped.items()
        if rows[0]["generation_pool"] == "lower_orders"
    }
    split_by_group = assign_lower_splits(lower_groups, seed, validation_groups=50)
    for group, rows in grouped.items():
        if rows[0]["generation_pool"] == "order4_ood":
            split_by_group[group] = "order4_ood_test"

    records: list[dict[str, Any]] = []
    for group in sorted(grouped):
        rows = sorted(grouped[group], key=lambda row: int(row["question_order"]))
        anchor, final = choose_counterfactual_containers(rows, group, seed)
        for row in rows:
            for intervention in ("observed", "hidden"):
                records.append(
                    make_record(
                        row,
                        group,
                        anchor,
                        final,
                        intervention,
                        split_by_group[group],
                    )
                )
    validate_records(records)
    return records


def validate_final_dataset(
    source_rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> None:
    expected_split_counts = {
        "train": 2000,
        "validation": 400,
        "order4_ood_test": 600,
    }
    split_counts = Counter(row["split"] for row in records)
    if dict(split_counts) != expected_split_counts:
        raise ValueError(f"Unexpected split counts: {dict(split_counts)}")

    order_counts = Counter(int(row["question_order"]) for row in records)
    if order_counts != Counter({order: 600 for order in range(5)}):
        raise ValueError(f"Unexpected order counts: {dict(order_counts)}")

    if any(
        int(row["question_order"]) == 4 and row["split"] != "order4_ood_test"
        for row in records
    ):
        raise ValueError("Some order-4 records are outside the OOD test split")
    if any(
        int(row["question_order"]) != 4 and row["split"] == "order4_ood_test"
        for row in records
    ):
        raise ValueError("The OOD test split contains lower-order records")

    groups_by_split: dict[str, set[str]] = defaultdict(set)
    for row in records:
        groups_by_split[row["split"]].add(row["source_group_id"])
        choices = {value for _, value in parse_choices(row["choices"])}
        if row["answer"] not in choices:
            raise ValueError(f"Answer not found in choices: {row['sample_id']}")
    split_names = list(groups_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = groups_by_split[left] & groups_by_split[right]
            if overlap:
                raise ValueError(f"Group leakage between {left} and {right}")

    if len(source_rows) != 1500 or len(records) != 3000:
        raise ValueError(
            f"Unexpected source/final sizes: {len(source_rows)}/{len(records)}"
        )


def build_metadata(
    repo: Path,
    seed: int,
    source_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    story_hashes: dict[str, str],
) -> dict[str, Any]:
    higher = [row for row in records if int(row["question_order"]) > 0]
    covered = [row for row in higher if row["shortcut_prediction"] is not None]
    commit_result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    source_commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    return {
        "name": "Hi-ToM Counterfactual 3000",
        "source_repository": "https://github.com/ying-hui-he/Hi-ToM_dataset",
        "source_checkout": str(repo),
        "source_commit": source_commit,
        "seed": seed,
        "base_story_counts": {
            "lower_orders": 300,
            "order4_ood": 300,
        },
        "generated_source_questions": len(source_rows),
        "generated_records": len(records),
        "split_counts": dict(Counter(row["split"] for row in records)),
        "order_counts": {
            str(order): count
            for order, count in sorted(
                Counter(int(row["question_order"]) for row in records).items()
            )
        },
        "split_order_counts": {
            f"{split}|order={order}": count
            for (split, order), count in sorted(
                Counter(
                    (row["split"], int(row["question_order"])) for row in records
                ).items()
            )
        },
        "intervention_counts": dict(
            Counter(row["intervention_type"] for row in records)
        ),
        "unique_source_story_hashes": len(set(story_hashes.values())),
        "higher_order_earliest_exit_shortcut_accuracy": sum(
            row["shortcut_prediction"] == row["answer"] for row in covered
        )
        / len(covered),
        "higher_order_last_mention_shortcut_accuracy": sum(
            row["last_mentioned_container"] == row["answer"] for row in higher
        )
        / len(higher),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hi-tom-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scenarios-per-stratum", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scenarios_per_stratum != 50:
        raise ValueError(
            "The requested 3,000-record split requires --scenarios-per-stratum 50"
        )
    source_rows, story_hashes = generate_source_rows(
        args.hi_tom_repo, args.seed, args.scenarios_per_stratum
    )
    records = build_counterfactual_records(source_rows, args.seed)
    validate_final_dataset(source_rows, records)
    metadata = build_metadata(
        args.hi_tom_repo, args.seed, source_rows, records, story_hashes
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "Hi-ToM_counterfactual_3000.json").write_text(
        json.dumps({"metadata": metadata, "data": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.output_dir / "generated_source.jsonl", source_rows)
    for split, filename in (
        ("train", "train.jsonl"),
        ("validation", "validation.jsonl"),
        ("order4_ood_test", "order4_ood_test.jsonl"),
    ):
        write_jsonl(
            args.output_dir / filename,
            (row for row in records if row["split"] == split),
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
