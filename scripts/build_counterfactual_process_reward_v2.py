#!/usr/bin/env python3
"""Rebuild counterfactual stories without intervention-position shortcuts.

The source dataset is treated as immutable. This script removes the old
counterfactual suffix and writes a new process-reward dataset whose paired
variants have the same event count, final-move position, and neutral tail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from build_hi_tom_counterfactual import (
    normalize_story,
    question_agents,
    question_object,
    serial_join,
    story_agents,
)


GENERATION_VERSION = "2.0"
OLD_HI_TOM_SUFFIX_EVENTS = 6
EXPLICIT_LABEL_CUES = (
    "none of the listed people saw",
    "every listed observer saw",
    "clearly observed this final move",
    "in a sealed room",
)
TARGET_RATIO_BUCKETS = (0.60, 0.68, 0.76, 0.84)
NEUTRAL_TEMPLATES = (
    "{first} checked the time.",
    "{second} sat near the doorway.",
    "{first} and {second} discussed the weather.",
    "A bell rang in the hallway.",
    "{second} picked up a magazine.",
)


def stable_int(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            records.append(value)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
            )
            count += 1
    return count


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_sentences(story: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=\.)\s+", story.strip())
        if part.strip()
    ]


def continuation_count(pair_id: str, seed: int) -> int:
    return 2 + stable_int(seed, pair_id, "continuation-count") % 5


def neutral_events(
    agents: list[str], count: int, pair_id: str, seed: int
) -> list[str]:
    if count < 1:
        return []
    first = agents[stable_int(seed, pair_id, "neutral-first") % len(agents)]
    remaining = [agent for agent in agents if agent != first]
    second = remaining[stable_int(seed, pair_id, "neutral-second") % len(remaining)]
    offset = stable_int(seed, pair_id, "neutral-template") % len(NEUTRAL_TEMPLATES)
    return [
        NEUTRAL_TEMPLATES[(offset + index) % len(NEUTRAL_TEMPLATES)].format(
            first=first, second=second
        )
        for index in range(count)
    ]


def choose_focus_and_control(
    agents: list[str], queried_agents: list[str]
) -> tuple[list[str], list[str]]:
    if queried_agents:
        focus = list(dict.fromkeys(queried_agents))
    else:
        # Order-0 labels do not depend on visibility, but retaining two groups
        # keeps both pair variants structurally symmetric.
        focus = agents[: max(1, len(agents) // 2)]
    control = [agent for agent in agents if agent not in focus]
    if not control:
        raise ValueError("Counterfactual construction needs at least one control agent")
    return focus, control


def move_group(group: list[str], source: str, destination: str) -> str:
    names = serial_join(group)
    return f"{names} left the {source} and entered the {destination}."


def numbered(events: list[str], start: int = 1) -> list[str]:
    return [f"{index} {event}" for index, event in enumerate(events, start=start)]


def strip_hi_tom_suffix(record: dict[str, Any]) -> list[str]:
    lines = [
        line.strip()
        for line in normalize_story(record["story"]).splitlines()
        if line.strip()
    ]
    if len(lines) <= OLD_HI_TOM_SUFFIX_EVENTS:
        raise ValueError(f"Hi-ToM story is too short: {record['global_sample_id']}")
    suffix = lines[-OLD_HI_TOM_SUFFIX_EVENTS:]
    if "entered the observation_room together" not in suffix[0]:
        raise ValueError(f"Unexpected Hi-ToM suffix: {record['global_sample_id']}")
    return [
        re.sub(r"^\d+\s+", "", line)
        for line in lines[:-OLD_HI_TOM_SUFFIX_EVENTS]
    ]


def first_object_mention(events: list[str], obj: str) -> int:
    pattern = re.compile(rf"\b{re.escape(obj)}\b", re.IGNORECASE)
    for index, event in enumerate(events, start=1):
        if pattern.search(event):
            return index
    raise ValueError(f"Object {obj!r} does not occur in the source prefix")


def required_hi_tom_prefix_length(
    events: list[str], obj: str, queried_agents: list[str]
) -> int:
    minimum = first_object_mention(events, obj)
    for agent in queried_agents:
        exit_pattern = re.compile(
            rf"^{re.escape(agent)} (?:exited|left|went out of)\b",
            re.IGNORECASE,
        )
        try:
            exit_index = next(
                index
                for index, event in enumerate(events, start=1)
                if exit_pattern.search(event)
            )
        except StopIteration as exc:
            raise ValueError(f"Could not find the first source exit for {agent}") from exc
        minimum = max(minimum, exit_index)
    return minimum


def choose_hi_tom_prefix_length(
    events: list[str], minimum: int, tail_count: int, group_id: str, seed: int
) -> int:
    target = TARGET_RATIO_BUCKETS[
        stable_int(seed, group_id, "position-bucket") % len(TARGET_RATIO_BUCKETS)
    ]
    # The final move is prefix+4 and total length is prefix+4+tail_count.
    ideal = round(target * tail_count / (1.0 - target) - 4)
    return max(minimum, min(len(events), ideal))


def hi_tom_story(
    record: dict[str, Any], prefix: list[str], tail_count: int, seed: int
) -> tuple[str, dict[str, Any]]:
    agents = story_agents("\n".join(numbered(prefix)))
    queried = question_agents(record["question"])
    focus, control = choose_focus_and_control(agents, queried)
    before = control if record["intervention_type"] == "observed" else focus
    after = focus if record["intervention_type"] == "observed" else control
    obj = question_object(record["question"])
    anchor = str(record["counterfactual_anchor_container"])
    final = str(record["counterfactual_container"])
    group = serial_join(agents)
    common = [
        f"After the earlier activity ended, {group} entered the observation_room together.",
        f"An automated system moved the {obj} to the {anchor}.",
        move_group(before, "observation_room", "waiting_room"),
        f"An automated system moved the {obj} to the {final}.",
        move_group(after, "observation_room", "waiting_room"),
    ]
    neutral = neutral_events(agents, tail_count - 1, record["global_pair_id"], seed)
    events = [*prefix, *common, *neutral]
    final_index = len(prefix) + 4
    return "\n".join(numbered(events)) + "\n\n", {
        "intervention_event_index": final_index,
        "story_event_count": len(events),
        "events_after_intervention": len(events) - final_index,
        "neutral_continuation_events": len(neutral),
    }


def parse_exploretom_people(record: dict[str, Any]) -> tuple[list[str], str]:
    events = split_sentences(record["story"])
    if len(events) < 6:
        raise ValueError(f"ExploreToM story is too short: {record['global_sample_id']}")
    entrants = []
    for event in events[:4]:
        match = re.match(r"^([A-Z][a-z]+) entered the ", event)
        if not match:
            raise ValueError(f"Unexpected ExploreToM entry event: {event!r}")
        entrants.append(match.group(1))
    # Upstream order is distractor, outer, inner, mover.
    distractor, outer, inner, mover = entrants
    queried = list(record["qprop=params"][0])
    expected = [outer] if int(record["question_order"]) == 1 else [outer, inner]
    if queried != expected:
        raise ValueError(
            f"Unexpected queried-agent order: {record['global_sample_id']}"
        )
    return [outer, inner, mover, distractor], mover


def exploretom_story(
    record: dict[str, Any], tail_count: int, seed: int
) -> tuple[str, dict[str, Any]]:
    agents, mover = parse_exploretom_people(record)
    outer, inner, _, distractor = agents
    queried = list(record["qprop=params"][0])
    focus, _ = choose_focus_and_control(agents, queried)
    control = [distractor]
    before = control if record["intervention_type"] == "observed" else focus
    after = focus if record["intervention_type"] == "observed" else control
    primary, secondary = record["rooms"]
    obj = str(record["qprop=params"][1])
    initial = next(
        value
        for value in re.findall(
            rf"moved the {re.escape(obj)} to the (.*?), which is also located in",
            record["story"],
            re.IGNORECASE,
        )
        if value not in {
            record["counterfactual_anchor_container"],
            record["counterfactual_final_container"],
        }
    )
    anchor = str(record["counterfactual_anchor_container"])
    final = str(record["counterfactual_final_container"])
    events = [
        f"{distractor} entered the {primary}.",
        f"{outer} entered the {primary}.",
        f"{inner} entered the {primary}.",
        f"{mover} entered the {primary}.",
        (
            f"{mover} moved the {obj} to the {initial}, which is also located "
            f"in the {primary}."
        ),
        (
            f"{mover} moved the {obj} to the {anchor}, which is also located "
            f"in the {primary}."
        ),
        move_group(before, primary, secondary),
        (
            f"{mover} moved the {obj} to the {final}, which is also located "
            f"in the {primary}."
        ),
        move_group(after, primary, secondary),
    ]
    neutral = neutral_events(agents, tail_count - 1, record["global_pair_id"], seed)
    events.extend(neutral)
    final_index = 8
    return " ".join(events), {
        "intervention_event_index": final_index,
        "story_event_count": len(events),
        "events_after_intervention": len(events) - final_index,
        "neutral_continuation_events": len(neutral),
    }


def compact_process_prompt(record: dict[str, Any]) -> str:
    instruction = (
        "Return exactly one JSON object and no markdown or extra text. Infer all values "
        "from the story. Copy names exactly. belief_chain lists characters from the "
        "outermost thinker to the innermost thinker. final_move_observed is true only "
        "when every character in that chain was present for the final object move.\n\n"
        "World-state schema:\n"
        '{"tom_order":0,"belief_chain":[],"object":"...","reasoning_mode":"world_state",'
        '"world_state":"...","answer":"..."}\n'
        "Belief schema:\n"
        '{"tom_order":1,"belief_chain":["..."],"object":"...","reasoning_mode":"belief",'
        '"final_move_observed":true,"nested_belief":"...","answer":"..."}'
    )
    return (
        f"{instruction}\n\nStory:\n{record['story'].rstrip()}\n\n"
        f"Question: {record['question']}\nChoices: {record['choices']}\n"
    )


def rebuild_standard_prompt(record: dict[str, Any]) -> str:
    return (
        "Read the following story and answer the multiple-choice question. "
        "Think step-by-step. Provide the answer first, and then explain it.\n"
        f"Story:\n{record['story'].rstrip()}\n\nQuestion: {record['question']}\n"
        f"Choices: {record['choices']}\n"
    )


class TokenCounter:
    def __init__(self, tokenizer_name: str | None):
        self.name = tokenizer_name or "conservative-character-estimate"
        self.tokenizer = None
        if tokenizer_name:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                local_files_only=Path(tokenizer_name).exists(),
                use_fast=True,
            )

    def raw_count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        # English BPEs are normally below this deliberately conservative estimate.
        return math.ceil(len(text.encode("utf-8")) / 3)

    def sequence_count(self, prompt: str, response: str) -> int:
        if self.tokenizer is None:
            return self.raw_count(prompt) + self.raw_count(response) + 32
        try:
            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            prompt_count = len(prompt_ids)
        except (ImportError, TypeError, ValueError):
            wrapper = (
                "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
                "You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
            )
            prompt_count = self.raw_count(wrapper)
        return prompt_count + self.raw_count(response) + 1


def enrich_record(
    record: dict[str, Any], story: str, audit: dict[str, Any], counter: TokenCounter
) -> dict[str, Any]:
    enriched = dict(record)
    enriched.update(audit)
    enriched["counterfactual_generation_version"] = GENERATION_VERSION
    enriched["story"] = story
    if enriched.get("source_dataset") == "exploretom":
        enriched["story_structure"] = story
    enriched["intervention_relative_position"] = round(
        audit["intervention_event_index"] / audit["story_event_count"], 6
    )
    enriched["prompt"] = rebuild_standard_prompt(enriched)
    enriched["process_prompt"] = compact_process_prompt(enriched)
    enriched["process_prompt_token_count"] = counter.raw_count(
        enriched["process_prompt"]
    )
    enriched["process_sequence_token_count"] = counter.sequence_count(
        enriched["process_prompt"], enriched["process_response"]
    )
    return enriched


def transform_pair(
    pair: list[dict[str, Any]], seed: int, counter: TokenCounter
) -> list[dict[str, Any]]:
    if len(pair) != 2 or {row["intervention_type"] for row in pair} != {
        "observed",
        "hidden",
    }:
        raise ValueError(f"Incomplete pair: {pair[0].get('global_pair_id', '<unknown>')}")
    pair_id = pair[0]["global_pair_id"]
    tail_count = continuation_count(pair_id, seed)
    source = pair[0]["source_dataset"]
    output = []
    if source == "hi-tom":
        base_by_type = {
            row["intervention_type"]: strip_hi_tom_suffix(row) for row in pair
        }
        if base_by_type["observed"] != base_by_type["hidden"]:
            raise ValueError(f"Pair has different Hi-ToM source stories: {pair_id}")
        base = base_by_type["observed"]
        obj = question_object(pair[0]["question"])
        queried = question_agents(pair[0]["question"])
        minimum = required_hi_tom_prefix_length(base, obj, queried)
        # Retain every queried agent's original first exit so that re-entry,
        # rather than the earliest-exit shortcut, remains necessary. Increase
        # the continuation when needed to keep the intervention below 90%.
        tail_count = max(
            tail_count,
            math.ceil((minimum + 4) * (1.0 / 0.90 - 1.0)),
        )
        if tail_count > 6:
            raise ValueError(f"Cannot position Hi-ToM intervention below 90%: {pair_id}")
        prefix_length = choose_hi_tom_prefix_length(
            base, minimum, tail_count, str(pair[0]["source_group_id"]), seed
        )
        prefix = base[:prefix_length]
        for row in pair:
            story, audit = hi_tom_story(row, prefix, tail_count, seed)
            audit["source_prefix_event_count"] = prefix_length
            output.append(enrich_record(row, story, audit, counter))
    elif source == "exploretom":
        for row in pair:
            story, audit = exploretom_story(row, tail_count, seed)
            audit["source_prefix_event_count"] = 6
            output.append(enrich_record(row, story, audit, counter))
    else:
        raise ValueError(f"Unknown source dataset: {source!r}")
    return output


def validate(
    records_by_split: dict[str, list[dict[str, Any]]], max_tokens: int
) -> dict[str, Any]:
    pairs: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    all_records = [row for rows in records_by_split.values() for row in rows]
    for split, rows in records_by_split.items():
        for row in rows:
            if row["split"] != split:
                raise ValueError(f"Split mismatch: {row['global_sample_id']}")
            if not 2 <= int(row["events_after_intervention"]) <= 6:
                raise ValueError(f"Continuation out of range: {row['global_sample_id']}")
            if row["process_prompt_token_count"] > max_tokens:
                raise ValueError(
                    f"Prompt exceeds {max_tokens} tokens: {row['global_sample_id']}"
                )
            if row["process_sequence_token_count"] > max_tokens:
                raise ValueError(
                    f"Sequence exceeds {max_tokens} tokens: {row['global_sample_id']}"
                )
            lowered = row["story"].lower()
            if any(cue in lowered for cue in EXPLICIT_LABEL_CUES):
                raise ValueError(f"Explicit visibility cue remains: {row['global_sample_id']}")
            if json.loads(row["process_response"]) != row["process_target"]:
                raise ValueError(f"Process response mismatch: {row['global_sample_id']}")
            target = row["process_target"]
            final = (
                row["counterfactual_container"]
                if row["source_dataset"] == "hi-tom"
                else row["counterfactual_final_container"]
            )
            anchor = row["counterfactual_anchor_container"]
            if int(row["question_order"]) == 0:
                if target.get("world_state") != final or target.get("answer") != final:
                    raise ValueError(
                        f"Incorrect world-state target: {row['global_sample_id']}"
                    )
            else:
                observed = row["intervention_type"] == "observed"
                expected = final if observed else anchor
                if (
                    target.get("final_move_observed") is not observed
                    or target.get("nested_belief") != expected
                    or target.get("answer") != expected
                ):
                    raise ValueError(f"Incorrect belief target: {row['global_sample_id']}")
            pairs[row["global_pair_id"]].append(row)

    for pair_id, pair in pairs.items():
        if len(pair) != 2 or {row["intervention_type"] for row in pair} != {
            "observed",
            "hidden",
        }:
            raise ValueError(f"Incomplete transformed pair: {pair_id}")
        observed = next(row for row in pair if row["intervention_type"] == "observed")
        hidden = next(row for row in pair if row["intervention_type"] == "hidden")
        for field in (
            "intervention_event_index",
            "story_event_count",
            "events_after_intervention",
            "intervention_relative_position",
        ):
            if observed[field] != hidden[field]:
                raise ValueError(f"Pair position mismatch for {field}: {pair_id}")
        observed_events = (
            [line for line in observed["story"].strip().splitlines() if line]
            if observed["source_dataset"] == "hi-tom"
            else split_sentences(observed["story"])
        )
        hidden_events = (
            [line for line in hidden["story"].strip().splitlines() if line]
            if hidden["source_dataset"] == "hi-tom"
            else split_sentences(hidden["story"])
        )
        if observed_events[-1] != hidden_events[-1]:
            raise ValueError(f"Pair has label-dependent final event: {pair_id}")
        if observed["source_dataset"] == "hi-tom":
            observed_bow = Counter(
                re.sub(r"^\d+\s+", "", event) for event in observed_events
            )
            hidden_bow = Counter(
                re.sub(r"^\d+\s+", "", event) for event in hidden_events
            )
        else:
            observed_bow = Counter(observed_events)
            hidden_bow = Counter(hidden_events)
        if observed_bow != hidden_bow:
            raise ValueError(f"Pair has label-dependent event vocabulary: {pair_id}")

    positions_by_type: defaultdict[str, Counter[tuple[int, int]]] = defaultdict(
        Counter
    )
    for row in all_records:
        positions_by_type[row["intervention_type"]][
            (row["intervention_event_index"], row["story_event_count"])
        ] += 1
    if positions_by_type["observed"] != positions_by_type["hidden"]:
        raise ValueError("Observed/hidden intervention-position distributions differ")

    return {
        "records": len(all_records),
        "pairs": len(pairs),
        "split_counts": dict(Counter(row["split"] for row in all_records)),
        "source_counts": dict(
            Counter(row["source_dataset"] for row in all_records)
        ),
        "order_counts": dict(
            Counter(str(row["question_order"]) for row in all_records)
        ),
        "intervention_counts": dict(
            Counter(row["intervention_type"] for row in all_records)
        ),
        "events_after_intervention_counts": dict(
            Counter(str(row["events_after_intervention"]) for row in all_records)
        ),
        "intervention_relative_position": {
            "min": min(row["intervention_relative_position"] for row in all_records),
            "max": max(row["intervention_relative_position"] for row in all_records),
            "mean": round(
                sum(row["intervention_relative_position"] for row in all_records)
                / len(all_records),
                6,
            ),
        },
        "process_prompt_tokens": {
            "max": max(row["process_prompt_token_count"] for row in all_records),
            "mean": round(
                sum(row["process_prompt_token_count"] for row in all_records)
                / len(all_records),
                3,
            ),
        },
        "process_sequence_tokens": {
            "max": max(row["process_sequence_token_count"] for row in all_records),
            "mean": round(
                sum(row["process_sequence_token_count"] for row in all_records)
                / len(all_records),
                3,
            ),
        },
        "pair_position_match_rate": 1.0,
        "pair_final_event_match_rate": 1.0,
        "pair_event_bag_match_rate": 1.0,
        "explicit_label_cue_count": 0,
    }


def build_dataset(
    input_dir: Path,
    output_dir: Path,
    seed: int,
    max_tokens: int,
    tokenizer_name: str | None,
) -> dict[str, Any]:
    source = {
        split: read_jsonl(input_dir / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    counter = TokenCounter(tokenizer_name)
    transformed: dict[str, list[dict[str, Any]]] = {}
    for split, rows in source.items():
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["global_pair_id"]].append(row)
        transformed[split] = []
        for pair_id in sorted(grouped):
            transformed[split].extend(transform_pair(grouped[pair_id], seed, counter))

    audit = validate(transformed, max_tokens)
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in transformed.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    manifest = {
        "name": "Counterfactual Process-Reward Dataset v2",
        "counterfactual_generation_version": GENERATION_VERSION,
        "seed": seed,
        "source_directory": str(input_dir),
        "source_files": {
            split: {
                "path": str(input_dir / f"{split}.jsonl"),
                "sha256": file_sha256(input_dir / f"{split}.jsonl"),
            }
            for split in source
        },
        "tokenizer": counter.name,
        "max_process_and_sequence_tokens": max_tokens,
        "continuation_event_range": [2, 6],
        **audit,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    readme = f"""# Counterfactual Process-Reward Dataset v2

This is an immutable-source rebuild of `{input_dir}`. The source directory was
not modified.

Paired observed/hidden stories have identical final-move positions and event
counts. The final move is followed by 2-6 events, including a label-independent
neutral tail, and the old explicit observed/hidden closing statements are gone.
`process_prompt` is compact and both it and the chat-formatted canonical
training sequence are capped at {max_tokens} tokens using `{counter.name}`.

Rebuild from the repository root with:

```bash
python3 scripts/build_counterfactual_process_reward_v2.py \\
  --input-dir {input_dir} \\
  --output-dir {output_dir} \\
  --tokenizer /path/to/Qwen2.5-tokenizer
```

See `manifest.json` for provenance, position balance, continuation counts, and
token-length statistics.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/counterfactual_process_reward"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/counterfactual_process_reward_v2"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--tokenizer",
        help="Local tokenizer path or Hugging Face ID; omit for a conservative estimate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        args.input_dir,
        args.output_dir,
        args.seed,
        args.max_tokens,
        args.tokenizer,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
