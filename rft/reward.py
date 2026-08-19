"""Deterministic process scorer used by sampling, training-data construction and eval."""

from __future__ import annotations

import json
import re
from typing import Any

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
FENCED_JSON_RE = re.compile(r"^```(?:json)?\s*(\{.*\})\s*```$", re.DOTALL | re.IGNORECASE)


def normalize(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def parse_prediction(output: str | dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    if isinstance(output, dict):
        return output, True
    if not isinstance(output, str):
        return None, False
    text = output.strip()
    strict_json = True
    match = FENCED_JSON_RE.fullmatch(text)
    if match:
        text = match.group(1)
        strict_json = False
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None, False
    return (parsed, strict_json) if isinstance(parsed, dict) else (None, False)


def valid_schema(prediction: dict[str, Any], fields: set[str], mode: str) -> bool:
    if set(prediction) != fields or prediction.get("reasoning_mode") != mode:
        return False
    if type(prediction.get("tom_order")) is not int:
        return False
    if not isinstance(prediction.get("belief_chain"), list) or not all(
        isinstance(item, str) for item in prediction["belief_chain"]
    ):
        return False
    string_fields = fields - {"tom_order", "belief_chain", "final_move_observed"}
    if not all(isinstance(prediction.get(field), str) for field in string_fields):
        return False
    return mode != "belief" or type(prediction.get("final_move_observed")) is bool


def normalized_chain(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return [normalize(item) for item in value]


def validate_target(target: dict[str, Any]) -> tuple[str, set[str]]:
    mode = target.get("reasoning_mode")
    if mode == "world_state":
        fields = WORLD_FIELDS
    elif mode == "belief":
        fields = BELIEF_FIELDS
    else:
        raise ValueError(f"Unknown target reasoning_mode: {mode!r}")
    if not valid_schema(target, fields, mode):
        raise ValueError("Target does not match the process-reward schema")
    order = target["tom_order"]
    chain = target["belief_chain"]
    if mode == "world_state":
        if order != 0 or chain:
            raise ValueError("World-state targets require order 0 and an empty chain")
        if normalize(target["world_state"]) != normalize(target["answer"]):
            raise ValueError("World-state target and answer must agree")
    elif order < 1 or len(chain) != order:
        raise ValueError("Belief-chain length must equal the positive ToM order")
    elif normalize(target["nested_belief"]) != normalize(target["answer"]):
        raise ValueError("Nested-belief target and answer must agree")
    return mode, fields


def score_process_output(output: str | dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Return a scalar reward in [0, 1] plus auditable checks/components."""
    mode, fields = validate_target(target)
    prediction, strict_json = parse_prediction(output)
    if prediction is None:
        return {
            "reward": 0.0,
            "reasoning_mode": mode,
            "checks": {"parseable_json": False, "format": False},
            "components": {"format": 0.0},
        }

    schema_ok = valid_schema(prediction, fields, mode)
    order_ok = type(prediction.get("tom_order")) is int and prediction["tom_order"] == target["tom_order"]
    chain_ok = normalized_chain(prediction.get("belief_chain")) == normalized_chain(target["belief_chain"])
    object_ok = normalize(prediction.get("object", "")) == normalize(target["object"])
    format_ok = strict_json and schema_ok

    if mode == "world_state":
        state_ok = normalize(prediction.get("world_state", "")) == normalize(target["world_state"])
        answer_ok = normalize(prediction.get("answer", "")) == normalize(target["answer"])
        checks = {
            "parseable_json": True,
            "format": format_ok,
            "tom_order": order_ok,
            "belief_chain": chain_ok,
            "object": object_ok,
            "world_state": state_ok,
            "answer": answer_ok,
        }
        components = {
            "format": 0.05 if format_ok else 0.0,
            "tom_order": 0.10 if order_ok else 0.0,
            "object": 0.10 if object_ok else 0.0,
            "world_state": 0.50 if state_ok else 0.0,
            "answer": 0.25 if state_ok and answer_ok else 0.0,
        }
    else:
        visibility_ok = (
            type(prediction.get("final_move_observed")) is bool
            and prediction["final_move_observed"] == target["final_move_observed"]
        )
        belief_ok = normalize(prediction.get("nested_belief", "")) == normalize(target["nested_belief"])
        answer_ok = normalize(prediction.get("answer", "")) == normalize(target["answer"])
        checks = {
            "parseable_json": True,
            "format": format_ok,
            "tom_order": order_ok,
            "belief_chain": chain_ok,
            "object": object_ok,
            "final_move_observed": visibility_ok,
            "nested_belief": belief_ok,
            "answer": answer_ok,
        }
        components = {
            "format": 0.05 if format_ok else 0.0,
            "tom_order": 0.05 if order_ok else 0.0,
            "belief_chain": 0.10 if chain_ok else 0.0,
            "object": 0.05 if object_ok else 0.0,
            "final_move_observed": 0.25 if visibility_ok else 0.0,
            "nested_belief": 0.35 if visibility_ok and belief_ok else 0.0,
            "answer": 0.15 if visibility_ok and belief_ok and answer_ok else 0.0,
        }

    return {
        "reward": round(sum(components.values()), 10),
        "reasoning_mode": mode,
        "checks": checks,
        "components": components,
    }


def process_reward(output: str | dict[str, Any], target: dict[str, Any]) -> float:
    return float(score_process_output(output, target)["reward"])
