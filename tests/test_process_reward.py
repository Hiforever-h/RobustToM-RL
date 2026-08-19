import importlib.util
import json
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "process_reward.py"
SPEC = importlib.util.spec_from_file_location("process_reward", SCRIPT)
reward = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(reward)


class ProcessRewardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.target = {
            "tom_order": 2,
            "belief_chain": ["Alice", "Bob"],
            "object": "passport",
            "reasoning_mode": "belief",
            "final_move_observed": False,
            "nested_belief": "archive drawer",
            "answer": "archive drawer",
        }

    def test_exact_belief_process_receives_full_reward(self) -> None:
        result = reward.score_process_output(json.dumps(self.target), self.target)
        self.assertEqual(result["reward"], 1.0)

    def test_answer_is_gated_when_visibility_is_wrong(self) -> None:
        prediction = dict(self.target)
        prediction["final_move_observed"] = True
        result = reward.score_process_output(prediction, self.target)
        self.assertEqual(result["reward"], 0.25)
        self.assertEqual(result["components"]["nested_belief"], 0.0)
        self.assertEqual(result["components"]["answer"], 0.0)

    def test_world_answer_is_gated_by_world_state(self) -> None:
        target = {
            "tom_order": 0,
            "belief_chain": [],
            "object": "passport",
            "reasoning_mode": "world_state",
            "world_state": "linen chest",
            "answer": "linen chest",
        }
        prediction = dict(target)
        prediction["world_state"] = "archive drawer"
        result = reward.score_process_output(prediction, target)
        self.assertEqual(result["reward"], 0.25)
        self.assertEqual(result["components"]["answer"], 0.0)

    def test_markdown_fence_loses_only_format_reward(self) -> None:
        output = f"```json\n{json.dumps(self.target)}\n```"
        result = reward.score_process_output(output, self.target)
        self.assertEqual(result["reward"], 0.95)

if __name__ == "__main__":
    unittest.main()
