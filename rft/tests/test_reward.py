import json
import unittest

from rft.reward import score_process_output


class RewardTest(unittest.TestCase):
    def setUp(self):
        self.target = {
            "tom_order": 2,
            "belief_chain": ["Alice", "Bob"],
            "object": "passport",
            "reasoning_mode": "belief",
            "final_move_observed": False,
            "nested_belief": "archive drawer",
            "answer": "archive drawer",
        }

    def test_exact_json_gets_full_reward(self):
        self.assertEqual(score_process_output(json.dumps(self.target), self.target)["reward"], 1.0)

    def test_fence_is_not_accepted(self):
        output = "```json\n" + json.dumps(self.target) + "\n```"
        result = score_process_output(output, self.target)
        self.assertEqual(result["reward"], 0.95)
        self.assertFalse(result["checks"]["format"])

    def test_visibility_gates_nested_state(self):
        prediction = dict(self.target, final_move_observed=True)
        result = score_process_output(prediction, self.target)
        self.assertEqual(result["reward"], 0.25)
        self.assertFalse(result["checks"]["final_move_observed"])

    def test_malformed_json_is_zero(self):
        self.assertEqual(score_process_output("not json", self.target)["reward"], 0.0)
