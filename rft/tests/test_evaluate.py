import json
import unittest

from rft.evaluate import evaluate_predictions


class EvaluateTest(unittest.TestCase):
    def test_perfect_counterfactual_pair(self):
        rows = []
        for side, observed, answer in (
            ("observed", True, "linen chest"),
            ("hidden", False, "archive drawer"),
        ):
            target = {
                "tom_order": 1,
                "belief_chain": ["Alice"],
                "object": "passport",
                "reasoning_mode": "belief",
                "final_move_observed": observed,
                "nested_belief": answer,
                "answer": answer,
            }
            rows.append(
                {
                    "global_sample_id": f"sample-{side}",
                    "global_pair_id": "pair-1",
                    "source_dataset": "hi-tom",
                    "question_order": 1,
                    "intervention_type": side,
                    "process_target": target,
                    "response": json.dumps(target),
                    "generation_reached_eos": True,
                    "token_count": 20,
                }
            )
        overall = evaluate_predictions(rows)["overall"]
        for key in (
            "parse_rate",
            "strict_format_rate",
            "full_reward_rate",
            "answer_accuracy",
            "core_state_accuracy",
            "pair_accuracy",
            "intervention_sensitivity",
            "answer_state_consistency",
            "eos_rate",
        ):
            self.assertEqual(overall[key], 1.0, key)

    def test_perfect_nested_belief_pair(self):
        rows = []
        for side, inner, answer in (
            ("observed", "linen chest", "archive drawer"),
            ("hidden", "metal trunk", "blue suitcase"),
        ):
            target = {
                "tom_order": 2,
                "belief_chain": ["Alice", "Bob"],
                "object": "passport",
                "reasoning_mode": "nested_belief",
                "belief_trace": [
                    {"belief_chain": ["Bob"], "location": inner},
                    {
                        "belief_chain": ["Alice", "Bob"],
                        "location": answer,
                    },
                ],
                "answer": answer,
            }
            rows.append(
                {
                    "global_sample_id": f"nested-{side}",
                    "global_pair_id": "nested-pair-1",
                    "source_dataset": "symbolic-tom-v3",
                    "question_order": 2,
                    "intervention_type": side,
                    "process_target": target,
                    "response": json.dumps(target),
                }
            )
        overall = evaluate_predictions(rows)["overall"]
        self.assertEqual(overall["core_state_accuracy"], 1.0)
        self.assertEqual(overall["answer_state_consistency"], 1.0)
        self.assertEqual(overall["pair_accuracy"], 1.0)
        self.assertEqual(overall["intervention_sensitivity"], 1.0)


if __name__ == "__main__":
    unittest.main()
