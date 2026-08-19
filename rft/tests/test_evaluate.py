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


if __name__ == "__main__":
    unittest.main()
