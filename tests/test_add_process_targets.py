import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "add_process_targets.py"
SPEC = importlib.util.spec_from_file_location("process_targets", SCRIPT)
targets = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(targets)


class ProcessTargetTest(unittest.TestCase):
    def test_hi_tom_belief_target(self) -> None:
        record = {
            "source_dataset": "hi-tom",
            "global_sample_id": "hi-tom:1-hidden",
            "question_order": 2,
            "question": "Where does Alice think Bob thinks the passport is?",
            "counterfactual_anchor_container": "archive_drawer",
            "counterfactual_container": "linen_chest",
            "intervention_type": "hidden",
            "answer": "archive_drawer",
        }
        self.assertEqual(
            targets.build_process_target(record),
            {
                "tom_order": 2,
                "belief_chain": ["Alice", "Bob"],
                "object": "passport",
                "reasoning_mode": "belief",
                "final_move_observed": False,
                "nested_belief": "archive_drawer",
                "answer": "archive_drawer",
            },
        )

    def test_exploretom_belief_target(self) -> None:
        record = {
            "source_dataset": "exploretom",
            "global_sample_id": "exploretom:1-observed",
            "question_order": 1,
            "qprop=params": [["Henry"], "passport", "container_location-True"],
            "counterfactual_anchor_container": "archive drawer",
            "counterfactual_final_container": "linen chest",
            "intervention_type": "observed",
            "answer": "linen chest",
        }
        target = targets.build_process_target(record)
        self.assertEqual(target["belief_chain"], ["Henry"])
        self.assertTrue(target["final_move_observed"])
        self.assertEqual(target["nested_belief"], "linen chest")

    def test_order_zero_uses_world_state(self) -> None:
        record = {
            "source_dataset": "hi-tom",
            "global_sample_id": "hi-tom:0-hidden",
            "question_order": 0,
            "question": "Where is the passport really?",
            "counterfactual_anchor_container": "archive_drawer",
            "counterfactual_container": "linen_chest",
            "intervention_type": "hidden",
            "answer": "linen_chest",
        }
        target = targets.build_process_target(record)
        self.assertEqual(target["reasoning_mode"], "world_state")
        self.assertEqual(target["belief_chain"], [])
        self.assertEqual(target["world_state"], "linen_chest")


if __name__ == "__main__":
    unittest.main()
