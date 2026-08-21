import csv
import json
import tempfile
import unittest
from pathlib import Path

from grpo.prompt import ORDER_TRACE_INSTRUCTION
from rft.prepare_hitom_eval import prepare_hitom_eval
from scripts.add_symbolic_v3_few_shots import FEW_SHOT_MARKER


class PrepareHiToMEvalTest(unittest.TestCase):
    def test_builds_order4_few_shot_dataset(self):
        fieldnames = [
            "deception",
            "story_length",
            "question_order",
            "sample_id",
            "story",
            "question",
            "choices",
            "answer",
        ]
        rows = [
            {
                "deception": "False",
                "story_length": "2",
                "question_order": "3",
                "sample_id": "7",
                "story": "A lower-order story.",
                "question": "Where does Alice think Bob thinks Carol thinks the key is?",
                "choices": "A. red_box, B. blue_box",
                "answer": "red_box",
            },
            {
                "deception": "True",
                "story_length": "2",
                "question_order": "4",
                "sample_id": "7",
                "story": "Alice, Bob, Carol, and David watched the key move to the blue_box.",
                "question": (
                    "Where does Alice think Bob thinks Carol thinks David thinks "
                    "the key is?"
                ),
                "choices": "A. red_box, B. blue_box",
                "answer": "blue_box",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "hi_tom.csv"
            with input_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            output_dir = root / "output"
            manifest = prepare_hitom_eval(
                input_path, output_dir, order=4, expected_count=1
            )
            built = [
                json.loads(line)
                for line in (output_dir / "test.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertEqual(manifest["count"], 1)
            self.assertEqual(len(built), 1)
            row = built[0]
            self.assertEqual(
                row["global_sample_id"],
                "hi-tom-order4:deception=true:length=2:sample=007",
            )
            self.assertEqual(
                row["belief_chain"], ["Alice", "Bob", "Carol", "David"]
            )
            self.assertEqual(row["object"], "key")
            self.assertEqual(row["gold_answer"], "blue_box")
            self.assertNotIn("process_target", row)
            self.assertEqual(row["process_prompt"].count(FEW_SHOT_MARKER), 1)
            self.assertEqual(row["process_prompt"].count(ORDER_TRACE_INSTRUCTION), 1)
            self.assertIn(
                "Choices: A. red_box, B. blue_box", row["process_prompt"]
            )


if __name__ == "__main__":
    unittest.main()
