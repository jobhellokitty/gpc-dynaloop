import unittest

from gpc_dynaloop.er_sft import build_step_entries, materialize_step


class ERSFTTest(unittest.TestCase):
    def setUp(self):
        self.record = {
            "messages": [
                {"role": "system", "content": "robot"},
                {"role": "user", "content": "<image>first"},
                {"role": "assistant", "content": "act one"},
                {"role": "user", "content": "<image>second"},
                {"role": "assistant", "content": "act two"},
            ],
            "images": ["./data/images/a.png", "./data/images/b.png"],
        }
        self.manifest = {
            "split": "train",
            "source_index": 7,
            "task_type": "single_search",
            "scene": "FloorPlan1",
        }

    def test_builds_one_entry_per_assistant_decision(self):
        entries = build_step_entries(self.record, self.manifest)
        self.assertEqual([entry["image_count"] for entry in entries], [1, 2])
        self.assertEqual(entries[1]["target_message_index"], 4)

    def test_materializes_only_history_through_target(self):
        entry = build_step_entries(self.record, self.manifest)[0]
        materialized = materialize_step(self.record, entry, "/dataset")
        self.assertEqual(len(materialized["messages"]), 3)
        self.assertEqual(
            materialized["messages"][1]["content"][0]["image"],
            "/dataset/data/images/a.png",
        )
        self.assertEqual(materialized["messages"][-1]["content"], "act one")


if __name__ == "__main__":
    unittest.main()
