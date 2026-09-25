import unittest

from gpc_dynaloop.er_subset import image_token_count, parse_image_path, scene_split


class ERSubsetTest(unittest.TestCase):
    def test_parse_image_path(self):
        parsed = parse_image_path(
            "./data/images/single_search/FloorPlan204_single_search_1_b/0_init.png"
        )
        self.assertEqual(parsed["task_type"], "single_search")
        self.assertEqual(parsed["scene"], "FloorPlan204")

    def test_scene_split_is_room_family_relative(self):
        config = {
            "train_offsets": [1, 24],
            "test_offsets": [25, 30],
        }
        self.assertEqual(scene_split("FloorPlan204", config), "train")
        self.assertEqual(scene_split("FloorPlan225", config), "test")
        self.assertEqual(scene_split("FloorPlan330", config), "test")

    def test_image_token_count(self):
        record = {
            "messages": [
                {"role": "user", "content": "<image>first"},
                {"role": "assistant", "content": "observe"},
                {"role": "user", "content": "<image>second"},
            ]
        }
        self.assertEqual(image_token_count(record), 2)


if __name__ == "__main__":
    unittest.main()
