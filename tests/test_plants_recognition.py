import unittest

from plants_recognition import recognize_plant, recognize_plants


class PlantRecognitionTests(unittest.TestCase):
    def test_tree_recognition(self):
        self.assertEqual(recognize_plant("Pine Tree"), "tree")

    def test_flower_recognition(self):
        self.assertEqual(recognize_plant("Red Rose"), "flower")

    def test_unknown_recognition(self):
        self.assertEqual(recognize_plant("Cactus"), "unknown")

    def test_ambiguous_tree_and_flower_keywords(self):
        self.assertEqual(recognize_plant("Rose Tree"), "unknown")

    def test_empty_and_whitespace_input(self):
        self.assertEqual(recognize_plant(""), "unknown")
        self.assertEqual(recognize_plant("   "), "unknown")

    def test_case_insensitive_recognition(self):
        self.assertEqual(recognize_plant("rOsE"), "flower")

    def test_batch_recognition(self):
        self.assertEqual(
            recognize_plants(["Oak", "Tulip", "Fern"]),
            {"Oak": "tree", "Tulip": "flower", "Fern": "unknown"},
        )

    def test_empty_batch(self):
        self.assertEqual(recognize_plants([]), {})


if __name__ == "__main__":
    unittest.main()
