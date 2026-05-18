import unittest

from plants_recognition import recognize_plant, recognize_plants


class PlantRecognitionTests(unittest.TestCase):
    def test_tree_recognition(self):
        self.assertEqual(recognize_plant("Pine Tree"), "tree")

    def test_flower_recognition(self):
        self.assertEqual(recognize_plant("Red Rose"), "flower")

    def test_unknown_recognition(self):
        self.assertEqual(recognize_plant("Cactus"), "unknown")

    def test_batch_recognition(self):
        self.assertEqual(
            recognize_plants(["Oak", "Tulip", "Fern"]),
            {"Oak": "tree", "Tulip": "flower", "Fern": "unknown"},
        )


if __name__ == "__main__":
    unittest.main()
