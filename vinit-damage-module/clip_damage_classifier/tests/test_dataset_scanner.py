"""
Unit tests for dataset scanning and metadata extraction.
"""
import unittest
from clip_damage_classifier.metadata_parser import parse_banknote_metadata

class TestScannerMetadata(unittest.TestCase):
    def test_denomination_and_generation_extraction(self):
        # Case 1: New 10 Rupees
        meta1 = parse_banknote_metadata("Spoilt New Indian Bank Notes/New 10 Rupees", "10_new_1.jpg")
        self.assertEqual(meta1["denomination"], "10")
        self.assertEqual(meta1["note_generation"], "new")

        # Case 2: Old 100 Rupees
        meta2 = parse_banknote_metadata("Spoilt Old Indian Bank Notes/Old 100 Rupees", "100_old_42.jpg")
        self.assertEqual(meta2["denomination"], "100")
        self.assertEqual(meta2["note_generation"], "old")

        # Case 3: 50 Rupees
        meta3 = parse_banknote_metadata("New 50 Rupees", "note50.png")
        self.assertEqual(meta3["denomination"], "50")

        # Case 4: Ambiguous path
        meta4 = parse_banknote_metadata("RandomFolder/Notes", "image001.jpg")
        self.assertEqual(meta4["denomination"], "")
        self.assertEqual(meta4["note_generation"], "")

if __name__ == "__main__":
    unittest.main()
