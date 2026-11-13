import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from src.topics.baseline import training_texts, keyword_cues, CUES, topic_inventory, SOURCE_COMMIT
from src.llm.schema import TOPICS


class TopicTests(unittest.TestCase):
    def test_inventory_hash_ignores_line_endings_but_detects_content_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "Dataset").mkdir()
            csv_path = source / "Dataset" / "articles.csv"
            csv_path.write_bytes(b'doc_id,text\n1,"first\nsecond"\n')
            with patch("src.topics.baseline.subprocess.check_output", side_effect=[SOURCE_COMMIT, ""] * 3):
                lf = topic_inventory(source)
                csv_path.write_bytes(csv_path.read_bytes().replace(b"\n", b"\r\n"))
                self.assertEqual(lf, topic_inventory(source))
                csv_path.write_bytes(b"doc_id,text\r\n1,changed\r\n")
                self.assertNotEqual(lf["articles.csv"]["sha256"], topic_inventory(source)["articles.csv"]["sha256"])

    def test_exploration_uses_only_unique_training_texts(self):
        rows = [{"doc_id": "1", "split": "train", "text_sha256": "a"},
                {"doc_id": "2", "split": "train", "text_sha256": "a"},
                {"doc_id": "3", "split": "test", "text_sha256": "b"},
                {"doc_id": "4", "split": "review_reserved", "text_sha256": "c"}]
        self.assertEqual([r["doc_id"] for r in training_texts(rows)], ["1"])

    def test_cues_are_vocabulary_bound_and_count_once(self):
        self.assertEqual(set(CUES), set(TOPICS))
        self.assertEqual(keyword_cues("Anxiety and anxiety."), ["Anxiety"])
        self.assertEqual(keyword_cues("ordinary product announcement"), [])


if __name__ == "__main__":
    unittest.main()
