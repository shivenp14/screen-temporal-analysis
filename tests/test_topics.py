import unittest
from src.topics.baseline import training_texts, keyword_cues, CUES
from src.llm.schema import TOPICS


class TopicTests(unittest.TestCase):
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
