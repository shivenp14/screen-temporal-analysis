import unittest

from src.sentiment.baseline import make_model, metrics, model_input, validate_records


class SentimentTests(unittest.TestCase):
    def test_vectorizer_does_not_learn_held_out_vocabulary(self):
        model = make_model()
        model.fit(["benefit support", "benefit support", "harm risk", "harm risk", "report facts", "report facts"],
                  ["Positive", "Positive", "Negative", "Negative", "Neutral", "Neutral"])
        model.predict(["unseenholdoutword support"])
        self.assertNotIn("unseenholdoutword", model["tfidf"].vocabulary_)

    def test_cross_split_groups_are_rejected(self):
        rows = [{"doc_id": str(i), "text": "article", "text_sha256": str(i), "duplicate_group": "shared",
                 "stance_4class": "Negative", "sentiment_3class": "Negative", "split": split}
                for i, split in enumerate(("train", "validation", "test"))]
        with self.assertRaises(ValueError):
            validate_records(rows)

    def test_missing_platform_class_stays_in_macro_denominator(self):
        result = metrics(["Negative", "Neutral"], ["Negative", "Neutral"])
        self.assertEqual(result["accuracy"], 1)
        self.assertAlmostEqual(result["macro_f1"], 2/3)
        self.assertEqual(result["per_class"]["Positive"]["support"], 0)

    def test_platform_aware_input_distinguishes_identical_cross_platform_text(self):
        article = {"text": "Identical report text", "platform": "snapchat"}
        other_platform = {"text": "Identical report text", "platform": "youtube"}
        self.assertNotEqual(model_input(article, True), model_input(other_platform, True))
        self.assertEqual(model_input(article, False), model_input(other_platform, False))


if __name__ == "__main__":
    unittest.main()
