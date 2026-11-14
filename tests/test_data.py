import unittest
from src.data.preprocess import duplicate_groups, parse_date, assign_splits
from src.analysis.temporal import summarize


class DataTests(unittest.TestCase):
    def test_date_failure_is_missing_not_filename_year(self):
        self.assertIsNone(parse_date("not a date"))
        self.assertIsNone(parse_date(None))
        self.assertEqual(parse_date("2024-12-31T23:30:00-05:00"), "2025-01-01")

    def test_similar_texts_group_and_different_text_does_not(self):
        a = " ".join(f"word{i}" for i in range(100))
        groups, _ = duplicate_groups({"a": a, "b": a + " extra", "c": "totally unrelated words"})
        self.assertEqual(groups["a"], groups["b"])
        self.assertNotEqual(groups["a"], groups["c"])

    def test_syndicated_copy_below_old_cutoff_stays_in_one_group(self):
        original = [f"word{i}" for i in range(165)]
        copy = original.copy()
        copy[80:85] = [f"replacement{i}" for i in range(5)]
        groups, _ = duplicate_groups({"original": " ".join(original), "copy": " ".join(copy)})
        self.assertEqual(groups["original"], groups["copy"])

    def test_group_reservation_and_partition_exclusivity(self):
        rows = [{"duplicate_group": f"{label}-{i}", "sentiment_3class": label}
                for label in ("Positive", "Negative", "Neutral") for i in range(30)]
        rows.append(dict(rows[0]))
        assign_splits(rows, {"Positive-0"})
        self.assertEqual(rows[0]["split"], "review_reserved")
        self.assertEqual(rows[-1]["split"], "review_reserved")
        self.assertEqual({r["split"] for r in rows}, {"train", "validation", "test", "review_reserved"})

    def test_temporal_denominators_and_missing_dates(self):
        rows = [{"platform": "youtube", "publication_date": "2024-01-10", "sentiment_3class": "Negative"},
                {"platform": "youtube", "publication_date": "2024-02-10", "sentiment_3class": "Neutral"},
                {"platform": "youtube", "publication_date": None, "sentiment_3class": "Positive"}]
        result = summarize(rows, "quarter")
        self.assertEqual({r["n"] for r in result}, {2})
        self.assertEqual(sum(r["proportion"] for r in result), 1)


if __name__ == "__main__":
    unittest.main()
