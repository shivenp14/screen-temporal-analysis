import unittest

from src.topics.prepare_pilot import select_packet


class TopicPacketTests(unittest.TestCase):
    def test_packet_uses_distinct_training_groups_and_is_deterministic(self):
        rows = [{"doc_id": f"id-{i:02d}", "platform": "youtube", "text": "anxiety and sleep disruption",
                 "duplicate_group": f"group-{i:02d}", "split": "train"}
                for i in range(52)]
        rows.extend([{"doc_id": "held-out", "platform": "youtube", "text": "anxiety",
                      "duplicate_group": "held-out-group", "split": "test"}])
        pilot, practice = select_packet(rows)
        pilot_again, practice_again = select_packet(rows)
        self.assertEqual([r["doc_id"] for r in pilot], [r["doc_id"] for r in pilot_again])
        self.assertEqual([r["doc_id"] for r in practice], [r["doc_id"] for r in practice_again])
        self.assertEqual((len(pilot), len(practice)), (40, 6))
        self.assertEqual(len({r["duplicate_group"] for r in pilot + practice}), 46)
        self.assertNotIn("held-out", {r["doc_id"] for r in pilot + practice})


if __name__ == "__main__":
    unittest.main()
