import unittest

from scripts.calibrate_place_recognition import select_threshold, threshold_metrics


class PlaceRecognitionCalibrationTests(unittest.TestCase):
    def test_precision_constrained_threshold_rejects_visual_alias(self):
        pairs = [
            {"label": "same_place", "similarity": 0.96},
            {"label": "same_place", "similarity": 0.93},
            {"label": "different_place", "similarity": 0.91},
            {"label": "different_place", "similarity": 0.40},
        ]
        selected, _ = select_threshold(pairs, target_precision=1.0)
        self.assertGreater(selected["threshold"], 0.91)
        self.assertEqual(selected["precision"], 1.0)
        self.assertEqual(selected["recall"], 1.0)

    def test_threshold_metrics_counts_false_merge(self):
        metrics = threshold_metrics(
            [
                {"label": "same_place", "similarity": 0.9},
                {"label": "different_place", "similarity": 0.85},
            ],
            0.8,
        )
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
