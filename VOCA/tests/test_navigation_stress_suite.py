import csv
import tempfile
import unittest
from pathlib import Path

from navigation_stress_suite import (
    deterministic_guard_cases,
    evaluate_revisit_pairs,
    summarize_stop_cases,
)


class NavigationStressSuiteTests(unittest.TestCase):
    def test_deterministic_guard_cases_pass(self):
        cases = deterministic_guard_cases()

        self.assertGreaterEqual(len(cases), 6)
        self.assertTrue(all(case["passed"] for case in cases))

    def test_revisit_summary_prioritizes_zero_false_merges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pairs.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["label", "similarity"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"label": "same_place", "similarity": 0.95},
                        {"label": "same_place", "similarity": 0.70},
                        {"label": "different_place", "similarity": 0.60},
                    ]
                )

            summary = evaluate_revisit_pairs(path, 0.84)

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["fp"], 0)
        self.assertAlmostEqual(summary["recall"], 0.5)

    def test_stop_summary_rejects_any_false_stop(self):
        passed = summarize_stop_cases(
            [
                {"expected_pass": True, "actual_pass": True},
                {"expected_pass": False, "actual_pass": False},
            ]
        )
        failed = summarize_stop_cases(
            [{"expected_pass": False, "actual_pass": True}]
        )

        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])


if __name__ == "__main__":
    unittest.main()
