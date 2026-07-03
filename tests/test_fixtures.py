import unittest
from collections import Counter

from goal_adapter_fixtures import REFINEMENT_CASES


class FirstStageFixtureTest(unittest.TestCase):
    def test_fixture_set_has_three_cases_per_first_stage_group(self):
        group_counts = Counter(case.get("case_group") for case in REFINEMENT_CASES)

        self.assertEqual(len(REFINEMENT_CASES), 12)
        self.assertEqual(
            group_counts,
            {
                "coarse_gps": 3,
                "coarse_object_point": 3,
                "tracking_loss": 3,
                "deadlock": 3,
            },
        )

    def test_fixture_case_names_are_unique(self):
        names = [case["name"] for case in REFINEMENT_CASES]

        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
