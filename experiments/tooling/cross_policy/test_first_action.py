import unittest

import numpy as np

from first_action_divergence import first_action_divergence


class FirstActionDivergence(unittest.TestCase):
    def test_identical_has_no_divergent_tick(self):
        a = np.arange(21.0).reshape(3, 7)
        result = first_action_divergence(a, a.copy())
        self.assertIsNone(result["first_divergent_tick"])
        self.assertEqual(result["first_action"]["l2"], 0.0)

    def test_stale_shape_equal_first_action_later_divergence(self):
        a = np.zeros((6, 7))
        b = a.copy()
        b[2, 3] = 0.5
        result = first_action_divergence(a, b)
        self.assertEqual(result["first_action"]["l2"], 0.0)
        self.assertEqual(result["first_divergent_tick"], 2)
        self.assertAlmostEqual(result["at_first_divergent_tick"]["max_abs"], 0.5)

    def test_different_lengths_use_common_prefix(self):
        result = first_action_divergence(np.zeros((6, 7)), np.zeros((4, 7)))
        self.assertIsNone(result["first_divergent_tick"])
        self.assertEqual(result["common_ticks"], 4)
        self.assertEqual(result["lengths"], [6, 4])

    def test_rejects_mismatched_dimension(self):
        with self.assertRaises(ValueError):
            first_action_divergence(np.zeros((2, 7)), np.zeros((2, 6)))


if __name__ == "__main__":
    unittest.main()
