import unittest

import numpy as np

from tools import evaluate_gdc_common_support as support
from tools import analyze_boundary_topology as topology


class SimpleCalibration:
    """Synthetic rect/Velodyne mapping: rect z becomes Velodyne x."""

    @staticmethod
    def project_image_to_rect(uv_depth):
        uv_depth = np.asarray(uv_depth)
        return np.column_stack((np.zeros(len(uv_depth)), np.zeros(len(uv_depth)), uv_depth[:, 2]))

    @staticmethod
    def project_rect_to_velo(points_rect):
        points_rect = np.asarray(points_rect)
        return np.column_stack((points_rect[:, 2], -points_rect[:, 0], -points_rect[:, 1]))


class GDCCommonSupportEvaluationTests(unittest.TestCase):
    @staticmethod
    def projection():
        return {
            "shape": (2, 4),
            "vertical_centers_deg": np.array([22.5, -22.5]),
            "azimuth_centers_deg": np.array([-135.0, -45.0, 45.0, 135.0]),
            "azimuth_mode": "full_360_front_centered",
            "vmin_deg": -45.0,
            "vmax_deg": 45.0,
            "azimuth_min_deg": -180.0,
            "azimuth_max_deg": 180.0,
        }

    def winners(self, points, indices=None, **overrides):
        if indices is None:
            indices = np.arange(len(points))
        return support.canonical_winner_map(
            points, indices, self.projection(), **overrides
        )

    def test_single_point_winner(self):
        reconstructed, winners = self.winners([[10.0, 0.0, 0.0]], [7])
        self.assertEqual(reconstructed[1, 2], 10.0)
        self.assertEqual(winners[1, 2], 7)
        self.assertEqual(np.count_nonzero(reconstructed), 1)

    def test_multiple_points_same_cell_nearest_wins(self):
        reconstructed, winners = self.winners(
            [[20.0, 0.0, 0.0], [10.0, 0.0, 0.0]], [3, 8]
        )
        self.assertEqual(reconstructed[1, 2], 10.0)
        self.assertEqual(winners[1, 2], 8)

    def test_nearest_ineligible_farther_eligible_cell_is_ineligible(self):
        _, winners = self.winners(
            [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], [0, 1]
        )
        eligible = support.gdc_eligible_canonical_mask(
            winners, np.array([False, True])
        )
        self.assertFalse(eligible[1, 2])

    def test_nearest_eligible_farther_ineligible_cell_is_eligible(self):
        _, winners = self.winners(
            [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], [0, 1]
        )
        eligible = support.gdc_eligible_canonical_mask(
            winners, np.array([True, False])
        )
        self.assertTrue(eligible[1, 2])

    def test_invalid_depth_is_excluded(self):
        depth = np.array([[np.nan, 0.0, -1.0, 10.0]], dtype=np.float32)
        reconstructed, winners = support.raw_depth_to_canonical_winner_map(
            depth, SimpleCalibration(), self.projection()
        )
        self.assertEqual(np.count_nonzero(reconstructed), 1)
        self.assertEqual(winners[1, 2], 3)

    def test_range_bounds_are_strict(self):
        depth = np.array([[0.1, 80.0, 10.0]], dtype=np.float32)
        reconstructed, winners = support.raw_depth_to_canonical_winner_map(
            depth,
            SimpleCalibration(),
            self.projection(),
            range_min=0.1,
            range_max=80.0,
        )
        self.assertEqual(np.count_nonzero(reconstructed), 1)
        self.assertEqual(reconstructed[1, 2], 10.0)
        self.assertEqual(winners[1, 2], 2)

    def test_projection_row_and_column_consistency(self):
        _, winners = self.winners(
            [[10.0, 0.0, 5.0], [10.0, 0.0, -5.0]], [4, 5]
        )
        self.assertEqual(winners[0, 2], 4)
        self.assertEqual(winners[1, 2], 5)

    def test_horizontal_azimuth_behavior(self):
        _, winners = self.winners(
            [[0.0, -10.0, 0.0], [0.0, 10.0, 0.0]], [6, 7]
        )
        self.assertEqual(winners[1, 1], 6)
        self.assertEqual(winners[1, 3], 7)

    def test_equal_float32_range_uses_smallest_flat_pixel(self):
        reconstructed, winners = self.winners(
            [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], [9, 2]
        )
        self.assertEqual(reconstructed[1, 2], 10.0)
        self.assertEqual(winners[1, 2], 2)

    def test_reconstructed_range_equality_fixture(self):
        depth = np.array([[10.0, 20.0]], dtype=np.float32)
        reconstructed, _ = support.raw_depth_to_canonical_winner_map(
            depth, SimpleCalibration(), self.projection()
        )
        audit = support.audit_reconstructed_raw_range(reconstructed, reconstructed)
        self.assertEqual(audit["mismatched_range_cell_count"], 0)
        self.assertEqual(audit["status"], "ok")

    def test_intentional_reconstruction_mismatch_fails_fast(self):
        reconstructed = np.zeros((2, 4), dtype=np.float32)
        artifact = reconstructed.copy()
        artifact[1, 2] = 10.0
        with self.assertRaisesRegex(ValueError, "reconstruction mismatch"):
            support.audit_reconstructed_raw_range(reconstructed, artifact)

    def test_source_rows_are_excluded_from_common_support(self):
        common = np.ones((3, 4), dtype=bool)
        common &= topology.hidden_rows_mask(common.shape, [1])
        eligible = np.ones_like(common)
        result = support.common_gdc_support_mask(common, eligible)
        self.assertFalse(np.any(result[1]))
        self.assertEqual(int(result.sum()), 8)

    def test_common_support_is_exact_intersection(self):
        common = np.array([[True, True, False, False]])
        eligible = np.array([[True, False, True, False]])
        result = support.common_gdc_support_mask(common, eligible)
        self.assertTrue(np.array_equal(result, common & eligible))

    def test_gt_does_not_affect_eligibility_mask(self):
        depth = np.array([[10.0, 20.0]], dtype=np.float32)
        calib = SimpleCalibration()
        before = support.production_gdc_eligibility(depth, calib)
        gt_a = np.zeros((2, 4), dtype=np.float32)
        gt_b = np.full((2, 4), 70.0, dtype=np.float32)
        self.assertFalse(np.array_equal(gt_a, gt_b))
        after = support.production_gdc_eligibility(depth, calib)
        self.assertTrue(np.array_equal(before, after))

    def test_gdc_and_rgc_outputs_do_not_affect_eligibility_mask(self):
        depth = np.array([[10.0, 20.0]], dtype=np.float32)
        calib = SimpleCalibration()
        before = support.production_gdc_eligibility(depth, calib)
        gdc_output = np.array([[1.0, 2.0]])
        rgc_output = np.array([[70.0, 60.0]])
        self.assertFalse(np.array_equal(gdc_output, rgc_output))
        after = support.production_gdc_eligibility(depth, calib)
        self.assertTrue(np.array_equal(before, after))

    def test_winner_reconstruction_is_deterministic(self):
        arguments = ([[20.0, 0.0, 0.0], [10.0, 0.0, 0.0]], [8, 3])
        first = self.winners(*arguments)
        second = self.winners(*arguments)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))


if __name__ == "__main__":
    unittest.main()
