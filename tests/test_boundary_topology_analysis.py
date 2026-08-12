import math
import unittest
from types import SimpleNamespace

import numpy as np

from tools import analyze_boundary_topology as analysis


class BoundaryTopologyAnalysisTests(unittest.TestCase):
    @staticmethod
    def graph_args(**overrides):
        values = {
            "range_min": 0.1,
            "range_max": 80.0,
            "boundary_log_thr": 0.2,
            "neighbor": "angular_grid8",
            "edge_spatial_mode": "angular",
            "sigma_angular": 0.01,
            "sigma_tangent": 1.0,
            "sigma_log_range": 0.3,
            "max_log_range_diff": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def projection(shape):
        height, width = shape
        return {
            "vertical_centers_deg": np.linspace(0.15, -0.15, height),
            "azimuth_centers_deg": np.linspace(-0.35, 0.35, width),
            "azimuth_mode": "full_360_front_centered",
        }

    def test_gt_boundary_detection_same_surface_and_jump(self):
        same = np.full((2, 4), 10.0, dtype=np.float32)
        self.assertFalse(np.any(analysis.gt_boundary_mask(same)))
        jump = same.copy()
        jump[:, 2:] = 20.0
        boundary = analysis.gt_boundary_mask(jump, boundary_log_thr=0.2)
        self.assertTrue(boundary[:, 1].all())
        self.assertTrue(boundary[:, 2].all())

    def test_horizontal_wrap_boundary_is_detected(self):
        gt = np.array([[30.0, 30.0, 30.0, 10.0]], dtype=np.float32)
        boundary = analysis.gt_boundary_mask(gt, boundary_log_thr=0.2)
        self.assertTrue(boundary[0, 0])
        self.assertTrue(boundary[0, -1])

    def test_periodic_boundary_distance_wraps_at_seam(self):
        boundary = np.zeros((1, 8), dtype=bool)
        boundary[0, 0] = True
        distance = analysis.periodic_boundary_distance(boundary)
        self.assertEqual(distance[0, 0], 0.0)
        self.assertEqual(distance[0, 1], 1.0)
        self.assertEqual(distance[0, 2], 2.0)
        self.assertEqual(distance[0, 7], 1.0)
        self.assertEqual(distance[0, 6], 2.0)

    def test_source_rows_are_excluded_from_common_domain(self):
        array = np.full((3, 4), 10.0, dtype=np.float32)
        common = analysis.common_hidden_valid_mask(array, [array, array, array], [1])
        self.assertFalse(np.any(common[1]))
        self.assertTrue(np.all(common[[0, 2]]))

    def test_harm_rate_counts_only_raw_regressions(self):
        gt = np.array([[10.0, 10.0, 10.0]])
        raw = np.array([[11.0, 11.0, 11.0]])
        method = np.array([[12.0, 10.0, 11.0 + 5e-7]])
        metrics, _ = analysis.comparison_metrics(
            method, gt, raw, np.ones(gt.shape, dtype=bool)
        )
        self.assertAlmostEqual(metrics["harm_rate"], 1.0 / 3.0)
        self.assertEqual(metrics["mean_harm_magnitude"], 1.0)
        self.assertEqual(metrics["mean_improvement_magnitude"], 1.0)

    def test_rgc_cross_boundary_ratio_is_exact(self):
        raw = np.full((2, 4), 10.0, dtype=np.float32)
        gt = raw.copy()
        gt[:, 2:] = 20.0
        rows, _ = analysis.rgc_graph_quality_rows(
            raw, gt, self.projection(raw.shape), self.graph_args()
        )
        full = next(row for row in rows if row["method"] == "rgc_full")
        self.assertEqual(
            full["cross_boundary_edge_ratio"],
            full["cross_boundary_edges"] / full["gt_valid_edges"],
        )
        self.assertGreater(full["cross_boundary_edges"], 0)

    def test_range_gate_reduces_cross_boundary_influence(self):
        raw = np.full((2, 8), 10.0, dtype=np.float32)
        raw[:, 4:] = 30.0
        gt = raw.copy()
        rows, _ = analysis.rgc_graph_quality_rows(
            raw, gt, self.projection(raw.shape), self.graph_args()
        )
        angular = next(row for row in rows if row["method"] == "rgc_angular_only_weight")
        full = next(row for row in rows if row["method"] == "rgc_full")
        self.assertLess(
            full["normalized_cross_boundary_influence"],
            angular["normalized_cross_boundary_influence"],
        )
        self.assertLess(
            full["mean_range_gate_cross_boundary"],
            full["mean_range_gate_same_surface"],
        )

    def test_angular_only_and_full_topology_are_identical(self):
        raw = np.full((3, 8), 10.0, dtype=np.float32)
        gt = raw.copy()
        rows, debug = analysis.rgc_graph_quality_rows(
            raw, gt, self.projection(raw.shape), self.graph_args()
        )
        angular, full = rows
        self.assertEqual(angular["N_edges_total"], full["N_edges_total"])
        self.assertEqual(angular["edge_endpoint_digest"], full["edge_endpoint_digest"])
        self.assertEqual(debug["edge_i"].shape, debug["edge_j"].shape)

    def test_invalid_gt_endpoint_is_excluded_from_graph_quality(self):
        raw = np.full((2, 5), 10.0, dtype=np.float32)
        gt = raw.copy()
        gt[0, 0] = 0.0
        rows, _ = analysis.rgc_graph_quality_rows(
            raw, gt, self.projection(raw.shape), self.graph_args()
        )
        full = next(row for row in rows if row["method"] == "rgc_full")
        self.assertLess(full["gt_valid_edges"], full["N_edges_total"])
        self.assertLess(full["mapping_ratio"], 1.0)

    def test_empty_boundary_has_explicit_nan_zero_semantics(self):
        boundary = np.zeros((2, 4), dtype=bool)
        distance = analysis.periodic_boundary_distance(boundary)
        self.assertTrue(np.isinf(distance).all())
        array = np.full((2, 4), 10.0)
        empty = analysis.region_mask(np.ones_like(boundary), distance, "boundary", 1)
        metrics, _ = analysis.comparison_metrics(array, array, array, empty)
        self.assertEqual(metrics["count"], 0)
        self.assertTrue(math.isnan(metrics["mae"]))
        self.assertTrue(math.isnan(metrics["harm_rate"]))

    def test_analysis_helpers_are_deterministic(self):
        gt = np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32)
        first_boundary = analysis.gt_boundary_mask(gt)
        second_boundary = analysis.gt_boundary_mask(gt)
        self.assertTrue(np.array_equal(first_boundary, second_boundary))
        self.assertTrue(np.array_equal(
            analysis.periodic_boundary_distance(first_boundary),
            analysis.periodic_boundary_distance(second_boundary),
        ))

    def test_gdc_graph_mapping_failure_is_explicit(self):
        row = analysis.unsupported_gdc_graph_row()
        self.assertEqual(row["status"], analysis.GDC_UNSUPPORTED_STATUS)
        self.assertEqual(row["mapping_ratio"], 0.0)
        self.assertTrue(math.isnan(row["normalized_cross_boundary_influence"]))


if __name__ == "__main__":
    unittest.main()
