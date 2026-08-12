import math
import unittest
from types import SimpleNamespace

import numpy as np

from tools import analyze_boundary_topology as analysis


class BoundaryTopologyAnalysisTests(unittest.TestCase):
    @staticmethod
    def activity(raw, method, gt, correction_tol=1e-3):
        raw = np.asarray(raw, dtype=np.float64)
        method = np.asarray(method, dtype=np.float64)
        gt = np.asarray(gt, dtype=np.float64)
        mask = np.ones(raw.shape, dtype=bool)
        metrics, _ = analysis.correction_activity_metrics(
            method, gt, raw, mask, correction_tol=correction_tol
        )
        return metrics

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

    def test_raw_activity_semantics(self):
        metrics = self.activity([11.0, 9.0], [11.0, 9.0], [10.0, 10.0])
        self.assertEqual(metrics["changed_ratio"], 0.0)
        self.assertEqual(metrics["unchanged_ratio"], 1.0)
        self.assertEqual(metrics["improved_ratio"], 0.0)
        self.assertEqual(metrics["harm_rate"], 0.0)
        self.assertEqual(metrics["neutral_ratio"], 1.0)
        self.assertEqual(metrics["mean_abs_correction"], 0.0)
        self.assertEqual(metrics["median_abs_correction"], 0.0)

    def test_changed_and_improved(self):
        metrics = self.activity([12.0], [11.0], [10.0])
        self.assertEqual(metrics["changed_pixels"], 1)
        self.assertEqual(metrics["improved_pixels"], 1)

    def test_changed_and_harmed(self):
        metrics = self.activity([11.0], [12.0], [10.0])
        self.assertEqual(metrics["changed_pixels"], 1)
        self.assertEqual(metrics["harmed_pixels"], 1)

    def test_changed_and_neutral(self):
        metrics = self.activity([9.0], [11.0], [10.0])
        self.assertEqual(metrics["changed_pixels"], 1)
        self.assertEqual(metrics["neutral_pixels"], 1)

    def test_unchanged_semantics_are_independent_of_error_outcome(self):
        metrics = self.activity([11.0], [11.0005], [10.0])
        self.assertEqual(metrics["changed_pixels"], 0)
        self.assertEqual(metrics["unchanged_pixels"], 1)
        self.assertEqual(metrics["harmed_pixels"], 1)

    def test_correction_tolerance_boundary_is_strict(self):
        exact = analysis._correction_activity_from_vectors([1e-3], [0.0])
        above = analysis._correction_activity_from_vectors(
            [np.nextafter(1e-3, math.inf)], [0.0]
        )
        self.assertEqual(exact["changed_pixels"], 0)
        self.assertEqual(above["changed_pixels"], 1)

    def test_error_tolerance_boundary_is_neutral(self):
        values = np.array([
            -analysis.HARM_TOL,
            analysis.HARM_TOL,
            np.nextafter(-analysis.HARM_TOL, -math.inf),
            np.nextafter(analysis.HARM_TOL, math.inf),
        ])
        metrics = analysis._correction_activity_from_vectors(
            np.ones(4), values
        )
        self.assertEqual(metrics["neutral_pixels"], 2)
        self.assertEqual(metrics["improved_pixels"], 1)
        self.assertEqual(metrics["harmed_pixels"], 1)

    def test_activity_count_identities(self):
        metrics = analysis._correction_activity_from_vectors(
            [0.0, 0.01, -0.02, 0.0005], [0.0, -1.0, 2.0, 0.0]
        )
        self.assertEqual(
            metrics["changed_pixels"] + metrics["unchanged_pixels"],
            metrics["pixels"],
        )
        self.assertEqual(
            metrics["improved_pixels"]
            + metrics["harmed_pixels"]
            + metrics["neutral_pixels"],
            metrics["pixels"],
        )

    def test_changed_conditional_identity(self):
        metrics = analysis._correction_activity_from_vectors(
            [0.01, -0.02, 0.03, 0.0], [-1.0, 2.0, 0.0, -1.0]
        )
        changed_outcomes = (
            metrics["changed_improved_pixels"]
            + metrics["changed_harmed_pixels"]
            + metrics["changed_neutral_pixels"]
        )
        self.assertEqual(changed_outcomes, metrics["changed_pixels"])
        self.assertAlmostEqual(
            metrics["changed_improvement_rate"]
            + metrics["changed_harm_rate"]
            + metrics["changed_neutral_rate"],
            1.0,
        )

    def test_net_error_reduction_is_exact(self):
        metrics = analysis._correction_activity_from_vectors(
            [1.0, 1.0], [-1.0, 2.0]
        )
        self.assertEqual(metrics["total_improvement"], 1.0)
        self.assertEqual(metrics["total_harm"], 2.0)
        self.assertEqual(metrics["net_error_reduction"], -1.0)
        self.assertEqual(metrics["mean_net_error_reduction"], -0.5)
        self.assertEqual(metrics["improvement_to_harm_ratio"], 0.5)

    def test_changed_only_correction_magnitude(self):
        metrics = analysis._correction_activity_from_vectors(
            [0.0, 0.002, -0.004], [0.0, 0.0, 0.0]
        )
        self.assertEqual(metrics["mean_abs_correction"], 0.002)
        self.assertEqual(metrics["mean_abs_correction_changed_only"], 0.003)
        self.assertEqual(metrics["median_abs_correction_changed_only"], 0.003)

    def test_empty_changed_set_has_nan_conditional_statistics(self):
        metrics = analysis._correction_activity_from_vectors(
            [0.0, 0.001], [0.0, 0.0]
        )
        self.assertEqual(metrics["changed_pixels"], 0)
        self.assertTrue(math.isnan(metrics["changed_improvement_rate"]))
        self.assertTrue(math.isnan(metrics["changed_harm_rate"]))
        self.assertTrue(math.isnan(metrics["mean_abs_correction_changed_only"]))
        self.assertTrue(math.isnan(metrics["median_abs_correction_changed_only"]))

    def test_source_row_exclusion_applies_to_activity(self):
        raw = np.full((3, 4), 10.0)
        method = raw.copy()
        method[1] += 1.0
        common = analysis.common_hidden_valid_mask(raw, [raw, method, raw], [1])
        metrics, _ = analysis.correction_activity_metrics(
            method, raw, raw, common
        )
        self.assertEqual(metrics["pixels"], 8)
        self.assertEqual(metrics["changed_pixels"], 0)

    def test_boundary_and_interior_activity_are_separate(self):
        gt = np.full((1, 16), 10.0)
        gt[:, 6:10] = 20.0
        raw = gt.copy()
        boundary = analysis.gt_boundary_mask(gt)
        distance = analysis.periodic_boundary_distance(boundary)
        common = np.ones(gt.shape, dtype=bool)
        boundary_region = analysis.region_mask(common, distance, "boundary", 1)
        interior_region = analysis.region_mask(common, distance, "interior", 1)
        method = raw.copy()
        method[boundary_region] += 0.1
        boundary_metrics, _ = analysis.correction_activity_metrics(
            method, gt, raw, boundary_region
        )
        interior_metrics, _ = analysis.correction_activity_metrics(
            method, gt, raw, interior_region
        )
        self.assertEqual(
            boundary_metrics["changed_pixels"], int(boundary_region.sum())
        )
        self.assertEqual(interior_metrics["changed_pixels"], 0)

    def test_activity_analysis_is_deterministic(self):
        arguments = ([0.0, 0.002, -0.003], [-1.0, 0.0, 2.0])
        first = analysis._correction_activity_from_vectors(*arguments)
        second = analysis._correction_activity_from_vectors(*arguments)
        for key in first:
            if math.isnan(first[key]):
                self.assertTrue(math.isnan(second[key]))
            else:
                self.assertEqual(first[key], second[key])

    def test_invalid_activity_tolerances_fail_fast(self):
        for correction_tol in (-1.0, math.inf, math.nan):
            with self.assertRaises(ValueError):
                analysis._correction_activity_from_vectors(
                    [0.0], [0.0], correction_tol=correction_tol
                )
        with self.assertRaises(ValueError):
            analysis._correction_activity_from_vectors(
                [0.0], [0.0], error_tol=-1.0
            )


if __name__ == "__main__":
    unittest.main()
