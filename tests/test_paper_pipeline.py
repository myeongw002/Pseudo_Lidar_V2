import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from range_gdc.evaluate_range_metrics import leakage_row
from range_gdc.make_anchor_from_gt_range import process_one
from range_gdc.range_gdc import RangeROIGDC, _apply_anchor_reject, select_residuals
from range_gdc.range_main_batch import npy_map, select_scene_ids
from range_gdc.range_projection import range_to_velo
from src.pseudo_lidar.depth_to_range_uniform import lidar_points_to_spherical_guide_uniform
from range_gdc.shared_canonical_anchor import (
    extract_shared_points, project_with_source_indices, shared_points_to_camera_depth,
    shared_points_to_gdc_depth,
)
from ptc2depthmap import get_depth_map
from tools import run_range_gdc_experiment as pipeline
from tools.audit_shared_anchor_protocol import audit_frame, aggregate_audit
from tools.run_fusion_comparison import aggregate_common_hidden_metrics


class PaperPipelineTests(unittest.TestCase):
    class FakeGDCCalibration:
        def project_velo_to_image(self, points):
            points = np.asarray(points, dtype=np.float32)
            return points[:, [1, 2]]

        def project_velo_to_rect(self, points):
            points = np.asarray(points, dtype=np.float32)
            return np.column_stack((points[:, 1], points[:, 2], points[:, 0]))

    def test_canonical_gdc_filters_velodyne_x_at_clip_distance(self):
        depth = shared_points_to_gdc_depth(
            np.array([[2.0, 1.0, 1.0, 0.1]], dtype=np.float32),
            self.FakeGDCCalibration(), (10, 10, 3),
        )
        self.assertFalse(np.any(depth > 0))

    def test_canonical_gdc_keeps_point_beyond_clip_distance_in_fov(self):
        depth = shared_points_to_gdc_depth(
            np.array([[3.0, 2.0, 2.0, 0.1]], dtype=np.float32),
            self.FakeGDCCalibration(), (10, 10, 3),
        )
        self.assertEqual(depth[2, 2], 3.0)

    def test_canonical_gdc_filters_point_outside_float_image_fov(self):
        depth = shared_points_to_gdc_depth(
            np.array([[3.0, 9.0, 1.0, 0.1]], dtype=np.float32),
            self.FakeGDCCalibration(), (10, 10, 3),
        )
        self.assertFalse(np.any(depth > 0))

    def test_canonical_gdc_pixel_collision_keeps_nearest_positive_camera_z(self):
        depth = shared_points_to_gdc_depth(
            np.array([[5.0, 1.0, 1.0, 0.1], [3.0, 1.0, 1.0, 0.2]], dtype=np.float32),
            self.FakeGDCCalibration(), (10, 10, 3),
        )
        self.assertEqual(depth[1, 1], 3.0)

    def test_production_canonical_gdc_and_audit_helper_are_identical(self):
        points = np.array([
            [1.0, 1.0, 1.0, 0.1], [5.0, 1.0, 1.0, 0.2],
            [3.0, 1.0, 1.0, 0.3], [4.0, 20.0, 1.0, 0.4],
        ], dtype=np.float32)
        calib = self.FakeGDCCalibration()
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        production = get_depth_map(points, calib, image, "nearest_positive")
        audit_expected = shared_points_to_gdc_depth(points, calib, image.shape)
        self.assertTrue(np.array_equal(production, audit_expected))

    def test_legacy_last_collision_policy_preserves_last_overwrite(self):
        points = np.array([
            [3.0, 1.0, 1.0, 0.1], [5.0, 1.0, 1.0, 0.2],
        ], dtype=np.float32)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        depth = get_depth_map(points, self.FakeGDCCalibration(), image, "legacy_last")
        self.assertEqual(depth[1, 1], 5.0)

    @staticmethod
    def shared_audit_fixture():
        points = np.array([[1.0, 0.0, 0.0, 0.1], [0.0, 2.0, 0.0, 0.2]], dtype=np.float32)
        source = np.full((2, 2), -1, dtype=np.int32)
        source[0, 0], source[0, 1] = 5, 9
        rgc = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32)
        gdc = np.array([[3.0, -1.0]], dtype=np.float32)
        return {
            "frame_id": "000000", "points": points, "source_index": source,
            "recorded_rgc_source_index": source.copy(), "rgc_anchor": rgc,
            "actual_gdc_depth": gdc.copy(), "expected_gdc_depth": gdc,
            "manifest_frame": {"sha256": "expected", "shared_sparse_point_count": 2},
            "actual_shared_sha256": "expected", "selected_rows": [0],
        }

    def test_shared_audit_detects_wrong_rgc_range_value(self):
        fixture = self.shared_audit_fixture()
        fixture["rgc_anchor"] = fixture["rgc_anchor"].copy()
        fixture["rgc_anchor"][0, 1] += 0.01
        row = audit_frame(**fixture)
        self.assertEqual(row["rgc_range_value_mismatch_count"], 1)
        self.assertGreater(row["rgc_range_value_max_abs_error"], 0.009)

    def test_shared_audit_detects_manifest_hash_mismatch(self):
        fixture = self.shared_audit_fixture()
        fixture["actual_shared_sha256"] = "modified-file"
        row = audit_frame(**fixture)
        self.assertFalse(row["shared_sparse_sha256_match"])
        self.assertEqual(row["shared_sparse_sha256_mismatch_count"], 1)

    def test_shared_audit_detects_source_index_point_count_mismatch(self):
        fixture = self.shared_audit_fixture()
        fixture["points"] = fixture["points"][:1]
        row = audit_frame(**fixture)
        self.assertEqual(row["selected_unique_source_index_count"], 2)
        self.assertEqual(row["shared_sparse_point_count"], 1)
        self.assertEqual(row["shared_point_count_mismatch_count"], 1)

    def test_shared_audit_detects_wrong_gdc_depth_value(self):
        fixture = self.shared_audit_fixture()
        fixture["actual_gdc_depth"] = fixture["actual_gdc_depth"].copy()
        fixture["actual_gdc_depth"][0, 0] += 0.02
        row = audit_frame(**fixture)
        self.assertEqual(row["gdc_depth_value_mismatch_count"], 1)
        self.assertEqual(row["gdc_both_valid_value_mismatch_count"], 1)
        self.assertEqual(row["gdc_expected_valid_only_count"], 0)
        self.assertEqual(row["gdc_actual_valid_only_count"], 0)
        self.assertGreater(row["gdc_depth_value_max_abs_error"], 0.019)

    def test_normal_shared_audit_has_zero_mismatch_counts(self):
        row = audit_frame(**self.shared_audit_fixture())
        summary = aggregate_audit([row])
        for key in (
            "shared_sparse_sha256_mismatch_count", "shared_point_count_mismatch_count",
            "rgc_anchor_not_from_shared_count", "rgc_range_value_mismatch_count",
            "gdc_anchor_not_from_shared_count", "gdc_depth_value_mismatch_count",
        ):
            self.assertEqual(summary[key], 0, key)

    def test_canonical_projection_keeps_nearest_source_collision_winner(self):
        # Same cell, deliberately reverse the nearest/farthest PCD order.
        points = np.array([[10.0, 0.0, 0.0, 0.1], [5.0, 0.0, 0.0, 0.2]], dtype=np.float32)
        ranges, source, count = project_with_source_indices(
            points, range_h=4, range_w=8, vmin_deg=-10, vmax_deg=10,
            azimuth_mode="full_360_front_centered", range_min=0.1, range_max=80.0,
        )
        row, col = np.argwhere(source == 1)[0]
        self.assertEqual(count[row, col], 2)
        self.assertEqual(source[row, col], 1)
        self.assertEqual(ranges[row, col], 5.0)

    def test_shared_pointcloud_preserves_original_records_and_deterministic_order(self):
        points = np.array([
            [8.0, 0.0, 0.0, 0.8], [5.0, 0.0, 0.0, 0.5], [6.0, 1.0, 0.0, 0.6],
        ], dtype=np.float32)
        _, source, _ = project_with_source_indices(
            points, range_h=4, range_w=64, vmin_deg=-10, vmax_deg=10,
            azimuth_mode="full_360_front_centered", range_min=0.1, range_max=80.0,
        )
        shared, indices = extract_shared_points(points, source, [1, 2])
        self.assertTrue(np.array_equal(indices, np.sort(np.unique(indices))))
        self.assertTrue(np.array_equal(shared, points[indices]))
        self.assertNotIn(0, indices.tolist())  # Collision loser cannot reappear.

    def test_camera_anchor_uses_nearest_positive_depth(self):
        class Calibration:
            def project_velo_to_image(self, points):
                return np.column_stack((np.zeros(len(points)), np.zeros(len(points))))
            def project_velo_to_rect(self, points):
                return np.asarray(points, dtype=np.float32)
        depth = shared_points_to_camera_depth(
            np.array([[1, 0, 9, 1], [2, 0, 3, 1], [3, 0, -2, 1]], dtype=np.float32),
            Calibration(), (2, 2, 3), invalid_value=-1,
        )
        self.assertEqual(depth[0, 0], 3.0)

    def test_shared_canonical_range_anchor_uses_only_shared_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pc, indexes, out, mask, meta = (root / name for name in ("pc", "index", "out", "mask", "meta"))
            for path in (pc, indexes): path.mkdir()
            original = np.array([[1, 0, 1, .1]], dtype=np.float32)
            original.tofile(pc / "000000.bin")
            source = np.full((4, 4), -1, dtype=np.int32); source[1, 2] = 0; source[3, 1] = 1
            np.save(indexes / "000000.npy", source)
            args = type("Args", (), {"expected_height": 4, "expected_width": 4, "source_index_dir": str(indexes),
                "selected_rows": [1], "shared_pointcloud_dir": str(pc), "invalid_value": 0.0,
                "output_range_path": str(out), "output_mask_path": str(mask), "meta_dir": str(meta)})()
            from range_gdc.make_anchor_from_gt_range import process_one_shared_canonical
            process_one_shared_canonical(0, args)
            anchor = np.load(out / "000000.npy")
            self.assertAlmostEqual(float(anchor[1, 2]), np.sqrt(2), places=6)
            self.assertFalse(np.any(anchor[[0, 2, 3]]))
    def test_canonical_runner_builds_confidence_hard_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split.txt"
            split.write_text("0\n")
            cfg = {
                "output_root": str(root / "output"), "kitti_root": str(root / "kitti"),
                "split_file": str(split), "data_tag": "unit", "range_anchor": {"selected_rows": [1]},
                "anchor": {"selected_lines": [1]}, "range_gdc": {"selection_mode": "confidence_hard"},
            }
            args = type("Args", (), {"config": str(root / "config.yaml"), "threads": 1, "data_tag": None, "output_root": None,
                "kitti_root": None, "split_file": None, "sdn_config": None, "sdn_checkpoint": None,
                "no_distance_eval": True})()
            with mock.patch.object(pipeline, "load_yaml", return_value=cfg):
                with mock.patch.object(pipeline, "read_split_ids", return_value=["000000"]):
                    with mock.patch.object(pipeline, "resolve", side_effect=lambda value: Path(value)):
                        context = pipeline.build_context(args)
            context["args"] = args
            commands = pipeline.build_stages(context)["range_gdc"].commands_fn()[0]
            self.assertIn("--selection_mode", commands)
            self.assertEqual(commands[commands.index("--selection_mode") + 1], "confidence_hard")

    def test_canonical_runner_uses_shared_anchor_stages_and_soft_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); split = root / "split.txt"; split.write_text("0\n")
            cfg = {"output_root": str(root / "out"), "kitti_root": str(root / "kitti"), "split_file": str(split),
                   "anchor": {"mode": "shared_canonical", "selected_rows": [1]},
                   "range_anchor": {"mode": "shared_canonical", "selected_rows": [1]},
                   "range_gdc": {"selection_mode": "soft", "projection": {"height": 4, "width": 8}}}
            args = type("Args", (), {"config": "x", "threads": 1, "data_tag": None, "output_root": None, "kitti_root": None,
                "split_file": None, "sdn_config": None, "sdn_checkpoint": None, "no_distance_eval": True})()
            with mock.patch.object(pipeline, "load_yaml", return_value=cfg), mock.patch.object(pipeline, "read_split_ids", return_value=["000000"]), mock.patch.object(pipeline, "resolve", side_effect=lambda value: Path(value)):
                context = pipeline.build_context(args)
            context["args"] = args
            stages = pipeline.build_stages(context)
            self.assertIn("canonical_shared_anchor", stages)
            self.assertIn("range_anchor_from_shared_anchor", stages)
            command = stages["range_anchor_from_shared_anchor"].commands_fn()[0]
            self.assertIn("shared_canonical", command)

    def test_checked_in_canonical_config_uses_soft_fusion(self):
        config = pipeline.load_yaml(str(Path(__file__).parents[1] / "configs" / "r64_pipeline_canonical.yaml"))
        self.assertEqual(config["range_gdc"]["selection_mode"], "soft")

    def test_abs_anchor_rejection_boundary(self):
        guide = np.array([[10.0, 10.0]], dtype=np.float32)
        anchor = np.array([[12.0, 11.999]], dtype=np.float32)
        kept, rejected = _apply_anchor_reject(np.ones_like(guide, dtype=bool), guide, anchor, "abs", 0.4, 2.0)
        self.assertEqual(kept.tolist(), [[False, True]])
        self.assertEqual(rejected, 1)

    def test_log_ratio_anchor_rejection_boundary(self):
        guide = np.array([[10.0, 10.0]], dtype=np.float32)
        anchor = np.array([[10.0 * np.exp(0.4), 10.0 * np.exp(0.399)]], dtype=np.float32)
        kept, rejected = _apply_anchor_reject(np.ones_like(guide, dtype=bool), guide, anchor, "log_ratio", 0.4, 2.0)
        self.assertEqual(kept.tolist(), [[False, True]])
        self.assertEqual(rejected, 1)

    def test_soft_fusion_uses_method_equation_for_all_valid_direct_proposals(self):
        final, _, _ = select_residuals(
            np.array([0.2, -0.2]), np.array([1.0, -1.0]), np.array([0.1, 0.9]),
            np.array([0.01, 0.01]), selection_mode="soft",
        )
        self.assertTrue(np.allclose(final, [0.28, -0.92]))

    def test_confidence_hard_selects_graph_direct_and_midpoint_blend(self):
        final, stats, _ = select_residuals(
            np.zeros(3), np.array([1.0, -1.0, 2.0]), np.array([0.9, 0.1, 0.5]),
            np.array([0.01, 0.01, 0.01]), selection_mode="confidence_hard",
        )
        self.assertTrue(np.allclose(final, [1.0, 0.0, 1.0]))
        self.assertEqual((stats["selection_direct_count"], stats["selection_graph_count"], stats["selection_blend_count"]), (1, 1, 1))

    @staticmethod
    def correction_fixture():
        guide = np.full((4, 5), 10.0, dtype=np.float32)
        anchor = np.zeros_like(guide)
        anchor[1, 2] = 11.0
        return guide, anchor

    def test_graph_only_does_not_construct_direct_branch(self):
        guide, anchor = self.correction_fixture()
        with mock.patch("range_gdc.range_gdc.build_direct_residual_transfer", side_effect=AssertionError):
            _, _, stats, debug = RangeROIGDC(guide, anchor, ablation_mode="graph_only", return_stats=True, return_debug=True)
        self.assertEqual(stats["selection_direct_count"], 0)
        self.assertTrue(np.all(debug["selection_graph_mask"]))

    def test_direct_only_does_not_construct_graph_branch_or_fallback(self):
        guide, anchor = self.correction_fixture()
        with mock.patch("range_gdc.range_gdc.build_spherical_graph_laplacian", side_effect=AssertionError):
            _, _, stats, debug = RangeROIGDC(guide, anchor, ablation_mode="direct_only", return_stats=True, return_debug=True)
        self.assertEqual(stats["selection_graph_count"], 0)
        self.assertTrue(np.allclose(debug["delta_final"], debug["delta_direct"]))

    def test_accepted_anchor_force_is_identical_across_variants(self):
        guide, anchor = self.correction_fixture()
        for mode in ("full", "graph_only", "direct_only"):
            corrected, _, stats = RangeROIGDC(guide, anchor, ablation_mode=mode, return_stats=True)
            self.assertEqual(stats["anchor_forced_count"], 1)
            self.assertEqual(corrected[1, 2], anchor[1, 2])

    def test_spherical_projection_round_trip(self):
        points = np.array([[10.0, 0.0, 0.0], [8.0, 2.0, -1.0]], dtype=np.float32)
        image, mask = lidar_points_to_spherical_guide_uniform(points, range_h=16, range_w=64, vmin_deg=-25, vmax_deg=5)
        args = type("Args", (), {"range_h": 16, "range_w": 64, "vmin_deg": -25, "vmax_deg": 5,
            "azimuth_mode": "full_360_front_centered", "azimuth_min_deg": None,
            "azimuth_max_deg": None, "meta_path": None})()
        reconstructed = range_to_velo(image, args)
        image2, mask2 = lidar_points_to_spherical_guide_uniform(reconstructed, range_h=16, range_w=64, vmin_deg=-25, vmax_deg=5)
        self.assertTrue(np.array_equal(mask, mask2))
        self.assertTrue(np.allclose(image[mask], image2[mask2], atol=1e-5))

    def test_fixed_row_anchor_has_no_values_outside_selected_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt_dir, out_dir, mask_dir, meta_dir = (root / name for name in ("gt", "anchor", "mask", "meta"))
            gt_dir.mkdir()
            gt = np.full((4, 5), 10.0, dtype=np.float32)
            np.save(gt_dir / "000000.npy", gt)
            args = type("Args", (), {"gt_range_path": str(gt_dir), "output_range_path": str(out_dir),
                "output_mask_path": str(mask_dir), "meta_dir": str(meta_dir), "expected_height": 4,
                "expected_width": 5, "selected_rows": [1, 3], "invalid_value": 0.0})()
            process_one(0, args)
            anchor = np.load(out_dir / "000000.npy")
            self.assertFalse(np.any(anchor[[0, 2]]))
            self.assertTrue(np.all(anchor[[1, 3]] == gt[[1, 3]]))

    def test_hidden_row_leakage_is_detected(self):
        gt = np.ones((4, 5), dtype=np.float32) * 10
        anchor = np.zeros_like(gt)
        anchor[1, 2] = 10
        args = type("Args", (), {"enable_leakage_check": True, "leakage_method": "range_gdc",
            "range_min": 0.1, "range_max": 80.0, "source_rows": 1, "row_offset": None,
            "row_stride": None, "source_row_indices": [0], "selected_rows_dir": None,
            "projection_selected_rows": None, "expected_height": 4, "expected_width": 5,
            "leakage_anchor_row_tolerance": 0, "leakage_zero_eps": 1e-6,
            "leakage_warn_err_zero_ratio": 0.01, "leakage_warn_err_5cm_ratio": 0.8})()
        with self.assertRaises(ValueError):
            leakage_row(0, {"range_gdc": gt.copy()}, gt, anchor, args)

    def test_range_main_batch_split_filtering_requires_every_input_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pred, anchor = root / "pred", root / "anchor"
            pred.mkdir()
            anchor.mkdir()
            for scene_id in (0, 1):
                np.save(pred / f"{scene_id:06d}.npy", np.ones((2, 2), dtype=np.float32))
                np.save(anchor / f"{scene_id:06d}.npy", np.ones((2, 2), dtype=np.float32))
            split = root / "split.txt"
            split.write_text("1\n")
            self.assertEqual(select_scene_ids(npy_map(pred), npy_map(anchor), str(split)), ["000001"])
            split.write_text("2\n")
            with self.assertRaises(FileNotFoundError):
                select_scene_ids(npy_map(pred), npy_map(anchor), str(split))

    def test_fusion_summary_uses_global_percentiles_not_frame_percentile_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            directories = {name: root / name for name in ("gt", "guide", "source", "hard", "soft")}
            for path in directories.values():
                path.mkdir()
            split = root / "split.txt"
            split.write_text("0\n1\n")
            for scene_id, error in ((0, 0.0), (1, 10.0)):
                gt = np.full((2, 2), 10.0, dtype=np.float32)
                guide = gt.copy()
                source = gt.copy()
                hard = gt.copy()
                soft = gt.copy()
                hard[1, 1] += error
                soft[1, 1] += error * 2.0
                for name, value in (("gt", gt), ("guide", guide), ("source", source), ("hard", hard), ("soft", soft)):
                    np.save(directories[name] / f"{scene_id:06d}.npy", value)
            meta = root / "projection_meta.npz"
            np.savez(meta, selected_rows=np.asarray([0], dtype=np.int32))
            rows = aggregate_common_hidden_metrics(
                split, directories["gt"], directories["guide"], directories["source"],
                {"confidence_hard": directories["hard"], "soft": directories["soft"]},
                range_min=0.1, range_max=80.0, selected_rows=[0], projection_meta_path=meta,
            )
            hard = next(row for row in rows if row["selection_mode"] == "confidence_hard")
            self.assertEqual(hard["eval_pixels"], 4)
            self.assertAlmostEqual(hard["p90_abs"], 7.0)
            self.assertNotAlmostEqual(hard["p90_abs"], 4.5)


if __name__ == "__main__":
    unittest.main()
