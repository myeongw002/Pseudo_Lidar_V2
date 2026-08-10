import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_r64_pipeline as pipeline
from gdc.gdc import anchor_accept_mask, subsample_mask_by_grid
from range_gdc.evaluate_range_metrics import area_supports, leakage_row
from range_gdc.range_gdc import _apply_anchor_reject
from range_gdc.range_projection import range_to_velo
from range_gdc.range_to_pointcloud import load_centers
from src.pseudo_lidar.depth_to_range_uniform import (
    lidar_points_to_spherical_guide_uniform,
    make_uniform_vertical_grid,
    projection_metadata,
)


class PaperPipelineTests(unittest.TestCase):
    def test_legacy_method_specific_defaults_are_preserved(self):
        cfg = pipeline.migrate_config({
            "gdc": {"disable_subsample": True},
            "range_gdc": {"anchor_reject": "none"},
        })
        self.assertTrue(cfg["original_gdc"]["run_naive"])
        self.assertFalse(cfg["original_gdc"]["run_optimized"])
        self.assertNotIn("anchor_reject", cfg["original_gdc"])
        self.assertEqual(cfg["range_gdc"]["anchor_reject"], "none")

    def test_shared_sparse_directory_is_both_anchor_source(self):
        cfg = pipeline.migrate_config({
            "output_root": "/tmp/paper",
            "kitti_root": "/tmp/kitti",
            "split_file": "split/test_1.txt",
            "sdn_config": "src/configs/sdn_kitti_train.config",
            "sdn_checkpoint": "/tmp/model.pth",
            "data_tag": "test",
            "anchor": {"mode": "shared_pointcloud", "selected_lines": [5, 7, 9, 11]},
            "original_gdc": {},
            "range_gdc": {},
        })
        paths = pipeline.collect_paths(cfg, ["gdc", "range_gdc"])
        depth_cmd = pipeline.build_ptc2depthmap_command(
            paths["anchor_sparse_ptc_path"], paths["anchor_depthmap_path"], paths, 1
        )
        range_cmd = pipeline.build_range_projection_command(
            "ptc-to-range", paths["anchor_sparse_ptc_path"],
            paths["anchor_range_root_path"], paths, {}, 1
        )
        self.assertIn(str(paths["anchor_sparse_ptc_path"]), depth_cmd)
        self.assertIn(str(paths["anchor_sparse_ptc_path"]), range_cmd)

    def test_mismatched_method_anchor_directory_is_rejected(self):
        cfg = pipeline.migrate_config({
            "output_root": "/tmp/paper_mismatch",
            "kitti_root": "/tmp/kitti",
            "split_file": "split/test_1.txt",
            "sdn_config": "src/configs/sdn_kitti_train.config",
            "sdn_checkpoint": "/tmp/model.pth", "data_tag": "test",
            "anchor": {"mode": "shared_pointcloud", "selected_lines": [5]},
            "original_gdc": {"shared_anchor_pointcloud_path": "/tmp/wrong"},
            "range_gdc": {},
        })
        paths = pipeline.collect_paths(cfg, ["gdc", "range_gdc"])
        with self.assertRaises(ValueError):
            pipeline.validate_shared_anchor_config(cfg, paths, ["gdc", "range_gdc"])

    def test_anchor_reject_modes_and_strict_boundaries(self):
        pred = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
        anchor = np.array([[12.0, 11.999, 10.0 * np.exp(0.4)]], dtype=np.float32)
        candidate = np.ones_like(pred, dtype=bool)
        self.assertTrue(np.all(anchor_accept_mask(pred, anchor, candidate, "none")))
        accepted_abs = anchor_accept_mask(pred, anchor, candidate, "abs", 2.0, 0.4)
        self.assertEqual(accepted_abs.tolist(), [[False, True, False]])
        accepted_log = anchor_accept_mask(pred, anchor, candidate, "log_ratio", 2.0, 0.4)
        self.assertFalse(bool(accepted_log[0, 2]))
        range_kept, rejected = _apply_anchor_reject(
            candidate, pred, anchor, "abs", 0.4, 2.0
        )
        self.assertFalse(bool(range_kept[0, 0]))
        self.assertEqual(rejected, 2)

    def _roundtrip_projection(self, mode, amin=None, amax=None):
        points = np.array([[10.0, 0.0, 0.0], [8.0, 2.0, -1.0]], dtype=np.float32)
        image, mask = lidar_points_to_spherical_guide_uniform(
            points, range_h=16, range_w=64, vmin_deg=-25, vmax_deg=5,
            azimuth_mode=mode, azimuth_min_deg=amin, azimuth_max_deg=amax,
            range_min=0.1, range_max=80.0,
        )
        args = SimpleNamespace(
            range_h=16, range_w=64, vmin_deg=-25, vmax_deg=5,
            azimuth_mode=mode, azimuth_min_deg=amin, azimuth_max_deg=amax,
            meta_path=None,
        )
        reconstructed = range_to_velo(image, args)
        image2, mask2 = lidar_points_to_spherical_guide_uniform(
            reconstructed, range_h=16, range_w=64, vmin_deg=-25, vmax_deg=5,
            azimuth_mode=mode, azimuth_min_deg=amin, azimuth_max_deg=amax,
            range_min=0.1, range_max=80.0,
        )
        self.assertTrue(np.array_equal(mask, mask2))
        self.assertTrue(np.allclose(image[mask], image2[mask2], atol=1e-5))

    def test_full_360_projection_inverse(self):
        self._roundtrip_projection("full_360_front_centered")

    def test_bounded_projection_inverse(self):
        self._roundtrip_projection("bounded", -45.0, 45.0)

    def test_metadata_free_vertical_fallback_uses_highest_row_zero(self):
        vertical, _ = load_centers(None, 8, 16, -24.9, 2.0,
                                   "full_360_front_centered", None, None)
        expected = make_uniform_vertical_grid(8, -24.9, 2.0)[1]
        self.assertTrue(np.array_equal(vertical, expected))
        self.assertGreater(vertical[0], vertical[-1])

    def test_deterministic_subsampling(self):
        rng = np.random.default_rng(4)
        points = np.column_stack((
            rng.uniform(-20, 20, 500), rng.uniform(-0.9, 2.4, 500),
            rng.uniform(1.1, 70, 500),
        ))
        a = subsample_mask_by_grid(points, "deterministic", 0)
        b = subsample_mask_by_grid(points, "deterministic", 999)
        self.assertTrue(np.array_equal(a, b))

    def test_common_hidden_valid_and_coverage_support(self):
        gt = np.ones((4, 5), dtype=np.float32) * 10
        guide = gt.copy()
        guide[1, :] = 0
        args = SimpleNamespace(
            range_min=0.1, range_max=80.0, source_rows=2, row_offset=None,
            row_stride=None, source_row_indices=[0], selected_rows_dir=None,
            projection_selected_rows=None,
        )
        supports, _, _, gt_valid = area_supports(0, gt, guide, guide, args)
        pred_a = gt > 0
        pred_b = gt > 0
        pred_b[2, 0] = False
        common_hidden = supports["hidden_rows_valid"] & gt_valid & pred_a & pred_b
        self.assertFalse(common_hidden[0].any())
        self.assertFalse(common_hidden[2, 0])
        denominator = int((gt_valid & supports["hidden_rows_valid"]).sum())
        self.assertEqual(int(common_hidden.sum()), denominator - 1)

    def test_hidden_row_anchor_leakage_is_detected(self):
        gt = np.ones((4, 5), dtype=np.float32) * 10
        pred = gt.copy()
        anchor = np.zeros_like(gt)
        anchor[1, 2] = 10
        args = SimpleNamespace(
            enable_leakage_check=True, leakage_method="range_gdc",
            range_min=0.1, range_max=80.0, source_rows=1,
            row_offset=None, row_stride=None, source_row_indices=[0],
            selected_rows_dir=None, projection_selected_rows=None,
            expected_height=4, expected_width=5,
            leakage_anchor_row_tolerance=0, leakage_zero_eps=1e-6,
            leakage_warn_err_zero_ratio=0.01,
            leakage_warn_err_5cm_ratio=0.8,
        )
        with self.assertRaises(ValueError):
            leakage_row(0, {"range_gdc": pred}, gt, anchor, args)

    def test_projection_metadata_fields(self):
        meta = projection_metadata(64, 1024, -24.9, 2.0,
                                   "full_360_front_centered")
        for key in (
            "azimuth_min_deg", "azimuth_max_deg", "azimuth_span_deg",
            "horizontal_resolution_deg", "vertical_resolution_deg",
            "row0_elevation_direction", "camera_fov_only",
            "valid_column_count", "valid_azimuth_span_deg",
        ):
            self.assertIn(key, meta)


if __name__ == "__main__":
    unittest.main()
