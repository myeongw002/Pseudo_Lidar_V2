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
from tools import run_range_gdc_experiment as pipeline
from tools.run_fusion_comparison import aggregate_common_hidden_metrics


class PaperPipelineTests(unittest.TestCase):
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
