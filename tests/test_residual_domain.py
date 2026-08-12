import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from range_gdc import range_main_batch
from range_gdc.range_gdc import RangeROIGDC


class ResidualDomainTests(unittest.TestCase):
    @staticmethod
    def fixture():
        guide = np.array([
            [10.0, 11.0, 20.0, 21.0],
            [10.5, 12.0, 22.0, 20.0],
            [9.0, 12.5, 24.0, 25.0],
        ], dtype=np.float32)
        anchor = np.zeros_like(guide)
        anchor[0, 1] = 12.0
        anchor[2, 2] = 23.0
        return guide, anchor

    @classmethod
    def run_domain(cls, residual_domain=None, delta_clip=None):
        guide, anchor = cls.fixture()
        kwargs = {
            "method": "spsolve",
            "anchor_reject": "none",
            "anchor_force_policy": "none",
            "delta_clip": delta_clip,
            "edge_range_mode": "log_gaussian",
            "return_stats": True,
            "return_debug": True,
        }
        if residual_domain is not None:
            kwargs["residual_domain"] = residual_domain
        return guide, anchor, RangeROIGDC(guide, anchor, **kwargs)

    def test_default_residual_domain_equals_explicit_log(self):
        guide, anchor = self.fixture()
        default = RangeROIGDC(
            guide, anchor, method="spsolve", return_stats=True,
            return_debug=True,
        )
        explicit = RangeROIGDC(
            guide, anchor, method="spsolve", residual_domain="log",
            return_stats=True, return_debug=True,
        )
        self.assertTrue(np.array_equal(default[0], explicit[0]))
        self.assertTrue(np.array_equal(default[1], explicit[1]))
        self.assertEqual((default[3]["L"] != explicit[3]["L"]).nnz, 0)
        for key in (
            "target_delta", "delta_graph", "delta_final", "edge_i",
            "edge_j", "edge_weight",
        ):
            self.assertTrue(np.array_equal(default[3][key], explicit[3][key]))
        self.assertEqual(default[2]["residual_domain"], "log")
        self.assertEqual(default[2]["method_tag"], "graph_log_range_residual")

    def test_linear_target_residual_is_anchor_minus_guide(self):
        guide, anchor, result = self.run_domain("linear", delta_clip=None)
        target_mask = result[3]["target_mask"]
        expected = anchor[target_mask].astype(np.float64) - guide[target_mask]
        self.assertTrue(np.array_equal(result[3]["target_delta"], expected))

    def test_log_target_residual_is_log_anchor_minus_log_guide(self):
        guide, anchor, result = self.run_domain("log", delta_clip=None)
        target_mask = result[3]["target_mask"]
        expected = (
            np.log(anchor[target_mask].astype(np.float64))
            - np.log(guide[target_mask].astype(np.float64))
        )
        self.assertTrue(np.array_equal(result[3]["target_delta"], expected))

    def test_domains_share_graph_and_anchor_acceptance(self):
        _, _, log_result = self.run_domain("log", delta_clip=None)
        _, _, linear_result = self.run_domain("linear", delta_clip=None)
        log_stats, linear_stats = log_result[2], linear_result[2]
        for key in (
            "N_nodes", "N_residual_targets", "N_edges_graph",
            "anchor_after_reject_count", "anchor_reject_count",
        ):
            self.assertEqual(log_stats[key], linear_stats[key])
        for key in ("edge_i", "edge_j", "edge_weight"):
            self.assertTrue(np.array_equal(
                log_result[3][key], linear_result[3][key]
            ))
        self.assertTrue(np.array_equal(
            log_result[3]["target_mask"], linear_result[3]["target_mask"]
        ))

    def test_linear_reconstruction_is_raw_plus_delta(self):
        guide, _, result = self.run_domain("linear", delta_clip=None)
        corrected, _, stats, debug = result
        expected = np.clip(
            guide[debug["guide_valid"]].astype(np.float64)
            + debug["delta_final"],
            0.1,
            80.0,
        ).astype(np.float32)
        self.assertTrue(np.array_equal(corrected[debug["guide_valid"]], expected))
        self.assertEqual(stats["method_tag"], "graph_linear_range_residual")

    def test_log_reconstruction_is_exp_of_log_raw_plus_delta(self):
        guide, _, result = self.run_domain("log", delta_clip=None)
        corrected, _, _, debug = result
        expected = np.clip(
            np.exp(
                np.log(guide[debug["guide_valid"]].astype(np.float64))
                + debug["delta_final"]
            ),
            0.1,
            80.0,
        ).astype(np.float32)
        self.assertTrue(np.array_equal(corrected[debug["guide_valid"]], expected))

    def test_invalid_residual_domain_is_rejected(self):
        guide, anchor = self.fixture()
        with self.assertRaisesRegex(ValueError, "residual_domain"):
            RangeROIGDC(guide, anchor, residual_domain="invalid")

    def test_cli_residual_domain_modes_and_default(self):
        required = [
            "--pred_path", "pred", "--anchor_path", "anchor",
            "--output_path", "out", "--mask_output_path", "mask",
        ]
        default = range_main_batch.parse_args(required)
        explicit_log = range_main_batch.parse_args(
            required + ["--residual_domain", "log"]
        )
        linear = range_main_batch.parse_args(
            required + ["--residual_domain", "linear"]
        )
        self.assertEqual(default.residual_domain, "log")
        self.assertEqual(explicit_log.residual_domain, "log")
        self.assertEqual(linear.residual_domain, "linear")

    def test_disable_delta_clip_resolves_to_none(self):
        required = [
            "--pred_path", "pred", "--anchor_path", "anchor",
            "--output_path", "out", "--mask_output_path", "mask",
        ]
        default = range_main_batch.parse_args(required)
        disabled = range_main_batch.parse_args(
            required + ["--disable_delta_clip"]
        )
        self.assertEqual(default.delta_clip, 0.3)
        self.assertIsNone(disabled.delta_clip)

    def test_process_one_passes_disabled_delta_clip_as_none(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pred_file = root / "pred.npy"
            anchor_file = root / "anchor.npy"
            guide, anchor = self.fixture()
            np.save(pred_file, guide)
            np.save(anchor_file, anchor)
            args = {
                "overwrite": True,
                "method": "spsolve",
                "range_min": 0.1,
                "range_max": 80.0,
                "anchor_reject": "none",
                "log_ratio_thr": 0.4,
                "abs_error_thr": 2.0,
                "lambda_anchor": 300.0,
                "lambda_prior": 0.1,
                "lambda_smooth": 1.0,
                "neighbor": "angular_grid8",
                "edge_spatial_mode": "angular",
                "edge_range_mode": "log_gaussian",
                "residual_domain": "linear",
                "sigma_angular": 0.01,
                "sigma_tangent": 1.0,
                "sigma_log_range": 0.3,
                "max_log_range_diff": None,
                "delta_clip": None,
                "anchor_force_policy": "accepted_only",
                "verbose": False,
            }
            mocked_result = (
                guide.copy(), np.ones_like(guide, dtype=bool), {}
            )
            with mock.patch.object(
                range_main_batch, "RangeROIGDC", return_value=mocked_result
            ) as mocked:
                range_main_batch.process_one((
                    "000000", str(pred_file), str(anchor_file),
                    str(root / "out"), str(root / "mask"), guide.shape,
                    None, None, "full_360_front_centered", args,
                ))
            self.assertIsNone(mocked.call_args.kwargs["delta_clip"])
            self.assertEqual(
                mocked.call_args.kwargs["residual_domain"], "linear"
            )


if __name__ == "__main__":
    unittest.main()
