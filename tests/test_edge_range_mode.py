import unittest

import numpy as np

from range_gdc import range_main_batch
from range_gdc.range_gdc import (
    RangeROIGDC,
    build_spherical_graph_laplacian,
    valid_range_mask,
)


class EdgeRangeModeTests(unittest.TestCase):
    @staticmethod
    def graph_fixture(edge_range_mode=None):
        guide = np.array([
            [10.0, 10.0, 20.0, 20.0],
            [10.0, 12.0, 24.0, 20.0],
            [8.0, 12.0, 24.0, 30.0],
        ], dtype=np.float32)
        guide_valid = valid_range_mask(guide)
        node_rows, node_cols = np.where(guide_valid)
        node_id = np.full(guide.shape, -1, dtype=np.int32)
        node_id[node_rows, node_cols] = np.arange(node_rows.size, dtype=np.int32)
        kwargs = {
            "neighbor": "angular_grid8",
            "edge_spatial_mode": "angular",
            "sigma_angular": 1.0,
            "sigma_log_range": 0.3,
            "max_log_range_diff": None,
        }
        if edge_range_mode is not None:
            kwargs["edge_range_mode"] = edge_range_mode
        return build_spherical_graph_laplacian(
            guide, guide_valid, node_id, node_rows, node_cols, **kwargs
        )

    def test_default_mode_equals_explicit_log_gaussian(self):
        default = self.graph_fixture()
        explicit = self.graph_fixture("log_gaussian")
        self.assertEqual((default[0] != explicit[0]).nnz, 0)
        self.assertEqual(default[1]["edge_range_mode"], "log_gaussian")
        for key in default[2]:
            self.assertTrue(np.array_equal(default[2][key], explicit[2][key]))

    def test_uniform_range_gate_is_one_on_every_retained_edge(self):
        _, stats, debug = self.graph_fixture("uniform")
        self.assertGreater(debug["range_gate"].size, 0)
        self.assertTrue(np.array_equal(
            debug["range_gate"], np.ones_like(debug["range_gate"])
        ))
        self.assertEqual(stats["range_gate_mean"], 1.0)

    def test_uniform_edge_weight_equals_spatial_weight(self):
        _, _, debug = self.graph_fixture("uniform")
        self.assertTrue(np.array_equal(
            debug["edge_weight"], debug["spatial_weight"]
        ))

    def test_log_gaussian_gate_and_weight_match_existing_formula(self):
        _, _, debug = self.graph_fixture("log_gaussian")
        expected_gate = np.exp(
            -(debug["log_range_diff"] ** 2) / (2.0 * 0.3 ** 2)
        )
        self.assertTrue(np.array_equal(debug["range_gate"], expected_gate))
        self.assertTrue(np.array_equal(
            debug["edge_weight"], debug["spatial_weight"] * expected_gate
        ))

    def test_modes_have_identical_edge_topology(self):
        full = self.graph_fixture("log_gaussian")[2]
        angular_only = self.graph_fixture("uniform")[2]
        self.assertTrue(np.array_equal(full["edge_i"], angular_only["edge_i"]))
        self.assertTrue(np.array_equal(full["edge_j"], angular_only["edge_j"]))

    def test_invalid_edge_range_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "edge_range_mode"):
            self.graph_fixture("invalid")
        guide = np.full((2, 3), 10.0, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "edge_range_mode"):
            RangeROIGDC(guide, np.zeros_like(guide), edge_range_mode="invalid")

    def test_range_gdc_default_output_preserves_log_gaussian_semantics(self):
        guide = np.array([
            [10.0, 10.0, 20.0, 20.0],
            [10.0, 12.0, 24.0, 20.0],
            [8.0, 12.0, 24.0, 30.0],
        ], dtype=np.float32)
        anchor = np.zeros_like(guide)
        anchor[1, 1] = 11.0
        default = RangeROIGDC(
            guide, anchor, method="spsolve", return_stats=True,
            return_debug=True,
        )
        explicit = RangeROIGDC(
            guide, anchor, method="spsolve",
            edge_range_mode="log_gaussian", return_stats=True,
            return_debug=True,
        )
        self.assertTrue(np.array_equal(default[0], explicit[0]))
        self.assertTrue(np.array_equal(default[1], explicit[1]))
        self.assertEqual((default[3]["L"] != explicit[3]["L"]).nnz, 0)
        self.assertTrue(np.array_equal(
            default[3]["edge_weight"], explicit[3]["edge_weight"]
        ))
        self.assertEqual(default[2]["edge_range_mode"], "log_gaussian")

    def test_cli_parser_accepts_both_modes_and_defaults_to_log_gaussian(self):
        required = [
            "--pred_path", "pred", "--anchor_path", "anchor",
            "--output_path", "out", "--mask_output_path", "mask",
        ]
        default = range_main_batch.parse_args(required)
        full = range_main_batch.parse_args(
            required + ["--edge_range_mode", "log_gaussian"]
        )
        uniform = range_main_batch.parse_args(
            required + ["--edge_range_mode", "uniform"]
        )
        self.assertEqual(default.edge_range_mode, "log_gaussian")
        self.assertEqual(full.edge_range_mode, "log_gaussian")
        self.assertEqual(uniform.edge_range_mode, "uniform")


if __name__ == "__main__":
    unittest.main()
