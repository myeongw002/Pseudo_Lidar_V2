import inspect
import subprocess
import sys

import numpy as np
from scipy import sparse

try:
    from .range_gdc import (
        RangeROIGDC,
        _apply_anchor_reject,
        build_graph_residual_system,
        build_spherical_graph_laplacian,
        valid_range_mask,
    )
except ImportError:
    from range_gdc import (
        RangeROIGDC,
        _apply_anchor_reject,
        build_graph_residual_system,
        build_spherical_graph_laplacian,
        valid_range_mask,
    )


def make_fixture():
    guide = np.zeros((8, 16), dtype=np.float32)
    anchor = np.zeros_like(guide)
    guide[2:6, 4:12] = 10.0
    anchor[3, 6] = 9.0
    return guide, anchor


def make_node_index(guide):
    guide_valid = valid_range_mask(guide)
    rows, cols = np.where(guide_valid)
    node_id = -np.ones_like(guide, dtype=np.int32)
    node_id[rows, cols] = np.arange(len(rows), dtype=np.int32)
    return guide_valid, rows, cols, node_id


def build_laplacian(guide, neighbor="angular_grid8", sigma_angular=0.01):
    guide_valid, rows, cols, node_id = make_node_index(guide)
    L, stats, debug = build_spherical_graph_laplacian(
        guide, guide_valid, node_id, rows, cols,
        neighbor=neighbor, edge_spatial_mode="angular",
        sigma_angular=sigma_angular, sigma_log_range=0.3,
    )
    return guide_valid, rows, cols, node_id, L, stats, debug


def test_graph_laplacian_shape_and_positive_weights():
    guide, _ = make_fixture()
    guide_valid, _, _, _, L, stats, debug = build_laplacian(guide)
    assert L.shape == (int(guide_valid.sum()), int(guide_valid.sum()))
    assert stats["N_edges_graph"] > 0
    assert np.all(np.isfinite(debug["edge_weight"]))
    assert np.all(debug["edge_weight"] > 0.0)


def test_horizontal_azimuth_wrap():
    guide = np.full((1, 4), 10.0, dtype=np.float32)
    _, _, _, _, _, _, debug = build_laplacian(
        guide, neighbor="angular_grid4", sigma_angular=10.0,
    )
    pairs = {frozenset((int(i), int(j))) for i, j in zip(debug["edge_i"], debug["edge_j"])}
    assert frozenset((0, 3)) in pairs


def test_same_surface_edge_is_stronger_than_range_discontinuity():
    guide = np.array([[10.0, 10.0, 30.0]], dtype=np.float32)
    _, _, _, _, _, _, debug = build_laplacian(
        guide, neighbor="angular_grid4", sigma_angular=10.0,
    )
    same = debug["edge_weight"][debug["log_range_diff"] == 0.0]
    discontinuous = debug["edge_weight"][debug["log_range_diff"] > 0.5]
    assert same.size and discontinuous.size
    assert float(np.min(same)) > float(np.max(discontinuous))


def test_graph_residual_system_shape_and_anchor_constraint():
    L = sparse.csr_matrix((3, 3), dtype=np.float64)
    A, b = build_graph_residual_system(
        L, np.array([1]), np.array([0.25]),
        lambda_anchor=300.0, lambda_prior=0.1, lambda_smooth=1.0,
    )
    assert A.shape == (3, 3) and b.shape == (3,)
    assert np.isclose(A[1, 1], 300.1)
    assert np.isclose(b[1], 75.0)
    assert np.count_nonzero(b) == 1


def test_binary_rejection_threshold_boundary():
    guide = np.array([[10.0, 10.0]], dtype=np.float32)
    anchor = np.array([[12.0, 11.999]], dtype=np.float32)
    kept, rejected = _apply_anchor_reject(
        np.ones_like(guide, dtype=bool), guide, anchor, "abs", 0.4, 2.0,
    )
    assert kept.tolist() == [[False, True]]
    assert rejected == 1


def test_delta_clipping():
    guide = np.array([[10.0]], dtype=np.float32)
    anchor = np.array([[30.0]], dtype=np.float32)
    corrected, _, _, debug = RangeROIGDC(
        guide, anchor, method="spsolve", anchor_reject="none",
        delta_clip=0.3, anchor_force_policy="none",
        return_stats=True, return_debug=True,
    )
    assert np.isclose(debug["target_delta"][0], 0.3)
    assert np.isclose(debug["delta_final"][0], 0.3)
    assert np.isclose(corrected[0, 0], 10.0 * np.exp(0.3), rtol=1e-6)


def test_accepted_only_force_uses_anchor_range():
    guide, anchor = make_fixture()
    corrected, _, stats = RangeROIGDC(
        guide, anchor, method="spsolve", anchor_reject="none",
        anchor_force_policy="accepted_only", return_stats=True,
    )
    target = valid_range_mask(guide) & valid_range_mask(anchor)
    assert np.array_equal(corrected[target], anchor[target])
    assert stats["anchor_forced_count"] == int(target.sum())


def test_rejected_anchor_is_not_forced():
    guide = np.array([[10.0]], dtype=np.float32)
    anchor = np.array([[12.0]], dtype=np.float32)
    corrected, _, stats = RangeROIGDC(
        guide, anchor, method="spsolve", anchor_reject="abs",
        abs_error_thr=2.0, anchor_force_policy="accepted_only", return_stats=True,
    )
    assert corrected[0, 0] == guide[0, 0]
    assert corrected[0, 0] != anchor[0, 0]
    assert stats["anchor_reject_count"] == 1
    assert stats["anchor_forced_count"] == 0


def test_output_mask_preserves_guide_valid_domain():
    guide, anchor = make_fixture()
    anchor[0, 0] = 5.0
    _, mask = RangeROIGDC(guide, anchor, method="spsolve", anchor_reject="none")
    assert np.array_equal(mask, valid_range_mask(guide))
    assert not mask[0, 0]


def test_empty_node_behavior():
    guide = np.zeros((2, 3), dtype=np.float32)
    corrected, mask, stats = RangeROIGDC(guide, guide, return_stats=True)
    assert np.array_equal(corrected, guide)
    assert not np.any(mask)
    assert stats["status"] == "empty_nodes"


def test_cg_and_spsolve_sanity():
    guide, anchor = make_fixture()
    outputs = []
    for method in ("cg", "spsolve"):
        corrected, mask, stats = RangeROIGDC(
            guide, anchor, method=method, anchor_reject="none", return_stats=True,
        )
        assert stats["solver_info"] == "0"
        assert np.all(np.isfinite(corrected[mask]))
        outputs.append(corrected)
    assert np.allclose(outputs[0], outputs[1], atol=1e-5, rtol=1e-5)


def test_legacy_api_and_cli_removed():
    parameters = inspect.signature(RangeROIGDC).parameters
    for name in (
        "transfer_k", "confidence_mode", "selection_mode", "ablation_mode",
        "force_anchor_value", "anchor_reliability_mode", "anchor_reliability_scale",
    ):
        assert name not in parameters
    result = subprocess.run(
        [sys.executable, "-m", "range_gdc.range_main_batch", "--help"],
        check=True, capture_output=True, text=True,
    )
    for flag in (
        "transfer_k", "confidence_mode", "selection_mode", "ablation_mode",
        "force_anchor_value", "anchor_reliability_mode", "anchor_reliability_scale",
    ):
        assert flag not in result.stdout


def main():
    test_graph_laplacian_shape_and_positive_weights()
    test_horizontal_azimuth_wrap()
    test_same_surface_edge_is_stronger_than_range_discontinuity()
    test_graph_residual_system_shape_and_anchor_constraint()
    test_binary_rejection_threshold_boundary()
    test_delta_clipping()
    test_accepted_only_force_uses_anchor_range()
    test_rejected_anchor_is_not_forced()
    test_output_mask_preserves_guide_valid_domain()
    test_empty_node_behavior()
    test_cg_and_spsolve_sanity()
    test_legacy_api_and_cli_removed()
    print("range_gdc_sanity: OK")


if __name__ == "__main__":
    main()
