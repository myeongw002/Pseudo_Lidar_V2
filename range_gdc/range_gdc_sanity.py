import subprocess
import sys

import numpy as np

try:
    from .range_gdc import (
        RangeROIGDC,
        build_direct_residual_transfer,
        build_graph_residual_system,
        build_spherical_graph_laplacian,
        select_residuals,
        valid_range_mask,
    )
except ImportError:
    from range_gdc import (
        RangeROIGDC,
        build_direct_residual_transfer,
        build_graph_residual_system,
        build_spherical_graph_laplacian,
        select_residuals,
        valid_range_mask,
    )


def make_fixture():
    H, W = 8, 16
    guide = np.zeros((H, W), dtype=np.float32)
    anchor = np.zeros((H, W), dtype=np.float32)
    guide[2:6, 4:12] = 10.0
    anchor[3, 6] = 9.0
    return guide, anchor


def make_node_index(guide):
    guide_valid = valid_range_mask(guide)
    rows, cols = np.where(guide_valid)
    node_id = -np.ones_like(guide, dtype=np.int32)
    node_id[rows, cols] = np.arange(len(rows), dtype=np.int32)
    return guide_valid, rows, cols, node_id


def test_graph_laplacian_shapes_and_positive_weights():
    guide, _ = make_fixture()
    guide_valid, rows, cols, node_id = make_node_index(guide)
    L, stats, debug = build_spherical_graph_laplacian(
        guide,
        guide_valid,
        node_id,
        rows,
        cols,
        neighbor="angular_grid8",
        edge_spatial_mode="angular",
        sigma_angular=0.01,
        sigma_log_range=0.3,
    )
    N = int(guide_valid.sum())
    assert L.shape == (N, N)
    assert stats["N_edges_graph"] > 0
    assert np.all(np.isfinite(debug["edge_weight"]))
    assert np.all(debug["edge_weight"] > 0.0)

    target_idx = np.array([node_id[3, 6]], dtype=np.int64)
    target_delta = np.array([np.log(9.0 / 10.0)], dtype=np.float64)
    A, b = build_graph_residual_system(
        L,
        target_node_indices=target_idx,
        target_delta=target_delta,
        lambda_anchor=300.0,
        lambda_prior=0.05,
        lambda_smooth=1.0,
    )
    assert A.shape == (N, N)
    assert b.shape == (N,)


def test_select_residuals_soft():
    final, stats, _ = select_residuals(
        np.array([0.0, 0.0]),
        np.array([1.0, -1.0]),
        np.array([0.25, 0.75]),
        np.array([0.01, 0.3]),
        selection_mode="soft",
    )
    assert np.allclose(final, [0.25, -0.75])
    assert stats["selection_direct_count"] + stats["selection_graph_count"] + stats["selection_blend_count"] == 2


def test_select_residuals_confidence_hard():
    graph = np.array([0.0, 0.0, 0.0])
    direct = np.array([1.0, -1.0, 2.0])
    conf = np.array([0.9, 0.1, 0.5])
    dlog = np.array([0.01, 0.01, 0.01])
    final, stats, _ = select_residuals(
        graph,
        direct,
        conf,
        dlog,
        selection_mode="confidence_hard",
        confidence_high_thr=0.8,
        confidence_low_thr=0.2,
    )
    assert np.allclose(final, [1.0, 0.0, 1.0])
    assert stats["selection_direct_count"] == 1
    assert stats["selection_graph_count"] == 1
    assert stats["selection_blend_count"] == 1


def test_select_residuals_log_range_piecewise():
    graph = np.array([0.0, 0.0, 0.0])
    direct = np.array([1.0, -1.0, 2.0])
    conf = np.array([0.4, 0.4, 0.4])
    dlog = np.array([0.01, 0.3, 0.1])
    final, stats, _ = select_residuals(
        graph,
        direct,
        conf,
        dlog,
        selection_mode="log_range_piecewise",
        direct_log_range_thr=0.05,
        graph_log_range_thr=0.2,
    )
    assert np.allclose(final, [1.0, 0.0, 0.8])
    assert stats["selection_direct_count"] == 1
    assert stats["selection_graph_count"] == 1
    assert stats["selection_blend_count"] == 1


def test_select_residuals_invalid_fallback():
    final_direct_nan, stats_direct_nan, _ = select_residuals(
        np.array([0.2]),
        np.array([np.nan]),
        np.array([1.0]),
        np.array([0.01]),
        selection_mode="log_range_piecewise",
    )
    assert np.allclose(final_direct_nan, [0.2])
    assert stats_direct_nan["selection_graph_count"] == 1
    assert stats_direct_nan["selection_invalid_direct_count"] == 1

    final_dlog_nan, stats_dlog_nan, _ = select_residuals(
        np.array([0.2]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([np.nan]),
        selection_mode="log_range_piecewise",
    )
    assert np.allclose(final_dlog_nan, [0.2])
    assert stats_dlog_nan["selection_graph_count"] == 1
    assert stats_dlog_nan["selection_invalid_dlog_count"] == 1


def test_range_consistent_direct_transfer():
    guide, anchor = make_fixture()
    corrected, mask, stats, debug = RangeROIGDC(
        guide,
        anchor,
        method="spsolve",
        anchor_reject="none",
        lambda_anchor=300.0,
        lambda_prior=0.05,
        lambda_smooth=1.0,
        sigma_conf_pixel=4.0,
        sigma_conf_log_range=0.2,
        selection_mode="log_range_piecewise",
        direct_log_range_thr=0.05,
        graph_log_range_thr=0.2,
        delta_clip=0.5,
        force_anchor_value=False,
        return_stats=True,
        return_debug=True,
    )
    hidden = debug["node_id"][3, 7]
    expected = np.log(9.0 / 10.0)
    assert stats["confidence_high_ratio"] > 0.0
    assert stats["selection_direct_count"] > 0
    assert abs(debug["delta_direct"][hidden] - expected) < 1e-6
    assert abs(debug["delta_final"][hidden] - debug["delta_direct"][hidden]) < abs(
        debug["delta_graph"][hidden] - debug["delta_direct"][hidden]
    )
    assert abs(float(corrected[3, 7]) - 9.0) < 0.2
    assert np.array_equal(mask, guide > 0)


def test_range_discontinuous_fallback():
    H, W = 6, 8
    guide = np.zeros((H, W), dtype=np.float32)
    anchor = np.zeros((H, W), dtype=np.float32)
    guide[2, 2] = 10.0
    guide[2, 3] = 40.0
    guide[2, 4] = 40.0
    anchor[2, 2] = 8.0

    corrected, _, _, debug = RangeROIGDC(
        guide,
        anchor,
        method="spsolve",
        anchor_reject="none",
        lambda_anchor=300.0,
        lambda_prior=0.05,
        lambda_smooth=1.0,
        sigma_conf_pixel=4.0,
        sigma_conf_log_range=0.05,
        selection_mode="log_range_piecewise",
        direct_log_range_thr=0.05,
        graph_log_range_thr=0.2,
        delta_clip=0.5,
        force_anchor_value=False,
        return_stats=True,
        return_debug=True,
    )
    target_node = debug["node_id"][2, 3]
    assert debug["confidence"][target_node] < 0.01
    assert debug["selection_graph_mask"][target_node]
    assert abs(debug["delta_final"][target_node] - debug["delta_graph"][target_node]) < abs(
        debug["delta_final"][target_node] - debug["delta_direct"][target_node]
    )
    assert corrected[2, 3] > 30.0


def test_confidence_monotonicity():
    raw_log_nodes = np.log(np.array([10.0, 10.1, 11.0, 16.5], dtype=np.float64))
    rows = np.array([0, 0, 0, 0], dtype=np.int64)
    cols = np.array([0, 1, 2, 3], dtype=np.int64)
    target_idx = np.array([0], dtype=np.int64)
    target_delta = np.array([np.log(9.0 / 10.0)], dtype=np.float64)
    _, conf, _, _, _, _ = build_direct_residual_transfer(
        raw_log_nodes,
        rows,
        cols,
        target_idx,
        target_delta,
        shape=(1, 16),
        transfer_k=1,
        transfer_neighbor_mode="rowcol",
        direct_weight_mode="nearest",
        confidence_mode="nearest",
        sigma_conf_pixel=100.0,
        sigma_conf_log_range=0.1,
    )
    assert conf[1] > conf[2] > conf[3]


def test_output_mask_excludes_guide_invalid_and_force_anchor():
    guide, anchor = make_fixture()
    corrected, mask, stats = RangeROIGDC(
        guide,
        anchor,
        method="spsolve",
        anchor_reject="none",
        force_anchor_value=True,
        return_stats=True,
    )
    target = (guide > 0) & (anchor > 0)
    expected = (guide > 0) & np.isfinite(corrected) & (corrected >= 1.0) & (corrected <= 80.0)
    assert np.array_equal(mask, expected)
    assert not np.any(mask[guide <= 0])
    assert np.allclose(corrected[target], anchor[target])
    assert stats["anchor_abs_error_after_force"] == 0.0


def test_anchor_overlap_before_reject_count():
    guide = np.zeros((4, 4), dtype=np.float32)
    anchor = np.zeros((4, 4), dtype=np.float32)
    guide[1, 1] = 10.0
    guide[1, 2] = 10.0
    anchor[1, 1] = 8.0
    anchor[1, 2] = 1.0

    _, _, stats = RangeROIGDC(
        guide,
        anchor,
        method="spsolve",
        anchor_reject="log_ratio",
        log_ratio_thr=0.4,
        force_anchor_value=False,
        return_stats=True,
    )
    assert stats["N_anchor_overlap"] == 2
    assert stats["N_residual_targets"] == 1
    assert stats["N_rejected_residual_targets"] == 1


def test_old_bias_cli_removed():
    cmd = [sys.executable, "-m", "range_gdc.range_main_batch", "--help"]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "global_bias_mode" not in result.stdout
    assert "bias_spherical_graph" not in result.stdout
    assert "transfer_k" in result.stdout
    assert "selection_mode" in result.stdout


def main():
    test_graph_laplacian_shapes_and_positive_weights()
    test_select_residuals_soft()
    test_select_residuals_confidence_hard()
    test_select_residuals_log_range_piecewise()
    test_select_residuals_invalid_fallback()
    test_range_consistent_direct_transfer()
    test_range_discontinuous_fallback()
    test_confidence_monotonicity()
    test_output_mask_excludes_guide_invalid_and_force_anchor()
    test_anchor_overlap_before_reject_count()
    test_old_bias_cli_removed()
    print("range_gdc_sanity: OK")


if __name__ == "__main__":
    main()
