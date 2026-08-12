import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve


ANCHOR_REJECT_MODES = {"log_ratio", "abs", "none"}
NEIGHBOR_MODES = {"angular_grid4", "angular_grid8"}
EDGE_SPATIAL_MODES = {"angular", "tangent"}
EDGE_RANGE_MODES = {"log_gaussian", "uniform"}
RESIDUAL_DOMAINS = {"log", "linear"}


def valid_range_mask(range_img, range_min=0.1, range_max=80.0):
    range_img = np.asarray(range_img)
    return (
        np.isfinite(range_img)
        & (range_img >= range_min)
        & (range_img <= range_max)
    )


def make_uniform_vertical_centers(height, vmin_deg=-24.9, vmax_deg=2.0):
    if vmax_deg <= vmin_deg:
        raise ValueError("vmax_deg must be larger than vmin_deg")
    edges_bottom_to_top = np.linspace(vmin_deg, vmax_deg, height + 1, dtype=np.float64)
    centers_bottom_to_top = 0.5 * (edges_bottom_to_top[:-1] + edges_bottom_to_top[1:])
    return centers_bottom_to_top[::-1]


def make_azimuth_centers(width, mode="full_360_front_centered"):
    if mode in {"front_center", "full_360_front_centered"}:
        edges = np.linspace(-180.0, 180.0, width + 1, dtype=np.float64)
    elif mode == "front_zero":
        edges = np.linspace(0.0, 360.0, width + 1, dtype=np.float64)
    else:
        raise ValueError(f"Unknown azimuth mode: {mode}")
    return 0.5 * (edges[:-1] + edges[1:])


def _angle_centers_for_shape(
    shape,
    *,
    vertical_centers_deg=None,
    azimuth_centers_deg=None,
    azimuth_mode="full_360_front_centered",
):
    H, W = shape
    if vertical_centers_deg is None:
        vertical_centers_deg = make_uniform_vertical_centers(H)
    if azimuth_centers_deg is None:
        azimuth_centers_deg = make_azimuth_centers(W, mode=azimuth_mode)

    vertical_centers_deg = np.asarray(vertical_centers_deg, dtype=np.float64)
    azimuth_centers_deg = np.asarray(azimuth_centers_deg, dtype=np.float64)
    if vertical_centers_deg.shape != (H,):
        raise ValueError(f"vertical_centers_deg shape must be {(H,)}, got {vertical_centers_deg.shape}")
    if azimuth_centers_deg.shape != (W,):
        raise ValueError(f"azimuth_centers_deg shape must be {(W,)}, got {azimuth_centers_deg.shape}")
    return np.deg2rad(vertical_centers_deg), np.deg2rad(azimuth_centers_deg)


def _wrapped_angle_diff(a, b):
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def _grid_offsets(neighbor):
    if neighbor == "angular_grid4":
        return [(0, 1), (1, 0)]
    if neighbor == "angular_grid8":
        return [(0, 1), (1, -1), (1, 0), (1, 1)]
    raise ValueError(f"neighbor must be one of {sorted(NEIGHBOR_MODES)}, got {neighbor!r}")


def _apply_anchor_reject(
    target_mask,
    guide_range,
    anchor_range,
    anchor_reject,
    log_ratio_thr,
    abs_error_thr,
    eps=1e-6,
):
    if anchor_reject not in ANCHOR_REJECT_MODES:
        raise ValueError(
            f"anchor_reject must be one of {sorted(ANCHOR_REJECT_MODES)}, got {anchor_reject!r}"
        )

    keep = np.asarray(target_mask, dtype=bool).copy()
    if not np.any(keep) or anchor_reject == "none":
        return keep, 0

    if anchor_reject == "log_ratio":
        diff = np.abs(
            np.log(np.maximum(anchor_range[keep], eps))
            - np.log(np.maximum(guide_range[keep], eps))
        )
        reject = diff >= float(log_ratio_thr)
    else:
        reject = np.abs(anchor_range[keep] - guide_range[keep]) >= float(abs_error_thr)

    coords = np.where(keep)
    if reject.size:
        keep[coords[0][reject], coords[1][reject]] = False
    return keep, int(reject.sum())


def _mean_abs_and_rmse(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return float(np.mean(np.abs(values))), float(np.sqrt(np.mean(values ** 2)))


def _vector_stats(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_abs_mean": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_abs_mean": float(np.mean(np.abs(values))),
    }


def build_spherical_graph_laplacian(
    guide_range,
    guide_valid,
    node_id,
    node_rows,
    node_cols,
    *,
    vertical_centers_deg=None,
    azimuth_centers_deg=None,
    azimuth_mode="full_360_front_centered",
    neighbor="angular_grid8",
    edge_spatial_mode="angular",
    sigma_angular=0.01,
    sigma_tangent=1.0,
    sigma_log_range=0.3,
    max_log_range_diff=None,
    edge_range_mode="log_gaussian",
):
    if neighbor not in NEIGHBOR_MODES:
        raise ValueError(f"neighbor must be one of {sorted(NEIGHBOR_MODES)}, got {neighbor!r}")
    if edge_spatial_mode not in EDGE_SPATIAL_MODES:
        raise ValueError(
            f"edge_spatial_mode must be one of {sorted(EDGE_SPATIAL_MODES)}, got {edge_spatial_mode!r}"
        )
    if edge_range_mode not in EDGE_RANGE_MODES:
        raise ValueError(
            f"edge_range_mode must be one of {sorted(EDGE_RANGE_MODES)}, "
            f"got {edge_range_mode!r}"
        )
    if sigma_angular <= 0:
        raise ValueError("sigma_angular must be positive")
    if sigma_tangent <= 0:
        raise ValueError("sigma_tangent must be positive")
    if sigma_log_range <= 0:
        raise ValueError("sigma_log_range must be positive")

    guide_range = np.asarray(guide_range, dtype=np.float64)
    H, W = guide_range.shape
    N = int(len(node_rows))
    elevation, azimuth = _angle_centers_for_shape(
        guide_range.shape,
        vertical_centers_deg=vertical_centers_deg,
        azimuth_centers_deg=azimuth_centers_deg,
        azimuth_mode=azimuth_mode,
    )

    rows = []
    cols = []
    data = []
    weights = []
    range_gates = []
    spatial_weights = []
    spatial_distances = []
    log_range_diffs = []
    zero_count = 0
    edge_i = []
    edge_j = []

    for r, c in zip(node_rows, node_cols):
        i = int(node_id[r, c])
        for dr, dc in _grid_offsets(neighbor):
            rr = int(r + dr)
            if rr < 0 or rr >= H:
                continue
            cc = int((c + dc) % W)
            if not guide_valid[rr, cc]:
                continue
            j = int(node_id[rr, cc])
            if j < 0 or j == i:
                continue

            theta_i = azimuth[c]
            theta_j = azimuth[cc]
            phi_i = elevation[r]
            phi_j = elevation[rr]
            d_theta = _wrapped_angle_diff(theta_i, theta_j)
            d_phi = phi_i - phi_j
            phi_bar = 0.5 * (phi_i + phi_j)
            d_ang_sq = (np.cos(phi_bar) * d_theta) ** 2 + d_phi ** 2

            ri = float(guide_range[r, c])
            rj = float(guide_range[rr, cc])
            if edge_spatial_mode == "tangent":
                r_bar = 0.5 * (ri + rj)
                d_spatial_sq = (r_bar ** 2) * d_ang_sq
                spatial_sigma = float(sigma_tangent)
            else:
                d_spatial_sq = d_ang_sq
                spatial_sigma = float(sigma_angular)

            log_diff = abs(float(np.log(ri) - np.log(rj)))
            spatial_weight = float(np.exp(-d_spatial_sq / (2.0 * spatial_sigma ** 2)))
            if edge_range_mode == "log_gaussian":
                range_gate = float(
                    np.exp(
                        -(log_diff ** 2)
                        / (2.0 * float(sigma_log_range) ** 2)
                    )
                )
            else:
                range_gate = 1.0
            weight = spatial_weight * range_gate
            if max_log_range_diff is not None and log_diff > float(max_log_range_diff):
                weight = 0.0

            if not np.isfinite(weight) or weight <= 0.0:
                zero_count += 1
                continue

            rows.extend([i, j, i, j])
            cols.extend([i, j, j, i])
            data.extend([weight, weight, -weight, -weight])
            weights.append(weight)
            range_gates.append(range_gate)
            spatial_weights.append(spatial_weight)
            spatial_distances.append(float(np.sqrt(max(d_spatial_sq, 0.0))))
            log_range_diffs.append(log_diff)
            edge_i.append(i)
            edge_j.append(j)

    L = sparse.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    stats = {
        "neighbor": neighbor,
        "edge_spatial_mode": edge_spatial_mode,
        "edge_range_mode": edge_range_mode,
        "sigma_angular": float(sigma_angular),
        "sigma_tangent": float(sigma_tangent),
        "sigma_log_range": float(sigma_log_range),
        "max_log_range_diff": "" if max_log_range_diff is None else float(max_log_range_diff),
        "N_edges_graph": int(weights.size),
        "edge_weight_mean": float(np.mean(weights)) if weights.size else np.nan,
        "edge_weight_std": float(np.std(weights)) if weights.size else np.nan,
        "edge_weight_min": float(np.min(weights)) if weights.size else np.nan,
        "edge_weight_max": float(np.max(weights)) if weights.size else np.nan,
        "edge_weight_zero_count": int(zero_count),
        "edge_weight_zero_ratio": float(zero_count / max(zero_count + int(weights.size), 1)),
        "range_gate_mean": float(np.mean(range_gates)) if range_gates else np.nan,
        "angular_weight_mean": float(np.mean(spatial_weights)) if spatial_weights else np.nan,
        "spatial_distance_mean": float(np.mean(spatial_distances)) if spatial_distances else np.nan,
        "log_range_diff_mean": float(np.mean(log_range_diffs)) if log_range_diffs else np.nan,
    }
    debug = {
        "edge_i": np.asarray(edge_i, dtype=np.int64),
        "edge_j": np.asarray(edge_j, dtype=np.int64),
        "edge_weight": weights,
        "range_gate": np.asarray(range_gates, dtype=np.float64),
        "spatial_weight": np.asarray(spatial_weights, dtype=np.float64),
        "spatial_distance": np.asarray(spatial_distances, dtype=np.float64),
        "log_range_diff": np.asarray(log_range_diffs, dtype=np.float64),
    }
    return L, stats, debug


def build_graph_residual_system(
    L,
    target_node_indices,
    target_delta,
    *,
    lambda_anchor=300.0,
    lambda_prior=0.05,
    lambda_smooth=1.0,
):
    N = L.shape[0]
    if L.shape != (N, N):
        raise ValueError(f"L must be square, got {L.shape}")
    if lambda_anchor < 0 or lambda_prior < 0 or lambda_smooth < 0:
        raise ValueError("lambda_anchor, lambda_prior, and lambda_smooth must be >= 0")

    I = sparse.eye(N, format="csr", dtype=np.float64)
    A = float(lambda_prior) * I + float(lambda_smooth) * L
    b = np.zeros((N,), dtype=np.float64)

    target_node_indices = np.asarray(target_node_indices, dtype=np.int64)
    target_delta = np.asarray(target_delta, dtype=np.float64)
    if target_node_indices.size:
        if target_delta.shape != target_node_indices.shape:
            raise ValueError("target_delta shape must match target_node_indices")
        anchor_diag = sparse.csr_matrix(
            (
                np.full(target_node_indices.shape, float(lambda_anchor), dtype=np.float64),
                (target_node_indices, target_node_indices),
            ),
            shape=(N, N),
        )
        A = A + anchor_diag
        b[target_node_indices] = float(lambda_anchor) * target_delta

    return A.tocsr(), b


def _solve_linear_system(A, b, method):
    if method == "spsolve":
        return spsolve(A, b), 0
    if method != "cg":
        raise ValueError("method must be cg or spsolve")
    try:
        x, info = cg(A, b, rtol=1e-6, atol=1e-8, maxiter=1000)
    except TypeError:
        x, info = cg(A, b, tol=1e-6, maxiter=1000)
    return x, info


def _solve_graph_delta(
    L,
    target_node_indices,
    target_delta,
    *,
    method,
    lambda_anchor,
    lambda_prior,
    lambda_smooth,
):
    A, b = build_graph_residual_system(
        L,
        target_node_indices=target_node_indices,
        target_delta=target_delta,
        lambda_anchor=lambda_anchor,
        lambda_prior=lambda_prior,
        lambda_smooth=lambda_smooth,
    )
    delta_graph, info = _solve_linear_system(A, b, method=method)
    system_residual = A @ delta_graph - b
    solve_residual = float(np.linalg.norm(system_residual) / max(np.linalg.norm(b), 1e-12))
    return delta_graph, info, solve_residual, A, b


def RangeROIGDC(
    pred_range,
    anchor_range,
    *,
    vertical_centers_deg=None,
    azimuth_centers_deg=None,
    azimuth_mode="full_360_front_centered",
    method="cg",
    range_min=0.1,
    range_max=80.0,
    anchor_reject="abs",
    log_ratio_thr=0.4,
    abs_error_thr=2.0,
    lambda_anchor=300.0,
    lambda_prior=0.1,
    lambda_smooth=1.0,
    neighbor="angular_grid8",
    edge_spatial_mode="angular",
    sigma_angular=0.01,
    sigma_tangent=1.0,
    sigma_log_range=0.3,
    max_log_range_diff=None,
    edge_range_mode="log_gaussian",
    residual_domain="log",
    delta_clip=0.3,
    anchor_force_policy="accepted_only",
    return_debug=False,
    return_stats=False,
    verbose=False,
):
    """Graph-regularized log- or linear-range residual correction."""
    del verbose
    t0 = time.perf_counter()
    guide_range = np.asarray(pred_range, dtype=np.float32)
    anchor_range = np.asarray(anchor_range, dtype=np.float32)
    if guide_range.shape != anchor_range.shape:
        raise ValueError(f"shape mismatch: pred_range={guide_range.shape}, anchor_range={anchor_range.shape}")
    if method not in {"cg", "spsolve"}:
        raise ValueError("method must be cg or spsolve")
    if edge_range_mode not in EDGE_RANGE_MODES:
        raise ValueError(
            f"edge_range_mode must be one of {sorted(EDGE_RANGE_MODES)}, "
            f"got {edge_range_mode!r}"
        )
    if residual_domain not in RESIDUAL_DOMAINS:
        raise ValueError(
            f"residual_domain must be one of {sorted(RESIDUAL_DOMAINS)}, "
            f"got {residual_domain!r}"
        )
    if delta_clip is not None and delta_clip <= 0:
        raise ValueError("delta_clip must be positive when set")
    if anchor_force_policy not in {"accepted_only", "all_valid", "none"}:
        raise ValueError(
            "anchor_force_policy must be accepted_only, all_valid, or none"
        )

    H, W_img = guide_range.shape
    stats = {
        "method_tag": f"graph_{residual_domain}_range_residual",
        "residual_domain": residual_domain,
        "status": "ok",
        "H": H,
        "W": W_img,
        "N_valid_pred": 0,
        "N_nodes": 0,
        "N_anchor_valid": 0,
        "N_anchor_overlap": 0,
        "N_residual_targets": 0,
        "N_rejected_residual_targets": 0,
        "anchor_candidate_count": 0,
        "anchor_overlap_count": 0,
        "anchor_before_reject_count": 0,
        "anchor_after_reject_count": 0,
        "anchor_reject_count": 0,
        "anchor_reject_mode": anchor_reject,
        "anchor_force_policy": anchor_force_policy,
        "anchor_forced_count": 0,
        "anchor_reject_ratio": 0.0,
        "output_valid_count": 0,
        "output_valid_ratio": 0.0,
        "lambda_anchor": float(lambda_anchor),
        "lambda_prior": float(lambda_prior),
        "lambda_smooth": float(lambda_smooth),
        "edge_range_mode": edge_range_mode,
        "delta_clip": "" if delta_clip is None else float(delta_clip),
        "delta_graph_mean": np.nan,
        "delta_graph_std": np.nan,
        "delta_graph_abs_mean": np.nan,
        "propagation_ratio_graph": np.nan,
        "delta_final_mean": np.nan,
        "delta_final_std": np.nan,
        "delta_final_abs_mean": np.nan,
        "anchor_abs_error_before": np.nan,
        "anchor_rmse_before": np.nan,
        "anchor_mae_before_reject": np.nan,
        "anchor_rmse_before_reject": np.nan,
        "anchor_abs_error_after_graph_solve": np.nan,
        "anchor_rmse_after_graph_solve": np.nan,
        "anchor_abs_error_after_graph": np.nan,
        "anchor_rmse_after_graph": np.nan,
        "anchor_abs_error_after_force": np.nan,
        "anchor_rmse_after_force": np.nan,
        "corrected_anchor_mae": np.nan,
        "corrected_anchor_rmse": np.nan,
        "accepted_anchor_mae_after_correction": np.nan,
        "accepted_anchor_rmse_after_correction": np.nan,
        "rejected_anchor_mae_after_correction": np.nan,
        "rejected_anchor_rmse_after_correction": np.nan,
        "all_overlap_anchor_mae_after_correction": np.nan,
        "all_overlap_anchor_rmse_after_correction": np.nan,
        "solver_info": "",
        "solve_residual": np.nan,
        "t_build_nodes": 0.0,
        "t_build_graph": 0.0,
        "t_graph_solve": 0.0,
        "t_total_correction": 0.0,
    }

    t_nodes0 = time.perf_counter()
    guide_valid = valid_range_mask(guide_range, range_min=range_min, range_max=range_max)
    anchor_valid = valid_range_mask(anchor_range, range_min=range_min, range_max=range_max)
    target_before_reject = guide_valid & anchor_valid
    target_mask, rejected_targets = _apply_anchor_reject(
        target_before_reject,
        guide_range=guide_range,
        anchor_range=anchor_range,
        anchor_reject=anchor_reject,
        log_ratio_thr=log_ratio_thr,
        abs_error_thr=abs_error_thr,
    )

    node_rows, node_cols = np.where(guide_valid)
    N = int(len(node_rows))
    node_id = -np.ones((H, W_img), dtype=np.int32)
    node_id[node_rows, node_cols] = np.arange(N, dtype=np.int32)
    stats.update(
        {
            "N_valid_pred": int(guide_valid.sum()),
            "N_nodes": N,
            "N_anchor_valid": int(anchor_valid.sum()),
            "N_anchor_overlap": int(target_before_reject.sum()),
            "N_residual_targets": int(target_mask.sum()),
            "N_rejected_residual_targets": int(rejected_targets),
            "anchor_candidate_count": int(anchor_valid.sum()),
            "anchor_overlap_count": int(target_before_reject.sum()),
            "anchor_before_reject_count": int(target_before_reject.sum()),
            "anchor_after_reject_count": int(target_mask.sum()),
            "anchor_reject_count": int(rejected_targets),
        }
    )
    if stats["N_anchor_overlap"]:
        stats["anchor_reject_ratio"] = (
            stats["N_anchor_overlap"] - stats["N_residual_targets"]
        ) / stats["N_anchor_overlap"]
    stats["t_build_nodes"] = time.perf_counter() - t_nodes0

    corrected_range = guide_range.copy()
    output_mask = np.zeros_like(guide_valid, dtype=bool)
    debug = {}
    if N == 0:
        stats["status"] = "empty_nodes"
        stats["t_total_correction"] = time.perf_counter() - t0
        result = (corrected_range, output_mask, stats)
        if return_debug:
            debug = {
                "target_mask": target_mask,
            }
            result = (*result, debug)
        return result if return_stats else (corrected_range, output_mask)

    if residual_domain == "log":
        raw_base = np.log(np.maximum(guide_range.astype(np.float64), 1e-6))
        anchor_base = np.log(np.maximum(anchor_range.astype(np.float64), 1e-6))
    else:
        raw_base = guide_range.astype(np.float64)
        anchor_base = anchor_range.astype(np.float64)
    raw_base_nodes = raw_base[guide_valid]
    target_node_indices = node_id[target_mask]
    target_delta_map = anchor_base - raw_base
    target_delta = target_delta_map[target_mask]
    if delta_clip is not None:
        target_delta = np.clip(target_delta, -float(delta_clip), float(delta_clip))

    stats.update(_vector_stats(target_delta, "residual_target"))
    before_reject_err = guide_range[target_before_reject] - anchor_range[target_before_reject]
    stats["anchor_mae_before_reject"], stats["anchor_rmse_before_reject"] = (
        _mean_abs_and_rmse(before_reject_err)
    )
    before_err = guide_range[target_mask] - anchor_range[target_mask]
    stats["anchor_abs_error_before"], stats["anchor_rmse_before"] = _mean_abs_and_rmse(before_err)

    t_graph0 = time.perf_counter()
    L, graph_stats, graph_debug = build_spherical_graph_laplacian(
        guide_range, guide_valid, node_id, node_rows, node_cols,
        vertical_centers_deg=vertical_centers_deg,
        azimuth_centers_deg=azimuth_centers_deg, azimuth_mode=azimuth_mode,
        neighbor=neighbor, edge_spatial_mode=edge_spatial_mode,
        sigma_angular=sigma_angular, sigma_tangent=sigma_tangent,
        sigma_log_range=sigma_log_range, max_log_range_diff=max_log_range_diff,
        edge_range_mode=edge_range_mode,
    )
    stats["t_build_graph"] = time.perf_counter() - t_graph0
    stats.update(graph_stats)
    t_solve0 = time.perf_counter()
    delta_graph, info, solve_residual, A, b = _solve_graph_delta(
        L, target_node_indices, target_delta, method=method,
        lambda_anchor=lambda_anchor, lambda_prior=lambda_prior,
        lambda_smooth=lambda_smooth,
    )
    if delta_clip is not None:
        delta_graph = np.clip(delta_graph, -float(delta_clip), float(delta_clip))
    stats["t_graph_solve"] = time.perf_counter() - t_solve0
    stats["solver_info"] = str(info)
    stats["solve_residual"] = solve_residual
    if info != 0:
        stats["status"] = f"solver_info_{info}"
    stats.update(_vector_stats(delta_graph, "delta_graph"))
    stats["propagation_ratio_graph"] = float(
        stats["delta_graph_abs_mean"] / max(stats["residual_target_abs_mean"], 1e-12)
    ) if np.isfinite(stats["delta_graph_abs_mean"]) and np.isfinite(stats["residual_target_abs_mean"]) else np.nan
    if residual_domain == "log":
        graph_node_values = np.exp(raw_base_nodes + delta_graph).astype(np.float32)
    else:
        graph_node_values = (raw_base_nodes + delta_graph).astype(np.float32)
    corrected_graph_solve = guide_range.copy()
    corrected_graph_solve[guide_valid] = np.clip(graph_node_values, range_min, range_max)
    graph_solve_err = corrected_graph_solve[target_mask] - anchor_range[target_mask]
    stats["anchor_abs_error_after_graph_solve"], stats["anchor_rmse_after_graph_solve"] = _mean_abs_and_rmse(graph_solve_err)

    # Preserve the former graph_only accepted-anchor treatment exactly: the
    # accepted target residual is inserted after the solve and clipped again.
    delta_final = np.where(np.isfinite(delta_graph), delta_graph, 0.0)
    if target_node_indices.size:
        delta_final[target_node_indices] = target_delta
    if delta_clip is not None:
        delta_final = np.clip(delta_final, -float(delta_clip), float(delta_clip))
    stats.update(_vector_stats(delta_final, "delta_final"))

    corrected_before_force = guide_range.copy()
    if residual_domain == "log":
        final_node_values = np.exp(raw_base_nodes + delta_final).astype(np.float32)
    else:
        final_node_values = (raw_base_nodes + delta_final).astype(np.float32)
    final_node_values = np.clip(final_node_values, range_min, range_max)
    corrected_before_force[guide_valid] = final_node_values
    graph_err = corrected_before_force[target_mask] - anchor_range[target_mask]
    stats["anchor_abs_error_after_graph"], stats["anchor_rmse_after_graph"] = _mean_abs_and_rmse(graph_err)

    corrected_range = corrected_before_force.copy()
    if anchor_force_policy == "accepted_only":
        force_mask = target_mask
    elif anchor_force_policy == "all_valid":
        force_mask = target_before_reject
    else:
        force_mask = np.zeros_like(target_mask, dtype=bool)
    corrected_range[force_mask] = anchor_range[force_mask]
    stats["anchor_forced_count"] = int(force_mask.sum())
    after_force_err = corrected_range[target_mask] - anchor_range[target_mask]
    stats["anchor_abs_error_after_force"], stats["anchor_rmse_after_force"] = _mean_abs_and_rmse(after_force_err)
    stats["corrected_anchor_mae"] = stats["anchor_abs_error_after_force"]
    stats["corrected_anchor_rmse"] = stats["anchor_rmse_after_force"]
    rejected_mask = target_before_reject & ~target_mask
    rejected_after_err = corrected_range[rejected_mask] - anchor_range[rejected_mask]
    all_overlap_after_err = (
        corrected_range[target_before_reject] - anchor_range[target_before_reject]
    )
    stats["accepted_anchor_mae_after_correction"], stats["accepted_anchor_rmse_after_correction"] = (
        _mean_abs_and_rmse(after_force_err)
    )
    stats["rejected_anchor_mae_after_correction"], stats["rejected_anchor_rmse_after_correction"] = (
        _mean_abs_and_rmse(rejected_after_err)
    )
    stats["all_overlap_anchor_mae_after_correction"], stats["all_overlap_anchor_rmse_after_correction"] = (
        _mean_abs_and_rmse(all_overlap_after_err)
    )

    output_mask = (
        guide_valid
        & np.isfinite(corrected_range)
        & (corrected_range >= range_min)
        & (corrected_range <= range_max)
    )
    stats["output_valid_count"] = int(output_mask.sum())
    stats["output_valid_ratio"] = float(output_mask.sum() / max(int(guide_valid.sum()), 1))
    stats["t_total_correction"] = time.perf_counter() - t0

    if return_debug:
        debug = {
            "node_rows": node_rows,
            "node_cols": node_cols,
            "node_id": node_id,
            "L": L,
            "A": A,
            "b": b,
            "delta_graph": delta_graph,
            "delta_final": delta_final,
            "target_node_indices": target_node_indices,
            "target_delta": target_delta,
            "guide_valid": guide_valid,
            "target_mask": target_mask,
            "force_mask": force_mask,
            **graph_debug,
        }

    if return_stats and return_debug:
        return corrected_range.astype(np.float32), output_mask, stats, debug
    if return_stats:
        return corrected_range.astype(np.float32), output_mask, stats
    return corrected_range.astype(np.float32), output_mask
