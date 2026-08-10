import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve
from scipy.spatial import cKDTree


ANCHOR_REJECT_MODES = {"log_ratio", "abs", "none"}
NEIGHBOR_MODES = {"angular_grid4", "angular_grid8"}
EDGE_SPATIAL_MODES = {"angular", "tangent"}
TRANSFER_NEIGHBOR_MODES = {"rowcol", "angular"}
DIRECT_WEIGHT_MODES = {"nearest", "weighted_knn"}
CONFIDENCE_MODES = {"nearest", "max_weight", "sum_weight"}
SELECTION_MODES = {"soft", "confidence_hard", "log_range_piecewise"}
ABLATION_MODES = {"full", "graph_only", "direct_only"}


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


def _quantile(values, q):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.quantile(values, q))


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
):
    if neighbor not in NEIGHBOR_MODES:
        raise ValueError(f"neighbor must be one of {sorted(NEIGHBOR_MODES)}, got {neighbor!r}")
    if edge_spatial_mode not in EDGE_SPATIAL_MODES:
        raise ValueError(
            f"edge_spatial_mode must be one of {sorted(EDGE_SPATIAL_MODES)}, got {edge_spatial_mode!r}"
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
            range_gate = float(np.exp(-(log_diff ** 2) / (2.0 * float(sigma_log_range) ** 2)))
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


def _angle_features(rows, cols, shape, *, vertical_centers_deg=None, azimuth_centers_deg=None, azimuth_mode="full_360_front_centered"):
    elevation, azimuth = _angle_centers_for_shape(
        shape,
        vertical_centers_deg=vertical_centers_deg,
        azimuth_centers_deg=azimuth_centers_deg,
        azimuth_mode=azimuth_mode,
    )
    elev = elevation[rows]
    az = azimuth[cols]
    return np.stack([np.cos(az), np.sin(az), elev], axis=1)


def _rowcol_augmented_features(rows, cols, width):
    base = np.stack([rows, cols], axis=1).astype(np.float64)
    left = np.stack([rows, cols - width], axis=1).astype(np.float64)
    right = np.stack([rows, cols + width], axis=1).astype(np.float64)
    features = np.concatenate([base, left, right], axis=0)
    source = np.concatenate(
        [
            np.arange(len(rows), dtype=np.int64),
            np.arange(len(rows), dtype=np.int64),
            np.arange(len(rows), dtype=np.int64),
        ]
    )
    return features, source


def _query_transfer_neighbors(
    node_rows,
    node_cols,
    target_rows,
    target_cols,
    shape,
    *,
    transfer_k,
    transfer_neighbor_mode,
    vertical_centers_deg=None,
    azimuth_centers_deg=None,
    azimuth_mode="full_360_front_centered",
):
    N = int(len(node_rows))
    M = int(len(target_rows))
    if M == 0:
        return np.full((N, 0), -1, dtype=np.int64), np.full((N, 0), np.nan, dtype=np.float64)

    k = min(max(int(transfer_k), 1), M)
    if transfer_neighbor_mode == "rowcol":
        target_features, source = _rowcol_augmented_features(target_rows, target_cols, shape[1])
        node_features = np.stack([node_rows, node_cols], axis=1).astype(np.float64)
        query_k = min(target_features.shape[0], max(k * 3, 1))
        tree = cKDTree(target_features)
        dist, idx = tree.query(node_features, k=query_k)
        if query_k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        neigh_idx = np.full((N, k), -1, dtype=np.int64)
        neigh_dist = np.full((N, k), np.nan, dtype=np.float64)
        for i in range(N):
            used = set()
            out = 0
            for d, raw_idx in zip(dist[i], idx[i]):
                src = int(source[int(raw_idx)])
                if src in used:
                    continue
                used.add(src)
                neigh_idx[i, out] = src
                neigh_dist[i, out] = float(d)
                out += 1
                if out >= k:
                    break
        return neigh_idx, neigh_dist

    if transfer_neighbor_mode == "angular":
        target_features = _angle_features(
            target_rows,
            target_cols,
            shape,
            vertical_centers_deg=vertical_centers_deg,
            azimuth_centers_deg=azimuth_centers_deg,
            azimuth_mode=azimuth_mode,
        )
        node_features = _angle_features(
            node_rows,
            node_cols,
            shape,
            vertical_centers_deg=vertical_centers_deg,
            azimuth_centers_deg=azimuth_centers_deg,
            azimuth_mode=azimuth_mode,
        )
        tree = cKDTree(target_features)
        dist, idx = tree.query(node_features, k=k)
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        return idx.astype(np.int64), dist.astype(np.float64)

    raise ValueError(
        f"transfer_neighbor_mode must be one of {sorted(TRANSFER_NEIGHBOR_MODES)}, got {transfer_neighbor_mode!r}"
    )


def build_direct_residual_transfer(
    raw_log_nodes,
    node_rows,
    node_cols,
    target_node_indices,
    target_delta,
    *,
    shape,
    transfer_k=1,
    transfer_neighbor_mode="rowcol",
    direct_weight_mode="nearest",
    confidence_mode="nearest",
    sigma_conf_pixel=2.0,
    sigma_conf_angular=0.01,
    sigma_conf_log_range=0.1,
    confidence_power=1.0,
    confidence_min=0.0,
    confidence_max=1.0,
    vertical_centers_deg=None,
    azimuth_centers_deg=None,
    azimuth_mode="full_360_front_centered",
):
    if transfer_neighbor_mode not in TRANSFER_NEIGHBOR_MODES:
        raise ValueError(f"transfer_neighbor_mode must be one of {sorted(TRANSFER_NEIGHBOR_MODES)}")
    if direct_weight_mode not in DIRECT_WEIGHT_MODES:
        raise ValueError(f"direct_weight_mode must be one of {sorted(DIRECT_WEIGHT_MODES)}")
    if confidence_mode not in CONFIDENCE_MODES:
        raise ValueError(f"confidence_mode must be one of {sorted(CONFIDENCE_MODES)}")
    if transfer_k <= 0:
        raise ValueError("transfer_k must be positive")
    if sigma_conf_pixel <= 0 or sigma_conf_angular <= 0 or sigma_conf_log_range <= 0:
        raise ValueError("confidence sigma values must be positive")
    if confidence_power <= 0:
        raise ValueError("confidence_power must be positive")
    if confidence_min < 0 or confidence_max > 1 or confidence_min > confidence_max:
        raise ValueError("confidence_min/max must satisfy 0 <= min <= max <= 1")

    N = int(len(raw_log_nodes))
    target_node_indices = np.asarray(target_node_indices, dtype=np.int64)
    target_delta = np.asarray(target_delta, dtype=np.float64)
    delta_direct = np.zeros((N,), dtype=np.float64)
    confidence = np.zeros((N,), dtype=np.float64)
    nearest_pixel_dist = np.full((N,), np.nan, dtype=np.float64)
    nearest_log_diff = np.full((N,), np.nan, dtype=np.float64)

    if target_node_indices.size == 0:
        return delta_direct, confidence, nearest_pixel_dist, nearest_log_diff, {
            "nearest_anchor_pixel_dist_mean": np.nan,
            "nearest_anchor_pixel_dist_median": np.nan,
            "nearest_anchor_log_range_diff_mean": np.nan,
            "nearest_anchor_log_range_diff_median": np.nan,
            "t_find_nearest_anchor": 0.0,
            "t_build_direct_transfer": 0.0,
        }, {}

    target_rows = node_rows[target_node_indices]
    target_cols = node_cols[target_node_indices]
    t_query0 = time.perf_counter()
    neigh_idx, neigh_dist = _query_transfer_neighbors(
        node_rows,
        node_cols,
        target_rows,
        target_cols,
        shape,
        transfer_k=transfer_k,
        transfer_neighbor_mode=transfer_neighbor_mode,
        vertical_centers_deg=vertical_centers_deg,
        azimuth_centers_deg=azimuth_centers_deg,
        azimuth_mode=azimuth_mode,
    )
    t_query = time.perf_counter() - t_query0

    t_weight0 = time.perf_counter()
    spatial_sigma = sigma_conf_pixel if transfer_neighbor_mode == "rowcol" else sigma_conf_angular
    weight_max = np.zeros((N,), dtype=np.float64)
    weight_sum = np.zeros((N,), dtype=np.float64)
    for i in range(N):
        valid = neigh_idx[i] >= 0
        if not np.any(valid):
            continue
        idx = neigh_idx[i, valid]
        dist = neigh_dist[i, valid]
        log_diff = np.abs(raw_log_nodes[i] - raw_log_nodes[target_node_indices[idx]])
        weights = np.exp(-(dist ** 2) / (2.0 * spatial_sigma ** 2)) * np.exp(
            -(log_diff ** 2) / (2.0 * sigma_conf_log_range ** 2)
        )
        weights = np.where(np.isfinite(weights), weights, 0.0)
        nearest_pixel_dist[i] = float(dist[0])
        nearest_log_diff[i] = float(log_diff[0])
        if direct_weight_mode == "weighted_knn" and float(np.sum(weights)) > 0.0:
            delta_direct[i] = float(np.sum(weights * target_delta[idx]) / np.sum(weights))
        else:
            delta_direct[i] = float(target_delta[idx[0]])
        weight_max[i] = float(np.max(weights)) if weights.size else 0.0
        weight_sum[i] = float(np.sum(weights))

    nearest_conf = np.exp(-(nearest_pixel_dist ** 2) / (2.0 * spatial_sigma ** 2)) * np.exp(
        -(nearest_log_diff ** 2) / (2.0 * sigma_conf_log_range ** 2)
    )
    nearest_conf = np.where(np.isfinite(nearest_conf), nearest_conf, 0.0)
    if confidence_mode == "max_weight":
        confidence = weight_max
    elif confidence_mode == "sum_weight":
        confidence = np.minimum(weight_sum, 1.0)
    else:
        confidence = nearest_conf
    confidence = np.clip(confidence ** float(confidence_power), confidence_min, confidence_max)

    stats = {
        "nearest_anchor_pixel_dist_mean": float(np.mean(nearest_pixel_dist[np.isfinite(nearest_pixel_dist)]))
        if np.any(np.isfinite(nearest_pixel_dist))
        else np.nan,
        "nearest_anchor_pixel_dist_median": _quantile(nearest_pixel_dist, 0.5),
        "nearest_anchor_log_range_diff_mean": float(np.mean(nearest_log_diff[np.isfinite(nearest_log_diff)]))
        if np.any(np.isfinite(nearest_log_diff))
        else np.nan,
        "nearest_anchor_log_range_diff_median": _quantile(nearest_log_diff, 0.5),
        "t_find_nearest_anchor": float(t_query),
        "t_build_direct_transfer": float(time.perf_counter() - t_weight0),
    }
    debug = {
        "transfer_neighbor_index": neigh_idx,
        "transfer_neighbor_distance": neigh_dist,
        "transfer_weight_max": weight_max,
        "transfer_weight_sum": weight_sum,
    }
    return delta_direct, confidence, nearest_pixel_dist, nearest_log_diff, stats, debug


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


def select_residuals(
    delta_graph,
    delta_direct,
    confidence,
    nearest_log_range_diff,
    *,
    selection_mode="confidence_hard",
    confidence_high_thr=0.8,
    confidence_low_thr=0.2,
    direct_log_range_thr=0.05,
    graph_log_range_thr=0.2,
):
    """Select final residuals from graph and direct proposals."""
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {sorted(SELECTION_MODES)}, got {selection_mode!r}")
    if confidence_low_thr > confidence_high_thr:
        raise ValueError("confidence_low_thr must be <= confidence_high_thr")
    if direct_log_range_thr >= graph_log_range_thr:
        raise ValueError("direct_log_range_thr must be < graph_log_range_thr")

    delta_graph = np.asarray(delta_graph, dtype=np.float64)
    delta_direct = np.asarray(delta_direct, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    nearest_log_range_diff = np.asarray(nearest_log_range_diff, dtype=np.float64)
    if not (
        delta_graph.shape
        == delta_direct.shape
        == confidence.shape
        == nearest_log_range_diff.shape
    ):
        raise ValueError("delta_graph, delta_direct, confidence, and nearest_log_range_diff must have the same shape")

    N = int(delta_graph.size)
    finite_graph = np.isfinite(delta_graph)
    finite_direct = np.isfinite(delta_direct)
    finite_conf = np.isfinite(confidence)
    finite_dlog = np.isfinite(nearest_log_range_diff)
    valid_direct = finite_graph & finite_direct & finite_conf & finite_dlog

    confidence_safe = np.where(finite_conf, confidence, 0.0)
    delta_graph_safe = np.where(finite_graph, delta_graph, 0.0)
    delta_direct_safe = np.where(finite_direct, delta_direct, delta_graph_safe)

    direct_mask = np.zeros((N,), dtype=bool)
    graph_mask = np.zeros((N,), dtype=bool)
    blend_mask = np.zeros((N,), dtype=bool)

    if selection_mode == "soft":
        direct_mask = valid_direct & (confidence_safe >= float(confidence_high_thr))
        graph_mask = (~valid_direct) | (confidence_safe <= float(confidence_low_thr))
        blend_mask = ~(direct_mask | graph_mask)
        delta_final = confidence_safe * delta_direct_safe + (1.0 - confidence_safe) * delta_graph_safe
        delta_final[~valid_direct] = delta_graph_safe[~valid_direct]

    elif selection_mode == "confidence_hard":
        direct_mask = valid_direct & (confidence_safe >= float(confidence_high_thr))
        graph_mask = (~valid_direct) | (confidence_safe <= float(confidence_low_thr))
        blend_mask = ~(direct_mask | graph_mask)
        delta_final = delta_graph_safe.copy()
        delta_final[direct_mask] = delta_direct_safe[direct_mask]
        delta_final[blend_mask] = (
            confidence_safe[blend_mask] * delta_direct_safe[blend_mask]
            + (1.0 - confidence_safe[blend_mask]) * delta_graph_safe[blend_mask]
        )

    else:
        dlog = nearest_log_range_diff
        direct_mask = valid_direct & (dlog < float(direct_log_range_thr))
        graph_mask = (~valid_direct) | (dlog >= float(graph_log_range_thr))
        blend_mask = ~(direct_mask | graph_mask)
        delta_final = delta_graph_safe.copy()
        delta_final[direct_mask] = delta_direct_safe[direct_mask]
        delta_final[blend_mask] = (
            confidence_safe[blend_mask] * delta_direct_safe[blend_mask]
            + (1.0 - confidence_safe[blend_mask]) * delta_graph_safe[blend_mask]
        )

    stats = {
        "selection_mode": selection_mode,
        "confidence_high_thr": float(confidence_high_thr),
        "confidence_low_thr": float(confidence_low_thr),
        "direct_log_range_thr": float(direct_log_range_thr),
        "graph_log_range_thr": float(graph_log_range_thr),
        "selection_direct_count": int(direct_mask.sum()),
        "selection_graph_count": int(graph_mask.sum()),
        "selection_blend_count": int(blend_mask.sum()),
        "selection_direct_ratio": float(direct_mask.sum() / max(N, 1)),
        "selection_graph_ratio": float(graph_mask.sum() / max(N, 1)),
        "selection_blend_ratio": float(blend_mask.sum() / max(N, 1)),
        "selection_invalid_direct_count": int((~finite_direct).sum()),
        "selection_invalid_conf_count": int((~finite_conf).sum()),
        "selection_invalid_dlog_count": int((~finite_dlog).sum()),
    }
    debug = {
        "selection_direct_mask": direct_mask,
        "selection_graph_mask": graph_mask,
        "selection_blend_mask": blend_mask,
    }
    return delta_final, stats, debug


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
    transfer_k=1,
    transfer_neighbor_mode="rowcol",
    direct_weight_mode="nearest",
    confidence_mode="nearest",
    sigma_conf_pixel=2.0,
    sigma_conf_angular=0.01,
    sigma_conf_log_range=0.05,
    confidence_power=2.0,
    confidence_min=0.0,
    confidence_max=1.0,
    selection_mode="confidence_hard",
    ablation_mode="full",
    confidence_high_thr=0.8,
    confidence_low_thr=0.2,
    direct_log_range_thr=0.05,
    graph_log_range_thr=0.2,
    delta_clip=0.3,
    force_anchor_value=None,
    anchor_force_policy="accepted_only",
    return_debug=False,
    return_stats=False,
    verbose=False,
):
    """Confidence-aware residual transfer GDC on guide-valid range bins.

    ``ablation_mode`` controls which residual branch is applied:
    ``full`` uses the original confidence-aware fusion,
    ``graph_only`` uses only the graph proposal, and
    ``direct_only`` uses only valid direct transfers. Nodes without a
    valid direct proposal remain uncorrected in ``direct_only`` mode.
    """
    del verbose
    t0 = time.perf_counter()
    guide_range = np.asarray(pred_range, dtype=np.float32)
    anchor_range = np.asarray(anchor_range, dtype=np.float32)
    if guide_range.shape != anchor_range.shape:
        raise ValueError(f"shape mismatch: pred_range={guide_range.shape}, anchor_range={anchor_range.shape}")
    if method not in {"cg", "spsolve"}:
        raise ValueError("method must be cg or spsolve")
    if delta_clip is not None and delta_clip <= 0:
        raise ValueError("delta_clip must be positive when set")
    if force_anchor_value is not None:
        # Backward-compatible alias used by older configs/tests.
        anchor_force_policy = "accepted_only" if force_anchor_value else "none"
    if anchor_force_policy not in {"accepted_only", "all_valid", "none"}:
        raise ValueError(
            "anchor_force_policy must be accepted_only, all_valid, or none"
        )
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {sorted(SELECTION_MODES)}, got {selection_mode!r}")
    if ablation_mode not in ABLATION_MODES:
        raise ValueError(
            f"ablation_mode must be one of {sorted(ABLATION_MODES)}, got {ablation_mode!r}"
        )
    if confidence_low_thr > confidence_high_thr:
        raise ValueError("confidence_low_thr must be <= confidence_high_thr")
    if direct_log_range_thr >= graph_log_range_thr:
        raise ValueError("direct_log_range_thr must be < graph_log_range_thr")

    H, W_img = guide_range.shape
    stats = {
        "method_tag": "confidence_residual_transfer",
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
        "transfer_k": int(transfer_k),
        "transfer_neighbor_mode": transfer_neighbor_mode,
        "direct_weight_mode": direct_weight_mode,
        "confidence_mode": confidence_mode,
        "sigma_conf_pixel": float(sigma_conf_pixel),
        "sigma_conf_angular": float(sigma_conf_angular),
        "sigma_conf_log_range": float(sigma_conf_log_range),
        "confidence_power": float(confidence_power),
        "confidence_min": float(confidence_min),
        "confidence_max": float(confidence_max),
        "selection_mode": selection_mode,
        "ablation_mode": ablation_mode,
        "confidence_high_thr": float(confidence_high_thr),
        "confidence_low_thr": float(confidence_low_thr),
        "direct_log_range_thr": float(direct_log_range_thr),
        "graph_log_range_thr": float(graph_log_range_thr),
        "selection_direct_count": 0,
        "selection_graph_count": 0,
        "selection_blend_count": 0,
        "selection_direct_ratio": 0.0,
        "selection_graph_ratio": 0.0,
        "selection_blend_ratio": 0.0,
        "selection_invalid_direct_count": 0,
        "selection_invalid_conf_count": 0,
        "selection_invalid_dlog_count": 0,
        "selection_uncorrected_count": 0,
        "selection_uncorrected_ratio": 0.0,
        "delta_clip": "" if delta_clip is None else float(delta_clip),
        "delta_graph_mean": np.nan,
        "delta_graph_std": np.nan,
        "delta_graph_abs_mean": np.nan,
        "propagation_ratio_graph": np.nan,
        "delta_direct_mean": np.nan,
        "delta_direct_std": np.nan,
        "delta_direct_abs_mean": np.nan,
        "nearest_anchor_pixel_dist_mean": np.nan,
        "nearest_anchor_pixel_dist_median": np.nan,
        "nearest_anchor_log_range_diff_mean": np.nan,
        "nearest_anchor_log_range_diff_median": np.nan,
        "confidence_mean": np.nan,
        "confidence_std": np.nan,
        "confidence_min_value": np.nan,
        "confidence_max_value": np.nan,
        "confidence_median": np.nan,
        "confidence_p10": np.nan,
        "confidence_p90": np.nan,
        "confidence_high_ratio": np.nan,
        "confidence_mid_ratio": np.nan,
        "confidence_low_ratio": np.nan,
        "delta_final_mean": np.nan,
        "delta_final_std": np.nan,
        "delta_final_abs_mean": np.nan,
        "delta_final_vs_graph_abs_mean": np.nan,
        "delta_final_vs_direct_abs_mean": np.nan,
        "anchor_abs_error_before": np.nan,
        "anchor_rmse_before": np.nan,
        "anchor_mae_before_reject": np.nan,
        "anchor_rmse_before_reject": np.nan,
        "anchor_abs_error_after_graph_solve": np.nan,
        "anchor_rmse_after_graph_solve": np.nan,
        "anchor_abs_error_after_blend": np.nan,
        "anchor_rmse_after_blend": np.nan,
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
        "t_find_nearest_anchor": 0.0,
        "t_build_direct_transfer": 0.0,
        "t_blend": 0.0,
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
            result = (*result, debug)
        return result if return_stats else (corrected_range, output_mask)

    raw_log = np.log(np.maximum(guide_range.astype(np.float64), 1e-6))
    anchor_log = np.log(np.maximum(anchor_range.astype(np.float64), 1e-6))
    raw_log_nodes = raw_log[guide_valid]
    target_node_indices = node_id[target_mask]
    target_delta_map = anchor_log - raw_log
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

    # Ablations do not construct the branch they are intended to remove.  This
    # keeps their computation and their final residual semantics unambiguous.
    L = A = b = None
    graph_debug = {}
    if ablation_mode == "direct_only":
        delta_graph = np.full((N,), np.nan, dtype=np.float64)
    else:
        t_graph0 = time.perf_counter()
        L, graph_stats, graph_debug = build_spherical_graph_laplacian(
            guide_range, guide_valid, node_id, node_rows, node_cols,
            vertical_centers_deg=vertical_centers_deg,
            azimuth_centers_deg=azimuth_centers_deg, azimuth_mode=azimuth_mode,
            neighbor=neighbor, edge_spatial_mode=edge_spatial_mode,
            sigma_angular=sigma_angular, sigma_tangent=sigma_tangent,
            sigma_log_range=sigma_log_range, max_log_range_diff=max_log_range_diff,
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
        graph_node_values = np.exp(raw_log_nodes + delta_graph).astype(np.float32)
        corrected_graph = guide_range.copy()
        corrected_graph[guide_valid] = np.clip(graph_node_values, range_min, range_max)
        graph_err = corrected_graph[target_mask] - anchor_range[target_mask]
        stats["anchor_abs_error_after_graph_solve"], stats["anchor_rmse_after_graph_solve"] = _mean_abs_and_rmse(graph_err)

    transfer_debug = {}
    if ablation_mode == "graph_only":
        delta_direct = np.full((N,), np.nan, dtype=np.float64)
        confidence = np.full((N,), np.nan, dtype=np.float64)
        nearest_dist = np.full((N,), np.nan, dtype=np.float64)
        nearest_log_diff = np.full((N,), np.nan, dtype=np.float64)
    else:
        delta_direct, confidence, nearest_dist, nearest_log_diff, direct_stats, transfer_debug = build_direct_residual_transfer(
            raw_log_nodes, node_rows, node_cols, target_node_indices, target_delta,
            shape=guide_range.shape, transfer_k=transfer_k,
            transfer_neighbor_mode=transfer_neighbor_mode,
            direct_weight_mode=direct_weight_mode, confidence_mode=confidence_mode,
            sigma_conf_pixel=sigma_conf_pixel, sigma_conf_angular=sigma_conf_angular,
            sigma_conf_log_range=sigma_conf_log_range, confidence_power=confidence_power,
            confidence_min=confidence_min, confidence_max=confidence_max,
            vertical_centers_deg=vertical_centers_deg,
            azimuth_centers_deg=azimuth_centers_deg, azimuth_mode=azimuth_mode,
        )
        if delta_clip is not None:
            delta_direct = np.clip(delta_direct, -float(delta_clip), float(delta_clip))
        stats["t_find_nearest_anchor"] = direct_stats.pop("t_find_nearest_anchor", 0.0)
        stats["t_build_direct_transfer"] = direct_stats.pop("t_build_direct_transfer", 0.0)
        stats.update(_vector_stats(delta_direct, "delta_direct"))
        stats.update(direct_stats)
    conf_finite = confidence[np.isfinite(confidence)]
    if conf_finite.size:
        stats["confidence_mean"] = float(np.mean(conf_finite))
        stats["confidence_std"] = float(np.std(conf_finite))
        stats["confidence_min_value"] = float(np.min(conf_finite))
        stats["confidence_max_value"] = float(np.max(conf_finite))
        stats["confidence_median"] = float(np.median(conf_finite))
        stats["confidence_p10"] = _quantile(conf_finite, 0.1)
        stats["confidence_p90"] = _quantile(conf_finite, 0.9)
        stats["confidence_high_ratio"] = float(np.mean(conf_finite >= 0.8))
        stats["confidence_mid_ratio"] = float(np.mean((conf_finite >= 0.2) & (conf_finite < 0.8)))
        stats["confidence_low_ratio"] = float(np.mean(conf_finite < 0.2))

    t_blend0 = time.perf_counter()
    if ablation_mode == "full":
        delta_final, selection_stats, selection_debug = select_residuals(
            delta_graph=delta_graph,
            delta_direct=delta_direct,
            confidence=confidence,
            nearest_log_range_diff=nearest_log_diff,
            selection_mode=selection_mode,
            confidence_high_thr=confidence_high_thr,
            confidence_low_thr=confidence_low_thr,
            direct_log_range_thr=direct_log_range_thr,
            graph_log_range_thr=graph_log_range_thr,
        )
        selection_stats["selection_uncorrected_count"] = 0
        selection_stats["selection_uncorrected_ratio"] = 0.0
        selection_debug["selection_uncorrected_mask"] = np.zeros((N,), dtype=bool)

    elif ablation_mode == "graph_only":
        finite_graph = np.isfinite(delta_graph)
        delta_final = np.where(finite_graph, delta_graph, 0.0)
        graph_mask = finite_graph.copy()
        direct_mask = np.zeros((N,), dtype=bool)
        blend_mask = np.zeros((N,), dtype=bool)
        uncorrected_mask = ~finite_graph
        selection_stats = {
            "selection_mode": selection_mode,
            "confidence_high_thr": float(confidence_high_thr),
            "confidence_low_thr": float(confidence_low_thr),
            "direct_log_range_thr": float(direct_log_range_thr),
            "graph_log_range_thr": float(graph_log_range_thr),
            "selection_direct_count": 0,
            "selection_graph_count": int(graph_mask.sum()),
            "selection_blend_count": 0,
            "selection_direct_ratio": 0.0,
            "selection_graph_ratio": float(graph_mask.sum() / max(N, 1)),
            "selection_blend_ratio": 0.0,
            "selection_invalid_direct_count": int((~np.isfinite(delta_direct)).sum()),
            "selection_invalid_conf_count": int((~np.isfinite(confidence)).sum()),
            "selection_invalid_dlog_count": int((~np.isfinite(nearest_log_diff)).sum()),
            "selection_uncorrected_count": int(uncorrected_mask.sum()),
            "selection_uncorrected_ratio": float(uncorrected_mask.sum() / max(N, 1)),
        }
        selection_debug = {
            "selection_direct_mask": direct_mask,
            "selection_graph_mask": graph_mask,
            "selection_blend_mask": blend_mask,
            "selection_uncorrected_mask": uncorrected_mask,
        }

    else:  # direct_only
        # A direct proposal is valid only when a nearest accepted anchor
        # was found. Do not fall back to the graph branch in this mode.
        valid_direct = (
            np.isfinite(delta_direct)
            & np.isfinite(nearest_log_diff)
        )
        delta_final = np.zeros_like(delta_graph, dtype=np.float64)
        delta_final[valid_direct] = delta_direct[valid_direct]
        direct_mask = valid_direct.copy()
        graph_mask = np.zeros((N,), dtype=bool)
        blend_mask = np.zeros((N,), dtype=bool)
        uncorrected_mask = ~valid_direct
        selection_stats = {
            "selection_mode": selection_mode,
            "confidence_high_thr": float(confidence_high_thr),
            "confidence_low_thr": float(confidence_low_thr),
            "direct_log_range_thr": float(direct_log_range_thr),
            "graph_log_range_thr": float(graph_log_range_thr),
            "selection_direct_count": int(direct_mask.sum()),
            "selection_graph_count": 0,
            "selection_blend_count": 0,
            "selection_direct_ratio": float(direct_mask.sum() / max(N, 1)),
            "selection_graph_ratio": 0.0,
            "selection_blend_ratio": 0.0,
            "selection_invalid_direct_count": int((~np.isfinite(delta_direct)).sum()),
            "selection_invalid_conf_count": int((~np.isfinite(confidence)).sum()),
            "selection_invalid_dlog_count": int((~np.isfinite(nearest_log_diff)).sum()),
            "selection_uncorrected_count": int(uncorrected_mask.sum()),
            "selection_uncorrected_ratio": float(uncorrected_mask.sum() / max(N, 1)),
        }
        selection_debug = {
            "selection_direct_mask": direct_mask,
            "selection_graph_mask": graph_mask,
            "selection_blend_mask": blend_mask,
            "selection_uncorrected_mask": uncorrected_mask,
        }

    stats.update(selection_stats)
    if target_node_indices.size:
        # Keep the accepted-anchor treatment identical for all variants.
        delta_final[target_node_indices] = target_delta
    if delta_clip is not None:
        delta_final = np.clip(delta_final, -float(delta_clip), float(delta_clip))
    stats["t_blend"] = time.perf_counter() - t_blend0
    stats.update(_vector_stats(delta_final, "delta_final"))
    stats["delta_final_vs_graph_abs_mean"] = float(np.mean(np.abs(delta_final - delta_graph))) if N else np.nan
    stats["delta_final_vs_direct_abs_mean"] = float(np.mean(np.abs(delta_final - delta_direct))) if N else np.nan

    corrected_before_force = guide_range.copy()
    final_node_values = np.exp(raw_log_nodes + delta_final).astype(np.float32)
    final_node_values = np.clip(final_node_values, range_min, range_max)
    corrected_before_force[guide_valid] = final_node_values
    blend_err = corrected_before_force[target_mask] - anchor_range[target_mask]
    stats["anchor_abs_error_after_blend"], stats["anchor_rmse_after_blend"] = _mean_abs_and_rmse(blend_err)

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
            "delta_direct": delta_direct,
            "confidence": confidence,
            "delta_final": delta_final,
            "nearest_anchor_pixel_dist": nearest_dist,
            "nearest_anchor_log_range_diff": nearest_log_diff,
            "target_node_indices": target_node_indices,
            "target_delta": target_delta,
            "guide_valid": guide_valid,
            "target_mask": target_mask,
            **selection_debug,
            **graph_debug,
            **transfer_debug,
        }

    if return_stats and return_debug:
        return corrected_range.astype(np.float32), output_mask, stats, debug
    if return_stats:
        return corrected_range.astype(np.float32), output_mask, stats
    return corrected_range.astype(np.float32), output_mask
