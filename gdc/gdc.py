'''
Correct predicted depthmaps with sparse LiDAR ground-truths
by Graph-based Depth Correction (GDC)

Author: Yurong You
Date: Feb 2020
'''

try:
    from pykdtree.kdtree import KDTree
except ImportError:  # SciPy is already required by GDC; keep pykdtree optional.
    from scipy.spatial import cKDTree as KDTree
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres, cg
from scipy.sparse import eye as seye
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
import time
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

def filter_mask(pc_rect):
    """Return index of points that lies within the region defined below."""
    valid_inds = (pc_rect[:, 2] < 80) * \
                 (pc_rect[:, 2] > 1) * \
                 (pc_rect[:, 0] < 40) * \
                 (pc_rect[:, 0] >= -40) * \
                 (pc_rect[:, 1] < 2.5) * \
                 (pc_rect[:, 1] >= -1)
    return valid_inds


GRID_SIZE = 0.1
index_field_sample = np.full(
    (35, int(80 / 0.1), int(80 / 0.1)), -1, dtype=np.int32)

def subsample_mask_by_grid(pc_rect, strategy="legacy_random", seed=None):
    N = pc_rect.shape[0]
    if strategy == "legacy_random":
        perm = np.random.permutation(N)
    elif strategy == "seeded_random":
        perm = np.random.default_rng(seed).permutation(N)
    elif strategy == "deterministic":
        perm = np.arange(N, dtype=np.int64)
    else:
        raise ValueError(
            "subsample_strategy must be legacy_random, seeded_random, or deterministic"
        )
    pc_rect = pc_rect[perm]

    range_filter = filter_mask(pc_rect)
    pc_rect = pc_rect[range_filter]

    pc_rect_quantized = np.floor(pc_rect[:, :3] / GRID_SIZE).astype(np.int32)
    pc_rect_quantized[:, 0] = pc_rect_quantized[:, 0] \
        + int(80 / GRID_SIZE / 2)
    pc_rect_quantized[:, 1] = pc_rect_quantized[:, 1] + int(1 / GRID_SIZE)

    index_field = index_field_sample.copy()

    index_field[pc_rect_quantized[:, 1],
                pc_rect_quantized[:, 2], pc_rect_quantized[:, 0]] = np.arange(pc_rect.shape[0])
    mask = np.zeros(perm.shape, dtype=bool)
    mask[perm[range_filter][index_field[index_field >= 0]]] = 1
    return mask


def anchor_accept_mask(pred_depth, anchor_depth, candidate_mask, mode="abs",
                       abs_error_thr=2.0, log_ratio_thr=0.4):
    """Apply the shared strict anchor policy in camera z-depth space.

    A value exactly on the threshold is rejected, preserving Original GDC's
    historical ``abs(pred-anchor) < 2`` boundary behavior.
    """
    if mode not in {"none", "abs", "log_ratio"}:
        raise ValueError("anchor_reject must be one of none, abs, log_ratio")
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    accepted = candidate_mask.copy()
    if mode == "none" or not np.any(candidate_mask):
        return accepted
    if mode == "abs":
        accepted[candidate_mask] &= (
            np.abs(pred_depth[candidate_mask] - anchor_depth[candidate_mask])
            < float(abs_error_thr)
        )
    else:
        eps = 1e-6
        diff = np.abs(
            np.log(np.maximum(pred_depth[candidate_mask], eps))
            - np.log(np.maximum(anchor_depth[candidate_mask], eps))
        )
        accepted[candidate_mask] &= diff < float(log_ratio_thr)
    return accepted


def filter_theta_mask(pc_rect, low, high):
    # though if we have to do this precisely, we should convert
    # point clouds to velodyne space, here we just use those in rect space,
    # since actually the velodyne and the cameras are very close to each other.

    x, y, z = pc_rect[:, 0], pc_rect[:, 1], pc_rect[:, 2]
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arcsin(y / d)
    return (theta >= low) * (theta < high)


def depth2ptc(depth, calib):
    """Convert a depth_map to a pointcloud."""
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth]).reshape((3, -1)).T
    return calib.project_image_to_rect(points)


def image_anchor_candidate_masks(
    pred_depth, anchor_depth, calib, consider_range=(-0.1, 3.0),
    subsample=False, subsample_strategy="legacy_random", subsample_seed=None,
):
    """Return production GDC anchor candidate, prediction, and overlap masks."""
    ptc_pred = depth2ptc(pred_depth, calib)
    consider_pl = (
        filter_mask(ptc_pred)
        * filter_theta_mask(
            ptc_pred,
            low=np.radians(consider_range[0]),
            high=np.radians(consider_range[1]),
        )
    ).reshape(pred_depth.shape)
    subsample_mask = None
    if subsample:
        subsample_mask = subsample_mask_by_grid(
            ptc_pred, strategy=subsample_strategy, seed=subsample_seed
        ).reshape(pred_depth.shape)
        consider_pl &= subsample_mask
    anchor_candidate = (
        (anchor_depth > 0)
        & filter_mask(depth2ptc(anchor_depth, calib)).reshape(anchor_depth.shape)
    )
    anchor_overlap = anchor_candidate & consider_pl
    return anchor_candidate, consider_pl, anchor_overlap, subsample_mask


def GDC(pred_depth, gt_depth, calib,
        k=10,
        W_tol=1e-5,
        recon_tol=1e-4,
        verbose=False,
        method='gmres',
        consider_range=(-0.1, 3.0),
        subsample=False,
        subsample_strategy="legacy_random",
        subsample_seed=None,
        anchor_reject="abs",
        abs_error_thr=2.0,
        log_ratio_thr=0.4,
        anchor_force_policy="accepted_only",
        subsample_output="preserve",
        return_stats=False,
        ):
    """
    Returns the depth map after Graph-based Depth Correction (GDC).

    Parameters:
        pred_depth - predicted depthmap
        gt_depth - lidar depthmap (-1 means no groundtruth)
        calib - calibration object
        k - k used in KNN
        W_tol - tolerance in solving reconstruction weights
        recon_tol - tolerance used in gmres / cg
        debug - if in debug mode (more info will show)
        verbose - if True, more info will show
        method - use cg or gmres to solve the second step
        consider_range - perform LLDC only on points whose pitch angles are
            within this range
        subsample - whether subsampling points by 0.1 m grids
        anchor_force_policy - which sparse anchors are copied to the final
            output: accepted_only, all_valid, or none
        subsample_output - preserve keeps unprocessed prediction pixels;
            sparse reproduces the legacy sparse-only optimized output

    Returns:
        new_depth_map - A refined depthmap with the same size of pred_depth
    """

    if anchor_force_policy not in {"accepted_only", "all_valid", "none"}:
        raise ValueError(
            "anchor_force_policy must be accepted_only, all_valid, or none"
        )
    if subsample_output not in {"preserve", "sparse"}:
        raise ValueError("subsample_output must be preserve or sparse")

    if verbose:
        print("warpping up depth infos...")

    ptc = depth2ptc(pred_depth, calib)
    anchor_candidate, consider_PL, anchor_overlap, subsample_mask = image_anchor_candidate_masks(
        pred_depth,
        gt_depth,
        calib,
        consider_range=consider_range,
        subsample=subsample,
        subsample_strategy=subsample_strategy,
        subsample_seed=subsample_seed,
    )
    gt_mask = anchor_accept_mask(
        pred_depth,
        gt_depth,
        anchor_overlap,
        mode=anchor_reject,
        abs_error_thr=abs_error_thr,
        log_ratio_thr=log_ratio_thr,
    )

    # we only consider points within certain ranges
    pred_mask = np.logical_not(gt_mask) * consider_PL

    x_info = np.concatenate((pred_depth[pred_mask], pred_depth[gt_mask]))
    gt_info = gt_depth[gt_mask]
    N_PL = pred_mask.sum()   # number of pseudo_lidar points
    N_L = gt_mask.sum()      # number of lidar points (groundtruth)
    ptc = np.concatenate(
        (ptc[pred_mask.reshape(-1)], ptc[gt_mask.reshape(-1)]))
    if verbose:
        print("N_PL={} N_L={}".format(N_PL, N_L))
        print("building up KDtree...")

    tree = KDTree(ptc)
    neighbors = tree.query(ptc, k=k+1)[1][:, 1:]

    if verbose:
        print("sovling W...")

    As = np.zeros((N_PL + N_L, k+2, k+2))
    bs = np.zeros((N_PL + N_L, k+2))
    As[:, :k, :k] = np.eye(k) * (1 + W_tol)
    As[:, k+1, :k] = 1
    As[:, :k, k+1] = 1
    bs[:, k+1] = 1
    bs[:, k] = x_info
    As[:, k, :k] = x_info[neighbors]
    As[:, :k, k] = x_info[neighbors]

    W = np.linalg.solve(As, bs[..., None])[:, :k, 0]

    if verbose:
        avg = 0
        for i in range(N_PL):
            avg += np.abs(W[i, :k].dot(x_info[neighbors[i]]) - x_info[i])
        print("average reconstruction diff: {:.3e}".format(avg / N_PL))
        print("building up sparse W...")

    # We devide the sparse W matrix into 4 parts:
    # [W_PLPL, W_LPL]
    # [W_PLL , W_LL ]
    idx_PLPL = neighbors[:N_PL] < N_PL
    indptr_PLPL = np.concatenate(([0], np.cumsum(idx_PLPL.sum(axis=1))))
    W_PLPL = csr_matrix((W[:N_PL][idx_PLPL], neighbors[:N_PL]
                         [idx_PLPL], indptr_PLPL), shape=(N_PL, N_PL))

    idx_LPL = neighbors[:N_PL] >= N_PL
    indptr_LPL = np.concatenate(([0], np.cumsum(idx_LPL.sum(axis=1))))
    W_LPL = csr_matrix((W[:N_PL][idx_LPL], neighbors[:N_PL]
                        [idx_LPL] - N_PL, indptr_LPL), shape=(N_PL, N_L))

    idx_PLL = neighbors[N_PL:] < N_PL
    indptr_PLL = np.concatenate(([0], np.cumsum(idx_PLL.sum(axis=1))))
    W_PLL = csr_matrix((W[N_PL:][idx_PLL], neighbors[N_PL:]
                        [idx_PLL], indptr_PLL), shape=(N_L, N_PL))

    idx_LL = neighbors[N_PL:] >= N_PL
    indptr_LL = np.concatenate(([0], np.cumsum(idx_LL.sum(axis=1))))
    W_LL = csr_matrix((W[N_PL:][idx_LL], neighbors[N_PL:]
                       [idx_LL] - N_PL, indptr_LL), shape=(N_L, N_L))

    if verbose:
        print("reconstructing depth...")

    A = sparse.vstack((seye(N_PL) - W_PLPL, W_PLL))
    b = np.concatenate((W_LPL.dot(gt_info), gt_info - W_LL.dot(gt_info)))

    ATA = LinearOperator((A.shape[1], A.shape[1]),
                         matvec=lambda x: A.T.dot(A.dot(x)))
    method = cg if method == 'cg' else gmres
    try:
        x_new, info = method(ATA, A.T.dot(b), x0=x_info[:N_PL], tol=recon_tol)
    except TypeError:
        x_new, info = method(ATA, A.T.dot(b), x0=x_info[:N_PL], rtol=recon_tol)
    if verbose:
        print(info)
        print('solve in error: {}'.format(np.linalg.norm(A.dot(x_new) - b)))

    if subsample and subsample_output == "sparse":
        new_depth_map = np.full_like(pred_depth, -1)
        new_depth_map[subsample_mask] = pred_depth[subsample_mask]
    else:
        new_depth_map = pred_depth.copy()
    new_depth_map[pred_mask] = x_new
    if anchor_force_policy == "accepted_only":
        force_mask = gt_mask
    elif anchor_force_policy == "all_valid":
        force_mask = gt_depth > 0
    else:
        force_mask = np.zeros_like(gt_mask, dtype=bool)
    new_depth_map[force_mask] = gt_depth[force_mask]

    candidate_count = int(anchor_candidate.sum())
    overlap_count = int(anchor_overlap.sum())
    accepted_count = int(gt_mask.sum())
    stats = {
        "anchor_candidate_count": candidate_count,
        "anchor_overlap_count": overlap_count,
        "anchor_before_reject_count": overlap_count,
        "anchor_after_reject_count": accepted_count,
        "anchor_reject_count": overlap_count - accepted_count,
        "anchor_reject_ratio": (
            (overlap_count - accepted_count) / overlap_count if overlap_count else np.nan
        ),
        "correction_node_count": int(N_PL + N_L),
        "anchor_force_policy": anchor_force_policy,
        "anchor_forced_count": int(force_mask.sum()),
        "subsample_output": subsample_output,
    }
    return (new_depth_map, stats) if return_stats else new_depth_map
