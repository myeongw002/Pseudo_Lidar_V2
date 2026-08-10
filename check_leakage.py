import argparse
from pathlib import Path

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None


def load_npy(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def valid_range(R, M=None, rmin=1.0, rmax=120.0):
    valid = np.isfinite(R) & (R >= rmin) & (R <= rmax)
    if M is not None:
        valid &= M.astype(bool)
    return valid


def summarize_ratio(name, mask):
    print(f"{name}: {int(mask.sum()):,}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--anchor_range", required=True)
    parser.add_argument("--anchor_mask", required=True)

    parser.add_argument("--pred_range", required=True)
    parser.add_argument("--pred_mask", required=True)

    parser.add_argument("--gt_range", required=True)
    parser.add_argument("--gt_mask", required=True)

    parser.add_argument("--stats_csv", default=None)
    parser.add_argument("--allow_leakage", action="store_true")

    parser.add_argument("--rmin", type=float, default=1.0)
    parser.add_argument("--rmax", type=float, default=120.0)

    args = parser.parse_args()
    failures = []

    R_anchor = load_npy(args.anchor_range)
    M_anchor = load_npy(args.anchor_mask).astype(bool)

    R_pred = load_npy(args.pred_range)
    M_pred = load_npy(args.pred_mask).astype(bool)

    R_gt = load_npy(args.gt_range)
    M_gt = load_npy(args.gt_mask).astype(bool)

    print("=== Shape check ===")
    print("anchor range:", R_anchor.shape)
    print("anchor mask :", M_anchor.shape)
    print("pred range  :", R_pred.shape)
    print("pred mask   :", M_pred.shape)
    print("gt range    :", R_gt.shape)
    print("gt mask     :", M_gt.shape)

    if R_pred.shape != R_gt.shape:
        raise ValueError(f"pred and gt shape mismatch: {R_pred.shape} vs {R_gt.shape}")

    if M_anchor.shape != R_gt.shape:
        print()
        print("[WARN] anchor mask shape != R64 shape")
        print("       이 경우 anchor가 32xW 원본 형태일 수 있음.")
        print("       Range GDC에 실제로 들어간 64xW anchor mask를 넣어서 다시 확인하는 게 좋음.")
        raise ValueError("anchor mask shape must match the evaluated range grid")

    H, W = R_gt.shape

    anchor_valid = valid_range(R_anchor, M_anchor, args.rmin, args.rmax)
    pred_valid = valid_range(R_pred, M_pred, args.rmin, args.rmax)
    gt_valid = valid_range(R_gt, M_gt, args.rmin, args.rmax)

    anchor_rows = np.where(anchor_valid.any(axis=1))[0]

    anchor_row_mask = np.zeros((H, W), dtype=bool)
    anchor_row_mask[anchor_rows, :] = True
    hidden_row_mask = ~anchor_row_mask

    print()
    print("=== 1, 2. Anchor row check ===")
    print("valid anchor rows:", anchor_rows.tolist())
    print("num valid anchor rows:", len(anchor_rows))
    print("anchor valid pixels:", int(anchor_valid.sum()))

    if len(anchor_rows) <= H // 2 + 2:
        print("[OK] anchor row 수가 low-res source 수준으로 보임.")
    else:
        print("[DANGER] anchor row 수가 너무 많음. R64/full GT가 anchor로 들어갔을 가능성 있음.")

    if len(anchor_rows) >= H - 2:
        print("[LEAKAGE SUSPECT] 거의 모든 row가 anchor로 valid함.")
        failures.append("anchor_valid_on_nearly_all_rows")

    print()
    print("=== Area counts ===")
    summarize_ratio("GT valid total", gt_valid)
    summarize_ratio("anchor rows & GT valid", anchor_row_mask & gt_valid)
    summarize_ratio("hidden rows & GT valid", hidden_row_mask & gt_valid)
    summarize_ratio("pred valid & GT valid", pred_valid & gt_valid)
    summarize_ratio("pred valid & GT valid & hidden rows", pred_valid & gt_valid & hidden_row_mask)

    print()
    print("=== 3. Hidden row err≈0 check ===")
    eval_hidden = pred_valid & gt_valid & hidden_row_mask

    if eval_hidden.sum() == 0:
        print("[ERROR] hidden row eval pixel이 없음.")
        failures.append("no_hidden_row_evaluation_pixels")
    else:
        err = np.abs(R_pred - R_gt)
        e = err[eval_hidden]

        print("hidden eval pixels:", int(eval_hidden.sum()))
        print("MAE:", float(np.mean(e)))
        print("RMSE:", float(np.sqrt(np.mean(e ** 2))))
        print("median:", float(np.median(e)))
        print("P90:", float(np.percentile(e, 90)))
        print("P95:", float(np.percentile(e, 95)))
        print("P99:", float(np.percentile(e, 99)))

        for thr in [1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]:
            ratio = float(np.mean(e < thr))
            print(f"ratio err < {thr:g}: {ratio:.6f}")

        if np.mean(e < 1e-6) > 0.01:
            print("[DANGER] hidden row에서 GT와 완전히 같은 값이 1% 이상 있음. leakage 강하게 의심.")
            failures.append("hidden_zero_error_ratio_high")
        elif np.mean(e < 1e-2) > 0.3:
            print("[WARN] hidden row에서 1cm 이하 오차 비율이 큼. leakage 또는 매우 강한 보정 확인 필요.")
        elif np.mean(e < 5e-2) > 0.8:
            print("[WARN] hidden row에서 5cm 이하 오차 비율이 매우 큼. anchor/GT leakage 확인 필요.")
        else:
            print("[OK] hidden row err≈0 비율은 과도하지 않음.")

    print()
    print("=== 4. Force anchor hidden-row check ===")

    hidden_anchor_valid = anchor_valid & hidden_row_mask
    print("anchor valid pixels in hidden rows:", int(hidden_anchor_valid.sum()))

    if hidden_anchor_valid.sum() == 0:
        print("[OK] hidden row에는 anchor valid가 없음. force anchor가 hidden row에 적용될 가능성 낮음.")
    else:
        print("[DANGER] hidden row에도 anchor valid가 있음. force_anchor가 hidden row에 적용될 수 있음.")
        failures.append("anchor_valid_in_hidden_rows")

        if eval_hidden.sum() > 0:
            err_anchor = np.abs(R_pred - R_anchor)
            m = hidden_anchor_valid & pred_valid
            if m.sum() > 0:
                print("hidden anchor & pred valid:", int(m.sum()))
                print("ratio pred == anchor in hidden anchor pixels:")
                print("  err < 1e-6:", float(np.mean(err_anchor[m] < 1e-6)))
                print("  err < 1e-3:", float(np.mean(err_anchor[m] < 1e-3)))

    if args.stats_csv is not None:
        print()
        print("=== Stats CSV force-anchor columns check ===")
        stats_path = Path(args.stats_csv)
        if not stats_path.exists():
            print("[WARN] stats_csv not found:", stats_path)
        elif pd is None:
            print("[WARN] pandas is unavailable; skipping optional stats_csv summary.")
        else:
            df = pd.read_csv(stats_path)
            cols = [
                "force_anchor_count_total",
                "force_anchor_count_anchor_rows",
                "force_anchor_count_hidden_rows",
                "N_anchor_valid",
                "N_anchor_overlap",
                "N_residual_targets",
            ]
            existing = [c for c in cols if c in df.columns]

            if not existing:
                print("[WARN] force-anchor 관련 column이 stats_csv에 없음.")
                print("       range_gdc_stats에 force_anchor_count_hidden_rows를 추가하는 게 좋음.")
            else:
                print(df[existing].describe().T)

                if "force_anchor_count_hidden_rows" in df.columns:
                    total_hidden_force = int(df["force_anchor_count_hidden_rows"].sum())
                    print("sum force_anchor_count_hidden_rows:", total_hidden_force)
                    if total_hidden_force == 0:
                        print("[OK] stats 기준 force anchor hidden row 적용 없음.")
                    else:
                        print("[DANGER] stats 기준 force anchor가 hidden row에 적용됨.")

    print()
    print("=== Final judgment hints ===")
    print("정상 조건:")
    print("  - valid anchor rows가 shared sparse source의 실제 occupied row와 일치")
    print("  - hidden row anchor valid pixels = 0")
    print("  - hidden row err < 1e-6 비율 ≈ 0")
    print("  - force_anchor_count_hidden_rows = 0")
    print()
    print("위험 조건:")
    print("  - num valid anchor rows ≈ 64")
    print("  - hidden row에도 anchor valid가 있음")
    print("  - hidden row에서 err≈0 비율이 큼")
    print("  - pred_range 경로가 R64 GT 또는 anchor range를 가리킴")
    if failures and not args.allow_leakage:
        raise SystemExit("Leakage check failed: " + ", ".join(sorted(set(failures))))


if __name__ == "__main__":
    main()
