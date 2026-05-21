"""
run_loocv.py
============
Leave-One-Subject-Out cross-validation runner.

For each user in VALID_PARTICIPANTS:
  1. Train on the OTHER 3 users
  2. Predict on the held-out user (baseline = no per-user calibration)
  3. Use first 30% of held-out user's data to calibrate
  4. Predict on the remaining 70% (calibrated)
  5. Save every prediction (true, pred, top-3, confidence, marker NaN rate)

Then runs error_analysis.py on the resulting predictions.

Usage:
    python run_loocv.py
    python run_loocv.py --output-dir results/my_experiment
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from train_v2 import (
    VALID_PARTICIPANTS,
    load_all_data,
    augment_dropout,
    aggregate_features,
    train_hand,
    predict_for_user,
    calibrate_user,
    compute_posture_baseline,
    participant_id,
    KEY_TO_ROW,
    KEY_TO_FINGER,
    PATH_PATTERN,
)

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════
def git_sha():
    """Capture current git commit for reproducibility. Returns 'unknown' if no git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def marker_nan_rate(raw_segments, keystroke_id):
    """Fraction of NaN cells in marker columns for one keystroke window.
    High NaN rate = marker occlusion = the q25 failure mode."""
    if raw_segments is None or raw_segments.empty:
        return np.nan
    seg = raw_segments[raw_segments["Keystroke_ID"] == keystroke_id]
    if seg.empty:
        return np.nan
    marker_cols = [
        c for c in seg.columns
        if any(m in c for m in ["I4", "M2", "M4", "R4", "L4"])
        and c.endswith(("_x", "_y", "_z"))
    ]
    if not marker_cols:
        return np.nan
    return float(seg[marker_cols].isna().mean().mean())


def build_predictions_df(df_eval, preds, proba, classes,
                         raw_segments, fold_label, prediction_type):
    """One row per keystroke with everything error analysis needs."""
    y_true = df_eval["Pressed_Letter"].values

    # Top-3 candidates
    top3_idx = np.argsort(-proba, axis=1)[:, :3]
    top3_letters = classes[top3_idx]
    top3_conf = np.take_along_axis(proba, top3_idx, axis=1)
    true_in_top3 = np.array([
        y_true[i] in top3_letters[i] for i in range(len(y_true))
    ])

    df_out = pd.DataFrame({
        "true_letter": y_true,
        "pred_letter": preds,
        "confidence": top3_conf[:, 0],
        "top2_letter": top3_letters[:, 1],
        "top3_letter": top3_letters[:, 2],
        "true_in_top3": true_in_top3,
        "true_row": [KEY_TO_ROW.get(str(k).lower(), "?") for k in y_true],
        "pred_row": [KEY_TO_ROW.get(str(k).lower(), "?") for k in preds],
        "true_finger": [KEY_TO_FINGER.get(str(k).lower(), "?") for k in y_true],
        "pred_finger": [KEY_TO_FINGER.get(str(k).lower(), "?") for k in preds],
        "fold": fold_label,
        "prediction_type": prediction_type,
    })

    # Marker quality (needs raw long-format segments)
    if raw_segments is not None and "Keystroke_ID" in df_eval.columns:
        df_out["marker_nan_rate"] = df_eval["Keystroke_ID"].apply(
            lambda kid: marker_nan_rate(raw_segments, kid)
        ).values
    else:
        df_out["marker_nan_rate"] = np.nan

    return df_out


# ═════════════════════════════════════════════════════════════════════════
# ONE FOLD
# ═════════════════════════════════════════════════════════════════════════
def run_one_fold(df_features_all, feature_cols, raw_segments,
                  held_out_user, hand, output_dir,
                  augment=True, posture_norm=True,
                  use_centroids=True, use_logit_bias=True,
                  calibration_fraction=0.30, centroid_alpha=1.0):
    """Train on N-1 users, evaluate on the held-out user."""
    start = time.time()
    print(f"\n{'='*60}\nFOLD: {hand} hand, held out = {held_out_user}\n{'='*60}")

    df = df_features_all.copy()
    df["pid"] = df["Filepath"].apply(participant_id)

    df_train = df[df["pid"] != held_out_user].drop(columns=["pid"]).reset_index(drop=True)
    df_test = df[df["pid"] == held_out_user].drop(columns=["pid"]).reset_index(drop=True)

    if df_test.empty:
        print(f"  ⚠️  No test data for {held_out_user} — skipping")
        return None
    if df_train.empty:
        print(f"  ⚠️  No training data — skipping")
        return None

    # Train
    expert = train_hand(
        df_train, feature_cols,
        hand_name=f"{hand}-LOOCV-{held_out_user}",
        augmented=augment, posture_norm=posture_norm,
    )

    # ── Baseline (posture norm only, no calibration) ────────────────
    posture_baseline_test = (
        compute_posture_baseline(df_test, feature_cols) if posture_norm else None
    )
    preds_base, proba_base, classes = predict_for_user(
        df_test, expert, posture_baseline=posture_baseline_test,
    )
    y_test = df_test["Pressed_Letter"].values
    baseline_acc = float((preds_base == y_test).mean())

# ── Split first: 30% calibration, 70% eval (so baseline & calibrated
    #    are measured on the SAME held-out 70% — fair comparison)
    n_calib = max(20, int(calibration_fraction * len(df_test)))
    n_calib = min(n_calib, len(df_test) - 1)

    df_calib = df_test.iloc[:n_calib].reset_index(drop=True)
    df_eval = df_test.iloc[n_calib:].reset_index(drop=True)
    y_test = df_test["Pressed_Letter"].values
    y_calib = y_test[:n_calib]
    y_eval = y_test[n_calib:]

    # ── Baseline = posture-norm only, evaluated on the SAME df_eval
    posture_baseline_test = (
        compute_posture_baseline(df_test, feature_cols) if posture_norm else None
    )
    preds_base, proba_base, classes = predict_for_user(
        df_eval, expert, posture_baseline=posture_baseline_test,
    )
    baseline_acc = float((preds_base == y_eval).mean())

    # ── Calibrated: optionally apply centroids + logit_bias on top of posture
    calib = calibrate_user(expert, df_calib, y_calib)

    preds_cal, proba_cal, _ = predict_for_user(
        df_eval, expert,
        posture_baseline=calib["posture_baseline"],
        logit_bias=(calib["logit_bias"] if use_logit_bias else None),
        class_centroids=(calib["class_centroids"] if use_centroids else None),
        centroid_idx=(calib["centroid_idx"] if use_centroids else None),
        centroid_alpha=centroid_alpha,
        centroid_counts=(calib["centroid_counts"] if use_centroids else None),
    )
    calibrated_acc = float((preds_cal == y_eval).mean())

    # ── Save predictions ────────────────────────────────────────────
    fold_label = f"{hand}_{held_out_user}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_pred_base = build_predictions_df(
        df_eval, preds_base, proba_base, classes,
        raw_segments, fold_label, "baseline",
    )
    df_pred_cal = build_predictions_df(
        df_eval, preds_cal, proba_cal, classes,
        raw_segments, fold_label, "calibrated",
    )
    df_pred = pd.concat([df_pred_base, df_pred_cal], ignore_index=True)
    pred_path = output_dir / f"predictions_{fold_label}.csv"
    df_pred.to_csv(pred_path, index=False)

    duration = time.time() - start
    print(f"\n  ✅ Baseline:   {baseline_acc:.2%}  ({len(y_eval)} samples, posture-only)")
    print(f"  ✅ Calibrated: {calibrated_acc:.2%} ({len(y_eval)} samples)")
    print(f"  ✅ Δ from calibration: {calibrated_acc - baseline_acc:+.2%}")
    print(f"  ⏱  {duration:.0f}s | saved → {pred_path}")

    return {
        "fold": fold_label,
        "hand": hand,
        "held_out_user": held_out_user,
        "n_train_samples": len(df_train),
        "n_test_samples": len(df_test),
        "n_calibration_samples": n_calib,
        "baseline_accuracy": baseline_acc,
        "calibrated_accuracy": calibrated_acc,
        "duration_seconds": duration,
        "predictions_path": str(pred_path),
    }


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Default: results/<timestamp>")
    parser.add_argument("--path-pattern", type=str, default=PATH_PATTERN)
    parser.add_argument("--users", type=str, default=None,
                        help="Comma-separated user IDs (default: all 4)")
    parser.add_argument("--no-centroids", action="store_true",
                        help="Disable per-user class centroids (Strategy C)")
    parser.add_argument("--no-logit-bias", action="store_true",
                        help="Disable vector-scaling logit bias from calibration")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Disable BOTH centroids and logit bias (posture-norm only)")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-posture-norm", action="store_true")
    parser.add_argument("--no-analysis", action="store_true",
                        help="Skip error analysis at the end")
    parser.add_argument("--centroid-alpha", type=float, default=1.0)
    
    args = parser.parse_args()
    # --no-calibration is a shortcut for --no-centroids --no-logit-bias
    if args.no_calibration:
        args.no_centroids = True
        args.no_logit_bias = True
    # Output dir
    if args.output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / ts
    else:
        output_dir = Path(args.output_dir)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "run.log"),
        ],
    )

    # Reproducibility metadata
    metadata = {
        "git_sha": git_sha(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n🚀 LOOCV starting | output: {output_dir} | git: {metadata['git_sha']}")

    users = args.users.split(",") if args.users else VALID_PARTICIPANTS

    # Load data ONCE (all users)
    print(f"\n📂 Loading data for {users}...")
    right_segs, left_segs = load_all_data(args.path_pattern, users)

    # Optional dropout augmentation
    if not args.no_augment:
        print("\n🎲 Applying dropout augmentation...")
        right_segs_aug = augment_dropout(right_segs, "R", rng=np.random.default_rng(42))
        left_segs_aug = augment_dropout(left_segs, "L", rng=np.random.default_rng(43))
    else:
        right_segs_aug = right_segs
        left_segs_aug = left_segs

    # Aggregate features
    print("\n📊 Aggregating features...")
    right_feat, right_cols = aggregate_features(right_segs_aug)
    left_feat, left_cols = aggregate_features(left_segs_aug)
    print(f"  Right: {len(right_feat)} keystrokes, {len(right_cols)} features")
    print(f"  Left:  {len(left_feat)} keystrokes, {len(left_cols)} features")

    # ── Run all folds ───────────────────────────────────────────────
    results = []
    hand_data = {
        "Right": (right_feat, right_cols, right_segs),
        "Left":  (left_feat, left_cols, left_segs),
    }

    for hand, (feat, cols, raw) in hand_data.items():
        if feat.empty:
            print(f"\n⚠️  No {hand} hand data — skipping")
            continue
        for user in users:
            try:
                result = run_one_fold(
                    feat, cols, raw, user, hand, predictions_dir,
                    augment=not args.no_augment,
                    posture_norm=not args.no_posture_norm,
                    use_centroids=not args.no_centroids,
                    use_logit_bias=not args.no_logit_bias,
                    centroid_alpha=args.centroid_alpha,
                )
                if result:
                    results.append(result)
            except Exception as e:
                log.exception(f"Fold {hand}/{user} FAILED: {e}")

    # ── Summary CSV ─────────────────────────────────────────────────
    if results:
        summary = pd.DataFrame(results)
        summary.to_csv(output_dir / "loocv_summary.csv", index=False)
        print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
        print(summary[["fold", "baseline_accuracy", "calibrated_accuracy",
                       "n_test_samples"]].to_string(index=False))
        print(f"\n  Mean baseline:   {summary['baseline_accuracy'].mean():.2%}")
        print(f"  Mean calibrated: {summary['calibrated_accuracy'].mean():.2%}")

    # ── Error analysis ──────────────────────────────────────────────
    if not args.no_analysis and results:
        print(f"\n📈 Running error analysis...")
        from error_analysis import run_full_analysis
        analysis_dir = output_dir / "analysis"
        run_full_analysis(predictions_dir, analysis_dir)
        print(f"\n✅ DONE")
        print(f"  Predictions: {predictions_dir}")
        print(f"  Analysis:    {analysis_dir}")
        print(f"  📄 Open the report: {analysis_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()