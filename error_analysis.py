"""
error_analysis.py
=================
Post-hoc analysis of predictions from run_loocv.py.

Covers 4 dimensions:
  1. Per-user confusion matrices + most-confused key pairs
  2. Error breakdown by row / finger / hand / user
  3. Error rate vs marker NaN rate (catches q25-style occlusion failures)
  4. Failure mode clustering:
       - STABLE / FIXED / FRAGILE / PERSISTENT (calibration impact)
       - RECOVERABLE / UNRECOVERABLE (is true label in top-3?)
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# LOAD
# ═════════════════════════════════════════════════════════════════════════
def load_predictions(predictions_dir):
    """Concatenate all per-fold prediction CSVs."""
    predictions_dir = Path(predictions_dir)
    csvs = sorted(predictions_dir.glob("predictions_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No prediction CSVs in {predictions_dir}. Run run_loocv.py first."
        )
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    df["correct"] = df["true_letter"] == df["pred_letter"]
    df[["hand", "user"]] = df["fold"].str.split("_", n=1, expand=True)
    return df


# ═════════════════════════════════════════════════════════════════════════
# 1. CONFUSION MATRICES + CONFUSED PAIRS
# ═════════════════════════════════════════════════════════════════════════
def confusion_matrix(df, labels=None):
    if labels is None:
        labels = sorted(set(df["true_letter"]) | set(df["pred_letter"]))
    cm = pd.crosstab(df["true_letter"], df["pred_letter"],
                     rownames=["true"], colnames=["pred"], dropna=False)
    return cm.reindex(index=labels, columns=labels, fill_value=0)


def most_confused_pairs(df, top_k=15):
    """Top K confused key pairs, symmetric (g↔h merges both directions)."""
    errors = df[~df["correct"]].copy()
    if errors.empty:
        return pd.DataFrame(columns=["key_a", "key_b", "count", "asymmetry"])

    errors["pair"] = errors.apply(
        lambda r: tuple(sorted([str(r["true_letter"]), str(r["pred_letter"])])),
        axis=1,
    )
    counts = errors["pair"].value_counts().head(top_k).reset_index()
    counts.columns = ["pair", "count"]
    counts[["key_a", "key_b"]] = pd.DataFrame(counts["pair"].tolist(),
                                                index=counts.index)

    # Asymmetry: how lopsided is the confusion?
    def _asym(row):
        a, b = row["key_a"], row["key_b"]
        ab = ((errors["true_letter"] == a) & (errors["pred_letter"] == b)).sum()
        ba = ((errors["true_letter"] == b) & (errors["pred_letter"] == a)).sum()
        total = ab + ba
        return (ab - ba) / total if total else 0.0

    counts["asymmetry"] = counts.apply(_asym, axis=1)
    return counts[["key_a", "key_b", "count", "asymmetry"]]


# ═════════════════════════════════════════════════════════════════════════
# 2. BREAKDOWN BY ROW / FINGER / HAND / USER
# ═════════════════════════════════════════════════════════════════════════
def breakdown_by(df, by):
    out = df.groupby(by).agg(
        n=("correct", "size"),
        n_correct=("correct", "sum"),
        accuracy=("correct", "mean"),
    ).reset_index()
    out["error_rate"] = 1 - out["accuracy"]
    return out.sort_values("error_rate", ascending=False)


def row_confusion(df):
    """True row → predicted row (row-normalised). Catches row collapse."""
    return pd.crosstab(df["true_row"], df["pred_row"],
                       rownames=["true_row"], colnames=["pred_row"],
                       normalize="index")


# ═════════════════════════════════════════════════════════════════════════
# 3. ERROR VS MARKER QUALITY
# ═════════════════════════════════════════════════════════════════════════
def errors_vs_marker_quality(df, bins=(0.0, 0.05, 0.15, 0.30, 0.50, 1.01)):
    df = df.dropna(subset=["marker_nan_rate"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["nan_bucket"] = pd.cut(df["marker_nan_rate"], bins=list(bins),
                                include_lowest=True, right=False)
    out = df.groupby("nan_bucket", observed=True).agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        mean_confidence=("confidence", "mean"),
    ).reset_index()
    out["error_rate"] = 1 - out["accuracy"]
    return out


# ═════════════════════════════════════════════════════════════════════════
# 4. FAILURE MODE CLUSTERING
# ═════════════════════════════════════════════════════════════════════════
def calibration_impact(df):
    """Classify each keystroke as STABLE / FIXED / FRAGILE / PERSISTENT.

    NOTE: only the calibrated subset has matching keystrokes in baseline
    (baseline runs on all df_test; calibrated runs on the 70% held-out
    after calibration). We can only compare on the calibrated subset.
    """
    # Re-derive keystroke index within (user, hand, prediction_type)
    df = df.copy()
    df["_idx"] = df.groupby(["user", "hand", "prediction_type"]).cumcount()

    base = df[df["prediction_type"] == "baseline"]
    cal = df[df["prediction_type"] == "calibrated"]

    # Align: calibrated samples start at offset = n_calibration into baseline
    # We approximate by aligning the LAST len(cal) rows of baseline per fold
    aligned = []
    for (user, hand), cal_g in cal.groupby(["user", "hand"]):
        base_g = base[(base["user"] == user) & (base["hand"] == hand)]
        n_cal = len(cal_g)
        if n_cal == 0 or len(base_g) < n_cal:
            continue
        base_tail = base_g.tail(n_cal).reset_index(drop=True)
        cal_g = cal_g.reset_index(drop=True)
        merged = pd.DataFrame({
            "user": user, "hand": hand,
            "correct_base": base_tail["correct"].values,
            "correct_cal": cal_g["correct"].values,
            "true_letter": cal_g["true_letter"].values,
        })
        aligned.append(merged)

    if not aligned:
        return pd.DataFrame()
    merged = pd.concat(aligned, ignore_index=True)

    def _cat(row):
        b, c = row["correct_base"], row["correct_cal"]
        if b and c: return "STABLE"
        if b and not c: return "FRAGILE"
        if not b and c: return "FIXED"
        return "PERSISTENT"
    merged["impact"] = merged.apply(_cat, axis=1)
    return merged


def recoverability(df):
    """For wrong predictions, was the true label in top-3?"""
    wrong = df[~df["correct"]].copy()
    if wrong.empty:
        return pd.DataFrame()
    out = wrong.groupby(["user", "hand", "prediction_type"]).agg(
        n_errors=("correct", "size"),
        n_recoverable=("true_in_top3", "sum"),
    ).reset_index()
    out["n_unrecoverable"] = out["n_errors"] - out["n_recoverable"]
    out["recoverable_rate"] = out["n_recoverable"] / out["n_errors"]
    return out


# ═════════════════════════════════════════════════════════════════════════
# REPORT
# ═════════════════════════════════════════════════════════════════════════
def write_markdown_report(df, output_path, title="LOOCV Error Analysis"):
    output_path = Path(output_path)
    cal = df[df["prediction_type"] == "calibrated"].copy()
    base = df[df["prediction_type"] == "baseline"].copy()

    lines = []
    a = lines.append

    a(f"# {title}\n")
    a(f"Baseline rows: {len(base):,} | Calibrated rows: {len(cal):,}\n")

    # Headline
    a("## Headline\n")
    a("| Stage | Accuracy | n |")
    a("|---|---|---|")
    a(f"| Baseline   | {base['correct'].mean():.2%} | {len(base):,} |")
    a(f"| Calibrated | {cal['correct'].mean():.2%}  | {len(cal):,} |")
    delta = cal['correct'].mean() - base['correct'].mean()
    a(f"| **Δ**      | **{delta:+.2%}** |   |\n")

    # Breakdowns
    a("## Breakdown (calibrated)\n")
    for by, label in [("user", "Per-user"), ("hand", "Per-hand"),
                       ("true_row", "Per-row"), ("true_finger", "Per-finger")]:
        a(f"### {label}\n")
        a(breakdown_by(cal, by).to_markdown(index=False, floatfmt=".3f"))
        a("\n")

    # Confused pairs
    a("## Most-confused key pairs (calibrated)\n")
    pairs = most_confused_pairs(cal, top_k=15)
    if not pairs.empty:
        a(pairs.to_markdown(index=False, floatfmt=".3f"))
    a("\n")

    # Row confusion
    a("## Row-level confusion (calibrated)\n")
    a(row_confusion(cal).to_markdown(floatfmt=".3f"))
    a("\n")

    # Marker quality
    a("## Error rate vs marker NaN rate (calibrated)\n")
    mq = errors_vs_marker_quality(cal)
    if not mq.empty:
        mq["nan_bucket"] = mq["nan_bucket"].astype(str)
        a(mq.to_markdown(index=False, floatfmt=".3f"))
    else:
        a("_No marker_nan_rate data._")
    a("\n")

    # Calibration impact
    a("## Calibration impact (STABLE / FIXED / FRAGILE / PERSISTENT)\n")
    impact = calibration_impact(df)
    if not impact.empty:
        summary = impact.groupby(["user", "hand", "impact"]).size().unstack(fill_value=0)
        a(summary.to_markdown())
    a("\n")

    # Recoverability
    a("## Recoverability (was true label in top-3 of wrong predictions?)\n")
    rec = recoverability(df)
    if not rec.empty:
        a(rec.to_markdown(index=False, floatfmt=".3f"))
    a("\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"  📄 Wrote report → {output_path}")
    return output_path


def run_full_analysis(predictions_dir, output_dir):
    """End-to-end: load all prediction CSVs, write CSVs + markdown report."""
    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(predictions_dir)
    cal = df[df["prediction_type"] == "calibrated"]

    # Per-hand confusion matrices
    for hand in cal["hand"].unique():
        cm = confusion_matrix(cal[cal["hand"] == hand])
        cm.to_csv(output_dir / f"confusion_{hand}.csv")

    # Per-user confusion matrices
    for (user, hand), g in cal.groupby(["user", "hand"]):
        confusion_matrix(g).to_csv(output_dir / f"confusion_{user}_{hand}.csv")

    # Confused pairs
    most_confused_pairs(cal, top_k=30).to_csv(
        output_dir / "confused_pairs.csv", index=False)

    # Breakdowns
    for by in ["user", "hand", "true_row", "true_finger"]:
        breakdown_by(cal, by).to_csv(
            output_dir / f"breakdown_by_{by}.csv", index=False)

    # Marker quality
    mq = errors_vs_marker_quality(cal)
    if not mq.empty:
        mq["nan_bucket"] = mq["nan_bucket"].astype(str)
        mq.to_csv(output_dir / "errors_vs_marker_quality.csv", index=False)

    # Calibration impact
    impact = calibration_impact(df)
    if not impact.empty:
        impact.to_csv(output_dir / "calibration_impact.csv", index=False)

    # Recoverability
    rec = recoverability(df)
    if not rec.empty:
        rec.to_csv(output_dir / "recoverability.csv", index=False)

    # Markdown report
    write_markdown_report(df, output_dir / "REPORT.md")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run error analysis on LOOCV predictions")
    parser.add_argument("--predictions-dir", type=str, required=True,
                        help="Path to predictions/ folder from run_loocv.py")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Default: same as predictions-dir but /analysis")
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    out_dir = Path(args.output_dir) if args.output_dir else pred_dir.parent / "analysis"
    run_full_analysis(pred_dir, out_dir)