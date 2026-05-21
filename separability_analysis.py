"""
separability_analysis.py
========================
Quantify the "mental keyboard drift" hypothesis with three metrics:

  1. WITHIN-USER SEPARABILITY — for each user, how well-separated are
     their own key clusters in feature space? (silhouette score)
     High = consistent typist; Low = chaotic mapping.

  2. INTER-USER DRIFT — for each key, how far apart do different users
     place that key in feature space? Measured in pooled-σ units, so
     it's directly comparable to the within-user separation.

  3. CROSS-USER COLLISION — for each (user_A, key_X) cluster, which
     (user_B, key_Y) clusters does it overlap with? This is the actual
     MECHANISM of LOOCV failure: when q11's 'a' lives where q14's 'q'
     lives, the model trained on q14 will predict 'q' for q11's 'a'.

The headline number for your paper:
    DRIFT_RATIO = mean(inter_user_drift) / mean(within_user_separation)
    If > 1.0: the same key across users is MORE different than two
              different keys within one user. Air-typing is a per-user
              motor task, not a universal one.

Usage:
    python separability_analysis.py
    python separability_analysis.py --output-dir results/separability
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

from train_v2 import (
    VALID_PARTICIPANTS,
    load_all_data,
    aggregate_features,
    participant_id,
    KEY_TO_FINGER,
    KEY_TO_ROW,
    PATH_PATTERN,
)

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# WHICH FEATURES DEFINE "FEATURE SPACE" FOR THESE METRICS?
# ═════════════════════════════════════════════════════════════════════════
# We use the angle-mean features (MCP/PIP/DIP flexion + abduction) for
# every finger. These are the features the model relies on most heavily
# and are interpretable as "hand posture at keystroke time".
#
# Velocity features are excluded here because they're noisy and we want
# the metrics to measure POSTURE drift, not timing drift.
def get_angle_features(df):
    """Return the columns to use as feature space + cleaned matrix."""
    angle_cols = [c for c in df.columns
                  if c.endswith("_Mean")
                  and any(k in c for k in ["MCP_Flexion", "PIP_Flexion",
                                            "DIP_Flexion", "MCP_Abduction"])
                  and "VEL_" not in c and "ACC_" not in c and "JRK_" not in c]
    if not angle_cols:
        return None, []
    X = df[angle_cols].fillna(0).values.astype(float)
    return X, angle_cols


# ═════════════════════════════════════════════════════════════════════════
# METRIC 1 — WITHIN-USER SEPARABILITY
# ═════════════════════════════════════════════════════════════════════════
def within_user_silhouette(df, hand_label):
    """Silhouette score for one user's key clusters.

    Returns dict: {user, hand, n_samples, n_classes, silhouette}
    silhouette ∈ [-1, 1]:
        > 0.5: well-separated clusters (good typist)
        0.2-0.5: weak structure
        < 0.2: clusters overlap heavily
        < 0.0: clusters are entangled (same-letter scatter > different-letter spread)
    """
    if df.empty:
        return None
    user = df["Filepath"].apply(participant_id).iloc[0]
    X, cols = get_angle_features(df)
    if X is None or len(X) < 20:
        return None
    y = df["Pressed_Letter"].values
    n_classes = len(set(y))
    if n_classes < 2:
        return None

    # Standardize so all angle features contribute equally
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    try:
        score = silhouette_score(X, y, metric="euclidean", sample_size=min(2000, len(X)))
    except Exception as e:
        log.warning(f"silhouette failed for {user}/{hand_label}: {e}")
        return None

    return {
        "user": user, "hand": hand_label,
        "n_samples": len(X), "n_classes": n_classes,
        "silhouette": float(score),
    }


# ═════════════════════════════════════════════════════════════════════════
# METRIC 2 — INTER-USER DRIFT (per key)
# ═════════════════════════════════════════════════════════════════════════
def inter_user_drift(per_user_df, hand_label):
    """For each key, measure how far apart users' cluster centers are.

    Returns DataFrame with columns:
      key, hand, n_users_with_key,
      inter_user_centroid_dist, within_user_letter_dist,
      drift_ratio, n_total_samples, row, finger

    drift_ratio = inter_user_centroid_dist / within_user_letter_dist
      > 1.0 → users disagree on this key more than two different letters
              disagree within one user
    """
    out_rows = []

    # Build unified standardized feature matrix across users
    all_dfs = []
    for user, df in per_user_df.items():
        if df.empty:
            continue
        sub = df.copy()
        sub["__user"] = user
        all_dfs.append(sub)
    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    X_all, cols = get_angle_features(combined)
    if X_all is None:
        return pd.DataFrame()

    mu, sd = X_all.mean(axis=0), X_all.std(axis=0) + 1e-9
    X_all = (X_all - mu) / sd
    combined["__row"] = np.arange(len(combined))

    # ── Precompute the DENOMINATOR once per user ──────────────────────
    # within_user_typical_dist[user] = mean pairwise distance between
    # that user's OWN letter centroids (= "how far apart are two letters
    # for this user, typically").
    within_user_typical_dist = {}
    for u in combined["__user"].unique():
        u_rows = combined[combined["__user"] == u]
        u_key_centroids = []
        for k_other, g in u_rows.groupby("Pressed_Letter"):
            if len(g) < 3:
                continue
            idx = g["__row"].values
            u_key_centroids.append(X_all[idx].mean(axis=0))
        if len(u_key_centroids) < 2:
            continue
        u_cs = np.array(u_key_centroids)
        u_d = cdist(u_cs, u_cs, metric="euclidean")
        n_u = len(u_cs)
        within_user_typical_dist[u] = float(
            u_d[np.triu_indices(n_u, k=1)].mean()
        )

    if not within_user_typical_dist:
        return pd.DataFrame()

    # ── Now compute drift PER KEY ─────────────────────────────────────
    keys = sorted(set(combined["Pressed_Letter"]))
    for key in keys:
        key_rows = combined[combined["Pressed_Letter"] == key]
        users_with_key = sorted(key_rows["__user"].unique())
        if len(users_with_key) < 2:
            continue

        # Per-user centroids for THIS key
        centroids = {}
        n_samples = 0
        for u in users_with_key:
            idx = key_rows[key_rows["__user"] == u]["__row"].values
            if len(idx) < 3:
                continue
            Xu = X_all[idx]
            centroids[u] = Xu.mean(axis=0)
            n_samples += len(idx)

        if len(centroids) < 2:
            continue

        # Numerator: mean pairwise distance between users' centroids
        cs = np.array(list(centroids.values()))
        d = cdist(cs, cs, metric="euclidean")
        n = len(cs)
        inter_user_dist = d[np.triu_indices(n, k=1)].mean()

        # Denominator: average of those users' typical letter-letter dists
        denoms = [within_user_typical_dist[u] for u in centroids
                  if u in within_user_typical_dist]
        if not denoms:
            continue
        within_user_letter_dist = float(np.mean(denoms))

        out_rows.append({
            "key": key,
            "hand": hand_label,
            "n_users_with_key": len(centroids),
            "inter_user_centroid_dist": float(inter_user_dist),
            "within_user_letter_dist": within_user_letter_dist,
            "drift_ratio": float(inter_user_dist /
                                  max(within_user_letter_dist, 1e-6)),
            "n_total_samples": int(n_samples),
            "row": KEY_TO_ROW.get(key, "?"),
            "finger": KEY_TO_FINGER.get(key, "?"),
        })

    return pd.DataFrame(out_rows).sort_values(
        "drift_ratio", ascending=False
    ).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════
# METRIC 3 — CROSS-USER COLLISION
# ═════════════════════════════════════════════════════════════════════════
def cross_user_collision(per_user_df, hand_label, top_k=20):
    """For each (user_A, key_X) cluster centroid, find the CLOSEST
    (user_B, key_Y) cluster centroid where user_A ≠ user_B.

    If key_X != key_Y, that's a "collision": q11's 'a' is closer to q14's
    'q' than to anyone's 'a'. THIS is the literal mechanism of LOOCV
    failure for that user.

    Returns DataFrame ranked by smallest collision distance:
      user_A, key_A, user_B, key_B, distance_sigma,
      A_finger, B_finger, A_row, B_row, same_key, same_finger, same_row
    """
    # Build unified standardized feature space
    all_dfs = []
    for user, df in per_user_df.items():
        if df.empty:
            continue
        sub = df.copy()
        sub["__user"] = user
        all_dfs.append(sub)
    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    X_all, cols = get_angle_features(combined)
    if X_all is None:
        return pd.DataFrame()

    mu, sd = X_all.mean(axis=0), X_all.std(axis=0) + 1e-9
    X_all = (X_all - mu) / sd
    combined["__row"] = np.arange(len(combined))

    # Build centroid per (user, key)
    centroid_rows = []
    for (user, key), g in combined.groupby(["__user", "Pressed_Letter"]):
        if len(g) < 5:  # need enough samples for a stable centroid
            continue
        idx = g["__row"].values
        c = X_all[idx].mean(axis=0)
        centroid_rows.append({"user": user, "key": key, "centroid": c, "n": len(idx)})

    if not centroid_rows:
        return pd.DataFrame()

    centroids = np.array([r["centroid"] for r in centroid_rows])
    info = [(r["user"], r["key"], r["n"]) for r in centroid_rows]

    # Pairwise distance matrix among centroids
    D = cdist(centroids, centroids, metric="euclidean")

    # For each centroid, find nearest OTHER-user centroid
    out = []
    for i, (uA, kA, nA) in enumerate(info):
        best_j, best_d = None, np.inf
        for j, (uB, kB, nB) in enumerate(info):
            if uA == uB:
                continue
            if D[i, j] < best_d:
                best_d = D[i, j]
                best_j = j
        if best_j is None:
            continue
        uB, kB, nB = info[best_j]
        out.append({
            "user_A": uA, "key_A": kA, "n_A": nA,
            "user_B": uB, "key_B": kB, "n_B": nB,
            "distance_sigma": float(best_d),
            "same_key": kA == kB,
            "A_finger": KEY_TO_FINGER.get(kA, "?"),
            "B_finger": KEY_TO_FINGER.get(kB, "?"),
            "A_row":    KEY_TO_ROW.get(kA, "?"),
            "B_row":    KEY_TO_ROW.get(kB, "?"),
            "hand":     hand_label,
        })

    df_out = pd.DataFrame(out)
    df_out["same_finger"] = df_out["A_finger"] == df_out["B_finger"]
    df_out["same_row"]    = df_out["A_row"]    == df_out["B_row"]

    # Sort by collision risk: smallest distance to a DIFFERENT key wins
    collisions = df_out[~df_out["same_key"]].sort_values("distance_sigma")
    return collisions.reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════
# HEADLINE NUMBER + REPORT
# ═════════════════════════════════════════════════════════════════════════
def compute_headline(within_df, drift_df):
    """The single number for the paper:
       DRIFT_RATIO = inter_user_drift_σ / within_user_separation_σ
       > 1.0 → users disagree on a key more than they disagree on letters.
    """
    if within_df.empty or drift_df.empty:
        return None

    # Within-user separation expressed in σ (silhouette is a ratio metric
    # but in different units, so we use the "intra/inter cluster distance"
    # interpretation: silhouette ~ (b-a)/max(a,b), so high silhouette →
    # b >> a. We use mean inter-cluster distance from within-user data
    # by approximating with (1 - silhouette) inverse — but it's cleaner to
    # just report the two numbers side by side.
    mean_within_silhouette = within_df["silhouette"].mean()
    mean_drift_sigma = drift_df["drift_ratio"].mean()

    return {
        "mean_within_user_silhouette":  float(mean_within_silhouette),
        "mean_inter_user_drift_sigma":  float(mean_drift_sigma),
        "n_keys_analyzed":              int(len(drift_df)),
        "n_user_hand_combos":           int(len(within_df)),
    }


def write_report(within_df, drift_df, collisions_df, headline, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    within_df.to_csv(out_dir / "within_user_silhouette.csv", index=False)
    drift_df.to_csv(out_dir / "inter_user_drift.csv", index=False)
    collisions_df.to_csv(out_dir / "cross_user_collisions.csv", index=False)

    lines = []
    a = lines.append

    a("# Mental Keyboard Drift Analysis\n")
    a("Three quantitative metrics that test whether each user has their")
    a("own private mental keyboard layout.\n")

    # Headline
    a("## Headline\n")
    if headline:
        a(f"- **Mean within-user silhouette**: `{headline['mean_within_user_silhouette']:+.3f}`")
        a(f"  (>0.5 = tight clusters, <0.2 = overlapping, <0 = chaotic)")
        a(f"- **Mean inter-user drift**: `{headline['mean_inter_user_drift_sigma']:.2f}σ`")
        a(f"  (>2σ = users place the same key in essentially different regions)")
        a(f"- Computed across `{headline['n_keys_analyzed']}` keys, "
          f"`{headline['n_user_hand_combos']}` user-hand combos\n")

    # Metric 1
    a("## Metric 1 — Within-User Silhouette (per user × hand)\n")
    a("How separable is one user's own keyboard?\n")
    a(within_df.sort_values(["hand", "silhouette"], ascending=[True, False])
              .to_markdown(index=False, floatfmt=".3f"))
    a("")

    # Metric 2 (worst-drift keys)
    a("## Metric 2 — Top-15 keys with highest inter-user drift\n")
    a("These are the keys where users disagree most on hand posture.\n")
    a(drift_df.head(15).to_markdown(index=False, floatfmt=".2f"))
    a("")

    a("### Drift by row\n")
    by_row = drift_df.groupby(["hand", "row"]).agg(
        mean_drift_ratio=("drift_ratio", "mean"),
        n_keys=("key", "count"),
    ).reset_index()
    a(by_row.to_markdown(index=False, floatfmt=".2f"))
    a("")

    a("### Drift by finger\n")
    by_finger = drift_df.groupby(["hand", "finger"]).agg(
        mean_drift_ratio=("drift_ratio", "mean"),
        n_keys=("key", "count"),
    ).reset_index()
    a(by_finger.to_markdown(index=False, floatfmt=".2f"))
    a("")

    # Metric 3
    a("## Metric 3 — Top-20 cross-user collisions\n")
    a("Each row shows a (user, key) whose centroid is CLOSER to a")
    a("DIFFERENT user's DIFFERENT key than to anyone else's version of")
    a("the same letter. These are the literal mechanism of LOOCV failure.\n")
    a(collisions_df.head(20).to_markdown(index=False, floatfmt=".2f"))
    a("")

    a("### Collision summary\n")
    total = len(collisions_df) + (collisions_df["same_key"] == False).sum() * 0  # placeholder
    # Count how often the nearest neighbor is on same finger / same row / different
    same_finger_pct = collisions_df["same_finger"].mean() * 100 if not collisions_df.empty else 0
    same_row_pct    = collisions_df["same_row"].mean() * 100 if not collisions_df.empty else 0
    a(f"- **{same_finger_pct:.1f}%** of collisions are *same-finger* "
      f"(another user's neighbor key on the same finger)")
    a(f"- **{same_row_pct:.1f}%** of collisions are *same-row*")
    a(f"- This tells you whether drift moves keys vertically (row errors) "
      f"or horizontally (finger errors).")
    a("")

    (out_dir / "REPORT.md").write_text("\n".join(lines))
    print(f"\n✅ Report written → {out_dir / 'REPORT.md'}")
    print(f"   CSVs:")
    print(f"     within_user_silhouette.csv")
    print(f"     inter_user_drift.csv")
    print(f"     cross_user_collisions.csv")


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-pattern", default=PATH_PATTERN)
    parser.add_argument("--users", default=None,
                        help="Comma-separated (default: 4 training users)")
    parser.add_argument("--output-dir", default="results/separability")
    args = parser.parse_args()

    users = args.users.split(",") if args.users else VALID_PARTICIPANTS

    # Load + aggregate (same path as keyboard_diagnostic.py)
    print(f"📂 Loading data for {users}...")
    right_segs, left_segs = load_all_data(args.path_pattern, users)
    print("📊 Aggregating features...")
    right_feat, _ = aggregate_features(right_segs)
    left_feat, _  = aggregate_features(left_segs)

    right_feat["pid"] = right_feat["Filepath"].apply(participant_id)
    left_feat["pid"]  = left_feat["Filepath"].apply(participant_id)

    # ── Metric 1: within-user silhouette ────────────────────────────────
    print("\n🔍 Computing within-user silhouettes...")
    within_rows = []
    for u in users:
        for hand, feat in [("Right", right_feat), ("Left", left_feat)]:
            sub = feat[feat["pid"] == u]
            r = within_user_silhouette(sub, hand)
            if r:
                within_rows.append(r)
                print(f"  {u} {hand}: silhouette = {r['silhouette']:+.3f} "
                      f"(n={r['n_samples']}, classes={r['n_classes']})")
    within_df = pd.DataFrame(within_rows)

    # ── Metric 2: inter-user drift per key ──────────────────────────────
    print("\n🔍 Computing inter-user drift per key...")
    drift_dfs = []
    for hand, feat in [("Right", right_feat), ("Left", left_feat)]:
        per_user = {u: feat[feat["pid"] == u].reset_index(drop=True)
                    for u in users}
        d = inter_user_drift(per_user, hand)
        drift_dfs.append(d)
        if not d.empty:
            print(f"  {hand}: mean drift_ratio = {d['drift_ratio'].mean():.2f} "
                  f"across {len(d)} keys "
                  f"(worst: {d.iloc[0]['key']} @ {d.iloc[0]['drift_ratio']:.2f})")
    drift_df = pd.concat(drift_dfs, ignore_index=True)

    # ── Metric 3: cross-user collisions ─────────────────────────────────
    print("\n🔍 Finding cross-user collisions...")
    collision_dfs = []
    for hand, feat in [("Right", right_feat), ("Left", left_feat)]:
        per_user = {u: feat[feat["pid"] == u].reset_index(drop=True)
                    for u in users}
        c = cross_user_collision(per_user, hand)
        collision_dfs.append(c)
        if not c.empty:
            print(f"  {hand}: {len(c)} centroids closer to another user's "
                  f"different key than to anyone's same key")
            top = c.iloc[0]
            print(f"     worst collision: ({top['user_A']},{top['key_A']}) ↔ "
                  f"({top['user_B']},{top['key_B']}) @ {top['distance_sigma']:.2f}σ")
    collisions_df = pd.concat(collision_dfs, ignore_index=True)

    # ── Headline ────────────────────────────────────────────────────────
    headline = compute_headline(within_df, drift_df)
    if headline:
        print("\n" + "=" * 60)
        print("HEADLINE")
        print("=" * 60)
        print(f"  Mean within-user silhouette:  "
              f"{headline['mean_within_user_silhouette']:+.3f}")
        print(f"  Mean inter-user drift:        "
              f"{headline['mean_inter_user_drift_sigma']:.2f}σ")
        print("=" * 60)

    # ── Persist ─────────────────────────────────────────────────────────
    write_report(within_df, drift_df, collisions_df, headline, args.output_dir)


if __name__ == "__main__":
    main()