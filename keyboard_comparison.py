"""
keyboard_comparison.py
======================
Two visualizations that test the "one model per user" hypothesis:

  FIGURE 1 — RECONSTRUCTED KEYBOARDS PER USER + OVERLAP
    For each user, reconstruct their "imagined keyboard" by plotting
    every key's centroid in a shared 2D angle space, sized by sample
    count. Then a 5th panel overlays all 4 users' centroids to expose
    cross-user drift visually.

  FIGURE 2 — PER-KEY BEST-FEATURE DISCOVERY
    For each letter, search over ALL pairs of available angle features
    and find the pair that gives the highest separability score
    (Fisher discriminant ratio against neighbor keys). Then visualize:
      (a) a heatmap of "which feature wins for which key"
      (b) the discriminant score for each (key, feature_pair) combo
    This answers: "do different letters need different features?"

Usage:
    python keyboard_comparison.py --hand Left
    python keyboard_comparison.py --hand Right --top-k-pairs 8
"""
import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from train_v2 import (
    VALID_PARTICIPANTS,
    load_all_data,
    aggregate_features,
    participant_id,
    KEY_TO_FINGER,
    PATH_PATTERN,
)


# ═════════════════════════════════════════════════════════════════════════
# TOUCH-TYPING FINGER ASSIGNMENTS (same as keyboard_diagnostic.py)
# ═════════════════════════════════════════════════════════════════════════
FINGER_KEYS = {
    "Left":  {"Little": ["q", "a", "z"],
              "Ring":   ["w", "s", "x"],
              "Middle": ["e", "d", "c"],
              "Index":  ["r", "t", "f", "g", "v", "b"]},
    "Right": {"Index":  ["y", "u", "h", "j", "n", "m"],
              "Middle": ["i", "k"],
              "Ring":   ["o", "l"],
              "Little": ["p"]},
}

# Per-finger angle features (each finger has 4: MCP flexion/abduction, PIP, DIP)
ANGLE_TYPES = ["MCP_Flexion", "MCP_Abduction", "PIP_Flexion", "DIP_Flexion"]
FINGER_NAMES = ["Index", "Middle", "Ring", "Little"]


def all_angle_features():
    """List every angle feature: finger × angle_type × _Mean."""
    return [f"{f}_{a}_Mean" for f in FINGER_NAMES for a in ANGLE_TYPES]


# ═════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════
def load_per_user(path_pattern, users):
    print(f"📂 Loading data for {users}...")
    right_segs, left_segs = load_all_data(path_pattern, users)
    print("📊 Aggregating features...")
    right_feat, _ = aggregate_features(right_segs)
    left_feat, _  = aggregate_features(left_segs)
    right_feat["pid"] = right_feat["Filepath"].apply(participant_id)
    left_feat["pid"]  = left_feat["Filepath"].apply(participant_id)
    return {u: (right_feat[right_feat["pid"] == u].reset_index(drop=True),
                left_feat[left_feat["pid"] == u].reset_index(drop=True))
            for u in users}


def get_hand_df(per_user, hand):
    """Combine all users' data for one hand into a single dataframe with 'user' column."""
    parts = []
    for u, (rdf, ldf) in per_user.items():
        df = rdf if hand == "Right" else ldf
        if df.empty:
            continue
        sub = df.copy()
        sub["user"] = u
        parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 1 — RECONSTRUCTED PER-USER KEYBOARDS + OVERLAP
# ═════════════════════════════════════════════════════════════════════════
# Strategy: for each user, project all of their keystrokes into the SAME
# 2D feature space (chosen to be informative across users), then mark each
# key's centroid. The "imagined keyboard" is the spatial arrangement of
# centroids — high MCP flexion = top row, low = bottom; abduction = column.
#
# We use principal-component projection on the GLOBAL feature matrix so
# every user is plotted in the same coordinate system → directly comparable.

def fit_global_pca(hand_df, hand):
    """Fit a 2D PCA on standardized angle features across ALL users.
    Returns (pca_components_2xD, feature_means, feature_stds, feature_cols)."""
    feature_cols = [c for c in all_angle_features() if c in hand_df.columns]
    X = hand_df[feature_cols].fillna(0).values.astype(float)
    # Standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Xs = (X - mu) / sd
    # PCA via SVD
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    # First 2 components
    components = Vt[:2]                # shape (2, D)
    explained = (S[:2] ** 2) / (S ** 2).sum()
    return components, mu, sd, feature_cols, explained


def project_user(df, components, mu, sd, feature_cols):
    """Project one user's keystrokes onto the shared 2D space."""
    X = df[feature_cols].fillna(0).values.astype(float)
    Xs = (X - mu) / sd
    return Xs @ components.T            # shape (n, 2)


def plot_user_keyboard(df, projected, ax, user, hand, color_map):
    """Plot one user's reconstructed keyboard: a centroid per key in 2D."""
    if df.empty:
        ax.text(0.5, 0.5, f"{user}: no data", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = df.copy()
    df["__pc1"] = projected[:, 0]
    df["__pc2"] = projected[:, 1]

    valid_keys = [k for finger in FINGER_KEYS[hand].values() for k in finger]
    for k in valid_keys:
        sub = df[df["Pressed_Letter"] == k]
        if len(sub) < 3:
            continue
        cx, cy = sub["__pc1"].mean(), sub["__pc2"].mean()
        sx, sy = sub["__pc1"].std(), sub["__pc2"].std()
        color = color_map[k]
        # Cloud (lightly)
        ax.scatter(sub["__pc1"], sub["__pc2"], color=color,
                   s=4, alpha=0.20, edgecolor="none")
        # Centroid ellipse (1σ)
        ax.add_patch(mpatches.Ellipse(
            (cx, cy), 2*sx, 2*sy, fill=True,
            facecolor=color, edgecolor=color, alpha=0.30, linewidth=0.5,
        ))
        # Key label
        ax.text(cx, cy, k.upper(), ha="center", va="center",
                fontsize=10, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor=color, alpha=0.85,
                          edgecolor="black", linewidth=0.6))

    ax.set_title(f"{user}'s reconstructed keyboard — {hand}", fontsize=10)
    ax.set_xlabel("PC1 (mostly row depth)", fontsize=8)
    ax.set_ylabel("PC2 (mostly column position)", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_overlap_keyboard(per_user_projected, ax, hand):
    """Overlay all users' key centroids in one plot, colored by user.
    Connects same-key centroids across users with thin lines to show drift."""
    cmap = plt.colormaps.get_cmap("tab10")
    user_colors = {u: cmap(i % 10) for i, u in enumerate(per_user_projected)}

    valid_keys = [k for finger in FINGER_KEYS[hand].values() for k in finger]

    # First pass: collect each user's centroid per key
    centroids = {k: {} for k in valid_keys}   # {key: {user: (x, y)}}
    for user, (df, proj) in per_user_projected.items():
        if df.empty:
            continue
        df = df.copy()
        df["__pc1"] = proj[:, 0]
        df["__pc2"] = proj[:, 1]
        for k in valid_keys:
            sub = df[df["Pressed_Letter"] == k]
            if len(sub) < 3:
                continue
            centroids[k][user] = (sub["__pc1"].mean(), sub["__pc2"].mean())

    # Draw drift lines (same-key centroids across users)
    for k, user_pts in centroids.items():
        if len(user_pts) < 2:
            continue
        pts = list(user_pts.values())
        # Draw lines between every pair of users' centroids for this key
        for (p1, p2) in itertools.combinations(pts, 2):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color="gray", alpha=0.20, linewidth=0.6, zorder=1)

    # Draw centroids as colored dots
    for user, (df, proj) in per_user_projected.items():
        if df.empty:
            continue
        df = df.copy()
        df["__pc1"] = proj[:, 0]
        df["__pc2"] = proj[:, 1]
        for k in valid_keys:
            if user not in centroids[k]:
                continue
            cx, cy = centroids[k][user]
            ax.scatter([cx], [cy], color=user_colors[user], s=60,
                       edgecolor="black", linewidth=0.5, zorder=3,
                       label=user if k == valid_keys[0] else None)
            ax.text(cx, cy, k.upper(), ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white", zorder=4)

    ax.set_title(f"All users overlaid — {hand}\n"
                 "(gray lines = inter-user drift per key)",
                 fontsize=10)
    ax.set_xlabel("PC1 (mostly row depth)", fontsize=8)
    ax.set_ylabel("PC2 (mostly column position)", fontsize=8)
    # Dedupe legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)


def build_figure_1(per_user, hand, output_path):
    """Five-panel layout: 4 user keyboards + 1 overlap."""
    hand_df = get_hand_df(per_user, hand)
    if hand_df.empty:
        print(f"⚠️  No data for {hand} hand")
        return

    components, mu, sd, feature_cols, explained = fit_global_pca(hand_df, hand)
    print(f"\n   PCA explained variance — PC1: {explained[0]:.1%}, "
          f"PC2: {explained[1]:.1%}, total: {sum(explained):.1%}")

    # Project each user
    per_user_projected = {}
    for u, (rdf, ldf) in per_user.items():
        df = rdf if hand == "Right" else ldf
        if df.empty:
            continue
        proj = project_user(df, components, mu, sd, feature_cols)
        per_user_projected[u] = (df, proj)

    # Color scheme — keys colored by finger
    finger_color = {"Index": "#e74c3c", "Middle": "#3498db",
                    "Ring": "#2ecc71", "Little": "#f39c12"}
    color_map = {}
    for finger, keys in FINGER_KEYS[hand].items():
        for k in keys:
            color_map[k] = finger_color[finger]

    n_users = len(per_user_projected)
    fig = plt.figure(figsize=(5.5 * (n_users + 1), 5.5))
    gs = fig.add_gridspec(1, n_users + 1, wspace=0.30)

    for i, (u, (df, proj)) in enumerate(per_user_projected.items()):
        ax = fig.add_subplot(gs[0, i])
        plot_user_keyboard(df, proj, ax, u, hand, color_map)

    # Overlap panel
    ax_overlap = fig.add_subplot(gs[0, -1])
    plot_overlap_keyboard(per_user_projected, ax_overlap, hand)

    # Legend for finger colors
    finger_handles = [
        mpatches.Patch(color=finger_color[f], label=f) for f in FINGER_NAMES
    ]
    fig.legend(handles=finger_handles, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Reconstructed Mental Keyboards — {hand} hand\n"
        f"PC1+PC2 capture {sum(explained):.0%} of angle-feature variance",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    print(f"✅ Saved → {output_path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 2 — PER-KEY BEST-FEATURE DISCOVERY
# ═════════════════════════════════════════════════════════════════════════
# For each letter, we ask: among ALL pairs of angle features, which pair
# best separates THIS letter from its neighbors (= other keys on the same
# finger)? We use the Fisher discriminant ratio:
#     J(f1, f2) = ||μ_key - μ_others||² / (Σ_key + Σ_others)
# computed on the 2D plane spanned by (f1, f2).
#
# If the BEST feature pair differs across keys, then one universal pair
# (which our model partly uses) is fundamentally suboptimal.

def fisher_score_2d(X_pos, X_neg):
    """Fisher discriminant ratio in 2D: separation between two clusters,
    normalized by their spread. Higher = more separable."""
    if len(X_pos) < 3 or len(X_neg) < 3:
        return np.nan
    mu_pos = X_pos.mean(axis=0)
    mu_neg = X_neg.mean(axis=0)
    between = np.sum((mu_pos - mu_neg) ** 2)
    within = X_pos.var(axis=0).sum() + X_neg.var(axis=0).sum() + 1e-9
    return between / within


def find_best_feature_per_key(hand_df, hand, top_k_pairs=6):
    """For each key, find the top-K feature pairs by Fisher score against
    its 'neighbors' (other keys typed by the same finger).

    Returns a DataFrame with columns:
      key, rank, feat_x, feat_y, score, neighbors
    """
    feature_cols = [c for c in all_angle_features() if c in hand_df.columns]
    all_pairs = list(itertools.combinations(feature_cols, 2))
    print(f"   Searching {len(all_pairs)} feature pairs × {sum(len(v) for v in FINGER_KEYS[hand].values())} keys...")

    rows = []
    for finger, keys in FINGER_KEYS[hand].items():
        for k in keys:
            X_pos = hand_df[hand_df["Pressed_Letter"] == k]
            if len(X_pos) < 5:
                continue
            # "Neighbors" = OTHER keys on the same finger (the hard cases)
            neighbors = [k2 for k2 in keys if k2 != k]
            X_neg = hand_df[hand_df["Pressed_Letter"].isin(neighbors)]
            if len(X_neg) < 5:
                continue

            scores = []
            for fx, fy in all_pairs:
                Xp = X_pos[[fx, fy]].fillna(0).values
                Xn = X_neg[[fx, fy]].fillna(0).values
                # Standardize jointly so scores are comparable across pairs
                X_all = np.vstack([Xp, Xn])
                mu, sd = X_all.mean(axis=0), X_all.std(axis=0) + 1e-9
                Xp_s = (Xp - mu) / sd
                Xn_s = (Xn - mu) / sd
                s = fisher_score_2d(Xp_s, Xn_s)
                scores.append((s, fx, fy))

            scores.sort(reverse=True, key=lambda t: t[0])
            for rank, (s, fx, fy) in enumerate(scores[:top_k_pairs], 1):
                rows.append({
                    "key": k, "finger": finger, "rank": rank,
                    "feat_x": fx, "feat_y": fy,
                    "score": s,
                    "neighbors": ",".join(neighbors),
                })
    return pd.DataFrame(rows)


def plot_best_feature_heatmap(best_df, ax):
    """Heatmap: rows = keys, cols = feature pairs. Cell = Fisher score.
    Shows whether one feature pair dominates or whether the 'best' pair
    varies across keys."""
    # Use the BEST pair per key as a label, build a wide matrix of top scores
    top1 = best_df[best_df["rank"] == 1].copy()
    top1["pair_label"] = top1.apply(
        lambda r: f"{r['feat_x'].replace('_Mean','')}\n×\n{r['feat_y'].replace('_Mean','')}",
        axis=1,
    )

    # Group: for each (key, pair) take its best score (rank 1)
    ax.barh(top1["key"], top1["score"], color="#3498db", edgecolor="black",
            linewidth=0.5)
    for i, (k, lbl, s) in enumerate(zip(top1["key"], top1["pair_label"], top1["score"])):
        ax.text(s + 0.1, i, lbl, va="center", fontsize=7,
                color="#333", family="monospace")
    ax.set_xlabel("Fisher discriminant ratio (higher = more separable)",
                  fontsize=9)
    ax.set_title("Best feature pair per key (vs same-finger neighbors)",
                 fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")


def plot_feature_pair_diversity(best_df, ax):
    """For each key, plot a bar showing its top-K Fisher scores.
    If scores are flat → many pairs work equally well.
    If scores drop sharply → ONE pair dominates (good).
    If top score is low → NO pair separates this key well (bad)."""
    keys = best_df["key"].unique()
    cmap = plt.colormaps.get_cmap("viridis")

    for i, k in enumerate(keys):
        sub = best_df[best_df["key"] == k].sort_values("rank")
        ranks = sub["rank"].values
        scores = sub["score"].values
        color = cmap(i / max(len(keys) - 1, 1))
        ax.plot(ranks, scores, marker="o", color=color, label=k.upper(),
                linewidth=1.5, markersize=4)

    ax.set_xlabel("Feature-pair rank (1 = best)", fontsize=9)
    ax.set_ylabel("Fisher score", fontsize=9)
    ax.set_title("Score decay across top feature pairs per key",
                 fontsize=11, fontweight="bold")
    ax.legend(ncol=3, fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)


def build_figure_2(per_user, hand, output_path, top_k_pairs=6):
    """Two-panel figure: best-feature heatmap + score-decay diagnostic."""
    hand_df = get_hand_df(per_user, hand)
    if hand_df.empty:
        print(f"⚠️  No data for {hand} hand")
        return

    best_df = find_best_feature_per_key(hand_df, hand, top_k_pairs=top_k_pairs)
    if best_df.empty:
        print("⚠️  No best-feature results")
        return

    # Save the table for inspection
    csv_path = output_path.with_suffix(".csv")
    best_df.to_csv(csv_path, index=False)
    print(f"   📊 Saved feature search table → {csv_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7),
                                     gridspec_kw={"width_ratios": [1.2, 1]})
    plot_best_feature_heatmap(best_df, ax1)
    plot_feature_pair_diversity(best_df, ax2)

    fig.suptitle(
        f"Per-Key Best-Feature Discovery — {hand} hand\n"
        f"Different letters need different features? (top-{top_k_pairs} pairs per key)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    print(f"✅ Saved → {output_path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-pattern", default=PATH_PATTERN)
    parser.add_argument("--users", default=None)
    parser.add_argument("--hand", default="Left", choices=["Left", "Right"])
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--top-k-pairs", type=int, default=6,
                        help="How many top-scoring feature pairs to show per key")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    users = args.users.split(",") if args.users else VALID_PARTICIPANTS
    per_user = load_per_user(args.path_pattern, users)

    print("\n" + "=" * 60)
    print(f"FIGURE 1: Reconstructed keyboards + overlap — {args.hand}")
    print("=" * 60)
    build_figure_1(per_user, args.hand,
                    output_dir / f"keyboards_{args.hand.lower()}.png")

    print("\n" + "=" * 60)
    print(f"FIGURE 2: Per-key best-feature discovery — {args.hand}")
    print("=" * 60)
    build_figure_2(per_user, args.hand,
                    output_dir / f"best_features_{args.hand.lower()}.png",
                    top_k_pairs=args.top_k_pairs)


if __name__ == "__main__":
    main()