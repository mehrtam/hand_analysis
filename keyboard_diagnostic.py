# """
# keyboard_diagnostic.py
# ======================
# Three-panel diagnostic dashboard:

#   1. PER-USER MENTAL KEYBOARD — for each user, plot every keystroke
#      in 2D angle space, colored by letter. Tight separated clouds = a
#      well-defined personal keyboard. Smeared clouds = chaotic mapping.

#   2. INTER-USER OVERLAP MAP — for one chosen letter, plot all 4 users'
#      keystrokes on the same axes, colored by user. Tells us whether q11's
#      "H" lives in the same angle-space region as q14's "H".

#   3. CONFUSION-AS-KEYBOARD — draw a QWERTY layout. For each confused pair
#      from a LOOCV results folder, draw an arrow between the two keys with
#      thickness ∝ error count. Patterns reveal row vs column confusions.

# Usage:
#     # Build dashboard from a LOOCV results folder + the raw data
#     python keyboard_diagnostic.py --results-dir results/20260518_131039

#     # Or just build the first two panels (don't need LOOCV results)
#     python keyboard_diagnostic.py --no-confusion
# """
# import argparse
# import logging
# from pathlib import Path

# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np
# import pandas as pd
# from matplotlib.lines import Line2D
# from matplotlib.patches import FancyArrowPatch, Rectangle

# from train_v2 import (
#     VALID_PARTICIPANTS,
#     load_all_data,
#     aggregate_features,
#     participant_id,
#     KEY_TO_FINGER,
#     KEY_TO_ROW,
#     PATH_PATTERN,
# )

# log = logging.getLogger(__name__)


# # ═════════════════════════════════════════════════════════════════════════
# # QWERTY LAYOUT (for confusion-as-keyboard view)
# # ═════════════════════════════════════════════════════════════════════════
# KEYBOARD_ROWS = [
#     "qwertyuiop",
#     "asdfghjkl",
#     "zxcvbnm",
# ]
# KEY_POS = {}  # (x, y) for each key in QWERTY layout
# for r_idx, row in enumerate(KEYBOARD_ROWS):
#     for c_idx, k in enumerate(row):
#         KEY_POS[k] = (c_idx + r_idx * 0.5, -r_idx)  # 0.5 stagger per row


# # ═════════════════════════════════════════════════════════════════════════
# # LOAD DATA + AGGREGATE FEATURES (uses your existing pipeline)
# # ═════════════════════════════════════════════════════════════════════════
# def load_features_per_user(path_pattern, users):
#     """Returns dict {user_id: (right_features_df, left_features_df)}.

#     Each features_df is the output of aggregate_features() filtered to
#     one user — one row per keystroke, columns = angle/velocity features.
#     """
#     print(f"📂 Loading data for {users}...")
#     right_segs, left_segs = load_all_data(path_pattern, users)
#     print("📊 Aggregating features...")
#     right_feat, _ = aggregate_features(right_segs)
#     left_feat, _  = aggregate_features(left_segs)

#     right_feat["pid"] = right_feat["Filepath"].apply(participant_id)
#     left_feat["pid"]  = left_feat["Filepath"].apply(participant_id)

#     per_user = {}
#     for u in users:
#         per_user[u] = (
#             right_feat[right_feat["pid"] == u].reset_index(drop=True),
#             left_feat[left_feat["pid"] == u].reset_index(drop=True),
#         )
#     return per_user


# # ═════════════════════════════════════════════════════════════════════════
# # WHICH ANGLE FEATURES TO PLOT?
# # ═════════════════════════════════════════════════════════════════════════
# # We let the user pick the X-axis and Y-axis angle. Defaults are chosen to
# # discriminate well for a given hand:
# #  - Right hand: Index MCP flexion (X) vs Index PIP flexion (Y)
# #    → captures the y/u/h/j/n/m diagonal
# #  - Left hand: Little MCP flexion (X) vs Index PIP flexion (Y)
# #    → spans the q/a/z column AND the index column
# DEFAULT_AXES = {
#     "Right": ("Index_MCP_Flexion_Mean",  "Index_PIP_Flexion_Mean"),
#     "Left":  ("Little_MCP_Flexion_Mean", "Index_PIP_Flexion_Mean"),
# }


# # ═════════════════════════════════════════════════════════════════════════
# # PANEL 1 — PER-USER MENTAL KEYBOARD
# # ═════════════════════════════════════════════════════════════════════════
# def plot_per_user_mental_keyboard(per_user_features, hand, axes_pair, ax):
#     """Sub-figure: one user's keystrokes in 2D angle space, colored by letter."""
#     user, (right_df, left_df) = per_user_features
#     df = right_df if hand == "Right" else left_df
#     x_col, y_col = axes_pair

#     if df.empty or x_col not in df.columns or y_col not in df.columns:
#         ax.text(0.5, 0.5, f"{user}: no data", ha="center", va="center",
#                 transform=ax.transAxes)
#         ax.set_title(f"{user} — {hand}")
#         return

#     keys = sorted(df["Pressed_Letter"].unique())
#     cmap = plt.colormaps.get_cmap("tab20")

#     for i, k in enumerate(keys):
#         sub = df[df["Pressed_Letter"] == k]
#         ax.scatter(sub[x_col], sub[y_col], color=cmap(i % 20),
#                    s=12, alpha=0.45, edgecolor="none")
#         # Annotate cluster center
#         cx, cy = sub[x_col].median(), sub[y_col].median()
#         ax.annotate(k, (cx, cy), fontsize=11, fontweight="bold",
#                     ha="center", va="center",
#                     bbox=dict(boxstyle="round,pad=0.15",
#                               facecolor="white", alpha=0.75, edgecolor="gray"))

#     ax.set_xlabel(x_col.replace("_Mean", "").replace("_", " "), fontsize=8)
#     ax.set_ylabel(y_col.replace("_Mean", "").replace("_", " "), fontsize=8)
#     ax.set_title(f"{user} — {hand} hand  (n={len(df)})", fontsize=10)
#     ax.grid(True, alpha=0.3)


# # ═════════════════════════════════════════════════════════════════════════
# # PANEL 2 — INTER-USER OVERLAP MAP
# # ═════════════════════════════════════════════════════════════════════════
# def plot_inter_user_overlap(per_user_features, hand, axes_pair, target_letter, ax):
#     """Plot all users' keystrokes for ONE letter on the same axes.
#     Different colors per user. Tight overlap = generalisable; spread = not."""
#     x_col, y_col = axes_pair
#     cmap = plt.colormaps.get_cmap("tab10")

#     user_colors = {}
#     any_data = False
#     for i, (user, (right_df, left_df)) in enumerate(per_user_features.items()):
#         df = right_df if hand == "Right" else left_df
#         if df.empty:
#             continue
#         sub = df[df["Pressed_Letter"] == target_letter]
#         if sub.empty:
#             continue
#         color = cmap(i % 10)
#         user_colors[user] = color
#         ax.scatter(sub[x_col], sub[y_col], color=color, s=25, alpha=0.55,
#                    edgecolor="black", linewidth=0.3, label=f"{user} (n={len(sub)})")

#         # Ellipse showing one std of this user's cloud
#         if len(sub) >= 5:
#             cx, cy = sub[x_col].mean(), sub[y_col].mean()
#             sx, sy = sub[x_col].std(), sub[y_col].std()
#             ax.add_patch(mpatches.Ellipse((cx, cy), 2*sx, 2*sy,
#                                           fill=False, color=color, lw=2, ls="--"))
#         any_data = True

#     if not any_data:
#         ax.text(0.5, 0.5, f"No '{target_letter}' data", ha="center", va="center",
#                 transform=ax.transAxes)
#     ax.set_xlabel(x_col.replace("_Mean", "").replace("_", " "), fontsize=8)
#     ax.set_ylabel(y_col.replace("_Mean", "").replace("_", " "), fontsize=8)
#     ax.set_title(f"All users typing '{target_letter}' — {hand}", fontsize=10)
#     ax.legend(fontsize=7, loc="best")
#     ax.grid(True, alpha=0.3)


# # ═════════════════════════════════════════════════════════════════════════
# # PANEL 3 — CONFUSION AS KEYBOARD
# # ═════════════════════════════════════════════════════════════════════════
# def plot_confusion_keyboard(confused_pairs_csv, ax, top_n=20):
#     """Draw QWERTY layout. For top N confused pairs, draw an arrow between the
#     two keys; arrow width = error count. Color codes the *type* of confusion:
#       - Vertical (column-mate, e.g. q-a, a-z):     ORANGE   (row error)
#       - Horizontal (row-mate, e.g. f-g, h-j):       BLUE     (column-within-row error)
#       - Diagonal:                                   GRAY     (mixed)
#     """
#     pairs = pd.read_csv(confused_pairs_csv)
#     pairs = pairs.head(top_n)

#     # Draw keys as squares
#     for k, (x, y) in KEY_POS.items():
#         rect = Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
#                          fill=True, facecolor="#f0f0f0",
#                          edgecolor="black", linewidth=1)
#         ax.add_patch(rect)
#         ax.text(x, y, k.upper(), ha="center", va="center",
#                 fontsize=11, fontweight="bold")

#     # Draw confusion arrows
#     max_count = pairs["count"].max() if not pairs.empty else 1
#     for _, r in pairs.iterrows():
#         ka, kb = r["key_a"], r["key_b"]
#         if ka not in KEY_POS or kb not in KEY_POS:
#             continue
#         xa, ya = KEY_POS[ka]
#         xb, yb = KEY_POS[kb]

#         # Color by confusion type
#         same_row = (KEY_TO_ROW.get(ka) == KEY_TO_ROW.get(kb))
#         same_finger = (KEY_TO_FINGER.get(ka) == KEY_TO_FINGER.get(kb))
#         if same_finger and not same_row:
#             color = "#e67e22"     # ORANGE: same finger, different row (row error)
#         elif same_row and not same_finger:
#             color = "#2980b9"     # BLUE: same row, different finger
#         else:
#             color = "#7f8c8d"     # GRAY: other

#         lw = 1 + 6 * (r["count"] / max_count)
#         arrow = FancyArrowPatch((xa, ya), (xb, yb),
#                                 connectionstyle="arc3,rad=0.2",
#                                 arrowstyle="-",
#                                 color=color, linewidth=lw, alpha=0.75)
#         ax.add_patch(arrow)

#         # Label the count at midpoint
#         mx, my = (xa + xb) / 2, (ya + yb) / 2
#         ax.text(mx, my + 0.05, str(int(r["count"])),
#                 fontsize=8, ha="center",
#                 bbox=dict(boxstyle="round,pad=0.1",
#                           facecolor="white", alpha=0.8, edgecolor="none"))

#     # Legend
#     legend_elements = [
#         Line2D([0], [0], color="#e67e22", lw=3,
#                label="Row error (same finger, ≠ row)"),
#         Line2D([0], [0], color="#2980b9", lw=3,
#                label="Column error (same row, ≠ finger)"),
#         Line2D([0], [0], color="#7f8c8d", lw=3, label="Other"),
#     ]
#     ax.legend(handles=legend_elements, loc="lower center",
#               bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)

#     ax.set_xlim(-1, 11)
#     ax.set_ylim(-3, 1)
#     ax.set_aspect("equal")
#     ax.axis("off")
#     ax.set_title(f"Confusion-as-keyboard (top {top_n} confused pairs)",
#                  fontsize=11)


# # ═════════════════════════════════════════════════════════════════════════
# # DASHBOARD ASSEMBLER
# # ═════════════════════════════════════════════════════════════════════════
# def build_dashboard(per_user_features, results_dir, output_path,
#                      hand="Left", target_letter="a"):
#     """Assemble all three panels into one figure."""
#     users = list(per_user_features.keys())
#     n_users = len(users)
#     axes_pair = DEFAULT_AXES[hand]

#     # Figure layout: panel 1 spans top (one column per user), panel 2 + 3 share row 2
#     fig = plt.figure(figsize=(5 * n_users, 14))
#     gs = fig.add_gridspec(3, n_users, hspace=0.35, wspace=0.3,
#                           height_ratios=[1.0, 1.0, 1.0])

#     # ── Panel 1: per-user mental keyboard (one subplot per user) ────────
#     for i, u in enumerate(users):
#         ax = fig.add_subplot(gs[0, i])
#         plot_per_user_mental_keyboard(
#             (u, per_user_features[u]), hand, axes_pair, ax,
#         )

#     # ── Panel 2: inter-user overlap for the target letter ───────────────
#     ax2 = fig.add_subplot(gs[1, :n_users // 2 + 1] if n_users > 1 else gs[1, :])
#     plot_inter_user_overlap(per_user_features, hand, axes_pair, target_letter, ax2)

#     # ── Panel 3: confusion-as-keyboard (uses LOOCV output) ──────────────
#     if results_dir is not None:
#         confused_csv = Path(results_dir) / "analysis" / "confused_pairs.csv"
#         if confused_csv.exists():
#             ax3 = fig.add_subplot(gs[1, n_users // 2 + 1:] if n_users > 1 else gs[2, :])
#             plot_confusion_keyboard(confused_csv, ax3, top_n=20)
#         else:
#             print(f"⚠️  No confused_pairs.csv found at {confused_csv}")

#     # ── Row 3: same dashboard for the OTHER hand ─────────────────────────
#     other_hand = "Right" if hand == "Left" else "Left"
#     axes_pair_other = DEFAULT_AXES[other_hand]
#     for i, u in enumerate(users):
#         ax = fig.add_subplot(gs[2, i])
#         plot_per_user_mental_keyboard(
#             (u, per_user_features[u]), other_hand, axes_pair_other, ax,
#         )

#     fig.suptitle(
#         f"Mental Keyboard Diagnostic  |  primary hand: {hand}  |  target letter: '{target_letter}'",
#         fontsize=14, fontweight="bold", y=0.995,
#     )
#     plt.savefig(output_path, dpi=120, bbox_inches="tight")
#     print(f"\n✅ Saved dashboard → {output_path}")
#     plt.close()


# # ═════════════════════════════════════════════════════════════════════════
# # MAIN
# # ═════════════════════════════════════════════════════════════════════════
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--path-pattern", default=PATH_PATTERN)
#     parser.add_argument("--users", default=None,
#                         help="Comma-separated user IDs (default: all 4)")
#     parser.add_argument("--results-dir", default=None,
#                         help="LOOCV results folder (for confusion panel). "
#                              "If omitted, only panels 1+2 are built.")
#     parser.add_argument("--hand", default="Left", choices=["Left", "Right"])
#     parser.add_argument("--target-letter", default="a",
#                         help="Letter to compare across users in panel 2")
#     parser.add_argument("--output", default="keyboard_diagnostic.png")
#     parser.add_argument("--no-confusion", action="store_true",
#                         help="Skip the confusion panel even if results-dir is given")
#     args = parser.parse_args()

#     users = args.users.split(",") if args.users else VALID_PARTICIPANTS
#     per_user = load_features_per_user(args.path_pattern, users)

#     results_dir = None if args.no_confusion else args.results_dir
#     build_dashboard(
#         per_user, results_dir, args.output,
#         hand=args.hand, target_letter=args.target_letter,
#     )


# if __name__ == "__main__":
#     main()
"""
keyboard_diagnostic.py
======================
Per-finger keyboard diagnostic. Each user gets a 1×4 row of panels —
one per finger — showing only that finger's keys in the angle space
of that finger. This is the correct view under the touch-typing
assumption: pinky angles for q/a/z, index angles for r/t/f/g/v/b, etc.

Panels:
  1. PER-USER PER-FINGER — does this user's hand consistently produce
     the right posture for the right key?
  2. INTER-USER OVERLAP — for one chosen letter, all users overlaid in
     the *correct* finger's axes.
  3. CONFUSION-AS-KEYBOARD — top confused pairs drawn on QWERTY.

Usage:
    python keyboard_diagnostic.py --no-confusion --target-letter q --hand Left
    python keyboard_diagnostic.py --results-dir results/20260518_131039 \
        --target-letter a --hand Left
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

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
# QWERTY LAYOUT (for the confusion-as-keyboard panel)
# ═════════════════════════════════════════════════════════════════════════
KEYBOARD_ROWS = [
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
]
KEY_POS = {}
for r_idx, row in enumerate(KEYBOARD_ROWS):
    for c_idx, k in enumerate(row):
        KEY_POS[k] = (c_idx + r_idx * 0.5, -r_idx)


# ═════════════════════════════════════════════════════════════════════════
# TOUCH-TYPING FINGER ASSIGNMENTS
# ═════════════════════════════════════════════════════════════════════════
# Each finger covers a column of 3 keys (top/home/bottom).
# Index covers 2 columns; everything else covers 1.
FINGER_KEYS = {
    "Left": {
        "Little": ["q", "a", "z"],
        "Ring":   ["w", "s", "x"],
        "Middle": ["e", "d", "c"],
        "Index":  ["r", "t", "f", "g", "v", "b"],
    },
    "Right": {
        "Index":  ["y", "u", "h", "j", "n", "m"],
        "Middle": ["i", "k"],
        "Ring":   ["o", "l"],
        "Little": ["p"],
    },
}

# Best 2D axes per finger:
#  - MCP Flexion → depth of reach (Top vs Home vs Bottom of the column)
#  - MCP Abduction → sideways spread (which column for 2-column fingers)
#  - PIP Flexion → curl tightness (also responds to row)
# For single-column fingers: MCP_Flexion × PIP_Flexion (both row-sensitive).
# For Index (2 columns): MCP_Flexion × MCP_Abduction (row × column).
FINGER_AXES = {
    "Little": ("Little_MCP_Flexion_Mean", "Little_PIP_Flexion_Mean"),
    "Ring":   ("Ring_MCP_Flexion_Mean",   "Ring_PIP_Flexion_Mean"),
    "Middle": ("Middle_MCP_Flexion_Mean", "Middle_PIP_Flexion_Mean"),
    "Index":  ("Index_MCP_Flexion_Mean",  "Index_MCP_Abduction_Mean"),
}


# ═════════════════════════════════════════════════════════════════════════
# DATA LOADING (uses your existing pipeline)
# ═════════════════════════════════════════════════════════════════════════
def load_features_per_user(path_pattern, users):
    """Return {user_id: (right_features_df, left_features_df)}."""
    print(f"📂 Loading data for {users}...")
    right_segs, left_segs = load_all_data(path_pattern, users)
    print("📊 Aggregating features...")
    right_feat, _ = aggregate_features(right_segs)
    left_feat, _  = aggregate_features(left_segs)

    right_feat["pid"] = right_feat["Filepath"].apply(participant_id)
    left_feat["pid"]  = left_feat["Filepath"].apply(participant_id)

    per_user = {}
    for u in users:
        per_user[u] = (
            right_feat[right_feat["pid"] == u].reset_index(drop=True),
            left_feat[left_feat["pid"] == u].reset_index(drop=True),
        )
    return per_user


def _finger_of(letter, hand):
    """Which finger types this letter on the given hand?"""
    for finger, keys in FINGER_KEYS.get(hand, {}).items():
        if letter in keys:
            return finger
    return None


# ═════════════════════════════════════════════════════════════════════════
# PANEL 1 — PER-USER PER-FINGER VIEW
# ═════════════════════════════════════════════════════════════════════════
def plot_user_per_finger(user, df, hand, fig, gs_slice):
    """One row of 4 panels per user. Each panel shows ONE finger's keys
    plotted against THAT finger's discriminative angles."""
    finger_order = (["Little", "Ring", "Middle", "Index"] if hand == "Left"
                    else ["Index", "Middle", "Ring", "Little"])

    if df is None or df.empty:
        ax = fig.add_subplot(gs_slice[0])
        ax.text(0.5, 0.5, f"{user}: no data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(f"{user} — {hand}")
        return

    cmap = plt.colormaps.get_cmap("tab10")

    for col, finger in enumerate(finger_order):
        ax = fig.add_subplot(gs_slice[col])
        x_col, y_col = FINGER_AXES[finger]
        keys = FINGER_KEYS[hand][finger]

        if x_col not in df.columns or y_col not in df.columns:
            ax.text(0.5, 0.5, "missing\nfeatures", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_title(f"{finger}", fontsize=9)
            continue

        for i, k in enumerate(keys):
            sub = df[df["Pressed_Letter"] == k]
            if sub.empty:
                continue
            color = cmap(i % 10)
            ax.scatter(sub[x_col], sub[y_col], color=color, s=14,
                       alpha=0.55, edgecolor="none")
            cx, cy = sub[x_col].median(), sub[y_col].median()
            ax.annotate(k.upper(), (cx, cy), fontsize=10, fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="white", alpha=0.85,
                                  edgecolor=color, linewidth=1.2))

        ax.set_xlabel(x_col.replace("_Mean", "").replace("_", " "), fontsize=7)
        ax.set_ylabel(y_col.replace("_Mean", "").replace("_", " "), fontsize=7)
        title = f"{user} — {finger}" if col == 0 else f"{finger}"
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)


# ═════════════════════════════════════════════════════════════════════════
# PANEL 2 — INTER-USER OVERLAP (correct finger's axes for target letter)
# ═════════════════════════════════════════════════════════════════════════
def plot_overlap_with_axes(per_user_features, hand, axes_pair, target_letter, ax):
    x_col, y_col = axes_pair
    cmap = plt.colormaps.get_cmap("tab10")
    any_data = False
    for i, (user, (right_df, left_df)) in enumerate(per_user_features.items()):
        df = right_df if hand == "Right" else left_df
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            continue
        sub = df[df["Pressed_Letter"] == target_letter]
        if sub.empty:
            continue
        color = cmap(i % 10)
        ax.scatter(sub[x_col], sub[y_col], color=color, s=30, alpha=0.55,
                   edgecolor="black", linewidth=0.3,
                   label=f"{user} (n={len(sub)})")
        if len(sub) >= 5:
            cx, cy = sub[x_col].mean(), sub[y_col].mean()
            sx, sy = sub[x_col].std(), sub[y_col].std()
            ax.add_patch(mpatches.Ellipse((cx, cy), 2*sx, 2*sy,
                                          fill=False, color=color, lw=2, ls="--"))
        any_data = True
    if not any_data:
        ax.text(0.5, 0.5, f"No '{target_letter}' data",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(x_col.replace("_Mean", "").replace("_", " "), fontsize=8)
    ax.set_ylabel(y_col.replace("_Mean", "").replace("_", " "), fontsize=8)
    ax.set_title(f"All users typing '{target_letter}' — {hand}", fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)


# ═════════════════════════════════════════════════════════════════════════
# PANEL 3 — CONFUSION AS KEYBOARD
# ═════════════════════════════════════════════════════════════════════════
def plot_confusion_keyboard(confused_pairs_csv, ax, top_n=20):
    pairs = pd.read_csv(confused_pairs_csv).head(top_n)

    for k, (x, y) in KEY_POS.items():
        rect = Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                         fill=True, facecolor="#f0f0f0",
                         edgecolor="black", linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y, k.upper(), ha="center", va="center",
                fontsize=11, fontweight="bold")

    max_count = pairs["count"].max() if not pairs.empty else 1
    for _, r in pairs.iterrows():
        ka, kb = r["key_a"], r["key_b"]
        if ka not in KEY_POS or kb not in KEY_POS:
            continue
        xa, ya = KEY_POS[ka]
        xb, yb = KEY_POS[kb]

        same_row = (KEY_TO_ROW.get(ka) == KEY_TO_ROW.get(kb))
        same_finger = (KEY_TO_FINGER.get(ka) == KEY_TO_FINGER.get(kb))
        if same_finger and not same_row:
            color = "#e67e22"
        elif same_row and not same_finger:
            color = "#2980b9"
        else:
            color = "#7f8c8d"

        lw = 1 + 6 * (r["count"] / max_count)
        arrow = FancyArrowPatch((xa, ya), (xb, yb),
                                connectionstyle="arc3,rad=0.2",
                                arrowstyle="-",
                                color=color, linewidth=lw, alpha=0.75)
        ax.add_patch(arrow)

        mx, my = (xa + xb) / 2, (ya + yb) / 2
        ax.text(mx, my + 0.05, str(int(r["count"])),
                fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.1",
                          facecolor="white", alpha=0.8, edgecolor="none"))

    legend_elements = [
        Line2D([0], [0], color="#e67e22", lw=3,
               label="Row error (same finger, ≠ row)"),
        Line2D([0], [0], color="#2980b9", lw=3,
               label="Column error (same row, ≠ finger)"),
        Line2D([0], [0], color="#7f8c8d", lw=3, label="Other"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)

    ax.set_xlim(-1, 11)
    ax.set_ylim(-3, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Confusion-as-keyboard (top {top_n} pairs)", fontsize=11)


# ═════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═════════════════════════════════════════════════════════════════════════
def build_dashboard(per_user_features, results_dir, output_path,
                     hand="Left", target_letter="a"):
    users = list(per_user_features.keys())
    n_users = len(users)
    n_fingers = 4

    fig = plt.figure(figsize=(3.5 * n_fingers, 4.5 * n_users + 8))
    total_rows = n_users + 2 + n_users
    gs = fig.add_gridspec(
        total_rows, n_fingers,
        hspace=0.55, wspace=0.35,
        height_ratios=[1.0] * n_users + [1.4, 1.4] + [1.0] * n_users,
    )

    # Top block: per-finger panels for the primary hand
    for r, u in enumerate(users):
        right_df, left_df = per_user_features[u]
        df = left_df if hand == "Left" else right_df
        row_slice = [gs[r, c] for c in range(n_fingers)]
        plot_user_per_finger(u, df, hand, fig, row_slice)

    # Middle row 1: inter-user overlap, in the target letter's finger axes
    target_finger = _finger_of(target_letter, hand) or "Index"
    overlap_axes = FINGER_AXES[target_finger]
    ax2 = fig.add_subplot(gs[n_users, :2])
    plot_overlap_with_axes(per_user_features, hand, overlap_axes,
                            target_letter, ax2)

    # Middle row 2: confusion keyboard (if results dir given)
    if results_dir is not None:
        confused_csv = Path(results_dir) / "analysis" / "confused_pairs.csv"
        if confused_csv.exists():
            ax3 = fig.add_subplot(gs[n_users, 2:])
            plot_confusion_keyboard(confused_csv, ax3, top_n=20)
        else:
            print(f"⚠️  No confused_pairs.csv at {confused_csv}")

    # Bottom block: per-finger panels for the OTHER hand
    other_hand = "Right" if hand == "Left" else "Left"
    for r, u in enumerate(users):
        right_df, left_df = per_user_features[u]
        df = left_df if other_hand == "Left" else right_df
        row_slice = [gs[n_users + 2 + r, c] for c in range(n_fingers)]
        plot_user_per_finger(u, df, other_hand, fig, row_slice)

    fig.suptitle(
        f"Mental Keyboard Diagnostic — Per-Finger View  "
        f"|  primary hand: {hand}  |  target letter: '{target_letter}'",
        fontsize=14, fontweight="bold", y=0.998,
    )
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\n✅ Saved dashboard → {output_path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-pattern", default=PATH_PATTERN)
    parser.add_argument("--users", default=None,
                        help="Comma-separated user IDs (default: all 4)")
    parser.add_argument("--results-dir", default=None,
                        help="LOOCV results folder (for confusion panel).")
    parser.add_argument("--hand", default="Left", choices=["Left", "Right"])
    parser.add_argument("--target-letter", default="a",
                        help="Letter to compare across users in the overlap panel")
    parser.add_argument("--output", default="keyboard_diagnostic.png")
    parser.add_argument("--no-confusion", action="store_true",
                        help="Skip the confusion panel even if results-dir is given")
    args = parser.parse_args()

    users = args.users.split(",") if args.users else VALID_PARTICIPANTS
    per_user = load_features_per_user(args.path_pattern, users)

    results_dir = None if args.no_confusion else args.results_dir
    build_dashboard(
        per_user, results_dir, args.output,
        hand=args.hand, target_letter=args.target_letter,
    )


if __name__ == "__main__":
    main()