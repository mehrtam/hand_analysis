"""
drift_figure.py
===============
Publication figure: QWERTY keyboard layout where each key is colored
by its inter-user drift_ratio. Higher = more inter-user disagreement
on hand posture for that key.

Reads:  results/separability/inter_user_drift.csv
Writes: figures/drift_keyboard.png and figures/drift_keyboard.pdf

Usage:
    python drift_figure.py
    python drift_figure.py --input results/separability/inter_user_drift.csv
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize


# ═════════════════════════════════════════════════════════════════════════
# PUBLICATION RC PARAMS (academic look)
# ═════════════════════════════════════════════════════════════════════════
mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":         10,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "axes.linewidth":    0.8,
    "axes.edgecolor":    "#333333",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":      42,   # editable text in vector PDFs
    "ps.fonttype":       42,
})


# ═════════════════════════════════════════════════════════════════════════
# KEYBOARD LAYOUT
# ═════════════════════════════════════════════════════════════════════════
# Standard staggered QWERTY. (col, row) where row 0 = top.
KEYBOARD_ROWS = [
    "qwertyuiop",   # row 0  (Top)
    "asdfghjkl",    # row 1  (Home)    — staggered 0.5 right
    "zxcvbnm",      # row 2  (Bottom)  — staggered 1.0 right
]
ROW_STAGGER = [0.0, 0.5, 1.0]

KEY_POS = {}
for r_idx, row in enumerate(KEYBOARD_ROWS):
    for c_idx, k in enumerate(row):
        x = c_idx + ROW_STAGGER[r_idx]
        y = -r_idx
        KEY_POS[k] = (x, y)


# ═════════════════════════════════════════════════════════════════════════
# ACADEMIC PALETTE
# ═════════════════════════════════════════════════════════════════════════
# Muted blue → neutral → muted orange. Centered at drift_ratio = 1.0
# (the "neutral" boundary: above this, users disagree on a key more than
# two letters disagree within one user).
ACADEMIC_BLUE   = "#3b6e8f"   # muted teal-blue
ACADEMIC_LIGHT  = "#f0ece2"   # warm off-white
ACADEMIC_ORANGE = "#c1652c"   # muted burnt orange
ACCENT_DARK     = "#1f2933"   # near-black for text

CMAP = LinearSegmentedColormap.from_list(
    "academic_drift",
    [ACADEMIC_BLUE, ACADEMIC_LIGHT, ACADEMIC_ORANGE],
    N=256,
)


def get_color(drift_ratio, vmin, vmax):
    """Map drift_ratio to a colour, centred so 1.0 lands at the neutral midpoint."""
    # Symmetric around 1.0: clip range so 1.0 is at colormap centre
    span = max(abs(vmin - 1.0), abs(vmax - 1.0))
    norm = Normalize(vmin=1.0 - span, vmax=1.0 + span)
    return CMAP(norm(drift_ratio))


# ═════════════════════════════════════════════════════════════════════════
# FIGURE
# ═════════════════════════════════════════════════════════════════════════
def make_figure(drift_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(drift_csv)
    # If a key appears for both hands (which shouldn't happen with this dataset,
    # but defensive code), take the mean. Otherwise use the value we have.
    df_keyed = df.groupby("key").agg(
        drift_ratio=("drift_ratio", "mean"),
        n_total_samples=("n_total_samples", "sum"),
        hand=("hand", "first"),
    ).reset_index()
    key_to_drift = dict(zip(df_keyed["key"], df_keyed["drift_ratio"]))
    key_to_n     = dict(zip(df_keyed["key"], df_keyed["n_total_samples"]))

    vmin = df_keyed["drift_ratio"].min()
    vmax = df_keyed["drift_ratio"].max()

    # 7-inch double-column width, height tuned for keyboard aspect ratio
    fig, ax = plt.subplots(figsize=(7.0, 3.3))

    key_size = 0.92  # leave a hair of whitespace between keys

    for k, (x, y) in KEY_POS.items():
        drift = key_to_drift.get(k)
        if drift is None:
            face = "#dddddd"
            edge = "#999999"
            text_color = "#999999"
            label_drift = "—"
        else:
            face = get_color(drift, vmin, vmax)
            edge = ACCENT_DARK
            # Decide text colour based on luminance of background
            r, g, b, _ = face
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = ACCENT_DARK if luminance > 0.55 else "white"
            label_drift = f"{drift:.2f}"

        # Rounded square key
        box = FancyBboxPatch(
            (x - key_size / 2, y - key_size / 2),
            key_size, key_size,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            linewidth=0.9, edgecolor=edge, facecolor=face,
        )
        ax.add_patch(box)

        # Key letter (uppercase, large)
        ax.text(x, y + 0.13, k.upper(),
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=text_color, family="sans-serif")

        # Drift ratio number (smaller, below letter)
        ax.text(x, y - 0.25, label_drift,
                ha="center", va="center",
                fontsize=8, color=text_color, family="sans-serif")

    # Set bounds with a little margin
    xs = [p[0] for p in KEY_POS.values()]
    ys = [p[1] for p in KEY_POS.values()]
    ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Colorbar ──────────────────────────────────────────────────────
    # Build a horizontal colorbar at the bottom
    span = max(abs(vmin - 1.0), abs(vmax - 1.0))
    norm = Normalize(vmin=1.0 - span, vmax=1.0 + span)
    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.30, 0.06, 0.42, 0.025])   # [left, bottom, w, h]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.outline.set_edgecolor(ACCENT_DARK)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=8, color=ACCENT_DARK, width=0.6)

    # Mark the neutral boundary explicitly
    cbar.ax.axvline(1.0, color=ACCENT_DARK, linewidth=1.0, linestyle="--", alpha=0.7)
    cbar.set_label(
        "Inter-user drift ratio  (1.00 = parity with within-user letter spread)",
        fontsize=9, color=ACCENT_DARK, labelpad=4,
    )

    # ── Title and subtitle ───────────────────────────────────────────
    fig.suptitle(
        "Per-key inter-user posture drift across N=4 users",
        fontsize=11.5, fontweight="bold", color=ACCENT_DARK, y=0.97,
    )
    fig.text(
        0.5, 0.905,
        f"Drift ratio > 1: same letter across users differs more than two letters within one user  "
        f"|  range: {vmin:.2f}–{vmax:.2f}",
        ha="center", fontsize=8.5, color="#555555", style="italic",
    )

    # ── Annotate the headline finding directly on the figure ─────────
    # Pull a couple of the worst keys to call out
    worst = df_keyed.sort_values("drift_ratio", ascending=False).iloc[0]
    fig.text(
        0.02, 0.02,
        f"Worst-drift key: '{worst['key'].upper()}' (drift = {worst['drift_ratio']:.2f})  "
        f"|  mean drift: {df_keyed['drift_ratio'].mean():.2f}  "
        f"|  N keys = {len(df_keyed)}",
        fontsize=7.5, color="#666666", family="sans-serif",
    )

    # Save in both raster and vector formats
    png_path = output_dir / "drift_keyboard.png"
    pdf_path = output_dir / "drift_keyboard.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"✅ Saved: {png_path}")
    print(f"✅ Saved: {pdf_path}")
    print(f"\nFigure summary:")
    print(f"   {len(df_keyed)} keys plotted")
    print(f"   drift range: {vmin:.2f} – {vmax:.2f}")
    print(f"   neutral boundary at 1.00 shown as dashed line on colorbar")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="results/separability/inter_user_drift.csv",
        help="Path to inter_user_drift.csv from separability_analysis.py",
    )
    parser.add_argument(
        "--output-dir", default="figures",
        help="Directory to write the figure(s) to",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Drift CSV not found at {input_path}. "
            f"Run separability_analysis.py first."
        )
    make_figure(input_path, args.output_dir)


if __name__ == "__main__":
    main()