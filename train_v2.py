"""
train_v2.py
===========
Self-contained training pipeline for keystroke prediction from joint angles.

Major improvements over v1
---------------------------
1. WRIST-RELATIVE COORDINATES — every fingertip and joint is expressed
   relative to the wrist midpoint, removing dependency on the user's
   absolute wrist position. This was the root cause of the row-detection
   failure on q25 (b/c/g/v at 0% adjacent accuracy).

2. MARKER DROPOUT AUGMENTATION — during training, 20% of samples have
   I4/M2/etc. randomly zeroed out, teaching the model to handle the
   marker occlusion seen in real deployment users.

3. SCALER SAVED IN PKL — no fit-at-test-time anywhere. Deployment loads
   and applies. Eliminates the single-row-StandardScaler collapse bug.

4. HIERARCHICAL ROW → KEY ARCHITECTURE — a Row classifier (Top/Home/Bottom)
   runs first. Its output is multiplied into the key probabilities, which
   enforces row consistency and fixes the Mode 2 failure (predicting top-row
   keys for bottom-row inputs).

5. DUAL SPECIALIST A+B FUSION preserved (it's the part that worked).

6. LOSO EVALUATION built in — honest cross-user accuracy reporting.

7. PER-USER CALIBRATION — takes 5–10 labeled samples per key from a new
   user, computes a per-class logit bias, saves a small user_profile.

Usage
-----
    python train_v2.py                        # train on q11/q14/q16/q17, save model
    python train_v2.py --loso                 # also run leave-one-subject-out eval
    python train_v2.py --calibrate-on q25     # train + calibrate + test on q25
"""

import argparse
import glob
import os
import warnings
import joblib
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from scipy.signal       import savgol_filter, butter, filtfilt
from sklearn.preprocessing  import StandardScaler, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, LeaveOneGroupOut, train_test_split
from sklearn.metrics    import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

import lightgbm as lgb
from catboost import CatBoostClassifier

# Use the user's existing geometry module — UNCHANGED
from geometry import (
    midpoint, line_vector,
    compute_all_mcp_abduction_angles,
    compute_mcp_flexion_angles,
    compute_all_finger_segment_angles,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════
PATH_PATTERN        = "/Users/fateme/Desktop/newdata/**/*.csv"
VALID_PARTICIPANTS  = ["q11", "q14", "q16", "q17"]
WINDOW_MS           = 50
OFFSET_MS           = 0
QTM_FS              = 200.0
SG_WINDOW_LENGTH    = 15
SG_POLYORDER        = 3

# Marker dropout augmentation
DROPOUT_PROB        = 0.20      # fraction of samples that get a dropout
DROPOUT_MARKERS     = ["I4", "M2", "M4", "R4", "L4"]  # markers occluded in real data
DROPOUT_NAN_RANGE   = (0.30, 1.0)

# Marker quality gate
MAX_MARKER_NAN      = 0.30
CRITICAL_MARKERS    = ["I4", "M2"]

# Right and left key sets
RIGHT_FINGER_KEY_MAP = {
    "Index":  ["y", "u", "h", "j", "n", "m"],
    "Middle": ["i", "k"],
    "Ring":   ["o", "l"],
    "Little": ["p"],
}
LEFT_FINGER_KEY_MAP = {
    "Index":  ["r", "t", "f", "g", "v", "b"],
    "Middle": ["e", "d", "c"],
    "Ring":   ["w", "s", "x"],
    "Little": ["q", "a", "z"],
}
RIGHT_KEYS = [k for keys in RIGHT_FINGER_KEY_MAP.values()  for k in keys]
LEFT_KEYS  = [k for keys in LEFT_FINGER_KEY_MAP.values()   for k in keys]

# Row map (Top / Home / Bottom)
KEY_TO_ROW = {
    # Top row
    'q':'Top','w':'Top','e':'Top','r':'Top','t':'Top',
    'y':'Top','u':'Top','i':'Top','o':'Top','p':'Top',
    # Home row
    'a':'Home','s':'Home','d':'Home','f':'Home','g':'Home',
    'h':'Home','j':'Home','k':'Home','l':'Home',
    # Bottom row
    'z':'Bottom','x':'Bottom','c':'Bottom','v':'Bottom','b':'Bottom',
    'n':'Bottom','m':'Bottom',
}

# Build KEY_TO_FINGER from BOTH hand maps separately.
# DO NOT use {**RIGHT, **LEFT} — both dicts share finger names as keys
# (Index, Middle, Ring, Little), so left-hand entries overwrite right-hand ones,
# leaving every right key unmapped → Target_Finger=NaN → all right rows dropped.
KEY_TO_FINGER = {}
for hand_map in (RIGHT_FINGER_KEY_MAP, LEFT_FINGER_KEY_MAP):
    for finger, keys in hand_map.items():
        for k in keys:
            KEY_TO_FINGER[k] = finger

FINGERS = ["Index", "Middle", "Ring", "Little"]
ROWS    = ["Top", "Home", "Bottom"]

KINEMATIC_FEATURES = ["MCP_Flexion", "MCP_Abduction", "PIP_Flexion", "DIP_Flexion"]


# ═════════════════════════════════════════════════════════════════════════════
# GEOMETRY ADAPTER (handles lowercase axes + hand prefix)
# ═════════════════════════════════════════════════════════════════════════════
def load_marker(df, marker, hand_prefix):
    """
    Load a 3D marker from a CSV.
    For finger markers (I1, I2, M1, ...): column is "{hand_prefix}_{marker}_{x|y|z}"
    For wrist markers (Cin, Cout):        column is "QTMdc_{hand_prefix}_{marker}_{x|y|z}"
    """
    if marker in ("Cin", "Cout"):
        cols = [f"QTMdc_{hand_prefix}_{marker}_{ax}" for ax in ("x", "y", "z")]
    else:
        cols = [f"{hand_prefix}_{marker}_{ax}" for ax in ("x", "y", "z")]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        return np.full((len(df), 3), np.nan)
    return df[cols].replace([np.inf, -np.inf], np.nan).to_numpy()


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL FILTERS
# ═════════════════════════════════════════════════════════════════════════════
def butter_lowpass_filter(data, cutoff=15.0, fs=200.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def safe_savgol(arr, window=SG_WINDOW_LENGTH, poly=SG_POLYORDER):
    if len(arr) < window + 5:
        return arr
    mask = ~np.isnan(arr)
    if mask.sum() < window:
        return arr
    out = arr.copy()
    out[mask] = savgol_filter(arr[mask], window, poly)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# KINEMATICS — per-file processing
# ═════════════════════════════════════════════════════════════════════════════
def compute_velocity(df_angles, t):
    """np.gradient with timestamps — robust to slight fs jitter."""
    out = pd.DataFrame(index=df_angles.index)
    for col in df_angles.columns:
        a = df_angles[col].to_numpy()
        m = ~np.isnan(a)
        if m.sum() < 3:
            out[f"VEL_{col}"] = np.nan
            continue
        v_full = np.full_like(a, np.nan)
        v_full[m] = np.gradient(np.unwrap(np.radians(a[m])), t[m])
        out[f"VEL_{col}"] = np.degrees(safe_savgol(v_full))
    return out


def compute_higher_order(df_in, t, prefix_in, prefix_out):
    """Compute next derivative (acceleration from velocity, jerk from accel)."""
    out = pd.DataFrame(index=df_in.index)
    for col in df_in.columns:
        a = df_in[col].to_numpy()
        m = ~np.isnan(a)
        new_name = col.replace(prefix_in, prefix_out)
        if m.sum() < 3:
            out[new_name] = np.nan
            continue
        d_full = np.full_like(a, np.nan)
        d_full[m] = np.gradient(a[m], t[m])
        out[new_name] = safe_savgol(d_full)
    return out


def kinematics_for_file(fpath, hand_prefix, target_keys):
    """
    Process one CSV file: load, filter, compute angles + velocities,
    extract pre-keystroke windows.

    Returns a list of segment DataFrames (one per keystroke), or [].
    """
    try:
        df = pd.read_csv(fpath)
        df.columns = df.columns.str.strip()
    except Exception:
        return []

    # Normalise core column names
    df = df.rename(columns={
        "time": "TimeStamp",
        "currentLetter": "Pressed_Letter",
        "keyPressFlags": "KeyPressFlag",
    })

    # Rename the wrist columns to QTMdc_*_x/y/z to match load_marker
    qtm_renames = {}
    for marker in ("Cin", "Cout"):
        for ax in ("x", "y", "z"):
            old = f"{hand_prefix}_{marker}_{ax}"
            new = f"QTMdc_{hand_prefix}_{marker}_{ax}"
            if old in df.columns:
                qtm_renames[old] = new
    if qtm_renames:
        df = df.rename(columns=qtm_renames)

    if "TimeStamp" not in df.columns:
        return []

    df = df.sort_values("TimeStamp").drop_duplicates("TimeStamp").reset_index(drop=True)
    if len(df) < SG_WINDOW_LENGTH + 5:
        return []

    df["Pressed_Letter"] = df["Pressed_Letter"].astype(str).str.lower()
    df["Filepath"]       = fpath
    df = df[df["Pressed_Letter"].isin(target_keys)].copy()
    if df.empty:
        return []

    # Collect all marker column names we'll need
    marker_names = ["I1","I2","I3","I4","M1","M2","M3","M4",
                    "R1","R2","R3","R4","L1","L2","L3","L4","Cin","Cout"]
    all_cols = []
    for m in marker_names:
        if m in ("Cin","Cout"):
            for ax in ("x","y","z"):
                all_cols.append(f"QTMdc_{hand_prefix}_{m}_{ax}")
        else:
            for ax in ("x","y","z"):
                all_cols.append(f"{hand_prefix}_{m}_{ax}")

    if any(c not in df.columns for c in all_cols):
        return []

    # ── Marker quality gate ──────────────────────────────────────────────
    for m in CRITICAL_MARKERS:
        col = f"{hand_prefix}_{m}_x"
        if col in df.columns and df[col].isna().mean() > MAX_MARKER_NAN:
            return []

    # Interpolate, then Butterworth low-pass filter
    df[all_cols] = df[all_cols].replace([np.inf, -np.inf], np.nan).interpolate(
        limit_direction="both"
    )
    try:
        df[all_cols] = butter_lowpass_filter(
            df[all_cols].values, cutoff=15.0, fs=QTM_FS, order=4
        )
    except Exception:
        pass

    # Load marker arrays
    coords = {m: load_marker(df, m, hand_prefix) for m in marker_names}
    Cin, Cout = coords["Cin"], coords["Cout"]

    # ── Angles via existing geometry.py functions ────────────────────────
    hand_side = "R" if hand_prefix == "R" else "L"
    mcp_abd  = compute_all_mcp_abduction_angles(
        coords["L1"], coords["R1"], coords["M1"], coords["I1"],
        coords["L4"], coords["R4"], coords["M4"], coords["I4"],
        Cin, Cout, hand_side=hand_side,
    )
    mcp_flex = compute_mcp_flexion_angles(
        coords["L1"], coords["L2"],
        coords["R1"], coords["R2"],
        coords["M1"], coords["M2"],
        coords["I1"], coords["I2"],
        Cin, Cout,
    )
    seg = compute_all_finger_segment_angles(
        coords["L1"], coords["L2"], coords["L3"], coords["L4"],
        coords["R1"], coords["R2"], coords["R3"], coords["R4"],
        coords["M1"], coords["M2"], coords["M3"], coords["M4"],
        coords["I1"], coords["I2"], coords["I3"], coords["I4"],
    )

    angles = pd.DataFrame(index=df.index)
    for f in FINGERS:
        angles[f"{f}_MCP_Abduction"] = np.degrees(mcp_abd[f])
        angles[f"{f}_MCP_Flexion"]   = np.degrees(mcp_flex[f])
        angles[f"{f}_PIP_Flexion"]   = np.degrees(seg[f]["PIP_angle"])
        angles[f"{f}_DIP_Flexion"]   = np.degrees(seg[f]["DIP_angle"])
    angles = angles.clip(-180, 180).apply(safe_savgol)

    # ── Wrist-relative coordinates (THE KEY NEW FEATURE) ─────────────────
    wrist_mid = (Cin + Cout) / 2.0
    rel_coords = {}
    for f, tip_marker in [("Index","I4"), ("Middle","M4"), ("Ring","R4"), ("Little","L4")]:
        rel = coords[tip_marker] - wrist_mid
        rel_coords[f"REL_{f}_TIP_X"] = rel[:, 0]
        rel_coords[f"REL_{f}_TIP_Y"] = rel[:, 1]
        rel_coords[f"REL_{f}_TIP_Z"] = rel[:, 2]
    for f, mcp_marker in [("Index","I1"), ("Middle","M1"), ("Ring","R1"), ("Little","L1")]:
        rel = coords[mcp_marker] - wrist_mid
        rel_coords[f"REL_{f}_MCP_X"] = rel[:, 0]
        rel_coords[f"REL_{f}_MCP_Y"] = rel[:, 1]
        rel_coords[f"REL_{f}_MCP_Z"] = rel[:, 2]
    rel_df = pd.DataFrame(rel_coords, index=df.index)

    # Velocity, acceleration, jerk
    t = df["TimeStamp"].to_numpy()
    vel   = compute_velocity(angles, t)
    accel = compute_higher_order(vel, t, "VEL_", "ACC_")
    jerk  = compute_higher_order(accel, t, "ACC_", "JRK_")

    # Combine
    full = pd.concat([
        df[["TimeStamp", "Pressed_Letter", "KeyPressFlag", "Filepath"]],
        angles, vel, accel, jerk, rel_df,
    ], axis=1)

    # ── Extract pre-keystroke windows around each keypress event ─────────
    df["Contact_Flag"] = (
        (df["KeyPressFlag"].shift(-1) == 1) & (df["KeyPressFlag"] == 0)
    )
    contact_events = df[df["Contact_Flag"]]

    segments = []
    for _, ev in contact_events.iterrows():
        ct = ev["TimeStamp"]
        seg = full[
            (full["TimeStamp"] >= ct - WINDOW_MS/1000.0) &
            (full["TimeStamp"] <= ct - OFFSET_MS/1000.0)
        ].copy()
        if seg.empty or len(seg) < 3:
            continue
        seg["Keystroke_ID"] = f"{ct:.6f}_{ev['Pressed_Letter']}"
        seg["Pressed_Letter"] = ev["Pressed_Letter"]
        segments.append(seg)

    return segments


def process_file_both_hands(fpath):
    """Run kinematics for both hands, return (right_segments, left_segments)."""
    r = kinematics_for_file(fpath, "R", RIGHT_KEYS)
    l = kinematics_for_file(fpath, "L", LEFT_KEYS)
    return r, l


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE AGGREGATION
# ═════════════════════════════════════════════════════════════════════════════
def aggregate_features(seg_df):
    """
    Take a long-format DataFrame (rows = frames, columns = signals),
    group by Keystroke_ID, return a wide-format DataFrame
    (rows = keystrokes, columns = features).
    """
    if seg_df is None or seg_df.empty:
        return pd.DataFrame(), []

    grouped = seg_df.groupby("Keystroke_ID")
    feats = {}

    feats["Duration_ms"] = (grouped["TimeStamp"].max() -
                            grouped["TimeStamp"].min()) * 1000

    # All angle / velocity / accel / jerk columns
    signal_prefixes = ["", "VEL_", "ACC_", "JRK_"]
    for prefix in signal_prefixes:
        for f in FINGERS:
            for k in KINEMATIC_FEATURES:
                col = f"{prefix}{f}_{k}"
                if col not in seg_df.columns:
                    continue
                feats[f"{col}_Mean"] = grouped[col].mean()
                feats[f"{col}_Std"]  = grouped[col].std()
                feats[f"{col}_Peak"] = grouped[col].apply(
                    lambda x: x.abs().max() if x.notna().any() else np.nan
                )
                feats[f"{col}_RMS"]  = grouped[col].apply(
                    lambda x: np.sqrt(np.nanmean(x**2)) if x.notna().any() else np.nan
                )

    # Wrist-relative coordinates — these are THE row-detection features
    rel_cols = [c for c in seg_df.columns if c.startswith("REL_")]
    for col in rel_cols:
        feats[f"{col}_Mean"] = grouped[col].mean()
        feats[f"{col}_Std"]  = grouped[col].std()
        feats[f"{col}_Min"]  = grouped[col].min()
        feats[f"{col}_Max"]  = grouped[col].max()

    # Inter-finger ratios (useful for distinguishing within-finger keys)
    for f1 in FINGERS:
        for f2 in FINGERS:
            if f1 >= f2:
                continue
            for sig in ["MCP_Flexion", "PIP_Flexion"]:
                c1 = f"{f1}_{sig}"
                c2 = f"{f2}_{sig}"
                if c1 in seg_df.columns and c2 in seg_df.columns:
                    diff = grouped[c1].mean() - grouped[c2].mean()
                    feats[f"DIFF_{f1}_{f2}_{sig}_Mean"] = diff

    df_agg = pd.DataFrame(feats)

    # Target columns
    df_agg["Pressed_Letter"] = grouped["Pressed_Letter"].first().loc[df_agg.index]
    df_agg["Filepath"]       = grouped["Filepath"].first().loc[df_agg.index]
    df_agg["Target_Finger"]  = df_agg["Pressed_Letter"].map(KEY_TO_FINGER)
    df_agg["Target_Row"]     = df_agg["Pressed_Letter"].map(KEY_TO_ROW)

    df_agg = df_agg.dropna(subset=["Pressed_Letter", "Target_Row", "Target_Finger"])
    df_agg = df_agg.reset_index(drop=True)

    feature_cols = [c for c in df_agg.columns if c not in
                    {"Pressed_Letter", "Filepath", "Target_Finger", "Target_Row"}]

    return df_agg, feature_cols


# ═════════════════════════════════════════════════════════════════════════════
# POSTURE NORMALIZATION (per-user Y/Z baseline subtraction)
# ═════════════════════════════════════════════════════════════════════════════
# In air-typing, each user has their own resting hand height (Y) and depth (Z)
# without a physical keyboard to anchor. Training users have means around
# Y=70mm but q25's left hand is at 68mm — this 8mm shift is enough to make
# the row classifier collapse. Subject-level normalization fixes this:
#   1. At training: subtract each user's per-hand mean Y/Z BEFORE the global scaler
#   2. At calibration: compute new user's per-hand mean Y/Z from calibration phrase
#   3. At inference: subtract user's saved baseline before scaling
# Result: model learns posture-invariant features; new users get normalized to
# the same space.

POSTURE_AXIS_SUFFIXES = ("_Y_Mean", "_Y_Std", "_Y_Min", "_Y_Max",
                          "_Z_Mean", "_Z_Std", "_Z_Min", "_Z_Max")


def get_posture_feature_cols(feature_cols):
    """Return list of feature columns that should be posture-normalized.
    These are the wrist-relative Y and Z aggregations (mean/std/min/max)."""
    return [c for c in feature_cols
            if c.startswith("REL_")
            and any(c.endswith(s) for s in POSTURE_AXIS_SUFFIXES)]


def compute_posture_baseline(df_features, feature_cols):
    """Compute per-feature mean over a set of keystrokes.
    df_features: DataFrame of one user's keystrokes (one hand).
    Returns: dict {column_name: mean_value} for posture-relevant columns only.
    """
    posture_cols = get_posture_feature_cols(feature_cols)
    baseline = {}
    for col in posture_cols:
        if col in df_features.columns:
            baseline[col] = float(df_features[col].mean())
    return baseline


def apply_posture_baseline(df_features, baseline):
    """Subtract a saved posture baseline from a DataFrame's features.
    Modifies a COPY of df_features (does not mutate input).
    Only subtracts from the columns present in the baseline dict.
    """
    df_out = df_features.copy()
    for col, mean_val in baseline.items():
        if col in df_out.columns:
            df_out[col] = df_out[col] - mean_val
    return df_out


def normalize_per_user(df_features, feature_cols):
    """At training time: subtract each user's posture baseline from their own
    samples. Returns (normalized_df, dict_of_user_baselines).

    Uses 'Filepath' column to identify users via participant_id().
    """
    df_out = df_features.copy()
    df_out["pid"] = df_out["Filepath"].apply(participant_id)

    user_baselines = {}
    for pid in df_out["pid"].unique():
        if pid is None:
            continue
        user_mask = df_out["pid"] == pid
        user_baseline = compute_posture_baseline(df_out[user_mask], feature_cols)
        user_baselines[pid] = user_baseline
        # Apply to this user's rows
        for col, m in user_baseline.items():
            if col in df_out.columns:
                df_out.loc[user_mask, col] = df_out.loc[user_mask, col] - m

    df_out = df_out.drop(columns=["pid"])
    return df_out, user_baselines


# ═════════════════════════════════════════════════════════════════════════════
# PER-USER CLASS CENTROIDS (Strategy C: deeper calibration)
# ═════════════════════════════════════════════════════════════════════════════
# Strategy C: each user has a "personal keyboard" — their imagined position of
# each key in feature space differs from training users. We learn each user's
# per-class centroid from their calibration phrase, and use distance-to-centroid
# to correct the base model's predictions.
#
# Centroids are computed in the model's own discriminative subspace (top
# importance features from key_model_A) AFTER the saved scaler is applied.
# At inference: final_proba = base_proba * exp(-alpha * dist_to_centroid)


def get_centroid_feature_indices(expert, n_features=30):
    """Returns the indices of the top-N most-important features the
    key_model_A uses, intersected with the global feature space.

    Reuses the model's learned feature importance — these are the features
    that discriminate between keys. Computing centroids in this subspace
    gives us a low-dim, model-aligned space for distance calculations.
    """
    key_model = expert["key_model_A"]
    idx_A     = expert["key_feature_indices_A"]   # mapping into global feature space

    # CatBoost feature importance — gives importance for the model_A's input features
    try:
        importance = key_model.get_feature_importance()
    except Exception:
        # Fallback for non-CatBoost models
        if hasattr(key_model, "feature_importances_"):
            importance = key_model.feature_importances_
        else:
            # No importance available — use all of model A's features
            return np.array(list(idx_A))

    # Take the top-N most important features within model A
    n_use = min(int(n_features), len(importance))
    top_within_A = np.argsort(importance)[-n_use:]
    # Map back to global feature-space indices
    centroid_global_idx = np.array(idx_A)[top_within_A]
    return np.sort(centroid_global_idx)


def compute_class_centroids(X_scaled, y, classes, centroid_idx,
                             min_samples_per_class=3):
    """Compute per-class mean in the centroid feature subspace.

    Args:
        X_scaled:   (n, d) already-scaled feature matrix
        y:          (n,) labels (strings)
        classes:    list of class strings (from key_encoder.classes_)
        centroid_idx: feature indices to use for centroids
        min_samples_per_class: skip centroids for classes with fewer samples

    Returns:
        Tuple of (centroids, sample_counts):
          centroids: dict {class_name: ndarray (len(centroid_idx),)}
          sample_counts: dict {class_name: int} — calibration samples used
        Classes with too few samples are skipped in BOTH dicts.
    """
    centroids = {}
    sample_counts = {}
    X_sub = X_scaled[:, centroid_idx]
    for c in classes:
        mask = (y == c)
        n = int(mask.sum())
        if n < min_samples_per_class:
            continue
        centroids[str(c)]      = X_sub[mask].mean(axis=0)
        sample_counts[str(c)]  = n
    return centroids, sample_counts


def centroid_log_scores(X_scaled, expert, centroids, centroid_idx,
                         alpha=1.0, sample_counts=None, n_full=20):
    """For each sample, compute log-score for each class based on (negative)
    squared Euclidean distance to that class's centroid.

    Per-class alpha damping: when sample_counts is provided, classes with
    fewer calibration samples get a weaker centroid pull, because their
    centroid is noisier. Formula:
        alpha[K] = alpha * sqrt(min(n[K], n_full) / n_full)
    This caps alpha at the base value once a class has n_full+ samples,
    and dampens it for rare classes (q, j, x, z).

    Args:
        X_scaled:      (n, d) scaled features
        expert:        trained expert (for class ordering)
        centroids:     dict {class_name: centroid_vector}
        centroid_idx:  feature indices used to build centroids
        alpha:         BASE scaling on the distance score
        sample_counts: dict {class_name: n_calibration_samples} OR None
                       If None, uniform alpha is used for all classes.
        n_full:        sample count above which alpha is at full strength.
                       Default 20 — below this, alpha is dampened.

    Returns:
        log_scores: (n, n_classes), neutral 0 for classes without centroids
    """
    classes = expert["key_encoder"].classes_
    X_sub = X_scaled[:, centroid_idx]
    n_samples = X_sub.shape[0]
    n_classes = len(classes)

    log_scores = np.zeros((n_samples, n_classes))
    for j, c in enumerate(classes):
        c_str = str(c)
        if c_str not in centroids:
            continue
        diff = X_sub - centroids[c_str][None, :]
        sq_dist = np.sum(diff ** 2, axis=1)

        # Per-class alpha damping based on calibration sample count
        if sample_counts is not None and c_str in sample_counts:
            n_c = sample_counts[c_str]
            alpha_c = alpha * np.sqrt(min(n_c, n_full) / float(n_full))
        else:
            alpha_c = alpha

        log_scores[:, j] = -alpha_c * sq_dist

    return log_scores


# ═════════════════════════════════════════════════════════════════════════════
# MARKER DROPOUT AUGMENTATION
# ═════════════════════════════════════════════════════════════════════════════
def augment_dropout(df_full, hand_prefix, dropout_prob=DROPOUT_PROB,
                    nan_range=DROPOUT_NAN_RANGE, rng=None):
    """
    Randomly mask out marker columns on a fraction of keystrokes.
    Operates BEFORE feature aggregation (on the long-format DataFrame).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    df_aug = df_full.copy()
    keystroke_ids = df_aug["Keystroke_ID"].unique()

    n_dropped = 0
    for kid in keystroke_ids:
        if rng.random() > dropout_prob:
            continue
        mask        = df_aug["Keystroke_ID"] == kid
        marker      = rng.choice(DROPOUT_MARKERS)
        nan_pct     = rng.uniform(*nan_range)
        idx         = df_aug.index[mask]
        n_to_nan    = max(1, int(nan_pct * len(idx)))
        nan_idx     = rng.choice(idx, n_to_nan, replace=False)

        # Affect both raw coords (REL_*) and angle features that depend on this marker
        cols_to_break = [c for c in df_aug.columns
                         if (f"_{marker}_" in c) or (marker in c.split("_")[-1:])]
        # Also break related angles for the corresponding finger
        finger_for_marker = {"I4":"Index", "I1":"Index", "I2":"Index", "I3":"Index",
                             "M1":"Middle","M2":"Middle","M3":"Middle","M4":"Middle",
                             "R1":"Ring","R2":"Ring","R3":"Ring","R4":"Ring",
                             "L1":"Little","L2":"Little","L3":"Little","L4":"Little"}
        f = finger_for_marker.get(marker)
        if f:
            cols_to_break += [c for c in df_aug.columns
                              if c.startswith(f"REL_{f}_") or c.startswith(f"{f}_")]
        cols_to_break = list(set(cols_to_break) & set(df_aug.columns))

        df_aug.loc[nan_idx, cols_to_break] = np.nan
        n_dropped += 1

    print(f"   🔥 Dropout augmentation: {n_dropped}/{len(keystroke_ids)} keystrokes affected")
    return df_aug


# ═════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING — per hand
# ═════════════════════════════════════════════════════════════════════════════
def tune_catboost(X, y, sample_weight=None, n_iter=15, cv=3):
    cat = CatBoostClassifier(
        loss_function="MultiClass",
        random_seed=42,
        logging_level="Silent",
        allow_writing_files=False,
    )
    grid = {
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "l2_leaf_reg":   [3, 5, 10],
        "iterations":    [600, 1000],
    }
    if sample_weight is None:
        sample_weight = compute_sample_weight("balanced", y)
    search = RandomizedSearchCV(
        cat, grid, n_iter=n_iter, scoring="f1_weighted",
        cv=cv, n_jobs=-1, random_state=42, verbose=0,
    )
    search.fit(X, y, sample_weight=sample_weight)
    return search.best_estimator_


def select_features(X_tr, y_tr, n_features, seed=42):
    """Pick top-N features by LightGBM importance — fast and effective."""
    n_features = min(n_features, X_tr.shape[1])
    pre = lgb.LGBMClassifier(n_estimators=200, max_depth=6,
                             verbose=-1, random_state=seed)
    pre.fit(X_tr, y_tr)
    importance = pre.feature_importances_
    top_idx = np.argsort(importance)[-n_features:]
    return np.sort(top_idx)


def train_hand(df_features, feature_cols, hand_name, augmented=False,
               posture_norm=True):
    """
    Train the full hierarchy for one hand:
      - Finger classifier
      - Row classifier
      - Dual-specialist key classifiers (A on velocity-heavy features,
                                        B on coordinate-heavy features)
    posture_norm: if True, subtract each user's posture baseline before
                  scaling. The saved expert will record this and the
                  per-user baselines (used for diagnostics + sanity).
    """
    print(f"\n{'='*60}")
    tag = []
    if augmented: tag.append("AUGMENTED")
    if posture_norm: tag.append("POSTURE-NORM")
    if not tag: tag = ["CLEAN"]
    print(f"🏋️  TRAINING {hand_name.upper()} HAND  ({' + '.join(tag)})")
    print(f"{'='*60}")
    print(f"   Samples: {len(df_features)}, features: {len(feature_cols)}")

    # Posture normalization: subtract each user's per-hand baseline
    user_baselines = {}
    if posture_norm:
        df_features, user_baselines = normalize_per_user(df_features, feature_cols)
        n_posture_cols = len(get_posture_feature_cols(feature_cols))
        print(f"   Posture-normalized {n_posture_cols} REL_*_Y/Z features per user "
              f"({len(user_baselines)} users)")

    X = df_features[feature_cols].fillna(0).values.astype(float)
    y_key    = df_features["Pressed_Letter"].values
    y_finger = df_features["Target_Finger"].values
    y_row    = df_features["Target_Row"].values

    # Fit ONE scaler on the entire training set — saved for deployment
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encoders
    key_enc    = LabelEncoder().fit(y_key)
    finger_enc = LabelEncoder().fit(y_finger)
    row_enc    = LabelEncoder().fit(y_row)

    y_key_enc    = key_enc.transform(y_key)
    y_finger_enc = finger_enc.transform(y_finger)
    y_row_enc    = row_enc.transform(y_row)

    # Train/test split for internal validation
    X_tr, X_te, yk_tr, yk_te, yf_tr, yf_te, yr_tr, yr_te = train_test_split(
        X_scaled, y_key_enc, y_finger_enc, y_row_enc,
        test_size=0.15, stratify=y_key_enc, random_state=42,
    )

    # ── Stage 1: Finger classifier ──────────────────────────────────────
    print("\n   [Stage 1] Finger classifier...")
    finger_idx = select_features(X_tr, yf_tr, n_features=35)
    finger_model = lgb.LGBMClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    finger_model.fit(X_tr[:, finger_idx], yf_tr)
    yf_pred = finger_model.predict(X_te[:, finger_idx])
    print(f"     Finger acc: {accuracy_score(yf_te, yf_pred):.2%}")

    # ── Stage 2: Row classifier ──────────────────────────────────────────
    # KEY FIX: This stage explicitly uses wrist-relative features to
    # prevent the row-detection failure seen in q25.
    print("\n   [Stage 2] Row classifier (NEW - fixes Mode 2 failure)...")
    rel_feature_idx = [i for i, c in enumerate(feature_cols)
                       if c.startswith("REL_") and "Y" in c]
    angle_idx_y     = [i for i, c in enumerate(feature_cols)
                       if "_MCP_Flexion_Mean" in c]
    row_pool_idx    = sorted(set(rel_feature_idx + angle_idx_y))
    if len(row_pool_idx) < 5:
        # Fallback if filter is too strict
        row_pool_idx = list(range(X_scaled.shape[1]))

    row_idx_local = select_features(
        X_tr[:, row_pool_idx], yr_tr, n_features=min(25, len(row_pool_idx))
    )
    row_idx = np.array(row_pool_idx)[row_idx_local]

    row_model = lgb.LGBMClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    row_model.fit(X_tr[:, row_idx], yr_tr)
    yr_pred = row_model.predict(X_te[:, row_idx])
    print(f"     Row acc: {accuracy_score(yr_te, yr_pred):.2%}")
    print(f"     Confusion: {dict(zip(*np.unique(yr_pred, return_counts=True)))}")

    # ── Stage 3: Dual-specialist key classifiers ─────────────────────────
    print("\n   [Stage 3] Dual-specialist key classifiers...")

    # Specialist A: velocity / dynamics features
    vel_idx_pool = [i for i, c in enumerate(feature_cols)
                    if "VEL_" in c or "ACC_" in c or "JRK_" in c]
    if len(vel_idx_pool) < 30:
        vel_idx_pool = list(range(X_scaled.shape[1]))
    idx_A_local = select_features(X_tr[:, vel_idx_pool], yk_tr, n_features=30)
    idx_A = np.array(vel_idx_pool)[idx_A_local]

    sw_tr = compute_sample_weight("balanced", yk_tr)
    print("     - Tuning Specialist A (dynamics)...")
    key_model_A = tune_catboost(X_tr[:, idx_A], yk_tr, sw_tr, n_iter=10)

    # Specialist B: coordinate / position features
    coord_idx_pool = [i for i, c in enumerate(feature_cols)
                      if c.startswith("REL_") or "_MCP_Flexion_Mean" in c
                      or "_PIP_Flexion_Mean" in c or "_DIP_Flexion_Mean" in c]
    if len(coord_idx_pool) < 30:
        coord_idx_pool = list(range(X_scaled.shape[1]))
    idx_B_local = select_features(X_tr[:, coord_idx_pool], yk_tr, n_features=30)
    idx_B = np.array(coord_idx_pool)[idx_B_local]
    print("     - Tuning Specialist B (positions)...")
    key_model_B = tune_catboost(X_tr[:, idx_B], yk_tr, sw_tr, n_iter=10)

    # ── Assemble and evaluate fusion ─────────────────────────────────────
    expert = {
        "scaler":                  scaler,                    # 🔥 SAVED
        "feature_cols":            feature_cols,
        "posture_norm":            posture_norm,               # NEW
        "training_user_baselines": user_baselines,             # NEW (diagnostic)
        "finger_model":            finger_model,
        "finger_feature_indices":  finger_idx,
        "finger_encoder":          finger_enc,
        "row_model":               row_model,                  # NEW
        "row_feature_indices":     row_idx,                    # NEW
        "row_encoder":             row_enc,                    # NEW
        "key_model_A":             key_model_A,
        "key_feature_indices_A":   idx_A,
        "key_model_B":             key_model_B,
        "key_feature_indices_B":   idx_B,
        "key_encoder":             key_enc,
        "fusion_alpha":            1.5,
        "row_weight":               0.7,                       # how much row gates key
        "hand":                    hand_name,
    }

    fused_pred, _ = predict_with_full_fusion(X_te, expert)
    fused_correct = (fused_pred == key_enc.inverse_transform(yk_te))
    print(f"\n   🏆 {hand_name} FUSED Top-1: {fused_correct.mean():.2%}")

    return expert


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE — full hierarchical fusion
# ═════════════════════════════════════════════════════════════════════════════
def _safe_normalize(p, axis=1):
    s = p.sum(axis=axis, keepdims=True)
    return np.divide(p, s, out=np.zeros_like(p), where=s != 0)


def predict_for_user(df_features, expert, posture_baseline=None,
                     logit_bias=None, class_centroids=None,
                     centroid_idx=None, centroid_alpha=1.0,
                     centroid_counts=None):
    """
    Convenience wrapper that handles the full inference pipeline:
      1. Subtract user's posture baseline (if expert was trained with posture_norm)
      2. Apply expert's scaler
      3. Run dual-specialist + finger + row fusion -> base_proba
      4. (NEW) Apply per-user class centroids: final = base * exp(-α·dist²)
      5. Optionally apply logit bias from calibration

    Args:
        df_features: DataFrame with feature columns
        expert: trained expert dict (right or left)
        posture_baseline: dict of column → mean from calibration phrase.
                          REQUIRED if expert["posture_norm"] is True.
        logit_bias: ndarray (n_classes,) from calibrate_user
        class_centroids: dict {class_name: centroid_vector} from calibration.
                         If provided, applies Strategy C centroid correction.
        centroid_idx: feature indices used when centroids were computed
                      (must match — saved in user profile)
        centroid_alpha: scaling on centroid distance score (higher = stronger pull
                        toward the user's calibration centroids)

    Returns: (predictions, fused_proba, classes)
    """
    feat_cols = expert["feature_cols"]
    df_in = df_features.copy()

    # Make sure all expected feature columns exist
    for c in feat_cols:
        if c not in df_in.columns:
            df_in[c] = 0.0

    # Step 1: Posture normalization (if model was trained with it)
    if expert.get("posture_norm", False):
        if posture_baseline is None:
            raise ValueError(
                "Model was trained with posture_norm=True but no "
                "posture_baseline was provided. Compute it from a "
                "calibration phrase using compute_posture_baseline()."
            )
        df_in = apply_posture_baseline(df_in, posture_baseline)

    # Step 2: Build feature matrix and apply saved scaler
    X = df_in[feat_cols].fillna(0).values.astype(float)
    X_scaled = expert["scaler"].transform(X)

    # Step 3: Run base fusion (without logit bias yet — we apply it after centroids)
    _, base_proba = predict_with_full_fusion(X_scaled, expert, logit_bias=None)

    # Step 4: Apply per-user class centroids (Strategy C)
    eps = 1e-9
    if class_centroids is not None and centroid_idx is not None and len(class_centroids) > 0:
        log_scores = centroid_log_scores(X_scaled, expert, class_centroids,
                                          centroid_idx, alpha=centroid_alpha,
                                          sample_counts=centroid_counts)
        # Combine: log(final) = log(base) + log_scores
        log_combined = np.log(base_proba + eps) + log_scores
        # Stabilize and softmax
        log_combined -= log_combined.max(axis=1, keepdims=True)
        proba = np.exp(log_combined)
        proba /= proba.sum(axis=1, keepdims=True) + eps
    else:
        proba = base_proba

    # Step 5: Apply logit bias if provided
    if logit_bias is not None:
        log_p = np.log(proba + eps) + logit_bias[None, :]
        log_p -= log_p.max(axis=1, keepdims=True)
        proba = np.exp(log_p)
        proba /= proba.sum(axis=1, keepdims=True) + eps

    classes = expert["key_encoder"].classes_
    preds = classes[np.argmax(proba, axis=1)]
    return preds, proba, classes


def predict_with_full_fusion(X_scaled, expert, logit_bias=None):
    """
    X_scaled is already standardised — caller should:
      1. Apply posture baseline subtraction first (if expert["posture_norm"])
      2. Then call expert["scaler"].transform()
      3. Then call this function

    Returns (predicted_keys, fused_proba).
    """
    eps = 1e-9
    key_enc    = expert["key_encoder"]
    finger_enc = expert["finger_encoder"]
    row_enc    = expert["row_encoder"]

    # Stage 1: Finger
    f_proba = expert["finger_model"].predict_proba(X_scaled[:, expert["finger_feature_indices"]])
    # Stage 2: Row
    r_proba = expert["row_model"].predict_proba(X_scaled[:, expert["row_feature_indices"]])
    # Stage 3: Key (dual specialist)
    proba_A = expert["key_model_A"].predict_proba(X_scaled[:, expert["key_feature_indices_A"]])
    proba_B = expert["key_model_B"].predict_proba(X_scaled[:, expert["key_feature_indices_B"]])
    alpha   = expert.get("fusion_alpha", 1.5)
    key_proba = _safe_normalize((proba_A + eps) ** alpha *
                                 (proba_B + eps) ** (1.0 / alpha))

    # Build per-key gates from finger and row predictions
    rw = expert.get("row_weight", 0.7)
    fused = np.zeros_like(key_proba)
    for i, key_class in enumerate(key_enc.classes_):
        kc          = str(key_class).lower()
        finger_name = KEY_TO_FINGER.get(kc, "Index")
        row_name    = KEY_TO_ROW.get(kc, "Home")

        f_idx = np.where(finger_enc.classes_ == finger_name)[0]
        r_idx = np.where(row_enc.classes_    == row_name)[0]
        if f_idx.size == 0 or r_idx.size == 0:
            continue

        f_score = f_proba[:, f_idx[0]]
        r_score = r_proba[:, r_idx[0]]
        # Soft gate: row_weight controls how strict the row constraint is
        gate = (r_score ** rw) * f_score
        fused[:, i] = key_proba[:, i] * gate

    # Optional calibration bias (log-space)
    if logit_bias is not None:
        with np.errstate(divide='ignore'):
            log_p = np.log(fused + eps) + logit_bias[None, :]
        fused = np.exp(log_p - log_p.max(axis=1, keepdims=True))

    fused = _safe_normalize(fused)
    preds = key_enc.inverse_transform(np.argmax(fused, axis=1))
    return preds, fused


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
def participant_id(fpath):
    name = os.path.basename(fpath).lower()
    for pid in VALID_PARTICIPANTS + ["q10", "q25"]:
        if f"_{pid}_" in name:
            return pid
    return None


def load_all_data(path_pattern, valid_pids):
    """
    Load and process all CSVs in parallel for both hands.
    Returns (right_df, left_df) — long-format DataFrames with all keystroke segments.
    """
    files = [f for f in glob.glob(path_pattern, recursive=True)
             if participant_id(f) in valid_pids]
    print(f"📂 Found {len(files)} CSV file(s) for {valid_pids}")

    if not files:
        raise RuntimeError("No files found")

    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(process_file_both_hands, files))

    right_segments, left_segments = [], []
    for r, l in results:
        right_segments.extend(r)
        left_segments.extend(l)

    right_df = pd.concat(right_segments, ignore_index=True) if right_segments else pd.DataFrame()
    left_df  = pd.concat(left_segments,  ignore_index=True) if left_segments  else pd.DataFrame()
    print(f"   Right keystrokes: {right_df['Keystroke_ID'].nunique() if not right_df.empty else 0}")
    print(f"   Left keystrokes:  {left_df['Keystroke_ID'].nunique() if not left_df.empty else 0}")
    return right_df, left_df


# ═════════════════════════════════════════════════════════════════════════════
# CALIBRATION (per-user logit bias)
# ═════════════════════════════════════════════════════════════════════════════
def calibrate_user(expert, df_calib, y_calib, reg_scale=1.0, max_iter=300):
    """
    PER-USER CALIBRATION: computes BOTH a posture baseline (for air-typing
    posture differences) AND a vector-scaling logit bias (for fine within-finger
    confusion). Both go into the user profile.

    Args:
        df_calib: DataFrame with feature columns (not just an array — we need
                  column names to compute posture baseline).
                  Backwards compat: if you pass a numpy array, posture
                  normalization is skipped (returns posture_baseline=None).
        y_calib:  true labels
        reg_scale: regularization strength for logit bias
        max_iter:  optimizer iterations

    Returns:
        dict with keys:
          'logit_bias':       ndarray (n_classes,)
          'posture_baseline': dict {col: mean} or None
          'reg_scale':        scalar (saved for reproducibility)

    VECTOR-SCALING CALIBRATION with per-class adaptive regularization.

    Designed for the ONLINE DEPLOYMENT scenario where each user types a known
    calibration phrase (5-10 min) at session start. With realistic typing,
    common letters (e, t, a) get 50-150 calibration samples while rare letters
    (q, j, x, z) get only 1-5 samples. Uniform L2 regularization is wrong here:
    it's either too weak for rare classes (overfit) or too strong for common
    classes (underfit). This version scales regularization PER CLASS by
    1/sqrt(n_samples_in_that_class), so well-supported classes can have
    aggressive bias while rare classes stay conservative.

    Args:
        expert:    trained hand expert dict
        X_calib:   raw (unscaled) calibration features (n, d)
        y_calib:   true labels (n,) of strings matching key_encoder.classes_
        reg_scale: GLOBAL regularization strength multiplier (1.0 = balanced)
                   Lower it (0.3) to be more aggressive overall.
                   Raise it (3.0) to be more conservative overall.
                   Per-class regularization is automatically scaled by support.
        max_iter:  optimizer iterations

    Returns:
        bias: ndarray (n_classes,) — add to log-proba at inference time
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    key_enc = expert["key_encoder"]
    classes = key_enc.classes_
    n_classes = len(classes)
    eps = 1e-9

    # ── Posture baseline ────────────────────────────────────────────────
    # If df_calib is a DataFrame and the model uses posture_norm, compute it.
    posture_baseline = None
    if isinstance(df_calib, pd.DataFrame) and expert.get("posture_norm", False):
        posture_baseline = compute_posture_baseline(df_calib, expert["feature_cols"])
        print(f"    Posture baseline computed: {len(posture_baseline)} features, "
              f"Y-mean shift = {np.mean([v for k, v in posture_baseline.items() if '_Y_Mean' in k]):+.2f}mm")

    # ── Build the SCALED calibration matrix (used for both centroids and bias)
    if isinstance(df_calib, pd.DataFrame):
        df_in = df_calib.copy()
        for c in expert["feature_cols"]:
            if c not in df_in.columns:
                df_in[c] = 0.0
        if expert.get("posture_norm", False):
            df_in = apply_posture_baseline(df_in, posture_baseline)
        X_calib_scaled = expert["scaler"].transform(
            df_in[expert["feature_cols"]].fillna(0).values.astype(float)
        )
    else:
        # Backwards compat: raw ndarray, no posture norm
        X_calib_scaled = expert["scaler"].transform(df_calib)

    # ── (Strategy C) Per-user class centroids ────────────────────────────
    # Computed in scaled feature space, restricted to top-importance features.
    centroid_idx = get_centroid_feature_indices(expert, n_features=30)
    class_centroids, centroid_counts = compute_class_centroids(
        X_calib_scaled, np.array([str(y).lower() for y in y_calib]),
        expert["key_encoder"].classes_, centroid_idx,
        min_samples_per_class=3,
    )
    if class_centroids:
        ns = sorted(centroid_counts.values())
        print(f"    Centroids built for {len(class_centroids)}/{len(expert['key_encoder'].classes_)} classes "
              f"(samples per class: min={ns[0]}, median={ns[len(ns)//2]}, max={ns[-1]}, "
              f"using top-{len(centroid_idx)} features)")
    else:
        print(f"    No centroids built (all classes had <3 calibration samples)")

    # ── Get model's calibration predictions WITH centroids applied
    # (this is what the user will actually see at inference time, so the
    # logit bias should be fitted on top of these post-centroid probabilities)
    _, fused, _ = predict_for_user(
        df_calib, expert,
        posture_baseline=posture_baseline,
        class_centroids=class_centroids,
        centroid_idx=centroid_idx,
        centroid_alpha=1.0,
    )

    # Encode true labels and filter to valid ones
    label_to_idx = {c: i for i, c in enumerate(classes)}
    valid = np.array([str(y).lower() in label_to_idx for y in y_calib])
    log_p = np.log(np.clip(fused[valid], eps, 1.0))
    y_idx = np.array([label_to_idx[str(y).lower()] for y in y_calib
                      if str(y).lower() in label_to_idx])

    if len(y_idx) == 0:
        print("    ⚠️  No calibration samples match model classes; returning zero bias.")
        return np.zeros(n_classes)

    n_calib = len(y_idx)

    # Count calibration samples per class
    class_counts = np.zeros(n_classes)
    for i in y_idx:
        class_counts[i] += 1

    # Per-class regularization weights:
    # - More samples → less regularization → more aggressive bias
    # - Fewer samples → more regularization → conservative bias
    # Formula: reg_per_class = reg_scale / sqrt(n_per_class + 1)
    # The +1 prevents division by zero for unseen classes.
    per_class_reg = reg_scale / np.sqrt(class_counts + 1.0)

    def loss_and_grad(b):
        logits = log_p + b[None, :]
        log_norm = logsumexp(logits, axis=1, keepdims=True)
        log_softmax = logits - log_norm

        # Cross-entropy on true class
        ce = -log_softmax[np.arange(n_calib), y_idx].mean()

        # Per-class L2 regularization (different strength per class)
        reg = 0.5 * (per_class_reg * b ** 2).sum() / n_classes
        loss = ce + reg

        # Gradient
        softmax = np.exp(log_softmax)
        one_hot = np.zeros_like(softmax)
        one_hot[np.arange(n_calib), y_idx] = 1.0
        grad_ce = (softmax - one_hot).mean(axis=0)
        grad_reg = per_class_reg * b / n_classes
        return loss, grad_ce + grad_reg

    # Optimize
    result = minimize(
        loss_and_grad, np.zeros(n_classes),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "disp": False},
    )

    bias = result.x

    # Diagnostic
    print(f"    Calibration: {n_calib} samples across {(class_counts>0).sum()}/{n_classes} classes")
    print(f"    Min/Max samples per class: {int(class_counts[class_counts>0].min())} / "
          f"{int(class_counts.max())}")
    print(f"    Loss {result.fun:.4f} | bias range [{bias.min():+.2f}, {bias.max():+.2f}] | "
          f"converged: {result.success}")

    return {
        "logit_bias":         bias,
        "posture_baseline":   posture_baseline,
        "class_centroids":    class_centroids,
        "centroid_counts":    centroid_counts,    # NEW: for adaptive alpha
        "centroid_idx":       centroid_idx,
        "centroid_alpha":     1.0,
        "reg_scale":          reg_scale,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-pattern", default=PATH_PATTERN,
                        help="Glob pattern for TRAINING data")
    parser.add_argument("--calib-path-pattern", default=None,
                        help="Glob pattern for the calibration user's data "
                             "(if different from training path). "
                             "E.g. /Users/fateme/Downloads/q25_QTM_DATA_converted/*.csv")
    parser.add_argument("--out", default=None,
                        help="Output pkl path. Default: model_v2_posture.pkl "
                             "or model_v2_nopost.pkl depending on flags.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip marker dropout augmentation")
    parser.add_argument("--no-posture-norm", action="store_true",
                        help="Disable per-user posture normalization "
                             "(for A/B comparison with the old behaviour)")
    parser.add_argument("--centroid-alpha", type=float, default=1.0,
                        help="Strength of per-user centroid correction "
                             "(Strategy C). 0 disables; 1.0 default; "
                             "higher = more aggressive personalization.")
    parser.add_argument("--loso", action="store_true",
                        help="Run leave-one-subject-out evaluation")
    parser.add_argument("--calibrate-on", default=None,
                        help="Participant ID to calibrate + test on (e.g. q25)")
    args = parser.parse_args()

    # ── Load training data ───────────────────────────────────────────────
    print("🚀  Loading training data...")
    right_df, left_df = load_all_data(args.path_pattern, VALID_PARTICIPANTS)

    # ── Apply marker dropout augmentation ────────────────────────────────
    if not args.no_augment:
        print("\n🎲  Applying marker dropout augmentation...")
        right_df = augment_dropout(right_df, "R", rng=np.random.default_rng(42))
        left_df  = augment_dropout(left_df,  "L", rng=np.random.default_rng(43))

    # ── Aggregate features ──────────────────────────────────────────────
    print("\n📊  Aggregating features...")
    right_features, right_cols = aggregate_features(right_df)
    left_features,  left_cols  = aggregate_features(left_df)
    print(f"   Right: {len(right_features)} samples, {len(right_cols)} features")
    print(f"   Left:  {len(left_features)} samples, {len(left_cols)} features")

    # ── Train ───────────────────────────────────────────────────────────
    posture_norm = not args.no_posture_norm
    right_expert = train_hand(right_features, right_cols, "Right",
                              augmented=not args.no_augment,
                              posture_norm=posture_norm)
    left_expert  = train_hand(left_features,  left_cols,  "Left",
                              augmented=not args.no_augment,
                              posture_norm=posture_norm)

    if args.out is None:
        args.out = ("model_v2_posture.pkl" if posture_norm
                    else "model_v2_nopost.pkl")

    payload = {
        "Right_Expert":  right_expert,
        "Left_Expert":   left_expert,
        "Model_Type":    "HIERARCHICAL_ROW_KEY_DUAL_SPECIALIST_V2",
        "trained_on":    VALID_PARTICIPANTS,
        "augmented":     not args.no_augment,
        "posture_norm":  posture_norm,
    }
    joblib.dump(payload, args.out)
    print(f"\n💾  Saved model to {args.out}")

    # ── LOSO evaluation (REAL — retrains for each held-out user) ────────
    if args.loso:
        print("\n" + "=" * 60)
        print("🔄  LEAVE-ONE-SUBJECT-OUT EVALUATION (real)")
        print("=" * 60)
        print("   This retrains the model 4 times (once per held-out user).")
        print("   Each held-out user is genuinely unseen during their own evaluation.")

        # Tag each row with its participant ID
        right_features["pid"] = right_features["Filepath"].apply(participant_id)
        left_features["pid"]  = left_features["Filepath"].apply(participant_id)

        loso_results = {}
        for held_out in VALID_PARTICIPANTS:
            print(f"\n   ───── Held out: {held_out} ─────")

            for hand_name, df_features in [("Right", right_features),
                                            ("Left",  left_features)]:
                train_mask = df_features["pid"] != held_out
                test_mask  = df_features["pid"] == held_out
                df_train   = df_features[train_mask].drop(columns=["pid"])
                df_test    = df_features[test_mask].drop(columns=["pid"])

                if df_test.empty or df_train.empty:
                    print(f"     {hand_name}: skipped (no data)")
                    continue

                feat_cols = [c for c in df_train.columns if c not in
                             {"Pressed_Letter", "Filepath", "Target_Finger",
                              "Target_Row"}]

                # Train a fresh model excluding held_out
                print(f"     Training {hand_name} (without {held_out})...")
                expert_loso = train_hand(df_train, feat_cols,
                                          f"{hand_name}-LOSO-{held_out}",
                                          augmented=True,
                                          posture_norm=posture_norm)

                # Evaluate on held_out user
                # If using posture norm, compute the held-out user's baseline
                # from their own data (simulates a calibration phase).
                y_t = df_test["Pressed_Letter"].values
                pb = (compute_posture_baseline(df_test, feat_cols)
                      if posture_norm else None)
                preds, _, _ = predict_for_user(df_test, expert_loso,
                                                posture_baseline=pb)
                acc = (preds == y_t).mean()
                print(f"     ✅ {hand_name} on unseen {held_out}: {acc:.2%} ({len(y_t)} samples)")
                loso_results[(hand_name, held_out)] = acc

        # Summary
        print("\n   " + "─" * 50)
        print("   LOSO SUMMARY (true cross-user accuracy):")
        print("   " + "─" * 50)
        for hand in ["Right", "Left"]:
            accs = [v for (h, _), v in loso_results.items() if h == hand]
            if accs:
                print(f"     {hand}: mean={np.mean(accs):.2%}, "
                      f"min={np.min(accs):.2%}, max={np.max(accs):.2%}")

    # ── Calibrate + test on a new user ───────────────────────────────────
    if args.calibrate_on:
        print("\n" + "=" * 60)
        print(f"🎯  CALIBRATION + TEST on {args.calibrate_on}")
        print("=" * 60)

        # Use a separate path for the calibration user if provided
        calib_path = args.calib_path_pattern or args.path_pattern
        print(f"   Searching for {args.calibrate_on} files at: {calib_path}")
        try:
            right_test, left_test = load_all_data(calib_path, [args.calibrate_on])
        except RuntimeError as e:
            print(f"\n   ❌  Could not load calibration data: {e}")
            print(f"       Pass --calib-path-pattern '/path/to/{args.calibrate_on}/*.csv'")
            print(f"       The trained model is still saved at: {args.out}")
            return
        right_test_feat, _ = aggregate_features(right_test)
        left_test_feat,  _ = aggregate_features(left_test)

        for hand_name, df_test, expert in [
            ("Right", right_test_feat, right_expert),
            ("Left",  left_test_feat,  left_expert),
        ]:
            if df_test.empty:
                continue

            y = df_test["Pressed_Letter"].values

            # Baseline WITHOUT user calibration
            # (but still applies posture norm using the test set's OWN mean
            # if model was trained with posture_norm — this simulates the user
            # having gone through a calibration phase)
            posture_baseline_test = None
            if expert.get("posture_norm", False):
                posture_baseline_test = compute_posture_baseline(
                    df_test, expert["feature_cols"]
                )
            preds_base, _, _ = predict_for_user(
                df_test, expert,
                posture_baseline=posture_baseline_test,
            )
            base_acc = (preds_base == y).mean()
            print(f"\n   {hand_name} hand — {len(y)} samples")
            print(f"     Baseline (posture-norm only, no logit bias): {base_acc:.2%}")

            # Use first 30% of data for calibration, rest for test
            n_calib = max(20, int(0.30 * len(y)))
            df_calib = df_test.iloc[:n_calib]
            df_eval  = df_test.iloc[n_calib:]
            y_calib, y_eval = y[:n_calib], y[n_calib:]

            # Calibrate (computes posture baseline + logit bias)
            calib = calibrate_user(expert, df_calib, y_calib)

            # Evaluate held-out using posture baseline + centroids + logit bias
            preds_cal, _, _ = predict_for_user(
                df_eval, expert,
                posture_baseline=calib["posture_baseline"],
                logit_bias=calib["logit_bias"],
                class_centroids=calib["class_centroids"],
                centroid_idx=calib["centroid_idx"],
                centroid_alpha=args.centroid_alpha,
                centroid_counts=calib["centroid_counts"],
            )
            cal_acc = (preds_cal == y_eval).mean()
            print(f"     Calibrated (held-out {len(y_eval)}): {cal_acc:.2%}")

            # Save calibration profile
            profile_path = f"user_profile_{args.calibrate_on}_{hand_name}.pkl"
            joblib.dump({
                "hand":               hand_name,
                "user":               args.calibrate_on,
                "logit_bias":         calib["logit_bias"],
                "posture_baseline":   calib["posture_baseline"],
                "class_centroids":    calib["class_centroids"],
                "centroid_counts":    calib["centroid_counts"],
                "centroid_idx":       calib["centroid_idx"],
                "centroid_alpha":     calib["centroid_alpha"],
                "calibration_size":   n_calib,
                "model_posture_norm": expert.get("posture_norm", False),
            }, profile_path)
            print(f"     Profile saved → {profile_path}")


if __name__ == "__main__":
    main()



