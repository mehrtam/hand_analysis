# geometry.py
import numpy as np
import pandas as pd  # only for types; not heavily used

def midpoint(p1, p2):
    return (p1 + p2) / 2

def line_vector(p1, p2):
    return p2 - p1

def angle_between_vectors(v1, v2):
    epsilon = 1e-8
    v1_u = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + epsilon)
    v2_u = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + epsilon)
    cos_theta = np.sum(v1_u * v2_u, axis=1)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def project_onto_plane(v, n):
    dot_product = np.sum(v * n, axis=1, keepdims=True)
    return v - dot_product * n

def plane_normal(p1, p2, p3):
    n = np.cross(p2 - p1, p3 - p1)
    epsilon = 1e-12
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    return n / (norm_n + epsilon)

def get_3d_cols(prefix):
    return [f"{prefix}_X", f"{prefix}_Y", f"{prefix}_Z"]

def load_marker_coords(df, marker_name):
    cols = get_3d_cols(marker_name)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return np.full((len(df), 3), np.nan)
    return df[cols].replace([np.inf, -np.inf], np.nan).to_numpy()

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2, axis=1)

def compute_all_mcp_abduction_angles(L1, R1, M1, I1, L4, R4, M4, I4, Cin, Cout, hand_side="R"):
    palm_mid = midpoint(Cin, Cout)
    mcp_bar_vec = line_vector(I1, L1)
    palm_normal_vec = plane_normal(I1, L1, palm_mid)

    fingers = {
        "Little": (L1, L4),
        "Ring":   (R1, R4),
        "Middle": (M1, M4),
        "Index":  (I1, I4),
    }

    results = {}
    for label, (mcp, tip) in fingers.items():
        v_mcp_tip = line_vector(mcp, tip)
        v_proj = project_onto_plane(v_mcp_tip, palm_normal_vec)
        angle = angle_between_vectors(v_proj, mcp_bar_vec)
        cross_prod = np.cross(mcp_bar_vec, v_proj)
        sign = np.sign(np.sum(cross_prod * palm_normal_vec, axis=1))
        sign[sign == 0] = 1
        results[label] = angle * sign if hand_side == "R" else angle * -sign

    return results

def compute_mcp_flexion_angles(L1, L2, R1, R2, M1, M2, I1, I2, Cin, Cout):
    palm_mid = midpoint(Cin, Cout)
    palm_normal_vec = plane_normal(I1, L1, palm_mid)

    fingers = {
        "Little": (L1, L2),
        "Ring":   (R1, R2),
        "Middle": (M1, M2),
        "Index":  (I1, I2),
    }

    results = {}
    for label, (mcp, pip) in fingers.items():
        v = line_vector(mcp, pip)
        angle_with_normal = angle_between_vectors(v, palm_normal_vec)
        results[label] = np.pi / 2 - angle_with_normal
    return results

def compute_all_finger_segment_angles(L1, L2, L3, L4,
                                      R1, R2, R3, R4,
                                      M1, M2, M3, M4,
                                      I1, I2, I3, I4):
    fingers = {
        "Little": (L1, L2, L3, L4),
        "Ring":   (R1, R2, R3, R4),
        "Middle": (M1, M2, M3, M4),
        "Index":  (I1, I2, I3, I4),
    }
    results = {}
    for label, (mcp, pip, dip, tip) in fingers.items():
        v1 = line_vector(mcp, pip)
        v2 = line_vector(pip, dip)
        v3 = line_vector(dip, tip)
        results[label] = {
            "PIP_angle": angle_between_vectors(v1, v2),
            "DIP_angle": angle_between_vectors(v2, v3),
        }
    return results
