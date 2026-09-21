"""
colors_db.py — база данных цветов LEGO (SQLite), конвертация в LAB,
подбор уникальных цветов палитры под кластеры K-Means.
"""
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
from skimage.color import rgb2lab, deltaE_ciede2000

from config import DB_PATH, SELLER_COLORS, PRICE_PER_PIECE_RUB


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def init_demo_db(path=DB_PATH):
    import os
    if os.path.exists(path):
        conn = sqlite3.connect(path)
        try:
            existing_count = pd.read_sql("SELECT COUNT(*) as c FROM colors", conn).iloc[0]["c"]
        except Exception:
            existing_count = -1
        conn.close()
        if existing_count == len(SELLER_COLORS):
            return
        os.remove(path)

    rows = []
    for i, (name, hexcode) in enumerate(SELLER_COLORS, start=1):
        r, g, b = hex_to_rgb(hexcode)
        rows.append({"color_id": i, "name_ru": name, "rgb_hex": hexcode, "r": r, "g": g, "b": b,
                      "price_per_piece_rub": PRICE_PER_PIECE_RUB, "qty_in_stock": 1000, "is_active": 1})
    df = pd.DataFrame(rows)
    lab_vals = rgb2lab((df[["r", "g", "b"]].values.reshape(-1, 1, 3) / 255.0)).reshape(-1, 3)
    df["L"], df["a"], df["b_lab"] = lab_vals[:, 0], lab_vals[:, 1], lab_vals[:, 2]
    conn = sqlite3.connect(path)
    df.to_sql("colors", conn, if_exists="replace", index=False)
    conn.close()


@st.cache_data
def load_colors(path=DB_PATH):
    conn = sqlite3.connect(path)
    df = pd.read_sql("SELECT * FROM colors WHERE is_active = 1", conn)
    conn.close()
    return df


def rgb_to_lab_fast(rgb):
    r, g, b = rgb / 255.0

    def to_linear(c):
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92

    r, g, b = to_linear(r), to_linear(g), to_linear(b)
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    x, y, z = x / 0.95047, y / 1.0, z / 1.08883

    def f(t):
        return t ** (1/3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])


def assign_unique_colors(centers_lab, colors_df):
    palette_lab_all = colors_df[["L", "a", "b_lab"]].values
    n_clusters = len(centers_lab)
    n_available = len(palette_lab_all)
    dist_matrix = np.zeros((n_clusters, n_available))
    for i, center_lab in enumerate(centers_lab):
        dist_matrix[i] = deltaE_ciede2000(center_lab[None, :], palette_lab_all)

    assigned_color_idx = [-1] * n_clusters
    used = set()
    flat_order = np.dstack(np.unravel_index(np.argsort(dist_matrix, axis=None), dist_matrix.shape))[0]
    for cluster_i, color_j in flat_order:
        cluster_i, color_j = int(cluster_i), int(color_j)
        if assigned_color_idx[cluster_i] == -1 and color_j not in used:
            assigned_color_idx[cluster_i] = color_j
            used.add(color_j)
        if all(a != -1 for a in assigned_color_idx):
            break
    for i in range(n_clusters):
        if assigned_color_idx[i] == -1:
            assigned_color_idx[i] = int(np.argmin(dist_matrix[i]))

    color_ids = colors_df["color_id"].values
    return [int(color_ids[idx]) for idx in assigned_color_idx]
