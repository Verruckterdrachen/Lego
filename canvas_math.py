"""
canvas_math.py — расчёт полотна LEGO Art (система готовых наборов 16x16),
подбор оптимального масштаба мозаики, DP-расчёт минимальной стоимости панелей.
"""
import streamlit as st
import pandas as pd

from config import (
    CANVAS_SETS, BEST_VALUE_SET, UNIT_STUDS, MAX_PANELS_PER_DIM, AR_TOLERANCE,
    PRICE_PER_PIECE_RUB, MIN_STUDS, MAX_STUDS,
)


def compute_adaptive_cell_px(canvas_w_studs, canvas_h_studs, target_width_px=900,
                              min_cell_px=4, max_cell_px=40):
    raw_cell = target_width_px / max(1, canvas_w_studs)
    return int(max(min_cell_px, min(max_cell_px, raw_cell)))


def get_catalog_table():
    rows = []
    for name, s in CANVAS_SETS.items():
        rows.append({
            "Набор": name, "Плиток": s["pieces"], "Цена, руб": s["price_rub"],
            "Руб/плитку": s["price_per_piece"], "★": "★" if name == BEST_VALUE_SET else "",
        })
    return pd.DataFrame(rows).sort_values("Руб/плитку").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def min_cost_for_n_panels(n_panels_needed):
    INF = float("inf")
    max_panels = n_panels_needed + max(s["pieces"] for s in CANVAS_SETS.values())
    dp = [INF] * (max_panels + 1)
    dp[0] = 0
    for total in range(max_panels):
        if dp[total] == INF:
            continue
        for s in CANVAS_SETS.values():
            nt = min(total + s["pieces"], max_panels)
            if dp[total] + s["price_rub"] < dp[nt]:
                dp[nt] = dp[total] + s["price_rub"]
    return min(dp[n_panels_needed:])


@st.cache_data(show_spinner=False)
def find_best_canvas_shape(mosaic_w, mosaic_h, max_panels_per_dim=MAX_PANELS_PER_DIM, ar_tolerance=AR_TOLERANCE):
    """Обязательное покрытие: canvas_w >= mosaic_w и canvas_h >= mosaic_h."""
    target_ar = mosaic_w / mosaic_h
    options = []
    for n_cols in range(1, max_panels_per_dim + 1):
        for n_rows in range(1, max_panels_per_dim + 1):
            cw, ch = n_cols * UNIT_STUDS, n_rows * UNIT_STUDS
            if cw < mosaic_w or ch < mosaic_h:
                continue
            ar_diff = abs(cw / ch - target_ar) / target_ar
            if ar_diff > ar_tolerance:
                continue
            n_panels = n_cols * n_rows
            cost = min_cost_for_n_panels(n_panels)
            excess = (cw - mosaic_w) + (ch - mosaic_h)
            options.append({"n_cols": n_cols, "n_rows": n_rows, "w": cw, "h": ch,
                             "n_panels": n_panels, "cost": cost, "ar_diff": ar_diff, "excess": excess})
    if not options:
        return None
    options.sort(key=lambda o: (o["cost"], o["excess"], o["ar_diff"]))
    return options[0]


@st.cache_data(show_spinner=False)
def find_best_canvas_covering(target_w, target_h, max_panels_per_dim=MAX_PANELS_PER_DIM, ar_tolerance=AR_TOLERANCE):
    target_ar = target_w / target_h
    options = []
    for n_cols in range(1, max_panels_per_dim + 1):
        for n_rows in range(1, max_panels_per_dim + 1):
            cw, ch = n_cols * UNIT_STUDS, n_rows * UNIT_STUDS
            if cw < target_w or ch < target_h:
                continue
            ar_diff = abs(cw / ch - target_ar) / target_ar
            if ar_diff > ar_tolerance:
                continue
            n_panels = n_cols * n_rows
            cost = min_cost_for_n_panels(n_panels)
            excess = (cw - target_w) + (ch - target_h)
            options.append({"n_cols": n_cols, "n_rows": n_rows, "w": cw, "h": ch,
                             "n_panels": n_panels, "cost": cost, "ar_diff": ar_diff, "excess": excess})
    if not options:
        return None
    options.sort(key=lambda o: (o["cost"], o["excess"], o["ar_diff"]))
    return options[0]


def evaluate_frame_strategy(mosaic_w, mosaic_h, depth, base_canvas=None, cost_per_pct_threshold=50):
    base = base_canvas or find_best_canvas_shape(mosaic_w, mosaic_h)
    if base is None:
        return None
    expand_target_w = mosaic_w + 2 * depth
    expand_target_h = mosaic_h + 2 * depth
    expand_canvas = find_best_canvas_covering(expand_target_w, expand_target_h)
    extra_cost = (expand_canvas["cost"] - base["cost"]) if expand_canvas else float("inf")
    shrink_internal_w = base["w"] - 2 * depth
    shrink_internal_h = base["h"] - 2 * depth
    shrink_valid = shrink_internal_w > 0 and shrink_internal_h > 0
    content_loss_pct = (round((1 - (shrink_internal_w * shrink_internal_h) / (mosaic_w * mosaic_h)) * 100, 1)
                         if shrink_valid else 100.0)

    if not shrink_valid:
        recommendation = "expand"
    elif not expand_canvas:
        recommendation = "shrink"
    elif extra_cost == 0:
        recommendation = "expand"
    elif content_loss_pct <= 0:
        recommendation = "shrink"
    else:
        recommendation = "expand" if (extra_cost / content_loss_pct) < cost_per_pct_threshold else "shrink"

    return {
        "base": base, "expand_canvas": expand_canvas, "extra_cost": extra_cost, "shrink_canvas": base,
        "shrink_internal_w": shrink_internal_w, "shrink_internal_h": shrink_internal_h,
        "content_loss_pct": content_loss_pct, "recommendation": recommendation,
    }


def compute_backing_cost_as_pieces(canvas_w_studs, canvas_h_studs, price_per_piece=PRICE_PER_PIECE_RUB):
    return round(canvas_w_studs * canvas_h_studs * price_per_piece, 2)


def get_scale_options(width_px, height_px, min_studs=MIN_STUDS, max_studs=MAX_STUDS):
    raw_options = []
    seen_studs = set()
    for i in range(1, 121):
        c = 0.005 * i
        w_studs = round(width_px * c)
        h_studs = round(height_px * c)
        if w_studs < min_studs or h_studs < min_studs:
            continue
        if w_studs > max_studs or h_studs > max_studs:
            continue
        key = (w_studs, h_studs)
        if key in seen_studs:
            continue
        seen_studs.add(key)
        raw_options.append({"coeff": c, "w_studs": w_studs, "h_studs": h_studs, "total_tiles": w_studs * h_studs})
    raw_options.sort(key=lambda o: o["total_tiles"])

    deduped = []
    seen_canvas = set()
    for o in raw_options:
        canvas = find_best_canvas_shape(o["w_studs"], o["h_studs"])
        if canvas is None:
            continue
        canvas_key = (canvas["w"], canvas["h"])
        if canvas_key in seen_canvas:
            continue
        seen_canvas.add(canvas_key)
        deduped.append(o)
    return deduped


def make_scale_label(w_studs, h_studs):
    best = find_best_canvas_shape(w_studs, h_studs)
    if best is None:
        return f"{w_studs}×{h_studs} — нет полотна"
    return (f"{best['n_cols']}×{best['n_rows']} панелей "
            f"({best['w']}×{best['h']} дет., {best['w']*0.8:.0f}×{best['h']*0.8:.0f} см, {best['cost']} руб)")
