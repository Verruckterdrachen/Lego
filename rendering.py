"""
rendering.py — рендер финального превью мозаики (кружки/квадраты на чёрной
подложке), overlay сетки панелей 16x16 и центровых линий, визуализация
кропа, перевод клика по превью в координаты студины.
"""
import numpy as np
from PIL import Image, ImageDraw

from config import (
    UNIT_STUDS, FRAME_BLACK, MATTING_WHITE, MOSAIC_BG_COLOR, CANVAS_GRID_LINE_COLOR,
    CENTERLINE_COLOR, CENTER_DOT_COLOR, CROP_DARKEN_COLOR, CROP_KEEP_BORDER_COLOR,
    PREVIEW_SUPERSAMPLE, COMPARE_PREVIEW_MAX_WIDTH_PX,
)
from canvas_math import compute_adaptive_cell_px


def render_crop_visualization(orig_img, crop_box, max_preview_w=COMPARE_PREVIEW_MAX_WIDTH_PX):
    img = orig_img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([0, 0, img.width, img.height], fill=CROP_DARKEN_COLOR)
    x0, y0, x1, y1 = crop_box
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 0))
    border_w = max(2, img.width // 200)
    draw.rectangle([x0, y0, x1, y1], outline=CROP_KEEP_BORDER_COLOR, width=border_w)
    result = Image.alpha_composite(img, overlay).convert("RGB")
    if result.width > max_preview_w:
        scale = max_preview_w / result.width
        result = result.resize((max_preview_w, round(result.height * scale)), Image.Resampling.LANCZOS)
    return result


def render_final_preview(pixel_ids, colors_df, canvas_w, canvas_h, frame_depth, matting_depth,
                          preview_mode="round"):
    cell_px = compute_adaptive_cell_px(canvas_w, canvas_h)
    hi_cell = cell_px * PREVIEW_SUPERSAMPLE
    color_map = {row["color_id"]: (row["r"], row["g"], row["b"]) for _, row in colors_df.iterrows()}
    total_inset = frame_depth + matting_depth
    canvas_img = Image.new("RGB", (canvas_w * hi_cell, canvas_h * hi_cell), FRAME_BLACK)
    draw = ImageDraw.Draw(canvas_img)

    if matting_depth > 0:
        mx0, my0 = frame_depth * hi_cell, frame_depth * hi_cell
        mx1, my1 = (canvas_w - frame_depth) * hi_cell, (canvas_h - frame_depth) * hi_cell
        draw.rectangle([mx0, my0, mx1, my1], fill=MATTING_WHITE)

    mosaic_h, mosaic_w = pixel_ids.shape
    px0, py0 = total_inset * hi_cell, total_inset * hi_cell
    draw.rectangle([px0, py0, px0 + mosaic_w * hi_cell, py0 + mosaic_h * hi_cell], fill=MOSAIC_BG_COLOR)

    margin = max(1, int(hi_cell * 0.03))
    for y in range(mosaic_h):
        for x in range(mosaic_w):
            rgb = color_map.get(pixel_ids[y, x], (200, 200, 200))
            x0, y0 = px0 + x * hi_cell, py0 + y * hi_cell
            if preview_mode == "round":
                cx, cy = x0 + hi_cell // 2, y0 + hi_cell // 2
                r = hi_cell // 2 - margin
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)
            else:
                draw.rectangle([x0 + margin, y0 + margin, x0 + hi_cell - margin, y0 + hi_cell - margin], fill=rgb)

    final = canvas_img.resize((canvas_w * cell_px, canvas_h * cell_px), Image.Resampling.LANCZOS)
    return final, cell_px


def _clamped_grid_line(draw, coord, is_vertical, img_w, img_h, color, width):
    half = max(1, width // 2)
    if is_vertical:
        x = max(half, min(coord, img_w - half))
        draw.line([(x, 0), (x, img_h)], fill=color, width=width)
    else:
        y = max(half, min(coord, img_h - half))
        draw.line([(0, y), (img_w, y)], fill=color, width=width)


def overlay_canvas_grid_and_center(preview_img, canvas_w_studs, canvas_h_studs, cell_px, show_grid, show_center):
    img = preview_img.convert("RGBA")
    if show_grid:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        line_width = max(3, cell_px // 3)
        for gx in range(0, canvas_w_studs + 1, UNIT_STUDS):
            _clamped_grid_line(draw, gx * cell_px, True, img.width, img.height, CANVAS_GRID_LINE_COLOR, line_width)
        for gy in range(0, canvas_h_studs + 1, UNIT_STUDS):
            _clamped_grid_line(draw, gy * cell_px, False, img.width, img.height, CANVAS_GRID_LINE_COLOR, line_width)
        img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")

    if show_center:
        draw2 = ImageDraw.Draw(img)
        cx_px, cy_px = img.width / 2, img.height / 2
        dash_len, gap_len = 14, 8
        y = 0
        while y < img.height:
            draw2.line([(cx_px, y), (cx_px, min(y + dash_len, img.height))], fill=CENTERLINE_COLOR, width=3)
            y += dash_len + gap_len
        x = 0
        while x < img.width:
            draw2.line([(x, cy_px), (min(x + dash_len, img.width), cy_px)], fill=CENTERLINE_COLOR, width=3)
            x += dash_len + gap_len
        dot_r = max(6, cell_px // 2)
        draw2.ellipse([cx_px - dot_r, cy_px - dot_r, cx_px + dot_r, cy_px + dot_r],
                       fill=CENTER_DOT_COLOR, outline=(0, 0, 0), width=2)
    return img


def click_to_stud_coords(click_x, click_y, cell_px, total_inset, mosaic_w, mosaic_h):
    """Переводит пиксель клика по previewimg в (row, col) pixel_ids. None, если клик по рамке/паспарту."""
    offset_px = total_inset * cell_px
    rel_x = click_x - offset_px
    rel_y = click_y - offset_px
    if rel_x < 0 or rel_y < 0:
        return None
    col = rel_x // cell_px
    row = rel_y // cell_px
    if 0 <= row < mosaic_h and 0 <= col < mosaic_w:
        return int(row), int(col)
    return None


def apply_manual_color_overrides(pixel_ids, overrides):
    """overrides: {(row, col): color_id}. Применяется к копии pixel_ids, не мутируя оригинал."""
    if not overrides:
        return pixel_ids
    result = pixel_ids.copy()
    for (row, col), color_id in overrides.items():
        if 0 <= row < result.shape[0] and 0 <= col < result.shape[1]:
            result[row, col] = color_id
    return result
