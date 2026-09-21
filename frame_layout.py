"""
frame_layout.py — раскладка Tile-деталей рамки и паспарту (жадный алгоритм
подбора длинных плоских деталей 8/6/4/2/1 студ, Г-образные углы 2x2).
"""
from PIL import ImageDraw

from config import (
    FRAME_TILE_SIZES, FRAME_CORNER_SIZE, TILE_STRAIGHT_PRICE_PER_PIECE_RUB,
    SEAM_HALO_COLOR, FRAME_SEAM_LINE_COLOR, MATTING_SEAM_LINE_COLOR,
    FRAME_SEAM_WIDTH_FACTOR, MATTING_SEAM_WIDTH_FACTOR,
)


def greedy_pack_from_corner_to_center(half_length, sizes=FRAME_TILE_SIZES):
    big_size = sizes[0]
    big_count = half_length // big_size
    remainder = half_length % big_size
    center_part = [big_size] * big_count
    if remainder == 0:
        return center_part
    edge_part = []
    rem = remainder
    for s in sizes[1:]:
        while rem >= s:
            edge_part.append(s)
            rem -= s
    return edge_part + center_part


def layout_straight_run(run_studs, sizes=FRAME_TILE_SIZES):
    if run_studs <= 0:
        return []
    if run_studs in sizes:
        return [run_studs]
    if run_studs % 2 == 1:
        half_len = (run_studs - 1) // 2
        half = greedy_pack_from_corner_to_center(half_len, sizes)
        return half + [1] + list(reversed(half))
    else:
        half_len = run_studs // 2
        half = greedy_pack_from_corner_to_center(half_len, sizes)
        return half + list(reversed(half))


def build_ring_layout(ring_w, ring_h, corner_size=FRAME_CORNER_SIZE, sizes=FRAME_TILE_SIZES):
    top_run = ring_w - 2 * corner_size
    left_run = ring_h - 2 * corner_size
    return {
        "top": layout_straight_run(top_run, sizes), "bottom": layout_straight_run(top_run, sizes),
        "left": layout_straight_run(left_run, sizes), "right": layout_straight_run(left_run, sizes),
        "corner_size": corner_size, "n_corners": 4,
    }


def build_multi_ring_layout(canvas_w, canvas_h, depth, corner_size=FRAME_CORNER_SIZE, sizes=FRAME_TILE_SIZES):
    rings = []
    for i in range(depth):
        ring_w = canvas_w - 2 * i
        ring_h = canvas_h - 2 * i
        if ring_w <= 2 * corner_size or ring_h <= 2 * corner_size:
            break
        rings.append(build_ring_layout(ring_w, ring_h, corner_size, sizes))
    return rings


def build_ring_usage_report(ring_layout):
    counts = {}
    for side in ("top", "bottom", "left", "right"):
        for piece_len in ring_layout[side]:
            counts[piece_len] = counts.get(piece_len, 0) + 1
    counts["corner_2x2"] = ring_layout["n_corners"]
    return counts


def compute_multi_ring_cost(rings, corner_price_per_piece):
    total_straight = 0
    total_corner = 0
    for ring in rings:
        usage = build_ring_usage_report(ring)
        total_straight += sum(v for k, v in usage.items() if k != "corner_2x2")
        total_corner += usage.get("corner_2x2", 0)
    cost = round(total_straight * TILE_STRAIGHT_PRICE_PER_PIECE_RUB + total_corner * corner_price_per_piece, 2)
    return cost, total_straight, total_corner


def draw_L_corner_contour(draw, off_x_studs, off_y_studs, cell_px, corner_position, color,
                           halo_color=SEAM_HALO_COLOR, corner_size=FRAME_CORNER_SIZE, line_width=3, halo_extra=3):
    c = corner_size
    if corner_position == "top-left":
        points = [(0, 0), (c, 0), (c, 1), (1, 1), (1, c), (0, c), (0, 0)]
    elif corner_position == "top-right":
        points = [(0, 0), (c, 0), (c, c), (c-1, c), (c-1, 1), (0, 1), (0, 0)]
    elif corner_position == "bottom-left":
        points = [(0, 0), (1, 0), (1, c-1), (c, c-1), (c, c), (0, c), (0, 0)]
    else:
        points = [(0, c-1), (c-1, c-1), (c-1, 0), (c, 0), (c, c), (0, c), (0, c-1)]
    scaled = [((off_x_studs + px) * cell_px, (off_y_studs + py) * cell_px) for px, py in points]
    draw.line(scaled, fill=halo_color, width=line_width + halo_extra * 2, joint="curve")
    draw.line(scaled, fill=color, width=line_width, joint="curve")


def overlay_frame_tile_layout(preview_img, canvas_w_studs, canvas_h_studs, frame_depth, matting_depth, cell_px,
                               show_frame=True, show_matting=True, show_labels=True):
    img = preview_img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    frame_seam_w = max(2, cell_px // FRAME_SEAM_WIDTH_FACTOR)
    matting_seam_w = max(1, cell_px // MATTING_SEAM_WIDTH_FACTOR)
    c = FRAME_CORNER_SIZE

    def draw_piece_rect(x0s, y0s, x1s, y1s, color, seam_w, label=None):
        x0, y0, x1, y1 = x0s * cell_px, y0s * cell_px, x1s * cell_px, y1s * cell_px
        draw.rectangle([x0, y0, x1, y1], outline=SEAM_HALO_COLOR, width=seam_w + 4)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=seam_w)
        if label and show_labels and cell_px > 10:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            draw.text((cx, cy), label, fill=color, anchor="mm")

    def draw_one_ring(ring_w, ring_h, off_x, off_y, color, seam_w):
        layout_top = layout_straight_run(ring_w - 2 * c)
        layout_left = layout_straight_run(ring_h - 2 * c)
        corner_specs = [("top-left", off_x, off_y), ("top-right", off_x + ring_w - c, off_y),
                         ("bottom-left", off_x, off_y + ring_h - c), ("bottom-right", off_x + ring_w - c, off_y + ring_h - c)]
        for pos, ox, oy in corner_specs:
            draw_L_corner_contour(draw, ox, oy, cell_px, pos, color, corner_size=c, line_width=seam_w)

        x_cursor = off_x + c
        for piece_len in layout_top:
            draw_piece_rect(x_cursor, off_y, x_cursor + piece_len, off_y + 1, color, seam_w, str(piece_len))
            x_cursor += piece_len
        x_cursor = off_x + c
        for piece_len in layout_top:
            draw_piece_rect(x_cursor, off_y + ring_h - 1, x_cursor + piece_len, off_y + ring_h, color, seam_w, str(piece_len))
            x_cursor += piece_len
        y_cursor = off_y + c
        for piece_len in layout_left:
            draw_piece_rect(off_x, y_cursor, off_x + 1, y_cursor + piece_len, color, seam_w, str(piece_len))
            y_cursor += piece_len
        y_cursor = off_y + c
        for piece_len in layout_left:
            draw_piece_rect(off_x + ring_w - 1, y_cursor, off_x + ring_w, y_cursor + piece_len, color, seam_w, str(piece_len))
            y_cursor += piece_len

    if show_frame:
        for i in range(frame_depth):
            ring_w, ring_h = canvas_w_studs - 2 * i, canvas_h_studs - 2 * i
            if ring_w <= 2 * c or ring_h <= 2 * c:
                break
            draw_one_ring(ring_w, ring_h, i, i, FRAME_SEAM_LINE_COLOR, frame_seam_w)

    if show_matting:
        for i in range(matting_depth):
            ring_w = canvas_w_studs - 2 * frame_depth - 2 * i
            ring_h = canvas_h_studs - 2 * frame_depth - 2 * i
            if ring_w <= 2 * c or ring_h <= 2 * c:
                break
            draw_one_ring(ring_w, ring_h, frame_depth + i, frame_depth + i, MATTING_SEAM_LINE_COLOR, matting_seam_w)

    return img.convert("RGB")
