"""
crop_geometry.py — вся математика кропа фото под целевые пропорции мозаики,
ручной сдвиг кропа вдоль подвижной оси, работа с "супер-областью" для кэша.
"""
import math


def compute_crop_box(orig_w_px, orig_h_px, target_w, target_h, ar_tolerance_pct=8.0):
    target_ar = target_w / target_h
    orig_ar = orig_w_px / orig_h_px
    ar_diff_pct = abs(target_ar - orig_ar) / orig_ar * 100
    if ar_diff_pct <= ar_tolerance_pct:
        return (0, 0, orig_w_px, orig_h_px), False, 0.0, ar_diff_pct
    if orig_ar > target_ar:
        new_w = round(orig_h_px * target_ar)
        crop_x = (orig_w_px - new_w) // 2
        box = (crop_x, 0, crop_x + new_w, orig_h_px)
        pct = round((orig_w_px - new_w) / orig_w_px * 100, 1)
    else:
        new_h = round(orig_w_px / target_ar)
        crop_y = (orig_h_px - new_h) // 2
        box = (0, crop_y, orig_w_px, crop_y + new_h)
        pct = round((orig_h_px - new_h) / orig_h_px * 100, 1)
    return box, True, pct, ar_diff_pct


def get_crop_movable_axis(base_crop_box, orig_w_px, orig_h_px):
    """
    Определяет, по какой оси возможен ручной сдвиг кропа.
    Обрезка всегда максимум по одной оси (либо X, либо Y).
    Возвращает ('x'|'y'|None, свободный_ход_px).
    """
    x0, y0, x1, y1 = base_crop_box
    crop_w, crop_h = x1 - x0, y1 - y0
    if crop_w < orig_w_px:
        return "x", orig_w_px - crop_w
    if crop_h < orig_h_px:
        return "y", orig_h_px - crop_h
    return None, 0


def compute_crop_step_px(base_crop_box, mosaic_target_w, mosaic_target_h, axis):
    """шаг сдвига кропа = ширина/высота ОДНОЙ детали (1 stud) мозаики в пикселях оригинала."""
    x0, y0, x1, y1 = base_crop_box
    crop_w, crop_h = x1 - x0, y1 - y0
    if axis == "x" and mosaic_target_w > 0:
        return max(1, round(crop_w / mosaic_target_w))
    if axis == "y" and mosaic_target_h > 0:
        return max(1, round(crop_h / mosaic_target_h))
    return 0


def get_crop_offset_bounds(base_crop_box, orig_w_px, orig_h_px, axis, step_px):
    """Минимально/максимально допустимый offset_studs для данной оси."""
    if axis is None or step_px <= 0:
        return 0, 0
    x0, y0, x1, y1 = base_crop_box
    if axis == "x":
        crop_w = x1 - x0
        min_x0, max_x0 = 0, orig_w_px - crop_w
        min_offset = math.ceil((min_x0 - x0) / step_px)
        max_offset = math.floor((max_x0 - x0) / step_px)
    else:
        crop_h = y1 - y0
        min_y0, max_y0 = 0, orig_h_px - crop_h
        min_offset = math.ceil((min_y0 - y0) / step_px)
        max_offset = math.floor((max_y0 - y0) / step_px)
    return min_offset, max_offset


def apply_crop_offset(base_crop_box, orig_w_px, orig_h_px, axis, offset_studs, step_px,
                       offset_min=None, offset_max=None):
    """
    Сдвигает базовый (центрированный) crop_box на offset_studs деталей вдоль axis.
    Крайние позиции диапазона прилипают к абсолютному краю оригинала (0 или
    orig_px - crop_size), промежуточные остаются кратными ровно одной детали.
    """
    x0, y0, x1, y1 = base_crop_box
    if axis is None or step_px <= 0 or offset_studs == 0:
        return base_crop_box, 0
    if offset_min is None or offset_max is None:
        offset_min, offset_max = get_crop_offset_bounds(base_crop_box, orig_w_px, orig_h_px, axis, step_px)
    offset_studs = max(offset_min, min(offset_max, offset_studs))
    if axis == "x":
        crop_w = x1 - x0
        if offset_studs <= offset_min:
            new_x0 = 0
        elif offset_studs >= offset_max:
            new_x0 = orig_w_px - crop_w
        else:
            new_x0 = x0 + offset_studs * step_px
        return (new_x0, y0, new_x0 + crop_w, y1), offset_studs
    else:
        crop_h = y1 - y0
        if offset_studs <= offset_min:
            new_y0 = 0
        elif offset_studs >= offset_max:
            new_y0 = orig_h_px - crop_h
        else:
            new_y0 = y0 + offset_studs * step_px
        return (x0, new_y0, x1, new_y0 + crop_h), offset_studs


def compute_full_span_box(base_crop_box, orig_w_px, orig_h_px, axis):
    """"Супер-область" — crop_box, растянутый на весь диапазон возможного сдвига по подвижной оси."""
    x0, y0, x1, y1 = base_crop_box
    if axis == "x":
        return (0, y0, orig_w_px, y1)
    elif axis == "y":
        return (x0, 0, x1, orig_h_px)
    return base_crop_box


def crop_window_from_full(crop_box, full_span_box, orig_w_px, orig_h_px, axis,
                           mosaic_target_w, mosaic_target_h, full_mosaic_w, full_mosaic_h):
    """Переводит текущий crop_box в точные индексы окна внутри БОЛЬшого pixel_ids_full."""
    cx0, cy0, cx1, cy1 = crop_box
    fx0, fy0, fx1, fy1 = full_span_box
    if axis == "x":
        if cx0 <= fx0:
            col0 = 0
        elif cx1 >= fx1:
            col0 = full_mosaic_w - mosaic_target_w
        else:
            px_per_stud = (fx1 - fx0) / full_mosaic_w if full_mosaic_w else 1
            col0 = round((cx0 - fx0) / px_per_stud) if px_per_stud else 0
        col0 = max(0, min(full_mosaic_w - mosaic_target_w, col0))
        return 0, full_mosaic_h, col0, col0 + mosaic_target_w
    elif axis == "y":
        if cy0 <= fy0:
            row0 = 0
        elif cy1 >= fy1:
            row0 = full_mosaic_h - mosaic_target_h
        else:
            px_per_stud = (fy1 - fy0) / full_mosaic_h if full_mosaic_h else 1
            row0 = round((cy0 - fy0) / px_per_stud) if px_per_stud else 0
        row0 = max(0, min(full_mosaic_h - mosaic_target_h, row0))
        return row0, row0 + mosaic_target_h, 0, full_mosaic_w
    else:
        return 0, mosaic_target_h, 0, mosaic_target_w


def compute_full_mosaic_dims(full_span_box, axis, mosaic_target_w, mosaic_target_h, step_px):
    """Размер "супер-мозаики" - на весь диапазон возможного сдвига кропа."""
    fx0, fy0, fx1, fy1 = full_span_box
    if axis == "x" and step_px > 0:
        full_w = max(mosaic_target_w, round((fx1 - fx0) / step_px))
        return full_w, mosaic_target_h
    elif axis == "y" and step_px > 0:
        full_h = max(mosaic_target_h, round((fy1 - fy0) / step_px))
        return mosaic_target_w, full_h
    return mosaic_target_w, mosaic_target_h
