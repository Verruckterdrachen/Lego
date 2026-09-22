import io
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.segmentation import slic
from skimage.restoration import denoise_bilateral
from sklearn.cluster import KMeans
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from config import QUANTIZE_MODE_LABELS, BAYER_4x4, BAYER_8x8, _BAYER_MATRICES, SLIC_MAX_DIM, SLIC_COMPACTNESS, SLIC_SIGMA, SLIC_SEGMENTS_PER_STUD
from colors_db import load_colors, rgb_to_lab_fast, assign_unique_colors


def resize_for_slic(img, max_dim=SLIC_MAX_DIM):
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    return img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)


@st.cache_data(show_spinner=False)
def compute_slic_segments(_img_bytes, w_studs, h_studs, crop_box=None):
    img = Image.open(io.BytesIO(_img_bytes)).convert("RGB")
    if crop_box:
        img = img.crop(crop_box)
    prepped = resize_for_slic(img)
    img_float = np.array(prepped).astype(float) / 255.0
    n_target_segments = max(50, int(w_studs * h_studs * SLIC_SEGMENTS_PER_STUD))
    segments = slic(img_float, n_segments=n_target_segments, compactness=SLIC_COMPACTNESS,
                     sigma=SLIC_SIGMA, start_label=0)

    seg_h, seg_w = segments.shape
    y_edges = np.linspace(0, seg_h, h_studs + 1).astype(int)
    x_edges = np.linspace(0, seg_w, w_studs + 1).astype(int)

    grid_segments = np.zeros((h_studs, w_studs), dtype=int)
    for ty in range(h_studs):
        for tx in range(w_studs):
            block = segments[y_edges[ty]:y_edges[ty+1], x_edges[tx]:x_edges[tx+1]]
            if block.size == 0:
                grid_segments[ty, tx] = grid_segments[ty, tx-1] if tx > 0 else 0
                continue
            vals, counts = np.unique(block, return_counts=True)
            majority_seg = vals[counts.argmax()]
            majority_frac = counts.max() / block.size
            if majority_frac < 0.65 and len(vals) > 1:
                second_idx = np.argsort(counts)[-2]
                grid_segments[ty, tx] = vals[second_idx] if counts[second_idx] / block.size >= 0.35 else majority_seg
            else:
                grid_segments[ty, tx] = majority_seg
    return grid_segments


def pre_sharpen(img, percent=150, radius=2, threshold=2):
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def pixelate_exact(img, target_w, target_h, crop_box=None, apply_sharpen=False):
    """apply_sharpen=False по умолчанию — UnsharpMask создаёт ореолы,
    которые при большом числе цветов превращаются в битые одиночные пиксели."""
    work = img.crop(crop_box) if crop_box else img
    if apply_sharpen:
        work = pre_sharpen(work)
    return work.resize((target_w, target_h), Image.Resampling.BOX)


def build_working_palette(small_img, n_colors, colors_df):
    """KMeans в Lab — perceptually uniform пространство.
    Используем IMPORTANCE-взвешенный сэмплинг: центральная 1/3 изображения
    (зона лица/объекта) имеет вес 3x, края — 1x. Это направляет палитру
    на значимые детали, а не на фон и волосы по площади."""
    arr_rgb = np.array(small_img.convert("RGB")).reshape(-1, 3).astype(float)
    h_img, w_img = small_img.size[1], small_img.size[0]

    # Важностная маска: центральные 60% по ширине и 50% по высоте — вес 3
    ys, xs = np.mgrid[0:h_img, 0:w_img]
    y_flat, x_flat = ys.flatten(), xs.flatten()
    cy, cx = h_img / 2.0, w_img / 2.0
    in_center = (
        (np.abs(y_flat - cy) < h_img * 0.25) &
        (np.abs(x_flat - cx) < w_img * 0.30)
    )
    weights = np.where(in_center, 3, 1).astype(int)
    weighted_rgb = np.repeat(arr_rgb, weights, axis=0)

    arr_lab = rgb2lab((weighted_rgb.reshape(-1, 1, 3) / 255.0)).reshape(-1, 3)
    n_unique_pixels = len(np.unique(arr_rgb, axis=0))
    k = max(1, min(n_colors, n_unique_pixels, len(colors_df)))
    km = KMeans(n_clusters=k, n_init=6, random_state=42).fit(arr_lab)
    centers_lab = km.cluster_centers_
    matched_color_ids = assign_unique_colors(centers_lab, colors_df)
    unique_ids = sorted(set(matched_color_ids))
    subset = colors_df[colors_df["color_id"].isin(unique_ids)].reset_index(drop=True)
    return subset, len(unique_ids), k


def smooth_before_pixelize(img, sigma_color=0.08, sigma_spatial=8, enabled=True):
    """Стадия A. sigma_spatial=8 — убирает шум кожи, но НЕ съедает глаза/зрачки."""
    if not enabled:
        return img
    arr = np.asarray(img.convert("RGB")).astype(float) / 255.0
    smoothed = denoise_bilateral(arr, sigma_color=sigma_color, sigma_spatial=sigma_spatial,
                                    channel_axis=-1)
    smoothed = np.clip(smoothed * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(smoothed, mode="RGB")


def smooth_super_strong(img, radius=2, enabled=True):
    """Доп. усиленное сглаживание (Gaussian). Радиус 1-2 безопасен для глаз, 3+ рискует."""
    if not enabled or radius <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _nearest_ids_for_pixels(lab_pixels, palette_lab, palette_color_ids):
    diffs = deltaE_ciede2000(lab_pixels[:, None, :], palette_lab[None, :, :])
    idx = diffs.argmin(axis=1)
    return palette_color_ids[idx], idx


def _build_edge_mask(pixel_ids):
    """Маска границ по Sobel на карте color_id.
    True = граница — эти пиксели не трогает cleanup."""
    from scipy.ndimage import sobel
    arr = pixel_ids.astype(float)
    sx = sobel(arr, axis=1)
    sy = sobel(arr, axis=0)
    magnitude = np.hypot(sx, sy)
    threshold = magnitude.mean() + magnitude.std() * 0.5
    return magnitude > threshold


def _build_gradient_mask(small_img):
    """Маска зон с плавным градиентом (кожа, фон) — там дозволен локальный dithering.
    Зоны с резкими границами (края, детали) — dithering отключён, только nearest.
    Возвращает bool-массив (h, w): True = плавная зона, dithering разрешён."""
    from scipy.ndimage import sobel
    arr_gray = np.array(small_img.convert("L")).astype(float)
    sx = sobel(arr_gray, axis=1)
    sy = sobel(arr_gray, axis=0)
    magnitude = np.hypot(sx, sy)
    # Пороговое значение: ниже среднего — плавная зона
    threshold = magnitude.mean() * 1.2
    return magnitude < threshold


def quantize_nearest(small_img, palette_df, segment_map=None):
    img_arr = np.array(small_img.convert("RGB")).astype(float)
    h, w, _ = img_arr.shape
    palette_lab = palette_df[["L", "a", "b_lab"]].values
    palette_color_ids = palette_df["color_id"].values
    lab_grid = rgb2lab(img_arr / 255.0).reshape(-1, 3)
    ids_flat, _ = _nearest_ids_for_pixels(lab_grid, palette_lab, palette_color_ids)
    return ids_flat.reshape(h, w).astype(np.int32)


def quantize_nearest_potts_fast(small_img, palette_df, segment_map=None,
                                  smoothness=2.0, n_iter=2):
    """Nearest-color + edge-aware Potts-регуляризация (векторизованная ICM).
    Убирает шумовые одиночные пиксели внутри однородных зон (кожа, фон, волосы),
    но НЕ сглаживает через цветовые границы объектов.
    smoothness=2.0, n_iter=2 оптимальны для портретов 48x64 — 80x100."""
    img_arr = np.array(small_img.convert("RGB")).astype(float)
    h, w, _ = img_arr.shape
    palette_lab = palette_df[["L", "a", "b_lab"]].values
    palette_color_ids = palette_df["color_id"].values
    n_colors = len(palette_df)

    lab_grid = rgb2lab(img_arr / 255.0).reshape(-1, 3)
    all_dists = deltaE_ciede2000(lab_grid[:, None, :], palette_lab[None, :, :])
    data_cost = all_dists.reshape(h, w, n_colors)
    current = all_dists.argmin(axis=1).reshape(h, w)

    for _ in range(n_iter):
        agree = np.zeros((h, w, n_colors), dtype=np.float32)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(current, shift=dy, axis=0) if dy else current
            shifted = np.roll(shifted, shift=dx, axis=1) if dx else shifted
            for ci in range(n_colors):
                agree[:, :, ci] += (shifted == ci).astype(np.float32)
        total_cost = data_cost - smoothness * agree
        current = total_cost.argmin(axis=2)

    return palette_color_ids[current].astype(np.int32)


def quantize_nearest_potts_v2(small_img, palette_df, segment_map=None,
                               smoothness=2.5, n_iter=3):
    """nearest_potts_v2 — улучшенный портретный режим.

    Отличия от v1 (nearest_potts):
    1. Importance mask: центральная зона лица имеет в 2x усиленный data_cost penalty
       → палитра «тянется» к оттенкам лица, а не к фону.
    2. Edge-weight: плавность (smoothness) адаптивна — на сильных Sobel-границах
       smoothness снижается до 0.3x, внутри плавных зон — полная.
       Это означает: регуляризатор не сглаживает через веки, губы, контур лица.
    3. Локальный dithering только в плавных зонах (кожа, фон) поверх Potts-результата.
       В зонах границ — строго nearest, без Bayer-шахматки.
    4. n_iter=3 вместо 2 — чуть больше сходимости без значительного роста времени.

    Параметры:
        smoothness: базовая сила регуляризации (2.5 для портрета 48-80 студов).
        n_iter: число итераций ICM (3 достаточно, 4+ не даёт заметного прироста).
    """
    img_arr = np.array(small_img.convert("RGB")).astype(float)
    h, w, _ = img_arr.shape
    palette_lab = palette_df[["L", "a", "b_lab"]].values
    palette_color_ids = palette_df["color_id"].values
    n_colors = len(palette_df)

    # --- 1. data cost ---
    lab_grid = rgb2lab(img_arr / 255.0).reshape(-1, 3)
    all_dists = deltaE_ciede2000(lab_grid[:, None, :], palette_lab[None, :, :])
    data_cost = all_dists.reshape(h, w, n_colors)  # (h, w, K)

    # --- 2. Importance mask (центр изображения — зона лица) ---
    ys, xs = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    dist_y = np.abs(ys - cy) / (h * 0.5)
    dist_x = np.abs(xs - cx) / (w * 0.5)
    # importance [0..1]: 1.0 = центр, 0.0 = край; sigmoid-like falloff
    importance = np.clip(1.0 - np.maximum(dist_y, dist_x), 0.0, 1.0)  # (h, w)
    # Усиливаем data_cost на краях: периферия «нагружена» меньше точностью
    # (центр важнее → там data_cost ощущается сильнее — не снижаем его, зато
    # на краях позволяем регуляризатору доминировать сильнее)
    importance_weight = 1.0 + importance[:, :, None] * 0.8  # (h, w, 1), диапазон 1.0–1.8
    data_cost = data_cost * importance_weight

    # --- 3. Adaptive edge-weight map (Sobel по Lab-яркости) ---
    lab_L = lab_grid[:, 0].reshape(h, w)
    from scipy.ndimage import sobel as _sobel
    sx = _sobel(lab_L, axis=1)
    sy = _sobel(lab_L, axis=0)
    edge_magnitude = np.hypot(sx, sy)  # (h, w)
    # Нормируем: edge_w [0..1], 0 = плоская зона, 1 = острая граница
    e_max = np.percentile(edge_magnitude, 95)
    edge_w = np.clip(edge_magnitude / (e_max + 1e-6), 0.0, 1.0)  # (h, w)
    # На острых границах smoothness → 0.3x; на плоских → 1.0x
    smooth_map = smoothness * (1.0 - edge_w * 0.7)  # (h, w)

    # --- 4. ICM Potts с адаптивным smoothness ---
    current = all_dists.argmin(axis=1).reshape(h, w)

    for _ in range(n_iter):
        agree = np.zeros((h, w, n_colors), dtype=np.float32)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(current, shift=dy, axis=0) if dy else current
            shifted = np.roll(shifted, shift=dx, axis=1) if dx else shifted
            for ci in range(n_colors):
                agree[:, :, ci] += (shifted == ci).astype(np.float32)
        # agree взвешен локальной силой сглаживания
        total_cost = data_cost - smooth_map[:, :, None] * agree
        current = total_cost.argmin(axis=2)

    pixel_ids = palette_color_ids[current].astype(np.int32)  # (h, w)

    # --- 5. Локальный dithering только в плавных зонах ---
    # Bayer 2x2 micro-dither: чуть «оживляет» градиенты кожи, не ломая границы
    flat_mask = edge_w < 0.25  # плавная зона
    if flat_mask.any():
        bayer2 = np.array([[0, 2], [3, 1]], dtype=float) / 4.0 - 0.5  # диапазон [-0.5, 0.5]
        tiled = np.tile(bayer2, (h // 2 + 1, w // 2 + 1))[:h, :w]
        # Только в плоских зонах: пересчитываем с малым смещением
        palette_rgb = palette_df[["r", "g", "b"]].values.astype(float)
        step = 8.0  # небольшой шаг — не создаёт шахматку, только мягкий переход
        perturbed_arr = np.clip(img_arr + tiled[:, :, None] * step, 0, 255)
        perturbed_lab = rgb2lab(perturbed_arr / 255.0).reshape(-1, 3)
        dither_dists = deltaE_ciede2000(perturbed_lab[:, None, :], palette_lab[None, :, :])
        dither_ids = palette_color_ids[dither_dists.argmin(axis=1)].reshape(h, w).astype(np.int32)
        # Применяем dither только там, где flat_mask И где dither меняет цвет
        changed = (dither_ids != pixel_ids) & flat_mask
        pixel_ids = np.where(changed, dither_ids, pixel_ids)

    return pixel_ids


def quantize_by_cluster_map(small_img, palette_df, n_colors_requested, segment_map=None):
    img_arr = np.array(small_img.convert("RGB")).astype(float)
    h, w, _ = img_arr.shape
    flat_rgb = img_arr.reshape(-1, 3)
    palette_lab = palette_df[["L", "a", "b_lab"]].values
    palette_color_ids = palette_df["color_id"].values

    k = max(1, min(n_colors_requested, len(np.unique(flat_rgb, axis=0)), len(palette_df)))
    labels = KMeans(n_clusters=k, n_init=6, random_state=42).fit_predict(flat_rgb)

    out_ids = np.zeros(h * w, dtype=np.int32)
    for cluster_id in range(k):
        mask = labels == cluster_id
        if not mask.any():
            continue
        mean_rgb = flat_rgb[mask].mean(axis=0)
        mean_lab = rgb2lab(mean_rgb.reshape(1, 1, 3) / 255.0).reshape(1, 3)
        color_id, _ = _nearest_ids_for_pixels(mean_lab, palette_lab, palette_color_ids)
        out_ids[mask] = color_id[0]
    return out_ids.reshape(h, w)


def quantize_by_superpixel_map(small_img, palette_df, n_colors_requested, segment_map=None,
                                  n_segments_multiplier=4.0, compactness=12, sigma=1.0):
    """Режим 'superpixel'. SLIC-суперпиксели + усреднение цвета."""
    img_arr = np.array(small_img.convert("RGB")).astype(float)
    h, w, _ = img_arr.shape
    palette_lab = palette_df[["L", "a", "b_lab"]].values
    palette_color_ids = palette_df["color_id"].values

    if segment_map is not None and segment_map.shape == (h, w):
        labels = segment_map
    else:
        img_float = img_arr / 255.0
        n_target = max(30, int(h * w * n_segments_multiplier / 64))
        labels = slic(img_float, n_segments=n_target, compactness=compactness, sigma=sigma, start_label=0)

    out_ids = np.zeros((h, w), dtype=np.int32)
    for lab_id in np.unique(labels):
        mask = labels == lab_id
        if not mask.any():
            continue
        mean_rgb = img_arr[mask].mean(axis=0)
        mean_lab = rgb2lab(mean_rgb.reshape(1, 1, 3) / 255.0).reshape(1, 3)
        color_id, _ = _nearest_ids_for_pixels(mean_lab, palette_lab, palette_color_ids)
        out_ids[mask] = color_id[0]
    return out_ids


def dither_bayer(small_img, palette_df, matrix_key="bayer4", segment_map=None):
    img_arr = np.array(small_img.convert("RGB")).astype(float)
    h, w, _ = img_arr.shape
    matrix = _BAYER_MATRICES[matrix_key]
    mh, mw = matrix.shape
    tiled = np.tile(matrix, (h // mh + 1, w // mw + 1))[:h, :w]
    palette_lab = palette_df[["L", "a", "b_lab"]].values
    palette_color_ids = palette_df["color_id"].values

    step = 255.0 / max(1, len(palette_df) ** (1 / 3))
    perturbed = np.clip(img_arr + tiled[:, :, None] * step, 0, 255)
    lab_grid = rgb2lab(perturbed / 255.0).reshape(-1, 3)
    ids_flat, _ = _nearest_ids_for_pixels(lab_grid, palette_lab, palette_color_ids)
    return ids_flat.reshape(h, w).astype(np.int32)


def dither_atkinson(small_img, palette_df, segment_map=None):
    img_arr = np.array(small_img.convert("RGB")).astype(float)
    h, w, _ = img_arr.shape
    palette_rgb = palette_df[["r", "g", "b"]].values.astype(float)
    palette_lab = palette_df[["L", "a", "b_lab"]].values
    palette_color_ids = palette_df["color_id"].values

    out_ids = np.zeros((h, w), dtype=np.int32)
    work = img_arr.copy()
    use_boundaries = segment_map is not None and segment_map.shape == (h, w)

    for y in range(h):
        for x in range(w):
            old_pixel = np.clip(work[y, x], 0, 255)
            lab_pixel = rgb_to_lab_fast(old_pixel)
            diffs = deltaE_ciede2000(lab_pixel[None, :], palette_lab)
            idx = diffs.argmin()
            out_ids[y, x] = palette_color_ids[idx]
            error = (old_pixel - palette_rgb[idx]) / 8.0
            cur_seg = segment_map[y, x] if use_boundaries else None

            def _add(ny, nx):
                if 0 <= ny < h and 0 <= nx < w:
                    if not use_boundaries or segment_map[ny, nx] == cur_seg:
                        work[ny, nx] += error

            _add(y, x + 1)
            _add(y, x + 2)
            _add(y + 1, x - 1)
            _add(y + 1, x)
            _add(y + 1, x + 1)
            _add(y + 2, x)
    return out_ids


def dither_floyd_steinberg(small_img, palette_df, segment_map=None, strength=1.0):
    img_arr = np.array(small_img.convert("RGB")).astype(float)
    h, w, _ = img_arr.shape

    palette_rgb = palette_df[["r", "g", "b"]].values.astype(float)
    palette_lab = palette_df[["L", "a", "b_lab"]].values
    palette_color_ids = palette_df["color_id"].values

    out_ids = np.zeros((h, w), dtype=np.int32)
    work = img_arr.copy()
    use_boundaries = segment_map is not None and segment_map.shape == (h, w)

    for y in range(h):
        xrange = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
        direction = 1 if y % 2 == 0 else -1
        for x in xrange:
            old_pixel = np.clip(work[y, x], 0, 255)
            lab_pixel = rgb_to_lab_fast(old_pixel)
            diffs = deltaE_ciede2000(lab_pixel[None, :], palette_lab)
            idx = diffs.argmin()
            out_ids[y, x] = palette_color_ids[idx]
            new_pixel = palette_rgb[idx]
            error = (old_pixel - new_pixel) * strength

            cur_seg = segment_map[y, x] if use_boundaries else None

            nx = x + direction
            if 0 <= nx < w and (not use_boundaries or segment_map[y, nx] == cur_seg):
                work[y, nx] += error * 7 / 16
            if y + 1 < h:
                nx_diag_back = x - direction
                if 0 <= nx_diag_back < w and (not use_boundaries or segment_map[y + 1, nx_diag_back] == cur_seg):
                    work[y + 1, nx_diag_back] += error * 3 / 16
                if not use_boundaries or segment_map[y + 1, x] == cur_seg:
                    work[y + 1, x] += error * 5 / 16
                nx_diag_fwd = x + direction
                if 0 <= nx_diag_fwd < w and (not use_boundaries or segment_map[y + 1, nx_diag_fwd] == cur_seg):
                    work[y + 1, nx_diag_fwd] += error * 1 / 16
    return out_ids


def dither_to_lego_palette(small_img, palette_df, segment_map=None, quantize_mode="fs_classic",
                             n_colors_requested=None):
    if quantize_mode == "superpixel":
        n = n_colors_requested if n_colors_requested is not None else len(palette_df)
        return quantize_by_superpixel_map(small_img, palette_df, n, segment_map=segment_map)
    if quantize_mode == "nearest":
        return quantize_nearest(small_img, palette_df, segment_map=segment_map)
    if quantize_mode == "nearest_potts":
        return quantize_nearest_potts_fast(small_img, palette_df, segment_map=segment_map)
    if quantize_mode == "nearest_potts_v2":
        return quantize_nearest_potts_v2(small_img, palette_df, segment_map=segment_map)
    if quantize_mode == "cluster":
        n = n_colors_requested if n_colors_requested is not None else len(palette_df)
        return quantize_by_cluster_map(small_img, palette_df, n, segment_map=segment_map)
    if quantize_mode in ("bayer4", "bayer8"):
        return dither_bayer(small_img, palette_df, matrix_key=quantize_mode, segment_map=segment_map)
    if quantize_mode == "atkinson":
        return dither_atkinson(small_img, palette_df, segment_map=segment_map)
    if quantize_mode == "fs_soft":
        return dither_floyd_steinberg(small_img, palette_df, segment_map=segment_map, strength=0.6)
    return dither_floyd_steinberg(small_img, palette_df, segment_map=segment_map, strength=1.0)


def majority_vote_cleanup_fast(pixel_ids, min_neighbor_fraction=0.5, iterations=1, enabled=True):
    """Чистка одиночных выбросов с защитой граничных пикселей.
    Пиксели вблизи границ (Sobel-маска) НЕ изменяются — сохраняет
    зрачки, веки, края губ и контур лица."""
    if not enabled:
        return pixel_ids

    edge_mask = _build_edge_mask(pixel_ids)

    def _mode_excluding_center(values):
        center = values[4]
        neighbors = np.delete(values, 4)
        vals, counts = np.unique(neighbors, return_counts=True)
        majority_val = vals[counts.argmax()]
        same_fraction = np.sum(neighbors == center) / len(neighbors)
        if same_fraction < min_neighbor_fraction and counts.max() / len(neighbors) >= min_neighbor_fraction:
            return majority_val
        return center

    result = pixel_ids.copy()
    for _ in range(max(1, iterations)):
        cleaned = ndimage.generic_filter(result, _mode_excluding_center, size=3, mode="nearest").astype(np.int32)
        cleaned[edge_mask] = result[edge_mask]
        result = cleaned
    return result


@st.cache_data(show_spinner=False)
def compute_pixel_ids_full(_img_bytes, full_span_box, full_mosaic_w, full_mosaic_h,
                            n_colors_requested, use_slic_boundaries, colors_db_signature,
                            smooth_enabled=True, quantize_mode="superpixel",
                            cleanup_enabled=True, cleanup_fraction=0.5,
                            extra_smooth_enabled=False, extra_smooth_radius=2,
                            apply_sharpen=False):
    """
    apply_sharpen=False по умолчанию — UnsharpMask + Lanczos создают ореолы,
    которые при большом числе цветов превращаются в битые одиночные пиксели.
    Передавайте apply_sharpen=True только через явный UI-чекбокс.
    """
    img = Image.open(io.BytesIO(_img_bytes)).convert("RGB")
    colors_df = load_colors()

    work_img = img.crop(full_span_box) if full_span_box else img
    work_img = smooth_before_pixelize(work_img, enabled=smooth_enabled)
    work_img = smooth_super_strong(work_img, radius=extra_smooth_radius, enabled=extra_smooth_enabled)

    # BOX-resample: нет ringing-ореолов (в отличие от Lanczos)
    small_full = pixelate_exact(work_img, full_mosaic_w, full_mosaic_h,
                                 crop_box=None, apply_sharpen=apply_sharpen)
    working_palette, actual_unique_colors, k_used = build_working_palette(small_full, n_colors_requested, colors_df)

    segment_map_full = None
    if use_slic_boundaries:
        segment_map_full = compute_slic_segments(_img_bytes, full_mosaic_w, full_mosaic_h,
                                                    crop_box=full_span_box)

    pixel_ids_full = dither_to_lego_palette(small_full, working_palette, segment_map=segment_map_full,
                                              quantize_mode=quantize_mode, n_colors_requested=n_colors_requested)

    if cleanup_enabled:
        pixel_ids_full = majority_vote_cleanup_fast(pixel_ids_full, min_neighbor_fraction=cleanup_fraction,
                                                       enabled=True)

    # Реальный счётчик: unique значения в итоговой карте, а не размер рабочей палитры
    actual_used_colors = int(len(np.unique(pixel_ids_full)))

    return pixel_ids_full, working_palette, actual_used_colors, k_used
