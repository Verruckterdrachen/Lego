"""
pixelization.py — пайплайн пикселизации LEGO-мозаики.

Патч nearest-potts-v2:
- quantize_nearest_potts_fast: полностью векторизованный ICM без цикла по n_colors,
  принимает smoothness из UI, поддерживает importance_mask.
- build_working_palette: KMeans в Lab + importance-weighted sampling
  (приоритет пикселям на глазах/губах/контуре лица).
- compute_importance_mask: быстрая Laplacian-маска деталей лица.
- majority_vote_cleanup_fast: Sobel edge-protection уже был, оставлен.
- Все остальные режимы без изменений.
"""
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


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def resize_for_slic(img, max_dim=SLIC_MAX_DIM):
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    return img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)


def compute_importance_mask(img_arr):
    """Маска важных деталей (глаза, губы, контур).
    Используется для importance-weighted палитры и защиты от cleanup.
    Возвращает float32 в [0, 1]: 1.0 = важный пиксель, 0.0 = фон/кожа.

    Алгоритм: Laplacian of Gaussian на яркостном канале.
    Лапласиан максимален там, где резкий переход — край брови, зрачок, губа.
    """
    gray = np.mean(img_arr, axis=2).astype(np.float32)
    # LoG: сначала Gaussian blur, затем Laplacian
    blurred = ndimage.gaussian_filter(gray, sigma=1.0)
    lap = ndimage.laplace(blurred)
    lap_abs = np.abs(lap)
    # нормализуем в [0, 1]
    mx = lap_abs.max()
    if mx < 1e-6:
        return np.zeros_like(lap_abs)
    mask = lap_abs / mx
    # Порог: верхние 20% градиента — «важные» пиксели
    threshold = np.percentile(mask, 80)
    return (mask >= threshold).astype(np.float32)


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
    return work.resize((target_w, target_h), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# Построение рабочей палитры
# ---------------------------------------------------------------------------

def build_working_palette(small_img, n_colors, colors_df, importance_mask=None):
    """KMeans в Lab (perceptually uniform) + importance-weighted sampling.

    importance_mask: float32 (h, w) в [0, 1] — важные пиксели (глаза, губы, контур).
    Пиксели с importance=1 получают вес IMPORTANCE_WEIGHT_FACTOR при обучении KMeans,
    чтобы центроиды палитры не уходили только в «большие» зоны кожи/волос/фона.

    Без importance_mask поведение идентично предыдущей версии.
    """
    IMPORTANCE_WEIGHT_FACTOR = 6  # важные пиксели весят в 6 раз больше при KMeans

    arr_rgb = np.array(small_img.convert("RGB")).reshape(-1, 3).astype(float)
    arr_lab = rgb2lab((arr_rgb.reshape(-1, 1, 3) / 255.0)).reshape(-1, 3)

    # Строим вектор весов
    if importance_mask is not None:
        flat_mask = importance_mask.reshape(-1).astype(float)
        weights = 1.0 + flat_mask * (IMPORTANCE_WEIGHT_FACTOR - 1.0)
    else:
        weights = np.ones(len(arr_lab), dtype=float)

    # Importance-weighted sampling: дублируем важные пиксели
    n_unique_pixels = len(np.unique(arr_rgb, axis=0))
    k = max(1, min(n_colors, n_unique_pixels, len(colors_df)))

    # Строим взвешенную выборку для KMeans
    weights_norm = weights / weights.sum()
    n_sample = min(len(arr_lab), max(k * 50, 2000))  # ограничиваем скорость
    sample_idx = np.random.default_rng(42).choice(len(arr_lab), size=n_sample,
                                                    replace=True, p=weights_norm)
    arr_lab_sampled = arr_lab[sample_idx]

    km = KMeans(n_clusters=k, n_init=6, random_state=42).fit(arr_lab_sampled)
    centers_lab = km.cluster_centers_
    matched_color_ids = assign_unique_colors(centers_lab, colors_df)
    unique_ids = sorted(set(matched_color_ids))
    subset = colors_df[colors_df["color_id"].isin(unique_ids)].reset_index(drop=True)
    return subset, len(unique_ids), k


# ---------------------------------------------------------------------------
# Сглаживание
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Внутренние утилиты квантизации
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Режимы квантизации
# ---------------------------------------------------------------------------

def quantize_nearest(small_img, palette_df, segment_map=None):
    img_arr = np.array(small_img.convert("RGB")).astype(float)
    h, w, _ = img_arr.shape
    palette_lab = palette_df[["L", "a", "b_lab"]].values
    palette_color_ids = palette_df["color_id"].values
    lab_grid = rgb2lab(img_arr / 255.0).reshape(-1, 3)
    ids_flat, _ = _nearest_ids_for_pixels(lab_grid, palette_lab, palette_color_ids)
    return ids_flat.reshape(h, w).astype(np.int32)


def quantize_nearest_potts_fast(small_img, palette_df, segment_map=None,
                                  smoothness=2.0, n_iter=3, importance_mask=None):
    """Nearest-color + edge-aware Potts-регуляризация (полностью векторизованный ICM).

    Улучшения v2 по сравнению с предыдущей версией:
    - Убран цикл `for ci in range(n_colors)` — теперь agree считается одним
      scatter_add по всем пикселям сразу: O(h*w*4) вместо O(h*w*n_colors*4).
    - Параметр smoothness приходит из UI-слайдера (0.5 .. 6.0).
    - importance_mask: пиксели с importance=1 имеют увеличенный штраф за смену цвета
      (нельзя «перекрасить» зрачок в цвет кожи при регуляризации).
    - n_iter по умолчанию увеличен до 3 — портреты 64x80 обрабатываются за ~0.3 сек.

    smoothness: чем выше — тем агрессивнее сглаживание однородных зон.
      Рекомендуется 1.5–3.0 для портретов, 3.0–5.0 для иллюстраций/фона.
    importance_mask: float32 (h, w) в [0, 1]. Если передан,
      штраф за смену цвета на важных пикселях умножается на IMPORTANCE_LOCK.
    """
    IMPORTANCE_LOCK = 4.0  # важные пиксели сопротивляются регуляризации в 4 раза сильнее

    img_arr = np.array(small_img.convert("RGB")).astype(float)
    h, w, _ = img_arr.shape
    palette_lab = palette_df[["L", "a", "b_lab"]].values
    palette_color_ids = palette_df["color_id"].values
    n_colors = len(palette_df)

    # Data cost: CIEDE2000 до каждого цвета палитры, shape (h*w, n_colors)
    lab_grid = rgb2lab(img_arr / 255.0).reshape(-1, 3)
    all_dists = deltaE_ciede2000(lab_grid[:, None, :], palette_lab[None, :, :])
    data_cost = all_dists.reshape(h, w, n_colors)  # (h, w, K)

    # Начальное присвоение: nearest-color
    current = all_dists.argmin(axis=1).reshape(h, w)  # (h, w) — индексы в палитре

    # Importance: если маски нет — единичная
    if importance_mask is not None:
        imp = importance_mask.astype(np.float32)  # (h, w)
    else:
        imp = np.zeros((h, w), dtype=np.float32)

    # ICM-итерации
    for _ in range(n_iter):
        # agree[y, x, ci] = число соседей пикселя (y,x), имеющих метку ci
        # Считаем через scatter по текущим меткам — без цикла по цветам
        agree = np.zeros((h, w, n_colors), dtype=np.float32)

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # shifted_labels — метки соседей
            shifted = np.roll(current, shift=dy, axis=0) if dy != 0 else current.copy()
            shifted = np.roll(shifted, shift=dx, axis=1) if dx != 0 else shifted

            # Граничные строки/столбцы при roll оборачиваются — зануляем их
            if dy == -1:
                shifted[-1, :] = current[-1, :]  # нижний сосед верхнего края = сам пиксель
            elif dy == 1:
                shifted[0, :] = current[0, :]
            if dx == -1:
                shifted[:, -1] = current[:, -1]
            elif dx == 1:
                shifted[:, 0] = current[:, 0]

            # scatter: для каждого пикселя добавляем 1 в agree по метке соседа
            # Используем np.add.at с flat-индексом
            flat_pixel_idx = np.arange(h * w)
            flat_label = shifted.reshape(-1)
            flat_agree = agree.reshape(h * w, n_colors)
            np.add.at(flat_agree, (flat_pixel_idx, flat_label), 1.0)
            agree = flat_agree.reshape(h, w, n_colors)

        # smoothness_map: для важных пикселей штраф за смену цвета выше
        # => уменьшаем agree-бонус для «чужих» цветов на важных пикселях
        # Эффективно: agree *= (smoothness + imp * (IMPORTANCE_LOCK-1) * smoothness)
        smoothness_map = smoothness * (1.0 + imp * (IMPORTANCE_LOCK - 1.0))  # (h, w)

        # total_cost = data_cost - smoothness_map[:,:,None] * agree
        total_cost = data_cost - smoothness_map[:, :, np.newaxis] * agree
        current = total_cost.argmin(axis=2)  # (h, w)

    return palette_color_ids[current].astype(np.int32)


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


# ---------------------------------------------------------------------------
# Диспетчер режимов
# ---------------------------------------------------------------------------

def dither_to_lego_palette(small_img, palette_df, segment_map=None, quantize_mode="nearest_potts",
                             n_colors_requested=None, potts_smoothness=2.0, importance_mask=None):
    """Диспетчер режимов квантизации.

    Новые параметры:
      potts_smoothness — сила Potts-регуляризации (только для nearest_potts).
      importance_mask  — маска важных деталей (только для nearest_potts и build_working_palette).
    """
    if quantize_mode == "superpixel":
        n = n_colors_requested if n_colors_requested is not None else len(palette_df)
        return quantize_by_superpixel_map(small_img, palette_df, n, segment_map=segment_map)
    if quantize_mode == "nearest":
        return quantize_nearest(small_img, palette_df, segment_map=segment_map)
    if quantize_mode == "nearest_potts":
        return quantize_nearest_potts_fast(small_img, palette_df, segment_map=segment_map,
                                            smoothness=potts_smoothness, importance_mask=importance_mask)
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


# ---------------------------------------------------------------------------
# Финальная чистка одиночных выбросов
# ---------------------------------------------------------------------------

def majority_vote_cleanup_fast(pixel_ids, min_neighbor_fraction=0.5, iterations=1, enabled=True,
                                  importance_mask=None):
    """Чистка одиночных выбросов с двойной защитой:
    1. Sobel edge-mask: пиксели на границах объектов НЕ изменяются.
    2. importance_mask: важные пиксели (глаза, губы, контур) НЕ изменяются.

    Это гарантирует, что cleanup не удалит зрачок, веко, край губы или
    линию брови — только шумовые «одиночки» внутри однородных зон.
    """
    if not enabled:
        return pixel_ids

    edge_mask = _build_edge_mask(pixel_ids)

    # Дополнительная защита важных пикселей
    if importance_mask is not None:
        protect_mask = edge_mask | (importance_mask > 0.5)
    else:
        protect_mask = edge_mask

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
        cleaned[protect_mask] = result[protect_mask]
        result = cleaned
    return result


# ---------------------------------------------------------------------------
# Главная функция (вызывается из app.py)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_pixel_ids_full(_img_bytes, full_span_box, full_mosaic_w, full_mosaic_h,
                            n_colors_requested, use_slic_boundaries, colors_db_signature,
                            smooth_enabled=True, quantize_mode="nearest_potts",
                            cleanup_enabled=True, cleanup_fraction=0.5,
                            extra_smooth_enabled=False, extra_smooth_radius=2,
                            apply_sharpen=False, potts_smoothness=2.0,
                            use_importance_mask=True):
    """
    Основной пайплайн пикселизации.

    Новые параметры:
      potts_smoothness  — сила Potts-регуляризации (float, только для nearest_potts).
      use_importance_mask — включить importance-weighted палитру и защиту деталей в cleanup.

    apply_sharpen=False по умолчанию — UnsharpMask + Lanczos создают ореолы,
    которые при большом числе цветов превращаются в битые одиночные пиксели.
    Передавайте apply_sharpen=True только через явный UI-чекбокс.
    """
    img = Image.open(io.BytesIO(_img_bytes)).convert("RGB")
    colors_df = load_colors()

    work_img = img.crop(full_span_box) if full_span_box else img
    work_img = smooth_before_pixelize(work_img, enabled=smooth_enabled)
    work_img = smooth_super_strong(work_img, radius=extra_smooth_radius, enabled=extra_smooth_enabled)

    small_full = pixelate_exact(work_img, full_mosaic_w, full_mosaic_h,
                                 crop_box=None, apply_sharpen=apply_sharpen)

    # Importance mask — строим ДО палитры
    img_arr_small = np.array(small_full.convert("RGB")).astype(float)
    importance_mask = compute_importance_mask(img_arr_small) if use_importance_mask else None

    working_palette, actual_unique_colors, k_used = build_working_palette(
        small_full, n_colors_requested, colors_df,
        importance_mask=importance_mask if use_importance_mask else None
    )

    segment_map_full = None
    if use_slic_boundaries:
        segment_map_full = compute_slic_segments(_img_bytes, full_mosaic_w, full_mosaic_h,
                                                    crop_box=full_span_box)

    pixel_ids_full = dither_to_lego_palette(
        small_full, working_palette,
        segment_map=segment_map_full,
        quantize_mode=quantize_mode,
        n_colors_requested=n_colors_requested,
        potts_smoothness=potts_smoothness,
        importance_mask=importance_mask if use_importance_mask else None,
    )

    if cleanup_enabled:
        pixel_ids_full = majority_vote_cleanup_fast(
            pixel_ids_full,
            min_neighbor_fraction=cleanup_fraction,
            enabled=True,
            importance_mask=importance_mask if use_importance_mask else None,
        )

    # Реальный счётчик: unique значения в итоговой карте, а не размер рабочей палитры
    actual_used_colors = len(np.unique(pixel_ids_full))

    return pixel_ids_full, working_palette, actual_used_colors, k_used
