"""
diagnostics.py — ДИАГНОСТИЧЕСКИЙ МОДУЛЬ (новый файл, ничего не заменяет).

Цель: показать, ГДЕ конкретно теряются детали и КАКИЕ цвета "съедают" палитру,
без изменения текущего пайплайна пикселизации. Подключается ПОСЛЕ
compute_pixel_ids_full() как отдельный анализ уже готового результата.

Как использовать в app.py:

    from diagnostics import build_diagnostics_report, render_diagnostics_ui

    pixel_ids_full, working_palette, actual_unique_colors, k_used = compute_pixel_ids_full(...)

    report = build_diagnostics_report(
        pixel_ids_full=pixel_ids_full,
        working_palette=working_palette,
        original_small_img=small_full,   # RGB-изображение ПОСЛЕ downscale, ДО квантизации
        colors_df=colors_df,
    )
    render_diagnostics_ui(report)

Ничего не меняет в логике квантизации — чисто read-only анализ.
"""
import numpy as np
import pandas as pd
import streamlit as st
from skimage.color import rgb2lab, deltaE_ciede2000


# ---------------------------------------------------------------------------
# 1. Базовая статистика: сколько цветов РЕАЛЬНО используется после чистки
# ---------------------------------------------------------------------------
def compute_actual_color_usage(pixel_ids_full, working_palette):
    """Честный подсчёт: сколько цветов физически осталось на канвасе
    ПОСЛЕ квантизации и cleanup, а не сколько было заявлено в палитре.
    Если actual_used < len(working_palette) — часть выбранных LEGO-цветов
    вообще не используется (палитра была подобрана неоптимально)."""
    unique_ids, counts = np.unique(pixel_ids_full, return_counts=True)
    total_studs = pixel_ids_full.size

    usage_df = pd.DataFrame({"color_id": unique_ids, "stud_count": counts})
    usage_df["share_pct"] = (usage_df["stud_count"] / total_studs * 100).round(2)
    usage_df = usage_df.merge(
        working_palette[["color_id", "name", "r", "g", "b"]] if "name" in working_palette.columns
        else working_palette[["color_id", "r", "g", "b"]],
        on="color_id", how="left"
    )
    usage_df = usage_df.sort_values("stud_count", ascending=False).reset_index(drop=True)

    n_allocated = len(working_palette)
    n_actually_used = len(unique_ids)
    n_wasted = n_allocated - n_actually_used

    return usage_df, {
        "n_allocated": n_allocated,
        "n_actually_used": n_actually_used,
        "n_wasted_slots": n_wasted,
        "waste_pct": round(n_wasted / max(1, n_allocated) * 100, 1),
    }


# ---------------------------------------------------------------------------
# 2. Доминирующие цвета — что "съедает" палитру
# ---------------------------------------------------------------------------
def top_dominant_colors(usage_df, top_n=10):
    """Топ-N цветов по покрытию площади. Если 2-3 цвета покрывают >60%
    канваса — палитра явно перекошена в сторону фона/волос, а не лица."""
    top = usage_df.head(top_n).copy()
    cum_share = top["share_pct"].cumsum()
    top["cumulative_pct"] = cum_share.round(2)
    return top


# ---------------------------------------------------------------------------
# 3. Локализация "потерянных" цветов: какие пиксели ИСХОДНОГО изображения
#    были ближе к цвету, который в итоге не попал в палитру / был перекрыт
# ---------------------------------------------------------------------------
def compute_quantization_error_map(original_small_img, pixel_ids_full, working_palette):
    """Для каждого стуба считает perceptual (CIEDE2000) расстояние между
    ИСХОДНЫМ цветом (до квантизации) и итоговым цветом LEGO-детали.
    Высокая ошибка = "здесь квантизация сильно искажает исходное фото".
    Именно тут физически видно, где именно "проседают детали"."""
    img_arr = np.array(original_small_img.convert("RGB")).astype(float)
    h, w, _ = img_arr.shape
    assert pixel_ids_full.shape == (h, w), "original_small_img и pixel_ids_full должны быть одного размера"

    orig_lab = rgb2lab(img_arr / 255.0)

    palette_lookup = working_palette.set_index("color_id")[["r", "g", "b"]]
    id_to_rgb = {cid: (row.r, row.g, row.b) for cid, row in palette_lookup.iterrows()}

    result_rgb = np.zeros((h, w, 3), dtype=float)
    for cid, rgb in id_to_rgb.items():
        mask = pixel_ids_full == cid
        result_rgb[mask] = rgb

    result_lab = rgb2lab(result_rgb / 255.0)
    error_map = deltaE_ciede2000(orig_lab.reshape(-1, 3), result_lab.reshape(-1, 3)).reshape(h, w)

    return error_map


def error_map_to_heatmap_image(error_map, max_error=25.0):
    """Конвертирует error_map (CIEDE2000, обычно 0-40) в RGB heatmap:
    зелёный = точное совпадение, жёлтый = среднее искажение, красный = сильное.
    max_error — верхняя граница нормализации (25 — эмпирически заметное искажение)."""
    from PIL import Image
    normalized = np.clip(error_map / max_error, 0, 1)

    heatmap = np.zeros((*error_map.shape, 3), dtype=np.uint8)
    heatmap[..., 0] = (normalized * 255).astype(np.uint8)                  # R растёт с ошибкой
    heatmap[..., 1] = ((1 - normalized) * 200).astype(np.uint8)            # G падает с ошибкой
    heatmap[..., 2] = 30

    return Image.fromarray(heatmap, mode="RGB")


# ---------------------------------------------------------------------------
# 4. Разбивка ошибки по зонам (face-region proxy без детектора лиц):
#    делим канвас на центральную зону (вероятно лицо/объект) и периферию
#    (вероятно фон/волосы) по эвристике — % от размера мозаики.
# ---------------------------------------------------------------------------
def compute_zone_error_breakdown(error_map, center_fraction=0.5):
    """Грубая эвристика без детектора лиц: сравнивает среднюю ошибку
    квантизации в центральной зоне канваса (где обычно лицо на портрете)
    против периферии (где обычно волосы/фон).
    Если центр_error >> периферия_error — значит квантизация "жертвует"
    лицом ради фона, что и является диагнозом текущей проблемы."""
    h, w = error_map.shape
    cy0 = int(h * (1 - center_fraction) / 2)
    cy1 = int(h * (1 + center_fraction) / 2)
    cx0 = int(w * (1 - center_fraction) / 2)
    cx1 = int(w * (1 + center_fraction) / 2)

    center_mask = np.zeros_like(error_map, dtype=bool)
    center_mask[cy0:cy1, cx0:cx1] = True

    center_error = error_map[center_mask].mean()
    periphery_error = error_map[~center_mask].mean()

    return {
        "center_mean_error": round(float(center_error), 2),
        "periphery_mean_error": round(float(periphery_error), 2),
        "center_worse_by_pct": round(
            (center_error - periphery_error) / max(0.01, periphery_error) * 100, 1
        ),
    }


# ---------------------------------------------------------------------------
# 5. Сводный отчёт — вызывается один раз, собирает всё вместе
# ---------------------------------------------------------------------------
def build_diagnostics_report(pixel_ids_full, working_palette, original_small_img, colors_df=None):
    usage_df, usage_stats = compute_actual_color_usage(pixel_ids_full, working_palette)
    top_colors = top_dominant_colors(usage_df, top_n=10)
    error_map = compute_quantization_error_map(original_small_img, pixel_ids_full, working_palette)
    zone_breakdown = compute_zone_error_breakdown(error_map)
    heatmap_img = error_map_to_heatmap_image(error_map)

    return {
        "usage_df": usage_df,
        "usage_stats": usage_stats,
        "top_colors": top_colors,
        "error_map": error_map,
        "zone_breakdown": zone_breakdown,
        "heatmap_img": heatmap_img,
        "mean_error_overall": round(float(error_map.mean()), 2),
        "max_error_overall": round(float(error_map.max()), 2),
        "p95_error": round(float(np.percentile(error_map, 95)), 2),
    }


# ---------------------------------------------------------------------------
# 6. UI-рендер для Streamlit — вставляется в отдельный st.expander()
# ---------------------------------------------------------------------------
def render_diagnostics_ui(report):
    """Рисует диагностическую панель. Вызывать после основного превью,
    внутри st.expander('🔬 Диагностика квантизации', expanded=False)."""
    st.markdown("### Сводка по использованию палитры")
    stats = report["usage_stats"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Выделено цветов", stats["n_allocated"])
    col2.metric("Реально используется", stats["n_actually_used"])
    col3.metric(
        "Пустых слотов", stats["n_wasted_slots"],
        delta=f"{stats['waste_pct']}% впустую",
        delta_color="inverse" if stats["n_wasted_slots"] > 0 else "off",
    )
    if stats["n_wasted_slots"] > 0:
        st.warning(
            f"{stats['n_wasted_slots']} цветов из палитры заняли слот, но не используются "
            f"на канвасе. Это значит K-Means выбрал кластеры, которые LEGO-палитра "
            f"'перетянула' на уже занятые соседние цвета — палитра расходуется неэффективно."
        )

    st.markdown("### Топ-10 цветов по занимаемой площади")
    st.caption(
        "Если 2-3 верхних цвета покрывают больше 50-60% канваса — "
        "палитра скорее всего перекошена в сторону фона/волос, а не лица."
    )
    display_cols = ["color_id", "stud_count", "share_pct", "cumulative_pct"]
    if "name" in report["top_colors"].columns:
        display_cols.insert(1, "name")
    st.dataframe(report["top_colors"][display_cols], use_container_width=True, hide_index=True)

    st.markdown("### Карта ошибки квантизации (CIEDE2000)")
    st.caption(
        "Зелёный = цвет практически не изменился при переходе к LEGO-палитре. "
        "Жёлтый/красный = сильное искажение — именно здесь пропадают детали лица "
        "(глаза, брови, губы, тени)."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(report["heatmap_img"], caption="Heatmap ошибки (красный = хуже)", use_container_width=True)
    with col_b:
        st.metric("Средняя ошибка (canvas)", report["mean_error_overall"])
        st.metric("95-й перцентиль ошибки", report["p95_error"])
        st.metric("Максимальная ошибка", report["max_error_overall"])

    st.markdown("### Центр (лицо) vs периферия (фон/волосы)")
    zb = report["zone_breakdown"]
    col_x, col_y, col_z = st.columns(3)
    col_x.metric("Ошибка в центре", zb["center_mean_error"])
    col_y.metric("Ошибка на периферии", zb["periphery_mean_error"])
    col_z.metric(
        "Центр хуже на",
        f"{zb['center_worse_by_pct']}%",
        delta_color="inverse" if zb["center_worse_by_pct"] > 0 else "normal",
    )
    if zb["center_worse_by_pct"] > 15:
        st.error(
            "Центральная зона (вероятно лицо) искажается заметно сильнее периферии. "
            "Это прямое подтверждение: build_working_palette() выделяет цвета "
            "пропорционально площади, а лицо — маленькая площадь по сравнению "
            "с волосами/фоном, поэтому теряет приоритет в K-Means. "
            "Решение: importance-weighted palette selection (см. Патч B)."
        )
    elif zb["center_worse_by_pct"] > 0:
        st.info(
            "Центр искажается немного сильнее периферии — умеренный перекос, "
            "но не критичный. Можно точечно усилить вес центра при выборе палитры."
        )
    else:
        st.success(
            "Центр не хуже периферии по ошибке — в этом случае проблема НЕ в "
            "распределении палитры, а скорее в resize/downscale стадии до квантизации."
        )
