"""

TEST COMMIT — проверка связки git + GitHub, дата 21.09.2026

app.py — главный файл. Streamlit UI и оркестрация пайплайна.
Вся логика разнесена по модулям: config, colors_db, canvas_math, crop_geometry,
pixelization, frame_layout, rendering, pricing, ui_widgets.

Запуск:
    pip install -r requirements.txt
    streamlit run app.py
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    PIXEL_EDITOR_AVAILABLE = True
except ImportError:
    PIXEL_EDITOR_AVAILABLE = False

from config import (
    DARK_THEME_CSS, RECOMMENDED_MIN_COLORS, QUANTIZE_MODE_LABELS,
    MIN_MARKUP_PERCENT, MAX_MARKUP_PERCENT, DEFAULT_MARKUP_PERCENT,
    FRAME_CORNER_SIZE, TILE_CORNER_BLACK_PRICE_PER_PIECE_RUB, TILE_CORNER_WHITE_PRICE_PER_PIECE_RUB,
    PRICE_PER_PIECE_RUB,
)
from colors_db import init_demo_db, load_colors
from canvas_math import (
    get_scale_options, make_scale_label, find_best_canvas_shape, evaluate_frame_strategy,
    compute_backing_cost_as_pieces,
)
from crop_geometry import (
    compute_crop_box, get_crop_movable_axis, compute_crop_step_px, get_crop_offset_bounds,
    apply_crop_offset, compute_full_span_box, compute_full_mosaic_dims, crop_window_from_full,
)
from pixelization import compute_pixel_ids_full
from frame_layout import build_multi_ring_layout, compute_multi_ring_cost, overlay_frame_tile_layout
from rendering import (
    render_crop_visualization, render_final_preview, overlay_canvas_grid_and_center,
    click_to_stud_coords, apply_manual_color_overrides,
)
from pricing import (
    build_usage_report, compute_cost_summary, compute_selling_price, color_swatch_html,
    build_usage_table_html, export_pdf,
)
from ui_widgets import metric_card, slider_with_buttons, select_slider_with_buttons


st.set_page_config(page_title="LEGO-портреты", layout="wide")
st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
init_demo_db()
colors_df = load_colors()

st.title("Конструктор LEGO-портретов")

uploaded_file = st.file_uploader("Загрузите фото клиента", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    section1_container = st.container()

    st.markdown("### 2. Масштаб и оформление")
    base_options = get_scale_options(img.width, img.height)
    if not base_options:
        st.error("Не удалось подобрать масштаб для этого фото.")
        st.stop()
    st.caption(f"Найдено {len(base_options)} вариантов масштаба.")

    cols1, cols2, cols3 = st.columns(3)
    with cols1:
        labels = [make_scale_label(o["w_studs"], o["h_studs"]) for o in base_options]
        scale_idx = select_slider_with_buttons("Масштаб мозаики", key="scale_idx", options_list=labels)
        base_scale = base_options[scale_idx]
    with cols2:
        frame_mode = st.radio("Оформление", options=["Строго по плиткам", "С чёрной рамкой", "Рамка + паспарту"],
                               key="frame_mode")
    with cols3:
        frame_depth_choice = 0
        matting_depth_choice = 0
        if frame_mode != "Строго по плиткам":
            frame_depth_choice = slider_with_buttons("Толщина рамки (студов)", key="frame_depth_val",
                                                        min_val=1, max_val=10, default_val=2)
        if frame_mode == "Рамка + паспарту":
            matting_depth_choice = slider_with_buttons("Толщина паспарту (студов)", key="matting_depth_val",
                                                          min_val=1, max_val=10, default_val=2)

    base_canvas = find_best_canvas_shape(base_scale["w_studs"], base_scale["h_studs"])
    if base_canvas is None:
        st.error("Не удалось подобрать полотно для этого масштаба.")
        st.stop()

    total_depth = frame_depth_choice + matting_depth_choice
    if total_depth == 0:
        canvas_w, canvas_h = base_canvas["w"], base_canvas["h"]
        canvas_cost = base_canvas["cost"]
        mosaic_target_w, mosaic_target_h = canvas_w, canvas_h
    else:
        strategy_info = evaluate_frame_strategy(base_scale["w_studs"], base_scale["h_studs"],
                                                    total_depth, base_canvas=base_canvas)
        if strategy_info is None:
            st.error("Не удалось подобрать стратегию рамки для этого масштаба.")
            st.stop()

        col_strat1, col_strat2 = st.columns([2, 1])
        with col_strat1:
            strategy_label = st.radio(
                f"Стратегия рамки {'(рекомендуется Expand)' if strategy_info['recommendation'] == 'expand' else '(рекомендуется Shrink)'}",
                options=["Expand — расширить полотно", "Shrink canvas — ужать мозаику"],
                index=0 if strategy_info["recommendation"] == "expand" else 1,
                key="frame_strategy_radio", horizontal=True,
            )
            use_expand = strategy_label.startswith("Expand")
        if use_expand and strategy_info["expand_canvas"]:
            canvas_w, canvas_h = strategy_info["expand_canvas"]["w"], strategy_info["expand_canvas"]["h"]
            canvas_cost = strategy_info["expand_canvas"]["cost"]
            mosaic_target_w = canvas_w - 2 * total_depth
            mosaic_target_h = canvas_h - 2 * total_depth
        else:
            canvas_w, canvas_h = strategy_info["shrink_canvas"]["w"], strategy_info["shrink_canvas"]["h"]
            canvas_cost = strategy_info["shrink_canvas"]["cost"]
            mosaic_target_w = strategy_info["shrink_internal_w"]
            mosaic_target_h = strategy_info["shrink_internal_h"]
        with col_strat2:
            metric_card("Доп. стоимость Expand", f"{strategy_info['extra_cost']} руб")
            metric_card("Потеря контента Shrink", f"{strategy_info['content_loss_pct']}%")

    if mosaic_target_w <= 0 or mosaic_target_h <= 0:
        st.error("Слишком большая толщина рамки/паспарту для этого полотна. Уменьшите толщину.")
        st.stop()

    base_crop_box, needs_crop, crop_pct, ar_diff_pct = compute_crop_box(img.width, img.height,
                                                                          mosaic_target_w, mosaic_target_h)
    crop_axis, crop_slack_px = get_crop_movable_axis(base_crop_box, img.width, img.height)
    crop_step_px = compute_crop_step_px(base_crop_box, mosaic_target_w, mosaic_target_h, crop_axis)
    crop_signature = (img.width, img.height, mosaic_target_w, mosaic_target_h, crop_axis)

    if st.session_state.get("crop_offset_signature") != crop_signature:
        st.session_state["crop_offset_signature"] = crop_signature
        st.session_state["crop_offset_studs"] = 0
        st.session_state["manual_color_overrides"] = {}

    offset_min, offset_max = get_crop_offset_bounds(base_crop_box, img.width, img.height, crop_axis, crop_step_px)
    st.session_state["crop_offset_studs"] = max(offset_min, min(offset_max, st.session_state.get("crop_offset_studs", 0)))

    def move_crop(delta, axis_required):
        if crop_axis != axis_required:
            return
        current = st.session_state.get("crop_offset_studs", 0)
        lo, hi = get_crop_offset_bounds(base_crop_box, img.width, img.height, crop_axis, crop_step_px)
        st.session_state["crop_offset_studs"] = max(lo, min(hi, current + delta))

    crop_box, applied_offset_studs = apply_crop_offset(base_crop_box, img.width, img.height, crop_axis,
                                                          st.session_state.get("crop_offset_studs", 0),
                                                          crop_step_px, offset_min, offset_max)
    full_span_box = compute_full_span_box(base_crop_box, img.width, img.height, crop_axis)
    full_mosaic_w, full_mosaic_h = compute_full_mosaic_dims(full_span_box, crop_axis,
                                                               mosaic_target_w, mosaic_target_h, crop_step_px)

    with section1_container:
        st.markdown("### 1. Оригинал и область кропа")
        col_orig, col_crop = st.columns(2)
        with col_orig:
            st.image(img, caption=f"Оригинал {img.width}x{img.height} px", use_container_width=True)
        with col_crop:
            if needs_crop:
                crop_preview = render_crop_visualization(img, crop_box)
                st.image(crop_preview, caption=f"Кроп: сохранится {100-crop_pct:.1f}%, обрежется {crop_pct}%",
                          use_container_width=True)

                can_up = crop_axis == "y" and st.session_state["crop_offset_studs"] > offset_min
                can_down = crop_axis == "y" and st.session_state["crop_offset_studs"] < offset_max
                can_left = crop_axis == "x" and st.session_state["crop_offset_studs"] > offset_min
                can_right = crop_axis == "x" and st.session_state["crop_offset_studs"] < offset_max

                _, col_up, _ = st.columns([1, 1, 1])
                with col_up:
                    st.button("⬆️", key="crop_move_up", use_container_width=True, disabled=not can_up,
                              on_click=move_crop, args=(-1, "y"))
                col_left, col_reset, col_right = st.columns([1, 1, 1])
                with col_left:
                    st.button("⬅️", key="crop_move_left", use_container_width=True, disabled=not can_left,
                              on_click=move_crop, args=(-1, "x"))
                with col_reset:
                    st.button("Сброс", key="crop_move_reset", use_container_width=True,
                              disabled=st.session_state["crop_offset_studs"] == 0,
                              on_click=lambda: st.session_state.update(crop_offset_studs=0))
                with col_right:
                    st.button("➡️", key="crop_move_right", use_container_width=True, disabled=not can_right,
                              on_click=move_crop, args=(1, "x"))
                _, col_down, _ = st.columns([1, 1, 1])
                with col_down:
                    st.button("⬇️", key="crop_move_down", use_container_width=True, disabled=not can_down,
                              on_click=move_crop, args=(1, "y"))

                if applied_offset_studs != 0:
                    axis_label = "вертикали" if crop_axis == "y" else "горизонтали"
                    direction = ("вниз" if applied_offset_studs > 0 else "вверх") if crop_axis == "y" else \
                                ("вправо" if applied_offset_studs > 0 else "влево")
                    at_edge = applied_offset_studs in (offset_min, offset_max)
                    edge_note = " (край)" if at_edge else ""
                    st.caption(f"Сдвиг {abs(applied_offset_studs)} студов {direction} по {axis_label}{edge_note}")
            else:
                st.image(img, caption=f"Пропорции совпадают (откл. {ar_diff_pct:.1f}%) — кроп не нужен",
                         use_container_width=True)

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        max_possible_colors = min(len(colors_df), mosaic_target_w * mosaic_target_h)
        n_colors_requested = slider_with_buttons(f"Цветов в палитре (max {len(colors_df)})", key="n_colors_val",
                                                    min_val=1, max_val=max_possible_colors,
                                                    default_val=min(30, max_possible_colors))
    with col_p2:
        use_slic_boundaries = st.checkbox("Границы объектов (SLIC)", value=True)
        show_canvas_grid = st.checkbox("Сетка панелей 16×16", value=True)
    with col_p3:
        show_center_lines = st.checkbox("Центровые линии", value=True)
        preview_mode_key = "round" if st.radio("Форма плиток", ["Круглые", "Квадратные"], horizontal=True) == "Круглые" else "square"

    st.markdown("##### Пикселизация — режимы (переключаются чекбоксами)")
    st.caption("Три независимые стадии: сглаживание фото → способ квантизации цвета → чистка одиночных выбросов.")
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        smooth_enabled = st.checkbox("Сгладить фото перед пикселизацией (bilateral)", value=True,
                                       help="Убирает шум/зерно фото внутри однородных зон, сохраняя контуры объектов. "
                                            "Уменьшает число случайных ярких/тёмных студин ещё до квантизации.")
    with col_q2:
        quantize_mode_options = list(QUANTIZE_MODE_LABELS.keys())
        quantize_mode = st.selectbox("Способ квантизации цвета", options=quantize_mode_options,
                                        index=quantize_mode_options.index("superpixel"),
                                        format_func=lambda k: QUANTIZE_MODE_LABELS[k],
                                        help="Суперпиксели SLIC — крупные плавные зоны тона на коже/фоне, мелкие "
                                             "детали (глаза, брови) сохраняются отдельными сегментами.")
        extra_smooth_enabled = st.checkbox("🎨 Доп. сглаживание фона (осторожно с портретами!)", value=False,
                                             help="⚠️ Gaussian blur не различает объекты — при большом радиусе может "
                                                  "смазать глаза/брови. Радиус 1-2 безопасен для портретов.")
        extra_smooth_radius = 2
        if extra_smooth_enabled:
            extra_smooth_radius = st.slider("Радиус сглаживания", min_value=1, max_value=6, value=2, step=1)
    with col_q3:
        cleanup_enabled = st.checkbox("Убрать одиночные пиксели (majority-vote)", value=True,
                                        help="Финальная 'ручная подчистка': студина, резко отличающаяся от 8 соседей, "
                                             "заменяется цветом большинства соседей.")
        cleanup_fraction = 0.5
        if cleanup_enabled:
            cleanup_fraction = st.slider("Порог чистки (доля соседей)", min_value=0.3, max_value=0.8,
                                            value=0.5, step=0.05,
                                            help="Выше — чистка агрессивнее (заменяет больше студин).")

    show_frame_layout = False
    show_matting_layout = False
    if frame_depth_choice > 0 or matting_depth_choice > 0:
        col_tl1, col_tl2, col_tl3 = st.columns(3)
        with col_tl1:
            if frame_depth_choice > 0:
                show_frame_layout = st.checkbox(
                    f"Раскладка Tile для рамки ({frame_depth_choice} дет.)",
                    value=True, key="show_frame_layout_cb",
                )
        with col_tl2:
            if matting_depth_choice > 0:
                show_matting_layout = st.checkbox(
                    f"Раскладка Tile для паспарту ({matting_depth_choice} дет.)",
                    value=True, key="show_matting_layout_cb",
                )
        with col_tl3:
            show_tile_labels = st.checkbox("Подписи размеров деталей", value=True, key="show_tile_labels_cb")
    else:
        show_tile_labels = True

    if n_colors_requested < RECOMMENDED_MIN_COLORS:
        st.info(f"Рекомендуется не менее {RECOMMENDED_MIN_COLORS} цветов для качественной передачи оттенков.")

    total_studs_estimate = mosaic_target_w * mosaic_target_h
    if total_studs_estimate > 15000:
        st.warning(f"⚠️ Очень крупная мозаика ({mosaic_target_w}×{mosaic_target_h} = "
                   f"{total_studs_estimate} студов) — обработка может занять заметное время.")

    colors_db_signature = (len(colors_df), tuple(colors_df["color_id"].tolist()),
                            smooth_enabled, quantize_mode, cleanup_enabled, cleanup_fraction,
                            extra_smooth_enabled, extra_smooth_radius)

    with st.spinner("Обработка изображения..."):
        pixel_ids_full, working_palette, actual_unique_colors, k_used = compute_pixel_ids_full(
            file_bytes, full_span_box, full_mosaic_w, full_mosaic_h,
            n_colors_requested, use_slic_boundaries, colors_db_signature,
            smooth_enabled=smooth_enabled, quantize_mode=quantize_mode,
            cleanup_enabled=cleanup_enabled, cleanup_fraction=cleanup_fraction,
            extra_smooth_enabled=extra_smooth_enabled, extra_smooth_radius=extra_smooth_radius,
        )

        row0, row1, col0, col1 = crop_window_from_full(crop_box, full_span_box, img.width, img.height,
                                                          crop_axis, mosaic_target_w, mosaic_target_h,
                                                          full_mosaic_w, full_mosaic_h)
        pixel_ids_auto = pixel_ids_full[row0:row1, col0:col1]

        n_manual_overrides = len(st.session_state.get("manual_color_overrides", {}))
        if n_manual_overrides > 0:
            col_info, col_reset_all = st.columns([3, 1])
            with col_info:
                st.caption(f"Применено {n_manual_overrides} ручных правок цвета.")
            with col_reset_all:
                if st.button("Сбросить все правки", key="pixel_editor_reset_all_btn", use_container_width=True):
                    st.session_state["manual_color_overrides"] = {}
                    st.rerun()

        manual_overrides = st.session_state.get("manual_color_overrides", {})
        pixel_ids = apply_manual_color_overrides(pixel_ids_auto, manual_overrides)

        preview_img, used_cell_px = render_final_preview(pixel_ids, working_palette, canvas_w, canvas_h,
                                                            frame_depth_choice, matting_depth_choice,
                                                            preview_mode=preview_mode_key)
        preview_img = overlay_canvas_grid_and_center(preview_img, canvas_w, canvas_h, used_cell_px,
                                                        show_canvas_grid, show_center_lines)
        if show_frame_layout or show_matting_layout:
            preview_img = overlay_frame_tile_layout(preview_img, canvas_w, canvas_h, frame_depth_choice,
                                                       matting_depth_choice, used_cell_px,
                                                       show_frame=show_frame_layout, show_matting=show_matting_layout,
                                                       show_labels=show_tile_labels)

        usage_report = build_usage_report(pixel_ids, working_palette)
        cost_summary = compute_cost_summary(usage_report)

        frame_tiles_cost = 0.0
        matting_tiles_cost = 0.0
        frame_tiles_straight_count = frame_tiles_corner_count = 0
        matting_tiles_straight_count = matting_tiles_corner_count = 0
        if frame_depth_choice > 0:
            frame_rings = build_multi_ring_layout(canvas_w, canvas_h, frame_depth_choice)
            frame_tiles_cost, frame_tiles_straight_count, frame_tiles_corner_count = compute_multi_ring_cost(
                frame_rings, TILE_CORNER_BLACK_PRICE_PER_PIECE_RUB)
        if matting_depth_choice > 0:
            matting_w_studs = canvas_w - 2 * frame_depth_choice
            matting_h_studs = canvas_h - 2 * frame_depth_choice
            if matting_w_studs > 2 * FRAME_CORNER_SIZE and matting_h_studs > 2 * FRAME_CORNER_SIZE:
                matting_rings = build_multi_ring_layout(matting_w_studs, matting_h_studs, matting_depth_choice)
                matting_tiles_cost, matting_tiles_straight_count, matting_tiles_corner_count = compute_multi_ring_cost(
                    matting_rings, TILE_CORNER_WHITE_PRICE_PER_PIECE_RUB)

        st.markdown("### 3. Превью мозаики")
        legend_bits = []
        if show_canvas_grid:
            legend_bits.append("красная сетка = панели 16×16")
        if show_center_lines:
            legend_bits.append("голубые линии = центр")
        if frame_depth_choice > 0:
            legend_bits.append("чёрная рамка")
        if matting_depth_choice > 0:
            legend_bits.append("белое паспарту")
        if show_frame_layout:
            legend_bits.append("зелёный контур — раскладка Tile рамки")
        if show_matting_layout:
            legend_bits.append("синий контур — раскладка Tile паспарту")
        st.caption(" · ".join(legend_bits))

        col_prev, col_stats = st.columns(2)
        with col_prev:
            pixel_edit_mode = st.checkbox(
                "✏️ Режим редактирования цвета (клик по деталям)", value=True,
                help="Выключите, чтобы превью стало обычной картинкой — тогда доступно "
                     "'Открыть изображение в новой вкладке' по правому клику браузера.",
            )
            total_inset_for_click = frame_depth_choice + matting_depth_choice
            if PIXEL_EDITOR_AVAILABLE and pixel_edit_mode:
                click_result = streamlit_image_coordinates(preview_img, key="pixel_editor_click", use_column_width=True)
                st.caption("Клик по детали мозаики, чтобы изменить её цвет вручную.")
                if click_result is not None:
                    displayed_w = click_result.get("width", preview_img.width) or preview_img.width
                    scale_factor = preview_img.width / displayed_w
                    real_click_x = click_result["x"] * scale_factor
                    real_click_y = click_result["y"] * scale_factor
                    clicked_stud = click_to_stud_coords(real_click_x, real_click_y, used_cell_px,
                                                           total_inset_for_click, mosaic_target_w, mosaic_target_h)
                    if clicked_stud is not None:
                        st.session_state["pixel_editor_selected"] = clicked_stud
                    else:
                        st.info("Клик попал на рамку/паспарту — выберите деталь внутри самой мозаики.")

                selected_stud = st.session_state.get("pixel_editor_selected")
                if selected_stud is not None:
                    sel_row, sel_col = selected_stud
                    if 0 <= sel_row < pixel_ids.shape[0] and 0 <= sel_col < pixel_ids.shape[1]:
                        current_color_id = int(pixel_ids[sel_row, sel_col])
                        current_row = colors_df[colors_df.color_id == current_color_id]
                        if not current_row.empty:
                            current_row = current_row.iloc[0]
                            col_cur1, col_cur2 = st.columns([1, 3])
                            with col_cur1:
                                st.markdown(color_swatch_html(current_row["rgb_hex"], size=32), unsafe_allow_html=True)
                            with col_cur2:
                                st.caption(f"Деталь ({sel_col}, {sel_row}): #{current_color_id} {current_row['name_ru']}")

                            color_options = colors_df.sort_values("color_id")
                            option_labels = [f"{r.color_id} — {r.name_ru}" for _, r in color_options.iterrows()]
                            option_ids = color_options["color_id"].tolist()
                            default_idx = option_ids.index(current_color_id) if current_color_id in option_ids else 0
                            new_color_choice = st.selectbox(
                                "Новый цвет детали", options=range(len(option_labels)),
                                format_func=lambda i: option_labels[i], index=default_idx,
                                key="pixel_editor_color_select",
                            )
                            new_color_id = option_ids[new_color_choice]

                            col_apply1, col_apply2 = st.columns(2)
                            with col_apply1:
                                if st.button("Применить", key="pixel_editor_apply_btn", use_container_width=True):
                                    overrides = st.session_state.get("manual_color_overrides", {})
                                    overrides[(sel_row, sel_col)] = new_color_id
                                    st.session_state["manual_color_overrides"] = overrides
                                    st.rerun()
                            with col_apply2:
                                has_override = (sel_row, sel_col) in st.session_state.get("manual_color_overrides", {})
                                if st.button("Сбросить эту деталь", key="pixel_editor_reset_one_btn",
                                              use_container_width=True, disabled=not has_override):
                                    overrides = st.session_state.get("manual_color_overrides", {})
                                    overrides.pop((sel_row, sel_col), None)
                                    st.session_state["manual_color_overrides"] = overrides
                                    st.rerun()
            else:
                st.image(preview_img, caption="Превью мозаики", use_container_width=True)
                if not PIXEL_EDITOR_AVAILABLE:
                    st.caption("Для точечного редактирования цвета установите: pip install streamlit-image-coordinates")
                else:
                    st.caption("Обычный режим просмотра — доступно 'Открыть изображение в новой вкладке'.")

        with col_stats:
            mc1, mc2 = st.columns(2)
            with mc1:
                metric_card("Полотно", f"{canvas_w*0.8:.1f}×{canvas_h*0.8:.1f} см", "accent")
                metric_card("Мозаика Ш×В", f"{mosaic_target_w}×{mosaic_target_h} ({mosaic_target_w*mosaic_target_h} дет.)")
                metric_card("Цветов использовано", actual_unique_colors)
            with mc2:
                metric_card("Толщина рамки+паспарту", f"{frame_depth_choice + matting_depth_choice} дет.")
                if needs_crop:
                    metric_card("Обрезано фото", f"{crop_pct}%", "warn")

            st.markdown("#### Себестоимость")
            backing_cost_as_pieces = compute_backing_cost_as_pieces(canvas_w, canvas_h)
            mc3, mc4 = st.columns(2)
            with mc3:
                metric_card("Полотно LEGO Art (реальная цена)", f"{canvas_cost:.2f} руб", "money")
            with mc4:
                metric_card("Аналитика: подложка по цене детали", f"{backing_cost_as_pieces:.2f} руб", "neutral")
            st.caption(f"Полотно {canvas_w}×{canvas_h} студов, аналитика по цене 1 детали "
                       f"({PRICE_PER_PIECE_RUB:.3f} руб) — только для сравнения, не заменяет реальную закупку.")

            if frame_depth_choice > 0 or matting_depth_choice > 0:
                st.markdown("#### Tile-раскладка рамки/паспарту")
                mc5, mc6 = st.columns(2)
                with mc5:
                    if frame_depth_choice > 0:
                        metric_card(f"Рамка ({frame_depth_choice} дет.), деталей Tile",
                                    f"{frame_tiles_straight_count + frame_tiles_corner_count}")
                with mc6:
                    if matting_depth_choice > 0:
                        metric_card(f"Паспарту ({matting_depth_choice} дет.), деталей Tile",
                                    f"{matting_tiles_straight_count + matting_tiles_corner_count}")

        st.markdown("### 4. Расход деталей и итоговая цена")
        col_d1, col_d2 = st.columns([3, 2])
        with col_d1:
            st.markdown(build_usage_table_html(usage_report), unsafe_allow_html=True)
        with col_d2:
            frame_matting_tiles_total_cost = round(frame_tiles_cost + matting_tiles_cost, 2)
            total_material_cost = round(cost_summary["total_cost_by_usage"] + canvas_cost + frame_matting_tiles_total_cost, 2)
            st.markdown("#### Три составляющие себестоимости")
            metric_card("1. Мозаика (детали по цветам)", f"{cost_summary['total_cost_by_usage']:.2f} руб")
            metric_card("2. Полотно LEGO Art", f"{canvas_cost:.2f} руб")
            metric_card("3. Tile-рамка/паспарту", f"{frame_matting_tiles_total_cost:.2f} руб")
            metric_card("Итоговая себестоимость", f"{total_material_cost:.2f} руб", "money")

            markup_percent = slider_with_buttons("Наценка, %", key="markup_percent_val",
                                                    min_val=MIN_MARKUP_PERCENT, max_val=MAX_MARKUP_PERCENT,
                                                    default_val=DEFAULT_MARKUP_PERCENT, step=5)
            selling_price, profit = compute_selling_price(total_material_cost, markup_percent)
            metric_card(f"Цена продажи ({markup_percent}%)", f"{selling_price:.2f} руб", "price")
            metric_card("Прибыль", f"{profit:.2f} руб", "money")

            pdf_buf = export_pdf(pixel_ids, colors_df, usage_report, canvas_w, canvas_h,
                                  frame_depth_choice, matting_depth_choice)
            st.download_button("📄 Скачать схему сборки PDF", data=pdf_buf, file_name="lego_assembly_scheme.pdf",
                                mime="application/pdf")

else:
    st.info("Загрузите фото клиента, чтобы начать расчёт.")
