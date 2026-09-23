"""
instructions_export.py — генерация печатных карточек-инструкций по сборке
LEGO-мозаики, отдельно для каждой панели 16x16 (деталь 65803).

Каждая карточка формата A5 (148x210 мм, "половина A4") содержит:
  1. Мини-план всех панелей полотна: уменьшенная копия ИТОГОВОЙ мозаики.
     Текущая панель затемнена (50% чёрного) с жирным белым номером,
     остальные — полупрозрачные, номер крупным чёрным шрифтом.
  2. Легенду цветов данной панели: горизонтальный flow-layout, прижата к
     правому краю листа. Размер кружка ФИКСИРОВАН (не зависит от n_colors).
     Правая колонка выравнивается АДАПТИВНО по ширине текста "xN": сначала
     измеряется самая широкая подпись именно в правой колонке (среди всех
     строк легенды — 1-, 2- или 3-значные числа могут отличаться), и весь
     блок легенды сдвигается так, чтобы правый край этой самой широкой
     подписи точно касался правого края листа. Короткие подписи (x3) в
     остальных строках правой колонки автоматически получают больше
     свободного места после текста — сам кружок/номер не двигается,
     сдвигается только якорная точка колонки. Это устраняет визуальный
     "недотяг" при однозначных числах и риск вылета текста за край листа
     при трёхзначных числах.
  3. Пронумерованную сетку самой панели, центрированную по вертикали в
     оставшемся пространстве между шапкой и нижним краем листа. Круг детали
     касается стенок ячейки впритык. Координаты и номер цвета — одним
     увеличенным шрифтом. Толстые красные направляющие по границам блоков
     4x4 (поверх коннекторов, до края плиты), короткие чёрные
     Technic-коннекторы, выступающие в основном НАРУЖУ от края плиты, на
     гранях, где есть соседняя панель — три позиции, симметричные
     относительно центра: между студами (3,4), (8,9), (13,14).

preview_mode: "round"  — кружки на чёрной подложке (имитация LEGO-стойки),
              "square" — сплошные квадраты без подложки.

Все данные (pixel_ids, palette_df) уже посчитаны на этапе превью/экспорта
основной мозаики (см. pricing.export_svg) — этот модуль их не пересчитывает,
только нарезает на панели и переверстывает в отдельный печатный SVG.
"""

import math
from config import UNIT_STUDS

PAGE_W_MM = 148.0
PAGE_H_MM = 210.0
MARGIN_MM = 8.0

MINI_PLAN_FRACTION = 0.40
HEADER_GAP_MM = 5.0
COORD_LABEL_MM = 7.0
HEADER_HEIGHT_MM = 48.0

LEGEND_FIXED_SWATCH_MM = 6.5
LEGEND_MAX_COLS = 5
LEGEND_ROW_GAP_MM = 1.3
LEGEND_COL_GAP_MM = 1.6
LEGEND_TEXT_GAP_MM = 0.5

# Эмпирический коэффициент ширины символа для "&#215;N" (умножение + цифры)
# при данном font-size, шрифт Arial Bold. Используется, чтобы точно измерить
# самую широкую подпись в правой колонке легенды и выровнять её впритык к
# правому краю листа, вместо использования единого предположения ширины
# текста для всех строк (что раньше давало лишний зазор у "x3" и риск
# вылета у "x132").
TEXT_CHAR_WIDTH_FACTOR = 0.62

BLOCK_SIZE = 4
CONNECTOR_GAPS = [(3, 4), (8, 9), (13, 14)]     # симметричны относительно центра 8.5
CONNECTOR_OUTWARD_FRACTION = 0.55
CONNECTOR_INWARD_FRACTION = 0.0
CONNECTOR_THICKNESS_FRACTION = 0.34

MINI_INACTIVE_OPACITY = 0.30
MINI_ACTIVE_DARKEN_OPACITY = 0.50
MINI_GRID_STROKE = "#1a1a1a"


def _luminance(hex_color):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _text_color_for_bg(hex_color):
    return "#111111" if _luminance(hex_color) > 140 else "#f5f5f5"


def slice_panels(pixel_ids, unit_studs=UNIT_STUDS):
    """
    Разбивает полную матрицу pixel_ids на панели unit_studs x unit_studs.
    Нумерация — слева-направо, сверху-вниз, с 1, без пропусков.
    """
    h, w = pixel_ids.shape
    panels_y = math.ceil(h / unit_studs)
    panels_x = math.ceil(w / unit_studs)
    panels = []
    idx = 1
    for prow in range(panels_y):
        for pcol in range(panels_x):
            y0, y1 = prow * unit_studs, min((prow + 1) * unit_studs, h)
            x0, x1 = pcol * unit_studs, min((pcol + 1) * unit_studs, w)
            sub = pixel_ids[y0:y1, x0:x1]
            panels.append({
                "index": idx, "row": prow, "col": pcol,
                "panels_x": panels_x, "panels_y": panels_y, "matrix": sub,
            })
            idx += 1
    return panels


def _panel_color_counts(matrix):
    """color_id -> count, только для цветов, реально присутствующих в панели."""
    unique, counts = _np_unique_counts(matrix)
    return sorted(zip(unique, counts), key=lambda t: t[0])


def _np_unique_counts(matrix):
    import numpy as np
    unique, counts = np.unique(matrix, return_counts=True)
    return unique.tolist(), counts.tolist()


def _svg_text(x, y, content, size, color, anchor="middle", weight="normal"):
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}" dominant-baseline="central">{content}</text>')


def _mm_to_px(mm_val, px_per_mm):
    return mm_val * px_per_mm


def _estimate_count_text_width(count, font_size):
    """
    Оценивает реальную ширину подписи "xN" (символ умножения + цифры count)
    при данном font_size — используется для адаптивного выравнивания правой
    колонки легенды по фактической ширине текста, а не по единому
    предположению для всех строк.
    """
    n_chars = 1 + len(str(count))  # "x" + цифры
    return n_chars * font_size * TEXT_CHAR_WIDTH_FACTOR


def _swatch_shape(cx, cy, size, hexcolor, preview_mode):
    """
    Рисует одну ячейку цвета в ЛЕГЕНДЕ — тем же способом, что ячейку в
    основной сетке: круг касается стенок своей ячейки впритык
    (r = size/2 - 0.3). Квадрат — сплошной без подложки.
    """
    els = []
    if preview_mode == "round":
        r = size / 2 - 0.3
        els.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" '
                    f'fill="{hexcolor}" stroke="#000000" stroke-width="0.5"/>')
    else:
        els.append(f'<rect x="{cx - size/2:.2f}" y="{cy - size/2:.2f}" '
                    f'width="{size:.2f}" height="{size:.2f}" fill="{hexcolor}" '
                    f'stroke="#333333" stroke-width="0.5"/>')
    return els


def build_instruction_card_svg(panel, palette_by_id, full_pixel_ids,
                                preview_mode="round", px_per_mm=3.78,
                                page_w_mm=PAGE_W_MM, page_h_mm=PAGE_H_MM,
                                margin_mm=MARGIN_MM):
    """
    Строит одну печатную карточку-инструкцию для одной панели.
    """
    idx = panel["index"]
    prow, pcol = panel["row"], panel["col"]
    panels_x, panels_y = panel["panels_x"], panel["panels_y"]
    matrix = panel["matrix"]
    rows_m, cols_m = matrix.shape

    W = _mm_to_px(page_w_mm, px_per_mm)
    H = _mm_to_px(page_h_mm, px_per_mm)
    M = _mm_to_px(margin_mm, px_per_mm)

    content_w = W - 2 * M
    content_h = H - 2 * M
    x0, y0 = M, M

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.2f}" height="{H:.2f}" '
        f'viewBox="0 0 {W:.2f} {H:.2f}">',
        f'<rect x="0" y="0" width="{W:.2f}" height="{H:.2f}" fill="#ffffff"/>',
    ]

    # ---- геометрия мини-плана ----
    mini_size = content_w * MINI_PLAN_FRACTION
    cell_mini = mini_size / max(panels_x, panels_y)
    mini_w = cell_mini * panels_x
    mini_h = cell_mini * panels_y

    header_h = _mm_to_px(HEADER_HEIGHT_MM, px_per_mm)

    counts = _panel_color_counts(matrix)
    n_colors = len(counts)

    legend_area_x0 = x0 + mini_w + _mm_to_px(6, px_per_mm)
    legend_area_w = (x0 + content_w) - legend_area_x0

    # ---- легенда: фиксированный размер свотча, единая сетка колонок ----
    swatch = _mm_to_px(LEGEND_FIXED_SWATCH_MM, px_per_mm)
    col_gap_px = _mm_to_px(LEGEND_COL_GAP_MM, px_per_mm)
    text_gap_px = _mm_to_px(LEGEND_TEXT_GAP_MM, px_per_mm)
    legend_font = max(7, swatch * 0.42)
    text_w_assumed = swatch * 0.85 + text_gap_px
    cell_w_tight = swatch + text_gap_px + text_w_assumed + col_gap_px
    row_gap_px = _mm_to_px(LEGEND_ROW_GAP_MM, px_per_mm)
    legend_row_h = swatch + row_gap_px

    cols_by_width = max(1, int(legend_area_w // cell_w_tight))
    cols_per_row = max(1, min(LEGEND_MAX_COLS, cols_by_width))
    rows_used = math.ceil(n_colors / cols_per_row) if cols_per_row and n_colors else 1

    legend_cell_w = legend_area_w / cols_per_row
    LEGEND_RIGHT_SHIFT_MM = 3.0
    legend_right_edge = x0 + content_w + _mm_to_px(LEGEND_RIGHT_SHIFT_MM, px_per_mm)

    # ---- адаптивное выравнивание правой колонки по фактической ширине
    # текста "xN": находим МАКСИМАЛЬНУЮ реальную ширину подписи среди ВСЕХ
    # элементов, которые окажутся в самой правой колонке (индекс
    # cols_per_row - 1), и используем именно её, чтобы вычислить точный
    # отступ якоря колонок от правого края листа. Более короткие подписи
    # (x3) в той же колонке автоматически получают больше свободного места
    # после текста — сам якорь колонки сдвигается вправо ровно настолько,
    # чтобы САМАЯ ДЛИННАЯ подпись касалась края, без риска вылета у x132
    # и без лишнего пустого зазора у x3, если он единственный в колонке.
    max_text_w_in_last_col = 0.0
    for i, (color_id, count) in enumerate(counts):
        row_i = i // cols_per_row
        n_in_this_row = min(cols_per_row, n_colors - row_i * cols_per_row)
        col_offset = cols_per_row - n_in_this_row
        col_i = i % cols_per_row
        is_last_col = (col_offset + col_i) == (cols_per_row - 1)
        if is_last_col:
            tw = _estimate_count_text_width(count, legend_font)
            max_text_w_in_last_col = max(max_text_w_in_last_col, tw)

    # правый край самой широкой подписи в правой колонке должен совпасть с
    # legend_right_edge; подпись начинается после круга+text_gap, поэтому
    # ширина одной "виртуальной" колонки = swatch + text_gap + фактический max_text_w
    actual_last_col_content_w = swatch + text_gap_px + max_text_w_in_last_col
    legend_grid_x0 = legend_right_edge - col_gap_px - actual_last_col_content_w \
        - (cols_per_row - 1) * legend_cell_w

    def _col_x(col_idx):
        return legend_grid_x0 + col_idx * legend_cell_w

    # ---- сетка: считаем размер, затем центрируем блок по вертикали ----
    coord_band_px = _mm_to_px(COORD_LABEL_MM, px_per_mm)
    area_top = y0 + header_h + _mm_to_px(HEADER_GAP_MM, px_per_mm)
    area_bottom = y0 + content_h
    area_avail_w = content_w - coord_band_px
    area_avail_h = area_bottom - area_top - coord_band_px

    cell = min(area_avail_w / cols_m, area_avail_h / rows_m)
    grid_w = cell * cols_m
    grid_h = cell * rows_m

    block_h = coord_band_px + grid_h
    block_top = area_top + (area_bottom - area_top - block_h) / 2  # центрирование

    grid_top = block_top + coord_band_px
    grid_left = x0 + (content_w - grid_w) / 2
    grid_bottom = grid_top + grid_h
    grid_right = grid_left + grid_w

    unified_font = max(9, cell * 0.42)

    # ---- 1. мини-план: реальное уменьшенное превью всей мозаики ----
    fh, fw = full_pixel_ids.shape
    px_per_stud_mini = cell_mini / UNIT_STUDS

    parts.append('<g data-block="mini-plan">')
    for r in range(fh):
        for c in range(fw):
            this_prow = r // UNIT_STUDS
            this_pcol = c // UNIT_STUDS
            is_current_panel = (this_prow == prow and this_pcol == pcol)
            color_id = int(full_pixel_ids[r, c])
            hexcolor = "#" + palette_by_id.get(color_id, "808080").lstrip("#")
            mx = x0 + c * px_per_stud_mini
            my = y0 + r * px_per_stud_mini
            opacity = "1" if is_current_panel else str(MINI_INACTIVE_OPACITY)
            parts.append(
                f'<rect x="{mx:.2f}" y="{my:.2f}" width="{px_per_stud_mini:.2f}" '
                f'height="{px_per_stud_mini:.2f}" fill="{hexcolor}" opacity="{opacity}"/>'
            )

    active_x0 = x0 + pcol * cell_mini
    active_y0 = y0 + prow * cell_mini
    parts.append(
        f'<rect x="{active_x0:.2f}" y="{active_y0:.2f}" width="{cell_mini:.2f}" '
        f'height="{cell_mini:.2f}" fill="#000000" opacity="{MINI_ACTIVE_DARKEN_OPACITY}"/>'
    )

    for r in range(panels_y):
        for c in range(panels_x):
            cx0 = x0 + c * cell_mini
            cy0 = y0 + r * cell_mini
            parts.append(
                f'<rect x="{cx0:.2f}" y="{cy0:.2f}" width="{cell_mini:.2f}" '
                f'height="{cell_mini:.2f}" fill="none" stroke="{MINI_GRID_STROKE}" '
                f'stroke-width="0.7"/>'
            )
    parts.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{mini_w:.2f}" '
                  f'height="{mini_h:.2f}" fill="none" stroke="{MINI_GRID_STROKE}" '
                  f'stroke-width="1.2"/>')

    for r in range(panels_y):
        for c in range(panels_x):
            cx0 = x0 + c * cell_mini
            cy0 = y0 + r * cell_mini
            is_current = (r == prow and c == pcol)
            panel_num = r * panels_x + c + 1
            if is_current:
                parts.append(_svg_text(cx0 + cell_mini / 2, cy0 + cell_mini / 2,
                                        str(panel_num), max(14, cell_mini * 0.46),
                                        "#ffffff", weight="bold"))
                parts.append(f'<rect x="{cx0:.2f}" y="{cy0:.2f}" width="{cell_mini:.2f}" '
                              f'height="{cell_mini:.2f}" fill="none" stroke="#ff2d2d" '
                              f'stroke-width="2"/>')
            else:
                parts.append(_svg_text(cx0 + cell_mini / 2, cy0 + cell_mini / 2,
                                        str(panel_num), max(12, cell_mini * 0.40),
                                        "#000000", weight="bold"))
    parts.append('</g>')

    # ---- 2. легенда: единая сетка колонок, адаптивный якорь правой
    # колонки по фактической ширине "xN" ----
    parts.append('<g data-block="legend">')
    for i, (color_id, count) in enumerate(counts):
        row_i = i // cols_per_row
        col_i = i % cols_per_row
        n_in_this_row = min(cols_per_row, n_colors - row_i * cols_per_row)
        col_offset = cols_per_row - n_in_this_row
        lx = _col_x(col_offset + col_i)
        ly = y0 + row_i * legend_row_h

        hexcolor = "#" + palette_by_id.get(color_id, "808080").lstrip("#")
        text_color = _text_color_for_bg(hexcolor)
        cx = lx + swatch / 2
        cy = ly + swatch / 2
        parts.extend(_swatch_shape(cx, cy, swatch, hexcolor, preview_mode))
        parts.append(_svg_text(cx, cy, str(color_id),
                                legend_font, text_color, weight="bold"))
        parts.append(_svg_text(lx + swatch + text_gap_px, cy,
                                f"&#215;{count}", legend_font, "#111111",
                                anchor="start"))
    parts.append('</g>')

    # ---- 3. сетка панели (блок центрирован по вертикали) ----
    parts.append('<g data-block="coords">')
    for c in range(cols_m):
        cx = grid_left + c * cell + cell / 2
        cy = grid_top - coord_band_px / 2
        parts.append(_svg_text(cx, cy, str(c + 1), unified_font, "#333333", weight="bold"))
    for r in range(rows_m):
        cx = grid_left - coord_band_px / 2
        cy = grid_top + r * cell + cell / 2
        parts.append(_svg_text(cx, cy, str(r + 1), unified_font, "#333333", weight="bold"))
    parts.append('</g>')

    parts.append('<g data-block="grid">')
    for r in range(rows_m):
        for c in range(cols_m):
            color_id = int(matrix[r, c])
            hexcolor = "#" + palette_by_id.get(color_id, "808080").lstrip("#")
            gx = grid_left + c * cell
            gy = grid_top + r * cell
            cx, cy = gx + cell / 2, gy + cell / 2
            if preview_mode == "round":
                parts.append(f'<rect x="{gx:.2f}" y="{gy:.2f}" width="{cell:.2f}" '
                              f'height="{cell:.2f}" fill="#000000"/>')
                r_dot = cell / 2 - 0.3
                parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r_dot:.2f}" '
                              f'fill="{hexcolor}" stroke="#00000055" stroke-width="0.3"/>')
            else:
                parts.append(f'<rect x="{gx:.2f}" y="{gy:.2f}" width="{cell:.2f}" '
                              f'height="{cell:.2f}" fill="{hexcolor}" '
                              f'stroke="#00000030" stroke-width="0.35"/>')
            text_color = _text_color_for_bg(hexcolor)
            parts.append(_svg_text(cx, cy, str(color_id),
                                    unified_font, text_color, weight="bold"))
    for c in range(cols_m + 1):
        gx = grid_left + c * cell
        parts.append(f'<line x1="{gx:.2f}" y1="{grid_top:.2f}" x2="{gx:.2f}" '
                      f'y2="{grid_bottom:.2f}" stroke="#8a8a8a" stroke-width="0.35" opacity="0.7"/>')
    for r in range(rows_m + 1):
        gy = grid_top + r * cell
        parts.append(f'<line x1="{grid_left:.2f}" y1="{gy:.2f}" x2="{grid_right:.2f}" '
                      f'y2="{gy:.2f}" stroke="#8a8a8a" stroke-width="0.35" opacity="0.7"/>')
    parts.append('</g>')

    # ---- коннекторы Technic (рисуются ДО красных направляющих) ----
    has_right_neighbor = (pcol + 1) < panels_x
    has_left_neighbor = pcol > 0
    has_bottom_neighbor = (prow + 1) < panels_y
    has_top_neighbor = prow > 0

    outward = cell * CONNECTOR_OUTWARD_FRACTION
    inward = cell * CONNECTOR_INWARD_FRACTION
    conn_len = outward + inward
    conn_thick = cell * CONNECTOR_THICKNESS_FRACTION

    parts.append('<g data-block="connectors">')

    def _draw_h_edge_connector(gap_before, gap_after, on_right):
        if gap_after > rows_m:
            return
        y_boundary = grid_top + gap_before * cell
        x_rect = (grid_right - inward) if on_right else (grid_left - outward)
        y_top = y_boundary - conn_thick / 2
        y_bot = y_boundary + conn_thick / 2
        r = conn_thick * 0.3
        if on_right:
            # прилегает к плите СЛЕВА (x_rect) -> левые 2 угла прямые, правые скруглены
            path = (f'M {x_rect:.2f} {y_top:.2f} '
                    f'L {x_rect+conn_len-r:.2f} {y_top:.2f} '
                    f'Q {x_rect+conn_len:.2f} {y_top:.2f} {x_rect+conn_len:.2f} {y_top+r:.2f} '
                    f'L {x_rect+conn_len:.2f} {y_bot-r:.2f} '
                    f'Q {x_rect+conn_len:.2f} {y_bot:.2f} {x_rect+conn_len-r:.2f} {y_bot:.2f} '
                    f'L {x_rect:.2f} {y_bot:.2f} Z')
        else:
            # прилегает к плите СПРАВА (x_rect+conn_len) -> правые 2 угла прямые, левые скруглены
            path = (f'M {x_rect+r:.2f} {y_top:.2f} '
                    f'L {x_rect+conn_len:.2f} {y_top:.2f} '
                    f'L {x_rect+conn_len:.2f} {y_bot:.2f} '
                    f'L {x_rect+r:.2f} {y_bot:.2f} '
                    f'Q {x_rect:.2f} {y_bot:.2f} {x_rect:.2f} {y_bot-r:.2f} '
                    f'L {x_rect:.2f} {y_top+r:.2f} '
                    f'Q {x_rect:.2f} {y_top:.2f} {x_rect+r:.2f} {y_top:.2f} Z')
        parts.append(f'<path d="{path}" fill="#0a0a0a"/>')

    def _draw_v_edge_connector(gap_before, gap_after, on_bottom):
        if gap_after > cols_m:
            return
        x_boundary = grid_left + gap_before * cell
        y_rect = (grid_bottom - inward) if on_bottom else (grid_top - outward)
        x_left = x_boundary - conn_thick / 2
        x_right = x_boundary + conn_thick / 2
        r = conn_thick * 0.3
        if on_bottom:
            # прилегает к плите СВЕРХУ (y_rect) -> верхние 2 угла прямые, нижние скруглены
            path = (f'M {x_left:.2f} {y_rect:.2f} '
                    f'L {x_right:.2f} {y_rect:.2f} '
                    f'L {x_right:.2f} {y_rect+conn_len-r:.2f} '
                    f'Q {x_right:.2f} {y_rect+conn_len:.2f} {x_right-r:.2f} {y_rect+conn_len:.2f} '
                    f'L {x_left+r:.2f} {y_rect+conn_len:.2f} '
                    f'Q {x_left:.2f} {y_rect+conn_len:.2f} {x_left:.2f} {y_rect+conn_len-r:.2f} '
                    f'Z')
        else:
            # прилегает к плите СНИЗУ (y_rect+conn_len) -> нижние 2 угла прямые, верхние скруглены
            path = (f'M {x_left:.2f} {y_rect+r:.2f} '
                    f'Q {x_left:.2f} {y_rect:.2f} {x_left+r:.2f} {y_rect:.2f} '
                    f'L {x_right-r:.2f} {y_rect:.2f} '
                    f'Q {x_right:.2f} {y_rect:.2f} {x_right:.2f} {y_rect+r:.2f} '
                    f'L {x_right:.2f} {y_rect+conn_len:.2f} '
                    f'L {x_left:.2f} {y_rect+conn_len:.2f} Z')
        parts.append(f'<path d="{path}" fill="#0a0a0a"/>')

    for gap_before, gap_after in CONNECTOR_GAPS:
        if has_right_neighbor:
            _draw_h_edge_connector(gap_before, gap_after, on_right=True)
        if has_left_neighbor:
            _draw_h_edge_connector(gap_before, gap_after, on_right=False)
        if has_bottom_neighbor:
            _draw_v_edge_connector(gap_before, gap_after, on_bottom=True)
        if has_top_neighbor:
            _draw_v_edge_connector(gap_before, gap_after, on_bottom=False)
    parts.append('</g>')

    # красные направляющие рисуются ПОСЛЕ коннекторов -> визуально ПОВЕРХ них
    parts.append('<g data-block="block-guides">')
    for c in range(BLOCK_SIZE, cols_m, BLOCK_SIZE):
        gx = grid_left + c * cell
        parts.append(f'<line x1="{gx:.2f}" y1="{grid_top:.2f}" x2="{gx:.2f}" '
                      f'y2="{grid_bottom:.2f}" stroke="#e6231e" '
                      f'stroke-width="{max(1.2, cell*0.06):.2f}" opacity="0.95"/>')
    for r in range(BLOCK_SIZE, rows_m, BLOCK_SIZE):
        gy = grid_top + r * cell
        parts.append(f'<line x1="{grid_left:.2f}" y1="{gy:.2f}" x2="{grid_right:.2f}" '
                      f'y2="{gy:.2f}" stroke="#e6231e" '
                      f'stroke-width="{max(1.2, cell*0.06):.2f}" opacity="0.95"/>')
    parts.append('</g>')

    parts.append(f'<rect x="{grid_left:.2f}" y="{grid_top:.2f}" width="{grid_w:.2f}" '
                  f'height="{grid_h:.2f}" fill="none" stroke="#000000" stroke-width="1"/>')

    parts.append('</svg>')
    return "".join(parts).encode("utf-8")


def build_all_instructions_svg(pixel_ids, palette_df, unit_studs=UNIT_STUDS,
                                preview_mode="round", px_per_mm=3.78,
                                page_w_mm=PAGE_W_MM, page_h_mm=PAGE_H_MM,
                                margin_mm=MARGIN_MM):
    """
    Строит инструкции по всем панелям полотна.

    Возвращает tuple:
      combined_svg  — один SVG-файл, где карточки идут одна под другой
                      по порядку номеров,
      per_panel     — список (panel_index, svg_bytes) для раздельного
                      экспорта (например, в ZIP).
    """
    palette_by_id = {int(row["color_id"]): row["rgb_hex"] for _, row in palette_df.iterrows()}
    panels = slice_panels(pixel_ids, unit_studs=unit_studs)

    per_panel = []
    for panel in panels:
        svg_bytes = build_instruction_card_svg(
            panel, palette_by_id, pixel_ids, preview_mode=preview_mode,
            px_per_mm=px_per_mm, page_w_mm=page_w_mm, page_h_mm=page_h_mm,
            margin_mm=margin_mm,
        )
        per_panel.append((panel["index"], svg_bytes))

    page_w_px = page_w_mm * px_per_mm
    page_h_px = page_h_mm * px_per_mm
    total_h_px = page_h_px * len(per_panel)

    combined_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_w_px:.2f}" '
        f'height="{total_h_px:.2f}" viewBox="0 0 {page_w_px:.2f} {total_h_px:.2f}">'
    ]
    for i, (panel_index, svg_bytes) in enumerate(per_panel):
        inner = svg_bytes.decode("utf-8")
        inner_body = inner[inner.find("<svg"):]
        inner_body = inner_body[inner_body.find(">") + 1: inner_body.rfind("</svg>")]
        y_offset = i * page_h_px
        combined_parts.append(f'<g transform="translate(0,{y_offset:.2f})">')
        combined_parts.append(f'<!-- page-break before panel {panel_index} -->')
        combined_parts.append(inner_body)
        combined_parts.append('</g>')
    combined_parts.append('</svg>')

    combined_svg = "".join(combined_parts).encode("utf-8")
    return combined_svg, per_panel
