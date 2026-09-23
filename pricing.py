"""
pricing.py — расчёт расхода деталей по цветам, себестоимости, цены продажи,
HTML-таблица расхода и экспорт PDF-схемы сборки.
"""
import io
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.units import mm
from reportlab.lib import colors as rl_colors
from reportlab.platypus import Table, TableStyle

from config import PRICE_PER_PIECE_RUB, UNIT_STUDS


def build_usage_report(pixel_ids, colors_df):
    unique, counts = np.unique(pixel_ids, return_counts=True)
    report = []
    for u, c in zip(unique, counts):
        row = colors_df[colors_df.color_id == u].iloc[0]
        cost = round(c * PRICE_PER_PIECE_RUB, 2)
        report.append({"color_id": int(u), "name": row["name_ru"], "hex": row["rgb_hex"],
                        "needed": int(c), "cost_rub": cost})
    return pd.DataFrame(report).sort_values("needed", ascending=False).reset_index(drop=True)


def compute_cost_summary(usage_report):
    total_pieces = int(usage_report["needed"].sum())
    total_cost_by_usage = round(usage_report["cost_rub"].sum(), 2)
    return {"total_pieces": total_pieces, "total_cost_by_usage": total_cost_by_usage}


def compute_selling_price(total_material_cost_rub, markup_percent):
    price = round(total_material_cost_rub * (1 + markup_percent / 100), 2)
    profit = round(price - total_material_cost_rub, 2)
    return price, profit


def color_swatch_html(hexcode, size=18):
    return (f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
            f'background-color:#{hexcode};border:1px solid #666"></div>')


def build_usage_table_html(usage_report):
    rows_html = []
    for _, row in usage_report.iterrows():
        swatch = color_swatch_html(row["hex"])
        rows_html.append(
            f'<tr><td style="text-align:center;padding:5px">{swatch}</td>'
            f'<td style="text-align:center;padding:5px;font-weight:600;color:#5AC4DA">{row["color_id"]}</td>'
            f'<td style="padding:5px">{row["name"]}</td>'
            f'<td style="padding:5px">#{row["hex"]}</td>'
            f'<td style="text-align:right;padding:5px">{row["needed"]}</td>'
            f'<td style="text-align:right;padding:5px">{row["cost_rub"]:.2f}</td></tr>'
        )
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:13px;'
        'background-color:#1a1c24;color:#e6e6e6">'
        '<thead><tr style="background-color:#262a35">'
        '<th style="padding:5px"></th><th style="padding:5px">№</th>'
        '<th style="padding:5px;text-align:left">Название</th>'
        '<th style="padding:5px;text-align:left">HEX</th>'
        '<th style="padding:5px">Нужно</th><th style="padding:5px">Руб</th>'
        '</tr></thead><tbody>' + "".join(rows_html) + '</tbody></table>'
    )

def export_svg(pixel_ids, palette_df, canvas_w, canvas_h, frame_depth, matting_depth,
                preview_mode="round", stud_size=10, group_by_color=True):
    total_inset = frame_depth + matting_depth
    svg_w = canvas_w * stud_size
    svg_h = canvas_h * stud_size

    palette_by_id = {
        int(row["color_id"]): row["rgb_hex"]
        for _, row in palette_df.iterrows()
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}">',
        f'<rect x="0" y="0" width="{svg_w}" height="{svg_h}" fill="#000000"/>',
    ]

    if matting_depth > 0:
        mx = frame_depth * stud_size
        mw = svg_w - 2 * mx
        mh = svg_h - 2 * mx
        parts.append(f'<rect x="{mx}" y="{mx}" width="{mw}" height="{mh}" fill="#FFFFFF"/>')

    h, w = pixel_ids.shape
    groups = {}
    for row in range(h):
        for col in range(w):
            color_id = int(pixel_ids[row, col])
            hex_color = palette_by_id.get(color_id, "808080")
            cx = (total_inset + col) * stud_size + stud_size / 2
            cy = (total_inset + row) * stud_size + stud_size / 2
            if preview_mode == "round":
                r = stud_size * 0.42
                el = f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="#{hex_color}"/>'
            else:
                x = (total_inset + col) * stud_size
                y = (total_inset + row) * stud_size
                el = f'<rect x="{x:.2f}" y="{y:.2f}" width="{stud_size}" height="{stud_size}" fill="#{hex_color}"/>'
            groups.setdefault(color_id, []).append(el)

    if group_by_color:
        for color_id, elements in sorted(groups.items()):
            parts.append(f'<g id="color_{color_id}" data-color-id="{color_id}">')
            parts.extend(elements)
            parts.append('</g>')
    else:
        for elements in groups.values():
            parts.extend(elements)

    parts.append('</svg>')
    svg_content = "\n".join(parts)
    return svg_content.encode("utf-8")

def export_pdf(pixel_ids, colors_df, usage_report, canvas_w, canvas_h, frame_depth, matting_depth):
    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=A4)
    page_w, page_h = A4
    total_inset = frame_depth + matting_depth

    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, page_h - 20 * mm, "LEGO-мозаика — схема сборки")
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, page_h - 28 * mm,
                 f"Полотно {canvas_w}x{canvas_h} студов ({canvas_w*0.8:.1f}x{canvas_h*0.8:.1f} см), "
                 f"рамка {frame_depth} дет., паспарту {matting_depth} дет.")

    grid_top = page_h - 40 * mm
    max_grid_width = page_w - 40 * mm
    cell = min(max_grid_width / canvas_w, (grid_top - 100 * mm) / canvas_h)
    color_map = {row["color_id"]: (row["r"]/255, row["g"]/255, row["b"]/255) for _, row in colors_df.iterrows()}

    grid_px_w, grid_px_h = canvas_w * cell, canvas_h * cell
    c.setFillColorRGB(0, 0, 0)
    c.rect(20*mm, grid_top - grid_px_h, grid_px_w, grid_px_h, fill=1, stroke=0)
    if matting_depth > 0:
        c.setFillColorRGB(1, 1, 1)
        c.rect(20*mm + frame_depth*cell, grid_top - grid_px_h + frame_depth*cell,
               grid_px_w - 2*frame_depth*cell, grid_px_h - 2*frame_depth*cell, fill=1, stroke=0)

    mosaic_h, mosaic_w = pixel_ids.shape
    for y in range(mosaic_h):
        for x in range(mosaic_w):
            r, g, b = color_map.get(pixel_ids[y, x], (0.7, 0.7, 0.7))
            px = 20 * mm + (total_inset + x) * cell
            py = grid_top - (total_inset + y + 1) * cell
            c.setFillColorRGB(r, g, b)
            c.setStrokeColorRGB(r, g, b)
            c.circle(px + cell/2, py + cell/2, cell/2 - 0.2*mm, fill=1, stroke=1)

    for gx in range(0, canvas_w + 1, UNIT_STUDS):
        x = 20*mm + gx*cell
        c.setStrokeColorRGB(0.9, 0.2, 0.2)
        c.setLineWidth(0.5)
        c.line(x, grid_top - grid_px_h, x, grid_top)
    for gy in range(0, canvas_h + 1, UNIT_STUDS):
        y = grid_top - gy*cell
        c.line(20*mm, y, 20*mm + grid_px_w, y)

    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, page_h - 20 * mm, "Расход деталей по цветам")

    table_data = [["№", "Название", "HEX", "Нужно", "Стоимость"]]
    for _, row in usage_report.iterrows():
        table_data.append([str(row["color_id"]), row["name"], "#" + row["hex"],
                            str(row["needed"]), f"{row['cost_rub']:.2f} руб"])
    t = Table(table_data, colWidths=[12*mm, 40*mm, 25*mm, 28*mm, 32*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    t.wrapOn(c, page_w, page_h)
    t.drawOn(c, 20 * mm, page_h - 30 * mm - len(table_data) * 7 * mm)

    c.save()
    buf.seek(0)
    return buf
