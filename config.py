"""
config.py — все константы, тема оформления, база 52 цветов LEGO и режимы пикселизации.
Не содержит логики обработки, только данные и настройки.
"""
import numpy as np
import pandas as pd

STUD_MM = 8.0

MIN_STUDS = 8

MAX_STUDS = 260

DB_PATH = "lego_colors.db"

TARGET_PREVIEW_WIDTH_PX = 900

MIN_CELL_PX = 4

MAX_CELL_PX = 40

PREVIEW_SUPERSAMPLE = 3

COMPARE_PREVIEW_MAX_WIDTH_PX = 460

RECOMMENDED_MIN_COLORS = 16

SLIC_MAX_DIM = 800

# Compactness=8 (было 15): сегменты лучше следуют контурам лица,
# меньше квадратных артефактов на скулах, веках, волосах.
SLIC_COMPACTNESS = 8

# Sigma=1 (было 2): меньше предсглаживания до SLIC →
# тонкие детали (веки, уголки губ) попадают в отдельные сегменты.
SLIC_SIGMA = 1

SLIC_SEGMENTS_PER_STUD = 1.2

# ---------------------------------------------------------------------------
# Potts-регуляризация v3.1
# ---------------------------------------------------------------------------

# Сила сглаживания внутри однородных SLIC-зон (кожа, фон, волосы).
# 3.0 (было 2.5) — чуть сильнее связность, меньше одиночных выбросов
# в плоских зонах кожи без потери деталей на SLIC-границах.
POTTS_SMOOTHNESS = 3.0

# Число ICM-итераций. 4 (было 3) — одна дополнительная итерация
# убирает оставшиеся выбросы в крупных однородных зонах (лоб, щёки).
POTTS_ITER = 4

# Сила локального FS-дизеринга в плавных зонах (0 = выключен).
# ОТКЛЮЧЕНО (было 0.45): конфликтовал с Potts-сглаживанием, которое
# уже выравнивает соседние пиксели в однородных зонах — вложенный
# дизеринг тут же вносил обратно зерно ошибки на те же пиксели,
# из-за чего на коже/лице получался шум вместо плавного тона.
POTTS_LOCAL_DITHER_STRENGTH = 0.0

LOT_SIZE = 1000

LOT_PRICE_RUB = 419

PRICE_PER_PIECE_RUB = LOT_PRICE_RUB / LOT_SIZE

DEFAULT_MARKUP_PERCENT = 25

MIN_MARKUP_PERCENT = 25

MAX_MARKUP_PERCENT = 100

AR_TOLERANCE = 0.35

AR_TOLERANCE_PCT_FOR_CROP = 8.0

EXPAND_COST_PER_PCT_THRESHOLD = 50

UNIT_STUDS = 16

UNIT_CM = UNIT_STUDS * STUD_MM / 10

MAX_PANELS_PER_DIM = 16

CANVAS_SETS = {
    "6pcs": {"pieces": 6, "price_rub": 669},
    "9pcs": {"pieces": 9, "price_rub": 1099},
    "12pcs": {"pieces": 12, "price_rub": 1419},
    "15pcs": {"pieces": 15, "price_rub": 1539},
    "20pcs": {"pieces": 20, "price_rub": 2089},
}
for _s in CANVAS_SETS.values():
    _s["price_per_piece"] = round(_s["price_rub"] / _s["pieces"], 2)

BEST_VALUE_SET = min(CANVAS_SETS, key=lambda k: CANVAS_SETS[k]["price_per_piece"])

MOSAIC_BG_COLOR = (0, 0, 0)

FRAME_BLACK = (0, 0, 0)

MATTING_WHITE = (255, 255, 255)

CANVAS_GRID_LINE_COLOR = (255, 40, 40, 110)

CENTERLINE_COLOR = (0, 200, 255)

CENTER_DOT_COLOR = (255, 220, 0)

CROP_DARKEN_COLOR = (0, 0, 0, 165)

CROP_KEEP_BORDER_COLOR = (255, 50, 50, 255)

FRAME_TILE_SIZES = (8, 6, 4, 2, 1)

FRAME_CORNER_SIZE = 2

TILE_STRAIGHT_PRICE_PER_PIECE_RUB = 339 / 50

TILE_CORNER_WHITE_PRICE_PER_PIECE_RUB = 132 / 50

TILE_CORNER_BLACK_PRICE_PER_PIECE_RUB = 133 / 50

FRAME_SEAM_LINE_COLOR = (40, 220, 90)

MATTING_SEAM_LINE_COLOR = (30, 30, 220)

SEAM_HALO_COLOR = (0, 0, 0)

FRAME_SEAM_WIDTH_FACTOR = 10

MATTING_SEAM_WIDTH_FACTOR = 16

DARK_THEME_CSS = """

"""

SELLER_COLORS = [
    ("White", "F2F3F2"), ("Light Light Gray", "E6E3E0"), ("Light Bluish Gray", "A0A5A9"),
    ("Dark Bluish Gray", "6C6E68"), ("Dark Dark Gray", "3E3C39"), ("Black", "05131D"),
    ("Light Royal Blue", "9FC3E9"), ("Dark Gray", "6D6E5C"), ("Skin White", "F6D7B3"),
    ("Flesh Pink", "F5C189"), ("Light Flesh", "F6D7B3"), ("Flesh Tan", "E4CD9E"),
    ("Flesh Red", "CC702A"), ("Nougat", "D09168"), ("Medium Dark Flesh", "AA7D55"),
    ("Dark Orange", "A83E16"), ("Brown", "582A12"), ("Dark Brown", "352100"),
    ("Tan", "E4CD9E"), ("Dark Tan", "958A73"), ("Bright Yellow", "F2CD37"),
    ("Yellow", "FFD500"), ("Dark Yellow", "C79923"), ("Bright Orange", "FE8A18"),
    ("Orange", "FF8000"), ("Light Pink", "FECCCF"), ("Bright Pink", "E4ADC8"),
    ("Dark Pink", "C870A0"), ("Magenta", "923978"), ("Red", "C91A09"),
    ("Dark Red", "720E0F"), ("Sand Red", "D67572"), ("Medium Lavender", "9391E4"),
    ("Purple", "81007B"), ("Dark Purple", "3F3691"), ("Medium Blue", "5A93DB"),
    ("Medium Azure", "36AEBF"), ("Navy Blue", "0A3463"), ("Dark Azure", "078BC9"),
    ("Bright Light Blue", "9FC3E9"), ("Blue", "0055BF"), ("Dark Blue", "0A1F3D"),
    ("Sand Blue", "6074A1"), ("Yellowish Green", "DFEEA5"), ("Lime", "BBE90B"),
    ("Olive Green", "9B9A5A"), ("Sand Green", "A0BCAC"), ("Light Aqua", "ADC3C0"),
    ("Bright Green", "4B9F4A"), ("Green", "237841"), ("Dark Green", "184632"),
    ("Army Green", "6C6E39"),
]

QUANTIZE_MODE_LABELS = {
    "segment_assign": "Сегменты + оптимальное назначение (Hungarian)",
    "nearest_potts": "Nearest + Potts + dithering (портреты, рекомендуется) ✨",
    "superpixel": "Суперпиксели SLIC + усреднение (плавные зоны)",
    "nearest": "Nearest-color (точный, есть выбросы)",
    "cluster": "Кластерная карта K-Means (зонально, хендмейд)",
    "bayer4": "Ordered/Bayer 4x4 (регулярный узор)",
    "bayer8": "Ordered/Bayer 8x8 (мягче, менее заметный)",
    "atkinson": "Atkinson dithering (легче FS, чище)",
    "fs_soft": "Floyd-Steinberg смягчённый (0.6x ошибки)",
    "fs_classic": "Floyd-Steinberg классический (старое поведение)",
}

BAYER_4x4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=float) / 16.0 - 0.5

BAYER_8x8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=float) / 64.0 - 0.5

_BAYER_MATRICES = {"bayer4": BAYER_4x4, "bayer8": BAYER_8x8}
