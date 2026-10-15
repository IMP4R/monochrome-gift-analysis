#!/usr/bin/env python3
"""Палитра моделей (с сеткой 8×8), цвет фонов, радужность, скор монохрома.

Механика v2.9:
- модель -> до 3 отдельных CAM02-UCS цветов; для каждого сохраняется распределение CIE L*
  по 20 полосам, поэтому светлый и тёмный оттенки не усредняются в ложный;
  share — доля красочной массы (показ), share_all — доля ВСЕХ пикселей (скор);
- СЕТКА 8×8 и collection-common остаются только для удаления повторяющегося
  центрального цвета коллекции; они не участвуют в тональном штрафе;
- hue-match засчитывается только при попадании реальной цветовой массы модели
  в тональный коридор фона: до 10 L* полный балл, 10–40 L* квадратичный спад;
- широкий диапазон одного hue-family штрафуется по lstd; насыщенность фона
  сверяется с family, кроме естественной тёмной тени того же hue;
- скор = min(100, 135 × max(0, match − 0.6 × mismatch) × (1 − 0.6 × rainbow));
- нейтральные не мешают; ахроматика — по perceptual lightness;
- итог умеренно умножается на мировой color-preference prior: 65% BCP-32 +
  35% YouGov (10 стран), диапазон коэффициента 0.88–1.12.

classic similarity() (bleach-hub) оставлен для сравнения."""
import colorsys
import copy
import io
import json
import re
import threading
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image
from colorspacious import cspace_convert, deltaE

OG_RE = re.compile(r'<meta property="og:image" content="([^"]+)"')

COLOR_CACHE_FILE = Path(__file__).parent / "cache" / "colors.json"
_color_cache = None
_cache_lock = threading.Lock()


def _load_cache():
    global _color_cache
    if _color_cache is None:
        try:
            _color_cache = json.loads(COLOR_CACHE_FILE.read_text())
        except Exception:
            _color_cache = {}
    return _color_cache


def cache_get(key: str):
    return _load_cache().get(key)


def cache_set(key: str, value: dict):
    with _cache_lock:
        _load_cache()[key] = value
        try:
            COLOR_CACHE_FILE.parent.mkdir(exist_ok=True)
            COLOR_CACHE_FILE.write_text(json.dumps(_color_cache, ensure_ascii=False))
        except Exception:
            pass


def model_slug_candidates(name: str):
    base = name.strip().lower()
    alnum = "".join(ch for ch in base if ch.isalnum())
    dashes = "-".join(base.split())
    plain = base.replace(" ", "")
    seen, out = set(), []
    for s in (alnum, dashes, plain):
        if s and s not in seen:
            seen.add(s)
            out.append(quote(s))
    return out


def _dominant_of(px):
    """Доминирующий цвет списка RGB-пикселей (бакеты по 24) или None."""
    if len(px) < 8:
        return None
    buckets = Counter((r // 24 * 24, g // 24 * 24, b // 24 * 24) for r, g, b in px)
    top = buckets.most_common(1)[0][0]
    sel = [p for p in px if (p[0] // 24 * 24, p[1] // 24 * 24, p[2] // 24 * 24) == top]
    n = len(sel)
    return tuple(round(sum(c) / n) for c in zip(*sel))


def image_dominant(content: bytes):
    img = Image.open(io.BytesIO(content)).convert("RGBA")
    img.thumbnail((96, 96))
    px = [(p[0], p[1], p[2]) for p in img.getdata() if p[3] > 24]
    if not px:
        return None
    return _dominant_of(px)


def center_average(content: bytes):
    """Средний цвет центральной области рендера фона (то, что видно за моделью)."""
    img = Image.open(io.BytesIO(content)).convert("RGB")
    w, h = img.size
    img = img.crop((int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7)))
    img = img.resize((1, 1))
    return img.getpixel((0, 0))


def tme_og_image_dominant(short_name: str, num: int):
    try:
        url = "https://t.me/nft/" + short_name + "-" + str(num)
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None
        m = OG_RE.search(r.text)
        if not m:
            return None
        img = requests.get(m.group(1), timeout=15)
        if img.status_code != 200:
            return None
        return image_dominant(img.content)
    except Exception:
        return None


def _hue_of(rgb):
    h, s, v = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
    return h * 360, s, v


def _rgb_of_hex(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


_CAM02_RGB_CACHE = {}
_LSTAR_RGB_CACHE = {}
_LCH_RGB_CACHE = {}


def _cam02_rgb(rgb):
    key = tuple(int(x) for x in rgb)
    value = _CAM02_RGB_CACHE.get(key)
    if value is None:
        value = np.asarray(cspace_convert(key, "sRGB255", "CAM02-UCS"), dtype=float)
        _CAM02_RGB_CACHE[key] = value
    return value


def _deltaE_rgb(a, b) -> float:
    """True Euclidean CAM02-UCS distance; each distinct RGB is converted once."""
    return float(np.linalg.norm(_cam02_rgb(a) - _cam02_rgb(b)))


TONE_BINS = 20  # полосы perceptual lightness L*: шаг 5 из диапазона 0..100
TONE_CENTERS = np.arange(2.5, 100.0, 5.0)
HUE_FULL_MATCH_DEG = 12.0
HUE_ZERO_MATCH_DEG = 45.0
CENTER_TONE_WEIGHT = 0.35
DARK_CHROMA_FLOOR = 0.50
PAIR_SCORE_SCALE = 110.0
PERCEPTUAL_COLOR_SIGMA = 18.0
PERCEPTUAL_COLOR_FLOOR = 0.20
PERCEPTUAL_CENTER_WEIGHT = 0.25
ACCENT_ALLOWANCE_MAX = 0.35
ACCENT_ALLOWANCE_RATIO = 0.70


def _lstar_values(rgb):
    """RGB-массив -> perceptual lightness CIE L* (0..100)."""
    x = np.asarray(rgb, dtype=float) / 255.0
    x = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    y = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    return np.where(y > 216 / 24389, 116 * np.cbrt(y) - 16, (24389 / 27) * y)


def _lstar_rgb(rgb):
    key = tuple(int(x) for x in rgb)
    value = _LSTAR_RGB_CACHE.get(key)
    if value is None:
        value = float(_lstar_values(np.asarray([key]))[0])
        _LSTAR_RGB_CACHE[key] = value
    return value


# Prior приятности отдельного ФОНА, не замена скору монохрома.
# BCP: 8 hues × 4 cuts. Рейтинги приблизительно оцифрованы по Fig. 1C
# Palmer & Schloss (2010), шкала -100..100. YouGov — точные средние процентов
# по 10 странам (n=10,868). Семантическая категория Telegram берётся из
# официального имени для 75 известных фонов; неизвестные цвета классифицируются
# по perceptual CIELCh, а не по HSV.
BCP_PREFERENCE = {
    "saturated": {"red": 32, "orange": 18, "yellow": 5, "chartreuse": -23,
                  "green": 8, "cyan": 34, "blue": 52, "purple": 35},
    "light": {"red": 10, "orange": 4, "yellow": 0, "chartreuse": -23,
              "green": 8, "cyan": 25, "blue": 38, "purple": 24},
    "muted": {"red": 8, "orange": 0, "yellow": -3, "chartreuse": -27,
              "green": 6, "cyan": 22, "blue": 35, "purple": 22},
    "dark": {"red": 40, "orange": -35, "yellow": -60, "chartreuse": -32,
             "green": 28, "cyan": 16, "blue": 20, "purple": 12},
}
YOUGOV_PERCENT = {
    "red": 12.3, "orange": 5.6, "yellow": 6.2, "green": 11.7,
    "blue": 27.6, "purple": 11.1, "pink": 7.1, "brown": 3.2,
    "black": 7.7, "white": 5.5,
}
YOUGOV_INTERPOLATED = {
    "chartreuse": (YOUGOV_PERCENT["yellow"] + YOUGOV_PERCENT["green"]) / 2,
    "cyan": (YOUGOV_PERCENT["green"] + YOUGOV_PERCENT["blue"]) / 2,
}
PREFERENCE_BCP_WEIGHT = 1.00
PREFERENCE_YOUGOV_WEIGHT = 0.00
PREFERENCE_COEFF_MIN = 1.00  # legacy: no multiplicative prior
PREFERENCE_COEFF_MAX = 1.00
MONO_TOTAL_WEIGHT = 0.80
COLOR_TOTAL_WEIGHT = 0.20
# Ten basic YouGov colors imply an expected neutral share near 10%. Centering
# there prevents blue's 27.6% outlier from incorrectly making green negative.
YOUGOV_NEUTRAL_PERCENT = 10.0

# name -> (user-facing family, BCP hue, YouGov category)
# BCP has cyan/chartreuse but no brown/pink; YouGov is the opposite. Keeping
# these axes separate prevents labels such as Dark Green from becoming lime.
TELEGRAM_BACKDROP_DEFINITIONS = {
    # Red / rose
    **{n: ("red", "red", "red") for n in (
        "Rosewood", "Fire Engine", "Carmine", "Mystic Pearl", "Strawberry",
        "Coral Red", "Raspberry", "Burgundy")},
    # Brown / tan: warm BCP hue, basic YouGov brown
    **{n: ("brown", "orange", "brown") for n in (
        "Chestnut", "Seal Brown", "Chocolate", "Cappuccino", "Desert Sand")},
    # Orange
    **{n: ("orange", "orange", "orange") for n in (
        "Tomato", "Burnt Sienna", "Carrot Juice", "Copper", "Persimmon",
        "Orange", "Caramel")},
    # Yellow / gold / olive
    **{n: ("yellow", "yellow", "yellow") for n in (
        "Old Gold", "Mustard", "Satin Gold", "Amber", "Pure Gold", "Light Olive")},
    # True yellow-green only; green-named colors below are not chartreuse
    **{n: ("chartreuse", "chartreuse", "chartreuse") for n in (
        "Khaki Green", "Lemongrass")},
    # Green, including dark/low-chroma and blue-green names
    **{n: ("green", "green", "green") for n in (
        "Rifle Green", "Camo Green", "Malachite", "Pistachio", "Shamrock Green",
        "Ranger Green", "Dark Green", "Hunter Green", "Mint Green", "Emerald",
        "Gunship Green", "Pine Green", "Pacific Green", "Jade Green", "Tactical Pine")},
    # Cyan / turquoise
    **{n: ("cyan", "cyan", "cyan") for n in (
        "Aquamarine", "Pacific Cyan", "Turquoise", "Deep Cyan", "Moonstone")},
    # Basic blue names whose BCP stimulus is the cyan-blue region
    **{n: ("blue", "cyan", "blue") for n in (
        "Sky Blue", "Azure Blue", "Celtic Blue", "Silver Blue", "Indigo Dye",
        "French Blue", "Sapphire")},
    # Deeper BCP blue
    **{n: ("blue", "blue", "blue") for n in (
        "Navy Blue", "Cobalt Blue", "Marine Blue", "Neon Blue")},
    # Purple / violet
    **{n: ("purple", "purple", "purple") for n in (
        "Electric Indigo", "Electric Purple", "Lavender", "Grape", "English Violet", "Purple", "Dark Lilac")},
    "Fandango": ("pink", "purple", "pink"),
    "Mexican Pink": ("pink", "purple", "pink"),
    # Achromatic official names. YouGov black/white is chosen from L*.
    **{n: ("neutral", None, None) for n in (
        "Ivory White", "Platinum", "Battleship Grey", "Feldgrau", "Gunmetal",
        "Steel Grey", "Roman Silver")},
}

FAMILY_TO_BCP = {
    "red": "red", "orange": "orange", "brown": "orange",
    "yellow": "yellow", "chartreuse": "chartreuse", "green": "green",
    "cyan": "cyan", "blue": "blue", "purple": "purple", "pink": "purple",
    "neutral": None,
}


def _lch_of(rgb):
    """sRGB255 -> cached (CIE L*, C*, hue angle)."""
    key = tuple(int(x) for x in rgb)
    value = _LCH_RGB_CACHE.get(key)
    if value is None:
        lab = np.asarray(cspace_convert(key, "sRGB255", "CIELab"), dtype=float)
        lstar, a, b = (float(x) for x in lab)
        value = (lstar, float(np.hypot(a, b)),
                 float((np.degrees(np.arctan2(b, a)) + 360.0) % 360.0))
        _LCH_RGB_CACHE[key] = value
    return value


def _fallback_color_definition(lstar, chroma, hue):
    """Perceptual fallback for unknown backdrop names.

    Boundaries are calibrated on the 73 official Telegram backdrop colors in
    CIELCh. Low chroma is neutral; chartreuse is intentionally narrow.
    """
    if chroma < 10.0:
        return "neutral", None, None
    if hue < 47.5 or hue >= 350.0:
        family = "red"
    elif hue < 78.0:
        family = "orange"
    elif hue < 105.0:
        family = "yellow"
    elif hue < 122.0:
        family = "chartreuse"
    elif hue < 178.0:
        family = "green"
    elif hue < 267.0:
        family = "cyan"
    elif hue < 300.0:
        family = "blue"
    elif hue < 340.0:
        family = "purple"
    elif lstar >= 58.0:
        family = "pink"
    else:
        family = "red"
    bcp_hue = FAMILY_TO_BCP[family]
    yougov_color = family if family not in ("cyan", "chartreuse") else family
    return family, bcp_hue, yougov_color


def _bcp_cut(sat, lstar, chroma):
    """Map a real backdrop to the closest BCP cut using lightness + chroma."""
    if lstar < 43.0:
        return "dark"
    if sat >= 0.65 or chroma >= 55.0:
        return "saturated"
    if lstar >= 67.5:
        return "light"
    return "muted"


def _normalize_bcp_rating(rating):
    # Preserve the semantic zero: rating 0 -> 0.5, observed -60/+52 -> 0/1.
    value = 0.5 + 0.5 * (rating / (52.0 if rating >= 0 else 60.0))
    return max(0.0, min(1.0, value))


def _normalize_yougov_percent(percent):
    # Neutral is the expected 10% share among ten basic colors. Scale each side
    # separately to brown=0 and blue=1 while preserving 10%=0.5.
    if percent >= YOUGOV_NEUTRAL_PERCENT:
        value = 0.5 + 0.5 * ((percent - YOUGOV_NEUTRAL_PERCENT) /
                             (YOUGOV_PERCENT["blue"] - YOUGOV_NEUTRAL_PERCENT))
    else:
        value = 0.5 * ((percent - YOUGOV_PERCENT["brown"]) /
                       (YOUGOV_NEUTRAL_PERCENT - YOUGOV_PERCENT["brown"]))
    return max(0.0, min(1.0, value))


def color_preference(rgb, backdrop_name=None):
    """BCP-32 + YouGov prior with exact Telegram definitions and CIELCh fallback."""
    if not rgb:
        return {"coefficient": 1.0, "raw_coefficient": 1.0, "combined": 0.5,
                "achromatic": True, "color_family": "neutral"}

    lstar, chroma, perceptual_hue = _lch_of(rgb)
    key = str(backdrop_name).strip() if backdrop_name else None
    definition = TELEGRAM_BACKDROP_DEFINITIONS.get(key)
    if definition is None:
        family, bcp_hue, yougov_color = _fallback_color_definition(
            lstar, chroma, perceptual_hue)
        source = "cielch_fallback"
    else:
        family, bcp_hue, yougov_color = definition
        source = "telegram_name"

    if family == "neutral" or bcp_hue is None:
        category = yougov_color or ("white" if lstar >= 60.0 else "black")
        return {
            "coefficient": 1.0, "raw_coefficient": 1.0, "hue_floor": None,
            "combined": 0.5, "achromatic": True, "color_family": "neutral",
            "classification_source": source, "perceptual_hue": round(perceptual_hue, 2),
            "perceptual_chroma": round(chroma, 2), "perceptual_lstar": round(lstar, 2),
            "bcp_hue": None, "bcp_cut": None, "bcp_rating": None,
            "bcp_normalized": 0.5, "bcp_score": 50.0, "yougov_color": category,
            "yougov_percent": YOUGOV_PERCENT[category],
            "yougov_normalized": round(_normalize_yougov_percent(YOUGOV_PERCENT[category]), 4),
        }

    _, sat, _ = _hue_of(rgb)
    cut = _bcp_cut(sat, lstar, chroma)
    bcp_rating = float(BCP_PREFERENCE[cut][bcp_hue])
    bcp_norm = _normalize_bcp_rating(bcp_rating)
    if yougov_color in YOUGOV_PERCENT:
        yougov_percent = YOUGOV_PERCENT[yougov_color]
    else:
        yougov_percent = YOUGOV_INTERPOLATED[yougov_color]
    yougov_norm = _normalize_yougov_percent(float(yougov_percent))

    combined = bcp_norm
    coefficient = 1.0  # visual and color scores are additive, never multiplicative
    return {
        "coefficient": round(coefficient, 4), "raw_coefficient": round(coefficient, 4),
        "hue_floor": None, "combined": round(combined, 4), "achromatic": False,
        "color_family": family, "classification_source": source,
        "perceptual_hue": round(perceptual_hue, 2),
        "perceptual_chroma": round(chroma, 2), "perceptual_lstar": round(lstar, 2),
        "bcp_hue": bcp_hue, "bcp_cut": cut, "bcp_rating": bcp_rating,
        "bcp_normalized": round(bcp_norm, 4), "bcp_score": round(100.0 * bcp_norm, 2), "yougov_color": yougov_color,
        "yougov_percent": round(float(yougov_percent), 2),
        "yougov_normalized": round(yougov_norm, 4),
    }


def bcp_color_score(bg_rgb, backdrop_name=None):
    """Independent 0..100 color pleasantness score from BCP-32 only."""
    return round(100.0 * float(color_preference(bg_rgb, backdrop_name).get("bcp_normalized", 0.5)), 2)


def combine_mono_color_scores(mono_score, color_score):
    """Transparent additive score: 80 visual monochrome points + 20 BCP points."""
    mono_points = MONO_TOTAL_WEIGHT * max(0.0, min(100.0, float(mono_score)))
    color_points = COLOR_TOTAL_WEIGHT * max(0.0, min(100.0, float(color_score)))
    return round(mono_points + color_points, 2), round(mono_points, 2), round(color_points, 2)


def apply_color_preference(score, bg_rgb, backdrop_name=None):
    """Compatibility helper using the new additive score, never a multiplier."""
    total, _, _ = combine_mono_color_scores(score, bcp_color_score(bg_rgb, backdrop_name))
    return total


MONO_FEATURE_NAMES = (
    "base", "classic", "rainbow", "neutral_share", "matched_share",
    "full_share", "mismatch_share", "weighted_hue_fit", "min_hue_dist",
    "weighted_perceptual_fit", "min_delta_e", "weighted_delta_e",
    "weighted_tone_fit", "min_lstar_gap", "center_delta_e",
    "center_lstar_gap", "dominant_share", "palette_entropy", "bg_lstar",
    "bg_chroma", "bg_sat", "model_achromatic", "bg_achromatic",
)
MONO_MODEL_PATH = Path(__file__).parent / "models" / "mono_logreg_v1.json"
_mono_model_cache = {"mtime": None, "data": None}


def _hex_rgb(value):
    if not value:
        return None
    value = str(value).lstrip("#")
    if len(value) != 6:
        return None
    try:
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def _hue_distance(a, b):
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def mono_features(bg_rgb, model_info, base_score=None, classic_score=None, bg_name=None):
    """Color-identity-free visual features used by the learned mono model.

    No BCP, YouGov, color family, or absolute hue is included. Absolute
    lightness/chroma/saturation remain because they affect human perception.
    """
    if not bg_rgb:
        return {name: 0.0 for name in MONO_FEATURE_NAMES}
    if base_score is None:
        base_score = pair_score(bg_rgb, model_info, apply_preference=False,
                                tone_tolerance_scale=1.0, bg_name=bg_name)
    if classic_score is None:
        model_rgb = model_info.get("rgb")
        classic_score = similarity(bg_rgb, model_rgb) if model_rgb else 0.0
    bh, bs, _ = _hue_of(bg_rgb)
    bg_lstar = _lstar_rgb(bg_rgb)
    _, bg_chroma, _ = _lch_of(bg_rgb)
    profile = color_preference(bg_rgb, bg_name)
    palette = [c for c in (model_info.get("palette") or []) if not c.get("common")]
    shares = np.asarray([float(c.get("share_all", c.get("share", 0.0)) or 0.0)
                         for c in palette], dtype=float)
    if palette:
        total = max(1e-9, float(shares.sum()))
        weights = shares / total
        hue_dists, delta_es, tone_fits, lstar_gaps, perceptual_fits = [], [], [], [], []
        for c in palette:
            hue = c.get("hue")
            hue_dists.append(_hue_distance(hue if hue is not None else bh, bh))
            crgb = _hex_rgb(c.get("hex"))
            delta_e = _deltaE_rgb(crgb, bg_rgb) if crgb else 100.0
            delta_es.append(delta_e)
            perceptual_fits.append(float(np.exp(-(delta_e ** 2) / (2.0 * 18.0 ** 2))))
            lstar = c.get("lstar")
            if lstar is None:
                lstar = _lstar_rgb(crgb) if crgb else bg_lstar
            gap = abs(float(lstar) - bg_lstar)
            lstar_gaps.append(gap)
            tone = _tone_fit(c.get("tone_hist"), bg_lstar, 1.0)
            tone_fits.append(float(tone) if tone is not None else _tone_delta_fit(gap, 1.0))
    else:
        weights = np.asarray([1.0])
        shares = np.asarray([0.0])
        hue_dists, delta_es, tone_fits, lstar_gaps, perceptual_fits = [180.0], [100.0], [0.0], [100.0], [0.0]
    hue_dists = np.asarray(hue_dists, dtype=float)
    delta_es = np.asarray(delta_es, dtype=float)
    tone_fits = np.asarray(tone_fits, dtype=float)
    lstar_gaps = np.asarray(lstar_gaps, dtype=float)
    perceptual_fits = np.asarray(perceptual_fits, dtype=float)
    matched_share = float(shares[hue_dists <= 45.0].sum()) if palette else 0.0
    full_share = float(shares[hue_dists <= 12.0].sum()) if palette else 0.0
    mismatch_share = float(shares[hue_dists > 45.0].sum()) if palette else 1.0
    hue_fit = np.where(hue_dists <= 12.0, 1.0,
                       np.maximum(0.0, 1.0 - (hue_dists - 12.0) / 33.0))
    entropy = (-sum(float(w) * np.log(max(float(w), 1e-12)) for w in weights) /
               np.log(3.0)) if len(weights) > 1 else 0.0
    center = model_info.get("center") or {}
    center_rgb = _hex_rgb(center.get("hex"))
    center_delta_e = _deltaE_rgb(center_rgb, bg_rgb) if center_rgb else 100.0
    center_lstar = center.get("lstar")
    if center_lstar is None:
        center_lstar = _lstar_rgb(center_rgb) if center_rgb else bg_lstar
    values = {
        "base": float(base_score or 0.0) / 100.0,
        "classic": float(classic_score or 0.0) / 100.0,
        "rainbow": float(model_info.get("rainbow") or 0.0),
        "neutral_share": float(model_info.get("neutral") or 0.0),
        "matched_share": matched_share,
        "full_share": full_share,
        "mismatch_share": mismatch_share,
        "weighted_hue_fit": float(np.dot(weights, hue_fit)),
        "min_hue_dist": float(hue_dists.min()) / 180.0,
        "weighted_perceptual_fit": float(np.dot(weights, perceptual_fits)),
        "min_delta_e": float(delta_es.min()) / 100.0,
        "weighted_delta_e": float(np.dot(weights, delta_es)) / 100.0,
        "weighted_tone_fit": float(np.dot(weights, tone_fits)),
        "min_lstar_gap": float(lstar_gaps.min()) / 100.0,
        "center_delta_e": float(center_delta_e) / 100.0,
        "center_lstar_gap": abs(float(center_lstar) - bg_lstar) / 100.0,
        "dominant_share": float(shares.max()) if palette else 0.0,
        "palette_entropy": float(entropy),
        "bg_lstar": float(bg_lstar) / 100.0,
        "bg_chroma": float(bg_chroma) / 100.0,
        "bg_sat": float(bs),
        "model_achromatic": float(bool(model_info.get("achromatic"))),
        "bg_achromatic": float(bool(profile.get("achromatic"))),
    }
    return {name: round(float(values[name]), 6) for name in MONO_FEATURE_NAMES}


def load_mono_model(path=None):
    path = Path(path) if path else MONO_MODEL_PATH
    try:
        mtime = path.stat().st_mtime
        if _mono_model_cache["mtime"] != mtime:
            data = json.loads(path.read_text(encoding="utf-8"))
            if tuple(data.get("feature_names") or ()) != MONO_FEATURE_NAMES:
                raise ValueError("mono model feature mismatch")
            _mono_model_cache.update(mtime=mtime, data=data)
        return _mono_model_cache["data"]
    except Exception:
        return None


def predict_monochrome(bg_rgb, model_info, base_score=None, classic_score=None,
                       bg_name=None, model_path=None):
    features = mono_features(bg_rgb, model_info, base_score, classic_score, bg_name)
    model = load_mono_model(model_path)
    if model:
        mean = model["mean"]
        scale = model["scale"]
        coefficients = model["coefficients"]
        z = float(model["intercept"])
        for i, name in enumerate(model["feature_names"]):
            z += float(coefficients[i]) * ((features[name] - float(mean[i])) /
                                           max(1e-9, float(scale[i])))
        probability = 1.0 / (1.0 + float(np.exp(-max(-30.0, min(30.0, z)))))
        version = model.get("model_version", "mono-logreg")
        threshold = float(model.get("decision_threshold", 0.5))
    else:
        base = float(base_score or 0.0)
        probability = 1.0 / (1.0 + float(np.exp(-(base - 20.0) / 10.0)))
        version = "base-fallback"
        threshold = 0.5
    return {
        "probability": round(probability, 6),
        "score": round(100.0 * probability, 2),
        "model_version": version,
        "decision_threshold": round(threshold, 6),
        "features": features,
    }


def _tone_stats(samples):
    """Не усредняем тона: сохраняем распределение L* по 20 полосам."""
    values = _lstar_values(np.asarray([[p[3], p[4], p[5]] for p in samples]))
    hist = np.histogram(values, bins=np.linspace(0, 100, TONE_BINS + 1))[0].astype(float)
    hist /= max(1.0, hist.sum())
    return (round(float(values.mean()), 2), round(float(values.std()), 2),
            [round(float(x), 4) for x in hist])


def _tone_delta_fit(delta_l, tolerance_scale=1.0):
    """Scalar tone fit with the same 10/40 L* corridor as the histogram."""
    scale = max(0.5, min(1.0, float(tolerance_scale)))
    full = 10.0 * scale
    cutoff = 40.0 * scale
    if delta_l <= full:
        return 1.0
    return max(0.0, 1.0 - (delta_l - full) / (cutoff - full)) ** 2


def _tone_fit(hist, bg_lstar, tolerance_scale=1.0):
    """Доля цветовой массы в тональном коридоре относительно фона.

    Базовые границы: полный балл до 10 L*, ноль после 40 L*. Для фонов с
    preference coefficient < 1 обе границы пропорционально сжимаются: чем
    менее приятен фон по BCP/YouGov, тем точнее модель должна совпасть по тону.
    """
    if not hist or len(hist) != TONE_BINS:
        return None
    scale = max(0.5, min(1.0, float(tolerance_scale)))
    full = 10.0 * scale
    cutoff = 40.0 * scale
    d = np.abs(TONE_CENTERS - bg_lstar)
    kernel = np.where(d <= full, 1.0,
                      np.maximum(0.0, 1.0 - (d - full) / (cutoff - full)) ** 2)
    return float(np.dot(np.asarray(hist, dtype=float), kernel))


def _merge_families(clusters, max_deg=25.0):
    """Сливаем кластеры одного цветового семейства (разница оттенка <= max_deg)."""
    merged = []
    for c in clusters:
        if merged and c.get("hue") is not None and merged[-1].get("hue") is not None:
            top = merged[-1]
            dh = c["hue"] - top["hue"]
            if dh > 180:
                c_h = c["hue"] - 360
            elif dh < -180:
                c_h = c["hue"] + 360
            else:
                c_h = c["hue"]
            d = min(abs(dh), 360 - abs(dh))
            if d <= max_deg:
                a_share, b_share = top["share"], c["share"]
                s = a_share + b_share
                sa = top.get("share_all", 0) + c.get("share_all", 0)
                top["hue"] = round(((top["hue"] * a_share + c_h * b_share) / s) % 360, 1)
                top["val"] = round((top.get("val", 0) * a_share + c.get("val", 0) * b_share) / s, 3)
                top["sat"] = round((top.get("sat", 0) * a_share + c.get("sat", 0) * b_share) / s, 3)
                top["vstd"] = round(max(top.get("vstd", 0), c.get("vstd", 0)), 3)
                ah, bh = top.get("tone_hist"), c.get("tone_hist")
                if ah and bh and len(ah) == TONE_BINS and len(bh) == TONE_BINS:
                    hist = (np.asarray(ah) * a_share + np.asarray(bh) * b_share) / s
                    hist /= max(1e-9, float(hist.sum()))
                    mean_l = float(np.dot(hist, TONE_CENTERS))
                    std_l = float(np.sqrt(np.dot(hist, (TONE_CENTERS - mean_l) ** 2)))
                    top["tone_hist"] = [round(float(x), 4) for x in hist]
                    top["lstar"] = round(mean_l, 2)
                    top["lstd"] = round(std_l, 2)
                top["share"] = round(s, 3)
                top["share_all"] = round(sa, 3)
                continue
        merged.append(dict(c))
    return merged


def palette_from_pixels(px):
    """Up to three separate perceptual colors; never average hue families.

    Pixels are quantized and clustered in CAM02-UCS. Six internal clusters
    preserve distinct colors and shades; the three largest are returned.
    Every representative is an actual color from the image, not an RGB mix.
    """
    total = len(px)
    if not total:
        return {"palette": [], "rainbow": 0.0, "achromatic": True,
                "rgb": None, "neutral": 1.0}

    arr = np.asarray(px, dtype=np.uint8)
    quantized = ((arr.astype(np.uint16) // 8) * 8 + 4).clip(0, 255).astype(np.uint8)
    rgbs, weights = np.unique(quantized, axis=0, return_counts=True)
    weights = weights.astype(float)
    cam = np.asarray(cspace_convert(rgbs, "sRGB255", "CAM02-UCS"), dtype=float)

    # Deterministic weighted k-means++: more internal clusters than output
    # colors prevent red/orange/gold or light/dark shades from being averaged.
    k = min(6, len(rgbs))
    first = int(np.argmax(weights))
    chosen = [first]
    centers = [cam[first].copy()]
    min_d2 = np.sum((cam - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        score = weights * min_d2
        score[chosen] = -1.0
        idx = int(np.argmax(score))
        chosen.append(idx)
        centers.append(cam[idx].copy())
        min_d2 = np.minimum(min_d2, np.sum((cam - cam[idx]) ** 2, axis=1))
    centers = np.asarray(centers)

    labels = np.zeros(len(rgbs), dtype=int)
    for _ in range(20):
        d2 = np.sum((cam[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1)
        new_centers = centers.copy()
        for ci in range(k):
            mask = new_labels == ci
            if np.any(mask):
                new_centers[ci] = np.average(cam[mask], axis=0, weights=weights[mask])
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers

    hsv = np.asarray([colorsys.rgb_to_hsv(*(rgb.astype(float) / 255.0))
                      for rgb in rgbs], dtype=float)
    pixel_colorful = (hsv[:, 1] >= 0.18) & (hsv[:, 2] >= 0.15)
    colorful_mass = float(weights[pixel_colorful].sum())
    neutral_share = round(max(0.0, 1.0 - colorful_mass / total), 3)

    # Rainbow is still measured from actual hue coverage, independent of the
    # three displayed clusters.
    bins = np.zeros(36, dtype=float)
    for h, w in zip(hsv[pixel_colorful, 0] * 360.0, weights[pixel_colorful]):
        bins[int(h // 10) % 36] += w
    sig = int(np.sum(bins / max(1.0, colorful_mass) >= 0.03))
    coverage = sig / 36.0
    rainbow = max(0.0, min(1.0, (coverage - 0.12) / 0.5))

    candidates = []
    tone_edges = np.linspace(0, 100, TONE_BINS + 1)
    for ci in range(k):
        mask = labels == ci
        mass = float(weights[mask].sum())
        if mass / total < 0.025:
            continue
        idxs = np.flatnonzero(mask)
        nearest = idxs[int(np.argmin(np.sum((cam[mask] - centers[ci]) ** 2, axis=1)))]
        rep = tuple(int(v) for v in rgbs[nearest])
        ws = weights[mask]
        hs = hsv[mask, 0] * 2.0 * np.pi
        sat = float(np.average(hsv[mask, 1], weights=ws))
        val = float(np.average(hsv[mask, 2], weights=ws))
        sin_h = float(np.average(np.sin(hs), weights=ws))
        cos_h = float(np.average(np.cos(hs), weights=ws))
        hue = float(np.degrees(np.arctan2(sin_h, cos_h)) % 360.0)
        lvals = _lstar_values(rgbs[mask])
        lstar = float(np.average(lvals, weights=ws))
        lstd = float(np.sqrt(np.average((lvals - lstar) ** 2, weights=ws)))
        hist = np.histogram(lvals, bins=tone_edges, weights=ws)[0].astype(float)
        hist /= max(1.0, float(hist.sum()))
        vv = hsv[mask, 2]
        vstd = float(np.sqrt(np.average((vv - val) ** 2, weights=ws)))
        _, chroma, _ = _lch_of(rep)
        is_neutral = chroma < 10.0 or sat < 0.16 or val < 0.15
        candidates.append({
            "hex": hexcode(rep), "_mass": mass,
            "share_all": round(mass / total, 3),
            "hue": None if is_neutral else round(hue, 1),
            "sat": None if is_neutral else round(sat, 3),
            "val": round(val, 3), "vstd": round(vstd, 3),
            "lstar": round(lstar, 2), "lstd": round(lstd, 2),
            "tone_hist": [round(float(x), 4) for x in hist],
        })

    candidates.sort(key=lambda c: -c["_mass"])
    clusters = candidates[:3]
    selected_mass = sum(c["_mass"] for c in clusters)
    for c in clusters:
        c["share"] = round(c["_mass"] / max(1.0, selected_mass), 3)
        c.pop("_mass", None)
    chromatic_share = sum(c["share_all"] for c in clusters if c.get("hue") is not None)
    achromatic = chromatic_share < 0.12
    main = next((c for c in clusters if c.get("hue") is not None),
                clusters[0] if clusters else None)
    main_rgb = _rgb_of_hex(main["hex"]) if main else None
    return {"palette": clusters, "rainbow": round(rainbow, 2),
            "achromatic": achromatic, "rgb": main_rgb,
            "neutral": neutral_share}


def center_family(img):
    """Доминантное семейство центральных ячеек сетки 8×8 (центральные 4×4).
    У персонажей это лицо (кожа); у Gearhead — тёмный экран (None, ахроматика)."""
    w, h = img.size
    crop = img.crop((int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)))
    px = [(p[0], p[1], p[2]) for p in crop.getdata() if p[3] > 24]
    if not px:
        return None
    info = palette_from_pixels(px)
    if info.get("achromatic") or not info["palette"]:
        return None
    top = dict(info["palette"][0])
    top.pop("share", None)
    top.pop("share_all", None)
    top.pop("tone_hist", None)  # для common нужны только hue/sat, не 20 полос
    top.pop("lstar", None)
    top.pop("lstd", None)
    return top  # {hex, hue, sat, val, vstd}


MODEL_WEBP_SIZE = 1024
MODEL_WEBP_QUALITY = 100
BACKDROP_WEBP_SIZE = 512
BACKDROP_WEBP_QUALITY = 100
MODEL_WEBP_LOSSLESS = True
BACKDROP_WEBP_LOSSLESS = True
IMAGE_ENCODING_VERSION = "lossless1"
_image_webp_lock = threading.Lock()


def model_webp_cache_path(short_name, model_name):
    return Path(__file__).parent / "cache" / (
        "modelwebp_lossless1_" + str(short_name).lower() + "_" + quote(str(model_name), safe="") + ".webp")


def _save_webp_image(image, path, max_size, quality, keep_alpha=False, lossless=False):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    image = image.convert("RGBA" if keep_alpha else "RGB")
    if max(image.size) > int(max_size):
        image.thumbnail((int(max_size), int(max_size)), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + str(threading.get_ident()) + ".tmp.webp")
    image.save(tmp, "WEBP", quality=int(quality),
               method=(0 if lossless else 4),
               exact=bool(keep_alpha), lossless=bool(lossless))
    with _image_webp_lock:
        if not path.exists() or path.stat().st_size == 0:
            tmp.replace(path)
        else:
            tmp.unlink(missing_ok=True)
    return path


def cache_model_webp_image(short_name, model_name, image):
    try:
        return _save_webp_image(image, model_webp_cache_path(short_name, model_name),
                                MODEL_WEBP_SIZE, MODEL_WEBP_QUALITY,
                                keep_alpha=True, lossless=MODEL_WEBP_LOSSLESS)
    except Exception:
        return None


def ensure_model_webp(short_name, model_name, source_url=None):
    """Fetch once and cache a full-resolution, pixel-exact lossless WebP."""
    path = model_webp_cache_path(short_name, model_name)
    if path.exists() and path.stat().st_size > 0:
        return path
    urls = []
    if source_url:
        urls.append(source_url)
    for slug in model_slug_candidates(model_name):
        url = ("https://storage.portal-market.com/portals-market/gifts/" +
               str(short_name).lower() + "/models/png/" + slug + ".png")
        if url not in urls:
            urls.append(url)
    for url in urls:
        try:
            response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if response.status_code != 200:
                continue
            image = Image.open(io.BytesIO(response.content)).convert("RGBA")
            return cache_model_webp_image(short_name, model_name, image)
        except Exception:
            continue
    return None


def ensure_backdrop_webp(png_path):
    """Convert cached 512px PNG background to pixel-exact lossless WebP."""
    png_path = Path(png_path)
    webp_path = png_path.with_suffix(".lossless.webp")
    if webp_path.exists() and webp_path.stat().st_size > 0:
        return webp_path
    if not png_path.exists():
        return None
    try:
        image = Image.open(png_path)
        return _save_webp_image(image, webp_path, BACKDROP_WEBP_SIZE,
                                BACKDROP_WEBP_QUALITY, keep_alpha=False,
                                lossless=BACKDROP_WEBP_LOSSLESS)
    except Exception:
        return None


def model_color_full(short_name: str, model_name: str):
    """-> (rgb, url, info{palette, rainbow, achromatic, neutral, center}). Кэш (m8|)."""
    ckey = "m8|" + short_name + "|" + model_name
    c = cache_get(ckey)
    if c is not None:
        # копии без флага common — он уровня коллекции и не должен жить в кэше
        pal = []
        for f in c.get("palette") or []:
            f2 = dict(f)
            f2.pop("common", None)
            pal.append(f2)
        info = {"palette": pal, "rainbow": c.get("rainbow"),
                "achromatic": c.get("achromatic"), "neutral": c.get("neutral"),
                "center": c.get("center")}
        return (tuple(c["rgb"]) if c.get("rgb") else None), c.get("url"), info
    rgb, url, info = None, None, {"palette": [], "rainbow": 0.0, "achromatic": False,
                                   "neutral": 1.0, "center": None}
    for slug in model_slug_candidates(model_name):
        u = ("https://storage.portal-market.com/portals-market/gifts/"
             + short_name + "/models/png/" + slug + ".png")
        try:
            r = requests.get(u, timeout=15)
            if r.status_code != 200:
                continue
            img = Image.open(io.BytesIO(r.content)).convert("RGBA")
            cache_model_webp_image(short_name, model_name, img)
            img.thumbnail((128, 128))
            px = [(p[0], p[1], p[2]) for p in img.getdata() if p[3] > 24]
            info = palette_from_pixels(px)
            info["center"] = center_family(img)
            rgb, url = info.get("rgb"), u
            break
        except Exception:
            continue
    # deepcopy палитры: collection_common помечает common на объектах моделей,
    # и без копии пометки протравляют кэш (в памяти и на диске)
    cache_set(ckey, {"rgb": list(rgb) if rgb else None, "url": url,
                     "palette": copy.deepcopy(info.get("palette")),
                     "rainbow": info.get("rainbow"),
                     "achromatic": info.get("achromatic"),
                     "neutral": info.get("neutral"), "center": info.get("center")})
    return rgb, url, info


def collection_common(models, min_ratio=0.55, hue_thresh=10.0, sat_thresh=0.2):
    """Common-цвет коллекции по ЦЕНТРАЛЬНЫМ доминантам моделей (сетка 8×8):
    кожа есть в центре у всех персонажей -> common; латунь Gearhead (в центре
    тёмный экран) и дерево Трэнта (sat 0.91 против 0.5 у кожи) не попадают.
    Кластер по окну (оттенок ±10°, насыщенность ±0.2); покрытие >= min_ratio
    моделей с центром -> common; далее common помечается у ВСЕХ семейств
    моделей в том же окне (кожа — это и есть их доминант)."""
    cents = []
    for m in models:
        cen = m.get("center")
        if cen and cen.get("hue") is not None and cen.get("hex"):
            cents.append((cen, m.get("name")))
    n_all = len([m for m in models if m.get("palette")])
    if not cents or n_all < 3:
        return []
    order = sorted(range(len(cents)), key=lambda i: (cents[i][0]["hue"], cents[i][0].get("sat") or 0))
    used = [False] * len(cents)
    out = []
    common_members = []
    for i in order:
        if used[i]:
            continue
        hi = cents[i][0]["hue"]
        si = cents[i][0].get("sat") or 0.5
        mem = []
        for j in range(len(cents)):
            if used[j]:
                continue
            hj = cents[j][0]["hue"]
            sj = cents[j][0].get("sat") or 0.5
            dh = abs(hi - hj) % 360
            dh = min(dh, 360 - dh)
            if dh <= hue_thresh and abs(si - sj) <= sat_thresh:
                mem.append(j)
                used[j] = True
        models_cov = {cents[j][1] for j in mem}
        if len(models_cov) / n_all >= min_ratio:
            for j in mem:
                common_members.append(cents[j][0])
            cen_hex = cents[i][0]["hex"]
            out.append({"hex": cen_hex, "rgb": list(_rgb_of_hex(cen_hex)),
                        "models": len(models_cov), "ratio": round(len(models_cov) / n_all, 2)})
    if common_members:
        # ОДНО окно вокруг медианы кластера (не per-member — те цепляют латунь
        # через самый насыщенный член). Кожа семейств sat 0.44-0.68 — внутри;
        # латунь Гирхеда 0.79 против медианы ~0.55 — снаружи, при ЛЮБОМ раскладе.
        mhs = sorted(mc["hue"] for mc in common_members)
        mss = sorted(mc.get("sat") or 0.5 for mc in common_members)
        med_h = mhs[len(mhs) // 2]
        med_s = mss[len(mss) // 2]
        for m in models:
            for c in m.get("palette") or []:
                if c.get("hue") is None or not c.get("hex"):
                    continue
                cs = c.get("sat") or 0.5
                dh = abs(c["hue"] - med_h) % 360
                dh = min(dh, 360 - dh)
                if dh <= hue_thresh and abs(cs - med_s) <= sat_thresh:
                    c["common"] = True
    return sorted(out, key=lambda x: -x["ratio"])


def fetch_lottie(url: str):
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def lottie_bg_color_from_data(data: dict):
    if not data:
        return None
    for layer in data.get("layers", []):
        if layer.get("nm") == "Background":
            for shape in layer.get("shapes", []):
                for it in shape.get("it", []):
                    if it.get("ty") == "gf":
                        k = it.get("g", {}).get("k", {}).get("k", [])
                        if len(k) >= 4:
                            return tuple(round(v * 255) for v in k[1:4])
    return None


GRADIENT_MAP_FILE = Path(__file__).parent / "cache" / "backdrop_gradients.json"
_gradient_map = None


def _load_gradient_map():
    global _gradient_map
    if _gradient_map is None:
        try:
            _gradient_map = json.loads(GRADIENT_MAP_FILE.read_text())
        except Exception:
            _gradient_map = {}
    return _gradient_map


def lottie_bg_gradient(data: dict):
    """Все стопы градиента слоя Background: [(pos, (r,g,b)), ...]."""
    if not data:
        return None
    for layer in data.get("layers", []):
        if layer.get("nm") == "Background":
            for shape in layer.get("shapes", []):
                for it in shape.get("it", []):
                    if it.get("ty") == "gf":
                        k = it.get("g", {}).get("k", {}).get("k", [])
                        n = len(k) // 4
                        if n >= 2:
                            return [(k[i * 4], tuple(round(v * 255) for v in k[i * 4 + 1:i * 4 + 4]))
                                    for i in range(n)]
    return None


def save_gradient(name: str, stops):
    """Запоминаем официальный градиент бэкдропа по имени (имена общие для всех
    коллекций Telegram)."""
    with _cache_lock:
        m = _load_gradient_map()
        if name not in m:
            m[name] = [[p, list(rgb)] for p, rgb in stops]
            try:
                GRADIENT_MAP_FILE.write_text(json.dumps(m, ensure_ascii=False))
            except Exception:
                pass


def synthetic_bg_png(name: str, rgb_hint=None, size: int = 512):
    """Синтез фона: радиальный градиент по сохранённым официальным стопам имени;
    если имени нет в карте — плоская заливка rgb_hint. Для коллекций без
    Fragment-lottie (finepen)."""
    stops = _load_gradient_map().get(name)
    if stops:
        stops = sorted(stops, key=lambda s: s[0])
        c1 = np.array(stops[0][1], dtype=float)
        c2 = np.array(stops[-1][1], dtype=float)
        yy, xx = np.mgrid[0:size, 0:size]
        d = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size * 0.7071)
        t = np.clip(d, 0, 1)[..., None]
        arr = (c1 + (c2 - c1) * t).astype("uint8")
    elif rgb_hint:
        arr = np.zeros((size, size, 3), dtype="uint8")
        arr[:, :] = rgb_hint
    else:
        return None
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, "PNG")
    return buf.getvalue()


def render_png_from_data(data: dict, keep=("Background",), frame: int = 3):
    """rlottie-рендер только указанных слоёв. Импорт ленивый — на Mac rlottie нет."""
    if not data:
        return None
    try:
        from rlottie_python import LottieAnimation
    except ImportError:
        return None
    try:
        layers = [l for l in data.get("layers", []) if l.get("nm") in keep]
        if not layers:
            return None
        slim = dict(data)
        slim["layers"] = layers
        img = LottieAnimation.from_data(json.dumps(slim)).render_pillow_frame(frame_num=frame)
        base = Image.new("RGBA", img.size, (17, 20, 26, 255))
        base.alpha_composite(img.convert("RGBA"))
        buf = io.BytesIO()
        base.convert("RGB").save(buf, "PNG")
        return buf.getvalue()
    except Exception:
        return None


def hexcode(rgb):
    return "#%02X%02X%02X" % tuple(rgb) if rgb else None


_PERCEPTUAL_COLOR_FIT_CACHE = {}


def _perceptual_color_fit(hex_color, bg_rgb):
    """CAM02-UCS fit: prevents hue-only false positives."""
    if not hex_color:
        return 1.0
    bg_key = tuple(int(v) for v in bg_rgb)
    key = (hex_color, bg_key)
    cached = _PERCEPTUAL_COLOR_FIT_CACHE.get(key)
    if cached is not None:
        return cached
    delta_e = _deltaE_rgb(_rgb_of_hex(hex_color), bg_key)
    fit = max(PERCEPTUAL_COLOR_FLOOR,
              float(np.exp(-(delta_e ** 2) /
                           (2.0 * PERCEPTUAL_COLOR_SIGMA ** 2))))
    _PERCEPTUAL_COLOR_FIT_CACHE[key] = fit
    return fit


def pair_score(bg_rgb, model_info, apply_preference=True, tone_tolerance_scale=None, bg_name=None) -> float:
    """v2.18: perceptual color fit + contrast-accent allowance + prior.

    Hue differences up to 12° are a full match. The tone score combines the
    full model distribution with the dominant central surface so shadows and
    outlines cannot make an unrelated dark background look ideal. The dark
    saturation exception is capped at a moderate 0.50. CAM02-UCS guards
    against hue-only false positives; a secondary contrasting accent up to
    35% is allowed when the dominant family matches the backdrop.

    Если preference coefficient ниже 1, тональный коридор 10/40 L* сжимается
    на тот же коэффициент. Положительный prior коридор не расширяет.

    Для каждого hue-family храним 20 полос perceptual lightness L*, поэтому
    светлый и тёмный оттенки больше не превращаются в ложный «средний» цвет.
    До 10 L* от фона — полный тональный балл; далее идёт квадратичный спад до
    нуля на 40 L*. Широкое семейство штрафуется по lstd, а светлые фоны ещё
    сравниваются с насыщенностью семейства. Для явно тёмного фона разрешён
    естественный уход оттенка к серому/чёрному. common и нейтральные не участвуют."""
    if not bg_rgb:
        return 0.0
    preference_profile = color_preference(bg_rgb, bg_name)
    if tone_tolerance_scale is None:
        tone_tolerance_scale = (min(1.0, preference_profile["coefficient"])
                                if apply_preference else 1.0)
    tone_tolerance_scale = max(0.5, min(1.0, float(tone_tolerance_scale)))

    def finish(raw):
        raw = round(min(100.0, max(0.0, float(raw))), 2)
        return apply_color_preference(raw, bg_rgb, bg_name) if apply_preference else raw

    bh, bs, bv = _hue_of(bg_rgb)
    bg_lstar = _lstar_rgb(bg_rgb)
    pal = model_info.get("palette") or []
    if not pal:
        return 0.0
    bg_achro = bool(preference_profile.get("achromatic"))
    if model_info.get("achromatic"):
        if not bg_achro:
            return finish(15.0)
        h0 = pal[0].get("hex")
        if not h0:
            return 0.0
        model_lstar = _lstar_rgb(_rgb_of_hex(h0))
        return finish(max(0.0, 100 - abs(model_lstar - bg_lstar) / 32.0 * 100))
    if bg_achro:
        # серый/ахроматичный фон + цветная модель — НЕ монохром
        # (голоса: 201 nope из 202). Ниже порога колоды, чтобы не засорять.
        return finish(5.0)
    center = model_info.get("center") or {}
    center_hue = center.get("hue")
    center_lstar = None
    center_rgb = None
    if center.get("hex"):
        center_rgb = _rgb_of_hex(center["hex"])
        center_lstar = _lstar_rgb(center_rgb)

    match = 0.0
    mismatch = 0.0
    matched_share = 0.0
    for c in pal:
        if c.get("hue") is None or c.get("common"):
            continue
        sa = c.get("share_all", c["share"])
        d = abs(c["hue"] - bh) % 360
        d = min(d, 360 - d)
        if d <= HUE_ZERO_MATCH_DEG:
            matched_share += sa
            sim_h = (1.0 if d <= HUE_FULL_MATCH_DEG else
                     max(0.0, 1.0 - (d - HUE_FULL_MATCH_DEG) /
                         (HUE_ZERO_MATCH_DEG - HUE_FULL_MATCH_DEG)))
            tone = _tone_fit(c.get("tone_hist"), bg_lstar, tone_tolerance_scale)
            if tone is None:  # совместимость со старым cache до пересчёта m7
                family_lstar = c.get("lstar")
                if family_lstar is None and c.get("hex"):
                    family_lstar = _lstar_rgb(_rgb_of_hex(c["hex"]))
                delta_l = abs((family_lstar if family_lstar is not None else bg_lstar) - bg_lstar)
                full_l = 10.0 * tone_tolerance_scale
                cutoff_l = 40.0 * tone_tolerance_scale
                tone = (1.0 if delta_l <= full_l else
                        max(0.0, 1.0 - (delta_l - full_l) / (cutoff_l - full_l)) ** 2)
            center_family = False
            if center_lstar is not None and center_hue is not None:
                center_dh = abs(c["hue"] - center_hue) % 360.0
                center_dh = min(center_dh, 360.0 - center_dh)
                if center_dh <= 25.0:
                    center_family = True
                    center_tone = _tone_delta_fit(
                        abs(center_lstar - bg_lstar), tone_tolerance_scale)
                    tone = ((1.0 - CENTER_TONE_WEIGHT) * tone +
                            CENTER_TONE_WEIGHT * center_tone)
            family_lstar = c.get("lstar")
            if family_lstar is None and c.get("hex"):
                family_lstar = _lstar_rgb(_rgb_of_hex(c["hex"]))
            lstd = c.get("lstd")
            if lstd is None:
                lstd = (c.get("vstd") or 0.0) * 100.0
            coherence = 1.0 if lstd <= 12.0 else max(0.30, 1.0 - (lstd - 12.0) / 35.0)
            sat_gap = abs((c.get("sat") if c.get("sat") is not None else bs) - bs)
            chroma = 1.0 if sat_gap <= 0.12 else max(0.10, 1.0 - (sat_gap - 0.12) / 0.35)
            # Тёмный фон может быть естественной тенью того же hue: при разнице
            # от 15 L* не требуем от него сохранять насыщенность светлой модели.
            if family_lstar is not None and family_lstar - bg_lstar >= 15.0:
                chroma = max(chroma, DARK_CHROMA_FLOOR)
            color_fit = _perceptual_color_fit(c.get("hex"), bg_rgb)
            if center_family and center_rgb is not None and center.get("hex"):
                center_fit = _perceptual_color_fit(center.get("hex"), bg_rgb)
                color_fit = ((1.0 - PERCEPTUAL_CENTER_WEIGHT) * color_fit +
                             PERCEPTUAL_CENTER_WEIGHT * center_fit)
            match += sa * sim_h * tone * coherence * chroma * color_fit
        else:
            mismatch += sa
    rainbow = model_info.get("rainbow") or 0.0
    accent_allowance = min(ACCENT_ALLOWANCE_MAX,
                           ACCENT_ALLOWANCE_RATIO * matched_share)
    effective_mismatch = max(0.0, mismatch - accent_allowance)
    score = min(100.0, PAIR_SCORE_SCALE *
                max(0.0, match - 0.6 * effective_mismatch) *
                (1 - 0.6 * rainbow))
    return finish(score)


def similarity(bg, fg) -> float:
    """classic similarityPercentage (bleach-hub) — оставлен для сравнения."""
    de = _deltaE_rgb(bg, fg)
    rgb_dist = float(np.linalg.norm(np.array(bg, dtype=float) - np.array(fg, dtype=float)))
    bh, bs, _ = _hue_of(bg)
    fh, fs, _ = _hue_of(fg)
    cam_sim = 100 * np.exp(-de ** 2 / (2 * 18.04 ** 2))
    rgb_sim = min(100, 100 - (rgb_dist / 150 * 100))
    raw = abs(bh - fh)
    hue_diff = min(raw, 360 - raw)
    hue_sim = 100 - min(100, hue_diff / 45 * 100)
    chroma_sim = 100 - min(100, abs(bs - fs) / 0.3 * 100)
    w = [0.4, 0.3, 0.2, 0.1] if (bs < 0.2 and fs < 0.2) else [0.5, 0.2, 0.2, 0.1]
    return round(w[0] * cam_sim + w[1] * rgb_sim + w[2] * hue_sim + w[3] * chroma_sim, 2)
