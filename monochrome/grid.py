#!/usr/bin/env python3
"""Монтаж топ-N пар коллекции: модель поверх чистого рендера фона, сетка с подписями.
Запуск на сервере: venv/bin/python grid.py <short_name> [topn]"""
import io
import json
import sys
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw

BASE = Path(__file__).parent
CACHE = BASE / "cache"
CELL = 256
COLS = 4
LABEL = 26


def load(src):
    if str(src).startswith("http"):
        r = requests.get(src, timeout=20)
        return Image.open(io.BytesIO(r.content)).convert("RGBA")
    return Image.open(src).convert("RGBA")


def main(short, topn=12):
    d = json.load(open(CACHE / ("ctor_" + short + ".json")))
    pairs = d["monochrome_pairs"][:topn]
    rows = (len(pairs) + COLS - 1) // COLS
    grid = Image.new("RGB", (COLS * CELL, rows * (CELL + LABEL)), (14, 17, 22))
    draw = ImageDraw.Draw(grid)
    for i, p in enumerate(pairs):
        x, y = (i % COLS) * CELL, (i // COLS) * (CELL + LABEL)
        bgp = CACHE / ("bg_" + short + "_" + quote(p["backdrop"], safe="") + ".png")
        try:
            cell = load(bgp).resize((CELL, CELL)).convert("RGBA")
        except Exception:
            cell = Image.new("RGBA", (CELL, CELL), (30, 34, 42, 255))
        if p.get("model_img"):
            try:
                mo = load(p["model_img"])
                mo.thumbnail((int(CELL * 0.8), int(CELL * 0.8)))
                cell.alpha_composite(mo, ((CELL - mo.width) // 2, (CELL - mo.height) // 2))
            except Exception:
                pass
        grid.paste(cell.convert("RGB"), (x, y))
        draw.text((x + 6, y + CELL + 6),
                  "{}. {} x {}  {}".format(i + 1, p["model"], p["backdrop"], p["score"]),
                  fill=(230, 233, 238))
    out = CACHE / ("grid_" + short + ".png")
    grid.save(out)
    print("saved", out, grid.size)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "libertyfigure",
         int(sys.argv[2]) if len(sys.argv) > 2 else 12)
