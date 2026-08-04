# -*- coding: utf-8 -*-
"""M3: 決定論的手書き風レンダラ(ブートストラップ)。

- 一様レンダリング原則: 全サンプルを単一パイプラインで一括レンダリング。
- ペア生成原則: スタイル・ジッタ乱数列は pair_id から導出し、誤りあり/なしの
  双子で完全共有する。
- 字形は暫定フォント(自前収集字形バンク=統合収集プログラムは TODO)。
- 撮像層は簡易ノイズのみ(本格的なドメインランダム化は Phase 0 後半)。
- 座標 GT: 文字単位 bbox から、ステップ bbox と誤りスパン bbox(跨行対応)を
  ピクセル絶対座標で出力する(設計書 v2.1 項目 0)。
"""
import hashlib
import os
import random

from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_CANDIDATES = [
    # Docker (Linux)
    "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
    "/usr/share/fonts/opentype/ipaexfont-mincho/ipaexm.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
    # Windows(ローカル開発時)
    "C:/Windows/Fonts/UDDigiKyokashoN-R.ttc",
    "C:/Windows/Fonts/YuGothM.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
]

PAGE_W, PAGE_H = 1240, 1754
MARGIN_X, TOP_Y = 150, 140
LINE_GAP = 96


def available_fonts():
    fonts = [p for p in FONT_CANDIDATES if os.path.exists(p)]
    if not fonts:
        raise RuntimeError("日本語フォントが見つからない(FONT_CANDIDATES を確認)")
    return fonts


def style_from_pair(pair_id):
    seed = int(hashlib.sha256(pair_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    fonts = available_fonts()
    return {
        "style_id": f"style-{seed:08x}",
        "font": fonts[rng.randrange(len(fonts))],
        "size": rng.randint(40, 48),
        "jitter": rng.uniform(0.6, 1.8),
        "ink": rng.choice([(25, 25, 30), (20, 20, 90), (45, 38, 30)]),
        "seed": seed,
    }


def _union(boxes):
    return [int(min(b[0] for b in boxes)), int(min(b[1] for b in boxes)),
            int(max(b[2] for b in boxes)), int(max(b[3] for b in boxes))]


def render_record(rec, out_png, debug_png=None):
    style = style_from_pair(rec["pair_id"])
    rng = random.Random(style["seed"])  # ジッタ列も双子で共有(ペア生成原則)
    font = ImageFont.truetype(style["font"], style["size"])
    img = Image.new("RGB", (PAGE_W, PAGE_H), (250, 248, 243))
    d = ImageDraw.Draw(img)

    if rec["control_flag"]["error_free"]:
        steps = rec["gold_solution"]
    else:
        steps = rec["mutant_solution"]
    lines = [("problem", rec["problem"]["text_ja"])] + \
        [(s["step_id"], s["text"]) for s in steps]

    char_boxes = {}  # key -> [(char_index, row, box), ...]
    y = TOP_Y
    for key, text in lines:
        x = MARGIN_X if key == "problem" else MARGIN_X + 60
        row = 0
        recs = []
        for i, ch in enumerate(text):
            adv = d.textlength(ch, font=font)
            if x + adv > PAGE_W - MARGIN_X:
                row += 1
                y += LINE_GAP
                x = MARGIN_X + 100
            if not ch.isspace():
                dx = rng.uniform(-style["jitter"], style["jitter"])
                dy = rng.gauss(0, style["jitter"] * 1.6)
                bx = font.getbbox(ch)
                d.text((x + dx, y + dy), ch, font=font, fill=style["ink"])
                recs.append((i, row,
                             (x + dx + bx[0], y + dy + bx[1],
                              x + dx + bx[2], y + dy + bx[3])))
            x += adv + rng.uniform(0, 1.2)
        char_boxes[key] = recs
        y += LINE_GAP

    boxes_px = []
    for key, _text in lines:
        if key == "problem":
            continue
        bs = char_boxes[key]
        if bs:
            boxes_px.append({"step_id": key, "bbox": _union([b[2] for b in bs])})

    error_span_boxes = []
    for ei, err in enumerate(rec["injected_errors"]):
        site = err["mutation_site"]
        lo, hi = err["span"]
        chs = [b for b in char_boxes[site] if lo <= b[0] < hi]
        pad = 1
        while not chs and pad <= 4:  # スパンが空白のみ等の場合は近傍へ拡張
            chs = [b for b in char_boxes[site]
                   if lo - pad <= b[0] < hi + pad]
            pad += 1
        if not chs:
            chs = char_boxes[site]
        rows = sorted({b[1] for b in chs})
        per_row = [_union([b[2] for b in chs if b[1] == r]) for r in rows]
        error_span_boxes.append({"error_ref": ei, "boxes": per_row,
                                 "multiline": len(rows) > 1})

    # 撮像層(簡易): 紙面ノイズ+軽いぼかし(幾何は不変=座標 GT を保存)
    for _ in range(400):
        nx, ny = rng.randrange(PAGE_W), rng.randrange(PAGE_H)
        g = rng.randint(185, 232)
        d.point((nx, ny), fill=(g, g, g))
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    img.save(out_png)

    if debug_png:
        dbg = img.copy()
        dd = ImageDraw.Draw(dbg)
        for b in boxes_px:
            dd.rectangle(b["bbox"], outline=(0, 150, 0), width=2)
        for es in error_span_boxes:
            for b in es["boxes"]:
                dd.rectangle(b, outline=(210, 30, 30), width=3)
        dbg.save(debug_png)

    return {
        "style_id": style["style_id"],
        "font": os.path.basename(style["font"]),
        "writer_consistent": True,
        "noise_profile": "bootstrap-dots-blur",
        "page": {"w": PAGE_W, "h": PAGE_H},
        "boxes_px": boxes_px,
        "error_span_boxes_px": error_span_boxes,
    }
