# -*- coding: utf-8 -*-
"""M3: 決定論的手書き風レンダラ(ブートストラップ)。

- 一様レンダリング原則: 全サンプルを単一パイプラインで一括レンダリング。
- ペア生成原則: スタイルと文字 ID ベース乱数を pair_id から導出し、誤り
  あり/なしの双子で非変更文字を共有する。
- 字形は暫定フォント(自前収集字形バンク=統合収集プログラムは TODO)。
- 撮像層は簡易ノイズのみ(本格的なドメインランダム化は Phase 0 後半)。
- 座標 GT: 文字ごとの実インク mask から bbox を作り、ステップ bbox と
  誤りスパン bbox(跨行対応)をピクセル絶対座標で出力する。
"""
import hashlib
import math
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
BBOX_MARGIN_PX = 2
BLUR_RADIUS = 0.4


def available_fonts():
    fonts = [p for p in FONT_CANDIDATES if os.path.exists(p)]
    if not fonts:
        raise RuntimeError("日本語フォントが見つからない(FONT_CANDIDATES を確認)")
    return fonts


def style_from_pair(pair_id):
    """pair_id だけから従来互換のスタイルを決定する。"""
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
        # 固定セルが既定。natural は render_record(..., pitch_mode="natural")
        # で従来の可変字送りを選べる。
        "pitch_mode": "grid",
    }


def _union(boxes):
    """複数 bbox の外接矩形を外向きに整数化する。"""
    return [math.floor(min(b[0] for b in boxes)),
            math.floor(min(b[1] for b in boxes)),
            math.ceil(max(b[2] for b in boxes)),
            math.ceil(max(b[3] for b in boxes))]


def _seed_from_parts(*parts):
    """型と区切りの曖昧さがない文字列から SHA-256 seed を作る。"""
    payload = "".join(f"{type(p).__name__}:{len(str(p))}:{p}|" for p in parts)
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _char_variation(pair_id, step_key, stable_char_index, char, jitter):
    """文字インスタンス ID ごとのジッタと字送りノイズを返す。"""
    rng = random.Random(_seed_from_parts(
        pair_id, step_key, stable_char_index, char))
    return (rng.uniform(-jitter, jitter),
            rng.gauss(0, jitter * 1.6),
            rng.uniform(0, 1.2))


def _exact_mutation_alignment(gold_text, mutant_text, span):
    """単一 mutation span の前後を照合し、文字 ID と grid cell を返す。

    span は mutant 側の半開区間である。span より後の文字は、変異後と変異前
    のスパン長差を引いて gold 側 index に揃える。変異スパン内は双子間共有の
    対象外なので専用 ID を付ける。挿入が既存セル数を超える場合は変異セル内に
    重ね、後続文字のアンカーを動かさない。
    """
    if (not isinstance(span, (list, tuple)) or len(span) != 2
            or not all(isinstance(value, int) for value in span)):
        return None
    lo, hi = span
    if not (0 <= lo < hi <= len(mutant_text)):
        return None
    if gold_text[:lo] != mutant_text[:lo]:
        return None

    suffix = mutant_text[hi:]
    if suffix and not gold_text.endswith(suffix):
        return None
    gold_hi = len(gold_text) - len(suffix)
    if gold_hi < lo:
        return None
    gold_span_len = gold_hi - lo
    mutant_span_len = hi - lo
    delta = mutant_span_len - gold_span_len

    aligned = []
    for index, char in enumerate(mutant_text):
        if index < lo:
            stable_id = grid_cell = index
        elif index >= hi:
            stable_id = grid_cell = index - delta
            if not (0 <= stable_id < len(gold_text)
                    and gold_text[stable_id] == char):
                return None
        else:
            offset = index - lo
            stable_id = f"mutation:{lo}:{offset}"
            # 変異領域が長くなっても後続の固定アンカーを押し出さない。
            grid_cell = lo + min(offset, max(0, gold_span_len - 1))
        aligned.append((stable_id, grid_cell))
    return aligned


def _character_alignment(rec, steps):
    """描画対象ステップごとの (stable ID, grid cell) と精度を返す。"""
    direct = {
        step["step_id"]: [(index, index) for index, _ in enumerate(step["text"])]
        for step in steps
    }
    if rec["control_flag"]["error_free"]:
        return direct, "exact"

    errors_by_site = {}
    for error in rec.get("injected_errors", []):
        errors_by_site.setdefault(error.get("mutation_site"), []).append(error)
    gold_by_id = {
        step["step_id"]: step["text"] for step in rec.get("gold_solution", [])
    }
    result = dict(direct)
    exact = True
    mutant_by_id = {step["step_id"]: step["text"] for step in steps}
    for site, errors in errors_by_site.items():
        # 現行 schema は 1 ステップ 1 mutation。複数 span は対応関係が一意に
        # ならないため、保守的に逐次 index へフォールバックする。
        if (len(errors) != 1 or site not in gold_by_id
                or site not in mutant_by_id):
            exact = False
            continue
        aligned = _exact_mutation_alignment(
            gold_by_id[site], mutant_by_id[site], errors[0].get("span"))
        if aligned is None:
            exact = False
            continue
        result[site] = aligned
    return result, "exact" if exact else "fallback"


def _draw_character(img, origin, char, font, ink):
    """文字を一時 L mask に描き、ページへ合成して実インク bbox を返す。"""
    ox, oy = origin
    metric = font.getbbox(char)
    if metric is None:
        return None
    allocation_pad = BBOX_MARGIN_PX + 4
    left = math.floor(ox + metric[0]) - allocation_pad
    top = math.floor(oy + metric[1]) - allocation_pad
    right = math.ceil(ox + metric[2]) + allocation_pad
    bottom = math.ceil(oy + metric[3]) + allocation_pad
    width, height = max(1, right - left), max(1, bottom - top)

    mask = Image.new("L", (width, height), 0)
    md = ImageDraw.Draw(mask)
    md.text((ox - left, oy - top), char, font=font, fill=255)
    local_bbox = mask.getbbox()
    if local_bbox is None:
        return None
    img.paste(ink, (left, top), mask)

    ink_bbox = (left + local_bbox[0], top + local_bbox[1],
                left + local_bbox[2], top + local_bbox[3])
    return [max(0, ink_bbox[0] - BBOX_MARGIN_PX),
            max(0, ink_bbox[1] - BBOX_MARGIN_PX),
            min(PAGE_W, ink_bbox[2] + BBOX_MARGIN_PX),
            min(PAGE_H, ink_bbox[3] + BBOX_MARGIN_PX)]


def _render_line(img, measure, pair_id, key, text, base_y, font, style,
                 alignment, pitch_mode):
    """1 論理行を描画し、文字 bbox と使用した物理行数を返す。"""
    start_x = MARGIN_X if key == "problem" else MARGIN_X + 60
    records = []
    max_row = 0

    if pitch_mode == "grid":
        pitch = style["size"] * 1.05
        # 右端セルの 1 pitch 分もページ内に収まる列数にする。
        columns = max(1, int((PAGE_W - MARGIN_X - start_x) // pitch))
        for index, char in enumerate(text):
            stable_id, grid_cell = alignment[index]
            row, column = divmod(grid_cell, columns)
            max_row = max(max_row, row)
            x = start_x + column * pitch
            y = base_y + row * LINE_GAP
            dx, dy, _advance_noise = _char_variation(
                pair_id, key, stable_id, char, style["jitter"])
            if not char.isspace():
                bbox = _draw_character(img, (x + dx, y + dy), char, font,
                                       style["ink"])
                if bbox is not None:
                    records.append({"i": index, "row": row, "char": char,
                                    "bbox": bbox})
        return records, max_row + 1

    x = start_x
    row = 0
    for index, char in enumerate(text):
        stable_id, _grid_cell = alignment[index]
        dx, dy, advance_noise = _char_variation(
            pair_id, key, stable_id, char, style["jitter"])
        advance = measure.textlength(char, font=font)
        if x + advance > PAGE_W - MARGIN_X:
            row += 1
            x = MARGIN_X + 100
        if not char.isspace():
            bbox = _draw_character(img, (x + dx, base_y + row * LINE_GAP + dy),
                                   char, font, style["ink"])
            if bbox is not None:
                records.append({"i": index, "row": row, "char": char,
                                "bbox": bbox})
        x += advance + advance_noise
    return records, row + 1


def render_record(rec, out_png, debug_png=None, pitch_mode=None):
    """1 レコードを PNG 化し、後方互換キーを含む座標メタデータを返す。"""
    style = style_from_pair(rec["pair_id"])
    if pitch_mode is not None:
        style["pitch_mode"] = pitch_mode
    pitch_mode = style["pitch_mode"]
    if pitch_mode not in {"grid", "natural"}:
        raise ValueError("pitch_mode は 'grid' または 'natural' を指定してください")

    font = ImageFont.truetype(style["font"], style["size"])
    img = Image.new("RGB", (PAGE_W, PAGE_H), (250, 248, 243))
    measure = ImageDraw.Draw(img)

    if rec["control_flag"]["error_free"]:
        steps = rec["gold_solution"]
    else:
        steps = rec["mutant_solution"]
    lines = [("problem", rec["problem"]["text_ja"])] + [
        (step["step_id"], step["text"]) for step in steps
    ]
    step_alignment, char_id_alignment = _character_alignment(rec, steps)
    alignments = {"problem": [(i, i) for i, _ in enumerate(lines[0][1])]}
    alignments.update(step_alignment)

    char_boxes = {}
    y = TOP_Y
    for key, text in lines:
        records, physical_rows = _render_line(
            img, measure, rec["pair_id"], key, text, y, font, style,
            alignments[key], pitch_mode)
        char_boxes[key] = records
        y += physical_rows * LINE_GAP

    boxes_px = []
    for key, _text in lines:
        if key == "problem":
            continue
        records = char_boxes[key]
        if records:
            boxes_px.append({
                "step_id": key,
                "bbox": _union([record["bbox"] for record in records]),
            })

    error_span_boxes = []
    for error_index, error in enumerate(rec.get("injected_errors", [])):
        site = error["mutation_site"]
        lo, hi = error["span"]
        chars = [record for record in char_boxes[site]
                 if lo <= record["i"] < hi]
        pad = 1
        while not chars and pad <= 4:  # スパンが空白のみ等の場合は近傍へ拡張
            chars = [record for record in char_boxes[site]
                     if lo - pad <= record["i"] < hi + pad]
            pad += 1
        if not chars:
            chars = char_boxes[site]
        rows = sorted({record["row"] for record in chars})
        per_row = [_union([record["bbox"] for record in chars
                           if record["row"] == row]) for row in rows]
        error_span_boxes.append({
            "error_ref": error_index,
            "boxes": per_row,
            "multiline": len(rows) > 1,
        })

    # 撮像層(簡易): 紙面ノイズ+軽いぼかし。背景乱数は文字列とは独立させ、
    # 双子で同一の撮像層を保つ。
    noise_rng = random.Random(_seed_from_parts(rec["pair_id"], "page-noise"))
    draw = ImageDraw.Draw(img)
    for _ in range(400):
        nx, ny = noise_rng.randrange(PAGE_W), noise_rng.randrange(PAGE_H)
        gray = noise_rng.randint(185, 232)
        draw.point((nx, ny), fill=(gray, gray, gray))
    img = img.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    img.save(out_png)

    if debug_png:
        debug = img.copy()
        debug_draw = ImageDraw.Draw(debug)
        for box in boxes_px:
            debug_draw.rectangle(box["bbox"], outline=(0, 150, 0), width=2)
        for error_span in error_span_boxes:
            for box in error_span["boxes"]:
                debug_draw.rectangle(box, outline=(210, 30, 30), width=3)
        debug.save(debug_png)

    return {
        "style_id": style["style_id"],
        "font": os.path.basename(style["font"]),
        "writer_consistent": True,
        "noise_profile": "bootstrap-dots-blur",
        "page": {"w": PAGE_W, "h": PAGE_H},
        "label_by_construction": True,
        "bbox_margin_px": BBOX_MARGIN_PX,
        "bbox_basis": "ink-mask",
        "char_boxes_px": {
            key: records for key, records in char_boxes.items()
            if key != "problem"
        },
        "char_id_alignment": char_id_alignment,
        "pitch_mode": pitch_mode,
        # 既存 consumer が利用するキーは維持する。
        "boxes_px": boxes_px,
        "error_span_boxes_px": error_span_boxes,
    }
