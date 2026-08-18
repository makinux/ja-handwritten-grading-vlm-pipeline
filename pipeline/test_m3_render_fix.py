# -*- coding: utf-8 -*-
"""m3_render の bbox・双子安定性・決定論を実データなしで検証する。"""
import copy
import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageChops

import m3_render


ROOT = Path(__file__).resolve().parents[1]
ETL_DATA_DIR = ROOT / "data/etl"
GLYPH_INDEX = ETL_DATA_DIR / "glyph_index.json"
AUDIT_REPORT = ROOT / "out/etl_audit_report.json"


class SkipETLTest(Exception):
    pass


# 外部データに依存しない、挿入長が変わる最小の双子レコード。
SYNTHETIC_ERROR_RECORD = {
    "sample_id": "synthetic-render-e1",
    "pair_id": "synthetic-render-pair",
    "problem": {"text_ja": "Q"},
    "gold_solution": [
        {"step_id": "s1", "text": "pre12suffix"},
        {"step_id": "s2", "text": "unchanged-step"},
    ],
    "mutant_solution": [
        {"step_id": "s1", "text": "pre999suffix"},
        {"step_id": "s2", "text": "unchanged-step"},
    ],
    "injected_errors": [{
        "type": "合成テスト",
        "mutation_site": "s1",
        "span": [3, 6],
    }],
    "control_flag": {"error_free": False},
}


def _control_twin():
    record = copy.deepcopy(SYNTHETIC_ERROR_RECORD)
    record["sample_id"] = "synthetic-render-e0"
    record["mutant_solution"] = []
    record["injected_errors"] = []
    record["control_flag"] = {"error_free": True}
    return record


def _path(label):
    return Path(__file__).with_name(f"._m3_render_test_{os.getpid()}_{label}.png")


def _render(record, label, **kwargs):
    path = _path(label)
    metadata = m3_render.render_record(record, str(path), **kwargs)
    return path, metadata


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mass(image):
    histogram = image.convert("RGB").histogram()
    return sum((index % 256) * count
               for index, count in enumerate(histogram))


def test_twin_boxes():
    """grid では変異スパン外の対応文字 bbox が完全一致する。"""
    gold_path, gold = _render(_control_twin(), "gold")
    mutant_path, mutant = _render(SYNTHETIC_ERROR_RECORD, "mutant")
    try:
        assert gold["pitch_mode"] == mutant["pitch_mode"] == "grid"
        assert gold["char_id_alignment"] == mutant["char_id_alignment"] == "exact"
        assert mutant["label_by_construction"] is True
        assert mutant["bbox_basis"] == "ink-mask"
        assert mutant["bbox_margin_px"] == 2

        gold_s1 = {item["i"]: item for item in gold["char_boxes_px"]["s1"]}
        mutant_s1 = {
            item["i"]: item for item in mutant["char_boxes_px"]["s1"]
        }
        # mutant span は 3 文字、gold span は 2 文字なので後続は +1 で対応。
        for gold_index in list(range(0, 3)) + list(range(5, 11)):
            mutant_index = gold_index if gold_index < 3 else gold_index + 1
            left, right = gold_s1[gold_index], mutant_s1[mutant_index]
            assert left["char"] == right["char"]
            assert left["row"] == right["row"]
            assert left["bbox"] == right["bbox"]

        assert (gold["char_boxes_px"]["s2"]
                == mutant["char_boxes_px"]["s2"])
    finally:
        gold_path.unlink(missing_ok=True)
        mutant_path.unlink(missing_ok=True)


def test_blurred_ink_mass():
    """20 文字で bbox 外周 1px のブラー後インク質量を検査する。"""
    visible = "ABCDEFGHIJKLMNOPQRST"
    spaced = "  ".join(visible)
    ink_record = {
        "sample_id": "synthetic-mass-ink",
        "pair_id": "synthetic-mass-pair",
        "problem": {"text_ja": " "},
        "gold_solution": [{"step_id": "s1", "text": spaced}],
        "mutant_solution": [],
        "injected_errors": [],
        "control_flag": {"error_free": True},
    }
    blank_record = copy.deepcopy(ink_record)
    blank_record["sample_id"] = "synthetic-mass-blank"
    blank_record["gold_solution"][0]["text"] = " " * len(spaced)

    ink_path, metadata = _render(ink_record, "mass_ink")
    blank_path, _blank_metadata = _render(blank_record, "mass_blank")
    try:
        with Image.open(ink_path) as ink_image, Image.open(blank_path) as blank_image:
            difference = ImageChops.difference(
                ink_image.convert("RGB"), blank_image.convert("RGB"))
            boxes = metadata["char_boxes_px"]["s1"]
            assert len(boxes) == 20
            for item in boxes:
                x1, y1, x2, y2 = item["bbox"]
                inner_mass = _mass(difference.crop((x1, y1, x2, y2)))
                expanded = (max(0, x1 - 1), max(0, y1 - 1),
                            min(difference.width, x2 + 1),
                            min(difference.height, y2 + 1))
                outer_mass = _mass(difference.crop(expanded)) - inner_mass
                ratio = outer_mass / max(1, inner_mass)
                assert ratio < 0.001, (
                    f"文字 {item['char']!r} の bbox 外インク比 {ratio:.6f}")
    finally:
        ink_path.unlink(missing_ok=True)
        blank_path.unlink(missing_ok=True)


def test_png_determinism():
    """同じレコードを 2 回描画した PNG はバイト単位で一致する。"""
    first_path, _ = _render(SYNTHETIC_ERROR_RECORD, "determinism_1")
    second_path, _ = _render(SYNTHETIC_ERROR_RECORD, "determinism_2")
    try:
        assert _sha256(first_path) == _sha256(second_path)
    finally:
        first_path.unlink(missing_ok=True)
        second_path.unlink(missing_ok=True)


def test_union_rounding():
    """外接矩形は左上を floor、右下を ceil する。"""
    actual = m3_render._union([
        (0.2, 1.8, 2.01, 3.001),
        (-0.2, 2.0, 1.0, 4.2),
    ])
    assert actual == [-1, 1, 3, 5], actual


def test_etl_problem_line_uses_font():
    """ETL 指定時も問題文を bank へ渡さず、解答行だけ ETL 描画する。"""
    class RecordingBank:
        def __init__(self):
            self.calls = []

        def get(self, char, pseudo_writer_id):
            self.calls.append((char, pseudo_writer_id))
            return Image.new("L", (12, 16), 255), {
                "family": "9G",
                "writer_key": "test",
            }

    bank = RecordingBank()
    previous_bank = m3_render._ETL_GLYPH_BANK
    m3_render._ETL_GLYPH_BANK = bank
    record = _simple_etl_record("解", "synthetic-etl-problem-font")
    record["problem"]["text_ja"] = "問題"
    path = None
    try:
        path, metadata = _render(
            record, "etl_problem_font", glyph_source="etl")
        assert [char for char, _writer in bank.calls] == ["解"]
        assert metadata["char_boxes_px"]["s1"][0]["glyph_source"] == "etl:9G"
        assert metadata["glyph_fallback_rate"] == 0.0
        assert "problem" not in metadata["char_boxes_px"]
    finally:
        m3_render._ETL_GLYPH_BANK = previous_bank
        if path is not None:
            path.unlink(missing_ok=True)


def test_9g_contrast_normalization_keeps_grayscale():
    """9G の p95 stretch は端の混入片を除きつつ階調を保持する。"""
    image = Image.new("L", (20, 20), 0)
    values = [34, 51, 68, 85, 102, 119, 136, 153] * 8
    image.putdata(
        [values.pop(0) if 6 <= x < 14 and 6 <= y < 14 else 0
         for y in range(20) for x in range(20)])
    normalized = m3_render.glyph_bank_module._preprocess_glyph(image, "9G")
    positive = sorted(value for value in normalized.tobytes() if value > 0)
    assert positive[-1] == 255
    assert len(set(positive)) >= 4
    assert m3_render.glyph_bank_module._percentile(positive, 0.95) == 255

    # 9G 実データにある隣接セル由来の明るい端片で背景推定を汚しても、
    # 中央の薄い鉛筆線を消さず、端片を crop bbox に含めない。
    contaminated = Image.new("L", (128, 127), 0)
    pixels = contaminated.load()
    for y in range(18, 112):
        pixels[0, y] = 153
        pixels[1, y] = 136
    shades = (34, 51, 68, 85, 102, 119)
    for y in range(35, 94):
        for x in range(34, 95):
            if (x + y) % 9 < 2 or x in {42, 70, 88}:
                pixels[x, y] = shades[(x + 2 * y) % len(shades)]
    normalized = m3_render.glyph_bank_module._preprocess_glyph(
        contaminated, "9G")
    positive = sorted(value for value in normalized.tobytes() if value > 0)
    assert 40 <= normalized.width < 90
    assert 40 <= normalized.height < 90
    assert positive[-1] == 255
    assert len(set(positive)) >= 4


def _require_etl():
    has_9g = any(ETL_DATA_DIR.rglob("ETL9G_01")) if ETL_DATA_DIR.is_dir() else False
    has_1 = any(ETL_DATA_DIR.rglob("ETL1C_01")) if ETL_DATA_DIR.is_dir() else False
    has_6 = any(ETL_DATA_DIR.rglob("ETL6C_01")) if ETL_DATA_DIR.is_dir() else False
    if not (has_9g and has_1 and has_6):
        raise SkipETLTest("data/etl の ETL9G/ETL1/ETL6 実データなし")
    if not GLYPH_INDEX.is_file():
        m3_render.glyph_bank_module.build_index(ETL_DATA_DIR, GLYPH_INDEX)


def _simple_etl_record(text, pair_id="synthetic-etl-pair"):
    return {
        "sample_id": f"{pair_id}-sample",
        "pair_id": pair_id,
        "problem": {"text_ja": " "},
        "gold_solution": [{"step_id": "s1", "text": text}],
        "mutant_solution": [],
        "injected_errors": [],
        "control_flag": {"error_free": True},
    }


def test_etl_twin_boxes():
    """ETL 疑似筆者でも双子の変異スパン外 bbox は完全一致する。"""
    _require_etl()
    gold_path, gold = _render(
        _control_twin(), "etl_gold", glyph_source="etl")
    mutant_path, mutant = _render(
        SYNTHETIC_ERROR_RECORD, "etl_mutant", glyph_source="etl")
    try:
        assert gold["writer_consistent"] is False
        assert mutant["writer_consistent"] is False
        assert gold["pseudo_writer_id"] == mutant["pseudo_writer_id"]
        gold_s1 = {item["i"]: item for item in gold["char_boxes_px"]["s1"]}
        mutant_s1 = {
            item["i"]: item for item in mutant["char_boxes_px"]["s1"]
        }
        for gold_index in list(range(0, 3)) + list(range(5, 11)):
            mutant_index = gold_index if gold_index < 3 else gold_index + 1
            left, right = gold_s1[gold_index], mutant_s1[mutant_index]
            assert left["char"] == right["char"]
            assert left["bbox"] == right["bbox"]
            assert left["glyph_source"] == right["glyph_source"]
        assert gold["char_boxes_px"]["s2"] == mutant["char_boxes_px"]["s2"]
    finally:
        gold_path.unlink(missing_ok=True)
        mutant_path.unlink(missing_ok=True)


def test_etl_png_determinism():
    """同じ ETL レコードを 2 回描画した PNG は byte 同一になる。"""
    _require_etl()
    record = _simple_etl_record("1+2=3/4", "synthetic-etl-determinism")
    first_path, _ = _render(record, "etl_det_1", glyph_source="etl")
    second_path, _ = _render(record, "etl_det_2", glyph_source="etl")
    try:
        assert _sha256(first_path) == _sha256(second_path)
    finally:
        first_path.unlink(missing_ok=True)
        second_path.unlink(missing_ok=True)


def test_etl_font_fallback_source():
    """ETL にない除算記号は char_boxes 上でも font fallback と記録する。"""
    _require_etl()
    path, metadata = _render(
        _simple_etl_record("1÷2", "synthetic-etl-fallback"),
        "etl_fallback", glyph_source="etl")
    try:
        by_char = {item["char"]: item for item in metadata["char_boxes_px"]["s1"]}
        assert by_char["÷"]["glyph_source"] == "font"
        assert by_char["1"]["glyph_source"].startswith("etl:")
        assert by_char["2"]["glyph_source"].startswith("etl:")
        assert metadata["glyph_fallback_rate"] == 1 / 3
    finally:
        path.unlink(missing_ok=True)


def test_etl_fallback_rate_matches_audit():
    """数字・演算子行の fallback 比率を全 family 合算監査集合と照合する。"""
    _require_etl()
    if not AUDIT_REPORT.is_file():
        raise SkipETLTest("out/etl_audit_report.json なし (先に etl_audit.py を実行)")
    report = json.loads(AUDIT_REPORT.read_text(encoding="utf-8"))
    available = set(report["available_unicode_character_list"])
    previous_remaining = {
        item["char"]
        for item in report["previous_oov_21"]["remaining_after_all_families"]
    }
    assert previous_remaining == set("x×、。")

    text = "1+2=3x×、。÷"
    expected_fallbacks = sum(char not in available for char in text)
    assert expected_fallbacks > 0
    path, metadata = _render(
        _simple_etl_record(text, "synthetic-etl-audit-rate"),
        "etl_audit_rate", glyph_source="etl")
    try:
        records = metadata["char_boxes_px"]["s1"]
        assert len(records) == len(text)
        actual_fallbacks = sum(
            item["glyph_source"] == "font" for item in records)
        assert actual_fallbacks == expected_fallbacks
        assert metadata["glyph_fallback_rate"] == expected_fallbacks / len(text)
    finally:
        path.unlink(missing_ok=True)


def main():
    tests = [
        test_twin_boxes,
        test_blurred_ink_mass,
        test_png_determinism,
        test_union_rounding,
        test_etl_problem_line_uses_font,
        test_9g_contrast_normalization_keeps_grayscale,
        test_etl_twin_boxes,
        test_etl_png_determinism,
        test_etl_font_fallback_source,
        test_etl_fallback_rate_matches_audit,
    ]
    passed = 0
    skipped = 0
    for test in tests:
        try:
            test()
        except SkipETLTest as exc:
            skipped += 1
            print(f"[SKIP] {test.__name__}: {exc}")
        else:
            passed += 1
            print(f"[PASS] {test.__name__}")
    print(f"PASS: {passed} tests; SKIP: {skipped} tests")


if __name__ == "__main__":
    main()
