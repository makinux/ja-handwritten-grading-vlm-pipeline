# -*- coding: utf-8 -*-
"""m3_render の bbox・双子安定性・決定論を実データなしで検証する。"""
import copy
import hashlib
import os
from pathlib import Path

from PIL import Image, ImageChops

import m3_render


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


def _render(record, label):
    path = _path(label)
    metadata = m3_render.render_record(record, str(path))
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


def main():
    tests = [
        test_twin_boxes,
        test_blurred_ink_mass,
        test_png_determinism,
        test_union_rounding,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"PASS: {len(tests)} tests")


if __name__ == "__main__":
    main()
