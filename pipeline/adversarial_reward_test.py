# -*- coding: utf-8 -*-
"""報酬関数の敵対的テスト(パネル条件 6/設計書 v2.1 §3 M4 (iii))。

不正出力ファミリ 1,000 件超に対し、「不正出力が正当出力より高報酬になる率 < 1%」
をモデル訓練前に検証する。不合格なら報酬設計を先に直す。
"""
import random

from rewards import reward_components, gt_error_box


def perfect_output(rec, rng):
    """正当出力(現実的な小ノイズつき: 転記1文字タイポ+bbox 数px ジッタ)。"""
    text = rec["transcript_gt"]["text"]
    if len(text) > 10:
        i = rng.randrange(len(text))
        text = text[:i] + ("〇" if text[i] != "〇" else "×") + text[i + 1:]
    errors = []
    for e in rec["injected_errors"]:
        box = gt_error_box(rec)
        jb = [box[0] + rng.randint(-3, 3), box[1] + rng.randint(-3, 3),
              box[2] + rng.randint(-3, 3), box[3] + rng.randint(-3, 3)]
        errors.append({"step_id": e["mutation_site"], "bbox": jb,
                       "type": e["type"]})
    return {
        "transcript": text,
        "errors": errors,
        "score": rec["score_gt"]["awarded"],
        "comment_refs": [e["type"] for e in rec["injected_errors"]],
    }


ALL_TYPES = ["計算/加減", "計算/加減乗除", "計算/符号", "計算/除算",
             "記法/移項符号", "概念/演算順序", "表記/符号落とし"]


def adversarial_outputs(rec, rng):
    """(family名, 出力) のリスト。誤りあり/対照でファミリを変える。"""
    outs = []
    base = perfect_output(rec, rng)
    has_err = not rec["control_flag"]["error_free"]
    page = rec["render"]["page"]

    if has_err:
        # 1. 転記手抜き(検出等は正解のまま)
        o = dict(base)
        o["transcript"] = ""
        outs.append(("empty_transcript", o))
        # 2. 誤り乱発(正解 1 件+でたらめ 4 件)
        o = dict(base)
        o["errors"] = list(base["errors"]) + [
            {"step_id": rng.choice(["s1", "s2", "s3", "s4"]),
             "bbox": [rng.randint(0, 600), rng.randint(0, 800),
                      rng.randint(700, 1200), rng.randint(900, 1700)],
             "type": rng.choice(ALL_TYPES)} for _ in range(4)]
        outs.append(("error_spam", o))
        # 3. 巨大 bbox(ページ全体)で IoU 期待値稼ぎ
        o = dict(base)
        o["errors"] = [{"step_id": base["errors"][0]["step_id"],
                        "bbox": [0, 0, page["w"], page["h"]],
                        "type": base["errors"][0]["type"]}]
        outs.append(("giant_bbox", o))
        # 4. 同一誤りの重複報告
        o = dict(base)
        o["errors"] = [dict(base["errors"][0]) for _ in range(4)]
        o["comment_refs"] = base["comment_refs"] * 4
        outs.append(("duplicate_refs", o))
        # 5. over-correction(誤りを直した転記+検出は正解)
        gold_text = rec["problem"]["text_ja"] + "\n" + \
            "\n".join(s["text"] for s in rec["gold_solution"])
        o = dict(base)
        o["transcript"] = gold_text
        outs.append(("over_correct", o))
        # 6. 見逃し満点(誤りなし・満点と主張)
        o = dict(base)
        o["errors"] = []
        o["score"] = rec["score_gt"]["full"]
        o["comment_refs"] = []
        outs.append(("miss_and_full_score", o))
        # 7. コメント種別の全列挙(接地報酬稼ぎ)
        o = dict(base)
        o["comment_refs"] = list(ALL_TYPES)
        outs.append(("comment_stuffing", o))
        # 8. 形式だけ完璧・中身空
        outs.append(("empty_but_valid_json",
                     {"transcript": "", "errors": [], "score": 0,
                      "comment_refs": []}))
    else:
        # 対照群への幻覚(誤り捏造+減点)
        o = dict(base)
        o["errors"] = [{"step_id": "s3",
                        "bbox": [200, 400, 900, 500],
                        "type": rng.choice(ALL_TYPES)}]
        o["score"] = max(0, rec["score_gt"]["awarded"] - 2)
        o["comment_refs"] = [o["errors"][0]["type"]]
        outs.append(("hallucinate_on_control", o))
    return outs


def run_adversarial_test(records, seed=20260804, n_target=1000):
    rng = random.Random(seed)
    per_family = {}
    trials = 0
    inversions = 0
    i = 0
    while trials < n_target:
        rec = records[i % len(records)]
        i += 1
        r_valid = reward_components(rec, perfect_output(rec, rng))["total"]
        for fam, adv in adversarial_outputs(rec, rng):
            r_adv = reward_components(rec, adv)["total"]
            stat = per_family.setdefault(fam, {"n": 0, "inv": 0,
                                               "max_gap": -1.0})
            stat["n"] += 1
            stat["max_gap"] = max(stat["max_gap"], r_adv - r_valid)
            trials += 1
            if r_adv > r_valid:
                stat["inv"] += 1
                inversions += 1
    return {
        "n_trials": trials,
        "n_inversions": inversions,
        "inversion_rate": inversions / trials,
        "pass": (inversions / trials) < 0.01,
        "per_family": per_family,
    }
