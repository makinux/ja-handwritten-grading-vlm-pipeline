# -*- coding: utf-8 -*-
"""G1 ゲートの実効性実測。

パネル確定所見「G1/G2 通過率は品質 KPI にならない——測るべきはゲートの
精度と再現率」への対応。テンプレート逐語化に既知のフォルトを注入し、
g1b_check_texts(逐語化照合)の検出率(再現率)と、良性言い換え・クリーン
出力への誤棄却率を測る。LLM 逐語化(Qwen3.6)接続後は同じハーネスを
実出力に対して回す。
"""
import random

import gen_core
from gen_core import DOMAINS, g1b_check_texts
from verbalizer import (inject_benign_paraphrase, inject_digit_change,
                        inject_minus_drop, inject_verbose_faithful)


def run_gate_efficacy(seed=20260804, n_per_arm=200):
    rng = random.Random(seed)
    arms = {
        "clean": {"n": 0, "flagged": 0},               # 誤棄却を測る
        "benign_paraphrase": {"n": 0, "flagged": 0},   # 誤棄却を測る
        "verbose_faithful": {"n": 0, "flagged": 0},     # 誤棄却を測る
        "digit_change": {"n": 0, "flagged": 0},        # 検出を測る
        "minus_drop": {"n": 0, "flagged": 0},          # 検出を測る
    }
    for arm in arms:
        for _ in range(n_per_arm):
            domain = rng.choice(list(DOMAINS))
            gen_fn, exec_fn, _ = DOMAINS[domain]
            problem = gen_fn(rng)
            steps, _ans = exec_fn(problem["params"], None)
            texts = [s["text"] for s in steps]
            if arm == "digit_change":
                texts, _ = inject_digit_change(texts, rng)
            elif arm == "minus_drop":
                texts, _ = inject_minus_drop(texts, rng)
            elif arm == "benign_paraphrase":
                texts, _ = inject_benign_paraphrase(texts, rng)
            elif arm == "verbose_faithful":
                texts, _ = inject_verbose_faithful(texts, rng)
            flagged = bool(g1b_check_texts(steps, texts))
            arms[arm]["n"] += 1
            arms[arm]["flagged"] += flagged

    recall_digit = arms["digit_change"]["flagged"] / arms["digit_change"]["n"]
    recall_minus = arms["minus_drop"]["flagged"] / arms["minus_drop"]["n"]
    fr_clean = arms["clean"]["flagged"] / arms["clean"]["n"]
    fr_benign = arms["benign_paraphrase"]["flagged"] / \
        arms["benign_paraphrase"]["n"]
    fr_verbose = arms["verbose_faithful"]["flagged"] / \
        arms["verbose_faithful"]["n"]
    return {
        "arms": arms,
        "recall_digit_change": recall_digit,
        "recall_minus_drop": recall_minus,
        "false_reject_clean": fr_clean,
        "false_reject_benign_paraphrase": fr_benign,
        "false_reject_verbose_faithful": fr_verbose,
        "pass": (recall_digit >= 0.99 and recall_minus >= 0.99
                 and fr_clean <= 0.01 and fr_benign <= 0.01
                 and fr_verbose <= 0.01),
    }
