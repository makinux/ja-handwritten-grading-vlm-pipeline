# -*- coding: utf-8 -*-
"""KPI 整合モンテカルロ(パネル実測パッケージ項目 H/設計書 v2.1 §7)。

検出 TPR×種別×深刻度の精度から「点数完全一致率」の含意値をシミュレートし、
仮目標 80% の実現可能領域を独立仮定/相関ありの両レジームで求める。
"""
import random

CONTROL = 0.30
TNR = 0.95  # 幻覚誤り率 ≤5% の含意


def simulate(tpr, acc_type, acc_sev, corr_easy=0.0, trials=20000, seed=7):
    """corr_easy: 「素直なサンプル」(全要素成功)の混合率による正の相関。"""
    rng = random.Random(seed)
    hit = 0
    for _ in range(trials):
        if rng.random() < CONTROL:
            hit += rng.random() < TNR
            continue
        k = 1 if rng.random() < 0.5 else 2
        if corr_easy and rng.random() < corr_easy:
            hit += 1
            continue
        denom = 1.0 - corr_easy
        p_t = max(0.0, (tpr - corr_easy) / denom) if corr_easy else tpr
        p_c = max(0.0, (acc_type - corr_easy) / denom) if corr_easy else acc_type
        p_s = max(0.0, (acc_sev - corr_easy) / denom) if corr_easy else acc_sev
        ok = True
        for _e in range(k):
            if not (rng.random() < p_t and rng.random() < p_c
                    and rng.random() < p_s):
                ok = False
                break
        hit += ok
    return hit / trials


def grid_report():
    lines = ["| TPR | 種別 | 深刻度 | 独立仮定 | 相関 0.5 |",
             "|---|---|---|---|---|"]
    feasible_ind = []
    for tpr in (0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97):
        for at in (0.80, 0.90, 0.95):
            for asv in (0.80, 0.90, 0.95):
                p_ind = simulate(tpr, at, asv, corr_easy=0.0)
                p_cor = simulate(tpr, at, asv, corr_easy=0.5)
                if p_ind >= 0.80:
                    feasible_ind.append((tpr, at, asv, p_ind))
                if (tpr, at, asv) in [(0.75, 0.80, 0.80), (0.85, 0.90, 0.90),
                                      (0.93, 0.95, 0.95), (0.95, 0.95, 0.95),
                                      (0.97, 0.95, 0.95)]:
                    lines.append(f"| {tpr} | {at} | {asv} | {p_ind:.3f} "
                                 f"| {p_cor:.3f} |")
    concl = [
        "",
        f"独立仮定で点数完全一致 ≥ 0.80 に到達する組合せ: "
        f"{len(feasible_ind)} 通り",
    ]
    if feasible_ind:
        t = min(f[0] for f in feasible_ind)
        concl.append(f"到達に必要な最小 TPR(グリッド内): {t}")
    concl.append(
        "結論: 下限 KPI(TPR 0.75〜0.85・種別/深刻度 0.8)では 80% は含意されず、"
        "各要素 ≥0.93〜0.95 か正の誤差相関が必要。パネル指摘(B 第1位(b)・"
        "A の反駁)と整合。§7 の仮目標は Phase 0 実測後に本表で再設定する。")
    return "\n".join(lines + concl)
