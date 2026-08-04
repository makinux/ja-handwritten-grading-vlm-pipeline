# -*- coding: utf-8 -*-
"""Phase 0 ブートストラップ実行:M1→G1→M2→G2→M3(座標GT)→検収→
報酬敵対的テスト→KPI モンテカルロ→レポート出力。

実行:  python pipeline/run_phase0_bootstrap.py  (リポジトリルートで)
Docker: README.md 参照。
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen_core
import m3_render
from adversarial_reward_test import run_adversarial_test
from gate_efficacy_test import run_gate_efficacy
from kpi_montecarlo import grid_report
from textonly_probe import run_probe

SEED = 20260804
N_TOTAL = 200
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "out")


def main():
    t0 = time.time()
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "debug"), exist_ok=True)

    # ---- M1/M2 + G1/G2
    kept, stats = gen_core.generate_batch(N_TOTAL, SEED)
    print(f"[gen] kept={len(kept)} G1 {stats['g1_pass']}/{stats['g1_pass']+stats['g1_fail']} "
          f"G2 {stats['g2_pass']}/{stats['g2_pass']+stats['g2_fail']} "
          f"control_ratio={stats['control_ratio']:.3f}")

    # ---- M3 レンダリング(全件)+ペア不変性チェック
    n_debug = 0
    style_by_pair = {}
    pair_style_ok = True
    for rec, _problem in kept:
        img_path = os.path.join(OUT, "images", rec["sample_id"] + ".png")
        dbg_path = None
        if n_debug < 6:
            dbg_path = os.path.join(OUT, "debug",
                                    rec["sample_id"] + "_boxes.png")
            n_debug += 1
        meta = m3_render.render_record(rec, img_path, dbg_path)
        rec["render"] = meta
        rec["render"]["image"] = os.path.relpath(img_path, OUT)
        prev = style_by_pair.get(rec["pair_id"])
        if prev is not None and prev != meta["style_id"]:
            pair_style_ok = False
        style_by_pair[rec["pair_id"]] = meta["style_id"]
    print(f"[render] {len(kept)} pages, pair-style invariance: {pair_style_ok}")

    # ---- 検収(パネル条件 1): 位置 IoU の GT が 100 サンプルで機械的に構成可能か
    err_recs = [r for r, _ in kept if not r["control_flag"]["error_free"]]
    check_n = min(100, len(err_recs))
    ok_n = 0
    for r in err_recs[:check_n]:
        es = r["render"]["error_span_boxes_px"]
        good = bool(es) and all(
            b[0] < b[2] and b[1] < b[3]
            and 0 <= b[0] and 0 <= b[1]
            and b[2] <= r["render"]["page"]["w"]
            and b[3] <= r["render"]["page"]["h"]
            for e in es for b in e["boxes"])
        ok_n += good
    iou_gt_ok = (ok_n == check_n)
    print(f"[検収] 誤りスパン座標 GT 構成可能: {ok_n}/{check_n}")

    # ---- データセット出力
    ds_path = os.path.join(OUT, "dataset.jsonl")
    with open(ds_path, "w", encoding="utf-8") as f:
        for rec, _p in kept:
            for e in rec["injected_errors"]:
                e.pop("_payload", None)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- 報酬の敵対的テスト(パネル条件 6)
    adv = run_adversarial_test([r for r, _ in kept], seed=SEED, n_target=1000)
    print(f"[adv] trials={adv['n_trials']} inversions={adv['n_inversions']} "
          f"rate={adv['inversion_rate']:.4f} pass={adv['pass']}")

    # ---- G1 ゲート実効性(フォルト注入。「通過率でなく精度を測る」)
    ge = run_gate_efficacy(seed=SEED)
    print(f"[gate] recall digit={ge['recall_digit_change']:.3f} "
          f"minus={ge['recall_minus_drop']:.3f} "
          f"false_reject clean={ge['false_reject_clean']:.3f} "
          f"benign={ge['false_reject_benign_paraphrase']:.3f} "
          f"verbose={ge['false_reject_verbose_faithful']:.3f} "
          f"pass={ge['pass']}")

    # ---- text-only プローブ(項目 D): 生成器リーク修正の前後比較
    probe_fixed = run_probe(2000, seed=777, fraction_golds=True)
    probe_legacy = run_probe(2000, seed=777, fraction_golds=False)
    print(f"[probe] AUC fixed={probe_fixed['auc']:.3f} "
          f"legacy={probe_legacy['auc']:.3f}")

    # ---- KPI モンテカルロ(項目 H)
    mc = grid_report()

    # ---- レポート
    ops = "\n".join(f"  - {k}: {v}" for k, v in
                    sorted(stats["per_operator"].items()))
    fam = "\n".join(
        f"| {k} | {v['n']} | {v['inv']} | {v['max_gap']:+.3f} |"
        for k, v in sorted(adv["per_family"].items()))
    report = f"""# Phase 0 ブートストラップ実行レポート

実行日時: {time.strftime('%Y-%m-%d %H:%M:%S')} / 乱数シード: {SEED} / 所要 {time.time()-t0:.1f} 秒

## 生成(M1→G1→M2→G2)

- 生成総数(G2 後): {len(kept)}(誤りあり {len(err_recs)} / 対照 {len(kept)-len(err_recs)})
- G1 通過: {stats['g1_pass']} / {stats['g1_pass']+stats['g1_fail']}
- G2 通過: {stats['g2_pass']} / {stats['g2_pass']+stats['g2_fail']}
- 対照群比率: {stats['control_ratio']:.3f}(目標 0.30±0.05)
- オペレータ別件数:
{ops}

注: 本ブートストラップは LLM 逐語化を含まない(テンプレート逐語化)ため、
G1/G2 の通過率が高いのは想定どおり。ゲートが実働するのは Qwen 系
逐語化・LLM 層オペレータの接続後であり、通過率はそのときの品質指標になる。

## レンダリング(M3)+座標 GT 検収

- レンダリング: {len(kept)} ページ(画像 out/images/、検証用オーバーレイ out/debug/ に 6 枚)
- ペア生成原則(双子のスタイル共有): {"OK" if pair_style_ok else "NG"}
- **検収(パネル条件 1)**: 誤りスパン座標 GT(error_span_boxes_px)が
  機械的に構成可能 — {ok_n}/{check_n} 件 {"→ 合格" if iou_gt_ok else "→ 不合格(要修正)"}

## 報酬関数の敵対的テスト(パネル条件 6)

- 試行 {adv['n_trials']} 件、逆転(不正 > 正当) {adv['n_inversions']} 件、
  逆転率 {adv['inversion_rate']:.4f} → 基準 < 0.01 に **{"合格" if adv['pass'] else "不合格"}**

| ファミリ | n | 逆転 | 最大ギャップ(不正-正当) |
|---|---|---|---|
{fam}

## G1 ゲート実効性(フォルト注入。パネル「通過率でなくゲート精度を測れ」対応)

| 系(各 200 件) | 期待動作 | 実測 |
|---|---|---|
| クリーン | 通す | 誤棄却率 {ge['false_reject_clean']:.3f} |
| 良性言い換え(数値不変) | 通す | 誤棄却率 {ge['false_reject_benign_paraphrase']:.3f} |
| 饒舌だが忠実(既知数値の再言及) | 通す | 誤棄却率 {ge['false_reject_verbose_faithful']:.3f} |
| 数値改変(1 桁ハルシネーション) | 棄却 | 検出率 {ge['recall_digit_change']:.3f} |
| 符号落とし(負号の欠落) | 棄却 | 検出率 {ge['recall_minus_drop']:.3f} |

基準(検出 ≥0.99 かつ誤棄却 ≤0.01)に **{"合格" if ge['pass'] else "不合格"}**。
LLM 逐語化(Qwen3.6)接続後は同じハーネスを実出力に対して回す
(接続点は `verbalizer.LLMVerbalizer`、実機未検証スタブ)。

## text-only プローブ(項目 D・診断)

- 旧生成器(一次方程式の正解が常に整数解): AUC **{probe_legacy['auc']:.3f}**
  — 「分数の出現≒誤り」という生成器由来の強リークを検出
- 修正後(分数解の正解を 40% 許可=非現実的制約の撤廃): AUC **{probe_fixed['auc']:.3f}**
- 修正後の上位識別 bigram(誤り側頻度/対照側頻度): {probe_fixed['top_error_bigrams'][:5]}
- 位置づけ: これは**診断であり合否閾値ではない**(設計書 v2.1 §3 M6)。残る
  予測可能性には「誤りが式を複雑化させる」という正当な因果信号が含まれる。
  最終判定は容量整合対照(同規模テキスト専用モデル vs VLM)で行う(Phase 1)。

## KPI 整合モンテカルロ(項目 H)

{mc}

## 既知の縮小(スタブ)と TODO

- LLM 逐語化は接続点のみ(`verbalizer.LLMVerbalizer`。vLLM 未接続・実機未検証)。
  LLM 層オペレータ(概念誤りの自由生成)は未実装
- 字形は暫定フォント(自前収集字形バンク=統合収集プログラム待ち。割付表は
  docs/collection_plan.csv)
- 撮像層は簡易ノイズのみ/縦書き・分数 2 次元レイアウト未実装
- 入力契約は docs/input_contract.md のドラフト(チーム確定要)
- **教員間一致 r の実測は TODO(教員手配待ち)** — タスク #1
"""
    rp = os.path.join(OUT, "phase0_report.md")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[done] report: {rp}")
    return 0 if (iou_gt_ok and adv["pass"] and ge["pass"]) else 1


if __name__ == "__main__":
    sys.exit(main())
