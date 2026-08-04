# 日本語手書き答案採点 VLM 訓練パイプライン

誤り注入+手書きレンダリングによる label-by-construction 型の VLM 訓練基盤。
設計・レビューの経緯と根拠はすべて設計書側に集約している。

## ドキュメント

| ファイル | 内容 |
|---|---|
| `design_doc_japanese_handwritten_grading_vlm_v2.md` | **設計書 第2.1版**(Web 裏取り+敵対的パネルレビュー反映済み) |
| `adversarial_panel_review_20260804.md` | 敵対的パネル(GPT-5.6 Sol × Claude Opus × 3 ラウンド)の統合報告 |
| `panel_records/` | パネル討論の全文記録(A/B × Round 1–3) |

## Phase 0 ブートストラップ(`pipeline/`)

設計書 §3 の M1→M2→M3 垂直スライスと、パネル必須条件の 2 ハーネスを実装:

- `gen_core.py` — M1 問題・正解生成(solution program、テンプレート逐語化)+ M2 誤り注入(プログラム・ミューテーション、`mutation_site`/`causally_affected_nodes` 分離=項目0)+ G1/G2 ゲート + ルーブリック項目充足による点数 GT
- `m3_render.py` — 決定論レンダラ(文字単位 bbox → ステップ bbox+誤りスパン bbox〔跨行対応〕、ペア生成原則・一様レンダリング原則)
- `rewards.py` — 7 項目報酬(正規化 CER+over-correction 罰、検出、IoU、種別、点数、コメント接地、形式)
- `adversarial_reward_test.py` — **報酬の敵対的テスト**(不正出力 1,000 件で逆転率 < 1% をモデル訓練前に検証。パネル条件 6)
- `kpi_montecarlo.py` — **KPI 整合モンテカルロ**(点数完全一致 80% の実現可能領域。パネル実測パッケージ項目 H)
- `verbalizer.py` — 逐語化の差し替え点(テンプレート実装+vLLM/Qwen3.6 用 OpenAI 互換クライアントの**実機未検証スタブ**)とフォルト注入
- `gate_efficacy_test.py` — **G1 ゲート実効性実測**(数値改変・符号落とし=検出すべき/良性言い換え=通すべき。「通過率でなくゲート精度を測る」)
- `textonly_probe.py` — **text-only プローブ**(項目 D。文字 bigram NB、pair 単位グループ分割。生成器リークの診断)
- `collection_plan.py` — 統合収集プログラム(1,350 枚)の割付表生成 → `docs/collection_plan.csv`
- `run_phase0_bootstrap.py` — 一括実行(200 サンプル生成 → ゲート → レンダリング → 検収 → 敵対的テスト → ゲート実効性 → プローブ → `out/phase0_report.md`)

`docs/` — 統合収集プログラム計画書(割付表+同意書チェックリスト)、入力契約ドラフト(C1 フル/C2 最小の 2 条件、step_id 意味論)、**ハードウェア計画(EVO-X2 単機・予算ゼロ運用パス)**。

## 現在の実行環境方針(予算ゼロ)

訓練用 H100 は当面調達しないため、[docs/hardware_plan.md](docs/hardware_plan.md) の
**EVO-X2 単機パス**で進行する:LLM 逐語化・生成・ゲート実測・ゼロショットベースラインは
EVO-X2(llama.cpp+Qwen3.6-35B-A3B GGUF)、小規模訓練は 2B LoRA(ROCm 実験枠/Kaggle 無料枠)、
後段最適化は RAFT/DPO を先行(GRPO・8B 高速反復・32B・V4-Flash 判定は予算確保後の条件付き工程として凍結)。
機材到着後の最初の一歩は `python pipeline/llm_smoke_test.py`(接続+G1b 実測)。

## 実行(Docker)

```bash
docker build -t ja-grading-phase0 .
docker run --rm -v "$(pwd):/work" ja-grading-phase0
```

PowerShell の場合は `-v "${PWD}:/work"`。**Git Bash(MSYS)からはパス変換でマウントが静かに失敗することがある**ため、`MSYS_NO_PATHCONV=1` を付けるか PowerShell を使うこと。成果物は `out/`(dataset.jsonl / images / debug / phase0_report.md)に出力される。

報酬の敵対的テストは、モデル訓練前に実バグを 3 件検出し、いずれも修正済み:
(1) コメント接地の一様 F1 が種別全列挙(水増し)に僅差で逆転される(+0.004)→ F0.5+全列挙罰へ。
(2) over-correction 罰のサイト文**部分文字列判定**が、短いサイト文("x=1")が他行("3x=12")に偶然含まれるケースで素通し(引き分け +0.000)。
(3) 置換した**全文 CER 比較判定**は、正当出力のタイポが偶然変異文字に当たると正当出力を誤罰(逆転 8 件に悪化)。
→ 最終形は**行単位のスパン局所判定**(サイト文に最も近い転記行が gold 側と mut 側のどちらに編集距離で近いか)。現行マージンは全ファミリ負(over_correct -0.200)。`out/phase0_report.md` 参照。

ローカル実行(開発時): Python 3.12+ と Pillow があれば `python pipeline/run_phase0_bootstrap.py`。

## 現時点のスコープ(スタブ)

- LLM 逐語化・LLM 層オペレータ(概念誤りの自由生成)は未接続 — vLLM+Qwen3.6 接続時に G1/G2 が実働する
- 字形は暫定フォント(IPAex/Noto/UD デジタル教科書体)— 自前収集字形バンク(統合収集プログラム)が正系統(設計書 §3 M3)
- 撮像層は簡易ノイズのみ。縦書き・分数等の 2 次元レイアウトは未実装
- **教員間一致 r の実測は TODO(教員手配待ち)** — 設計書 §11 参照
