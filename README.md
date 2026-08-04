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
- `run_phase0_bootstrap.py` — 一括実行(200 サンプル生成 → ゲート → レンダリング → 検収 → テスト → `out/phase0_report.md`)

## 実行(Docker)

```bash
docker build -t ja-grading-phase0 .
docker run --rm -v "$(pwd):/work" ja-grading-phase0
```

PowerShell の場合は `-v "${PWD}:/work"`。**Git Bash(MSYS)からはパス変換でマウントが静かに失敗することがある**ため、`MSYS_NO_PATHCONV=1` を付けるか PowerShell を使うこと。成果物は `out/`(dataset.jsonl / images / debug / phase0_report.md)に出力される。

既知のチューニング・バックログ: 報酬の敵対的テストは合格(逆転率 0.10% < 1%)だが、`comment_stuffing`(+0.004 で 1 件逆転)と `over_correct`(余裕 0.012)のマージンが薄い。コメント接地の precision 重視化と over-correction 罰の増強を Phase 1 の報酬チューニングで扱う。

ローカル実行(開発時): Python 3.12+ と Pillow があれば `python pipeline/run_phase0_bootstrap.py`。

## 現時点のスコープ(スタブ)

- LLM 逐語化・LLM 層オペレータ(概念誤りの自由生成)は未接続 — vLLM+Qwen3.6 接続時に G1/G2 が実働する
- 字形は暫定フォント(IPAex/Noto/UD デジタル教科書体)— 自前収集字形バンク(統合収集プログラム)が正系統(設計書 §3 M3)
- 撮像層は簡易ノイズのみ。縦書き・分数等の 2 次元レイアウトは未実装
- **教員間一致 r の実測は TODO(教員手配待ち)** — 設計書 §11 参照
