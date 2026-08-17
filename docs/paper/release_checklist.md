# 公開準備チェックリスト(GitHub public 化+arXiv)

作成: 2026-08-14。スキャン実施済み(生 IP=0 件・メール=0 件・実ユーザー名=committed ファイルに 0 件)。

## A. リポジトリ汎名化(公開前必須・機械的)

- [x] 内部サーバエイリアス EVO-X2機/TITAN V機 → `server-igpu`(EVO-X2)/`server-titanv` 等へ一括置換(対象: docs/4 件・panel_records/b_r1.md・train/2 件)
- [x] 別プロジェクト名 "別プロジェクト" への言及を削除(panel_records/b_r1.md ほか)
- [ ] 運用ログ的記述(ポート番号・kill 手順・ホスト RAM 事情)は docs/ の運用ノートに残すか公開版で簡約するか判断
- [ ] 未追跡の生ログ(panel_records/*_raw.md、r*_prompt_*.md)は**コミットしない**(ホスト名等を含む)
- [x] 最終スキャン再実行(2026-08-17 実施・残存 0 件): `grep -rE "192\.168|(内部ホスト)|nss|@nssv" --include="*"`(.git 除外)で 0 件確認

## B. ライセンス・体裁(要・社内判断)

- [x] LICENSE 追加(MIT・会社決定 2026-08-17)(候補: Apache-2.0/MIT——**会社判断**)
- [x] README を公開向けに書き直し(概要・論文リンク・再現手順・引用情報)
- [ ] CITATION.cff 追加
- [ ] フォント: リポジトリに同梱なし(Dockerfile で IPAex/Noto を導入)——ライセンス問題なし ✓
- [ ] 生成データ(out/)は非同梱のまま。データセット公開は別判断(HF Datasets 等、権利・品質確認後)

## C. 承認(公開前必須・非技術)

- [x] 社内公開承認(2026-08-17 承認済み)(知財・共同研究交渉との整合、特に収集計画・教員実測に関わる記述)
- [x] 論文著者・所属の確定(和山亮介/株式会社ノーザンシステムサービス)
- [ ] arXiv アカウント/endorsement の確認(カテゴリ候補: cs.CL、cross: cs.CV)

## D. 論文側(nlp_annual_draft.md の TODO と連動)

- [x] LaTeX 整形版作成(docs/paper/nlp_manuscript.tex。公式 sty は CFP 公開後に差し替え)・図 4 点作成済み
- [x] 関連研究の文献検証・確定
- [x] 追加シード訓練 3 本(表 2 の検出率差の分散注記)——TITAN V機 で各 ~3h・機械実行可
- [ ] arXiv 英語版の作成(和文確定後に翻訳+英文校正)
- [ ] GitHub URL・arXiv ID の相互記載

## 推奨順序

和文ドラフト確定 → 社内承認申請(並行: 追加シード実験) → リポジトリ汎名化+LICENSE → GitHub public → arXiv 英語版投稿 → 会議投稿(投稿先確定後)
