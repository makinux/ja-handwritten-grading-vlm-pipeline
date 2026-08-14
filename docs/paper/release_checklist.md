# 公開準備チェックリスト(GitHub public 化+arXiv)

作成: 2026-08-14。スキャン実施済み(生 IP=0 件・メール=0 件・実ユーザー名=committed ファイルに 0 件)。

## A. リポジトリ汎名化(公開前必須・機械的)

- [ ] 内部サーバエイリアス dev187/dev216 → `server-igpu`(EVO-X2)/`server-titanv` 等へ一括置換(対象: docs/4 件・panel_records/b_r1.md・train/2 件)
- [ ] 別プロジェクト名 "physmoe" への言及を削除(panel_records/b_r1.md ほか)
- [ ] 運用ログ的記述(ポート番号・kill 手順・ホスト RAM 事情)は docs/ の運用ノートに残すか公開版で簡約するか判断
- [ ] 未追跡の生ログ(panel_records/*_raw.md、r*_prompt_*.md)は**コミットしない**(ホスト名等を含む)
- [ ] 最終スキャン再実行: `grep -rE "192\.168|reCAPTURER|nss|@nssv" --include="*"`(.git 除外)で 0 件確認

## B. ライセンス・体裁(要・社内判断)

- [ ] LICENSE 追加(候補: Apache-2.0/MIT——**会社判断**)
- [ ] README を公開向けに書き直し(概要・論文リンク・再現手順・引用情報)
- [ ] CITATION.cff 追加
- [ ] フォント: リポジトリに同梱なし(Dockerfile で IPAex/Noto を導入)——ライセンス問題なし ✓
- [ ] 生成データ(out/)は非同梱のまま。データセット公開は別判断(HF Datasets 等、権利・品質確認後)

## C. 承認(公開前必須・非技術)

- [ ] 社内公開承認(知財・共同研究交渉との整合、特に収集計画・教員実測に関わる記述)
- [ ] 論文著者・所属の確定
- [ ] arXiv アカウント/endorsement の確認(カテゴリ候補: cs.CL、cross: cs.CV)

## D. 論文側(nlp_annual_draft.md の TODO と連動)

- [ ] 学会スタイル適用(和文 4 ページ)・図 2 点作成
- [ ] 関連研究の文献検証・確定
- [ ] 追加シード訓練 2〜3 本(表 2 の検出率差の分散注記)——dev216 で各 ~3h・機械実行可
- [ ] arXiv 英語版の作成(和文確定後に翻訳+英文校正)
- [ ] GitHub URL・arXiv ID の相互記載

## 推奨順序

和文ドラフト確定 → 社内承認申請(並行: 追加シード実験) → リポジトリ汎名化+LICENSE → GitHub public → arXiv 英語版投稿 → 年次大会投稿(〆切 1 月中旬目安)
