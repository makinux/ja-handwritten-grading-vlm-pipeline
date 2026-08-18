# 実装タスク: ETL1/ETL6 対応+疑似筆者グリフバンク・レンダラ

背景: ETL9 実データ監査(out/etl_audit_report.json)で2つの設計変更が確定した。
(a) 筆者は各自約152字種しか書いておらず「筆者ID=ページスタイル」は不成立 → **疑似筆者方式**へ。
(b) 答案コーパスの OOV 率 70.9%(数字・演算子・句読点)→ **ETL1/ETL6 で数字・カタカナ・英字・記号を補い**、残余はフォント fallback。
データは data/etl/ に配置済み: ETL9G_01..50(8199B)、ETL9B_1..5(576B)、ETL1/ETL1/ETL1C_01..13(2052B, M-type)、ETL6/ETL6/ETL6C_01..12(2052B, M-type)。ETL1INFO/ETL6INFO はテキスト。

## タスク1: etl_audit.py の ETL1/ETL6(M-type)対応

- M-type 2052 バイトレコードのパーサを追加(AIST 公式フォーマット: <http://etlcdb.db.aist.go.jp/etlcdb/etln/form_m.htm>)。要点: bytes 1-2 Data Number, 3-4 Character Code(ASCII 2文字), 7-8 Serial Sheet Number, 31-32 JIS Code 等、画像は 64x63 4bit(2016 バイト)。**Character Code の文字種判定は CO59/JIS の変換表を docstring に明記**し、ETL1(カタカナ・数字・英字・記号)/ETL6(同系)を Unicode へ正しくマップする(半角カナは全角カタカナへ正規化)。マップ不能コードは undecodable として報告。
- レコード長 2052 の自動判別を _discover_files に追加(ETL1 と ETL6 の区別はパス名 or INFO ファイルで。INFO はスキャン対象から除外)。
- 監査出力に family 別の distinct 文字・writer 統計を追加し、**OOV 計算は全 family の合算文字集合**で行う。さらに「前回 OOV 21 字種のうち ETL1/6 でカバーされた字種/残余」を明示するフィールドを追加。
- 実行して out/etl_audit_report.json を更新し、残余 OOV を最終メッセージで報告(コーパスは out/etl_corpus.jsonl)。

## タスク2: pipeline/glyph_bank.py(新規)

ETL 実筆グリフを引くモジュール。要件:

1. **インデックス構築** `build_index(data_dir, out_path)`: 全 family を走査し、char(Unicode)→ [{family, file, offset, writer_key}] を JSON で data/etl/glyph_index.json に保存(data/ は gitignore 済み)。writer_key は 9G では「シート様式ID+ファイル内出現順から導出した一意キー」でよい(厳密筆者同定は将来課題、キーの決定論性のみ必須)。ETL1/6 は Serial Sheet Number。
2. **決定論的グリフ取得** `GlyphBank.get(char, pseudo_writer_id)`: hash(pseudo_writer_id, char) から候補リストの1つを決定論的に選び、PIL の L 画像(インク=白)で返す。同一 (pseudo_writer_id, char) は常に同一グリフ(双子共有の要)。付帯情報 {family, writer_key} も返す。
3. **前処理**: 4bit 階調→L 変換、背景推定して 0 基準化、インク bbox でクロップ、コントラスト正規化。9B は使わない(9G 既定)。
4. 遅延ロード・LRU キャッシュ(ファイルハンドルはオフセット読みで、全量をメモリに載せない)。

## タスク3: m3_render.py への統合

- `render_record(..., glyph_source="font"|"etl")` を追加(既定は "font" のまま=既存挙動不変)。
- "etl" 時: pseudo_writer_id = pair_id から導出(双子共有)。各文字を GlyphBank から取得し、フォントサイズ相当へ**インク高さ基準でスケール**して grid セルに配置(ジッタは既存の文字ID乱数を流用)。グリフが無い文字は**フォントで fallback** し、char_boxes_px の各エントリに `glyph_source: "etl:9G"|"etl:1"|"etl:6"|"font"` を記録。戻り値に `writer_consistent: false`(etl 時)、`pseudo_writer_id`、`glyph_fallback_rate` を追加。
- インク色は既存 style["ink"] を適用(グリフ L 画像をアルファとして着色 paste)。bbox は既存の ink-mask 方式と同じ基準(マージン2px)で。
- 決定論維持(同一レコード→バイト同一 PNG)。

## タスク4: テスト

- test_m3_render_fix.py に追加(実 ETL データが data/etl にある前提で良い。無い環境では skip 表示):
  (a) glyph_source="etl" で双子の変異スパン外 bbox 完全一致
  (b) 同一レコード2回で PNG SHA-256 一致
  (c) fallback 文字(例: "÷")の glyph_source が "font" と記録される
  (d) 数字・演算子を含む行で glyph_fallback_rate が監査の残余 OOV と整合(ゼロでないこと自体は許容)
- 実行して PASS を確認。既存4テストも PASS のまま。

## 完了条件

- `python pipeline/etl_audit.py --data-dir data/etl --corpus out/etl_corpus.jsonl --output out/etl_audit_report.json` が ETL1/6 込みで完走し、残余 OOV を報告
- `python pipeline/test_m3_render_fix.py` 全 PASS
- glyph_index.json 構築の所要時間とサイズを報告
- 変更ファイルと要件対応箇所を最終メッセージで報告。コミットはしない(レビュー後に実施)
