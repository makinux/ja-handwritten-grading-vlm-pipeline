# 実装タスク: レンダラ/スキーマ修正(設計書 §2)+ ETL 監査スクリプト

あなたはこのリポジトリの実装担当。docs/t2i_handwriting_plan.md(第2版)の §2「レンダラ/スキーマ修正」と §8 優先4 の道具を実装せよ。対象は `pipeline/m3_render.py` の改修と、新規 `pipeline/etl_audit.py` の作成。既存の呼び出し互換性(run_phase0_bootstrap.py 等からの利用)を壊さないこと。コメント・docstring は既存コードと同様に日本語。

## タスク1: m3_render.py の改修

背景(敵対的パネルで確定したコードの欠陥、panel_records/t2i_synthesis.md 参照):
- 全文字 bbox が出力されていない(内部変数 char_boxes のみ)
- bbox が font.getbbox(フォントメトリクス)由来で、実インクと乖離。最後に GaussianBlur(0.4) が掛かるため可視インクが bbox 外に滲む
- `_union` が右端・下端を int() 切り捨て(外接矩形なら floor(min)/ceil(max) が正しい)
- 乱数を文字ごとに逐次消費するため、誤りあり/なし双子で変異点以降のジッタ列・折返しがずれる(ペア生成原則違反)

実装要件:

1. **文字IDベース乱数**: 各文字のジッタ (dx, dy) と字送りノイズを、逐次消費の rng ではなく `hash(pair_id, step_key, stable_char_index, char)` から決定論的に導出する。**stable_char_index は双子間で非変更文字が同一 ID になるよう定義する**: レコードには `gold_solution` / `mutant_solution` と `injected_errors[].{mutation_site, span}` があるので、変異ステップでは span より前の文字は同一インデックス、span 以降の文字は gold 側インデックス = mutant インデックス − (変異後スパン長 − 変異前スパン長) で対応付ける(変異前スパン長は gold ステップとの差分から算出。取れない場合は保守的に「span 以降も mutant インデックスをそのまま使う」へフォールバックし、その旨をメタデータ `char_id_alignment: "exact"|"fallback"` に記録)。非変異ステップと問題文は素直にインデックス一致。
2. **折返しの双子安定化**: 行折返し位置が双子間でずれる問題への対策として、折返し判定を「gold テキストのレイアウトを基準に両者へ適用」するのではなく、シンプルに**固定文字送り(グリッド)モード**を追加する: `style` に `pitch_mode: "grid"` を追加し、grid モードでは各文字の描画原点を固定ピッチ(フォントサイズ×1.05 目安)で決める。既定は grid(双子安定)。従来の可変送りは `pitch_mode: "natural"` として残す。
3. **実インク bbox(mask 由来)**: 各文字を一時レイヤ(L モード)に描画し、非零画素の外接矩形から bbox を得る。ブラー(σ=0.4)の裾を考慮し、**確定 bbox = 実インク bbox を全方向に margin=2px 膨張**したものとし、メタデータに `bbox_margin_px: 2` と `bbox_basis: "ink-mask"` を記録する(質量99.9%基準の近似。σ=0.4 の 3σ≈1.2px なので 2px で十分)。
4. **_union の修正**: min 側 floor、max 側 ceil。
5. **出力スキーマ拡張**(render_record の戻り値):
   - `label_by_construction: true`(本レンダラは構成的。将来の Edit 系列は false を入れる)
   - `char_boxes_px`: step_key ごとの全文字 bbox のリスト `[{i, row, char, bbox}]`(問題文は除いてよい)
   - `char_id_alignment`(上記1)
   - `pitch_mode`
   - 既存キー(boxes_px, error_span_boxes_px 等)は互換維持。ステップ bbox・誤りスパン bbox は新しい文字 bbox から従来同様に合成する
6. **決定論の維持**: 同一入力レコード→バイト同一 PNG(既存の設計原則)。スタイル導出(style_from_pair)は互換のまま。
7. **検証**: `pipeline/` に `test_m3_render_fix.py` を新規作成し、以下を自動検証する(pytest 形式でなく `python pipeline/test_m3_render_fix.py` で走る素朴なスクリプトでよい、既存のテストスクリプトの流儀に合わせる):
   - (a) 双子レコード(error_free true/false の対)で、**変異スパン外の文字の bbox が完全一致**すること(grid モード)
   - (b) 全文字について、ブラー後画像の当該 bbox 外周 1px 帯のインク質量が bbox 内質量の 0.1% 未満であること(サンプル 20 文字で可)
   - (c) 同一レコード 2 回レンダリングで PNG の SHA-256 一致
   - (d) _union の ceil 化(単体)
   テスト用レコードはファイル冒頭に最小の合成 dict で埋め込む(実データ不要)。

## タスク2: pipeline/etl_audit.py の新規作成

ETL9G/ETL9B の実データ監査スクリプト(設計書 §3 の着手前監査)。データ未入手でも書ける部分を実装し、`--data-dir` にファイルが無ければ使い方を表示して正常終了する。

- ETL9G: 1レコード 8199 バイト、JIS X 0208 区点コード・シリアルシート番号・128×127 4bit 画像等の公知フォーマットをパースする(フォーマット定義は docstring に明記)
- ETL9B: 1レコード 576 バイト、64×63 1bit
- 出力(JSON + 標準出力サマリ): (a) distinct シート番号(=writer 代理キー)数、(b) writer×字種クラスの被覆行列の統計(全クラス揃う writer 数)、(c) 指定コーパス(`--corpus` に JSONL、各行の `text` フィールド)に対する OOV 文字一覧と OOV 率、(d) `--export-samples N` で 9G/9B の同一文字サンプル PNG を並べて出力(目視比較用)
- 依存は標準ライブラリ+PIL のみ

## 完了条件

- `python pipeline/test_m3_render_fix.py` が PASS
- `python pipeline/etl_audit.py --help` 相当(引数なし実行)がエラーなく使い方を表示
- 既存スクリプトのうち m3_render を import するもの(grep で確認せよ)が壊れていないこと(少なくとも import と主要関数呼び出しの смoke)
- 変更ファイル一覧と、各要件(1〜7, タスク2)への対応箇所を最終メッセージで報告

コミットはするな(レビュー後に実施する)。
