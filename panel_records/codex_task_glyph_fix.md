# 修正タスク: グリフバンク描画の2点修正

デモ(out/etl_demo_page.png)の目視レビューで2点の修正が必要になった。

1. **問題文は常にフォント描画にする**: 設計(docs/t2i_handwriting_plan.md §4「問題文は印刷フォントのまま残す」)どおり、`glyph_source="etl"` でも key=="problem" の行は必ずフォントで描画する。char_boxes_px の problem 行は従来どおり出力対象外なので provenance 影響なし。docstring に一行明記。

2. **ETL9G グリフのコントラスト正規化を強化**: 現状、9G の鉛筆書きグリフが薄く掠れて合成される(数字系 ETL1/6 は問題なし)。glyph_bank の前処理で、インク画素の強度分布に基づくレスケーリングを入れる(例: 背景推定後、インク画素の p95 強度が 255 になるよう線形ストレッチ+必要なら軽いガンマ)。二値化はしない(階調=筆致テクスチャは保持)。9G の「亜」等の画数の多い字が 40px 級に縮小されても視認できることを、テスト用に out/etl_glyph_contrast_check.png(修正前後の並置、9G の漢字 5 字分)を出力して確認する。

完了条件: `python pipeline/test_m3_render_fix.py` 全 PASS(problem 行フォント化に伴い期待値がずれるテストがあれば修正)、デモ再生成(out/etl_demo_page.png、out/dataset.jsonl の最初の error_free=false レコード、glyph_source="etl")、比較 PNG 出力。コミットはしない。
