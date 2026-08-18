修正完了しました。コミットはしていません。

- `problem` 行を `glyph_source="etl"` でも常にフォント描画に固定し、docstringへ明記。
- provenance・`char_boxes_px`・fallback率集計から problem 行を除外。
- ETL9Gを背景中央値基準＋p95ストレッチ＋γ=0.75で強調。階調は保持。
- 端の混入筆跡による誤った背景推定・cropも補正。
- 回帰テスト追加。

検証結果:

```text
PASS: 10 tests; SKIP: 0 tests
```

生成物:

- `out/etl_demo_page.png`
  - `out/dataset.jsonl` 1行目
  - `sample_id=math-g7-lin-000000-e1`
  - `error_free=false`
  - `glyph_source="etl"`
- `out/etl_glyph_contrast_check.png`
  - ETL9G「亜・議・識・曜・鑑」の修正前後比較
  - 40px縮小でも視認可能なことを目視確認済み

`py_compile`とPNG整合性確認も通過しています。