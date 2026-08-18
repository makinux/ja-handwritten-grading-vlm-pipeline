実装完了しました。コミットはしていません。

### 変更ファイル
- `pipeline/realism.py`（新規）
- `pipeline/m3_render.py`
- `pipeline/test_m3_render_fix.py`
- `out/realism_demo_off.png`
- `out/realism_demo_on.png`

### 要件対応
- ベースラインスプライン、サイズドリフト、微回転、4×4弾性変形、滲み・かすれ、セル内字間補正を実装
- `REALISM_LIMITS` に字形ラベル保護上限を集約し、超過時は `ValueError`
- 全乱数を疑似筆者／pair・行キー・安定文字ID・変形名から決定論的に導出
- 各mask変形後にbboxを再計算し、最終mask＋2pxマージンからGTを生成
- font/ETL共通のL-mask経路へ統合
- メタデータへ `realism.applied`、`strength_profile_id`、実効パラメータを追加
- `realism=None` は変更前PNGとfont/ETL双方でSHA-256完全一致

### 検証
`python pipeline/test_m3_render_fix.py`

- **PASS: 14 tests**
- **SKIP: 0**
- 双子bbox一致、PNG決定性、bbox外インク比0.1%未満、上限違反を確認
- デモは `out/dataset.jsonl` の最初の誤りありレコード `math-g7-lin-000000-e1` から生成・目視確認済みです。