実装完了しました。コミットはしていません。

### 変更ファイル

- `pipeline/m3_render.py`
- `pipeline/test_m3_render_fix.py`
- `pipeline/etl_audit.py`

### 要件対応

1. **文字IDベース乱数**  
   `m3_render.py:87-162`。mutation span 前後を照合し、後続文字を gold index に整列。不確定時は `char_id_alignment="fallback"`。

2. **grid 固定文字送り**  
   `m3_render.py:195-239`。既定は `grid`、`render_record(..., pitch_mode="natural")` で従来方式を選択可能。

3. **ink-mask bbox**  
   `m3_render.py:166-192`。一時 L mask の実インク領域を取得し、全方向2px拡張。`bbox_basis` と `bbox_margin_px` を出力。

4. **`_union` の floor/ceil 化**  
   `m3_render.py:63-68`。

5. **出力スキーマ拡張**  
   `m3_render.py:328-347`。`label_by_construction`、`char_boxes_px`、`char_id_alignment`、`pitch_mode` を追加。既存キーは維持。

6. **決定論維持**  
   文字乱数と撮像ノイズを独立したSHA-256 seedから生成。

7. **自動テスト追加**  
   `test_m3_render_fix.py:64-153`。双子bbox、ブラー後インク質量、PNG SHA-256、`_union` を検査。

8. **ETL監査**  
   `etl_audit.py` に9G/9Bパーサ、writer被覆、OOV、比較PNG、JSON/標準出力を実装。[AIST ETL9G仕様](https://etlcdb.db.aist.go.jp/etlcdb/etln/form_e9g.htm)、[ETL9B仕様](https://etlcdb.db.aist.go.jp/etlcdb/etln/form_e9b.htm)準拠です。

### 検証結果

- `python pipeline/test_m3_render_fix.py` → **PASS: 4 tests**
- `python pipeline/etl_audit.py` → 使い方を表示、終了コード0
- ETL合成レコード smoke → パース、dummy除外、OOV、JSON、比較PNGすべてPASS
- `run_phase0_bootstrap.py` / `run_render.py` → import・実レコードrender smoke PASS
- `py_compile` / `git diff --check` → PASS

実ETLデータは未配置のため、607,200件の本走査自体は未実施です。