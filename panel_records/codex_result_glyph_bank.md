実装完了しました。コミットはしていません。

### 実装内容

- `pipeline/etl_audit.py`
  - ETL1/ETL6 M-type 2052-byte パーサ追加。
  - 半角カナの全角正規化、Character Code/JIS X 0201 判定、変換不能 `/X` の報告。
  - INFO 除外と2052-byte自動検出。
  - family別文字・writer統計、全family合算OOV、前回OOV 21字種の補完状況を追加。
  - 9G writer keyをファイル＋sheet run順から4,000件として算出。
  - 依頼文と異なるbyte位置は、[AIST公式M-type仕様](https://etlcdb.db.aist.go.jp/etlcdb/etln/form_m.htm)および実データに従い、Serial Sheet Number=bytes 5–6、JIS X 0201=byte 7を採用。

- `pipeline/glyph_bank.py`（新規）
  - 9G/ETL1/ETL6の決定論的インデックス構築。
  - SHA-256による疑似筆者グリフ選択。
  - 背景除去、0基準化、crop、コントラスト正規化。
  - 遅延offset read、ファイルハンドル・画像LRUキャッシュ。
  - 9Bは除外。

- `pipeline/m3_render.py`
  - `glyph_source="font"|"etl"` を追加。既定は従来どおりfont。
  - 疑似筆者ID、ETL描画、font fallback、インク色合成、ink-mask bboxを実装。
  - `glyph_source`、`writer_consistent`、`pseudo_writer_id`、`glyph_fallback_rate` を出力。

- `pipeline/test_m3_render_fix.py`
  - ETL双子bbox一致、PNG決定性、`÷` fallback、監査OOV整合の4テストを追加。
  - データ不在時はSKIP表示。

### 実行結果

監査結果:

- OOV: **2,848 / 28,214 = 10.0943%**
- 前回21字種のうちETL1/6で補完: **17字種**
- 残余OOV: **`x`, `×`, `、`, `。`**
- 更新先: `out/etl_audit_report.json`

グリフインデックス:

- `data/etl/glyph_index.json`
- 構築時間: **7.180秒**
- サイズ: **85,588,705 bytes（81.62 MiB）**
- 登録: **906,181グリフ、3,151文字**
- writer keys: 9G=4,000、ETL1=1,445、ETL6=1,383

テスト:

```text
PASS: 8 tests; SKIP: 0 tests
```

`py_compile` と `git diff --check` も成功しています。