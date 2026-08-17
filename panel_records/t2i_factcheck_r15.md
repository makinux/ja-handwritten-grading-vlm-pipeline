# ファシリテータによる事実確認(Round 1.5)— 両パネリストへの中立情報

以下はファシリテータが一次資料・実コードに当たって独立検証した結果である。意見ではなく事実確認として扱うこと。

1. **m3_render.py のコード監査(パネリストA #4・#5 のレンダラ部分)**: 原文照合で**確認**。
   - 戻り値は `boxes_px`(ステップbbox)と `error_span_boxes_px` のみで、全文字bboxは出力されていない
   - bbox は `font.getbbox()`(フォントメトリクス)由来で、最後に `GaussianBlur(0.4)` が全面に掛かる
   - `_union` は右端・下端も `int()` 切り捨て
   - 乱数は文字ごとに逐次消費(非空白文字は dx,dy の2回+全文字で字送り1回)のため、**変異点以降は現行レンダラでも双子のジッタ列・折返しがずれる**。空白⇔非空白の置換では消費数自体が変わる

2. **ETL 利用規約原文**(etlcdb.db.aist.go.jp/download2/ を直接取得): 「Use of Database is allowed for free」(第3条)で**目的制限の文言なし**(研究/商用の区別なし)。第5条「Distribution of Database should only be made through this web site. Unauthorized distribution and publication of the data itself beyond the scope of quotation and/or direct URL links to data files are prohibited.」。**学習済みモデル・派生物・機械学習への言及は規約全体に存在しない**。改定日 2025-03-28(第8条)。
   → 設計書の「商用含め無条件無料」は目的制限については正確。ただし派生物(LoRA・合成画像)の扱いは「明示許可」ではなく「未規定」であり、パネリストAの「未確認」指摘とパネリストBの「照会を今日出すべき」は文言上の裏付けがある。

3. **DiffSynth-Studio train.py**(GitHub main の原文取得): `parser.add_argument("--zero_cond_t", default=False, action="store_true", help="A special parameter introduced by Qwen-Image-Edit-2511. Please enable it for this model.")` — **store_true フラグであり、設計書の `--zero_cond_t 1` は誤記(Aの指摘 #9 は正しい)**。

4. **diffusers `QwenImageEditPlusPipeline`**(GitHub main の原文取得): `calculate_dimensions(target_area=1024*1024, ratio)` で縦横比保存のまま面積1024²へ、`round(w/32)*32` で**32の倍数**へ丸め。**リサイズをスキップする分岐は存在しない**。したがって ~1MP「以内」の入力は**拡大**される。恒等化は「計算式の不動点となる寸法(面積≒1024²かつ32倍数)」を入力すること自体では達成可能だが、設計書の「16の倍数・~1MP以内」という記述は誤り(Aの指摘 #1 は正しい)。ComfyUI 側の前処理経路は未検証。
