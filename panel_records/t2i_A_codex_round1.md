## 中心的反対

**「Qwen-Image-Edit を通した最終画像でも label-by-construction・文字 bbox GT・双子同一性を維持できる」という中心主張に反対する。** Edit は入力グリフを条件にするだけで、文字列・局所幾何・非変更領域を保存する拘束条件ではない。最終データは **label-by-construction ではなく label-by-OCR-screening** になる。Stage A を実験として行うことには反対しないが、現行 Go/No-Go 条件のまま bbox 教師付き主データへ採用するのは不可。

- **事実**: Qwen-Image-Edit は意味条件と VAE 条件を使う確率的生成器であり、文字ごとの恒等写像や既知の座標写像を出力しない。[公式モデルカード](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)にも、日本語の全字形保存や bbox 保存保証はない。
- **推論**: OCR 合格は「OCR が GT と同じ文字列を返した」ことしか保証せず、実際の字形、局所 bbox、非変更文字の同一性は保証しない。OCR が文脈で異体・崩れを補完する場合もある。
- **確信度: 99%**
- **反証条件**: 500パッチ、計1万文字以上を漢字画数・かな・数字・演算子・句読点で層化し、変換後を人手で文字単位に再 bbox 化する。文字一致率99.9%以上、全誤りスパンの99%以上で補正後 bbox IoU≥0.9、非変更文字の99%以上で位置ずれ≤2 pxを同時達成すること。Kaggle上限15 GPU時間、外部費用0円。安価な代理は100パッチ・2,000文字、3 GPU時間以内だが、これだけでは1%以下を確証できない。

## 座標計算の不整合

### 1. 「約1MP以内・16の倍数ならリサイズ恒等」は誤り

- **事実**: 現行 Diffusers の `QwenImageEditPlusPipeline` は既定で入力縦横比から面積 `1024×1024` の寸法を計算し、まず32の倍数へ丸める。さらに出力寸法は VAE/patch 制約の倍数へ切り下げ、条件画像もリサイズする。[実装](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py)  
  したがって「1MP**以内**」ではなく、少なくとも既定経路では「縦横比に応じた約1MPへの拡大・縮小」である。16の倍数だけでも足りず、寸法計算部分は32単位で丸める。
- **推論**: ComfyUI、Diffusers、musubi で前処理が完全に同じとは限らない。musubi は2511について「公式1Mサイズへのリサイズ」を必須としている。[musubi文書](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/qwen_image.md)
- **確信度: 98%**
- **反証条件**: 使用予定の固定済み ComfyUI workflow に、`641×257`、`1024×1024`、`2048×512`、`1536×640` の座標格子画像を各10枚入力し、各ノード直前直後の実テンソル寸法・crop・pad・出力寸法をログする。全経路で拡縮率1、crop/pad 0ならその寸法についてのみ恒等と認める。50実行未満、費用0円、1時間以内。

### 2. `ページ座標 = offset + 元bbox` は diffusion 後には成立しない

正しくは

\[
B_{\text{page}}=O+T(B_{\text{source}})
\]

であり、`O+B` は変換 \(T\) が恒等の場合だけ成立する。Diffusion は局所的なストローク移動・拡幅・欠落・追加を起こし、\(T\) は通常、単一の平行移動でもアフィン変換でもない。

- **事実**: 位相相関で推定できるのは基本的にパッチ全体の平行移動であり、文字ごとの非剛体変形は補正できない。
- **推論**: 行パッチへの分割は最大移動可能範囲を狭めるだけで、パッチ内 bbox の正しさを構造的には保証しない。
- **確信度: 99%**
- **反証条件**: 200パッチについて変換後の全文字を人手または独立した文字セグメンタで再 bbox 化し、(a)補正なし、(b)位相相関、(c)局所 optical flow の3方式を比較する。位相相関だけで文字中心残差の99パーセンタイル≤2 px、bbox IoUの1パーセンタイル≥0.9なら反対を撤回する。費用0円、手作業4～8時間。代理は50パッチ・500文字。

### 3. 「数pxは IoU 的に許容」は bbox サイズ依存であり、一般には偽

24×20 px の box を縦横4 pxずつずらすと IoU はちょうど0.5、5 pxなら約0.42になる。10×10 px の小記号では縦横3 pxずれで約0.325である。よって「数px」は小さい演算子・小数点・符号では許容範囲ではない。

- **確信度: 100%**
- **反証条件**: 現行10,000件の全 `error_span_boxes_px` について、1～8 pxの水平・垂直・対角シフトを再計算し、採用予定のずれ量でも95%以上が IoU≥0.5を保つこと。CPUのみ、費用0円、数分。安価な代理は全 box サイズ分布からの解析計算で十分。

### 4. 現行 `m3_render.py` 自体も「ピクセル厳密 bbox」ではない

- `char_boxes` は内部変数であり、返されるのはステップ bbox と誤りスパン bboxだけで、全文字 bbox は出力されていない（`pipeline/m3_render.py:68-115,135-`）。
- bbox はフォントメトリクスから作り、最後に `GaussianBlur(0.4)` をかけているため、可視インクは bbox 外へ広がり得る（同`:83-87,122`）。
- `_union` の右端・下端も `int()` で切り捨てており、外接矩形なら `floor(min), ceil(max)` が必要（同`:49-51`）。

したがって「FTと無関係な既存の座標帳簿へオフセットを足すだけ」ではなく、まず bbox 定義と出力スキーマの修正が必要である。

- **確信度: 99%**
- **反証条件**: 100ページを二値化して実インク外接矩形とメタデータ bbox を比較し、全可視インク包含率100%かつ余白規則が仕様化されていること。費用0円、CPU数分。

## ペア生成原則

### 5. 同一 seed・同一 workflow は双子の同一性を保証しない

同じノイズでも条件画像が異なれば denoising 軌道が異なる。誤り文字以外の筆圧、字形、背景まで変わり得る。これは乱数共有であって、出力スタイルの拘束ではない。

現行レンダラも完全ではない。`m3_render.py:74-88` は文字ごとに乱数を順次消費するため、文字数、空白、改行、glyph advance が変わる変異では、変異点以後の jitter と折返し位置が双子間でずれる。`style_from_pair` が共有するのはフォント・サイズ等であり、画像同一性ではない。

- **事実**: 同 seed は同じ初期乱数を与える。
- **推論**: 同じ出力スタイル・非変更領域を与える、という主張は成立しない。
- **確信度: 99%**
- **反証条件**: 置換・挿入・削除・空白変更を各50組、計200双子生成し、位置合わせ後の「変異マスク外」差分面積率≤1%、SSIM≥0.99、書き手埋め込み距離が異seed対照より有意に小さいことを要求する。Kaggle 10 GPU時間以内、0円。代理は各型10組、計40組。

## モデル仕様・実装対応

### 6. Qwen-Image-Edit-2511 の日付は誤り、その他は一部確認

- **誤り**: 公開日は設計書の2025-12-26ではなく、公式リポジトリでは **2025-12-23**。[Qwen公式履歴](https://github.com/QwenLM/Qwen-Image)
- **確認**: 20B、Apache-2.0、公式 safetensors、ComfyUIネイティブworkflowは確認できる。[モデルカード](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)、[ComfyUI公式手順](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511)
- **条件付き**: GGUF は存在するがコミュニティ量子化であり、Q4は約11.5～13GB。公式BF16/FP8サポートと「GGUFが公式・ネイティブ」を混同している。[2511 GGUF例](https://huggingface.co/vantagewithai/Qwen-Image-Edit-2511-GGUF/tree/main)
- **確認**: Lightning 4-step LoRA の「40 step比で約10倍」は配布元の主張と一致する。ただし、文字忠実度が同等という意味ではない。[Lightningモデルカード](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)
- **確認**: Qwen-Image-2512 は2025-12-31公開。Qwen-Image-2.0 は2026-02-10発表だが、2026-08-17時点で公式ウェイト公開は確認できない。[Qwen公式履歴](https://github.com/QwenLM/Qwen-Image)
- **確信度: 99%**
- **反証条件**: 公式Qwenリポジトリまたは公式HFの公開履歴が12月26日を初回公開日として示せば日付判定を撤回する。サンプル不要、費用0円。

### 7. 日本語実証として引用された2記事は証拠能力が不足

- Zenn記事は **Edit-2509** の少数例であり、2511、答案ページ、CER、複雑度別漢字、bbox、双子再現性を評価していない。[記事](https://zenn.dev/kota_iizuka/articles/33219ebb8aff99)
- Hatena記事は冒頭で Qwen-Image は日本語を直接書けないと述べ、Text Rendererを介した少数の成功例を示すもの。縦書き入力対応ノードの存在は確認できるが、縦書き文字の網羅的忠実度の実証ではない。[記事](https://nowokay.hatenablog.com/entry/2026/01/05/110344)
- 2511が「選択された人気LoRAを統合」は公式記述どおりだが、その例は照明・視点・設計などであり、**日本語手書きLoRAの統合や答案忠実度向上は未確認**。
- **確信度: 98%**
- **反証条件**: 2511・4-stepで最低500パッチ、1万文字の direct T2I 対 glyph-conditioned edit の対比較を行い、層化CER、文字欠落率、複雑漢字の stroke/component 誤り、bboxずれを公開する。Kaggle 15 GPU時間以内、0円。代理100パッチでは傾向確認のみ。

### 8. Kaggle T4 + Q4 を「現実解」とするには実測が欠ける

Q4本体約12～13GBだけでは VRAM見積りにならない。別途 Qwen2.5-VL-7B text encoder、VAE、LoRA、activation、latent が必要で、16GB T4ではCPU/RAM offloadが前提になる。さらにComfyUI公式workflowはBF16本体を案内しており、GGUF利用には通常追加ノードが必要である。

一方、EVO-X2を「分オーダーで事実上不可」と扱うのも未検証である。AMDは2026年時点で Ryzen AI Max 系について Windows/LinuxのPyTorch、学習・推論対応を掲げている。[ROCm対応表](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility.html) 128GB統合メモリの容量面では、少なくとも即座に除外できない。

- **確信度: 90%**
- **反証条件**: 同じ50パッチ・4-step・同じ量子化で、EVO-X2 ROCm、Kaggle T4、可能ならT4×2を実測し、初回ロード時間、秒/枚、ピークVRAM/RAM、失敗率を記録する。各環境3時間上限、費用0円。50枚を完走できなければ「数十～数百枚PoCに十分」は反証されたと扱う。

## LoRA 学習

### 9. ツール対応は部分的に正しいが、DiffSynth のCLI表記が誤り

- musubi の `--model_version edit-2511`、control画像、FP8、block swap、Windows対応は文書に存在する。ただし同文書は2511学習を未検証扱いしており、動作保証ではない。[musubi文書](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/qwen_image.md)
- DiffSynth-Studio の2511 LoRA例は存在するが、`--zero_cond_t` は値を取らない `store_true` フラグである。設計書の **`--zero_cond_t 1` はそのままでは誤ったCLI**。[公式学習スクリプト](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_training/train.py)、[2511実行例](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_training/lora/Qwen-Image-Edit-2511.sh)
- **確信度: 99%**
- **反証条件**: 現行mainの `train.py --help` が `--zero_cond_t 1` を受理すること、または公式例が値付き構文へ更新されること。データ不要、費用0円。

### 10. 「24GB VRAM+64GB RAMが可能ライン」「T4/EVO不可」は過度に断定的

musubi は低VRAM向けに FP8、gradient checkpointing、block swap、CPU offloadを提供しているため、16GB未満でも**容量上は**動く可能性がある。逆に24GBなら必ず実用速度という保証もない。RunPodの現行Secure Cloud 4090は24GB VRAMだが、掲載RAMは41GBで、設計書の64GB条件を満たさない。[RunPod料金表](https://www.runpod.io/pricing)

「2000 step=2～3時間」も解像度・swap数・OS・optimizerで大幅に変わる。公開実測には1024²・1500 stepで約2時間53分～5時間19分の例があり、固定値にはできない。[実測例](https://pc.watch.impress.co.jp/docs/column/nishikawa/2062248.html)

- **確信度: 95%**
- **反証条件**:
  - 無料代理: T4とEVO-X2で64ペア・100 stepを実行し、OOM、step時間、RAMを測る。各3時間、0円。
  - 本検証: 4090で対象解像度・rank16・2511・1000 stepを1回。ピークVRAM≤23GB、RAM≤契約値、外挿2000 step≤3時間を満たすこと。クラウド費用上限3,000円。上限内で完走不能なら時間・費用主張を棄却。

### 11. 「500～3,000ペアで足りる」は未検証のまま設計前提になっている

スタイルLoRA一般の相場を、文字内容保存・局所座標保存を要求する制御付き編集へ移植する根拠がない。また、出力教師がETL孤立字の貼合せなら、LoRAが学ぶのも主に孤立字貼合せの分布であり、教師に存在しない連綿・文脈依存・自然な字間を生成する根拠はない。

- **事実**: 同一テキストの機械生成ペアは大量に作れる。
- **推論**: それは独立した実筆分布が無限に得られることを意味しない。字形は各字最大約200例に強く再利用される。
- **確信度: 98%**
- **反証条件**: 500/1,000/3,000ペアの学習曲線を同一300パッチheld-outで比較し、CER、局所bbox、実答案との書き手埋め込み距離、人手自然度を測る。3,000まで単調改善し、500で既に最終値の95%以上なら見込みを支持する。3 run合計クラウド費用上限3,000円は厳しいため、安価な代理として512px・各500 stepで傾向を確認し、本検証は予算確保後とすべき。

## ETL の内容・ライセンス

### 12. ETL規約改定日は正しいが、「商用含め無条件無料」は過大解釈

公式規約には2025-03-28改定と無料利用が明記され、旧目的制限は現行文面にない。一方で、著作権はAISTに残り、データ配布経路・再公開は禁止され、参照条件もある。[AIST公式規約](https://etlcdb.db.aist.go.jp/download2/)

現行文面は、次を明示していない。

- 商用モデル学習が「使用」に含まれるか
- ETL由来の合成画像を顧客や社外へ配布できるか
- ETLを強く記憶したLoRAが「データ自体の加工再配布」に当たるか
- クラウドGPUへのアップロードが許されるか

したがって「商用含め無条件」「禁止はデータ自体の再配布のみ」は未確認。設計書自身がAIST照会を未決事項にしており、決定事項との内部整合もない。

- **確信度: 90%**
- **反証条件**: AISTから、企業の商用製品開発、外部クラウド処理、LoRA、合成画像の社外提供を個別に許容する書面回答を取得する。サンプル不要、照会費0円。その回答が得られればこの法務上の反対は撤回する。

### 13. ETL9の文字数は正しいが、筆者ID・文字集合・B版選択に未解決点がある

- **確認**: ETL9は2,965漢字+71ひらがな=3,036クラス、607,200標本、各クラス200標本である。[AIST ETL9仕様](https://etlcdb.db.aist.go.jp/etlcdb/etln/etl9/etl9.htm)
- **欠落**: ETL9だけでは数字、英字、カタカナ、主要演算子・数学記号を網羅しない。答案レンダラの全面置換素材にはならない。
- **未確認**: 現行AIST概要は筆記者を延べ4,000人と記す一方、ファイル構造は200系列として扱われる。レコードには `Serial Sheet Number` はあるが、設計書が仮定する「全3,036字を同一人物が書いた200個の writer_id」対応は、生データで照合されていない。[ETL9G形式](https://etlcdb.db.aist.go.jp/etlcdb/etln/form_e9g.htm)
- **設計上不利**: ETL9Bは64×63の二値、ETL9Gは128×127の4bit階調である。高解像度答案の主素材にBを選ぶ合理性が示されておらず、拡大時の階段状輪郭を増やす。[AIST概要](https://etlcdb.db.aist.go.jp/the-etl-character-database/)
- **確信度: 96%**
- **反証条件**: 公式全データを走査し、(a)同一writerキーごとに3,036クラスが1つずつ揃う200組、(b)性別・年齢・職業等のmetadata整合、(c)予定10,000答案のETL9 OOV率を算出する。CPUのみ、0円、全607,200レコード。OOV率0%かつ200完全writer群なら筆者共有主張を支持する。併せて9B/9Gを各100ページ盲検比較する。

### 14. Stage B' の label/bbox は構成的に保存可能だが、「完全保存」は実装条件付き

ETLグリフを直接貼るだけなら、ラベルと配置セルは既知であり、この部分は diffusion より堅い。ただし以下が必要である。

- glyph crop・拡縮・回転後の alpha mask から bbox を再計算
- 欠字のフォントfallbackを provenance に記録
- 同一writerが存在しない文字集合を混ぜた場合は `writer_consistent=false`
- 64×63二値を高解像度へ拡大した際の補間方式固定
- 現行の長さ依存乱数消費を文字IDベース乱数へ変更

「CPUなので30万頁と両立」も throughput 実測がない。

- **確信度: 95%**
- **反証条件**: 1,000ページを本番解像度・全metadata出力込みでレンダリングし、ピークRAM、頁/秒、欠字率、bbox包含率を測る。30万頁への線形外挿が許容時間内で、欠字率0または仕様化されたfallbackのみなら成立。EVO-X2、0円、通常1時間未満。

## その他のデータセット・規約

### 15. TUAT Kondate の価格・製品開発権は確認できるが、クラウド学習と衝突する

一般価格50万円、製品開発とデータを直接含まない成果物の販売権は公式条件どおり。一方、データのコピーを機関敷地外へ持ち出せない条件がある。[TUAT公式利用条件](https://web.tuat.ac.jp/~nakagawa/database/jp/kondate_proc.html)

したがって、Kondate由来画像をRunPodへアップロードする上位版LoRAは、通常の契約文面とは衝突する可能性が高い。「社外配布だけ要交渉」では不足し、**外部クラウド処理自体を契約時に確認**すべきである。

- **確信度: 99%**
- **反証条件**: TUATから、指定クラウド事業者への暗号化アップロードと学習処理を許す書面を取得する。費用は照会0円、ライセンス購入50万円。このプロジェクトの数千円規模では実行不能なので、当面の実行計画から外すべき。

### 16. ライセンス表には過度な一括化がある

- MathWriting は原則CC BY-NC-SA 4.0、CROHME 2014/2023もNC系であり、商用製品を目的とする企業R&Dでの不使用判断は妥当。[MathWriting公式README](https://github.com/google-research/google-research/blob/master/mathwriting/archive_readme.md)、[CROHME-2014](https://tc11.cvc.uab.es/datasets/CROHME-2014_2)
- CASIA-HWDBの無償版は学術研究限定・製品開発禁止だが、商用利用は有償問い合わせの余地があるため、表の「商用出口×」は「無償契約では×」が正確。[CASIA公式申請書](https://nlpr.ia.ac.cn/databases/download/CASIA-HWDB-Chinese.pdf)
- 「Kuzushiji/NDL=CC BY-SA」はデータセットを特定しておらず不正確。KMNIST/Kuzushiji-KanjiはCC BY-SA 4.0だが、東京大学史料編纂所の2023くずし字データセットはCC BY 4.0、NDLの一部OCR学習データはPublic Domainである。[KMNIST](https://github.com/rois-codh/kmnist)、[史料編纂所データ](https://lab.hi.u-tokyo.ac.jp/datasets/kuzushiji)、[NDL](https://lab.ndl.go.jp/data_set/ocr_en/r3_text/)
- **確信度: 98%**
- **反証条件**: データセット名・版・URL・取得日・ライセンスファイルSHAを固定したSBOMを作る。データ不要、費用0円、半日。名称を一括したままでは検証不能。

## 図形・検品

### 17. matplotlib xkcd の決定性は確認できるが、「座標GT完全保存」は別問題

ローカルのMatplotlib 3.11.1で別Pythonプロセスから同一図を2回生成したところ、PNGのSHA-256は一致した。決定性という主張はこの環境では再現した。しかし xkcd sketch は表示線を意図座標から揺らすため、ピクセルbboxや交点位置を元の直線座標からそのまま流用できない。描画後path/maskから再計算すべきである。

また「図形は意味レベルなので文字ほど忠実性リスクがない」は誤り。グラフの傾き・交点・目盛、幾何の平行・直角、筆算の桁位置は局所幾何そのものがラベルである。

- **確信度: 97%**
- **反証条件**: グラフ100、幾何100、筆算100の計300図で、傾き符号、交点誤差、角度誤差、桁alignmentを生成前後比較する。Qwen版が意味ラベル100%、座標許容99%以上なら第二候補を認める。Kaggle 10 GPU時間以内、0円。代理は各20図。

### 18. 検品ループの文献根拠が誤って束ねられている

- 2026 StyleText はOCRベースの意味フィルタを使うため根拠になる。[StyleText](https://arxiv.org/abs/2605.17309)
- SynthOCR-Gen はUnicode・フォント・劣化処理による合成器であり、diffusion出力をOCRで棄却する方式の根拠ではない。[SynthOCR-Gen](https://arxiv.org/abs/2601.16113)
- JSSODaもLLM生成日本語をフォント描画した単純合成OCRデータで、Qwen Edit後のbbox回収や rejection loop の前例ではない。[JSSODa](https://huggingface.co/datasets/llm-jp/JSSODa)

したがって「3研究で標準化したパターン」は誤り。検品自体は必要だが、引用のうち直接支えるのは主にStyleTextである。

さらにQwen3-VLだけを判定器にすると、Qwen系表現・言語事前分布に依存した相関誤りが残る。OCR合格サンプルだけ残すと、乱雑な筆跡が選択的に除去され、目的の分布から遠ざかる。

- **確信度: 98%**
- **反証条件**: Qwen3-VL判定後の採択400件・棄却100件を、独立OCR一種と人手で再評価する。false accept≤0.1%、false reject≤5%、画数・乱雑度別の棄却率差≤5ポイントなら単独判定を支持する。外部費用0円、手作業5～10時間。代理は採択100・棄却50。

## 外部API規約の記述

「競合モデル条項でグレー」はサービス種別を分ける必要がある。

- OpenAI個人向け規約と現行Services Agreementには、競合モデル開発へのOutput利用制限が明記されているため、該当するなら「グレー」より明確に禁止側である。[OpenAI Terms of Use](https://openai.com/policies/terms-of-use/)、[Services Agreement](https://openai.com/en-GB/policies/services-agreement/)
- Googleの一般向け生成AI追加規約は、生成物を機械学習モデル開発に使うこと自体を禁止している。[Google Generative AI Terms](https://policies.google.com/terms/generative-ai?hl=en-US)
- ただし企業契約・サービス固有契約では条件が異なり得る。「両社規約」と一括せず、実際に利用する商品・契約版を固定すべきである。
- 「日本語文字品質で最高水準」は比較ベンチマークがなく未検証。
- **確信度: 95%**
- **反証条件**: 対象API名、契約種別、規約版日を固定し、提供者から本VLM学習への利用許可を取得する。費用0円の問い合わせで足りる。もっとも外部API不使用方針だけで不採用判断は成立する。

## 採択に必要な修正

1. Qwen Edit後のデータを `label_by_construction=false`、`label_validation=ocr+human/audit` として別系列に隔離する。
2. bbox教師付き主系列は Stage B' と決定的撮像層に限定し、Qwen系列は bboxなしの転記・スタイル補助、または厳格再アノテーション済み分だけにする。
3. `offset+bbox`、位相相関、同seedによる「保存」「完全共有」という表現を削除する。
4. Stage A の合格条件に、CERだけでなく文字単位人手監査、局所bbox、非変更領域、双子差分、棄却選択バイアスを加える。
5. ETL9Gを主候補として9Bと比較し、数学記号・数字・カタカナのfallbackとwriter整合性を明示する。
6. DiffSynthの引数を `--zero_cond_t` に訂正し、EVO-X2/T4/4090の100-step smoke test後にだけ資源見積りを確定する。
7. AISTとTUATについて、生成画像・LoRA・クラウド処理を明記した書面許諾が得られるまで商用可と確定しない。

## 自分の議論の最も弱い点

最も弱いのは **ETL規約からLoRA・合成画像・クラウド処理まで未許諾と読む部分**である。現行文面の「無料で使用できる」は広く、AISTが意図として商用学習と派生成果物を全面許可している可能性はある。AISTから「企業の商用モデル学習、外部クラウド処理、学習済みLoRA、生成画像の社外提供を許可する」と明記した回答が得られれば、この法務上の反対は崩れる。ただし、その回答が得られても diffusion 後の label・bbox・双子同一性に関する技術的反対は崩れない。