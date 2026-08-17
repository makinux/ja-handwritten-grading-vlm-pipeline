# 日本語手書き数学答案の誤り診断 VLM — label-by-construction 合成データパイプライン

日本語の手書き数学答案(中学 1 年:正負の数・一次方程式)の**誤り位置・誤り種別・採点**を行う視覚言語モデル(VLM)を、実答案の注釈なしで訓練するための合成データパイプラインと訓練・評価ハーネスです。解法を実行可能なプログラムとして生成し、誤りをプログラム変異として注入し、決定論的レンダリングにより全文字の座標を既知にします——**ラベルは注釈されるのではなく、構成から導かれます**。

論文(言語処理学会年次大会 予稿・準備中):**「構成によるラベルは観測から識別不能でありうる:手書き風合成答案による数学誤り診断 VLM の学習」** 和山亮介・Fable 5(株式会社ノーザンシステムサービス)。原稿とレビュー記録は [docs/paper/](docs/paper/) を参照。

## 主要な結果

- 9,204 件の合成データによる QLoRA SFT(Qwen3-VL-2B)で、ゼロショットでは全モデル 0.000 の**誤り位置特定を IoU@0.5 0.795** へ、幻覚率を 31.4%→**0%**、点数完全一致を 34.5%→**93.5%** へ(配備形態 GGUF 8bit+llama.cpp で測定)
- **ラベル識別可能性の失敗モード**を生成器全件走査で定量化:概念的誤解ラベルの 95.6% が機械的誤記ラベルと答案表面で文字列一致(観測同値)。生成経路(provenance)を予測ターゲットにしてはならない
- 修正は**再訓練不要**:推論時のラベル写像+過剰主張ペナルティ(識別不能な事例への細粒度断定=不正解)で種別一致 0.970・過剰主張 0%

## リポジトリ構成

```
pipeline/          生成(M1解法→M2誤り注入→逐語化→M3レンダ)・ゲート・報酬・評価
  gen_core.py            生成器と G1/G2 ゲート
  verbalizer.py          合成方式逐語化(LLM は説明句のみ・式は機械合成)
  llm_mutation.py        LLM 概念オペレータ(AST ホワイトリスト検証)
  relabel_identifiable.py 識別可能粒度への再ラベル(観測同値スキャン)
  run_generation.py / run_render.py  生成・レンダリング実行
  run_zeroshot_eval.py   評価ハーネス(座標系/システム/タキソノミ/後処理を切替可)
  rewards.py             報酬(三値化: 一致/粗すぎ部分点/過剰主張0)+敵対的テスト
  build_sft_dataset.py   SFT データセット構築(messages 形式・相対座標)
train/             QLoRA 訓練スクリプト(単機 GPU 用・前後評価内蔵)
kaggle/            Kaggle T4 用ノートブック(同一ハイパラの代替経路)
docs/              設計書・各フェーズ報告・論文原稿(docs/paper/)
panel_records/     敵対的パネルの全記録(設計・タキソノミ・原稿レビュー)
```

## クイックスタート

```bash
# 環境(Docker)
docker build -t ja-grading .
docker run --rm -v "$(pwd)":/work -w /work ja-grading python pipeline/run_phase0_bootstrap.py

# 生成(要: OpenAI 互換 API の LLM サーバ。例: llama.cpp + Qwen3.6-35B)
VLLM_BASE_URL=http://127.0.0.1:8080/v1 VLLM_MODEL=<alias> \
  python pipeline/run_generation.py --n 1000 --chunk 200 --seed 7 --out out/gen
python pipeline/run_render.py --in out/gen --out out/gen/render

# 識別可能粒度への再ラベル
python pipeline/relabel_identifiable.py --records out/gen --out-map out/map.jsonl --out-stats out/stats.json

# SFT データ構築 → 訓練(単機 GPU)
python pipeline/build_sft_dataset.py --records out/gen/render/records --images out/gen/render/images \
  --out out/sft --relabel-map out/map.jsonl
python train/train_qlora.py --data-dir out/sft --out-dir out/run1  # 詳細: train/README_titanv.md
```

依存: Python 3.11+/生成・評価は標準ライブラリ+Pillow、訓練は torch/transformers/peft/bitsandbytes(バージョンは train/README_titanv.md 参照)。日本語フォント(IPAex/Noto CJK)が必要です(Dockerfile が導入)。

## 再現性

- 報酬関数は訓練前に敵対的テスト(不正出力 1,000 件・逆転 0/1000)を通過
- 評価ハーネスの既定は常に後方互換(変更時は byte 一致検証)
- 3 シードの一次成果物: [docs/paper/evidence/](docs/paper/evidence/)
- 全設計判断は敵対的パネル記録([panel_records/](panel_records/))に監査証跡つきで残存

## ライセンス

MIT License(© 2026 Northern System Service Co., Ltd.)。[LICENSE](LICENSE) を参照。
