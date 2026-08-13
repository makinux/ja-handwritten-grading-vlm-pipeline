# Kaggle T4 で Qwen3-VL-2B を QLoRA 学習する

`qwen3vl_2b_sft.ipynb` は、日本語手書き答案の採点 SFT データを使い、Kaggle の無料 GPU 枠で `unsloth/Qwen3-VL-2B-Instruct` を 4-bit QLoRA 学習してビフォー／アフター評価するノートブックです。

## 1. データセットを Kaggle に登録する

手元のバンドル zip に、展開後の最上位が次の構成になることを確認します。

```text
train.jsonl
val.jsonl
test.jsonl
dataset_card.md
images/
  {sample_id}.png
  ...
```

各 JSONL は UTF-8 の 1 行 1 JSON で、`image` の値（例: `images/abc.png`）から同じバンドル内の画像を参照できる必要があります。

1. Kaggle にサインインし、**Datasets** → **New Dataset** を開きます。
2. バンドル zip をアップロードします。Kaggle 側で展開されない場合は、ローカルで zip を展開し、上記のファイルと `images/` をまとめてアップロードします。
3. Dataset title を **ja-handwriting-grading-sft** にして作成します。
4. Dataset のファイル一覧で、`train.jsonl` などが余分な親ディレクトリや zip の内側ではなくデータセット直下に見えることを確認します。

この名前ならノートブックの既定パス `/kaggle/input/ja-handwriting-grading-sft` を変更せずに使えます。別名にした場合は、設定セルの `DATASET_DIR` を Kaggle が表示する実際のマウント先に変更してください。

## 2. ノートブックを実行する

1. Kaggle の **Code** → **New Notebook** で新規ノートブックを作ります。
2. **File** → **Import Notebook**（または Upload）から `qwen3vl_2b_sft.ipynb` をアップロードします。
3. 右側パネルの **Add Input** / **Add Data** で、先ほど作成した `ja-handwriting-grading-sft` Dataset を追加します。
4. **Notebook options** の **Accelerator** を **GPU T4 x2** に設定します。Internet が無効なら、Unsloth とモデルを取得できるよう **Internet on** にします。
5. 最初の設定セルを確認します。既定は `TRAIN_LIMIT=3000`、`EPOCHS=1` です。`TRAIN_LIMIT=None` にすると `train.jsonl` 全件を使います。
6. **Run All** を実行します。インストール、モデル読込、学習、ベースモデル評価、学習済み LoRA 評価、結果保存の順に進みます。

`TRAIN_LIMIT` はシャッフル前の `train.jsonl` 先頭 N 件に適用されます。画像は学習バッチごとに遅延ロードされるため、全画像が RAM に常駐することはありません。Kaggle の表示が T4 x2 でも、通常のノートブック実行では Unsloth/Trainer が利用する GPU 構成に従います。

## 3. 所要時間の目安

- `TRAIN_LIMIT=3000`: 学習 約 2〜4 時間、ビフォー／アフター評価 約 1〜2 時間
- 合計は Kaggle の週 30 時間枠内を想定していますが、混雑、生成長、Kaggle イメージ、GPU の割当状況で変動します。
- 全件（9,204 件）の学習はこの初回確認ではなく、次段の実験で行うことを推奨します。

## 4. 成果物を回収する

実行完了後、右側の **Output** または `/kaggle/working` のファイル一覧から次をダウンロードします。

- `/kaggle/working/lora_adapters/` — LoRA adapter と processor 設定
- `/kaggle/working/eval_results.json` — ベース／学習後の集約指標、評価対象 ID、parse failure 詳細

必要なら **Save Version** を実行して出力を永続化してからダウンロードしてください。セッション終了後は `/kaggle/working` の未保存ファイルが失われることがあります。

## 5. トラブルシュート

### CUDA out of memory (OOM)

設定セルの `MAX_PIXELS` を小さくします（例: `640*896`、さらに必要なら `512*768`）。`BATCH=1` は維持し、実効バッチや所要時間との兼ね合いで `GRAD_ACCUM` も調整してください。`GRAD_ACCUM` 自体は 1 回の forward の画像メモリを大きくは減らさないため、OOM には `MAX_PIXELS` を下げるのが第一選択です。まだ不足する場合は `MAX_SEQ` も下げます。

### セッションが切れる、時間内に終わらない

`TRAIN_LIMIT` を 3000 より小さくして再実行します。評価時間が問題なら `EVAL_N` も下げられます。途中で切れた学習は自動再開されないため、完走可能な件数から始めてください。

### Dataset が見つからない

右側パネルで Dataset が追加済みか、`/kaggle/input/ja-handwriting-grading-sft/train.jsonl` が存在するかを確認します。Dataset 名を変えた場合は `DATASET_DIR` も変更します。

### インストールまたはモデル取得に失敗する

Internet を on にしてセッションを再起動し、先頭から Run All します。Kaggle のベースイメージ更新直後は依存関係が変わることがあるため、インストールセルのエラーを解消する前に後続セルだけを実行しないでください。
