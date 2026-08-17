# TITAN V機 Qwen3-VL-2B QLoRA 実行手順

この手順は TITAN V機 上で `train/train_qlora.py` を単機・`nohup` 実行するためのものです。スクリプトは Kaggle の `kaggle/qwen3vl_2b_sft.ipynb` と同じ学習条件および評価指標で、LoRA 付与前の base model と学習後 adapter を比較します。

## 1. 前提

- Ubuntu 20.04
- NVIDIA TITAN V 12 GB（Volta、sm_70）
- NVIDIA ドライバ 570 系
- miniconda3 導入済み

TITAN V は BF16 および FlashAttention の対象外です。スクリプトは FP16 と SDPA に固定し、Unsloth / TRL / Triton を使用しません。

## 2. 環境構築

```bash
conda create -n ja python=3.11 -y && conda activate ja
conda install -y pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install "transformers>=4.57,<5" peft bitsandbytes accelerate pillow
```

sm_70 は PyTorch 2.5 系 + CUDA 12.1 までを安全側の構成として使います。CUDA 12.6 / 12.8 系 wheel や PyTorch 2.7 以降は Volta サポート外となる恐れがあるため使用しません。また、Transformers 5.x は PyTorch の要求バージョンが上がるため `<5` に固定します。

PyTorch は pip ではなく conda で導入します。pip の `torch==2.5.1` は依存 `nvidia-cudnn-cu12==9.1.0.70` を固定要求しますが、この版は index から削除済みで解決不能です(2026-08 に TITAN V機 で実測)。conda の pytorch チャネルは 2.5.1 が最終リリースで、cudnn を conda パッケージとして同梱するためこの問題を回避できます。導入後、GPU と compute capability を確認できます。

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))  # (7, 0)
PY
```

GPU を使わない指標テストも先に実行できます。

```bash
python train/train_qlora.py --self-test
```

## 3. データ配置

Task G の出力を次の形で配置します。

```text
~/ja-grading/data/sft/
├── train.jsonl
├── val.jsonl
├── test.jsonl
└── images/
    ├── <sample_id>.png
    └── ...
```

各 JSONL は 1 行 1 JSON object で、`image` はデータディレクトリからの相対パスです。学習データは shuffle せず `train.jsonl` の先頭から使用します。

## 4. スモークテスト

リポジトリのルート（例: `~/ja-grading`）で実行します。`--smoke` は実効値を `train-limit=8`、`eval-n=4`、`max-new-tokens=128` に上書きします。

```bash
cd ~/ja-grading
python train/train_qlora.py \
  --data-dir ~/ja-grading/data/sft \
  --out-dir ~/ja-grading/out/TITAN V機-smoke \
  --smoke
```

## 5. 本番実行と監視

既定値は train 3,000 件、1 epoch、LoRA `r=16` / `alpha=16`、学習率 `2e-4`、batch 1、gradient accumulation 8、評価 200 件です。

```bash
mkdir -p ~/ja-grading/out/TITAN V機
cd ~/ja-grading
nohup python train/train_qlora.py \
  --data-dir ~/ja-grading/data/sft \
  --out-dir ~/ja-grading/out/TITAN V機 \
  > ~/ja-grading/out/train.log 2>&1 &
echo $!
```

ログと GPU は別端末から監視します。

```bash
tail -f ~/ja-grading/out/train.log
watch -n 2 nvidia-smi
```

目安は `TRAIN_LIMIT=3000` で訓練 2～4 時間、before / after 評価を合わせて約 1 時間です。画像内容や生成長により増減します。

## 6. 成果物

- `~/ja-grading/out/TITAN V機/lora_adapters/`: LoRA adapter と processor
- `~/ja-grading/out/TITAN V機/checkpoints/`: Trainer checkpoint（最大 2 世代）
- `~/ja-grading/out/TITAN V機/eval_results.json`: before / after の行別出力、集約指標、実効設定、所要時間

## 7. トラブルシュート

### CUDA OOM

評価時は batch OOM を検知すると自動で batch 1 に切り替えます。それでも不足する場合は画像上限を下げ、評価 batch を明示的に 1 にします。`--grad-accum 8` は維持してください。

```bash
python train/train_qlora.py \
  --data-dir ~/ja-grading/data/sft \
  --out-dir ~/ja-grading/out/TITAN V機 \
  --max-pixels $((640*905)) --eval-batch 1 --grad-accum 8
```

### bitsandbytes が sm_70 で失敗する

4-bit 量子化を外して FP16 LoRA に退避します。FP16 model は VRAM 使用量が増えるため、必要なら `--max-pixels` も下げます。

```bash
python train/train_qlora.py \
  --data-dir ~/ja-grading/data/sft \
  --out-dir ~/ja-grading/out/TITAN V機-fp16 \
  --no-quant --max-pixels $((640*905)) --eval-batch 1
```

### FP16 の loss がスパイクする

学習率を `1e-4` に下げます。

```bash
python train/train_qlora.py \
  --data-dir ~/ja-grading/data/sft \
  --out-dir ~/ja-grading/out/TITAN V機-lr1e4 \
  --lr 1e-4
```

### checkpoint から再開する

同じ `--out-dir` の `checkpoints/checkpoint-*` が存在する状態で `--resume` を指定します。checkpoint がなければ新規学習として開始します。既存の before 評価を保持して再開時間を短縮する場合は `--skip-before-eval` も付けます。

```bash
nohup python train/train_qlora.py \
  --data-dir ~/ja-grading/data/sft \
  --out-dir ~/ja-grading/out/TITAN V機 \
  --resume --skip-before-eval \
  > ~/ja-grading/out/train.log 2>&1 &
```
