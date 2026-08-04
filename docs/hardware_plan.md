# ハードウェア計画:EVO-X2 単機・予算ゼロ運用パス

**方針**: 支出をほぼゼロ(電気代+任意で無料クラウド枠)に抑えたまま、設計書 v2.1 の
Phase 0 を完結させ、Phase 1 を縮小構成で前進させる。H100 級を要する工程
(GRPO/RLVR・8B の高速反復・32B・V4-Flash 判定)は**予算確保後の条件付き工程として凍結**し、
それまでに Go/No-Go の判定材料を揃える。

この順序は敵対的パネルの確定所見——「安価な代替(rejection sampling+SFT/DPO)を先に
実測し、GRPO はそれに勝つ見込みが立ってから投資する」「大きな支出は Phase 0 の実測
合格後に」——とそのまま整合する。予算制約は計画の歪みではなく、パネルが要求した
意思決定順序の強制執行になっている。

## 機材

GMKtec EVO-X2 / AMD Ryzen AI Max+ 395(Strix Halo)/ 128GB LPDDR5X /
Radeon 8060S iGPU(40CU, RDNA3.5)

- 実効メモリ帯域 ~215 GB/s。LLM 推論は帯域律速のため、**アクティブ 3B の MoE
  (Qwen3.6-35B-A3B)が最適合**(30B 級 MoE で ~100 tok/s の実測報告)
- 8B dense 推論 ~48 tok/s、70B dense ~5 tok/s(参考)
- 128GB ユニファイドメモリ → Q8 の 35B MoE+長コンテキストに余裕。8B BF16 推論可

## セットアップ手順

1. **OS**: Ubuntu 24.04 LTS を推奨。iGPU への割当メモリを 96〜110GB に設定
   (BIOS の UMA 設定+カーネルパラメータ `amdttm.pages_limit`/GTT)
2. **推論ランタイム**: llama.cpp(**Vulkan/RADV 既定で十分**。gfx1151 では token
   生成は Vulkan が ROCm と同等以上。長文プロンプト処理を上げたい場合のみ ROCm ビルド)
3. **モデル取得**(いずれも llama.cpp 用 GGUF):
   - 標準: `ggml-org/Qwen3.6-35B-A3B-GGUF` の Q8_0(~37GB)
   - 高速: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`(MTP で 1.5〜2 倍速)
4. **サーバ起動**(OpenAI 互換 API):

   ```bash
   llama-server -m Qwen3.6-35B-A3B-Q8_0.gguf \
     --host 0.0.0.0 --port 8080 -c 8192 -np 4 --jinja
   ```

5. **接続確認**(パイプライン側はコード変更不要。`verbalizer.LLMVerbalizer` は
   OpenAI 互換 API を前提に実装済み):

   ```bash
   VLLM_BASE_URL=http://<EVO-X2>:8080/v1 python pipeline/llm_smoke_test.py
   ```

   スモークテストが通ったら、同じ URL で `gate_efficacy_test` 系のハーネスを
   実 LLM 出力に対して回す(テンプレートでは常に 100% だった G1/G2 通過率が、
   ここで初めて実測の品質指標になる)。

## 何がどこまで行けるか(EVO-X2 単機)

| 工程 | 可否 | 所要目安 | 備考 |
|---|---|---|---|
| LLM 逐語化・概念誤りオペレータの開発 | ◎ | 即日〜 | Qwen3.6-35B-A3B Q8。構造化出力は llama-server の JSON schema 強制+既存 G2 で受ける |
| Phase 0 の 1 万サンプル生成(実 LLM) | ◎ | 1〜3 日 | ~100 tok/s 前提。MTP 版で短縮可 |
| G1/G2 の実測(通過率+ゲート精度) | ◎ | 生成と同時 | Phase 0 完了条件の中核 |
| ゼロショット・ベースライン(Qwen3-VL 2B/4B/8B、Sarashina2.2-Vision) | ◎ | 数時間〜1 日/モデル | GGUF 推論。**KPI 仮目標を実測で確定**(設計書 §7 の前提) |
| レンダリング(M3・30 万頁) | ◎ | CPU で並列 | GPU 不要 |
| SFT データ量産 | △ | 3〜5 万件/週 | 常時稼働で蓄積。Phase 1' は 3 万件から開始 |
| 2B(→4B)LoRA SFT | △ | 数日/epoch | 実験枠: ROCm/TheRock の gfx1151 PyTorch。**動かなければ即 Kaggle 無料枠へ切替**(T4×2・週 30h、QLoRA で 2B は ~1.5 週分の枠で 1 run) |
| RAFT(rejection sampling+再SFT)/DPO | △ | 推論は高速・訓練は上記経路 | **パネルの「安価な代替」比較を GRPO 抜きで先に実施**(SFT vs RAFT vs DPO) |
| GRPO(RLVR)・8B 高速反復・32B | ✕ | — | CUDA+H100 待ち。**凍結**(予算確保後の条件付き工程) |
| V4-Flash 自前ホスト判定 | ✕ | — | Q4 でも ~160GB。判定は教員+決定論指標で進行(元々非ゲート設計) |
| 教員 r 実測・収集 1,350 枚 | GPU 不要 | — | 謝金予算(数十万〜¥100 万)が必要になった時点で別途判断 |

## 縮小マイルストーン(予算ゼロ版)

- **Phase 0'(機材到着〜+1 ヶ月)**: スモークテスト → 実 LLM でゲート実測 →
  概念誤りオペレータ → 1 万サンプル → ゼロショット・ベースライン → **KPI 確定**
- **Phase 1'(+2〜3 ヶ月)**: SFT データ 3 万件 → 2B LoRA SFT(EVO-X2 or Kaggle)→
  RAFT/DPO を同一チェックポイントから比較 → 合成 held-out で SFT vs RAFT vs DPO を
  報告(**GRPO 列は「予算確保後」として空欄のまま提示**する——空欄自体が投資判断の材料)
- **成果主張の上限**: 「2B 縮小構成での方式比較と改善傾向」まで。8B 本番・sim2real
  定量(収集 1,350 枚)・32B は予算確保後。誇張しない(パネル条件:主張スコープの限定)

## 注意・リスク

- **ROCm gfx1151 の PyTorch 訓練は成熟途上**。半日試して動かなければ深追いせず
  Kaggle/Colab 無料枠(CUDA・安定)へ切り替える(ここにこだわるのが最大の時間リスク)
- 常時稼働の消費電力 ~140〜200W(電気代 月 2〜3 千円程度)
- llama-server の JSON 強制は文法ベース——スキーマ逸脱は既存の G2/構造検査が棄却する
- 外部 API は使わない前提(GPT-5.6 評価も凍結。もともと非ゲートの補助であり計画に影響しない)
