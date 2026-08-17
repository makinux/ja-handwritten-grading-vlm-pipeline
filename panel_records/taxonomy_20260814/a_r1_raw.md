2026-08-14T02:07:44.544137Z ERROR codex_models_manager::cache: failed to load models cache: missing field `supports_reasoning_summaries` at line 86 column 5
OpenAI Codex v0.144.1
--------
workdir: D:\home\日本語手書きテストデータセット作成パイプライン
model: gpt-5.6-sol
provider: openai
approval: never
sandbox: read-only
reasoning effort: high
reasoning summaries: none
session id: 019ffe06-7a73-7152-b1a6-cfd2482dc13e
--------
user
# 設計判断ブリーフ: 誤り種別タキソノミの評価粒度(パネリスト A 用)

あなたは設計レビューパネルの一員。以下は自己完結の判断材料である。あなたの最終メッセージが討論記録としてそのまま保存される。前置き・挨拶・要約の反復は不要。日本語で書くこと。

## システム背景(要点)

日本語手書き数学答案の採点 VLM を、label-by-construction 合成データ(誤り注入+決定論レンダリング)で訓練するプロジェクト。誤りは「オペレータ」で注入され、各オペレータに種別ラベルと深刻度が紐付く。現行 9 クラス:

- 概念/演算順序、概念/移項(深刻度3・LLM概念オペレータ)、概念/符号・乗算(深刻度3・LLM)、表記/符号落とし、計算/加減、計算/加減乗除、計算/符号(深刻度2・機械変異)、計算/除算、記法/移項符号(深刻度2・機械変異)
- 「概念/移項」=「移項しても符号は変わらない」等の**誤解**。LLM が誤解に基づく説明句を生成し、それが答案の文面(逐語化)に反映される。ただし**誤り式そのもののスパンは「記法/移項符号」(単純な符号書き間違い)とほぼ同一の見た目**。判別信号は説明句の言い回しに薄く存在する。
- モデル出力契約: `{transcript, errors:[{step_id,bbox,type}], score}`。深刻度は出力しない。点数 GT はルーブリック項目充足から計算され、**種別ラベルに依存しない**(=評価タキソノミ変更は点数系に波及しない)。
- 現行データは **1 答案 1 誤りが主流**(複数誤り・同種誤りの反復は未導入)。「概念的誤解=同種の誤りを一貫して繰り返す」という人間的な判別手がかりが構造的に存在しない。

## 実測(2026-08-13/14、Qwen3-VL-2B QLoRA SFT 後)

- HF fp16 ハーネス(test 先頭 200 件): 種別マクロ 0.760。内訳: 概念/移項 6/13(残 7 は記法/移項符号へ)、記法/移項符号 13/22(残 9 は概念/移項へ)→ **移項ファミリとしては 35/35=100%**。計算/符号 0/3(全て概念/符号・乗算へ)。その他クラスは 0.83〜1.0。
- llama.cpp 配備ハーネス(seed77・n=200): 種別マクロ 0.647〜0.656。**混同の方向がランタイム間で逆転**(このハーネスでは概念/移項 0.94・記法/移項符号 0.00)——下位区分の判別が本質的に不安定である証拠。
- 位置特定(IoU@0.5 0.795〜0.948)・検出(BACC 0.99)・点数(完全一致 0.935)は高水準。**種別のみ KPI 未達**。
- 転記はほぼ完全(CER 0.000〜0.007)= 説明句の文字は読めているのに、種別判別に活用できていない。
- 参考: 詳細は `docs/phase1_2b_sft_report.md`(追記 1〜3)、オペレータ定義は `pipeline/gen_core.py`(必要なら読んでよい)。

## 設計書の現行規定

- KPI(合成系・内部回帰指標): 種別 macro 一致 ≥ 0.85
- RL 用報酬表: 「誤り種別 | 分類一致(**上位階層一致に部分報酬**)」= 階層扱いが既に予定されている
- 対外的に主張可能な性能は人手参照系のみ(合成系 KPI は開発内部指標)

## 教育的文脈

概念的誤解(再教育が必要)と記法・計算ミス(注意喚起で足りる)の区別は、採点フィードバックの質に直結し、製品価値の一部。ただし**人間の教員がこの区別を答案表面からどの程度判別できるかは未測定**(教員間一致 r の実測は外部依存で未実施)。

## 実行可能性の制約(反証条件はこのスケールで実行可能であること)

- 評価再実行(n=200): 約 20 分・無料(手元サーバ)
- データ再生成(1 万件): 約 7 時間+再訓練 約 2.5 時間 = 数日内で反復可能
- 人手ラベルの追加調達は不可(計画済みの収集 1,350 頁のみ)。教員実測は待ち状態。
- タキソノミ定義・評価スクリプトの改定: 実装コスト小

## 検討対象の選択肢(攻撃対象として評価せよ。正しいと仮定するな)

(a) KPI をファミリ粒度へ改定(下位区分は参考指標へ降格)
(b) 生成側で判別可能性を強制(概念系に誤解の語りを必須化・同種誤りの反復による文脈設計)
(c) 現行維持(sim2real 後に再判定)
(d) 階層 KPI(ファミリ ≥0.85 必須+下位区分は分解報告のみ)
(e) 下位区分の操作的再定義(答案から観測可能な定義に書き直す)

組合せ・新案も可。**少なくとも 1 つの選択肢を明確に否定すること。**

## 成果物要件

- 推奨する設計判断(単一の明確な決定。段階条件があれば明記)
- 主要主張ごとに: 事実/推測の区別、確信度(%)、**具体的で安価に実行可能な反証条件**(何を・何件・どれだけの手間で測れば崩れるかを明記。汎用的な「反証する証拠が出たら」は不可)
- **自説の最弱点**を明示し、何が観測されたら崩れるか

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'rg -n "概念/移項|記法/移項符号|計算/符号|概念/符号・乗算|error_type|severity" pipeline/gen_core.py docs/phase1_2b_sft_report.md' in D:\home\日本語手書きテストデータセット作成パイプライン
 succeeded in 324ms:
docs/phase1_2b_sft_report.md:46:4. **残る弱点は隣接クラス混同の 2 種のみ。**「概念/移項」13 件中 12 件を「記法/移項符号」
docs/phase1_2b_sft_report.md:47:   と誤分類、「計算/符号」3 件全てを「概念/符号・乗算」と誤分類。いずれも誤りスパンの
docs/phase1_2b_sft_report.md:68:- **種別の残課題は「移項ファミリ内の下位区分」に収斂した。**「概念/移項」は 1/13→6/13 と
docs/phase1_2b_sft_report.md:69:  改善する一方、「記法/移項符号」が 20/22→13/22 と逆方向の混同が発生(双方向化)。
docs/phase1_2b_sft_report.md:71:  いずれか)。「計算/符号」(n=3)も全て符号系隣接クラス「概念/符号・乗算」への混同。
docs/phase1_2b_sft_report.md:74:  統合した場合のマクロ ≈0.85、符号系(計算/符号+概念/符号・乗算)も統合すれば ≈0.97。
docs/phase1_2b_sft_report.md:111:- 種別の混同は本ハーネスでも移項ファミリ内に収斂(方向は逆転: 概念/移項 0.94、
docs/phase1_2b_sft_report.md:112:  記法/移項符号 0.00)。下位区分の不安定性がランタイムを跨いで再現し、
pipeline/gen_core.py:62:def mut_spec(op_id, site, err_type, severity, payload=None):
pipeline/gen_core.py:64:            "severity": severity, "payload": payload or {}}
pipeline/gen_core.py:150:        mut_spec("op-transpose-sign", "s2", "記法/移項符号", 2),
pipeline/gen_core.py:241:        muts.append(mut_spec("op-mult-sign-drop", "s2", "計算/符号", 2))
pipeline/gen_core.py:263:            "op-llm-transpose-misconception", "s2", "概念/移項", 3,
pipeline/gen_core.py:278:            "op-llm-product-misconception", "s2", "概念/符号・乗算", 3,
pipeline/gen_core.py:400:            "severity": mut["severity"],
pipeline/gen_core.py:543:            "type": err["type"], "severity": err["severity"],

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$f='"'pipeline/gen_core.py'; "'$lines=Get-Content -LiteralPath $f -Encoding UTF8; foreach($range in @(@(125,165),@(220,292))){ for($i=$range[0];$i -le $range[1];$i++){ '"'{0,4}: {1}' -f "'$i,$lines[$i-1] }; '"'' }" in D:\home\日本語手書きテストデータセット作成パイプライン
 succeeded in 410ms:
 125: 
 126:     # s3 右辺の計算
 127:     if m == "op-arith-slip" and site == "s3":
 128:         d = d + pay["delta"]
 129:     tk = Tok()
 130:     steps.append(step("s3", "simplify", f"{tk.n(a)}x = {tk.n(d)}", tk.nums))
 131: 
 132:     # s4 除算
 133:     ans = Fraction(d, a)
 134:     if m == "op-divide-error" and site == "s4":
 135:         if pay["kind"] == "mul":
 136:             ans = Fraction(d * a)
 137:         else:
 138:             ans = Fraction(d, a) + pay["delta"]
 139:     if m == "op-final-sign-drop" and site == "s4":
 140:         ans = abs(ans)
 141:     tk = Tok()
 142:     steps.append(step("s4", "divide", f"x = {tk.n(ans)}", tk.nums))
 143:     return steps, ans
 144: 
 145: 
 146: def linear_mutations(rng, params):
 147:     a, b, c = params["a"], params["b"], params["c"]
 148:     gold_ans = Fraction(c - b, a)
 149:     muts = [
 150:         mut_spec("op-transpose-sign", "s2", "記法/移項符号", 2),
 151:         mut_spec("op-arith-slip", "s3", "計算/加減", 1,
 152:                  {"delta": rng.choice([-2, -1, 1, 2])}),
 153:         mut_spec("op-divide-error", "s4", "計算/除算", 2,
 154:                  {"kind": rng.choice(["mul", "off"]),
 155:                   "delta": rng.choice([-1, 1])}),
 156:     ]
 157:     if gold_ans < 0:
 158:         muts.append(mut_spec("op-final-sign-drop", "s4", "表記/符号落とし", 1))
 159:     return muts
 160: 
 161: 
 162: # --------------------------------------------------------- integer arithmetic
 163: # 正負の数: p + q × r(演算の優先順位)
 164: 
 165: 

 220:                           f"{tk.n(p)} + {paren(tk.n(mv))} = {tk.n(ans)}", tk.nums))
 221: 
 222:     if m == "op-final-sign-drop" and site == "s4":
 223:         ans = abs(ans)
 224:     tk = Tok()
 225:     steps.append(step("s4", "answer", f"答え {tk.n(ans)}", tk.nums))
 226:     return steps, Fraction(ans)
 227: 
 228: 
 229: def m2_applies(mut, site):
 230:     return mut and mut.get("site") == site
 231: 
 232: 
 233: def arith_mutations(rng, params):
 234:     p, q, r = params["p"], params["q"], params["r"]
 235:     gold = p + q * r
 236:     muts = [
 237:         mut_spec("op-arith-slip", rng.choice(["s2", "s3"]), "計算/加減乗除", 1,
 238:                  {"delta": rng.choice([-2, -1, 1, 2])}),
 239:     ]
 240:     if q * r < 0:
 241:         muts.append(mut_spec("op-mult-sign-drop", "s2", "計算/符号", 2))
 242:     if r != 1 and q != 0:
 243:         muts.append(mut_spec("op-order-of-ops", "s2", "概念/演算順序", 3))
 244:     if gold < 0:
 245:         muts.append(mut_spec("op-final-sign-drop", "s4", "表記/符号落とし", 1))
 246:     return muts
 247: 
 248: 
 249: def _llm_mut_from_proposal(problem, proposal):
 250:     """LLM 提案を検証し、再実行可能な mutation spec へ変換する。"""
 251:     if not isinstance(proposal, dict):
 252:         raise ValueError("LLM proposal must be a dictionary")
 253:     params = problem["params"]
 254:     if problem["domain"] == "一次方程式":
 255:         b, c = (params[key] for key in ("b", "c"))
 256:         value, tokens, canonical = safe_parse_expr(proposal.get("expr"))
 257:         allowed = {abs(b), abs(c)}
 258:         if not set(abs(token) for token in tokens) <= allowed:
 259:             raise ValueError("linear proposal contains a disallowed constant")
 260:         if value == Fraction(c - b):
 261:             raise ValueError("linear proposal equals the gold transpose value")
 262:         return mut_spec(
 263:             "op-llm-transpose-misconception", "s2", "概念/移項", 3,
 264:             {"expr_canonical": canonical, "expr_tokens": tokens,
 265:              "d_value": str(value)})
 266: 
 267:     if problem["domain"] == "正負の数":
 268:         q, r = (params[key] for key in ("q", "r"))
 269:         value = proposal.get("value")
 270:         if type(value) is not int:
 271:             raise ValueError("arithmetic proposal value must be an integer")
 272:         gold = q * r
 273:         if abs(value) > abs(gold) + 30:
 274:             raise ValueError("arithmetic proposal value is out of range")
 275:         if value == gold:
 276:             raise ValueError("arithmetic proposal equals the gold product")
 277:         return mut_spec(
 278:             "op-llm-product-misconception", "s2", "概念/符号・乗算", 3,
 279:             {"mv_value": value})
 280: 
 281:     raise ValueError(f"unsupported LLM mutation domain: {problem['domain']}")
 282: 
 283: 
 284: def _propose_llm_mutation(llm_mutator, problem):
 285:     """領域別の提案 API を呼び、検証済み mutation spec を返す。"""
 286:     params = problem["params"]
 287:     if problem["domain"] == "一次方程式":
 288:         proposal = llm_mutator.propose_linear_transpose(
 289:             problem, params["a"], params["b"], params["c"])
 290:     elif problem["domain"] == "正負の数":
 291:         proposal = llm_mutator.propose_arith_product(
 292:             problem, params["q"], params["r"])


exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'rg -n -C 4 "explanation|説明|misconception|誤解|narr|語り" pipeline docs/phase1_2b_sft_report.md' in D:\home\日本語手書きテストデータセット作成パイプライン
 succeeded in 284ms:
docs/phase1_2b_sft_report.md-44-3. **採点が完全一致 1.00。** ルーブリック条件付き採点(配点サンプリング)が形式ごと
docs/phase1_2b_sft_report.md-45-   学習された。ゼロショットの不安定さ(モデル規模と単調でない)は訓練で標準化された。
docs/phase1_2b_sft_report.md-46-4. **残る弱点は隣接クラス混同の 2 種のみ。**「概念/移項」13 件中 12 件を「記法/移項符号」
docs/phase1_2b_sft_report.md-47-   と誤分類、「計算/符号」3 件全てを「概念/符号・乗算」と誤分類。いずれも誤りスパンの
docs/phase1_2b_sft_report.md:48:   見た目がほぼ同一で、説明句(逐語化の誤解の語り)の読解が必要な区別。対策候補:
docs/phase1_2b_sft_report.md-49-   フルデータ(9,204)訓練での該当クラス増量/クラス定義の再検討(統合 or 判別特徴の明示)。
docs/phase1_2b_sft_report.md-50-5. **TITAN V 単機で 1 サイクル 65 分。** 2B 縮小構成の反復実験は TITAN V機 で十分高速
docs/phase1_2b_sft_report.md-51-   (Kaggle 想定 4-6h → 実測 1h)。フルデータでも ~2.5-3h/サイクルの見込み。
docs/phase1_2b_sft_report.md-52-
--
docs/phase1_2b_sft_report.md-68-- **種別の残課題は「移項ファミリ内の下位区分」に収斂した。**「概念/移項」は 1/13→6/13 と
docs/phase1_2b_sft_report.md-69-  改善する一方、「記法/移項符号」が 20/22→13/22 と逆方向の混同が発生(双方向化)。
docs/phase1_2b_sft_report.md-70-  ただし**ファミリレベルでは 35/35=100%**(gt が移項系の全サンプルで、予測も移項系の
docs/phase1_2b_sft_report.md-71-  いずれか)。「計算/符号」(n=3)も全て符号系隣接クラス「概念/符号・乗算」への混同。
docs/phase1_2b_sft_report.md:72:  → モデルは誤りの所在とファミリを完全に特定しており、下位区分(概念的誤解 vs 記法ミス)が
docs/phase1_2b_sft_report.md-73-  **答案表面から観測可能か**というタキソノミ設計の問題に帰着する。参考値: 移項 2 クラスを
docs/phase1_2b_sft_report.md-74-  統合した場合のマクロ ≈0.85、符号系(計算/符号+概念/符号・乗算)も統合すれば ≈0.97。
docs/phase1_2b_sft_report.md-75-- 対応候補: (a) 評価タキソノミをファミリ粒度へ改定(観測可能性基準)、(b) 逐語化の
docs/phase1_2b_sft_report.md:76:  説明句に下位区分の判別特徴を強制(生成側の対処)、(c) 現行維持で sim2real 後に再判定。
docs/phase1_2b_sft_report.md-77-  **設計判断事項として次回レビューに付す。**
docs/phase1_2b_sft_report.md-78-
docs/phase1_2b_sft_report.md-79-成果物: TITAN V機 `~/ja-grading/out/TITAN V機-full/{lora_adapters,eval_results.json}`、
docs/phase1_2b_sft_report.md-80-ローカル `out/TITAN V機_sft/eval_results_full.json`。
--
pipeline\build_sft_dataset.py-465-        "",
pipeline\build_sft_dataset.py-466-        "## 入力契約",
pipeline\build_sft_dataset.py-467-        "",
pipeline\build_sft_dataset.py-468-        "ユーザーテキストは `run_zeroshot_eval.py` の C1 `build_prompt` を再利用し、"
pipeline\build_sft_dataset.py:469:        "SFT 用に相対座標規約だけを追記している。assistant は説明や Markdown を含まない JSON 文字列である。",
pipeline\build_sft_dataset.py-470-        "",
pipeline\build_sft_dataset.py-471-    ])
pipeline\build_sft_dataset.py-472-    return "\n".join(lines)
pipeline\build_sft_dataset.py-473-
--
pipeline\gen_core.py-104-    steps.append(step("s1", "setup",
pipeline\gen_core.py-105-                      f"{tk.n(a)}x {sb} {tk.n(abs(b))} = {tk.n(c)}", tk.nums))
pipeline\gen_core.py-106-
pipeline\gen_core.py-107-    # s2 移項: 正しくは c から b を引く(表示は abs(b) と演算子)
pipeline\gen_core.py:108:    if m == "op-llm-transpose-misconception" and site == "s2":
pipeline\gen_core.py-109-        tk = Tok()
pipeline\gen_core.py-110-        a_text = tk.n(a)
pipeline\gen_core.py-111-        for token in pay["expr_tokens"]:
pipeline\gen_core.py-112-            tk.n(token)
--
pipeline\gen_core.py-201-                          f"{paren(tk.n(t))} × {paren(tk.n(r))} = {tk.n(ans)}",
pipeline\gen_core.py-202-                          tk.nums))
pipeline\gen_core.py-203-    else:
pipeline\gen_core.py-204-        mv = q * r
pipeline\gen_core.py:205:        if m == "op-llm-product-misconception" and site == "s2":
pipeline\gen_core.py-206-            mv = pay["mv_value"]
pipeline\gen_core.py-207-        if m == "op-mult-sign-drop" and site == "s2":
pipeline\gen_core.py-208-            mv = abs(mv)
pipeline\gen_core.py-209-        if m == "op-arith-slip" and site == "s2":
--
pipeline\gen_core.py-259-            raise ValueError("linear proposal contains a disallowed constant")
pipeline\gen_core.py-260-        if value == Fraction(c - b):
pipeline\gen_core.py-261-            raise ValueError("linear proposal equals the gold transpose value")
pipeline\gen_core.py-262-        return mut_spec(
pipeline\gen_core.py:263:            "op-llm-transpose-misconception", "s2", "概念/移項", 3,
pipeline\gen_core.py-264-            {"expr_canonical": canonical, "expr_tokens": tokens,
pipeline\gen_core.py-265-             "d_value": str(value)})
pipeline\gen_core.py-266-
pipeline\gen_core.py-267-    if problem["domain"] == "正負の数":
--
pipeline\gen_core.py-274-            raise ValueError("arithmetic proposal value is out of range")
pipeline\gen_core.py-275-        if value == gold:
pipeline\gen_core.py-276-            raise ValueError("arithmetic proposal equals the gold product")
pipeline\gen_core.py-277-        return mut_spec(
pipeline\gen_core.py:278:            "op-llm-product-misconception", "s2", "概念/符号・乗算", 3,
pipeline\gen_core.py-279-            {"mv_value": value})
pipeline\gen_core.py-280-
pipeline\gen_core.py-281-    raise ValueError(f"unsupported LLM mutation domain: {problem['domain']}")
pipeline\gen_core.py-282-
--
pipeline\llm_mutation.py-122-        prompt = (
pipeline\llm_mutation.py-123-            f"一次方程式 {a}x {sign} {abs(b)} = {c} の移項で、中学生が"
pipeline\llm_mutation.py-124-            "やりがちな誤りを 1 つ提案してください。誤った移項後の右辺の式だけを"
pipeline\llm_mutation.py-125-            "次の JSON で返してください: "
pipeline\llm_mutation.py:126:            '{"expr": "<右辺の式(例: 20 + 8)>", "reason": "<どんな誤解か一言>"}。'
pipeline\llm_mutation.py-127-            f"使ってよい数は {c} と {abs(b)}（符号は変えてよい）だけです。"
pipeline\llm_mutation.py-128-            f"正しい式 {c} - {b} は禁止です。$ や LaTeX 記法は使わないでください。"
pipeline\llm_mutation.py-129-        )
pipeline\llm_mutation.py-130-        return self._chat_json(prompt)
pipeline\llm_mutation.py-131-
pipeline\llm_mutation.py-132-    def propose_arith_product(self, problem, q, r):
pipeline\llm_mutation.py-133-        prompt = (
pipeline\llm_mutation.py:134:            f"{q} × {r} の計算で、符号や九九の誤解による誤った結果を 1 つ"
pipeline\llm_mutation.py-135-            "提案してください。次の JSON だけを返してください: "
pipeline\llm_mutation.py:136:            '{"value": <整数>, "reason": "<どんな誤解か一言>"}。'
pipeline\llm_mutation.py-137-            f"正解 {q * r} は禁止です。"
pipeline\llm_mutation.py-138-        )
pipeline\llm_mutation.py-139-        return self._chat_json(prompt)
pipeline\llm_mutation.py-140-
--
pipeline\verbalizer.py-5-限定される。本モジュールはその差し替え点を定義する。
pipeline\verbalizer.py-6-
pipeline\verbalizer.py-7-- TemplateVerbalizer : 現行のテンプレート逐語化(gen_core が生成した text を
pipeline\verbalizer.py-8-  そのまま用いる)。ブートストラップの既定。
pipeline\verbalizer.py:9:- LLMVerbalizer      : 説明句を vLLM の OpenAI 互換 API(Qwen3.6 系)で生成し、
pipeline\verbalizer.py-10-  gen_core が生成した式を後置して逐語化するクライアント。
pipeline\verbalizer.py-11-  環境変数 VLLM_BASE_URL(例 http://gpu-host:8000/v1)と VLLM_MODEL を要する。
pipeline\verbalizer.py-12-  ※ GPU サーバ未接続のため実機未検証。接続後は必ず g1b_check_texts の
pipeline\verbalizer.py-13-  ゲートを通してから採用すること(原則 2: LLM 出力は検証ゲートを通過する
--
pipeline\verbalizer.py-26-    """vLLM 呼び出しに使う temperature を環境変数から返す。"""
pipeline\verbalizer.py-27-    return float(os.environ.get("VLLM_TEMPERATURE", "0.3"))
pipeline\verbalizer.py-28-
pipeline\verbalizer.py-29-
pipeline\verbalizer.py:30:def _compose_step_text(explanation, template):
pipeline\verbalizer.py:31:    """LLM の説明句に、未包含の場合だけ正解の式テンプレートを後置する。"""
pipeline\verbalizer.py:32:    explanation = explanation.strip()
pipeline\verbalizer.py:33:    if explanation.endswith("。"):
pipeline\verbalizer.py:34:        explanation = explanation[:-1]
pipeline\verbalizer.py:35:    if normalize_text(template) in normalize_text(explanation):
pipeline\verbalizer.py:36:        return explanation
pipeline\verbalizer.py:37:    return explanation + template
pipeline\verbalizer.py-38-
pipeline\verbalizer.py-39-
pipeline\verbalizer.py-40-class TemplateVerbalizer:
pipeline\verbalizer.py-41-    """テンプレート逐語化(恒等変換)。"""
--
pipeline\verbalizer.py-62-
pipeline\verbalizer.py-63-class LLMVerbalizer:
pipeline\verbalizer.py-64-    """vLLM(OpenAI 互換)クライアント。
pipeline\verbalizer.py-65-
pipeline\verbalizer.py:66:    プロンプトは各式の直前に置く短い説明句だけを生成させ、正しい式テンプレートを
pipeline\verbalizer.py-67-    プログラム側で後置する。出力は g1b_check_texts で検証する。
pipeline\verbalizer.py-68-    thinking は既定で無効化し、VLLM_ENABLE_THINKING=1 の場合のみサーバ既定に戻す。
pipeline\verbalizer.py-69-
pipeline\verbalizer.py-70-    環境変数: VLLM_BASE_URL / VLLM_MODEL / VLLM_MAX_TOKENS /
--
pipeline\verbalizer.py-104-        if content is None or content == "":
pipeline\verbalizer.py-105-            raise ValueError(
pipeline\verbalizer.py-106-                "LLM 応答の content が空(thinking による文脈枯渇の可能性。"
pipeline\verbalizer.py-107-                "VLLM_MAX_TOKENS/サーバの -c を確認)")
pipeline\verbalizer.py:108:        explanations = json.loads(content)["steps"]
pipeline\verbalizer.py:109:        if len(explanations) != len(steps):
pipeline\verbalizer.py-110-            raise ValueError("LLM 逐語化のステップ数が不一致")
pipeline\verbalizer.py-111-        return [
pipeline\verbalizer.py:112:            _compose_step_text(explanation, step["text"])
pipeline\verbalizer.py:113:            for explanation, step in zip(explanations, steps)
pipeline\verbalizer.py-114-        ]
pipeline\verbalizer.py-115-
pipeline\verbalizer.py-116-    @staticmethod
pipeline\verbalizer.py-117-    def _build_prompt(problem, steps):
pipeline\verbalizer.py-118-        lines = [
pipeline\verbalizer.py:119:            "次の数学の解答ステップについて、各ステップの式の直前に置く短い説明句を書いてください。",
pipeline\verbalizer.py-120-            "各ステップの式はこちらで後置するので、式そのものは書かないでください。",
pipeline\verbalizer.py-121-            "「次に掛け算を計算すると、」「両辺から 6 を引いて、」のように、",
pipeline\verbalizer.py:122:            "式に自然につながる形（読点や「と」など）で終わる説明句にしてください。",
pipeline\verbalizer.py-123-            "数値に言及する場合は、与えたステップに現れる数値のみを使ってください。",
pipeline\verbalizer.py-124-            "答えのステップは「したがって、」のような結びの句でかまいません。",
pipeline\verbalizer.py:125:            "出力は JSON {\"steps\": [各ステップの説明句]} のみ。",
pipeline\verbalizer.py-126-            f"問題: {problem['problem_text']}",
pipeline\verbalizer.py-127-            "ステップ:",
pipeline\verbalizer.py-128-        ]
pipeline\verbalizer.py-129-        for s in steps:
--
pipeline\verbalizer.py-173-    return out, i
pipeline\verbalizer.py-174-
pipeline\verbalizer.py-175-
pipeline\verbalizer.py-176-def inject_verbose_faithful(texts, rng):
pipeline\verbalizer.py:177:    """既知数値を説明で再言及する良性の逐語化: G1 は通すべき。"""
pipeline\verbalizer.py-178-    out = list(texts)
pipeline\verbalizer.py-179-    known = [token for text in texts for token in NUM_RE.findall(text)]
pipeline\verbalizer.py-180-    token = rng.choice(known)
pipeline\verbalizer.py-181-    if rng.choice([False, True]):
--
pipeline\run_zeroshot_eval.py-23-画像に実際に書かれている内容を忠実に転記し、模範解答で答案を上書きしないでください。
pipeline\run_zeroshot_eval.py-24-問題文、step_id 付き模範解答、採点基準と答案画像を照合してください。
pipeline\run_zeroshot_eval.py-25-誤りごとに、対応する模範解答の step_id、答案画像上の誤り箇所の bbox、誤り種別を返してください。
pipeline\run_zeroshot_eval.py-26-bbox はページ左上を原点とするピクセル座標 [x0,y0,x1,y1] です。
pipeline\run_zeroshot_eval.py:27:誤りがなければ errors は空配列にしてください。出力は説明や Markdown を含まない JSON オブジェクトのみとしてください。"""
pipeline\run_zeroshot_eval.py-28-
pipeline\run_zeroshot_eval.py-29-OUTPUT_INSTRUCTION = """出力 JSON の形式:
pipeline\run_zeroshot_eval.py-30-{"transcript": str, "errors": [{"step_id": str|null, "bbox": [x0,y0,x1,y1], "type": str}], "score": int}"""
pipeline\run_zeroshot_eval.py-31-

codex
## 設計判断

**(d) 階層 KPI を直ちに採用し、(e) 観測可能な下位区分へ再定義する。次の生成・訓練サイクルで (b) を実施する。**  
現時点ではファミリ一致だけを必須 KPI とし、下位区分は診断指標とする。下位区分は後述の昇格条件を満たした場合に限り、別の必須 KPI に戻す。

### 評価階層

第1階層は、注入元ではなく答案上で観測できる誤った変換によって定義する。

- 演算順序
- 移項
  - 現行の「概念/移項」「記法/移項符号」を統合
- 符号・乗算
  - 現行の「概念/符号・乗算」「計算/符号」を統合
- 符号落とし
- 加減
- 加減乗除
- 除算

第2階層は、原因を断定せず、答案内の証拠を表す。

- `移項/概念根拠あり`
- `移項/原因不明`
- `符号・乗算/概念根拠あり`
- `符号・乗算/原因不明`

`概念根拠あり` は、次のいずれかが答案中に観測できる場合に限定する。

1. 「移項しても符号は変わらない」等、誤った一般則を明示している。
2. 独立した2箇所以上で同じ誤規則を一貫して適用している。

単発の誤り式だけなら、生成オペレータが概念系でも `原因不明` とする。逆に、機械変異で作った式に明示的な誤規則の説明を組み合わせたなら `概念根拠あり` とする。`operator_id` と深刻度は生成 provenance として保持するが、VLM の種別 GT にはしない。

## KPI の運用

- 必須 KPI：第1階層のクラス別一致マクロ ≥ 0.85
- 診断指標：第2階層の完全一致マクロ、クラス別 recall、混同行列
- HF と llama.cpp の**双方で**必須値を満たすこと。平均して帳尻を合わせない。
- 現行 n=200 は継続比較用に残すが、少数クラス n=3 で設計判断しない。別に n=400 の固定層化テストを作り、移項と符号・乗算を各100件以上、残りを各40件前後含める。評価負荷は両ランタイム合計で約80分。
- 点数 GT は変更しない。旧・新 evaluator で同じ200件の `score` と bbox 評価が完全一致することを改定時の回帰テストにする。

第2階層を必須 KPI に昇格する条件は、未見の説明表現を用いた100組の対照ペアを含む n=400 で、次をすべて満たすこととする。

- 両ランタイムで第2階層マクロ ≥ 0.85
- 各対象 subtype の recall ≥ 0.80
- ランタイム間のマクロ差 ≤ 0.05
- 同じ誤り式に対して、説明・反復文脈だけを変えた対照ペアの判定正解率 ≥ 0.90

一度の1万件再生成と再訓練、約10時間の計算で判定可能である。

## 選択肢への判断

- **(c) 現行維持は明確に否定する。** sim2real は入力分布を変えるが、レンダリングから観測できない生成 provenance を予測可能にはしない。現行 KPI を維持したまま待つのは、測定不能性とモデル不足を混同する。
- (a) 単独採用も否定する。ファミリだけに恒久統合すると、教育的に重要な概念根拠を捨てる。今回は「必須 KPI をファミリへ移す」が、「観測可能な下位区分は残して再昇格させる」判断である。
- (b) 単独では不十分。概念例だけに定型句を追加すると、概念理解ではなく語句検出の合成ショートカットになる。
- (d) は即時採用する。
- (e) は下位 KPI 再昇格の前提とする。

なお現行コードでは、`pipeline/llm_mutation.py` が `reason` を要求している一方、`pipeline/gen_core.py` の mutation payload には式・値しか保持されず、`pipeline/verbalizer.py` にも operator の理由が明示的には渡らない。したがって現状の「誤解の語り」は生成され得ても、必須化されてはいない。(b) では、検証済みの誤規則を構造化 payload に保持し、verbalizer に渡す必要がある。

## 主要主張と反証条件

### 1. 現行の概念対ミス分類は、モデル性能より教師ラベルの観測不能性が支配的である

- **事実、確信度99%：** 移項35件はファミリ一致100%、下位区分だけが双方向に混同し、ランタイム間で混同方向も逆転している。CER はほぼゼロである。
- **推測、確信度95%：** 単発答案から生成原因を推定させる現行教師信号には、安定した決定境界がない。
- **反証条件：** 現行定義のまま移項60組、符号60組の式条件を合わせた答案を作り、未見説明句で評価する。内部の独立した2名が transcript のみから balanced accuracy ≥0.85、κ≥0.80で分類し、既存モデルも両ランタイムでマクロ ≥0.85・差≤0.05なら、この主張は崩れる。生成・評価・盲検確認で半日以内。

### 2. ファミリ KPI は検出能力の内部回帰指標として妥当である

- **事実、確信度99%：** 位置、検出、点数は高水準で、点数 GT は種別非依存である。
- **推測、確信度92%：** 現状の主要な学習済み能力は「どこで、どの数学的変換を誤ったか」であり、第1階層がそれを最も直接に測る。
- **反証条件：** subtype 誤分類だが family 正解の50件について、実際に生成するフィードバックを盲検比較する。10件以上で、同一 family 判定のままでは誤った再教育または不適切な注意喚起になるなら、family 単独の必須 KPI は不十分であり、下位区分を即時のリリースゲートに追加する。

### 3. 観測証拠による再定義と文脈生成で、下位区分は再び学習可能になる

- **推測、確信度78%：** 明示的誤規則と反復適用を教師信号にすれば、潜在的な注入原因ではなく入力中の証拠を分類できる。
- **反証条件：** 上記仕様で1万件を再生成・再訓練し、未見表現を含む n=400 で昇格条件のいずれかを満たさなければ、この設計は少なくとも2Bモデルには不成立と判断する。追加反復で曖昧に延命せず、下位区分を参考指標のまま固定する。

### 4. 現行維持より今の再定義の方が合理的である

- **推測、確信度94%：** 数日内に反復可能で、点数系への波及もなく、待つより安価に仮説を切り分けられる。
- **反証条件：** 定義を変えず、クラス増量だけを行う1回の再生成・再訓練で、未見説明句 n=400 に対して両ランタイムの9クラスマクロ ≥0.85、対象4クラス recall ≥0.80、ランタイム差≤0.05を満たすなら、再定義の必要性は弱まり、現行分類を維持してよい。

## 自説の最弱点

**`原因不明` は観測上は正しいが、実際には概念的誤解を持つ生徒を過小診断する可能性がある。確信度65%。** 人間は単発答案から、筆跡上の訂正、途中式の選び方、授業文脈など、現在の合成データにない信号を使えるかもしれない。

計画済み1,350頁のうち、移項・符号系を最低100答案抽出し、2名の教員が現行の「概念／単純ミス」を独立判定したとき、κ≥0.70かつ単発・説明なし答案でも一致率≥0.80になれば、自説の中心である「原因を答案表面からは断定しない」は崩れる。その場合は、人間が使った観測特徴を生成器へ追加し、旧来の概念対ミス区分を第2階層の必須 KPI として復活させる。
tokens used
38,307
## 設計判断

**(d) 階層 KPI を直ちに採用し、(e) 観測可能な下位区分へ再定義する。次の生成・訓練サイクルで (b) を実施する。**  
現時点ではファミリ一致だけを必須 KPI とし、下位区分は診断指標とする。下位区分は後述の昇格条件を満たした場合に限り、別の必須 KPI に戻す。

### 評価階層

第1階層は、注入元ではなく答案上で観測できる誤った変換によって定義する。

- 演算順序
- 移項
  - 現行の「概念/移項」「記法/移項符号」を統合
- 符号・乗算
  - 現行の「概念/符号・乗算」「計算/符号」を統合
- 符号落とし
- 加減
- 加減乗除
- 除算

第2階層は、原因を断定せず、答案内の証拠を表す。

- `移項/概念根拠あり`
- `移項/原因不明`
- `符号・乗算/概念根拠あり`
- `符号・乗算/原因不明`

`概念根拠あり` は、次のいずれかが答案中に観測できる場合に限定する。

1. 「移項しても符号は変わらない」等、誤った一般則を明示している。
2. 独立した2箇所以上で同じ誤規則を一貫して適用している。

単発の誤り式だけなら、生成オペレータが概念系でも `原因不明` とする。逆に、機械変異で作った式に明示的な誤規則の説明を組み合わせたなら `概念根拠あり` とする。`operator_id` と深刻度は生成 provenance として保持するが、VLM の種別 GT にはしない。

## KPI の運用

- 必須 KPI：第1階層のクラス別一致マクロ ≥ 0.85
- 診断指標：第2階層の完全一致マクロ、クラス別 recall、混同行列
- HF と llama.cpp の**双方で**必須値を満たすこと。平均して帳尻を合わせない。
- 現行 n=200 は継続比較用に残すが、少数クラス n=3 で設計判断しない。別に n=400 の固定層化テストを作り、移項と符号・乗算を各100件以上、残りを各40件前後含める。評価負荷は両ランタイム合計で約80分。
- 点数 GT は変更しない。旧・新 evaluator で同じ200件の `score` と bbox 評価が完全一致することを改定時の回帰テストにする。

第2階層を必須 KPI に昇格する条件は、未見の説明表現を用いた100組の対照ペアを含む n=400 で、次をすべて満たすこととする。

- 両ランタイムで第2階層マクロ ≥ 0.85
- 各対象 subtype の recall ≥ 0.80
- ランタイム間のマクロ差 ≤ 0.05
- 同じ誤り式に対して、説明・反復文脈だけを変えた対照ペアの判定正解率 ≥ 0.90

一度の1万件再生成と再訓練、約10時間の計算で判定可能である。

## 選択肢への判断

- **(c) 現行維持は明確に否定する。** sim2real は入力分布を変えるが、レンダリングから観測できない生成 provenance を予測可能にはしない。現行 KPI を維持したまま待つのは、測定不能性とモデル不足を混同する。
- (a) 単独採用も否定する。ファミリだけに恒久統合すると、教育的に重要な概念根拠を捨てる。今回は「必須 KPI をファミリへ移す」が、「観測可能な下位区分は残して再昇格させる」判断である。
- (b) 単独では不十分。概念例だけに定型句を追加すると、概念理解ではなく語句検出の合成ショートカットになる。
- (d) は即時採用する。
- (e) は下位 KPI 再昇格の前提とする。

なお現行コードでは、`pipeline/llm_mutation.py` が `reason` を要求している一方、`pipeline/gen_core.py` の mutation payload には式・値しか保持されず、`pipeline/verbalizer.py` にも operator の理由が明示的には渡らない。したがって現状の「誤解の語り」は生成され得ても、必須化されてはいない。(b) では、検証済みの誤規則を構造化 payload に保持し、verbalizer に渡す必要がある。

## 主要主張と反証条件

### 1. 現行の概念対ミス分類は、モデル性能より教師ラベルの観測不能性が支配的である

- **事実、確信度99%：** 移項35件はファミリ一致100%、下位区分だけが双方向に混同し、ランタイム間で混同方向も逆転している。CER はほぼゼロである。
- **推測、確信度95%：** 単発答案から生成原因を推定させる現行教師信号には、安定した決定境界がない。
- **反証条件：** 現行定義のまま移項60組、符号60組の式条件を合わせた答案を作り、未見説明句で評価する。内部の独立した2名が transcript のみから balanced accuracy ≥0.85、κ≥0.80で分類し、既存モデルも両ランタイムでマクロ ≥0.85・差≤0.05なら、この主張は崩れる。生成・評価・盲検確認で半日以内。

### 2. ファミリ KPI は検出能力の内部回帰指標として妥当である

- **事実、確信度99%：** 位置、検出、点数は高水準で、点数 GT は種別非依存である。
- **推測、確信度92%：** 現状の主要な学習済み能力は「どこで、どの数学的変換を誤ったか」であり、第1階層がそれを最も直接に測る。
- **反証条件：** subtype 誤分類だが family 正解の50件について、実際に生成するフィードバックを盲検比較する。10件以上で、同一 family 判定のままでは誤った再教育または不適切な注意喚起になるなら、family 単独の必須 KPI は不十分であり、下位区分を即時のリリースゲートに追加する。

### 3. 観測証拠による再定義と文脈生成で、下位区分は再び学習可能になる

- **推測、確信度78%：** 明示的誤規則と反復適用を教師信号にすれば、潜在的な注入原因ではなく入力中の証拠を分類できる。
- **反証条件：** 上記仕様で1万件を再生成・再訓練し、未見表現を含む n=400 で昇格条件のいずれかを満たさなければ、この設計は少なくとも2Bモデルには不成立と判断する。追加反復で曖昧に延命せず、下位区分を参考指標のまま固定する。

### 4. 現行維持より今の再定義の方が合理的である

- **推測、確信度94%：** 数日内に反復可能で、点数系への波及もなく、待つより安価に仮説を切り分けられる。
- **反証条件：** 定義を変えず、クラス増量だけを行う1回の再生成・再訓練で、未見説明句 n=400 に対して両ランタイムの9クラスマクロ ≥0.85、対象4クラス recall ≥0.80、ランタイム差≤0.05を満たすなら、再定義の必要性は弱まり、現行分類を維持してよい。

## 自説の最弱点

**`原因不明` は観測上は正しいが、実際には概念的誤解を持つ生徒を過小診断する可能性がある。確信度65%。** 人間は単発答案から、筆跡上の訂正、途中式の選び方、授業文脈など、現在の合成データにない信号を使えるかもしれない。

計画済み1,350頁のうち、移項・符号系を最低100答案抽出し、2名の教員が現行の「概念／単純ミス」を独立判定したとき、κ≥0.70かつ単発・説明なし答案でも一致率≥0.80になれば、自説の中心である「原因を答案表面からは断定しない」は崩れる。その場合は、人間が使った観測特徴を生成器へ追加し、旧来の概念対ミス区分を第2階層の必須 KPI として復活させる。
