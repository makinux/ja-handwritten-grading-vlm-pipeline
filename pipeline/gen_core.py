# -*- coding: utf-8 -*-
"""M1(問題・正解生成)+ M2(誤り注入)+ G1/G2 ゲート。

設計書 v2.1 §3 M1/M2 準拠のブートストラップ実装。
- 解答は「実行可能なステップ列(solution program)」として構築し、
  日本語文はテンプレート逐語化(LLM 逐語化は将来差し替えのスタブ)。
- 誤り注入はプログラム・ミューテーション。ミュータント再実行で下流を再計算し、
  mutation_site / causally_affected_nodes を分離して記録する(項目0)。
- 点数 GT はルーブリック項目充足集合から計算する(v2.1 の方向に準拠)。
"""
from fractions import Fraction
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import difflib
import os
import random
import re
import unicodedata

from llm_mutation import safe_parse_expr

NUM_RE = re.compile(r"-?\d+")

CONFIG = {
    # 一次方程式の正解に分数解を許す。text-only プローブが検出した
    # 「正解は常に整数解→分数の出現が誤りを示唆する」というテキストリーク
    # の除去(人工的な分布均衡化ではなく、生成器の非現実的制約の撤廃)。
    "linear_fraction_golds": True,
}

# ---------------------------------------------------------------- formatting


class Tok:
    """テキストへ数値を出力しつつ、逐語化照合(G1b)用に期待トークンを登録する。"""

    def __init__(self):
        self.nums = []

    def n(self, v):
        if isinstance(v, Fraction) and v.denominator != 1:
            self.nums.append(str(v.numerator))
            self.nums.append(str(v.denominator))
            return f"{v.numerator}/{v.denominator}"
        iv = int(v)
        self.nums.append(str(iv))
        return str(iv)


def paren(s):
    return f"({s})" if s.startswith("-") else s


def step(step_id, op, text, numbers):
    return {"step_id": step_id, "op": op, "text": text, "latex": text,
            "numbers": list(numbers)}


# ------------------------------------------------------------- mutation spec


def mut_spec(op_id, site, err_type, severity, payload=None):
    return {"operator": op_id, "site": site, "type": err_type,
            "severity": severity, "payload": payload or {}}


# ----------------------------------------------------------- linear equation
# 一次方程式 ax + b = c(a は ±2..5、x 整数、b≠0)


def gen_linear_problem(rng):
    a = rng.choice([2, 3, 4, 5, -2, -3])
    b = rng.choice([v for v in range(-9, 10) if v != 0])
    if CONFIG["linear_fraction_golds"] and rng.random() < 0.4:
        c = rng.randint(-20, 20)
        if (c - b) % abs(a) == 0:  # 非整数解を強制
            c += 1
    else:
        x = rng.randint(-9, 9)
        c = a * x + b
    params = {"a": a, "b": b, "c": c}
    tk = Tok()
    sb = "+" if b > 0 else "-"
    eq = f"{tk.n(a)}x {sb} {tk.n(abs(b))} = {tk.n(c)}"
    return {
        "domain": "一次方程式", "grade": 7, "generator": "bootstrap/linear_1var@v0.1",
        "params": params,
        "problem_text": f"次の方程式を解きなさい。 {eq}",
        "problem_numbers": tk.nums,
        "answer_key": str(Fraction(c - b, a)) if Fraction(c - b, a).denominator != 1
        else str(int(Fraction(c - b, a))),
    }


def exec_linear(params, mut):
    a, b, c = params["a"], params["b"], params["c"]
    m = (mut or {}).get("operator")
    site = (mut or {}).get("site")
    pay = (mut or {}).get("payload", {})
    steps = []

    tk = Tok()
    sb = "+" if b > 0 else "-"
    steps.append(step("s1", "setup",
                      f"{tk.n(a)}x {sb} {tk.n(abs(b))} = {tk.n(c)}", tk.nums))

    # s2 移項: 正しくは c から b を引く(表示は abs(b) と演算子)
    if m == "op-llm-transpose-misconception" and site == "s2":
        tk = Tok()
        a_text = tk.n(a)
        for token in pay["expr_tokens"]:
            tk.n(token)
        steps.append(step("s2", "transpose",
                          f"{a_text}x = {pay['expr_canonical']}", tk.nums))
        d = Fraction(pay["d_value"])
    else:
        op2 = "-" if b > 0 else "+"
        if m == "op-transpose-sign" and site == "s2":
            op2 = "+" if op2 == "-" else "-"
        tk = Tok()
        steps.append(step(
            "s2", "transpose",
            f"{tk.n(a)}x = {tk.n(c)} {op2} {tk.n(abs(b))}", tk.nums))
        d = c - abs(b) if op2 == "-" else c + abs(b)

    # s3 右辺の計算
    if m == "op-arith-slip" and site == "s3":
        d = d + pay["delta"]
    tk = Tok()
    steps.append(step("s3", "simplify", f"{tk.n(a)}x = {tk.n(d)}", tk.nums))

    # s4 除算
    ans = Fraction(d, a)
    if m == "op-divide-error" and site == "s4":
        if pay["kind"] == "mul":
            ans = Fraction(d * a)
        else:
            ans = Fraction(d, a) + pay["delta"]
    if m == "op-final-sign-drop" and site == "s4":
        ans = abs(ans)
    tk = Tok()
    steps.append(step("s4", "divide", f"x = {tk.n(ans)}", tk.nums))
    return steps, ans


def linear_mutations(rng, params):
    a, b, c = params["a"], params["b"], params["c"]
    gold_ans = Fraction(c - b, a)
    muts = [
        mut_spec("op-transpose-sign", "s2", "記法/移項符号", 2),
        mut_spec("op-arith-slip", "s3", "計算/加減", 1,
                 {"delta": rng.choice([-2, -1, 1, 2])}),
        mut_spec("op-divide-error", "s4", "計算/除算", 2,
                 {"kind": rng.choice(["mul", "off"]),
                  "delta": rng.choice([-1, 1])}),
    ]
    if gold_ans < 0:
        muts.append(mut_spec("op-final-sign-drop", "s4", "表記/符号落とし", 1))
    return muts


# --------------------------------------------------------- integer arithmetic
# 正負の数: p + q × r(演算の優先順位)


def gen_arith_problem(rng):
    nz = [v for v in range(-9, 10) if v != 0]
    p, q, r = rng.choice(nz), rng.choice(nz), rng.choice(nz)
    params = {"p": p, "q": q, "r": r}
    tk = Tok()
    expr = f"{tk.n(p)} + {paren(tk.n(q))} × {paren(tk.n(r))}"
    return {
        "domain": "正負の数", "grade": 7, "generator": "bootstrap/int_arith@v0.1",
        "params": params,
        "problem_text": f"次の計算をしなさい。 {expr}",
        "problem_numbers": tk.nums,
        "answer_key": str(p + q * r),
    }


def exec_arith(params, mut):
    p, q, r = params["p"], params["q"], params["r"]
    m = (mut or {}).get("operator")
    site = (mut or {}).get("site")
    pay = (mut or {}).get("payload", {})
    steps = []

    tk = Tok()
    steps.append(step("s1", "setup",
                      f"{tk.n(p)} + {paren(tk.n(q))} × {paren(tk.n(r))}", tk.nums))

    if m == "op-order-of-ops" and site == "s2":
        # 概念誤り: 左から計算 (p + q) × r
        t = p + q
        tk = Tok()
        steps.append(step("s2", "add_first",
                          f"{tk.n(p)} + {paren(tk.n(q))} = {tk.n(t)}", tk.nums))
        ans = t * r
        tk = Tok()
        steps.append(step("s3", "mul_second",
                          f"{paren(tk.n(t))} × {paren(tk.n(r))} = {tk.n(ans)}",
                          tk.nums))
    else:
        mv = q * r
        if m == "op-llm-product-misconception" and site == "s2":
            mv = pay["mv_value"]
        if m == "op-mult-sign-drop" and site == "s2":
            mv = abs(mv)
        if m == "op-arith-slip" and site == "s2":
            mv = mv + pay["delta"]
        tk = Tok()
        steps.append(step("s2", "multiply",
                          f"{paren(tk.n(q))} × {paren(tk.n(r))} = {tk.n(mv)}",
                          tk.nums))
        ans = p + mv
        if m == "op-arith-slip" and site == "s3":
            ans = ans + pay["delta"]
        tk = Tok()
        steps.append(step("s3", "add",
                          f"{tk.n(p)} + {paren(tk.n(mv))} = {tk.n(ans)}", tk.nums))

    if m == "op-final-sign-drop" and site == "s4":
        ans = abs(ans)
    tk = Tok()
    steps.append(step("s4", "answer", f"答え {tk.n(ans)}", tk.nums))
    return steps, Fraction(ans)


def m2_applies(mut, site):
    return mut and mut.get("site") == site


def arith_mutations(rng, params):
    p, q, r = params["p"], params["q"], params["r"]
    gold = p + q * r
    muts = [
        mut_spec("op-arith-slip", rng.choice(["s2", "s3"]), "計算/加減乗除", 1,
                 {"delta": rng.choice([-2, -1, 1, 2])}),
    ]
    if q * r < 0:
        muts.append(mut_spec("op-mult-sign-drop", "s2", "計算/符号", 2))
    if r != 1 and q != 0:
        muts.append(mut_spec("op-order-of-ops", "s2", "概念/演算順序", 3))
    if gold < 0:
        muts.append(mut_spec("op-final-sign-drop", "s4", "表記/符号落とし", 1))
    return muts


def _llm_mut_from_proposal(problem, proposal):
    """LLM 提案を検証し、再実行可能な mutation spec へ変換する。"""
    if not isinstance(proposal, dict):
        raise ValueError("LLM proposal must be a dictionary")
    params = problem["params"]
    if problem["domain"] == "一次方程式":
        b, c = (params[key] for key in ("b", "c"))
        value, tokens, canonical = safe_parse_expr(proposal.get("expr"))
        allowed = {abs(b), abs(c)}
        if not set(abs(token) for token in tokens) <= allowed:
            raise ValueError("linear proposal contains a disallowed constant")
        if value == Fraction(c - b):
            raise ValueError("linear proposal equals the gold transpose value")
        return mut_spec(
            "op-llm-transpose-misconception", "s2", "概念/移項", 3,
            {"expr_canonical": canonical, "expr_tokens": tokens,
             "d_value": str(value)})

    if problem["domain"] == "正負の数":
        q, r = (params[key] for key in ("q", "r"))
        value = proposal.get("value")
        if type(value) is not int:
            raise ValueError("arithmetic proposal value must be an integer")
        gold = q * r
        if abs(value) > abs(gold) + 30:
            raise ValueError("arithmetic proposal value is out of range")
        if value == gold:
            raise ValueError("arithmetic proposal equals the gold product")
        return mut_spec(
            "op-llm-product-misconception", "s2", "概念/符号・乗算", 3,
            {"mv_value": value})

    raise ValueError(f"unsupported LLM mutation domain: {problem['domain']}")


def _propose_llm_mutation(llm_mutator, problem):
    """領域別の提案 API を呼び、検証済み mutation spec を返す。"""
    params = problem["params"]
    if problem["domain"] == "一次方程式":
        proposal = llm_mutator.propose_linear_transpose(
            problem, params["a"], params["b"], params["c"])
    elif problem["domain"] == "正負の数":
        proposal = llm_mutator.propose_arith_product(
            problem, params["q"], params["r"])
    else:
        raise ValueError(f"unsupported LLM mutation domain: {problem['domain']}")
    return _llm_mut_from_proposal(problem, proposal)


def _mutation_changes_site(problem, mutation, exec_fn):
    """G2 で除外される、mutation_site が不変の候補を事前判定する。"""
    gold_steps, _ = exec_fn(problem["params"], None)
    mutant_steps, _ = exec_fn(problem["params"], mutation)
    site = mutation["site"]
    gold_by_id = {step["step_id"]: step["text"] for step in gold_steps}
    mutant_by_id = {step["step_id"]: step["text"] for step in mutant_steps}
    return gold_by_id[site] != mutant_by_id.get(site)


# ------------------------------------------------------------------ assembly

DOMAINS = {
    "一次方程式": (gen_linear_problem, exec_linear, linear_mutations),
    "正負の数": (gen_arith_problem, exec_arith, arith_mutations),
}

AFFECTED = {  # mutation_site -> causally affected downstream nodes
    "s1": ["s2", "s3", "s4"],
    "s2": ["s3", "s4"],
    "s3": ["s4"],
    "s4": [],
}


def rubric_for(rng, domain):
    presets = [(2, 4, 4), (3, 4, 3), (2, 5, 3)]
    w1, w2, w3 = rng.choice(presets)
    item2 = "変形過程" if domain == "一次方程式" else "途中の計算"
    return {
        "fn_id": f"rb-{'lin' if domain == '一次方程式' else 'ar'}-w{w1}{w2}{w3}",
        "weights": {"立式": w1, item2: w2, "最終解": w3},
        "item2": item2,
        "text_ja": f"採点基準:立式 {w1} 点、{item2} {w2} 点、最終解 {w3} 点(計 10 点)",
    }


def span_of_mutation(gold_text, mut_text):
    """site ステップの gold/mutant テキスト差分から誤りスパン [start, end) を得る。"""
    sm = difflib.SequenceMatcher(a=gold_text, b=mut_text, autojunk=False)
    lo, hi = None, None
    for tag, _a1, _a2, b1, b2 in sm.get_opcodes():
        if tag == "equal":
            continue
        lo = b1 if lo is None else min(lo, b1)
        hi = b2 if hi is None else max(hi, b2)
    if lo is None:  # 差分なし(起きないはず)→ 全体
        lo, hi = 0, len(mut_text)
    if hi <= lo:
        hi = min(len(mut_text), lo + 1)
    return [lo, hi]


def score_by_rubric(rubric, domain, site, affected, final_ok):
    """ルーブリック項目充足集合から点数を決定論的に計算する。"""
    dirty = set([site] + affected) if site else set()
    w = rubric["weights"]
    item2 = rubric["item2"]
    award = 0
    award += w["立式"] if "s1" not in dirty else 0
    mid = ["s2", "s3"]
    clean_mid = len([s for s in mid if s not in dirty])
    award += (w[item2] * clean_mid) // len(mid)
    award += w["最終解"] if final_ok else 0
    return award


def make_record(idx, problem, mut, rng):
    """1 サンプル(誤りあり or 対照)の完全レコードを構築する。"""
    domain = problem["domain"]
    _, exec_fn, _ = DOMAINS[domain]
    gold_steps, gold_ans = exec_fn(problem["params"], None)
    if mut is not None:
        mut_steps, mut_ans = exec_fn(problem["params"], mut)
    else:
        mut_steps, mut_ans = gold_steps, gold_ans

    site = mut["site"] if mut else None
    affected = []
    if mut:
        gold_by_id = {s["step_id"]: s for s in gold_steps}
        affected = [sid for sid in AFFECTED[site]
                    if gold_by_id[sid]["text"] !=
                    {s["step_id"]: s for s in mut_steps}[sid]["text"]]

    # ルーブリックは (問題インデックス, 領域) から決定論的に導出し、
    # ペアの双子(誤りあり/なし)で必ず同一になるようにする
    rubric = rubric_for(random.Random(f"rub-{idx}-{domain}"), domain)
    final_ok = (mut_ans == gold_ans)
    awarded = score_by_rubric(rubric, domain, site, affected, final_ok) \
        if mut else 10

    errors = []
    if mut:
        site_gold = {s["step_id"]: s for s in gold_steps}[site]["text"]
        site_mut = {s["step_id"]: s for s in mut_steps}[site]["text"]
        errors.append({
            "type": mut["type"],
            "operator": mut["operator"],
            "mutation_site": site,
            "causally_affected_nodes": affected,
            "span": span_of_mutation(site_gold, site_mut),
            "severity": mut["severity"],
            "propagates": len(affected) > 0,
        })

    transcript = problem["problem_text"] + "\n" + \
        "\n".join(s["text"] for s in mut_steps)

    tag = "e1" if mut else "e0"
    dom_tag = "lin" if domain == "一次方程式" else "ar"
    return {
        "sample_id": f"math-g7-{dom_tag}-{idx:06d}-{tag}",
        "pair_id": f"math-g7-{dom_tag}-{idx:06d}",
        "problem": {
            "text_ja": problem["problem_text"], "grade": problem["grade"],
            "domain": domain, "generator": problem["generator"],
        },
        "gold_solution": gold_steps,
        "answer_key": {"final": problem["answer_key"],
                       "verified_by": "fraction-exact"},
        "rubric": {k: rubric[k] for k in ("fn_id", "weights", "text_ja")},
        "injected_errors": errors,
        "mutant_solution": mut_steps if mut else [],
        "transcript_gt": {"text": transcript,
                          "normalization": "NFKC+strip-space"},
        "score_gt": {"full": 10, "awarded": awarded},
        "control_flag": {"error_free": mut is None},
        "provenance": {"pipeline": "phase0-bootstrap-v0.2", "gates_passed": [],
                       "verbalizer": "template"},
    }


# ---------------------------------------------------------------------- G1


def g1_gate(problem):
    """(a) プログラム実行と最終解の一致 (b) 逐語化照合 (c) スキーマ検査"""
    reasons = []
    domain = problem["domain"]
    _, exec_fn, _ = DOMAINS[domain]
    steps, ans = exec_fn(problem["params"], None)

    # (a) 実行検証(Fraction 厳密算術)
    if domain == "一次方程式":
        a, b, c = (problem["params"][k] for k in ("a", "b", "c"))
        if a * ans + b != c:
            reasons.append("G1a: 解が方程式を満たさない")
    else:
        p, q, r = (problem["params"][k] for k in ("p", "q", "r"))
        if ans != p + q * r:
            reasons.append("G1a: 計算結果が一致しない")
    key = problem["answer_key"]
    ans_str = str(ans) if ans.denominator != 1 else str(int(ans))
    if ans_str != key:
        reasons.append("G1a: answer_key と実行結果が不一致")

    # (b) 逐語化照合: テキスト中の数値トークン列 == プログラム登録トークン列
    if NUM_RE.findall(problem["problem_text"]) != problem["problem_numbers"]:
        reasons.append("G1b: 問題文の数値がプログラム値と不一致")
    for sid in g1b_check_texts(steps, [s["text"] for s in steps]):
        reasons.append(f"G1b: {sid} の数値が不一致")

    # (c) スキーマ
    for k in ("domain", "problem_text", "answer_key", "params"):
        if k not in problem:
            reasons.append(f"G1c: フィールド欠落 {k}")
    return (len(reasons) == 0), reasons


def g1b_check_texts(steps, texts):
    """逐語化照合 v2 の本体(LLM 逐語化の受け入れゲート)。

    各文で期待トークン列(numbers)が抽出列の部分列になり、かつ抽出された
    全トークンが全ステップの既知数値(符号反転を含む)であることを確認する。
    answer 以外はテンプレート式そのものの包含も要求し、answer は「答」の
    文字と最終値トークンの包含を要求する。
    いずれかを満たさないステップ ID のリストを返す。
    """
    known = set()
    for s in steps:
        for token in s["numbers"]:
            known.add(token)
            known.add(token[1:] if token.startswith("-") else "-" + token)

    fails = []
    for s, text in zip(steps, texts):
        actual = NUM_RE.findall(text)
        expected = s["numbers"]
        template = s.get("text_template", s["text"])
        expected_pos = 0
        for token in actual:
            if expected_pos < len(expected) and token == expected[expected_pos]:
                expected_pos += 1
        if s.get("op") == "answer":
            form_ok = ("答" in normalize_text(text)
                       and bool(expected) and expected[-1] in actual)
        else:
            form_ok = normalize_text(template) in normalize_text(text)
        if (expected_pos != len(expected)
                or any(token not in known for token in actual)
                or not form_ok):
            fails.append(s["step_id"])
    return fails


# ---------------------------------------------------------------------- G2


def g2_gate(record, problem):
    """(a) diff 封じ込め (b) ミュータント再実行整合+誤りの成立 (c) は バッチ側"""
    if record["control_flag"]["error_free"]:
        return True, []
    reasons = []
    err = record["injected_errors"][0]
    site, affected = err["mutation_site"], err["causally_affected_nodes"]

    gold = {s["step_id"]: s["text"] for s in record["gold_solution"]}
    mut = {s["step_id"]: s["text"] for s in record["mutant_solution"]}
    changed = [sid for sid in gold if gold[sid] != mut.get(sid, "")]

    # (a) 変更ステップ ⊆ {site} ∪ affected、かつ site は必ず変更されている
    allowed = set([site] + affected)
    if not set(changed) <= allowed:
        reasons.append(f"G2a: 許容外の変更 {sorted(set(changed) - allowed)}")
    if site not in changed:
        reasons.append("G2a: mutation_site に変更がない(誤りが不成立)")

    # (b) 再実行の決定性(同一 mutation で再実行して一致)
    domain = problem["domain"]
    _, exec_fn, _ = DOMAINS[domain]
    re_steps, _ = exec_fn(problem["params"], _rebuild_mut(err))
    if [s["text"] for s in re_steps] != \
            [s["text"] for s in record["mutant_solution"]]:
        reasons.append("G2b: ミュータント再実行が一致しない")

    # (b') スパンの健全性
    lo, hi = err["span"]
    if not (0 <= lo < hi <= len(mut[site])):
        reasons.append("G2b: 誤りスパンが不正")
    return (len(reasons) == 0), reasons


def _rebuild_mut(err):
    return {"operator": err["operator"], "site": err["mutation_site"],
            "type": err["type"], "severity": err["severity"],
            "payload": err.get("_payload", {})}


# -------------------------------------------------------------- verbalization


def _verbalize_steps(verbalizer, problem, steps, retries):
    """逐語化して G1b を通ったステップだけを採用する。

    初回にまとめて逐語化し、不合格ステップだけを ``retries`` 回まで
    再生成する。呼び出し失敗や不正な戻り値も、その回の対象ステップが
    不合格だったものとして扱う。
    """
    templates = [s["text"] for s in steps]
    texts = list(templates)
    pending = list(range(len(steps)))
    passed = set()

    for _attempt in range(max(0, int(retries)) + 1):
        if not pending:
            break
        target_steps = [steps[i] for i in pending]
        try:
            candidates = verbalizer.verbalize(problem, target_steps)
            if (not isinstance(candidates, (list, tuple))
                    or len(candidates) != len(target_steps)
                    or any(not isinstance(text, str) or not text
                           for text in candidates)):
                raise ValueError("verbalizer returned invalid step texts")
            failed_ids = set(g1b_check_texts(target_steps, candidates))
        except Exception:
            candidates = [None] * len(target_steps)
            failed_ids = {s["step_id"] for s in target_steps}

        next_pending = []
        for original_i, candidate, target in zip(
                pending, candidates, target_steps):
            if target["step_id"] in failed_ids:
                next_pending.append(original_i)
            else:
                texts[original_i] = candidate
                passed.add(original_i)
        pending = next_pending

    fallback_ids = [steps[i]["step_id"] for i in pending]
    return texts, len(steps), len(passed), fallback_ids


def _rebuild_transcript(record):
    steps = (record["gold_solution"]
             if record["control_flag"]["error_free"]
             else record["mutant_solution"])
    record["transcript_gt"]["text"] = (record["problem"]["text_ja"] + "\n" +
                                         "\n".join(s["text"] for s in steps))


def _ordered_step_ids(record, ids):
    wanted = set(ids)
    return [s["step_id"] for s in record["gold_solution"]
            if s["step_id"] in wanted]


def _verbalize_pair(pair, verbalizer, retries):
    """1 ペアを逐語化し、そのペアだけの統計を返す。"""
    attempted = passed = fallback_steps = 0
    representative, problem = pair[0]
    gold_templates = {
        s["step_id"]: s["text"] for s in representative["gold_solution"]
    }
    gold_texts, n_attempted, n_passed, gold_fallback = _verbalize_steps(
        verbalizer, problem, representative["gold_solution"], retries)
    attempted += n_attempted
    passed += n_passed
    fallback_steps += len(gold_fallback)
    gold_by_id = {
        s["step_id"]: text
        for s, text in zip(representative["gold_solution"], gold_texts)
    }

    for record, _ in pair:
        # テンプレート原文は、共有した gold 文を上書きする前に保存する。
        for gold_step in record["gold_solution"]:
            gold_step["text_template"] = gold_templates[gold_step["step_id"]]
            gold_step["text"] = gold_by_id[gold_step["step_id"]]

        mutant_fallback = []
        if not record["control_flag"]["error_free"]:
            error = record["injected_errors"][0]
            affected = set([error["mutation_site"]] +
                           error["causally_affected_nodes"])
            target_steps = [s for s in record["mutant_solution"]
                            if s["step_id"] in affected]
            mutant_texts, n_attempted, n_passed, mutant_fallback = \
                _verbalize_steps(verbalizer, problem, target_steps, retries)
            attempted += n_attempted
            passed += n_passed
            fallback_steps += len(mutant_fallback)
            mutant_by_id = {s["step_id"]: text
                            for s, text in zip(target_steps, mutant_texts)}

            # 非変異ステップは文字列そのものを shared gold から継ぎ合わせる。
            for mutant_step in record["mutant_solution"]:
                sid = mutant_step["step_id"]
                mutant_step["text_template"] = mutant_step.get(
                    "text_template", mutant_step["text"])
                mutant_step["text"] = (mutant_by_id[sid] if sid in affected
                                       else gold_by_id[sid])
            respan_after_verbalization(record)

        fallback_parts = []
        ordered_gold_fallback = _ordered_step_ids(record, gold_fallback)
        ordered_mutant_fallback = _ordered_step_ids(record, mutant_fallback)
        if ordered_gold_fallback:
            fallback_parts.append("gold:" + ",".join(ordered_gold_fallback))
        if ordered_mutant_fallback:
            fallback_parts.append("mut:" + ",".join(ordered_mutant_fallback))
        record["provenance"]["verbalizer"] = (
            "llm+fallback(" + "|".join(fallback_parts) + ")"
            if fallback_parts else "llm")
        _rebuild_transcript(record)

    return attempted, passed, fallback_steps


def _verbalize_kept_pairs(kept, verbalizer, retries):
    """G2 通過後のレコードを pair_id 単位で差分逐語化する。"""
    groups = {}
    for record, problem in kept:
        groups.setdefault(record["pair_id"], []).append((record, problem))

    concurrency = int(os.environ.get("VLLM_CONCURRENCY", "4"))
    if concurrency < 1:
        raise ValueError("VLLM_CONCURRENCY must be at least 1")

    pairs = list(groups.values())
    if concurrency == 1:
        pair_stats = (
            _verbalize_pair(pair, verbalizer, retries) for pair in pairs)
        totals = [0, 0, 0]
        for stats in pair_stats:
            for i, value in enumerate(stats):
                totals[i] += value
        return tuple(totals)

    # 各ワーカーは自ペアのレコードだけを書き換え、統計はメインスレッドで合算する。
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        pair_stats = executor.map(
            lambda pair: _verbalize_pair(pair, verbalizer, retries), pairs)
        totals = [0, 0, 0]
        for stats in pair_stats:
            for i, value in enumerate(stats):
                totals[i] += value

    return tuple(totals)


def _whole_step_span(text):
    return [0, max(1, len(text))]


def _template_expression_span(text, template):
    """空白の揺れを許して、原文中のテンプレート式の範囲を返す。"""
    if not template:
        return None
    # 複数桁の数値は分割せず、それ以外は記号・変数を一文字ずつ扱う。
    # これにより ``20`` 自体は保持しつつ、各トークン間の空白を許容する。
    tokens = re.findall(r"\d+|[^\d\s]", template)
    if not tokens:
        return None
    pattern = r"\s*".join(re.escape(token) for token in tokens)
    match = re.search(pattern, text)
    return [match.start(), match.end()] if match else None


def _matches_in_span(text, search_span=None):
    matches = list(NUM_RE.finditer(text))
    if search_span is None:
        return matches
    lo, hi = search_span
    return [match for match in matches
            if lo <= match.start() and match.end() <= hi]


def _number_token_positions(template):
    """空白区切りの canonical 式で各数値が属するトークン位置を返す。"""
    tokens = template.split()
    positions = []
    for token_i, token in enumerate(tokens):
        positions.extend([token_i] * len(NUM_RE.findall(token)))
    return tokens, positions


def _changed_operator_pair(gold_template, mutant_template):
    """対応する隣接数値間で演算子が最初に異なる数値 index 対を返す。"""
    gold_tokens, gold_positions = _number_token_positions(gold_template)
    mutant_tokens, mutant_positions = _number_token_positions(mutant_template)
    if len(gold_positions) != len(mutant_positions):
        return None

    operators = {"+", "-", "×", "*", "/", "="}
    for number_i in range(len(gold_positions) - 1):
        gold_between = gold_tokens[
            gold_positions[number_i] + 1:gold_positions[number_i + 1]]
        mutant_between = mutant_tokens[
            mutant_positions[number_i] + 1:mutant_positions[number_i + 1]]
        gold_ops = [token for token in gold_between if token in operators]
        mutant_ops = [token for token in mutant_between if token in operators]
        if gold_ops != mutant_ops:
            return number_i, number_i + 1
    return None


def _operator_span(text, gold_site, mutant_site, search_span=None):
    """canonical 式で変化した演算子を逐語化後の式内へ係留する。"""
    gold_template = gold_site.get("text_template", gold_site.get("text", ""))
    mutant_template = mutant_site.get(
        "text_template", mutant_site.get("text", ""))
    pair = _changed_operator_pair(gold_template, mutant_template)
    matches = _matches_in_span(text, search_span)
    if pair is None or pair[1] >= len(matches):
        return None
    left, right = matches[pair[0]], matches[pair[1]]
    raw_lo, raw_hi = left.end(), right.start()
    between = text[raw_lo:raw_hi]
    leading = len(between) - len(between.lstrip())
    trailing = len(between) - len(between.rstrip())
    lo, hi = raw_lo + leading, raw_hi - trailing
    if hi <= lo:
        # 両端トリムで空になった場合も、安全な 1 文字を確保する。
        lo = min(raw_lo, max(0, len(text) - 1))
        hi = min(len(text), lo + 1)
    return [lo, hi] if 0 <= lo < hi <= len(text) else None


def respan_after_verbalization(record):
    """逐語化後の mutation_site 文に対して誤り span を再計算する。"""
    if record["control_flag"]["error_free"] or not record["injected_errors"]:
        return record

    error = record["injected_errors"][0]
    site = error["mutation_site"]
    gold_site = next((s for s in record["gold_solution"]
                      if s["step_id"] == site), None)
    mutant_site = next((s for s in record["mutant_solution"]
                        if s["step_id"] == site), None)
    if gold_site is None or mutant_site is None:
        return record

    text = mutant_site["text"]
    gold_template = gold_site.get("text_template", gold_site.get("text", ""))
    mutant_template = mutant_site.get(
        "text_template", mutant_site.get("text", ""))
    expression_span = _template_expression_span(text, mutant_template)
    span = None
    if NUM_RE.findall(gold_template) == NUM_RE.findall(mutant_template):
        span = _operator_span(text, gold_site, mutant_site, expression_span)
    else:
        gold_numbers = Counter(gold_site.get("numbers", []))
        matches = _matches_in_span(text, expression_span)
        # 既知数値の再言及は無視し、gold にない値を最優先で拾う。
        for match in matches:
            if gold_numbers[match.group(0)] == 0:
                span = [match.start(), match.end()]
                break
        # 変異値が別の gold 値と同値なら、多重度の超過を次善候補にする。
        if span is None:
            remaining = gold_numbers.copy()
            for match in matches:
                token = match.group(0)
                if remaining[token] > 0:
                    remaining[token] -= 1
                else:
                    span = [match.start(), match.end()]
                    break

    # 式を特定できた場合は、個別箇所を特定できなくても span を式内に保つ。
    # 式が見つからない場合だけ従来どおり全文を最終フォールバックにする。
    error["span"] = span or expression_span or _whole_step_span(text)
    return record


# ------------------------------------------------------------------- batch


def generate_batch(n_total, seed, control_ratio=0.30, verbalizer=None,
                   verbalize_retries=2, llm_mutator=None, llm_mut_prob=0.25):
    """誤りあり/なしを 70/30 で生成。対照はペア生成原則に従い誤りサンプルと
    (問題・スタイル乱数) を共有する。"""
    rng = random.Random(seed)
    if not 0 <= llm_mut_prob <= 1:
        raise ValueError("llm_mut_prob must be between 0 and 1")
    n_err = round(n_total * (1 - control_ratio))
    n_ctrl = n_total - n_err

    records, problems = [], []
    g1_pass = g1_fail = 0
    llm_mut_attempted = llm_mut_accepted = llm_mut_fallback = 0
    for i in range(n_err):
        domain = rng.choice(list(DOMAINS))
        gen_fn, exec_fn, mut_fn = DOMAINS[domain]
        problem = gen_fn(rng)
        ok, reasons = g1_gate(problem)
        if not ok:
            g1_fail += 1
            continue
        g1_pass += 1
        # 先に既存オペレータを選び、LLM 不合格時の復帰先を確保する。
        mutations = mut_fn(rng, problem["params"])
        mut = rng.choice(mutations)
        if not _mutation_changes_site(problem, mut, exec_fn):
            # 例: 0 を掛けても 0 のままになる除算誤り。乱数を追加消費せず、
            # 同じ候補集合内の成立する変異へ切り替える。
            mut = next(candidate for candidate in mutations
                       if _mutation_changes_site(problem, candidate, exec_fn))
        if llm_mutator is not None and rng.random() < llm_mut_prob:
            llm_mut_attempted += 1
            llm_mut = None
            # 初回呼び出しに加えて、最大 2 回リトライする。
            for _attempt in range(3):
                try:
                    llm_mut = _propose_llm_mutation(llm_mutator, problem)
                    break
                except Exception:
                    # API 障害・JSON 不正・検証不合格はいずれもデータにしない。
                    continue
            if llm_mut is None:
                llm_mut_fallback += 1
            else:
                mut = llm_mut
                llm_mut_accepted += 1
        rec = make_record(i, problem, mut, rng)
        rec["injected_errors"][0]["_payload"] = mut["payload"]
        rec["provenance"]["gates_passed"].append("G1")
        records.append(rec)
        problems.append(problem)

    # 対照群: 誤りサンプルからランダムに選んでペアの双子を作る
    twin_idx = rng.sample(range(len(records)), min(n_ctrl, len(records)))
    for i in twin_idx:
        problem = problems[i]
        rec = make_record(int(records[i]["pair_id"].split("-")[-1]),
                          problem, None, rng)
        rec["pair_id"] = records[i]["pair_id"]
        rec["provenance"]["gates_passed"].append("G1")
        records.append(rec)
        problems.append(problem)

    # G2
    g2_pass = g2_fail = 0
    kept = []
    fail_reasons = []
    for rec, problem in zip(records, problems):
        ok, reasons = g2_gate(rec, problem)
        if ok:
            if not rec["control_flag"]["error_free"]:
                rec["provenance"]["gates_passed"] += ["G2a", "G2b"]
            g2_pass += 1
            kept.append((rec, problem))
        else:
            g2_fail += 1
            fail_reasons.append((rec["sample_id"], reasons))

    verbalize_attempted = verbalize_pass = verbalize_fallback_steps = 0
    if verbalizer is not None:
        (verbalize_attempted, verbalize_pass,
         verbalize_fallback_steps) = _verbalize_kept_pairs(
             kept, verbalizer, verbalize_retries)

    n_ctrl_kept = sum(1 for r, _ in kept if r["control_flag"]["error_free"])
    stats = {
        "n_generated": len(records),
        "g1_pass": g1_pass, "g1_fail": g1_fail,
        "g2_pass": g2_pass, "g2_fail": g2_fail,
        "g2_fail_reasons": fail_reasons[:5],
        "llm_mut_attempted": llm_mut_attempted,
        "llm_mut_accepted": llm_mut_accepted,
        "llm_mut_fallback": llm_mut_fallback,
        "verbalize_attempted": verbalize_attempted,
        "verbalize_pass": verbalize_pass,
        "verbalize_fallback_steps": verbalize_fallback_steps,
        "control_ratio": n_ctrl_kept / max(1, len(kept)),
        "per_operator": _count_ops(kept),
    }
    return kept, stats


def _count_ops(kept):
    counts = {}
    for rec, _ in kept:
        if rec["control_flag"]["error_free"]:
            counts["(control)"] = counts.get("(control)", 0) + 1
        else:
            op = rec["injected_errors"][0]["operator"]
            counts[op] = counts.get(op, 0) + 1
    return counts


def normalize_text(s):
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", "", s)
