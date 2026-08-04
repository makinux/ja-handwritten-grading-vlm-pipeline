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
import difflib
import random
import re
import unicodedata

NUM_RE = re.compile(r"-?\d+")

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
    x = rng.randint(-9, 9)
    b = rng.choice([v for v in range(-9, 10) if v != 0])
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
    op2 = "-" if b > 0 else "+"
    if m == "op-transpose-sign" and site == "s2":
        op2 = "+" if op2 == "-" else "-"
    tk = Tok()
    steps.append(step("s2", "transpose",
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
        "provenance": {"pipeline": "phase0-bootstrap-v0.1", "gates_passed": []},
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
    for s in steps:
        if NUM_RE.findall(s["text"]) != s["numbers"]:
            reasons.append(f"G1b: {s['step_id']} の数値が不一致")

    # (c) スキーマ
    for k in ("domain", "problem_text", "answer_key", "params"):
        if k not in problem:
            reasons.append(f"G1c: フィールド欠落 {k}")
    return (len(reasons) == 0), reasons


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


# ------------------------------------------------------------------- batch


def generate_batch(n_total, seed, control_ratio=0.30):
    """誤りあり/なしを 70/30 で生成。対照はペア生成原則に従い誤りサンプルと
    (問題・スタイル乱数) を共有する。"""
    rng = random.Random(seed)
    n_err = round(n_total * (1 - control_ratio))
    n_ctrl = n_total - n_err

    records, problems = [], []
    g1_pass = g1_fail = 0
    for i in range(n_err):
        domain = rng.choice(list(DOMAINS))
        gen_fn, _, mut_fn = DOMAINS[domain]
        problem = gen_fn(rng)
        ok, reasons = g1_gate(problem)
        if not ok:
            g1_fail += 1
            continue
        g1_pass += 1
        mut = rng.choice(mut_fn(rng, problem["params"]))
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

    n_ctrl_kept = sum(1 for r, _ in kept if r["control_flag"]["error_free"])
    stats = {
        "n_generated": len(records),
        "g1_pass": g1_pass, "g1_fail": g1_fail,
        "g2_pass": g2_pass, "g2_fail": g2_fail,
        "g2_fail_reasons": fail_reasons[:5],
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
