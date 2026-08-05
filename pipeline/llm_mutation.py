# -*- coding: utf-8 -*-
"""LLM 層の概念誤り提案と、提案された算術式の安全な検証。"""
import ast
from fractions import Fraction
import json
import os
import re
import urllib.request


_NUMBER_RE = re.compile(r"-?\d+")
_BINOP_TEXT = {ast.Add: "+", ast.Sub: "-", ast.Mult: "×"}


def safe_parse_expr(s):
    """制限付き算術式を検証し、値・数値列・正規形を返す。"""
    if not isinstance(s, str):
        raise ValueError("expression must be a string")
    if len(s) > 40:
        raise ValueError("expression is too long")

    source = s.replace("×", "*").replace("　", " ").replace("$", "").strip()
    if not source:
        raise ValueError("expression is empty")
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError("invalid expression syntax") from exc

    constants = 0

    def evaluate(node):
        nonlocal constants
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if type(node.value) is not int:
                raise ValueError("only integer constants are allowed")
            constants += 1
            if constants > 4:
                raise ValueError("too many constants")
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.USub, ast.UAdd):
            value = evaluate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOP_TEXT:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            return left * right
        raise ValueError(f"disallowed expression node: {type(node).__name__}")

    def render(node):
        if isinstance(node, ast.Expression):
            return render(node.body)
        if isinstance(node, ast.Constant):
            return str(node.value)
        if isinstance(node, ast.UnaryOp):
            sign = "-" if isinstance(node.op, ast.USub) else "+"
            inner = render(node.operand)
            if isinstance(node.operand, ast.Constant):
                return sign + inner
            return f"{sign}({inner})"
        if isinstance(node, ast.BinOp):
            left = render(node.left)
            right = render(node.right)
            if isinstance(node.left, ast.BinOp):
                left = f"({left})"
            if isinstance(node.right, ast.BinOp):
                right = f"({right})"
            return f"{left} {_BINOP_TEXT[type(node.op)]} {right}"
        # evaluate() が先に全ノードを検証するため、ここには到達しない。
        raise ValueError(f"disallowed expression node: {type(node).__name__}")

    value = evaluate(tree)
    canonical = render(tree)
    tokens = [int(token) for token in _NUMBER_RE.findall(canonical)]
    return value, tokens, canonical


class LLMMutationProposer:
    """OpenAI 互換 API を使って、誤りの構造だけを提案するクライアント。"""

    def __init__(self, base_url=None, model=None, timeout=None):
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL")
        self.model = model or os.environ.get("VLLM_MODEL", "Qwen/Qwen3.6-27B")
        self.max_tokens = int(os.environ.get("VLLM_MAX_TOKENS", "900"))
        self.timeout = (timeout if timeout is not None
                        else int(os.environ.get("VLLM_TIMEOUT", "60")))
        if not self.base_url:
            raise RuntimeError("VLLM_BASE_URL が未設定(GPU サーバ接続後に使用する)")

    def _chat_json(self, prompt):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
        }
        if os.environ.get("VLLM_ENABLE_THINKING") != "1":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"].get("content")
        if not content:
            raise ValueError("LLM response content is empty")
        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError("LLM response must be a JSON object")
        return result

    def propose_linear_transpose(self, problem, a, b, c):
        sign = "+" if b > 0 else "-"
        prompt = (
            f"一次方程式 {a}x {sign} {abs(b)} = {c} の移項で、中学生が"
            "やりがちな誤りを 1 つ提案してください。誤った移項後の右辺の式だけを"
            "次の JSON で返してください: "
            '{"expr": "<右辺の式(例: 20 + 8)>", "reason": "<どんな誤解か一言>"}。'
            f"使ってよい数は {c} と {abs(b)}（符号は変えてよい）だけです。"
            f"正しい式 {c} - {b} は禁止です。$ や LaTeX 記法は使わないでください。"
        )
        return self._chat_json(prompt)

    def propose_arith_product(self, problem, q, r):
        prompt = (
            f"{q} × {r} の計算で、符号や九九の誤解による誤った結果を 1 つ"
            "提案してください。次の JSON だけを返してください: "
            '{"value": <整数>, "reason": "<どんな誤解か一言>"}。'
            f"正解 {q * r} は禁止です。"
        )
        return self._chat_json(prompt)


class FakeMutationProposer:
    """リトライを確実に通す、オフラインテスト用の決定論的提案器。"""

    def __init__(self):
        self.calls = 0
        self.call_count = 0

    def _first_call(self):
        self.calls += 1
        self.call_count += 1
        return self.calls == 1

    def propose_linear_transpose(self, problem, a, b, c):
        if self._first_call():
            return {"expr": f"{c} + y", "reason": "未許可の変数を使う"}
        expr = f"{c} + {abs(b)}"
        # b < 0 では上式が正解になるため、同じ許可定数で確実に誤答にする。
        if c + abs(b) == c - b:
            expr = f"{c} - {abs(b)}"
        return {"expr": expr, "reason": "移項しても符号を正しく変えられない"}

    def propose_arith_product(self, problem, q, r):
        if self._first_call():
            return {"value": "y", "reason": "整数ではない値を返す"}
        gold = q * r
        value = abs(gold) if abs(gold) != gold else gold + 1
        return {"value": value, "reason": "積の符号または九九を取り違える"}
