# -*- coding: utf-8 -*-
"""逐語化(verbalization)インターフェース。

設計書 v2.1 §3 M1: LLM の役割は「正しさが外部保証された骨格の日本語化」に
限定される。本モジュールはその差し替え点を定義する。

- TemplateVerbalizer : 現行のテンプレート逐語化(gen_core が生成した text を
  そのまま用いる)。ブートストラップの既定。
- LLMVerbalizer      : vLLM の OpenAI 互換 API(Qwen3.6 系)を呼ぶクライアント。
  環境変数 VLLM_BASE_URL(例 http://gpu-host:8000/v1)と VLLM_MODEL を要する。
  ※ GPU サーバ未接続のため実機未検証。接続後は必ず g1b_check_texts の
  ゲートを通してから採用すること(原則 2: LLM 出力は検証ゲートを通過する
  まで データではない)。
- フォルト注入(inject_*): G1 ゲートの実効性実測(gate_efficacy_test.py)用。
"""
import json
import os
import random
import urllib.request

from gen_core import NUM_RE


class TemplateVerbalizer:
    """テンプレート逐語化(恒等変換)。"""

    name = "template"

    def verbalize(self, problem, steps):
        return [s["text"] for s in steps]


class FakeVerbalizer:
    """ネットワークを使わない、G1b 通過確認用の決定論的逐語化器。"""

    name = "fake"

    def verbalize(self, problem, steps):
        texts = []
        for step in steps:
            token = step["numbers"][0] if step.get("numbers") else None
            prefix = f"つまり {token} を使うと、" if token is not None else "つまり "
            texts.append(prefix + step["text"])
        return texts


class LLMVerbalizer:
    """vLLM(OpenAI 互換)クライアント。

    プロンプトはプログラムのステップ構造を渡し、数値を一切変えずに
    自然な日本語へ言い換えるよう指示する。出力は g1b_check_texts で検証する。
    thinking は既定で無効化し、VLLM_ENABLE_THINKING=1 の場合のみサーバ既定に戻す。

    環境変数: VLLM_BASE_URL / VLLM_MODEL / VLLM_MAX_TOKENS /
    VLLM_ENABLE_THINKING / VLLM_TIMEOUT
    """

    name = "llm"

    def __init__(self, base_url=None, model=None, timeout=None):
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL")
        self.model = model or os.environ.get("VLLM_MODEL", "Qwen/Qwen3.6-27B")
        self.max_tokens = int(os.environ.get("VLLM_MAX_TOKENS", "900"))
        self.timeout = (timeout if timeout is not None
                        else int(os.environ.get("VLLM_TIMEOUT", "60")))
        if not self.base_url:
            raise RuntimeError(
                "VLLM_BASE_URL が未設定(GPU サーバ接続後に使用する)")

    def verbalize(self, problem, steps):
        prompt = self._build_prompt(problem, steps)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
        }
        if os.environ.get("VLLM_ENABLE_THINKING") != "1":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.loads(r.read().decode("utf-8"))
        content = body["choices"][0]["message"].get("content")
        if content is None or content == "":
            raise ValueError(
                "LLM 応答の content が空(thinking による文脈枯渇の可能性。"
                "VLLM_MAX_TOKENS/サーバの -c を確認)")
        texts = json.loads(content)["steps"]
        if len(texts) != len(steps):
            raise ValueError("LLM 逐語化のステップ数が不一致")
        return texts

    @staticmethod
    def _build_prompt(problem, steps):
        lines = [
            "次の数学の解答ステップを、中学生が書く自然な日本語の答案文にしてください。",
            "各ステップには、与えたステップの数式を一字一句そのまま含め、",
            "その前後に自然な説明を書き添えてください。式の数値・記号を変えないでください。",
            "出力は JSON {\"steps\": [各ステップの文字列]} のみ。",
            f"問題: {problem['problem_text']}",
            "ステップ:",
        ]
        for s in steps:
            lines.append(f"- {s['step_id']}: {s['text']}")
        return "\n".join(lines)


# ------------------------------------------------ フォルト注入(G1 実効性用)


def inject_digit_change(texts, rng):
    """数値ハルシネーション: どこか 1 桁を別の数字に置換(検出されるべき)。"""
    out = list(texts)
    idxs = [i for i, t in enumerate(out) if NUM_RE.search(t)]
    i = rng.choice(idxs)
    t = out[i]
    pos = [j for j, ch in enumerate(t) if ch.isdigit()]
    j = rng.choice(pos)
    new = str((int(t[j]) + rng.randint(1, 8)) % 10)
    out[i] = t[:j] + new + t[j + 1:]
    return out, i


def inject_minus_drop(texts, rng):
    """符号落とし: 負数の '-' を 1 箇所削除(検出されるべき)。
    負数がなければ digit_change に切替。"""
    out = list(texts)
    idxs = [i for i, t in enumerate(out)
            if any(m.group(0).startswith("-") for m in NUM_RE.finditer(t))]
    if not idxs:
        return inject_digit_change(texts, rng)
    i = rng.choice(idxs)
    t = out[i]
    for m in NUM_RE.finditer(t):
        if m.group(0).startswith("-"):
            out[i] = t[:m.start()] + t[m.start() + 1:]
            break
    return out, i


def inject_benign_paraphrase(texts, rng):
    """良性の言い換え(数値不変): G1 は通すべき(誤棄却の測定用)。"""
    out = list(texts)
    i = rng.randrange(len(out))
    prefix = rng.choice(["よって ", "したがって ", "ゆえに "])
    out[i] = prefix + out[i]
    return out, i


def inject_verbose_faithful(texts, rng):
    """既知数値を説明で再言及する良性の逐語化: G1 は通すべき。"""
    out = list(texts)
    known = [token for text in texts for token in NUM_RE.findall(text)]
    token = rng.choice(known)
    if rng.choice([False, True]):
        token = token[1:] if token.startswith("-") else "-" + token
    i = rng.randrange(len(out))
    prefix = rng.choice([
        f"両辺から {token} を引くと、",
        f"両辺を {token} で割って、",
        f"左辺の {token} を消すために、",
    ])
    out[i] = prefix + out[i]
    return out, i
