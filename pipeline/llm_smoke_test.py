# -*- coding: utf-8 -*-
"""EVO-X2(llama-server)接続スモークテスト。

使い方:
  VLLM_BASE_URL=http://<host>:8080/v1 python pipeline/llm_smoke_test.py [--n 10]
  python pipeline/llm_smoke_test.py --dry   # サーバなしで経路のみ確認

実 LLM 逐語化に対して G1b(逐語化照合)を回し、通過率を表示する。
これが「テンプレートでは常に 100% だった G1 が実測値になる」最初の一歩。
"""
import argparse
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_core import DOMAINS, g1b_check_texts
from verbalizer import LLMVerbalizer, TemplateVerbalizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--url", default=None,
                    help="OpenAI 互換 API のベース URL(既定: VLLM_BASE_URL)")
    ap.add_argument("--dry", action="store_true",
                    help="サーバなしでテンプレート逐語化を同経路で流す")
    args = ap.parse_args()

    if args.dry:
        vb = TemplateVerbalizer()
        print("[smoke] dry-run(テンプレート逐語化)")
    else:
        try:
            vb = LLMVerbalizer(base_url=args.url)
        except RuntimeError as e:
            print(f"[smoke] 接続設定エラー: {e}")
            return 2
        print(f"[smoke] LLM 逐語化: {vb.base_url} / {vb.model}")

    rng = random.Random(20260804)
    n_pass = 0
    for i in range(args.n):
        domain = rng.choice(list(DOMAINS))
        gen_fn, exec_fn, _ = DOMAINS[domain]
        problem = gen_fn(rng)
        steps, _ = exec_fn(problem["params"], None)
        try:
            texts = vb.verbalize(problem, steps)
        except Exception as e:
            print(f"  [{i+1}] {domain}: 呼び出し失敗 — {type(e).__name__}: {e}")
            continue
        fails = g1b_check_texts(steps, texts)
        ok = not fails
        n_pass += ok
        mark = "PASS" if ok else f"FAIL({','.join(fails)})"
        print(f"  [{i+1}] {domain}: {mark}")
        if not ok:
            for s, t in zip(steps, texts):
                if s["step_id"] in fails:
                    print(f"        期待 {s['numbers']} / 出力: {t}")

    print(f"[smoke] G1b 通過率: {n_pass}/{args.n}")
    return 0 if n_pass > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
