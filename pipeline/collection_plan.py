# -*- coding: utf-8 -*-
"""統合収集プログラムの割付表生成(パネル条件 7/設計書 v2.1 §3 M6)。

総計 1,350 枚:
- 書き写しブロック 900 枚 = 書き手 60 名 × 15 枚(内容 item-disjoint)
    - うち書き手 20 名(G01–G20)= 字形供給用(字形バンク/レンダラ開発)
    - うち書き手 40 名(E01–E40)= 評価専用(sim2real 定量測定)
- 公平性ブロック 330 枚 = 書き手 30 名(F01–F30、評価専用)× 共通 11 内容
    - 同一内容×書き手間比較(点数一致 98% vs 95% を ±2pt で分離)
- 自力解答ブロック 120 枚 = 書き手 30 名(G01–G20+C01–C10)× 4 枚
    - 時間制限 2 枚+訂正可 2 枚。定性インベントリ(訂正行動)専用。
      統計的合否判定には使わない。

書き手の用途間分離(パネル確定所見):
評価専用(E, F)の書き手・ページは、字形供給(G)・校正(C)と完全分離。
"""
import csv
import os
import random

OUT_CSV = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "docs", "collection_plan.csv")


def build_rows(seed=20260804):
    rng = random.Random(seed)
    rows = []

    # 書き写しブロック 900(item-disjoint 内容 U0001..U0900)
    contents = [f"U{i:04d}" for i in range(1, 901)]
    rng.shuffle(contents)
    writers = [f"G{i:02d}" for i in range(1, 21)] + \
        [f"E{i:02d}" for i in range(1, 41)]
    ci = 0
    for w in writers:
        purpose = "字形供給" if w.startswith("G") else "評価専用"
        for k in range(15):
            rows.append({
                "page_id": f"COPY-{w}-{k+1:02d}",
                "block": "書き写し", "writer_id": w,
                "content_id": contents[ci], "condition": "copy",
                "purpose": purpose,
            })
            ci += 1

    # 公平性ブロック 330(共通内容 FC01..FC11 × F01..F30 全交差)
    for w in (f"F{i:02d}" for i in range(1, 31)):
        for c in range(1, 12):
            rows.append({
                "page_id": f"FAIR-{w}-FC{c:02d}",
                "block": "公平性", "writer_id": w,
                "content_id": f"FC{c:02d}", "condition": "copy",
                "purpose": "評価専用",
            })

    # 自力解答ブロック 120(G01–G20 + C01–C10 × 4 枚)
    solvers = [f"G{i:02d}" for i in range(1, 21)] + \
        [f"C{i:02d}" for i in range(1, 11)]
    for w in solvers:
        for k, cond in enumerate(
                ["timed_solve", "timed_solve",
                 "corrected_solve", "corrected_solve"]):
            rows.append({
                "page_id": f"SOLV-{w}-{k+1}",
                "block": "自力解答", "writer_id": w,
                "content_id": f"S{rng.randint(1, 60):03d}",
                "condition": cond,
                "purpose": "定性インベントリ(レンダラ校正)",
            })
    return rows


def main():
    rows = build_rows()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    blocks = {}
    eval_writers, noneval_writers = set(), set()
    for r in rows:
        blocks[r["block"]] = blocks.get(r["block"], 0) + 1
        if r["purpose"] == "評価専用":
            eval_writers.add(r["writer_id"])
        else:
            noneval_writers.add(r["writer_id"])
    overlap = eval_writers & noneval_writers
    print(f"[plan] total={len(rows)} blocks={blocks}")
    print(f"[plan] writers: 評価専用={len(eval_writers)} "
          f"非評価={len(noneval_writers)} 重複={sorted(overlap)}")
    assert len(rows) == 1350 and not overlap, "割付の不変条件違反"
    print(f"[plan] -> {OUT_CSV}")


if __name__ == "__main__":
    main()
