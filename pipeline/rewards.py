# -*- coding: utf-8 -*-
"""報酬関数(設計書 v2.1 §3 M4 の 7 項目)。

モデル出力の想定形式(ブートストラップでは機械検証可能な構造のみ):
{"transcript": str,
 "errors": [{"step_id": str, "bbox": [x0,y0,x1,y1], "type": str}],
 "score": int,
 "comment_refs": [誤り種別文字列, ...]}   # コメント接地の機械検証プロキシ
"""
from gen_core import normalize_text

WEIGHTS = {
    "transcript": 0.25,
    "detection": 0.15,
    "location": 0.15,
    "type": 0.10,
    "score": 0.20,
    "comment": 0.05,
    "format": 0.10,
}


def cer(ref, hyp):
    """文字誤り率(Levenshtein / len(ref))。"""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        rc = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if rc == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / n


def iou(a, b):
    if not a or not b:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def gt_error_box(rec):
    es = rec.get("render", {}).get("error_span_boxes_px", [])
    if not es or not es[0]["boxes"]:
        return None
    return _union(es[0]["boxes"])


def reward_components(rec, out):
    comps = {k: 0.0 for k in WEIGHTS}
    ok = (isinstance(out, dict) and isinstance(out.get("transcript"), str)
          and isinstance(out.get("errors"), list) and "score" in out)
    comps["format"] = 1.0 if ok else 0.0
    if not ok:
        comps["total"] = 0.0
        return comps

    has_err = not rec["control_flag"]["error_free"]

    # --- 忠実転記(正規化 CER 連続報酬+over-correction 罰)
    ref = normalize_text(rec["transcript_gt"]["text"])
    hyp = normalize_text(out["transcript"])
    t_reward = max(0.0, 1.0 - cer(ref, hyp))
    if has_err:
        site = rec["injected_errors"][0]["mutation_site"]
        gold_site = normalize_text(
            {s["step_id"]: s["text"] for s in rec["gold_solution"]}[site])
        mut_site = normalize_text(
            {s["step_id"]: s["text"] for s in rec["mutant_solution"]}[site])
        if gold_site != mut_site and gold_site in hyp and mut_site not in hyp:
            t_reward = max(0.0, t_reward - 0.5)  # 誤りを勝手に「修正」した
    comps["transcript"] = t_reward

    # --- 誤り有無(過剰報告への罰つき)
    preds = out["errors"]
    n_gt = len(rec["injected_errors"])
    if has_err:
        det = (1.0 if preds else 0.0) - 0.25 * max(0, len(preds) - n_gt)
    else:
        det = 1.0 if not preds else 0.0
    comps["detection"] = max(0.0, det)

    # --- 位置・種別(対照群で予測なしなら規約上 1.0)
    if not has_err:
        loc = typ = 1.0 if not preds else 0.0
    elif not preds:
        loc = typ = 0.0
    else:
        gt = rec["injected_errors"][0]
        gbox = gt_error_box(rec)
        best_iou = max((iou(p.get("bbox"), gbox) for p in preds), default=0.0)
        step_hit = any(p.get("step_id") == gt["mutation_site"] for p in preds)
        loc = 0.5 * (1.0 if best_iou >= 0.5 else best_iou) + \
            0.5 * (1.0 if step_hit else 0.0)
        types = [p.get("type", "") for p in preds]
        if gt["type"] in types:
            typ = 1.0
        elif any(t.split("/")[0] == gt["type"].split("/")[0]
                 for t in types if t):
            typ = 0.5
        else:
            typ = 0.0
    comps["location"], comps["type"] = loc, typ

    # --- 点数(完全一致+隣接部分報酬)
    gt_score = rec["score_gt"]["awarded"]
    ps = out.get("score")
    if ps == gt_score:
        comps["score"] = 1.0
    elif isinstance(ps, int) and abs(ps - gt_score) <= 1:
        comps["score"] = 0.5

    # --- コメント接地(言及種別の F1 =機械検証プロキシ)
    gt_refs = {e["type"] for e in rec["injected_errors"]}
    pr = set(out.get("comment_refs", []))
    if not gt_refs and not pr:
        comps["comment"] = 1.0
    elif gt_refs and pr:
        tp = len(gt_refs & pr)
        prec = tp / len(pr)
        recall = tp / len(gt_refs)
        comps["comment"] = (2 * prec * recall / (prec + recall)
                            if prec + recall else 0.0)

    comps["total"] = sum(WEIGHTS[k] * comps[k] for k in WEIGHTS)
    return comps
