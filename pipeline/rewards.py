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

SUBTYPE_TO_FAMILY = {
    "概念/移項": "移項",
    "記法/移項符号": "移項",
    "概念/符号・乗算": "符号・乗算",
    "計算/符号": "符号・乗算",
}
FAMILY_TYPES = frozenset(SUBTYPE_TO_FAMILY.values())
_AUTO_TYPE_REWARD_MODE = object()


def lev(a, b):
    """Levenshtein 距離。"""
    n, m = len(a), len(b)
    if n == 0:
        return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ac = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ac == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def cer(ref, hyp):
    """文字誤り率(Levenshtein / len(ref))。"""
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return lev(ref, hyp) / len(ref)


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


def _type_reward(gt, predicted_types, mode):
    if mode == "legacy":
        if gt["type"] in predicted_types:
            return 1.0, False
        if any(value.split("/")[0] == gt["type"].split("/")[0]
               for value in predicted_types if value):
            return 0.5, False
        return 0.0, False

    identifiable = gt.get("type_identifiable")
    if not isinstance(identifiable, str):
        # 未注釈レコードでは従来ラベルを用い、既存データを壊さない。
        return _type_reward(gt, predicted_types, "legacy")
    if gt.get("ambiguous_cross_family") is True:
        return 0.0, True
    if identifiable in FAMILY_TYPES:
        over_claim = any(
            SUBTYPE_TO_FAMILY.get(value) == identifiable
            for value in predicted_types
        )
        if over_claim:
            return 0.0, False
        return (1.0 if identifiable in predicted_types else 0.0), False
    if identifiable in predicted_types:
        return 1.0, False
    family = SUBTYPE_TO_FAMILY.get(identifiable)
    if family is not None and family in predicted_types:
        return 0.5, False
    return 0.0, False


def reward_components(rec, out, type_reward_mode=_AUTO_TYPE_REWARD_MODE):
    automatic_type_mode = type_reward_mode is _AUTO_TYPE_REWARD_MODE
    if (not automatic_type_mode
            and type_reward_mode not in ("legacy", "identifiable")):
        raise ValueError(
            "type_reward_mode must be 'legacy' or 'identifiable'")
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
        # over-correction 判定(スパン局所): 転記の各行のうちサイト文に最も
        # 近い行を取り、その行が「紙に書かれた誤答(mut_site)」より
        # 「修正後の正解(gold_site)」に編集距離で厳密に近い場合のみ罰する。
        # 履歴: (1) 部分文字列判定は短いサイト文("x=1")が他行("3x=12")に
        # 偶然含まれると素通し。(2) 全文 CER 比較は、正当出力のタイポが偶然
        # 変異文字に当たると(1 文字変異では gold と mut が距離 1 のため)
        # 正当出力を誤罰。いずれも敵対的テストが検出→行単位の局所判定へ。
        # 1 文字変異へのタイポは gold/mut と同距離になり罰されない。
        # 罰は 0.8(原則 5 の中核リスク)。
        site = rec["injected_errors"][0]["mutation_site"]
        gold_site = normalize_text(
            {s["step_id"]: s["text"] for s in rec["gold_solution"]}[site])
        mut_site = normalize_text(
            {s["step_id"]: s["text"] for s in rec["mutant_solution"]}[site])
        hyp_lines = [normalize_text(l) for l in out["transcript"].splitlines()]
        hyp_lines = [l for l in hyp_lines if l]
        if hyp_lines and gold_site != mut_site:
            best = min(hyp_lines,
                       key=lambda l: min(lev(l, mut_site), lev(l, gold_site)))
            if lev(best, gold_site) < lev(best, mut_site):
                t_reward = max(0.0, t_reward - 0.8)
    comps["transcript"] = t_reward

    # --- 誤り有無(過剰報告への罰つき)
    preds = out["errors"]
    n_gt = len(rec["injected_errors"])
    if has_err:
        det = (1.0 if preds else 0.0) - 0.25 * max(0, len(preds) - n_gt)
    else:
        det = 1.0 if not preds else 0.0
    comps["detection"] = max(0.0, det)

    # 引数省略時、注釈済みレコードは既存 adversarial test を変更せず
    # identifiable モードで検証できるよう自動切替する。未注釈時は legacy。
    # 明示指定された legacy/identifiable は注釈の有無より優先する。
    first_error = (rec.get("injected_errors") or [{}])[0]
    effective_type_mode = type_reward_mode
    if automatic_type_mode:
        effective_type_mode = (
            "identifiable"
            if isinstance(first_error, dict)
            and "type_identifiable" in first_error
            else "legacy"
        )

    # --- 位置・種別(対照群で予測なしなら規約上 1.0)
    type_excluded = bool(
        has_err and effective_type_mode == "identifiable"
        and isinstance(first_error, dict)
        and first_error.get("ambiguous_cross_family") is True)
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
        typ, type_excluded = _type_reward(gt, types, effective_type_mode)
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
        # F0.5(precision 重視)+種別全列挙(水増し)への明示罰。
        # 一様 F1 では comment_stuffing が正当出力を +0.004 上回る逆転を
        # 敵対的テストが検出したため precision 側を強化
        b2 = 0.25
        f = ((1 + b2) * prec * recall / (b2 * prec + recall)
             if (b2 * prec + recall) else 0.0)
        if len(pr) > 2 * max(1, len(gt_refs)):
            f *= 0.2
        comps["comment"] = f

    if type_excluded:
        active_weight = sum(
            weight for key, weight in WEIGHTS.items() if key != "type")
        comps["total"] = sum(
            WEIGHTS[key] * comps[key] for key in WEIGHTS if key != "type"
        ) / active_weight
    else:
        comps["total"] = sum(WEIGHTS[k] * comps[k] for k in WEIGHTS)
    return comps
