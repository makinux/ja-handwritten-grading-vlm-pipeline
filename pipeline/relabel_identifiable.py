# -*- coding: utf-8 -*-
"""誤り GT を観測から識別可能な最も細かい粒度へ再ラベルする。"""

import argparse
import collections
import copy
import glob
import json
import os
import re
import sys


TRANSPOSE_LLM = "op-llm-transpose-misconception"
TRANSPOSE_SIGN = "op-transpose-sign"
PRODUCT_LLM = "op-llm-product-misconception"
MULT_SIGN_DROP = "op-mult-sign-drop"
ARITH_SLIP = "op-arith-slip"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", required=True,
        help="chunk_*.jsonl を含む入力ディレクトリ",
    )
    parser.add_argument("--out-map", required=True, help="再ラベル map JSONL")
    parser.add_argument("--out-stats", required=True, help="集計 JSON")
    parser.add_argument(
        "--annotate-records",
        help="injected_errors[0] に識別可能ラベルを追記する出力ディレクトリ",
    )
    return parser.parse_args(argv)


def _atomic_write_text(path, text):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    temporary = path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_chunks(record_dir):
    paths = sorted(glob.glob(os.path.join(record_dir, "chunk_*.jsonl")))
    if not paths:
        raise ValueError(
            "入力ディレクトリに chunk_*.jsonl がありません: " + record_dir)
    chunks = []
    seen = set()
    for path in paths:
        records = []
        with open(path, "r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: JSON が不正です") from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{path}:{line_number}: JSON object ではありません")
                sample_id = record.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(
                        f"{path}:{line_number}: sample_id が不正です")
                if sample_id in seen:
                    raise ValueError(f"sample_id が重複しています: {sample_id}")
                seen.add(sample_id)
                records.append(record)
        chunks.append((path, records))
    return chunks


def _site_latex(record, solution_key, mutation_site):
    steps = record.get(solution_key)
    if not isinstance(steps, list):
        raise ValueError(
            f"{record['sample_id']}: {solution_key} が配列ではありません")
    for step in steps:
        if isinstance(step, dict) and step.get("step_id") == mutation_site:
            latex = step.get("latex")
            if not isinstance(latex, str):
                raise ValueError(
                    f"{record['sample_id']}: {solution_key}/{mutation_site} "
                    "の latex が不正です")
            return latex
    raise ValueError(
        f"{record['sample_id']}: {solution_key} に {mutation_site} がありません")


def _flip_last_rhs_binary_sign(latex):
    """最後の ``=`` より右にある最後の二項 +/- だけを反転する。"""
    equals = latex.rfind("=")
    if equals < 0:
        raise ValueError("latex に = がありません")
    rhs = latex[equals + 1:]
    candidates = []
    for index, char in enumerate(rhs):
        if char not in "+-":
            continue
        previous = index - 1
        while previous >= 0 and rhs[previous].isspace():
            previous -= 1
        # 先頭符号、括弧直後、別演算子直後の符号は単項演算子である。
        if previous < 0 or rhs[previous] in "=+-*/×÷(":
            continue
        candidates.append(index)
    if not candidates:
        raise ValueError("latex の右辺に二項 +/- がありません")
    index = candidates[-1]
    flipped = "-" if rhs[index] == "+" else "+"
    return latex[:equals + 1] + rhs[:index] + flipped + rhs[index + 1:]


def _without_space(value):
    return "".join(value.split())


def _integers(latex):
    return [int(value) for value in re.findall(r"[+-]?\d+", latex)]


def _domain(record):
    value = record.get("domain")
    if value is not None:
        return value
    problem = record.get("problem", {})
    return problem.get("domain") if isinstance(problem, dict) else None


def _relabel_record(record):
    errors = record.get("injected_errors")
    if not isinstance(errors, list):
        raise ValueError(
            f"{record['sample_id']}: injected_errors が配列ではありません")
    if not errors:
        return None, None
    error = errors[0]
    if not isinstance(error, dict):
        raise ValueError(
            f"{record['sample_id']}: injected_errors[0] が不正です")
    legacy = error.get("type")
    operator = error.get("operator")
    site = error.get("mutation_site")
    if not isinstance(legacy, str) or not isinstance(operator, str):
        raise ValueError(
            f"{record['sample_id']}: type/operator が不正です")

    identifiable = legacy
    subtype_retained = False
    unidentifiable = False
    pair_bucket = None

    if operator == TRANSPOSE_LLM:
        gold = _site_latex(record, "gold_solution", site)
        mutant = _site_latex(record, "mutant_solution", site)
        try:
            mechanical = _flip_last_rhs_binary_sign(gold)
        except ValueError as exc:
            raise ValueError(f"{record['sample_id']}: {exc}") from exc
        if _without_space(mechanical) == _without_space(mutant):
            identifiable = "移項"
            unidentifiable = True
            pair_bucket = "equivalent"
        else:
            identifiable = legacy
            subtype_retained = True
            pair_bucket = "subtype_retained"
    elif operator == TRANSPOSE_SIGN:
        identifiable = "移項"
        unidentifiable = True
    elif operator == PRODUCT_LLM:
        gold = _site_latex(record, "gold_solution", site)
        mutant = _site_latex(record, "mutant_solution", site)
        gold_numbers = _integers(gold)
        mutant_numbers = _integers(mutant)
        if len(gold_numbers) < 2 or not mutant_numbers:
            raise ValueError(
                f"{record['sample_id']}: 積の q/r/mv を抽出できません")
        q, r = gold_numbers[:2]
        mv = mutant_numbers[-1]
        product = q * r
        if product < 0 and mv == abs(product):
            identifiable = "符号・乗算"
            unidentifiable = True
            pair_bucket = "negative_equivalent"
        elif product > 0:
            identifiable = "符号・乗算"
            unidentifiable = True
            pair_bucket = "positive_false_identifiable"
        elif product < 0:
            identifiable = legacy
            subtype_retained = True
            pair_bucket = "subtype_retained"
        else:
            raise ValueError(
                f"{record['sample_id']}: q*r=0 は再ラベル規則の範囲外です")
        product_delta_ambiguous = abs(mv - product) in (1, 2)
    elif operator == MULT_SIGN_DROP:
        identifiable = "符号・乗算"
        unidentifiable = True
        product_delta_ambiguous = False
    else:
        product_delta_ambiguous = False

    ambiguous = bool(
        (operator == PRODUCT_LLM and product_delta_ambiguous)
        or (operator == ARITH_SLIP and site == "s2"
            and _domain(record) == "正負の数")
    )
    mapping = {
        "sample_id": record["sample_id"],
        "type_identifiable": identifiable,
        "subtype_retained": subtype_retained,
        "ambiguous_cross_family": ambiguous,
        "type_legacy": legacy,
        "operator": operator,
    }
    details = {
        "unidentifiable": unidentifiable,
        "pair_bucket": pair_bucket,
        "ambiguous_product": bool(
            operator == PRODUCT_LLM and product_delta_ambiguous),
        "ambiguous_arith": bool(
            operator == ARITH_SLIP and site == "s2"
            and _domain(record) == "正負の数"),
    }
    return mapping, details


def _build_outputs(chunks):
    mappings = []
    distribution = collections.Counter()
    transpose = collections.Counter()
    product = collections.Counter()
    ambiguous_product = 0
    ambiguous_arith = 0
    subtype_count = 0
    layer0_violations = 0
    total_records = 0

    for _path, records in chunks:
        total_records += len(records)
        for record in records:
            mapping, details = _relabel_record(record)
            if mapping is None:
                continue
            mappings.append(mapping)
            distribution[mapping["type_identifiable"]] += 1
            subtype_count += int(mapping["subtype_retained"])
            operator = mapping["operator"]
            if operator == TRANSPOSE_LLM:
                transpose[details["pair_bucket"]] += 1
            elif operator == PRODUCT_LLM:
                product[details["pair_bucket"]] += 1
            ambiguous_product += int(details["ambiguous_product"])
            ambiguous_arith += int(details["ambiguous_arith"])
            if details["unidentifiable"] and mapping["subtype_retained"]:
                layer0_violations += 1

    error_records = len(mappings)
    stats = {
        "input_chunks": len(chunks),
        "records_total": total_records,
        "error_records": error_records,
        "new_gt_distribution": dict(sorted(distribution.items())),
        "transpose_pair": {
            "total": sum(transpose.values()),
            "equivalent": transpose["equivalent"],
            "subtype_retained": transpose["subtype_retained"],
        },
        "product_pair": {
            "total": sum(product.values()),
            "negative_equivalent": product["negative_equivalent"],
            "positive_false_identifiable": product[
                "positive_false_identifiable"],
            "family_total": (
                product["negative_equivalent"]
                + product["positive_false_identifiable"]),
            "subtype_retained": product["subtype_retained"],
        },
        "subtype_coverage": {
            "count": subtype_count,
            "total": error_records,
            "rate": subtype_count / error_records if error_records else None,
        },
        "ambiguous_cross_family": {
            "total": ambiguous_product + ambiguous_arith,
            "llm_product_delta_1_or_2": ambiguous_product,
            "arith_slip_s2_signed_numbers": ambiguous_arith,
        },
        "layer0_data_health": {
            "unidentifiable_with_subtype_gt": layer0_violations,
            "passed": layer0_violations == 0,
        },
    }
    assert layer0_violations == 0, (
        "層 0 データ健全性検査失敗: 識別不能標本に下位区分 GT が付いています")
    return mappings, stats


def _write_annotations(chunks, output_dir, mappings):
    source_dirs = {os.path.abspath(os.path.dirname(path)) for path, _ in chunks}
    if os.path.abspath(output_dir) in source_dirs:
        raise ValueError("--annotate-records は入力ディレクトリと別にしてください")
    os.makedirs(output_dir, exist_ok=True)
    by_id = {value["sample_id"]: value for value in mappings}
    for input_path, records in chunks:
        annotated = []
        for source in records:
            record = copy.deepcopy(source)
            mapping = by_id.get(record["sample_id"])
            if mapping is not None:
                error = record["injected_errors"][0]
                error["type_identifiable"] = mapping["type_identifiable"]
                error["ambiguous_cross_family"] = mapping[
                    "ambiguous_cross_family"]
            annotated.append(record)
        text = "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in annotated
        )
        _atomic_write_text(
            os.path.join(output_dir, os.path.basename(input_path)), text)


def main(argv=None):
    args = _parse_args(argv)
    try:
        chunks = _read_chunks(args.records)
        mappings, stats = _build_outputs(chunks)
        map_text = "".join(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
            for value in mappings
        )
        _atomic_write_text(args.out_map, map_text)
        _atomic_write_text(
            args.out_stats,
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        )
        if args.annotate_records:
            _write_annotations(chunks, args.annotate_records, mappings)
    except (OSError, ValueError, TypeError, KeyError, AssertionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"records={stats['records_total']} errors={stats['error_records']} "
        f"subtypes={stats['subtype_coverage']['count']} "
        f"ambiguous={stats['ambiguous_cross_family']['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
