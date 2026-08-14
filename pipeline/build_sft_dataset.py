# -*- coding: utf-8 -*-
"""Render 済みレコードを VLM の SFT 用 messages JSONL に変換する。"""

import argparse
import collections
import json
import math
import os
import random
import sys


try:
    from .run_zeroshot_eval import build_prompt as _eval_build_prompt
    from .run_zeroshot_eval import _read_records as _eval_read_records
except ImportError:  # ``python pipeline/build_sft_dataset.py`` での実行用
    from run_zeroshot_eval import build_prompt as _eval_build_prompt
    from run_zeroshot_eval import _read_records as _eval_read_records


RELATIVE_COORDS_INSTRUCTION = (
    "bbox は 0-1000 の相対座標 [x0,y0,x1,y1] です"
    "（ページ左上が原点です）。"
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", required=True,
        help="render メタを含む chunk_*.jsonl のディレクトリ",
    )
    parser.add_argument(
        "--images", required=True,
        help="{sample_id}.png を含むディレクトリ",
    )
    parser.add_argument("--out", required=True, help="出力ディレクトリ")
    parser.add_argument("--exclude-eval-seed", type=int)
    parser.add_argument("--exclude-eval-n", type=int)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--relabel-map",
        help="指定時は教師 errors[0].type を type_identifiable に置換する",
    )
    parser.add_argument(
        "--verify-images", action="store_true",
        help="全参照画像の存在を確認し、欠損があれば失敗する",
    )
    args = parser.parse_args(argv)

    if (args.exclude_eval_seed is None) != (args.exclude_eval_n is None):
        parser.error(
            "--exclude-eval-seed と --exclude-eval-n は同時に指定してください")
    if args.exclude_eval_n is not None and args.exclude_eval_n < 0:
        parser.error("--exclude-eval-n は 0 以上で指定してください")
    if not math.isfinite(args.val_frac) or not 0.0 <= args.val_frac <= 1.0:
        parser.error("--val-frac は 0 以上 1 以下で指定してください")
    return args


def build_prompt(record, coords_mode="pixel"):
    """C1 と同じ本文を作り、必要なら座標規約だけ SFT 用に追加する。

    ``pixel`` は eval の ``build_prompt`` と完全に同じ文字列を返す。
    eval の既定契約を変えずに SFT では ``relative`` を指定する。
    """
    prompt = _eval_build_prompt(record)
    if coords_mode == "pixel":
        return prompt
    if coords_mode == "relative":
        return prompt + "\n\n" + RELATIVE_COORDS_INSTRUCTION
    raise ValueError("coords_mode must be 'pixel' or 'relative'")


def _load_records(record_dir):
    paths, records, malformed = _eval_read_records(record_dir)
    if not paths:
        raise ValueError(
            "入力ディレクトリに chunk_*.jsonl がありません: " + record_dir)
    if malformed:
        raise ValueError(f"JSON として読めないレコードが {malformed} 件あります")
    return paths, records


def _load_relabel_map(path):
    mappings = {}
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: JSON が不正です") from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: JSON object ではありません")
            sample_id = value.get("sample_id")
            identifiable = value.get("type_identifiable")
            if (not isinstance(sample_id, str) or not sample_id
                    or not isinstance(identifiable, str)):
                raise ValueError(f"{path}:{line_number}: map 行が不正です")
            if sample_id in mappings:
                raise ValueError(f"relabel map の sample_id 重複: {sample_id}")
            mappings[sample_id] = identifiable
    return mappings


def _apply_relabel_map(records, mappings):
    for record in records:
        errors = record.get("injected_errors")
        if not errors:
            continue
        sample_id = record.get("sample_id")
        if sample_id not in mappings:
            raise ValueError(
                f"{sample_id}: relabel map に誤りレコードがありません")
        if not isinstance(errors, list) or not isinstance(errors[0], dict):
            raise ValueError(f"{sample_id}: injected_errors[0] が不正です")
        errors[0]["type"] = mappings[sample_id]


def _require_nonempty_string(record, key, position):
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"record {position}: {key} が空または文字列ではありません")
    return value


def _page_size(record, sample_id):
    try:
        page = record["render"]["page"]
        width = float(page["w"])
        height = float(page["h"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{sample_id}: render.page.w/h が不正です") from exc
    if (not math.isfinite(width) or not math.isfinite(height)
            or width <= 0 or height <= 0):
        raise ValueError(f"{sample_id}: render.page.w/h が正の有限値ではありません")
    return width, height


def _union_boxes(boxes, sample_id, error_index):
    if not isinstance(boxes, list) or not boxes:
        raise ValueError(
            f"{sample_id}: error {error_index} の boxes が空または不正です")
    parsed = []
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(
                f"{sample_id}: error {error_index} に不正な bbox があります")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in box):
            raise ValueError(
                f"{sample_id}: error {error_index} に不正な bbox があります")
        parsed.append(box)
    return [
        min(box[0] for box in parsed),
        min(box[1] for box in parsed),
        max(box[2] for box in parsed),
        max(box[3] for box in parsed),
    ]


def _relative_bbox(box, width, height):
    dimensions = (width, height, width, height)
    return [
        max(0, min(1000, int(round(value * 1000 / dimension))))
        for value, dimension in zip(box, dimensions)
    ]


def _teacher_value(record):
    sample_id = record["sample_id"]
    width, height = _page_size(record, sample_id)

    injected_errors = record.get("injected_errors")
    if not isinstance(injected_errors, list):
        raise ValueError(f"{sample_id}: injected_errors が配列ではありません")
    spans = record.get("render", {}).get("error_span_boxes_px")
    if not isinstance(spans, list):
        raise ValueError(
            f"{sample_id}: render.error_span_boxes_px が配列ではありません")

    boxes_by_ref = {}
    for position, span in enumerate(spans):
        if not isinstance(span, dict):
            raise ValueError(
                f"{sample_id}: error_span_boxes_px[{position}] が不正です")
        reference = span.get("error_ref", position)
        boxes_by_ref[reference] = _union_boxes(
            span.get("boxes"), sample_id, reference)

    errors = []
    for index, error in enumerate(injected_errors):
        if not isinstance(error, dict):
            raise ValueError(f"{sample_id}: injected_errors[{index}] が不正です")
        if index not in boxes_by_ref:
            raise ValueError(f"{sample_id}: error {index} の bbox がありません")
        errors.append({
            "step_id": error.get("mutation_site"),
            "bbox": _relative_bbox(boxes_by_ref[index], width, height),
            "type": error.get("type"),
        })

    try:
        transcript = record["transcript_gt"]["text"]
        score = record["score_gt"]["awarded"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{sample_id}: transcript_gt/score_gt が不正です") from exc
    if not isinstance(transcript, str):
        raise ValueError(f"{sample_id}: transcript_gt.text が文字列ではありません")

    # この挿入順が assistant の教師 JSON の固定キー順である。
    return {"transcript": transcript, "errors": errors, "score": score}


def _record_operator(record):
    errors = record.get("injected_errors", [])
    if not errors:
        return None
    return errors[0].get("operator") if isinstance(errors[0], dict) else None


def _is_error_free(record):
    flag = record.get("control_flag", {}).get("error_free")
    if isinstance(flag, bool):
        return flag
    return not bool(record.get("injected_errors"))


def _validate_records(records):
    seen_sample_ids = set()
    for position, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"record {position}: JSON object ではありません")
        sample_id = _require_nonempty_string(record, "sample_id", position)
        _require_nonempty_string(record, "pair_id", position)
        if sample_id in seen_sample_ids:
            raise ValueError(f"sample_id が重複しています: {sample_id}")
        seen_sample_ids.add(sample_id)
        teacher = _teacher_value(record)
        if _is_error_free(record) and teacher["errors"]:
            raise ValueError(
                f"{sample_id}: error_free=true ですが injected_errors があります")
        if not _is_error_free(record) and not teacher["errors"]:
            raise ValueError(
                f"{sample_id}: error_free=false ですが injected_errors が空です")


def _eval_selected_ids(records, seed, count):
    # IMPORTANT: run_zeroshot_eval.py main の選択手順と相互に同期すること。
    # 同じ読込順の全レコードを Random(seed).shuffle し、先頭 n 件を取る。
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return {record["sample_id"] for record in shuffled[:count]}


def _split_records(records, val_frac, seed, eval_seed=None, eval_n=None):
    selected_ids = set()
    test_pair_ids = set()
    if eval_seed is not None:
        selected_ids = _eval_selected_ids(records, eval_seed, eval_n)
        test_pair_ids = {
            record["pair_id"] for record in records
            if record["sample_id"] in selected_ids
        }

    remaining_pair_ids = sorted({
        record["pair_id"] for record in records
        if record["pair_id"] not in test_pair_ids
    })
    random.Random(seed).shuffle(remaining_pair_ids)
    val_pair_count = int(round(len(remaining_pair_ids) * val_frac))
    val_pair_ids = set(remaining_pair_ids[:val_pair_count])

    splits = {"train": [], "val": [], "test": []}
    for record in records:
        pair_id = record["pair_id"]
        if pair_id in test_pair_ids:
            splits["test"].append(record)
        elif pair_id in val_pair_ids:
            splits["val"].append(record)
        else:
            splits["train"].append(record)
    return splits, selected_ids, test_pair_ids, val_pair_ids


def _verify_images(records, image_dir):
    missing = []
    for record in records:
        path = os.path.join(image_dir, record["sample_id"] + ".png")
        if not os.path.isfile(path):
            missing.append(path)
    return missing


def _sft_record(record):
    teacher_text = json.dumps(
        _teacher_value(record), ensure_ascii=False, separators=(",", ":"))
    return {
        "id": record["sample_id"],
        "image": "images/" + record["sample_id"] + ".png",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": build_prompt(record, coords_mode="relative"),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": teacher_text}],
            },
        ],
        "meta": {
            "pair_id": record["pair_id"],
            "domain": record.get("problem", {}).get("domain"),
            "operator": _record_operator(record),
            "error_free": _is_error_free(record),
        },
    }


def _atomic_write_jsonl(path, records):
    temporary = path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(
                    _sft_record(record), ensure_ascii=False,
                    separators=(",", ":")) + "\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_text(path, text):
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


def _distribution(records, value_getter, omit_none=False):
    counter = collections.Counter()
    for record in records:
        value = value_getter(record)
        if value is None and omit_none:
            continue
        counter[str(value) if value is not None else "<none>"] += 1
    return dict(sorted(counter.items()))


def _split_stats(records):
    error_free = sum(_is_error_free(record) for record in records)
    errors = len(records) - error_free
    return {
        "records": len(records),
        "pairs": len({record["pair_id"] for record in records}),
        "error_records": errors,
        "control_records": error_free,
        "error_to_control_ratio": (
            errors / error_free if error_free else None),
        "operator_distribution": _distribution(
            records, _record_operator, omit_none=True),
        "domain_distribution": _distribution(
            records, lambda record: record.get("problem", {}).get("domain")),
    }


def _build_stats(records, splits, paths, args, selected_ids, test_pair_ids,
                 val_pair_ids):
    overall = _split_stats(records)
    return {
        "total_records": len(records),
        "total_pairs": overall["pairs"],
        "input_chunks": len(paths),
        "counts": {name: len(values) for name, values in splits.items()},
        "pair_counts": {
            name: len({record["pair_id"] for record in values})
            for name, values in splits.items()
        },
        "error_records": overall["error_records"],
        "control_records": overall["control_records"],
        "error_to_control_ratio": overall["error_to_control_ratio"],
        "operator_distribution": overall["operator_distribution"],
        "domain_distribution": overall["domain_distribution"],
        "splits": {
            name: _split_stats(values) for name, values in splits.items()
        },
        "split_method": {
            "group_key": "pair_id",
            "train_val_seed": args.seed,
            "val_fraction_of_remaining_pairs": args.val_frac,
            "validation_pairs": len(val_pair_ids),
        },
        "coordinates": {
            "convention": "0-1000 relative coordinates",
            "format": "[x0,y0,x1,y1]",
            "origin": "top-left",
            "conversion": "round(pixel * 1000 / page_dimension), clamped to 0..1000",
        },
        "eval_exclusion": {
            "enabled": args.exclude_eval_seed is not None,
            "seed": args.exclude_eval_seed,
            "requested_n": args.exclude_eval_n,
            "selected_samples": len(selected_ids),
            "isolated_pairs": len(test_pair_ids),
            "isolated_records": len(splits["test"]),
            "selected_sample_ids": sorted(selected_ids),
        },
    }


def _ratio_text(stats):
    ratio = stats["error_to_control_ratio"]
    suffix = "N/A" if ratio is None else f"{ratio:.4f}:1"
    return (
        f"{stats['error_records']} / {stats['control_records']} "
        f"({suffix})"
    )


def _dataset_card(stats, args):
    lines = [
        "# SFT Dataset Card",
        "",
        "## 件数とラベル構成",
        "",
        "| split | records | pairs | 誤りあり / 対照 (比率) |",
        "|---|---:|---:|---:|",
    ]
    for name in ("train", "val", "test"):
        values = stats["splits"][name]
        lines.append(
            f"| {name} | {values['records']} | {values['pairs']} | "
            f"{_ratio_text(values)} |")
    lines.extend([
        f"| **total** | **{stats['total_records']}** | "
        f"**{stats['total_pairs']}** | "
        f"**{_ratio_text(stats)}** |",
        "",
        "全体の pair 数: "
        + str(sum(stats["pair_counts"].values()))
        + "（pair_id は split 間で排他的）。",
        "",
        "## オペレータ分布（誤りありレコード）",
        "",
        "| operator | records |",
        "|---|---:|",
    ])
    operators = stats["operator_distribution"]
    if operators:
        lines.extend(f"| {operator} | {count} |"
                     for operator, count in operators.items())
    else:
        lines.append("| N/A | 0 |")

    split_method = stats["split_method"]
    exclusion = stats["eval_exclusion"]
    lines.extend([
        "",
        "## 分割方法",
        "",
        "すべての分割は `pair_id` 単位で行い、同一 pair が複数 split に入らないようにした。",
    ])
    if exclusion["enabled"]:
        lines.append(
            "まず `run_zeroshot_eval.py` と同じ、入力順の全レコードを "
            f"`random.Random({exclusion['seed']}).shuffle` して先頭 "
            f"{exclusion['requested_n']} 件を取る手順を再現した。選択された "
            f"{exclusion['selected_samples']} sample の "
            f"{exclusion['isolated_pairs']} pair に属する全 "
            f"{exclusion['isolated_records']} レコードを `test` に隔離した。")
    else:
        lines.append("eval 集合の除外指定はなく、`test` は空である。")
    lines.append(
        "残りの pair_id を辞書順に並べてから "
        f"`random.Random({split_method['train_val_seed']}).shuffle` し、"
        f"pair 数の {split_method['val_fraction_of_remaining_pairs']:.6g} "
        "（round で整数化）を `val`、残りを `train` とした。")
    lines.extend([
        "",
        "## 座標規約",
        "",
        "教師出力の `bbox` はページ左上を原点とする 0–1000 の相対整数座標 "
        "`[x0,y0,x1,y1]` である。各誤りの複数 pixel bbox の外接矩形を取り、"
        "各軸を `round(pixel * 1000 / page_dimension)` で変換して 0–1000 にクランプした。",
        "画像はコピーせず、各 JSONL から `images/{sample_id}.png` を参照する。",
        "",
        "## 入力契約",
        "",
        "ユーザーテキストは `run_zeroshot_eval.py` の C1 `build_prompt` を再利用し、"
        "SFT 用に相対座標規約だけを追記している。assistant は説明や Markdown を含まない JSON 文字列である。",
        "",
    ])
    return "\n".join(lines)


def main(argv=None):
    args = _parse_args(argv)
    try:
        paths, records = _load_records(args.records)
        if args.relabel_map:
            _apply_relabel_map(
                records, _load_relabel_map(args.relabel_map))
        _validate_records(records)

        if args.verify_images:
            missing = _verify_images(records, args.images)
            if missing:
                print(f"参照画像が {len(missing)} 件不足しています:", file=sys.stderr)
                for path in missing:
                    print(path, file=sys.stderr)
                return 1

        splits, selected_ids, test_pair_ids, val_pair_ids = _split_records(
            records, args.val_frac, args.seed,
            args.exclude_eval_seed, args.exclude_eval_n,
        )
        split_pair_sets = {
            name: {record["pair_id"] for record in values}
            for name, values in splits.items()
        }
        if (split_pair_sets["train"] & split_pair_sets["val"]
                or split_pair_sets["train"] & split_pair_sets["test"]
                or split_pair_sets["val"] & split_pair_sets["test"]):
            raise AssertionError("split 間で pair_id が重複しています")

        os.makedirs(args.out, exist_ok=True)
        for name in ("train", "val", "test"):
            _atomic_write_jsonl(
                os.path.join(args.out, name + ".jsonl"), splits[name])

        stats = _build_stats(
            records, splits, paths, args, selected_ids, test_pair_ids,
            val_pair_ids,
        )
        _atomic_write_text(
            os.path.join(args.out, "stats.json"),
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        )
        _atomic_write_text(
            os.path.join(args.out, "dataset_card.md"),
            _dataset_card(stats, args),
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"records={len(records)} train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])} "
        f"out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
