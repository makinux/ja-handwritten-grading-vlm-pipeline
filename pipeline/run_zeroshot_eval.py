# -*- coding: utf-8 -*-
"""OpenAI 互換 VLM API を用いた C1 ゼロショット評価ランナー。"""
import argparse
import base64
import glob
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rewards import cer, iou, lev, normalize_text


SYSTEM_PROMPT = """あなたは日本語の手書き数学答案を厳密に採点する採点者です。
画像に実際に書かれている内容を忠実に転記し、模範解答で答案を上書きしないでください。
問題文、step_id 付き模範解答、採点基準と答案画像を照合してください。
誤りごとに、対応する模範解答の step_id、答案画像上の誤り箇所の bbox、誤り種別を返してください。
bbox はページ左上を原点とするピクセル座標 [x0,y0,x1,y1] です。
誤りがなければ errors は空配列にしてください。出力は説明や Markdown を含まない JSON オブジェクトのみとしてください。"""

OUTPUT_INSTRUCTION = """出力 JSON の形式:
{"transcript": str, "errors": [{"step_id": str|null, "bbox": [x0,y0,x1,y1], "type": str}], "score": int}"""


# Keep this text synchronized with build_sft_dataset.py's
# RELATIVE_COORDS_INSTRUCTION. Importing that module here would create a
# circular import because it imports build_prompt from this module.
RELATIVE_COORDS_INSTRUCTION = 'bbox は 0-1000 の相対座標 [x0,y0,x1,y1] です（ページ左上が原点です）。'


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True,
                        help="render メタを含む chunk_*.jsonl のディレクトリ")
    parser.add_argument("--images", required=True,
                        help="{sample_id}.png を含むディレクトリ")
    parser.add_argument("--url", default="http://127.0.0.1:8081/v1",
                        help="OpenAI 互換 API のベース URL")
    parser.add_argument("--model", default="qwen3-vl-8b")
    parser.add_argument("--coords", choices=("pixel", "relative"),
                        default="pixel")
    parser.add_argument("--system", choices=("on", "off"), default="on")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mock", action="store_true",
                        help="API を呼ばず GT と同じ予測を合成する")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    if args.n < 0:
        parser.error("--n は 0 以上で指定してください")
    if args.max_tokens < 1:
        parser.error("--max-tokens は 1 以上で指定してください")
    if args.timeout <= 0:
        parser.error("--timeout は 0 より大きくしてください")
    return args


def _atomic_write_text(path, text):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.replace(temporary, path)


def _read_records(record_dir):
    paths = sorted(glob.glob(os.path.join(record_dir, "chunk_*.jsonl")))
    records = []
    malformed = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise TypeError("record is not an object")
                    records.append(value)
                except (json.JSONDecodeError, TypeError):
                    malformed += 1
    return paths, records, malformed


def _page_size(record):
    try:
        page = record["render"]["page"]
        width = float(page["w"])
        height = float(page["h"])
        if not math.isfinite(width) or not math.isfinite(height):
            return None
        if width <= 0 or height <= 0:
            return None
        return width, height
    except (KeyError, TypeError, ValueError):
        return None


def _eligible(record, image_dir):
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        return None, "sample_id_missing"
    if not isinstance(record.get("render"), dict) or _page_size(record) is None:
        return None, "render_metadata_missing"
    image_path = os.path.join(image_dir, sample_id + ".png")
    if not os.path.isfile(image_path):
        return None, "image_missing"
    return image_path, None


def _solution_text(record):
    lines = []
    for step in record.get("gold_solution", []):
        if isinstance(step, dict):
            lines.append(f"- {step.get('step_id')}: {step.get('text', '')}")
    return "\n".join(lines)


def build_prompt(record, coords_mode="pixel"):
    """C1 のテキスト入力を構築する。"""
    prompt = "\n\n".join([
        "問題文:\n" + str(record.get("problem", {}).get("text_ja", "")),
        "模範解答:\n" + _solution_text(record),
        "採点基準:\n" + str(record.get("rubric", {}).get("text_ja", "")),
        OUTPUT_INSTRUCTION,
    ])
    if coords_mode == "pixel":
        return prompt
    if coords_mode == "relative":
        return prompt + "\n\n" + RELATIVE_COORDS_INSTRUCTION
    raise ValueError("coords_mode must be 'pixel' or 'relative'")


def _image_data_uri(image_path):
    with open(image_path, "rb") as stream:
        encoded = base64.b64encode(stream.read()).decode("ascii")
    return "data:image/png;base64," + encoded


def build_messages(record, image_path, coords_mode="pixel", system="on"):
    """OpenAI Chat Completions 用のマルチモーダル messages を返す。"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_prompt(record, coords_mode)},
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_uri(image_path)},
                },
            ],
        },
    ]
    if system == "on":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return messages


def _endpoint(base_url):
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _api_request(args, record, image_path):
    payload = {
        "model": args.model,
        "messages": build_messages(record, image_path, args.coords, args.system),
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {
            "enable_thinking": os.environ.get("VLLM_ENABLE_THINKING") == "1",
        },
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(
        _endpoint(args.url), data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return response.read().decode("utf-8", errors="replace"), None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return body, f"HTTPError {exc.code}: {exc.reason}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return "", f"{type(exc).__name__}: {exc}"


def _content_from_response(raw_response):
    envelope = json.loads(raw_response)
    content = envelope["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        if parts:
            return "".join(parts)
    raise TypeError("message.content is not text")


def _parse_json_content(content):
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise TypeError("content is not a string or object")
    text = content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as first_error:
        if text.startswith("```") and text.endswith("```"):
            inner = text[3:-3].strip()
            if inner.lower().startswith("json"):
                inner = inner[4:].lstrip()
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                pass
        start = text.find("{")
        if start >= 0:
            try:
                value, _end = json.JSONDecoder().raw_decode(text[start:])
                return value
            except json.JSONDecodeError:
                pass
        raise first_error


def _sanitize_bbox(value, page_size, coords_mode="pixel"):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           for v in value):
        return None
    coords = [float(v) for v in value]
    if not all(math.isfinite(v) for v in coords):
        return None
    width, height = ((1000.0, 1000.0) if coords_mode == "relative"
                     else page_size)
    coords[0] = min(width, max(0.0, coords[0]))
    coords[2] = min(width, max(0.0, coords[2]))
    coords[1] = min(height, max(0.0, coords[1]))
    coords[3] = min(height, max(0.0, coords[3]))
    if coords[2] <= coords[0] or coords[3] <= coords[1]:
        return None
    return [int(v) if v.is_integer() else v for v in coords]


def _as_int(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _sanitize_output(value, record, coords_mode="pixel"):
    if not isinstance(value, dict):
        raise TypeError("output JSON is not an object")
    transcript = value.get("transcript")
    errors = value.get("errors")
    if not isinstance(transcript, str):
        raise TypeError("transcript is not a string")
    if not isinstance(errors, list):
        raise TypeError("errors is not an array")
    page_size = _page_size(record)
    sanitized_errors = []
    for error in errors:
        if not isinstance(error, dict):
            error = {}
        step_id = error.get("step_id")
        if step_id is not None and not isinstance(step_id, str):
            step_id = str(step_id)
        error_type = error.get("type", "")
        if not isinstance(error_type, str):
            error_type = str(error_type)
        sanitized_errors.append({
            "step_id": step_id,
            "bbox": _sanitize_bbox(error.get("bbox"), page_size, coords_mode),
            "type": error_type,
        })
    return {
        "transcript": transcript,
        "errors": sanitized_errors,
        "score": _as_int(value.get("score")),
    }


def _union_boxes(boxes):
    valid = [box for box in boxes
             if isinstance(box, (list, tuple)) and len(box) == 4]
    if not valid:
        return None
    return [
        min(box[0] for box in valid), min(box[1] for box in valid),
        max(box[2] for box in valid), max(box[3] for box in valid),
    ]


def _relative_bbox(box, width, height):
    dimensions = (width, height, width, height)
    return [
        max(0, min(1000, int(round(value * 1000 / dimension))))
        for value, dimension in zip(box, dimensions)
    ]


def _gt_error_boxes(record, coords_mode="pixel"):
    spans = record.get("render", {}).get("error_span_boxes_px", [])
    by_ref = {}
    for position, span in enumerate(spans if isinstance(spans, list) else []):
        if not isinstance(span, dict):
            continue
        reference = span.get("error_ref", position)
        by_ref[reference] = _union_boxes(span.get("boxes", []))
    boxes = [by_ref.get(index) for index, _error in
             enumerate(record.get("injected_errors", []))]
    if coords_mode == "pixel":
        return boxes
    if coords_mode == "relative":
        width, height = _page_size(record)
        return [_relative_bbox(box, width, height) if box is not None else None
                for box in boxes]
    raise ValueError("coords_mode must be 'pixel' or 'relative'")


def _mock_output(record, coords_mode="pixel"):
    boxes = _gt_error_boxes(record, coords_mode)
    errors = []
    for index, error in enumerate(record.get("injected_errors", [])):
        errors.append({
            "step_id": error.get("mutation_site"),
            "bbox": boxes[index] if index < len(boxes) else None,
            "type": error.get("type", ""),
        })
    return {
        "transcript": record["transcript_gt"]["text"],
        "errors": errors,
        "score": record["score_gt"]["awarded"],
    }


def _is_over_correction(record, transcript):
    if not record.get("injected_errors"):
        return False
    try:
        site = record["injected_errors"][0]["mutation_site"]
        gold_site = normalize_text({
            step["step_id"]: step["text"]
            for step in record["gold_solution"]
        }[site])
        mut_site = normalize_text({
            step["step_id"]: step["text"]
            for step in record["mutant_solution"]
        }[site])
    except (KeyError, TypeError):
        return False
    hyp_lines = [normalize_text(line) for line in transcript.splitlines()]
    hyp_lines = [line for line in hyp_lines if line]
    if not hyp_lines or gold_site == mut_site:
        return False
    best = min(hyp_lines,
               key=lambda line: min(lev(line, mut_site), lev(line, gold_site)))
    return lev(best, gold_site) < lev(best, mut_site)


def _individual_metrics(record, output, coords_mode="pixel"):
    has_error = not bool(record.get("control_flag", {}).get("error_free"))
    predictions = output["errors"]
    detected = bool(predictions)
    reference = normalize_text(record["transcript_gt"]["text"])
    hypothesis = normalize_text(output["transcript"])
    gt_score = record["score_gt"]["awarded"]
    score = output["score"]

    bbox_iou = None
    bbox_hit = None
    if has_error and detected:
        gt_boxes = [box for box in _gt_error_boxes(record, coords_mode) if box]
        bbox_iou = max(
            (iou(prediction.get("bbox"), gt_box)
             for prediction in predictions for gt_box in gt_boxes),
            default=0.0,
        )
        bbox_hit = bbox_iou >= 0.5

    gt_types = [error.get("type", "")
                for error in record.get("injected_errors", [])]
    predicted_types = {prediction.get("type", "")
                       for prediction in predictions}
    type_hits = [{"type": error_type,
                  "exact_match": error_type in predicted_types}
                 for error_type in gt_types]
    return {
        "has_error": has_error,
        "detected_error": detected,
        "detection_correct": detected if has_error else not detected,
        "transcript_cer": cer(reference, hypothesis),
        "over_correction": (_is_over_correction(record, output["transcript"])
                            if has_error else None),
        "bbox_iou": bbox_iou,
        "bbox_iou_at_0_5": bbox_hit,
        "type_exact_matches": type_hits,
        "score_exact_match": score == gt_score,
        "score_within_1": (score is not None and abs(score - gt_score) <= 1),
    }


def _mean(values):
    return sum(values) / len(values) if values else None


def _aggregate(result_rows):
    parsed = [row for row in result_rows if not row["parse_failure"]]
    metrics = [row["metrics"] for row in parsed]
    error_metrics = [value for value in metrics if value["has_error"]]
    controls = [value for value in metrics if not value["has_error"]]
    tpr = _mean([1.0 if value["detected_error"] else 0.0
                 for value in error_metrics])
    tnr = _mean([0.0 if value["detected_error"] else 1.0
                 for value in controls])
    bacc = (tpr + tnr) / 2.0 if tpr is not None and tnr is not None else None

    type_groups = {}
    for value in metrics:
        for hit in value["type_exact_matches"]:
            type_groups.setdefault(hit["type"], []).append(
                1.0 if hit["exact_match"] else 0.0)
    type_by_class = {key: _mean(values)
                     for key, values in sorted(type_groups.items())}

    cers = [value["transcript_cer"] for value in metrics]
    located = [value for value in error_metrics
               if value["detected_error"]]
    result = {
        "transcript_cer_mean": _mean(cers),
        "transcript_cer_median": statistics.median(cers) if cers else None,
        "tpr": tpr,
        "tnr": tnr,
        "bacc": bacc,
        "hallucinated_error_rate": None if tnr is None else 1.0 - tnr,
        "over_correction_rate": _mean([
            1.0 if value["over_correction"] else 0.0
            for value in error_metrics
        ]),
        "bbox_iou_mean": _mean([value["bbox_iou"] for value in located]),
        "iou_at_0_5": _mean([
            1.0 if value["bbox_iou_at_0_5"] else 0.0
            for value in located
        ]),
        "type_macro_exact_match": _mean(list(type_by_class.values())),
        "type_exact_match_by_class": type_by_class,
        "score_exact_match": _mean([
            1.0 if value["score_exact_match"] else 0.0 for value in metrics
        ]),
        "score_within_1": _mean([
            1.0 if value["score_within_1"] else 0.0 for value in metrics
        ]),
        "n_error_records": len(error_metrics),
        "n_control_records": len(controls),
        "n_location_records": len(located),
    }
    return result


def _format_metric(value):
    if value is None:
        return "N/A"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def _summary_markdown(summary):
    labels = [
        ("coords", "coords"),
        ("system", "system"),
        ("評価成功件数", "n_evaluated"),
        ("parse failures", "parse_failures"),
        ("欠損スキップ", "skipped_missing"),
        ("Transcript CER 平均", "transcript_cer_mean"),
        ("Transcript CER 中央値", "transcript_cer_median"),
        ("TPR", "tpr"), ("TNR", "tnr"), ("BACC", "bacc"),
        ("幻覚誤り率", "hallucinated_error_rate"),
        ("Over-correction 率", "over_correction_rate"),
        ("BBox IoU 平均", "bbox_iou_mean"),
        ("IoU@0.5", "iou_at_0_5"),
        ("種別 macro 完全一致", "type_macro_exact_match"),
        ("点数完全一致", "score_exact_match"),
        ("点数 ±1 一致", "score_within_1"),
        ("経過秒", "elapsed_seconds"),
        ("件/秒", "records_per_second"),
    ]
    lines = ["# ゼロショット評価サマリー", "", "| 項目 | 値 |",
             "|---|---:|"]
    lines.extend(f"| {label} | {_format_metric(summary.get(key))} |"
                 for label, key in labels)
    lines.extend(["", "## 種別ごとの完全一致率", "",
                  "| 種別 | 完全一致率 |", "|---|---:|"])
    values = summary.get("type_exact_match_by_class", {})
    if values:
        lines.extend(f"| {key} | {_format_metric(value)} |"
                     for key, value in values.items())
    else:
        lines.append("| N/A | N/A |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = _parse_args(argv)
    started = time.perf_counter()
    os.makedirs(args.out, exist_ok=True)

    paths, records, malformed_records = _read_records(args.records)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    selected = records[:args.n]

    skip_reasons = {}
    tasks = []
    for record in selected:
        image_path, reason = _eligible(record, args.images)
        if reason:
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        else:
            tasks.append((record, image_path))

    print(
        f"[load] chunks={len(paths)} records={len(records)} "
        f"selected={len(selected)} eligible={len(tasks)} coords={args.coords} "
        f"system={args.system} "
        f"skipped={sum(skip_reasons.values())} malformed={malformed_records}",
        flush=True,
    )

    result_rows = []
    results_path = os.path.join(args.out, "results.jsonl")
    with open(results_path + ".tmp", "w", encoding="utf-8", newline="\n") as stream:
        for index, (record, image_path) in enumerate(tasks, 1):
            request_error = None
            raw_response = ""
            raw_content = None
            parsed_output = None
            failure_reason = None
            try:
                if args.mock:
                    value = _mock_output(record, args.coords)
                    raw_content = json.dumps(value, ensure_ascii=False)
                    raw_response = raw_content
                else:
                    raw_response, request_error = _api_request(
                        args, record, image_path)
                    if request_error:
                        raise RuntimeError(request_error)
                    raw_content = _content_from_response(raw_response)
                    value = _parse_json_content(raw_content)
                parsed_output = _sanitize_output(value, record, args.coords)
            except (json.JSONDecodeError, KeyError, IndexError, TypeError,
                    RuntimeError, ValueError) as exc:
                failure_reason = f"{type(exc).__name__}: {exc}"

            row = {
                "sample_id": record.get("sample_id"),
                "has_error": not bool(
                    record.get("control_flag", {}).get("error_free")),
                "raw_response": raw_response,
                "raw_content": raw_content,
                "parsed_output": parsed_output,
                "parse_failure": failure_reason is not None,
                "failure_reason": failure_reason,
                "metrics": (_individual_metrics(record, parsed_output,
                                                  args.coords)
                            if parsed_output is not None else None),
            }
            result_rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 20 == 0:
                elapsed = time.perf_counter() - started
                failures = sum(1 for result in result_rows
                               if result["parse_failure"])
                print(
                    f"[progress] {index}/{len(tasks)} "
                    f"parse_failures={failures} "
                    f"elapsed={elapsed:.1f}s rate={index / elapsed:.2f}/s",
                    flush=True,
                )
    os.replace(results_path + ".tmp", results_path)

    elapsed = time.perf_counter() - started
    parse_failures = sum(1 for row in result_rows if row["parse_failure"])
    evaluated = len(result_rows) - parse_failures
    summary = {
        "n": evaluated,
        "n_requested": args.n,
        "n_available": len(records),
        "n_selected": len(selected),
        "n_attempted": len(result_rows),
        "n_evaluated": evaluated,
        "parse_failures": parse_failures,
        "skipped_missing": sum(skip_reasons.values()),
        "skip_reasons": skip_reasons,
        "malformed_input_records": malformed_records,
        "seed": args.seed,
        "mock": args.mock,
        "model": args.model,
        "coords": args.coords,
        "system": args.system,
        "elapsed_seconds": elapsed,
        "records_per_second": (len(result_rows) / elapsed if elapsed else 0.0),
    }
    summary.update(_aggregate(result_rows))
    _atomic_write_text(
        os.path.join(args.out, "eval_summary.json"),
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        os.path.join(args.out, "eval_summary.md"),
        _summary_markdown(summary),
    )
    print(
        f"[done] attempted={len(result_rows)} evaluated={evaluated} "
        f"parse_failures={parse_failures} skipped={summary['skipped_missing']} "
        f"coords={args.coords} system={args.system} "
        f"elapsed={elapsed:.2f}s rate={summary['records_per_second']:.2f}/s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
