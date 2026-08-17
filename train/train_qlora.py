#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-VL-2B QLoRA training/evaluation for TITAN V機.

Hardware contract: one NVIDIA TITAN V 12 GB (Volta, sm_70), NVIDIA driver
570 series.  Volta has no BF16 support, so this script always uses fp16
(bf16=False).  FlashAttention is unavailable, so attention is fixed to SDPA.
Unsloth, TRL, and Triton are deliberately not used; the training stack is
transformers + PEFT + bitsandbytes + accelerate (HF Trainer) only.

The pure-Python data, JSON, and metric functions above ``run_training`` do not
import torch.  This is intentional: ``--self-test`` must work on a CPU-only
machine without the ML stack installed.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence


REFERENCE_ZEROSHOT_2B = {
    "transcript_cer_mean": 0.024,
    "bacc": 0.843,
    "hallucinated_error_rate": 0.314,
    "score_exact_match": 0.345,
    "iou_at_0_5": 0.0,
    "type_macro_exact_match": 0.0,
}


def log(message: str) -> None:
    """Print one nohup-friendly progress line."""
    print(message, flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON objects in input order; images remain unopened."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: JSON object ではありません")
            records.append(value)
    return records


def dataset_image_path(data_dir: Path, record: dict[str, Any]) -> Path:
    relative = record.get("image")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{record.get('id')}: image が不正です")
    root = data_dir.resolve()
    path = (root / relative).resolve()
    if os.path.commonpath([str(root), str(path)]) != str(root):
        raise ValueError(f"Dataset 外の画像は参照できません: {relative}")
    return path


def conversation_with_image(
    record: dict[str, Any], image: Any, include_assistant: bool = True
) -> list[dict[str, Any]]:
    messages = copy.deepcopy(record["messages"])
    if not include_assistant:
        messages = [m for m in messages if m.get("role") != "assistant"]
    replacements = 0
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                part["image"] = image
                replacements += 1
    if replacements != 1:
        raise ValueError(
            f"{record.get('id')}: image placeholder は 1 個必要です "
            f"(実際 {replacements})"
        )
    return messages


# The functions in this section are synchronized with pipeline/rewards.py and
# the evaluation cell in kaggle/qwen3vl_2b_sft.ipynb.  Dataset bboxes are
# already normalized to 0--1000, so both sides are compared in that space.
def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", "", value)


def lev(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def cer(ref: str, hyp: str) -> float:
    return (0.0 if not hyp else 1.0) if not ref else lev(ref, hyp) / len(ref)


def iou(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if not a or not b:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_json_robust(content: Any) -> Any:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise TypeError("model output is not text or object")
    text = content.strip()
    attempts = [text]
    if text.startswith("```") and text.endswith("```"):
        inner = text[3:-3].strip()
        if inner.lower().startswith("json"):
            inner = inner[4:].lstrip()
        attempts.append(inner)
    for candidate in attempts:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start >= 0:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
        return value
    raise json.JSONDecodeError("JSON object not found", text, 0)


def sanitize_bbox(value: Any) -> list[int | float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        return None
    coords = [float(v) for v in value]
    if not all(math.isfinite(v) for v in coords):
        return None
    coords = [min(1000.0, max(0.0, v)) for v in coords]
    if coords[2] <= coords[0] or coords[3] <= coords[1]:
        return None
    return [int(v) if v.is_integer() else v for v in coords]


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def sanitize_output(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("output JSON is not an object")
    if not isinstance(value.get("transcript"), str):
        raise TypeError("transcript is not a string")
    if not isinstance(value.get("errors"), list):
        raise TypeError("errors is not an array")
    errors = []
    for error in value["errors"]:
        if not isinstance(error, dict):
            error = {}
        step_id = error.get("step_id")
        if step_id is not None and not isinstance(step_id, str):
            step_id = str(step_id)
        error_type = error.get("type", "")
        if not isinstance(error_type, str):
            error_type = str(error_type)
        errors.append(
            {
                "step_id": step_id,
                "bbox": sanitize_bbox(error.get("bbox")),
                "type": error_type,
            }
        )
    return {
        "transcript": value["transcript"],
        "errors": errors,
        "score": _as_int(value.get("score")),
    }


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "".join(parts)
    raise TypeError("message content is not text")


def teacher_output(record: dict[str, Any]) -> dict[str, Any]:
    assistants = [
        message
        for message in record.get("messages", [])
        if message.get("role") == "assistant"
    ]
    if len(assistants) != 1:
        raise ValueError(f"{record.get('id')}: assistant message は 1 個必要です")
    return sanitize_output(parse_json_robust(_content_text(assistants[0]["content"])))


def individual_metrics(
    reference: dict[str, Any], output: dict[str, Any]
) -> dict[str, Any]:
    has_error = bool(reference["errors"])
    predictions = output["errors"]
    detected = bool(predictions)
    bbox_iou = None
    bbox_hit = None
    if has_error and detected:
        gt_boxes = [
            error.get("bbox") for error in reference["errors"] if error.get("bbox")
        ]
        bbox_iou = max(
            (
                iou(prediction.get("bbox"), gt_box)
                for prediction in predictions
                for gt_box in gt_boxes
            ),
            default=0.0,
        )
        bbox_hit = bbox_iou >= 0.5
    gt_types = [error.get("type", "") for error in reference["errors"]]
    predicted_types = {error.get("type", "") for error in predictions}
    type_hits = [
        {"type": error_type, "exact_match": error_type in predicted_types}
        for error_type in gt_types
    ]
    gt_score = reference["score"]
    score = output["score"]
    return {
        "has_error": has_error,
        "detected_error": detected,
        "transcript_cer": cer(
            normalize_text(reference["transcript"]),
            normalize_text(output["transcript"]),
        ),
        "bbox_iou": bbox_iou,
        "bbox_iou_at_0_5": bbox_hit,
        "type_exact_matches": type_hits,
        "score_exact_match": score == gt_score,
        "score_within_1": (
            score is not None and gt_score is not None and abs(score - gt_score) <= 1
        ),
    }


def aggregate_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    parsed = [row for row in rows if not row["parse_failure"]]
    metrics = [row["metrics"] for row in parsed]
    error_metrics = [value for value in metrics if value["has_error"]]
    controls = [value for value in metrics if not value["has_error"]]
    tpr = _mean([float(value["detected_error"]) for value in error_metrics])
    tnr = _mean([float(not value["detected_error"]) for value in controls])
    bacc = (tpr + tnr) / 2.0 if tpr is not None and tnr is not None else None
    type_groups: dict[str, list[float]] = {}
    for value in metrics:
        for hit in value["type_exact_matches"]:
            type_groups.setdefault(hit["type"], []).append(float(hit["exact_match"]))
    type_by_class = {
        key: _mean(values) for key, values in sorted(type_groups.items())
    }
    located = [value for value in error_metrics if value["detected_error"]]
    return {
        "n_attempted": len(rows),
        "n_evaluated": len(parsed),
        "parse_failures": len(rows) - len(parsed),
        "transcript_cer_mean": _mean(
            [value["transcript_cer"] for value in metrics]
        ),
        "tpr": tpr,
        "tnr": tnr,
        "bacc": bacc,
        "hallucinated_error_rate": None if tnr is None else 1.0 - tnr,
        "iou_at_0_5": _mean(
            [float(value["bbox_iou_at_0_5"]) for value in located]
        ),
        "type_macro_exact_match": _mean(list(type_by_class.values())),
        "type_exact_match_by_class": type_by_class,
        "score_exact_match": _mean(
            [float(value["score_exact_match"]) for value in metrics]
        ),
        "score_within_1": _mean(
            [float(value["score_within_1"]) for value in metrics]
        ),
        "n_error_records": len(error_metrics),
        "n_control_records": len(controls),
        "n_location_records": len(located),
    }


def _meta_eval_selected(record: dict[str, Any]) -> bool:
    """Recognize Task G's explicit evaluation-selection metadata."""
    meta = record.get("meta")
    if not isinstance(meta, dict):
        return False
    for key in ("eval_selected", "selected_for_eval", "is_eval"):
        if meta.get(key) is True:
            return True
    evaluation = meta.get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("selected") is True:
        return True
    return meta.get("split") in {"eval", "evaluation"}


def select_eval_records(
    records: Sequence[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    # Synchronized with kaggle/qwen3vl_2b_sft.ipynb evaluation cell: metadata-
    # selected records come first in input order, then the beginning of
    # test.jsonl fills the remainder, with duplicate IDs removed.
    priority = [record for record in records if _meta_eval_selected(record)]
    chosen: list[dict[str, Any]] = []
    seen = set()
    for record in priority + list(records):
        key = record.get("id")
        if key in seen:
            continue
        chosen.append(record)
        seen.add(key)
        if len(chosen) >= count:
            break
    return chosen


def _assert_close(actual: float | None, expected: float, name: str) -> None:
    if actual is None or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(f"{name}: expected {expected}, got {actual}")
    log(f"PASS: {name}")


def run_self_test() -> int:
    """Exercise metrics and parsing without importing torch/transformers."""
    forbidden_before = [name for name in ("torch", "transformers", "peft") if name in sys.modules]
    if forbidden_before:
        raise AssertionError(f"heavy modules imported before self-test: {forbidden_before}")

    _assert_close(cer("abc", "adc"), 1 / 3, "CER substitution")
    _assert_close(cer("", "x"), 1.0, "CER empty reference")
    _assert_close(cer("", ""), 0.0, "CER two empty strings")
    _assert_close(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0, "IoU identical")
    _assert_close(iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3, "IoU overlap")
    _assert_close(iou(None, [0, 0, 1, 1]), 0.0, "IoU missing bbox")

    fenced = '```json\n{"transcript":"x","errors":[],"score":2}\n```'
    assert parse_json_robust(fenced)["score"] == 2
    log("PASS: robust parse fenced JSON")
    prefixed = 'answer follows: {"transcript":"x","errors":[],"score":3} trailing'
    assert parse_json_robust(prefixed)["score"] == 3
    log("PASS: robust parse prefixed JSON")
    assert parse_json_robust({"transcript": "x"})["transcript"] == "x"
    log("PASS: robust parse object passthrough")
    assert sanitize_bbox([-1, 2, 1001, 900]) == [0, 2, 1000, 900]
    assert sanitize_bbox([0, 0, 0, 4]) is None
    log("PASS: bbox sanitation")

    ref_error = {
        "transcript": "１２ + 3", "errors": [{"bbox": [0, 0, 10, 10], "type": "calc"}], "score": 4
    }
    pred_error = {
        "transcript": "12+3", "errors": [{"bbox": [0, 0, 10, 10], "type": "calc"}], "score": 3
    }
    ref_control = {"transcript": "ok", "errors": [], "score": 5}
    pred_control = {"transcript": "ok", "errors": [], "score": 5}
    m1 = individual_metrics(ref_error, pred_error)
    m2 = individual_metrics(ref_control, pred_control)
    assert m1["bbox_iou_at_0_5"] is True and m1["score_within_1"] is True
    assert m1["score_exact_match"] is False and m1["transcript_cer"] == 0.0
    log("PASS: individual IoU/type/score/CER metrics")
    rows = [
        {"parse_failure": False, "metrics": m1},
        {"parse_failure": False, "metrics": m2},
        {"parse_failure": True, "metrics": None},
    ]
    summary = aggregate_metrics(rows)
    _assert_close(summary["tpr"], 1.0, "TPR")
    _assert_close(summary["tnr"], 1.0, "TNR")
    _assert_close(summary["bacc"], 1.0, "BACC")
    _assert_close(summary["type_macro_exact_match"], 1.0, "type exact macro")
    _assert_close(summary["score_exact_match"], 0.5, "score exact")
    _assert_close(summary["score_within_1"], 1.0, "score within one")
    assert summary["parse_failures"] == 1
    log("PASS: parse failure aggregation")

    records = [
        {"id": "a", "meta": {}},
        {"id": "b", "meta": {"eval_selected": True}},
        {"id": "c", "meta": {}},
    ]
    assert [r["id"] for r in select_eval_records(records, 2)] == ["b", "a"]
    log("PASS: notebook-synchronized eval selection")
    forbidden_after = [name for name in ("torch", "transformers", "peft") if name in sys.modules]
    if forbidden_after:
        raise AssertionError(f"heavy modules imported by self-test: {forbidden_after}")
    log("SELF-TEST: ALL PASS (torch/transformers/peft were not imported)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--train-limit", type=int, default=3000)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=768 * 1086)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--eval-n", type=int, default=200)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-quant", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-before-eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.data_dir is None or args.out_dir is None:
        raise SystemExit("--data-dir and --out-dir are required unless --self-test is used")
    positive = (
        "train_limit", "epochs", "lora_r", "lora_alpha", "lr", "batch",
        "grad_accum", "max_pixels", "min_pixels", "eval_n", "eval_batch",
        "max_new_tokens",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be greater than zero")
    if args.min_pixels > args.max_pixels:
        raise SystemExit("--min-pixels must not exceed --max-pixels")


def _is_vision_name(name: str) -> bool:
    padded = f".{name.lower()}."
    return any(
        token in padded
        for token in (
            ".visual.", ".vision.", ".vision_model.", ".vision_tower.", ".merger."
        )
    )


def _trainable_lora_names(model: Any) -> list[str]:
    return [name for name, parameter in model.named_parameters() if "lora_" in name and parameter.requires_grad]


def _attach_language_lora(model: Any, args: argparse.Namespace, LoraConfig: Any, get_peft_model: Any) -> Any:
    target_suffixes = [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ]

    def config(targets: list[str]) -> Any:
        return LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
            task_type="CAUSAL_LM",
        )

    model = get_peft_model(model, config(target_suffixes))
    bad = [name for name in _trainable_lora_names(model) if _is_vision_name(name)]
    if bad:
        log("[lora] vision/merger adapters detected; rebuilding with exact language-only names:")
        for name in bad:
            log(f"  exclude: {name}")
        if not hasattr(model, "unload"):
            raise RuntimeError("PEFT model cannot unload unsafe vision adapters")
        model = model.unload()
        exact_targets = [
            name
            for name, _module in model.named_modules()
            if name.rsplit(".", 1)[-1] in target_suffixes and not _is_vision_name(name)
        ]
        if not exact_targets:
            raise RuntimeError("language LoRA target modules were not found")
        model = get_peft_model(model, config(exact_targets))

    remaining = [name for name in _trainable_lora_names(model) if _is_vision_name(name)]
    if remaining:
        raise RuntimeError(f"vision/merger LoRA exclusion failed: {remaining[:5]}")
    names = _trainable_lora_names(model)
    if not names:
        raise RuntimeError("no trainable LoRA parameters were attached")
    log(f"[lora] verified language-only adapters: {len(names)} trainable tensors")
    model.print_trainable_parameters()
    return model


def _configure_processor(processor: Any, args: argparse.Namespace) -> None:
    image_processor = getattr(processor, "image_processor", processor)
    image_processor.min_pixels = args.min_pixels
    image_processor.max_pixels = args.max_pixels
    if hasattr(processor, "min_pixels"):
        processor.min_pixels = args.min_pixels
    if hasattr(processor, "max_pixels"):
        processor.max_pixels = args.max_pixels
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    log(f"[processor] min_pixels={args.min_pixels} max_pixels={args.max_pixels}")


def _move_batch(batch: Any, device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _make_train_collator(
    processor: Any, data_dir: Path, max_seq: int, torch: Any, Image: Any
) -> Any:
    class AssistantOnlyVisionCollator:
        """Open only the current batch and mask everything before assistant output."""

        def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
            images = []
            sources = []
            try:
                records = [feature["record"] for feature in features]
                for record in records:
                    source = Image.open(dataset_image_path(data_dir, record))
                    sources.append(source)
                    images.append(source.convert("RGB"))
                full_prompts = [
                    processor.apply_chat_template(
                        conversation_with_image(record, image),
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    for record, image in zip(records, images)
                ]
                user_prompts = [
                    processor.apply_chat_template(
                        conversation_with_image(record, image, include_assistant=False),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for record, image in zip(records, images)
                ]
                # The full sequence and user-only+generation-prompt token-length
                # difference defines the assistant labels, as in the notebook.
                prefix_lengths = []
                for prompt, image in zip(user_prompts, images):
                    prefix = processor(
                        text=[prompt], images=[image], padding=False,
                        truncation=True, max_length=max_seq, return_tensors="pt"
                    )
                    prefix_lengths.append(int(prefix["input_ids"].shape[1]))
                batch = processor(
                    text=full_prompts,
                    images=images,
                    padding=True,
                    truncation=True,
                    max_length=max_seq,
                    return_tensors="pt",
                )
                labels = batch["input_ids"].clone()
                for row, prefix_length in enumerate(prefix_lengths):
                    labels[row, : min(prefix_length, labels.shape[1])] = -100
                labels[batch["attention_mask"] == 0] = -100
                batch["labels"] = labels
                return dict(batch)
            finally:
                for image in images:
                    image.close()
                for source in sources:
                    source.close()

    return AssistantOnlyVisionCollator()


def _infer_batch(
    model: Any,
    processor: Any,
    records: Sequence[dict[str, Any]],
    data_dir: Path,
    max_new_tokens: int,
    torch: Any,
    Image: Any,
) -> list[str]:
    images = []
    sources = []
    try:
        for record in records:
            source = Image.open(dataset_image_path(data_dir, record))
            sources.append(source)
            images.append(source.convert("RGB"))
        prompts = [
            processor.apply_chat_template(
                conversation_with_image(record, image, include_assistant=False),
                tokenize=False,
                add_generation_prompt=True,
            )
            for record, image in zip(records, images)
        ]
        inputs = processor(
            text=prompts, images=images, padding=True, return_tensors="pt"
        )
        device = next(model.parameters()).device
        inputs = _move_batch(inputs, device)
        input_width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        return processor.batch_decode(
            generated[:, input_width:], skip_special_tokens=True
        )
    finally:
        for image in images:
            image.close()
        for source in sources:
            source.close()


def _is_cuda_oom(exc: BaseException, torch: Any) -> bool:
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", ())
    return (bool(oom_type) and isinstance(exc, oom_type)) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def evaluate_records(
    model: Any,
    processor: Any,
    records: Sequence[dict[str, Any]],
    data_dir: Path,
    eval_batch: int,
    max_new_tokens: int,
    torch: Any,
    Image: Any,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    batch_size = eval_batch
    cursor = 0
    while cursor < len(records):
        chunk = records[cursor : cursor + batch_size]
        batch_failure = None
        try:
            outputs = _infer_batch(
                model, processor, chunk, data_dir, max_new_tokens, torch, Image
            )
        except Exception as exc:  # inference/IO failure becomes parse failure
            if _is_cuda_oom(exc, torch) and batch_size > 1:
                log(f"[eval] CUDA OOM at batch={batch_size}; retrying with batch=1")
                batch_size = 1
                torch.cuda.empty_cache()
                continue
            outputs = [""] * len(chunk)
            batch_failure = f"{type(exc).__name__}: {exc}"
            if _is_cuda_oom(exc, torch):
                torch.cuda.empty_cache()
        for record, raw in zip(chunk, outputs):
            reference = teacher_output(record)
            parsed_output = None
            failure_reason = batch_failure
            if failure_reason is None:
                try:
                    parsed_output = sanitize_output(parse_json_robust(raw))
                except Exception as exc:
                    failure_reason = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "id": record.get("id"),
                    "parse_failure": failure_reason is not None,
                    "failure_reason": failure_reason,
                    "raw_output": raw,
                    "parsed_output": parsed_output,
                    "metrics": (
                        individual_metrics(reference, parsed_output)
                        if parsed_output is not None
                        else None
                    ),
                }
            )
            completed = len(rows)
            if completed % 10 == 0 or completed == len(records):
                failures = sum(row["parse_failure"] for row in rows)
                log(f"  {completed}/{len(records)} parse_failures={failures}")
        cursor += len(chunk)
    summary = aggregate_metrics(rows)
    summary["temperature"] = 0.0
    summary["max_new_tokens"] = max_new_tokens
    summary["effective_eval_batch"] = batch_size
    return {"summary": summary, "rows": rows}


def _write_results(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    temporary.replace(path)
    log(f"[results] saved: {path}")


def _metric_table(before: Any, after: Any) -> None:
    metric_rows = [
        ("transcript CER (mean)", "transcript_cer_mean"), ("TPR", "tpr"),
        ("TNR", "tnr"), ("BACC", "bacc"),
        ("hallucination rate", "hallucinated_error_rate"),
        ("score exact", "score_exact_match"), ("score ±1", "score_within_1"),
        ("bbox IoU@0.5", "iou_at_0_5"),
        ("type exact (macro)", "type_macro_exact_match"),
        ("parse failures", "parse_failures"),
    ]

    def display(value: Any) -> str:
        if value is None:
            return "N/A"
        return f"{value:.6f}" if isinstance(value, float) else str(value)

    log(f"{'metric':<28} {'before':>12} {'after':>12}")
    log("-" * 54)
    before_summary = before.get("summary", {}) if isinstance(before, dict) else {}
    after_summary = after.get("summary", {}) if isinstance(after, dict) else {}
    for label, key in metric_rows:
        log(f"{label:<28} {display(before_summary.get(key)):>12} {display(after_summary.get(key)):>12}")


def run_training(args: argparse.Namespace) -> int:
    # Heavy imports are deliberately delayed until after --self-test handling.
    import gc

    import torch
    from PIL import Image
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required (expected NVIDIA TITAN V)")
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.out_dir / "checkpoints"
    adapters_dir = args.out_dir / "lora_adapters"
    results_path = args.out_dir / "eval_results.json"

    for required in ("train.jsonl", "val.jsonl", "test.jsonl"):
        path = args.data_dir / required
        if not path.is_file():
            raise FileNotFoundError(path)
    train_records = read_jsonl(args.data_dir / "train.jsonl")[: args.train_limit]
    test_records = read_jsonl(args.data_dir / "test.jsonl")
    eval_records = select_eval_records(test_records, args.eval_n)
    if not train_records:
        raise ValueError("学習レコードが 0 件です")
    if not eval_records:
        raise ValueError("評価レコードが 0 件です")
    for record in train_records:
        path = dataset_image_path(args.data_dir, record)
        if not path.is_file():
            raise FileNotFoundError(path)
    log(f"[data] train={len(train_records)} (unshuffled prefix), eval={len(eval_records)}/{len(test_records)}")

    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    _configure_processor(processor, args)
    quantization_config = None
    if not args.no_quant:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    model_kwargs = {
        "attn_implementation": "sdpa",
        "torch_dtype": torch.float16,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": 0}
    log(f"[model] loading {args.model_id} quantized={not args.no_quant} fp16 SDPA")
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    if args.no_quant:
        model.to(torch.device("cuda:0"))

    config = {
        "model_id": args.model_id,
        "data_dir": str(args.data_dir),
        "out_dir": str(args.out_dir),
        "train_limit": args.train_limit,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.0,
        "learning_rate": args.lr,
        "batch": args.batch,
        "gradient_accumulation": args.grad_accum,
        "max_seq": 4096,
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
        "eval_n": args.eval_n,
        "eval_batch": args.eval_batch,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "quantized_4bit": not args.no_quant,
        "no_quant": args.no_quant,
        "smoke": args.smoke,
        "skip_before_eval": args.skip_before_eval,
        "resume": args.resume,
        "fp16": True,
        "bf16": False,
        "attn_implementation": "sdpa",
        "optim": "paged_adamw_8bit",
        # Notebook cell 5 uses linear (not cosine); keep comparison exact.
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.03,
        "reference_zeroshot_2b": REFERENCE_ZEROSHOT_2B,
    }
    runtime = {
        "train_seconds": None,
        "eval_seconds_before": None,
        "eval_seconds_after": None,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "n_train_used": len(train_records),
    }
    existing = {}
    if results_path.is_file():
        try:
            existing = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log("[results] existing eval_results.json is unreadable; replacing it")
    if args.skip_before_eval and isinstance(existing.get("runtime"), dict):
        runtime["eval_seconds_before"] = existing["runtime"].get(
            "eval_seconds_before"
        )
    before_result = existing.get("before") if args.skip_before_eval else None
    results = {
        "config": config,
        "eval_ids": [record.get("id") for record in eval_records],
        "before": before_result,
        "after": None,
        "runtime": runtime,
    }

    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "left"
    if args.skip_before_eval:
        log("[before] skipped (existing result preserved when present)")
    else:
        log("[before] evaluating base model before LoRA attachment")
        model.eval()
        started = time.monotonic()
        before_result = evaluate_records(
            model, processor, eval_records, args.data_dir, args.eval_batch,
            args.max_new_tokens, torch, Image,
        )
        runtime["eval_seconds_before"] = time.monotonic() - started
        results["before"] = before_result
        _write_results(results_path, results)
        gc.collect()
        torch.cuda.empty_cache()

    if not args.no_quant:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = _attach_language_lora(model, args, LoraConfig, get_peft_model)

    class LazyJSONLVisionDataset(Dataset):
        def __len__(self) -> int:
            return len(train_records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {"record": train_records[index]}

    tokenizer.padding_side = "right"
    collator = _make_train_collator(processor, args.data_dir, 4096, torch, Image)
    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=LazyJSONLVisionDataset(),
        data_collator=collator,
        processing_class=processor,
    )
    checkpoint_exists = any(checkpoints_dir.glob("checkpoint-*"))
    resume = bool(args.resume and checkpoint_exists)
    if args.resume and not checkpoint_exists:
        log("[train] --resume requested, but no checkpoint exists; starting fresh")
    log(f"[train] starting resume_from_checkpoint={resume}")
    started = time.monotonic()
    train_result = trainer.train(resume_from_checkpoint=True if resume else None)
    runtime["train_seconds"] = time.monotonic() - started
    log(f"[train] metrics={train_result.metrics}")
    adapters_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapters_dir)
    processor.save_pretrained(adapters_dir)
    log(f"[train] saved adapters and processor: {adapters_dir}")

    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    model.eval()
    tokenizer.padding_side = "left"
    gc.collect()
    torch.cuda.empty_cache()
    log("[after] evaluating trained LoRA")
    started = time.monotonic()
    after_result = evaluate_records(
        model, processor, eval_records, args.data_dir, args.eval_batch,
        args.max_new_tokens, torch, Image,
    )
    runtime["eval_seconds_after"] = time.monotonic() - started
    results["after"] = after_result
    _write_results(results_path, results)
    _metric_table(results.get("before"), after_result)
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.smoke:
        args.train_limit = 8
        args.eval_n = 4
        args.max_new_tokens = 128
        log("[config] smoke overrides: train_limit=8 eval_n=4 max_new_tokens=128")
    _validate_args(args)
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
