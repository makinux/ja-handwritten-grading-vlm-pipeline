# -*- coding: utf-8 -*-
"""生成済み JSONL をチャンク単位で並列レンダリングする。"""
import argparse
import glob
import json
import multiprocessing
import os
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m3_render import render_record


def _line_count(path):
    with open(path, "r", encoding="utf-8") as stream:
        return sum(1 for _line in stream)


def _atomic_write_text(path, text):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.replace(temporary, path)


def _write_jsonl(path, values):
    text = "".join(
        json.dumps(value, ensure_ascii=False) + "\n" for value in values
    )
    _atomic_write_text(path, text)


def _error_text(exc):
    return f"{type(exc).__name__}: {exc}"


def _render_chunk(task):
    """ワーカープロセス内で一つのチャンクを最後まで処理する。"""
    (input_path, output_path, image_dir, debug_dir, expected_count,
     first_ordinal, debug_every) = task
    completed = []
    failures = []
    debug_count = 0

    with open(input_path, "r", encoding="utf-8") as stream:
        for local_index in range(expected_count):
            line = stream.readline()
            if not line:
                failures.append({
                    "sample_id": f"{os.path.basename(input_path)}:{local_index + 1}",
                    "error": "EOFError: 入力レコードが予定件数より少ない",
                })
                continue

            record = None
            try:
                record = json.loads(line)
                sample_id = record["sample_id"]
                image_path = os.path.join(image_dir, sample_id + ".png")
                ordinal = first_ordinal + local_index + 1
                debug_path = None
                if ordinal % debug_every == 0:
                    debug_path = os.path.join(
                        debug_dir, sample_id + "_boxes.png")
                record["render"] = render_record(
                    record, image_path, debug_png=debug_path)
                completed.append(record)
                if debug_path is not None:
                    debug_count += 1
            except Exception as exc:
                if isinstance(record, dict):
                    sample_id = str(record.get(
                        "sample_id",
                        f"{os.path.basename(input_path)}:{local_index + 1}",
                    ))
                else:
                    sample_id = f"{os.path.basename(input_path)}:{local_index + 1}"
                failures.append({
                    "sample_id": sample_id,
                    "error": _error_text(exc),
                })

    # 完了ファイルを最後に確定し、行数をレジューム判定に利用する。
    _write_jsonl(output_path, completed)
    return {
        "chunk": os.path.basename(input_path),
        "total": expected_count,
        "successful": len(completed),
        "failures": failures,
        "debug_images": debug_count,
        "status": "rendered",
    }


def _skipped_chunk_result(task):
    (input_path, _output_path, _image_dir, debug_dir, expected_count,
     first_ordinal, debug_every) = task
    debug_count = 0
    with open(input_path, "r", encoding="utf-8") as stream:
        for local_index in range(expected_count):
            line = stream.readline()
            if not line:
                break
            ordinal = first_ordinal + local_index + 1
            if ordinal % debug_every:
                continue
            try:
                sample_id = json.loads(line)["sample_id"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            debug_path = os.path.join(debug_dir, sample_id + "_boxes.png")
            if os.path.isfile(debug_path):
                debug_count += 1
    return {
        "chunk": os.path.basename(input_path),
        "total": expected_count,
        "successful": expected_count,
        "failures": [],
        "debug_images": debug_count,
        "status": "skipped",
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_dir", required=True,
                        help="chunk_*.jsonl の入力ディレクトリ")
    parser.add_argument("--out", dest="output_dir", required=True,
                        help="レンダリング結果の出力ディレクトリ")
    default_workers = max(1, (os.cpu_count() or 2) // 2)
    parser.add_argument("--workers", type=int, default=default_workers,
                        help="並列ワーカー数")
    parser.add_argument("--debug-every", type=int, default=500,
                        help="N 件ごとに bbox オーバーレイを保存")
    parser.add_argument("--limit", type=int,
                        help="先頭 M 件だけを処理")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers は 1 以上で指定してください")
    if args.debug_every < 1:
        parser.error("--debug-every は 1 以上で指定してください")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit は 0 以上で指定してください")
    return args


def _make_tasks(args, input_paths, image_dir, record_dir, debug_dir):
    tasks = []
    remaining = args.limit
    first_ordinal = 0
    for input_path in input_paths:
        input_count = _line_count(input_path)
        if remaining is None:
            expected_count = input_count
        else:
            expected_count = min(input_count, remaining)
        if expected_count == 0:
            if remaining == 0:
                break
            continue
        output_path = os.path.join(record_dir, os.path.basename(input_path))
        tasks.append((
            input_path, output_path, image_dir, debug_dir, expected_count,
            first_ordinal, args.debug_every,
        ))
        first_ordinal += expected_count
        if remaining is not None:
            remaining -= expected_count
            if remaining == 0:
                break
    return tasks


def main(argv=None):
    args = _parse_args(argv)
    started = time.perf_counter()
    image_dir = os.path.join(args.output_dir, "images")
    record_dir = os.path.join(args.output_dir, "records")
    debug_dir = os.path.join(args.output_dir, "debug")
    for path in (args.output_dir, image_dir, record_dir, debug_dir):
        os.makedirs(path, exist_ok=True)

    input_paths = sorted(glob.glob(os.path.join(
        args.input_dir, "chunk_*.jsonl")))
    tasks = _make_tasks(
        args, input_paths, image_dir, record_dir, debug_dir)

    pending = []
    results = []
    for task in tasks:
        output_path = task[1]
        expected_count = task[4]
        if (os.path.isfile(output_path)
                and _line_count(output_path) == expected_count):
            result = _skipped_chunk_result(task)
            results.append(result)
            print(f"[{result['chunk']}] skip ({expected_count} 件)", flush=True)
        else:
            pending.append(task)

    if pending:
        try:
            with multiprocessing.Pool(processes=args.workers) as pool:
                rendered_results = pool.map(_render_chunk, pending)
        except PermissionError as exc:
            # 制限付き Windows 環境では Pool の名前付きパイプ作成自体が
            # 拒否されることがある。通常実行では上の Pool 経路を使う。
            print(
                f"[warn] multiprocessing.Pool を開始できないため逐次実行: "
                f"{_error_text(exc)}",
                file=sys.stderr,
                flush=True,
            )
            rendered_results = [_render_chunk(task) for task in pending]
        results.extend(rendered_results)
        for result in rendered_results:
            print(
                f"[{result['chunk']}] render "
                f"成功={result['successful']} "
                f"失敗={len(result['failures'])}",
                flush=True,
            )

    failures = []
    for result in results:
        failures.extend(result["failures"])
    failures.sort(key=lambda value: value["sample_id"])
    _write_jsonl(os.path.join(args.output_dir, "failures.jsonl"), failures)

    total = sum(result["total"] for result in results)
    successful = sum(result["successful"] for result in results)
    debug_count = sum(result["debug_images"] for result in results)
    elapsed = time.perf_counter() - started
    summary = {
        "total_records": total,
        "successful": successful,
        "failed": len(failures),
        "elapsed_seconds": elapsed,
        "pages_per_second": successful / elapsed if elapsed else 0.0,
        "debug_images": debug_count,
    }
    _atomic_write_text(
        os.path.join(args.output_dir, "render_summary.json"),
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(
        f"[done] 総件数={total} 成功={successful} 失敗={len(failures)} "
        f"経過秒={elapsed:.2f} 頁毎秒={summary['pages_per_second']:.2f} "
        f"debug枚数={debug_count}",
        flush=True,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
