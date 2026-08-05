# -*- coding: utf-8 -*-
"""大規模データセットをチャンク単位で生成・再開するランナー。"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_core import generate_batch
from llm_mutation import FakeMutationProposer, LLMMutationProposer
from verbalizer import FakeVerbalizer, LLMVerbalizer


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _atomic_write_text(path, text):
    """同じディレクトリの一時ファイル経由で成果物を確定する。"""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.replace(temporary, path)


def _line_count(path):
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for _line in stream)


def _without_payload(value):
    """出力用オブジェクトから内部キー ``_payload`` を再帰的に除く。"""
    if isinstance(value, dict):
        return {key: _without_payload(item) for key, item in value.items()
                if key != "_payload"}
    if isinstance(value, list):
        return [_without_payload(item) for item in value]
    return value


def _make_clients(fake):
    if fake:
        return FakeVerbalizer(), FakeMutationProposer()
    return LLMVerbalizer(), LLMMutationProposer()


def _write_chunk(path, records):
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    _atomic_write_text(path, "".join(line + "\n" for line in lines))


def _load_chunk(path, expected_count, chunk_index):
    records = []
    prefix = f"c{chunk_index:03d}-"
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: JSON の読み込みに失敗") from exc
            if not record["sample_id"].startswith(prefix):
                raise ValueError(f"{path}: sample_id の接頭辞が不正")
            if not record["pair_id"].startswith(prefix):
                raise ValueError(f"{path}: pair_id の接頭辞が不正")
            records.append(record)
    if len(records) != expected_count:
        raise ValueError(
            f"{path}: 期待行数 {expected_count} に対して {len(records)} 行")
    return records


def _namespace_records(kept, chunk_index, expected_count):
    prefix = f"c{chunk_index:03d}-"
    records = []
    for record, _problem in kept:
        record["sample_id"] = prefix + record["sample_id"]
        record["pair_id"] = prefix + record["pair_id"]
        records.append(_without_payload(record))
    if len(records) != expected_count:
        raise RuntimeError(
            f"生成件数が不足: 期待 {expected_count} 件、実際 {len(records)} 件")
    sample_ids = [record["sample_id"] for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("チャンク内で sample_id が重複")
    return records


def _progress(chunk_index, chunk_count, count, stats, elapsed, status):
    verbal_attempted = stats.get("verbalize_attempted", 0)
    llm_attempted = stats.get("llm_mut_attempted", 0)
    print(
        f"[chunk {chunk_index + 1}/{chunk_count} {status}] "
        f"件数={count} "
        f"verbalize通過率={_ratio(stats.get('verbalize_pass', 0), verbal_attempted):.3f} "
        f"fallback率={_ratio(stats.get('verbalize_fallback_steps', 0), verbal_attempted):.3f} "
        f"llm_mut受理率={_ratio(stats.get('llm_mut_accepted', 0), llm_attempted):.3f} "
        f"経過秒={elapsed:.2f}",
        flush=True,
    )


def _add_stats(total, stats):
    for key in (
            "n_generated", "g1_pass", "g1_fail", "g2_pass", "g2_fail",
            "verbalize_attempted", "verbalize_pass",
            "verbalize_fallback_steps", "llm_mut_attempted",
            "llm_mut_accepted", "llm_mut_fallback"):
        total[key] += stats.get(key, 0)
    total["per_operator"].update(stats.get("per_operator", {}))


def _add_records(total, records):
    for record in records:
        sample_id = record["sample_id"]
        if sample_id in total["sample_ids"]:
            raise RuntimeError(f"sample_id が全体で重複: {sample_id}")
        total["sample_ids"].add(sample_id)
        total["pair_chunks"][record["pair_id"]].add(sample_id.split("-", 1)[0])
        if record["control_flag"]["error_free"]:
            total["controls"] += 1
        else:
            total["errors"] += 1


def _build_summary(args, totals, chunk_summaries, elapsed):
    count = totals["errors"] + totals["controls"]
    for pair_id, chunk_prefixes in totals["pair_chunks"].items():
        if len(chunk_prefixes) != 1:
            raise RuntimeError(f"pair_id が複数チャンクにまたがる: {pair_id}")

    return {
        "requested_n": args.n,
        "total_records": count,
        "chunks": chunk_summaries,
        "samples": {
            "error": totals["errors"],
            "control": totals["controls"],
            "error_to_control_ratio": _ratio(
                totals["errors"], totals["controls"]),
        },
        "g1": {
            "pass": totals["g1_pass"],
            "fail": totals["g1_fail"],
            "pass_rate": _ratio(
                totals["g1_pass"], totals["g1_pass"] + totals["g1_fail"]),
        },
        "g2": {
            "pass": totals["g2_pass"],
            "fail": totals["g2_fail"],
            "pass_rate": _ratio(
                totals["g2_pass"], totals["g2_pass"] + totals["g2_fail"]),
        },
        "verbalize": {
            "attempted": totals["verbalize_attempted"],
            "passed": totals["verbalize_pass"],
            "fallback": totals["verbalize_fallback_steps"],
            "pass_rate": _ratio(
                totals["verbalize_pass"], totals["verbalize_attempted"]),
            "fallback_rate": _ratio(
                totals["verbalize_fallback_steps"],
                totals["verbalize_attempted"]),
        },
        "llm_mut": {
            "attempted": totals["llm_mut_attempted"],
            "accepted": totals["llm_mut_accepted"],
            "fallback": totals["llm_mut_fallback"],
            "accept_rate": _ratio(
                totals["llm_mut_accepted"], totals["llm_mut_attempted"]),
        },
        "per_operator": dict(sorted(totals["per_operator"].items())),
        "total_elapsed_seconds": elapsed,
        "records_per_second": _ratio(count, elapsed),
    }


def _summary_markdown(summary):
    samples = summary["samples"]
    verbal = summary["verbalize"]
    llm_mut = summary["llm_mut"]
    operators = "\n".join(
        f"| `{operator}` | {count} |"
        for operator, count in summary["per_operator"].items())
    return f"""# Generation summary

- 総件数: {summary['total_records']}
- 誤り / 対照: {samples['error']} / {samples['control']} ({samples['error_to_control_ratio']:.3f}:1)
- G1: {summary['g1']['pass']} pass / {summary['g1']['fail']} fail (通過率 {summary['g1']['pass_rate']:.3f})
- G2: {summary['g2']['pass']} pass / {summary['g2']['fail']} fail (通過率 {summary['g2']['pass_rate']:.3f})
- verbalize: attempted {verbal['attempted']} / pass {verbal['passed']} / fallback {verbal['fallback']} (通過率 {verbal['pass_rate']:.3f}, fallback率 {verbal['fallback_rate']:.3f})
- llm_mut: attempted {llm_mut['attempted']} / accepted {llm_mut['accepted']} / fallback {llm_mut['fallback']} (受理率 {llm_mut['accept_rate']:.3f})
- 総経過時間: {summary['total_elapsed_seconds']:.2f} 秒
- スループット: {summary['records_per_second']:.2f} 件/秒

## オペレータ別件数

| オペレータ | 件数 |
|---|---:|
{operators}
"""


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True, help="総生成件数")
    parser.add_argument("--chunk", type=int, required=True, help="チャンク件数")
    parser.add_argument("--seed", type=int, required=True, help="基準乱数シード")
    parser.add_argument("--out", type=Path, required=True, help="出力ディレクトリ")
    parser.add_argument("--llm-mut-prob", type=float, default=0.25)
    parser.add_argument("--fake", action="store_true",
                        help="オフライン用 Fake クライアントを使う")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--no-resume", action="store_true",
                        help="既存チャンクを無視して全件再生成する")
    resume.add_argument("--resume", action="store_false", dest="no_resume",
                        help="有効な既存チャンクを再利用する（既定）")
    parser.set_defaults(no_resume=False)
    args = parser.parse_args(argv)
    if args.n <= 0:
        parser.error("--n は 1 以上にしてください")
    if args.chunk <= 0:
        parser.error("--chunk は 1 以上にしてください")
    if not 0 <= args.llm_mut_prob <= 1:
        parser.error("--llm-mut-prob は 0 以上 1 以下にしてください")
    return args


def main(argv=None):
    args = _parse_args(argv)
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    chunk_count = (args.n + args.chunk - 1) // args.chunk
    totals = defaultdict(int)
    totals["per_operator"] = Counter()
    totals["sample_ids"] = set()
    totals["pair_chunks"] = defaultdict(set)
    chunk_summaries = []

    for chunk_index in range(chunk_count):
        expected_count = min(args.chunk, args.n - chunk_index * args.chunk)
        chunk_path = args.out / f"chunk_{chunk_index:03d}.jsonl"
        stats_path = args.out / f"stats_{chunk_index:03d}.json"

        can_resume = (not args.no_resume and chunk_path.exists()
                      and _line_count(chunk_path) == expected_count)
        if can_resume:
            if not stats_path.exists():
                raise RuntimeError(
                    f"{chunk_path} は有効ですが {stats_path} がありません")
            records = _load_chunk(chunk_path, expected_count, chunk_index)
            with stats_path.open("r", encoding="utf-8") as stream:
                stats = json.load(stream)
            chunk_elapsed = stats.get("elapsed_seconds", 0.0)
            status = "skip"
        else:
            last_error = None
            for attempt in range(2):
                chunk_started = time.perf_counter()
                try:
                    verbalizer, llm_mutator = _make_clients(args.fake)
                    kept, stats = generate_batch(
                        expected_count,
                        args.seed + chunk_index,
                        verbalizer=verbalizer,
                        llm_mutator=llm_mutator,
                        llm_mut_prob=args.llm_mut_prob,
                    )
                    records = _namespace_records(
                        kept, chunk_index, expected_count)
                    chunk_elapsed = time.perf_counter() - chunk_started
                    stats.update({
                        "chunk_index": chunk_index,
                        "seed": args.seed + chunk_index,
                        "requested": expected_count,
                        "written": len(records),
                        "elapsed_seconds": chunk_elapsed,
                    })
                    # chunk ファイルを最後に確定し、resume の完了マーカーにする。
                    _atomic_write_text(
                        stats_path,
                        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                    )
                    _write_chunk(chunk_path, records)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == 0:
                        print(
                            f"[chunk {chunk_index + 1}/{chunk_count}] 失敗: "
                            f"{type(exc).__name__}: {exc} / 1 回リトライします",
                            file=sys.stderr,
                            flush=True,
                        )
            if last_error is not None:
                print(
                    f"[中断] chunk_{chunk_index:03d} が再試行後も失敗しました: "
                    f"{type(last_error).__name__}: {last_error}\n"
                    f"原因を解消後、--resume でこのチャンクから再開できます。",
                    file=sys.stderr,
                )
                return 1
            status = "generated"

        _add_stats(totals, stats)
        _add_records(totals, records)
        chunk_summaries.append({
            "index": chunk_index,
            "seed": args.seed + chunk_index,
            "records": len(records),
            "status": status,
            "elapsed_seconds": chunk_elapsed,
        })
        _progress(chunk_index, chunk_count, len(records), stats,
                  chunk_elapsed, status)

    elapsed = time.perf_counter() - started
    summary = _build_summary(args, totals, chunk_summaries, elapsed)
    _atomic_write_text(
        args.out / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(args.out / "summary.md", _summary_markdown(summary))
    print(
        f"[done] 総件数={summary['total_records']} "
        f"経過秒={elapsed:.2f} 件/秒={summary['records_per_second']:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
