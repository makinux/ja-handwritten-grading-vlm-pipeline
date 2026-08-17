# -*- coding: utf-8 -*-
"""ETL9G/ETL9B の writer 被覆・OOV・画像差を監査する。

公知の固定長フォーマット(AIST ETL Character Database)を次のように読む。
整数と画像内の bit 順序はいずれも big-endian である。

ETL9G (G-type、1 レコード 8199 bytes):
  bytes   1-2: Serial Sheet Number (2-byte integer)
          3-4: JIS X 0208 区点コード
         5-12: JIS Typical Reading (ASCII)
        13-16: Serial Data Number (4-byte integer)
        17-30: 評価・筆記者属性・日付・シート内位置
        31-64: 未定義
       65-8192: 128x127、4 bit/pixel、上位 nibble 先行の画像
     8193-8199: 不確定領域

ETL9B (B-type、1 レコード 576 bytes):
  bytes   1-2: Serial Sheet Number (2-byte integer)
          3-4: JIS X 0208 区点コード
          5-8: JIS Typical Reading (ASCII)
          9-512: 64x63、1 bit/pixel、MSB 先行の画像
        513-576: 不確定領域
  各ファイルの先頭 1 レコードは dummy。ETL9B-5 の末尾 3036 件
  (分割版では ETL9B-5.2 の末尾)はモデル画像なので writer 監査から除外する。

フォーマット原典:
  https://etlcdb.db.aist.go.jp/etlcdb/etln/form_e9g.htm
  https://etlcdb.db.aist.go.jp/etlcdb/etln/form_e9b.htm

引数なし、または --data-dir に対象ファイルがない場合は使い方を表示して
正常終了する。依存は Python 標準ライブラリと Pillow のみ。
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import statistics
import sys

from PIL import Image, ImageDraw


ETL9G_RECORD_SIZE = 8199
ETL9B_RECORD_SIZE = 576
ETL9_CLASS_COUNT = 3036
ETL9G_IMAGE_SIZE = (128, 127)
ETL9B_IMAGE_SIZE = (64, 63)

_FILE_PATTERNS = {
    "9G": re.compile(
        r"^etl9g[-_]?\d+(?:\.\d+)?(?:\.(?:bin|dat|raw))?$", re.I),
    "9B": re.compile(
        r"^etl9b[-_]?\d+(?:\.\d+)?(?:\.(?:bin|dat|raw))?$", re.I),
}


@dataclass
class ETLRecord:
    """監査に必要な ETL 1 レコード分の情報。"""

    family: str
    sheet_number: int
    jis_code: bytes
    char: str | None
    reading: str
    serial_data_number: int | None = None
    metadata: dict | None = None
    image: Image.Image | None = None

    @property
    def class_key(self):
        return self.jis_code.hex().upper()


@lru_cache(maxsize=None)
def decode_jis_x0208(code):
    """2-byte の JIS X 0208 区点コードを Unicode 1 文字へ変換する。"""
    if len(code) != 2 or any(value < 0x21 or value > 0x7E for value in code):
        return None
    try:
        decoded = (b"\x1b$B" + code + b"\x1b(B").decode("iso2022_jp")
    except UnicodeDecodeError:
        return None
    return decoded if len(decoded) == 1 else None


def _decode_ascii(raw):
    return raw.rstrip(b"\x00 ").decode("ascii", errors="replace")


def _unpack_4bit(data):
    pixels = bytearray(len(data) * 2)
    for index, value in enumerate(data):
        pixels[index * 2] = (value >> 4) * 17
        pixels[index * 2 + 1] = (value & 0x0F) * 17
    return Image.frombytes("L", ETL9G_IMAGE_SIZE, bytes(pixels))


def _unpack_1bit(data):
    # Pillow mode "1" は各 byte の MSB から読む。L 化して比較 PNG の
    # リサイズ時にも二値値を明示的に維持する。
    return Image.frombytes("1", ETL9B_IMAGE_SIZE, data).convert("L")


def parse_etl9g_record(raw, include_image=False):
    """8199-byte ETL9G レコードをパースする。"""
    if len(raw) != ETL9G_RECORD_SIZE:
        raise ValueError(
            f"ETL9G レコード長が不正です: {len(raw)} bytes")
    code = bytes(raw[2:4])
    image = _unpack_4bit(raw[64:8192]) if include_image else None
    return ETLRecord(
        family="9G",
        sheet_number=int.from_bytes(raw[0:2], "big"),
        jis_code=code,
        char=decode_jis_x0208(code),
        reading=_decode_ascii(raw[4:12]),
        serial_data_number=int.from_bytes(raw[12:16], "big"),
        metadata={
            "individual_evaluation": raw[16],
            "group_evaluation": raw[17],
            "sex": raw[18],
            "age": raw[19],
            "industry_code": int.from_bytes(raw[20:22], "big"),
            "occupation_code": int.from_bytes(raw[22:24], "big"),
            "sheet_gathering_date": int.from_bytes(raw[24:26], "big"),
            "scanning_date": int.from_bytes(raw[26:28], "big"),
            "sample_position_x": raw[28],
            "sample_position_y": raw[29],
        },
        image=image,
    )


def parse_etl9b_record(raw, include_image=False):
    """576-byte ETL9B レコードをパースする。"""
    if len(raw) != ETL9B_RECORD_SIZE:
        raise ValueError(
            f"ETL9B レコード長が不正です: {len(raw)} bytes")
    code = bytes(raw[2:4])
    image = _unpack_1bit(raw[8:512]) if include_image else None
    return ETLRecord(
        family="9B",
        sheet_number=int.from_bytes(raw[0:2], "big"),
        jis_code=code,
        char=decode_jis_x0208(code),
        reading=_decode_ascii(raw[4:8]),
        image=image,
    )


class AuditStats:
    """writer(シート番号)×字種クラスの疎な被覆行列を集計する。"""

    def __init__(self, family):
        self.family = family
        self.files = []
        self.records = 0
        self.skipped_dummy_records = 0
        self.skipped_model_records = 0
        self.invalid_sheet_records = 0
        self.coverage = defaultdict(set)
        self.classes = set()
        self.characters = set()
        self.undecodable_codes = Counter()

    def add(self, record):
        self.records += 1
        class_key = record.class_key
        self.classes.add(class_key)
        if record.char is None:
            self.undecodable_codes[class_key] += 1
        else:
            self.characters.add(record.char)
        if record.sheet_number <= 0:
            self.invalid_sheet_records += 1
        else:
            self.coverage[record.sheet_number].add(class_key)

    @staticmethod
    def _percentile(sorted_values, ratio):
        if not sorted_values:
            return 0
        index = round((len(sorted_values) - 1) * ratio)
        return sorted_values[index]

    def report(self):
        counts = sorted(len(values) for values in self.coverage.values())
        observed_classes = len(self.classes)
        writer_count = len(self.coverage)
        pair_count = sum(counts)
        denominator = writer_count * observed_classes
        complete_observed = sum(
            1 for values in self.coverage.values()
            if len(values) == observed_classes)
        complete_expected = sum(
            1 for values in self.coverage.values()
            if len(values) >= ETL9_CLASS_COUNT)
        return {
            "format": self.family,
            "files": self.files,
            "file_count": len(self.files),
            "records": self.records,
            "skipped_dummy_records": self.skipped_dummy_records,
            "skipped_model_records": self.skipped_model_records,
            "records_with_invalid_sheet_number": self.invalid_sheet_records,
            "writer_proxy": "serial_sheet_number",
            "distinct_sheet_numbers": writer_count,
            "distinct_character_classes": observed_classes,
            "decoded_unicode_characters": len(self.characters),
            "undecodable_jis_codes": [
                {"jis_code": code, "records": count}
                for code, count in sorted(self.undecodable_codes.items())
            ],
            "coverage": {
                "expected_etl9_classes": ETL9_CLASS_COUNT,
                "writer_class_pairs": pair_count,
                "observed_matrix_density": (
                    pair_count / denominator if denominator else 0.0),
                "classes_per_writer": {
                    "min": min(counts) if counts else 0,
                    "p05": self._percentile(counts, 0.05),
                    "median": statistics.median(counts) if counts else 0,
                    "mean": statistics.fmean(counts) if counts else 0.0,
                    "p95": self._percentile(counts, 0.95),
                    "max": max(counts) if counts else 0,
                },
                "writers_with_all_observed_classes": complete_observed,
                "writers_with_all_3036_classes": complete_expected,
            },
        }


class SampleMatcher:
    """少数の 9G/9B 同一文字を、可能なら同一 sheet で対応付ける。"""

    def __init__(self, requested):
        self.requested = requested
        self.g_order = []
        self.g_by_code = {}
        self.b_exact = {}
        self.b_fallback = {}

    def wants_g_image(self, basic_record):
        return (len(self.g_order) < self.requested
                and basic_record.class_key not in self.g_by_code)

    def add_g(self, record):
        if record.class_key in self.g_by_code:
            return
        self.g_order.append(record.class_key)
        self.g_by_code[record.class_key] = record

    def wants_b_image(self, basic_record):
        code = basic_record.class_key
        if code not in self.g_by_code:
            return False
        target = self.g_by_code[code]
        return (code not in self.b_fallback
                or (basic_record.sheet_number == target.sheet_number
                    and code not in self.b_exact))

    def add_b(self, record):
        code = record.class_key
        self.b_fallback.setdefault(code, record)
        target = self.g_by_code[code]
        if record.sheet_number == target.sheet_number:
            self.b_exact.setdefault(code, record)

    def pairs(self):
        result = []
        for code in self.g_order:
            b_record = self.b_exact.get(code) or self.b_fallback.get(code)
            if b_record is not None:
                result.append((self.g_by_code[code], b_record))
        return result


def _discover_files(data_dir):
    found = {"9G": [], "9B": []}
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        for family, pattern in _FILE_PATTERNS.items():
            if pattern.match(path.name):
                found[family].append(path)
                break
    return found


def _etl9b_data_limit(record_count):
    """dummy 後の実筆レコード上限を返す。"""
    data_count = max(0, record_count - 1)
    if data_count == 124476:  # ETL9B-5: 121440 + model 3036
        return 121440
    if data_count == 63756:   # ETL9B-5.2: 60720 + model 3036
        return 60720
    return data_count


def _scan_file(path, family, stats, samples):
    record_size = (ETL9G_RECORD_SIZE if family == "9G"
                   else ETL9B_RECORD_SIZE)
    size = path.stat().st_size
    if size == 0 or size % record_size:
        raise ValueError(
            f"{path}: ファイル長 {size} が {record_size} の倍数ではありません")
    record_count = size // record_size
    data_limit = record_count if family == "9G" else _etl9b_data_limit(
        record_count)
    parser = parse_etl9g_record if family == "9G" else parse_etl9b_record
    stats.files.append(str(path))
    print(f"[scan] {family} {path} ({record_count:,} records)", file=sys.stderr)

    with path.open("rb") as stream:
        for record_index in range(record_count):
            raw = stream.read(record_size)
            if len(raw) != record_size:
                raise ValueError(f"{path}: record {record_index} が途中で終了しました")
            if family == "9B" and record_index == 0:
                stats.skipped_dummy_records += 1
                continue
            data_index = record_index - 1 if family == "9B" else record_index
            if family == "9B" and data_index >= data_limit:
                stats.skipped_model_records += 1
                continue

            basic = parser(raw, include_image=False)
            include_image = (samples.wants_g_image(basic) if family == "9G"
                             else samples.wants_b_image(basic))
            record = parser(raw, include_image=True) if include_image else basic
            stats.add(record)
            if include_image:
                if family == "9G":
                    samples.add_g(record)
                else:
                    samples.add_b(record)


def _audit_corpus(corpus_path, available_characters):
    total = 0
    oov_count = 0
    oov = Counter()
    line_count = 0
    with corpus_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{corpus_path}:{line_number}: JSON が不正です: {exc}") from exc
            text = value.get("text") if isinstance(value, dict) else None
            if not isinstance(text, str):
                raise ValueError(
                    f"{corpus_path}:{line_number}: text フィールドが文字列ではありません")
            line_count += 1
            for char in text:
                # 空白はレンダリング対象グリフではないため OOV 分母から除く。
                if char.isspace():
                    continue
                total += 1
                if char not in available_characters:
                    oov_count += 1
                    oov[char] += 1
    return {
        "path": str(corpus_path),
        "jsonl_records": line_count,
        "non_whitespace_characters": total,
        "oov_occurrences": oov_count,
        "oov_rate": oov_count / total if total else 0.0,
        "distinct_oov_characters": len(oov),
        "oov_characters": [
            {
                "char": char,
                "codepoint": "+".join(f"U+{ord(value):04X}" for value in char),
                "count": count,
            }
            for char, count in sorted(oov.items(), key=lambda item: ord(item[0]))
        ],
    }


def _safe_label_char(char):
    return char if char is not None else "(decode-error)"


def _export_samples(samples, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, (g_record, b_record) in enumerate(samples.pairs(), 1):
        label_height = 24
        gap = 12
        canvas = Image.new("L", (128 * 2 + gap, label_height + 127), 255)
        draw = ImageDraw.Draw(canvas)
        draw.text((2, 5), f"9G {g_record.class_key} sheet={g_record.sheet_number}",
                  fill=0)
        draw.text((128 + gap + 2, 5),
                  f"9B sheet={b_record.sheet_number}", fill=0)
        canvas.paste(g_record.image, (0, label_height))
        b_scaled = b_record.image.resize((128, 126), Image.Resampling.NEAREST)
        canvas.paste(b_scaled, (128 + gap, label_height))
        filename = f"sample_{index:03d}_{g_record.class_key}.png"
        path = output_dir / filename
        canvas.save(path)
        reports.append({
            "file": str(path),
            "jis_code": g_record.class_key,
            "char": _safe_label_char(g_record.char),
            "etl9g_sheet_number": g_record.sheet_number,
            "etl9b_sheet_number": b_record.sheet_number,
            "same_sheet_number": (
                g_record.sheet_number == b_record.sheet_number),
        })
    return reports


def _write_json(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        help="展開済み ETL9G/ETL9B 固定長ファイルのディレクトリ")
    parser.add_argument(
        "--corpus", type=Path,
        help="各行に text フィールドを持つ監査対象 JSONL")
    parser.add_argument(
        "--output", type=Path, default=Path("etl_audit_report.json"),
        help="監査 JSON の出力先(既定: etl_audit_report.json)")
    parser.add_argument(
        "--export-samples", type=int, default=0, metavar="N",
        help="9G/9B の同一文字比較 PNG を N 組出力")
    parser.add_argument(
        "--sample-dir", type=Path,
        help="比較 PNG 出力先(既定: JSON 出力先と同階層の etl_audit_samples)")
    return parser


def _print_summary(report, output_path):
    print("ETL9 実データ監査サマリ")
    for family in ("9G", "9B"):
        data = report["formats"][family]
        coverage = data["coverage"]
        print(
            f"  ETL{family}: files={data['file_count']}, "
            f"records={data['records']:,}, "
            f"distinct sheets={data['distinct_sheet_numbers']:,}, "
            f"classes={data['distinct_character_classes']:,}, "
            f"all 3036 classes={coverage['writers_with_all_3036_classes']:,}")
        distribution = coverage["classes_per_writer"]
        print(
            "    classes/writer: "
            f"min={distribution['min']}, median={distribution['median']}, "
            f"mean={distribution['mean']:.2f}, max={distribution['max']}")
    corpus = report.get("corpus")
    if corpus is not None:
        print(
            f"  OOV: {corpus['oov_occurrences']:,}/"
            f"{corpus['non_whitespace_characters']:,} "
            f"({corpus['oov_rate']:.4%}), "
            f"distinct={corpus['distinct_oov_characters']:,}")
    else:
        print("  OOV: --corpus 未指定")
    exports = report["sample_exports"]
    print(
        f"  比較 PNG: {exports['exported']}/{exports['requested']} 組")
    print(f"  JSON: {output_path}")


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.export_samples < 0:
        parser.error("--export-samples は 0 以上で指定してください")
    if args.data_dir is None:
        parser.print_help()
        return 0
    if not args.data_dir.is_dir():
        print(f"ETL データディレクトリがありません: {args.data_dir}\n")
        parser.print_help()
        return 0

    files = _discover_files(args.data_dir)
    if not files["9G"] and not files["9B"]:
        print(f"ETL9G/ETL9B 固定長ファイルがありません: {args.data_dir}\n")
        parser.print_help()
        return 0
    if args.corpus is not None and not args.corpus.is_file():
        parser.error(f"--corpus が見つかりません: {args.corpus}")

    stats = {family: AuditStats(family) for family in ("9G", "9B")}
    samples = SampleMatcher(args.export_samples)
    try:
        # 先に 9G の比較候補を選び、9B 走査中に同じ JIS code/sheet を拾う。
        for family in ("9G", "9B"):
            for path in files[family]:
                _scan_file(path, family, stats[family], samples)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    available_characters = stats["9G"].characters | stats["9B"].characters
    corpus_report = None
    if args.corpus is not None:
        try:
            corpus_report = _audit_corpus(args.corpus, available_characters)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    sample_dir = args.sample_dir or args.output.with_name("etl_audit_samples")
    try:
        exported = (_export_samples(samples, sample_dir)
                    if args.export_samples else [])
        report = {
            "schema_version": 1,
            "data_dir": str(args.data_dir),
            "formats": {
                family: stats[family].report() for family in ("9G", "9B")
            },
            "available_unicode_characters": len(available_characters),
            "corpus": corpus_report,
            "sample_exports": {
                "requested": args.export_samples,
                "exported": len(exported),
                "directory": str(sample_dir) if args.export_samples else None,
                "pairs": exported,
            },
        }
        _write_json(args.output, report)
    except OSError as exc:
        print(f"error: 出力に失敗しました: {exc}", file=sys.stderr)
        return 1

    _print_summary(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
