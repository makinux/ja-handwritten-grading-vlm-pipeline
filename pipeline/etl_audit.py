# -*- coding: utf-8 -*-
"""ETL1/ETL6/ETL9G/ETL9B の writer 被覆・OOV・画像差を監査する。

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

ETL1/ETL6 (M-type、1 レコード 2052 bytes):
  bytes   1-2: Data Number
          3-4: Character Code (2-byte ASCII mnemonic)
          5-6: Serial Sheet Number
            7: JIS X 0201 code
            8: extended EBCDIC code
          9-30: 評価・筆記者属性・通番・日付・画像レベル
         31-32: 未定義 (配布実データでは 0)
        33-2048: 64x63、4 bit/pixel、上位 nibble 先行の画像
      2049-2052: 不確定領域

M-type の文字変換表の選択はデータ family で決める。AIST の一覧と Q&A に
よれば CO-59→EUC 表を使うのは K-type の ETL2 だけであり、M-type の
ETL1/ETL6 は JIS X 0201/extended EBCDIC 表である。本実装は bytes 3-4 の
Character Code をクラス識別と既知の例外 (括弧、円記号等) の判定に使い、
通常は byte 7 の JIS X 0201 を Unicode へ変換する。半角カナは全角カタカナ
へ正規化する。Character Code ``/X`` は JIS/EBCDIC が共に 0 で、配布資料
から Unicode を一意に決められないため undecodable として報告する。

依頼時の要点には Serial Sheet Number=bytes 7-8、JIS Code=bytes 31-32 と
あったが、公式 M-type 表および配置済み実データは上記 (bytes 5-6/byte 7、
bytes 31-32 は未定義) で一致するため、公式位置を採用している。

フォーマット原典:
  https://etlcdb.db.aist.go.jp/etlcdb/etln/form_m.htm
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
import unicodedata

from PIL import Image, ImageDraw


ETLM_RECORD_SIZE = 2052
ETL9G_RECORD_SIZE = 8199
ETL9B_RECORD_SIZE = 576
ETL9_CLASS_COUNT = 3036
ETL_EXPECTED_CLASS_COUNTS = {"1": 99, "6": 114, "9G": 3036, "9B": 3036}
ETLM_IMAGE_SIZE = (64, 63)
ETL9G_IMAGE_SIZE = (128, 127)
ETL9B_IMAGE_SIZE = (64, 63)

PREVIOUS_OOV_CHARACTERS = tuple(
    "()+-/0123456789:=x\u00d7\u3001\u3002")

_FILE_PATTERNS = {
    "1": re.compile(
        r"^etl1c?[-_]?\d+(?:\.\d+)?(?:\.(?:bin|dat|raw))?$", re.I),
    "6": re.compile(
        r"^etl6c?[-_]?\d+(?:\.\d+)?(?:\.(?:bin|dat|raw))?$", re.I),
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
    character_code: bytes | None = None
    ebcdic_code: int | None = None

    @property
    def class_key(self):
        code = self.character_code if self.character_code is not None else self.jis_code
        return code.hex().upper()


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


@lru_cache(maxsize=None)
def decode_jis_x0201(code):
    """1-byte JIS X 0201 を Unicode へ変換し、半角カナを全角化する。"""
    if not isinstance(code, int) or not 0 <= code <= 0xFF:
        return None
    # JIS Roman は ASCII と異なる 2 箇所を明示する。
    if code == 0x5C:
        return "\u00a5"
    if code == 0x7E:
        return "\u203e"
    if 0x20 <= code <= 0x7D:
        return chr(code)
    if not 0xA1 <= code <= 0xDF:
        return None
    try:
        decoded = bytes([code]).decode("shift_jis")
    except UnicodeDecodeError:
        return None
    # NFKC は半角カナを全角へ変換する。単独の濁点/半濁点だけは結合文字に
    # なるため、独立グリフとして扱える spacing mark にする。
    if code == 0xDE:
        return "\u309b"
    if code == 0xDF:
        return "\u309c"
    normalized = unicodedata.normalize("NFKC", decoded)
    return normalized if len(normalized) == 1 else None


_M_CHARACTER_CODE_OVERRIDES = {
    b"$Y": "\u00a5",
    b"-6": " ",
    b"''": '"',
    b"((": "[",
    b"))": "]",
    b"(K": "\u300c",
    b")K": "\u300d",
    b",,": "\u309b",
    b",0": "\u309c",
    # /X は画像上も JIS X 0201 外の記号だが、公式 M-type レコード内の
    # JIS/extended-EBCDIC が 0 のため推測で Unicode を割り当てない。
    b"/X": None,
}


@lru_cache(maxsize=None)
def decode_m_character(character_code, jis_x0201_code):
    """M-type Character Code/JIS X 0201 の組を Unicode へ変換する。

    変換表の family 判定は次の通り。

    * K-type/ETL2: CO-59→EUC 表 (この関数の対象外)
    * M-type/ETL1, ETL6: JIS X 0201 と extended EBCDIC

    M-type の ASCII Character Code は正解クラスを表す 2-byte mnemonic
    なので、単純な ``"A "``/``"0 "``/``"( "`` はそれを優先する。
    これにより、配布レコードで JIS byte が欠損した少数例と、JIS byte の
    左右括弧が逆転している例も画像ラベルどおりになる。カナ mnemonic
    (``" A"``, ``"KA"`` 等) とその他記号は JIS X 0201 表を使う。
    CO-59 表を M-type に適用してはならない。
    """
    character_code = bytes(character_code)
    if len(character_code) != 2:
        return None
    if character_code in _M_CHARACTER_CODE_OVERRIDES:
        return _M_CHARACTER_CODE_OVERRIDES[character_code]
    if character_code[1] == 0x20 and 0x21 <= character_code[0] <= 0x7E:
        # 末尾 blank の 1 文字 mnemonic。先頭 blank のカナ母音とは別物。
        return chr(character_code[0])
    return decode_jis_x0201(jis_x0201_code)


def _decode_ascii(raw):
    return raw.rstrip(b"\x00 ").decode("ascii", errors="replace")


def _unpack_4bit(data, image_size=ETL9G_IMAGE_SIZE):
    pixels = bytearray(len(data) * 2)
    for index, value in enumerate(data):
        pixels[index * 2] = (value >> 4) * 17
        pixels[index * 2 + 1] = (value & 0x0F) * 17
    return Image.frombytes("L", image_size, bytes(pixels))


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


def parse_etlm_record(raw, family, include_image=False):
    """2052-byte ETL1/ETL6 M-type レコードをパースする。

    AIST 公式 ``form_m.htm`` の byte 位置を使う。Character Code の
    CO-59/JIS 変換表選択と Unicode 正規化規則は
    :func:`decode_m_character` の docstring を参照。
    """
    if family not in {"1", "6"}:
        raise ValueError(f"M-type family は '1' または '6' です: {family!r}")
    if len(raw) != ETLM_RECORD_SIZE:
        raise ValueError(
            f"ETL{family} M-type レコード長が不正です: {len(raw)} bytes")
    character_code = bytes(raw[2:4])
    jis_code = raw[6]
    image = (_unpack_4bit(raw[32:2048], ETLM_IMAGE_SIZE)
             if include_image else None)
    return ETLRecord(
        family=family,
        sheet_number=int.from_bytes(raw[4:6], "big"),
        jis_code=bytes([jis_code]),
        char=decode_m_character(character_code, jis_code),
        reading=_decode_ascii(character_code),
        serial_data_number=int.from_bytes(raw[12:16], "big"),
        metadata={
            "data_number": int.from_bytes(raw[0:2], "big"),
            "individual_evaluation": raw[8],
            "group_evaluation": raw[9],
            "sex": raw[10],
            "age": raw[11],
            "industry_code": int.from_bytes(raw[16:18], "big"),
            "occupation_code": int.from_bytes(raw[18:20], "big"),
            "sheet_gathering_date": int.from_bytes(raw[20:22], "big"),
            "scanning_date": int.from_bytes(raw[22:24], "big"),
            "sample_position_y": raw[24],
            "sample_position_x": raw[25],
            "minimum_scanned_level": raw[26],
            "maximum_scanned_level": raw[27],
        },
        image=image,
        character_code=character_code,
        ebcdic_code=raw[7],
    )


def parse_etl1_record(raw, include_image=False):
    """2052-byte ETL1 M-type record parser。"""
    return parse_etlm_record(raw, "1", include_image)


def parse_etl6_record(raw, include_image=False):
    """2052-byte ETL6 M-type record parser。"""
    return parse_etlm_record(raw, "6", include_image)


class AuditStats:
    """writer(シート番号)×字種クラスの疎な被覆行列を集計する。"""

    def __init__(self, family):
        self.family = family
        self.files = []
        self.records = 0
        self.skipped_dummy_records = 0
        self.skipped_model_records = 0
        self.invalid_sheet_records = 0
        self.sheet_numbers = set()
        self.coverage = defaultdict(set)
        self.classes = set()
        self.characters = set()
        self.undecodable_codes = Counter()
        self.undecodable_details = {}

    def add(self, record, writer_key=None):
        self.records += 1
        class_key = record.class_key
        self.classes.add(class_key)
        if record.char is None:
            self.undecodable_codes[class_key] += 1
            self.undecodable_details.setdefault(class_key, {
                "class_code": class_key,
                "character_code": (
                    record.character_code.decode("ascii", errors="replace")
                    if record.character_code is not None else None),
                "jis_code": record.jis_code.hex().upper(),
            })
        else:
            self.characters.add(record.char)
        if record.sheet_number <= 0:
            self.invalid_sheet_records += 1
        else:
            self.sheet_numbers.add(record.sheet_number)
            self.coverage[
                record.sheet_number if writer_key is None else writer_key
            ].add(class_key)

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
        expected_classes = ETL_EXPECTED_CLASS_COUNTS[self.family]
        complete_expected = sum(
            1 for values in self.coverage.values()
            if len(values) >= expected_classes)
        writer_proxy = (
            "deterministic file + sheet-form run occurrence (pseudo key)"
            if self.family == "9G" else "serial_sheet_number")
        return {
            "format": self.family,
            "files": self.files,
            "file_count": len(self.files),
            "records": self.records,
            "skipped_dummy_records": self.skipped_dummy_records,
            "skipped_model_records": self.skipped_model_records,
            "records_with_invalid_sheet_number": self.invalid_sheet_records,
            "writer_proxy": writer_proxy,
            "distinct_sheet_numbers": len(self.sheet_numbers),
            "distinct_writer_keys": writer_count,
            "distinct_character_classes": observed_classes,
            "decoded_unicode_characters": len(self.characters),
            "unicode_characters": sorted(
                self.characters, key=lambda char: tuple(map(ord, char))),
            "undecodable_jis_codes": [
                {**self.undecodable_details[code], "records": count}
                for code, count in sorted(self.undecodable_codes.items())
            ],
            "coverage": {
                "expected_classes": expected_classes,
                "expected_etl9_classes": (
                    ETL9_CLASS_COUNT if self.family in {"9G", "9B"} else None),
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
                "writers_with_all_expected_classes": complete_expected,
                "writers_with_all_3036_classes": (
                    complete_expected if expected_classes == ETL9_CLASS_COUNT else 0),
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
    """固定長 ETL ファイルを検出する (INFO は常に除外)。

    2052-byte M-type はファイル長で自動判別した後、パス/ファイル名中の
    ETL1/ETL6 を使って family を分ける。これは同一 M-type レイアウトだけ
    では ETL1 と ETL6 を識別できないためである。
    """
    found = {family: [] for family in ("1", "6", "9G", "9B")}
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.upper().endswith("INFO"):
            continue
        for family in ("9G", "9B"):
            pattern = _FILE_PATTERNS[family]
            if pattern.match(path.name):
                found[family].append(path)
                break
        else:
            size = path.stat().st_size
            if not size or size % ETLM_RECORD_SIZE:
                continue
            path_marker = "/".join(part.upper() for part in path.parts)
            family = None
            if re.search(r"(?:^|[/_])ETL1(?:C|[/_]|$)", path_marker):
                family = "1"
            elif re.search(r"(?:^|[/_])ETL6(?:C|[/_]|$)", path_marker):
                family = "6"
            if family is not None and _FILE_PATTERNS[family].match(path.name):
                found[family].append(path)
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
    record_sizes = {
        "1": ETLM_RECORD_SIZE,
        "6": ETLM_RECORD_SIZE,
        "9G": ETL9G_RECORD_SIZE,
        "9B": ETL9B_RECORD_SIZE,
    }
    record_size = record_sizes[family]
    size = path.stat().st_size
    if size == 0 or size % record_size:
        raise ValueError(
            f"{path}: ファイル長 {size} が {record_size} の倍数ではありません")
    record_count = size // record_size
    data_limit = (_etl9b_data_limit(record_count)
                  if family == "9B" else record_count)
    if family == "9G":
        parser = parse_etl9g_record
    elif family == "9B":
        parser = parse_etl9b_record
    else:
        parser = lambda raw, include_image=False: parse_etlm_record(
            raw, family, include_image)
    stats.files.append(str(path))
    print(f"[scan] {family} {path} ({record_count:,} records)", file=sys.stderr)

    last_9g_sheet = None
    sheet_run_occurrences = Counter()
    writer_key = None
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
            include_image = (
                samples.wants_g_image(basic) if family == "9G" else
                samples.wants_b_image(basic) if family == "9B" else False)
            record = parser(raw, include_image=True) if include_image else basic
            if family == "9G" and record.sheet_number != last_9g_sheet:
                sheet_run_occurrences[record.sheet_number] += 1
                writer_key = (
                    f"{path.name}#sheet={record.sheet_number}:"
                    f"occ={sheet_run_occurrences[record.sheet_number]}")
                last_9g_sheet = record.sheet_number
            stats.add(record, writer_key=writer_key if family == "9G" else None)
            if include_image:
                if family == "9G":
                    samples.add_g(record)
                elif family == "9B":
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


def _codepoint(char):
    return "+".join(f"U+{ord(value):04X}" for value in char)


def _previous_oov_coverage(stats):
    family_characters = {
        family: stats[family].characters for family in ("1", "6", "9G", "9B")
    }
    all_characters = set().union(*family_characters.values())
    m_characters = family_characters["1"] | family_characters["6"]

    def entry(char):
        return {
            "char": char,
            "codepoint": _codepoint(char),
            "families": [
                family for family in ("1", "6", "9G", "9B")
                if char in family_characters[family]
            ],
        }

    covered = [char for char in PREVIOUS_OOV_CHARACTERS if char in m_characters]
    remaining = [
        char for char in PREVIOUS_OOV_CHARACTERS if char not in all_characters]
    return {
        "baseline_distinct_characters": len(PREVIOUS_OOV_CHARACTERS),
        "baseline_characters": [entry(char) for char in PREVIOUS_OOV_CHARACTERS],
        "covered_by_etl1_etl6_count": len(covered),
        "covered_by_etl1_etl6": [entry(char) for char in covered],
        "remaining_count": len(remaining),
        "remaining_after_all_families": [entry(char) for char in remaining],
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
        help="展開済み ETL1/ETL6/ETL9G/ETL9B 固定長ファイルのディレクトリ")
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
    print("ETL 実データ監査サマリ")
    for family in ("1", "6", "9G", "9B"):
        data = report["formats"][family]
        coverage = data["coverage"]
        print(
            f"  ETL{family}: files={data['file_count']}, "
            f"records={data['records']:,}, "
            f"distinct sheets={data['distinct_sheet_numbers']:,}, "
            f"writer keys={data['distinct_writer_keys']:,}, "
            f"classes={data['distinct_character_classes']:,}, "
            f"decoded chars={data['decoded_unicode_characters']:,}, "
            f"all expected classes={coverage['writers_with_all_expected_classes']:,}")
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
    previous = report["previous_oov_21"]
    covered = "".join(item["char"] for item in previous["covered_by_etl1_etl6"])
    remaining = "".join(
        item["char"] for item in previous["remaining_after_all_families"])
    print(
        f"  前回 OOV 21 字種: ETL1/6 covered="
        f"{previous['covered_by_etl1_etl6_count']} ({covered}), "
        f"remaining={previous['remaining_count']} ({remaining})")
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
    if not any(files.values()):
        print(f"ETL 固定長ファイルがありません: {args.data_dir}\n")
        parser.print_help()
        return 0
    if args.corpus is not None and not args.corpus.is_file():
        parser.error(f"--corpus が見つかりません: {args.corpus}")

    families = ("1", "6", "9G", "9B")
    stats = {family: AuditStats(family) for family in families}
    samples = SampleMatcher(args.export_samples)
    try:
        # 先に 9G の比較候補を選び、9B 走査中に同じ JIS code/sheet を拾う。
        for family in families:
            for path in files[family]:
                _scan_file(path, family, stats[family], samples)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    available_characters = set().union(
        *(stats[family].characters for family in families))
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
            "schema_version": 2,
            "data_dir": str(args.data_dir),
            "formats": {
                family: stats[family].report() for family in families
            },
            "available_unicode_characters": len(available_characters),
            "available_unicode_character_list": sorted(
                available_characters, key=lambda char: tuple(map(ord, char))),
            "previous_oov_21": _previous_oov_coverage(stats),
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
