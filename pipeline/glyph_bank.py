# -*- coding: utf-8 -*-
"""ETL 実筆グリフの決定論的インデックスと遅延ローダ。

インデックス対象は 9G、ETL1、ETL6。9B は 9G の二値派生データなので、
同じ文字集合の既定ソースには高階調の 9G を使い、9B は登録しない。
画像はインデックス構築時には読まず、``GlyphBank.get`` で選ばれた 1 レコード
だけを offset read する。返す L 画像は背景 0、インク白である。
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from PIL import Image

try:  # ``python pipeline/glyph_bank.py`` と package import の両方に対応。
    from . import etl_audit
except ImportError:  # pragma: no cover - script execution path
    import etl_audit


INDEX_SCHEMA_VERSION = 1
INDEX_FAMILIES = ("9G", "1", "6")
DEFAULT_INDEX_PATH = Path(__file__).resolve().parents[1] / "data/etl/glyph_index.json"


def _unicode_sort_key(char):
    return tuple(map(ord, char))


def _relative_file(path, data_dir):
    return path.resolve().relative_to(data_dir.resolve()).as_posix()


def _scan_index_file(path, family, data_dir, glyphs, counters):
    relative_file = _relative_file(path, data_dir)
    if family == "9G":
        record_size = etl_audit.ETL9G_RECORD_SIZE
    else:
        record_size = etl_audit.ETLM_RECORD_SIZE
    size = path.stat().st_size
    if not size or size % record_size:
        raise ValueError(
            f"{path}: ファイル長 {size} が record size {record_size} の倍数ではありません")

    last_sheet = None
    sheet_occurrences = Counter()
    writer_key = None
    with path.open("rb") as stream:
        for record_index in range(size // record_size):
            raw = stream.read(record_size)
            if len(raw) != record_size:
                raise ValueError(f"{path}: record {record_index} が途中で終了しました")
            if family == "9G":
                record = etl_audit.parse_etl9g_record(raw, include_image=False)
                # 9G の Serial Sheet Number は 1..20 のシート様式 ID であり
                # writer ID ではない。同じ様式の run 出現順とファイル名を足し、
                # 決定論的かつ一意な将来交換可能キーを作る。
                if record.sheet_number != last_sheet:
                    sheet_occurrences[record.sheet_number] += 1
                    writer_key = (
                        f"{relative_file}#sheet={record.sheet_number}:"
                        f"occ={sheet_occurrences[record.sheet_number]}")
                    last_sheet = record.sheet_number
            else:
                record = etl_audit.parse_etlm_record(
                    raw, family, include_image=False)
                # ETL1/6 では公式 Serial Sheet Number をそのまま writer key にする。
                writer_key = record.sheet_number

            counters["records_scanned"][family] += 1
            counters["writer_keys"][family].add(writer_key)
            if record.char is None:
                counters["undecodable_records"][family] += 1
                continue
            glyphs[record.char].append({
                "family": family,
                "file": relative_file,
                "offset": record_index * record_size,
                "writer_key": writer_key,
            })
            counters["indexed_records"][family] += 1


def build_index(data_dir, out_path):
    """全 9G/ETL1/ETL6 レコードの char→candidate JSON を構築する。

    戻り値は所要時間、サイズ、件数を含む小さな summary dict。候補配列は
    family、ファイル、offset の順が常に一定になるように走査する。
    """
    started = time.perf_counter()
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"ETL data directory がありません: {data_dir}")
    files = etl_audit._discover_files(data_dir)
    missing = [family for family in INDEX_FAMILIES if not files[family]]
    if missing:
        raise FileNotFoundError(
            f"glyph index に必要な family がありません: {', '.join(missing)}")

    glyphs = defaultdict(list)
    counters = {
        "records_scanned": Counter(),
        "indexed_records": Counter(),
        "undecodable_records": Counter(),
        "writer_keys": {family: set() for family in INDEX_FAMILIES},
    }
    for family in INDEX_FAMILIES:
        for path in files[family]:
            _scan_index_file(path, family, data_dir, glyphs, counters)

    ordered_glyphs = {
        char: glyphs[char] for char in sorted(glyphs, key=_unicode_sort_key)
    }
    document = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "data_dir": ".",
        "families": list(INDEX_FAMILIES),
        "record_counts": {
            name: {family: counter[family] for family in INDEX_FAMILIES}
            for name, counter in counters.items() if name != "writer_keys"
        },
        "distinct_writer_keys": {
            family: len(counters["writer_keys"][family])
            for family in INDEX_FAMILIES
        },
        "distinct_characters": len(ordered_glyphs),
        "glyphs": ordered_glyphs,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(out_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
    os.replace(temporary, out_path)
    elapsed = time.perf_counter() - started
    return {
        "path": str(out_path),
        "elapsed_seconds": elapsed,
        "size_bytes": out_path.stat().st_size,
        "distinct_characters": len(ordered_glyphs),
        "records_scanned": sum(counters["records_scanned"].values()),
        "indexed_records": sum(counters["indexed_records"].values()),
        "undecodable_records": sum(counters["undecodable_records"].values()),
        "record_counts": document["record_counts"],
        "distinct_writer_keys": document["distinct_writer_keys"],
    }


def _percentile(sorted_values, ratio):
    if not sorted_values:
        return 0
    return sorted_values[round((len(sorted_values) - 1) * ratio)]


def _preprocess_glyph(image, family):
    """背景推定、0 基準化、bbox crop、階調保持の contrast 正規化を行う。"""
    image = image.convert("L")
    width, height = image.size
    values = list(image.getdata())
    border = []
    for y in range(height):
        for x in range(width):
            if x < 2 or y < 2 or x >= width - 2 or y >= height - 2:
                border.append(values[y * width + x])
    border.sort()
    background = statistics.median(border) if border else 0
    # ETL の 4-bit 値は L 化後に 17 刻み。9G には隣接セル由来の筆跡片が
    # border に入る例があるため、高位 percentile を背景値としては使わない。
    # 背景中央値の 1 階調上だけを floor にし、薄い鉛筆線を残す。
    if family == "9G":
        floor = min(254, int(background) + 17)
    else:
        # ETL1/6 は従来どおり border の高位側をノイズ floor にする。
        floor = min(254, max(
            int(background), _percentile(border, 0.99) + 17))

    above = [max(0, value - floor) for value in values]
    central_margin = (
        max(2, min(image.size) // 16)
        if family == "9G" and min(image.size) >= 32 else 0)
    if central_margin:
        # white point も端の混入片ではなく、本来の文字がある中央領域の
        # インク分布から求める。
        positive = sorted(
            above[y * width + x]
            for y in range(central_margin, height - central_margin)
            for x in range(central_margin, width - central_margin)
            if above[y * width + x] > 0)
    else:
        positive = sorted(value for value in above if value > 0)
    if not positive:
        positive = sorted(value for value in above if value > 0)
    if not positive:
        return Image.new("L", (1, 1), 0)
    # 鉛筆筆跡の多い 9G は高位側の外れ値に引っ張られると、縮小後の中間階調が
    # 掠れてしまう。インク画素 p95 を white point とし、軽い gamma で中間調を
    # 持ち上げる。ETL1/6 は従来の正規化を保つ。
    if family == "9G":
        white_point_ratio = 0.95
        gamma = 0.75
    else:
        white_point_ratio = 0.985
        gamma = 1.0
    white_point = max(1, _percentile(positive, white_point_ratio))
    normalized = bytearray(len(above))
    for index, value in enumerate(above):
        linear = min(1.0, value / white_point)
        scaled = min(255, round(255 * (linear ** gamma)))
        # ごく弱い残留背景を除き、crop がスキャナノイズまで含まないようにする。
        normalized[index] = 0 if scaled < 10 else scaled
    result = Image.frombytes("L", image.size, bytes(normalized))
    if central_margin:
        # 9G の一部レコードには左右端に隣接セルの筆跡片が入る。筆跡値は
        # 変更せず、中央領域だけから crop bbox を決めて 40px 級への縮小時に
        # 本体が必要以上に小さくならないようにする。
        margin = central_margin
        central = result.crop((
            margin, margin, result.width - margin, result.height - margin))
        central_bbox = central.getbbox()
        if central_bbox is None:
            bbox = result.getbbox()
        else:
            pad = 2
            bbox = (
                max(0, central_bbox[0] + margin - pad),
                max(0, central_bbox[1] + margin - pad),
                min(result.width, central_bbox[2] + margin + pad),
                min(result.height, central_bbox[3] + margin + pad),
            )
    else:
        bbox = result.getbbox()
    return result.crop(bbox) if bbox is not None else Image.new("L", (1, 1), 0)


class GlyphBank:
    """JSON index を用いた決定論的 ETL glyph loader。

    ``get(char, pseudo_writer_id)`` は ``(PIL.Image[L] | None, metadata | None)``
    を返す。候補がないときだけ ``(None, None)``。ファイルハンドルと前処理済み
    glyph は独立した LRU に置き、全画像をメモリへ載せない。
    """

    def __init__(self, index_path=DEFAULT_INDEX_PATH, data_dir=None,
                 cache_size=512, max_open_files=8):
        self.index_path = Path(index_path)
        if not self.index_path.is_file():
            raise FileNotFoundError(f"glyph index がありません: {self.index_path}")
        with self.index_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if document.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise ValueError(
                f"未対応 glyph index schema: {document.get('schema_version')!r}")
        glyphs = document.get("glyphs")
        if not isinstance(glyphs, dict):
            raise ValueError("glyph index に glyphs mapping がありません")
        self.glyphs = glyphs
        self.data_dir = (Path(data_dir) if data_dir is not None
                         else self.index_path.parent)
        self.cache_size = max(1, int(cache_size))
        self.max_open_files = max(1, int(max_open_files))
        self._glyph_cache = OrderedDict()
        self._handles = OrderedDict()

    def __contains__(self, char):
        return char in self.glyphs

    def _handle(self, relative_file):
        handle = self._handles.pop(relative_file, None)
        if handle is None:
            handle = (self.data_dir / Path(relative_file)).open("rb")
        self._handles[relative_file] = handle
        while len(self._handles) > self.max_open_files:
            _old_path, old_handle = self._handles.popitem(last=False)
            old_handle.close()
        return handle

    def _load(self, candidate):
        cache_key = (
            candidate["family"], candidate["file"], int(candidate["offset"]))
        cached = self._glyph_cache.pop(cache_key, None)
        if cached is not None:
            self._glyph_cache[cache_key] = cached
            return cached.copy()

        family, relative_file, offset = cache_key
        record_size = (etl_audit.ETL9G_RECORD_SIZE if family == "9G"
                       else etl_audit.ETLM_RECORD_SIZE)
        handle = self._handle(relative_file)
        handle.seek(offset)
        raw = handle.read(record_size)
        if len(raw) != record_size:
            raise ValueError(
                f"{relative_file}@{offset}: glyph record が途中で終了しました")
        if family == "9G":
            image = etl_audit._unpack_4bit(
                raw[64:8192], etl_audit.ETL9G_IMAGE_SIZE)
        elif family in {"1", "6"}:
            image = etl_audit._unpack_4bit(
                raw[32:2048], etl_audit.ETLM_IMAGE_SIZE)
        else:
            raise ValueError(f"index 内の未対応 family: {family!r}")
        image = _preprocess_glyph(image, family)
        self._glyph_cache[cache_key] = image
        while len(self._glyph_cache) > self.cache_size:
            self._glyph_cache.popitem(last=False)
        return image.copy()

    def get(self, char, pseudo_writer_id):
        candidates = self.glyphs.get(char)
        if not candidates:
            return None, None
        payload = (
            f"writer:{len(str(pseudo_writer_id))}:{pseudo_writer_id}|"
            f"char:{len(char)}:{char}").encode("utf-8")
        selected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        candidate = candidates[selected % len(candidates)]
        image = self._load(candidate)
        return image, {
            "family": candidate["family"],
            "writer_key": candidate["writer_key"],
        }

    def close(self):
        while self._handles:
            _path, handle = self._handles.popitem(last=False)
            handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):  # pragma: no cover - best-effort interpreter cleanup
        try:
            self.close()
        except Exception:
            pass


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_INDEX_PATH.parent)
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX_PATH)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    summary = build_index(args.data_dir, args.output)
    print(
        f"glyph index: {summary['indexed_records']:,} records, "
        f"{summary['distinct_characters']:,} chars, "
        f"{summary['size_bytes'] / (1024 * 1024):.2f} MiB, "
        f"{summary['elapsed_seconds']:.3f}s -> {summary['path']}")
    if summary["undecodable_records"]:
        print(f"undecodable skipped: {summary['undecodable_records']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
