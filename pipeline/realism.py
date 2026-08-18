# -*- coding: utf-8 -*-
"""決定論的な手続き的リアリズム変形。

``REALISM_LIMITS`` は、強い変形で正解文字そのものが別字形に見えることを
避ける、**字形ラベル保護のための上限**である。呼び出し側の設定値は必ず
この上限（波長だけは下限）で検証し、暗黙の clamp は行わない。

乱数に相当する値はすべて SHA-256 による名前付き seed から直接導出する。
一つの PRNG を順番に消費しないため、変形の追加・無効化で他の変形結果が
変わらず、同じ疑似筆者 ID・行キー・安定文字 ID の双子は完全に共有される。
"""
from dataclasses import dataclass
import hashlib
import json
import math

from PIL import Image, ImageChops, ImageFilter


# 字形ラベル保護のための上限。公開設定名と一対一に対応させる。
REALISM_LIMITS = {
    "baseline_amplitude_px": 6.0,
    "baseline_min_wavelength_px": 300.0,
    "size_drift_ratio": 0.08,
    "rotation_degrees": 4.0,
    "elastic_grid_size": 4,
    "elastic_displacement_ratio": 0.04,
    "bleed_dilation_px": 1,
    "bleed_blur_radius_px": 0.8,
    "bleed_strength": 0.32,
    "fading_strength": 0.16,
    "spacing_offset_ratio": 0.06,
}


DEFAULT_REALISM = {
    "baseline_amplitude_px": 4.5,
    "baseline_wavelength_px": 420.0,
    "size_drift_ratio": 0.055,
    "rotation_degrees": 2.5,
    "elastic_grid_size": 4,
    "elastic_displacement_ratio": 0.025,
    "bleed_dilation_px": 1,
    "bleed_blur_radius_px": 0.55,
    "bleed_strength": 0.22,
    "fading_strength": 0.10,
    "spacing_offset_ratio": 0.045,
}


_UPPER_LIMIT_KEYS = {
    key for key in REALISM_LIMITS
    if key not in {"baseline_min_wavelength_px", "elastic_grid_size"}
}
_SETTING_KEYS = set(DEFAULT_REALISM) | {"enabled"}


def _seed_from_parts(*parts):
    """m3_render._seed_from_parts と同じ形式で 64-bit seed を作る。"""
    payload = "".join(
        f"{type(part).__name__}:{len(str(part))}:{part}|" for part in parts
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    )


def _unit_from_parts(*parts):
    # 上位 53 bit を使い、Python/NumPy の PRNG 実装には依存しない。
    return (_seed_from_parts(*parts) >> 11) / float(1 << 53)


def _signed_from_parts(*parts):
    return 2.0 * _unit_from_parts(*parts) - 1.0


def _number(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"realism.{name} は有限の数値で指定してください")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"realism.{name} は有限の数値で指定してください")
    return result


def validate_settings(settings):
    """設定を検証・正規化し、上限外なら ``ValueError`` を送出する。"""
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise ValueError("realism は None または dict で指定してください")
    unknown = sorted(set(settings) - _SETTING_KEYS)
    if unknown:
        raise ValueError(f"未知の realism 設定: {', '.join(unknown)}")
    if "enabled" in settings and not isinstance(settings["enabled"], bool):
        raise ValueError("realism.enabled は bool で指定してください")
    if settings.get("enabled") is False:
        return None

    result = dict(DEFAULT_REALISM)
    result.update({key: value for key, value in settings.items()
                   if key != "enabled"})

    for key in _UPPER_LIMIT_KEYS:
        value = _number(key, result[key])
        if not 0.0 <= value <= float(REALISM_LIMITS[key]):
            raise ValueError(
                f"realism.{key}={value} は字形保護上限 "
                f"{REALISM_LIMITS[key]} を超えています"
            )
        result[key] = value

    wavelength = _number(
        "baseline_wavelength_px", result["baseline_wavelength_px"]
    )
    minimum = REALISM_LIMITS["baseline_min_wavelength_px"]
    if wavelength < minimum:
        raise ValueError(
            f"realism.baseline_wavelength_px={wavelength} は字形保護下限 "
            f"{minimum} 未満です"
        )
    result["baseline_wavelength_px"] = wavelength

    grid_size = result["elastic_grid_size"]
    if (isinstance(grid_size, bool) or not isinstance(grid_size, int)
            or grid_size != REALISM_LIMITS["elastic_grid_size"]):
        raise ValueError("realism.elastic_grid_size は 4 固定です")
    dilation = result["bleed_dilation_px"]
    if (not float(dilation).is_integer()
            or int(dilation) > REALISM_LIMITS["bleed_dilation_px"]):
        raise ValueError("realism.bleed_dilation_px は 0 または 1 です")
    result["bleed_dilation_px"] = int(dilation)
    return result


@dataclass(frozen=True)
class RealismProfile:
    """疑似筆者ごとに固定された実効強度プロファイル。"""

    seed_identity: str
    profile_id: str
    parameters: dict


def _writer_scaled(seed_identity, name, configured, low=0.68):
    if configured == 0:
        return 0.0
    factor = low + (1.0 - low) * _unit_from_parts(
        seed_identity, "profile", name, "writer-strength"
    )
    return configured * factor


def build_profile(settings, seed_identity):
    """検証済み設定から、順序非依存な筆者別プロファイルを作る。"""
    normalized = validate_settings(settings)
    if normalized is None:
        return None

    parameters = dict(normalized)
    for name in (
        "baseline_amplitude_px",
        "size_drift_ratio",
        "rotation_degrees",
        "elastic_displacement_ratio",
        "bleed_blur_radius_px",
        "bleed_strength",
        "fading_strength",
        "spacing_offset_ratio",
    ):
        parameters[name] = _writer_scaled(
            seed_identity, name, float(normalized[name])
        )
    # 長い方へのみ筆者差を付け、300 px 下限を常に維持する。
    parameters["baseline_wavelength_px"] = normalized[
        "baseline_wavelength_px"
    ] * (1.0 + 0.18 * _unit_from_parts(
        seed_identity, "profile", "baseline_wavelength_px", "writer-strength"
    ))

    canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(
        f"{seed_identity}|{canonical}".encode("utf-8")
    ).hexdigest()[:16]
    return RealismProfile(
        seed_identity=str(seed_identity),
        profile_id=f"realism-{digest}",
        parameters=parameters,
    )


def metadata_summary(profile):
    """render metadata 用の JSON 化可能な要約を返す。"""
    if profile is None:
        return {"applied": False, "strength_profile_id": None}
    return {
        "applied": True,
        "strength_profile_id": profile.profile_id,
        "parameters": {
            key: (round(value, 6) if isinstance(value, float) else value)
            for key, value in profile.parameters.items()
        },
    }


def _catmull_rom(p0, p1, p2, p3, t):
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
    )


def _smooth_field(profile, key, row, x, name, amplitude, wavelength):
    """等間隔制御点を Catmull-Rom 補間する低周波スプライン。"""
    if amplitude == 0:
        return 0.0
    coordinate = x / wavelength
    knot = math.floor(coordinate)
    t = coordinate - knot
    controls = []
    for control in range(knot - 1, knot + 3):
        # 制御点 ID はレコード間で安定した仮想文字 ID として扱う。
        stable_control_id = f"row:{row}:control:{control}"
        controls.append(_signed_from_parts(
            profile.seed_identity, key, stable_control_id, name
        ))
    value = _catmull_rom(*controls, t)
    return max(-amplitude, min(amplitude, amplitude * value))


def placement_parameters(profile, key, stable_char_id, row, x, pitch,
                         pitch_mode):
    """文字のアンカーを動かさず、セル内配置と緩やかな drift を返す。"""
    if profile is None:
        return {
            "baseline_offset_px": 0.0,
            "spacing_offset_px": 0.0,
            "size_scale": 1.0,
            "rotation_degrees": 0.0,
        }
    values = profile.parameters
    wavelength = values["baseline_wavelength_px"]
    baseline = _smooth_field(
        profile, key, row, x, "baseline-spline",
        values["baseline_amplitude_px"], wavelength,
    )
    size_smooth = _smooth_field(
        profile, key, row, x, "size-drift-spline", 1.0,
        max(REALISM_LIMITS["baseline_min_wavelength_px"], wavelength * 1.2),
    )
    # 安定文字 ID 成分は 10% に留め、隣接文字の大きさが飛ばないようにする。
    size_local = _signed_from_parts(
        profile.seed_identity, key, stable_char_id, "size-drift-local"
    )
    size_value = max(-1.0, min(1.0, 0.9 * size_smooth + 0.1 * size_local))
    size_scale = 1.0 + values["size_drift_ratio"] * size_value
    rotation = values["rotation_degrees"] * _signed_from_parts(
        profile.seed_identity, key, stable_char_id, "rotation"
    )
    spacing = 0.0
    if pitch_mode == "grid":
        spacing = pitch * values["spacing_offset_ratio"] * _signed_from_parts(
            profile.seed_identity, key, stable_char_id, "spacing-offset"
        )
    return {
        "baseline_offset_px": baseline,
        "spacing_offset_px": spacing,
        "size_scale": size_scale,
        "rotation_degrees": rotation,
    }


def _crop_mask(mask, offset_x, offset_y):
    bbox = mask.getbbox()
    if bbox is None:
        return mask, offset_x, offset_y
    return (
        mask.crop(bbox),
        offset_x + bbox[0],
        offset_y + bbox[1],
    )


def _resize_about_center(mask, scale, offset_x, offset_y):
    if abs(scale - 1.0) < 1e-12:
        return _crop_mask(mask, offset_x, offset_y)
    old_width, old_height = mask.size
    new_size = (
        max(1, round(old_width * scale)),
        max(1, round(old_height * scale)),
    )
    resized = mask.resize(new_size, Image.Resampling.LANCZOS)
    offset_x += (old_width - new_size[0]) / 2.0
    offset_y += (old_height - new_size[1]) / 2.0
    return _crop_mask(resized, offset_x, offset_y)


def _rotate_about_center(mask, angle, offset_x, offset_y):
    if abs(angle) < 1e-12:
        return _crop_mask(mask, offset_x, offset_y)
    old_width, old_height = mask.size
    rotated = mask.rotate(
        angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=0
    )
    offset_x += (old_width - rotated.width) / 2.0
    offset_y += (old_height - rotated.height) / 2.0
    return _crop_mask(rotated, offset_x, offset_y)


def _elastic_warp(mask, profile, key, stable_char_id, ratio,
                  grid_size, offset_x, offset_y):
    if ratio == 0:
        return _crop_mask(mask, offset_x, offset_y)
    maximum = mask.width * ratio
    pad = max(2, math.ceil(maximum) + 2)
    padded = Image.new("L", (mask.width + 2 * pad, mask.height + 2 * pad), 0)
    padded.paste(mask, (pad, pad))
    offset_x -= pad
    offset_y -= pad
    width, height = padded.size

    dx_grid = [[0.0] * grid_size for _ in range(grid_size)]
    dy_grid = [[0.0] * grid_size for _ in range(grid_size)]
    for grid_y in range(grid_size):
        for grid_x in range(grid_size):
            point = f"grid:{grid_y}:{grid_x}"
            dx_grid[grid_y][grid_x] = maximum * _signed_from_parts(
                profile.seed_identity, key, stable_char_id,
                f"elastic-x:{point}",
            )
            dy_grid[grid_y][grid_x] = maximum * _signed_from_parts(
                profile.seed_identity, key, stable_char_id,
                f"elastic-y:{point}",
            )

    x_nodes = [i * width / (grid_size - 1) for i in range(grid_size)]
    y_nodes = [i * height / (grid_size - 1) for i in range(grid_size)]
    x_bounds = [round(value) for value in x_nodes]
    y_bounds = [round(value) for value in y_nodes]
    x_bounds[0], x_bounds[-1] = 0, width
    y_bounds[0], y_bounds[-1] = 0, height
    mesh = []
    for grid_y in range(grid_size - 1):
        for grid_x in range(grid_size - 1):
            box = (
                x_bounds[grid_x], y_bounds[grid_y],
                x_bounds[grid_x + 1], y_bounds[grid_y + 1],
            )
            if box[0] >= box[2] or box[1] >= box[3]:
                continue

            def source_point(point_x, point_y):
                return (
                    x_nodes[point_x] - dx_grid[point_y][point_x],
                    y_nodes[point_y] - dy_grid[point_y][point_x],
                )

            # Pillow QUAD 順: 左上、左下、右下、右上。
            top_left = source_point(grid_x, grid_y)
            bottom_left = source_point(grid_x, grid_y + 1)
            bottom_right = source_point(grid_x + 1, grid_y + 1)
            top_right = source_point(grid_x + 1, grid_y)
            quad = (*top_left, *bottom_left, *bottom_right, *top_right)
            mesh.append((box, quad))
    result = padded.transform(
        padded.size, Image.Transform.MESH, mesh,
        resample=Image.Resampling.BICUBIC, fillcolor=0,
    )
    return _crop_mask(result, offset_x, offset_y)


def _bleed(mask, dilation_px, blur_radius, strength, offset_x, offset_y):
    if strength == 0 or (dilation_px == 0 and blur_radius == 0):
        return _crop_mask(mask, offset_x, offset_y)
    pad = dilation_px + math.ceil(blur_radius * 3.0) + 2
    expanded = Image.new("L", (mask.width + 2 * pad, mask.height + 2 * pad), 0)
    expanded.paste(mask, (pad, pad))
    offset_x -= pad
    offset_y -= pad
    spread = expanded
    if dilation_px:
        spread = spread.filter(ImageFilter.MaxFilter(2 * dilation_px + 1))
    if blur_radius:
        spread = spread.filter(ImageFilter.GaussianBlur(blur_radius))
    # 芯の階調を維持し、外側に増えた濃度だけを指定強度で足す。
    image = Image.blend(expanded, ImageChops.lighter(expanded, spread), strength)
    return _crop_mask(image, offset_x, offset_y)


def _fade(mask, profile, key, stable_char_id, strength, offset_x, offset_y):
    if strength == 0:
        return _crop_mask(mask, offset_x, offset_y)
    required = mask.width * mask.height
    random_bytes = bytearray()
    block = 0
    while len(random_bytes) < required:
        # block 番号を transformation 名に含め、各ブロックを直接導出する。
        value = _seed_from_parts(
            profile.seed_identity, key, stable_char_id,
            f"fading-noise-mask:block:{block}",
        )
        random_bytes.extend(value.to_bytes(8, "big"))
        block += 1
    lookup = bytes(
        round(255.0 * (1.0 - strength * (value / 255.0) ** 3))
        for value in range(256)
    )
    noise_bytes = bytes(random_bytes[:required]).translate(lookup)
    noise_mask = Image.frombytes("L", mask.size, noise_bytes)
    image = ImageChops.multiply(mask, noise_mask)
    return _crop_mask(image, offset_x, offset_y)


def transform_mask(mask, profile, key, stable_char_id, size_scale,
                   rotation_degrees):
    """全変形を L mask に適用し、最終 mask と左上 offset を返す。

    各段の直後に ``getbbox`` で実インク領域を再計算して crop する。返却 mask
    も最終的な滲み・かすれ後の bbox に一致するため、呼び出し側はこの mask
    から 2 px margin 付き座標 GT を作れる。
    """
    if profile is None:
        return mask, (0.0, 0.0)
    offset_x = offset_y = 0.0
    mask, offset_x, offset_y = _crop_mask(mask, offset_x, offset_y)
    mask, offset_x, offset_y = _resize_about_center(
        mask, size_scale, offset_x, offset_y
    )
    mask, offset_x, offset_y = _rotate_about_center(
        mask, rotation_degrees, offset_x, offset_y
    )
    values = profile.parameters
    mask, offset_x, offset_y = _elastic_warp(
        mask, profile, key, stable_char_id,
        values["elastic_displacement_ratio"],
        values["elastic_grid_size"], offset_x, offset_y,
    )
    mask, offset_x, offset_y = _bleed(
        mask, values["bleed_dilation_px"],
        values["bleed_blur_radius_px"], values["bleed_strength"],
        offset_x, offset_y,
    )
    mask, offset_x, offset_y = _fade(
        mask, profile, key, stable_char_id, values["fading_strength"],
        offset_x, offset_y,
    )
    return mask, (offset_x, offset_y)
