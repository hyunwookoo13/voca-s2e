from __future__ import annotations

import argparse
import json
import struct
import zlib
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

from goal_adapter.schema import GoalAdapterInput


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class HabitatSmokeSample:
    image_path: Path
    input_path: Path
    adapter_input: GoalAdapterInput


def collect_habitat_smoke_sample(
    observation: Mapping[str, Any],
    context: Mapping[str, Any],
    output_dir: str | Path,
    image_key: str = "rgb",
    image_filename: str = "current_rgb.png",
    input_filename: str = "goal_adapter_input.json",
) -> HabitatSmokeSample:
    if image_key not in observation:
        raise ValueError(f"observation must include '{image_key}'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / image_filename
    write_rgb_png(observation[image_key], image_path)

    payload = dict(_require_mapping(context, "context_json"))
    payload["current_rgb"] = str(image_path.resolve())
    adapter_input = GoalAdapterInput.from_json(payload)

    input_path = output_dir / input_filename
    input_path.write_text(
        json.dumps(adapter_input.to_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return HabitatSmokeSample(
        image_path=image_path,
        input_path=input_path,
        adapter_input=adapter_input,
    )


def write_rgb_png(image: Any, output_path: str | Path) -> None:
    width, height, rgb_bytes = _rgb_image_bytes(image)
    raw_rows = bytearray()
    stride = width * 3
    for row_index in range(height):
        raw_rows.append(0)
        start = row_index * stride
        raw_rows.extend(rgb_bytes[start : start + stride])

    png = bytearray(PNG_SIGNATURE)
    png.extend(_png_chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)))
    png.extend(_png_chunk(b"IDAT", zlib.compress(bytes(raw_rows))))
    png.extend(_png_chunk(b"IEND", b""))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(png))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect a Habitat smoke-test RGB image and GoalAdapter input JSON.",
    )
    parser.add_argument(
        "--observation-json",
        required=True,
        help="JSON file containing a Habitat-like observation object with an rgb field.",
    )
    parser.add_argument(
        "--context-json",
        required=True,
        help="JSON file containing GoalAdapterInput context fields except current_rgb.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where current_rgb.png and goal_adapter_input.json will be written.",
    )
    parser.add_argument(
        "--image-key",
        default="rgb",
        help="Observation key containing the RGB or RGBA frame.",
    )
    args = parser.parse_args(argv)

    sample = collect_habitat_smoke_sample(
        observation=_read_json_object(args.observation_json, "observation_json"),
        context=_read_json_object(args.context_json, "context_json"),
        output_dir=args.output_dir,
        image_key=args.image_key,
    )
    print(f"Wrote RGB image to {sample.image_path}")
    print(f"Wrote GoalAdapter input to {sample.input_path}")
    return 0


def _rgb_image_bytes(image: Any) -> tuple[int, int, bytes]:
    if hasattr(image, "tolist"):
        image = image.tolist()
    if _is_string_like(image) or not isinstance(image, Sequence) or not image:
        raise ValueError("rgb image must be a non-empty HxWx3 or HxWx4 array")

    height = len(image)
    first_row = _require_row(image[0], "rgb[0]")
    width = len(first_row)
    if width == 0:
        raise ValueError("rgb image rows must be non-empty")

    rgb_bytes = bytearray()
    for row_index, row_value in enumerate(image):
        row = _require_row(row_value, f"rgb[{row_index}]")
        if len(row) != width:
            raise ValueError("rgb image rows must have a consistent width")
        for column_index, pixel_value in enumerate(row):
            pixel = _require_pixel(pixel_value, f"rgb[{row_index}][{column_index}]")
            rgb_bytes.extend(pixel[:3])
    return width, height, bytes(rgb_bytes)


def _require_row(value: Any, field_name: str) -> Sequence[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if _is_string_like(value) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a row of RGB pixels")
    return value


def _require_pixel(value: Any, field_name: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if _is_string_like(value) or not isinstance(value, Sequence) or len(value) not in {3, 4}:
        raise ValueError(f"{field_name} must be an RGB or RGBA pixel")
    components: list[int] = []
    for component in value[:3]:
        if not isinstance(component, Integral):
            raise ValueError(f"{field_name} must contain integer RGB values")
        component = int(component)
        if component < 0 or component > 255:
            raise ValueError(f"{field_name} RGB values must be in the 0..255 range")
        components.append(component)
    return components


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack("!I", len(data)) + chunk_type + data + struct.pack("!I", crc)


def _read_json_object(path: str | Path, field_name: str) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _require_mapping(payload, field_name)


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _is_string_like(value: Any) -> bool:
    return isinstance(value, (str, bytes, bytearray))


if __name__ == "__main__":
    raise SystemExit(main())
