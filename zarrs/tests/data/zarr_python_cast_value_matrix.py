#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cast-value-rs",
#     "ml-dtypes",
#     "numpy",
#     "zarr==3.2.1",
# ]
# ///

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.codecs import BytesCodec, CastValue
from zarr.core.dtype import get_data_type_from_json

DTYPES = [
    "int2",
    "int4",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint2",
    "uint4",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float4_e2m1fn",
    "float6_e2m3fn",
    "float6_e3m2fn",
    "float8_e3m4",
    "float8_e4m3",
    "float8_e4m3b11fnuz",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
    "bfloat16",
    "float16",
    "float32",
    "float64",
]

ROUNDINGS = [
    "nearest-even",
    "towards-zero",
    "towards-positive",
    "towards-negative",
    "nearest-away",
]

OUT_OF_RANGE = [None, "clamp", "wrap"]

ROOT = Path(__file__).parent / "zarr_python_compat/cast_value_matrix"
ARRAYS = ROOT / "arrays"
MANIFEST = ROOT / "manifest.json"


def dtype_info(name: str) -> tuple[Any, np.dtype[Any]] | None:
    try:
        zdtype = get_data_type_from_json(name, zarr_format=3)
        return zdtype, np.dtype(zdtype.to_native_dtype())
    except Exception:
        return None


def is_float(dtype: np.dtype[Any]) -> bool:
    return np.issubdtype(dtype, np.floating)


def is_integer(dtype: np.dtype[Any]) -> bool:
    return np.issubdtype(dtype, np.integer)


def is_unsigned(dtype: np.dtype[Any]) -> bool:
    return np.issubdtype(dtype, np.unsignedinteger)


def safe_name(*parts: str | None) -> str:
    return "__".join("none" if part is None else part.replace("-", "_") for part in parts)


def source_values(source_dtype: np.dtype[Any], out_of_range: str | None, map_kind: str) -> list[Any]:
    if map_kind == "nan_to_zero":
        return [0.0, 1.0, math.nan, 3.0]
    if out_of_range is None:
        return [0, 1, 2, 3]
    if is_float(source_dtype):
        return [0.0, 1.5, 127.5, 300.0, -3.0]
    info = np.iinfo(source_dtype)
    values = [0, 1]
    if info.min < 0:
        values.append(max(info.min, -3))
    values.append(min(info.max, 300))
    values.append(min(info.max, 127))
    return values


def scalar_map_for(
    source_dtype: np.dtype[Any],
    target_dtype: np.dtype[Any],
    map_kind: str,
) -> dict[str, list[tuple[Any, Any]]] | None:
    if map_kind == "none":
        return None
    if map_kind == "finite":
        return {"encode": [(1.0 if is_float(source_dtype) else 1, 2.0 if is_float(target_dtype) else 2)]}
    if map_kind == "nan_to_zero":
        return {"encode": [("NaN", 0.0 if is_float(target_dtype) else 0)]}
    raise ValueError(map_kind)


def expected_bytes(array: Any, source_dtype: np.dtype[Any]) -> list[int]:
    values = np.asarray(array[:], dtype=source_dtype)
    return list(values.tobytes())


def create_case(
    source: str,
    source_dtype: np.dtype[Any],
    target: str,
    target_dtype: np.dtype[Any],
    rounding: str,
    out_of_range: str | None,
    map_kind: str,
) -> dict[str, Any]:
    name = safe_name(source, "to", target, rounding, out_of_range, map_kind)
    case_path = ARRAYS / f"{name}.zarr"
    config: dict[str, Any] = {
        "source_dtype": source,
        "target_dtype": target,
        "rounding": rounding,
        "out_of_range": out_of_range,
        "scalar_map": map_kind,
        "path": str(case_path.relative_to(ROOT)),
    }
    values = source_values(source_dtype, out_of_range, map_kind)
    scalar_map = scalar_map_for(source_dtype, target_dtype, map_kind)
    array = zarr.create_array(
        case_path,
        shape=(len(values),),
        chunks=(len(values),),
        dtype=source,
        fill_value=0,
        filters=[
            CastValue(
                data_type=target,
                rounding=rounding,  # type: ignore[arg-type]
                out_of_range=out_of_range,  # type: ignore[arg-type]
                scalar_map=scalar_map,
            )
        ],
        serializer=BytesCodec(endian="little"),
        compressors=[],
        zarr_format=3,
        overwrite=True,
    )
    array[:] = np.asarray(values, dtype=source_dtype)
    expected = expected_bytes(array, source_dtype)
    encoded = case_path.joinpath("c/0").read_bytes()

    return {
        **config,
        "input": [str(value) if isinstance(value, float) and math.isnan(value) else value for value in values],
        "expected_decoded_bytes": expected,
        "encoded_bytes": list(encoded),
    }


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ARRAYS.mkdir(parents=True)

    dtype_map: dict[str, tuple[Any, np.dtype[Any]]] = {}
    unsupported_dtypes: dict[str, str] = {}
    for name in DTYPES:
        info = dtype_info(name)
        if info is None:
            unsupported_dtypes[name] = f"not supported by zarr-python {zarr.__version__}"
        else:
            dtype_map[name] = info

    cases = []
    filtered = 0
    for source, (_, source_dtype) in dtype_map.items():
        for target, (_, target_dtype) in dtype_map.items():
            for rounding in ROUNDINGS:
                for out_of_range in OUT_OF_RANGE:
                    for map_kind in ("none", "finite", "nan_to_zero"):
                        if not should_generate_case(
                            source_dtype,
                            target_dtype,
                            out_of_range,
                            map_kind,
                        ):
                            filtered += 1
                            continue
                        cases.append(
                            create_case(
                                source,
                                source_dtype,
                                target,
                                target_dtype,
                                rounding,
                                out_of_range,
                                map_kind,
                            )
                        )

    manifest = {
        "version": 1,
        "generator": Path(__file__).name,
        "zarr_python": zarr.__version__,
        "supported_dtypes": sorted(dtype_map),
        "unsupported_dtypes": unsupported_dtypes,
        "roundings": ROUNDINGS,
        "out_of_range": OUT_OF_RANGE,
        "cases": cases,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {MANIFEST}")
    print(f"Filtered {filtered} unsupported combinations before generation")
    print(f"Generated {len(cases)} cases")


def should_generate_case(
    source_dtype: np.dtype[Any],
    target_dtype: np.dtype[Any],
    out_of_range: str | None,
    map_kind: str,
) -> bool:
    if out_of_range == "wrap" and not is_integer(target_dtype):
        return False
    if map_kind == "nan_to_zero" and not (is_float(source_dtype) and is_integer(target_dtype)):
        return False
    return True


if __name__ == "__main__":
    main()
