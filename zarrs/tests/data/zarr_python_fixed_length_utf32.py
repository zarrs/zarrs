#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "zarr==3.1.3",
# ]
# ///

"""Generate fixed_length_utf32 test arrays for zarr-python compatibility tests.

Generates:
- V3 fixed_length_utf32 (little-endian) with length=4 (4 code points = 16 bytes)
- V2 <U4 (little-endian, 4 code points)
- V2 >U4 (big-endian, 4 code points)
"""

import json
import numpy as np
import zarr
from zarr.dtype import FixedLengthUTF32

# Test data: mix of ASCII, emoji, short strings, and empty string
test_data = ["abc", "🎉", "hi", "te", ""]
code_points = 4  # number of UTF-32 code points per element

# ---- V3 fixed_length_utf32 (little-endian) ----
path_v3_le = "tests/data/zarr_python_compat/fixed_length_utf32_v3_le.zarr"
array_v3_le = zarr.create_array(
    path_v3_le,
    dtype=FixedLengthUTF32(length=code_points, endianness="little"),
    shape=(len(test_data),),
    chunks=(len(test_data),),
    compressors=[],
    zarr_format=3,
    overwrite=True,
)
array_v3_le[:] = test_data

# Verify round-trip in Python
assert list(array_v3_le[:]) == test_data, "V3 LE round-trip failed"

with open(f"{path_v3_le}/zarr.json") as f:
    meta_v3_le = json.load(f)
print(f"V3 LE data_type: {meta_v3_le['data_type']}")
print(f"V3 LE codecs: {meta_v3_le['codecs']}")

# ---- V2 <U4 (little-endian) ----
path_v2_le = "tests/data/zarr_python_compat/fixed_length_utf32_v2_le.zarr"
array_v2_le = zarr.create_array(
    path_v2_le,
    dtype=np.dtype(f"<U{code_points}"),
    shape=(len(test_data),),
    chunks=(len(test_data),),
    compressors=None,
    zarr_format=2,
    overwrite=True,
)
array_v2_le[:] = test_data

# Verify round-trip in Python
assert list(array_v2_le[:]) == test_data, "V2 LE round-trip failed"

with open(f"{path_v2_le}/.zarray") as f:
    meta_v2_le = json.load(f)
print(f"V2 <U4 dtype: {meta_v2_le['dtype']}")

# ---- V2 >U4 (big-endian) ----
path_v2_be = "tests/data/zarr_python_compat/fixed_length_utf32_v2_be.zarr"
array_v2_be = zarr.create_array(
    path_v2_be,
    dtype=np.dtype(f">U{code_points}"),
    shape=(len(test_data),),
    chunks=(len(test_data),),
    compressors=None,
    zarr_format=2,
    overwrite=True,
)
array_v2_be[:] = test_data

# Verify round-trip in Python
assert list(array_v2_be[:]) == test_data, "V2 BE round-trip failed"

with open(f"{path_v2_be}/.zarray") as f:
    meta_v2_be = json.load(f)
print(f"V2 >U4 dtype: {meta_v2_be['dtype']}")

print("All arrays generated successfully!")
