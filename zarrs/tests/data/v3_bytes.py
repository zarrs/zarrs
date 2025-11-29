#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "zarr==3.1.3",
# ]
# ///

import zarr
from zarr.dtype import ArrayBytesVariableLength

data = [
    b"New York",
    b"Los Angeles",
    b"Chicago",
]

path_out = "tests/data/zarr_python_compat/v3_bytes.zarr"

array = zarr.create_array(
    path_out,
    dtype=ArrayBytesVariableLength(),
    shape=(len(data),),
    chunks=(1000,),
    compressors=[],
    zarr_format=3,
    overwrite=True,
)
array[:] = data
print(array.info)
