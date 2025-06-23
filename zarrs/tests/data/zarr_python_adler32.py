#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "zarr==3.0.8",
#     "numcodecs==0.16.1",
# ]
# ///

import numpy as np
import zarr
import numcodecs
# import numcodecs.zarr3

z = zarr.create_array(
    "zarrs/tests/data/zarr_python_compat/adler32.zarr",
    shape=(100, 100),
    chunks=(50, 50),
    dtype=np.uint16,
    zarr_format=2,
    # zarr_format=3,
    fill_value=0,
    overwrite=True,
    compressors=[numcodecs.Adler32()],
    # compressors=[numcodecs.zarr3.Adler32()],
)
z[:] = np.arange(100 * 100).reshape(100, 100)
