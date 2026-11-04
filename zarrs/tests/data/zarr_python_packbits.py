#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "zarr==3.1.3",
#     "numcodecs==0.16.3",
# ]
# ///

import numpy as np
import zarr
import numcodecs
from zarr.codecs import numcodecs as numcodecs_v3

print(zarr.__version__, numcodecs.__version__)

# Chunks of 20 elements, so each chunk has 4 padding bits
z = zarr.create_array(
    "zarrs/tests/data/zarr_python_compat/packbits.zarr",
    shape=(10, 9),
    chunks=(5, 4),
    dtype=bool,
    zarr_format=2,
    fill_value=False,
    overwrite=True,
    filters=[numcodecs.PackBits()],
    compressors=None,
)
z[:] = (np.arange(10 * 9).reshape(10, 9) * 7 % 11) < 4

z = zarr.create_array(
    "zarrs/tests/data/zarr_python_compat/packbits_v3.zarr",
    shape=(10, 9),
    chunks=(5, 4),
    dtype=bool,
    zarr_format=3,
    fill_value=False,
    overwrite=True,
    filters=[numcodecs_v3.PackBits()],
    compressors=None,
)
z[:] = (np.arange(10 * 9).reshape(10, 9) * 7 % 11) < 4
