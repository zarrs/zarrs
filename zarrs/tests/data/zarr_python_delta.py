#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "zarr==3.1.6",
#     "numcodecs==0.16.5",
# ]
# ///

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import zarr
import numcodecs
from zarr.codecs.numcodecs import Delta
from zarr.codecs import BytesCodec

# --- Zarr V2 ---

# V2 delta (dtype=int16)
z = zarr.open_array(
    "zarrs/tests/data/zarr_python_compat/delta_i2.zarr",
    mode="w",
    shape=(100, 100),
    chunks=(50, 50),
    dtype=np.int16,
    zarr_format=2,
    fill_value=0,
    filters=[numcodecs.Delta(dtype="<i2")],
    compressor=None,
)
z[:] = np.arange(100 * 100, dtype=np.int16).reshape(100, 100)

# V2 delta (dtype=int16, astype=int32)
z = zarr.open_array(
    "zarrs/tests/data/zarr_python_compat/delta_i2_astype_i4.zarr",
    mode="w",
    shape=(100, 100),
    chunks=(50, 50),
    dtype=np.int16,
    zarr_format=2,
    fill_value=0,
    filters=[numcodecs.Delta(dtype="<i2", astype="<i4")],
    compressor=None,
)
z[:] = np.arange(100 * 100, dtype=np.int16).reshape(100, 100)

# V2 delta (dtype=uint16, astype=uint32)
z = zarr.open_array(
    "zarrs/tests/data/zarr_python_compat/delta_u2_astype_u4.zarr",
    mode="w",
    shape=(100, 100),
    chunks=(50, 50),
    dtype=np.uint16,
    zarr_format=2,
    fill_value=0,
    filters=[numcodecs.Delta(dtype="<u2", astype="<u4")],
    compressor=None,
)
z[:] = np.arange(100 * 100, dtype=np.uint16).reshape(100, 100)

# --- Zarr V3 ---

# V3 numcodecs.delta (dtype=int16)
z = zarr.open_array(
    "zarrs/tests/data/zarr_python_compat/delta_v3_i2.zarr",
    mode="w",
    shape=(100, 100),
    chunks=(50, 50),
    dtype=np.int16,
    zarr_format=3,
    fill_value=0,
    codecs=[Delta(dtype="<i2"), BytesCodec()],
)
z[:] = np.arange(100 * 100, dtype=np.int16).reshape(100, 100)

# V3 numcodecs.delta (dtype=int16, astype=int32)
z = zarr.open_array(
    "zarrs/tests/data/zarr_python_compat/delta_v3_i2_astype_i4.zarr",
    mode="w",
    shape=(100, 100),
    chunks=(50, 50),
    dtype=np.int16,
    zarr_format=3,
    fill_value=0,
    codecs=[Delta(dtype="<i2", astype="<i4"), BytesCodec()],
)
z[:] = np.arange(100 * 100, dtype=np.int16).reshape(100, 100)

# V3 numcodecs.delta (dtype=uint16, astype=uint32)
z = zarr.open_array(
    "zarrs/tests/data/zarr_python_compat/delta_v3_u2_astype_u4.zarr",
    mode="w",
    shape=(100, 100),
    chunks=(50, 50),
    dtype=np.uint16,
    zarr_format=3,
    fill_value=0,
    codecs=[Delta(dtype="<u2", astype="<u4"), BytesCodec()],
)
z[:] = np.arange(100 * 100, dtype=np.uint16).reshape(100, 100)
