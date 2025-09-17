#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "zarr@git+https://github.com/d-v-b/zarr-python.git@feat/fixed-length-strings",
# ]
# ///

import numpy as np
import zarr
print(zarr.__version__)

for unit in ["Y", "W", "D", "h", "m", "s", "ms", "us", "ns"]: # "M" skipped due to case insensitive filesystem issues
    for scale_factor in ["", "10"]:
        if scale_factor == "10" and unit not in ["ms", "us"]:
            continue
        z = zarr.create_array(
            f"zarrs/tests/data/zarr_python_compat/datetime64[{scale_factor}{unit}].zarr",
            shape=(6),
            chunks=(5),
            dtype=f'datetime64[{scale_factor}{unit}]',
            zarr_format=3,
            fill_value=np.datetime64('NaT'),
            overwrite=True,
        )
        z[:] = np.array([
            np.datetime64(0, 'Y'), # 1970, epoch
            np.datetime64('nat'),
            np.datetime64('2005-02-03'), 
            np.datetime64('2005-02-03T04:05'),
            np.datetime64('2005-02-03T04:05:06'),
            np.datetime64('nat'),
        ])

for unit in ["ms", "us", "ns", "ps"]:
    for scale_factor in ["", "10"]:
        if scale_factor == "10" and unit not in ["ms", "us"]:
            continue
        z = zarr.create_array(
            f"zarrs/tests/data/zarr_python_compat/timedelta64[{scale_factor}{unit}].zarr",
            shape=(11),
            chunks=(5),
            dtype=f'timedelta64[{scale_factor}{unit}]',
            zarr_format=3,
            fill_value=np.timedelta64('NaT'),
            overwrite=True,
        )
        z[:] = np.array([
            np.timedelta64(8760, 'h'),
            np.timedelta64(2, 'W'),
            np.timedelta64(3, 'D'),
            np.timedelta64(4, 'h'),
            np.timedelta64(5, 'm'),
            np.timedelta64(6, 's'),
            np.timedelta64(7_000, 'ms'),
            np.timedelta64(8_000_000, 'us'),
            np.timedelta64(9_000_000_000, 'ns'),
            np.timedelta64(0),
            np.timedelta64('NaT'),
        ])
