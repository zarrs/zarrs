#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "zarr==3.0.1",
# ]
# ///

import zarr
import numpy as np

# Generate string arrays with different fill values
path_out = "tests/data/zarr_python_compat/str_v2_fv_0.zarr"
array = zarr.create_array(
    path_out,
    dtype=str,
    shape=(5,),
    chunks=(2,),
    filters=zarr.codecs.vlen_utf8.VLenUTF8(),
    compressors=[None],
    fill_value=0,
    zarr_format=2,
    overwrite=True,
)
array[:3] = ["a", "bb", ""]
print(array.info)
# assert (array[:] == ["a", "bb", "", "", ""]).all() # FAILURE

path_out = "tests/data/zarr_python_compat/str_v2_fv_null.zarr"
array = zarr.create_array(
    path_out,
    dtype=str,
    shape=(5,),
    chunks=(2,),
    filters=zarr.codecs.vlen_utf8.VLenUTF8(),
    compressors=[None],
    fill_value=None,
    zarr_format=2,
    overwrite=True,
)
array[:3] = ["a", "bb", ""]
print(array.info)
print(array[:])
assert (array[:] == ["a", "bb", "", "", ""]).all()

# bools
path_out = "tests/data/zarr_python_compat/bool_v2_fv_null.zarr"
array = zarr.create_array(
    path_out,
    dtype=bool,
    shape=(5,),
    chunks=(2,),
    compressors=[None],
    fill_value=None,
    zarr_format=2,
    overwrite=True,
)
# Only write to first 2 elements, leaving the rest as fill values
array[:2] = [True, False]
print(f"Bool array: {array[:]}")

# ints
path_out = "tests/data/zarr_python_compat/int_v2_fv_null.zarr"
array = zarr.create_array(
    path_out,
    dtype=np.int32,
    shape=(5,),
    chunks=(2,),
    compressors=[None],
    fill_value=None,
    zarr_format=2,
    overwrite=True,
)
array[:2] = [42, 123]
print(f"Int array: {array[:]}")

# floats
path_out = "tests/data/zarr_python_compat/float_v2_fv_null.zarr"
array = zarr.create_array(
    path_out,
    dtype=np.float32,
    shape=(5,),
    chunks=(2,),
    compressors=[None],
    fill_value=None,
    zarr_format=2,
    overwrite=True,
)
array[:2] = [3.14, 2.71]
print(f"Float array: {array[:]}")
