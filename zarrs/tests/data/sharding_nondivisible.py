#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "zarr @ git+https://github.com/zarr-developers/zarr-python@refs/pull/4198/head",
# ]
# ///

# Sharded arrays with subchunk shapes that do not evenly divide the shard shape (zarr-specs #370).
# Run from `zarrs/`: `./tests/data/sharding_nondivisible.py`

import numpy as np
import zarr
from zarr.codecs import BytesCodec, ShardingCodec

OUT = "tests/data/zarr_python_compat/sharding_nondivisible"


def write(name, shape, **kwargs):
    data = np.arange(np.prod(shape), dtype="uint16").reshape(shape)
    array = zarr.create_array(
        f"{OUT}_{name}.zarr", shape=shape, dtype="uint16", fill_value=0, compressors=None,
        overwrite=True, **kwargs,
    )
    array[...] = data


write("1d", (23,), shards={"shape": (12,), "index_location": "end"}, chunks=(5,))
write("2d", (23, 17), shards=(12, 12), chunks=(5, 5))
write("larger_than_shard", (10,), shards=(4,), chunks=(6,))
nested = ShardingCodec(chunk_shape=(5,), codecs=(ShardingCodec(chunk_shape=(2,), codecs=(BytesCodec(),)),))
write("nested", (24,), chunks=(12,), serializer=nested, filters=None)
with zarr.config.set({"array.rectilinear_chunks": True}):
    write("rectilinear", (10,), shards=[[4, 6]], chunks=(4,))
