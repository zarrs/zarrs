#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "zarr==3.1.2",
#   "shapely==2.1.2",
# ]
# ///

import numpy as np
import shapely
import zarr
from zarr.dtype import VariableLengthUTF8

# Root of the Zarr store
root = zarr.open_group("issue_287.zarr", mode="w", zarr_format=3)

meta = root.create_group("meta")

date_data = np.array(["2023-01-01", "2023-01-02", "2023-01-03"], dtype="datetime64[ms]")
meta.create_dataset("date", shape=date_data.shape, data=date_data)

collection_data = ["collection_a", "collection_b", "collection_c"]
meta.create_dataset(
    "collection",
    shape=len(collection_data),
    data=collection_data,
    dtype=VariableLengthUTF8(),
)


bbox_data = shapely.to_wkt(
    [
        shapely.box(-10.0, -10.0, 10.0, 10.0),
        shapely.box(-20.0, -20.0, 20.0, 20.0),
        shapely.box(-30.0, -30.0, 30.0, 30.0),
    ]
)

meta.create_dataset(
    "bbox",
    data=bbox_data,
    shape=len(bbox_data),
    dtype=VariableLengthUTF8(),
)
