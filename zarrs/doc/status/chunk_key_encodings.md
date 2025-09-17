| Chunk Key Encoding       | ZEP       | V3      | V2      | Feature Flag |
| ------------------------ | --------- | ------- | ------- | ------------ |
| [`default`]              | [ZEP0001] | &check; |         |              |
| [`v2`]                   | [ZEP0001] | &check; | &check; |              |
| 🚧[`zarrs.default_suffix`] |           | &check; |         |              |

[`default`]: crate::array::chunk_key_encoding::DefaultChunkKeyEncoding
[`v2`]: crate::array::chunk_key_encoding::V2ChunkKeyEncoding
[`zarrs.default_suffix`]: crate::array::chunk_key_encoding::DefaultSuffixChunkKeyEncoding
[ZEP0001]: https://zarr.dev/zeps/accepted/ZEP0001.html
