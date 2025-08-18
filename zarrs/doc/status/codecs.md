| Codec Type     | V3 `name`                          | V2 `id`                             | Feature Flag* |
| -------------- | ---------------------------------- | ----------------------------------- | ------------- |
| Array to Array | [`transpose`]                      | `transpose`                         | **transpose** |
|                | 🚧[`reshape`]                      | -                                   |               |
|                | 🚧[`numcodecs.fixedscaleoffset`]   | `fixedscaleoffset`                  |               |
|                | 🚧[`numcodecs.bitround`]†          | `bitround`                          | bitround      |
|                | 🚧[`zarrs.squeeze`]                | -                                   |               |
| Array to Bytes | [`bytes`]                          | -                                   |               |
|                | [`sharding_indexed`]               | -                                   | **sharding**  |
|                | 🚧[`vlen-array`]                   | `vlen-array`                        |               |
|                | [`vlen-bytes`]                     | `vlen-bytes`                        |               |
|                | [`vlen-utf8`]                      | `vlen-utf8`                         |               |
|                | [`packbits`]                       | `packbits`                          |               |
|                | 🚧[`numcodecs.pcodec`]             | `pcodec`                            | pcodec        |
|                | 🚧[`numcodecs.zfpy`]               | `zfpy`                              | zfp           |
|                | 🚧[`zarrs.vlen`]                   | -                                   |               |
|                | 🚧[`zarrs.vlen_v2`]                | -                                   |               |
|                | [`zfp`]                            | -                                   | zfp           |
| Bytes to Bytes | [`blosc`]                          | `blosc`                             | **blosc**     |
|                | [`crc32c`]                         | `crc32c`                            | **crc32c**    |
|                | [`gzip`]                           | `gzip`                              | **gzip**      |
|                | [`zstd`]                           | `zstd`                              | **zstd**      |
|                | 🚧[`numcodecs.adler32`]            | `adler32`                           | adler32       |
|                | 🚧[`numcodecs.bz2`]                | `bz2`                               | bz2           |
|                | 🚧[`numcodecs.fletcher32`]         | `fletcher32`                        | fletcher32    |
|                | 🚧[`numcodecs.shuffle`]            | `shuffle`                           |               |
|                | 🚧[`numcodecs.zlib`]               | `zlib`                              | zlib          |
|                | 🚧[`zarrs.gdeflate`]               | -                                   | gdeflate      |

<sup>\* Bolded feature flags are part of the default set of features.</sup>
<sup>† `numcodecs.bitround` supports additional data types not supported by `zarr-python`/`numcodecs`</sup>

[Zarr V3.0 Blosc]: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/blosc/index.html
[Zarr V3.0 Bytes]: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/bytes/index.html
[Zarr V3.0 CRC32C]: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/crc32c/index.html
[Zarr V3.0 Gzip]: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/gzip/index.html
[Zarr V3.0 Sharding]: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html
[Zarr V3.0 Transpose]: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/transpose/index.html

[zarr-extensions/codecs/vlen-bytes]: https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/vlen-bytes
[zarr-extensions/codecs/vlen-utf8]: https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/vlen-utf8
[zarr-extensions/codecs/packbits]: https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/packbits
[zarr-extensions/codecs/zfp]: https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/zfp
[zarr-extensions/codecs/zstd]: https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/zstd

[`transpose`]: crate::array::codec::array_to_array::transpose
[`reshape`]: crate::array::codec::array_to_array::reshape
[`numcodecs.bitround`]: crate::array::codec::array_to_array::bitround
[`numcodecs.fixedscaleoffset`]: crate::array::codec::array_to_array::fixedscaleoffset
[`zarrs.squeeze`]: crate::array::codec::array_to_array::squeeze

[`bytes`]: crate::array::codec::array_to_bytes::bytes
[`vlen-array`]: crate::array::codec::array_to_bytes::vlen_array
[`vlen-bytes`]: crate::array::codec::array_to_bytes::vlen_bytes
[`vlen-utf8`]: crate::array::codec::array_to_bytes::vlen_utf8
[`sharding_indexed`]: crate::array::codec::array_to_bytes::sharding
[`numcodecs.pcodec`]: crate::array::codec::array_to_bytes::pcodec
[`numcodecs.zfpy`]: crate::array::codec::array_to_bytes::zfpy
[`packbits`]: crate::array::codec::array_to_bytes::packbits
[`zarrs.vlen`]: crate::array::codec::array_to_bytes::vlen
[`zarrs.vlen_v2`]: crate::array::codec::array_to_bytes::vlen_v2
[`zfp`]: crate::array::codec::array_to_bytes::zfp

[`blosc`]: crate::array::codec::bytes_to_bytes::blosc
[`crc32c`]: crate::array::codec::bytes_to_bytes::crc32c
[`gzip`]: crate::array::codec::bytes_to_bytes::gzip
[`zstd`]: crate::array::codec::bytes_to_bytes::zstd
[`numcodecs.adler32`]: crate::array::codec::bytes_to_bytes::adler32
[`numcodecs.bz2`]: crate::array::codec::bytes_to_bytes::gzip
[`numcodecs.fletcher32`]: crate::array::codec::bytes_to_bytes::fletcher32
[`numcodecs.shuffle`]: crate::array::codec::bytes_to_bytes::shuffle
[`numcodecs.zlib`]: crate::array::codec::bytes_to_bytes::zlib
[`zarrs.gdeflate`]: crate::array::codec::bytes_to_bytes::gdeflate

`zarrs` supports arrays created with `zarr-python` 3.0.0+ and `numcodecs` 0.15.1+ with various `numcodecs.zarr3` codecs.
