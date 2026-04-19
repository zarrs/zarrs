# Codec & Data Type Compatibility Matrix

## How Compatibility is Evaluated

Each codec/data type combination is tested by:
1. Creating a small test array with representative values for the data type
2. Encoding the array using the codec, using default codecs for the data type where needed
3. Decoding the encoded data back to an array
4. Verifying the decoded array matches the original (round-trip test)

Results:
- **✓ supported**: The codec successfully encodes and decodes the data type
- **✗ unsupported**: The codec explicitly does not support the data type
- **💥 failure**: The codec claims support but the round-trip test failed
- **- not tested**: The combination has not been tested

## Contents

- [Array-to-Array Codecs](#array-to-array-codecs)
- [Array-to-Bytes Codecs](#array-to-bytes-codecs)
- [Bytes-to-Bytes Codecs](#bytes-to-bytes-codecs)

---

## Array-to-Array Codecs

| Data Type | bitround | cast_value | numcodecs.fixedscaleoffset | reshape | zarrs.squeeze | transpose |
|-----------|---|---|---|---|---|---|
| bfloat16 | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| bool | ✗ | - | ✗ | ✓ | ✓ | ✓ |
| bytes | - | - | - | ✓ | ✓ | ✓ |
| complex128 | ✓ | - | ✗ | ✓ | ✓ | ✓ |
| complex64 | ✓ | - | ✗ | ✓ | ✓ | ✓ |
| complex_bfloat16 | ✓ | - | ✗ | ✓ | ✓ | ✓ |
| complex_float16 | ✓ | - | ✗ | ✓ | ✓ | ✓ |
| complex_float32 | ✓ | - | ✗ | ✓ | ✓ | ✓ |
| complex_float4_e2m1fn | - | - | - | - | - | - |
| complex_float64 | ✓ | - | ✗ | ✓ | ✓ | ✓ |
| complex_float6_e2m3fn | - | - | - | - | - | - |
| complex_float6_e3m2fn | - | - | - | - | - | - |
| complex_float8_e3m4 | - | - | - | - | - | - |
| complex_float8_e4m3 | ✗ | - | ✗ | ✓ | ✓ | ✓ |
| complex_float8_e4m3b11fnuz | - | - | - | - | - | - |
| complex_float8_e4m3fnuz | - | - | - | - | - | - |
| complex_float8_e5m2 | ✗ | - | ✗ | ✓ | ✓ | ✓ |
| complex_float8_e5m2fnuz | - | - | - | - | - | - |
| complex_float8_e8m0fnu | - | - | - | - | - | - |
| float16 | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| float32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| float4_e2m1fn | - | - | - | - | - | - |
| float64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| float6_e2m3fn | - | - | - | - | - | - |
| float6_e3m2fn | - | - | - | - | - | - |
| float8_e3m4 | - | - | - | - | - | - |
| float8_e4m3 | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| float8_e4m3b11fnuz | - | - | - | - | - | - |
| float8_e4m3fnuz | - | - | - | - | - | - |
| float8_e5m2 | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| float8_e5m2fnuz | - | - | - | - | - | - |
| float8_e8m0fnu | - | - | - | - | - | - |
| int16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int2 | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| int32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int4 | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| int64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int8 | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| numpy.datetime64 | ✓ | - | ✗ | ✓ | ✓ | ✓ |
| numpy.timedelta64 | ✓ | - | ✗ | ✓ | ✓ | ✓ |
| r24 | ✗ | - | ✗ | ✓ | ✓ | ✓ |
| string | - | - | - | ✓ | ✓ | ✓ |
| uint16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| uint2 | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| uint32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| uint4 | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| uint64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| uint8 | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| zarrs.optional(float32) | - | - | - | ✓ | ✓ | ✗ |
| zarrs.optional(string) | - | - | - | ✓ | ✓ | ✗ |
| zarrs.optional(uint8) | - | - | - | ✓ | ✓ | ✗ |
| zarrs.optional(zarrs.optional(float32)) | - | - | - | ✓ | ✓ | ✗ |

## Array-to-Bytes Codecs

| Data Type | bytes | zarrs.optional | packbits | numcodecs.pcodec | sharding_indexed | zarrs.vlen | vlen-array | vlen-bytes | vlen-utf8 | zarrs.vlen_v2 | zfp |
|-----------|---|---|---|---|---|---|---|---|---|---|---|
| bfloat16 | ✓ | - | ✓ | ✗ | ✓ | - | - | - | - | - | ✗ |
| bool | ✓ | - | ✓ | ✗ | ✓ | - | - | - | - | - | ✗ |
| bytes | - | - | - | - | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| complex128 | ✓ | - | ✓ | ✓ | ✓ | - | - | - | - | - | ✗ |
| complex64 | ✓ | - | ✓ | ✓ | ✓ | - | - | - | - | - | ✗ |
| complex_bfloat16 | ✓ | - | ✓ | ✗ | ✓ | - | - | - | - | - | ✗ |
| complex_float16 | ✓ | - | ✓ | ✓ | ✓ | - | - | - | - | - | ✗ |
| complex_float32 | ✓ | - | ✓ | ✓ | ✓ | - | - | - | - | - | ✗ |
| complex_float4_e2m1fn | - | - | - | - | - | - | - | - | - | - | - |
| complex_float64 | ✓ | - | ✓ | ✓ | ✓ | - | - | - | - | - | ✗ |
| complex_float6_e2m3fn | - | - | - | - | - | - | - | - | - | - | - |
| complex_float6_e3m2fn | - | - | - | - | - | - | - | - | - | - | - |
| complex_float8_e3m4 | - | - | - | - | - | - | - | - | - | - | - |
| complex_float8_e4m3 | ✓ | - | ✓ | ✗ | ✓ | - | - | - | - | - | ✗ |
| complex_float8_e4m3b11fnuz | - | - | - | - | - | - | - | - | - | - | - |
| complex_float8_e4m3fnuz | - | - | - | - | - | - | - | - | - | - | - |
| complex_float8_e5m2 | ✓ | - | ✓ | ✗ | ✓ | - | - | - | - | - | ✗ |
| complex_float8_e5m2fnuz | - | - | - | - | - | - | - | - | - | - | - |
| complex_float8_e8m0fnu | - | - | - | - | - | - | - | - | - | - | - |
| float16 | ✓ | - | ✓ | ✓ | ✓ | - | - | - | - | - | ✗ |
| float32 | ✓ | - | ✓ | ✓ | ✓ | - | - | - | - | - | ✓ |
| float4_e2m1fn | - | - | - | - | - | - | - | - | - | - | - |
| float64 | ✓ | - | ✓ | ✓ | ✓ | - | - | - | - | - | ✓ |
| float6_e2m3fn | - | - | - | - | - | - | - | - | - | - | - |
| float6_e3m2fn | - | - | - | - | - | - | - | - | - | - | - |
| float8_e3m4 | - | - | - | - | - | - | - | - | - | - | - |
| float8_e4m3 | ✓ | - | ✓ | ✗ | ✓ | - | - | - | - | - | ✗ |
| float8_e4m3b11fnuz | - | - | - | - | - | - | - | - | - | - | - |
| float8_e4m3fnuz | - | - | - | - | - | - | - | - | - | - | - |
| float8_e5m2 | ✓ | - | ✓ | ✗ | ✓ | - | - | - | - | - | ✗ |
| float8_e5m2fnuz | - | - | - | - | - | - | - | - | - | - | - |
| float8_e8m0fnu | - | - | - | - | - | - | - | - | - | - | - |
| int16 | ✓ | - | ✓ | ✓ | ✓ | - | - | - | - | - | ✓ |
| int2 | ✓ | - | ✓ | ✗ | ✓ | - | - | - | - | - | ✗ |
| int32 | ✓ | - | ✓ | ✓ | ✓ | - | - | - | - | - | ✓ |
| int4 | ✓ | - | ✓ | ✗ | ✓ | - | - | - | - | - | ✗ |
| int64 | ✓ | - | ✓ | ✓ | ✓ | - | - | - | - | - | ✓ |
| int8 | ✓ | - | ✓ | ✗ | ✓ | - | - | - | - | - | ✓ |
| numpy.datetime64 | ✓ | - | ✓ | ✓ | ✓ | - | - | - | - | - | ✓ |
| numpy.timedelta64 | ✓ | - | ✓ | ✓ | ✓ | - | - | - | - | - | ✓ |
| r24 | ✓ | - | ✗ | ✗ | ✓ | - | - | - | - | - | ✗ |
| string | - | - | - | - | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| uint16 | ✓ | - | ✓ | ✓ | ✓ | - | - | - | - | - | ✓ |
| uint2 | ✓ | - | ✓ | ✗ | ✓ | - | - | - | - | - | ✗ |
| uint32 | ✓ | - | ✓ | ✓ | ✓ | - | - | - | - | - | ✓ |
| uint4 | ✓ | - | ✓ | ✗ | ✓ | - | - | - | - | - | ✗ |
| uint64 | ✓ | - | ✓ | ✓ | ✓ | - | - | - | - | - | ✓ |
| uint8 | ✓ | - | ✓ | ✗ | ✓ | - | - | - | - | - | ✓ |
| zarrs.optional(float32) | - | ✓ | - | - | - | - | - | - | - | - | - |
| zarrs.optional(string) | - | ✓ | - | - | - | - | - | - | - | - | - |
| zarrs.optional(uint8) | - | ✓ | - | - | - | - | - | - | - | - | - |
| zarrs.optional(zarrs.optional(float32)) | - | ✓ | - | - | - | - | - | - | - | - | - |

## Bytes-to-Bytes Codecs

| Data Type | numcodecs.adler32 | blosc | numcodecs.bz2 | crc32c | numcodecs.fletcher32 | zarrs.gdeflate | gzip | numcodecs.shuffle | numcodecs.zlib | zstd |
|-----------|---|---|---|---|---|---|---|---|---|---|
| bfloat16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| bool | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| bytes | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - | ✓ | ✓ |
| complex128 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| complex64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| complex_bfloat16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| complex_float16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| complex_float32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| complex_float4_e2m1fn | - | - | - | - | - | - | - | - | - | - |
| complex_float64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| complex_float6_e2m3fn | - | - | - | - | - | - | - | - | - | - |
| complex_float6_e3m2fn | - | - | - | - | - | - | - | - | - | - |
| complex_float8_e3m4 | - | - | - | - | - | - | - | - | - | - |
| complex_float8_e4m3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| complex_float8_e4m3b11fnuz | - | - | - | - | - | - | - | - | - | - |
| complex_float8_e4m3fnuz | - | - | - | - | - | - | - | - | - | - |
| complex_float8_e5m2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| complex_float8_e5m2fnuz | - | - | - | - | - | - | - | - | - | - |
| complex_float8_e8m0fnu | - | - | - | - | - | - | - | - | - | - |
| float16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| float32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| float4_e2m1fn | - | - | - | - | - | - | - | - | - | - |
| float64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| float6_e2m3fn | - | - | - | - | - | - | - | - | - | - |
| float6_e3m2fn | - | - | - | - | - | - | - | - | - | - |
| float8_e3m4 | - | - | - | - | - | - | - | - | - | - |
| float8_e4m3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| float8_e4m3b11fnuz | - | - | - | - | - | - | - | - | - | - |
| float8_e4m3fnuz | - | - | - | - | - | - | - | - | - | - |
| float8_e5m2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| float8_e5m2fnuz | - | - | - | - | - | - | - | - | - | - |
| float8_e8m0fnu | - | - | - | - | - | - | - | - | - | - |
| int16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| numpy.datetime64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| numpy.timedelta64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| r24 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| string | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - | ✓ | ✓ |
| uint16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| uint2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| uint32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| uint4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| uint64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| uint8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| zarrs.optional(float32) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - | ✓ | ✓ |
| zarrs.optional(string) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - | ✓ | ✓ |
| zarrs.optional(uint8) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - | ✓ | ✓ |
| zarrs.optional(zarrs.optional(float32)) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - | ✓ | ✓ |

