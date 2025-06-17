| [`DataType`]                  | V3 `data_type` `name`      | V2 `dtype`  | [`ElementOwned`] / [`Element`] (Feature Flag) |
| ----------------------------- | -------------------------- | ----------- | --------------------------------------------- |
| [`Bool`]                      | bool                       | \|b1        | [`bool`]                                      |
| [`Int2`]                      | int2                       |             | [`i8`]                                        |
| [`Int4`]                      | int4                       |             | [`i8`]                                        |
| [`Int8`]                      | int8                       | \|i1        | [`i8`]                                        |
| [`Int16`]                     | int16                      | >i2 <i2     | [`i16`]                                       |
| [`Int32`]                     | int32                      | >i4 <i4     | [`i32`]                                       |
| [`Int64`]                     | int64                      | >i8 <i8     | [`i64`]                                       |
| [`UInt2`]                     | uint2                      |             | [`u8`]                                        |
| [`UInt4`]                     | uint4                      |             | [`u8`]                                        |
| [`UInt8`]                     | uint8                      | \|u1        | [`u8`]                                        |
| [`UInt16`]                    | uint16                     | >u2 <u2     | [`u16`]                                       |
| [`UInt32`]                    | uint32                     | >u4 <u4     | [`u32`]                                       |
| [`UInt64`]                    | uint64                     | >u8 <u8     | [`u64`]                                       |
| [`Float4E2M1FN`]†             | float4_e2m1fn              |             |                                               |
| [`Float6E2M3FN`]†             | float6_e2m3fn              |             |                                               |
| [`Float6E3M2FN`]†             | float6_e3m2fn              |             |                                               |
| [`Float8E3M4`]†               | float8_e3m4                |             |                                               |
| [`Float8E4M3`]†               | float8_e4m3                |             |                                               |
| [`Float8E4M3B11FNUZ`]†        | float8_e4m3b11fnuz         |             |                                               |
| [`Float8E4M3FNUZ`]†           | float8_e4m3fnuz            |             |                                               |
| [`Float8E5M2`]†               | float8_e5m2                |             |                                               |
| [`Float8E5M2FNUZ`]†           | float8_e5m2fnuz            |             |                                               |
| [`Float8E8M0FNU`]†            | float8_e8m0fnu             |             |                                               |
| [`BFloat16`]                  | bfloat16                   |             | [`half::bf16`]                                |
| [`Float16`]                   | float16                    | >f2 <f2     | [`half::f16`]                                 |
| [`Float32`]                   | float32                    | >f4 <f4     | [`f32`]                                       |
| [`Float64`]                   | float64                    | >f8 <f8     | [`f64`]                                       |
| [`Complex64`]                 | complex64                  | >c8 <c8     | [`Complex<f32>`]                              |
| [`Complex128`]                | complex128                 | >c16 <c16   | [`Complex<f64>`]                              |
| [`ComplexBFloat16`]           | complex_bfloat16           |             | [`Complex<half::bf16>`]                       |
| [`ComplexFloat16`]            | complex_float16            |             | [`Complex<half::f16>`]                        |
| [`ComplexFloat32`]            | complex_float32            |             | [`Complex<f32>`]                              |
| [`ComplexFloat64`]            | complex_float64            |             | [`Complex<f64>`]                              |
| [`ComplexFloat4E2M1FN`]†      | complex_float4_e2m1fn      |             |                                               |
| [`ComplexFloat6E2M3FN`]†      | complex_float6_e2m3fn      |             |                                               |
| [`ComplexFloat6E3M2FN`]†      | complex_float6_e3m2fn      |             |                                               |
| [`ComplexFloat8E3M4`]†        | complex_float8_e3m4        |             |                                               |
| [`ComplexFloat8E4M3`]†        | complex_float8_e4m3        |             |                                               |
| [`ComplexFloat8E4M3B11FNUZ`]† | complex_float8_e4m3b11fnuz |             |                                               |
| [`ComplexFloat8E4M3FNUZ`]†    | complex_float8_e4m3fnuz    |             |                                               |
| [`ComplexFloat8E5M2`]†        | complex_float8_e5m2        |             |                                               |
| [`ComplexFloat8E5M2FNUZ`]†    | complex_float8_e5m2fnuz    |             |                                               |
| [`ComplexFloat8E8M0FNU`]†     | complex_float8_e8m0fnu     |             |                                               |
| [`RawBits`]                   | r*                         |             | `[u8; N]` / `&[u8; N]`                        |
| [`String`]                    | string                     | \|O         | [`String`] / [`&str`]                         |
| [`Bytes`]                     | bytes<br>binary            | \|VX        | [`Vec<u8>`] / `&[u8]`                         |
| [`NumpyDateTime64`]           | numpy.datetime64           |             | [`i64`]<br>[`chrono::DateTime<Utc>`] (`chrono`)<br>[`jiff::Timestamp`] (`jiff`)  |
| [`NumpyTimeDelta64`]          | numpy.timedelta64          |             | [`i64`]<br>[`chrono::TimeDelta`] (`chrono`)<br>[`jiff::SignedDuration`] (`jiff`) |

<sup>† Additional features (e.g. `float8`) may be required to parse floating point fill values. All subfloat types support hex string fill values.</sup>

[`DataType`]: crate::array::DataType

[`Bool`]: crate::array::DataType::Bool
[`Int2`]: crate::array::DataType::Int2
[`Int4`]: crate::array::DataType::Int4
[`Int8`]: crate::array::DataType::Int8
[`Int16`]: crate::array::DataType::Int16
[`Int32`]: crate::array::DataType::Int32
[`Int64`]: crate::array::DataType::Int64
[`UInt2`]: crate::array::DataType::UInt2
[`UInt4`]: crate::array::DataType::UInt4
[`UInt8`]: crate::array::DataType::UInt8
[`UInt16`]: crate::array::DataType::UInt16
[`UInt32`]: crate::array::DataType::UInt32
[`UInt64`]: crate::array::DataType::UInt64
[`Float4E2M1FN`]: crate::array::DataType::Float4E2M1FN
[`Float6E2M3FN`]: crate::array::DataType::Float6E2M3FN
[`Float6E3M2FN`]: crate::array::DataType::Float6E3M2FN
[`Float8E3M4`]: crate::array::DataType::Float8E3M4
[`Float8E4M3`]: crate::array::DataType::Float8E4M3
[`Float8E4M3B11FNUZ`]: crate::array::DataType::Float8E4M3B11FNUZ
[`Float8E4M3FNUZ`]: crate::array::DataType::Float8E4M3FNUZ
[`Float8E5M2`]: crate::array::DataType::Float8E5M2
[`Float8E5M2FNUZ`]: crate::array::DataType::Float8E5M2FNUZ
[`Float8E8M0FNU`]: crate::array::DataType::Float8E8M0FNU
[`BFloat16`]: crate::array::DataType::BFloat16
[`Float16`]: crate::array::DataType::Float16
[`Float32`]: crate::array::DataType::Float32
[`Float64`]: crate::array::DataType::Float64
[`ComplexBFloat16`]: crate::array::DataType::ComplexBFloat16
[`ComplexFloat16`]: crate::array::DataType::ComplexFloat16
[`ComplexFloat32`]: crate::array::DataType::ComplexFloat32
[`ComplexFloat64`]: crate::array::DataType::ComplexFloat64
[`ComplexFloat4E2M1FN`]: crate::array::DataType::ComplexFloat4E2M1FN`
[`ComplexFloat6E2M3FN`]: crate::array::DataType::ComplexFloat6E2M3FN`
[`ComplexFloat6E3M2FN`]: crate::array::DataType::ComplexFloat6E3M2FN`
[`ComplexFloat8E3M4`]: crate::array::DataType::ComplexFloat8E3M4`
[`ComplexFloat8E4M3`]: crate::array::DataType::ComplexFloat8E4M3`
[`ComplexFloat8E4M3B11FNUZ`]: crate::array::DataType::ComplexFloat8E4M3B11FNUZ
[`ComplexFloat8E4M3FNUZ`]: crate::array::DataType::ComplexFloat8E4M3FNUZ`
[`ComplexFloat8E5M2`]: crate::array::DataType::ComplexFloat8E5M2`
[`ComplexFloat8E5M2FNUZ`]: crate::array::DataType::ComplexFloat8E5M2FNUZ`
[`ComplexFloat8E8M0FNU`]: crate::array::DataType::ComplexFloat8E8M0FNU`
[`Complex64`]: crate::array::DataType::Complex64
[`Complex128`]: crate::array::DataType::Complex128
[`RawBits`]: crate::array::DataType::RawBits
[`String`]: crate::array::DataType::String
[`Bytes`]: crate::array::DataType::Bytes
[`NumpyDateTime64`]: crate::array::DataType::NumpyDateTime64
[`NumpyTimeDelta64`]: crate::array::DataType::NumpyTimeDelta64

[`Element`]: crate::array::Element
[`ElementOwned`]: crate::array::ElementOwned

[`Complex<half::bf16>`]: num::complex::Complex<half::bf16>
[`Complex<half::f16>`]: num::complex::Complex<half::f16>
[`Complex<f32>`]: num::complex::Complex<f32>
[`Complex<f64>`]: num::complex::Complex<f64>
[`Complex<f32>`]: num::complex::Complex<f32>
[`Complex<f64>`]: num::complex::Complex<f64>

[ZEP0001]: https://zarr.dev/zeps/accepted/ZEP0001.html
[zarr-specs #130]: https://github.com/zarr-developers/zarr-specs/issues/130
[ZEP0007 (draft)]: https://github.com/zarr-developers/zeps/pull/47
[data-types/string]: https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/string
[data-types/bytes]: https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/bytes
[data-types/complex_bfloat16]: https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/complex_bfloat16
[data-types/complex_float16]: https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/complex_float16
[data-types/complex_float32]: https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/complex_float32
[data-types/complex_float64]: https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/complex_float64
