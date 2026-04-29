| [`DataType`]                  | V3 `data_type` `name`        | V2 `dtype`    | [`ElementOwned`] / [`Element`]<br>(Feature Flag) |
| ----------------------------- | ---------------------------- | ------------- | ------------------------------------------------ |
| [`Bool`]                      | `bool`                       | `\|b1`        | [`bool`]                                         |
| [`Int2`]                      | `int2`                       |               | [`i8`]                                           |
| [`Int4`]                      | `int4`                       |               | [`i8`]                                           |
| [`Int8`]                      | `int8`                       | `\|i1`        | [`i8`]                                           |
| [`Int16`]                     | `int16`                      | `>i2` `<i2`   | [`i16`]                                          |
| [`Int32`]                     | `int32`                      | `>i4` `<i4`   | [`i32`]                                          |
| [`Int64`]                     | `int64`                      | `>i8` `<i8`   | [`i64`]                                          |
| [`UInt2`]                     | `uint2`                      |               | [`u8`]                                           |
| [`UInt4`]                     | `uint4`                      |               | [`u8`]                                           |
| [`UInt8`]                     | `uint8`                      | `\|u1`        | [`u8`]                                           |
| [`UInt16`]                    | `uint16`                     | `>u2` `<u2`   | [`u16`]                                          |
| [`UInt32`]                    | `uint32`                     | `>u4` `<u4`   | [`u32`]                                          |
| [`UInt64`]                    | `uint64`                     | `>u8` `<u8`   | [`u64`]                                          |
| [`Float4E2M1FN`]†             | `float4_e2m1fn`              |               | [`microfloat::f4e2m1fn`] (`microfloat`)          |
| [`Float6E2M3FN`]†             | `float6_e2m3fn`              |               | [`microfloat::f6e2m3fn`] (`microfloat`)          |
| [`Float6E3M2FN`]†             | `float6_e3m2fn`              |               | [`microfloat::f6e3m2fn`] (`microfloat`)          |
| [`Float8E3M4`]†               | `float8_e3m4`                |               | [`microfloat::f8e3m4`] (`microfloat`)            |
| [`Float8E4M3`]†               | `float8_e4m3`                |               |  [`microfloat::f8e4m3`] (`microfloat`)<br>[`float8::F8E4M3`] (`float8`) |
| [`Float8E4M3B11FNUZ`]†        | `float8_e4m3b11fnuz`         |               | [`microfloat::f8e4m3b11fnuz`] (`microfloat`)     |
| [`Float8E4M3FNUZ`]†           | `float8_e4m3fnuz`            |               | [`microfloat::f8e4m3fnuz`] (`microfloat`)        |
| [`Float8E5M2`]†               | `float8_e5m2`                |               | [`microfloat::f8e5m2`] (`microfloat`)<br>[`float8::F8E5M2`] (`float8`) |
| [`Float8E5M2FNUZ`]†           | `float8_e5m2fnuz`            |               | [`microfloat::f8e5m2fnuz`] (`microfloat`)        |
| [`Float8E8M0FNU`]†            | `float8_e8m0fnu`             |               | [`microfloat::f8e8m0fnu`] (`microfloat`)         |
| [`BFloat16`]                  | `bfloat16`                   |               | [`half::bf16`]                                   |
| [`Float16`]                   | `float16`                    | `>f2` `<f2`   | [`half::f16`]                                    |
| [`Float32`]                   | `float32`                    | `>f4` `<f4`   | [`f32`]                                          |
| [`Float64`]                   | `float64`                    | `>f8` `<f8`   | [`f64`]                                          |
| [`Complex64`]                 | `complex64`                  | `>c8` `<c8`   | [`Complex<f32>`]                                 |
| [`Complex128`]                | `complex128`                 | `>c16` `<c16` | [`Complex<f64>`]                                 |
| [`ComplexBFloat16`]           | `complex_bfloat16`           |               | [`Complex<half::bf16>`]                          |
| [`ComplexFloat16`]            | `complex_float16`            |               | [`Complex<half::f16>`]                           |
| [`ComplexFloat32`]            | `complex_float32`            |               | [`Complex<f32>`]                                 |
| [`ComplexFloat64`]            | `complex_float64`            |               | [`Complex<f64>`]                                 |
| [`ComplexFloat4E2M1FN`]†      | `complex_float4_e2m1fn`      |               | [`Complex<microfloat::f4e2m1fn>`] (`microfloat`) |
| [`ComplexFloat6E2M3FN`]†      | `complex_float6_e2m3fn`      |               | [`Complex<microfloat::f6e2m3fn>`] (`microfloat`) |
| [`ComplexFloat6E3M2FN`]†      | `complex_float6_e3m2fn`      |               | [`Complex<microfloat::f6e3m2fn>`] (`microfloat`) |
| [`ComplexFloat8E3M4`]†        | `complex_float8_e3m4`        |               | [`Complex<microfloat::f8e3m4>`] (`microfloat`)   |
| [`ComplexFloat8E4M3`]†        | `complex_float8_e4m3`        |               | [`Complex<float8::F8E4M3>`] (`float8`)<br>[`Complex<microfloat::f8e4m3>`] (`microfloat`) |
| [`ComplexFloat8E4M3B11FNUZ`]† | `complex_float8_e4m3b11fnuz` |               | [`Complex<microfloat::f8e4m3b11fnuz>`] (`microfloat`) |
| [`ComplexFloat8E4M3FNUZ`]†    | `complex_float8_e4m3fnuz`    |               | [`Complex<microfloat::f8e4m3fnuz>`] (`microfloat`) |
| [`ComplexFloat8E5M2`]†        | `complex_float8_e5m2`        |               | [`Complex<float8::F8E5M2>`] (`float8`)<br>[`Complex<microfloat::f8e5m2>`] (`microfloat`) |
| [`ComplexFloat8E5M2FNUZ`]†    | `complex_float8_e5m2fnuz`    |               | [`Complex<microfloat::f8e5m2fnuz>`] (`microfloat`) |
| [`ComplexFloat8E8M0FNU`]†     | `complex_float8_e8m0fnu`     |               | [`Complex<microfloat::f8e8m0fnu>`] (`microfloat`) |
| [`Optional`]                  | 🚧`zarrs.optional`           |               | [`Option`]                                       |
| [`RawBits`]                   | `r*`                         | `\|V*`        | `[u8; N]` / `&[u8; N]`                           |
| [`String`]                    | `string`                     | `\|O`         | [`String`] / [`&str`]                            |
| [`Bytes`]                     | `bytes`<br>~~`binary`~~<br>🚧`variable_length_bytes` | `\|VX`        | [`Vec<u8>`] / `&[u8]`                            |
| [`NumpyDateTime64`]           | `numpy.datetime64`           |               | [`i64`]<br>[`chrono::DateTime<Utc>`] (`chrono`)<br>[`jiff::Timestamp`] (`jiff`)  |
| [`NumpyTimeDelta64`]          | `numpy.timedelta64`          |               | [`i64`]<br>[`chrono::TimeDelta`] (`chrono`)<br>[`jiff::SignedDuration`] (`jiff`) |

<sup>† Additional features (e.g. `float8`) may be required to parse floating point fill values. All subfloat types support hex string fill values.</sup>

[`DataType`]: crate::array::data_type

[`Bool`]: crate::array::data_type::BoolDataType
[`Int2`]: crate::array::data_type::Int2DataType
[`Int4`]: crate::array::data_type::Int4DataType
[`Int8`]: crate::array::data_type::Int8DataType
[`Int16`]: crate::array::data_type::Int16DataType
[`Int32`]: crate::array::data_type::Int32DataType
[`Int64`]: crate::array::data_type::Int64DataType
[`UInt2`]: crate::array::data_type::UInt2DataType
[`UInt4`]: crate::array::data_type::UInt4DataType
[`UInt8`]: crate::array::data_type::UInt8DataType
[`UInt16`]: crate::array::data_type::UInt16DataType
[`UInt32`]: crate::array::data_type::UInt32DataType
[`UInt64`]: crate::array::data_type::UInt64DataType
[`Float4E2M1FN`]: crate::array::data_type::Float4E2M1FNDataType
[`Float6E2M3FN`]: crate::array::data_type::Float6E2M3FNDataType
[`Float6E3M2FN`]: crate::array::data_type::Float6E3M2FNDataType
[`Float8E3M4`]: crate::array::data_type::Float8E3M4DataType
[`Float8E4M3`]: crate::array::data_type::Float8E4M3DataType
[`Float8E4M3B11FNUZ`]: crate::array::data_type::Float8E4M3B11FNUZDataType
[`Float8E4M3FNUZ`]: crate::array::data_type::Float8E4M3FNUZDataType
[`Float8E5M2`]: crate::array::data_type::Float8E5M2DataType
[`Float8E5M2FNUZ`]: crate::array::data_type::Float8E5M2FNUZDataType
[`Float8E8M0FNU`]: crate::array::data_type::Float8E8M0FNUDataType
[`BFloat16`]: crate::array::data_type::BFloat16DataType
[`Float16`]: crate::array::data_type::Float16DataType
[`Float32`]: crate::array::data_type::Float32DataType
[`Float64`]: crate::array::data_type::Float64DataType
[`ComplexBFloat16`]: crate::array::data_type::ComplexBFloat16DataType
[`ComplexFloat16`]: crate::array::data_type::ComplexFloat16DataType
[`ComplexFloat32`]: crate::array::data_type::ComplexFloat32DataType
[`ComplexFloat64`]: crate::array::data_type::ComplexFloat64DataType
[`ComplexFloat4E2M1FN`]: crate::array::data_type::ComplexFloat4E2M1FN`DataType
[`ComplexFloat6E2M3FN`]: crate::array::data_type::ComplexFloat6E2M3FN`DataType
[`ComplexFloat6E3M2FN`]: crate::array::data_type::ComplexFloat6E3M2FN`DataType
[`ComplexFloat8E3M4`]: crate::array::data_type::ComplexFloat8E3M4`DataType
[`ComplexFloat8E4M3`]: crate::array::data_type::ComplexFloat8E4M3`DataType
[`ComplexFloat8E4M3B11FNUZ`]: crate::array::data_type::ComplexFloat8E4M3B11FNUZDataType
[`ComplexFloat8E4M3FNUZ`]: crate::array::data_type::ComplexFloat8E4M3FNUZ`DataType
[`ComplexFloat8E5M2`]: crate::array::data_type::ComplexFloat8E5M2`DataType
[`ComplexFloat8E5M2FNUZ`]: crate::array::data_type::ComplexFloat8E5M2FNUZ`DataType
[`ComplexFloat8E8M0FNU`]: crate::array::data_type::ComplexFloat8E8M0FNU`DataType
[`Complex64`]: crate::array::data_type::Complex64DataType
[`Complex128`]: crate::array::data_type::Complex128DataType
[`Optional`]: crate::array::data_type::OptionalDataType
[`RawBits`]: crate::array::data_type::RawBitsDataType
[`String`]: crate::array::data_type::StringDataType
[`Bytes`]: crate::array::data_type::BytesDataType
[`NumpyDateTime64`]: crate::array::data_type::NumpyDateTime64DataType
[`NumpyTimeDelta64`]: crate::array::data_type::NumpyTimeDelta64DataType

[`Element`]: crate::array::Element
[`ElementOwned`]: crate::array::ElementOwned

[`Complex<half::bf16>`]: num::complex::Complex<half::bf16>
[`Complex<half::f16>`]: num::complex::Complex<half::f16>
[`Complex<f32>`]: num::complex::Complex<f32>
[`Complex<f64>`]: num::complex::Complex<f64>
[`Complex<f32>`]: num::complex::Complex<f32>
[`Complex<f64>`]: num::complex::Complex<f64>
[`Complex<float8::F8E4M3>`]: num::complex::Complex<float8::F8E4M3>
[`Complex<float8::F8E5M2>`]: num::complex::Complex<float8::F8E5M2>
[`Complex<microfloat::f4e2m1fn>`]: num::complex::Complex<microfloat::f4e2m1fn>
[`Complex<microfloat::f6e2m3fn>`]: num::complex::Complex<microfloat::f6e2m3fn>
[`Complex<microfloat::f6e3m2fn>`]: num::complex::Complex<microfloat::f6e3m2fn>
[`Complex<microfloat::f8e3m4>`]: num::complex::Complex<microfloat::f8e3m4>
[`Complex<microfloat::f8e4m3>`]: num::complex::Complex<microfloat::f8e4m3>
[`Complex<microfloat::f8e4m3b11fnuz>`]: num::complex::Complex<microfloat::f8e4m3b11fnuz>
[`Complex<microfloat::f8e4m3fnuz>`]: num::complex::Complex<microfloat::f8e4m3fnuz>
[`Complex<microfloat::f8e5m2>`]: num::complex::Complex<microfloat::f8e5m2>
[`Complex<microfloat::f8e5m2fnuz>`]: num::complex::Complex<microfloat::f8e5m2fnuz>
[`Complex<microfloat::f8e8m0fnu>`]: num::complex::Complex<microfloat::f8e8m0fnu>
[`microfloat::f4e2m1fn`]: microfloat::f4e2m1fn
[`microfloat::f6e2m3fn`]: microfloat::f6e2m3fn
[`microfloat::f6e3m2fn`]: microfloat::f6e3m2fn
[`microfloat::f8e3m4`]: microfloat::f8e3m4
[`microfloat::f8e4m3`]: microfloat::f8e4m3
[`microfloat::f8e4m3b11fnuz`]: microfloat::f8e4m3b11fnuz
[`microfloat::f8e4m3fnuz`]: microfloat::f8e4m3fnuz
[`microfloat::f8e5m2`]: microfloat::f8e5m2
[`microfloat::f8e5m2fnuz`]: microfloat::f8e5m2fnuz
[`microfloat::f8e8m0fnu`]: microfloat::f8e8m0fnu

[ZEP0001]: https://zarr.dev/zeps/accepted/ZEP0001.html
[zarr-specs #130]: https://github.com/zarr-developers/zarr-specs/issues/130
[ZEP0007 (draft)]: https://github.com/zarr-developers/zeps/pull/47
[data-types/string]: https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/string
[data-types/bytes]: https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/bytes
[data-types/complex_bfloat16]: https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/complex_bfloat16
[data-types/complex_float16]: https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/complex_float16
[data-types/complex_float32]: https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/complex_float32
[data-types/complex_float64]: https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/complex_float64
