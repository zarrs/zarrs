use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use super::*;
use crate::array::codec::BytesCodec;
use crate::array::{
    ArrayBytes, ArraySubset, DataType, Element, FillValue, data_type, transmute_to_bytes_vec,
};
use zarrs_codec::{
    ArrayPartialDecoderTraits, ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, CodecOptions,
};
use zarrs_storage::StorageError;

fn nz_shape(shape: &[u64]) -> Vec<NonZeroU64> {
    shape
        .iter()
        .map(|&dim| NonZeroU64::new(dim).unwrap())
        .collect()
}

fn encoded_u32_values(shape: &[u64]) -> Vec<u32> {
    morton_order(shape)
        .unwrap()
        .into_iter()
        .map(|element| u32::try_from(element.decoded_linear_index).unwrap())
        .collect()
}

#[test]
fn morton_order_power_of_two_anisotropic() {
    // Shape [x=2, y=4, z=8] uses bit planes:
    //
    //   z2, y1, z1, x0, y0, z0
    //
    // The first 8 encoded positions therefore keep high z/y planes at zero and
    // visit the 2x2x2 block at x=0..2, y=0..2, z=0..2:
    //
    //   encoded:  0  1  2  3   4   5   6   7
    //   coord:   000 001 010 011 100 101 110 111  (x y z low bits)
    //   rowmaj:  0  1  8  9  32  33  40  41
    assert_eq!(
        encoded_u32_values(&[2, 4, 8])[0..8],
        [0, 1, 8, 9, 32, 33, 40, 41]
    );

    // Shape [x=8, y=2, z=4] uses bit planes:
    //
    //   x2, x1, z1, x0, y0, z0
    //
    // The first 8 encoded positions again have high planes zero, then enumerate
    // the low x/y/z bits:
    //
    //   encoded:  0  1  2  3  4  5   6   7
    //   coord:   000 001 010 011 100 101 110 111  (x y z low bits)
    //   rowmaj:  0  1  4  5  8  9  12  13
    assert_eq!(
        encoded_u32_values(&[8, 2, 4])[0..8],
        [0, 1, 4, 5, 8, 9, 12, 13]
    );
}

#[test]
fn morton_order_skips_padding_for_non_power_of_two() {
    // Shape [3, 3] is traversed as if padded to [4, 4], but padded coordinates
    // are skipped and do not consume encoded elements:
    //
    //   decoded row-major values:
    //      y=0:   0  1  2
    //      y=1:   3  4  5
    //      y=2:   6  7  8
    //
    //   padded 4x4 Morton order, shown as decoded values or xx for padding:
    //      0, 1, 3, 4, 2, xx, 5, xx, 6, 7, xx, xx, 8, xx, xx, xx
    //
    //   compact valid order:
    //      0, 1, 3, 4, 2, 5, 6, 7, 8
    let values = encoded_u32_values(&[3, 3]);
    assert_eq!(values, [0, 1, 3, 4, 2, 5, 6, 7, 8]);
    assert_eq!(values.len(), 9);
}

fn round_trip(
    bytes: ArrayBytes<'static>,
    shape: &[u64],
    data_type: &DataType,
    fill_value: FillValue,
) {
    let codec = MortonCodec::new();
    let shape = nz_shape(shape);
    let encoded = codec
        .encode(
            bytes.clone(),
            &shape,
            data_type,
            &fill_value,
            &CodecOptions::default(),
        )
        .unwrap();
    let decoded = codec
        .decode(
            encoded,
            &shape,
            data_type,
            &fill_value,
            &CodecOptions::default(),
        )
        .unwrap();
    assert_eq!(decoded, bytes);
}

#[test]
fn codec_morton_round_trip_fixed_variable_and_optional() {
    let fixed_data_type = data_type::uint32();
    let fixed = transmute_to_bytes_vec((0..18).collect::<Vec<u32>>()).into();
    round_trip(fixed, &[3, 3, 2], &fixed_data_type, FillValue::from(0u32));

    let variable_data_type = data_type::string();
    let strings = (0..12).map(|index| format!("s{index}")).collect::<Vec<_>>();
    let variable = String::into_array_bytes(&variable_data_type, strings).unwrap();
    round_trip(variable, &[3, 4], &variable_data_type, FillValue::from(""));

    let optional_data_type = data_type::uint16().to_optional();
    let optional_values = (0..12)
        .map(|index| (index % 3 != 0).then_some(u16::try_from(index).unwrap()))
        .collect::<Vec<_>>();
    let optional = Option::<u16>::into_array_bytes(&optional_data_type, optional_values).unwrap();
    round_trip(optional, &[3, 4], &optional_data_type, None::<u16>.into());
}

#[test]
fn codec_morton_partial_decode_regions() {
    // Decoded row-major chunk shape [3, 5]:
    //
    //   r0:  0  1  2  3  4
    //   r1:  5  6  7  8  9
    //   r2: 10 11 12 13 14
    //
    // The cases below verify full-chunk, center/right, corner, and top-left
    // subsets are returned in decoded row-major order after Morton reads.
    let codec = Arc::new(MortonCodec::new());
    let shape = nz_shape(&[3, 5]);
    let data_type = data_type::uint32();
    let fill_value = FillValue::from(0u32);
    let elements = (0..15).collect::<Vec<u32>>();
    let bytes: ArrayBytes = transmute_to_bytes_vec(elements.clone()).into();
    let encoded = codec
        .encode(
            bytes,
            &shape,
            &data_type,
            &fill_value,
            &CodecOptions::default(),
        )
        .unwrap();

    let bytes_codec = Arc::new(BytesCodec::default());
    let encoded_shape = codec.encoded_shape(&shape).unwrap();
    let input_handle = bytes_codec
        .partial_decoder(
            Arc::new(encoded.into_fixed().unwrap()),
            &encoded_shape,
            &data_type,
            &fill_value,
            &CodecOptions::default(),
        )
        .unwrap();
    let partial_decoder = codec
        .partial_decoder(
            input_handle,
            &shape,
            &data_type,
            &fill_value,
            &CodecOptions::default(),
        )
        .unwrap();

    let cases = [
        (
            ArraySubset::new_with_ranges(&[0..3, 0..5]),
            elements.clone(),
        ),
        (
            ArraySubset::new_with_ranges(&[1..3, 2..5]),
            vec![7, 8, 9, 12, 13, 14],
        ),
        (ArraySubset::new_with_ranges(&[2..3, 4..5]), vec![14]),
        (
            ArraySubset::new_with_ranges(&[0..2, 0..2]),
            vec![0, 1, 5, 6],
        ),
    ];

    for (subset, expected) in cases {
        let decoded = partial_decoder
            .partial_decode(&subset, &CodecOptions::default())
            .unwrap();
        let decoded = crate::array::convert_from_bytes_slice::<u32>(&decoded.into_fixed().unwrap());
        assert_eq!(decoded, expected);
    }
}

#[test]
fn codec_morton_partial_encode_updates_correct_elements() {
    // Start with a 4x4 decoded chunk:
    //
    //      0   1   2   3
    //      4   5   6   7
    //      8   9  10  11
    //     12  13  14  15
    //
    // Write subset rows 1..3, cols 1..3 with decoded-order values:
    //
    //     100 101
    //     102 103
    //
    // Expected decoded result:
    //
    //      0   1   2   3
    //      4 100 101   7
    //      8 102 103  11
    //     12  13  14  15
    let codec = Arc::new(MortonCodec::new());
    let shape = nz_shape(&[4, 4]);
    let data_type = data_type::uint32();
    let fill_value = FillValue::from(0u32);
    let original = (0..16).collect::<Vec<u32>>();
    let bytes: ArrayBytes = transmute_to_bytes_vec(original.clone()).into();
    let encoded = codec
        .encode(
            bytes,
            &shape,
            &data_type,
            &fill_value,
            &CodecOptions::default(),
        )
        .unwrap();

    let bytes_codec = Arc::new(BytesCodec::default());
    let encoded_shape = codec.encoded_shape(&shape).unwrap();
    let input_output_handle = bytes_codec
        .partial_encoder(
            Arc::new(Mutex::new(Some(encoded.into_fixed().unwrap().into_owned()))),
            &encoded_shape,
            &data_type,
            &fill_value,
            &CodecOptions::default(),
        )
        .unwrap();
    let partial_encoder = codec
        .partial_encoder(
            input_output_handle.clone(),
            &shape,
            &data_type,
            &fill_value,
            &CodecOptions::default(),
        )
        .unwrap();

    let update_subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
    let update_values = vec![100u32, 101, 102, 103];
    let update_bytes: ArrayBytes = transmute_to_bytes_vec(update_values).into();
    partial_encoder
        .partial_encode(&update_subset, &update_bytes, &CodecOptions::default())
        .unwrap();

    let partial_decoder = partial_encoder.into_dyn_decoder();
    let decoded = partial_decoder
        .partial_decode(
            &ArraySubset::new_with_ranges(&[0..4, 0..4]),
            &CodecOptions::default(),
        )
        .unwrap();
    let decoded = crate::array::convert_from_bytes_slice::<u32>(&decoded.into_fixed().unwrap());
    let expected = vec![0, 1, 2, 3, 4, 100, 101, 7, 8, 102, 103, 11, 12, 13, 14, 15];
    assert_eq!(decoded, expected);
}

struct RecordingDecoder {
    bytes: ArrayBytes<'static>,
    data_type: DataType,
    requested_runs: Mutex<Vec<Vec<(u64, u64)>>>,
}

impl ArrayPartialDecoderTraits for RecordingDecoder {
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn exists(&self) -> Result<bool, StorageError> {
        Ok(true)
    }

    fn size_held(&self) -> usize {
        self.bytes.size()
    }

    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let runs = indexer
            .iter_contiguous_linearised_indices(&[16])?
            .collect::<Vec<_>>();
        self.requested_runs.lock().unwrap().push(runs);
        self.bytes
            .extract_array_subset(indexer, &[16], &self.data_type)
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

#[test]
fn codec_morton_partial_decode_uses_exact_coalesced_runs() {
    // For shape [4, 4], decoded row-major values are:
    //
    //      0  1  2  3
    //      4  5  6  7
    //      8  9 10 11
    //     12 13 14 15
    //
    // Compact Morton encoded stream is:
    //
    //      pos:    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
    //      value:  0  1  4  5  2  3  6  7  8  9 12 13 10 11 14 15
    //
    // Requesting decoded row 0 is values [0, 1, 2, 3], which maps exactly to
    // encoded runs [0..2] and [4..6]. The recording decoder checks those runs.
    let codec = Arc::new(MortonCodec::new());
    let shape = nz_shape(&[4, 4]);
    let data_type = data_type::uint32();
    let fill_value = FillValue::from(0u32);
    let bytes: ArrayBytes = transmute_to_bytes_vec((0..16).collect::<Vec<u32>>()).into();
    let encoded = codec
        .encode(
            bytes,
            &shape,
            &data_type,
            &fill_value,
            &CodecOptions::default(),
        )
        .unwrap();
    let recording_decoder = Arc::new(RecordingDecoder {
        bytes: encoded,
        data_type: data_type.clone(),
        requested_runs: Mutex::new(Vec::new()),
    });
    let partial_decoder = codec
        .partial_decoder(
            recording_decoder.clone(),
            &shape,
            &data_type,
            &fill_value,
            &CodecOptions::default(),
        )
        .unwrap();

    let decoded = partial_decoder
        .partial_decode(
            &ArraySubset::new_with_ranges(&[0..1, 0..4]),
            &CodecOptions::default(),
        )
        .unwrap();
    let decoded = crate::array::convert_from_bytes_slice::<u32>(&decoded.into_fixed().unwrap());
    assert_eq!(decoded, vec![0, 1, 2, 3]);
    assert_eq!(
        recording_decoder.requested_runs.lock().unwrap().as_slice(),
        &[vec![(0, 2), (4, 2)]]
    );
}
