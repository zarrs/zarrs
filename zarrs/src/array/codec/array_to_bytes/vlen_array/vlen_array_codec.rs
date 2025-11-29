use crate::registry::codec::VLEN_ARRAY;

use crate::array::codec::array_to_bytes::vlen_v2::vlen_v2_macros;

vlen_v2_macros::vlen_v2_codec!(VlenArrayCodec, VLEN_ARRAY);
