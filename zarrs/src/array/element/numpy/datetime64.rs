#![allow(unused_imports)]

use crate::array::{
    ArrayBytes, ArrayError, DataType, Element, ElementOwned, convert_from_bytes_slice,
    data_type::NumpyDateTime64DataType,
};
use zarrs_plugin::ExtensionIdentifier;

#[cfg(feature = "chrono")]
impl Element for chrono::DateTime<chrono::Utc> {
    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ArrayError> {
        use chrono::DateTime;

        // Self::validate_data_type(data_type)?;
        let Some(dt) = data_type.as_any().downcast_ref::<NumpyDateTime64DataType>() else {
            return Err(ArrayError::IncompatibleElementType);
        };
        let (unit, scale_factor) = (dt.unit, dt.scale_factor);
        let mut bytes: Vec<u8> = Vec::with_capacity(elements.len() * size_of::<u64>());
        let scale_factor = i64::from(scale_factor.get());
        for element in elements {
            if element == &Self::MIN_UTC {
                bytes.extend_from_slice(&i64::MIN.to_ne_bytes());
            } else {
                let value = super::chrono_timedelta_to_int(
                    // why is this API self?
                    (*element).signed_duration_since(DateTime::UNIX_EPOCH),
                    unit,
                    scale_factor,
                )
                .ok_or_else(|| {
                    ArrayError::Other("unsupported chrono::DateTime unit or offset".to_string())
                })?;
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
        }
        Ok(bytes.into())
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ArrayError> {
        Ok(Self::to_array_bytes(data_type, &elements)?.into_owned())
    }

    fn validate_data_type(data_type: &DataType) -> Result<(), ArrayError> {
        if data_type.identifier() == NumpyDateTime64DataType::IDENTIFIER {
            Ok(())
        } else {
            Err(ArrayError::IncompatibleElementType)
        }
    }
}

#[cfg(feature = "chrono")]
impl ElementOwned for chrono::DateTime<chrono::Utc> {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ArrayError> {
        use chrono::{DateTime, NaiveDateTime};

        let Some(dt) = data_type.as_any().downcast_ref::<NumpyDateTime64DataType>() else {
            return Err(ArrayError::IncompatibleElementType);
        };
        let (unit, scale_factor) = (dt.unit, dt.scale_factor);

        let bytes = bytes.into_fixed()?;
        let elements = convert_from_bytes_slice::<i64>(&bytes);

        let scale_factor = i64::from(scale_factor.get());
        let datetimes = elements
            .into_iter()
            .map(|i| {
                if i == i64::MIN {
                    Ok(Self::MIN_UTC)
                } else {
                    let timedelta = super::int_to_chrono_timedelta(i, unit, scale_factor)
                        .ok_or_else(|| {
                            ArrayError::Other(
                                "unsupported chrono::DateTime unit or offset".to_string(),
                            )
                        })?;

                    Ok(Self::default() + timedelta)
                }
            })
            .collect::<Result<Vec<_>, ArrayError>>()?;

        Ok(datetimes)
    }
}

#[cfg(feature = "jiff")]
impl Element for jiff::Timestamp {
    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ArrayError> {
        use jiff::{Span, Timestamp, Unit};

        // Self::validate_data_type(data_type)?;
        let Some(dt) = data_type.as_any().downcast_ref::<NumpyDateTime64DataType>() else {
            return Err(ArrayError::IncompatibleElementType);
        };
        let (unit, scale_factor) = (dt.unit, dt.scale_factor);
        let mut bytes: Vec<u8> = Vec::with_capacity(elements.len() * size_of::<u64>());
        let error = |e: jiff::Error| ArrayError::Other(e.to_string());
        let scale_factor = i64::from(scale_factor.get());
        for element in elements {
            if element == &Timestamp::MIN {
                bytes.extend_from_slice(&i64::MIN.to_ne_bytes());
            } else {
                let value = super::jiff_duration_to_int(
                    element.duration_since(Timestamp::UNIX_EPOCH),
                    unit,
                    scale_factor,
                )
                .map_err(error)?;
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
        }
        Ok(bytes.into())
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ArrayError> {
        Ok(Self::to_array_bytes(data_type, &elements)?.into_owned())
    }

    fn validate_data_type(data_type: &DataType) -> Result<(), ArrayError> {
        if data_type.identifier() == NumpyDateTime64DataType::IDENTIFIER {
            Ok(())
        } else {
            Err(ArrayError::IncompatibleElementType)
        }
    }
}

#[cfg(feature = "jiff")]
impl ElementOwned for jiff::Timestamp {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ArrayError> {
        use jiff::{SignedDuration, Span, Timestamp};

        // Self::validate_data_type(data_type)?;
        let Some(dt) = data_type.as_any().downcast_ref::<NumpyDateTime64DataType>() else {
            return Err(ArrayError::IncompatibleElementType);
        };
        let (unit, scale_factor) = (dt.unit, dt.scale_factor);

        let bytes = bytes.into_fixed()?;
        let elements = convert_from_bytes_slice::<i64>(&bytes);
        let scale_factor = i64::from(scale_factor.get());
        let timestamps = elements
            .into_iter()
            .map(|i| {
                if i == i64::MIN {
                    Ok(Timestamp::MIN)
                } else {
                    Timestamp::from_duration(super::int_to_jiff_duration(i, unit, scale_factor)?)
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ArrayError::Other(e.to_string()))?;
        Ok(timestamps)
    }
}
