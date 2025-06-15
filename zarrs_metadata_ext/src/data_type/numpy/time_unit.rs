use derive_more::Display;
use serde::{Deserialize, Serialize};

/// A `NumPy` time unit (for `datetime64`/`timedelta64`).
#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Clone, Copy, Eq, PartialEq, Debug, Display)]
pub enum NumpyTimeUnit {
    #[serde(rename = "generic")]
    Generic,
    #[serde(rename = "Y")]
    Year,
    #[serde(rename = "M")]
    Month,
    #[serde(rename = "W")]
    Week,
    #[serde(rename = "D")]
    Day,
    #[serde(rename = "h")]
    Hour,
    #[serde(rename = "m")]
    Minute,
    #[serde(rename = "s")]
    Second,
    #[serde(rename = "ms")]
    Millisecond,
    #[serde(rename = "us", alias = "μs")]
    Microsecond,
    #[serde(rename = "ns")]
    Nanosecond,
    #[serde(rename = "ps")]
    Picosecond,
    #[serde(rename = "fs")]
    Femtosecond,
    #[serde(rename = "as")]
    Attosecond,
}
