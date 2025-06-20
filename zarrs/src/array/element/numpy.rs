#[allow(unused)]
use zarrs_metadata_ext::data_type::NumpyTimeUnit;

mod datetime64;
mod timedelta64;

#[cfg(feature = "chrono")]
pub(super) fn chrono_timedelta_to_int(
    timedelta: chrono::TimeDelta,
    unit: NumpyTimeUnit,
    scale_factor: i64,
) -> Option<i64> {
    match unit {
        NumpyTimeUnit::Generic | NumpyTimeUnit::Year | NumpyTimeUnit::Month => None,
        NumpyTimeUnit::Week => timedelta.num_days().checked_mul(7),
        NumpyTimeUnit::Day => Some(timedelta.num_days()),
        NumpyTimeUnit::Hour => Some(timedelta.num_hours()),
        NumpyTimeUnit::Minute => Some(timedelta.num_minutes()),
        NumpyTimeUnit::Second => Some(timedelta.num_seconds()),
        NumpyTimeUnit::Millisecond => Some(timedelta.num_milliseconds()),
        NumpyTimeUnit::Microsecond => timedelta.num_microseconds(),
        NumpyTimeUnit::Nanosecond => timedelta.num_nanoseconds(),
        NumpyTimeUnit::Picosecond => timedelta.checked_mul(1_000)?.num_nanoseconds(),
        NumpyTimeUnit::Femtosecond => timedelta.checked_mul(1_000_000)?.num_nanoseconds(),
        NumpyTimeUnit::Attosecond => timedelta.checked_mul(1_000_000_000)?.num_nanoseconds(),
    }?
    .checked_div(scale_factor)
}

#[cfg(feature = "chrono")]
pub(super) fn int_to_chrono_timedelta(
    i: i64,
    unit: NumpyTimeUnit,
    scale_factor: i64,
) -> Option<chrono::TimeDelta> {
    use chrono::TimeDelta;
    let i = i.saturating_mul(scale_factor);
    match unit {
        NumpyTimeUnit::Generic | NumpyTimeUnit::Year | NumpyTimeUnit::Month => None, // year/month units are not strictly correct with this API
        NumpyTimeUnit::Week => TimeDelta::try_weeks(i),
        NumpyTimeUnit::Day => TimeDelta::try_days(i),
        NumpyTimeUnit::Hour => TimeDelta::try_hours(i),
        NumpyTimeUnit::Minute => TimeDelta::try_minutes(i),
        NumpyTimeUnit::Second => TimeDelta::try_seconds(i),
        NumpyTimeUnit::Millisecond => TimeDelta::try_milliseconds(i),
        NumpyTimeUnit::Microsecond => Some(TimeDelta::microseconds(i)),
        NumpyTimeUnit::Nanosecond => Some(TimeDelta::nanoseconds(i)),
        NumpyTimeUnit::Picosecond => Some(TimeDelta::nanoseconds(i / 1_000)),
        NumpyTimeUnit::Femtosecond => Some(TimeDelta::nanoseconds(i / 1_000_000)),
        NumpyTimeUnit::Attosecond => Some(TimeDelta::nanoseconds(i / 1_000_000_000)),
    }
}

#[cfg(feature = "jiff")]
pub(super) fn jiff_duration_to_int(
    duration: jiff::SignedDuration,
    unit: NumpyTimeUnit,
    scale_factor: i64,
) -> Result<i64, jiff::Error> {
    use jiff::Unit;
    let span = jiff::Span::try_from(duration)?;
    let delta = match unit {
        NumpyTimeUnit::Year => span.total(Unit::Year)?,
        NumpyTimeUnit::Month => span.total(Unit::Month)?,
        NumpyTimeUnit::Week => span.total(Unit::Week)?,
        NumpyTimeUnit::Day => span.total(Unit::Day)?,
        NumpyTimeUnit::Hour => span.total(Unit::Hour)?,
        NumpyTimeUnit::Minute => span.total(Unit::Minute)?,
        NumpyTimeUnit::Second => span.total(Unit::Second)?,
        NumpyTimeUnit::Millisecond => span.total(Unit::Millisecond)?,
        NumpyTimeUnit::Microsecond => span.total(Unit::Microsecond)?,
        NumpyTimeUnit::Nanosecond => span.total(Unit::Nanosecond)?,
        NumpyTimeUnit::Picosecond => span.total(Unit::Nanosecond)? * 1e3,
        NumpyTimeUnit::Femtosecond => span.total(Unit::Nanosecond)? * 1e6,
        NumpyTimeUnit::Attosecond => span.total(Unit::Nanosecond)? * 1e9,
        NumpyTimeUnit::Generic => Err(jiff::Error::from_args(format_args!(
            "datetime64 generic unit is not supported"
        )))?,
    };
    #[allow(clippy::cast_precision_loss)]
    let delta = (delta / scale_factor as f64).trunc();
    #[allow(clippy::cast_possible_truncation)]
    Ok(delta as i64)
}

#[cfg(feature = "jiff")]
pub(super) fn int_to_jiff_duration(
    i: i64,
    unit: NumpyTimeUnit,
    scale_factor: i64,
) -> Result<jiff::SignedDuration, jiff::Error> {
    const EPOCH: jiff::civil::Date = jiff::civil::date(1970, 1, 1);
    let span = jiff::Span::new();
    let i = i.saturating_mul(scale_factor);
    match unit {
        NumpyTimeUnit::Generic => Err(jiff::Error::from_args(format_args!(
            "datetime64 generic unit is not supported"
        )))?,
        NumpyTimeUnit::Year => span.try_years(i),
        NumpyTimeUnit::Month => span.try_months(i),
        NumpyTimeUnit::Week => span.try_weeks(i),
        NumpyTimeUnit::Day => span.try_days(i),
        NumpyTimeUnit::Hour => span.try_hours(i),
        NumpyTimeUnit::Minute => span.try_minutes(i),
        NumpyTimeUnit::Second => span.try_seconds(i),
        NumpyTimeUnit::Millisecond => span.try_milliseconds(i),
        NumpyTimeUnit::Microsecond => span.try_microseconds(i),
        NumpyTimeUnit::Nanosecond => span.try_nanoseconds(i),
        NumpyTimeUnit::Picosecond => span.try_nanoseconds(i / 1_000),
        NumpyTimeUnit::Femtosecond => span.try_nanoseconds(i / 1_000_000),
        NumpyTimeUnit::Attosecond => span.try_nanoseconds(i / 1_000_000_000),
    }?
    .to_duration(EPOCH)
}
