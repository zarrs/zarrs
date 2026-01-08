#![allow(missing_docs)]
#![allow(unused)]

use std::error::Error;
use std::sync::Arc;

use zarrs::array::data_type;
use zarrs::metadata_ext::data_type::NumpyTimeUnit;

#[cfg(feature = "jiff")]
fn try_numpy_to_jiff_unit(unit: NumpyTimeUnit) -> Result<jiff::Unit, NumpyTimeUnit> {
    match unit {
        NumpyTimeUnit::Generic
        | NumpyTimeUnit::Picosecond
        | NumpyTimeUnit::Femtosecond
        | NumpyTimeUnit::Attosecond => Err(unit),
        NumpyTimeUnit::Year => Ok(jiff::Unit::Year),
        NumpyTimeUnit::Month => Ok(jiff::Unit::Month),
        NumpyTimeUnit::Week => Ok(jiff::Unit::Week),
        NumpyTimeUnit::Day => Ok(jiff::Unit::Day),
        NumpyTimeUnit::Hour => Ok(jiff::Unit::Hour),
        NumpyTimeUnit::Minute => Ok(jiff::Unit::Minute),
        NumpyTimeUnit::Second => Ok(jiff::Unit::Second),
        NumpyTimeUnit::Millisecond => Ok(jiff::Unit::Millisecond),
        NumpyTimeUnit::Microsecond => Ok(jiff::Unit::Microsecond),
        NumpyTimeUnit::Nanosecond => Ok(jiff::Unit::Nanosecond),
    }
}

#[cfg(any(feature = "chrono", feature = "jiff"))]
#[test]
fn zarr_python_v3_numpy_datetime_read() -> Result<(), Box<dyn Error>> {
    for (path, unit) in [
        (
            "tests/data/zarr_python_compat/datetime64[Y].zarr",
            NumpyTimeUnit::Year,
        ),
        // (
        //     "tests/data/zarr_python_compat/datetime64[M].zarr",
        //     NumpyTimeUnit::Month,
        // ),
        (
            "tests/data/zarr_python_compat/datetime64[W].zarr",
            NumpyTimeUnit::Week,
        ),
        (
            "tests/data/zarr_python_compat/datetime64[D].zarr",
            NumpyTimeUnit::Day,
        ),
        (
            "tests/data/zarr_python_compat/datetime64[h].zarr",
            NumpyTimeUnit::Hour,
        ),
        (
            "tests/data/zarr_python_compat/datetime64[s].zarr",
            NumpyTimeUnit::Second,
        ),
        (
            "tests/data/zarr_python_compat/datetime64[ms].zarr",
            NumpyTimeUnit::Millisecond,
        ),
        (
            "tests/data/zarr_python_compat/datetime64[us].zarr",
            NumpyTimeUnit::Microsecond,
        ),
        (
            "tests/data/zarr_python_compat/datetime64[ns].zarr",
            NumpyTimeUnit::Nanosecond,
        ),
        (
            "tests/data/zarr_python_compat/datetime64[10ms].zarr",
            NumpyTimeUnit::Millisecond,
        ),
        (
            "tests/data/zarr_python_compat/datetime64[10us].zarr",
            NumpyTimeUnit::Microsecond,
        ),
    ] {
        let store = Arc::new(zarrs::filesystem::FilesystemStore::new(path)?);
        let array = zarrs::array::Array::open(store.clone(), "/")?;
        let subset_all = array.subset_all();

        #[cfg(feature = "chrono")]
        // The underlying chrono API does not support year/month
        if !matches!(unit, NumpyTimeUnit::Year | NumpyTimeUnit::Month) {
            use chrono::{DateTime, Utc};

            let elements = array.retrieve_array_subset::<Vec<DateTime<Utc>>>(&subset_all)?;
            println!("{elements:?}");

            // Only subminute are not rounded
            if !matches!(unit, |NumpyTimeUnit::Week| NumpyTimeUnit::Day
                | NumpyTimeUnit::Hour
                | NumpyTimeUnit::Minute)
            {
                assert_eq!(
                    elements,
                    &[
                        DateTime::UNIX_EPOCH,
                        DateTime::<Utc>::MIN_UTC,
                        DateTime::parse_from_rfc3339("2005-02-03T00:00:00Z")
                            .unwrap()
                            .to_utc(),
                        DateTime::parse_from_rfc3339("2005-02-03T04:05:00Z")
                            .unwrap()
                            .to_utc(),
                        DateTime::parse_from_rfc3339("2005-02-03T04:05:06Z")
                            .unwrap()
                            .to_utc(),
                        DateTime::<Utc>::MIN_UTC,
                    ]
                );
            }
        }

        #[cfg(feature = "jiff")]
        {
            use jiff::{Timestamp, TimestampRound, Unit};
            let elements = array.retrieve_array_subset::<Vec<jiff::Timestamp>>(&subset_all)?;
            println!("{path:?}");
            println!("{elements:?}");

            // The jiff rounding API does not support years/months/weeks/days
            // https://github.com/BurntSushi/jiff/issues/1
            if !matches!(
                unit,
                NumpyTimeUnit::Year
                    | NumpyTimeUnit::Month
                    | NumpyTimeUnit::Week
                    | NumpyTimeUnit::Day
            ) {
                let round = TimestampRound::new().smallest(try_numpy_to_jiff_unit(unit).unwrap());
                assert_eq!(
                    elements,
                    &[
                        Timestamp::UNIX_EPOCH,
                        Timestamp::MIN,
                        "2005-02-03T00:00:00Z"
                            .parse::<Timestamp>()
                            .unwrap()
                            .round(round)
                            .unwrap(),
                        "2005-02-03T04:05Z"
                            .parse::<Timestamp>()
                            .unwrap()
                            .round(round)
                            .unwrap(),
                        "2005-02-03T04:05:06Z"
                            .parse::<Timestamp>()
                            .unwrap()
                            .round(round)
                            .unwrap(),
                        Timestamp::MIN,
                    ]
                );
            }
        }
    }

    Ok(())
}

#[cfg(any(feature = "chrono", feature = "jiff"))]
#[test]
fn zarr_python_v3_numpy_datetime_write() -> Result<(), Box<dyn Error>> {
    for unit in [
        NumpyTimeUnit::Year,
        NumpyTimeUnit::Month,
        NumpyTimeUnit::Week,
        NumpyTimeUnit::Day,
        NumpyTimeUnit::Hour,
        NumpyTimeUnit::Minute,
        NumpyTimeUnit::Second,
        NumpyTimeUnit::Millisecond,
        NumpyTimeUnit::Microsecond,
        NumpyTimeUnit::Nanosecond,
    ] {
        println!("{unit:?}");
        use zarrs::array::ArrayBuilder;
        use zarrs::storage::store::MemoryStore;
        use zarrs_data_type::FillValue;

        let store = Arc::new(MemoryStore::new());
        let array = ArrayBuilder::new(
            vec![6],
            vec![5],
            data_type::numpy_datetime64(unit, 1.try_into().unwrap()),
            i64::MIN,
        )
        .build(store.clone(), "/")?;
        array.store_metadata()?;

        #[cfg(feature = "chrono")]
        // The underlying chrono API does not support year/month
        if !matches!(unit, NumpyTimeUnit::Year | NumpyTimeUnit::Month) {
            use chrono::{DateTime, Utc};
            let elements = [
                DateTime::UNIX_EPOCH,
                DateTime::<Utc>::MIN_UTC,
                DateTime::parse_from_rfc3339("2005-02-03T00:00:00Z")
                    .unwrap()
                    .to_utc(),
                DateTime::parse_from_rfc3339("2005-02-03T04:05:00Z")
                    .unwrap()
                    .to_utc(),
                DateTime::parse_from_rfc3339("2005-02-03T04:05:06Z")
                    .unwrap()
                    .to_utc(),
                DateTime::<Utc>::MIN_UTC,
            ];

            array.store_array_subset(&array.subset_all(), &elements)?;
            if !matches!(unit, |NumpyTimeUnit::Week| NumpyTimeUnit::Day
                | NumpyTimeUnit::Hour
                | NumpyTimeUnit::Minute)
            {
                assert_eq!(
                    array.retrieve_array_subset::<Vec<DateTime<Utc>>>(&array.subset_all())?,
                    elements
                );
            }
        }

        #[cfg(feature = "jiff")]
        // The jiff rounding API does not support years/months/weeks/days
        // https://github.com/BurntSushi/jiff/issues/1
        if !matches!(
            unit,
            NumpyTimeUnit::Year | NumpyTimeUnit::Month | NumpyTimeUnit::Week | NumpyTimeUnit::Day
        ) {
            use jiff::{Timestamp, TimestampRound};
            let round = TimestampRound::new().smallest(try_numpy_to_jiff_unit(unit).unwrap());
            let elements = [
                "2005-02-25T00:00:00Z"
                    .parse::<Timestamp>()
                    .unwrap()
                    .round(round)
                    .unwrap(),
                Timestamp::UNIX_EPOCH,
                Timestamp::MIN,
                "2005-02-01T00:00:00Z"
                    .parse::<Timestamp>()
                    .unwrap()
                    .round(round)
                    .unwrap(),
                "2005-02-25T03:00:00Z"
                    .parse::<Timestamp>()
                    .unwrap()
                    .round(round)
                    .unwrap(),
                Timestamp::MIN,
            ];

            array.store_array_subset(&array.subset_all(), &elements)?;

            if !matches!(unit, NumpyTimeUnit::Hour | NumpyTimeUnit::Minute) {
                assert_eq!(
                    array.retrieve_array_subset::<Vec<jiff::Timestamp>>(&array.subset_all())?,
                    elements
                );
            }
        }
    }

    Ok(())
}

#[cfg(any(feature = "chrono", feature = "jiff"))]
#[test]
fn zarr_python_v3_numpy_timedelta_read() -> Result<(), Box<dyn Error>> {
    for (path, unit) in [
        (
            "tests/data/zarr_python_compat/timedelta64[ms].zarr",
            NumpyTimeUnit::Millisecond,
        ),
        (
            "tests/data/zarr_python_compat/timedelta64[us].zarr",
            NumpyTimeUnit::Microsecond,
        ),
        (
            "tests/data/zarr_python_compat/timedelta64[ns].zarr",
            NumpyTimeUnit::Nanosecond,
        ),
        (
            "tests/data/zarr_python_compat/timedelta64[10ms].zarr",
            NumpyTimeUnit::Millisecond,
        ),
        (
            "tests/data/zarr_python_compat/timedelta64[ps].zarr",
            NumpyTimeUnit::Picosecond,
        ),
        (
            "tests/data/zarr_python_compat/timedelta64[10us].zarr",
            NumpyTimeUnit::Microsecond,
        ),
    ] {
        let store = Arc::new(zarrs::filesystem::FilesystemStore::new(path)?);
        let array = zarrs::array::Array::open(store.clone(), "/")?;
        let subset_all = array.subset_all();

        #[cfg(feature = "chrono")]
        {
            use chrono::TimeDelta;
            let elements = array.retrieve_array_subset::<Vec<TimeDelta>>(&subset_all)?;

            let start_elem = if matches!(unit, NumpyTimeUnit::Picosecond) {
                // first element overflows in numpy
                1
            } else {
                0
            };

            println!("{path:?}");
            println!("{elements:?}");
            assert_eq!(
                &elements[start_elem..],
                &[
                    TimeDelta::hours(24 * 365),
                    TimeDelta::hours(24 * 7 * 2),
                    TimeDelta::hours(24 * 3),
                    TimeDelta::hours(4),
                    TimeDelta::minutes(5),
                    TimeDelta::seconds(6),
                    TimeDelta::milliseconds(7_000),
                    TimeDelta::microseconds(8_000_000),
                    TimeDelta::nanoseconds(9_000_000_000),
                    TimeDelta::default(),
                    TimeDelta::MIN,
                ][start_elem..]
            );
        }

        #[cfg(feature = "jiff")]
        if !matches!(unit, NumpyTimeUnit::Picosecond) {
            use jiff::{SignedDuration, Timestamp, TimestampRound, Unit};
            let elements = array.retrieve_array_subset::<Vec<SignedDuration>>(&subset_all)?;

            println!("{path:?}");
            println!("{elements:?}");
            let round = TimestampRound::new().smallest(try_numpy_to_jiff_unit(unit).unwrap());
            assert_eq!(
                elements,
                &[
                    SignedDuration::from_hours(24 * 365),
                    SignedDuration::from_hours(24 * 7 * 2),
                    SignedDuration::from_hours(24 * 3),
                    SignedDuration::from_hours(4),
                    SignedDuration::from_mins(5),
                    SignedDuration::from_secs(6),
                    SignedDuration::from_millis(7_000),
                    SignedDuration::from_micros(8_000_000),
                    SignedDuration::from_nanos(9_000_000_000),
                    SignedDuration::ZERO,
                    SignedDuration::MIN,
                ]
            );
        }
    }

    Ok(())
}

#[cfg(any(feature = "chrono", feature = "jiff"))]
#[test]
fn zarr_python_v3_numpy_timedelta_write() -> Result<(), Box<dyn Error>> {
    for scale_factor in [1, 2] {
        for unit in [
            NumpyTimeUnit::Second,
            NumpyTimeUnit::Millisecond,
            NumpyTimeUnit::Microsecond,
            NumpyTimeUnit::Nanosecond,
        ] {
            use jiff::{Timestamp, TimestampRound};
            use zarrs::array::ArrayBuilder;
            use zarrs::storage::store::MemoryStore;
            use zarrs_data_type::FillValue;

            let store = Arc::new(MemoryStore::new());
            let array = ArrayBuilder::new(
                vec![11],
                vec![5],
                data_type::numpy_timedelta64(unit, scale_factor.try_into().unwrap()),
                i64::MIN,
            )
            .build(store.clone(), "/")?;
            array.store_metadata()?;

            #[cfg(feature = "chrono")]
            {
                use chrono::TimeDelta;
                let elements = [
                    TimeDelta::hours(24 * 365),
                    TimeDelta::hours(24 * 7 * 2),
                    TimeDelta::hours(24 * 3),
                    TimeDelta::hours(4),
                    TimeDelta::minutes(5),
                    TimeDelta::seconds(6),
                    TimeDelta::milliseconds(14_000),
                    TimeDelta::microseconds(8_000_000),
                    TimeDelta::nanoseconds(18_000_000_000),
                    TimeDelta::default(),
                    TimeDelta::MIN,
                ];

                array.store_array_subset(&array.subset_all(), &elements)?;
                assert_eq!(
                    array.retrieve_array_subset::<Vec<TimeDelta>>(&array.subset_all())?,
                    elements
                );
            }

            #[cfg(feature = "jiff")]
            {
                use jiff::SignedDuration;
                let elements = [
                    SignedDuration::from_hours(24 * 365),
                    SignedDuration::from_hours(24 * 7 * 2),
                    SignedDuration::from_hours(24 * 3),
                    SignedDuration::from_hours(4),
                    SignedDuration::from_mins(50),
                    SignedDuration::from_secs(6),
                    SignedDuration::from_millis(14_000),
                    SignedDuration::from_micros(8_000_000),
                    SignedDuration::from_nanos(18_000_000_000),
                    SignedDuration::ZERO,
                    SignedDuration::MIN,
                ];

                array.store_array_subset(&array.subset_all(), &elements)?;
                assert_eq!(
                    array.retrieve_array_subset::<Vec<SignedDuration>>(&array.subset_all())?,
                    elements
                );
            }
        }
    }

    Ok(())
}
