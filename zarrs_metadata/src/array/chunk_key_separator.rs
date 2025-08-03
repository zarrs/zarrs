use derive_more::Display;

/// A chunk key separator. `/` or `.`.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Display)]
pub enum ChunkKeySeparator {
    /// The slash '/' character.
    #[display("/")]
    Slash,
    /// The dot '.' character.
    #[display(".")]
    Dot,
}

impl From<ChunkKeySeparator> for char {
    fn from(value: ChunkKeySeparator) -> Self {
        match value {
            ChunkKeySeparator::Slash => '/',
            ChunkKeySeparator::Dot => '.',
        }
    }
}

impl TryFrom<char> for ChunkKeySeparator {
    type Error = char;

    fn try_from(separator: char) -> Result<Self, Self::Error> {
        if separator == '/' {
            Ok(Self::Slash)
        } else if separator == '.' {
            Ok(Self::Dot)
        } else {
            Err(separator)
        }
    }
}

impl serde::Serialize for ChunkKeySeparator {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Slash => s.serialize_char('/'),
            Self::Dot => s.serialize_char('.'),
        }
    }
}

impl<'de> serde::Deserialize<'de> for ChunkKeySeparator {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(d)?;
        if let serde_json::Value::String(separator) = value {
            if separator == "/" {
                return Ok(Self::Slash);
            } else if separator == "." {
                return Ok(Self::Dot);
            }
        }
        Err(serde::de::Error::custom(
            "chunk key separator must be a `.` or `/`.",
        ))
    }
}
