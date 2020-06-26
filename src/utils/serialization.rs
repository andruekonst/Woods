//! Save and load serializable objects.
//! 
//! # File formats
//! All available formats are in [`SERIALIZATION_FORMATS`].
//! 
//! List of currently implemented formats:
//! 1. `json`
//! 2. `bincode`
//! 
//! # Example
//! ```
//! let params = TreeParameters::new(None, None);
//! save(&params, "params.json", Some("json"));
//! ```

use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fs::File;
use std::fmt;
use pyo3::prelude::{PyResult, PyErr};
use pyo3::exceptions;

/// Unknown serialization file format error.
#[derive(Debug, Clone)]
pub struct UnknownFormatError {
    /// File format, *not a fmt*.
    pub format: String,
}

impl UnknownFormatError {
    /// Make file format error by format name.
    fn new(format: &str) -> Self {
        UnknownFormatError {
            format: format.into()
        }
    }
}

/// Available serialization formats.
pub const SERIALIZATION_FORMATS: &[&str; 2] = &["json", "bincode"];
/// Default serialization format (`json`).
pub const DEFAULT_SERIALIZATION_FORMAT: &str = "json";

impl fmt::Display for UnknownFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Incorrect format: `{}`. Please, use one of: {:?}",
               self.format,
               SERIALIZATION_FORMATS)
    }
}

impl Into<PyErr> for UnknownFormatError {
    /// Cast unknown format error to Python's `ValueError`.
    fn into(self) -> PyErr {
        PyErr::new::<exceptions::ValueError, _>(self.to_string())
    }
}

/// Save serializable object to file (`filename`), using
/// `format` file format.
pub fn save<T: Serialize>(what: &T, filename: &str, format: Option<&str>) -> PyResult<()> {
    let file = File::create(filename)?;
    let f = format.unwrap_or(DEFAULT_SERIALIZATION_FORMAT);
    match f {
        "json" => serde_json::to_writer(file, what).unwrap(),
        "bincode" => bincode::serialize_into(file, what).unwrap(),
        _ => {
            if SERIALIZATION_FORMATS.contains(&f) {
                unimplemented!("Format `{}` serialization", f);
            }
            return Err(UnknownFormatError::new(f).into());
        }
    }
    Ok(())
}

/// Load serializable object from file (`filename`) with `format` file format.
pub fn load<T: DeserializeOwned>(what: &mut T, filename: &str, format: Option<&str>) -> PyResult<()> {
    let file = File::open(filename)?;
    let f = format.unwrap_or(DEFAULT_SERIALIZATION_FORMAT);
     match f {
        "json" => {
            *what = serde_json::from_reader::<_, T>(file).unwrap();
        },
        "bincode" => {
            *what = bincode::deserialize_from::<_, T>(file).unwrap();
        },
        _ => {
            if SERIALIZATION_FORMATS.contains(&f) {
                unimplemented!("Format `{}` deserialization", f);
            }
            return Err(UnknownFormatError::new(f).into());
        }
    };
    Ok(())
}
