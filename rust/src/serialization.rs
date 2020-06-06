use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fs::File;
use std::fmt;
use pyo3::prelude::{PyResult, PyErr};
use pyo3::exceptions;

#[derive(Debug, Clone)]
pub struct UnknownFormatError {
    pub format: String,
}

impl UnknownFormatError {
    fn new(format: &str) -> Self {
        UnknownFormatError {
            format: format.into()
        }
    }
}

pub const SERIALIZATION_FORMATS: &[&str; 2] = &["json", "bincode"];

impl fmt::Display for UnknownFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Incorrect format: `{}`. Please, use one of: {:?}",
               self.format,
               SERIALIZATION_FORMATS)
    }
}

impl Into<PyErr> for UnknownFormatError {
    fn into(self) -> PyErr {
        PyErr::new::<exceptions::ValueError, _>(self.to_string())
    }
}

pub fn save<T: Serialize>(what: &T, filename: &str, format: Option<&str>) -> PyResult<()> {
    let file = File::create(filename)?;
    let f = format.unwrap_or("json");
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

pub fn load<T: DeserializeOwned>(what: &mut T, filename: &str, format: Option<&str>) -> PyResult<()> {
    let file = File::open(filename)?;
    let f = format.unwrap_or("json");
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
