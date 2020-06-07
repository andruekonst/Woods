//! Numeric utils: default floating point type (`D`) and `NonNan` wrapper.
//! 

use std::cmp::Ordering;

/// Default floating point type.
pub type D = f64;

/// Numeric data type wrapper with ordering for non-`NaN` numbers.
/// 
/// **NaN-safeness should be checked manually before wrapping into `NonNan`**
/// 
/// IEEE-754 floating point numbers like `f64` don't implement [`Ord`] trait.
/// This wrapper implements [`Ord`] trait for all non-`NaN` numbers.
/// `NaN` number will result it `panic`.
/// 
#[derive(PartialEq,PartialOrd)]
pub struct NonNan(D);

impl Eq for NonNan {}

impl Clone for NonNan {
    fn clone(&self) -> Self {
        NonNan::from(self.0)
    }
}

impl Ord for NonNan {
    fn cmp(&self, other: &NonNan) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl From<D> for NonNan {
    fn from(item: D) -> Self {
        NonNan(item)
    }
}

impl From<&D> for NonNan {
    fn from(item: &D) -> Self {
        NonNan(*item)
    }
}

impl Into<D> for NonNan {
    fn into(self) -> D {
        self.0
    }
}