use std::cmp::Ordering;

pub type D = f64;

#[derive(PartialEq,PartialOrd)]
pub struct NonNan(D);

impl NonNan {
    pub fn new(val: D) -> Option<NonNan> {
        Some(NonNan(val))
        // if val.is_nan() {
        //     None
        // } else {
        //     Some(NonNan(val))
        // }
    }
}

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
        NonNan::new(item).unwrap()
    }
}

impl Into<D> for NonNan {
    fn into(self) -> D {
        self.0
    }
}