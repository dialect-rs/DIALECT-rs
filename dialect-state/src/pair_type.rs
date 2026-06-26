//! Classification of FMO fragment pairs.

use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PairType {
    Pair,
    ESD,
    None,
}

impl fmt::Display for PairType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            PairType::Pair => write!(f, "Pair"),
            PairType::ESD => write!(f, "ESD"),
            PairType::None => write!(f, "None"),
        }
    }
}
