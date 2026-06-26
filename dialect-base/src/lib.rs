//! Foundational, dependency-free building blocks shared by all method
//! families in dialect: physical constants, default settings, the chemical
//! `Element` type, small array utilities and parameterization configuration.

pub mod array_helper;
pub mod config;
pub mod constants;
pub mod defaults;
pub mod elements;

use std::env;
use std::fmt;
use std::time::Instant;

/// Returns the dialect source directory from the environment variable
/// configured in [`defaults::SOURCE_DIR_VARIABLE`].
pub fn get_path_prefix() -> String {
    let key: &str = defaults::SOURCE_DIR_VARIABLE;
    match env::var(key) {
        Ok(val) => val,
        Err(_e) => panic!("The environment variable {} was not set", key),
    }
}

/// A simple timer based on std::time::Instant, to implement the std::fmt::Display trait on
pub struct Timer {
    pub time: Instant,
}

impl Timer {
    pub fn start() -> Self {
        Timer {
            time: Instant::now(),
        }
    }
}

// Implement `Display` for Instant.
impl fmt::Display for Timer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Use `self.number` to refer to each positional data point.
        write!(
            f,
            "{:>68} {:>8.2} s",
            "elapsed time:",
            self.time.elapsed().as_secs_f32()
        )
    }
}
