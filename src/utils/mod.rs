pub(crate) mod array_helper;
mod tests;
mod zbrent;

pub use array_helper::ToOwnedF;
use std::time::Instant;
use std::{env, fmt};
pub use zbrent::zbrent;

use crate::defaults;
pub use tests::*;

pub enum Calculation {
    Converged,
    NotConverged,
}
/// A simple timer based on std::time::Instant, to implement the std::fmt::Display trait on
pub struct Timer {
    time: Instant,
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

/// Helper function that prepends the path of the `tincr` source directory to be able to
/// use a relative path to the parameter files. The environment variable `TINCR_SRC_DIR` should be
/// set, so that the parameter files can be found.
pub fn get_path_prefix() -> String {
    let key: &str = defaults::SOURCE_DIR_VARIABLE;
    match env::var(key) {
        Ok(val) => val,
        Err(_e) => panic!("The environment variable {} was not set", key),
    }
}
