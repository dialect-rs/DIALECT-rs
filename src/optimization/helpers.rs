//! System-specific optimization glue. The generic optimizers (GDIIS,
//! L-BFGS, model Hessians, line-search macros, XYZOutput) live in
//! dialect_utilities::optimization::helpers and are re-exported here.

pub use dialect_utilities::optimization::helpers::*;

use crate::fmo::SuperSystem;
use crate::initialization::System;
use dialect_utilities::{impl_line_search, impl_wolfe_line_search};
use ndarray::prelude::*;

impl System {
    impl_line_search!();
    impl_wolfe_line_search!();
}

impl SuperSystem<'_> {
    impl_line_search!();
    impl_wolfe_line_search!();
}


