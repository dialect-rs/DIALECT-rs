//! Snapshot of a DFTB `System`: geometry, MO coefficients, CI vectors and
//! the previous step's couplings (filled by the main crate).

use dialect_dftb_core::atom::Atom;
use ndarray::prelude::*;

#[derive(Debug, Clone)]
pub struct OldSystem {
    pub atoms: Vec<Atom>,
    pub orbs: Array2<f64>,
    pub ci_coefficients: Array2<f64>,
    pub old_scalar_couplings: Option<Array2<f64>>,
    pub old_nacv: Option<Vec<Array1<f64>>>,
}
