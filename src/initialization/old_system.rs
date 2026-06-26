//! Snapshot of a DFTB System for step-to-step state in dynamics and
//! couplings. The plain data struct lives in dialect-state.

pub use dialect_state::old_system::OldSystem;

use crate::initialization::System;
use ndarray::prelude::*;

pub fn new_old_system(

        system: &System,
        old_scalar_couplings: Option<Array2<f64>>,
        old_nacv: Option<Vec<Array1<f64>>>,
    ) -> OldSystem {
        OldSystem {
            atoms: system.atoms.clone(),
            orbs: system.properties.orbs().unwrap().to_owned(),
            ci_coefficients: system.properties.ci_coefficients().unwrap().to_owned(),
            old_scalar_couplings,
            old_nacv,
        }
    }
