use crate::initialization::{Atom, System};
use ndarray::prelude::*;

#[derive(Debug, Clone)]
pub struct OldSystem {
    pub atoms: Vec<Atom>,
    pub orbs: Array2<f64>,
    pub ci_coefficients: Array2<f64>,
    pub old_scalar_couplings: Option<Array2<f64>>,
    pub old_nacv: Option<Vec<Array1<f64>>>,
}

impl OldSystem {
    pub fn new(
        system: &System,
        old_scalar_couplings: Option<Array2<f64>>,
        old_nacv: Option<Vec<Array1<f64>>>,
    ) -> Self {
        OldSystem {
            atoms: system.atoms.clone(),
            orbs: system.properties.orbs().unwrap().to_owned(),
            ci_coefficients: system.properties.ci_coefficients().unwrap().to_owned(),
            old_scalar_couplings,
            old_nacv,
        }
    }
}
