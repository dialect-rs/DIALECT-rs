use crate::fmo::helpers::MolecularSlice;
use crate::fmo::{Monomer, ReducedBasisState, SuperSystem};
use crate::initialization::Atom;
use ndarray::prelude::*;

#[derive(Debug, Clone)]
pub struct OldSupersystem {
    /// Vector with the data and the positions of the individual
    /// atoms that are stored as [Atom](crate::initialization::Atom)
    pub atoms: Vec<Atom>,
    /// List of individuals fragments which are stored as a [Monomer](crate::fmo::Monomer)
    pub monomers: Vec<OldMonomer>,
    pub basis_states: Vec<ReducedBasisState>,
    pub last_scalar_coupling: Option<Array2<f64>>,
}

impl OldSupersystem {
    pub fn new(system: &SuperSystem) -> Self {
        let mut monomers: Vec<OldMonomer> = Vec::new();

        for monomer in system.monomers.iter() {
            monomers.push(OldMonomer::new(monomer));
        }

        let mut last_scalar_coupling: Option<Array2<f64>> = Option::None;
        if system.properties.last_scalar_coupling().is_some() {
            last_scalar_coupling =
                Some(system.properties.last_scalar_coupling().unwrap().to_owned());
        }

        OldSupersystem {
            atoms: system.atoms.clone(),
            monomers: monomers,
            basis_states: system.properties.basis_states().unwrap().to_vec(),
            last_scalar_coupling,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OldMonomer {
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of atomic orbitals
    pub n_orbs: usize,
    /// Index of the monomer in the [SuperSystem]
    pub index: usize,
    /// Different Slices that correspond to this monomer
    pub slice: MolecularSlice,
    pub orbs: Array2<f64>,
    pub occ_indices: Vec<usize>,
    pub virt_indices: Vec<usize>,
    pub tdm: Array2<f64>,
}

impl OldMonomer {
    pub fn new(monomer: &Monomer) -> Self {
        OldMonomer {
            n_atoms: monomer.n_atoms,
            n_orbs: monomer.n_orbs,
            index: monomer.index,
            slice: monomer.slice.clone(),
            orbs: monomer.properties.orbs().unwrap().to_owned(),
            occ_indices: monomer.properties.occ_indices().unwrap().to_vec(),
            virt_indices: monomer.properties.virt_indices().unwrap().to_vec(),
            tdm: monomer.properties.ci_coefficients().unwrap().to_owned(),
        }
    }
}
