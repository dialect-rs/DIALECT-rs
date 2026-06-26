//! Snapshots of the FMO supersystem (monomers, pairs, basis states) for
//! step-to-step state in dynamics and couplings (filled by the main crate).

use crate::basis_states::ReducedBasisState;
use crate::slices::MolecularSlice;
use dialect_dftb_core::atom::Atom;
use hashbrown::HashMap;
use ndarray::prelude::*;

#[derive(Debug, Clone)]
pub struct OldSupersystem {
    /// Vector with the data and the positions of the individual
    /// atoms that are stored as [Atom](crate::initialization::Atom)
    pub atoms: Vec<Atom>,
    /// List of individuals fragments which are stored as a [Monomer](crate::fmo::Monomer)
    pub monomers: Vec<OldMonomer>,
    pub pairs: Vec<OldPair>,
    pub esd_pairs: Vec<OldEsdPair>,
    pub basis_states: Vec<ReducedBasisState>,
    pub last_scalar_coupling: Option<Array2<f64>>,
    pub nacv_storage: HashMap<(usize, usize), Array1<f64>>,
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

#[derive(Debug, Clone)]
pub struct OldPair {
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of atomic orbitals
    pub n_orbs: usize,
    /// Index of the monomers in the [SuperSystem]
    pub index_1: usize,
    pub index_2: usize,
    pub s_i_ij: Array2<f64>,
    pub s_j_ij: Array2<f64>,
    pub nocc: usize,
    pub nvirt: usize,
    pub orbs: Array2<f64>,
}

#[derive(Debug, Clone)]
pub struct OldEsdPair {
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of atomic orbitals
    pub n_orbs: usize,
    /// Index of the monomers in the [SuperSystem]
    pub index_1: usize,
    pub index_2: usize,
    pub s_i_ij: Option<Array2<f64>>,
    pub s_j_ij: Option<Array2<f64>>,
    pub nocc: Option<usize>,
    pub nvirt: Option<usize>,
    pub orbs: Option<Array2<f64>>,
}
