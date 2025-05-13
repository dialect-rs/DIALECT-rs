#![allow(warnings)]

use crate::fmo::helpers::MolecularSlice;
use crate::fmo::{ESDPair, Monomer, Pair, ReducedBasisState, SuperSystem};
use crate::initialization::Atom;
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

impl OldSupersystem {
    pub fn new(system: &SuperSystem) -> Self {
        let mut monomers: Vec<OldMonomer> = Vec::new();
        let mut pairs: Vec<OldPair> = Vec::new();
        let mut esd_pairs: Vec<OldEsdPair> = Vec::new();

        for monomer in system.monomers.iter() {
            monomers.push(OldMonomer::new(monomer));
        }

        for pair in system.pairs.iter() {
            pairs.push(OldPair::new(pair));
        }

        for esd_pair in system.esd_pairs.iter() {
            esd_pairs.push(OldEsdPair::new(esd_pair));
        }

        let mut last_scalar_coupling: Option<Array2<f64>> = Option::None;
        if system.properties.last_scalar_coupling().is_some() {
            last_scalar_coupling =
                Some(system.properties.last_scalar_coupling().unwrap().to_owned());
        }

        OldSupersystem {
            atoms: system.atoms.clone(),
            monomers: monomers,
            pairs: pairs,
            esd_pairs: esd_pairs,
            basis_states: system.properties.basis_states().unwrap().to_vec(),
            last_scalar_coupling,
            nacv_storage: HashMap::new(),
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

impl OldPair {
    pub fn new(pair: &Pair) -> Self {
        OldPair {
            n_atoms: pair.n_atoms,
            n_orbs: pair.n_orbs,
            index_1: pair.i,
            index_2: pair.j,
            s_i_ij: pair.properties.s_i_ij().unwrap().to_owned(),
            s_j_ij: pair.properties.s_j_ij().unwrap().to_owned(),
            nocc: pair.properties.occ_indices().unwrap().len(),
            nvirt: pair.properties.virt_indices().unwrap().len(),
            orbs: pair.properties.orbs().unwrap().to_owned(),
        }
    }
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

impl OldEsdPair {
    pub fn new(pair: &ESDPair) -> Self {
        let s_i_ij: Option<Array2<f64>> = if pair.properties.s_i_ij().is_some() {
            Some(pair.properties.s_i_ij().unwrap().to_owned())
        } else {
            None
        };

        let s_j_ij: Option<Array2<f64>> = if pair.properties.s_j_ij().is_some() {
            Some(pair.properties.s_j_ij().unwrap().to_owned())
        } else {
            None
        };

        let nocc: Option<usize> = if pair.properties.occ_indices().is_some() {
            Some(pair.properties.occ_indices().unwrap().len())
        } else {
            None
        };

        let nvirt: Option<usize> = if pair.properties.virt_indices().is_some() {
            Some(pair.properties.virt_indices().unwrap().len())
        } else {
            None
        };

        let orbs: Option<Array2<f64>> = if pair.properties.orbs().is_some() {
            Some(pair.properties.orbs().unwrap().to_owned())
        } else {
            None
        };

        OldEsdPair {
            n_atoms: pair.n_atoms,
            n_orbs: pair.n_orbs,
            index_1: pair.i,
            index_2: pair.j,
            s_i_ij,
            s_j_ij,
            nocc,
            nvirt,
            orbs,
        }
    }
}
