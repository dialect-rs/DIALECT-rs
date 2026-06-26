//! Snapshot constructors for the FMO supersystem types. The plain data
//! structs live in dialect-state; building them requires the live
//! SuperSystem/Monomer/Pair types, so the constructors stay here.
#![allow(warnings)]

pub use dialect_state::old_supersystem::{OldEsdPair, OldMonomer, OldPair, OldSupersystem};

use crate::fmo::helpers::MolecularSlice;
use crate::fmo::{ESDPair, Monomer, Pair, ReducedBasisState, SuperSystem};
use crate::initialization::Atom;
use hashbrown::HashMap;
use ndarray::prelude::*;

pub fn new_old_supersystem(system: &SuperSystem) -> OldSupersystem {
        let mut monomers: Vec<OldMonomer> = Vec::new();
        let mut pairs: Vec<OldPair> = Vec::new();
        let mut esd_pairs: Vec<OldEsdPair> = Vec::new();

        for monomer in system.monomers.iter() {
            monomers.push(new_old_monomer(monomer));
        }

        for pair in system.pairs.iter() {
            pairs.push(new_old_pair(pair));
        }

        for esd_pair in system.esd_pairs.iter() {
            esd_pairs.push(new_old_esd_pair(esd_pair));
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

pub fn new_old_monomer(monomer: &Monomer) -> OldMonomer {
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

pub fn new_old_pair(pair: &Pair) -> OldPair {
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

pub fn new_old_esd_pair(pair: &ESDPair) -> OldEsdPair {
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
