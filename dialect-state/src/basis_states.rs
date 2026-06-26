//! Reduced descriptions of locally-excited and charge-transfer basis
//! states used by the FMO-LCMO machinery and stored in `Properties`.

use nalgebra::Vector3;
use ndarray::prelude::*;
use ndarray::Slice;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug)]
pub struct ChargeTransferPair {
    pub m_h: usize,
    pub m_l: usize,
    pub state_index: usize,
    pub state_energy: f64,
    pub eigenvectors: Array2<f64>,
    pub q_tr: Array1<f64>,
    pub tr_dipole: Vector3<f64>,
    /// [Slice](ndarray::prelude::Slice) for occupied orbitals corresponding to this molecular unit
    pub occ_orb: Slice,
    /// [Slice](ndarray::prelude::Slice) for virtual orbitals corresponding to this molecular unit
    pub virt_orb: Slice,
    pub occ_indices: Vec<usize>,
    pub virt_indices: Vec<usize>,
}

impl PartialEq for ChargeTransferPair {
    fn eq(&self, other: &Self) -> bool {
        self.m_h == other.m_h && self.m_l == other.m_l && self.state_index == other.state_index
    }
}

impl Display for ChargeTransferPair {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CT {}: {} -> {}",
            self.state_index,
            self.m_h + 1,
            self.m_l + 1
        )
    }
}

#[derive(Clone, Debug)]
pub enum ReducedBasisState {
    LE(ReducedLE),
    CT(ChargeTransferPair),
}

#[derive(Clone, Debug)]
pub struct ReducedLE {
    pub energy: f64,
    pub monomer_index: usize,
    pub state_index: usize,
    pub state_coefficient: f64,
    pub homo: usize,
}

#[derive(Clone, Debug)]
pub struct ReducedCT {
    pub energy: f64,
    pub monomer_index_h: usize,
    pub monomer_index_e: usize,
    pub state_index: usize,
    pub state_coefficient: f64,
}

impl Display for ReducedBasisState {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            ReducedBasisState::LE(state) => write!(f, "{}", state),
            ReducedBasisState::CT(state) => write!(f, "{}", state),
        }
    }
}

impl Display for ReducedLE {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "LE(S{}) on Frag. {:>4}",
            self.state_index + 1,
            self.monomer_index + 1
        )
    }
}

impl Display for ReducedCT {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "CT(Nr.{}) between Frag.: {} -> {}",
            self.state_index, self.monomer_index_h, self.monomer_index_e
        )
    }
}
