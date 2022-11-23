use crate::fmo::helpers::MolecularSlice;
use crate::initialization::parameters::{RepulsivePotential, SlaterKoster};
use crate::initialization::{Atom, Geometry};
use crate::io::{frame_to_coordinates, Configuration};
use crate::properties::Properties;
use crate::scc::gamma_approximation::GammaFunction;
use chemfiles::Frame;
use hashbrown::HashMap;
use ndarray::prelude::*;
use ndarray::{Slice, SliceInfo};
use ndarray_rand::rand_distr::Gamma;
use std::ops::Range;

/// Type that holds a molecular system that contains all data for the quantum chemical routines.
///
/// This type is only used for FMO calculations. This type is a similar to the [System] type that
/// is used in non-FMO calculations
#[derive(Debug, Clone)]
pub struct Monomer<'a> {
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of atomic orbitals
    pub n_orbs: usize,
    /// Index of the monomer in the [SuperSystem]
    pub index: usize,
    /// Different Slices that correspond to this monomer
    pub slice: MolecularSlice,
    /// Type that holds the calculated properties e.g. gamma matrix, overlap matrix and so on.
    pub properties: Properties,
    /// Repulsive potential type. Type that contains the repulsion energy and its derivative
    /// w.r.t. d/dR for each unique pair of atoms as a spline.
    pub vrep: &'a RepulsivePotential,
    /// Slater-Koster parameters for the H0 and overlap matrix elements. For each unique atom pair
    /// the matrix elements and their derivatives can be splined.
    pub slako: &'a SlaterKoster,
    /// Type of Gamma function. This can be either `Gaussian` or `Slater` type.
    pub gammafunction: GammaFunction,
    /// Gamma function for the long-range correction. Only used if long-range correction is requested
    pub gammafunction_lc: Option<GammaFunction>,
}

impl<'a> Monomer<'a> {
    pub fn new(
        n_atoms: usize,
        n_orbs: usize,
        index: usize,
        slice: MolecularSlice,
        properties: Properties,
        vrep: &'a RepulsivePotential,
        slako: &'a SlaterKoster,
        gammafunc: GammaFunction,
        gamma_lc: Option<GammaFunction>,
    ) -> Self {
        Self {
            n_atoms,
            n_orbs,
            slice,
            index,
            properties,
            vrep,
            slako,
            gammafunction: gammafunc,
            gammafunction_lc: gamma_lc,
        }
    }

    pub fn set_mo_indices(&mut self, n_elec: usize) {
        // get the indices of the occupied and virtual orbitals
        let mut occ_indices: Vec<usize> = Vec::new();
        let mut virt_indices: Vec<usize> = Vec::new();
        (0..self.n_orbs).for_each(|index| {
            if index < (n_elec / 2) {
                occ_indices.push(index)
            } else {
                virt_indices.push(index)
            }
        });
        self.properties.set_occ_indices(occ_indices);
        self.properties.set_virt_indices(virt_indices);
    }
}

impl PartialEq for Monomer<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for Monomer<'_> {}
