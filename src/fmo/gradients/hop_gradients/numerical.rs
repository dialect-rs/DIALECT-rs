#![allow(dead_code)]
#![allow(warnings)]
//! Numerical gradient testing for FMO-DFTB with HOP.
//!
//! Provides energy and gradient wrappers compatible with `assert_deriv_7point`,
//! plus a `test_gs_gradient_hop` entry point for numerical vs analytical comparison.

use crate::fmo::SuperSystem;
use crate::gradients::numerical::assert_deriv_7point;
use ndarray::prelude::*;

impl SuperSystem<'_> {
    /// Energy wrapper for HOP numerical gradient testing.
    ///
    /// Updates atom positions, then runs the full FMO-DFTB HOP SCC
    /// (monomer SCC, pair SCC, embedding, ESD, dispersion) and returns the total energy.
    pub fn gs_energy_wrapper_hop(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for pair in self.esd_pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.run_scc_hop().unwrap()
    }

    /// Analytical gradient wrapper for HOP numerical gradient testing.
    ///
    /// Runs the full FMO-DFTB HOP SCC and computes the analytical gradient.
    /// Returns the gradient as a flat array [3 * n_atoms].
    pub fn gs_gradient_wrapper_hop(&mut self) -> Array1<f64> {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for pair in self.esd_pairs.iter_mut() {
            pair.properties.reset();
        }
        self.run_gradient_hop().unwrap()
    }

    /// Test the FMO-DFTB HOP gradient against the 7-point numerical stencil.
    ///
    /// Uses O(h^6) central differences for maximum accuracy. Recommended stepsize: 0.01.
    pub fn test_gs_gradient_hop(&mut self) {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for pair in self.esd_pairs.iter_mut() {
            pair.properties.reset();
        }

        assert_deriv_7point(
            self,
            SuperSystem::gs_energy_wrapper_hop,
            SuperSystem::gs_gradient_wrapper_hop,
            self.get_xyz(),
            0.01,
            1e-6,
        );
    }
}
