//! Cartesian optimization impls for the DFTB system types. The driver
//! macros live in dialect-utilities (shared with the NDDO/OMx crates).

use crate::fmo::SuperSystem;
use crate::initialization::System;
use crate::io::Configuration;
use crate::optimization::helpers::*;
use crate::scc::scc_routine::RestrictedSCC;
use dialect_base::constants;
use dialect_utilities::{impl_cartesian_loop, impl_cartesian_loop_v2, impl_cartesian_loop_v2_fmo, impl_cartesian_loop_v3, impl_optimize, impl_optimize_cartesian, impl_optimize_cartesian_v2, impl_optimize_cartesian_v3};
use log::{log_enabled, warn, Level};
use ndarray::prelude::*;

impl System {
    impl_optimize!();
    impl_cartesian_loop!();
    impl_optimize_cartesian!();
    impl_cartesian_loop_v2!();
    impl_optimize_cartesian_v2!();
    impl_cartesian_loop_v3!();
    impl_optimize_cartesian_v3!();

    /// Standard model Hessian diagonal for non-FMO systems.
    /// Uses element-based force constants.
    pub fn model_hessian_diag(&self) -> Array1<f64> {
        let n = self.atoms.len() * 3;
        let mut diag = Array1::zeros(n);
        for (i, atom) in self.atoms.iter().enumerate() {
            let k = match atom.number {
                1 => 0.5,          // H
                6 | 7 | 8 => 0.35, // C, N, O
                _ => 0.25,         // Heavier elements
            };
            diag[3 * i] = k;
            diag[3 * i + 1] = k;
            diag[3 * i + 2] = k;
        }
        diag
    }

    /// Full model Hessian for non-FMO systems.
    pub fn model_hessian_full(&self, coords: &Array1<f64>) -> Array2<f64> {
        let atomic_numbers: Vec<u8> = self.atoms.iter().map(|a| a.number).collect();
        build_model_hessian_full(coords, &atomic_numbers)
    }

    pub fn opt_energy_and_gradient(&mut self, state: usize) -> (f64, Array1<f64>) {
        let (energy, gradient): (f64, Array1<f64>) = if state == 0 {
            // ground state energy and gradient
            self.prepare_scc();
            let tmp_energy = self.run_scc().unwrap();
            let tmp_gradient = self.ground_state_gradient(false);

            (tmp_energy, tmp_gradient)
        } else {
            // excited state calculation
            let excited_state: usize = state - 1;
            self.prepare_scc();
            let mut tmp_energy = self.run_scc().unwrap();

            // calculate excited states
            self.calculate_excited_states(false);
            tmp_energy += self.properties.ci_eigenvalue(excited_state).unwrap();

            let mut tmp_gradient = self.ground_state_gradient(true);
            tmp_gradient = tmp_gradient + self.calculate_excited_state_gradient(excited_state);

            (tmp_energy, tmp_gradient)
        };
        self.properties.reset_reduced();

        (energy, gradient)
    }

    pub fn calculate_energy_line_search(&mut self, state: usize) -> f64 {
        let energy: f64 = if state == 0 {
            // ground state energy and gradient
            self.prepare_scc();
            self.run_scc().unwrap()
        } else {
            // excited state calculation
            let excited_state: usize = state - 1;
            self.prepare_scc();
            let mut tmp_energy: f64 = self.run_scc().unwrap();

            // calculate excited states
            self.calculate_excited_states(false);
            tmp_energy += self.properties.ci_eigenvalue(excited_state).unwrap();
            tmp_energy
        };
        self.properties.reset_reduced();

        energy
    }
}


impl SuperSystem<'_> {
    impl_optimize!();
    impl_cartesian_loop!();
    impl_optimize_cartesian!();
    impl_cartesian_loop_v2_fmo!();
    impl_optimize_cartesian_v2!();
    impl_cartesian_loop_v3!();
    impl_optimize_cartesian_v3!();

    /// FMO-aware model Hessian diagonal.
    pub fn model_hessian_diag(&self) -> Array1<f64> {
        let n_atoms = self.atoms.len();
        let n = n_atoms * 3;
        let mut diag = Array1::zeros(n);

        let avg_monomer_size = if self.monomers.len() > 0 {
            n_atoms as f64 / self.monomers.len() as f64
        } else {
            n_atoms as f64
        };
        let alpha = avg_monomer_size / (avg_monomer_size + 3.0);
        let k_inter = 0.05;

        for (i, atom) in self.atoms.iter().enumerate() {
            let k_intra = match atom.number {
                1 => 0.5,
                6 | 7 | 8 => 0.35,
                _ => 0.25,
            };
            let k = alpha * k_intra + (1.0 - alpha) * k_inter;
            diag[3 * i] = k;
            diag[3 * i + 1] = k;
            diag[3 * i + 2] = k;
        }

        diag
    }

    /// FMO-aware full model Hessian.
    /// Uses monomer atom ranges to assign different force constants for
    /// intramolecular (same monomer) vs intermolecular (different monomer) pairs.
    /// Intramolecular: strong directional bond force constants.
    /// Intermolecular: weaker hydrogen bond / vdW force constants.
    pub fn model_hessian_full(&self, coords: &Array1<f64>) -> Array2<f64> {
        let n_atoms = self.atoms.len();
        let atomic_numbers: Vec<u8> = self.atoms.iter().map(|a| a.number).collect();

        // Build monomer membership array: monomer_of[i] = which monomer atom i belongs to
        let mut monomer_of = vec![usize::MAX; n_atoms];
        for (mon_idx, monomer) in self.monomers.iter().enumerate() {
            for atom_idx in monomer.slice.atom_as_range() {
                monomer_of[atom_idx] = mon_idx;
            }
        }

        build_model_hessian_full_fmo(coords, &atomic_numbers, &monomer_of)
    }

    pub fn opt_energy_and_gradient(&mut self, state: usize) -> (f64, Array1<f64>) {
        let (energy, gradient): (f64, Array1<f64>) = if state == 0 {
            // the geometry changed since the last evaluation: rebuild the
            // pair / ESD-pair classification before the SCC (same pattern
            // as the dynamics interface)
            self.redefine_pairs();
            if self.config.fmo.covalent_fragmentation {
                // ghost-aware HOP SCC + analytic gradient; the HOP data
                // (ghost atoms, extended gamma, ZREF) is rebuilt from the
                // current geometry inside the call
                let tmp_gradient = self.run_gradient_hop().unwrap();
                let tmp_energy = self.properties.last_energy().unwrap();
                (tmp_energy, tmp_gradient)
            } else {
                // ground state energy and gradient
                self.prepare_scc();
                let tmp_energy = self.run_scc().unwrap();
                let tmp_gradient = self.ground_state_gradient();

                (tmp_energy, tmp_gradient)
            }
        } else {
            panic!(
                "The optimization procedure for the fmo systems is restricted to the ground
            state"
            );
        };
        for monomer in self.monomers.iter_mut() {
            monomer.properties.reset_reduced();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset_reduced();
        }
        self.properties.reset_reduced();

        (energy, gradient)
    }

    pub fn calculate_energy_line_search(&mut self, state: usize) -> f64 {
        let energy: f64 = if state == 0 {
            // rebuild the pair classification for the trial geometry
            self.redefine_pairs();
            if self.config.fmo.covalent_fragmentation {
                // ghost-aware HOP SCC (same dispatch as the sp jobtype)
                self.run_scc_hop().unwrap()
            } else {
                // ground state energy and gradient
                self.prepare_scc();
                self.run_scc().unwrap()
            }
        } else {
            panic!(
                "The optimization procedure for the fmo systems is restricted to the ground
            state"
            );
        };
        for monomer in self.monomers.iter_mut() {
            monomer.properties.reset_reduced();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset_reduced();
        }
        self.properties.reset_reduced();

        energy
    }
}

