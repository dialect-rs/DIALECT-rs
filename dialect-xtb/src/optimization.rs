//! Cartesian geometry optimization for the xTB system types, generated
//! by the shared driver macros from dialect-utilities.

use crate::fmo::supersystem::XtbSuperSystem;
use crate::initialization::system::XtbSystem;
use dialect_base::constants;
use dialect_config::Configuration;
use dialect_utilities::optimization::helpers::*;
use dialect_utilities::scc_interface::RestrictedSCC;
use dialect_utilities::{impl_cartesian_loop, impl_cartesian_loop_v2, impl_cartesian_loop_v2_fmo, impl_cartesian_loop_v3, impl_line_search, impl_optimize, impl_optimize_cartesian, impl_optimize_cartesian_v2, impl_optimize_cartesian_v3, impl_wolfe_line_search};
use log::{debug, info, log_enabled, warn, Level};
use ndarray::prelude::*;

impl XtbSystem {
    impl_line_search!();
    impl_wolfe_line_search!();
}

impl XtbSuperSystem<'_> {
    impl_line_search!();
    impl_wolfe_line_search!();
}

impl XtbSystem {
    impl_optimize!();
    impl_cartesian_loop!();
    impl_optimize_cartesian!();
    impl_cartesian_loop_v2!();
    impl_optimize_cartesian_v2!();
    impl_cartesian_loop_v3!();
    impl_optimize_cartesian_v3!();

    /// Standard model Hessian diagonal for non-FMO systems.
    pub fn model_hessian_diag(&self) -> Array1<f64> {
        let n = self.atoms.len() * 3;
        let mut diag = Array1::zeros(n);
        for (i, atom) in self.atoms.iter().enumerate() {
            let k = match atom.number {
                1 => 0.5,
                6 | 7 | 8 => 0.35,
                _ => 0.25,
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

    pub fn opt_energy_and_gradient(&mut self, _state: usize) -> (f64, Array1<f64>) {
        // ground state energy and gradient
        self.prepare_scc();
        let energy = self.run_scc().unwrap();
        let gradient = self.ground_state_gradient();
        self.properties.reset_reduced();

        (energy, gradient)
    }

    pub fn calculate_energy_line_search(&mut self, _state: usize) -> f64 {
        // ground state energy
        self.prepare_scc();
        let energy = self.run_scc().unwrap();
        self.properties.reset_reduced();

        energy
    }
}

impl XtbSuperSystem<'_> {
    impl_optimize!();
    impl_cartesian_loop!();
    impl_optimize_cartesian!();
    impl_cartesian_loop_v2_fmo!();
    impl_optimize_cartesian_v2!();
    impl_cartesian_loop_v3!();
    impl_optimize_cartesian_v3!();

    /// FMO-aware model Hessian diagonal (same logic as SuperSystem).
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

    /// FMO-aware full model Hessian (same logic as SuperSystem).
    pub fn model_hessian_full(&self, coords: &Array1<f64>) -> Array2<f64> {
        let n_atoms = self.atoms.len();
        let atomic_numbers: Vec<u8> = self.atoms.iter().map(|a| a.number).collect();

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
            // pair / ESD-pair / trimer classification (and HOP ghost
            // positions) before the SCC
            self.update_fragmentation();
            // ground state energy and gradient; HOP systems need the
            // ghost-aware SCC driver (same dispatch as the sp jobtype)
            let tmp_energy = if self.config.fmo.covalent_fragmentation {
                self.run_scc_hop().unwrap()
            } else {
                self.prepare_scc();
                self.run_scc().unwrap()
            };
            let tmp_gradient = self.ground_state_gradient();

            (tmp_energy, tmp_gradient)
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
            // rebuild the fragment classification for the trial geometry
            self.update_fragmentation();
            // ground state energy; HOP systems need the ghost-aware SCC
            if self.config.fmo.covalent_fragmentation {
                self.run_scc_hop().unwrap()
            } else {
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
