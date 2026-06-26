//! FMO-DFTB SCC with Hybrid Orbital Projection (HOP).
//!
//! This module implements FMO-DFTB with covalent fragmentation using the HOP method.
//! Ghost hydrogen atoms are placed at BAA positions, and the supersystem gamma matrix
//! includes these ghost atoms — matching the `dftbfo.src` implementation.
//!
//! # Architecture
//!
//! - `hop_data`: Ghost atom creation, extended arrays, ZREF, gamma_ext
//! - `monomer`: Per-monomer SCC with HOP projector and extended atoms
//! - `pair`: Per-pair SCC with partial/healed bond classification
//! - `supersystem`: SCC loop orchestration and energy assembly

pub mod hop_data;
pub mod monomer;
pub mod pair;
pub mod supersystem;

use crate::fmo::scc::helpers::get_dispersion_energy;
use crate::fmo::scc::logging;
use crate::fmo::SuperSystem;
use crate::gradients::dispersion::gradient_disp;
use crate::scc::scc_routine::SCCError;
use crate::utils::Timer;
use hop_data::build_hop_data;
use log::info;
use ndarray::prelude::*;

impl SuperSystem<'_> {
    /// Run the complete FMO-DFTB SCC calculation with HOP.
    ///
    /// This is the top-level entry point, analogous to `run_scc()` for non-HOP FMO.
    ///
    /// Energy components:
    /// - Monomer energies (SCF + repulsive, real atoms only for repulsive)
    /// - Pair delta energies (E_pair - E_mono_I - E_mono_J)
    /// - Embedding energy (interfragment ESP * delta_dq, real atoms only)
    /// - ESD energy (extended dq including ghosts)
    /// - Dispersion energy (D3 correction, all real atoms)
    pub fn run_scc_hop(&mut self) -> Result<f64, SCCError> {
        let timer = Timer::start();
        let max_iter = self.config.scf.scf_max_cycles;

        // 1. Build HOP data: ghost atoms, extended gamma, ZREF
        let mut hop_data = build_hop_data(self);

        info!(
            "{: <45} {} bond(s), {} extended atoms",
            "HOP boundary treatment:",
            hop_data.detached_bonds.len(),
            hop_data.n_ext_atoms
        );
        logging::fmo_scc_init(max_iter);

        // 2. Prepare monomers and run self-consistent monomer SCC
        // Also prepares the standard (non-HOP) monomers for pair type detection
        self.prepare_scc_hop_monomers();

        let (monomer_energies, mono_states) = self.monomer_scc_hop(max_iter, &mut hop_data);

        // 3. Run pair SCC calculations
        let (pair_delta_energies, pair_states) = self.pair_scc_hop(&hop_data, &mono_states);

        // 4. Compute embedding energy
        let embedding = self.embedding_energy_hop(&hop_data, &mono_states, &pair_states);

        // 5. Compute ESD energy
        let esd_energy = self.esd_energy_hop(&hop_data);

        // 6. Dispersion energy
        let mut e_disp = 0.0;
        if self.config.dispersion.use_dispersion {
            e_disp = get_dispersion_energy(&self.atoms, &self.config);
        }

        // 7. Total energy
        let total_energy =
            monomer_energies + pair_delta_energies + embedding + esd_energy + e_disp;

        logging::fmo_scc_end(
            timer,
            monomer_energies,
            pair_delta_energies,
            embedding,
            esd_energy,
            e_disp,
        );

        self.properties.set_last_energy(total_energy);

        Ok(total_energy)
    }

    /// Run FMO-DFTB SCC with HOP and compute the analytical gradient.
    ///
    /// Performs the full SCC calculation, then computes the gradient using the
    /// converged SCC states (density matrices, charges, orbitals).
    pub fn run_gradient_hop(&mut self) -> Result<Array1<f64>, SCCError> {
        let timer = Timer::start();
        let max_iter = self.config.scf.scf_max_cycles;

        // 1. Build HOP data
        let mut hop_data = build_hop_data(self);

        info!(
            "{: <45} {} bond(s), {} extended atoms",
            "HOP boundary treatment:",
            hop_data.detached_bonds.len(),
            hop_data.n_ext_atoms
        );
        logging::fmo_scc_init(max_iter);

        // 2. Prepare monomers and run SCC
        self.prepare_scc_hop_monomers();
        let (monomer_energies, mono_states) = self.monomer_scc_hop(max_iter, &mut hop_data);

        // 3. Run pair SCC
        let (pair_delta_energies, pair_states) = self.pair_scc_hop(&hop_data, &mono_states);

        // 4. Energy components
        let embedding = self.embedding_energy_hop(&hop_data, &mono_states, &pair_states);
        let esd_energy = self.esd_energy_hop(&hop_data);
        let mut e_disp = 0.0;
        if self.config.dispersion.use_dispersion {
            e_disp = get_dispersion_energy(&self.atoms, &self.config);
        }

        let total_energy =
            monomer_energies + pair_delta_energies + embedding + esd_energy + e_disp;

        logging::fmo_scc_end(
            timer,
            monomer_energies,
            pair_delta_energies,
            embedding,
            esd_energy,
            e_disp,
        );
        self.properties.set_last_energy(total_energy);

        // 5. Compute gradient
        let grad_timer = Timer::start();
        let mut grad = self.ground_state_gradient_fmo_hop(&hop_data, &mono_states, &pair_states);

        // 6. Add dispersion gradient
        if self.config.dispersion.use_dispersion {
            let disp_grad = gradient_disp(&self.atoms, &self.config);
            grad += &disp_grad;
        }

        info!("{: <45} {}", "FMO-DFTB HOP gradient:", grad_timer);

        Ok(grad)
    }

    /// Prepare the standard (non-HOP) monomer data needed for pair type detection.
    ///
    /// This sets up the basic monomer gamma matrices and overlap matrices
    /// that are used by `get_pair_type()` and pair SCC setup.
    fn prepare_scc_hop_monomers(&mut self) {
        use crate::scc::gamma_approximation::gamma_atomwise;
        use crate::scc::h0_and_s::h0_and_s;

        let atoms = &self.atoms;
        for mol in self.monomers.iter_mut() {
            let mol_atoms = &atoms[mol.slice.atom_as_range()];

            // Only compute if not already set
            if mol.properties.gamma().is_none() {
                let gamma = gamma_atomwise(&mol.gammafunction, mol_atoms, mol.n_atoms);
                mol.properties.set_gamma(gamma);
            }
            if mol.properties.s().is_none() || mol.properties.h0().is_none() {
                let (s, h0) = h0_and_s(mol.n_orbs, mol_atoms, mol.slako);
                mol.properties.set_s(s);
                mol.properties.set_h0(h0);
            }
        }
    }
}
