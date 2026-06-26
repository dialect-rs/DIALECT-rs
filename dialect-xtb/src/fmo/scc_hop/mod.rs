//! FMO-xTB SCC with Hybrid Orbital Projection (HOP).
//!
//! This module implements FMO-xTB with covalent fragmentation using the HOP method.
//! Ghost atoms are placed at BDA positions (same element as BDA), and the extended
//! gamma_shell matrix includes these ghost atoms.
//!
//! # Architecture
//!
//! - `hop_data`: Ghost atom creation, extended arrays, ZREF, gamma_shell_ext
//! - `monomer`: Per-monomer SCC with HOP projector and extended atoms
//! - `pair`: Per-pair SCC with partial/healed bond classification
//! - `supersystem`: SCC loop orchestration and energy assembly

pub mod hop_data;
pub mod monomer;
pub mod pair;
pub mod supersystem;
pub mod trimer;

use dialect_utilities::scc_interface::SCCError;
use dialect_base::Timer;
use crate::fmo::supersystem::XtbSuperSystem;
use crate::scc::halogen_correction::get_halogen_correction;
use crate::scc::scc_helpers::get_dispersion_energy_xtb;
use dialect_utilities::fmo_logging as logging;
use hop_data::{build_xtb_hop_data, XtbHopData};
use log::info;
use monomer::XtbMonomerHopScc;
use ndarray::prelude::*;
use pair::XtbPairHopScc;
use trimer::XtbTrimerHopScc;

impl XtbSuperSystem<'_> {
    /// Run the complete FMO-xTB SCC calculation with HOP.
    ///
    /// Energy components:
    /// - Monomer energies (SCF + repulsive with ZREF/QREF scaling)
    /// - Pair delta energies (E_pair - E_mono_I - E_mono_J)
    /// - Embedding energy (interfragment ESP * CTIJ, POPMAT-based)
    /// - ESD energy (extended dq_shell including ghosts)
    /// - Dispersion energy (D3 correction, all real atoms)
    /// - Halogen correction
    pub fn run_scc_hop(&mut self) -> Result<f64, SCCError> {
        let timer = Timer::start();
        let max_iter = self.config.scf.scf_max_cycles;

        // 1. Build HOP data: ghost atoms, extended gamma_shell, ZREF
        let mut hop_data = build_xtb_hop_data(self);

        info!(
            "{: <45} {} bond(s), {} extended atoms",
            "HOP boundary treatment:",
            hop_data.detached_bonds.len(),
            hop_data.n_ext_atoms
        );
        logging::fmo_scc_init(max_iter);

        // 2. Run self-consistent monomer SCC
        let (monomer_energies, mono_states) = self.monomer_scc_hop(max_iter, &mut hop_data);

        // 3. Run pair SCC calculations
        let (pair_delta_energies, pair_states) = self.pair_scc_hop(&hop_data, &mono_states);

        // 4. Compute embedding energy (FMO2-level, unscaled)
        let embedding = self.embedding_energy_hop(&hop_data, &mono_states, &pair_states);

        // 5. Compute ESD energy
        let esd_energy = self.esd_energy_hop(&hop_data);

        // 6. Trimer (FMO3) contributions
        let (trimer_delta_energies, trimer_embedding) = if self.config.fmo.use_three_body {
            let (tri_delta, tri_states) =
                self.trimer_scc_hop(&hop_data, &mono_states, &pair_states);
            let tri_emb =
                self.trimer_embedding_energy_hop(&hop_data, &mono_states, &pair_states, &tri_states);
            (tri_delta, tri_emb)
        } else {
            (0.0, 0.0)
        };

        // 7. Dispersion + halogen correction
        let e_disp = get_dispersion_energy_xtb(&self.atoms, &self.config);
        let e_halogen = get_halogen_correction(&self.atoms);

        // 8. Total energy
        let total_energy = monomer_energies
            + pair_delta_energies
            + embedding
            + esd_energy
            + trimer_delta_energies
            + trimer_embedding
            + e_disp
            + e_halogen;

        // Print the SCC summary (monomer / pair / trimer energies), matching
        // the non-HOP FMO-xTB layout.
        if self.config.fmo.use_three_body {
            logging::fmo_scc_end_trimer(
                timer,
                monomer_energies,
                pair_delta_energies,
                embedding,
                esd_energy,
                e_disp,
                trimer_delta_energies,
                trimer_embedding,
            );
        } else {
            logging::fmo_scc_end(
                timer,
                monomer_energies,
                pair_delta_energies,
                embedding,
                esd_energy,
                e_disp,
            );
        }

        self.properties.set_last_energy(total_energy);

        Ok(total_energy)
    }

    /// Run FMO-xTB HOP SCC and return all states needed for the gradient.
    ///
    /// Returns (energy, hop_data, mono_states, pair_states, trimer_states).
    pub fn run_scc_hop_for_gradient(
        &mut self,
    ) -> Result<
        (
            f64,
            XtbHopData,
            Vec<XtbMonomerHopScc>,
            Vec<XtbPairHopScc>,
            Vec<XtbTrimerHopScc>,
        ),
        SCCError,
    > {
        let timer = Timer::start();
        let max_iter = self.config.scf.scf_max_cycles;

        // 1. Build HOP data
        let mut hop_data = build_xtb_hop_data(self);

        info!(
            "{: <45} {} bond(s), {} extended atoms",
            "HOP boundary treatment:",
            hop_data.detached_bonds.len(),
            hop_data.n_ext_atoms
        );
        logging::fmo_scc_init(max_iter);

        // 2. Monomer SCC
        let (monomer_energies, mono_states) = self.monomer_scc_hop(max_iter, &mut hop_data);

        // 3. Pair SCC
        let (pair_delta_energies, pair_states) = self.pair_scc_hop(&hop_data, &mono_states);

        // 4. Embedding energy
        let embedding = self.embedding_energy_hop(&hop_data, &mono_states, &pair_states);

        // 5. ESD energy
        let esd_energy = self.esd_energy_hop(&hop_data);

        // 6. Trimer contributions
        let (trimer_delta_energies, trimer_embedding, trimer_states) =
            if self.config.fmo.use_three_body {
                let (tri_delta, tri_states) =
                    self.trimer_scc_hop(&hop_data, &mono_states, &pair_states);
                let tri_emb = self.trimer_embedding_energy_hop(
                    &hop_data, &mono_states, &pair_states, &tri_states,
                );
                (tri_delta, tri_emb, tri_states)
            } else {
                (0.0, 0.0, Vec::new())
            };

        // 7. Dispersion + halogen
        let e_disp = get_dispersion_energy_xtb(&self.atoms, &self.config);
        let e_halogen = get_halogen_correction(&self.atoms);

        // 8. Total energy
        let total_energy = monomer_energies
            + pair_delta_energies
            + embedding
            + esd_energy
            + trimer_delta_energies
            + trimer_embedding
            + e_disp
            + e_halogen;

        if self.config.fmo.use_three_body {
            logging::fmo_scc_end_trimer(
                timer,
                monomer_energies,
                pair_delta_energies,
                embedding,
                esd_energy,
                e_disp,
                trimer_delta_energies,
                trimer_embedding,
            );
        } else {
            logging::fmo_scc_end(
                timer,
                monomer_energies,
                pair_delta_energies,
                embedding,
                esd_energy,
                e_disp,
            );
        }

        self.properties.set_last_energy(total_energy);

        Ok((
            total_energy,
            hop_data,
            mono_states,
            pair_states,
            trimer_states,
        ))
    }
}
