//! FMO-DFTB SCC loop orchestration with HOP.
//!
//! Implements the self-consistent monomer loop, pair SCC, embedding energy,
//! and ESD energy — all using extended atom arrays that include ghost atoms.

use crate::scc::mixer::BuildMixer;
use super::hop_data::{get_repulsive_energy_scaled, HopData};
use super::monomer::{monomer_scc_step_hop, prepare_monomer_hop, MonomerHopScc};
use super::pair::{prepare_pair_hop, run_pair_scc_hop, PairHopScc};
use crate::fmo::scc::logging;
use crate::fmo::SuperSystem;
use ndarray::prelude::*;
use rayon::prelude::*;

impl SuperSystem<'_> {
    /// Run the self-consistent monomer SCC loop with HOP.
    ///
    /// For each SCC iteration:
    /// 1. Compute global ESP from gamma_ext . dq_ext
    /// 2. For each monomer, extract its ESP slice and run one SCC step
    /// 3. Collect updated dq (including ghosts) back into dq_ext
    ///
    /// Returns (monomer_energies, updated_hop_data_with_dq, monomer_scc_states)
    pub fn monomer_scc_hop(
        &self,
        max_iter: usize,
        hop_data: &mut HopData,
    ) -> (f64, Vec<MonomerHopScc>) {
        let scf_config = self.config.scf;

        // Prepare all monomer HOP SCC states [parallel]
        let mut mono_states: Vec<MonomerHopScc> = (0..self.n_mol)
            .into_par_iter()
            .map(|frag_idx| {
                let frag_info = &hop_data.frag_info[frag_idx];
                let frag_atom_range = self.monomers[frag_idx].slice.atom_as_range();
                prepare_monomer_hop(
                    frag_idx,
                    frag_info,
                    hop_data,
                    &self.gammafunction,
                    &self.gammafunction_lc,
                    self.monomers[frag_idx].slako,
                    &self.atoms,
                    &frag_atom_range,
                    &self.config.broyden,
                )
            })
            .collect();

        // Create Anderson accelerators for LC (one per monomer, persists across SCC iterations)
        let mut accels: Vec<Option<crate::scc::mixer::AndersonAccel>> = (0..self.n_mol)
            .map(|frag_idx| {
                if self.gammafunction_lc.is_some() {
                    let dim = mono_states[frag_idx].n_ext_orbs * mono_states[frag_idx].n_ext_orbs;
                    Some(self.config.mix_config.build_mixer(dim).unwrap())
                } else {
                    None
                }
            })
            .collect();

        let mut converged = vec![false; self.n_mol];

        // SCC loop
        for _iter in 0..max_iter {
            // Global ESP: gamma_ext . dq_ext (frozen for this iteration)
            let esp_ext: Array1<f64> = hop_data.gamma_ext.dot(&hop_data.dq_ext);

            // Pre-compute external ESP for each monomer from frozen global ESP
            let v_esp_exts: Vec<Array1<f64>> = (0..self.n_mol)
                .map(|frag_idx| {
                    let fi = &hop_data.frag_info[frag_idx];
                    let ext_range = &fi.ext_range;
                    let esp_full_slice = esp_ext.slice(s![ext_range.start..ext_range.end]);
                    let gamma_self = hop_data.gamma_ext.slice(s![
                        ext_range.start..ext_range.end,
                        ext_range.start..ext_range.end
                    ]);
                    let dq_self = hop_data.dq_ext.slice(s![ext_range.start..ext_range.end]);
                    &esp_full_slice - &gamma_self.dot(&dq_self)
                })
                .collect();

            // Run all monomer SCC steps in parallel
            let results: Vec<bool> = mono_states
                .par_iter_mut()
                .zip(accels.par_iter_mut())
                .zip(v_esp_exts.par_iter())
                .map(|((mono, accel), v_esp)| {
                    monomer_scc_step_hop(mono, v_esp.view(), scf_config, accel)
                })
                .collect();

            // Update dq_ext with all monomers' new charges
            for frag_idx in 0..self.n_mol {
                converged[frag_idx] = results[frag_idx];
                let fi = &hop_data.frag_info[frag_idx];
                let ext_range = &fi.ext_range;
                hop_data
                    .dq_ext
                    .slice_mut(s![ext_range.start..ext_range.end])
                    .assign(&mono_states[frag_idx].dq);
            }

            let n_converged = converged.iter().filter(|&&c| c).count();
            logging::fmo_monomer_iteration(_iter, n_converged, self.n_mol);
            if n_converged == self.n_mol {
                break;
            }
        }

        if converged.contains(&false) {
            panic!("HOP Monomer SCC routine did NOT converge!");
        }

        // Compute monomer energies (SCF + repulsive for all extended atoms including ghosts)
        // Repulsive energy with ZREF/QREF scaling (DFTB_EREP convention).
        let e_reps: Vec<f64> = (0..self.n_mol)
            .into_par_iter()
            .map(|frag_idx| {
                let mono = &mono_states[frag_idx];
                let fi = &hop_data.frag_info[frag_idx];
                let zref_slice = hop_data.zref.slice(s![fi.ext_range.start..fi.ext_range.end]);
                let qref_slice = hop_data.qref.slice(s![fi.ext_range.start..fi.ext_range.end]);
                get_repulsive_energy_scaled(
                    &mono.ext_atoms, mono.n_ext_atoms, self.monomers[frag_idx].vrep,
                    zref_slice, qref_slice,
                )
            })
            .collect();

        let mut monomer_energies = 0.0;
        for frag_idx in 0..self.n_mol {
            let total = mono_states[frag_idx].last_energy + e_reps[frag_idx];
            mono_states[frag_idx].last_energy = total;
            monomer_energies += total;
        }

        (monomer_energies, mono_states)
    }

    /// Compute ESP_Q for each monomer in extended coordinates.
    ///
    /// esp_q_I = gamma_ext[ext_I, :] . dq_ext - gamma_ext[ext_I, ext_I] . dq_I
    ///
    /// This is the ESP from all OTHER fragments' extended charges acting on I.
    fn compute_esp_q_hop(&self, hop_data: &HopData) -> Vec<Array1<f64>> {
        (0..self.n_mol)
            .into_par_iter()
            .map(|frag_idx| {
                let fi = &hop_data.frag_info[frag_idx];
                let ext_range = &fi.ext_range;

                let esp_full: Array1<f64> = hop_data
                    .gamma_ext
                    .slice(s![ext_range.start..ext_range.end, ..])
                    .dot(&hop_data.dq_ext);

                let esp_self: Array1<f64> = hop_data
                    .gamma_ext
                    .slice(s![
                        ext_range.start..ext_range.end,
                        ext_range.start..ext_range.end
                    ])
                    .dot(&hop_data.dq_ext.slice(s![ext_range.start..ext_range.end]));

                &esp_full - &esp_self
            })
            .collect()
    }

    /// Run pair SCC calculations with HOP.
    ///
    /// For each close pair, prepare and converge the pair SCC.
    /// Returns the total pair delta energy: sum(E_pair - E_mono_I - E_mono_J).
    pub fn pair_scc_hop(
        &self,
        hop_data: &HopData,
        mono_states: &[MonomerHopScc],
    ) -> (f64, Vec<PairHopScc>) {
        let scf_config = self.config.scf;
        let esp_q_list = self.compute_esp_q_hop(hop_data);

        // Run all pair SCCs in parallel
        let pair_results: Vec<(PairHopScc, f64)> = self
            .pairs
            .par_iter()
            .map(|pair| {
                let fi_i = &hop_data.frag_info[pair.i];
                let fi_j = &hop_data.frag_info[pair.j];

                let frag_range_i = self.monomers[pair.i].slice.atom_as_range();
                let frag_range_j = self.monomers[pair.j].slice.atom_as_range();

                let mut pair_scc = prepare_pair_hop(
                    pair.i,
                    pair.j,
                    hop_data,
                    mono_states[pair.i].dq.view(),
                    mono_states[pair.j].dq.view(),
                    esp_q_list[pair.i].view(),
                    esp_q_list[pair.j].view(),
                    fi_i,
                    fi_j,
                    &self.gammafunction,
                    &self.gammafunction_lc,
                    self.monomers[pair.i].slako,
                    &self.atoms,
                    &frag_range_i,
                    &frag_range_j,
                    &self.config.broyden,
                );

                let pair_energy = run_pair_scc_hop(&mut pair_scc, scf_config, self.monomers[pair.i].vrep, &self.config.mix_config);

                let delta_e = pair_energy
                    - mono_states[pair.i].last_energy
                    - mono_states[pair.j].last_energy;

                (pair_scc, delta_e)
            })
            .collect();

        let mut pair_states: Vec<PairHopScc> = Vec::with_capacity(pair_results.len());
        let mut total_pair_delta = 0.0;
        for (pair_scc, delta_e) in pair_results.into_iter() {
            pair_states.push(pair_scc);
            total_pair_delta += delta_e;
        }

        (total_pair_delta, pair_states)
    }

    /// Compute the embedding energy with HOP (dDIJ*VIJ).
    ///
    /// Uses POPMAT differences (not dq differences) to correctly handle
    /// ZREF changes at healed BDA atoms, and includes ghost atom CTIJ
    /// contributions.
    ///
    /// CTIJ_A = POPMAT_pair(A) - POPMAT_mono(A)
    ///        = (dq_pair + ZREF_pair) - (dq_mono + ZREF_mono)
    ///
    /// For healed BDAs: ZREF_pair = ZREF_mono + 1, so CTIJ includes a +1 correction.
    /// For healed ghosts: ghost doesn't exist in pair, so CTIJ = -POPMAT_mono_ghost.
    /// For partial ghosts: CTIJ = POPMAT_pair_ghost - POPMAT_mono_ghost.
    pub fn embedding_energy_hop(
        &self,
        hop_data: &HopData,
        mono_states: &[MonomerHopScc],
        pair_states: &[PairHopScc],
    ) -> f64 {
        let esp_q_list = self.compute_esp_q_hop(hop_data);
        let gamma_ext = &hop_data.gamma_ext;
        let mut embedding = 0.0;

        for (pair_idx, pair) in self.pairs.iter().enumerate() {
            let fi_i = &hop_data.frag_info[pair.i];
            let fi_j = &hop_data.frag_info[pair.j];
            let ext_i = &fi_i.ext_range;
            let ext_j = &fi_j.ext_range;
            let n_real_i = fi_i.n_real_atoms;
            let n_real_j = fi_j.n_real_atoms;

            let dq_ext_j = hop_data.dq_ext.slice(s![ext_j.start..ext_j.end]);
            let dq_ext_i = hop_data.dq_ext.slice(s![ext_i.start..ext_i.end]);

            // ESP on I from all fragments except I and J (covers all ext_I atoms incl. ghosts)
            let gamma_ij = gamma_ext.slice(s![ext_i.start..ext_i.end, ext_j.start..ext_j.end]);
            let esp_i: Array1<f64> = &esp_q_list[pair.i] - &gamma_ij.dot(&dq_ext_j);

            // ESP on J from all fragments except I and J (covers all ext_J atoms incl. ghosts)
            let gamma_ji = gamma_ext.slice(s![ext_j.start..ext_j.end, ext_i.start..ext_i.end]);
            let esp_j: Array1<f64> = &esp_q_list[pair.j] - &gamma_ji.dot(&dq_ext_i);

            let ps = &pair_states[pair_idx];
            let mut emb_pair = 0.0;

            // Fragment I real atoms: CTIJ = POPMAT_pair - POPMAT_mono
            for a in 0..n_real_i {
                let popmat_pair = ps.dq[a] + ps.zref[a];
                let mono_zref = hop_data.zref[ext_i.start + a];
                let popmat_mono = mono_states[pair.i].dq[a] + mono_zref;
                let ctij = popmat_pair - popmat_mono;
                emb_pair += ctij * esp_i[a];
            }

            // Fragment J real atoms: CTIJ = POPMAT_pair - POPMAT_mono
            for a in 0..n_real_j {
                let pair_a = n_real_i + a;
                let popmat_pair = ps.dq[pair_a] + ps.zref[pair_a];
                let mono_zref = hop_data.zref[ext_j.start + a];
                let popmat_mono = mono_states[pair.j].dq[a] + mono_zref;
                let ctij = popmat_pair - popmat_mono;
                emb_pair += ctij * esp_j[a];
            }

            // === Ghost atoms of both I and J ===
            // Iterate detached_bonds in same order as prepare_pair_hop to match pair ghost indices.
            // For each bond where BAA is in I or J, the bond creates a ghost in that monomer.
            // If the bond is healed (BDA also in pair): ghost doesn't exist in pair → CTIJ = -POPMAT_mono
            // If partial (BDA outside pair): ghost exists in pair → CTIJ = POPMAT_pair - POPMAT_mono
            let mut mono_ghost_i_idx = 0usize;
            let mut mono_ghost_j_idx = 0usize;
            let mut pair_ghost_idx = ps.n_real_atoms; // pair ghosts start after real atoms

            for bond in &hop_data.detached_bonds {
                let bda_in_pair = bond.bda_fragment == pair.i || bond.bda_fragment == pair.j;

                if bond.baa_fragment == pair.i {
                    // Ghost in monomer I
                    let mono_ghost_local = n_real_i + mono_ghost_i_idx;
                    let mono_zref_g = hop_data.zref[ext_i.start + mono_ghost_local];
                    let popmat_mono_g = mono_states[pair.i].dq[mono_ghost_local] + mono_zref_g;

                    let ctij = if bda_in_pair {
                        // Healed: ghost doesn't exist in pair
                        -popmat_mono_g
                    } else {
                        // Partial: ghost exists in pair
                        let popmat_pair_g = ps.dq[pair_ghost_idx] + ps.zref[pair_ghost_idx];
                        pair_ghost_idx += 1;
                        popmat_pair_g - popmat_mono_g
                    };

                    emb_pair += ctij * esp_i[mono_ghost_local];
                    mono_ghost_i_idx += 1;
                } else if bond.baa_fragment == pair.j {
                    // Ghost in monomer J
                    let mono_ghost_local = n_real_j + mono_ghost_j_idx;
                    let mono_zref_g = hop_data.zref[ext_j.start + mono_ghost_local];
                    let popmat_mono_g = mono_states[pair.j].dq[mono_ghost_local] + mono_zref_g;

                    let ctij = if bda_in_pair {
                        // Healed: ghost doesn't exist in pair
                        -popmat_mono_g
                    } else {
                        // Partial: ghost exists in pair
                        let popmat_pair_g = ps.dq[pair_ghost_idx] + ps.zref[pair_ghost_idx];
                        pair_ghost_idx += 1;
                        popmat_pair_g - popmat_mono_g
                    };

                    emb_pair += ctij * esp_j[mono_ghost_local];
                    mono_ghost_j_idx += 1;
                }
            }

            embedding += emb_pair;
        }

        embedding
    }

    /// Compute ESD energy with HOP.
    ///
    /// Uses extended dq arrays (including ghost charges):
    ///   esd = sum over ESD pairs: dq_ext_I . gamma_ext[ext_I, ext_J] . dq_ext_J
    pub fn esd_energy_hop(&self, hop_data: &HopData) -> f64 {
        let gamma_ext = &hop_data.gamma_ext;
        let mut esd_energy = 0.0;

        for esd_pair in &self.esd_pairs {
            let ext_i = &hop_data.frag_info[esd_pair.i].ext_range;
            let ext_j = &hop_data.frag_info[esd_pair.j].ext_range;

            let dq_i = hop_data.dq_ext.slice(s![ext_i.start..ext_i.end]);
            let dq_j = hop_data.dq_ext.slice(s![ext_j.start..ext_j.end]);
            let gamma_ij =
                gamma_ext.slice(s![ext_i.start..ext_i.end, ext_j.start..ext_j.end]);

            let e = dq_i.dot(&gamma_ij).dot(&dq_j);
            esd_energy += e;
        }

        esd_energy
    }
}
