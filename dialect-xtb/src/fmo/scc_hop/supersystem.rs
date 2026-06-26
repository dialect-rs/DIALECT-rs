//! FMO-xTB SCC loop orchestration with HOP.
//!
//! Implements the self-consistent monomer loop, pair SCC, embedding energy,
//! and ESD energy — all using extended atom arrays that include ghost atoms.

use super::hop_data::{
    calculate_repulsive_energy_xtb_scaled, get_frag_shell_range, XtbHopData,
};
use super::monomer::{monomer_scc_step_hop_xtb, prepare_monomer_hop_xtb, XtbMonomerHopScc};
use super::pair::{prepare_pair_hop_xtb, run_pair_scc_hop_xtb, XtbPairHopScc};
use super::trimer::{prepare_trimer_hop_xtb, run_trimer_scc_hop_xtb, XtbTrimerHopScc};
use dialect_state::PairType;
use dialect_utilities::fmo_logging::fmo_monomer_iteration;
use dialect_utilities::scc_helpers::aovec_to_aomat;
use dialect_utilities::mulliken::shell_to_ao_values;
use crate::fmo::supersystem::XtbSuperSystem;
use crate::scc::gamma_matrix::gamma_shell_dsymv;
use crate::initialization::basis::Basis;
use ndarray::prelude::*;
use rayon::prelude::*;

/// Count shells per atom for atoms in the given range.
fn count_shells_per_atom(ext_basis: &Basis, ext_range: &std::ops::Range<usize>) -> Vec<usize> {
    let n_atoms = ext_range.end - ext_range.start;
    let mut counts = vec![0usize; n_atoms];
    for shell in &ext_basis.shells {
        if shell.atom_index >= ext_range.start && shell.atom_index < ext_range.end {
            counts[shell.atom_index - ext_range.start] += 1;
        }
    }
    counts
}

impl XtbSuperSystem<'_> {
    /// Run the self-consistent monomer SCC loop with HOP.
    pub fn monomer_scc_hop(
        &self,
        max_iter: usize,
        hop_data: &mut XtbHopData,
    ) -> (f64, Vec<XtbMonomerHopScc>) {
        let scf_charge_conv = self.config.scf.scf_charge_conv;
        let scf_energy_conv = self.config.scf.scf_energy_conv;
        let temperature = self.config.scf.electronic_temperature;

        // Prepare all monomer HOP SCC states [parallel]
        let mut mono_states: Vec<XtbMonomerHopScc> = (0..self.n_mol)
            .into_par_iter()
            .map(|frag_idx| {
                let frag_info = &hop_data.frag_info[frag_idx];
                let frag_atom_range = self.monomers[frag_idx].slice.atom_as_range();
                prepare_monomer_hop_xtb(
                    frag_idx,
                    frag_info,
                    hop_data,
                    &self.monomers[frag_idx].gammafunction,
                    &self.atoms,
                    &frag_atom_range,
                    &self.config.broyden,
                )
            })
            .collect();

        let mut converged = vec![false; self.n_mol];

        // SCC loop
        for _iter in 0..max_iter {
            // Compute global shell-level ESP from gamma_shell_ext . dq_shell_ext
            let v_shell_global: Array1<f64> = gamma_shell_dsymv(
                &hop_data.gamma_shell_ext.view(),
                &hop_data.dq_shell_ext.view(),
            );

            // Pre-compute ESP for each monomer (includes self-interaction, like non-HOP)
            let v_esp_list: Vec<Array2<f64>> = (0..self.n_mol)
                .map(|frag_idx| {
                    let fi = &hop_data.frag_info[frag_idx];
                    let shell_range = get_frag_shell_range(&hop_data.ext_basis, &fi.ext_range);
                    let n_ext_orbs = fi.n_ext_orbs;

                    // Full ESP slice for this fragment's shells (self + external)
                    let esp_full = v_shell_global.slice(s![shell_range.start..shell_range.end]);

                    // Convert shell→AO using monomer's extended basis
                    let mono = &mono_states[frag_idx];
                    let v_ao = shell_to_ao_values(&mono.basis, n_ext_orbs, esp_full);
                    aovec_to_aomat(v_ao.view(), n_ext_orbs)
                })
                .collect();

            // Run all monomer SCC steps in parallel
            let results: Vec<bool> = mono_states
                .par_iter_mut()
                .zip(v_esp_list.par_iter())
                .map(|(mono, v_esp)| {
                    monomer_scc_step_hop_xtb(
                        mono,
                        v_esp.clone(),
                        temperature,
                        scf_charge_conv,
                        scf_energy_conv,
                    )
                })
                .collect();

            // Update dq_shell_ext and dq_ext with all monomers' new charges
            for frag_idx in 0..self.n_mol {
                converged[frag_idx] = results[frag_idx];
                let fi = &hop_data.frag_info[frag_idx];
                let shell_range = get_frag_shell_range(&hop_data.ext_basis, &fi.ext_range);

                // Update shell-level charges
                hop_data
                    .dq_shell_ext
                    .slice_mut(s![shell_range.start..shell_range.end])
                    .assign(&mono_states[frag_idx].dq_shell);

                // Update atom-level charges
                hop_data
                    .dq_ext
                    .slice_mut(s![fi.ext_range.start..fi.ext_range.end])
                    .assign(&mono_states[frag_idx].dq);
            }

            let n_converged = converged.iter().filter(|&&c| c).count();
            fmo_monomer_iteration(_iter, n_converged, self.n_mol);
            if n_converged == self.n_mol {
                break;
            }
        }

        if converged.contains(&false) {
            panic!("HOP Monomer SCC routine did NOT converge!");
        }

        // Compute monomer energies: SCF + repulsive with ZREF/QREF scaling
        let e_reps: Vec<f64> = (0..self.n_mol)
            .into_par_iter()
            .map(|frag_idx| {
                let mono = &mono_states[frag_idx];
                let fi = &hop_data.frag_info[frag_idx];
                let zref_slice =
                    hop_data.zref.slice(s![fi.ext_range.start..fi.ext_range.end]);
                let qref_slice =
                    hop_data.qref.slice(s![fi.ext_range.start..fi.ext_range.end]);
                calculate_repulsive_energy_xtb_scaled(&mono.ext_atoms, zref_slice, qref_slice)
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

    /// Compute ESP_Q at shell level for each monomer in extended coordinates.
    /// esp_q_shell_I = gamma_shell_ext[shell_I, :] . dq_shell_ext - gamma_self . dq_I
    fn compute_esp_q_shell_hop(
        &self,
        hop_data: &XtbHopData,
    ) -> Vec<Array1<f64>> {
        (0..self.n_mol)
            .into_par_iter()
            .map(|frag_idx| {
                let fi = &hop_data.frag_info[frag_idx];
                let shell_range = get_frag_shell_range(&hop_data.ext_basis, &fi.ext_range);

                let esp_full: Array1<f64> = hop_data
                    .gamma_shell_ext
                    .slice(s![shell_range.start..shell_range.end, ..])
                    .dot(&hop_data.dq_shell_ext);

                let esp_self: Array1<f64> = hop_data
                    .gamma_shell_ext
                    .slice(s![
                        shell_range.start..shell_range.end,
                        shell_range.start..shell_range.end
                    ])
                    .dot(
                        &hop_data
                            .dq_shell_ext
                            .slice(s![shell_range.start..shell_range.end]),
                    );

                &esp_full - &esp_self
            })
            .collect()
    }

    /// Run pair SCC calculations with HOP.
    pub fn pair_scc_hop(
        &self,
        hop_data: &XtbHopData,
        mono_states: &[XtbMonomerHopScc],
    ) -> (f64, Vec<XtbPairHopScc>) {
        let max_iter = self.config.scf.scf_max_cycles;
        let temperature = self.config.scf.electronic_temperature;
        let scf_charge_conv = self.config.scf.scf_charge_conv;
        let scf_energy_conv = self.config.scf.scf_energy_conv;
        let esp_q_shell_list = self.compute_esp_q_shell_hop(hop_data);

        let pair_results: Vec<(XtbPairHopScc, f64)> = self
            .pairs
            .par_iter()
            .map(|pair| {
                let fi_i = &hop_data.frag_info[pair.i];
                let fi_j = &hop_data.frag_info[pair.j];
                let frag_range_i = self.monomers[pair.i].slice.atom_as_range();
                let frag_range_j = self.monomers[pair.j].slice.atom_as_range();

                let mut pair_scc = prepare_pair_hop_xtb(
                    pair.i,
                    pair.j,
                    hop_data,
                    mono_states[pair.i].dq.view(),
                    mono_states[pair.j].dq.view(),
                    mono_states[pair.i].dq_shell.view(),
                    mono_states[pair.j].dq_shell.view(),
                    esp_q_shell_list[pair.i].view(),
                    esp_q_shell_list[pair.j].view(),
                    fi_i,
                    fi_j,
                    &self.monomers[pair.i].gammafunction,
                    &self.atoms,
                    &frag_range_i,
                    &frag_range_j,
                    hop_data.cn_ext.view(),
                );

                let pair_energy = run_pair_scc_hop_xtb(
                    &mut pair_scc,
                    max_iter,
                    temperature,
                    scf_charge_conv,
                    scf_energy_conv,
                    &self.config.broyden,
                );

                let delta_e = pair_energy
                    - mono_states[pair.i].last_energy
                    - mono_states[pair.j].last_energy;

                (pair_scc, delta_e)
            })
            .collect();

        let mut pair_states: Vec<XtbPairHopScc> = Vec::with_capacity(pair_results.len());
        let mut total_pair_delta = 0.0;
        for (pair_scc, delta_e) in pair_results.into_iter() {
            pair_states.push(pair_scc);
            total_pair_delta += delta_e;
        }

        (total_pair_delta, pair_states)
    }

    /// Compute embedding energy with HOP (shell-level).
    ///
    /// Uses shell-level CTIJ = delta_dq_shell (pair - mono) dotted with
    /// shell-level ESP. In xTB, ESP is shell-dependent due to different
    /// Hubbard parameters per shell, so atom-level computation is incorrect.
    ///
    /// For FMO3: pass `pair_scal` with SCAL = 1 - n_trimers_containing_pair.
    /// Each pair's embedding contribution is multiplied by its SCAL factor.
    /// This replaces the separate trimer_embedding calculation, avoiding
    /// ghost-structure mismatch between trimer and monomer SCC contexts.
    pub fn embedding_energy_hop(
        &self,
        hop_data: &XtbHopData,
        mono_states: &[XtbMonomerHopScc],
        pair_states: &[XtbPairHopScc],
    ) -> f64 {
        self.embedding_energy_hop_scaled(hop_data, mono_states, pair_states, None)
    }

    /// Compute embedding energy with HOP and optional SCAL factors.
    pub fn embedding_energy_hop_scaled(
        &self,
        hop_data: &XtbHopData,
        mono_states: &[XtbMonomerHopScc],
        pair_states: &[XtbPairHopScc],
        pair_scal: Option<&[f64]>,
    ) -> f64 {
        let esp_q_shell_list = self.compute_esp_q_shell_hop(hop_data);
        let gamma_shell_ext = &hop_data.gamma_shell_ext;
        let mut embedding = 0.0;

        for (pair_idx, pair) in self.pairs.iter().enumerate() {
            let fi_i = &hop_data.frag_info[pair.i];
            let fi_j = &hop_data.frag_info[pair.j];
            let shell_range_i = get_frag_shell_range(&hop_data.ext_basis, &fi_i.ext_range);
            let shell_range_j = get_frag_shell_range(&hop_data.ext_basis, &fi_j.ext_range);

            let dq_shell_ext_j = hop_data
                .dq_shell_ext
                .slice(s![shell_range_j.start..shell_range_j.end]);
            let dq_shell_ext_i = hop_data
                .dq_shell_ext
                .slice(s![shell_range_i.start..shell_range_i.end]);

            // ESP on I from all except I and J (shell-level)
            let gamma_ij_shell = gamma_shell_ext.slice(s![
                shell_range_i.start..shell_range_i.end,
                shell_range_j.start..shell_range_j.end
            ]);
            let esp_shell_i: Array1<f64> = &esp_q_shell_list[pair.i] - &gamma_ij_shell.dot(&dq_shell_ext_j);

            let gamma_ji_shell = gamma_shell_ext.slice(s![
                shell_range_j.start..shell_range_j.end,
                shell_range_i.start..shell_range_i.end
            ]);
            let esp_shell_j: Array1<f64> = &esp_q_shell_list[pair.j] - &gamma_ji_shell.dot(&dq_shell_ext_i);

            let ps = &pair_states[pair_idx];
            let mut emb_pair = 0.0;

            // ---- Real shells: use delta_dq_shell_real (shell-level CTIJ) ----
            let n_rs_i = ps.n_real_shells_i;
            let n_rs_j = ps.n_real_shells_j;
            let ddq = &ps.delta_dq_shell_real;
            emb_pair += ddq.slice(s![..n_rs_i]).dot(&esp_shell_i.slice(s![..n_rs_i]));
            emb_pair += ddq.slice(s![n_rs_i..n_rs_i + n_rs_j]).dot(&esp_shell_j.slice(s![..n_rs_j]));

            // ---- Ghost shells: compute CTIJ at shell level per ghost ----
            let n_real_i = fi_i.n_real_atoms;
            let n_real_j = fi_j.n_real_atoms;

            // Count shells per atom for each fragment (needed for ghost shell counts)
            let spa_i = count_shells_per_atom(&hop_data.ext_basis, &fi_i.ext_range);
            let spa_j = count_shells_per_atom(&hop_data.ext_basis, &fi_j.ext_range);

            let n_real_shells_pair = ps.n_real_shells_i + ps.n_real_shells_j;
            let mut mono_ghost_shell_i = n_rs_i;
            let mut mono_ghost_shell_j = n_rs_j;
            let mut pair_ghost_shell = n_real_shells_pair;
            let mut ghost_idx_i = 0usize;
            let mut ghost_idx_j = 0usize;

            for bond in &hop_data.detached_bonds {
                let bda_in_pair = bond.bda_fragment == pair.i || bond.bda_fragment == pair.j;

                if bond.baa_fragment == pair.i {
                    let n_ghost_shells = spa_i[n_real_i + ghost_idx_i];
                    for gs in 0..n_ghost_shells {
                        let mono_dq = mono_states[pair.i].dq_shell[mono_ghost_shell_i + gs];
                        let ctij_s = if bda_in_pair {
                            // Healed: CTIJ = -(dq + q_ref) (POPMAT convention)
                            let q_ref = mono_states[pair.i].q_ref_shell[mono_ghost_shell_i + gs];
                            -(mono_dq + q_ref)
                        } else {
                            let pair_dq = ps.dq_shell[pair_ghost_shell + gs];
                            pair_dq - mono_dq
                        };
                        emb_pair += ctij_s * esp_shell_i[mono_ghost_shell_i + gs];
                    }
                    // Compensate BDA real shells with +q_ref (baa=I → BDA in J)
                    if bda_in_pair {
                        let bda_local = bond.bda_global - hop_data.monomer_indices[pair.j][0];
                        let mut k = 0;
                        for (si, sh) in mono_states[pair.j].basis.shells.iter().enumerate() {
                            if sh.atom_index == bda_local && si < n_rs_j && k < n_ghost_shells {
                                emb_pair += mono_states[pair.i].q_ref_shell[mono_ghost_shell_i + k]
                                    * esp_shell_j[si];
                                k += 1;
                            }
                        }
                    }
                    if !bda_in_pair {
                        pair_ghost_shell += n_ghost_shells;
                    }
                    mono_ghost_shell_i += n_ghost_shells;
                    ghost_idx_i += 1;
                } else if bond.baa_fragment == pair.j {
                    let n_ghost_shells = spa_j[n_real_j + ghost_idx_j];
                    for gs in 0..n_ghost_shells {
                        let mono_dq = mono_states[pair.j].dq_shell[mono_ghost_shell_j + gs];
                        let ctij_s = if bda_in_pair {
                            // Healed: CTIJ = -(dq + q_ref) (POPMAT convention)
                            let q_ref = mono_states[pair.j].q_ref_shell[mono_ghost_shell_j + gs];
                            -(mono_dq + q_ref)
                        } else {
                            let pair_dq = ps.dq_shell[pair_ghost_shell + gs];
                            pair_dq - mono_dq
                        };
                        emb_pair += ctij_s * esp_shell_j[mono_ghost_shell_j + gs];
                    }
                    // Compensate BDA real shells with +q_ref (baa=J → BDA in I)
                    if bda_in_pair {
                        let bda_local = bond.bda_global - hop_data.monomer_indices[pair.i][0];
                        let mut k = 0;
                        for (si, sh) in mono_states[pair.i].basis.shells.iter().enumerate() {
                            if sh.atom_index == bda_local && si < n_rs_i && k < n_ghost_shells {
                                emb_pair += mono_states[pair.j].q_ref_shell[mono_ghost_shell_j + k]
                                    * esp_shell_i[si];
                                k += 1;
                            }
                        }
                    }
                    if !bda_in_pair {
                        pair_ghost_shell += n_ghost_shells;
                    }
                    mono_ghost_shell_j += n_ghost_shells;
                    ghost_idx_j += 1;
                }
            }

            let scal = pair_scal.map_or(1.0, |s| s[pair_idx]);
            embedding += scal * emb_pair;
        }

        embedding
    }

    /// Compute ESD energy with HOP (shell-level).
    ///
    /// Uses extended dq_shell arrays (including ghost charges):
    ///   esd = 0.5 * dq_shell_ext . gamma_shell_ext . dq_shell_ext
    ///       - self_energies - close_pair_energies
    ///
    /// For FMO3: pass `pair_scal` to scale close-pair subtraction by SCAL.
    /// The ESD distant pairs are implicitly SCAL-scaled because
    /// ESD = total - self - SCAL*close_pairs, equivalent to
    /// SCAL*distant + (1-SCAL)*close = distant + (1-SCAL)*(close - distant).
    pub fn esd_energy_hop(
        &self,
        hop_data: &XtbHopData,
    ) -> f64 {
        self.esd_energy_hop_scaled(hop_data, None)
    }

    /// Compute ESD energy with HOP and optional SCAL factors.
    pub fn esd_energy_hop_scaled(
        &self,
        hop_data: &XtbHopData,
        pair_scal: Option<&[f64]>,
    ) -> f64 {
        let gamma_shell_ext = &hop_data.gamma_shell_ext;

        // Total: 0.5 * dq^T . G . dq
        let v_shell = gamma_shell_dsymv(&gamma_shell_ext.view(), &hop_data.dq_shell_ext.view());
        let e_total = 0.5 * hop_data.dq_shell_ext.dot(&v_shell);

        // Subtract self (diagonal blocks)
        let mut e_self = 0.0;
        for frag_idx in 0..self.n_mol {
            let fi = &hop_data.frag_info[frag_idx];
            let shell_range = get_frag_shell_range(&hop_data.ext_basis, &fi.ext_range);
            let dq_i = hop_data
                .dq_shell_ext
                .slice(s![shell_range.start..shell_range.end]);
            let g_ii = gamma_shell_ext.slice(s![
                shell_range.start..shell_range.end,
                shell_range.start..shell_range.end
            ]);
            e_self += 0.5 * dq_i.dot(&g_ii.dot(&dq_i));
        }

        // Subtract close pair contributions (SCAL-scaled for FMO3)
        let mut e_close = 0.0;
        for (pair_idx, pair) in self.pairs.iter().enumerate() {
            let fi_i = &hop_data.frag_info[pair.i];
            let fi_j = &hop_data.frag_info[pair.j];
            let shell_range_i = get_frag_shell_range(&hop_data.ext_basis, &fi_i.ext_range);
            let shell_range_j = get_frag_shell_range(&hop_data.ext_basis, &fi_j.ext_range);
            let dq_i = hop_data
                .dq_shell_ext
                .slice(s![shell_range_i.start..shell_range_i.end]);
            let dq_j = hop_data
                .dq_shell_ext
                .slice(s![shell_range_j.start..shell_range_j.end]);
            let g_ij = gamma_shell_ext.slice(s![
                shell_range_i.start..shell_range_i.end,
                shell_range_j.start..shell_range_j.end
            ]);
            let scal = pair_scal.map_or(1.0, |s| s[pair_idx]);
            e_close += scal * dq_i.dot(&g_ij.dot(&dq_j));
        }

        e_total - e_self - e_close
    }

    /// Run trimer SCC calculations with HOP (FMO3).
    pub fn trimer_scc_hop(
        &self,
        hop_data: &XtbHopData,
        mono_states: &[XtbMonomerHopScc],
        pair_states: &[XtbPairHopScc],
    ) -> (f64, Vec<XtbTrimerHopScc>) {
        let max_iter = self.config.scf.scf_max_cycles;
        let temperature = self.config.scf.electronic_temperature;
        let scf_charge_conv = self.config.scf.scf_charge_conv;
        let scf_energy_conv = self.config.scf.scf_energy_conv;
        let esp_q_shell_list = self.compute_esp_q_shell_hop(hop_data);

        let trimer_results: Vec<(XtbTrimerHopScc, f64)> = self
            .trimers
            .par_iter()
            .map(|trimer| {
                let fi_i = &hop_data.frag_info[trimer.i];
                let fi_j = &hop_data.frag_info[trimer.j];
                let fi_k = &hop_data.frag_info[trimer.k];
                let frag_range_i = self.monomers[trimer.i].slice.atom_as_range();
                let frag_range_j = self.monomers[trimer.j].slice.atom_as_range();
                let frag_range_k = self.monomers[trimer.k].slice.atom_as_range();

                let mut tri_scc = prepare_trimer_hop_xtb(
                    trimer.i,
                    trimer.j,
                    trimer.k,
                    hop_data,
                    mono_states[trimer.i].dq.view(),
                    mono_states[trimer.j].dq.view(),
                    mono_states[trimer.k].dq.view(),
                    mono_states[trimer.i].dq_shell.view(),
                    mono_states[trimer.j].dq_shell.view(),
                    mono_states[trimer.k].dq_shell.view(),
                    esp_q_shell_list[trimer.i].view(),
                    esp_q_shell_list[trimer.j].view(),
                    esp_q_shell_list[trimer.k].view(),
                    fi_i,
                    fi_j,
                    fi_k,
                    &self.monomers[trimer.i].gammafunction,
                    &self.atoms,
                    &frag_range_i,
                    &frag_range_j,
                    &frag_range_k,
                    hop_data.cn_ext.view(),
                );

                let tri_energy = run_trimer_scc_hop_xtb(
                    &mut tri_scc,
                    max_iter,
                    temperature,
                    scf_charge_conv,
                    scf_energy_conv,
                    &self.config.broyden,
                );

                // Get monomer energies
                let e_i = mono_states[trimer.i].last_energy;
                let e_j = mono_states[trimer.j].last_energy;
                let e_k = mono_states[trimer.k].last_energy;

                // Get pair delta energies for sub-pairs
                let gamma_shell_ext = &hop_data.gamma_shell_ext;

                let delta_ij = self.get_pair_delta_hop(
                    trimer.i, trimer.j, hop_data, mono_states, pair_states, gamma_shell_ext,
                );
                let delta_ik = self.get_pair_delta_hop(
                    trimer.i, trimer.k, hop_data, mono_states, pair_states, gamma_shell_ext,
                );
                let delta_jk = self.get_pair_delta_hop(
                    trimer.j, trimer.k, hop_data, mono_states, pair_states, gamma_shell_ext,
                );

                let delta_e = tri_energy - e_i - e_j - e_k - delta_ij - delta_ik - delta_jk;

                (tri_scc, delta_e)
            })
            .collect();

        let mut trimer_states: Vec<XtbTrimerHopScc> = Vec::with_capacity(trimer_results.len());
        let mut total_trimer_delta = 0.0;
        for (tri_scc, delta_e) in trimer_results.into_iter() {
            trimer_states.push(tri_scc);
            total_trimer_delta += delta_e;
        }

        (total_trimer_delta, trimer_states)
    }

    /// Get pair delta energy for a sub-pair in a trimer.
    /// If the pair is a close pair, use its SCC energy. Otherwise use ESD (shell-level).
    fn get_pair_delta_hop(
        &self,
        frag_a: usize,
        frag_b: usize,
        hop_data: &XtbHopData,
        mono_states: &[XtbMonomerHopScc],
        pair_states: &[XtbPairHopScc],
        gamma_shell_ext: &Array2<f64>,
    ) -> f64 {
        if self.properties.type_of_pair(frag_a, frag_b) == PairType::Pair {
            let index = self.properties.index_of_pair(frag_a, frag_b);
            pair_states[index].last_energy
                - mono_states[frag_a].last_energy
                - mono_states[frag_b].last_energy
        } else {
            // ESD approximation: dq_shell_A . gamma_ext[A,B] . dq_shell_B
            let fi_a = &hop_data.frag_info[frag_a];
            let fi_b = &hop_data.frag_info[frag_b];
            let shell_range_a = get_frag_shell_range(&hop_data.ext_basis, &fi_a.ext_range);
            let shell_range_b = get_frag_shell_range(&hop_data.ext_basis, &fi_b.ext_range);
            let dq_a = hop_data
                .dq_shell_ext
                .slice(s![shell_range_a.start..shell_range_a.end]);
            let dq_b = hop_data
                .dq_shell_ext
                .slice(s![shell_range_b.start..shell_range_b.end]);
            let g_ab = gamma_shell_ext.slice(s![
                shell_range_a.start..shell_range_a.end,
                shell_range_b.start..shell_range_b.end
            ]);
            dq_a.dot(&g_ab.dot(&dq_b))
        }
    }

    /// Compute FMO3 embedding energy using CTMUL (matches gradient exactly).
    ///
    /// Reuses the same `compute_ctmul_xtb_hop` function from the gradient code
    /// to ensure identical ghost atom handling. The embedding energy is:
    ///   E_emb = 0.5 * dq_shell_ext . gamma_shell_ext . ctmul_ext
    /// minus self-interaction terms.
    ///
    /// This replaces the separate SCAL*embedding + CTIJK decomposition with
    /// a single unified computation that handles all ghost contributions correctly.
    pub fn embedding_energy_hop_ctmul(
        &self,
        hop_data: &XtbHopData,
        mono_states: &[XtbMonomerHopScc],
        pair_states: &[XtbPairHopScc],
        trimer_states: &[XtbTrimerHopScc],
        pair_scal: &[f64],
    ) -> f64 {
        use crate::fmo::gradients::hop_gradients::helpers::compute_ctmul_xtb_hop;

        // Compute CTMUL using the same function as the gradient
        let ctmul_ext = compute_ctmul_xtb_hop(
            hop_data, mono_states, pair_states, trimer_states, pair_scal,
        );

        // Embedding = sum_frags( ESP_external_I . CTMUL_I )
        // where ESP_external_I = gamma_ext[I,:] . dq_ext - gamma_ext[I,I] . dq_I
        let esp_q_shell_list = self.compute_esp_q_shell_hop(hop_data);
        let mut embedding = 0.0;
        for frag_idx in 0..self.n_mol {
            let fi = &hop_data.frag_info[frag_idx];
            let shell_range = get_frag_shell_range(&hop_data.ext_basis, &fi.ext_range);
            let esp = &esp_q_shell_list[frag_idx];
            let ctmul_frag = ctmul_ext.slice(s![shell_range.start..shell_range.end]);
            embedding += esp.dot(&ctmul_frag);
        }
        embedding
    }

    /// Compute trimer embedding energy with HOP (FMO3) — old formula.
    ///
    /// Uses delta_dq_shell_real from trimer SCC. Subtracts pair embeddings.
    /// NOTE: This function has ghost-structure mismatch issues for >3 fragment systems.
    /// Prefer trimer_ctijk_energy_hop + SCAL-scaled embedding instead.
    pub fn trimer_embedding_energy_hop(
        &self,
        hop_data: &XtbHopData,
        mono_states: &[XtbMonomerHopScc],
        pair_states: &[XtbPairHopScc],
        trimer_states: &[XtbTrimerHopScc],
    ) -> f64 {
        let esp_q_shell_list = self.compute_esp_q_shell_hop(hop_data);
        let gamma_shell_ext = &hop_data.gamma_shell_ext;

        let embedding_vec: Vec<f64> = self
            .trimers
            .par_iter()
            .enumerate()
            .map(|(tri_idx, trimer)| {
                let fi_i = &hop_data.frag_info[trimer.i];
                let fi_j = &hop_data.frag_info[trimer.j];
                let fi_k = &hop_data.frag_info[trimer.k];
                let shell_range_i = get_frag_shell_range(&hop_data.ext_basis, &fi_i.ext_range);
                let shell_range_j = get_frag_shell_range(&hop_data.ext_basis, &fi_j.ext_range);
                let shell_range_k = get_frag_shell_range(&hop_data.ext_basis, &fi_k.ext_range);

                let dq_shell_ext_i = hop_data
                    .dq_shell_ext
                    .slice(s![shell_range_i.start..shell_range_i.end]);
                let dq_shell_ext_j = hop_data
                    .dq_shell_ext
                    .slice(s![shell_range_j.start..shell_range_j.end]);
                let dq_shell_ext_k = hop_data
                    .dq_shell_ext
                    .slice(s![shell_range_k.start..shell_range_k.end]);

                // ESP on I from all except I, J, K
                let esp_i: Array1<f64> = &esp_q_shell_list[trimer.i]
                    - &gamma_shell_ext
                        .slice(s![
                            shell_range_i.start..shell_range_i.end,
                            shell_range_j.start..shell_range_j.end
                        ])
                        .dot(&dq_shell_ext_j)
                    - &gamma_shell_ext
                        .slice(s![
                            shell_range_i.start..shell_range_i.end,
                            shell_range_k.start..shell_range_k.end
                        ])
                        .dot(&dq_shell_ext_k);

                let esp_j: Array1<f64> = &esp_q_shell_list[trimer.j]
                    - &gamma_shell_ext
                        .slice(s![
                            shell_range_j.start..shell_range_j.end,
                            shell_range_i.start..shell_range_i.end
                        ])
                        .dot(&dq_shell_ext_i)
                    - &gamma_shell_ext
                        .slice(s![
                            shell_range_j.start..shell_range_j.end,
                            shell_range_k.start..shell_range_k.end
                        ])
                        .dot(&dq_shell_ext_k);

                let esp_k: Array1<f64> = &esp_q_shell_list[trimer.k]
                    - &gamma_shell_ext
                        .slice(s![
                            shell_range_k.start..shell_range_k.end,
                            shell_range_i.start..shell_range_i.end
                        ])
                        .dot(&dq_shell_ext_i)
                    - &gamma_shell_ext
                        .slice(s![
                            shell_range_k.start..shell_range_k.end,
                            shell_range_j.start..shell_range_j.end
                        ])
                        .dot(&dq_shell_ext_j);

                // delta_dq_shell from trimer (real shells only)
                let ts = &trimer_states[tri_idx];
                let ns_i = ts.n_real_shells_i;
                let ns_j = ts.n_real_shells_j;
                let ns_k = ts.n_real_shells_k;
                let ddq = &ts.delta_dq_shell_real;
                let ddq_i = ddq.slice(s![..ns_i]);
                let ddq_j = ddq.slice(s![ns_i..ns_i + ns_j]);
                let ddq_k = ddq.slice(s![ns_i + ns_j..ns_i + ns_j + ns_k]);

                // Trimer embedding: ESP_I . ddq_I + ESP_J . ddq_J + ESP_K . ddq_K
                // Real shells first
                let mut embedding_terms: f64 = 0.0;
                embedding_terms += esp_i.slice(s![..ns_i]).dot(&ddq_i);
                embedding_terms += esp_j.slice(s![..ns_j]).dot(&ddq_j);
                embedding_terms += esp_k.slice(s![..ns_k]).dot(&ddq_k);

                // Ghost shell CTIJK contributions (matching gradient's compute_ctmul_xtb_hop)
                {
                    use super::hop_data::get_frag_shell_range;
                    let spa_i = count_shells_per_atom(&hop_data.ext_basis, &fi_i.ext_range);
                    let spa_j = count_shells_per_atom(&hop_data.ext_basis, &fi_j.ext_range);
                    let spa_k = count_shells_per_atom(&hop_data.ext_basis, &fi_k.ext_range);
                    let mut mono_ghost_offset_i = fi_i.n_real_shells;
                    let mut mono_ghost_offset_j = fi_j.n_real_shells;
                    let mut mono_ghost_offset_k = fi_k.n_real_shells;
                    let mut tri_ghost_offset =
                        ts.n_real_shells_i + ts.n_real_shells_j + ts.n_real_shells_k;
                    let mut ghost_idx_i = 0usize;
                    let mut ghost_idx_j = 0usize;
                    let mut ghost_idx_k = 0usize;
                    let tri_frags = [trimer.i, trimer.j, trimer.k];

                    for bond in &hop_data.detached_bonds {
                        let bda_in_tri = tri_frags.contains(&bond.bda_fragment);

                        if bond.baa_fragment == trimer.i {
                            let n_gs = spa_i[fi_i.n_real_atoms + ghost_idx_i];
                            for s in 0..n_gs {
                                let esp_val = esp_i[mono_ghost_offset_i + s];
                                let ctijk_ghost = if bda_in_tri {
                                    // Healed: CTIJK = -mono_dq (ghost disappears in trimer)
                                    -mono_states[trimer.i].dq_shell[mono_ghost_offset_i + s]
                                } else {
                                    // Partial: CTIJK = tri_dq - mono_dq
                                    ts.dq_shell[tri_ghost_offset + s]
                                        - mono_states[trimer.i].dq_shell[mono_ghost_offset_i + s]
                                };
                                embedding_terms += esp_val * ctijk_ghost;
                            }
                            if !bda_in_tri {
                                tri_ghost_offset += n_gs;
                            }
                            mono_ghost_offset_i += n_gs;
                            ghost_idx_i += 1;
                        } else if bond.baa_fragment == trimer.j {
                            let n_gs = spa_j[fi_j.n_real_atoms + ghost_idx_j];
                            for s in 0..n_gs {
                                let esp_val = esp_j[mono_ghost_offset_j + s];
                                let ctijk_ghost = if bda_in_tri {
                                    -mono_states[trimer.j].dq_shell[mono_ghost_offset_j + s]
                                } else {
                                    ts.dq_shell[tri_ghost_offset + s]
                                        - mono_states[trimer.j].dq_shell[mono_ghost_offset_j + s]
                                };
                                embedding_terms += esp_val * ctijk_ghost;
                            }
                            if !bda_in_tri {
                                tri_ghost_offset += n_gs;
                            }
                            mono_ghost_offset_j += n_gs;
                            ghost_idx_j += 1;
                        } else if bond.baa_fragment == trimer.k {
                            let n_gs = spa_k[fi_k.n_real_atoms + ghost_idx_k];
                            for s in 0..n_gs {
                                let esp_val = esp_k[mono_ghost_offset_k + s];
                                let ctijk_ghost = if bda_in_tri {
                                    -mono_states[trimer.k].dq_shell[mono_ghost_offset_k + s]
                                } else {
                                    ts.dq_shell[tri_ghost_offset + s]
                                        - mono_states[trimer.k].dq_shell[mono_ghost_offset_k + s]
                                };
                                embedding_terms += esp_val * ctijk_ghost;
                            }
                            if !bda_in_tri {
                                tri_ghost_offset += n_gs;
                            }
                            mono_ghost_offset_k += n_gs;
                            ghost_idx_k += 1;
                        }
                    }
                }

                // Subtract pair embeddings (same pattern as non-HOP)
                let emb_ij = self.pair_embedding_hop(
                    trimer.i, trimer.j, hop_data, pair_states, mono_states,
                    &esp_q_shell_list, gamma_shell_ext,
                    &shell_range_i, &shell_range_j,
                    fi_i, fi_j,
                );
                let emb_ik = self.pair_embedding_hop(
                    trimer.i, trimer.k, hop_data, pair_states, mono_states,
                    &esp_q_shell_list, gamma_shell_ext,
                    &shell_range_i, &shell_range_k,
                    fi_i, fi_k,
                );
                let emb_jk = self.pair_embedding_hop(
                    trimer.j, trimer.k, hop_data, pair_states, mono_states,
                    &esp_q_shell_list, gamma_shell_ext,
                    &shell_range_j, &shell_range_k,
                    fi_j, fi_k,
                );

                embedding_terms -= emb_ij + emb_ik + emb_jk;
                embedding_terms
            })
            .collect();

        embedding_vec.iter().sum()
    }

    /// Compute pair embedding energy for trimer subtraction.
    /// Returns 0.0 for ESD pairs.
    #[allow(clippy::too_many_arguments)]
    fn pair_embedding_hop(
        &self,
        frag_a: usize,
        frag_b: usize,
        hop_data: &XtbHopData,
        pair_states: &[XtbPairHopScc],
        mono_states: &[XtbMonomerHopScc],
        esp_q_shell_list: &[Array1<f64>],
        gamma_shell_ext: &Array2<f64>,
        shell_range_a: &std::ops::Range<usize>,
        shell_range_b: &std::ops::Range<usize>,
        fi_a: &super::hop_data::XtbHopFragInfo,
        fi_b: &super::hop_data::XtbHopFragInfo,
    ) -> f64 {
        if self.properties.type_of_pair(frag_a, frag_b) != PairType::Pair {
            return 0.0;
        }

        let index = self.properties.index_of_pair(frag_a, frag_b);
        let ps = &pair_states[index];
        // mono_states used for ghost shell dq and q_ref access

        // Map frag_a/frag_b to pair's I/J ordering
        let (pair_i, pair_j) = (ps.i, ps.j);
        let (fi_i, fi_j, sr_i, sr_j) = if frag_a == pair_i {
            (fi_a, fi_b, shell_range_a, shell_range_b)
        } else {
            (fi_b, fi_a, shell_range_b, shell_range_a)
        };
        // ESP on I excluding J (shell-level)
        let dq_j_range = hop_data.dq_shell_ext.slice(s![sr_j.start..sr_j.end]);
        let corr_i = gamma_shell_ext
            .slice(s![sr_i.start..sr_i.end, sr_j.start..sr_j.end])
            .dot(&dq_j_range);
        let esp_i: Array1<f64> = &esp_q_shell_list[pair_i] - &corr_i;

        let dq_i_range = hop_data.dq_shell_ext.slice(s![sr_i.start..sr_i.end]);
        let corr_j = gamma_shell_ext
            .slice(s![sr_j.start..sr_j.end, sr_i.start..sr_i.end])
            .dot(&dq_i_range);
        let esp_j: Array1<f64> = &esp_q_shell_list[pair_j] - &corr_j;

        let ddq = &ps.delta_dq_shell_real;
        let n_rs_i = ps.n_real_shells_i;
        let n_rs_j = ps.n_real_shells_j;

        // Real shells
        let mut emb = ddq.slice(s![..n_rs_i]).dot(&esp_i.slice(s![..n_rs_i]))
            + ddq.slice(s![n_rs_i..]).dot(&esp_j.slice(s![..n_rs_j]));

        // Ghost shells (matching embedding_energy_hop ghost handling)
        let n_real_i = fi_i.n_real_atoms;
        let n_real_j = fi_j.n_real_atoms;
        let spa_i = count_shells_per_atom(&hop_data.ext_basis, &fi_i.ext_range);
        let spa_j = count_shells_per_atom(&hop_data.ext_basis, &fi_j.ext_range);

        let n_real_shells_pair = n_rs_i + n_rs_j;
        let mut mono_ghost_shell_i = n_rs_i;
        let mut mono_ghost_shell_j = n_rs_j;
        let mut pair_ghost_shell = n_real_shells_pair;
        let mut ghost_idx_i = 0usize;
        let mut ghost_idx_j = 0usize;

        for bond in &hop_data.detached_bonds {
            let bda_in_pair = bond.bda_fragment == pair_i || bond.bda_fragment == pair_j;

            if bond.baa_fragment == pair_i {
                let n_ghost_shells = spa_i[n_real_i + ghost_idx_i];
                // Use mono_states from hop_data (we need dq_shell and q_ref_shell)
                // We access these through the pair_states' stored mono references
                // Actually, we need access to mono_states. Let's use hop_data.dq_shell_ext instead.
                // Ghost CTIJ: for healed bonds, CTIJ = -(dq + q_ref)
                // For partial bonds, CTIJ = pair_dq - mono_dq
                for gs in 0..n_ghost_shells {
                    let mono_dq = mono_states[pair_i].dq_shell[mono_ghost_shell_i + gs];
                    let ctij_s = if bda_in_pair {
                        let q_ref = mono_states[pair_i].q_ref_shell[mono_ghost_shell_i + gs];
                        -(mono_dq + q_ref)
                    } else {
                        let pair_dq = ps.dq_shell[pair_ghost_shell + gs];
                        pair_dq - mono_dq
                    };
                    emb += ctij_s * esp_i[mono_ghost_shell_i + gs];
                }
                // Compensate BDA real shells with +q_ref (baa=I → BDA in J)
                if bda_in_pair {
                    let bda_local = bond.bda_global - hop_data.monomer_indices[pair_j][0];
                    let mut k = 0;
                    for (si, sh) in mono_states[pair_j].basis.shells.iter().enumerate() {
                        if sh.atom_index == bda_local && si < n_rs_j && k < n_ghost_shells {
                            emb += mono_states[pair_i].q_ref_shell[mono_ghost_shell_i + k]
                                * esp_j[si];
                            k += 1;
                        }
                    }
                }
                if !bda_in_pair { pair_ghost_shell += n_ghost_shells; }
                mono_ghost_shell_i += n_ghost_shells;
                ghost_idx_i += 1;
            } else if bond.baa_fragment == pair_j {
                let n_ghost_shells = spa_j[n_real_j + ghost_idx_j];
                for gs in 0..n_ghost_shells {
                    let mono_dq = mono_states[pair_j].dq_shell[mono_ghost_shell_j + gs];
                    let ctij_s = if bda_in_pair {
                        let q_ref = mono_states[pair_j].q_ref_shell[mono_ghost_shell_j + gs];
                        -(mono_dq + q_ref)
                    } else {
                        let pair_dq = ps.dq_shell[pair_ghost_shell + gs];
                        pair_dq - mono_dq
                    };
                    emb += ctij_s * esp_j[mono_ghost_shell_j + gs];
                }
                // Compensate BDA real shells with +q_ref (baa=J → BDA in I)
                if bda_in_pair {
                    let bda_local = bond.bda_global - hop_data.monomer_indices[pair_i][0];
                    let mut k = 0;
                    for (si, sh) in mono_states[pair_i].basis.shells.iter().enumerate() {
                        if sh.atom_index == bda_local && si < n_rs_i && k < n_ghost_shells {
                            emb += mono_states[pair_j].q_ref_shell[mono_ghost_shell_j + k]
                                * esp_i[si];
                            k += 1;
                        }
                    }
                }
                if !bda_in_pair { pair_ghost_shell += n_ghost_shells; }
                mono_ghost_shell_j += n_ghost_shells;
                ghost_idx_j += 1;
            }
        }

        emb
    }
}
