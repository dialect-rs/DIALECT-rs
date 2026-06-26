//! FMO-DFTB gradient with Hybrid Orbital Projection (HOP).
//!
//! Adapts the non-HOP loop-fused gradient (`fmo_gradient.rs`) for covalent fragmentation:
//! - Extended atom lists (real + ghost boundary atoms)
//! - ZREF/QREF-scaled repulsive energy gradients
//! - POPMAT-based CTIJ/CTMUL/embedding
//! - Ghost-to-BAA scatter for local→global gradient mapping
//! - HOP projector gradient (HOPSDER + HOPCODER)

pub mod helpers;
pub mod hop_projector;
pub mod interfragment;
pub mod monomer;
pub mod numerical;
pub mod pair;
pub mod response_hop;

use crate::fmo::scc_hop::hop_data::HopData;
use crate::fmo::scc_hop::monomer::MonomerHopScc;
use crate::fmo::scc_hop::pair::PairHopScc;
use crate::fmo::SuperSystem;
use crate::initialization::Atom;
use rayon::prelude::*;
use ndarray::prelude::*;

use helpers::{
    build_monomer_local_to_global_dftb, build_pair_local_to_global_dftb, compute_ctmul_hop,
    compute_esp_q_hop, compute_espgrad_shiftct_hop, compute_shiftct_hop,
    get_pair_ghost_baa_globals, scatter_to_global,
};
use hop_projector::compute_hop_gradient_fmo_dftb;
use interfragment::interfragment_gradient_hop;
use monomer::monomer_gradient_combined_hop;
use pair::pair_gradient_combined_hop;
use response_hop::response_gradient_hop_total;

impl SuperSystem<'_> {
    /// Compute complete FMO-DFTB HOP gradient.
    ///
    /// Follows the same structure as `ground_state_gradient_fmo()` plus HOP additions:
    /// 1. CTMUL_ext (extended, POPMAT-based)
    /// 2. SHIFTCT + ESPGRAD per monomer (extended atoms)
    /// 3. Monomer gradients → scatter real+ghost to global
    /// 4. Pair gradients
    /// 5. Pair delta + CTIJ (ghost scatter + monomer ghost subtraction for healed bonds)
    /// 6. Inter-fragment gradient (extended CTMUL + ESD with ghost charges)
    /// 7. HOP projector gradient
    /// 8. Response gradient (SCZV Z-vector)
    /// 9. Assembly
    pub fn ground_state_gradient_fmo_hop(
        &self,
        hop_data: &HopData,
        mono_states: &[MonomerHopScc],
        pair_states: &[PairHopScc],
    ) -> Array1<f64> {
        let atoms: &[Atom] = &self.atoms[..];
        let n_atoms_total = atoms.len();
        let n_grad = 3 * n_atoms_total;
        let detached_bonds = &hop_data.detached_bonds;

        // Step 1: Compute CTMUL for all extended atoms (POPMAT-based)
        let ctmul_ext = compute_ctmul_hop(hop_data, mono_states, pair_states, &self.pairs);

        // Step 2: Compute SHIFTCT + ESPGRAD per monomer [parallel]
        let shiftcts: Vec<Array1<f64>> = (0..self.n_mol)
            .into_par_iter()
            .map(|frag_idx| {
                let embed = compute_shiftct_hop(frag_idx, hop_data, ctmul_ext.view());
                let espgrad = compute_espgrad_shiftct_hop(
                    frag_idx, hop_data, mono_states, pair_states, &self.pairs,
                );
                &embed + &espgrad
            })
            .collect();

        // Compute ESP for each monomer (external ESP from all other fragments) [parallel]
        let esp_q_list: Vec<Array1<f64>> = (0..self.n_mol)
            .into_par_iter()
            .map(|frag_idx| compute_esp_q_hop(frag_idx, hop_data))
            .collect();

        // Step 3: Monomer gradients (extended atoms) [parallel]
        let monomer_results: Vec<(Array1<f64>, Array1<f64>)> = (0..self.n_mol)
            .into_par_iter()
            .map(|frag_idx| {
                monomer_gradient_combined_hop(
                    &mono_states[frag_idx],
                    hop_data,
                    frag_idx,
                    shiftcts[frag_idx].view(),
                    esp_q_list[frag_idx].view(),
                    &self.gammafunction,
                    &self.gammafunction_lc,
                    self.monomers[frag_idx].slako,
                    self.monomers[frag_idx].vrep,
                )
            })
            .collect();

        // Assemble monomer gradients: scatter real + ghost to global
        let mut monomer_grad_total = Array1::<f64>::zeros(n_grad);
        let mut addlag_total = Array1::<f64>::zeros(n_grad);

        let local_to_globals: Vec<Vec<usize>> = (0..self.n_mol)
            .map(|frag_idx| {
                let frag_range = self.monomers[frag_idx].slice.atom_as_range();
                build_monomer_local_to_global_dftb(&frag_range, detached_bonds, frag_idx)
            })
            .collect();

        for (frag_idx, (mon_grad, add_grad)) in monomer_results.iter().enumerate() {
            let ltg = &local_to_globals[frag_idx];
            let n_ext = mono_states[frag_idx].n_ext_atoms;
            scatter_to_global(&mut monomer_grad_total, mon_grad, ltg, n_ext);
            scatter_to_global(&mut addlag_total, add_grad, ltg, n_ext);
        }

        // Step 4: Pair gradients (extended atoms)
        let pair_ghost_baa: Vec<Vec<usize>> = self
            .pairs
            .iter()
            .map(|pair| get_pair_ghost_baa_globals(hop_data, pair.i, pair.j))
            .collect();

        let pair_local_to_globals: Vec<Vec<usize>> = self
            .pairs
            .iter()
            .zip(pair_ghost_baa.iter())
            .map(|(pair, ghost_baa)| {
                let frag_range_i = self.monomers[pair.i].slice.atom_as_range();
                let frag_range_j = self.monomers[pair.j].slice.atom_as_range();
                build_pair_local_to_global_dftb(&frag_range_i, &frag_range_j, ghost_baa)
            })
            .collect();

        let pair_results: Vec<(Array1<f64>, Array1<f64>)> = self
            .pairs
            .par_iter()
            .enumerate()
            .map(|(pair_idx, pair)| {
                pair_gradient_combined_hop(
                    &pair_states[pair_idx],
                    hop_data,
                    mono_states,
                    n_atoms_total,
                    esp_q_list[pair.i].view(),
                    esp_q_list[pair.j].view(),
                    &self.gammafunction,
                    &self.gammafunction_lc,
                    self.monomers[pair.i].slako,
                    self.monomers[pair.i].vrep,
                    &pair_local_to_globals[pair_idx],
                )
            })
            .collect();

        // Step 5: Pair delta + CTIJ accumulation
        let mut pair_delta_total = Array1::<f64>::zeros(n_grad);
        let mut ctij_total = Array1::<f64>::zeros(n_grad);

        for (pair_idx, pair) in self.pairs.iter().enumerate() {
            let (pair_grad_local, ctij_grad_global) = &pair_results[pair_idx];
            let fi_i = &hop_data.frag_info[pair.i];
            let fi_j = &hop_data.frag_info[pair.j];
            let n_real_i = fi_i.n_real_atoms;
            let n_real_j = fi_j.n_real_atoms;
            let mon_i_grad = &monomer_results[pair.i].0;
            let mon_j_grad = &monomer_results[pair.j].0;
            let ghost_baa = &pair_ghost_baa[pair_idx];

            // I's real atoms: pair[local] - mono_I[local]
            for (local_idx, global_idx) in self.monomers[pair.i].slice.atom_as_range().enumerate() {
                for k in 0..3 {
                    pair_delta_total[3 * global_idx + k] +=
                        pair_grad_local[3 * local_idx + k] - mon_i_grad[3 * local_idx + k];
                }
            }

            // J's real atoms: pair offset = n_real_i
            for (local_idx, global_idx) in self.monomers[pair.j].slice.atom_as_range().enumerate() {
                for k in 0..3 {
                    pair_delta_total[3 * global_idx + k] +=
                        pair_grad_local[3 * (n_real_i + local_idx) + k]
                            - mon_j_grad[3 * local_idx + k];
                }
            }

            // Pair ghost atoms → BAA global
            let n_real_atoms = n_real_i + n_real_j;
            for (ghost_idx, &baa_global) in ghost_baa.iter().enumerate() {
                let pair_ghost_local = n_real_atoms + ghost_idx;
                for k in 0..3 {
                    pair_delta_total[3 * baa_global + k] +=
                        pair_grad_local[3 * pair_ghost_local + k];
                }
            }

            // Subtract monomer ghost contributions for all bonds where BAA is in this pair.
            // Ghost lives in BAA's monomer at BDA's position.
            // Covers both partial_BAA bonds (pair has ghost, need delta) and healed bonds
            // (pair has no ghost, need to undo monomer ghost contribution).
            for bond in detached_bonds {
                let baa_in_pair = bond.baa_fragment == pair.i || bond.baa_fragment == pair.j;
                if !baa_in_pair {
                    continue;
                }
                let baa_frag = bond.baa_fragment;
                let mono_grad = &monomer_results[baa_frag].0;
                let fi_baa = &hop_data.frag_info[baa_frag];
                let mono_bonds: Vec<_> = detached_bonds
                    .iter()
                    .filter(|b| b.baa_fragment == baa_frag)
                    .collect();
                if let Some(ghost_idx) = mono_bonds.iter().position(|b|
                    b.bda_global == bond.bda_global && b.baa_global == bond.baa_global
                ) {
                    let local_idx = fi_baa.n_real_atoms + ghost_idx;
                    for k in 0..3 {
                        pair_delta_total[3 * bond.bda_global + k] -=
                            mono_grad[3 * local_idx + k];
                    }
                }
            }

            // CTIJ contribution
            ctij_total += ctij_grad_global;
        }

        // Note: addlag is a monomer-only contribution (not part of pair delta).
        // Ghost atom addlag contributions are legitimate and should NOT be undone
        // for healed bonds. The pair gradient does not compute addlag, so there is
        // no corresponding pair-level term to delta against.

        // Step 6: Inter-fragment gradient (extended CTMUL + ESD with ghost charges)
        let interfrag_grad = interfragment_gradient_hop(
            atoms,
            hop_data,
            mono_states,
            &self.esd_pairs,
            ctmul_ext.view(),
            &self.gammafunction,
        );

        // Step 7: HOP projector gradient
        let mut hop_grad = Array1::<f64>::zeros(n_grad);
        compute_hop_gradient_fmo_dftb(
            hop_data,
            mono_states,
            pair_states,
            &self.pairs,
            atoms,
            self.monomers[0].slako, // All monomers share the same SK parameters
            &mut hop_grad,
        );

        // Step 8: Response gradient (SCZV Z-vector contribution)
        let response_grad = response_gradient_hop_total(
            hop_data,
            mono_states,
            &self.pairs,
            atoms,
            &shiftcts,
            &self.gammafunction,
            &self.gammafunction_lc,
            self.monomers[0].slako,
        );

        // Step 9: Assembly
        &monomer_grad_total
            + &pair_delta_total
            + &ctij_total
            + &interfrag_grad
            + &addlag_total
            + &hop_grad
            + &response_grad
    }
}
