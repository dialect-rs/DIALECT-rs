//! Pair gradient with HOP for FMO-DFTB.
//!
//! Adapts `pair_gradient_combined()` from `fmo_gradient.rs` for extended pairs:
//! - Extended atom lists (real_I + real_J + partial ghost atoms)
//! - POPMAT-based CTIJ computation
//! - ZREF/QREF-scaled repulsive energy gradient
//! - Ghost atom contributions to SK integrals, gamma, and CTIJ

use super::helpers::{
    build_orbital_offsets, build_shift_ao_matrix, compute_w_matrix, grad_repulsive_energy_scaled,
};
use crate::defaults::PROXIMITY_CUTOFF;
use crate::fmo::scc_hop::hop_data::HopData;
use crate::fmo::scc_hop::monomer::MonomerHopScc;
use crate::fmo::scc_hop::pair::PairHopScc;
use crate::gradients::helpers::compute_lr_coefficients_onthefly;
use crate::initialization::parameters::{RepulsivePotential, SlaterKoster};
use crate::param::slako_transformations::{
    directional_cosines, slako_transformation_gradients_fast, SplineCache,
};
use crate::scc::gamma_approximation::GammaFunction;
use ndarray::prelude::*;

/// Combined pair gradient with HOP: SCC + CTIJ + gamma + v_rep.
///
/// Returns `(pair_grad_local, ctij_grad_global)`.
/// - `pair_grad_local`: gradient in local pair coordinates `[3 * n_ext_pair_atoms]`
/// - `ctij_grad_global`: CTIJ contribution in global coordinates `[3 * n_atoms_total]`
pub fn pair_gradient_combined_hop(
    ps: &PairHopScc,
    hop_data: &HopData,
    mono_states: &[MonomerHopScc],
    n_atoms_total: usize,
    esp_q_i: ArrayView1<f64>,
    esp_q_j: ArrayView1<f64>,
    gammafunction: &GammaFunction,
    gammafunction_lc: &Option<GammaFunction>,
    slako: &SlaterKoster,
    vrep: &RepulsivePotential,
    local_to_global: &[usize],
) -> (Array1<f64>, Array1<f64>) {
    let ext_atoms = &ps.ext_atoms;
    let n_atoms_pair = ps.n_ext_atoms;
    let n_orbs = ps.n_ext_orbs;
    let n_real_i = ps.n_real_i;
    let n_real_j = ps.n_real_j;
    let n_real_atoms = ps.n_real_atoms;

    // Get stored SCC properties
    let p: ArrayView2<f64> = ps.p.view();
    let gamma: ArrayView2<f64> = ps.gamma.view();
    let dq: ArrayView1<f64> = ps.dq.view();
    let orbs: ArrayView2<f64> = ps.orbs.as_ref().expect("pair orbs not stored").view();
    let orbe: ArrayView1<f64> = ps.orbe.as_ref().expect("pair orbe not stored").view();
    let f: &[f64] = &ps.f;
    let nocc = ps.n_elec / 2;

    // Compute total shift = gamma_pair * dq_pair + ESP from K != I,J
    let intra_shift: Array1<f64> = gamma.dot(&dq);

    // ESP from K!=I,J: remove J's contribution from I's ESP and vice versa
    let fi_i = &hop_data.frag_info[ps.i];
    let fi_j = &hop_data.frag_info[ps.j];
    let gamma_ext = &hop_data.gamma_ext;
    let ext_i = &fi_i.ext_range;
    let ext_j = &fi_j.ext_range;
    let dq_ext_j = hop_data.dq_ext.slice(s![ext_j.start..ext_j.end]);
    let dq_ext_i = hop_data.dq_ext.slice(s![ext_i.start..ext_i.end]);

    let gamma_ij_block = gamma_ext.slice(s![ext_i.start..ext_i.end, ext_j.start..ext_j.end]);
    let gamma_ji_block = gamma_ext.slice(s![ext_j.start..ext_j.end, ext_i.start..ext_i.end]);

    let mut esp_from_k = Array1::<f64>::zeros(n_atoms_pair);

    // Real atoms in I: esp_q_I (extended) minus gamma_IJ . dq_J
    let esp_i_full: Array1<f64> = &esp_q_i - &gamma_ij_block.dot(&dq_ext_j);
    esp_from_k.slice_mut(s![..n_real_i]).assign(&esp_i_full.slice(s![..n_real_i]));

    // Real atoms in J: esp_q_J minus gamma_JI . dq_I
    let esp_j_full: Array1<f64> = &esp_q_j - &gamma_ji_block.dot(&dq_ext_i);
    esp_from_k.slice_mut(s![n_real_i..n_real_atoms]).assign(&esp_j_full.slice(s![..n_real_j]));

    // Ghost atoms in pair: ESP from all fragments except I and J
    // (same as in prepare_pair_hop)
    let mut ghost_idx_in_pair = n_real_atoms;
    for bond in &hop_data.detached_bonds {
        let bda_in_pair = bond.bda_fragment == ps.i || bond.bda_fragment == ps.j;
        let baa_in_pair = bond.baa_fragment == ps.i || bond.baa_fragment == ps.j;
        if !bda_in_pair && baa_in_pair {
            // Partial-BAA: ghost exists in pair
            let bda_frag = bond.bda_fragment;
            let bda_ext_range = &hop_data.frag_info[bda_frag].ext_range;
            let bda_frag_atom_start = hop_data.monomer_indices[bda_frag][0];
            let bda_local_in_frag = bond.bda_global - bda_frag_atom_start;
            let bda_ext_idx = bda_ext_range.start + bda_local_in_frag;

            let full_esp: f64 = gamma_ext.row(bda_ext_idx).dot(&hop_data.dq_ext);
            let esp_from_i: f64 = gamma_ext.slice(s![bda_ext_idx, ext_i.start..ext_i.end]).dot(&dq_ext_i);
            let esp_from_j: f64 = gamma_ext.slice(s![bda_ext_idx, ext_j.start..ext_j.end]).dot(&dq_ext_j);
            esp_from_k[ghost_idx_in_pair] = full_esp - esp_from_i - esp_from_j;
            ghost_idx_in_pair += 1;
        }
    }

    let total_shift: Array1<f64> = &intra_shift + &esp_from_k;

    // Build shift AO matrix and W matrix
    let shift_ao = build_shift_ao_matrix(total_shift.view(), ext_atoms, n_orbs);
    let w = compute_w_matrix(orbs, orbe, f, n_orbs, nocc);

    // WRK1 = shift_ao * P - W
    let mut wrk1 = Array2::<f64>::zeros([n_orbs, n_orbs]);
    for mu in 0..n_orbs {
        for nu in 0..n_orbs {
            wrk1[[mu, nu]] = shift_ao[[mu, nu]] * p[[mu, nu]] - w[[mu, nu]];
        }
    }

    // Build CTIJ using POPMAT differences (matching embedding_energy_hop)
    let mut ctij = Array1::<f64>::zeros(n_atoms_pair);
    let mut dq_monomer = Array1::<f64>::zeros(n_atoms_pair);

    // I real atoms
    for a in 0..n_real_i {
        let popmat_pair = dq[a] + ps.zref[a];
        let mono_zref = hop_data.zref[fi_i.ext_range.start + a];
        let popmat_mono = mono_states[ps.i].dq[a] + mono_zref;
        ctij[a] = popmat_pair - popmat_mono;
        dq_monomer[a] = mono_states[ps.i].dq[a];
    }

    // J real atoms
    for a in 0..n_real_j {
        let pair_a = n_real_i + a;
        let popmat_pair = dq[pair_a] + ps.zref[pair_a];
        let mono_zref = hop_data.zref[fi_j.ext_range.start + a];
        let popmat_mono = mono_states[ps.j].dq[a] + mono_zref;
        ctij[pair_a] = popmat_pair - popmat_mono;
        dq_monomer[pair_a] = mono_states[ps.j].dq[a];
    }

    // Ghost atoms: partial-BAA bonds
    let mut pair_ghost_idx_ct = n_real_atoms;
    let mut mono_ghost_i_idx = 0usize;
    let mut mono_ghost_j_idx = 0usize;
    for bond in &hop_data.detached_bonds {
        let bda_in_pair = bond.bda_fragment == ps.i || bond.bda_fragment == ps.j;
        let baa_in_pair = bond.baa_fragment == ps.i || bond.baa_fragment == ps.j;

        if bond.baa_fragment == ps.i {
            if !bda_in_pair && baa_in_pair {
                // Partial: ghost exists in pair
                let mono_ghost_local = n_real_i + mono_ghost_i_idx;
                let mono_zref_g = hop_data.zref[fi_i.ext_range.start + mono_ghost_local];
                let popmat_mono_g = mono_states[ps.i].dq[mono_ghost_local] + mono_zref_g;
                let popmat_pair_g = dq[pair_ghost_idx_ct] + ps.zref[pair_ghost_idx_ct];
                ctij[pair_ghost_idx_ct] = popmat_pair_g - popmat_mono_g;
                dq_monomer[pair_ghost_idx_ct] = mono_states[ps.i].dq[mono_ghost_local];
                pair_ghost_idx_ct += 1;
            }
            mono_ghost_i_idx += 1;
        } else if bond.baa_fragment == ps.j {
            if !bda_in_pair && baa_in_pair {
                // Partial: ghost exists in pair
                let mono_ghost_local = n_real_j + mono_ghost_j_idx;
                let mono_zref_g = hop_data.zref[fi_j.ext_range.start + mono_ghost_local];
                let popmat_mono_g = mono_states[ps.j].dq[mono_ghost_local] + mono_zref_g;
                let popmat_pair_g = dq[pair_ghost_idx_ct] + ps.zref[pair_ghost_idx_ct];
                ctij[pair_ghost_idx_ct] = popmat_pair_g - popmat_mono_g;
                dq_monomer[pair_ghost_idx_ct] = mono_states[ps.j].dq[mono_ghost_local];
                pair_ghost_idx_ct += 1;
            }
            mono_ghost_j_idx += 1;
        }
    }

    // For healed bonds: subtract ghost monomer POPMAT from BDA's CTIJ.
    // When a bond is healed (both BDA and BAA in pair), the pair has no ghost,
    // but BAA's monomer has a ghost at BDA's position. Its POPMAT must be
    // subtracted for the correct charge transfer delta.
    // (compute_ctmul_hop handles this for CTMUL; we must do the same for CTIJ.)
    for bond in &hop_data.detached_bonds {
        let bda_in_pair = bond.bda_fragment == ps.i || bond.bda_fragment == ps.j;
        let baa_in_pair = bond.baa_fragment == ps.i || bond.baa_fragment == ps.j;
        if !(bda_in_pair && baa_in_pair) {
            continue;
        }

        let baa_frag = bond.baa_fragment;
        let fi_baa = &hop_data.frag_info[baa_frag];
        let n_real_baa = fi_baa.n_real_atoms;
        let ghost_count = hop_data
            .detached_bonds
            .iter()
            .filter(|b| b.baa_fragment == baa_frag)
            .position(|b| b.bda_global == bond.bda_global && b.baa_global == bond.baa_global)
            .expect("ghost not found in BAA monomer");
        let ghost_local_in_mono = n_real_baa + ghost_count;

        let mono_zref_g = hop_data.zref[fi_baa.ext_range.start + ghost_local_in_mono];
        let popmat_mono_g = mono_states[baa_frag].dq[ghost_local_in_mono] + mono_zref_g;

        let bda_pair_local = if bond.bda_fragment == ps.i {
            let bda_frag_start = hop_data.monomer_indices[ps.i][0];
            bond.bda_global - bda_frag_start
        } else {
            let bda_frag_start = hop_data.monomer_indices[ps.j][0];
            n_real_i + (bond.bda_global - bda_frag_start)
        };

        ctij[bda_pair_local] -= popmat_mono_g;
    }

    // For healed bonds: add ghost's dq to BDA's dq_monomer for ESPGRAD.
    // Matches DFTB_ESPGRAD bond-dependent atom handling (dftbfo.src:3762-3829).
    // When a BDA has a ghost in the other pair fragment, the ESPGRAD must use the
    // TOTAL charge at the BDA's position = dq_monomer[BDA] + dq_ghost.
    for bond in &hop_data.detached_bonds {
        let bda_in_pair = bond.bda_fragment == ps.i || bond.bda_fragment == ps.j;
        let ghost_frag_in_pair = bond.baa_fragment == ps.i || bond.baa_fragment == ps.j;
        if !(bda_in_pair && ghost_frag_in_pair) {
            continue;
        }
        // Ghost is at BDA's position in BAA's monomer
        let baa_frag = bond.baa_fragment;
        let fi_baa = &hop_data.frag_info[baa_frag];
        let n_real_baa = fi_baa.n_real_atoms;
        let ghost_count = hop_data
            .detached_bonds
            .iter()
            .filter(|b| b.baa_fragment == baa_frag)
            .position(|b| b.bda_global == bond.bda_global && b.baa_global == bond.baa_global)
            .expect("ghost not found in BAA monomer");
        let ghost_local_in_mono = n_real_baa + ghost_count;
        let dq_ghost = mono_states[baa_frag].dq[ghost_local_in_mono];

        let bda_pair_local = if bond.bda_fragment == ps.i {
            let bda_frag_start = hop_data.monomer_indices[ps.i][0];
            bond.bda_global - bda_frag_start
        } else {
            let bda_frag_start = hop_data.monomer_indices[ps.j][0];
            n_real_i + (bond.bda_global - bda_frag_start)
        };

        dq_monomer[bda_pair_local] += dq_ghost;
    }

    // LC-DFTB: precompute coefficients for overlap and gamma_lr derivative terms
    let (coeff_s_lr, coeff_g_lr) = if let Some(ref gamma_lr_ao) = ps.gamma_lr_ao {
        let diff_p: Array2<f64> = &ps.p - &ps.p_ref;
        let (cs, cg) = compute_lr_coefficients_onthefly(
            diff_p.view(),
            ps.s.view(),
            gamma_lr_ao.view(),
        );
        (Some(cs), Some(cg))
    } else {
        (None, None)
    };

    let orbital_offsets = build_orbital_offsets(ext_atoms);

    let mut grad_local = Array1::<f64>::zeros(3 * n_atoms_pair);
    let mut ctij_grad_global = Array1::<f64>::zeros(3 * n_atoms_total);

    // === Single atom-pair loop ===
    for i in 0..n_atoms_pair {
        let atomi = &ext_atoms[i];
        let mu_start = orbital_offsets[i];

        for j in (i + 1)..n_atoms_pair {
            let atomj = &ext_atoms[j];

            let r_vec = atomi - atomj;
            let dist = r_vec.norm();

            // --- SK integral contributions ---
            if dist < PROXIMITY_CUTOFF {
                let nu_start = orbital_offsets[j];

                let (r, x, y, z) = if atomi <= atomj {
                    directional_cosines(&atomi.xyz, &atomj.xyz)
                } else {
                    directional_cosines(&atomj.xyz, &atomi.xyz)
                };

                let skt = slako.get(atomi.kind, atomj.kind);
                let h_cache = SplineCache::new(r, &skt.h_spline);
                let s_cache = SplineCache::new(r, &skt.s_spline);

                let mut mu = mu_start;
                for orbi in atomi.valorbs.iter() {
                    let mut nu = nu_start;
                    for orbj in atomj.valorbs.iter() {
                        let (dh_i, ds_i, dh_j, ds_j) = if atomi <= atomj {
                            let h_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &h_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                            );
                            let s_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &s_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                            );
                            (
                                [-h_grad[0], -h_grad[1], -h_grad[2]],
                                [-s_grad[0], -s_grad[1], -s_grad[2]],
                                h_grad,
                                s_grad,
                            )
                        } else {
                            let h_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &h_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                            );
                            let s_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &s_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                            );
                            (
                                h_grad,
                                s_grad,
                                [-h_grad[0], -h_grad[1], -h_grad[2]],
                                [-s_grad[0], -s_grad[1], -s_grad[2]],
                            )
                        };

                        let p_mn = p[[mu, nu]];
                        let p_nm = p[[nu, mu]];
                        let wrk_mn = wrk1[[mu, nu]];
                        let wrk_nm = wrk1[[nu, mu]];

                        for dir in 0..3 {
                            let val_i = dh_i[dir] * (p_mn + p_nm) + ds_i[dir] * (wrk_mn + wrk_nm);
                            let val_j = dh_j[dir] * (p_mn + p_nm) + ds_j[dir] * (wrk_mn + wrk_nm);

                            grad_local[3 * i + dir] += val_i;
                            grad_local[3 * j + dir] += val_j;
                        }

                        // LC-DFTB: overlap derivative contribution
                        if let Some(ref coeff_s) = coeff_s_lr {
                            let coeff_mu_nu = coeff_s[[mu, nu]];
                            for dir in 0..3 {
                                grad_local[3 * i + dir] -= 0.0625 * ds_i[dir] * coeff_mu_nu;
                                grad_local[3 * j + dir] -= 0.0625 * ds_j[dir] * coeff_mu_nu;
                            }
                        }

                        nu += 1;
                    }
                    mu += 1;
                }
            }

            // --- Gamma derivative (pair Coulomb) ---
            let dgamma_dr = gammafunction.deriv(dist, atomi.number, atomj.number);
            let coulomb_factor = dq[i] * dq[j] * dgamma_dr / dist;

            for dir in 0..3 {
                grad_local[3 * i + dir] += coulomb_factor * r_vec[dir];
                grad_local[3 * j + dir] -= coulomb_factor * r_vec[dir];
            }

            // --- CTIJ gamma gradient ---
            // CTIJ[a] * dq_mon[c] * dgamma_ac/dR → global coordinates
            let ctij_factor_a = ctij[i] * dq_monomer[j] * dgamma_dr / dist;
            let ctij_factor_b = ctij[j] * dq_monomer[i] * dgamma_dr / dist;

            let global_i = local_to_global[i];
            let global_j = local_to_global[j];

            for dir in 0..3 {
                ctij_grad_global[3 * global_i + dir] -= ctij_factor_a * r_vec[dir];
                ctij_grad_global[3 * global_j + dir] += ctij_factor_a * r_vec[dir];

                ctij_grad_global[3 * global_j + dir] += ctij_factor_b * r_vec[dir];
                ctij_grad_global[3 * global_i + dir] -= ctij_factor_b * r_vec[dir];
            }
        }
    }

    // LC-DFTB: gamma_lr derivative contribution (separate atom-pair loop, no cutoff)
    if let Some(ref coeff_g) = coeff_g_lr {
        let gamma_lc = gammafunction_lc.as_ref().unwrap();
        for i in 0..n_atoms_pair {
            let atomi = &ext_atoms[i];
            let mu_start_i = orbital_offsets[i];
            for j in (i + 1)..n_atoms_pair {
                let atomj = &ext_atoms[j];
                let r_vec = atomi - atomj;
                let dist = r_vec.norm();
                if dist < 1e-10 {
                    continue;
                }
                let nu_start_j = orbital_offsets[j];
                let gamma_lr_deriv = gamma_lc.deriv(dist, atomi.number, atomj.number);

                // Sum coeff_g over orbital block (mu in i, nu in j)
                let mut block_sum = 0.0;
                for mu_off in 0..atomi.n_orbs {
                    for nu_off in 0..atomj.n_orbs {
                        block_sum += coeff_g[[mu_start_i + mu_off, nu_start_j + nu_off]];
                    }
                }

                let val = -0.0625 * block_sum * gamma_lr_deriv / dist;
                for dir in 0..3 {
                    grad_local[3 * i + dir] += val * r_vec[dir];
                    grad_local[3 * j + dir] -= val * r_vec[dir];
                }
            }
        }
    }

    // --- Repulsive gradient with ZREF/QREF scaling ---
    let rep_grad =
        grad_repulsive_energy_scaled(ext_atoms, n_atoms_pair, vrep, ps.zref.view(), ps.qref.view());
    grad_local += &rep_grad;

    (grad_local, ctij_grad_global)
}
