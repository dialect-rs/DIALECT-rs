//! Monomer gradient with HOP for FMO-DFTB.
//!
//! Adapts `monomer_gradient_combined()` from `fmo_gradient.rs` for extended atoms:
//! - Extended atom lists (real + ghost boundary atoms)
//! - ZREF/QREF-scaled repulsive energy gradient
//! - Ghost-ghost and ghost-real SK integral contributions
//! - Extended gamma gradient with ghost charges

use super::helpers::{
    build_orbital_offsets, build_shift_ao_matrix, compute_w_matrix, grad_repulsive_energy_scaled,
};
use crate::defaults::PROXIMITY_CUTOFF;
use crate::fmo::scc_hop::hop_data::HopData;
use crate::fmo::scc_hop::monomer::MonomerHopScc;
use crate::gradients::helpers::compute_lr_coefficients_onthefly;
use crate::initialization::parameters::{RepulsivePotential, SlaterKoster};
use crate::param::slako_transformations::{
    directional_cosines, slako_transformation_gradients_fast, SplineCache,
};
use crate::scc::gamma_approximation::GammaFunction;
use ndarray::prelude::*;

/// Combined monomer gradient with HOP: SCC + addlag + gamma + v_rep.
///
/// Returns `(monomer_grad, addlag_grad)` — both sized `[3 * n_ext_atoms]` (including ghost entries).
///
/// The caller maps these to global coordinates via `local_to_global` and `scatter_to_global`.
pub fn monomer_gradient_combined_hop(
    mono: &MonomerHopScc,
    hop_data: &HopData,
    frag_idx: usize,
    shiftct: ArrayView1<f64>,
    esp_q_ext: ArrayView1<f64>,
    gammafunction: &GammaFunction,
    gammafunction_lc: &Option<GammaFunction>,
    slako: &SlaterKoster,
    vrep: &RepulsivePotential,
) -> (Array1<f64>, Array1<f64>) {
    let ext_atoms = &mono.ext_atoms;
    let n_atoms = mono.n_ext_atoms;
    let n_orbs = mono.n_ext_orbs;
    let fi = &hop_data.frag_info[frag_idx];

    // Get stored SCC properties
    let p: ArrayView2<f64> = mono.p.view();
    let gamma: ArrayView2<f64> = mono.gamma.view();
    let dq: ArrayView1<f64> = mono.dq.view();
    let orbs: ArrayView2<f64> = mono.orbs.as_ref().expect("orbs not stored").view();
    let orbe: ArrayView1<f64> = mono.orbe.as_ref().expect("orbe not stored").view();
    let f: &[f64] = &mono.f;
    let s: ArrayView2<f64> = mono.s.view();
    let nocc = mono.n_elec / 2;

    // === Pre-compute work matrices ===

    // W = energy-weighted density matrix
    let w = compute_w_matrix(orbs, orbe, f, n_orbs, nocc);

    // Total shift for monomer = gamma * dq + esp_q_ext (intra + inter-fragment)
    let intra_shift: Array1<f64> = gamma.dot(&dq);
    let total_shift: Array1<f64> = &intra_shift + &esp_q_ext;

    // Monomer shift in AO basis
    let shift_ao = build_shift_ao_matrix(total_shift.view(), ext_atoms, n_orbs);

    // WRK1_monomer = shift_ao * P - W
    let mut wrk1_monomer = Array2::<f64>::zeros([n_orbs, n_orbs]);
    for mu in 0..n_orbs {
        for nu in 0..n_orbs {
            wrk1_monomer[[mu, nu]] = shift_ao[[mu, nu]] * p[[mu, nu]] - w[[mu, nu]];
        }
    }

    // Addlag: WRK1_addlag = shiftct_ao * P - 0.5 * P * (shiftct_ao * S) * P
    let shiftct_ao = build_shift_ao_matrix(shiftct, ext_atoms, n_orbs);
    let shift_s = &shiftct_ao * &s;
    let d_shift_s = p.dot(&shift_s);
    let d_shift_s_d = d_shift_s.dot(&p);
    let mut wrk1_addlag = Array2::<f64>::zeros([n_orbs, n_orbs]);
    for mu in 0..n_orbs {
        for nu in 0..n_orbs {
            wrk1_addlag[[mu, nu]] =
                shiftct_ao[[mu, nu]] * p[[mu, nu]] - 0.5 * d_shift_s_d[[mu, nu]];
        }
    }

    // LC-DFTB: precompute coefficients for overlap and gamma_lr derivative terms
    let (coeff_s_lr, coeff_g_lr) = if let Some(ref gamma_lr_ao) = mono.gamma_lr_ao {
        let diff_p: Array2<f64> = &mono.p - &mono.p_ref;
        let (cs, cg) = compute_lr_coefficients_onthefly(diff_p.view(), s, gamma_lr_ao.view());
        (Some(cs), Some(cg))
    } else {
        (None, None)
    };

    // Build orbital offsets
    let orbital_offsets = build_orbital_offsets(ext_atoms);

    // Initialize gradients
    let mut grad_monomer = Array1::<f64>::zeros(3 * n_atoms);
    let mut grad_addlag = Array1::<f64>::zeros(3 * n_atoms);

    // === Single atom-pair loop: SK integrals + gamma + v_rep ===
    // Iterates over ALL atom pairs including ghost atoms.
    for i in 0..n_atoms {
        let atomi = &ext_atoms[i];
        let mu_start = orbital_offsets[i];

        for j in (i + 1)..n_atoms {
            let atomj = &ext_atoms[j];

            let r_vec = atomi - atomj;
            let dist = r_vec.norm();

            // --- SK integral contributions (within proximity cutoff) ---
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
                        let wrk_mn = wrk1_monomer[[mu, nu]];
                        let wrk_nm = wrk1_monomer[[nu, mu]];
                        let wrk_add_mn = wrk1_addlag[[mu, nu]];
                        let wrk_add_nm = wrk1_addlag[[nu, mu]];

                        for dir in 0..3 {
                            let dh_i_d = dh_i[dir];
                            let ds_i_d = ds_i[dir];
                            let dh_j_d = dh_j[dir];
                            let ds_j_d = ds_j[dir];

                            let mon_i = dh_i_d * (p_mn + p_nm) + ds_i_d * (wrk_mn + wrk_nm);
                            let mon_j = dh_j_d * (p_mn + p_nm) + ds_j_d * (wrk_mn + wrk_nm);

                            grad_monomer[3 * i + dir] += mon_i;
                            grad_monomer[3 * j + dir] += mon_j;

                            let add_i = ds_i_d * (wrk_add_mn + wrk_add_nm);
                            let add_j = ds_j_d * (wrk_add_mn + wrk_add_nm);

                            grad_addlag[3 * i + dir] += add_i;
                            grad_addlag[3 * j + dir] += add_j;
                        }

                        // LC-DFTB: overlap derivative contribution
                        if let Some(ref coeff_s) = coeff_s_lr {
                            let coeff_mu_nu = coeff_s[[mu, nu]];
                            for dir in 0..3 {
                                grad_monomer[3 * i + dir] -= 0.0625 * ds_i[dir] * coeff_mu_nu;
                                grad_monomer[3 * j + dir] -= 0.0625 * ds_j[dir] * coeff_mu_nu;
                            }
                        }

                        nu += 1;
                    }
                    mu += 1;
                }
            }

            // --- Gamma derivative (all atom pairs including ghosts) ---
            let dgamma_dr = gammafunction.deriv(dist, atomi.number, atomj.number);
            let gamma_factor = dq[i] * dq[j] * dgamma_dr / dist;

            for dir in 0..3 {
                grad_monomer[3 * i + dir] += gamma_factor * r_vec[dir];
                grad_monomer[3 * j + dir] -= gamma_factor * r_vec[dir];
            }
        }
    }

    // LC-DFTB: gamma_lr derivative contribution (separate atom-pair loop, no cutoff)
    if let Some(ref coeff_g) = coeff_g_lr {
        let gamma_lc = gammafunction_lc.as_ref().unwrap();
        for i in 0..n_atoms {
            let atomi = &ext_atoms[i];
            let mu_start_i = orbital_offsets[i];
            for j in (i + 1)..n_atoms {
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
                    grad_monomer[3 * i + dir] += val * r_vec[dir];
                    grad_monomer[3 * j + dir] -= val * r_vec[dir];
                }
            }
        }
    }

    // --- Repulsive gradient with ZREF/QREF scaling ---
    let zref_slice = hop_data.zref.slice(s![fi.ext_range.start..fi.ext_range.end]);
    let qref_slice = hop_data.qref.slice(s![fi.ext_range.start..fi.ext_range.end]);
    let rep_grad = grad_repulsive_energy_scaled(ext_atoms, n_atoms, vrep, zref_slice, qref_slice);
    grad_monomer += &rep_grad;

    (grad_monomer, grad_addlag)
}
