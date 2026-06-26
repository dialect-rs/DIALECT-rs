//! Low-memory, loop-fused FMO-DFTB gradient implementation.
//!
//! The implementation:
//! 1. Computes SK integrals on-the-fly (no [3N,M,M] arrays)
//! 2. Fuses monomer gradient + dftb_addlag + gamma + v_rep into one atom-pair loop
//! 3. Fuses pair gradient + CTIJ + gamma + v_rep into one atom-pair loop
//! 4. Fuses CTMUL + ESD into one inter-fragment loop

use crate::defaults::PROXIMITY_CUTOFF;
use crate::fmo::helpers::get_pair_slice;
use crate::fmo::{ESDPair, Monomer, Pair, SuperSystem};
use crate::gradients::helpers::compute_lr_coefficients_onthefly;
use crate::initialization::Atom;
use crate::param::slako_transformations::{
    directional_cosines, slako_transformation_gradients_fast, SplineCache,
};
use ndarray::prelude::*;
use rayon::prelude::*;

// ============================================================================
// Helper functions
// ============================================================================

/// Build orbital-to-atom mapping.
fn build_orb_to_atom_map(atoms: &[Atom], n_orbs: usize) -> Vec<usize> {
    let mut orb_to_atom = vec![0usize; n_orbs];
    let mut orb_idx = 0;
    for (atom_idx, atom) in atoms.iter().enumerate() {
        for _ in 0..atom.n_orbs {
            orb_to_atom[orb_idx] = atom_idx;
            orb_idx += 1;
        }
    }
    orb_to_atom
}

/// Build orbital offsets: orbital_offsets[i] = first orbital index of atom i.
/// orbital_offsets[n_atoms] = n_orbs.
fn build_orbital_offsets(atoms: &[Atom]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(atoms.len() + 1);
    offsets.push(0);
    for atom in atoms {
        offsets.push(offsets.last().unwrap() + atom.n_orbs);
    }
    offsets
}

/// Build shift matrix in AO basis from atom-based shift vector.
/// SHIFTIJ = 0.5 * (shift[I(mu)] + shift[I(nu)])
fn build_shift_ao_matrix(shift: ArrayView1<f64>, atoms: &[Atom], n_orbs: usize) -> Array2<f64> {
    let orb_to_atom = build_orb_to_atom_map(atoms, n_orbs);
    let mut shift_ao = Array2::<f64>::zeros([n_orbs, n_orbs]);
    for mu in 0..n_orbs {
        let s_mu = shift[orb_to_atom[mu]];
        for nu in 0..n_orbs {
            shift_ao[[mu, nu]] = 0.5 * (s_mu + shift[orb_to_atom[nu]]);
        }
    }
    shift_ao
}

/// Compute energy-weighted density matrix W = sum_k f_k * eps_k * C_k * C_k^T
fn compute_w_matrix(
    orbs: ArrayView2<f64>,
    orbe: ArrayView1<f64>,
    f: &[f64],
    n_orbs: usize,
    nocc: usize,
) -> Array2<f64> {
    let mut w = Array2::<f64>::zeros([n_orbs, n_orbs]);
    for k in 0..nocc {
        let weight = f[k] * orbe[k];
        for mu in 0..n_orbs {
            for nu in 0..=mu {
                let val = weight * orbs[[mu, k]] * orbs[[nu, k]];
                w[[mu, nu]] += val;
                if mu != nu {
                    w[[nu, mu]] += val;
                }
            }
        }
    }
    w
}

/// Compute CTMUL for all atoms: CTMUL[a] = sum over pairs of (dq_pair[a] - dq_mon[a]).
fn compute_ctmul(monomers: &[Monomer], pairs: &[Pair], n_atoms_total: usize) -> Vec<f64> {
    let mut ctmul = vec![0.0f64; n_atoms_total];
    for pair in pairs.iter() {
        let m_i = &monomers[pair.i];
        let m_j = &monomers[pair.j];
        let dq_pair: ArrayView1<f64> = pair.properties.dq().unwrap();
        let dq_i: ArrayView1<f64> = m_i.properties.dq().unwrap();
        let dq_j: ArrayView1<f64> = m_j.properties.dq().unwrap();

        for (local_a, global_a) in m_i.slice.atom_as_range().enumerate() {
            ctmul[global_a] += dq_pair[local_a] - dq_i[local_a];
        }
        for (local_b, global_b) in m_j.slice.atom_as_range().enumerate() {
            let pair_idx = m_i.n_atoms + local_b;
            ctmul[global_b] += dq_pair[pair_idx] - dq_j[local_b];
        }
    }
    ctmul
}

/// Compute SHIFTCT for a monomer using the FMO formula.
/// SHIFTCT[j] = sum over ALL atoms a: gamma(j,a) * CTMUL[a]
/// Compute SHIFTCT for a monomer using the FMO formula.
/// SHIFTCT[j] = sum over ALL atoms a: gamma(j,a) * CTMUL[a]
///
/// The row slice of the precomputed supersystem gamma includes monomer
/// `m`'s own atoms with the analytic on-site diagonal -- same approach
/// as the FMO-xTB implementation (no per-monomer gamma recomputation).
fn compute_shiftct(m: &Monomer, gamma_super: ArrayView2<f64>, ctmul: &[f64]) -> Array1<f64> {
    let gamma_m_all = gamma_super.slice(s![m.slice.atom, ..]);
    gamma_m_all.dot(&ndarray::aview1(ctmul))
}

/// Compute ESPGRAD contribution to SHIFTCT for a monomer.
/// SHIFTCT[j] -= gamma(j,a) * CTIJ[a] for each pair containing this monomer.
/// Compute ESPGRAD contribution to SHIFTCT for a monomer.
///
/// Subtracts the pair self-interaction terms using slices of the
/// precomputed supersystem gamma (incl. the on-site diagonal blocks
/// when `m` is a member of the pair) -- same approach as the FMO-xTB
/// implementation.
fn compute_espgrad_shiftct(
    m: &Monomer,
    gamma_super: ArrayView2<f64>,
    monomers: &[Monomer],
    pairs: &[Pair],
) -> Array1<f64> {
    let n_atoms_m = m.n_atoms;
    let mut espgrad_shiftct = Array1::<f64>::zeros(n_atoms_m);

    for pair in pairs.iter() {
        let (is_mon_i, is_mon_j) = (pair.i == m.index, pair.j == m.index);
        if !is_mon_i && !is_mon_j {
            continue;
        }

        let m_i = &monomers[pair.i];
        let m_j = &monomers[pair.j];

        let dq_pair: ArrayView1<f64> = pair.properties.dq().unwrap();
        let dq_i: ArrayView1<f64> = m_i.properties.dq().unwrap();
        let dq_j: ArrayView1<f64> = m_j.properties.dq().unwrap();

        let n_atoms_i = m_i.n_atoms;

        // CT density difference on the pair, split into the blocks that
        // live on monomer i and on monomer j.
        let ctij_on_i: Array1<f64> = &dq_pair.slice(s![..n_atoms_i]) - &dq_i;
        let ctij_on_j: Array1<f64> = &dq_pair.slice(s![n_atoms_i..]) - &dq_j;

        let gamma_m_mi = gamma_super.slice(s![m.slice.atom, m_i.slice.atom]);
        let gamma_m_mj = gamma_super.slice(s![m.slice.atom, m_j.slice.atom]);

        espgrad_shiftct -= &gamma_m_mi.dot(&ctij_on_i);
        espgrad_shiftct -= &gamma_m_mj.dot(&ctij_on_j);
    }

    espgrad_shiftct
}

// ============================================================================
// Function A: monomer_gradient_combined
// ============================================================================

/// Combined monomer gradient: monomer SCC + dftb_addlag + gamma derivative + v_rep.
///
/// Returns (monomer_grad, addlag_grad) — both computed in the same atom-pair loop.
/// - monomer_grad: dH·P + dS·WRK1_monomer + dgamma/dR·dq·dq + v_rep
/// - addlag_grad: dS·WRK1_addlag
///
/// The caller uses:
/// - monomer_for_total = monomer_grad + addlag_grad
/// - monomer_for_delta_subtraction = monomer_grad (no addlag)
fn monomer_gradient_combined(
    m: &Monomer,
    atoms: &[Atom],
    shiftct: ArrayView1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let m_atoms = &atoms[m.slice.atom_as_range()];
    let n_atoms = m.n_atoms;
    let n_orbs = m.n_orbs;

    // Get properties
    let p: ArrayView2<f64> = m.properties.p().unwrap();
    let gamma: ArrayView2<f64> = m.properties.gamma().unwrap();
    let dq: ArrayView1<f64> = m.properties.dq().unwrap();
    let orbs: ArrayView2<f64> = m.properties.orbs().unwrap();
    let orbe: ArrayView1<f64> = m.properties.orbe().unwrap();
    let f: &[f64] = m.properties.occupation().unwrap();
    let nocc = m.properties.occ_indices().unwrap().len();
    let s: ArrayView2<f64> = m.properties.s().unwrap();
    let esp_q: ArrayView1<f64> = m.properties.esp_q().unwrap();

    // === Pre-compute work matrices (all O(M^2)) ===

    // W = energy-weighted density matrix
    let w = compute_w_matrix(orbs, orbe, f, n_orbs, nocc);

    // Total shift for monomer = gamma * dq + esp_q (intra + inter-fragment)
    let intra_shift: Array1<f64> = gamma.dot(&dq);
    let total_shift: Array1<f64> = &intra_shift + &esp_q;

    // Monomer shift in AO basis
    let shift_ao = build_shift_ao_matrix(total_shift.view(), m_atoms, n_orbs);

    // WRK1_monomer = shift_ao * P - W (element-wise shift_ao * P, minus W)
    let mut wrk1_monomer = Array2::<f64>::zeros([n_orbs, n_orbs]);
    for mu in 0..n_orbs {
        for nu in 0..n_orbs {
            wrk1_monomer[[mu, nu]] = shift_ao[[mu, nu]] * p[[mu, nu]] - w[[mu, nu]];
        }
    }

    // Addlag: WRK1_addlag = shiftct_ao * P - 0.5 * P * (shiftct_ao * S) * P
    let shiftct_ao = build_shift_ao_matrix(shiftct, m_atoms, n_orbs);
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
    let (coeff_s_lr, coeff_g_lr) = if m.gammafunction_lc.is_some() {
        let diff_p: Array2<f64> = &p - &m.properties.p_ref().unwrap();
        let gamma_lr_ao: ArrayView2<f64> = m.properties.gamma_lr_ao().unwrap();
        let (cs, cg) = compute_lr_coefficients_onthefly(diff_p.view(), s, gamma_lr_ao);
        (Some(cs), Some(cg))
    } else {
        (None, None)
    };

    // Build orbital offsets
    let orbital_offsets = build_orbital_offsets(m_atoms);

    // Initialize gradients
    let mut grad_monomer = Array1::<f64>::zeros(3 * n_atoms);
    let mut grad_addlag = Array1::<f64>::zeros(3 * n_atoms);

    // === Single atom-pair loop: SK integrals + gamma + v_rep ===
    for i in 0..n_atoms {
        let atomi = &m_atoms[i];
        let mu_start = orbital_offsets[i];

        for j in (i + 1)..n_atoms {
            let atomj = &m_atoms[j];

            let r_vec = atomi - atomj;
            let dist = r_vec.norm();

            // --- SK integral contributions (within proximity cutoff) ---
            if dist < PROXIMITY_CUTOFF {
                let nu_start = orbital_offsets[j];

                // Directional cosines (ordered by atom type for consistent SK tables)
                let (r, x, y, z) = if atomi <= atomj {
                    directional_cosines(&atomi.xyz, &atomj.xyz)
                } else {
                    directional_cosines(&atomj.xyz, &atomi.xyz)
                };

                // Pre-compute spline caches (once per atom pair)
                let skt = m.slako.get(atomi.kind, atomj.kind);
                let h_cache = SplineCache::new(r, &skt.h_spline);
                let s_cache = SplineCache::new(r, &skt.s_spline);

                // Iterate over orbital pairs
                let mut mu = mu_start;
                for orbi in atomi.valorbs.iter() {
                    let mut nu = nu_start;
                    for orbj in atomj.valorbs.iter() {
                        // Compute dH and dS gradients with proper sign handling
                        let (dh_i, ds_i, dh_j, ds_j) = if atomi <= atomj {
                            let h_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &h_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                            );
                            let s_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &s_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                            );
                            // atomi <= atomj: r points from i to j
                            // dX/dr_i = -dX/dr, dX/dr_j = +dX/dr
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
                            // atomj < atomi: r points from j to i
                            // dX/dr_i = +dX/dr, dX/dr_j = -dX/dr
                            (
                                h_grad,
                                s_grad,
                                [-h_grad[0], -h_grad[1], -h_grad[2]],
                                [-s_grad[0], -s_grad[1], -s_grad[2]],
                            )
                        };

                        // Monomer terms: dH*P + dS*WRK1_monomer (for both (mu,nu) and (nu,mu))
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

                            // Monomer gradient: sum_{mu,nu} (dH*P + dS*WRK1_monomer)
                            let mon_i = dh_i_d * (p_mn + p_nm) + ds_i_d * (wrk_mn + wrk_nm);
                            let mon_j = dh_j_d * (p_mn + p_nm) + ds_j_d * (wrk_mn + wrk_nm);

                            grad_monomer[3 * i + dir] += mon_i;
                            grad_monomer[3 * j + dir] += mon_j;

                            // Addlag gradient: sum_{mu,nu} dS*WRK1_addlag
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

            // --- Gamma derivative ---
            // dgamma/dR_i = (dgamma/dr) * (R_i - R_j) / dist
            // Full sum: 0.5 * sum_{a,b} dgamma * dq_a * dq_b
            // For i<j, both (i,j) and (j,i) contribute: factor = dq_i * dq_j (no 0.5)
            let dgamma_dr = m.gammafunction.deriv(dist, atomi.number, atomj.number);
            let gamma_factor = dq[i] * dq[j] * dgamma_dr / dist;

            for dir in 0..3 {
                grad_monomer[3 * i + dir] += gamma_factor * r_vec[dir];
                grad_monomer[3 * j + dir] -= gamma_factor * r_vec[dir];
            }

            // --- Repulsive potential ---
            let v_ij_deriv = m.vrep.get(atomi.kind, atomj.kind).spline_deriv(dist);
            let rep_factor = v_ij_deriv / dist;

            for dir in 0..3 {
                let val = rep_factor * r_vec[dir];
                grad_monomer[3 * i + dir] += val;
                grad_monomer[3 * j + dir] -= val;
            }
        }
    }

    // LC-DFTB: gamma_lr derivative contribution (separate atom-pair loop, no cutoff)
    if let Some(ref coeff_g) = coeff_g_lr {
        let gamma_lc = m.gammafunction_lc.as_ref().unwrap();
        for i in 0..n_atoms {
            let atomi = &m_atoms[i];
            let mu_start_i = orbital_offsets[i];
            for j in (i + 1)..n_atoms {
                let atomj = &m_atoms[j];
                let r_vec = atomi - atomj;
                let dist = r_vec.norm();
                let nu_start_j = orbital_offsets[j];
                let gamma_lr_deriv = gamma_lc.deriv(dist, atomi.number, atomj.number);
                let mut coeff_sum = 0.0;
                for mu_off in 0..atomi.n_orbs {
                    for nu_off in 0..atomj.n_orbs {
                        coeff_sum += coeff_g[[mu_start_i + mu_off, nu_start_j + nu_off]];
                    }
                }
                let contrib = -0.0625 * gamma_lr_deriv * coeff_sum / dist;
                for dir in 0..3 {
                    grad_monomer[3 * i + dir] += contrib * r_vec[dir];
                    grad_monomer[3 * j + dir] -= contrib * r_vec[dir];
                }
            }
        }
    }

    (grad_monomer, grad_addlag)
}

// ============================================================================
// Function B: pair_gradient_combined
// ============================================================================

/// Combined pair gradient: pair SCC + CTIJ + gamma derivative + v_rep.
///
/// Returns (pair_grad_local, ctij_grad_global).
/// - pair_grad_local: gradient in local pair coordinates [3*n_pair_atoms]
/// - ctij_grad_global: CTIJ contribution in global coordinates [3*n_atoms_total]
fn pair_gradient_combined(
    pair: &Pair,
    pair_atoms: &[Atom],
    m_i: &Monomer,
    m_j: &Monomer,
    atoms: &[Atom],
) -> (Array1<f64>, Array1<f64>) {
    let n_atoms_pair = pair.n_atoms;
    let n_orbs = pair.n_orbs;
    let n_atoms_i = m_i.n_atoms;
    let n_atoms_total = atoms.len();

    // Get properties
    let p: ArrayView2<f64> = pair.properties.p().unwrap();
    let gamma: ArrayView2<f64> = pair.properties.gamma().unwrap();
    let dq: ArrayView1<f64> = pair.properties.dq().unwrap();
    let orbs: ArrayView2<f64> = pair.properties.orbs().unwrap();
    let orbe: ArrayView1<f64> = pair.properties.orbe().unwrap();
    let f: &[f64] = pair.properties.occupation().unwrap();
    let nocc = pair.properties.occ_indices().unwrap().len();

    // Compute total shift = gamma_pair * dq_pair + ESP from K != I,J
    let intra_shift: Array1<f64> = gamma.dot(&dq);
    let gamma_ij: ArrayView2<f64> = gamma.slice(s![0..n_atoms_i, n_atoms_i..]);
    let dq_mon_i: ArrayView1<f64> = m_i.properties.dq().unwrap();
    let dq_mon_j: ArrayView1<f64> = m_j.properties.dq().unwrap();

    let mut esp_from_k: Array1<f64> = Array1::zeros(n_atoms_pair);
    let esp_i: ArrayView1<f64> = m_i.properties.esp_q().unwrap();
    esp_from_k
        .slice_mut(s![0..n_atoms_i])
        .assign(&(&esp_i - &gamma_ij.dot(&dq_mon_j)));
    let esp_j: ArrayView1<f64> = m_j.properties.esp_q().unwrap();
    esp_from_k
        .slice_mut(s![n_atoms_i..])
        .assign(&(&esp_j - &gamma_ij.t().dot(&dq_mon_i)));

    let total_shift: Array1<f64> = &intra_shift + &esp_from_k;

    // Build shift AO matrix and W matrix
    let shift_ao = build_shift_ao_matrix(total_shift.view(), pair_atoms, n_orbs);
    let w = compute_w_matrix(orbs, orbe, f, n_orbs, nocc);

    // WRK1 = shift_ao * P - W
    let mut wrk1 = Array2::<f64>::zeros([n_orbs, n_orbs]);
    for mu in 0..n_orbs {
        for nu in 0..n_orbs {
            wrk1[[mu, nu]] = shift_ao[[mu, nu]] * p[[mu, nu]] - w[[mu, nu]];
        }
    }

    // LC-DFTB: precompute coefficients for overlap and gamma_lr derivative terms
    let (coeff_s_lr, coeff_g_lr) = if pair.gammafunction_lc.is_some() {
        let diff_p: Array2<f64> = &p - &pair.properties.p_ref().unwrap();
        let gamma_lr_ao: ArrayView2<f64> = pair.properties.gamma_lr_ao().unwrap();
        let (cs, cg) = compute_lr_coefficients_onthefly(
            diff_p.view(),
            pair.properties.s().unwrap(),
            gamma_lr_ao,
        );
        (Some(cs), Some(cg))
    } else {
        (None, None)
    };

    // Build CTIJ and monomer charges arrays
    let mut ctij = Array1::<f64>::zeros(n_atoms_pair);
    let mut dq_monomer = Array1::<f64>::zeros(n_atoms_pair);
    for local_a in 0..n_atoms_i {
        ctij[local_a] = dq[local_a] - dq_mon_i[local_a];
        dq_monomer[local_a] = dq_mon_i[local_a];
    }
    for local_b in 0..m_j.n_atoms {
        let pair_idx = n_atoms_i + local_b;
        ctij[pair_idx] = dq[pair_idx] - dq_mon_j[local_b];
        dq_monomer[pair_idx] = dq_mon_j[local_b];
    }

    // Local-to-global atom index mapping
    let local_to_global: Vec<usize> = m_i
        .slice
        .atom_as_range()
        .chain(m_j.slice.atom_as_range())
        .collect();

    let orbital_offsets = build_orbital_offsets(pair_atoms);

    let mut grad_local = Array1::<f64>::zeros(3 * n_atoms_pair);
    let mut ctij_grad_global = Array1::<f64>::zeros(3 * n_atoms_total);

    // === Single atom-pair loop ===
    for i in 0..n_atoms_pair {
        let atomi = &pair_atoms[i];
        let mu_start = orbital_offsets[i];

        for j in (i + 1)..n_atoms_pair {
            let atomj = &pair_atoms[j];

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

                let skt = pair.slako.get(atomi.kind, atomj.kind);
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
            // dgamma/dR_i = (dgamma/dr) * (R_i - R_j) / dist
            let dgamma_dr = pair.gammafunction.deriv(dist, atomi.number, atomj.number);
            let coulomb_factor = dq[i] * dq[j] * dgamma_dr / dist;

            for dir in 0..3 {
                grad_local[3 * i + dir] += coulomb_factor * r_vec[dir];
                grad_local[3 * j + dir] -= coulomb_factor * r_vec[dir];
            }

            // --- CTIJ gamma gradient ---
            // CTIJ[a] * dq_mon[c] * dgamma_ac/dR
            // grad[a] -= factor * (atom_a - atom_c), grad[c] += factor * (atom_a - atom_c)
            let ctij_factor_a = ctij[i] * dq_monomer[j] * dgamma_dr / dist;
            let ctij_factor_b = ctij[j] * dq_monomer[i] * dgamma_dr / dist;

            let global_i = local_to_global[i];
            let global_j = local_to_global[j];

            for dir in 0..3 {
                // CTIJ[i] * dq_mon[j]: a=i, c=j, dx = atom_i - atom_j = r_vec
                ctij_grad_global[3 * global_i + dir] -= ctij_factor_a * r_vec[dir];
                ctij_grad_global[3 * global_j + dir] += ctij_factor_a * r_vec[dir];

                // CTIJ[j] * dq_mon[i]: a=j, c=i, dx = atom_j - atom_i = -r_vec
                ctij_grad_global[3 * global_j + dir] += ctij_factor_b * r_vec[dir];
                ctij_grad_global[3 * global_i + dir] -= ctij_factor_b * r_vec[dir];
            }

            // --- Repulsive potential ---
            let v_ij_deriv = pair.vrep.get(atomi.kind, atomj.kind).spline_deriv(dist);
            let rep_factor = v_ij_deriv / dist;

            for dir in 0..3 {
                let val = rep_factor * r_vec[dir];
                grad_local[3 * i + dir] += val;
                grad_local[3 * j + dir] -= val;
            }
        }
    }

    // LC-DFTB: gamma_lr derivative contribution (separate atom-pair loop, no cutoff)
    if let Some(ref coeff_g) = coeff_g_lr {
        let gamma_lc = pair.gammafunction_lc.as_ref().unwrap();
        let orbital_offsets = build_orbital_offsets(pair_atoms);
        for i in 0..n_atoms_pair {
            let atomi = &pair_atoms[i];
            let mu_start_i = orbital_offsets[i];
            for j in (i + 1)..n_atoms_pair {
                let atomj = &pair_atoms[j];
                let r_vec = atomi - atomj;
                let dist = r_vec.norm();
                let nu_start_j = orbital_offsets[j];
                let gamma_lr_deriv = gamma_lc.deriv(dist, atomi.number, atomj.number);
                let mut coeff_sum = 0.0;
                for mu_off in 0..atomi.n_orbs {
                    for nu_off in 0..atomj.n_orbs {
                        coeff_sum += coeff_g[[mu_start_i + mu_off, nu_start_j + nu_off]];
                    }
                }
                let contrib = -0.0625 * gamma_lr_deriv * coeff_sum / dist;
                for dir in 0..3 {
                    grad_local[3 * i + dir] += contrib * r_vec[dir];
                    grad_local[3 * j + dir] -= contrib * r_vec[dir];
                }
            }
        }
    }

    (grad_local, ctij_grad_global)
}

// ============================================================================
// Function C: interfragment_gradient (CTMUL + ESD fused)
// ============================================================================

/// Combined inter-fragment gradient: CTMUL embedding + ES-dimer.
///
/// For each monomer I, loops over all atoms a and monomer atoms c:
/// - CTMUL contribution: ctmul[a] * dq_I[c] * dgamma_ac/dR
/// - ESD contribution (if a's fragment and I are an ESD pair): dq_J[a] * dq_I[c] * dgamma_ac/dR
fn interfragment_gradient(
    atoms: &[Atom],
    monomers: &[Monomer],
    esd_pairs: &[ESDPair],
    ctmul: &[f64],
) -> Array1<f64> {
    let n_atoms_total = atoms.len();
    let mut gradient = Array1::<f64>::zeros(3 * n_atoms_total);

    // Build atom-to-fragment mapping
    let mut atom_to_frag = vec![0usize; n_atoms_total];
    for m in monomers.iter() {
        for global_idx in m.slice.atom_as_range() {
            atom_to_frag[global_idx] = m.index;
        }
    }

    // Build ESD pair lookup
    use std::collections::HashSet;
    let mut esd_lookup: HashSet<(usize, usize)> = HashSet::new();
    for esd in esd_pairs.iter() {
        esd_lookup.insert((esd.i, esd.j));
        esd_lookup.insert((esd.j, esd.i));
    }

    // For each monomer I
    for m_i in monomers.iter() {
        let dq_i: ArrayView1<f64> = m_i.properties.dq().unwrap();

        // For each atom a (from ALL fragments)
        for global_a in 0..n_atoms_total {
            let atom_a = &atoms[global_a];
            let frag_a = atom_to_frag[global_a];

            // Check if (frag_a, m_i.index) is an ESD pair.
            // Only count ESD when frag_a > m_i.index to avoid double-counting:
            // the explicit gradient counts each ESD pair once, but our monomer loop would
            // count it twice (once when m_i=I with atoms from J, once when m_i=J
            // with atoms from I). We pick frag_a > m_i.index so that the gamma
            // function used is m_i's (matching the explicit gradient which uses the lower-index
            // fragment's gamma).
            let is_esd = frag_a > m_i.index && esd_lookup.contains(&(m_i.index, frag_a));

            // Get monomer charge for atom a (for ESD term)
            let dq_esd_a = if is_esd {
                let m_a = &monomers[frag_a];
                let local_a = global_a - m_a.slice.atom_as_range().start;
                m_a.properties.dq().unwrap()[local_a]
            } else {
                0.0
            };

            let ct_a = ctmul[global_a];

            // Skip if neither CTMUL nor ESD contribute
            if ct_a.abs() < 1e-14 && dq_esd_a.abs() < 1e-14 {
                continue;
            }

            // For each atom c in monomer I
            for (local_c, global_c) in m_i.slice.atom_as_range().enumerate() {
                if global_a == global_c {
                    continue;
                }

                let atom_c = &atoms[global_c];
                let dq_c = dq_i[local_c];

                let dx = atom_a.xyz[0] - atom_c.xyz[0];
                let dy = atom_a.xyz[1] - atom_c.xyz[1];
                let dz = atom_a.xyz[2] - atom_c.xyz[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist < 1e-10 {
                    continue;
                }

                let dgamma_dr = m_i.gammafunction.deriv(dist, atom_a.number, atom_c.number);

                let total_factor = (ct_a + dq_esd_a) * dq_c * dgamma_dr / dist;

                gradient[3 * global_a + 0] += total_factor * dx;
                gradient[3 * global_a + 1] += total_factor * dy;
                gradient[3 * global_a + 2] += total_factor * dz;
                gradient[3 * global_c + 0] -= total_factor * dx;
                gradient[3 * global_c + 1] -= total_factor * dy;
                gradient[3 * global_c + 2] -= total_factor * dz;
            }
        }
    }

    gradient
}

// ============================================================================
// Assembly: ground_state_gradient_fmo
// ============================================================================

impl SuperSystem<'_> {
    /// Compute complete FMO gradient using low-memory, loop-fused approach.
    ///
    /// Total = Monomer + Pair_delta + CTIJ + CTMUL + ESD + Addlag
    pub fn ground_state_gradient_fmo(&mut self) -> Array1<f64> {
        let atoms: &[Atom] = &self.atoms[..];
        let n_atoms_total = atoms.len();
        let n_grad = 3 * n_atoms_total;

        // Step 1: Compute CTMUL for all atoms
        let ctmul = compute_ctmul(&self.monomers, &self.pairs, n_atoms_total);

        // Step 2: Compute SHIFTCT per monomer (needed for addlag) [parallel]
        let monomers_ref = &self.monomers;
        let pairs_ref = &self.pairs;
        let gamma_super: ArrayView2<f64> = self.properties.gamma().unwrap();
        let shiftcts: Vec<Array1<f64>> = self
            .monomers
            .par_iter()
            .map(|m| {
                let espgrad = compute_espgrad_shiftct(m, gamma_super, monomers_ref, pairs_ref);
                let embed = compute_shiftct(m, gamma_super, &ctmul);
                &espgrad + &embed
            })
            .collect();

        // Step 3: Monomer gradients (includes monomer + addlag + gamma + v_rep) [parallel]
        let monomer_results: Vec<(Array1<f64>, Array1<f64>)> = self
            .monomers
            .par_iter()
            .zip(shiftcts.par_iter())
            .map(|(m, shiftct)| monomer_gradient_combined(m, atoms, shiftct.view()))
            .collect();

        // Assemble monomer gradients into global arrays
        let mut monomer_grad_total = Array1::<f64>::zeros(n_grad);
        let mut addlag_total = Array1::<f64>::zeros(n_grad);

        for (m, (mon_grad, add_grad)) in self.monomers.iter().zip(monomer_results.iter()) {
            for (local_idx, global_idx) in m.slice.atom_as_range().enumerate() {
                for k in 0..3 {
                    monomer_grad_total[3 * global_idx + k] += mon_grad[3 * local_idx + k];
                    addlag_total[3 * global_idx + k] += add_grad[3 * local_idx + k];
                }
            }
        }

        // Step 4: Pair gradients (includes pair + CTIJ + gamma + v_rep) [parallel]
        let pair_results: Vec<(Array1<f64>, Array1<f64>)> = self
            .pairs
            .par_iter()
            .map(|pair| {
                let m_i = &self.monomers[pair.i];
                let m_j = &self.monomers[pair.j];
                let pair_atoms: Vec<Atom> =
                    get_pair_slice(atoms, m_i.slice.atom_as_range(), m_j.slice.atom_as_range());
                pair_gradient_combined(pair, &pair_atoms, m_i, m_j, atoms)
            })
            .collect();

        // Step 5: Pair delta + CTIJ accumulation (sequential — small reduction)
        let mut pair_delta_total = Array1::<f64>::zeros(n_grad);
        let mut ctij_total = Array1::<f64>::zeros(n_grad);

        for (pair, (pair_grad_local, ctij_grad_global)) in
            self.pairs.iter().zip(pair_results.iter())
        {
            let m_i = &self.monomers[pair.i];
            let m_j = &self.monomers[pair.j];

            let mon_i_grad = &monomer_results[pair.i].0;
            let mon_j_grad = &monomer_results[pair.j].0;

            // Pair delta = pair_grad_local - monomer_grad_I - monomer_grad_J
            for (local_idx, global_idx) in m_i.slice.atom_as_range().enumerate() {
                for k in 0..3 {
                    pair_delta_total[3 * global_idx + k] +=
                        pair_grad_local[3 * local_idx + k] - mon_i_grad[3 * local_idx + k];
                }
            }
            for (local_idx, global_idx) in m_j.slice.atom_as_range().enumerate() {
                for k in 0..3 {
                    pair_delta_total[3 * global_idx + k] += pair_grad_local
                        [3 * (m_i.n_atoms + local_idx) + k]
                        - mon_j_grad[3 * local_idx + k];
                }
            }

            // CTIJ is already in global coordinates
            ctij_total = &ctij_total + ctij_grad_global;
        }
        // Step 6: Inter-fragment gradient (CTMUL + ESD fused)
        let interfrag_grad = interfragment_gradient(atoms, &self.monomers, &self.esd_pairs, &ctmul);

        // Step 7: Assembly
        // the FMO formula: total = monomer + pair_delta + CTIJ + CTMUL_embed + ESD + addlag
        &monomer_grad_total + &pair_delta_total + &ctij_total + &interfrag_grad + &addlag_total
    }

    // ========================================================================
    // Response gradient: low-memory, on-the-fly implementation
    // ========================================================================

    /// Compute response gradient for a single monomer using on-the-fly SK integrals.
    ///
    /// Replaces `calculate_z_times_b()`, `calculate_z_times_f()`, and
    /// `add_gamma_derivative_contribution()` from the response gradient module with a single
    /// atom-pair loop that computes SK integrals via SplineCache.
    ///
    /// Returns gradient in global coordinates [n_atoms_total, 3].
    fn response_gradient_onthefly(&self, z_vectors: &Vec<Array1<f64>>) -> Array2<f64> {
        let n_atoms_total = self.atoms.len();
        let atoms = &self.atoms[..];

        // Parallel per-monomer gradient computation
        let local_grads: Vec<Array2<f64>> = self
            .monomers
            .par_iter()
            .enumerate()
            .map(|(idx_i, m_i)| {
                let z_i = &z_vectors[idx_i];
                let m_atoms = &atoms[m_i.slice.atom_as_range()];
                let n_atoms = m_i.n_atoms;
                let n_orbs = m_i.n_orbs;
                let nocc: usize = m_i.properties.occ_indices().unwrap().len();
                let nvirt: usize = m_i.properties.virt_indices().unwrap().len();
                let orbs: ArrayView2<f64> = m_i.properties.orbs().unwrap();
                let orbe: ArrayView1<f64> = m_i.properties.orbe().unwrap();
                let s: ArrayView2<f64> = m_i.properties.s().unwrap();
                let gamma: ArrayView2<f64> = m_i.properties.gamma().unwrap();
                let dq: ArrayView1<f64> = m_i.properties.dq().unwrap();
                let esp_q: ArrayView1<f64> = m_i.properties.esp_q().unwrap();

                // === Pre-compute work matrices (all O(M^2)) ===

                // Reshape Z to matrix form [nvirt, nocc]
                let z_mat = z_i.view().into_shape([nvirt, nocc]).unwrap();
                let c_virt = orbs.slice(s![.., nocc..]);
                let c_occ = orbs.slice(s![.., ..nocc]);

                // Z_AO (unsymmetric) for Z×F and Z×B: C_virt * Z * C_occ^T
                let z_ao: Array2<f64> = c_virt.dot(&z_mat.dot(&c_occ.t()));

                // W^Z in AO basis: WZ_ai = Z_ai * eps_i
                let mut wz_mat = Array2::<f64>::zeros([nvirt, nocc]);
                for i_occ in 0..nocc {
                    for a in 0..nvirt {
                        wz_mat[[a, i_occ]] = z_mat[[a, i_occ]] * orbe[i_occ];
                    }
                }
                let mut wz_ao: Array2<f64> = c_virt.dot(&wz_mat.dot(&c_occ.t()));

                // LC-DFTB: add D×F_lc(Z)×D correction to energy-weighted density.
                // This is the "LC response addlag" — the Pulay force correction from
                // the LC exchange Fock perturbation by Z.
                // TRI2 += -0.5 * D * DFTB_LCSHIFT(Z) * D (dftbfo.src:5796-5837)
                if m_i.gammafunction_lc.is_some() {
                    let p_ao: ArrayView2<f64> = m_i.properties.p().unwrap();
                    let gamma_lr_ao: ArrayView2<f64> = m_i.properties.gamma_lr_ao().unwrap();

                    // Symmetrize Z in AO basis (uses triangular Z = symmetric)
                    let z_sym: Array2<f64> = 0.5 * (&z_ao + &z_ao.t());

                    // LCSHIFT formula (DFTB_LCSHIFT2):
                    // lwrk = γ⊙(S·Z·S) + [2·γ⊙(S·Z) + S·(γ⊙Z)]·S
                    let sz: Array2<f64> = s.dot(&z_sym);
                    let szs: Array2<f64> = sz.dot(&s);

                    let term_ab: Array2<f64> = &gamma_lr_ao * &szs; // γ⊙(S·Z·S)

                    let gz: Array2<f64> = &gamma_lr_ao * &z_sym; // γ⊙Z
                    let sgz: Array2<f64> = s.dot(&gz); // S·(γ⊙Z)
                    let gsz: Array2<f64> = &gamma_lr_ao * &sz; // γ⊙(S·Z)
                    let lwrk: Array2<f64> = &term_ab + &(2.0 * &gsz + &sgz).dot(&s);

                    // F_lc(Z) = -0.125 * sym(lwrk)
                    let f_lc_z: Array2<f64> = -0.125 * (&lwrk + &lwrk.t());

                    let dfd: Array2<f64> = p_ao.dot(&f_lc_z).dot(&p_ao);
                    wz_ao = wz_ao + (-0.5) * &dfd;
                }

                // Ground-state shift: shift_A = sum_B gamma_AB * dq_B + esp_q_A
                let shift_gs: Array1<f64> = gamma.dot(&dq) + &esp_q;

                // Build shift in AO basis
                let shift_ao = build_shift_ao_matrix(shift_gs.view(), m_atoms, n_orbs);

                // Combined work matrix for Z×B: shift_ao * z_ao - wz_ao
                let mut wrk_response = Array2::<f64>::zeros([n_orbs, n_orbs]);
                for mu in 0..n_orbs {
                    for nu in 0..n_orbs {
                        wrk_response[[mu, nu]] =
                            shift_ao[[mu, nu]] * z_ao[[mu, nu]] - wz_ao[[mu, nu]];
                    }
                }

                // Z×G (intra): symmetrized Z -> Mulliken charges
                let z_ao_sym: Array2<f64> = 0.5 * (&z_ao + &z_ao.t());
                let zs: Array2<f64> = z_ao_sym.dot(&s);
                let mut q_z = Array1::<f64>::zeros(n_atoms);
                let mut mu_idx = 0;
                for (a_idx, atom) in m_atoms.iter().enumerate() {
                    for _ in 0..atom.n_orbs {
                        q_z[a_idx] += zs[[mu_idx, mu_idx]];
                        mu_idx += 1;
                    }
                }

                // Build orbital offsets
                let orbital_offsets = build_orbital_offsets(m_atoms);

                // LC-DFTB response precomputation
                // f_lc_response: [M,M] matrix for Z×B overlap exchange (fused into SK loop)
                // w_lc_zg: [M,M] matrix for Z×G gamma_lr derivative (separate atom-pair loop)
                let (f_lc_response, w_lc_zg) = if m_i.gammafunction_lc.is_some() {
                    let d_ao: ArrayView2<f64> = m_i.properties.p().unwrap();
                    let d0_ao: ArrayView2<f64> = m_i.properties.p_ref().unwrap();
                    let diff_d: Array2<f64> = &d_ao - &d0_ao;
                    let gamma_lr_ao: ArrayView2<f64> = m_i.properties.gamma_lr_ao().unwrap();

                    // === LC Z×B: compute f_lc_total ===
                    // Following calculate_z_times_b (DFTB_LCGRAD1 MODE=1):
                    // f_lc = (Z·S·γ)·ΔD^T + (Z·S)·(ΔD·γ)^T + Z·(S·ΔD·γ) + (Z·γ)·(S·ΔD)
                    let zs_lr: Array2<f64> = z_ao.dot(&s);
                    let sd_lr: Array2<f64> = s.dot(&diff_d);

                    let zs_gamma: Array2<f64> = &zs_lr * &gamma_lr_ao;
                    let d_gamma: Array2<f64> = &diff_d * &gamma_lr_ao;
                    let sd_gamma: Array2<f64> = &sd_lr * &gamma_lr_ao;
                    let z_gamma: Array2<f64> = &z_ao * &gamma_lr_ao;

                    let f_lc_1 = zs_gamma.dot(&diff_d.t());
                    let f_lc_2 = zs_lr.dot(&d_gamma.t());
                    let f_lc_3 = z_ao.dot(&sd_gamma);
                    let f_lc_4 = z_gamma.dot(&sd_lr);
                    let f_lc_total: Array2<f64> = &f_lc_1 + &f_lc_2 + &f_lc_3 + &f_lc_4;

                    // === LC Z×G: compute W matrix ===
                    // Following calculate_lc_grad2 (DFTB_LCGRAD2):
                    // W[mu,nu] = zs*sd + ds*sz + szs*diff_d + sds*z_ao  (element-wise)
                    let ds_lr: Array2<f64> = diff_d.dot(&s);
                    let sz_lr: Array2<f64> = s.dot(&z_ao);
                    let szs: Array2<f64> = sz_lr.dot(&s); // S @ z_ao @ S
                    let sds: Array2<f64> = sd_lr.dot(&s); // S @ diff_d @ S

                    let w_lc: Array2<f64> =
                        &zs_lr * &sd_lr + &ds_lr * &sz_lr + &szs * &diff_d + &sds * &z_ao;

                    (Some(f_lc_total), Some(w_lc))
                } else {
                    (None, None)
                };

                // Local gradient for this monomer
                let mut grad_local = Array2::<f64>::zeros([n_atoms, 3]);

                // === Single atom-pair loop: Z×F + Z×B + Z×G (intra) ===
                for i in 0..n_atoms {
                    let atomi = &m_atoms[i];
                    let mu_start = orbital_offsets[i];

                    for j in (i + 1)..n_atoms {
                        let atomj = &m_atoms[j];

                        let r_vec = atomi - atomj;
                        let dist = r_vec.norm();

                        // --- SK integral contributions (Z×F + Z×B) ---
                        if dist < PROXIMITY_CUTOFF {
                            let nu_start = orbital_offsets[j];

                            let (r, x, y, z_cos) = if atomi <= atomj {
                                directional_cosines(&atomi.xyz, &atomj.xyz)
                            } else {
                                directional_cosines(&atomj.xyz, &atomi.xyz)
                            };

                            let skt = m_i.slako.get(atomi.kind, atomj.kind);
                            let h_cache = SplineCache::new(r, &skt.h_spline);
                            let s_cache = SplineCache::new(r, &skt.s_spline);

                            let mut mu = mu_start;
                            for orbi in atomi.valorbs.iter() {
                                let mut nu = nu_start;
                                for orbj in atomj.valorbs.iter() {
                                    let (dh_i, ds_i, dh_j, ds_j) = if atomi <= atomj {
                                        let h_grad = slako_transformation_gradients_fast(
                                            r, x, y, z_cos, &h_cache, orbi.l, orbi.m, orbj.l,
                                            orbj.m,
                                        );
                                        let s_grad = slako_transformation_gradients_fast(
                                            r, x, y, z_cos, &s_cache, orbi.l, orbi.m, orbj.l,
                                            orbj.m,
                                        );
                                        (
                                            [-h_grad[0], -h_grad[1], -h_grad[2]],
                                            [-s_grad[0], -s_grad[1], -s_grad[2]],
                                            h_grad,
                                            s_grad,
                                        )
                                    } else {
                                        let h_grad = slako_transformation_gradients_fast(
                                            r, x, y, z_cos, &h_cache, orbj.l, orbj.m, orbi.l,
                                            orbi.m,
                                        );
                                        let s_grad = slako_transformation_gradients_fast(
                                            r, x, y, z_cos, &s_cache, orbj.l, orbj.m, orbi.l,
                                            orbi.m,
                                        );
                                        (
                                            h_grad,
                                            s_grad,
                                            [-h_grad[0], -h_grad[1], -h_grad[2]],
                                            [-s_grad[0], -s_grad[1], -s_grad[2]],
                                        )
                                    };

                                    // Z×F: dH * z_ao (both (mu,nu) and (nu,mu) contributions)
                                    // Z×B: dS * wrk_response (both (mu,nu) and (nu,mu) contributions)
                                    let z_mn = z_ao[[mu, nu]];
                                    let z_nm = z_ao[[nu, mu]];
                                    let w_mn = wrk_response[[mu, nu]];
                                    let w_nm = wrk_response[[nu, mu]];

                                    for dir in 0..3 {
                                        let val_i =
                                            dh_i[dir] * (z_mn + z_nm) + ds_i[dir] * (w_mn + w_nm);
                                        let val_j =
                                            dh_j[dir] * (z_mn + z_nm) + ds_j[dir] * (w_mn + w_nm);

                                        grad_local[[i, dir]] += val_i;
                                        grad_local[[j, dir]] += val_j;
                                    }

                                    // LC-DFTB: overlap exchange response (Z×B_lr)
                                    if let Some(ref f_lc) = f_lc_response {
                                        let f_mn = f_lc[[mu, nu]];
                                        let f_nm = f_lc[[nu, mu]];
                                        let coeff = -0.125 * (f_mn + f_nm);
                                        for dir in 0..3 {
                                            grad_local[[i, dir]] += ds_i[dir] * coeff;
                                            grad_local[[j, dir]] += ds_j[dir] * coeff;
                                        }
                                    }

                                    nu += 1;
                                }
                                mu += 1;
                            }
                        }

                        // --- Z×G (intra): gamma derivative ---
                        let dgamma_dr = m_i.gammafunction.deriv(dist, atomi.number, atomj.number);
                        let factor = (dq[i] * q_z[j] + q_z[i] * dq[j]) * dgamma_dr / dist;

                        for dir in 0..3 {
                            grad_local[[i, dir]] += factor * r_vec[dir];
                            grad_local[[j, dir]] -= factor * r_vec[dir];
                        }
                    }
                }

                // --- LC Z×G: gamma_lr derivative (separate atom-pair loop, no cutoff) ---
                // Following calculate_lc_grad2 (DFTB_LCGRAD2):
                // For each atom pair, sum W over the orbital block and apply dgamma_lr/dr
                if let Some(ref w_lc) = w_lc_zg {
                    let gamma_lc = m_i.gammafunction_lc.as_ref().unwrap();
                    let scal: f64 = -0.125;

                    for i in 0..n_atoms {
                        let atomi = &m_atoms[i];
                        for j in (i + 1)..n_atoms {
                            let atomj = &m_atoms[j];
                            let r_vec = atomi - atomj;
                            let dist = r_vec.norm();
                            let dgamma_lr_dr = gamma_lc.deriv(dist, atomi.number, atomj.number);

                            // Sum W over orbital block (j, i) — matching
                            // calculate_lc_grad2 convention where mu∈iat(=j), nu∈jat(=i)
                            let mu_j = orbital_offsets[j];
                            let nu_i = orbital_offsets[i];
                            let mut tmp = 0.0;
                            for mu_off in 0..m_atoms[j].n_orbs {
                                for nu_off in 0..m_atoms[i].n_orbs {
                                    tmp += w_lc[[mu_j + mu_off, nu_i + nu_off]];
                                }
                            }

                            let val = tmp * dgamma_lr_dr * scal / dist;
                            for dir in 0..3 {
                                grad_local[[i, dir]] += r_vec[dir] * val;
                                grad_local[[j, dir]] -= r_vec[dir] * val;
                            }
                        }
                    }
                }

                grad_local
            })
            .collect();

        // Reduce per-monomer local gradients into global gradient
        let mut gradient = Array2::<f64>::zeros([n_atoms_total, 3]);
        for (m_i, grad_local) in self.monomers.iter().zip(local_grads.iter()) {
            for (local_idx, global_idx) in m_i.slice.atom_as_range().enumerate() {
                for k in 0..3 {
                    gradient[[global_idx, k]] += grad_local[[local_idx, k]];
                }
            }
        }

        // Inter-fragment Z×G contribution (reuse existing on-the-fly implementation)
        let grad_inter = self.calculate_inter_fragment_response_gradient(z_vectors);
        gradient = gradient + grad_inter;

        gradient
    }

    // ========================================================================
    // SCZV solver: matrix-free conjugate gradient
    // ========================================================================

    /// Compute the matrix-vector product A_I · v for a single fragment,
    /// without forming the full A matrix.
    ///
    /// A_I · v = diag(eps_i - eps_a) · v  +  coulomb_coeff · Q_I^T · gamma_I · Q_I · v
    ///
    /// Cost: O(n_atoms * nnum) instead of O(nnum^2) to form A.
    fn orbital_hessian_matvec(
        m_i: &Monomer,
        qvo_i: &Array2<f64>,      // [n_atoms_i, nnum]
        gamma_i: ArrayView2<f64>, // [n_atoms_i, n_atoms_i]
        v: &Array1<f64>,          // [nnum]
    ) -> Array1<f64> {
        let nocc = m_i.properties.occ_indices().unwrap().len();
        let nvirt = m_i.properties.virt_indices().unwrap().len();
        let orbe: ArrayView1<f64> = m_i.properties.orbe().unwrap();

        // Diagonal (one-electron) term: (eps_i - eps_a) * v_ai
        let mut result = Array1::<f64>::zeros(v.len());
        for a in 0..nvirt {
            for i in 0..nocc {
                let idx = a * nocc + i;
                result[idx] = (orbe[i] - orbe[nocc + a]) * v[idx];
            }
        }

        // Coulomb term: -1 * Q^T @ gamma @ Q @ v
        let qv: Array1<f64> = qvo_i.dot(v);
        let g_qv: Array1<f64> = gamma_i.dot(&qv);
        let qt_g_qv: Array1<f64> = qvo_i.t().dot(&g_qv);
        result = result - &qt_g_qv;

        result
    }

    /// Matrix-free SCZV solver using preconditioned conjugate gradient.
    ///
    /// Solves the coupled system:
    ///   A_I · Z_I + sum_{K≠I} A_{K,I}^T · Z_K = L_I   for all fragments I
    ///
    /// Uses Jacobi (diagonal) preconditioner M^{-1} = 1/(eps_i - eps_a)
    /// to reduce iteration count.
    pub fn solve_sczv_cg(
        &self,
        lagrangian: &Vec<Array1<f64>>,
        qvo_vec: &Vec<Array2<f64>>,
    ) -> Vec<Array1<f64>> {
        let maxiter = 500;
        let threshold = 1.0e-8;
        let gamma_full: ArrayView2<f64> = self.properties.gamma().unwrap();
        let n_frag = self.monomers.len();

        // Scale Lagrangian by factor 4 (SCAL=4.0 for RHF closed-shell)
        let rhs: Vec<Array1<f64>> = lagrangian.iter().map(|l| 4.0 * l).collect();

        // Segment sizes for each fragment
        let seg_sizes: Vec<usize> = self
            .monomers
            .iter()
            .map(|m| {
                let nocc = m.properties.occ_indices().unwrap().len();
                let nvirt = m.properties.virt_indices().unwrap().len();
                nvirt * nocc
            })
            .collect();

        // Precompute per-fragment gamma views
        let gammas: Vec<ArrayView2<f64>> = self
            .monomers
            .iter()
            .map(|m| m.properties.gamma().unwrap())
            .collect();

        // Precompute LC exchange intermediates: gamma_lr @ Q products
        // Note: uses m_i.properties.q_oo()/q_vo() for the gamma-contracted side,
        // while qvo_vec (from the response Lagrangian) is used as the
        // other Q_vo factor. These may differ due to MO sign conventions.
        let lc_data: Vec<Option<(Array2<f64>, Array2<f64>)>> =
            if self.config.lc.long_range_correction {
                self.monomers
                    .iter()
                    .map(|m_i| {
                        let gamma_lr: ArrayView2<f64> = m_i.properties.gamma_lr().unwrap();
                        let qoo = m_i.properties.q_oo().unwrap();
                        let qvo_local = m_i.properties.q_vo().unwrap();
                        let g_oo = gamma_lr.dot(&qoo); // [n_atoms, nocc^2]
                        let g_vo = gamma_lr.dot(&qvo_local); // [n_atoms, nvirt*nocc]
                        Some((g_oo, g_vo))
                    })
                    .collect()
            } else {
                vec![None; n_frag]
            };

        // Full matrix-vector product: computes A·z for the coupled system
        let matvec = |z_vecs: &Vec<Array1<f64>>| -> Vec<Array1<f64>> {
            // Step 1: Compute QINDZ_K = Q_K @ Z_K for all fragments
            let qindz: Vec<Array1<f64>> = (0..n_frag).map(|k| qvo_vec[k].dot(&z_vecs[k])).collect();

            // Step 2: For each fragment I, compute A_I·z_I + inter-fragment coupling
            (0..n_frag)
                .into_par_iter()
                .map(|idx_i| {
                    let m_i = &self.monomers[idx_i];

                    // Intra-fragment: A_I · z_I (diagonal + Coulomb)
                    let mut result_i = Self::orbital_hessian_matvec(
                        m_i,
                        &qvo_vec[idx_i],
                        gammas[idx_i],
                        &z_vecs[idx_i],
                    );

                    // LC exchange contribution to intra-fragment A_I · z_I
                    if let Some((ref g_oo, ref g_vo)) = lc_data[idx_i] {
                        let nocc = m_i.properties.occ_indices().unwrap().len();
                        let nvirt = m_i.properties.virt_indices().unwrap().len();
                        let n_atoms_i = m_i.n_atoms;
                        let qvv = m_i.properties.q_vv().unwrap();
                        let v_mat = z_vecs[idx_i].view().into_shape((nvirt, nocc)).unwrap();

                        let mut exc_result = Array2::<f64>::zeros((nvirt, nocc));

                        for a in 0..n_atoms_i {
                            // Term 1: (ab|ij) exchange
                            let g_oo_a = g_oo.slice(s![a, ..]).into_shape((nocc, nocc)).unwrap();
                            let qvv_a = qvv.slice(s![a, ..]).into_shape((nvirt, nvirt)).unwrap();
                            let w1 = v_mat.dot(&g_oo_a.t());
                            exc_result += &qvv_a.dot(&w1);

                            // Term 2: (aj|bi) exchange
                            let g_vo_a = g_vo.slice(s![a, ..]).into_shape((nvirt, nocc)).unwrap();
                            let qvo_a = qvo_vec[idx_i]
                                .slice(s![a, ..])
                                .into_shape((nvirt, nocc))
                                .unwrap();
                            let w2 = g_vo_a.t().dot(&v_mat);
                            exc_result += &qvo_a.dot(&w2.t());
                        }

                        let exc_flat = exc_result.into_shape(nvirt * nocc).unwrap();
                        result_i += &exc_flat;
                    }

                    // Inter-fragment: sum_{K≠I} (-1) * Q_I^T @ gamma_IK @ QINDZ_K
                    let mut shift_i = Array1::<f64>::zeros(m_i.n_atoms);
                    for idx_k in 0..n_frag {
                        if idx_k != idx_i {
                            let m_k = &self.monomers[idx_k];
                            let gamma_ik = gamma_full.slice(s![m_i.slice.atom, m_k.slice.atom]);
                            shift_i += &gamma_ik.dot(&qindz[idx_k]);
                        }
                    }
                    result_i -= &qvo_vec[idx_i].t().dot(&shift_i);

                    result_i
                })
                .collect()
        };

        // Jacobi preconditioner: M^{-1} = 1/(eps_i - eps_a)
        let inv_diag: Vec<Array1<f64>> = self
            .monomers
            .iter()
            .map(|m| {
                let nocc = m.properties.occ_indices().unwrap().len();
                let nvirt = m.properties.virt_indices().unwrap().len();
                let orbe: ArrayView1<f64> = m.properties.orbe().unwrap();
                let mut inv_d = Array1::<f64>::zeros(nvirt * nocc);
                for a in 0..nvirt {
                    for i in 0..nocc {
                        inv_d[a * nocc + i] = 1.0 / (orbe[i] - orbe[nocc + a]);
                    }
                }
                inv_d
            })
            .collect();

        // Apply preconditioner: z_i = M^{-1} @ r_i (element-wise)
        let precond = |r_vecs: &Vec<Array1<f64>>| -> Vec<Array1<f64>> {
            r_vecs
                .iter()
                .zip(inv_diag.iter())
                .map(|(ri, inv_d)| ri * inv_d)
                .collect()
        };

        // Dot product over all fragments
        let dot_all = |a: &Vec<Array1<f64>>, b: &Vec<Array1<f64>>| -> f64 {
            a.iter().zip(b.iter()).map(|(ai, bi)| ai.dot(bi)).sum()
        };

        // Vector operations over all fragments
        let vec_sub = |a: &Vec<Array1<f64>>, b: &Vec<Array1<f64>>| -> Vec<Array1<f64>> {
            a.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect()
        };
        let vec_add = |a: &Vec<Array1<f64>>, b: &Vec<Array1<f64>>| -> Vec<Array1<f64>> {
            a.iter().zip(b.iter()).map(|(ai, bi)| ai + bi).collect()
        };
        let vec_scale = |s: f64, a: &Vec<Array1<f64>>| -> Vec<Array1<f64>> {
            a.iter().map(|ai| s * ai).collect()
        };

        // Initialize with zero Z-vectors
        let mut z_vecs: Vec<Array1<f64>> = seg_sizes.iter().map(|&n| Array1::zeros(n)).collect();

        // Short-circuit: if RHS (Lagrangian) is identically zero, the Z-vectors
        // are also zero. This avoids 0/0 = NaN in the CG iteration when there
        // are no close pairs (e.g. dilute clusters).
        let rhs_norm = dot_all(&rhs, &rhs).sqrt();
        if rhs_norm == 0.0 {
            return z_vecs;
        }

        // Preconditioned CG: r = b - A·x, z = M^{-1}·r, p = z
        let az = matvec(&z_vecs);
        let mut r = vec_sub(&rhs, &az);
        let mut z_pre = precond(&r);
        let mut p = z_pre.clone();
        let mut rz = dot_all(&r, &z_pre); // r^T · M^{-1} · r

        let abs_threshold = threshold * rhs_norm;

        for _iter in 0..maxiter {
            let residual = dot_all(&r, &r).sqrt();
            if residual < abs_threshold {
                // eprintln!(
                //     "  SCZV-CG converged after {} iterations (residual = {:.2e})",
                //     iter, residual
                // );
                break;
            }

            let ap = matvec(&p);
            let pap = dot_all(&p, &ap);
            let alpha = rz / pap;

            z_vecs = vec_add(&z_vecs, &vec_scale(alpha, &p));
            r = vec_sub(&r, &vec_scale(alpha, &ap));

            z_pre = precond(&r);
            let rz_new = dot_all(&r, &z_pre);
            let beta = rz_new / rz;
            rz = rz_new;

            p = vec_add(&z_pre, &vec_scale(beta, &p));
        }

        z_vecs
    }

    /// Calculate response Lagrangian L^K according to Eq. (15) in Nishimoto 2015.
    /// L^K_ai = sum_{IJ} gamma_{K,IJ} * ΔΔq^{IJ} * Q^K_ai
    ///
    /// This builds the Lagrangian from dimer-monomer charge differences.
    pub fn calculate_response_lagrangian_new(&self) -> (Vec<Array1<f64>>, Vec<Array2<f64>>) {
        // Build Q_vo (virtual-occupied transition charges) for each monomer
        let mut qvo_vector: Vec<Array2<f64>> = Vec::with_capacity(self.monomers.len());

        for (_idx_i, m_i) in self.monomers.iter().enumerate() {
            let orbs: ArrayView2<f64> = m_i.properties.orbs().unwrap();
            let s: ArrayView2<f64> = m_i.properties.s().unwrap();
            let nocc: usize = m_i.properties.occ_indices().unwrap().len();
            let nvirt: usize = m_i.properties.virt_indices().unwrap().len();
            let m_i_atoms: &[Atom] = &self.atoms[m_i.slice.atom_as_range()];

            // S dot C: S_mu,nu * C_nu,i -> dimension [n_orbs, n_mo]
            let s_c: Array2<f64> = s.dot(&orbs);
            let mut mu: usize = 0;

            // Q_ai^A = sum_mu in A [ C_mu,a * (SC)_mu,i + C_mu,i * (SC)_mu,a ]
            let mut qvo: Array2<f64> = Array2::zeros([m_i.n_atoms, nvirt * nocc]);

            for (a_idx, atom) in m_i_atoms.iter().enumerate() {
                let end: usize = mu + atom.n_orbs;

                // Virtual coefficients on atom A
                let orbs_mu_virt = orbs.slice(s![mu..end, nocc..]);
                // SC for occupied on atom A
                let s_c_mu_occ = s_c.slice(s![mu..end, ..nocc]);
                // Term 1: C_mu,a * (SC)_mu,i
                let term1: Array2<f64> = orbs_mu_virt.t().dot(&s_c_mu_occ);

                // Occupied coefficients on atom A
                let orbs_mu_occ = orbs.slice(s![mu..end, ..nocc]);
                // SC for virtual on atom A
                let s_c_mu_virt = s_c.slice(s![mu..end, nocc..]);
                // Term 2: C_mu,i * (SC)_mu,a
                let term_2: Array2<f64> = orbs_mu_occ.t().dot(&s_c_mu_virt);

                let sum: Array2<f64> = term1 + term_2.t();
                let sum_1d: Array1<f64> = sum.into_shape([nvirt * nocc]).unwrap();

                qvo.slice_mut(s![a_idx, ..]).assign(&sum_1d);
                mu += atom.n_orbs;
            }
            qvo_vector.push(qvo);
        }

        // Calculate Lagrangian for each monomer K
        let mut lagrangian_vector: Vec<Array1<f64>> = Vec::with_capacity(self.monomers.len());
        let gamma_full: ArrayView2<f64> = self.properties.gamma().unwrap();

        for (idx_k, m_k) in self.monomers.iter().enumerate() {
            let gamma_slice: ArrayView2<f64> = gamma_full.slice(s![m_k.slice.atom, ..]);
            let qvo_k: ArrayView2<f64> = qvo_vector[idx_k].view();
            let nocc: usize = m_k.properties.occ_indices().unwrap().len();
            let nvirt: usize = m_k.properties.virt_indices().unwrap().len();

            // Sum over all pairs IJ where K is not involved
            let lagrangian_k: Array1<f64> = self
                .pairs
                .par_iter()
                .map(|pair| {
                    let m_i: &Monomer = &self.monomers[pair.i];
                    let m_j: &Monomer = &self.monomers[pair.j];

                    if m_i.index != m_k.index && m_j.index != m_k.index {
                        // Build gamma array for atoms in I and J
                        let mut g_arr: Array2<f64> =
                            Array2::zeros([m_k.n_atoms, m_i.n_atoms + m_j.n_atoms]);
                        g_arr
                            .slice_mut(s![.., ..m_i.n_atoms])
                            .assign(&gamma_slice.slice(s![.., m_i.slice.atom]));
                        g_arr
                            .slice_mut(s![.., m_i.n_atoms..])
                            .assign(&gamma_slice.slice(s![.., m_j.slice.atom]));

                        // ΔΔq^IJ = q^IJ - q^I - q^J (dimer - monomer charges)
                        let ddq: ArrayView1<f64> = pair.properties.delta_dq().unwrap();

                        // ESP on K from charge differences in IJ
                        // DFTB_ESPGRAD line 3638: SHIFTCT(J) -= gamma * CTIJ
                        // The negative sign is critical - SHIFTCT accumulates negatively
                        let esp: Array1<f64> = g_arr.dot(&ddq);

                        // L^K_ai = 0.5 * sum_A in K SHIFTCT(A) * Q^K_ai,A
                        // Factor 0.5 matches DFTB_ZVLAGGET (line 4650)
                        // Negative sign matches ESPGRAD accumulation
                        -0.5 * esp.dot(&qvo_k)
                    } else {
                        Array1::zeros(nvirt * nocc)
                    }
                })
                .reduce(|| Array1::zeros(nvirt * nocc), |a, b| a + b);

            lagrangian_vector.push(lagrangian_k);
        }

        (lagrangian_vector, qvo_vector)
    }

    /// Calculate inter-fragment response gradient contribution.
    /// Corresponds to DFTB_ZVEC_KFG subroutine.
    ///
    /// For each monomer I, compute Q^Z (Mulliken charges from Z-vector), then
    /// calculate gradient from gamma derivatives with charges in all other fragments:
    /// G^a += sum_{J!=I} Q^Z_I * dq_J * dgamma_IJ/dR
    pub fn calculate_inter_fragment_response_gradient(
        &self,
        z_vectors: &Vec<Array1<f64>>,
    ) -> Array2<f64> {
        let n_atoms_total = self.atoms.len();
        let mut gradient = Array2::zeros([n_atoms_total, 3]);

        // Note: gamma_full could be used to avoid recomputing gamma values,
        // but for now we use the monomer's gammafunction to compute dgamma/dr directly
        let _gamma_full: ArrayView2<f64> = self.properties.gamma().unwrap();

        // First, compute Q^Z (Mulliken charges from Z-vector) for each monomer
        let mut q_z_all: Vec<Array1<f64>> = Vec::with_capacity(self.monomers.len());

        for (idx_i, m_i) in self.monomers.iter().enumerate() {
            let z_i = &z_vectors[idx_i];
            let nocc: usize = m_i.properties.occ_indices().unwrap().len();
            let nvirt: usize = m_i.properties.virt_indices().unwrap().len();
            let orbs: ArrayView2<f64> = m_i.properties.orbs().unwrap();
            let s: ArrayView2<f64> = m_i.properties.s().unwrap();
            let atoms: &[Atom] = &self.atoms[m_i.slice.atom_as_range()];

            // Reshape Z to matrix form
            let z_mat = z_i.view().into_shape([nvirt, nocc]).unwrap().to_owned();

            // Construct Z in AO basis and symmetrize (matching TRI1)
            let c_virt = orbs.slice(s![.., nocc..]);
            let c_occ = orbs.slice(s![.., ..nocc]);
            let z_ao_unsym: Array2<f64> = c_virt.dot(&z_mat.dot(&c_occ.t()));
            let z_ao_sym: Array2<f64> = 0.5 * (&z_ao_unsym + &z_ao_unsym.t());

            // Calculate Mulliken charges from Z: Q^Z_A = sum_{μ in A} (Z_sym * S)_{μμ}
            // Following DFTB_ZVEC_KFG lines 5828-5832
            let zs: Array2<f64> = z_ao_sym.dot(&s);
            let mut q_z: Array1<f64> = Array1::zeros(m_i.n_atoms);
            let mut mu = 0;
            for (a_idx, atom) in atoms.iter().enumerate() {
                for _ in 0..atom.n_orbs {
                    q_z[a_idx] += zs[[mu, mu]];
                    mu += 1;
                }
            }
            q_z_all.push(q_z);
        }

        // Now compute inter-fragment gamma derivative contribution
        // For each pair of monomers (I, J), compute:
        // G^a += Q^Z_I * dq_J * dgamma_IJ/dR
        for (idx_i, m_i) in self.monomers.iter().enumerate() {
            let q_z_i: &Array1<f64> = &q_z_all[idx_i];
            let atoms_i: &[Atom] = &self.atoms[m_i.slice.atom_as_range()];

            for (idx_j, m_j) in self.monomers.iter().enumerate() {
                if idx_i == idx_j {
                    continue; // Skip self-interaction
                }

                let dq_j: ArrayView1<f64> = m_j.properties.dq().unwrap();
                let atoms_j: &[Atom] = &self.atoms[m_j.slice.atom_as_range()];

                // Compute gamma derivative contribution for all atom pairs (I, J)
                for (local_i, (global_i, atom_i)) in
                    m_i.slice.atom_as_range().zip(atoms_i.iter()).enumerate()
                {
                    for (local_j, (global_j, atom_j)) in
                        m_j.slice.atom_as_range().zip(atoms_j.iter()).enumerate()
                    {
                        // Distance vector from atom_j to atom_i (following the reference convention)
                        // VEC(K) = FMOC(K,IAG) - FMOC(K,JAG) = r_I - r_J
                        let dx = atom_i.xyz[0] - atom_j.xyz[0];
                        let dy = atom_i.xyz[1] - atom_j.xyz[1];
                        let dz = atom_i.xyz[2] - atom_j.xyz[2];
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                        if dist < 1e-10 {
                            continue; // Skip if atoms are at the same position
                        }

                        // Compute dgamma/dR using the same gamma function as monomer
                        // For now, use a finite-difference approximation of gamma derivative
                        // dgamma/dR ≈ dgamma/dr * r_vec / r
                        let dgamma_dr =
                            self.compute_dgamma_dr(atom_i, atom_j, dist, &m_i.gammafunction);

                        // Contribution: Q^Z_I * dq_J * dgamma/dR
                        let factor = q_z_i[local_i] * dq_j[local_j] * dgamma_dr / dist;

                        // Apply to gradient (Newton's third law)
                        gradient[[global_i, 0]] += factor * dx;
                        gradient[[global_i, 1]] += factor * dy;
                        gradient[[global_i, 2]] += factor * dz;
                        gradient[[global_j, 0]] -= factor * dx;
                        gradient[[global_j, 1]] -= factor * dy;
                        gradient[[global_j, 2]] -= factor * dz;
                    }
                }
            }
        }

        gradient
    }

    /// Compute dgamma/dr for a pair of atoms.
    /// This is the radial derivative of the gamma function.
    fn compute_dgamma_dr(
        &self,
        atom_i: &Atom,
        atom_j: &Atom,
        dist: f64,
        gammafunction: &crate::scc::gamma_approximation::GammaFunction,
    ) -> f64 {
        // Use the gamma function's derivative
        // gamma(r) depends on the Hubbard parameters of both atoms
        // For standard DFTB, gamma = 1/r - S(r, tau_i, tau_j) where S is the short-range correction
        // dgamma/dr = -1/r^2 - dS/dr

        // Get the derivative using the gamma function's deriv method
        // deriv(r, z_a, z_b) where z_a, z_b are atomic numbers
        gammafunction.deriv(dist, atom_i.number, atom_j.number)
    }

    /// Compute complete FMO response gradient using low-memory approach.
    ///
    /// Steps:
    /// 1. Build Lagrangian
    /// 2. Solve Z-vectors via matrix-free CG
    /// 3. On-the-fly gradient (fused SK loop + gamma derivative)
    /// 4. Apply -1.0 sign convention fix
    pub fn response_gradient_fmo(&mut self) -> Array1<f64> {
        let n_atoms_total = self.atoms.len();
        let n_grad = 3 * n_atoms_total;

        let (lagrangian_vec, qvo_vec) = self.calculate_response_lagrangian_new();

        let z_vectors = self.solve_sczv_cg(&lagrangian_vec, &qvo_vec);

        let mut gradient_2d = self.response_gradient_onthefly(&z_vectors);

        // Response addlag: dS/dR * (shift_z * P - 0.5 * P * (shift_z * S) * P)
        // where shift_z = gamma * Q_Z (Z-induced shift from all fragments).
        // This is DFTB_ZVEC_KGRAD.
        self.add_response_addlag(&z_vectors, &mut gradient_2d);

        // Apply -1.0 sign convention fix (MO sign difference vs the reference)
        let gradient_2d = -1.0 * gradient_2d;

        gradient_2d
            .into_shape([n_grad])
            .expect("Failed to reshape response gradient")
    }

    /// Response addlag: gradient from Z-vector-induced shift acting on density.
    ///
    /// Computes SHIFTZ = gamma * Q_Z (intra + inter-fragment), then evaluates
    /// dS/dR * (shift_z_ao * P - 0.5 * P * (shift_z_ao * S) * P).
    /// This is the DFTB_ZVEC_KGRAD contribution.
    fn add_response_addlag(
        &self,
        z_vectors: &[Array1<f64>],
        gradient: &mut Array2<f64>,
    ) {
        let atoms = &self.atoms[..];

        // Step 1: Compute Q_Z for all fragments
        let mut q_z_all: Vec<Array1<f64>> = Vec::with_capacity(self.monomers.len());
        for (idx_i, m_i) in self.monomers.iter().enumerate() {
            let z_i = &z_vectors[idx_i];
            let nocc = m_i.properties.occ_indices().unwrap().len();
            let nvirt = m_i.properties.virt_indices().unwrap().len();
            let orbs: ArrayView2<f64> = m_i.properties.orbs().unwrap();
            let s: ArrayView2<f64> = m_i.properties.s().unwrap();
            let m_atoms = &atoms[m_i.slice.atom_as_range()];

            let z_mat = z_i.view().into_shape([nvirt, nocc]).unwrap();
            let c_virt = orbs.slice(s![.., nocc..]);
            let c_occ = orbs.slice(s![.., ..nocc]);
            let z_ao_unsym = c_virt.dot(&z_mat.dot(&c_occ.t()));
            let z_ao_sym = 0.5 * (&z_ao_unsym + &z_ao_unsym.t());
            let zs = z_ao_sym.dot(&s);

            let mut q_z = Array1::<f64>::zeros(m_i.n_atoms);
            let mut mu = 0;
            for (a_idx, atom) in m_atoms.iter().enumerate() {
                for _ in 0..atom.n_orbs {
                    q_z[a_idx] += zs[[mu, mu]];
                    mu += 1;
                }
            }
            q_z_all.push(q_z);
        }

        // Step 2: Compute SHIFTZ for each fragment
        // SHIFTZ[a] = sum over ALL atoms b: gamma(a,b) * Q_Z[b]
        let mut shiftz_all: Vec<Array1<f64>> = Vec::with_capacity(self.monomers.len());
        for (frag_k, m_k) in self.monomers.iter().enumerate() {
            let gamma_k: ArrayView2<f64> = m_k.properties.gamma().unwrap();
            let atoms_k = &atoms[m_k.slice.atom_as_range()];

            // Intra: gamma_local * Q_Z_local
            let mut shiftz = gamma_k.dot(&q_z_all[frag_k]);

            // Inter: sum over atoms in other fragments
            for (frag_j, m_j) in self.monomers.iter().enumerate() {
                if frag_j == frag_k {
                    continue;
                }
                let atoms_j = &atoms[m_j.slice.atom_as_range()];
                let q_z_j = &q_z_all[frag_j];

                for (a, atom_a) in atoms_k.iter().enumerate() {
                    for (b, atom_b) in atoms_j.iter().enumerate() {
                        let dx = atom_a.xyz[0] - atom_b.xyz[0];
                        let dy = atom_a.xyz[1] - atom_b.xyz[1];
                        let dz = atom_a.xyz[2] - atom_b.xyz[2];
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                        let gamma_ab =
                            m_k.gammafunction.eval(dist, atom_a.number, atom_b.number);
                        shiftz[a] += gamma_ab * q_z_j[b];
                    }
                }
            }
            shiftz_all.push(shiftz);
        }

        // Step 3: For each fragment, compute addlag gradient
        for (frag_idx, m_i) in self.monomers.iter().enumerate() {
            let m_atoms = &atoms[m_i.slice.atom_as_range()];
            let n_atoms = m_i.n_atoms;
            let n_orbs = m_i.n_orbs;
            let p: ArrayView2<f64> = m_i.properties.p().unwrap();
            let s: ArrayView2<f64> = m_i.properties.s().unwrap();
            let shiftz = &shiftz_all[frag_idx];

            // Build shift_z_ao[mu,nu] = 0.5 * (shiftz[atom_mu] + shiftz[atom_nu])
            let shift_z_ao = build_shift_ao_matrix(shiftz.view(), m_atoms, n_orbs);

            // wrk1 = shift_z_ao * P - 0.5 * P * (shift_z_ao * S) * P
            let shift_s = &shift_z_ao * &s;
            let d_shift_s_d = p.dot(&shift_s).dot(&p);

            let mut wrk1 = Array2::<f64>::zeros([n_orbs, n_orbs]);
            for mu in 0..n_orbs {
                for nu in 0..n_orbs {
                    wrk1[[mu, nu]] =
                        shift_z_ao[[mu, nu]] * p[[mu, nu]] - 0.5 * d_shift_s_d[[mu, nu]];
                }
            }

            // SK loop for dS/dR * wrk1
            let orbital_offsets = build_orbital_offsets(m_atoms);

            for i_atom in 0..n_atoms {
                let atomi = &m_atoms[i_atom];
                let mu_start = orbital_offsets[i_atom];
                let global_i = m_i.slice.atom_as_range().start + i_atom;

                for j_atom in (i_atom + 1)..n_atoms {
                    let atomj = &m_atoms[j_atom];
                    let r_vec = atomi - atomj;
                    let dist = r_vec.norm();

                    if dist >= PROXIMITY_CUTOFF {
                        continue;
                    }

                    let nu_start = orbital_offsets[j_atom];
                    let global_j = m_i.slice.atom_as_range().start + j_atom;

                    let (r, x, y, z) = if atomi <= atomj {
                        directional_cosines(&atomi.xyz, &atomj.xyz)
                    } else {
                        directional_cosines(&atomj.xyz, &atomi.xyz)
                    };

                    let skt = m_i.slako.get(atomi.kind, atomj.kind);
                    let s_cache = SplineCache::new(r, &skt.s_spline);

                    let mut mu = mu_start;
                    for orbi in atomi.valorbs.iter() {
                        let mut nu = nu_start;
                        for orbj in atomj.valorbs.iter() {
                            let (ds_i, ds_j) = if atomi <= atomj {
                                let s_grad = slako_transformation_gradients_fast(
                                    r, x, y, z, &s_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                                );
                                ([-s_grad[0], -s_grad[1], -s_grad[2]], s_grad)
                            } else {
                                let s_grad = slako_transformation_gradients_fast(
                                    r, x, y, z, &s_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                                );
                                (s_grad, [-s_grad[0], -s_grad[1], -s_grad[2]])
                            };

                            let wrk_mn = wrk1[[mu, nu]];
                            let wrk_nm = wrk1[[nu, mu]];

                            for dir in 0..3 {
                                gradient[[global_i, dir]] += ds_i[dir] * (wrk_mn + wrk_nm);
                                gradient[[global_j, dir]] += ds_j[dir] * (wrk_mn + wrk_nm);
                            }

                            nu += 1;
                        }
                        mu += 1;
                    }
                }
            }
        }
    }
}
