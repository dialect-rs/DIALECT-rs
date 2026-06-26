//! Response gradient for FMO-xTB HOP.
//!
//! Implements Z-vector (SCZV) response gradient reading from new HOP SCC structs.
//! Sub-functions: Lagrangian, SCZV-CG solver, on-the-fly gradient, response addlag,
//! inter-fragment response, HOP response.

use super::helpers::{
    build_monomer_local_to_global, build_pair_ctij_full, compute_esp_q_shell_hop,
    compute_occ_virt_from_f,
};
use dialect_base::defaults::PROXIMITY_CUTOFF;
use dialect_utilities::mulliken::shell_to_ao_values;
use crate::fmo::gradients::fmo_gradient_shell::{
    get_pi_term_gradient_inline_shell, get_self_energy_cn_grad_coeff_shell,
};
use crate::fmo::scc_hop::hop_data::{
    compute_bda_dd_xtb, compute_rotated_sp3_xtb, get_frag_shell_range, XtbHopData,
};
use crate::fmo::scc_hop::monomer::XtbMonomerHopScc;
use crate::fmo::scc_hop::pair::XtbPairHopScc;
use crate::fmo::scc_hop::trimer::XtbTrimerHopScc;
use crate::gradients::ground_state::aovec_to_aomat;
use crate::gradients::hop_gradient::hop_gradient_single_bond_general;
use crate::hop::HOP_SHIFT;
use crate::initialization::atom::XtbAtom;
use crate::integrals::{calc_overlap_derivative_d_shells, obara_saika_derivatives_all};
use crate::parameters::*;
use crate::scc::gamma_matrix::XtbGammaFunction;
use crate::scc::hamiltonian::{
    calculate_pair_scaling_param, get_hueckel_constants_new, get_pi_term,
    get_self_energy_values_new,
};
use nalgebra::Vector3;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::ops::AddAssign;

// ============================================================================
// QVO computation from new structs
// ============================================================================

/// Compute Q_VO (transition charges) at AO and shell level for a monomer HOP SCC state.
///
/// Q_VO_{mu,ai} = C_virt_mu,a * (S @ C_occ)_mu,i + C_occ_mu,i * (S @ C_virt)_mu,a
fn compute_qvo_shell_hop(
    mono: &XtbMonomerHopScc,
) -> (Array2<f64>, Array2<f64>) {
    let (occ, virt) = compute_occ_virt_from_f(&mono.f);
    let nocc = occ.len();
    let nvirt = virt.len();
    let nvo = nvirt * nocc;
    let n_orbs = mono.n_ext_orbs;
    let n_shells = mono.basis.shells.len();

    let orbs = mono.orbs.as_ref().unwrap();
    let s = &mono.s;

    let c_occ = orbs.select(Axis(1), &occ);
    let c_virt = orbs.select(Axis(1), &virt);
    let sc_occ = s.dot(&c_occ);
    let sc_virt = s.dot(&c_virt);

    let mut qvo_ao = Array2::<f64>::zeros([n_orbs, nvo]);
    for a in 0..nvirt {
        for i in 0..nocc {
            let ai = a * nocc + i;
            for mu in 0..n_orbs {
                qvo_ao[[mu, ai]] = c_virt[[mu, a]] * sc_occ[[mu, i]]
                    + c_occ[[mu, i]] * sc_virt[[mu, a]];
            }
        }
    }

    // Aggregate to shell level
    let mut qvo_shell = Array2::<f64>::zeros([n_shells, nvo]);
    for (s_idx, shell) in mono.basis.shells.iter().enumerate() {
        for mu in shell.sph_start..shell.sph_end {
            for ai in 0..nvo {
                qvo_shell[[s_idx, ai]] += qvo_ao[[mu, ai]];
            }
        }
    }

    (qvo_ao, qvo_shell)
}

// ============================================================================
// R1. Lagrangian
// ============================================================================

/// Calculate the response Lagrangian for each monomer.
///
/// L_K[ai] = -0.5 * SCAL * sum_{pairs not containing K} ESP_on_K(CTIJ) . Q_VO_K
pub fn calculate_response_lagrangian_xtb_hop(
    mono_states: &[XtbMonomerHopScc],
    pair_states: &[XtbPairHopScc],
    trimer_states: &[XtbTrimerHopScc],
    hop_data: &XtbHopData,
    pair_scal: &[f64],
    use_three_body: bool,
) -> (Vec<Array1<f64>>, Vec<Array2<f64>>, Vec<Array2<f64>>) {
    let n_frag = mono_states.len();

    // Build QVO for each monomer
    let qvo_data: Vec<(Array2<f64>, Array2<f64>)> =
        mono_states.iter().map(|m| compute_qvo_shell_hop(m)).collect();

    let qvo_ao_vec: Vec<Array2<f64>> = qvo_data.iter().map(|(ao, _)| ao.clone()).collect();
    let qvo_shell_vec: Vec<Array2<f64>> = qvo_data.iter().map(|(_, sh)| sh.clone()).collect();

    // Lagrangian per monomer K
    let mut lagrangian_vec: Vec<Array1<f64>> = (0..n_frag)
        .into_par_iter()
        .map(|idx_k| {
            let mono_k = &mono_states[idx_k];
            let (occ_k, virt_k) = compute_occ_virt_from_f(&mono_k.f);
            let nocc = occ_k.len();
            let nvirt = virt_k.len();
            let nvo = nvirt * nocc;
            let n_shells_k = mono_k.basis.shells.len();
            let qvo_shell_k = &qvo_shell_vec[idx_k];

            let fi_k = &hop_data.frag_info[idx_k];
            let sr_k = get_frag_shell_range(&hop_data.ext_basis, &fi_k.ext_range);
            let n_k_shells = sr_k.end - sr_k.start;

            let mut lag_k = Array1::<f64>::zeros(nvo);

            // Pairs not containing K (SCAL-scaled)
            for (pair_idx, ps) in pair_states.iter().enumerate() {
                let scal = pair_scal[pair_idx];
                if scal.abs() < 1e-14 {
                    continue;
                }
                if ps.i == idx_k || ps.j == idx_k {
                    continue;
                }

                let fi_i = &hop_data.frag_info[ps.i];
                let fi_j = &hop_data.frag_info[ps.j];
                let sr_i = get_frag_shell_range(&hop_data.ext_basis, &fi_i.ext_range);
                let sr_j = get_frag_shell_range(&hop_data.ext_basis, &fi_j.ext_range);

                // Full CTIJ including ghost shells
                let (ctij_i, ctij_j) = build_pair_ctij_full(ps, mono_states, hop_data);

                let n_i_shells = sr_i.end - sr_i.start;
                let n_j_shells = sr_j.end - sr_j.start;

                let gamma_k_i = hop_data.gamma_shell_ext.slice(s![
                    sr_k.start..sr_k.start + n_k_shells,
                    sr_i.start..sr_i.start + n_i_shells
                ]);
                let gamma_k_j = hop_data.gamma_shell_ext.slice(s![
                    sr_k.start..sr_k.start + n_k_shells,
                    sr_j.start..sr_j.start + n_j_shells
                ]);

                let ctij_i_sl = ctij_i.slice(s![..n_i_shells]);
                let ctij_j_sl = ctij_j.slice(s![..n_j_shells]);

                let ctij_i_use = ctij_i_sl.to_owned();
                let ctij_j_use = ctij_j_sl.to_owned();

                let esp_shell_on_k: Array1<f64> =
                    gamma_k_i.dot(&ctij_i_use) + gamma_k_j.dot(&ctij_j_use);

                let qvo_k_sl = qvo_shell_k.slice(s![..n_k_shells, ..]);
                let lag_contribution = -0.5 * scal * esp_shell_on_k.dot(&qvo_k_sl);

                lag_k += &lag_contribution;
            }

            // Trimers not containing K
            if use_three_body {
                for ts in trimer_states.iter() {
                    if ts.i == idx_k || ts.j == idx_k || ts.k == idx_k {
                        continue;
                    }

                    let fi_ti = &hop_data.frag_info[ts.i];
                    let fi_tj = &hop_data.frag_info[ts.j];
                    let fi_tk = &hop_data.frag_info[ts.k];
                    let sr_ti = get_frag_shell_range(&hop_data.ext_basis, &fi_ti.ext_range);
                    let sr_tj = get_frag_shell_range(&hop_data.ext_basis, &fi_tj.ext_range);
                    let sr_tk = get_frag_shell_range(&hop_data.ext_basis, &fi_tk.ext_range);

                    let ddq = &ts.delta_dq_shell_real;
                    let n_rs_i = ts.n_real_shells_i;
                    let n_rs_j = ts.n_real_shells_j;

                    // Real shells
                    let g_k_ti = hop_data.gamma_shell_ext.slice(s![
                        sr_k.start..sr_k.start + n_k_shells,
                        sr_ti.start..sr_ti.start + fi_ti.n_real_shells
                    ]);
                    let g_k_tj = hop_data.gamma_shell_ext.slice(s![
                        sr_k.start..sr_k.start + n_k_shells,
                        sr_tj.start..sr_tj.start + fi_tj.n_real_shells
                    ]);
                    let g_k_tk = hop_data.gamma_shell_ext.slice(s![
                        sr_k.start..sr_k.start + n_k_shells,
                        sr_tk.start..sr_tk.start + fi_tk.n_real_shells
                    ]);

                    let mut esp_on_k: Array1<f64> = g_k_ti.dot(&ddq.slice(s![..n_rs_i]))
                        + g_k_tj.dot(&ddq.slice(s![n_rs_i..n_rs_i + n_rs_j]))
                        + g_k_tk.dot(&ddq.slice(s![n_rs_i + n_rs_j..]));

                    // Ghost shell contributions (matching compute_ctmul_xtb_hop trimer section)
                    let tri_frags = [ts.i, ts.j, ts.k];
                    let tri_fi = [fi_ti, fi_tj, fi_tk];
                    let tri_sr = [&sr_ti, &sr_tj, &sr_tk];
                    let tri_spa: Vec<Vec<usize>> = tri_fi.iter()
                        .map(|f| super::helpers::shells_per_atom_in_range(&hop_data.ext_basis, &f.ext_range))
                        .collect();
                    let mut mono_ghost_offsets = [fi_ti.n_real_shells, fi_tj.n_real_shells, fi_tk.n_real_shells];
                    let mut tri_ghost_offset = n_rs_i + n_rs_j + (ts.n_real_shells_k);
                    let mut ghost_idxs = [0usize; 3];

                    for bond in &hop_data.detached_bonds {
                        let bda_in_tri = tri_frags.contains(&bond.bda_fragment);
                        for fp in 0..3 {
                            if bond.baa_fragment == tri_frags[fp] {
                                let n_gs = tri_spa[fp][tri_fi[fp].n_real_atoms + ghost_idxs[fp]];
                                let ghost_global_start = tri_sr[fp].start + mono_ghost_offsets[fp];

                                // Compute ghost CTIJK (matching compute_ctmul_xtb_hop)
                                let ctijk_ghost: Vec<f64> = if bda_in_tri {
                                    // Healed: CTIJK_ghost = -mono_dq
                                    (0..n_gs).map(|s|
                                        -mono_states[tri_frags[fp]].dq_shell[mono_ghost_offsets[fp] + s]
                                    ).collect()
                                } else {
                                    // Partial: CTIJK_ghost = tri_dq - mono_dq
                                    (0..n_gs).map(|s| {
                                        let tri_val = ts.dq_shell[tri_ghost_offset + s];
                                        let mono_val = mono_states[tri_frags[fp]].dq_shell[mono_ghost_offsets[fp] + s];
                                        tri_val - mono_val
                                    }).collect()
                                };

                                // Add gamma[K, ghost_shells] . ctijk_ghost to ESP
                                for (s, &ctijk_val) in ctijk_ghost.iter().enumerate() {
                                    if ctijk_val.abs() < 1e-30 { continue; }
                                    let gs = ghost_global_start + s;
                                    for local_k in 0..n_k_shells {
                                        esp_on_k[local_k] += hop_data.gamma_shell_ext
                                            [[sr_k.start + local_k, gs]] * ctijk_val;
                                    }
                                }

                                if !bda_in_tri { tri_ghost_offset += n_gs; }
                                mono_ghost_offsets[fp] += n_gs;
                                ghost_idxs[fp] += 1;
                                break;
                            }
                        }
                    }

                    let qvo_k_sl2 = qvo_shell_k.slice(s![..n_k_shells, ..]);
                    lag_k += &(-0.5 * esp_on_k.dot(&qvo_k_sl2));
                }
            }

            lag_k
        })
        .collect();

    (lagrangian_vec, qvo_ao_vec, qvo_shell_vec)
}

// ============================================================================
// R2. SCZV-CG solver
// ============================================================================

/// Orbital Hessian matvec: A_I · v
fn orbital_hessian_matvec(
    mono: &XtbMonomerHopScc,
    qvo_shell: &Array2<f64>,
    v: &Array1<f64>,
) -> Array1<f64> {
    let (occ, virt) = compute_occ_virt_from_f(&mono.f);
    let nocc = occ.len();
    let nvirt = virt.len();
    let orbe = mono.orbe.as_ref().unwrap();
    let gamma_shell = &mono.gamma_shell;

    // Diagonal: (eps_i - eps_a) * v_ai
    let mut result = Array1::<f64>::zeros(v.len());
    for a in 0..nvirt {
        for i in 0..nocc {
            let idx = a * nocc + i;
            result[idx] = (orbe[occ[i]] - orbe[virt[a]]) * v[idx];
        }
    }

    // Coulomb: -Q_shell^T @ gamma_shell @ Q_shell @ v
    let qv_shell = qvo_shell.dot(v);
    let g_qv_shell = gamma_shell.dot(&qv_shell);
    let qt_g_qv = qvo_shell.t().dot(&g_qv_shell);
    result -= &qt_g_qv;

    result
}

/// Solve SCZV equations with preconditioned CG.
pub fn solve_sczv_cg_xtb_hop(
    mono_states: &[XtbMonomerHopScc],
    hop_data: &XtbHopData,
    lagrangian: &[Array1<f64>],
    qvo_ao_vec: &[Array2<f64>],
    qvo_shell_vec: &[Array2<f64>],
    gammafunction: &XtbGammaFunction,
) -> Vec<Array1<f64>> {
    let maxiter = 500;
    let threshold = 1.0e-8;
    let n_frag = mono_states.len();

    let rhs: Vec<Array1<f64>> = lagrangian.iter().map(|l| 4.0 * l).collect();

    let seg_sizes: Vec<usize> = mono_states
        .iter()
        .map(|m| {
            let (occ, virt) = compute_occ_virt_from_f(&m.f);
            virt.len() * occ.len()
        })
        .collect();

    // Atom-level QVO for third-order
    let qvo_atom_vec: Vec<Array2<f64>> = mono_states
        .iter()
        .enumerate()
        .map(|(idx, m)| {
            let nvo = seg_sizes[idx];
            let mut qvo_atom = Array2::<f64>::zeros([m.n_ext_atoms, nvo]);
            for shell in m.basis.shells.iter() {
                let at = shell.atom_index;
                for mu in shell.sph_start..shell.sph_end {
                    for ai in 0..nvo {
                        qvo_atom[[at, ai]] += qvo_ao_vec[idx][[mu, ai]];
                    }
                }
            }
            qvo_atom
        })
        .collect();

    // Third-order factors
    let third_factor_vec: Vec<Array1<f64>> = mono_states
        .iter()
        .map(|m| {
            let mut factors = Array1::<f64>::zeros(m.n_ext_atoms);
            for (at, atom) in m.ext_atoms.iter().enumerate() {
                let hubb_deriv = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
                factors[at] = 2.0 * hubb_deriv * m.dq[at];
            }
            factors
        })
        .collect();

    // Shell ranges in ext_basis
    let shell_ranges: Vec<std::ops::Range<usize>> = hop_data
        .frag_info
        .iter()
        .map(|fi| get_frag_shell_range(&hop_data.ext_basis, &fi.ext_range))
        .collect();

    let matvec = |z_vecs: &Vec<Array1<f64>>| -> Vec<Array1<f64>> {
        // Inter-fragment coupling: full shells (including ghosts)
        let qindz_shell: Vec<Array1<f64>> = (0..n_frag)
            .map(|k| qvo_shell_vec[k].dot(&z_vecs[k]))
            .collect();

        (0..n_frag)
            .into_par_iter()
            .map(|idx_i| {
                let mono_i = &mono_states[idx_i];

                // Intra-fragment (full shells — monomer's own gamma includes ghosts)
                let mut result_i = orbital_hessian_matvec(
                    mono_i,
                    &qvo_shell_vec[idx_i],
                    &z_vecs[idx_i],
                );

                // Third-order (atom-level)
                let qv_atom = qvo_atom_vec[idx_i].dot(&z_vecs[idx_i]);
                let g_qv_atom = &third_factor_vec[idx_i] * &qv_atom;
                result_i -= &qvo_atom_vec[idx_i].t().dot(&g_qv_atom);

                // Inter-fragment coupling
                let n_shells_i = mono_i.basis.shells.len();
                let sr_i = &shell_ranges[idx_i];
                let mut shift_shell_i = Array1::<f64>::zeros(n_shells_i);

                for idx_k in 0..n_frag {
                    if idx_k == idx_i {
                        continue;
                    }
                    let sr_k = &shell_ranges[idx_k];
                    let gamma_ik = hop_data.gamma_shell_ext.slice(s![
                        sr_i.start..sr_i.end,
                        sr_k.start..sr_k.end
                    ]);
                    let shift_from_k = gamma_ik.dot(&qindz_shell[idx_k]);
                    shift_shell_i.add_assign(&shift_from_k);
                }
                result_i -= &qvo_shell_vec[idx_i].t().dot(&shift_shell_i);

                result_i
            })
            .collect()
    };

    // Jacobi preconditioner
    let inv_diag: Vec<Array1<f64>> = mono_states
        .iter()
        .map(|m| {
            let (occ, virt) = compute_occ_virt_from_f(&m.f);
            let orbe = m.orbe.as_ref().unwrap();
            let mut inv_d = Array1::<f64>::zeros(virt.len() * occ.len());
            for a in 0..virt.len() {
                for i in 0..occ.len() {
                    inv_d[a * occ.len() + i] = 1.0 / (orbe[occ[i]] - orbe[virt[a]]);
                }
            }
            inv_d
        })
        .collect();

    let precond = |r_vecs: &Vec<Array1<f64>>| -> Vec<Array1<f64>> {
        r_vecs
            .iter()
            .zip(inv_diag.iter())
            .map(|(ri, inv_d)| ri * inv_d)
            .collect()
    };

    let dot_all = |a: &Vec<Array1<f64>>, b: &Vec<Array1<f64>>| -> f64 {
        a.iter().zip(b.iter()).map(|(ai, bi)| ai.dot(bi)).sum()
    };
    let vec_sub = |a: &Vec<Array1<f64>>, b: &Vec<Array1<f64>>| -> Vec<Array1<f64>> {
        a.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect()
    };
    let vec_add = |a: &Vec<Array1<f64>>, b: &Vec<Array1<f64>>| -> Vec<Array1<f64>> {
        a.iter().zip(b.iter()).map(|(ai, bi)| ai + bi).collect()
    };
    let vec_scale = |s: f64, a: &Vec<Array1<f64>>| -> Vec<Array1<f64>> {
        a.iter().map(|ai| s * ai).collect()
    };

    let mut z_vecs: Vec<Array1<f64>> = seg_sizes.iter().map(|&n| Array1::zeros(n)).collect();

    let az = matvec(&z_vecs);
    let mut r = vec_sub(&rhs, &az);
    let mut z_pre = precond(&r);
    let mut p = z_pre.clone();
    let mut rz = dot_all(&r, &z_pre);
    let rhs_norm = dot_all(&rhs, &rhs).sqrt();
    let abs_threshold = threshold * rhs_norm;

    for _iter in 0..maxiter {
        let residual = dot_all(&r, &r).sqrt();
        if residual < abs_threshold {
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

// ============================================================================
// R3. Response gradient on-the-fly
// ============================================================================

/// Per-monomer response gradient: Z×H + Z×B terms with shell-pair loop.
pub fn response_gradient_onthefly_xtb_hop(
    mono_states: &[XtbMonomerHopScc],
    hop_data: &XtbHopData,
    z_vectors: &[Array1<f64>],
    gammafunction: &XtbGammaFunction,
    cn_numbers_global: ArrayView1<f64>,
    cn_grad_global: ArrayView2<f64>,
    n_atoms_total: usize,
    frag_atom_ranges: &[std::ops::Range<usize>],
) -> Vec<(Array2<f64>, Array1<f64>)> {
    mono_states
        .par_iter()
        .enumerate()
        .map(|(idx_i, mono)| {
            let z_i = &z_vectors[idx_i];
            let n_atoms = mono.n_ext_atoms;
            let n_real_atoms = mono.n_real_atoms;
            let n_orbs = mono.n_ext_orbs;
            let n_shells = mono.basis.shells.len();
            let (occ, virt) = compute_occ_virt_from_f(&mono.f);
            let nocc = occ.len();
            let nvirt = virt.len();

            let orbs = mono.orbs.as_ref().unwrap();
            let orbe = mono.orbe.as_ref().unwrap();
            let s = &mono.s;
            let gamma_shell = &mono.gamma_shell;
            let dq_shell = &mono.dq_shell;
            let dq = &mono.dq;

            // ESP_Q for this monomer
            let esp_q_shell = compute_esp_q_shell_hop(idx_i, hop_data);

            // Z in MO → AO
            let z_mat = z_i.view().into_shape([nvirt, nocc]).unwrap();
            let c_virt = orbs.select(Axis(1), &virt);
            let c_occ = orbs.select(Axis(1), &occ);
            let z_ao: Array2<f64> = c_virt.dot(&z_mat.dot(&c_occ.t()));

            // WZ: energy-weighted Z
            let mut wz_mat = Array2::<f64>::zeros([nvirt, nocc]);
            for (i_idx, &i_mo) in occ.iter().enumerate() {
                for a in 0..nvirt {
                    wz_mat[[a, i_idx]] = z_mat[[a, i_idx]] * orbe[i_mo];
                }
            }
            let wz_ao: Array2<f64> = c_virt.dot(&wz_mat.dot(&c_occ.t()));

            // Shift matching SCC Hamiltonian
            let shift_shell: Array1<f64> = gamma_shell.dot(dq_shell) + &esp_q_shell;
            let mut shift_vec = shell_to_ao_values(&mono.basis, n_orbs, shift_shell.view());

            // Third-order shift
            for shell in mono.basis.shells.iter() {
                let at = shell.atom_index;
                let hubb_deriv = COUL_THIRD_ORDER_ATOM[mono.ext_atoms[at].number as usize - 1];
                let epot = hubb_deriv * dq[at] * dq[at];
                for mu in shell.sph_start..shell.sph_end {
                    shift_vec[mu] -= epot;
                }
            }

            let shift_ao_mat = aovec_to_aomat(shift_vec.view(), n_orbs) * 0.5;
            let wrk_response: Array2<f64> = &shift_ao_mat * &z_ao - &wz_ao;

            // Q^Z
            let z_sym: Array2<f64> = 0.5 * (&z_ao + &z_ao.t());
            let zs = z_sym.dot(s);
            let mut q_z_ao = Array1::<f64>::zeros(n_orbs);
            for mu in 0..n_orbs {
                q_z_ao[mu] = zs[[mu, mu]];
            }
            let mut q_z_shell = vec![0.0; n_shells];
            for (s_idx, shell) in mono.basis.shells.iter().enumerate() {
                for mu in shell.sph_start..shell.sph_end {
                    q_z_shell[s_idx] += q_z_ao[mu];
                }
            }

            // CN numbers: real from global, ghost = 0
            let atom_start = frag_atom_ranges[idx_i].start;
            let mut cn_numbers = Array1::<f64>::zeros(n_atoms);
            cn_numbers
                .slice_mut(s![..n_real_atoms])
                .assign(&cn_numbers_global.slice(s![frag_atom_ranges[idx_i].clone()]));

            let mut grad_local = Array2::<f64>::zeros([n_atoms, 3]);
            let mut cn_factors: Vec<f64> = vec![0.0; n_atoms];

            // === Shell-pair loop ===
            for (shell_i_idx, shell_i) in mono.basis.shells.iter().enumerate() {
                let atomi = &mono.ext_atoms[shell_i.atom_index];
                let at_i = shell_i.atom_index;
                let cn_1 = cn_numbers[at_i];

                for (shell_j_idx, shell_j) in mono.basis.shells.iter().enumerate() {
                    let atomj = &mono.ext_atoms[shell_j.atom_index];
                    let at_j = shell_j.atom_index;
                    let cn_2 = cn_numbers[at_j];

                    let r_vector: Vector3<f64> = atomi - atomj;
                    let distance = r_vector.norm();
                    if distance >= PROXIMITY_CUTOFF {
                        continue;
                    }

                    let self_energy_term = get_self_energy_values_new(
                        atomi.number, atomj.number, cn_1, cn_2,
                        shell_i.shell_index, shell_j.shell_index,
                    );
                    let cn_coeff_i =
                        get_self_energy_cn_grad_coeff_shell(atomi.number, shell_i.shell_index);
                    let cn_coeff_j =
                        get_self_energy_cn_grad_coeff_shell(atomj.number, shell_j.shell_index);

                    let is_same_shell = shell_i.sph_start == shell_j.sph_start
                        && shell_i.sph_end == shell_j.sph_end;

                    let (scaling_constant, en_term, hueckel_const, pi_term) = if !is_same_shell {
                        // xTB applies the element-pair scaling only to valence-valence
                        // shell pairs; pairs involving a polarization shell use 1.0.
                        let sc = if shell_i.polarization || shell_j.polarization {
                            1.0
                        } else {
                            calculate_pair_scaling_param(
                                atomi.number,
                                atomj.number,
                                shell_i.angular_momentum,
                                shell_j.angular_momentum,
                                shell_i.shell_index,
                                shell_j.shell_index,
                            )
                        };
                        let pauling_diff = (PAULING_EN[atomi.number as usize - 1]
                            - PAULING_EN[atomj.number as usize - 1])
                            .powi(2);
                        let en = if !shell_i.polarization && !shell_j.polarization {
                            1.0 + EN_SHELL_PARAM * pauling_diff
                        } else {
                            1.0
                        };
                        let hc = get_hueckel_constants_new(
                            atomi.number, atomj.number,
                            shell_i.angular_momentum, shell_j.angular_momentum,
                            shell_i.polarization, shell_j.polarization,
                        );
                        let pt = get_pi_term(
                            distance, atomi.number as usize, atomj.number as usize,
                            shell_i.angular_momentum, shell_j.angular_momentum,
                        );
                        (sc, en, hc, pt)
                    } else {
                        (0.0, 0.0, 0.0, 0.0)
                    };

                    let h0_val =
                        scaling_constant * hueckel_const * self_energy_term * en_term * pi_term;
                    let h_val_cn = scaling_constant * hueckel_const * en_term * pi_term;

                    let pi_grad: [f64; 3] = if at_i != at_j && !is_same_shell {
                        get_pi_term_gradient_inline_shell(
                            &r_vector, distance,
                            atomi.number as usize, atomj.number as usize,
                            shell_i.angular_momentum, shell_j.angular_momentum,
                        )
                    } else {
                        [0.0, 0.0, 0.0]
                    };
                    let pi_factor =
                        scaling_constant * hueckel_const * self_energy_term * en_term;

                    let mut diag_sp_sum: f64 = 0.0;
                    let mut off_sp_sum: f64 = 0.0;
                    let mut shell_pi_sp_sum: f64 = 0.0;
                    let mut shell_ds_contrib: [f64; 3] = [0.0; 3];

                    for idx_i_ao in shell_i.sph_start..shell_i.sph_end {
                        let idx_i_local = idx_i_ao - shell_i.sph_start;
                        for idx_j_ao in shell_j.sph_start..shell_j.sph_end {
                            let idx_j_local = idx_j_ao - shell_j.sph_start;
                            let z_ij = z_ao[[idx_i_ao, idx_j_ao]];
                            let s_ij = s[[idx_i_ao, idx_j_ao]];

                            if idx_i_ao == idx_j_ao {
                                diag_sp_sum += s_ij * z_ij;
                            } else {
                                off_sp_sum += s_ij * z_ij;
                                if at_i != at_j
                                    && shell_i.angular_momentum < 2
                                    && shell_j.angular_momentum < 2
                                {
                                    let orbital1 = &mono.basis.basis_functions
                                        [shell_i.start + idx_i_local];
                                    let orbital2 = &mono.basis.basis_functions
                                        [shell_j.start + idx_j_local];
                                    let norm_prod =
                                        orbital1.contracted_norm * orbital2.contracted_norm;
                                    let w_ij = wrk_response[[idx_i_ao, idx_j_ao]];
                                    let combined = h0_val * z_ij + w_ij;
                                    let ds_all =
                                        obara_saika_derivatives_all(orbital1, orbital2);
                                    for dir in 0..3 {
                                        shell_ds_contrib[dir] +=
                                            ds_all[dir] * norm_prod * combined;
                                    }
                                    shell_pi_sp_sum += s_ij
                                        * 0.5
                                        * (z_ao[[idx_i_ao, idx_j_ao]]
                                            + z_ao[[idx_j_ao, idx_i_ao]]);
                                }
                            }
                        }
                    }

                    // D-orbital handling
                    let shell_i_has_d = shell_i.angular_momentum >= 2;
                    let shell_j_has_d = shell_j.angular_momentum >= 2;
                    let either_has_d = shell_i_has_d || shell_j_has_d;

                    if at_i != at_j && either_has_d && shell_i_idx < shell_j_idx {
                        let ds_d =
                            calc_overlap_derivative_d_shells(&mono.basis, shell_i, shell_j);
                        let sph_dim_i = shell_i.sph_end - shell_i.sph_start;
                        let sph_dim_j = shell_j.sph_end - shell_j.sph_start;
                        for sph_i in 0..sph_dim_i {
                            let ii = shell_i.sph_start + sph_i;
                            for sph_j in 0..sph_dim_j {
                                let jj = shell_j.sph_start + sph_j;
                                let z_ij = z_ao[[ii, jj]];
                                let z_ji = z_ao[[jj, ii]];
                                let w_ij = wrk_response[[ii, jj]];
                                let w_ji = wrk_response[[jj, ii]];
                                for dir in 0..3 {
                                    let ds_val_i = ds_d[[dir, sph_i, sph_j]];
                                    let ds_val_j = ds_d[[3 + dir, sph_i, sph_j]];
                                    let combined =
                                        h0_val * (z_ij + z_ji) + (w_ij + w_ji);
                                    grad_local[[at_i, dir]] += ds_val_i * combined;
                                    grad_local[[at_j, dir]] += ds_val_j * combined;
                                }
                                shell_pi_sp_sum +=
                                    s[[ii, jj]] * 0.5 * (z_ij + z_ji);
                            }
                        }
                    }

                    if at_i != at_j {
                        for dir in 0..3 {
                            grad_local[[at_i, dir]] += shell_ds_contrib[dir];
                            grad_local[[at_j, dir]] -= shell_ds_contrib[dir];
                        }
                        let pi_contrib = 2.0 * pi_factor * shell_pi_sp_sum;
                        for dir in 0..3 {
                            grad_local[[at_i, dir]] += pi_grad[dir] * pi_contrib;
                        }
                        if either_has_d && shell_i_idx < shell_j_idx {
                            for dir in 0..3 {
                                grad_local[[at_j, dir]] -= pi_grad[dir] * pi_contrib;
                            }
                        }
                    }

                    if diag_sp_sum.abs() > 1e-15 {
                        cn_factors[at_i] += cn_coeff_i * diag_sp_sum;
                    }
                    if off_sp_sum.abs() > 1e-15 {
                        let off_factor = 0.5 * h_val_cn * off_sp_sum;
                        cn_factors[at_i] += off_factor * cn_coeff_i;
                        cn_factors[at_j] += off_factor * cn_coeff_j;
                    }
                }
            }

            // CN gradient: only real atoms
            let mut cn_grad_contribution = Array1::<f64>::zeros(3 * n_atoms_total);
            for at in 0..n_real_atoms {
                if cn_factors[at].abs() > 1e-15 {
                    let global_at = atom_start + at;
                    let cn_grad_at = cn_grad_global.slice(s![.., global_at]);
                    for k in 0..(3 * n_atoms_total) {
                        cn_grad_contribution[k] += cn_factors[at] * cn_grad_at[k];
                    }
                }
            }

            // Gamma derivative: dq_shell × q_z_shell
            for (si_idx, shell_i) in mono.basis.shells.iter().enumerate() {
                let atomi = &mono.ext_atoms[shell_i.atom_index];
                let at_i = shell_i.atom_index;
                for (sj_idx, shell_j) in mono.basis.shells.iter().enumerate() {
                    let atomj = &mono.ext_atoms[shell_j.atom_index];
                    let at_j = shell_j.atom_index;
                    if at_i != at_j {
                        let r_vec: Vector3<f64> = atomi - atomj;
                        let dist = r_vec.norm();
                        let inv = 1.0 / dist;
                        let e = [r_vec.x * inv, r_vec.y * inv, r_vec.z * inv];
                        let gd = gammafunction.deriv(
                            dist, atomi.number, shell_i.angular_momentum as u8,
                            atomj.number, shell_j.angular_momentum as u8,
                        );
                        let factor = (dq_shell[si_idx] * q_z_shell[sj_idx]
                            + q_z_shell[si_idx] * dq_shell[sj_idx])
                            * 0.5
                            * gd;
                        for dir in 0..3 {
                            grad_local[[at_i, dir]] += e[dir] * factor;
                            grad_local[[at_j, dir]] -= e[dir] * factor;
                        }
                    }
                }
            }

            (grad_local, cn_grad_contribution)
        })
        .collect()
}

// ============================================================================
// R4. Response addlag
// ============================================================================

/// Response addlag: dS/dR × (shiftz*P - 0.5*P*(shiftz*S)*P)
/// where shiftz = gamma_intra · Q_Z_self + gamma_ext · Q_Z_other (full intra+inter).
pub fn add_response_addlag_xtb_hop(
    mono_states: &[XtbMonomerHopScc],
    hop_data: &XtbHopData,
    z_vectors: &[Array1<f64>],
    gammafunction: &XtbGammaFunction,
    n_atoms_total: usize,
    frag_atom_ranges: &[std::ops::Range<usize>],
) -> Array1<f64> {
    let n_grad = 3 * n_atoms_total;
    let n_frag = mono_states.len();
    let mut addlag_total = Array1::<f64>::zeros(n_grad);

    // Pre-compute Q_Z at shell level — zero ghost shells (matching DFTB convention)
    let q_z_shell_all: Vec<Array1<f64>> = mono_states
        .iter()
        .enumerate()
        .map(|(idx, mono)| {
            let z_i = &z_vectors[idx];
            let (occ, virt) = compute_occ_virt_from_f(&mono.f);
            let nocc = occ.len();
            let nvirt = virt.len();
            let orbs = mono.orbs.as_ref().unwrap();
            let s_mat = &mono.s;

            let z_mat = z_i.view().into_shape([nvirt, nocc]).unwrap();
            let c_virt = orbs.select(Axis(1), &virt);
            let c_occ = orbs.select(Axis(1), &occ);
            let z_ao_unsym = c_virt.dot(&z_mat.dot(&c_occ.t()));
            let z_sym: Array2<f64> = 0.5 * (&z_ao_unsym + &z_ao_unsym.t());
            let zs = z_sym.dot(s_mat);

            let mut q_z = Array1::<f64>::zeros(mono.basis.shells.len());
            for (s_idx, shell) in mono.basis.shells.iter().enumerate() {
                for mu in shell.sph_start..shell.sph_end {
                    q_z[s_idx] += zs[[mu, mu]];
                }
            }
            q_z
        })
        .collect();

    // Shell ranges in ext_basis for each fragment
    let shell_ranges: Vec<std::ops::Range<usize>> = hop_data
        .frag_info
        .iter()
        .map(|fi| get_frag_shell_range(&hop_data.ext_basis, &fi.ext_range))
        .collect();

    let addlag_results: Vec<Array1<f64>> = mono_states
        .par_iter()
        .enumerate()
        .map(|(idx, mono)| {
        let z_i = &z_vectors[idx];
        let (occ, virt) = compute_occ_virt_from_f(&mono.f);
        let nocc = occ.len();
        let nvirt = virt.len();
        let n_orbs = mono.n_ext_orbs;

        let orbs = mono.orbs.as_ref().unwrap();
        let s_mat = &mono.s;
        let p_mat = &mono.p;
        let gamma_shell = &mono.gamma_shell;
        let q_z_shell = &q_z_shell_all[idx];

        // SHIFTZ = gamma_intra · Q_Z_self (intra-fragment, uses full Q_Z including ghosts)
        let mut shiftz_shell = gamma_shell.dot(q_z_shell);

        // Add inter-fragment contribution: gamma_ext[frag_i, frag_j] · Q_Z_j
        let sr_i = &shell_ranges[idx];
        let fi_i = &hop_data.frag_info[idx];
        for (jdx, _mono_j) in mono_states.iter().enumerate() {
            if jdx == idx {
                continue;
            }
            let sr_j = &shell_ranges[jdx];
            let gamma_ij = hop_data.gamma_shell_ext.slice(s![
                sr_i.start..sr_i.end,
                sr_j.start..sr_j.end
            ]);
            let qz_j = &q_z_shell_all[jdx];
            shiftz_shell += &gamma_ij.dot(qz_j);
        }

        // Third-order contribution to SHIFTZ (atom-level, subtracted)
        // δV_third[mu] = -2 * hubb[atom(mu)] * dq[atom(mu)] * Q_Z_atom[atom(mu)]
        // Compute Q_Z at atom level from shell Q_Z
        let mut q_z_atom = Array1::<f64>::zeros(mono.n_ext_atoms);
        for (s_idx, shell) in mono.basis.shells.iter().enumerate() {
            q_z_atom[shell.atom_index] += q_z_shell[s_idx];
        }
        let mut shiftz_ao = shell_to_ao_values(&mono.basis, n_orbs, shiftz_shell.view());
        for shell in mono.basis.shells.iter() {
            let at = shell.atom_index;
            let hubb_deriv = COUL_THIRD_ORDER_ATOM[mono.ext_atoms[at].number as usize - 1];
            let third_shift = 2.0 * hubb_deriv * mono.dq[at] * q_z_atom[at];
            for mu in shell.sph_start..shell.sph_end {
                shiftz_ao[mu] -= third_shift;
            }
        }
        let shiftz_mat = aovec_to_aomat(shiftz_ao.view(), n_orbs) * 0.5;

        // WRK = shiftz*P - 0.5*P*(shiftz*S)*P
        let shift_s = &shiftz_mat * s_mat;
        let d_shift_s = p_mat.dot(&shift_s);
        let d_shift_s_d = d_shift_s.dot(p_mat);
        let wrk = &(&shiftz_mat * p_mat) - &(0.5 * &d_shift_s_d);

        // Shell-pair loop (only dS/dR × wrk, same as addlag in monomer gradient)
        let n_atoms = mono.n_ext_atoms;
        let mut local_grad = Array1::<f64>::zeros(3 * n_atoms);

        for (shell_i_idx, shell_i) in mono.basis.shells.iter().enumerate() {
            let atomi = &mono.ext_atoms[shell_i.atom_index];
            let at_i = shell_i.atom_index;
            for (shell_j_idx, shell_j) in mono.basis.shells.iter().enumerate() {
                let atomj = &mono.ext_atoms[shell_j.atom_index];
                let at_j = shell_j.atom_index;
                if at_i == at_j {
                    continue;
                }
                let r_vec: Vector3<f64> = atomi - atomj;
                let dist = r_vec.norm();
                if dist >= PROXIMITY_CUTOFF {
                    continue;
                }
                let is_same = shell_i.sph_start == shell_j.sph_start
                    && shell_i.sph_end == shell_j.sph_end;
                if is_same {
                    continue;
                }

                let mut shell_ds_contrib: [f64; 3] = [0.0; 3];
                // sp block
                if shell_i.angular_momentum < 2 && shell_j.angular_momentum < 2 {
                    for idx_i in shell_i.sph_start..shell_i.sph_end {
                        let il = idx_i - shell_i.sph_start;
                        for idx_j in shell_j.sph_start..shell_j.sph_end {
                            let jl = idx_j - shell_j.sph_start;
                            if idx_i != idx_j {
                                let w_ij = wrk[[idx_i, idx_j]];
                                let o1 = &mono.basis.basis_functions[shell_i.start + il];
                                let o2 = &mono.basis.basis_functions[shell_j.start + jl];
                                let np = o1.contracted_norm * o2.contracted_norm;
                                let ds = obara_saika_derivatives_all(o1, o2);
                                for dir in 0..3 {
                                    shell_ds_contrib[dir] += ds[dir] * np * w_ij;
                                }
                            }
                        }
                    }
                }
                for dir in 0..3 {
                    local_grad[3 * at_i + dir] += shell_ds_contrib[dir];
                    local_grad[3 * at_j + dir] -= shell_ds_contrib[dir];
                }

                // D-orbital handling
                let either_d = shell_i.angular_momentum >= 2 || shell_j.angular_momentum >= 2;
                if either_d && shell_i_idx < shell_j_idx {
                    let ds_d = calc_overlap_derivative_d_shells(&mono.basis, shell_i, shell_j);
                    let sph_i = shell_i.sph_end - shell_i.sph_start;
                    let sph_j = shell_j.sph_end - shell_j.sph_start;
                    for si in 0..sph_i {
                        let ii = shell_i.sph_start + si;
                        for sj in 0..sph_j {
                            let jj = shell_j.sph_start + sj;
                            let w_ij = wrk[[ii, jj]];
                            for dir in 0..3 {
                                local_grad[3 * at_i + dir] +=
                                    2.0 * ds_d[[dir, si, sj]] * w_ij;
                                local_grad[3 * at_j + dir] +=
                                    2.0 * ds_d[[3 + dir, si, sj]] * w_ij;
                            }
                        }
                    }
                }
            }
        }

        // Scatter to global
        let ltg = build_monomer_local_to_global(
            frag_atom_ranges[idx].clone(),
            hop_data,
            idx,
        );
        let mut frag_global = Array1::<f64>::zeros(n_grad);
        for local_idx in 0..n_atoms {
            let global_idx = ltg[local_idx];
            for k in 0..3 {
                frag_global[3 * global_idx + k] += local_grad[3 * local_idx + k];
            }
        }
        frag_global
    }).collect();

    for frag_grad in &addlag_results {
        addlag_total += frag_grad;
    }

    addlag_total
}

// ============================================================================
// R5. Inter-fragment response gradient
// ============================================================================

/// Inter-fragment response: Q_Z[I] × dq[J] × dgamma/dR (real shells only).
pub fn inter_frag_response_gradient_xtb_hop(
    mono_states: &[XtbMonomerHopScc],
    hop_data: &XtbHopData,
    z_vectors: &[Array1<f64>],
    gammafunction: &XtbGammaFunction,
    n_atoms_total: usize,
    frag_atom_ranges: &[std::ops::Range<usize>],
) -> Array2<f64> {
    let atoms = &hop_data.ext_atoms;
    let n_frags = mono_states.len();

    // Compute Q^Z at shell level — zero ghost shells (matching DFTB convention)
    let q_z_shell_all: Vec<Vec<f64>> = mono_states
        .iter()
        .enumerate()
        .map(|(idx, mono)| {
            let z_i = &z_vectors[idx];
            let (occ, virt) = compute_occ_virt_from_f(&mono.f);
            let nocc = occ.len();
            let nvirt = virt.len();
            let orbs = mono.orbs.as_ref().unwrap();
            let s_mat = &mono.s;

            let z_mat = z_i.view().into_shape([nvirt, nocc]).unwrap();
            let c_virt = orbs.select(Axis(1), &virt);
            let c_occ = orbs.select(Axis(1), &occ);
            let z_unsym = c_virt.dot(&z_mat.dot(&c_occ.t()));
            let z_sym: Array2<f64> = 0.5 * (&z_unsym + &z_unsym.t());
            let zs = z_sym.dot(s_mat);

            let mut qz = vec![0.0; mono.basis.shells.len()];
            for (s_idx, shell) in mono.basis.shells.iter().enumerate() {
                for mu in shell.sph_start..shell.sph_end {
                    qz[s_idx] += zs[[mu, mu]];
                }
            }
            qz
        })
        .collect();

    let q_z_shell_all_proj: Vec<Vec<f64>> = q_z_shell_all.clone();
    let dq_shell_all_proj: Vec<Array1<f64>> =
        mono_states.iter().map(|m| m.dq_shell.clone()).collect();

    // Build local-to-global atom mappings (real atoms + ghost → BDA)
    let ltg_maps: Vec<Vec<usize>> = (0..n_frags)
        .map(|idx| {
            build_monomer_local_to_global(
                frag_atom_ranges[idx].clone(),
                hop_data,
                idx,
            )
        })
        .collect();

    // Parallel fold — all shells (real + ghost)
    let gradient: Array2<f64> = (0..n_frags)
        .into_par_iter()
        .fold(
            || Array2::<f64>::zeros([n_atoms_total, 3]),
            |mut gradient, idx_i| {
                let mono_i = &mono_states[idx_i];
                let fi_i = &hop_data.frag_info[idx_i];
                let ltg_i = &ltg_maps[idx_i];

                for (idx_j, mono_j) in mono_states.iter().enumerate() {
                    if idx_i == idx_j {
                        continue;
                    }
                    let fi_j = &hop_data.frag_info[idx_j];
                    let ltg_j = &ltg_maps[idx_j];

                    for (s_idx, shell_s) in mono_i.basis.shells.iter().enumerate() {
                        let local_s = shell_s.atom_index;
                        let global_s = ltg_i[local_s];
                        let atom_s = &hop_data.ext_atoms[fi_i.ext_range.start + local_s];

                        for (t_idx, shell_t) in mono_j.basis.shells.iter().enumerate() {
                            let local_t = shell_t.atom_index;
                            let global_t = ltg_j[local_t];
                            let atom_t = &hop_data.ext_atoms[fi_j.ext_range.start + local_t];

                            let dx = atom_s.xyz[0] - atom_t.xyz[0];
                            let dy = atom_s.xyz[1] - atom_t.xyz[1];
                            let dz = atom_s.xyz[2] - atom_t.xyz[2];
                            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                            if dist < 1e-10 {
                                continue;
                            }

                            let dgamma = gammafunction.deriv(
                                dist,
                                atom_s.number,
                                shell_s.angular_momentum as u8,
                                atom_t.number,
                                shell_t.angular_momentum as u8,
                            );

                            let factor = q_z_shell_all_proj[idx_i][s_idx]
                                * dq_shell_all_proj[idx_j][t_idx]
                                * dgamma
                                / dist;

                            gradient[[global_s, 0]] += factor * dx;
                            gradient[[global_s, 1]] += factor * dy;
                            gradient[[global_s, 2]] += factor * dz;
                            gradient[[global_t, 0]] -= factor * dx;
                            gradient[[global_t, 1]] -= factor * dy;
                            gradient[[global_t, 2]] -= factor * dz;
                        }
                    }
                }
                gradient
            },
        )
        .reduce(
            || Array2::<f64>::zeros([n_atoms_total, 3]),
            |mut a, b| {
                a += &b;
                a
            },
        );

    gradient
}

// ============================================================================
// R6. HOP response: Tr(Z_sym × dP_HOP/dR)
// ============================================================================

/// HOP response gradient: Tr(Z_sym × dP_HOP/dR) for each monomer.
///
/// Includes BOTH BDA bonds (bond-pointing hybrid, sign=+1) and
/// ghost bonds (3 non-bond hybrids, sign=-1) — matching DFTB's compute_z_hop_gradient.
pub fn hop_response_gradient_xtb_hop(
    mono_states: &[XtbMonomerHopScc],
    hop_data: &XtbHopData,
    z_vectors: &[Array1<f64>],
    atoms: &[XtbAtom],
    n_atoms_total: usize,
    frag_atom_ranges: &[std::ops::Range<usize>],
) -> Array2<f64> {
    let mut hop_grad = Array2::<f64>::zeros([n_atoms_total, 3]);

    if hop_data.detached_bonds.is_empty() {
        return hop_grad;
    }

    for (idx, mono) in mono_states.iter().enumerate() {
        // BDA bonds: this fragment owns the BDA
        let bda_bonds: Vec<_> = hop_data
            .detached_bonds
            .iter()
            .filter(|b| b.bda_fragment == idx)
            .collect();
        // Ghost bonds: this fragment owns the BAA (ghost at BDA position)
        let ghost_bonds: Vec<(usize, &crate::hop::DetachedBond)> = {
            let mut ghost_idx = 0usize;
            hop_data
                .detached_bonds
                .iter()
                .filter_map(|b| {
                    if b.baa_fragment == idx {
                        let gi = ghost_idx;
                        ghost_idx += 1;
                        Some((gi, b))
                    } else {
                        None
                    }
                })
                .collect()
        };

        if bda_bonds.is_empty() && ghost_bonds.is_empty() {
            continue;
        }

        let z_i = &z_vectors[idx];
        let (occ, virt) = compute_occ_virt_from_f(&mono.f);
        let nocc = occ.len();
        let nvirt = virt.len();
        let orbs = mono.orbs.as_ref().unwrap();
        let s_mat = &mono.s;

        let z_mat = z_i.view().into_shape([nvirt, nocc]).unwrap();
        let c_virt = orbs.select(Axis(1), &virt);
        let c_occ = orbs.select(Axis(1), &occ);
        let z_unsym = c_virt.dot(&z_mat.dot(&c_occ.t()));
        let z_sym: Array2<f64> = 0.5 * (&z_unsym + &z_unsym.t());

        let n_atoms = mono.n_ext_atoms;
        let mut local_grad = Array1::<f64>::zeros(3 * n_atoms);

        // BDA bonds: DD_bda = shift * |c><c|, coeff_sign = +1
        for bond in &bda_bonds {
            let bda_local = bond.bda_global - frag_atom_ranges[idx].start;
            let bda_pos = &atoms[bond.bda_global].xyz;
            let baa_pos = &atoms[bond.baa_global].xyz;
            let bond_vec = *baa_pos - *bda_pos;
            let rotated_sp3 = compute_rotated_sp3_xtb(&bond_vec);
            let dd_bda = compute_bda_dd_xtb(rotated_sp3.view(), HOP_SHIFT);
            let mut baa_grad_3 = [0.0f64; 3];

            hop_gradient_single_bond_general(
                z_sym.view(),
                s_mat.view(),
                &mono.basis,
                bda_local,
                bda_pos,
                baa_pos,
                dd_bda.view(),
                rotated_sp3.view(),
                &[1, 2, 0],
                1.0, // BDA: positive HOPCODER sign
                &mut local_grad,
                &mut baa_grad_3,
            );

            for k in 0..3 {
                hop_grad[[bond.baa_global, k]] += baa_grad_3[k];
            }
        }

        // Ghost bonds: DD_ghost = shift * (I - |c><c|), coeff_sign = -1
        for &(gi, bond) in &ghost_bonds {
            let ghost_local = mono.n_real_atoms + gi;
            let bda_pos = &atoms[bond.bda_global].xyz;
            let baa_pos = &atoms[bond.baa_global].xyz;
            let bond_vec = *baa_pos - *bda_pos;
            let rotated_sp3 = compute_rotated_sp3_xtb(&bond_vec);
            let dd_ghost =
                crate::fmo::scc_hop::hop_data::compute_ghost_nonbond_dd_xtb(
                    &bond_vec, HOP_SHIFT,
                );
            let mut baa_grad_3 = [0.0f64; 3];

            hop_gradient_single_bond_general(
                z_sym.view(),
                s_mat.view(),
                &mono.basis,
                ghost_local,
                bda_pos, // ghost is at BDA position
                baa_pos,
                dd_ghost.view(),
                rotated_sp3.view(),
                &[1, 2, 0],
                -1.0, // Ghost: negative HOPCODER sign
                &mut local_grad,
                &mut baa_grad_3,
            );

            // BAA gradient for ghost bond
            for k in 0..3 {
                hop_grad[[bond.baa_global, k]] += baa_grad_3[k];
            }
        }

        // Scatter: real atoms → global, ghost → BDA
        let ltg = build_monomer_local_to_global(
            frag_atom_ranges[idx].clone(),
            hop_data,
            idx,
        );
        for local_idx in 0..mono.n_real_atoms {
            let global_idx = ltg[local_idx];
            for k in 0..3 {
                hop_grad[[global_idx, k]] += local_grad[3 * local_idx + k];
            }
        }
        // Ghost atoms → BDA global
        for &(gi, bond) in &ghost_bonds {
            let local_idx = mono.n_real_atoms + gi;
            for k in 0..3 {
                hop_grad[[bond.bda_global, k]] += local_grad[3 * local_idx + k];
            }
        }
    }

    hop_grad
}

// ============================================================================
// Total response gradient assembly
// ============================================================================

/// Complete FMO-xTB HOP response gradient.
///
/// Assembly: -(intra_response + inter_response + hop_response + response_addlag + cn_response)
pub fn response_gradient_xtb_hop_total(
    mono_states: &[XtbMonomerHopScc],
    pair_states: &[XtbPairHopScc],
    trimer_states: &[XtbTrimerHopScc],
    hop_data: &XtbHopData,
    pair_scal: &[f64],
    gammafunction: &XtbGammaFunction,
    cn_numbers_global: ArrayView1<f64>,
    cn_grad_global: ArrayView2<f64>,
    atoms: &[XtbAtom],
    n_atoms_total: usize,
    frag_atom_ranges: &[std::ops::Range<usize>],
    use_three_body: bool,
) -> Array1<f64> {
    let n_grad = 3 * n_atoms_total;

    let (lagrangian_vec, qvo_ao_vec, qvo_shell_vec) =
        calculate_response_lagrangian_xtb_hop(
            mono_states, pair_states, trimer_states, hop_data, pair_scal, use_three_body,
        );

    // Check if all zero
    let all_zero = lagrangian_vec
        .iter()
        .all(|l| l.mapv(|x| x.abs()).sum() < 1e-30);
    if all_zero {
        return Array1::zeros(n_grad);
    }

    let z_vectors = solve_sczv_cg_xtb_hop(
        mono_states, hop_data, &lagrangian_vec, &qvo_ao_vec, &qvo_shell_vec, gammafunction,
    );

    let local_grads = response_gradient_onthefly_xtb_hop(
        mono_states, hop_data, &z_vectors, gammafunction,
        cn_numbers_global, cn_grad_global, n_atoms_total, frag_atom_ranges,
    );

    // Scatter local gradients to global
    let mut gradient_2d = Array2::<f64>::zeros([n_atoms_total, 3]);
    let mut cn_grad_total = Array1::<f64>::zeros(n_grad);

    for (idx, (grad_local, cn_glob)) in local_grads.iter().enumerate() {
        let ltg = build_monomer_local_to_global(
            frag_atom_ranges[idx].clone(),
            hop_data,
            idx,
        );
        // Scatter real atoms
        for (local_idx, global_idx) in frag_atom_ranges[idx].clone().enumerate() {
            for k in 0..3 {
                gradient_2d[[global_idx, k]] += grad_local[[local_idx, k]];
            }
        }
        // Ghost atoms → BDA
        let mut ghost_count = 0;
        for bond in &hop_data.detached_bonds {
            if bond.baa_fragment == idx {
                let local_idx = mono_states[idx].n_real_atoms + ghost_count;
                for k in 0..3 {
                    gradient_2d[[bond.bda_global, k]] += grad_local[[local_idx, k]];
                }
                ghost_count += 1;
            }
        }
        cn_grad_total += cn_glob;
    }

    for at in 0..n_atoms_total {
        for d in 0..3 {
            gradient_2d[[at, d]] += cn_grad_total[3 * at + d];
        }
    }

    let inter_grad = inter_frag_response_gradient_xtb_hop(
        mono_states, hop_data, &z_vectors, gammafunction, n_atoms_total, frag_atom_ranges,
    );

    let hop_resp = hop_response_gradient_xtb_hop(
        mono_states, hop_data, &z_vectors, atoms, n_atoms_total, frag_atom_ranges,
    );

    let addlag_resp = add_response_addlag_xtb_hop(
        mono_states, hop_data, &z_vectors, gammafunction, n_atoms_total, frag_atom_ranges,
    );

    let total_2d = gradient_2d + inter_grad + hop_resp;
    let total_2d = -1.0 * total_2d; // MO sign convention

    let mut result = total_2d
        .into_shape([n_grad])
        .expect("Failed to reshape xTB HOP response gradient");

    result -= &addlag_resp;

    result
}
