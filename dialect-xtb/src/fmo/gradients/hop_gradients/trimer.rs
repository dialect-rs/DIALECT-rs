//! Trimer gradient for FMO-xTB HOP.
//!
//! Same pattern as pair but with 3 fragments. CTIJK instead of CTIJ.

use super::helpers::compute_esp_q_shell_hop;
use super::helpers::grad_repulsive_energy_xtb_scaled;
use dialect_base::defaults::PROXIMITY_CUTOFF;
use dialect_utilities::mulliken::shell_to_ao_values;
use crate::fmo::gradients::fmo_gradient_shell::{
    get_pi_term_gradient_inline_shell, get_self_energy_cn_grad_coeff_shell,
};
use crate::fmo::scc_hop::hop_data::{get_frag_shell_range, XtbHopData};
use crate::fmo::scc_hop::monomer::XtbMonomerHopScc;
use crate::fmo::scc_hop::trimer::XtbTrimerHopScc;
use crate::gradients::ground_state::aovec_to_aomat;
use crate::gradients::helpers::coul_third_order_grad_contribution_xtb;
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

/// Combined trimer gradient: SCC + CTIJK.
///
/// Returns (trimer_grad_local, ctijk_grad_global, cn_grad_contribution).
pub fn trimer_gradient_combined_xtb_hop(
    ts: &XtbTrimerHopScc,
    hop_data: &XtbHopData,
    mono_states: &[XtbMonomerHopScc],
    n_atoms_total: usize,
    gammafunction: &XtbGammaFunction,
    cn_numbers_global: ArrayView1<f64>,
    cn_grad_global: ArrayView2<f64>,
    local_to_global: &[usize],
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n_atoms_trimer = ts.n_ext_atoms;
    let n_orbs = ts.n_ext_orbs;
    let n_real_atoms_trimer = ts.n_real_atoms;
    let n_real_shells_i = ts.n_real_shells_i;
    let n_real_shells_j = ts.n_real_shells_j;
    let n_real_shells_k = ts.n_real_shells_k;
    let n_real_shells_trimer = n_real_shells_i + n_real_shells_j + n_real_shells_k;
    let n_shells_trimer = ts.basis.shells.len();

    let p: ArrayView2<f64> = ts.p.view();
    let s: ArrayView2<f64> = ts.s.view();
    let orbe: ArrayView1<f64> = ts.orbe.as_ref().unwrap().view();
    let orbs: ArrayView2<f64> = ts.orbs.as_ref().unwrap().view();
    let occupations = Array1::from(ts.f.clone());
    let gamma_shell: ArrayView2<f64> = ts.gamma_shell.view();
    let dq_shell_trimer: ArrayView1<f64> = ts.dq_shell.view();
    let dq: ArrayView1<f64> = ts.dq.view();

    // W
    let weighted_orbe = &orbe * &occupations;
    let worbe_2d = Array2::from_diag(&weighted_orbe);
    let w: Array2<f64> = orbs.dot(&worbe_2d.dot(&orbs.t()));

    let gamma_dq_shell: Array1<f64> = gamma_shell.dot(&dq_shell_trimer);

    // ESP from L≠I,J,K
    let fi_i = &hop_data.frag_info[ts.i];
    let fi_j = &hop_data.frag_info[ts.j];
    let fi_k = &hop_data.frag_info[ts.k];

    let dq_mon_i_shell = mono_states[ts.i].dq_shell.slice(s![..fi_i.n_real_shells]);
    let dq_mon_j_shell = mono_states[ts.j].dq_shell.slice(s![..fi_j.n_real_shells]);
    let dq_mon_k_shell = mono_states[ts.k].dq_shell.slice(s![..fi_k.n_real_shells]);

    let esp_q_i = compute_esp_q_shell_hop(ts.i, hop_data);
    let esp_q_j = compute_esp_q_shell_hop(ts.j, hop_data);
    let esp_q_k = compute_esp_q_shell_hop(ts.k, hop_data);

    let n_s_i = n_real_shells_i;
    let n_s_j = n_real_shells_j;
    let gamma_ij = gamma_shell.slice(s![..n_s_i, n_s_i..n_s_i + n_s_j]);
    let gamma_ik =
        gamma_shell.slice(s![..n_s_i, n_s_i + n_s_j..n_real_shells_trimer]);
    let gamma_jk = gamma_shell
        .slice(s![n_s_i..n_s_i + n_s_j, n_s_i + n_s_j..n_real_shells_trimer]);

    // Compute ESP from L≠I,J,K using gamma_shell_ext to include ghost shell subtraction.
    // esp_q_I includes gamma(I_ext, ALL_ext)×dq_all - gamma(I_ext, I_ext)×dq_I.
    // We need to subtract J and K contributions INCLUDING their ghost shells.
    let gamma_shell_ext = &hop_data.gamma_shell_ext;
    let ext_sr_i = get_frag_shell_range(&hop_data.ext_basis, &fi_i.ext_range);
    let ext_sr_j = get_frag_shell_range(&hop_data.ext_basis, &fi_j.ext_range);
    let ext_sr_k = get_frag_shell_range(&hop_data.ext_basis, &fi_k.ext_range);
    let dq_ext_j = hop_data.dq_shell_ext.slice(s![ext_sr_j.start..ext_sr_j.end]);
    let dq_ext_k = hop_data.dq_shell_ext.slice(s![ext_sr_k.start..ext_sr_k.end]);
    let dq_ext_i = hop_data.dq_shell_ext.slice(s![ext_sr_i.start..ext_sr_i.end]);

    // For I: subtract J_ext and K_ext contributions (real + ghost shells of J and K)
    let gamma_i_jext = gamma_shell_ext.slice(s![
        ext_sr_i.start..ext_sr_i.start + fi_i.n_real_shells,
        ext_sr_j.start..ext_sr_j.end
    ]);
    let gamma_i_kext = gamma_shell_ext.slice(s![
        ext_sr_i.start..ext_sr_i.start + fi_i.n_real_shells,
        ext_sr_k.start..ext_sr_k.end
    ]);
    let esp_from_l_i: Array1<f64> = &esp_q_i.slice(s![..fi_i.n_real_shells])
        - &gamma_i_jext.dot(&dq_ext_j)
        - &gamma_i_kext.dot(&dq_ext_k);

    // For J: subtract I_ext and K_ext contributions
    let gamma_j_iext = gamma_shell_ext.slice(s![
        ext_sr_j.start..ext_sr_j.start + fi_j.n_real_shells,
        ext_sr_i.start..ext_sr_i.end
    ]);
    let gamma_j_kext = gamma_shell_ext.slice(s![
        ext_sr_j.start..ext_sr_j.start + fi_j.n_real_shells,
        ext_sr_k.start..ext_sr_k.end
    ]);
    let esp_from_l_j: Array1<f64> = &esp_q_j.slice(s![..fi_j.n_real_shells])
        - &gamma_j_iext.dot(&dq_ext_i)
        - &gamma_j_kext.dot(&dq_ext_k);

    // For K: subtract I_ext and K_ext contributions
    let gamma_k_iext = gamma_shell_ext.slice(s![
        ext_sr_k.start..ext_sr_k.start + fi_k.n_real_shells,
        ext_sr_i.start..ext_sr_i.end
    ]);
    let gamma_k_jext = gamma_shell_ext.slice(s![
        ext_sr_k.start..ext_sr_k.start + fi_k.n_real_shells,
        ext_sr_j.start..ext_sr_j.end
    ]);
    let esp_from_l_k: Array1<f64> = &esp_q_k.slice(s![..fi_k.n_real_shells])
        - &gamma_k_iext.dot(&dq_ext_i)
        - &gamma_k_jext.dot(&dq_ext_j);

    let mut total_shift_shell = Array1::<f64>::zeros(n_shells_trimer);
    total_shift_shell
        .slice_mut(s![..n_s_i])
        .assign(&(&gamma_dq_shell.slice(s![..n_s_i]) + &esp_from_l_i));
    total_shift_shell
        .slice_mut(s![n_s_i..n_s_i + n_s_j])
        .assign(&(&gamma_dq_shell.slice(s![n_s_i..n_s_i + n_s_j]) + &esp_from_l_j));
    total_shift_shell
        .slice_mut(s![n_s_i + n_s_j..n_real_shells_trimer])
        .assign(
            &(&gamma_dq_shell.slice(s![n_s_i + n_s_j..n_real_shells_trimer]) + &esp_from_l_k),
        );
    if n_shells_trimer > n_real_shells_trimer {
        total_shift_shell
            .slice_mut(s![n_real_shells_trimer..])
            .assign(&gamma_dq_shell.slice(s![n_real_shells_trimer..]));

        // Reconstruct ghost ESP from hop_data (same as prepare_trimer_hop_xtb)
        let gamma_shell_ext = &hop_data.gamma_shell_ext;
        let ext_shell_range_i = get_frag_shell_range(&hop_data.ext_basis, &fi_i.ext_range);
        let ext_shell_range_j = get_frag_shell_range(&hop_data.ext_basis, &fi_j.ext_range);
        let ext_shell_range_k = get_frag_shell_range(&hop_data.ext_basis, &fi_k.ext_range);

        let mut tri_ghost_idx = 0usize;
        for bond in &hop_data.detached_bonds {
            let bda_in = bond.bda_fragment == ts.i
                || bond.bda_fragment == ts.j
                || bond.bda_fragment == ts.k;
            let baa_in = bond.baa_fragment == ts.i
                || bond.baa_fragment == ts.j
                || bond.baa_fragment == ts.k;
            if baa_in && !bda_in {
                let bda_frag = bond.bda_fragment;
                let bda_frag_start = hop_data.monomer_indices[bda_frag][0];
                let bda_local_in_frag = bond.bda_global - bda_frag_start;
                let bda_ext_idx =
                    hop_data.frag_info[bda_frag].ext_range.start + bda_local_in_frag;

                let bda_ext_shells: Vec<usize> = hop_data
                    .ext_basis
                    .shells
                    .iter()
                    .enumerate()
                    .filter(|(_, sh)| sh.atom_index == bda_ext_idx)
                    .map(|(idx, _)| idx)
                    .collect();

                let ghost_local = n_real_atoms_trimer + tri_ghost_idx;
                let ghost_tri_shells: Vec<usize> = ts
                    .basis
                    .shells
                    .iter()
                    .enumerate()
                    .filter(|(_, sh)| sh.atom_index == ghost_local)
                    .map(|(idx, _)| idx)
                    .collect();

                for (&bda_sh, &ghost_sh) in
                    bda_ext_shells.iter().zip(ghost_tri_shells.iter())
                {
                    let full_esp: f64 =
                        gamma_shell_ext.row(bda_sh).dot(&hop_data.dq_shell_ext);
                    let esp_from_i: f64 = gamma_shell_ext
                        .slice(s![bda_sh, ext_shell_range_i.start..ext_shell_range_i.end])
                        .dot(
                            &hop_data.dq_shell_ext.slice(
                                s![ext_shell_range_i.start..ext_shell_range_i.end],
                            ),
                        );
                    let esp_from_j: f64 = gamma_shell_ext
                        .slice(s![bda_sh, ext_shell_range_j.start..ext_shell_range_j.end])
                        .dot(
                            &hop_data.dq_shell_ext.slice(
                                s![ext_shell_range_j.start..ext_shell_range_j.end],
                            ),
                        );
                    let esp_from_k: f64 = gamma_shell_ext
                        .slice(s![bda_sh, ext_shell_range_k.start..ext_shell_range_k.end])
                        .dot(
                            &hop_data.dq_shell_ext.slice(
                                s![ext_shell_range_k.start..ext_shell_range_k.end],
                            ),
                        );
                    total_shift_shell[ghost_sh] +=
                        full_esp - esp_from_i - esp_from_j - esp_from_k;
                }

                tri_ghost_idx += 1;
            }
        }
    }
    let total_shift = shell_to_ao_values(&ts.basis, n_orbs, total_shift_shell.view());
    let coulomb_mat: Array2<f64> = aovec_to_aomat(total_shift.view(), n_orbs) * 0.5;

    // Third order
    let hubbard_derivatives: Array1<f64> = ts
        .ext_atoms
        .iter()
        .map(|a| COUL_THIRD_ORDER_ATOM[a.number as usize - 1])
        .collect();
    let dq2_gamma =
        coul_third_order_grad_contribution_xtb(&ts.basis, dq, hubbard_derivatives.view());

    let effective_mat: Array2<f64> = &(-&w) + &(&coulomb_mat * &p) - &(0.5 * &dq2_gamma * &p);

    // CTIJK
    let mut delta_dq_shell = Array1::<f64>::zeros(n_shells_trimer);
    delta_dq_shell
        .slice_mut(s![..n_real_shells_trimer])
        .assign(&ts.delta_dq_shell_real);
    let mut dq_mon_shell_padded = Array1::<f64>::zeros(n_shells_trimer);
    dq_mon_shell_padded
        .slice_mut(s![..n_s_i])
        .assign(&dq_mon_i_shell);
    dq_mon_shell_padded
        .slice_mut(s![n_s_i..n_s_i + n_s_j])
        .assign(&dq_mon_j_shell);
    dq_mon_shell_padded
        .slice_mut(s![n_s_i + n_s_j..n_real_shells_trimer])
        .assign(&dq_mon_k_shell);

    // CN numbers
    let mut cn_numbers = Array1::<f64>::zeros(n_atoms_trimer);
    for (local_idx, &global_idx) in local_to_global.iter().enumerate().take(n_real_atoms_trimer) {
        cn_numbers[local_idx] = cn_numbers_global[global_idx];
    }

    let mut grad_local = Array1::<f64>::zeros(3 * n_atoms_trimer);
    let mut ctijk_grad_global = Array1::<f64>::zeros(3 * n_atoms_total);
    let mut cn_factors: Vec<f64> = vec![0.0; n_atoms_trimer];

    // === Shell-pair loop (identical structure to pair, but with trimer basis) ===
    for (shell_i_idx, shell_i) in ts.basis.shells.iter().enumerate() {
        let atomi = &ts.ext_atoms[shell_i.atom_index];
        let at_i = shell_i.atom_index;
        let cn_1 = cn_numbers[at_i];

        for (shell_j_idx, shell_j) in ts.basis.shells.iter().enumerate() {
            let atomj = &ts.ext_atoms[shell_j.atom_index];
            let at_j = shell_j.atom_index;
            let cn_2 = cn_numbers[at_j];

            let r_vector: Vector3<f64> = atomi - atomj;
            let distance: f64 = r_vector.norm();
            if distance >= PROXIMITY_CUTOFF {
                continue;
            }

            let self_energy_term = get_self_energy_values_new(
                atomi.number,
                atomj.number,
                cn_1,
                cn_2,
                shell_i.shell_index,
                shell_j.shell_index,
            );
            let cn_coeff_i = get_self_energy_cn_grad_coeff_shell(atomi.number, shell_i.shell_index);
            let cn_coeff_j = get_self_energy_cn_grad_coeff_shell(atomj.number, shell_j.shell_index);

            let is_same_shell =
                shell_i.sph_start == shell_j.sph_start && shell_i.sph_end == shell_j.sph_end;

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
                    atomi.number,
                    atomj.number,
                    shell_i.angular_momentum,
                    shell_j.angular_momentum,
                    shell_i.polarization,
                    shell_j.polarization,
                );
                let pt = get_pi_term(
                    distance,
                    atomi.number as usize,
                    atomj.number as usize,
                    shell_i.angular_momentum,
                    shell_j.angular_momentum,
                );
                (sc, en, hc, pt)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

            let h0_val = scaling_constant * hueckel_const * self_energy_term * en_term * pi_term;
            let h_val_cn = scaling_constant * hueckel_const * en_term * pi_term;

            let pi_grad: [f64; 3] = if at_i != at_j && !is_same_shell {
                get_pi_term_gradient_inline_shell(
                    &r_vector,
                    distance,
                    atomi.number as usize,
                    atomj.number as usize,
                    shell_i.angular_momentum,
                    shell_j.angular_momentum,
                )
            } else {
                [0.0, 0.0, 0.0]
            };
            let pi_factor = scaling_constant * hueckel_const * self_energy_term * en_term;

            let mut diag_sp_sum: f64 = 0.0;
            let mut off_sp_sum: f64 = 0.0;
            let mut shell_pi_sp_sum: f64 = 0.0;
            let mut shell_ds_contrib: [f64; 3] = [0.0; 3];

            for idx_i in shell_i.sph_start..shell_i.sph_end {
                let idx_i_local = idx_i - shell_i.sph_start;
                for idx_j in shell_j.sph_start..shell_j.sph_end {
                    let idx_j_local = idx_j - shell_j.sph_start;
                    let p_ij = p[[idx_i, idx_j]];
                    let s_ij = s[[idx_i, idx_j]];
                    if idx_i == idx_j {
                        diag_sp_sum += s_ij * p_ij;
                    } else {
                        off_sp_sum += s_ij * p_ij;
                        if at_i != at_j
                            && shell_i.angular_momentum < 2
                            && shell_j.angular_momentum < 2
                        {
                            let orbital1 =
                                &ts.basis.basis_functions[shell_i.start + idx_i_local];
                            let orbital2 =
                                &ts.basis.basis_functions[shell_j.start + idx_j_local];
                            let norm_prod = orbital1.contracted_norm * orbital2.contracted_norm;
                            let eff_ij = effective_mat[[idx_i, idx_j]];
                            let combined = h0_val * p_ij + eff_ij;
                            let ds_all = obara_saika_derivatives_all(orbital1, orbital2);
                            for dir in 0..3 {
                                shell_ds_contrib[dir] += ds_all[dir] * norm_prod * combined;
                            }
                            shell_pi_sp_sum += s_ij * p_ij;
                        }
                    }
                }
            }

            // D-orbital handling
            let shell_i_has_d = shell_i.angular_momentum >= 2;
            let shell_j_has_d = shell_j.angular_momentum >= 2;
            let either_has_d = shell_i_has_d || shell_j_has_d;
            if at_i != at_j && either_has_d && shell_i_idx < shell_j_idx {
                let ds_d = calc_overlap_derivative_d_shells(&ts.basis, shell_i, shell_j);
                let sph_dim_i = shell_i.sph_end - shell_i.sph_start;
                let sph_dim_j = shell_j.sph_end - shell_j.sph_start;
                for sph_i in 0..sph_dim_i {
                    let idx_i = shell_i.sph_start + sph_i;
                    for sph_j in 0..sph_dim_j {
                        let idx_j = shell_j.sph_start + sph_j;
                        let p_ij = p[[idx_i, idx_j]];
                        let eff_ij = effective_mat[[idx_i, idx_j]];
                        for dir in 0..3 {
                            let ds_val_i = 2.0 * ds_d[[dir, sph_i, sph_j]];
                            let ds_val_j = 2.0 * ds_d[[3 + dir, sph_i, sph_j]];
                            let combined = h0_val * p_ij + eff_ij;
                            grad_local[3 * at_i + dir] += ds_val_i * combined;
                            grad_local[3 * at_j + dir] += ds_val_j * combined;
                        }
                        shell_pi_sp_sum += s[[idx_i, idx_j]] * p_ij;
                    }
                }
            }

            if at_i != at_j {
                for dir in 0..3 {
                    grad_local[3 * at_i + dir] += shell_ds_contrib[dir];
                    grad_local[3 * at_j + dir] -= shell_ds_contrib[dir];
                }
                let pi_contrib = 2.0 * pi_factor * shell_pi_sp_sum;
                for dir in 0..3 {
                    grad_local[3 * at_i + dir] += pi_grad[dir] * pi_contrib;
                }
                if either_has_d && shell_i_idx < shell_j_idx {
                    for dir in 0..3 {
                        grad_local[3 * at_j + dir] -= pi_grad[dir] * pi_contrib;
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
    for at in 0..n_real_atoms_trimer {
        if cn_factors[at].abs() > 1e-15 {
            let global_at = local_to_global[at];
            let cn_grad_at: ArrayView1<f64> = cn_grad_global.slice(s![.., global_at]);
            for k in 0..(3 * n_atoms_total) {
                cn_grad_contribution[k] += cn_factors[at] * cn_grad_at[k];
            }
        }
    }

    // Gamma gradient + CTIJK
    for (shell_i_idx, shell_i) in ts.basis.shells.iter().enumerate() {
        let atomi = &ts.ext_atoms[shell_i.atom_index];
        let at_i = shell_i.atom_index;
        for (shell_j_idx, shell_j) in ts.basis.shells.iter().enumerate() {
            let atomj = &ts.ext_atoms[shell_j.atom_index];
            let at_j = shell_j.atom_index;
            if at_i != at_j {
                let r_vector: Vector3<f64> = atomi - atomj;
                let distance: f64 = r_vector.norm();
                let inv_dist = 1.0 / distance;
                let e_ij = [
                    r_vector.x * inv_dist,
                    r_vector.y * inv_dist,
                    r_vector.z * inv_dist,
                ];
                let gamma_deriv = gammafunction.deriv(
                    distance,
                    atomi.number,
                    shell_i.angular_momentum as u8,
                    atomj.number,
                    shell_j.angular_momentum as u8,
                );
                // Trimer Coulomb
                let shell_dq_prod =
                    dq_shell_trimer[shell_i_idx] * dq_shell_trimer[shell_j_idx];
                let shell_gamma_contrib = 0.5 * gamma_deriv * shell_dq_prod;
                for dir in 0..3 {
                    grad_local[3 * at_i + dir] += e_ij[dir] * shell_gamma_contrib;
                    grad_local[3 * at_j + dir] -= e_ij[dir] * shell_gamma_contrib;
                }
                // CTIJK
                let shell_ctijk_contrib = -gamma_deriv
                    * delta_dq_shell[shell_i_idx]
                    * dq_mon_shell_padded[shell_j_idx];
                let global_i = local_to_global[at_i];
                let global_j = local_to_global[at_j];
                for dir in 0..3 {
                    ctijk_grad_global[3 * global_i + dir] += e_ij[dir] * shell_ctijk_contrib;
                    ctijk_grad_global[3 * global_j + dir] -= e_ij[dir] * shell_ctijk_contrib;
                }
            }
        }
    }

    // Ghost FMODE: compensate for ghost shell contributions in interfrag gradient.
    // The interfrag gradient includes ghost shells using gamma at the BDA position.
    // We add a matching CTIJK subtraction using gamma at BDA position so they cancel.
    // Handles BOTH healed bonds (BDA inside trimer) and partial bonds (BAA inside,
    // BDA outside trimer). For partial bonds, the BDA/ghost is at an external position.
    {
        use super::helpers::shells_per_atom_in_range;
        let tri_frags = [ts.i, ts.j, ts.k];
        let tri_fi = [&hop_data.frag_info[ts.i], &hop_data.frag_info[ts.j], &hop_data.frag_info[ts.k]];

        for bond in &hop_data.detached_bonds {
            let bda_in_tri = tri_frags.contains(&bond.bda_fragment);
            let baa_in_tri = tri_frags.contains(&bond.baa_fragment);
            if !baa_in_tri { continue; } // BAA must be in the trimer

            let baa_frag = bond.baa_fragment;
            let bda_global = bond.bda_global;

            // Ghost shell info: ghost is in BAA's monomer
            let baa_frag_pos = tri_frags.iter().position(|&f| f == baa_frag).unwrap();
            let fi_baa = tri_fi[baa_frag_pos];
            let spa_baa = shells_per_atom_in_range(&hop_data.ext_basis, &fi_baa.ext_range);

            let mut ghost_count = 0usize;
            for b2 in &hop_data.detached_bonds {
                if b2.baa_fragment == baa_frag {
                    if std::ptr::eq(b2, bond) { break; }
                    ghost_count += 1;
                }
            }
            let n_ghost_shells = spa_baa[fi_baa.n_real_atoms + ghost_count];
            let ghost_shell_start = fi_baa.n_real_shells
                + (0..ghost_count).map(|g| spa_baa[fi_baa.n_real_atoms + g]).sum::<usize>();

            // BDA atom info — may be inside or outside the trimer
            let bda_frag = bond.bda_fragment;
            let bda_atom_ref = &hop_data.ext_atoms[hop_data.frag_info[bda_frag].ext_range.start
                + (bda_global - hop_data.monomer_indices[bda_frag][0])];

            if bda_in_tri {
                // HEALED bond: BDA is a real atom in the trimer.
                // Loop over (shell_i, shell_j on BDA) using trimer basis.
                let bda_frag_pos = tri_frags.iter().position(|&f| f == bda_frag).unwrap();
                let bda_local = {
                    let mut offset = 0;
                    for fp in 0..3 {
                        if fp == bda_frag_pos { break; }
                        offset += tri_fi[fp].n_real_atoms;
                    }
                    offset + (bda_global - hop_data.monomer_indices[bda_frag][0])
                };

                for (shell_i_idx, shell_i) in ts.basis.shells.iter().enumerate() {
                    let at_i = shell_i.atom_index;
                    if at_i == bda_local { continue; }
                    let atomi = &ts.ext_atoms[at_i];

                    for (shell_j_idx, shell_j) in ts.basis.shells.iter().enumerate() {
                        if shell_j.atom_index != bda_local { continue; }

                        let atomj = &ts.ext_atoms[bda_local];
                        let r_vector: Vector3<f64> = atomi - atomj;
                        let distance = r_vector.norm();
                        if distance < 1e-10 { continue; }
                        let inv_dist = 1.0 / distance;
                        let e_ij = [r_vector.x * inv_dist, r_vector.y * inv_dist, r_vector.z * inv_dist];

                        let gamma_deriv = gammafunction.deriv(
                            distance, atomi.number, shell_i.angular_momentum as u8,
                            atomj.number, shell_j.angular_momentum as u8,
                        );

                        let sj_on_bda = ts.basis.shells.iter()
                            .take(shell_j_idx)
                            .filter(|s| s.atom_index == bda_local)
                            .count();

                        if sj_on_bda < n_ghost_shells
                            && ghost_shell_start + sj_on_bda < mono_states[baa_frag].dq_shell.len()
                        {
                            let ghost_dq = mono_states[baa_frag].dq_shell[ghost_shell_start + sj_on_bda];
                            let ghost_ctijk = -gamma_deriv * delta_dq_shell[shell_i_idx] * ghost_dq;
                            let global_i = local_to_global[at_i];
                            let global_j = local_to_global[bda_local];
                            for dir in 0..3 {
                                ctijk_grad_global[3 * global_i + dir] += e_ij[dir] * ghost_ctijk;
                                ctijk_grad_global[3 * global_j + dir] -= e_ij[dir] * ghost_ctijk;
                            }
                        }
                    }
                }
            } else {
                // PARTIAL bond: BDA is OUTSIDE the trimer (ghost at BDA position).
                // Compute gamma derivatives between trimer atoms and external BDA.
                // Ghost shells correspond to the BDA atom's element/basis.
                let bda_ext_shells: Vec<(u8, u8)> = {
                    let bda_ext_idx = hop_data.frag_info[bda_frag].ext_range.start
                        + (bda_global - hop_data.monomer_indices[bda_frag][0]);
                    hop_data.ext_basis.shells.iter()
                        .filter(|sh| sh.atom_index == bda_ext_idx)
                        .map(|sh| (bda_atom_ref.number, sh.angular_momentum as u8))
                        .collect()
                };

                for (shell_i_idx, shell_i) in ts.basis.shells.iter().enumerate() {
                    let at_i = shell_i.atom_index;
                    let atomi = &ts.ext_atoms[at_i];

                    let dx = atomi.xyz[0] - bda_atom_ref.xyz[0];
                    let dy = atomi.xyz[1] - bda_atom_ref.xyz[1];
                    let dz = atomi.xyz[2] - bda_atom_ref.xyz[2];
                    let distance = (dx*dx + dy*dy + dz*dz).sqrt();
                    if distance < 1e-10 { continue; }
                    let inv_dist = 1.0 / distance;
                    let e_ij = [dx * inv_dist, dy * inv_dist, dz * inv_dist];

                    for (gs, &(bda_z, bda_l)) in bda_ext_shells.iter().enumerate() {
                        if gs >= n_ghost_shells { break; }
                        if ghost_shell_start + gs >= mono_states[baa_frag].dq_shell.len() { break; }

                        let gamma_deriv = gammafunction.deriv(
                            distance, atomi.number, shell_i.angular_momentum as u8,
                            bda_z, bda_l,
                        );

                        let ghost_dq = mono_states[baa_frag].dq_shell[ghost_shell_start + gs];
                        let ghost_ctijk = -gamma_deriv * delta_dq_shell[shell_i_idx] * ghost_dq;
                        let global_i = local_to_global[at_i];
                        for dir in 0..3 {
                            ctijk_grad_global[3 * global_i + dir] += e_ij[dir] * ghost_ctijk;
                            ctijk_grad_global[3 * bda_global + dir] -= e_ij[dir] * ghost_ctijk;
                        }
                    }
                }
            }
        }
    }

    // Repulsive energy gradient with ZREF/QREF scaling
    let grad_rep = grad_repulsive_energy_xtb_scaled(&ts.ext_atoms, ts.zref.view(), ts.qref.view());
    grad_local += &grad_rep;

    (grad_local, ctijk_grad_global, cn_grad_contribution)
}
