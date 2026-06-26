//! Monomer gradient for FMO-xTB HOP.
//!
//! Reads from `XtbMonomerHopScc` struct instead of `properties`.
//! Returns (scc_grad + repulsive, addlag_grad) both sized [3 * n_ext_atoms].

use super::helpers::{compute_occ_virt_from_f, grad_repulsive_energy_xtb_scaled};
use dialect_base::defaults::PROXIMITY_CUTOFF;
use dialect_utilities::mulliken::shell_to_ao_values;
use crate::fmo::gradients::fmo_gradient_shell::{
    get_pi_term_gradient_inline_shell, get_self_energy_cn_grad_coeff_shell,
};
use crate::fmo::scc_hop::hop_data::XtbHopData;
use crate::fmo::scc_hop::monomer::XtbMonomerHopScc;
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

/// Combined monomer gradient: SCC + addlag, fused into shell-pair loops.
///
/// Returns (monomer_grad, addlag_grad, cn_grad_contribution):
/// - monomer_grad: SCC + repulsive gradient [3*n_ext_atoms]
/// - addlag_grad: embedding addlag gradient [3*n_ext_atoms]
/// - cn_grad_contribution: CN gradient projected to global [3*n_atoms_total]
pub fn monomer_gradient_combined_xtb_hop(
    mono: &XtbMonomerHopScc,
    hop_data: &XtbHopData,
    frag_idx: usize,
    shiftct_shell: ArrayView1<f64>,
    esp_q_shell: ArrayView1<f64>,
    cn_numbers_global: ArrayView1<f64>,
    cn_grad_global: ArrayView2<f64>,
    gammafunction: &XtbGammaFunction,
    n_atoms_total: usize,
    frag_atom_range: std::ops::Range<usize>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n_atoms = mono.n_ext_atoms;
    let n_real_atoms = mono.n_real_atoms;
    let n_orbs = mono.n_ext_orbs;
    let n_shells = mono.basis.shells.len();
    let n_real_shells = mono.n_real_shells;

    // Direct access to SCC results (full-sized, includes ghosts)
    let p: ArrayView2<f64> = mono.p.view();
    let s: ArrayView2<f64> = mono.s.view();
    let orbe: ArrayView1<f64> = mono.orbe.as_ref().unwrap().view();
    let orbs: ArrayView2<f64> = mono.orbs.as_ref().unwrap().view();
    let f_occ = &mono.f;
    let gamma_shell: ArrayView2<f64> = mono.gamma_shell.view();
    let dq_shell: ArrayView1<f64> = mono.dq_shell.view();
    let dq: ArrayView1<f64> = mono.dq.view();

    // Compute W = C·diag(f·ε)·C^T
    let occupations = Array1::from(f_occ.clone());
    let weighted_orbe = &orbe * &occupations;
    let worbe_2d = Array2::from_diag(&weighted_orbe);
    let w: Array2<f64> = orbs.dot(&worbe_2d.dot(&orbs.t()));

    // Coulomb shift: gamma_shell · dq_shell + esp_q_shell
    // For HOP: dq_shell is full-sized (includes ghost charges), no truncation needed
    let total_shift_shell: Array1<f64> = gamma_shell.dot(&dq_shell) + &esp_q_shell;
    let total_shift = shell_to_ao_values(&mono.basis, n_orbs, total_shift_shell.view());
    let coulomb_mat: Array2<f64> = aovec_to_aomat(total_shift.view(), n_orbs) * 0.5;

    // Third order using full dq and ext_atoms
    let hubbard_derivatives: Array1<f64> = mono
        .ext_atoms
        .iter()
        .map(|a| COUL_THIRD_ORDER_ATOM[a.number as usize - 1])
        .collect();
    let dq2_gamma =
        coul_third_order_grad_contribution_xtb(&mono.basis, dq, hubbard_derivatives.view());

    // Effective matrix: -W + coulomb*P - 0.5*third*P
    let effective_mat: Array2<f64> = &(-&w) + &(&coulomb_mat * &p) - &(0.5 * &dq2_gamma * &p);

    // Addlag: shiftct_ao*P - 0.5*P*(shiftct_ao*S)*P
    let shiftct_ao = shell_to_ao_values(&mono.basis, n_orbs, shiftct_shell);
    let shiftct_ao_mat: Array2<f64> = aovec_to_aomat(shiftct_ao.view(), n_orbs) * 0.5;
    let shift_s: Array2<f64> = &shiftct_ao_mat * &s;
    let d_shift_s: Array2<f64> = p.dot(&shift_s);
    let d_shift_s_d: Array2<f64> = d_shift_s.dot(&p);
    let wrk1_addlag: Array2<f64> = &(&shiftct_ao_mat * &p) - &(0.5 * &d_shift_s_d);

    // CN numbers: real atoms from global, ghost = 0
    let atom_start = frag_atom_range.start;
    let mut cn_numbers = Array1::<f64>::zeros(n_atoms);
    cn_numbers
        .slice_mut(s![..n_real_atoms])
        .assign(&cn_numbers_global.slice(s![frag_atom_range.clone()]));

    let mut grad_monomer = Array1::<f64>::zeros(3 * n_atoms);
    let mut grad_addlag = Array1::<f64>::zeros(3 * n_atoms);
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
            let mut shell_ds_addlag_contrib: [f64; 3] = [0.0; 3];

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
                        if at_i != at_j {
                            if shell_i.angular_momentum < 2 && shell_j.angular_momentum < 2 {
                                let orbital1 =
                                    &mono.basis.basis_functions[shell_i.start + idx_i_local];
                                let orbital2 =
                                    &mono.basis.basis_functions[shell_j.start + idx_j_local];
                                let norm_prod = orbital1.contracted_norm * orbital2.contracted_norm;
                                let eff_ij = effective_mat[[idx_i, idx_j]];
                                let combined_factor = h0_val * p_ij + eff_ij;
                                let add_factor = wrk1_addlag[[idx_i, idx_j]];
                                let ds_all = obara_saika_derivatives_all(orbital1, orbital2);
                                for dir in 0..3 {
                                    shell_ds_contrib[dir] +=
                                        ds_all[dir] * norm_prod * combined_factor;
                                    shell_ds_addlag_contrib[dir] +=
                                        ds_all[dir] * norm_prod * add_factor;
                                }
                                shell_pi_sp_sum += s_ij * p_ij;
                            }
                        }
                    }
                }
            }

            // D-orbital handling
            let shell_i_has_d = shell_i.angular_momentum >= 2;
            let shell_j_has_d = shell_j.angular_momentum >= 2;
            let either_has_d = shell_i_has_d || shell_j_has_d;
            if at_i != at_j && either_has_d && shell_i_idx < shell_j_idx {
                let ds_d = calc_overlap_derivative_d_shells(&mono.basis, shell_i, shell_j);
                let sph_dim_i = shell_i.sph_end - shell_i.sph_start;
                let sph_dim_j = shell_j.sph_end - shell_j.sph_start;
                for sph_i in 0..sph_dim_i {
                    let idx_i = shell_i.sph_start + sph_i;
                    for sph_j in 0..sph_dim_j {
                        let idx_j = shell_j.sph_start + sph_j;
                        let p_ij = p[[idx_i, idx_j]];
                        let eff_ij = effective_mat[[idx_i, idx_j]];
                        let add_ij = wrk1_addlag[[idx_i, idx_j]];
                        for dir in 0..3 {
                            let ds_val_i = 2.0 * ds_d[[dir, sph_i, sph_j]];
                            let ds_val_j = 2.0 * ds_d[[3 + dir, sph_i, sph_j]];
                            let combined_factor = h0_val * p_ij + eff_ij;
                            grad_monomer[3 * at_i + dir] += ds_val_i * combined_factor;
                            grad_monomer[3 * at_j + dir] += ds_val_j * combined_factor;
                            grad_addlag[3 * at_i + dir] += ds_val_i * add_ij;
                            grad_addlag[3 * at_j + dir] += ds_val_j * add_ij;
                        }
                        shell_pi_sp_sum += s[[idx_i, idx_j]] * p_ij;
                    }
                }
            }

            if at_i != at_j {
                for dir in 0..3 {
                    grad_monomer[3 * at_i + dir] += shell_ds_contrib[dir];
                    grad_monomer[3 * at_j + dir] -= shell_ds_contrib[dir];
                    grad_addlag[3 * at_i + dir] += shell_ds_addlag_contrib[dir];
                    grad_addlag[3 * at_j + dir] -= shell_ds_addlag_contrib[dir];
                }
                let pi_contrib = 2.0 * pi_factor * shell_pi_sp_sum;
                for dir in 0..3 {
                    grad_monomer[3 * at_i + dir] += pi_grad[dir] * pi_contrib;
                }
                if either_has_d && shell_i_idx < shell_j_idx {
                    for dir in 0..3 {
                        grad_monomer[3 * at_j + dir] -= pi_grad[dir] * pi_contrib;
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

    // CN gradient: only real atoms have global CN indices
    let mut cn_grad_contribution = Array1::<f64>::zeros(3 * n_atoms_total);
    for at in 0..n_real_atoms {
        if cn_factors[at].abs() > 1e-15 {
            let global_at = atom_start + at;
            let cn_grad_at: ArrayView1<f64> = cn_grad_global.slice(s![.., global_at]);
            for k in 0..(3 * n_atoms_total) {
                cn_grad_contribution[k] += cn_factors[at] * cn_grad_at[k];
            }
        }
    }

    // Gamma gradient using full dq_shell (including ghost charges)
    for (shell_i_idx, shell_i) in mono.basis.shells.iter().enumerate() {
        let atomi = &mono.ext_atoms[shell_i.atom_index];
        let at_i = shell_i.atom_index;
        for (shell_j_idx, shell_j) in mono.basis.shells.iter().enumerate() {
            let atomj = &mono.ext_atoms[shell_j.atom_index];
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
                let shell_dq_prod = dq_shell[shell_i_idx] * dq_shell[shell_j_idx];
                let shell_gamma_contrib = 0.5 * gamma_deriv * shell_dq_prod;
                for dir in 0..3 {
                    grad_monomer[3 * at_i + dir] += e_ij[dir] * shell_gamma_contrib;
                    grad_monomer[3 * at_j + dir] -= e_ij[dir] * shell_gamma_contrib;
                }
            }
        }
    }

    // Repulsive energy gradient with ZREF/QREF scaling
    let fi = &hop_data.frag_info[frag_idx];
    let zref = hop_data.zref.slice(s![fi.ext_range.clone()]);
    let qref = hop_data.qref.slice(s![fi.ext_range.clone()]);
    let grad_rep = grad_repulsive_energy_xtb_scaled(&mono.ext_atoms, zref, qref);
    grad_monomer += &grad_rep;

    (grad_monomer, grad_addlag, cn_grad_contribution)
}
