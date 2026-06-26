//! Pair gradient for FMO-xTB HOP.
//!
//! Reads from `XtbPairHopScc` struct. Returns (pair_grad_local, ctij_grad_global, cn_grad_global).

use super::helpers::{compute_esp_q_shell_hop, grad_repulsive_energy_xtb_scaled, shells_per_atom_in_range};
use dialect_base::defaults::PROXIMITY_CUTOFF;
use dialect_utilities::mulliken::shell_to_ao_values;
use crate::fmo::gradients::fmo_gradient_shell::{
    get_pi_term_gradient_inline_shell, get_self_energy_cn_grad_coeff_shell,
};
use crate::fmo::scc_hop::hop_data::{get_frag_shell_range, XtbHopData};
use crate::fmo::scc_hop::monomer::XtbMonomerHopScc;
use crate::fmo::scc_hop::pair::XtbPairHopScc;
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

/// Combined pair gradient: pair SCC + CTIJ.
///
/// Returns (pair_grad_local, ctij_grad_global, cn_grad_contribution):
/// - pair_grad_local: SCC + repulsive [3*n_ext_atoms_pair]
/// - ctij_grad_global: CTIJ gamma derivative in global coords [3*n_atoms_total]
/// - cn_grad_contribution: CN gradient projected to global [3*n_atoms_total]
pub fn pair_gradient_combined_xtb_hop(
    ps: &XtbPairHopScc,
    hop_data: &XtbHopData,
    mono_states: &[XtbMonomerHopScc],
    n_atoms_total: usize,
    gammafunction: &XtbGammaFunction,
    cn_numbers_global: ArrayView1<f64>,
    cn_grad_global: ArrayView2<f64>,
    local_to_global: &[usize],
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n_atoms_pair = ps.n_ext_atoms;
    let n_orbs = ps.n_ext_orbs;
    let n_real_atoms_pair = ps.n_real_atoms;
    let n_real_shells_i = ps.n_real_shells_i;
    let n_real_shells_j = ps.n_real_shells_j;
    let n_real_shells_pair = n_real_shells_i + n_real_shells_j;
    let n_shells_pair = ps.basis.shells.len();

    // Direct access to SCC results
    let p: ArrayView2<f64> = ps.p.view();
    let s: ArrayView2<f64> = ps.s.view();
    let orbe: ArrayView1<f64> = ps.orbe.as_ref().unwrap().view();
    let orbs: ArrayView2<f64> = ps.orbs.as_ref().unwrap().view();
    let occupations = Array1::from(ps.f.clone());
    let gamma_shell: ArrayView2<f64> = ps.gamma_shell.view();
    let dq_shell_pair: ArrayView1<f64> = ps.dq_shell.view();
    let dq: ArrayView1<f64> = ps.dq.view();

    // Compute W
    let weighted_orbe = &orbe * &occupations;
    let worbe_2d = Array2::from_diag(&weighted_orbe);
    let w: Array2<f64> = orbs.dot(&worbe_2d.dot(&orbs.t()));

    // Compute total shift at shell level: gamma_shell · dq_shell (internal) + ESP from K≠I,J
    let gamma_dq_shell: Array1<f64> = gamma_shell.dot(&dq_shell_pair);

    // ESP from K≠I,J: reconstruct from hop_data
    // esp_from_k_i = esp_q_I - gamma_ext_IJ · dq_ext_J (all shells, real+ghost)
    // esp_from_k_j = esp_q_J - gamma_ext_JI · dq_ext_I (all shells, real+ghost)
    let esp_q_i = compute_esp_q_shell_hop(ps.i, hop_data);
    let esp_q_j = compute_esp_q_shell_hop(ps.j, hop_data);

    let fi_i = &hop_data.frag_info[ps.i];
    let fi_j = &hop_data.frag_info[ps.j];

    // Ext gamma and shell ranges (needed for ghost ESP subtraction)
    let gamma_shell_ext = &hop_data.gamma_shell_ext;
    let ext_shell_range_i = get_frag_shell_range(&hop_data.ext_basis, &fi_i.ext_range);
    let ext_shell_range_j = get_frag_shell_range(&hop_data.ext_basis, &fi_j.ext_range);

    // Monomer dq_shell (real shells only)
    let dq_mon_i_shell = mono_states[ps.i].dq_shell.slice(s![..fi_i.n_real_shells]);
    let dq_mon_j_shell = mono_states[ps.j].dq_shell.slice(s![..fi_j.n_real_shells]);

    // gamma cross-block from pair's local gamma (real shells only)
    let gamma_ij = gamma_shell.slice(s![..n_real_shells_i, n_real_shells_i..n_real_shells_pair]);

    // esp_q from hop_data has n_ext_shells_I entries; we only need n_real_shells_I
    // Start with real-shell subtraction
    let mut esp_from_k_i: Array1<f64> =
        &esp_q_i.slice(s![..fi_i.n_real_shells]) - &gamma_ij.dot(&dq_mon_j_shell);
    let mut esp_from_k_j: Array1<f64> =
        &esp_q_j.slice(s![..fi_j.n_real_shells]) - &gamma_ij.t().dot(&dq_mon_i_shell);

    // Also subtract J's GHOST shell ESP contribution from esp_from_k_i
    // (the SCC uses ext gamma with all shells, but above only subtracted real shells)
    if fi_j.n_ghost_atoms > 0 {
        let j_ghost_shell_start = ext_shell_range_j.start + fi_j.n_real_shells;
        let j_ghost_shell_end = ext_shell_range_j.end;
        let gamma_ireal_jghost = gamma_shell_ext.slice(s![
            ext_shell_range_i.start..ext_shell_range_i.start + fi_i.n_real_shells,
            j_ghost_shell_start..j_ghost_shell_end
        ]);
        let dq_j_ghost = hop_data.dq_shell_ext.slice(s![j_ghost_shell_start..j_ghost_shell_end]);
        esp_from_k_i -= &gamma_ireal_jghost.dot(&dq_j_ghost);
    }
    // Also subtract I's GHOST shell ESP contribution from esp_from_k_j
    if fi_i.n_ghost_atoms > 0 {
        let i_ghost_shell_start = ext_shell_range_i.start + fi_i.n_real_shells;
        let i_ghost_shell_end = ext_shell_range_i.end;
        let gamma_jreal_ighost = gamma_shell_ext.slice(s![
            ext_shell_range_j.start..ext_shell_range_j.start + fi_j.n_real_shells,
            i_ghost_shell_start..i_ghost_shell_end
        ]);
        let dq_i_ghost = hop_data.dq_shell_ext.slice(s![i_ghost_shell_start..i_ghost_shell_end]);
        esp_from_k_j -= &gamma_jreal_ighost.dot(&dq_i_ghost);
    }

    // Combine: real shells get gamma_dq + ESP, ghost shells get just gamma_dq
    let mut total_shift_shell = Array1::<f64>::zeros(n_shells_pair);
    total_shift_shell
        .slice_mut(s![..n_real_shells_i])
        .assign(&(&gamma_dq_shell.slice(s![..n_real_shells_i]) + &esp_from_k_i));
    total_shift_shell
        .slice_mut(s![n_real_shells_i..n_real_shells_pair])
        .assign(
            &(&gamma_dq_shell.slice(s![n_real_shells_i..n_real_shells_pair]) + &esp_from_k_j),
        );
    // Ghost shells: gamma_dq + ESP from K≠I,J (must match SCC preparation)
    if n_shells_pair > n_real_shells_pair {
        total_shift_shell
            .slice_mut(s![n_real_shells_pair..])
            .assign(&gamma_dq_shell.slice(s![n_real_shells_pair..]));

        let mut pair_ghost_idx = 0usize;
        for bond in &hop_data.detached_bonds {
            let bda_in = bond.bda_fragment == ps.i || bond.bda_fragment == ps.j;
            let baa_in = bond.baa_fragment == ps.i || bond.baa_fragment == ps.j;
            if baa_in && !bda_in {
                // Ghost at BDA position; BDA is in an external fragment
                let bda_frag = bond.bda_fragment;
                let bda_frag_start = hop_data.monomer_indices[bda_frag][0];
                let bda_local_in_frag = bond.bda_global - bda_frag_start;
                let bda_ext_idx =
                    hop_data.frag_info[bda_frag].ext_range.start + bda_local_in_frag;

                // BDA's shells in ext_basis
                let bda_ext_shells: Vec<usize> = hop_data
                    .ext_basis
                    .shells
                    .iter()
                    .enumerate()
                    .filter(|(_, sh)| sh.atom_index == bda_ext_idx)
                    .map(|(idx, _)| idx)
                    .collect();

                // Ghost's shells in pair basis
                let ghost_local = n_real_atoms_pair + pair_ghost_idx;
                let ghost_pair_shells: Vec<usize> = ps
                    .basis
                    .shells
                    .iter()
                    .enumerate()
                    .filter(|(_, sh)| sh.atom_index == ghost_local)
                    .map(|(idx, _)| idx)
                    .collect();

                for (&bda_sh, &ghost_sh) in
                    bda_ext_shells.iter().zip(ghost_pair_shells.iter())
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
                    total_shift_shell[ghost_sh] += full_esp - esp_from_i - esp_from_j;
                }

                pair_ghost_idx += 1;
            }
        }
    }
    let total_shift = shell_to_ao_values(&ps.basis, n_orbs, total_shift_shell.view());
    let coulomb_mat: Array2<f64> = aovec_to_aomat(total_shift.view(), n_orbs) * 0.5;

    // Third order
    let hubbard_derivatives: Array1<f64> = ps
        .ext_atoms
        .iter()
        .map(|a| COUL_THIRD_ORDER_ATOM[a.number as usize - 1])
        .collect();
    let dq2_gamma =
        coul_third_order_grad_contribution_xtb(&ps.basis, dq, hubbard_derivatives.view());

    let effective_mat: Array2<f64> = &(-&w) + &(&coulomb_mat * &p) - &(0.5 * &dq2_gamma * &p);

    // CTIJ: delta_dq_shell_real (pair - monomer, real shells only)
    // Pad to full n_shells_pair
    let mut delta_dq_shell = Array1::<f64>::zeros(n_shells_pair);
    delta_dq_shell
        .slice_mut(s![..n_real_shells_pair])
        .assign(&ps.delta_dq_shell_real);

    // dq_mon_shell padded for pair ordering
    let mut dq_mon_shell_padded = Array1::<f64>::zeros(n_shells_pair);
    dq_mon_shell_padded
        .slice_mut(s![..n_real_shells_i])
        .assign(&dq_mon_i_shell);
    dq_mon_shell_padded
        .slice_mut(s![n_real_shells_i..n_real_shells_pair])
        .assign(&dq_mon_j_shell);

    // Ghost contributions to CTIJ:
    // - Partial bonds (BAA in pair, BDA outside → ghost in pair):
    //     delta_dq_shell[ghost] = pair.dq_shell[ghost] - mono.dq_shell[ghost]
    //     dq_mon_shell[ghost] = mono.dq_shell[ghost]
    // - Healed bonds (both in pair → no ghost in pair):
    //     delta_dq_shell[BDA_shell] -= mono.dq_shell[ghost_shell]
    //     dq_mon_shell[BDA_shell] += mono.dq_shell[ghost_shell]
    {
        let spa_i = shells_per_atom_in_range(&hop_data.ext_basis, &fi_i.ext_range);
        let spa_j = shells_per_atom_in_range(&hop_data.ext_basis, &fi_j.ext_range);
        let n_real_i_atoms = fi_i.n_real_atoms;
        let n_real_j_atoms = fi_j.n_real_atoms;

        let mut pair_ghost_shell_offset = n_real_shells_pair;
        let mut mono_ghost_shell_offset_i = fi_i.n_real_shells;
        let mut mono_ghost_shell_offset_j = fi_j.n_real_shells;
        let mut ghost_idx_i = 0usize;
        let mut ghost_idx_j = 0usize;

        for bond in &hop_data.detached_bonds {
            let bda_in_pair = bond.bda_fragment == ps.i || bond.bda_fragment == ps.j;

            if bond.baa_fragment == ps.i {
                let n_ghost_shells = spa_i[fi_i.n_real_atoms + ghost_idx_i];

                if bda_in_pair {
                    // Healed bond: ghost exists in monomer I but not in pair.
                    // BDA is in frag J (since baa=I, bda!=I => bda=J for healed).
                    // Adjust BDA's shells: subtract ghost dq from delta, add to dq_mon.
                    let bda_frag_start = hop_data.monomer_indices[ps.j][0];
                    let bda_local_in_pair = ps.n_real_i + (bond.bda_global - bda_frag_start);
                    let bda_pair_shells: Vec<usize> = ps
                        .basis
                        .shells
                        .iter()
                        .enumerate()
                        .filter(|(_, sh)| sh.atom_index == bda_local_in_pair)
                        .map(|(idx, _)| idx)
                        .collect();
                    for (k, &bda_sh) in bda_pair_shells.iter().enumerate() {
                        if k < n_ghost_shells {
                            let ghost_dq =
                                mono_states[ps.i].dq_shell[mono_ghost_shell_offset_i + k];
                            delta_dq_shell[bda_sh] -= ghost_dq;
                            dq_mon_shell_padded[bda_sh] += ghost_dq;
                        }
                    }
                } else {
                    // Partial bond: fill ghost entries directly
                    for k in 0..n_ghost_shells {
                        let pair_sh = pair_ghost_shell_offset + k;
                        let mono_sh = mono_ghost_shell_offset_i + k;
                        let pair_dq = ps.dq_shell[pair_sh];
                        let mono_dq = mono_states[ps.i].dq_shell[mono_sh];
                        delta_dq_shell[pair_sh] = pair_dq - mono_dq;
                        dq_mon_shell_padded[pair_sh] = mono_dq;
                    }
                    pair_ghost_shell_offset += n_ghost_shells;
                }
                mono_ghost_shell_offset_i += n_ghost_shells;
                ghost_idx_i += 1;
            } else if bond.baa_fragment == ps.j {
                let n_ghost_shells = spa_j[fi_j.n_real_atoms + ghost_idx_j];

                if bda_in_pair {
                    // Healed bond: ghost exists in monomer J but not in pair.
                    // BDA is in frag I (since baa=J, bda!=J => bda=I for healed).
                    let bda_frag_start = hop_data.monomer_indices[ps.i][0];
                    let bda_local_in_pair = bond.bda_global - bda_frag_start;
                    let bda_pair_shells: Vec<usize> = ps
                        .basis
                        .shells
                        .iter()
                        .enumerate()
                        .filter(|(_, sh)| sh.atom_index == bda_local_in_pair)
                        .map(|(idx, _)| idx)
                        .collect();
                    for (k, &bda_sh) in bda_pair_shells.iter().enumerate() {
                        if k < n_ghost_shells {
                            let ghost_dq =
                                mono_states[ps.j].dq_shell[mono_ghost_shell_offset_j + k];
                            delta_dq_shell[bda_sh] -= ghost_dq;
                            dq_mon_shell_padded[bda_sh] += ghost_dq;
                        }
                    }
                } else {
                    // Partial bond: fill ghost entries directly
                    for k in 0..n_ghost_shells {
                        let pair_sh = pair_ghost_shell_offset + k;
                        let mono_sh = mono_ghost_shell_offset_j + k;
                        let pair_dq = ps.dq_shell[pair_sh];
                        let mono_dq = mono_states[ps.j].dq_shell[mono_sh];
                        delta_dq_shell[pair_sh] = pair_dq - mono_dq;
                        dq_mon_shell_padded[pair_sh] = mono_dq;
                    }
                    pair_ghost_shell_offset += n_ghost_shells;
                }
                mono_ghost_shell_offset_j += n_ghost_shells;
                ghost_idx_j += 1;
            }
        }
    }

    // CN numbers: real atoms from global, ghost = 0
    let mut cn_numbers = Array1::<f64>::zeros(n_atoms_pair);
    for (local_idx, &global_idx) in local_to_global.iter().enumerate().take(n_real_atoms_pair) {
        cn_numbers[local_idx] = cn_numbers_global[global_idx];
    }

    let mut grad_local = Array1::<f64>::zeros(3 * n_atoms_pair);
    let mut ctij_grad_global = Array1::<f64>::zeros(3 * n_atoms_total);
    let mut cn_factors: Vec<f64> = vec![0.0; n_atoms_pair];

    // === Shell-pair loop ===
    for (shell_i_idx, shell_i) in ps.basis.shells.iter().enumerate() {
        let atomi = &ps.ext_atoms[shell_i.atom_index];
        let at_i = shell_i.atom_index;
        let cn_1 = cn_numbers[at_i];

        for (shell_j_idx, shell_j) in ps.basis.shells.iter().enumerate() {
            let atomj = &ps.ext_atoms[shell_j.atom_index];
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
                        if at_i != at_j {
                            if shell_i.angular_momentum < 2 && shell_j.angular_momentum < 2 {
                                let orbital1 =
                                    &ps.basis.basis_functions[shell_i.start + idx_i_local];
                                let orbital2 =
                                    &ps.basis.basis_functions[shell_j.start + idx_j_local];
                                let norm_prod = orbital1.contracted_norm * orbital2.contracted_norm;
                                let eff_ij = effective_mat[[idx_i, idx_j]];
                                let combined_factor = h0_val * p_ij + eff_ij;
                                let ds_all = obara_saika_derivatives_all(orbital1, orbital2);
                                for dir in 0..3 {
                                    shell_ds_contrib[dir] +=
                                        ds_all[dir] * norm_prod * combined_factor;
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
                let ds_d = calc_overlap_derivative_d_shells(&ps.basis, shell_i, shell_j);
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
                            let combined_factor = h0_val * p_ij + eff_ij;
                            grad_local[3 * at_i + dir] += ds_val_i * combined_factor;
                            grad_local[3 * at_j + dir] += ds_val_j * combined_factor;
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
    for at in 0..n_real_atoms_pair {
        if cn_factors[at].abs() > 1e-15 {
            let global_at = local_to_global[at];
            let cn_grad_at: ArrayView1<f64> = cn_grad_global.slice(s![.., global_at]);
            for k in 0..(3 * n_atoms_total) {
                cn_grad_contribution[k] += cn_factors[at] * cn_grad_at[k];
            }
        }
    }

    // Gamma gradient + CTIJ
    for (shell_i_idx, shell_i) in ps.basis.shells.iter().enumerate() {
        let atomi = &ps.ext_atoms[shell_i.atom_index];
        let at_i = shell_i.atom_index;
        for (shell_j_idx, shell_j) in ps.basis.shells.iter().enumerate() {
            let atomj = &ps.ext_atoms[shell_j.atom_index];
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
                // Pair Coulomb
                let shell_dq_prod = dq_shell_pair[shell_i_idx] * dq_shell_pair[shell_j_idx];
                let shell_gamma_contrib = 0.5 * gamma_deriv * shell_dq_prod;
                for dir in 0..3 {
                    grad_local[3 * at_i + dir] += e_ij[dir] * shell_gamma_contrib;
                    grad_local[3 * at_j + dir] -= e_ij[dir] * shell_gamma_contrib;
                }
                // CTIJ: global coords via local_to_global
                let shell_ctij_contrib =
                    -gamma_deriv * delta_dq_shell[shell_i_idx] * dq_mon_shell_padded[shell_j_idx];
                let global_i = local_to_global[at_i];
                let global_j = local_to_global[at_j];
                for dir in 0..3 {
                    ctij_grad_global[3 * global_i + dir] += e_ij[dir] * shell_ctij_contrib;
                    ctij_grad_global[3 * global_j + dir] -= e_ij[dir] * shell_ctij_contrib;
                }
            }
        }
    }

    // Repulsive energy gradient with ZREF/QREF scaling
    let grad_rep = grad_repulsive_energy_xtb_scaled(&ps.ext_atoms, ps.zref.view(), ps.qref.view());
    grad_local += &grad_rep;

    (grad_local, ctij_grad_global, cn_grad_contribution)
}
