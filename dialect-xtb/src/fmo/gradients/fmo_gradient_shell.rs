//! Low-memory FMO-xTB ground-state gradient implementation using shell-level gamma.
//!
//! Gradient formula:
//!   total = monomer + pair_delta + addlag + CTIJ + CTMUL_embed + ESD + dispersion

use dialect_base::defaults::PROXIMITY_CUTOFF;
use dialect_utilities::fmo_helpers::{get_pair_slice_xtb, get_trimer_slice_xtb};
use dialect_state::PairType;
use dialect_utilities::mulliken::{shell_to_ao_charges, shell_to_ao_values};
use crate::fmo::gradients::monomer::grad_repulsive_energy_xtb;
use crate::fmo::monomer::XtbMonomer;
use crate::fmo::pair::XtbESDPair;
use crate::fmo::pair::XtbPair;
use crate::fmo::supersystem::XtbSuperSystem;
use crate::fmo::trimer::XtbTrimer;
use crate::gradients::ground_state::aovec_to_aomat;
use crate::gradients::halogen_bonding::gradient_halogen_bonding_xtb;
use crate::gradients::helpers::{coul_third_order_grad_contribution_xtb, gradient_disp3_xtb};
use crate::initialization::atom::XtbAtom;
use crate::initialization::basis::create_basis_set;
use crate::initialization::basis::Basis;
use crate::integrals::calc_overlap_derivative_d_shells;
use crate::integrals::obara_saika_derivatives_all;
use crate::parameters::*;
use crate::scc::gamma_matrix::gamma_gradient_xtb_double_contracted;
use crate::scc::hamiltonian::{
    calculate_pair_scaling_param, get_hueckel_constants_new, get_pi_term,
    get_self_energy_values_new,
};
use nalgebra::Vector3;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;

// ============================================================================
// Helper functions
// ============================================================================

/// Compute CTMUL in shell basis: ctmul_shell[s] = sum over pairs/trimers of delta_dq_shell[s].
///
/// For each pair (I,J), delta_dq_shell = dq_pair_shell - concat(dq_I_shell, dq_J_shell).
/// For each trimer (I,J,K), delta_dq_shell = dq_trimer_shell - concat(dq_I_shell, dq_J_shell, dq_K_shell).
///
/// For FMO3, pairs are scaled by SCAL = 1 - n_trimers_containing_pair.
/// Trimers are always added with SCAL = 1.
pub fn compute_ctmul_shell(
    monomers: &[XtbMonomer],
    pairs: &[XtbPair],
    trimers: &[XtbTrimer],
    pair_scal: &[f64],
    n_shells_total: usize,
) -> Array1<f64> {
    let mut ctmul_shell = Array1::<f64>::zeros(n_shells_total);

    // Pair contributions (scaled by SCAL for FMO3)
    for (pair_idx, pair) in pairs.iter().enumerate() {
        let scal = pair_scal[pair_idx];
        if scal.abs() < 1e-14 {
            continue;
        }
        let m_i = &monomers[pair.i];
        let m_j = &monomers[pair.j];
        let delta_dq_shell: ArrayView1<f64> = pair.properties.delta_dq_shell().unwrap();
        let n_shells_i = m_i.n_real_shells;

        let mut slice_i = ctmul_shell.slice_mut(s![m_i.slice.shell]);
        slice_i += &(scal * &delta_dq_shell.slice(s![..n_shells_i]));

        let mut slice_j = ctmul_shell.slice_mut(s![m_j.slice.shell]);
        slice_j += &(scal * &delta_dq_shell.slice(s![n_shells_i..]));
    }

    // Trimer contributions (SCAL = 1)
    for trimer in trimers.iter() {
        let m_i = &monomers[trimer.i];
        let m_j = &monomers[trimer.j];
        let m_k = &monomers[trimer.k];
        let delta_dq_shell: ArrayView1<f64> = trimer.properties.delta_dq_shell().unwrap();
        let n_shells_i = m_i.n_real_shells;
        let n_shells_j = m_j.n_real_shells;

        let mut slice_i = ctmul_shell.slice_mut(s![m_i.slice.shell]);
        slice_i += &delta_dq_shell.slice(s![..n_shells_i]);

        let mut slice_j = ctmul_shell.slice_mut(s![m_j.slice.shell]);
        slice_j += &delta_dq_shell.slice(s![n_shells_i..n_shells_i + n_shells_j]);

        let mut slice_k = ctmul_shell.slice_mut(s![m_k.slice.shell]);
        slice_k += &delta_dq_shell.slice(s![n_shells_i + n_shells_j..]);
    }

    ctmul_shell
}

/// Compute SHIFTCT + ESPGRAD correction for a monomer, using shell-level gamma.
///
/// SHIFTCT_shell[s_I] = sum_t gamma_shell_super[s_I, t] * ctmul_shell[t]  (all shells t)
/// ESPGRAD_shell[s_I] -= sum_{t in pair} gamma_shell_super[s_I, t] * ctij_shell[t] * SCAL
/// ESPGRAD_shell[s_I] -= sum_{t in trimer} gamma_shell_super[s_I, t] * ctijk_shell[t]
///
/// Result is expanded to AO level via shell_to_ao_values for the caller.
fn compute_shiftct_espgrad_shell(
    m: &XtbMonomer,
    gamma_shell_super: ArrayView2<f64>,
    ctmul_shell: ArrayView1<f64>,
    monomers: &[XtbMonomer],
    pairs: &[XtbPair],
    trimers: &[XtbTrimer],
    pair_scal: &[f64],
) -> Array1<f64> {
    // SHIFTCT: gamma_shell_super[I_shells, :] @ ctmul_shell
    let gamma_i_all = gamma_shell_super.slice(s![m.slice.shell, ..]);
    let mut shiftct_shell: Array1<f64> = gamma_i_all.dot(&ctmul_shell);

    // ESPGRAD correction: subtract self-interaction for each pair containing this monomer
    for (pair_idx, pair) in pairs.iter().enumerate() {
        let scal = pair_scal[pair_idx];
        if scal.abs() < 1e-14 {
            continue;
        }

        let (is_mon_i, is_mon_j) = (pair.i == m.index, pair.j == m.index);
        if !is_mon_i && !is_mon_j {
            continue;
        }

        let m_i = &monomers[pair.i];
        let m_j = &monomers[pair.j];
        let delta_dq_shell: ArrayView1<f64> = pair.properties.delta_dq_shell().unwrap();
        let n_shells_i = m_i.n_real_shells;

        let gamma_i_mi = gamma_shell_super.slice(s![m.slice.shell, m_i.slice.shell]);
        let gamma_i_mj = gamma_shell_super.slice(s![m.slice.shell, m_j.slice.shell]);

        let ctij_on_i = delta_dq_shell.slice(s![..n_shells_i]);
        let ctij_on_j = delta_dq_shell.slice(s![n_shells_i..]);

        shiftct_shell -= &(scal * &gamma_i_mi.dot(&ctij_on_i));
        shiftct_shell -= &(scal * &gamma_i_mj.dot(&ctij_on_j));
    }

    // ESPGRAD correction for trimers (SCAL = 1)
    for trimer in trimers.iter() {
        let is_in = trimer.i == m.index || trimer.j == m.index || trimer.k == m.index;
        if !is_in {
            continue;
        }

        let m_i = &monomers[trimer.i];
        let m_j = &monomers[trimer.j];
        let m_k = &monomers[trimer.k];
        let delta_dq_shell: ArrayView1<f64> = trimer.properties.delta_dq_shell().unwrap();
        let n_shells_i = m_i.n_real_shells;
        let n_shells_j = m_j.n_real_shells;

        let gamma_m_mi = gamma_shell_super.slice(s![m.slice.shell, m_i.slice.shell]);
        let gamma_m_mj = gamma_shell_super.slice(s![m.slice.shell, m_j.slice.shell]);
        let gamma_m_mk = gamma_shell_super.slice(s![m.slice.shell, m_k.slice.shell]);

        let ctijk_on_i = delta_dq_shell.slice(s![..n_shells_i]);
        let ctijk_on_j = delta_dq_shell.slice(s![n_shells_i..n_shells_i + n_shells_j]);
        let ctijk_on_k = delta_dq_shell.slice(s![n_shells_i + n_shells_j..]);

        shiftct_shell -= &gamma_m_mi.dot(&ctijk_on_i);
        shiftct_shell -= &gamma_m_mj.dot(&ctijk_on_j);
        shiftct_shell -= &gamma_m_mk.dot(&ctijk_on_k);
    }

    // Expand to AO level for caller
    shell_to_ao_values(&m.basis, m.n_orbs, shiftct_shell.view())
}

/// Helper: get the self-energy CN gradient coefficient
pub fn get_self_energy_cn_grad_coeff_shell(z: u8, shell_idx: usize) -> f64 {
    let z_idx: usize = (z - 1) as usize;
    -HAMILTONIAN_KCN_VALUES[z_idx][shell_idx]
}

/// Get pi term gradient
pub fn get_pi_term_gradient_inline_shell(
    r_vector: &Vector3<f64>,
    r_ab: f64,
    z_1: usize,
    z_2: usize,
    l_1: usize,
    l_2: usize,
) -> [f64; 3] {
    use dialect_base::constants::BOHR_TO_ANGS;

    let z_idx_1: usize = z_1 - 1;
    let z_idx_2: usize = z_2 - 1;

    let k_poly_1: f64 = HAMILTONIAN_SHELL_POLYNOMIALS[z_idx_1][l_1] * 0.01;
    let k_poly_2: f64 = HAMILTONIAN_SHELL_POLYNOMIALS[z_idx_2][l_2] * 0.01;

    let cov_1: f64 = COV_RADII[z_idx_1] / BOHR_TO_ANGS;
    let cov_2: f64 = COV_RADII[z_idx_2] / BOHR_TO_ANGS;
    let cov_sum: f64 = cov_1 + cov_2;
    let distance_term: f64 = (r_ab / cov_sum).sqrt();

    let deriv_val: f64 = (1.0 + k_poly_1 * distance_term) * k_poly_2
        / (2.0 * cov_sum * distance_term)
        + (1.0 + k_poly_2 * distance_term) * k_poly_1 / (2.0 * distance_term * cov_sum);

    let inv_r = 1.0 / r_ab;
    [
        r_vector.x * inv_r * deriv_val,
        r_vector.y * inv_r * deriv_val,
        r_vector.z * inv_r * deriv_val,
    ]
}

// ============================================================================
// Helper: transition charges for response gradient
// ============================================================================

/// Compute AO-level virtual-occupied transition charges.
///
/// Returns array [n_orbs, nvirt*nocc].
pub fn compute_qvo_ao_shell(m: &XtbMonomer) -> Array2<f64> {
    let orbs: ArrayView2<f64> = m.properties.orbs().unwrap();
    let s: ArrayView2<f64> = m.properties.s().unwrap();
    let nocc = m.properties.occ_indices().unwrap().len();
    let nvirt = m.properties.virt_indices().unwrap().len();
    let n_orbs = m.n_orbs;

    let sc = s.dot(&orbs); // [n_orbs, n_mo]
    let c_occ = orbs.slice(s![.., ..nocc]); // [n_orbs, nocc]
    let c_virt = orbs.slice(s![.., nocc..nocc + nvirt]); // [n_orbs, nvirt]
    let sc_occ = sc.slice(s![.., ..nocc]); // [n_orbs, nocc]
    let sc_virt = sc.slice(s![.., nocc..nocc + nvirt]); // [n_orbs, nvirt]

    let mut qvo_ao = Array2::<f64>::zeros([n_orbs, nvirt * nocc]);
    for mu in 0..n_orbs {
        for a in 0..nvirt {
            for i in 0..nocc {
                let idx = a * nocc + i;
                qvo_ao[[mu, idx]] =
                    c_virt[[mu, a]] * sc_occ[[mu, i]] + c_occ[[mu, i]] * sc_virt[[mu, a]];
            }
        }
    }
    qvo_ao
}

/// Compute both AO-level and shell-level virtual-occupied transition charges.
///
/// Returns (qvo_ao[n_orbs, nvo], qvo_shell[n_shells, nvo]).
/// - qvo_ao: needed for Z→AO transformation and atom-level third-order coupling
/// - qvo_shell: needed for Lagrangian contraction, A-matrix product, Z-vector inter-fragment coupling
pub fn compute_qvo_shell(m: &XtbMonomer) -> (Array2<f64>, Array2<f64>) {
    let qvo_ao = compute_qvo_ao_shell(m);

    let nocc = m.properties.occ_indices().unwrap().len();
    let nvirt = m.properties.virt_indices().unwrap().len();
    let nvo = nvirt * nocc;
    let n_shells = m.basis.shells.len();

    // Aggregate to shell level
    let mut qvo_shell = Array2::<f64>::zeros([n_shells, nvo]);
    for (s_idx, shell) in m.basis.shells.iter().enumerate() {
        for mu in shell.sph_start..shell.sph_end {
            for ai in 0..nvo {
                qvo_shell[[s_idx, ai]] += qvo_ao[[mu, ai]];
            }
        }
    }

    (qvo_ao, qvo_shell)
}

// ============================================================================
// Function A: monomer_gradient_combined_shell
// ============================================================================

/// Combined monomer gradient: monomer SCC + addlag, fused into shell-pair loops.
///
/// Uses gamma_shell for shift computation instead of gamma_ao.
///
/// Returns (monomer_scc_grad, addlag_grad, cn_grad_global):
/// - monomer_scc_grad: dH·P + dS·(-W + coulomb·P - third·P) + dgamma/dR·dq·dq + v_rep [3*n_atoms_monomer]
/// - addlag_grad: dS·WRK1_addlag + dgamma_shell·ctij_shiftct_correction [3*n_atoms_monomer]
/// - cn_grad_global: CN gradient contribution in global coordinates [3*n_atoms_total]
fn monomer_gradient_combined_shell(
    m: &XtbMonomer,
    atoms: &[XtbAtom],
    shiftct: ArrayView1<f64>,
    cn_numbers_global: ArrayView1<f64>,
    cn_grad_global: ArrayView2<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let m_atoms = &atoms[m.slice.atom_as_range()];
    let n_atoms = m.n_atoms;
    let n_orbs = m.n_orbs;

    // Get SCC properties
    let p: ArrayView2<f64> = m.properties.p().unwrap();
    let dq: ArrayView1<f64> = m.properties.dq().unwrap();
    let s: ArrayView2<f64> = m.properties.s().unwrap();
    let orbe: ArrayView1<f64> = m.properties.orbe().unwrap();
    let orbs: ArrayView2<f64> = m.properties.orbs().unwrap();
    let occupations: Array1<f64> = Array::from(m.properties.occupation().unwrap().to_vec());

    // Shell-level properties for shift computation
    let gamma_shell: ArrayView2<f64> = m.properties.gamma_shell().unwrap();
    let dq_shell: ArrayView1<f64> = m.properties.dq_shell().unwrap();
    let esp_q_shell: ArrayView1<f64> = m.properties.esp_q().unwrap();

    // Compute energy-weighted density matrix W
    let weighted_orbe = &orbe * &occupations;
    let worbe_2d: Array2<f64> = Array2::from_diag(&weighted_orbe);
    let w: Array2<f64> = orbs.dot(&worbe_2d.dot(&orbs.t()));

    // Coulomb potential: gamma_shell . dq_shell + esp_q_shell, then expand to AO
    let total_shift_shell: Array1<f64> = gamma_shell.dot(&dq_shell) + &esp_q_shell;
    let total_shift: Array1<f64> = shell_to_ao_values(&m.basis, n_orbs, total_shift_shell.view());
    let coulomb_mat: Array2<f64> = aovec_to_aomat(total_shift.view(), n_orbs) * 0.5;

    // Third order terms
    let mut hubbard_derivatives: Array1<f64> = Array1::zeros(n_atoms);
    for (val, atom) in hubbard_derivatives.iter_mut().zip(m_atoms.iter()) {
        *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
    }
    let dq2_gamma: Array2<f64> =
        coul_third_order_grad_contribution_xtb(&m.basis, dq, hubbard_derivatives.view());

    // Effective matrix: -W + coulomb*P - 0.5*third*P
    let effective_mat: Array2<f64> = &(-&w) + &(&coulomb_mat * &p) - &(0.5 * &dq2_gamma * &p);

    // Addlag: WRK1_addlag = shiftct_ao * P - 0.5 * P * (shiftct_ao * S) * P
    let shiftct_ao_mat: Array2<f64> = aovec_to_aomat(shiftct, n_orbs) * 0.5;
    let shift_s: Array2<f64> = &shiftct_ao_mat * &s;
    let d_shift_s: Array2<f64> = p.dot(&shift_s);
    let d_shift_s_d: Array2<f64> = d_shift_s.dot(&p);
    let wrk1_addlag: Array2<f64> = &(&shiftct_ao_mat * &p) - &(0.5 * &d_shift_s_d);

    // Use global coordination numbers for this monomer's atoms
    let atom_range = m.slice.atom_as_range();
    let atom_start = atom_range.start;
    let cn_numbers: Array1<f64> = cn_numbers_global.slice(s![atom_range.clone()]).to_owned();
    let n_atoms_total = atoms.len();

    // Initialize gradients and CN factors
    let mut grad_monomer = Array1::<f64>::zeros(3 * n_atoms);
    let mut grad_addlag = Array1::<f64>::zeros(3 * n_atoms);
    let mut cn_factors: Vec<f64> = vec![0.0; n_atoms];

    // === Shell-pair loop: overlap-dependent terms ===
    for (shell_i_idx, shell_i) in m.basis.shells.iter().enumerate() {
        let atomi = &m_atoms[shell_i.atom_index];
        let at_i = shell_i.atom_index;
        let cn_1 = cn_numbers[at_i];

        for (shell_j_idx, shell_j) in m.basis.shells.iter().enumerate() {
            let atomj = &m_atoms[shell_j.atom_index];
            let at_j = shell_j.atom_index;
            let cn_2 = cn_numbers[at_j];

            let r_vector: Vector3<f64> = atomi - atomj;
            let distance: f64 = r_vector.norm();

            if distance >= PROXIMITY_CUTOFF {
                continue;
            }

            // Precompute self-energy term
            let self_energy_term = get_self_energy_values_new(
                atomi.number,
                atomj.number,
                cn_1,
                cn_2,
                shell_i.shell_index,
                shell_j.shell_index,
            );

            // CN gradient coefficients
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

            // Pi gradient
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

            // Shell-level accumulators
            let mut diag_sp_sum: f64 = 0.0;
            let mut off_sp_sum: f64 = 0.0;
            let mut shell_pi_sp_sum: f64 = 0.0;
            let mut shell_ds_contrib: [f64; 3] = [0.0; 3];
            let mut shell_ds_addlag_contrib: [f64; 3] = [0.0; 3];

            // Loop over AO pairs within shell pair
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
                                    &m.basis.basis_functions[shell_i.start + idx_i_local];
                                let orbital2 =
                                    &m.basis.basis_functions[shell_j.start + idx_j_local];
                                let norm_prod = orbital1.contracted_norm * orbital2.contracted_norm;

                                // SCC: h0*P + effective_mat
                                let eff_ij = effective_mat[[idx_i, idx_j]];
                                let combined_factor = h0_val * p_ij + eff_ij;

                                // Addlag factor
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
                let ds_d = calc_overlap_derivative_d_shells(&m.basis, shell_i, shell_j);
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

            // Apply shell-level contributions
            if at_i != at_j {
                for dir in 0..3 {
                    grad_monomer[3 * at_i + dir] += shell_ds_contrib[dir];
                    grad_monomer[3 * at_j + dir] -= shell_ds_contrib[dir];

                    grad_addlag[3 * at_i + dir] += shell_ds_addlag_contrib[dir];
                    grad_addlag[3 * at_j + dir] -= shell_ds_addlag_contrib[dir];
                }

                // Pi gradient
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

            // Deferred CN gradient accumulation
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

    // Apply deferred CN gradient contributions using global CN gradients.
    let mut cn_grad_contribution = Array1::<f64>::zeros(3 * n_atoms_total);
    for at in 0..n_atoms {
        if cn_factors[at].abs() > 1e-15 {
            let global_at = atom_start + at;
            let cn_grad_at: ArrayView1<f64> = cn_grad_global.slice(s![.., global_at]);
            for k in 0..(3 * n_atoms_total) {
                cn_grad_contribution[k] += cn_factors[at] * cn_grad_at[k];
            }
        }
    }

    // Gamma gradient - separate loop without PROXIMITY_CUTOFF
    // Uses stored dq_shell directly instead of computing from dq_ao
    for (shell_i_idx, shell_i) in m.basis.shells.iter().enumerate() {
        let atomi = &m_atoms[shell_i.atom_index];
        let at_i = shell_i.atom_index;

        for (shell_j_idx, shell_j) in m.basis.shells.iter().enumerate() {
            let atomj = &m_atoms[shell_j.atom_index];
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

                let gamma_deriv = m.gammafunction.deriv(
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

    // Repulsive energy gradient
    let grad_rep = grad_repulsive_energy_xtb(m_atoms, n_atoms);
    grad_monomer += &grad_rep;

    (grad_monomer, grad_addlag, cn_grad_contribution)
}

// ============================================================================
// Function B: pair_gradient_combined_shell
// ============================================================================

/// Combined pair gradient: pair SCC + CTIJ, fused into shell-pair loops.
///
/// Uses gamma_shell for shift computation instead of gamma_ao.
///
/// Returns (pair_grad_local, ctij_grad_global, cn_grad_global).
fn pair_gradient_combined_shell(
    pair: &XtbPair,
    pair_atoms: &[XtbAtom],
    m_i: &XtbMonomer,
    m_j: &XtbMonomer,
    atoms: &[XtbAtom],
    cn_numbers_global: ArrayView1<f64>,
    cn_grad_global: ArrayView2<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n_atoms_pair = pair.n_atoms;
    let n_orbs = pair.n_orbs;
    let n_atoms_i = m_i.n_atoms;
    let n_atoms_total = atoms.len();

    // Get SCC properties
    let p: ArrayView2<f64> = pair.properties.p().unwrap();
    let dq: ArrayView1<f64> = pair.properties.dq().unwrap();
    let s: ArrayView2<f64> = pair.properties.s().unwrap();
    let orbe: ArrayView1<f64> = pair.properties.orbe().unwrap();
    let orbs: ArrayView2<f64> = pair.properties.orbs().unwrap();
    let occupations: Array1<f64> = Array::from(pair.properties.occupation().unwrap().to_vec());

    // Shell-level properties for shift computation
    let gamma_shell: ArrayView2<f64> = pair.properties.gamma_shell().unwrap();
    let dq_shell_pair: ArrayView1<f64> = pair.properties.dq_shell().unwrap();
    let n_shells_i = m_i.n_shells;
    let n_shells_pair = pair.basis.shells.len();

    // Compute W
    let weighted_orbe = &orbe * &occupations;
    let worbe_2d: Array2<f64> = Array2::from_diag(&weighted_orbe);
    let w: Array2<f64> = orbs.dot(&worbe_2d.dot(&orbs.t()));

    // Compute total shift at shell level
    let gamma_dq_shell: Array1<f64> = gamma_shell.dot(&dq_shell_pair);

    // ESP from K≠I,J: monomer esp_q (shell) minus inter-monomer gamma_shell block
    let gamma_ij_shell = gamma_shell.slice(s![..n_shells_i, n_shells_i..]);
    let dq_mon_i_shell: ArrayView1<f64> = m_i.properties.dq_shell().unwrap();
    let dq_mon_j_shell: ArrayView1<f64> = m_j.properties.dq_shell().unwrap();
    let esp_i_shell: ArrayView1<f64> = m_i.properties.esp_q().unwrap();
    let esp_j_shell: ArrayView1<f64> = m_j.properties.esp_q().unwrap();

    let esp_from_k_shell_i: Array1<f64> = &esp_i_shell - &gamma_ij_shell.dot(&dq_mon_j_shell);
    let esp_from_k_shell_j: Array1<f64> = &esp_j_shell - &gamma_ij_shell.t().dot(&dq_mon_i_shell);

    // Combine at shell level, then expand to AO
    let mut total_shift_shell = Array1::<f64>::zeros(n_shells_pair);
    total_shift_shell
        .slice_mut(s![..n_shells_i])
        .assign(&(&gamma_dq_shell.slice(s![..n_shells_i]) + &esp_from_k_shell_i));
    total_shift_shell
        .slice_mut(s![n_shells_i..])
        .assign(&(&gamma_dq_shell.slice(s![n_shells_i..]) + &esp_from_k_shell_j));
    let total_shift: Array1<f64> =
        shell_to_ao_values(&pair.basis, n_orbs, total_shift_shell.view());
    let coulomb_mat: Array2<f64> = aovec_to_aomat(total_shift.view(), n_orbs) * 0.5;

    // Third order terms
    let mut hubbard_derivatives: Array1<f64> = Array1::zeros(n_atoms_pair);
    for (val, atom) in hubbard_derivatives.iter_mut().zip(pair_atoms.iter()) {
        *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
    }
    let dq2_gamma: Array2<f64> =
        coul_third_order_grad_contribution_xtb(&pair.basis, dq, hubbard_derivatives.view());

    // Effective matrix
    let effective_mat: Array2<f64> = &(-&w) + &(&coulomb_mat * &p) - &(0.5 * &dq2_gamma * &p);

    // Shell-level CTIJ and monomer charges for gamma derivative
    // Use stored properties instead of summing from AO
    let delta_dq_shell: ArrayView1<f64> = pair.properties.delta_dq_shell().unwrap();

    // Build dq_mon_shell by concatenating monomer shell charges
    let mut dq_mon_shell_vec: Vec<f64> = Vec::with_capacity(n_shells_pair);
    for &val in dq_mon_i_shell.iter() {
        dq_mon_shell_vec.push(val);
    }
    for &val in dq_mon_j_shell.iter() {
        dq_mon_shell_vec.push(val);
    }

    // Local-to-global atom index mapping
    let local_to_global: Vec<usize> = m_i
        .slice
        .atom_as_range()
        .chain(m_j.slice.atom_as_range())
        .collect();

    // Use global coordination numbers for the pair's atoms
    let cn_numbers: Array1<f64> = {
        let mut cn = Array1::<f64>::zeros(n_atoms_pair);
        for (local_idx, &global_idx) in local_to_global.iter().enumerate() {
            cn[local_idx] = cn_numbers_global[global_idx];
        }
        cn
    };

    // Initialize gradients
    let mut grad_local = Array1::<f64>::zeros(3 * n_atoms_pair);
    let mut ctij_grad_global = Array1::<f64>::zeros(3 * n_atoms_total);
    let mut cn_factors: Vec<f64> = vec![0.0; n_atoms_pair];

    // === Shell-pair loop ===
    for (shell_i_idx, shell_i) in pair.basis.shells.iter().enumerate() {
        let atomi = &pair_atoms[shell_i.atom_index];
        let at_i = shell_i.atom_index;
        let cn_1 = cn_numbers[at_i];

        for (shell_j_idx, shell_j) in pair.basis.shells.iter().enumerate() {
            let atomj = &pair_atoms[shell_j.atom_index];
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
                                    &pair.basis.basis_functions[shell_i.start + idx_i_local];
                                let orbital2 =
                                    &pair.basis.basis_functions[shell_j.start + idx_j_local];
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
                let ds_d = calc_overlap_derivative_d_shells(&pair.basis, shell_i, shell_j);
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

            // Apply shell-level contributions
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

            // CN factors
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

    // Apply deferred CN gradient using global CN gradients
    let mut cn_grad_contribution = Array1::<f64>::zeros(3 * n_atoms_total);
    for at in 0..n_atoms_pair {
        if cn_factors[at].abs() > 1e-15 {
            let global_at = local_to_global[at];
            let cn_grad_at: ArrayView1<f64> = cn_grad_global.slice(s![.., global_at]);
            for k in 0..(3 * n_atoms_total) {
                cn_grad_contribution[k] += cn_factors[at] * cn_grad_at[k];
            }
        }
    }

    // Gamma gradient + CTIJ gamma gradient — separate loop without PROXIMITY_CUTOFF
    for (shell_i_idx, shell_i) in pair.basis.shells.iter().enumerate() {
        let atomi = &pair_atoms[shell_i.atom_index];
        let at_i = shell_i.atom_index;

        for (shell_j_idx, shell_j) in pair.basis.shells.iter().enumerate() {
            let atomj = &pair_atoms[shell_j.atom_index];
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

                let gamma_deriv = pair.gammafunction.deriv(
                    distance,
                    atomi.number,
                    shell_i.angular_momentum as u8,
                    atomj.number,
                    shell_j.angular_momentum as u8,
                );

                // Pair Coulomb: 0.5 * dgamma * dq_pair_shell[i] * dq_pair_shell[j]
                let shell_dq_prod = dq_shell_pair[shell_i_idx] * dq_shell_pair[shell_j_idx];
                let shell_gamma_contrib = 0.5 * gamma_deriv * shell_dq_prod;
                for dir in 0..3 {
                    grad_local[3 * at_i + dir] += e_ij[dir] * shell_gamma_contrib;
                    grad_local[3 * at_j + dir] -= e_ij[dir] * shell_gamma_contrib;
                }

                // CTIJ: -dgamma * ctij_shell[i] * dq_mon_shell[j]
                let shell_ctij_contrib =
                    -gamma_deriv * delta_dq_shell[shell_i_idx] * dq_mon_shell_vec[shell_j_idx];

                let global_i = local_to_global[at_i];
                let global_j = local_to_global[at_j];

                for dir in 0..3 {
                    ctij_grad_global[3 * global_i + dir] += e_ij[dir] * shell_ctij_contrib;
                    ctij_grad_global[3 * global_j + dir] -= e_ij[dir] * shell_ctij_contrib;
                }
            }
        }
    }

    // Repulsive energy gradient
    let grad_rep = grad_repulsive_energy_xtb(pair_atoms, n_atoms_pair);
    grad_local += &grad_rep;

    (grad_local, ctij_grad_global, cn_grad_contribution)
}

// ============================================================================
// Function B2: trimer_gradient_combined_shell (for FMO3)
// ============================================================================

/// Combined trimer gradient using gamma_shell for shift computation.
fn trimer_gradient_combined_shell(
    trimer: &XtbTrimer,
    trimer_atoms: &[XtbAtom],
    m_i: &XtbMonomer,
    m_j: &XtbMonomer,
    m_k: &XtbMonomer,
    atoms: &[XtbAtom],
    cn_numbers_global: ArrayView1<f64>,
    cn_grad_global: ArrayView2<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n_atoms_trimer = trimer.n_atoms;
    let n_orbs = trimer.n_orbs;
    let n_atoms_total = atoms.len();

    // Get SCC properties
    let p: ArrayView2<f64> = trimer.properties.p().unwrap();
    let dq: ArrayView1<f64> = trimer.properties.dq().unwrap();
    let s: ArrayView2<f64> = trimer.properties.s().unwrap();
    let orbe: ArrayView1<f64> = trimer.properties.orbe().unwrap();
    let orbs: ArrayView2<f64> = trimer.properties.orbs().unwrap();
    let occupations: Array1<f64> = Array::from(trimer.properties.occupation().unwrap().to_vec());

    // Shell-level properties
    let gamma_shell: ArrayView2<f64> = trimer.properties.gamma_shell().unwrap();
    let dq_shell_trimer: ArrayView1<f64> = trimer.properties.dq_shell().unwrap();
    let n_s_i = m_i.n_shells;
    let n_s_j = m_j.n_shells;
    let n_shells_trimer = trimer.basis().shells.len();

    // Compute W
    let weighted_orbe = &orbe * &occupations;
    let worbe_2d: Array2<f64> = Array2::from_diag(&weighted_orbe);
    let w: Array2<f64> = orbs.dot(&worbe_2d.dot(&orbs.t()));

    // Compute total shift at shell level
    let gamma_dq_shell: Array1<f64> = gamma_shell.dot(&dq_shell_trimer);

    // ESP from L≠I,J,K: subtract inter-monomer gamma_shell contributions from monomer esp_q
    let dq_mon_i_shell: ArrayView1<f64> = m_i.properties.dq_shell().unwrap();
    let dq_mon_j_shell: ArrayView1<f64> = m_j.properties.dq_shell().unwrap();
    let dq_mon_k_shell: ArrayView1<f64> = m_k.properties.dq_shell().unwrap();

    let gamma_ij_shell = gamma_shell.slice(s![..n_s_i, n_s_i..n_s_i + n_s_j]);
    let gamma_ik_shell = gamma_shell.slice(s![..n_s_i, n_s_i + n_s_j..]);
    let gamma_jk_shell = gamma_shell.slice(s![n_s_i..n_s_i + n_s_j, n_s_i + n_s_j..]);

    let esp_i_shell: ArrayView1<f64> = m_i.properties.esp_q().unwrap();
    let esp_j_shell: ArrayView1<f64> = m_j.properties.esp_q().unwrap();
    let esp_k_shell: ArrayView1<f64> = m_k.properties.esp_q().unwrap();

    let esp_from_l_i: Array1<f64> =
        &esp_i_shell - &gamma_ij_shell.dot(&dq_mon_j_shell) - &gamma_ik_shell.dot(&dq_mon_k_shell);
    let esp_from_l_j: Array1<f64> = &esp_j_shell
        - &gamma_ij_shell.t().dot(&dq_mon_i_shell)
        - &gamma_jk_shell.dot(&dq_mon_k_shell);
    let esp_from_l_k: Array1<f64> = &esp_k_shell
        - &gamma_ik_shell.t().dot(&dq_mon_i_shell)
        - &gamma_jk_shell.t().dot(&dq_mon_j_shell);

    let mut total_shift_shell = Array1::<f64>::zeros(n_shells_trimer);
    total_shift_shell
        .slice_mut(s![..n_s_i])
        .assign(&(&gamma_dq_shell.slice(s![..n_s_i]) + &esp_from_l_i));
    total_shift_shell
        .slice_mut(s![n_s_i..n_s_i + n_s_j])
        .assign(&(&gamma_dq_shell.slice(s![n_s_i..n_s_i + n_s_j]) + &esp_from_l_j));
    total_shift_shell
        .slice_mut(s![n_s_i + n_s_j..])
        .assign(&(&gamma_dq_shell.slice(s![n_s_i + n_s_j..]) + &esp_from_l_k));
    let total_shift: Array1<f64> =
        shell_to_ao_values(trimer.basis(), n_orbs, total_shift_shell.view());
    let coulomb_mat: Array2<f64> = aovec_to_aomat(total_shift.view(), n_orbs) * 0.5;

    // Third order terms
    let mut hubbard_derivatives: Array1<f64> = Array1::zeros(n_atoms_trimer);
    for (val, atom) in hubbard_derivatives.iter_mut().zip(trimer_atoms.iter()) {
        *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
    }
    let dq2_gamma: Array2<f64> =
        coul_third_order_grad_contribution_xtb(trimer.basis(), dq, hubbard_derivatives.view());

    // Effective matrix
    let effective_mat: Array2<f64> = &(-&w) + &(&coulomb_mat * &p) - &(0.5 * &dq2_gamma * &p);

    // Shell-level CTIJK and monomer charges for gamma derivative
    let delta_dq_shell: ArrayView1<f64> = trimer.properties.delta_dq_shell().unwrap();

    // Build dq_mon_shell by concatenating monomer shell charges
    let mut dq_mon_shell_vec: Vec<f64> = Vec::with_capacity(n_shells_trimer);
    for &val in dq_mon_i_shell.iter() {
        dq_mon_shell_vec.push(val);
    }
    for &val in dq_mon_j_shell.iter() {
        dq_mon_shell_vec.push(val);
    }
    for &val in dq_mon_k_shell.iter() {
        dq_mon_shell_vec.push(val);
    }

    // Local-to-global atom index mapping
    let local_to_global: Vec<usize> = m_i
        .slice
        .atom_as_range()
        .chain(m_j.slice.atom_as_range())
        .chain(m_k.slice.atom_as_range())
        .collect();

    let cn_numbers: Array1<f64> = {
        let mut cn = Array1::<f64>::zeros(n_atoms_trimer);
        for (local_idx, &global_idx) in local_to_global.iter().enumerate() {
            cn[local_idx] = cn_numbers_global[global_idx];
        }
        cn
    };

    let mut grad_local = Array1::<f64>::zeros(3 * n_atoms_trimer);
    let mut ctijk_grad_global = Array1::<f64>::zeros(3 * n_atoms_total);
    let mut cn_factors: Vec<f64> = vec![0.0; n_atoms_trimer];

    // === Shell-pair loop ===
    for (shell_i_idx, shell_i) in trimer.basis().shells.iter().enumerate() {
        let atomi = &trimer_atoms[shell_i.atom_index];
        let at_i = shell_i.atom_index;
        let cn_1 = cn_numbers[at_i];

        for (shell_j_idx, shell_j) in trimer.basis().shells.iter().enumerate() {
            let atomj = &trimer_atoms[shell_j.atom_index];
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
                                    &trimer.basis().basis_functions[shell_i.start + idx_i_local];
                                let orbital2 =
                                    &trimer.basis().basis_functions[shell_j.start + idx_j_local];
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
                let ds_d = calc_overlap_derivative_d_shells(trimer.basis(), shell_i, shell_j);
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

    // CN gradient
    let mut cn_grad_contribution = Array1::<f64>::zeros(3 * n_atoms_total);
    for at in 0..n_atoms_trimer {
        if cn_factors[at].abs() > 1e-15 {
            let global_at = local_to_global[at];
            let cn_grad_at: ArrayView1<f64> = cn_grad_global.slice(s![.., global_at]);
            for k in 0..(3 * n_atoms_total) {
                cn_grad_contribution[k] += cn_factors[at] * cn_grad_at[k];
            }
        }
    }

    // Gamma gradient + CTIJK
    for (shell_i_idx, shell_i) in trimer.basis().shells.iter().enumerate() {
        let atomi = &trimer_atoms[shell_i.atom_index];
        let at_i = shell_i.atom_index;

        for (shell_j_idx, shell_j) in trimer.basis().shells.iter().enumerate() {
            let atomj = &trimer_atoms[shell_j.atom_index];
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
                let gamma_deriv = trimer.gammafunction.deriv(
                    distance,
                    atomi.number,
                    shell_i.angular_momentum as u8,
                    atomj.number,
                    shell_j.angular_momentum as u8,
                );

                let shell_dq_prod = dq_shell_trimer[shell_i_idx] * dq_shell_trimer[shell_j_idx];
                let shell_gamma_contrib = 0.5 * gamma_deriv * shell_dq_prod;
                for dir in 0..3 {
                    grad_local[3 * at_i + dir] += e_ij[dir] * shell_gamma_contrib;
                    grad_local[3 * at_j + dir] -= e_ij[dir] * shell_gamma_contrib;
                }

                let shell_ctijk_contrib =
                    -gamma_deriv * delta_dq_shell[shell_i_idx] * dq_mon_shell_vec[shell_j_idx];
                let global_i = local_to_global[at_i];
                let global_j = local_to_global[at_j];
                for dir in 0..3 {
                    ctijk_grad_global[3 * global_i + dir] += e_ij[dir] * shell_ctijk_contrib;
                    ctijk_grad_global[3 * global_j + dir] -= e_ij[dir] * shell_ctijk_contrib;
                }
            }
        }
    }

    let grad_rep = grad_repulsive_energy_xtb(trimer_atoms, n_atoms_trimer);
    grad_local += &grad_rep;

    (grad_local, ctijk_grad_global, cn_grad_contribution)
}

// ============================================================================
// Function C: interfragment_gradient_xtb_shell (CTMUL + ESD fused)
// ============================================================================

/// Combined CTMUL embedding + ES-dimer gradient using shell-level charges.
fn interfragment_gradient_xtb_shell(
    atoms: &[XtbAtom],
    monomers: &[XtbMonomer],
    esd_pairs: &[XtbESDPair],
    ctmul_shell: ArrayView1<f64>,
    dq_global_shell: ArrayView1<f64>,
    super_basis: &Basis,
) -> Array1<f64> {
    let n_atoms_total = atoms.len();

    // Build atom-to-fragment mapping
    let mut atom_to_frag = vec![0usize; n_atoms_total];
    for m in monomers.iter() {
        for global_idx in m.slice.atom_as_range() {
            atom_to_frag[global_idx] = m.index;
        }
    }

    // Build ESD pair lookup
    let mut esd_lookup: HashSet<(usize, usize)> = HashSet::new();
    for esd in esd_pairs.iter() {
        esd_lookup.insert((esd.i, esd.j));
        esd_lookup.insert((esd.j, esd.i));
    }

    // Parallel fold/reduce over monomers — each thread gets a thread-local gradient
    let gradient: Array1<f64> = monomers
        .par_iter()
        .fold(
            || Array1::<f64>::zeros(3 * n_atoms_total),
            |mut gradient, m_i| {
                let dq_i_shell: ArrayView1<f64> = m_i.properties.dq_shell().unwrap();
                let m_i_atom_start = m_i.slice.atom_as_range().start;

                for (s_a_idx, shell_a) in super_basis.shells.iter().enumerate() {
                    let at_a = shell_a.atom_index;
                    let frag_a = atom_to_frag[at_a];

                    let is_esd = frag_a > m_i.index && esd_lookup.contains(&(m_i.index, frag_a));
                    let dq_esd_shell_a = if is_esd {
                        dq_global_shell[s_a_idx]
                    } else {
                        0.0
                    };

                    let ct_a = ctmul_shell[s_a_idx];

                    if ct_a.abs() < 1e-14 && dq_esd_shell_a.abs() < 1e-14 {
                        continue;
                    }

                    let atom_a = &atoms[at_a];

                    for (s_c_idx, shell_c) in m_i.basis.shells.iter().enumerate() {
                        let local_c = shell_c.atom_index;
                        let global_c = m_i_atom_start + local_c;

                        if at_a == global_c {
                            continue;
                        }

                        let atom_c = &atoms[global_c];

                        let dx = atom_a.xyz[0] - atom_c.xyz[0];
                        let dy = atom_a.xyz[1] - atom_c.xyz[1];
                        let dz = atom_a.xyz[2] - atom_c.xyz[2];
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                        if dist < 1e-10 {
                            continue;
                        }

                        let dgamma_dr = m_i.gammafunction.deriv(
                            dist,
                            atom_a.number,
                            shell_a.angular_momentum as u8,
                            atom_c.number,
                            shell_c.angular_momentum as u8,
                        );

                        let total_factor =
                            (ct_a + dq_esd_shell_a) * dq_i_shell[s_c_idx] * dgamma_dr / dist;

                        gradient[3 * at_a + 0] += total_factor * dx;
                        gradient[3 * at_a + 1] += total_factor * dy;
                        gradient[3 * at_a + 2] += total_factor * dz;
                        gradient[3 * global_c + 0] -= total_factor * dx;
                        gradient[3 * global_c + 1] -= total_factor * dy;
                        gradient[3 * global_c + 2] -= total_factor * dz;
                    }
                }
                gradient
            },
        )
        .reduce(
            || Array1::<f64>::zeros(3 * n_atoms_total),
            |mut a, b| {
                a += &b;
                a
            },
        );

    gradient
}

// ============================================================================
// Function D: Parallel CN gradient (replaces serial version from hamiltonian.rs)
// ============================================================================

/// Parallel version of `calculate_coordination_number_gradients`.
///
/// Uses `axis_chunks_iter_mut` to split the output `[3N, N]` matrix into
/// non-overlapping 3-row chunks — one per atom `i`. Each chunk is processed
/// independently with zero extra memory.
pub fn calculate_coordination_number_gradients_parallel(atoms: &[XtbAtom]) -> Array2<f64> {
    use dialect_base::constants::BOHR_TO_ANGS;

    let n = atoms.len();
    let mut grad_cn: Array2<f64> = Array2::zeros([3 * n, n]);

    grad_cn
        .axis_chunks_iter_mut(Axis(0), 3)
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut chunk)| {
            let atomi = &atoms[i];
            let cov_i: f64 = COV_RADII_CN[atomi.number as usize - 1] / BOHR_TO_ANGS;
            let mut grad_i = [0.0f64; 3];

            for (j, atomj) in atoms.iter().enumerate() {
                if i != j {
                    let cov_j: f64 = COV_RADII_CN[atomj.number as usize - 1] / BOHR_TO_ANGS;

                    let dx = atomi.xyz.x - atomj.xyz.x;
                    let dy = atomi.xyz.y - atomj.xyz.y;
                    let dz = atomi.xyz.z - atomj.xyz.z;
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                    let exp_val: f64 =
                        (-16.0 * (4.0 / 3.0 * (cov_i + cov_j) / distance - 1.0)).exp();
                    let deriv_val: f64 = -64.0 * (cov_i + cov_j) * exp_val
                        / (3.0 * distance.powi(2) * (exp_val + 1.0).powi(2));

                    let inv_dist = deriv_val / distance;
                    let vx = dx * inv_dist;
                    let vy = dy * inv_dist;
                    let vz = dz * inv_dist;

                    chunk[[0, j]] = vx;
                    chunk[[1, j]] = vy;
                    chunk[[2, j]] = vz;

                    grad_i[0] += vx;
                    grad_i[1] += vy;
                    grad_i[2] += vz;
                }
            }

            chunk[[0, i]] = grad_i[0];
            chunk[[1, i]] = grad_i[1];
            chunk[[2, i]] = grad_i[2];
        });

    grad_cn
}


// ============================================================================
// Assembly: ground_state_gradient_fmo + Response gradient
// ============================================================================

impl XtbSuperSystem<'_> {
    /// Compute complete FMO2-xTB gradient using gamma_shell.
    /// Returns (gradient, cn_grad_global) so the CN gradient can be reused by the response gradient.
    pub fn ground_state_gradient_fmo_shell(&mut self) -> (Array1<f64>, Array2<f64>) {
        let atoms: &[XtbAtom] = &self.atoms[..];
        let n_atoms_total = atoms.len();
        let n_grad = 3 * n_atoms_total;

        let cn_numbers_global: ArrayView1<f64> = self.properties.cn().unwrap();
        let cn_grad_global: Array2<f64> = calculate_coordination_number_gradients_parallel(atoms);

        // Step 0: Compute pair SCAL factors for FMO3
        let pair_scal: Vec<f64> = if self.config.fmo.use_three_body {
            let mut scal = vec![1.0f64; self.pairs.len()];
            for trimer in self.trimers.iter() {
                for &(a, b) in &[
                    (trimer.i, trimer.j),
                    (trimer.i, trimer.k),
                    (trimer.j, trimer.k),
                ] {
                    if self.properties.type_of_pair(a, b) == PairType::Pair {
                        let idx = self.properties.index_of_pair(a, b);
                        scal[idx] -= 1.0;
                    }
                }
            }
            scal
        } else {
            vec![1.0f64; self.pairs.len()]
        };

        // Step 1: Compute CTMUL_shell
        let n_shells_total = self.basis.shells.len();
        let trimers_ref = &self.trimers;
        let ctmul_shell = compute_ctmul_shell(
            &self.monomers,
            &self.pairs,
            trimers_ref,
            &pair_scal,
            n_shells_total,
        );

        // Step 2: Compute SHIFTCT + ESPGRAD per monomer using gamma_shell_super
        let gamma_shell_super: ArrayView2<f64> = self.properties.gamma_shell().unwrap();
        let monomers_ref = &self.monomers;
        let pairs_ref = &self.pairs;
        let shiftcts: Vec<Array1<f64>> = self
            .monomers
            .par_iter()
            .map(|m| {
                compute_shiftct_espgrad_shell(
                    m,
                    gamma_shell_super,
                    ctmul_shell.view(),
                    monomers_ref,
                    pairs_ref,
                    trimers_ref,
                    &pair_scal,
                )
            })
            .collect();

        // Step 3: Monomer gradients [parallel]
        let cn_grad_view = cn_grad_global.view();
        let monomer_results: Vec<(Array1<f64>, Array1<f64>, Array1<f64>)> = self
            .monomers
            .par_iter()
            .zip(shiftcts.par_iter())
            .map(|(m, shiftct)| {
                monomer_gradient_combined_shell(
                    m,
                    atoms,
                    shiftct.view(),
                    cn_numbers_global,
                    cn_grad_view,
                )
            })
            .collect();

        // Assemble monomer gradients
        let mut monomer_grad_total = Array1::<f64>::zeros(n_grad);
        let mut addlag_total = Array1::<f64>::zeros(n_grad);
        let mut cn_grad_total = Array1::<f64>::zeros(n_grad);

        for (m, (mon_grad, add_grad, cn_glob)) in self.monomers.iter().zip(monomer_results.iter()) {
            for (local_idx, global_idx) in m.slice.atom_as_range().enumerate() {
                for k in 0..3 {
                    monomer_grad_total[3 * global_idx + k] += mon_grad[3 * local_idx + k];
                    addlag_total[3 * global_idx + k] += add_grad[3 * local_idx + k];
                }
            }
            cn_grad_total += cn_glob;
        }

        // Step 4: Pair gradients [parallel]
        let pair_results: Vec<(Array1<f64>, Array1<f64>, Array1<f64>)> = self
            .pairs
            .par_iter()
            .map(|pair| {
                let m_i = &self.monomers[pair.i];
                let m_j = &self.monomers[pair.j];
                let pair_atoms: Vec<XtbAtom> =
                    get_pair_slice_xtb(atoms, m_i.slice.atom_as_range(), m_j.slice.atom_as_range());
                pair_gradient_combined_shell(
                    pair,
                    &pair_atoms,
                    m_i,
                    m_j,
                    atoms,
                    cn_numbers_global,
                    cn_grad_view,
                )
            })
            .collect();

        // Step 5: Pair delta + CTIJ accumulation
        let mut pair_delta_total = Array1::<f64>::zeros(n_grad);
        let mut ctij_total = Array1::<f64>::zeros(n_grad);

        for (pair_idx, (pair, (pair_grad_local, ctij_grad_global, pair_cn_glob))) in
            self.pairs.iter().zip(pair_results.iter()).enumerate()
        {
            let m_i = &self.monomers[pair.i];
            let m_j = &self.monomers[pair.j];
            let mon_i_grad = &monomer_results[pair.i].0;
            let mon_j_grad = &monomer_results[pair.j].0;

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

            let scal = pair_scal[pair_idx];
            ctij_total += &(scal * ctij_grad_global);

            let mon_i_cn = &monomer_results[pair.i].2;
            let mon_j_cn = &monomer_results[pair.j].2;
            cn_grad_total += &(pair_cn_glob - mon_i_cn - mon_j_cn);
        }

        // Step 6: Inter-fragment gradient using shell-level charges
        // Build dq_shell_global from stored monomer dq_shell
        let mut dq_shell_global = Array1::<f64>::zeros(n_shells_total);
        for m in self.monomers.iter() {
            let dq_s: ArrayView1<f64> = m.properties.dq_shell().unwrap();
            dq_shell_global.slice_mut(s![m.slice.shell]).assign(&dq_s);
        }

        let interfrag_grad = interfragment_gradient_xtb_shell(
            atoms,
            &self.monomers,
            &self.esd_pairs,
            ctmul_shell.view(),
            dq_shell_global.view(),
            &self.basis,
        );

        // Step 7: Global dispersion
        let disp_grad = gradient_disp3_xtb(&self.atoms, &self.config);
        let halogen_grad = gradient_halogen_bonding_xtb(&self.atoms);

        // Step 8: FMO3 three-body correction
        let trimer_contribution = if self.config.fmo.use_three_body {
            let trimer_results: Vec<(Array1<f64>, Array1<f64>, Array1<f64>)> = self
                .trimers
                .par_iter()
                .map(|trimer| {
                    let m_i = &self.monomers[trimer.i];
                    let m_j = &self.monomers[trimer.j];
                    let m_k = &self.monomers[trimer.k];
                    let trimer_atoms: Vec<XtbAtom> = get_trimer_slice_xtb(
                        atoms,
                        m_i.slice.atom_as_range(),
                        m_j.slice.atom_as_range(),
                        m_k.slice.atom_as_range(),
                    );
                    trimer_gradient_combined_shell(
                        trimer,
                        &trimer_atoms,
                        m_i,
                        m_j,
                        m_k,
                        atoms,
                        cn_numbers_global,
                        cn_grad_view,
                    )
                })
                .collect();

            // ESD pair gradients for trimer subtraction
            // Reconstruct AO charges from shell charges for gamma_gradient_xtb_double_contracted
            let esd_pair_grads: Vec<Array1<f64>> = self
                .esd_pairs
                .par_iter()
                .map(|esd_pair| {
                    let m_i = &self.monomers[esd_pair.i];
                    let m_j = &self.monomers[esd_pair.j];
                    let dq_i_shell = m_i.properties.dq_shell().unwrap();
                    let dq_j_shell = m_j.properties.dq_shell().unwrap();
                    let dq_i = shell_to_ao_charges(&m_i.basis, m_i.n_orbs, dq_i_shell);
                    let dq_j = shell_to_ao_charges(&m_j.basis, m_j.n_orbs, dq_j_shell);

                    let pair_atoms = get_pair_slice_xtb(
                        atoms,
                        m_i.slice.atom_as_range(),
                        m_j.slice.atom_as_range(),
                    );
                    let basis: Basis = create_basis_set(&pair_atoms);

                    let grad_gamma_term_i = gamma_gradient_xtb_double_contracted(
                        m_i.gammafunction,
                        &pair_atoms,
                        &basis,
                        dq_i.view(),
                        dq_j.view(),
                        (0, m_i.n_orbs),
                        (m_i.n_orbs, basis.nbas),
                        (0, m_i.n_atoms),
                    );
                    let grad_gamma_term_j = gamma_gradient_xtb_double_contracted(
                        m_i.gammafunction,
                        &pair_atoms,
                        &basis,
                        dq_i.view(),
                        dq_j.view(),
                        (0, m_i.n_orbs),
                        (m_i.n_orbs, basis.nbas),
                        (m_i.n_atoms, pair_atoms.len()),
                    );

                    let mut ij_grad: Array1<f64> = Array1::zeros(3 * m_i.n_atoms + 3 * m_j.n_atoms);
                    ij_grad
                        .slice_mut(s![..3 * m_i.n_atoms])
                        .assign(&grad_gamma_term_i);
                    ij_grad
                        .slice_mut(s![3 * m_i.n_atoms..])
                        .assign(&grad_gamma_term_j);
                    ij_grad
                })
                .collect();

            // Per-pair delta gradients
            let mut per_pair_delta_global: Vec<Array1<f64>> = Vec::with_capacity(self.pairs.len());
            for (pair, (pair_grad_local, _ctij_grad_global, pair_cn_glob)) in
                self.pairs.iter().zip(pair_results.iter())
            {
                let m_i = &self.monomers[pair.i];
                let m_j = &self.monomers[pair.j];
                let mon_i_grad = &monomer_results[pair.i].0;
                let mon_j_grad = &monomer_results[pair.j].0;
                let mon_i_cn = &monomer_results[pair.i].2;
                let mon_j_cn = &monomer_results[pair.j].2;

                let mut delta = Array1::<f64>::zeros(n_grad);
                for (local_idx, global_idx) in m_i.slice.atom_as_range().enumerate() {
                    for k in 0..3 {
                        delta[3 * global_idx + k] =
                            pair_grad_local[3 * local_idx + k] - mon_i_grad[3 * local_idx + k];
                    }
                }
                for (local_idx, global_idx) in m_j.slice.atom_as_range().enumerate() {
                    for k in 0..3 {
                        delta[3 * global_idx + k] = pair_grad_local
                            [3 * (m_i.n_atoms + local_idx) + k]
                            - mon_j_grad[3 * local_idx + k];
                    }
                }
                delta += &(pair_cn_glob - mon_i_cn - mon_j_cn);
                per_pair_delta_global.push(delta);
            }

            // Trimer delta + CTIJK assembly
            let mut trimer_delta_total = Array1::<f64>::zeros(n_grad);
            let mut ctijk_total = Array1::<f64>::zeros(n_grad);

            for (trimer, (tri_grad_local, tri_ctijk_glob, tri_cn)) in
                self.trimers.iter().zip(trimer_results.iter())
            {
                let m_i = &self.monomers[trimer.i];
                let m_j = &self.monomers[trimer.j];
                let m_k = &self.monomers[trimer.k];

                let mut delta_global = Array1::<f64>::zeros(n_grad);
                let mon_i_grad = &monomer_results[trimer.i].0;
                let mon_j_grad = &monomer_results[trimer.j].0;
                let mon_k_grad = &monomer_results[trimer.k].0;

                for (local_idx, global_idx) in m_i.slice.atom_as_range().enumerate() {
                    for k in 0..3 {
                        delta_global[3 * global_idx + k] =
                            tri_grad_local[3 * local_idx + k] - mon_i_grad[3 * local_idx + k];
                    }
                }
                for (local_idx, global_idx) in m_j.slice.atom_as_range().enumerate() {
                    for k in 0..3 {
                        delta_global[3 * global_idx + k] = tri_grad_local
                            [3 * (m_i.n_atoms + local_idx) + k]
                            - mon_j_grad[3 * local_idx + k];
                    }
                }
                for (local_idx, global_idx) in m_k.slice.atom_as_range().enumerate() {
                    for k in 0..3 {
                        delta_global[3 * global_idx + k] = tri_grad_local
                            [3 * (m_i.n_atoms + m_j.n_atoms + local_idx) + k]
                            - mon_k_grad[3 * local_idx + k];
                    }
                }

                delta_global += &(tri_cn
                    - &monomer_results[trimer.i].2
                    - &monomer_results[trimer.j].2
                    - &monomer_results[trimer.k].2);

                for &(a, b) in &[
                    (trimer.i, trimer.j),
                    (trimer.i, trimer.k),
                    (trimer.j, trimer.k),
                ] {
                    if self.properties.type_of_pair(a, b) == PairType::Pair {
                        let idx = self.properties.index_of_pair(a, b);
                        delta_global -= &per_pair_delta_global[idx];
                    } else {
                        let idx = self.properties.index_of_esd_pair(a, b);
                        let esd_grad_local = &esd_pair_grads[idx];
                        let m_a = &self.monomers[a];
                        let m_b = &self.monomers[b];

                        let mut esd_global = Array1::<f64>::zeros(n_grad);
                        for (local_idx, global_idx) in m_a.slice.atom_as_range().enumerate() {
                            for k in 0..3 {
                                esd_global[3 * global_idx + k] = esd_grad_local[3 * local_idx + k];
                            }
                        }
                        for (local_idx, global_idx) in m_b.slice.atom_as_range().enumerate() {
                            for k in 0..3 {
                                esd_global[3 * global_idx + k] =
                                    esd_grad_local[3 * (m_a.n_atoms + local_idx) + k];
                            }
                        }
                        delta_global -= &esd_global;
                    }
                }

                trimer_delta_total += &delta_global;
                ctijk_total += tri_ctijk_glob;
            }

            &trimer_delta_total + &ctijk_total
        } else {
            Array1::zeros(n_grad)
        };

        // Step 9: Assembly
        let total_gradient = &monomer_grad_total
            + &pair_delta_total
            + &ctij_total
            + &interfrag_grad
            + &addlag_total
            + &disp_grad
            + &halogen_grad
            + &cn_grad_total
            + &trimer_contribution;

        (total_gradient, cn_grad_global)
    }

    // ========================================================================
    // Response gradient: xTB version using gamma_shell
    // ========================================================================

    /// Calculate response Lagrangian using gamma_shell and qvo_shell.
    ///
    /// Returns (lagrangians, qvo_ao_vectors, qvo_shell_vectors).
    fn calculate_response_lagrangian_xtb_shell(
        &self,
    ) -> (Vec<Array1<f64>>, Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let gamma_shell_super: ArrayView2<f64> = self.properties.gamma_shell().unwrap();

        let pair_scal: Vec<f64> = if self.config.fmo.use_three_body {
            let mut scal = vec![1.0f64; self.pairs.len()];
            for trimer in self.trimers.iter() {
                for &(a, b) in &[
                    (trimer.i, trimer.j),
                    (trimer.i, trimer.k),
                    (trimer.j, trimer.k),
                ] {
                    if self.properties.type_of_pair(a, b) == PairType::Pair {
                        let idx = self.properties.index_of_pair(a, b);
                        scal[idx] -= 1.0;
                    }
                }
            }
            scal
        } else {
            vec![1.0f64; self.pairs.len()]
        };

        // Build qvo_ao and qvo_shell for each monomer
        let qvo_data: Vec<(Array2<f64>, Array2<f64>)> =
            self.monomers.iter().map(|m| compute_qvo_shell(m)).collect();

        let qvo_ao_vec: Vec<Array2<f64>> = qvo_data.iter().map(|(ao, _)| ao.clone()).collect();
        let qvo_shell_vec: Vec<Array2<f64>> = qvo_data.iter().map(|(_, sh)| sh.clone()).collect();

        // Calculate Lagrangian for each monomer K using shell-level gamma and qvo_shell
        let lagrangian_vec: Vec<Array1<f64>> = self
            .monomers
            .par_iter()
            .enumerate()
            .map(|(idx_k, m_k)| {
                let nocc = m_k.properties.occ_indices().unwrap().len();
                let nvirt = m_k.properties.virt_indices().unwrap().len();
                let nvo = nvirt * nocc;
                let qvo_shell_k: &Array2<f64> = &qvo_shell_vec[idx_k];

                let gamma_k_all_shell = gamma_shell_super.slice(s![m_k.slice.shell, ..]);

                let mut lag_k = Array1::<f64>::zeros(nvo);

                // Pairs (scaled by SCAL for FMO3)
                for (pair_idx, pair) in self.pairs.iter().enumerate() {
                    let scal = pair_scal[pair_idx];
                    if scal.abs() < 1e-14 {
                        continue;
                    }

                    let m_i = &self.monomers[pair.i];
                    let m_j = &self.monomers[pair.j];

                    if m_i.index == m_k.index || m_j.index == m_k.index {
                        continue;
                    }

                    let delta_dq_shell: ArrayView1<f64> = pair.properties.delta_dq_shell().unwrap();
                    let n_shells_i = m_i.n_shells;

                    let gamma_k_i_shell = gamma_k_all_shell.slice(s![.., m_i.slice.shell]);
                    let gamma_k_j_shell = gamma_k_all_shell.slice(s![.., m_j.slice.shell]);

                    let esp_shell_on_k: Array1<f64> = gamma_k_i_shell
                        .dot(&delta_dq_shell.slice(s![..n_shells_i]))
                        + gamma_k_j_shell.dot(&delta_dq_shell.slice(s![n_shells_i..]));

                    // [n_shells_K] dot [n_shells_K, nvo] = [nvo]
                    lag_k += &(-0.5 * scal * esp_shell_on_k.dot(qvo_shell_k));
                }

                // FMO3: Trimers not containing K
                if self.config.fmo.use_three_body {
                    for trimer in self.trimers.iter() {
                        let m_ti = &self.monomers[trimer.i];
                        let m_tj = &self.monomers[trimer.j];
                        let m_tk = &self.monomers[trimer.k];

                        if m_ti.index == m_k.index
                            || m_tj.index == m_k.index
                            || m_tk.index == m_k.index
                        {
                            continue;
                        }

                        let delta_dq_shell: ArrayView1<f64> =
                            trimer.properties.delta_dq_shell().unwrap();
                        let n_s_i = m_ti.n_shells;
                        let n_s_j = m_tj.n_shells;

                        let gamma_k_ti = gamma_k_all_shell.slice(s![.., m_ti.slice.shell]);
                        let gamma_k_tj = gamma_k_all_shell.slice(s![.., m_tj.slice.shell]);
                        let gamma_k_tk = gamma_k_all_shell.slice(s![.., m_tk.slice.shell]);

                        let esp_shell_on_k: Array1<f64> = gamma_k_ti
                            .dot(&delta_dq_shell.slice(s![..n_s_i]))
                            + gamma_k_tj.dot(&delta_dq_shell.slice(s![n_s_i..n_s_i + n_s_j]))
                            + gamma_k_tk.dot(&delta_dq_shell.slice(s![n_s_i + n_s_j..]));

                        lag_k += &(-0.5 * esp_shell_on_k.dot(qvo_shell_k));
                    }
                }

                lag_k
            })
            .collect();

        (lagrangian_vec, qvo_ao_vec, qvo_shell_vec)
    }

    /// Compute A_I · v using gamma_shell: O(n_shells²) instead of O(n_orbs²).
    fn orbital_hessian_matvec_xtb_shell(
        m_i: &XtbMonomer,
        qvo_shell: &Array2<f64>,
        gamma_shell: ArrayView2<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        let nocc = m_i.properties.occ_indices().unwrap().len();
        let nvirt = m_i.properties.virt_indices().unwrap().len();
        let orbe: ArrayView1<f64> = m_i.properties.orbe().unwrap();

        // Diagonal: (eps_i - eps_a) * v_ai
        let mut result = Array1::<f64>::zeros(v.len());
        for a in 0..nvirt {
            for i in 0..nocc {
                let idx = a * nocc + i;
                result[idx] = (orbe[i] - orbe[nocc + a]) * v[idx];
            }
        }

        // Coulomb: -Q_shell^T @ gamma_shell @ Q_shell @ v
        let qv_shell: Array1<f64> = qvo_shell.dot(v); // [n_shells]
        let g_qv_shell: Array1<f64> = gamma_shell.dot(&qv_shell); // [n_shells]
        let qt_g_qv: Array1<f64> = qvo_shell.t().dot(&g_qv_shell); // [nvo]
        result -= &qt_g_qv;

        result
    }

    /// Matrix-free SCZV solver using gamma_shell for Coulomb and inter-fragment coupling.
    fn solve_sczv_cg_xtb_shell(
        &self,
        lagrangian: &[Array1<f64>],
        qvo_ao_vec: &[Array2<f64>],
        qvo_shell_vec: &[Array2<f64>],
    ) -> Vec<Array1<f64>> {
        let maxiter = 500;
        let threshold = 1.0e-8;
        let gamma_shell_super: ArrayView2<f64> = self.properties.gamma_shell().unwrap();
        let n_frag = self.monomers.len();

        let rhs: Vec<Array1<f64>> = lagrangian.iter().map(|l| 4.0 * l).collect();

        let seg_sizes: Vec<usize> = self
            .monomers
            .iter()
            .map(|m| {
                let nocc = m.properties.occ_indices().unwrap().len();
                let nvirt = m.properties.virt_indices().unwrap().len();
                nvirt * nocc
            })
            .collect();

        // Per-fragment gamma_shell
        let gammas_shell: Vec<ArrayView2<f64>> = self
            .monomers
            .iter()
            .map(|m| m.properties.gamma_shell().unwrap())
            .collect();

        // Atom-level third-order (unchanged, uses qvo_ao)
        let atoms = &self.atoms[..];
        let qvo_atom_vec: Vec<Array2<f64>> = self
            .monomers
            .iter()
            .enumerate()
            .map(|(idx, m)| {
                let n_atoms = m.n_atoms;
                let nvo = seg_sizes[idx];
                let qvo_ao = &qvo_ao_vec[idx];
                let mut qvo_atom = Array2::<f64>::zeros([n_atoms, nvo]);
                for shell in m.basis.shells.iter() {
                    let at = shell.atom_index;
                    for mu in shell.sph_start..shell.sph_end {
                        for ai in 0..nvo {
                            qvo_atom[[at, ai]] += qvo_ao[[mu, ai]];
                        }
                    }
                }
                qvo_atom
            })
            .collect();

        let third_factor_vec: Vec<Array1<f64>> = self
            .monomers
            .iter()
            .map(|m| {
                let m_atoms = &atoms[m.slice.atom_as_range()];
                let dq: ArrayView1<f64> = m.properties.dq().unwrap();
                let mut factors = Array1::<f64>::zeros(m.n_atoms);
                for (at, atom) in m_atoms.iter().enumerate() {
                    let hubb_deriv = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
                    factors[at] = 2.0 * hubb_deriv * dq[at];
                }
                factors
            })
            .collect();

        // Full matrix-vector product using gamma_shell
        let matvec = |z_vecs: &Vec<Array1<f64>>| -> Vec<Array1<f64>> {
            // Shell-level QINDZ
            let qindz_shell: Vec<Array1<f64>> = (0..n_frag)
                .map(|k| qvo_shell_vec[k].dot(&z_vecs[k]))
                .collect();

            (0..n_frag)
                .into_par_iter()
                .map(|idx_i| {
                    let m_i = &self.monomers[idx_i];

                    // Intra-fragment: A_I · z_I using gamma_shell
                    let mut result_i = Self::orbital_hessian_matvec_xtb_shell(
                        m_i,
                        &qvo_shell_vec[idx_i],
                        gammas_shell[idx_i],
                        &z_vecs[idx_i],
                    );

                    // Third-order (atom-level, unchanged)
                    let qvo_atom = &qvo_atom_vec[idx_i];
                    let third_factor = &third_factor_vec[idx_i];
                    let qv_atom: Array1<f64> = qvo_atom.dot(&z_vecs[idx_i]);
                    let g_qv_atom: Array1<f64> = third_factor * &qv_atom;
                    result_i -= &qvo_atom.t().dot(&g_qv_atom);

                    // Inter-fragment: using gamma_shell_super and qvo_shell
                    let mut shift_shell_i = Array1::<f64>::zeros(m_i.n_shells);
                    for idx_k in 0..n_frag {
                        if idx_k != idx_i {
                            let m_k = &self.monomers[idx_k];
                            let gamma_ik_shell =
                                gamma_shell_super.slice(s![m_i.slice.shell, m_k.slice.shell]);
                            shift_shell_i += &gamma_ik_shell.dot(&qindz_shell[idx_k]);
                        }
                    }
                    result_i -= &qvo_shell_vec[idx_i].t().dot(&shift_shell_i);

                    result_i
                })
                .collect()
        };

        // Jacobi preconditioner
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

        // Preconditioned CG
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

    /// Per-monomer response gradient using gamma_shell for shift computation.
    fn response_gradient_onthefly_xtb_shell(
        &self,
        z_vectors: &[Array1<f64>],
        cn_numbers_global: ArrayView1<f64>,
        cn_grad_global: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_atoms_total = self.atoms.len();
        let atoms = &self.atoms[..];

        let local_grads: Vec<(Array2<f64>, Array1<f64>)> = self
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

                // Shell-level shift: gamma_shell · dq_shell + esp_q_shell
                let gamma_shell: ArrayView2<f64> = m_i.properties.gamma_shell().unwrap();
                let dq_shell: ArrayView1<f64> = m_i.properties.dq_shell().unwrap();
                let esp_q_shell: ArrayView1<f64> = m_i.properties.esp_q().unwrap();

                // Z in MO → AO basis
                let z_mat = z_i.view().into_shape([nvirt, nocc]).unwrap();
                let c_virt = orbs.slice(s![.., nocc..nocc + nvirt]);
                let c_occ = orbs.slice(s![.., ..nocc]);
                let z_ao: Array2<f64> = c_virt.dot(&z_mat.dot(&c_occ.t()));

                // WZ_AO: energy-weighted Z
                let mut wz_mat = Array2::<f64>::zeros([nvirt, nocc]);
                for i_occ in 0..nocc {
                    for a in 0..nvirt {
                        wz_mat[[a, i_occ]] = z_mat[[a, i_occ]] * orbe[i_occ];
                    }
                }
                let wz_ao: Array2<f64> = c_virt.dot(&wz_mat.dot(&c_occ.t()));

                // Build AO-level shift from shell-level
                let shift_shell: Array1<f64> = gamma_shell.dot(&dq_shell) + &esp_q_shell;
                let mut shift_vec = shell_to_ao_values(&m_i.basis, n_orbs, shift_shell.view());

                // Subtract third-order Coulomb shift (atom-level, unchanged)
                let dq_atom: ArrayView1<f64> = m_i.properties.dq().unwrap();
                for shell in m_i.basis.shells.iter() {
                    let at = shell.atom_index;
                    let hubb_deriv = COUL_THIRD_ORDER_ATOM[m_atoms[at].number as usize - 1];
                    let epot = hubb_deriv * dq_atom[at] * dq_atom[at];
                    for mu in shell.sph_start..shell.sph_end {
                        shift_vec[mu] -= epot;
                    }
                }

                let shift_ao_mat: Array2<f64> = aovec_to_aomat(shift_vec.view(), n_orbs) * 0.5;
                let wrk_response: Array2<f64> = &shift_ao_mat * &z_ao - &wz_ao;

                // Q^Z at AO level
                let z_sym: Array2<f64> = 0.5 * (&z_ao + &z_ao.t());
                let zs: Array2<f64> = z_sym.dot(&s);
                let mut q_z_ao = Array1::<f64>::zeros(n_orbs);
                for mu in 0..n_orbs {
                    q_z_ao[mu] = zs[[mu, mu]];
                }

                // Shell-level Q^Z (dq_shell from stored properties)
                let n_shells = m_i.basis.shells.len();
                let mut q_z_shell = vec![0.0; n_shells];
                for (shell_idx, shell) in m_i.basis.shells.iter().enumerate() {
                    for mu in shell.sph_start..shell.sph_end {
                        q_z_shell[shell_idx] += q_z_ao[mu];
                    }
                }

                let atom_range = m_i.slice.atom_as_range();
                let atom_start = atom_range.start;
                let cn_numbers: Array1<f64> = cn_numbers_global.slice(s![atom_range]).to_owned();

                let mut grad_local = Array2::<f64>::zeros([n_atoms, 3]);
                let mut cn_factors: Vec<f64> = vec![0.0; n_atoms];

                // === Shell-pair loop ===
                for (shell_i_idx, shell_i) in m_i.basis.shells.iter().enumerate() {
                    let atomi = &m_atoms[shell_i.atom_index];
                    let at_i = shell_i.atom_index;
                    let cn_1 = cn_numbers[at_i];

                    for (shell_j_idx, shell_j) in m_i.basis.shells.iter().enumerate() {
                        let atomj = &m_atoms[shell_j.atom_index];
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
                        let cn_coeff_i =
                            get_self_energy_cn_grad_coeff_shell(atomi.number, shell_i.shell_index);
                        let cn_coeff_j =
                            get_self_energy_cn_grad_coeff_shell(atomj.number, shell_j.shell_index);

                        let is_same_shell = shell_i.sph_start == shell_j.sph_start
                            && shell_i.sph_end == shell_j.sph_end;

                        let (scaling_constant, en_term, hueckel_const, pi_term) = if !is_same_shell
                        {
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

                        let h0_val =
                            scaling_constant * hueckel_const * self_energy_term * en_term * pi_term;
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
                        let pi_factor =
                            scaling_constant * hueckel_const * self_energy_term * en_term;

                        let mut diag_sp_sum: f64 = 0.0;
                        let mut off_sp_sum: f64 = 0.0;
                        let mut shell_pi_sp_sum: f64 = 0.0;
                        let mut shell_ds_contrib: [f64; 3] = [0.0; 3];

                        for idx_i in shell_i.sph_start..shell_i.sph_end {
                            let idx_i_local = idx_i - shell_i.sph_start;
                            for idx_j in shell_j.sph_start..shell_j.sph_end {
                                let idx_j_local = idx_j - shell_j.sph_start;
                                let z_ij = z_ao[[idx_i, idx_j]];
                                let s_ij = s[[idx_i, idx_j]];

                                if idx_i == idx_j {
                                    diag_sp_sum += s_ij * z_ij;
                                } else {
                                    off_sp_sum += s_ij * z_ij;
                                    if at_i != at_j {
                                        if shell_i.angular_momentum < 2
                                            && shell_j.angular_momentum < 2
                                        {
                                            let orbital1 = &m_i.basis.basis_functions
                                                [shell_i.start + idx_i_local];
                                            let orbital2 = &m_i.basis.basis_functions
                                                [shell_j.start + idx_j_local];
                                            let norm_prod =
                                                orbital1.contracted_norm * orbital2.contracted_norm;
                                            let w_ij = wrk_response[[idx_i, idx_j]];
                                            let combined = h0_val * z_ij + w_ij;
                                            let ds_all =
                                                obara_saika_derivatives_all(orbital1, orbital2);
                                            for dir in 0..3 {
                                                shell_ds_contrib[dir] +=
                                                    ds_all[dir] * norm_prod * combined;
                                            }
                                            shell_pi_sp_sum += s_ij
                                                * 0.5
                                                * (z_ao[[idx_i, idx_j]] + z_ao[[idx_j, idx_i]]);
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
                            let ds_d =
                                calc_overlap_derivative_d_shells(&m_i.basis, shell_i, shell_j);
                            let sph_dim_i = shell_i.sph_end - shell_i.sph_start;
                            let sph_dim_j = shell_j.sph_end - shell_j.sph_start;
                            for sph_i in 0..sph_dim_i {
                                let idx_i = shell_i.sph_start + sph_i;
                                for sph_j in 0..sph_dim_j {
                                    let idx_j = shell_j.sph_start + sph_j;
                                    let z_ij = z_ao[[idx_i, idx_j]];
                                    let z_ji = z_ao[[idx_j, idx_i]];
                                    let w_ij = wrk_response[[idx_i, idx_j]];
                                    let w_ji = wrk_response[[idx_j, idx_i]];
                                    for dir in 0..3 {
                                        let ds_val_i = ds_d[[dir, sph_i, sph_j]];
                                        let ds_val_j = ds_d[[3 + dir, sph_i, sph_j]];
                                        let combined = h0_val * (z_ij + z_ji) + (w_ij + w_ji);
                                        grad_local[[at_i, dir]] += ds_val_i * combined;
                                        grad_local[[at_j, dir]] += ds_val_j * combined;
                                    }
                                    shell_pi_sp_sum += s[[idx_i, idx_j]] * 0.5 * (z_ij + z_ji);
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

                // CN gradient
                let mut cn_grad_contribution = Array1::<f64>::zeros(3 * n_atoms_total);
                for at in 0..n_atoms {
                    if cn_factors[at].abs() > 1e-15 {
                        let global_at = atom_start + at;
                        let cn_grad_at: ArrayView1<f64> = cn_grad_global.slice(s![.., global_at]);
                        for k in 0..(3 * n_atoms_total) {
                            cn_grad_contribution[k] += cn_factors[at] * cn_grad_at[k];
                        }
                    }
                }

                // Gamma derivative loop using stored dq_shell
                for (shell_i_idx, shell_i) in m_i.basis.shells.iter().enumerate() {
                    let atomi = &m_atoms[shell_i.atom_index];
                    let at_i = shell_i.atom_index;

                    for (shell_j_idx, shell_j) in m_i.basis.shells.iter().enumerate() {
                        let atomj = &m_atoms[shell_j.atom_index];
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
                            let gamma_deriv = m_i.gammafunction.deriv(
                                distance,
                                atomi.number,
                                shell_i.angular_momentum as u8,
                                atomj.number,
                                shell_j.angular_momentum as u8,
                            );
                            let factor = (dq_shell[shell_i_idx] * q_z_shell[shell_j_idx]
                                + q_z_shell[shell_i_idx] * dq_shell[shell_j_idx])
                                * 0.5
                                * gamma_deriv;
                            for dir in 0..3 {
                                grad_local[[at_i, dir]] += e_ij[dir] * factor;
                                grad_local[[at_j, dir]] -= e_ij[dir] * factor;
                            }
                        }
                    }
                }

                (grad_local, cn_grad_contribution)
            })
            .collect();

        // Reduce into global gradient
        let mut gradient = Array2::<f64>::zeros([n_atoms_total, 3]);
        let mut cn_grad_total = Array1::<f64>::zeros(3 * n_atoms_total);
        for (m_i, (grad_local, cn_glob)) in self.monomers.iter().zip(local_grads.iter()) {
            for (local_idx, global_idx) in m_i.slice.atom_as_range().enumerate() {
                for k in 0..3 {
                    gradient[[global_idx, k]] += grad_local[[local_idx, k]];
                }
            }
            cn_grad_total += cn_glob;
        }

        for at in 0..n_atoms_total {
            for d in 0..3 {
                gradient[[at, d]] += cn_grad_total[3 * at + d];
            }
        }

        gradient
    }

    /// Inter-fragment response gradient: Z×G between monomers using stored dq_shell.
    fn inter_fragment_response_gradient_xtb_shell(&self, z_vectors: &[Array1<f64>]) -> Array2<f64> {
        let n_atoms_total = self.atoms.len();
        let atoms = &self.atoms[..];

        // Compute Q^Z at shell level and use stored dq_shell
        let mut q_z_shell_all: Vec<Vec<f64>> = Vec::with_capacity(self.monomers.len());
        let mut dq_shell_all: Vec<ArrayView1<f64>> = Vec::with_capacity(self.monomers.len());

        for (idx_i, m_i) in self.monomers.iter().enumerate() {
            let z_i = &z_vectors[idx_i];
            let nocc: usize = m_i.properties.occ_indices().unwrap().len();
            let nvirt: usize = m_i.properties.virt_indices().unwrap().len();
            let orbs: ArrayView2<f64> = m_i.properties.orbs().unwrap();
            let s: ArrayView2<f64> = m_i.properties.s().unwrap();

            let z_mat = z_i.view().into_shape([nvirt, nocc]).unwrap();
            let c_virt = orbs.slice(s![.., nocc..nocc + nvirt]);
            let c_occ = orbs.slice(s![.., ..nocc]);
            let z_ao_unsym: Array2<f64> = c_virt.dot(&z_mat.dot(&c_occ.t()));
            let z_ao_sym: Array2<f64> = 0.5 * (&z_ao_unsym + &z_ao_unsym.t());
            let zs: Array2<f64> = z_ao_sym.dot(&s);

            let n_shells = m_i.basis.shells.len();
            let mut q_z_shell = vec![0.0; n_shells];
            for (s_idx, shell) in m_i.basis.shells.iter().enumerate() {
                for mu in shell.sph_start..shell.sph_end {
                    q_z_shell[s_idx] += zs[[mu, mu]];
                }
            }

            q_z_shell_all.push(q_z_shell);
            dq_shell_all.push(m_i.properties.dq_shell().unwrap());
        }

        // Parallel fold/reduce over outer monomer loop
        let gradient: Array2<f64> = self
            .monomers
            .par_iter()
            .enumerate()
            .fold(
                || Array2::<f64>::zeros([n_atoms_total, 3]),
                |mut gradient, (idx_i, m_i)| {
                    let m_i_atom_start = m_i.slice.atom_as_range().start;

                    for (idx_j, m_j) in self.monomers.iter().enumerate() {
                        if idx_i == idx_j {
                            continue;
                        }

                        let m_j_atom_start = m_j.slice.atom_as_range().start;

                        for (s_idx, shell_s) in m_i.basis.shells.iter().enumerate() {
                            let local_s = shell_s.atom_index;
                            let global_s = m_i_atom_start + local_s;
                            let atom_s = &atoms[global_s];

                            for (t_idx, shell_t) in m_j.basis.shells.iter().enumerate() {
                                let local_t = shell_t.atom_index;
                                let global_t = m_j_atom_start + local_t;
                                let atom_t = &atoms[global_t];

                                let dx = atom_s.xyz[0] - atom_t.xyz[0];
                                let dy = atom_s.xyz[1] - atom_t.xyz[1];
                                let dz = atom_s.xyz[2] - atom_t.xyz[2];
                                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                                if dist < 1e-10 {
                                    continue;
                                }

                                let dgamma_dr = m_i.gammafunction.deriv(
                                    dist,
                                    atom_s.number,
                                    shell_s.angular_momentum as u8,
                                    atom_t.number,
                                    shell_t.angular_momentum as u8,
                                );

                                let factor = q_z_shell_all[idx_i][s_idx]
                                    * dq_shell_all[idx_j][t_idx]
                                    * dgamma_dr
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

    /// Complete FMO2-xTB response gradient using gamma_shell.
    pub fn response_gradient_fmo_xtb_shell(&mut self, cn_grad_global: &Array2<f64>) -> Array1<f64> {
        let n_atoms_total = self.atoms.len();
        let n_grad = 3 * n_atoms_total;

        let cn_numbers_global: ArrayView1<f64> = self.properties.cn().unwrap();

        let (lagrangian_vec, qvo_ao_vec, qvo_shell_vec) =
            self.calculate_response_lagrangian_xtb_shell();

        // If all Lagrangians are zero, skip response
        let all_zero = lagrangian_vec
            .iter()
            .all(|l| l.mapv(|x| x.abs()).sum() < 1e-30);
        if all_zero {
            return Array1::zeros(n_grad);
        }

        let z_vectors = self.solve_sczv_cg_xtb_shell(&lagrangian_vec, &qvo_ao_vec, &qvo_shell_vec);

        let gradient_2d = self.response_gradient_onthefly_xtb_shell(
            &z_vectors,
            cn_numbers_global,
            cn_grad_global.view(),
        );

        let inter_grad = self.inter_fragment_response_gradient_xtb_shell(&z_vectors);

        let addlag_resp = self.add_response_addlag_xtb_shell(&z_vectors);

        let total_2d = gradient_2d + inter_grad;
        let total_2d = -1.0 * total_2d;

        let mut result = total_2d
            .into_shape([n_grad])
            .expect("Failed to reshape xTB response gradient");

        // Response addlag is already in 1D global coordinates
        result -= &addlag_resp;

        result
    }

    /// Response addlag: dS/dR * (SHIFTZ * P - 0.5 * P * (SHIFTZ * S) * P)
    ///
    /// SHIFTZ = gamma_intra · Q_Z + gamma_inter · Q_Z_other - third_order
    ///
    /// This is the DFTB_ZVEC_KGRAD term from. It accounts for the
    /// density-dependent embedding potential's effect on the response gradient
    /// through the overlap derivative.
    fn add_response_addlag_xtb_shell(&self, z_vectors: &[Array1<f64>]) -> Array1<f64> {
        let n_atoms_total = self.atoms.len();
        let n_grad = 3 * n_atoms_total;
        let atoms = &self.atoms[..];

        // Pre-compute Q_Z at shell level for all monomers
        let q_z_shell_all: Vec<Array1<f64>> = self
            .monomers
            .iter()
            .enumerate()
            .map(|(idx, m_i)| {
                let z_i = &z_vectors[idx];
                let nocc = m_i.properties.occ_indices().unwrap().len();
                let nvirt = m_i.properties.virt_indices().unwrap().len();
                let orbs: ArrayView2<f64> = m_i.properties.orbs().unwrap();
                let s_mat: ArrayView2<f64> = m_i.properties.s().unwrap();

                let z_mat = z_i.view().into_shape([nvirt, nocc]).unwrap();
                let c_virt = orbs.slice(s![.., nocc..nocc + nvirt]);
                let c_occ = orbs.slice(s![.., ..nocc]);
                let z_ao_unsym = c_virt.dot(&z_mat.dot(&c_occ.t()));
                let z_sym: Array2<f64> = 0.5 * (&z_ao_unsym + &z_ao_unsym.t());
                let zs = z_sym.dot(&s_mat);

                let n_shells = m_i.basis.shells.len();
                let mut q_z = Array1::<f64>::zeros(n_shells);
                for (s_idx, shell) in m_i.basis.shells.iter().enumerate() {
                    for mu in shell.sph_start..shell.sph_end {
                        q_z[s_idx] += zs[[mu, mu]];
                    }
                }
                q_z
            })
            .collect();

        // For each monomer, compute SHIFTZ and addlag gradient
        let local_grads: Vec<Array1<f64>> = self
            .monomers
            .par_iter()
            .enumerate()
            .map(|(idx, m_i)| {
                let n_atoms = m_i.n_atoms;
                let n_orbs = m_i.n_orbs;
                let m_atoms = &atoms[m_i.slice.atom_as_range()];
                let p: ArrayView2<f64> = m_i.properties.p().unwrap();
                let s_mat: ArrayView2<f64> = m_i.properties.s().unwrap();
                let gamma_shell: ArrayView2<f64> = m_i.properties.gamma_shell().unwrap();
                let dq_atom: ArrayView1<f64> = m_i.properties.dq().unwrap();
                let q_z_shell = &q_z_shell_all[idx];

                // Intra-fragment: gamma_shell · Q_Z
                let mut shiftz_shell: Array1<f64> = gamma_shell.dot(q_z_shell);

                // Inter-fragment: gamma(shell_I, shell_J) · Q_Z_J for each J≠I
                let atom_start_i = m_i.slice.atom_as_range().start;
                for (jdx, m_j) in self.monomers.iter().enumerate() {
                    if jdx == idx {
                        continue;
                    }
                    let atom_start_j = m_j.slice.atom_as_range().start;
                    let q_z_j = &q_z_shell_all[jdx];

                    for (s_idx, shell_s) in m_i.basis.shells.iter().enumerate() {
                        let atom_s = &atoms[atom_start_i + shell_s.atom_index];
                        let mut sum = 0.0;
                        for (t_idx, shell_t) in m_j.basis.shells.iter().enumerate() {
                            let atom_t = &atoms[atom_start_j + shell_t.atom_index];
                            let dx = atom_s.xyz[0] - atom_t.xyz[0];
                            let dy = atom_s.xyz[1] - atom_t.xyz[1];
                            let dz = atom_s.xyz[2] - atom_t.xyz[2];
                            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                            let gamma_val = m_i.gammafunction.eval(
                                dist,
                                atom_s.number,
                                shell_s.angular_momentum as u8,
                                atom_t.number,
                                shell_t.angular_momentum as u8,
                            );
                            sum += gamma_val * q_z_j[t_idx];
                        }
                        shiftz_shell[s_idx] += sum;
                    }
                }

                // Third-order contribution: subtract 2 * hubb * dq * Q_Z_atom
                let mut q_z_atom = Array1::<f64>::zeros(n_atoms);
                for (s_idx, shell) in m_i.basis.shells.iter().enumerate() {
                    q_z_atom[shell.atom_index] += q_z_shell[s_idx];
                }

                let mut shiftz_ao =
                    shell_to_ao_values(&m_i.basis, n_orbs, shiftz_shell.view());
                for shell in m_i.basis.shells.iter() {
                    let at = shell.atom_index;
                    let hubb_deriv = COUL_THIRD_ORDER_ATOM[m_atoms[at].number as usize - 1];
                    let third_shift = 2.0 * hubb_deriv * dq_atom[at] * q_z_atom[at];
                    for mu in shell.sph_start..shell.sph_end {
                        shiftz_ao[mu] -= third_shift;
                    }
                }

                let shiftz_mat = aovec_to_aomat(shiftz_ao.view(), n_orbs) * 0.5;

                // WRK = shiftz*P - 0.5*P*(shiftz*S)*P
                let shift_s = &shiftz_mat * &s_mat;
                let d_shift_s = p.dot(&shift_s);
                let d_shift_s_d = d_shift_s.dot(&p);
                let wrk = &(&shiftz_mat * &p) - &(0.5 * &d_shift_s_d);

                // Shell-pair loop: dS/dR * wrk
                let mut local_grad = Array1::<f64>::zeros(3 * n_atoms);

                for (shell_i_idx, shell_i) in m_i.basis.shells.iter().enumerate() {
                    let atomi = &m_atoms[shell_i.atom_index];
                    let at_i = shell_i.atom_index;
                    for (shell_j_idx, shell_j) in m_i.basis.shells.iter().enumerate() {
                        let atomj = &m_atoms[shell_j.atom_index];
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

                        let mut shell_ds_contrib = [0.0f64; 3];
                        if shell_i.angular_momentum < 2 && shell_j.angular_momentum < 2 {
                            for idx_i in shell_i.sph_start..shell_i.sph_end {
                                let il = idx_i - shell_i.sph_start;
                                for idx_j in shell_j.sph_start..shell_j.sph_end {
                                    let jl = idx_j - shell_j.sph_start;
                                    if idx_i != idx_j {
                                        let w_ij = wrk[[idx_i, idx_j]];
                                        let o1 =
                                            &m_i.basis.basis_functions[shell_i.start + il];
                                        let o2 =
                                            &m_i.basis.basis_functions[shell_j.start + jl];
                                        let np =
                                            o1.contracted_norm * o2.contracted_norm;
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
                        let either_d =
                            shell_i.angular_momentum >= 2 || shell_j.angular_momentum >= 2;
                        if either_d && shell_i_idx < shell_j_idx {
                            let ds_d = calc_overlap_derivative_d_shells(
                                &m_i.basis, shell_i, shell_j,
                            );
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

                local_grad
            })
            .collect();

        // Scatter to global
        let mut result = Array1::<f64>::zeros(n_grad);
        for (m_i, local_grad) in self.monomers.iter().zip(local_grads.iter()) {
            for (local_idx, global_idx) in m_i.slice.atom_as_range().enumerate() {
                for k in 0..3 {
                    result[3 * global_idx + k] += local_grad[3 * local_idx + k];
                }
            }
        }

        result
    }
} // end impl XtbSuperSystem
