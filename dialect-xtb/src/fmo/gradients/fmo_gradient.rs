//! Low-memory FMO2-xTB ground-state gradient implementation.
//!
//! This module implements the correct FMO2 gradient following the FMO-DFTB formula,
//! adapted for xTB's AO-level charges and shell-level gamma derivatives.
//!
//! Gradient formula:
//!   total = monomer + pair_delta + addlag + CTIJ + CTMUL_embed + ESD + dispersion
//!
//! Key differences from DFTB:
//! - Charges are AO-level: dq_ao[n_orbs], gamma_ao[n_orbs, n_orbs]
//! - Gamma derivatives are shell-level: gamma_func.deriv(r, Z_a, l_a, Z_b, l_b)
//! - No grad_dq (that is a response term)
//!
//! This replaces the old supersystem.rs implementation which incorrectly included
//! grad_dq response terms and lacked proper CTMUL/SHIFTCT/addlag.

use dialect_base::defaults::PROXIMITY_CUTOFF;
use dialect_utilities::fmo_helpers::{get_pair_slice_xtb, get_trimer_slice_xtb};
use dialect_state::PairType;
use crate::fmo::gradients::monomer::{get_grad_dq_xtb_onthefly, grad_repulsive_energy_xtb};
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

/// Individual gradient components from `ground_state_gradient_fmo_parts()`.
pub struct FmoGradientParts {
    pub monomer_scc: Array1<f64>,
    pub addlag: Array1<f64>,
    pub pair_delta: Array1<f64>,
    pub ctij: Array1<f64>,
    pub interfrag: Array1<f64>,
    pub dispersion: Array1<f64>,
    /// CN correction gradient from using supersystem-level coordination numbers
    pub cn_correction: Array1<f64>,
    /// Per-pair pair delta gradients in global coordinates (for per-pair diagnostics)
    pub per_pair_delta: Vec<Array1<f64>>,
    /// Trimer delta gradient (FMO3 three-body correction)
    pub trimer_delta: Array1<f64>,
    /// Trimer embedding gradient (FMO3)
    pub trimer_embedding: Array1<f64>,
}

// ============================================================================
// Helper functions
// ============================================================================

/// Compute CTMUL in AO basis: ctmul_ao[mu] = sum over pairs/trimers of delta_dq[mu].
///
/// For each pair (I,J), delta_dq = dq_pair_ao - concat(dq_I_ao, dq_J_ao).
/// For each trimer (I,J,K), delta_dq = dq_trimer_ao - concat(dq_I_ao, dq_J_ao, dq_K_ao).
///
/// For FMO3, pairs are scaled by SCAL = 1 - n_trimers_containing_pair.
/// Trimers are always added with SCAL = 1.
fn compute_ctmul_ao(
    monomers: &[XtbMonomer],
    pairs: &[XtbPair],
    trimers: &[XtbTrimer],
    pair_scal: &[f64],
    n_orbs_total: usize,
) -> Array1<f64> {
    let mut ctmul_ao = Array1::<f64>::zeros(n_orbs_total);

    // Pair contributions (scaled by SCAL for FMO3)
    for (pair_idx, pair) in pairs.iter().enumerate() {
        let scal = pair_scal[pair_idx];
        if scal.abs() < 1e-14 {
            continue;
        }
        let m_i = &monomers[pair.i];
        let m_j = &monomers[pair.j];
        let delta_dq: ArrayView1<f64> = pair.properties.delta_dq().unwrap();

        let mut slice_i = ctmul_ao.slice_mut(s![m_i.slice.orb]);
        slice_i += &(scal * &delta_dq.slice(s![..m_i.n_orbs]));

        let mut slice_j = ctmul_ao.slice_mut(s![m_j.slice.orb]);
        slice_j += &(scal * &delta_dq.slice(s![m_i.n_orbs..]));
    }

    // Trimer contributions (SCAL = 1)
    for trimer in trimers.iter() {
        let m_i = &monomers[trimer.i];
        let m_j = &monomers[trimer.j];
        let m_k = &monomers[trimer.k];
        let delta_dq: ArrayView1<f64> = trimer.properties.delta_dq().unwrap();

        let mut slice_i = ctmul_ao.slice_mut(s![m_i.slice.orb]);
        slice_i += &delta_dq.slice(s![..m_i.n_orbs]);

        let mut slice_j = ctmul_ao.slice_mut(s![m_j.slice.orb]);
        slice_j += &delta_dq.slice(s![m_i.n_orbs..m_i.n_orbs + m_j.n_orbs]);

        let mut slice_k = ctmul_ao.slice_mut(s![m_k.slice.orb]);
        slice_k += &delta_dq.slice(s![m_i.n_orbs + m_j.n_orbs..]);
    }

    ctmul_ao
}

/// Compute SHIFTCT + ESPGRAD correction for a monomer, in AO basis.
///
/// SHIFTCT[mu_I] = sum_nu gamma_ao_super[mu_I, nu] * ctmul_ao[nu]  (all orbitals nu)
/// ESPGRAD[mu_I] -= sum_{nu in pair} gamma_ao_super[mu_I, nu] * ctij_ao[nu] * SCAL  (per pair containing I)
/// ESPGRAD[mu_I] -= sum_{nu in trimer} gamma_ao_super[mu_I, nu] * ctijk_ao[nu]  (per trimer containing I)
///
/// Net result = embedding potential from charge transfer, excluding self-interaction.
fn compute_shiftct_espgrad_ao(
    m: &XtbMonomer,
    gamma_ao_super: ArrayView2<f64>,
    ctmul_ao: ArrayView1<f64>,
    monomers: &[XtbMonomer],
    pairs: &[XtbPair],
    trimers: &[XtbTrimer],
    pair_scal: &[f64],
) -> Array1<f64> {
    // SHIFTCT: gamma_ao_super[I_orbs, :] @ ctmul_ao
    let gamma_i_all = gamma_ao_super.slice(s![m.slice.orb, ..]);
    let mut shiftct: Array1<f64> = gamma_i_all.dot(&ctmul_ao);

    // ESPGRAD correction: subtract self-interaction for each pair containing this monomer
    // Scaled by SCAL for FMO3
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
        let delta_dq: ArrayView1<f64> = pair.properties.delta_dq().unwrap();

        let gamma_i_mi = gamma_ao_super.slice(s![m.slice.orb, m_i.slice.orb]);
        let gamma_i_mj = gamma_ao_super.slice(s![m.slice.orb, m_j.slice.orb]);

        let ctij_on_i = delta_dq.slice(s![..m_i.n_orbs]);
        let ctij_on_j = delta_dq.slice(s![m_i.n_orbs..]);

        shiftct -= &(scal * &gamma_i_mi.dot(&ctij_on_i));
        shiftct -= &(scal * &gamma_i_mj.dot(&ctij_on_j));
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
        let delta_dq: ArrayView1<f64> = trimer.properties.delta_dq().unwrap();

        let gamma_m_mi = gamma_ao_super.slice(s![m.slice.orb, m_i.slice.orb]);
        let gamma_m_mj = gamma_ao_super.slice(s![m.slice.orb, m_j.slice.orb]);
        let gamma_m_mk = gamma_ao_super.slice(s![m.slice.orb, m_k.slice.orb]);

        let ctijk_on_i = delta_dq.slice(s![..m_i.n_orbs]);
        let ctijk_on_j = delta_dq.slice(s![m_i.n_orbs..m_i.n_orbs + m_j.n_orbs]);
        let ctijk_on_k = delta_dq.slice(s![m_i.n_orbs + m_j.n_orbs..]);

        shiftct -= &gamma_m_mi.dot(&ctijk_on_i);
        shiftct -= &gamma_m_mj.dot(&ctijk_on_j);
        shiftct -= &gamma_m_mk.dot(&ctijk_on_k);
    }

    shiftct
}

/// Helper: get the self-energy CN gradient coefficient
fn get_self_energy_cn_grad_coeff(z: u8, shell_idx: usize) -> f64 {
    let z_idx: usize = (z - 1) as usize;
    -HAMILTONIAN_KCN_VALUES[z_idx][shell_idx]
}

/// Get pi term gradient
fn get_pi_term_gradient_inline(
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
// Helper: AO-level transition charges for response gradient
// ============================================================================

/// Compute AO-level virtual-occupied transition charges.
///
/// DFTB sums over AOs per atom: `qvo[A, ai] = Σ_{μ∈A} [C_μa·(SC)_μi + C_μi·(SC)_μa]`.
/// xTB keeps per-AO: `qvo_ao[μ, ai] = C_μa·(SC)_μi + C_μi·(SC)_μa`.
///
/// Returns array [n_orbs, nvirt*nocc].
fn compute_qvo_ao(m: &XtbMonomer) -> Array2<f64> {
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

    // qvo_ao[mu, a*nocc + i] = C_virt[mu,a] * SC_occ[mu,i] + C_occ[mu,i] * SC_virt[mu,a]
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

// ============================================================================
// Function A: monomer_gradient_combined
// ============================================================================

/// Combined monomer gradient: monomer SCC + addlag, fused into shell-pair loops.
///
/// Returns (monomer_scc_grad, addlag_grad, cn_grad_global):
/// - monomer_scc_grad: dH·P + dS·(-W + coulomb·P - third·P) + dgamma/dR·dq·dq + v_rep [3*n_atoms_monomer]
/// - addlag_grad: dS·WRK1_addlag + dgamma_shell·ctij_shiftct_correction [3*n_atoms_monomer]
/// - cn_grad_global: CN gradient contribution in global coordinates [3*n_atoms_total]
///
/// The caller uses:
/// - monomer_for_total = monomer_scc_grad + addlag_grad + cn_grad_global (scattered)
/// - monomer_for_delta_subtraction = monomer_scc_grad + cn_grad_global (no addlag)
fn monomer_gradient_combined(
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
    let gamma_ao: ArrayView2<f64> = m.properties.gamma_ao().unwrap();
    let p: ArrayView2<f64> = m.properties.p().unwrap();
    let dq_ao: ArrayView1<f64> = m.properties.dq_ao().unwrap();
    let dq: ArrayView1<f64> = m.properties.dq().unwrap();
    let s: ArrayView2<f64> = m.properties.s().unwrap();
    let orbe: ArrayView1<f64> = m.properties.orbe().unwrap();
    let orbs: ArrayView2<f64> = m.properties.orbs().unwrap();
    let occupations: Array1<f64> = Array::from(m.properties.occupation().unwrap().to_vec());
    let esp_q: ArrayView1<f64> = m.properties.esp_q().unwrap();

    // Compute energy-weighted density matrix W
    let weighted_orbe = &orbe * &occupations;
    let worbe_2d: Array2<f64> = Array2::from_diag(&weighted_orbe);
    let w: Array2<f64> = orbs.dot(&worbe_2d.dot(&orbs.t()));

    // Coulomb potential: gamma_ao . dq_ao, then aovec_to_aomat
    let gamma_dq: Array1<f64> = gamma_ao.dot(&dq_ao);
    // Total shift for monomer: gamma_ao . dq_ao + esp_q (intra + inter-fragment)
    let total_shift: Array1<f64> = &gamma_dq + &esp_q;
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

    // Precompute shell-level dq sums for gamma gradient
    let n_shells = m.basis.shells.len();
    let mut dq_shell: Vec<f64> = vec![0.0; n_shells];
    for (shell_idx, shell) in m.basis.shells.iter().enumerate() {
        for idx in shell.sph_start..shell.sph_end {
            dq_shell[shell_idx] += dq_ao[idx];
        }
    }

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
            let cn_coeff_i = get_self_energy_cn_grad_coeff(atomi.number, shell_i.shell_index);
            let cn_coeff_j = get_self_energy_cn_grad_coeff(atomj.number, shell_j.shell_index);

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
                get_pi_term_gradient_inline(
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
    // cn_grad_global is [3*N_total, N_total], so moving any atom affects CN of any atom.
    // We accumulate into a global-sized array.
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

    // NOTE: Dispersion is NOT included here — it's added globally once in the assembly function

    (grad_monomer, grad_addlag, cn_grad_contribution)
}

// ============================================================================
// Function B: pair_gradient_combined
// ============================================================================

/// Combined pair gradient: pair SCC + CTIJ, fused into shell-pair loops.
///
/// Returns (pair_grad_local, ctij_grad_global, cn_grad_global).
/// - pair_grad_local: gradient in local pair coordinates [3*n_pair_atoms]
/// - ctij_grad_global: CTIJ contribution in global coordinates [3*n_atoms_total]
/// - cn_grad_global: CN gradient contribution in global coordinates [3*n_atoms_total]
fn pair_gradient_combined(
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
    let gamma_ao: ArrayView2<f64> = pair.properties.gamma_ao().unwrap();
    let p: ArrayView2<f64> = pair.properties.p().unwrap();
    let dq_ao: ArrayView1<f64> = pair.properties.dq_ao().unwrap();
    let dq: ArrayView1<f64> = pair.properties.dq().unwrap();
    let s: ArrayView2<f64> = pair.properties.s().unwrap();
    let orbe: ArrayView1<f64> = pair.properties.orbe().unwrap();
    let orbs: ArrayView2<f64> = pair.properties.orbs().unwrap();
    let occupations: Array1<f64> = Array::from(pair.properties.occupation().unwrap().to_vec());

    // Compute W
    let weighted_orbe = &orbe * &occupations;
    let worbe_2d: Array2<f64> = Array2::from_diag(&weighted_orbe);
    let w: Array2<f64> = orbs.dot(&worbe_2d.dot(&orbs.t()));

    // Compute total shift = gamma_pair * dq_pair + ESP from K != I,J
    let gamma_dq: Array1<f64> = gamma_ao.dot(&dq_ao);

    // ESP from K: esp_q_I/J already has full supersystem potential minus self-interaction
    // For pair, we need: esp_from_K = esp_q(monomer) - gamma(I,J)*dq_J
    // This is the ESP from all K≠I,J
    let dq_mon_i_ao: ArrayView1<f64> = m_i.properties.dq_ao().unwrap();
    let dq_mon_j_ao: ArrayView1<f64> = m_j.properties.dq_ao().unwrap();
    let gamma_ij: ArrayView2<f64> = gamma_ao.slice(s![0..m_i.n_orbs, m_i.n_orbs..]);

    let mut esp_from_k: Array1<f64> = Array1::zeros(n_orbs);
    let esp_i: ArrayView1<f64> = m_i.properties.esp_q().unwrap();
    esp_from_k
        .slice_mut(s![0..m_i.n_orbs])
        .assign(&(&esp_i - &gamma_ij.dot(&dq_mon_j_ao)));
    let esp_j: ArrayView1<f64> = m_j.properties.esp_q().unwrap();
    esp_from_k
        .slice_mut(s![m_i.n_orbs..])
        .assign(&(&esp_j - &gamma_ij.t().dot(&dq_mon_i_ao)));

    let total_shift: Array1<f64> = &gamma_dq + &esp_from_k;
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

    // Build CTIJ and monomer charges in shell basis for gamma gradient
    let delta_dq: ArrayView1<f64> = pair.properties.delta_dq().unwrap();

    // Shell-level CTIJ and monomer charges for gamma derivative
    let n_shells = pair.basis.shells.len();
    let mut ctij_shell: Vec<f64> = vec![0.0; n_shells];
    let mut dq_mon_shell: Vec<f64> = vec![0.0; n_shells];
    let mut dq_shell: Vec<f64> = vec![0.0; n_shells];

    // Build concatenated monomer dq_ao for dq_mon
    let mut dq_mon_ao_concat: Array1<f64> = Array1::zeros(n_orbs);
    dq_mon_ao_concat
        .slice_mut(s![..m_i.n_orbs])
        .assign(&dq_mon_i_ao);
    dq_mon_ao_concat
        .slice_mut(s![m_i.n_orbs..])
        .assign(&dq_mon_j_ao);

    for (s_idx, shell) in pair.basis.shells.iter().enumerate() {
        for mu in shell.sph_start..shell.sph_end {
            ctij_shell[s_idx] += delta_dq[mu];
            dq_mon_shell[s_idx] += dq_mon_ao_concat[mu];
            dq_shell[s_idx] += dq_ao[mu];
        }
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

            let cn_coeff_i = get_self_energy_cn_grad_coeff(atomi.number, shell_i.shell_index);
            let cn_coeff_j = get_self_energy_cn_grad_coeff(atomj.number, shell_j.shell_index);

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
                get_pi_term_gradient_inline(
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
                let shell_dq_prod = dq_shell[shell_i_idx] * dq_shell[shell_j_idx];
                let shell_gamma_contrib = 0.5 * gamma_deriv * shell_dq_prod;
                for dir in 0..3 {
                    grad_local[3 * at_i + dir] += e_ij[dir] * shell_gamma_contrib;
                    grad_local[3 * at_j + dir] -= e_ij[dir] * shell_gamma_contrib;
                }

                // CTIJ: -dgamma * ctij_shell[i] * dq_mon_shell[j]
                // The double loop naturally gives both (a,c) and (c,a) terms
                let shell_ctij_contrib =
                    -gamma_deriv * ctij_shell[shell_i_idx] * dq_mon_shell[shell_j_idx];

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

    // NOTE: Dispersion NOT included here

    (grad_local, ctij_grad_global, cn_grad_contribution)
}

// ============================================================================
// Function B2: trimer_gradient_combined (for FMO3)
// ============================================================================

/// Combined trimer gradient: trimer SCC fused into shell-pair loops.
///
/// Returns (trimer_grad_local, ctijk_grad_global, cn_grad_global):
/// - trimer_grad_local: gradient in local trimer coordinates [3*n_trimer_atoms]
/// - ctijk_grad_global: CTIJK gamma gradient in global coordinates [3*n_atoms_total]
///   (analogous to pair's ctij_grad: -dgamma * ctijk_shell[i] * dq_mon_shell[j])
/// - cn_grad_global: CN gradient contribution in global coordinates [3*n_atoms_total]
fn trimer_gradient_combined(
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
    let gamma_ao: ArrayView2<f64> = trimer.properties.gamma_ao().unwrap();
    let p: ArrayView2<f64> = trimer.properties.p().unwrap();
    let dq_ao: ArrayView1<f64> = trimer.properties.dq_ao().unwrap();
    let dq: ArrayView1<f64> = trimer.properties.dq().unwrap();
    let s: ArrayView2<f64> = trimer.properties.s().unwrap();
    let orbe: ArrayView1<f64> = trimer.properties.orbe().unwrap();
    let orbs: ArrayView2<f64> = trimer.properties.orbs().unwrap();
    let occupations: Array1<f64> = Array::from(trimer.properties.occupation().unwrap().to_vec());

    // Compute W
    let weighted_orbe = &orbe * &occupations;
    let worbe_2d: Array2<f64> = Array2::from_diag(&weighted_orbe);
    let w: Array2<f64> = orbs.dot(&worbe_2d.dot(&orbs.t()));

    // Compute total shift = gamma_trimer * dq_trimer + ESP from L != I,J,K
    let gamma_dq: Array1<f64> = gamma_ao.dot(&dq_ao);

    // ESP from L: For trimer, subtract inter-monomer gamma contributions from monomer esp_q
    let dq_mon_i_ao: ArrayView1<f64> = m_i.properties.dq_ao().unwrap();
    let dq_mon_j_ao: ArrayView1<f64> = m_j.properties.dq_ao().unwrap();
    let dq_mon_k_ao: ArrayView1<f64> = m_k.properties.dq_ao().unwrap();

    let gamma_ij: ArrayView2<f64> =
        gamma_ao.slice(s![0..m_i.n_orbs, m_i.n_orbs..m_i.n_orbs + m_j.n_orbs]);
    let gamma_ik: ArrayView2<f64> = gamma_ao.slice(s![0..m_i.n_orbs, m_i.n_orbs + m_j.n_orbs..]);
    let gamma_jk: ArrayView2<f64> = gamma_ao.slice(s![
        m_i.n_orbs..m_i.n_orbs + m_j.n_orbs,
        m_i.n_orbs + m_j.n_orbs..
    ]);

    let mut esp_from_l: Array1<f64> = Array1::zeros(n_orbs);
    let esp_i: ArrayView1<f64> = m_i.properties.esp_q().unwrap();
    esp_from_l
        .slice_mut(s![0..m_i.n_orbs])
        .assign(&(&esp_i - &gamma_ij.dot(&dq_mon_j_ao) - &gamma_ik.dot(&dq_mon_k_ao)));
    let esp_j: ArrayView1<f64> = m_j.properties.esp_q().unwrap();
    esp_from_l
        .slice_mut(s![m_i.n_orbs..m_i.n_orbs + m_j.n_orbs])
        .assign(&(&esp_j - &gamma_ij.t().dot(&dq_mon_i_ao) - &gamma_jk.dot(&dq_mon_k_ao)));
    let esp_k: ArrayView1<f64> = m_k.properties.esp_q().unwrap();
    esp_from_l
        .slice_mut(s![m_i.n_orbs + m_j.n_orbs..])
        .assign(&(&esp_k - &gamma_ik.t().dot(&dq_mon_i_ao) - &gamma_jk.t().dot(&dq_mon_j_ao)));

    let total_shift: Array1<f64> = &gamma_dq + &esp_from_l;
    let coulomb_mat: Array2<f64> = aovec_to_aomat(total_shift.view(), n_orbs) * 0.5;

    // Third order terms
    let mut hubbard_derivatives: Array1<f64> = Array1::zeros(n_atoms_trimer);
    for (val, atom) in hubbard_derivatives.iter_mut().zip(trimer_atoms.iter()) {
        *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
    }
    let dq2_gamma: Array2<f64> =
        coul_third_order_grad_contribution_xtb(&trimer.basis(), dq, hubbard_derivatives.view());

    // Effective matrix
    let effective_mat: Array2<f64> = &(-&w) + &(&coulomb_mat * &p) - &(0.5 * &dq2_gamma * &p);

    // Shell-level dq, CTIJK, and monomer charges for gamma gradient
    let delta_dq: ArrayView1<f64> = trimer.properties.delta_dq().unwrap();
    let n_shells = trimer.basis().shells.len();
    let mut dq_shell: Vec<f64> = vec![0.0; n_shells];
    let mut ctijk_shell: Vec<f64> = vec![0.0; n_shells];
    let mut dq_mon_shell: Vec<f64> = vec![0.0; n_shells];

    // Build concatenated monomer dq_ao
    let mut dq_mon_ao_concat: Array1<f64> = Array1::zeros(n_orbs);
    dq_mon_ao_concat
        .slice_mut(s![..m_i.n_orbs])
        .assign(&dq_mon_i_ao);
    dq_mon_ao_concat
        .slice_mut(s![m_i.n_orbs..m_i.n_orbs + m_j.n_orbs])
        .assign(&dq_mon_j_ao);
    dq_mon_ao_concat
        .slice_mut(s![m_i.n_orbs + m_j.n_orbs..])
        .assign(&dq_mon_k_ao);

    for (s_idx, shell) in trimer.basis().shells.iter().enumerate() {
        for mu in shell.sph_start..shell.sph_end {
            dq_shell[s_idx] += dq_ao[mu];
            ctijk_shell[s_idx] += delta_dq[mu];
            dq_mon_shell[s_idx] += dq_mon_ao_concat[mu];
        }
    }

    // Local-to-global atom index mapping
    let local_to_global: Vec<usize> = m_i
        .slice
        .atom_as_range()
        .chain(m_j.slice.atom_as_range())
        .chain(m_k.slice.atom_as_range())
        .collect();

    // Use global coordination numbers for the trimer's atoms
    let cn_numbers: Array1<f64> = {
        let mut cn = Array1::<f64>::zeros(n_atoms_trimer);
        for (local_idx, &global_idx) in local_to_global.iter().enumerate() {
            cn[local_idx] = cn_numbers_global[global_idx];
        }
        cn
    };

    // Initialize gradients
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

            let cn_coeff_i = get_self_energy_cn_grad_coeff(atomi.number, shell_i.shell_index);
            let cn_coeff_j = get_self_energy_cn_grad_coeff(atomj.number, shell_j.shell_index);

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
                get_pi_term_gradient_inline(
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
                let ds_d = calc_overlap_derivative_d_shells(&trimer.basis(), shell_i, shell_j);
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
    for at in 0..n_atoms_trimer {
        if cn_factors[at].abs() > 1e-15 {
            let global_at = local_to_global[at];
            let cn_grad_at: ArrayView1<f64> = cn_grad_global.slice(s![.., global_at]);
            for k in 0..(3 * n_atoms_total) {
                cn_grad_contribution[k] += cn_factors[at] * cn_grad_at[k];
            }
        }
    }

    // Gamma gradient — separate loop without PROXIMITY_CUTOFF
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

                // Trimer Coulomb: 0.5 * dgamma * dq_trimer_shell[i] * dq_trimer_shell[j]
                let shell_dq_prod = dq_shell[shell_i_idx] * dq_shell[shell_j_idx];
                let shell_gamma_contrib = 0.5 * gamma_deriv * shell_dq_prod;
                for dir in 0..3 {
                    grad_local[3 * at_i + dir] += e_ij[dir] * shell_gamma_contrib;
                    grad_local[3 * at_j + dir] -= e_ij[dir] * shell_gamma_contrib;
                }

                // CTIJK: -dgamma * ctijk_shell[i] * dq_mon_shell[j]
                // The double loop naturally gives both (a,c) and (c,a) terms
                let shell_ctijk_contrib =
                    -gamma_deriv * ctijk_shell[shell_i_idx] * dq_mon_shell[shell_j_idx];

                let global_i = local_to_global[at_i];
                let global_j = local_to_global[at_j];

                for dir in 0..3 {
                    ctijk_grad_global[3 * global_i + dir] += e_ij[dir] * shell_ctijk_contrib;
                    ctijk_grad_global[3 * global_j + dir] -= e_ij[dir] * shell_ctijk_contrib;
                }
            }
        }
    }

    // Repulsive energy gradient
    let grad_rep = grad_repulsive_energy_xtb(trimer_atoms, n_atoms_trimer);
    grad_local += &grad_rep;

    // NOTE: Dispersion NOT included here

    (grad_local, ctijk_grad_global, cn_grad_contribution)
}

// ============================================================================
// Function C: interfragment_gradient_xtb (CTMUL + ESD fused)
// ============================================================================

/// Combined CTMUL embedding + ES-dimer gradient.
///
/// For each monomer I, loops over all shells (from ANY fragment, including I itself)
/// and monomer I shells:
/// - CTMUL: ctmul_shell[s_a] * dq_I_shell[s_c] * dgamma/dR
/// - ESD: dq_J_shell[s_a] * dq_I_shell[s_c] * dgamma/dR  (if fragments are ESD pair)
///
/// IMPORTANT: Intra-fragment CTMUL contributions (where s_a belongs to monomer I)
/// are included because they cancel with the intra-fragment CTIJ terms from
/// pair_gradient_combined. Only ESD is restricted to inter-fragment pairs.
fn interfragment_gradient_xtb(
    atoms: &[XtbAtom],
    monomers: &[XtbMonomer],
    esd_pairs: &[XtbESDPair],
    ctmul_ao: ArrayView1<f64>,
    dq_ao_global: ArrayView1<f64>,
    super_basis: &Basis,
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
    let mut esd_lookup: HashSet<(usize, usize)> = HashSet::new();
    for esd in esd_pairs.iter() {
        esd_lookup.insert((esd.i, esd.j));
        esd_lookup.insert((esd.j, esd.i));
    }

    // Pre-compute shell-level ctmul sums using supersystem basis
    let n_super_shells = super_basis.shells.len();
    let mut ctmul_shell: Vec<f64> = vec![0.0; n_super_shells];
    for (s_idx, shell) in super_basis.shells.iter().enumerate() {
        for mu in shell.sph_start..shell.sph_end {
            ctmul_shell[s_idx] += ctmul_ao[mu];
        }
    }

    // Pre-compute shell-level dq sums from global dq_ao using supersystem basis
    // These are the monomer charges in global orbital indexing
    let mut dq_global_shell: Vec<f64> = vec![0.0; n_super_shells];
    for (s_idx, shell) in super_basis.shells.iter().enumerate() {
        for mu in shell.sph_start..shell.sph_end {
            dq_global_shell[s_idx] += dq_ao_global[mu];
        }
    }

    // For each monomer I
    for m_i in monomers.iter() {
        let dq_i_ao: ArrayView1<f64> = m_i.properties.dq_ao().unwrap();

        // Pre-compute shell-level charges for monomer I (local basis)
        let n_shells_i = m_i.basis.shells.len();
        let mut dq_i_shell: Vec<f64> = vec![0.0; n_shells_i];
        for (s_idx, shell) in m_i.basis.shells.iter().enumerate() {
            for mu in shell.sph_start..shell.sph_end {
                dq_i_shell[s_idx] += dq_i_ao[mu];
            }
        }

        // Atom offset for monomer I in global coordinates
        let m_i_atom_start = m_i.slice.atom_as_range().start;

        // For each supersystem shell s_a (from ANY fragment, including I itself)
        for (s_a_idx, shell_a) in super_basis.shells.iter().enumerate() {
            let at_a = shell_a.atom_index; // global atom index
            let frag_a = atom_to_frag[at_a];

            // NOTE: Do NOT skip intra-fragment (frag_a == m_i.index)!
            // Intra-fragment CTMUL contributions are needed to cancel with
            // intra-fragment CTIJ terms from pair_gradient_combined.

            // ESD check: only count when frag_a > m_i.index to avoid double-counting
            let is_esd = frag_a > m_i.index && esd_lookup.contains(&(m_i.index, frag_a));

            // For ESD, we use the global dq_ao summed over the supersystem shell
            let dq_esd_shell_a = if is_esd {
                dq_global_shell[s_a_idx]
            } else {
                0.0
            };

            let ct_a = ctmul_shell[s_a_idx];

            // Skip if neither CTMUL nor ESD contribute
            if ct_a.abs() < 1e-14 && dq_esd_shell_a.abs() < 1e-14 {
                continue;
            }

            let atom_a = &atoms[at_a];

            // For each shell s_c in monomer I
            for (s_c_idx, shell_c) in m_i.basis.shells.iter().enumerate() {
                let local_c = shell_c.atom_index; // local atom index within monomer I
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

                // Shell-level gamma derivative
                let dgamma_dr = m_i.gammafunction.deriv(
                    dist,
                    atom_a.number,
                    shell_a.angular_momentum as u8,
                    atom_c.number,
                    shell_c.angular_momentum as u8,
                );

                let total_factor = (ct_a + dq_esd_shell_a) * dq_i_shell[s_c_idx] * dgamma_dr / dist;

                gradient[3 * at_a + 0] += total_factor * dx;
                gradient[3 * at_a + 1] += total_factor * dy;
                gradient[3 * at_a + 2] += total_factor * dz;
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

impl XtbSuperSystem<'_> {
    /// Compute complete FMO2-xTB gradient using the correct FMO formula.
    ///
    /// Total = Monomer + Pair_delta + CTIJ + CTMUL_embed + ESD + Addlag + Dispersion + CN_correction
    ///
    /// No grad_dq (response) terms are included.
    pub fn ground_state_gradient_fmo(&mut self) -> Array1<f64> {
        let atoms: &[XtbAtom] = &self.atoms[..];
        let n_atoms_total = atoms.len();
        let n_grad = 3 * n_atoms_total;

        // Compute global CN and CN gradient once
        let cn_numbers_global: ArrayView1<f64> = self.properties.cn().unwrap();
        let cn_grad_global: Array2<f64> =
            crate::gradients::hamiltonian::calculate_coordination_number_gradients(atoms);

        // Step 0: Compute pair SCAL factors for FMO3
        // SCAL = 1 - n_trimers_containing_pair (inclusion-exclusion)
        let pair_scal: Vec<f64> = if self.config.fmo.use_three_body {
            let mut scal = vec![1.0f64; self.pairs.len()];
            for trimer in self.trimers.iter() {
                for &(a, b) in &[
                    (trimer.i, trimer.j),
                    (trimer.i, trimer.k),
                    (trimer.j, trimer.k),
                ] {
                    if self.properties.type_of_pair_reduced(a, b) == PairType::Pair {
                        let idx = self.properties.index_of_pair(a, b);
                        scal[idx] -= 1.0;
                    }
                }
            }
            scal
        } else {
            vec![1.0f64; self.pairs.len()]
        };

        // Step 1: Compute CTMUL_ao for all orbitals (with trimers and SCAL for FMO3)
        let trimers_ref = &self.trimers;
        let ctmul_ao = compute_ctmul_ao(
            &self.monomers,
            &self.pairs,
            trimers_ref,
            &pair_scal,
            self.n_orbs,
        );

        // Step 2: Compute SHIFTCT + ESPGRAD per monomer [parallel]
        let gamma_ao_super: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let monomers_ref = &self.monomers;
        let pairs_ref = &self.pairs;
        let shiftcts: Vec<Array1<f64>> = self
            .monomers
            .par_iter()
            .map(|m| {
                compute_shiftct_espgrad_ao(
                    m,
                    gamma_ao_super,
                    ctmul_ao.view(),
                    monomers_ref,
                    pairs_ref,
                    trimers_ref,
                    &pair_scal,
                )
            })
            .collect();

        // Step 3: Monomer gradients (SCC + addlag fused) [parallel]
        let cn_grad_view = cn_grad_global.view();
        let monomer_results: Vec<(Array1<f64>, Array1<f64>, Array1<f64>)> = self
            .monomers
            .par_iter()
            .zip(shiftcts.par_iter())
            .map(|(m, shiftct)| {
                monomer_gradient_combined(m, atoms, shiftct.view(), cn_numbers_global, cn_grad_view)
            })
            .collect();

        // Assemble monomer gradients into global arrays
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

        // Step 4: Pair gradients (SCC + CTIJ fused) [parallel]
        let pair_results: Vec<(Array1<f64>, Array1<f64>, Array1<f64>)> = self
            .pairs
            .par_iter()
            .map(|pair| {
                let m_i = &self.monomers[pair.i];
                let m_j = &self.monomers[pair.j];
                let pair_atoms: Vec<XtbAtom> =
                    get_pair_slice_xtb(atoms, m_i.slice.atom_as_range(), m_j.slice.atom_as_range());
                pair_gradient_combined(
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

        // Step 5: Pair delta + CTIJ accumulation (sequential reduction)
        let mut pair_delta_total = Array1::<f64>::zeros(n_grad);
        let mut ctij_total = Array1::<f64>::zeros(n_grad);

        for (pair_idx, (pair, (pair_grad_local, ctij_grad_global, pair_cn_glob))) in
            self.pairs.iter().zip(pair_results.iter()).enumerate()
        {
            let m_i = &self.monomers[pair.i];
            let m_j = &self.monomers[pair.j];

            let mon_i_grad = &monomer_results[pair.i].0;
            let mon_j_grad = &monomer_results[pair.j].0;

            // Pair delta = pair_grad - monomer_grad_I - monomer_grad_J
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

            // CTIJ already in global coordinates, scaled by pair_scal for FMO3
            let scal = pair_scal[pair_idx];
            ctij_total += &(scal * ctij_grad_global);

            // CN delta: pair_cn - mon_I_cn - mon_J_cn
            let mon_i_cn = &monomer_results[pair.i].2;
            let mon_j_cn = &monomer_results[pair.j].2;
            cn_grad_total += &(pair_cn_glob - mon_i_cn - mon_j_cn);
        }

        // Step 6: Inter-fragment gradient (CTMUL + ESD fused)
        let dq_ao_global: ArrayView1<f64> = self.properties.dq_ao().unwrap();
        let interfrag_grad = interfragment_gradient_xtb(
            atoms,
            &self.monomers,
            &self.esd_pairs,
            ctmul_ao.view(),
            dq_ao_global,
            &self.basis,
        );

        // Step 7: Global dispersion
        let disp_grad = gradient_disp3_xtb(&self.atoms, &self.config);
        let halogen_grad = gradient_halogen_bonding_xtb(&self.atoms);

        // Step 8: FMO3 three-body correction (SCAL-based, no separate embedding)
        //
        // With the SCAL approach, the FMO3 embedding is handled via:
        // - CTMUL includes trimer delta_dq (Step 1) and pair delta_dq * SCAL
        // - SHIFTCT subtracts self-interaction for trimers (Step 2)
        // - CTIJ is scaled by SCAL (Step 5)
        // - CTIJK gradient is added here (new)
        // - Trimer SCC delta is the many-body subtraction (standard)
        //
        // No trimer_embedding_gradient_low_memory() call needed.
        let trimer_contribution = if self.config.fmo.use_three_body {
            // Step 8b: Trimer gradients + CTIJK [parallel]
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
                    trimer_gradient_combined(
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

            // Step 8c: ESD pair gradients for trimer subtraction
            // Only the dgamma/dR term is needed here — the grad_dq (response) terms
            // are handled by the Lagrangian/SCZV-CG framework, matching the FMO2 ESD
            // gradient in interfragment_gradient_xtb (Step 6) which also only has dgamma/dR.
            let dq_ao_global: ArrayView1<f64> = self.properties.dq_ao().unwrap();
            let esd_pair_grads: Vec<Array1<f64>> = self
                .esd_pairs
                .par_iter()
                .map(|esd_pair| {
                    let m_i = &self.monomers[esd_pair.i];
                    let m_j = &self.monomers[esd_pair.j];
                    let dq_i: ArrayView1<f64> = dq_ao_global.slice(s![m_i.slice.orb]);
                    let dq_j: ArrayView1<f64> = dq_ao_global.slice(s![m_j.slice.orb]);

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
                        dq_i,
                        dq_j,
                        (0, m_i.n_orbs),
                        (m_i.n_orbs, basis.nbas),
                        (0, m_i.n_atoms),
                    );
                    let grad_gamma_term_j = gamma_gradient_xtb_double_contracted(
                        m_i.gammafunction,
                        &pair_atoms,
                        &basis,
                        dq_i,
                        dq_j,
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

            // Step 8d: Build per-pair delta gradients in global coordinates (for trimer subtraction)
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
                // Include CN delta in pair delta
                delta += &(pair_cn_glob - mon_i_cn - mon_j_cn);
                per_pair_delta_global.push(delta);
            }

            // Step 8e: Trimer delta + CTIJK assembly
            let mut trimer_delta_total = Array1::<f64>::zeros(n_grad);
            let mut ctijk_total = Array1::<f64>::zeros(n_grad);

            for (trimer, (tri_grad_local, tri_ctijk_glob, tri_cn)) in
                self.trimers.iter().zip(trimer_results.iter())
            {
                let m_i = &self.monomers[trimer.i];
                let m_j = &self.monomers[trimer.j];
                let m_k = &self.monomers[trimer.k];

                // Scatter trimer grad to global and subtract monomer grads
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

                // CN delta: tri_cn - mon_I_cn - mon_J_cn - mon_K_cn
                delta_global += &(tri_cn
                    - &monomer_results[trimer.i].2
                    - &monomer_results[trimer.j].2
                    - &monomer_results[trimer.k].2);

                // Subtract pair/ESD delta gradients for (I,J), (I,K), (J,K)
                for &(a, b) in &[
                    (trimer.i, trimer.j),
                    (trimer.i, trimer.k),
                    (trimer.j, trimer.k),
                ] {
                    if self.properties.type_of_pair_reduced(a, b) == PairType::Pair {
                        let idx = self.properties.index_of_pair(a, b);
                        delta_global -= &per_pair_delta_global[idx];
                    } else {
                        // ESD pair: scatter local ESD gradient to global and subtract
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

                // CTIJK already in global coordinates (SCAL = 1 for trimers)
                ctijk_total += tri_ctijk_glob;
            }
            // let trimer_rms = (trimer_delta_total.mapv(|x| x * x).sum()
            //     / trimer_delta_total.len() as f64)
            //     .sqrt();
            // let ctijk_rms =
            //     (ctijk_total.mapv(|x| x * x).sum() / ctijk_total.len() as f64).sqrt();
            // eprintln!("  FMO3 trimer delta RMS:  {:.6e}", trimer_rms);
            // eprintln!("  FMO3 CTIJK grad RMS:    {:.6e}", ctijk_rms);

            &trimer_delta_total + &ctijk_total
        } else {
            Array1::zeros(n_grad)
        };

        // Step 9: Assembly
        // total = monomer + pair_delta + CTIJ + CTMUL_embed + ESD + addlag + dispersion + cn_correction + trimer
        // let cn_rms = (cn_grad_total.mapv(|x| x * x).sum() / cn_grad_total.len() as f64).sqrt();
        // eprintln!("  CN correction RMS in total: {:.6e}", cn_rms);
        &monomer_grad_total
            + &pair_delta_total
            + &ctij_total
            + &interfrag_grad
            + &addlag_total
            + &disp_grad
            + &halogen_grad
            + &cn_grad_total
            + &trimer_contribution
    }

    // ========================================================================
    // Response gradient: xTB version (AO-level charges)
    // ========================================================================

    /// Calculate response Lagrangian for xTB at AO level.
    ///
    /// For each monomer K, the Lagrangian is:
    ///   L^K_ai = -0.5 * Σ_{pairs IJ, K∉{I,J}} [gamma_ao_super[K_orbs, IJ_orbs] · delta_dq_ao[IJ]] · qvo_ao_K
    ///
    /// Returns (lagrangians, qvo_ao_vectors) for each monomer.
    fn calculate_response_lagrangian_xtb(&self) -> (Vec<Array1<f64>>, Vec<Array2<f64>>) {
        let gamma_ao_super: ArrayView2<f64> = self.properties.gamma_ao().unwrap();

        // Compute pair SCAL factors for FMO3
        let pair_scal: Vec<f64> = if self.config.fmo.use_three_body {
            let mut scal = vec![1.0f64; self.pairs.len()];
            for trimer in self.trimers.iter() {
                for &(a, b) in &[
                    (trimer.i, trimer.j),
                    (trimer.i, trimer.k),
                    (trimer.j, trimer.k),
                ] {
                    if self.properties.type_of_pair_reduced(a, b) == PairType::Pair {
                        let idx = self.properties.index_of_pair(a, b);
                        scal[idx] -= 1.0;
                    }
                }
            }
            scal
        } else {
            vec![1.0f64; self.pairs.len()]
        };

        // Build qvo_ao for each monomer
        let qvo_ao_vec: Vec<Array2<f64>> =
            self.monomers.iter().map(|m| compute_qvo_ao(m)).collect();

        // Calculate Lagrangian for each monomer K
        // L_K = -0.5 * [Σ_{pairs not containing K} SCAL * ESP_on_K(CTIJ)
        //             + Σ_{trimers not containing K} ESP_on_K(CTIJK)] . qvo_K
        let lagrangian_vec: Vec<Array1<f64>> = self
            .monomers
            .iter()
            .enumerate()
            .map(|(idx_k, m_k)| {
                let nocc = m_k.properties.occ_indices().unwrap().len();
                let nvirt = m_k.properties.virt_indices().unwrap().len();
                let nvo = nvirt * nocc;
                let qvo_k: &Array2<f64> = &qvo_ao_vec[idx_k];

                // gamma_ao_super slice for K's orbitals vs all orbitals
                let gamma_k_all = gamma_ao_super.slice(s![m_k.slice.orb, ..]);

                let mut lag_k = Array1::<f64>::zeros(nvo);

                // Sum over all pairs IJ where K is not involved, scaled by SCAL
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

                    let delta_dq: ArrayView1<f64> = pair.properties.delta_dq().unwrap();

                    let gamma_k_i = gamma_k_all.slice(s![.., m_i.slice.orb]);
                    let gamma_k_j = gamma_k_all.slice(s![.., m_j.slice.orb]);

                    let esp_on_k: Array1<f64> = gamma_k_i.dot(&delta_dq.slice(s![..m_i.n_orbs]))
                        + gamma_k_j.dot(&delta_dq.slice(s![m_i.n_orbs..]));

                    lag_k += &(-0.5 * scal * esp_on_k.dot(qvo_k));
                }

                // FMO3: Sum over all trimers not containing K (SCAL = 1 for trimers)
                if self.config.fmo.use_three_body {
                    for trimer in self.trimers.iter() {
                        let m_ti = &self.monomers[trimer.i];
                        let m_tj = &self.monomers[trimer.j];
                        let m_tk = &self.monomers[trimer.k];

                        // Skip trimers that contain this monomer K
                        if m_ti.index == m_k.index
                            || m_tj.index == m_k.index
                            || m_tk.index == m_k.index
                        {
                            continue;
                        }

                        let delta_dq_ijk: ArrayView1<f64> = trimer.properties.delta_dq().unwrap();

                        // ESP on K from trimer's CTIJK (direct, no subtraction of pairs)
                        let gamma_k_ti = gamma_k_all.slice(s![.., m_ti.slice.orb]);
                        let gamma_k_tj = gamma_k_all.slice(s![.., m_tj.slice.orb]);
                        let gamma_k_tk = gamma_k_all.slice(s![.., m_tk.slice.orb]);

                        let esp_on_k: Array1<f64> = gamma_k_ti
                            .dot(&delta_dq_ijk.slice(s![..m_ti.n_orbs]))
                            + gamma_k_tj.dot(
                                &delta_dq_ijk.slice(s![m_ti.n_orbs..m_ti.n_orbs + m_tj.n_orbs]),
                            )
                            + gamma_k_tk.dot(&delta_dq_ijk.slice(s![m_ti.n_orbs + m_tj.n_orbs..]));

                        lag_k += &(-0.5 * esp_on_k.dot(qvo_k));
                    }
                }

                lag_k
            })
            .collect();

        (lagrangian_vec, qvo_ao_vec)
    }

    /// Compute the matrix-vector product A_I · v for a single xTB fragment.
    ///
    /// A_I · v = diag(eps_i - eps_a) · v  -  Q_ao^T · gamma_ao · Q_ao · v
    fn orbital_hessian_matvec_xtb(
        m_i: &XtbMonomer,
        qvo_ao: &Array2<f64>,
        gamma_ao: ArrayView2<f64>,
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

        // Coulomb: -1 * Q_ao^T @ gamma_ao @ Q_ao @ v
        let qv: Array1<f64> = qvo_ao.dot(v); // [n_orbs]
        let g_qv: Array1<f64> = gamma_ao.dot(&qv); // [n_orbs]
        let qt_g_qv: Array1<f64> = qvo_ao.t().dot(&g_qv); // [nvo]
        result -= &qt_g_qv;

        result
    }

    /// Matrix-free SCZV solver for xTB using preconditioned conjugate gradient.
    ///
    /// Solves the coupled system:
    ///   A_I · Z_I + sum_{K≠I} A_{K,I}^T · Z_K = L_I   for all fragments I
    ///
    /// Uses AO-level gamma_ao instead of atom-level gamma.
    fn solve_sczv_cg_xtb(
        &self,
        lagrangian: &[Array1<f64>],
        qvo_ao_vec: &[Array2<f64>],
    ) -> Vec<Array1<f64>> {
        let maxiter = 500;
        let threshold = 1.0e-8;
        let gamma_ao_super: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let n_frag = self.monomers.len();

        // Scale Lagrangian by factor 4 (SCAL=4.0 for RHF closed-shell)
        let rhs: Vec<Array1<f64>> = lagrangian.iter().map(|l| 4.0 * l).collect();

        // Segment sizes
        let seg_sizes: Vec<usize> = self
            .monomers
            .iter()
            .map(|m| {
                let nocc = m.properties.occ_indices().unwrap().len();
                let nvirt = m.properties.virt_indices().unwrap().len();
                nvirt * nocc
            })
            .collect();

        // Precompute per-fragment gamma_ao views
        let gammas_ao: Vec<ArrayView2<f64>> = self
            .monomers
            .iter()
            .map(|m| m.properties.gamma_ao().unwrap())
            .collect();

        // Precompute atom-level transition charges and third-order Coulomb factors
        // Q_atom[A, ai] = Σ_{μ∈A} Q_ao[μ, ai]
        // third_factor[A] = 2 * Gamma_A * dq_A
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

        // Full matrix-vector product: computes A·z for the coupled system
        let matvec = |z_vecs: &Vec<Array1<f64>>| -> Vec<Array1<f64>> {
            // Step 1: Compute QINDZ_K = Q_ao_K @ Z_K for all fragments
            let qindz: Vec<Array1<f64>> =
                (0..n_frag).map(|k| qvo_ao_vec[k].dot(&z_vecs[k])).collect();

            // Step 2: For each fragment I, compute A_I·z_I + inter-fragment coupling
            (0..n_frag)
                .into_par_iter()
                .map(|idx_i| {
                    let m_i = &self.monomers[idx_i];

                    // Intra-fragment: A_I · z_I (diagonal + Coulomb)
                    let mut result_i = Self::orbital_hessian_matvec_xtb(
                        m_i,
                        &qvo_ao_vec[idx_i],
                        gammas_ao[idx_i],
                        &z_vecs[idx_i],
                    );

                    // Third-order Coulomb coupling:
                    // A_3rd · v = -Σ_A 2·Γ_A·dq_A · Q_atom_A^T · (Q_atom_A · v)
                    let qvo_atom = &qvo_atom_vec[idx_i];
                    let third_factor = &third_factor_vec[idx_i];
                    let qv_atom: Array1<f64> = qvo_atom.dot(&z_vecs[idx_i]); // [n_atoms]
                    let g_qv_atom: Array1<f64> = third_factor * &qv_atom; // element-wise
                    result_i -= &qvo_atom.t().dot(&g_qv_atom); // [nvo]

                    // Inter-fragment: sum_{K≠I} (-1) * Q_I^T @ gamma_ao[I,K] @ QINDZ_K
                    let mut shift_i = Array1::<f64>::zeros(m_i.n_orbs);
                    for idx_k in 0..n_frag {
                        if idx_k != idx_i {
                            let m_k = &self.monomers[idx_k];
                            let gamma_ik = gamma_ao_super.slice(s![m_i.slice.orb, m_k.slice.orb]);
                            shift_i += &gamma_ik.dot(&qindz[idx_k]);
                        }
                    }
                    result_i -= &qvo_ao_vec[idx_i].t().dot(&shift_i);

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

        // Apply preconditioner
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

        // Preconditioned CG
        let az = matvec(&z_vecs);
        let mut r = vec_sub(&rhs, &az);
        let mut z_pre = precond(&r);
        let mut p = z_pre.clone();
        let mut rz = dot_all(&r, &z_pre);

        let rhs_norm = dot_all(&rhs, &rhs).sqrt();
        let abs_threshold = threshold * rhs_norm;

        for iter in 0..maxiter {
            let residual = dot_all(&r, &r).sqrt();
            if residual < abs_threshold {
                // eprintln!(
                //     "  xTB SCZV-CG converged after {} iterations (residual = {:.2e})",
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

    /// Compute per-monomer response gradient (Z×F + Z×B + Z×G intra) using on-the-fly
    /// shell-pair loops, mirroring monomer_gradient_combined structure.
    ///
    /// Returns gradient in global coordinates [n_atoms_total, 3].
    fn response_gradient_onthefly_xtb(
        &self,
        z_vectors: &[Array1<f64>],
        qvo_ao_vec: &[Array2<f64>],
        cn_numbers_global: ArrayView1<f64>,
        cn_grad_global: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_atoms_total = self.atoms.len();
        let atoms = &self.atoms[..];

        // Parallel per-monomer gradient computation
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
                let gamma_ao: ArrayView2<f64> = m_i.properties.gamma_ao().unwrap();
                let dq_ao: ArrayView1<f64> = m_i.properties.dq_ao().unwrap();
                let esp_q: ArrayView1<f64> = m_i.properties.esp_q().unwrap();

                // === Pre-compute work matrices ===

                // Z in MO → AO basis
                let z_mat = z_i.view().into_shape([nvirt, nocc]).unwrap();
                let c_virt = orbs.slice(s![.., nocc..nocc + nvirt]);
                let c_occ = orbs.slice(s![.., ..nocc]);

                // Z_AO = C_virt * Z_MO * C_occ^T
                let z_ao: Array2<f64> = c_virt.dot(&z_mat.dot(&c_occ.t()));

                // WZ_AO: energy-weighted Z
                let mut wz_mat = Array2::<f64>::zeros([nvirt, nocc]);
                for i_occ in 0..nocc {
                    for a in 0..nvirt {
                        wz_mat[[a, i_occ]] = z_mat[[a, i_occ]] * orbe[i_occ];
                    }
                }
                let wz_ao: Array2<f64> = c_virt.dot(&wz_mat.dot(&c_occ.t()));

                // AO-level shift: gamma_ao @ dq_ao + esp_q - third_order_shift
                let mut shift_vec: Array1<f64> = gamma_ao.dot(&dq_ao) + &esp_q;

                // Subtract third-order Coulomb shift: epot[A] = Gamma_A * dq_A^2
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

                // Work matrix: wrk_response = shift_ao * z_ao - wz_ao
                let wrk_response: Array2<f64> = &shift_ao_mat * &z_ao - &wz_ao;

                // Q^Z at AO level: q_z_ao[mu] = (Z_sym · S)_{mu,mu}
                let z_sym: Array2<f64> = 0.5 * (&z_ao + &z_ao.t());
                let zs: Array2<f64> = z_sym.dot(&s);
                let mut q_z_ao = Array1::<f64>::zeros(n_orbs);
                for mu in 0..n_orbs {
                    q_z_ao[mu] = zs[[mu, mu]];
                }

                // Shell-level Q^Z and dq for gamma derivative
                let n_shells = m_i.basis.shells.len();
                let mut q_z_shell: Vec<f64> = vec![0.0; n_shells];
                let mut dq_shell: Vec<f64> = vec![0.0; n_shells];
                for (shell_idx, shell) in m_i.basis.shells.iter().enumerate() {
                    for mu in shell.sph_start..shell.sph_end {
                        q_z_shell[shell_idx] += q_z_ao[mu];
                        dq_shell[shell_idx] += dq_ao[mu];
                    }
                }

                // Use global coordination numbers for this monomer's atoms
                let atom_range = m_i.slice.atom_as_range();
                let atom_start = atom_range.start;
                let cn_numbers: Array1<f64> = cn_numbers_global.slice(s![atom_range]).to_owned();

                let mut grad_local = Array2::<f64>::zeros([n_atoms, 3]);
                let mut cn_factors: Vec<f64> = vec![0.0; n_atoms];

                // === Shell-pair loop: overlap-dependent Z×F + Z×B ===
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

                        // Same Hamiltonian parameters as ground state
                        let self_energy_term = get_self_energy_values_new(
                            atomi.number,
                            atomj.number,
                            cn_1,
                            cn_2,
                            shell_i.shell_index,
                            shell_j.shell_index,
                        );

                        let cn_coeff_i =
                            get_self_energy_cn_grad_coeff(atomi.number, shell_i.shell_index);
                        let cn_coeff_j =
                            get_self_energy_cn_grad_coeff(atomj.number, shell_j.shell_index);

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
                            get_pi_term_gradient_inline(
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

                        // Shell-level accumulators (replace P with z_ao, effective_mat with wrk_response)
                        let mut diag_sp_sum: f64 = 0.0;
                        let mut off_sp_sum: f64 = 0.0;
                        let mut shell_pi_sp_sum: f64 = 0.0;
                        let mut shell_ds_contrib: [f64; 3] = [0.0; 3];

                        // Loop over AO pairs within shell pair
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

                                            // Response: h0*z_ao + wrk_response (instead of h0*P + effective_mat)
                                            let w_ij = wrk_response[[idx_i, idx_j]];
                                            let combined = h0_val * z_ij + w_ij;

                                            let ds_all =
                                                obara_saika_derivatives_all(orbital1, orbital2);

                                            for dir in 0..3 {
                                                shell_ds_contrib[dir] +=
                                                    ds_all[dir] * norm_prod * combined;
                                            }

                                            // Symmetrize z_ao for pi gradient: z_ao is NOT symmetric
                                            // for response (z_ao = C_virt * Z_mo * C_occ^T), but
                                            // Tr(Z * dH/dR) = Tr(Z_sym * dH/dR) since H is symmetric.
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

                                    // For d-shells with shell_i_idx < shell_j_idx, we process each
                                    // pair once. Factor 2 accounts for both orderings, but z_ao and
                                    // wrk_response are NOT symmetric for response gradient. Use
                                    // symmetrized values: (val[i,j] + val[j,i]) instead of 2*val[i,j].
                                    for dir in 0..3 {
                                        let ds_val_i = ds_d[[dir, sph_i, sph_j]];
                                        let ds_val_j = ds_d[[3 + dir, sph_i, sph_j]];

                                        let combined = h0_val * (z_ij + z_ji) + (w_ij + w_ji);
                                        grad_local[[at_i, dir]] += ds_val_i * combined;
                                        grad_local[[at_j, dir]] += ds_val_j * combined;
                                    }

                                    // Symmetrize z_ao for pi gradient (same reason as s-p case)
                                    shell_pi_sp_sum += s[[idx_i, idx_j]] * 0.5 * (z_ij + z_ji);
                                }
                            }
                        }

                        // Apply shell-level contributions
                        if at_i != at_j {
                            for dir in 0..3 {
                                grad_local[[at_i, dir]] += shell_ds_contrib[dir];
                                grad_local[[at_j, dir]] -= shell_ds_contrib[dir];
                            }

                            // Pi gradient
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

                        // CN factors (same deferred pattern as ground state but with z_ao)
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
                for at in 0..n_atoms {
                    if cn_factors[at].abs() > 1e-15 {
                        let global_at = atom_start + at;
                        let cn_grad_at: ArrayView1<f64> = cn_grad_global.slice(s![.., global_at]);
                        for k in 0..(3 * n_atoms_total) {
                            cn_grad_contribution[k] += cn_factors[at] * cn_grad_at[k];
                        }
                    }
                }

                // === Gamma derivative loop: intra-fragment Z×G ===
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

        // Reduce per-monomer local gradients into global gradient
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

        // Add CN gradient contribution to the 2D gradient
        for at in 0..n_atoms_total {
            for d in 0..3 {
                gradient[[at, d]] += cn_grad_total[3 * at + d];
            }
        }

        gradient
    }

    /// Inter-fragment response gradient: Z×G between monomers.
    ///
    /// For each monomer pair (I, J):
    ///   G^a += Σ_{shell_s∈I, shell_t∈J} q_z_shell_I[s] * dq_shell_J[t] * dgamma/dR
    fn inter_fragment_response_gradient_xtb(&self, z_vectors: &[Array1<f64>]) -> Array2<f64> {
        let n_atoms_total = self.atoms.len();
        let atoms = &self.atoms[..];
        let mut gradient = Array2::<f64>::zeros([n_atoms_total, 3]);

        // Compute Q^Z (Mulliken charges from Z-vector) at shell level for each monomer
        let mut q_z_shell_all: Vec<Vec<f64>> = Vec::with_capacity(self.monomers.len());
        let mut dq_shell_all: Vec<Vec<f64>> = Vec::with_capacity(self.monomers.len());

        for (idx_i, m_i) in self.monomers.iter().enumerate() {
            let z_i = &z_vectors[idx_i];
            let nocc: usize = m_i.properties.occ_indices().unwrap().len();
            let nvirt: usize = m_i.properties.virt_indices().unwrap().len();
            let orbs: ArrayView2<f64> = m_i.properties.orbs().unwrap();
            let s: ArrayView2<f64> = m_i.properties.s().unwrap();
            let dq_ao: ArrayView1<f64> = m_i.properties.dq_ao().unwrap();

            // Z_AO symmetrized → Mulliken
            let z_mat = z_i.view().into_shape([nvirt, nocc]).unwrap();
            let c_virt = orbs.slice(s![.., nocc..nocc + nvirt]);
            let c_occ = orbs.slice(s![.., ..nocc]);
            let z_ao_unsym: Array2<f64> = c_virt.dot(&z_mat.dot(&c_occ.t()));
            let z_ao_sym: Array2<f64> = 0.5 * (&z_ao_unsym + &z_ao_unsym.t());
            let zs: Array2<f64> = z_ao_sym.dot(&s);

            // Sum to shell level
            let n_shells = m_i.basis.shells.len();
            let mut q_z_shell = vec![0.0; n_shells];
            let mut dq_shell = vec![0.0; n_shells];
            for (s_idx, shell) in m_i.basis.shells.iter().enumerate() {
                for mu in shell.sph_start..shell.sph_end {
                    q_z_shell[s_idx] += zs[[mu, mu]];
                    dq_shell[s_idx] += dq_ao[mu];
                }
            }

            q_z_shell_all.push(q_z_shell);
            dq_shell_all.push(dq_shell);
        }

        // Inter-fragment gamma derivative contribution
        for (idx_i, m_i) in self.monomers.iter().enumerate() {
            let m_i_atom_start = m_i.slice.atom_as_range().start;

            for (idx_j, m_j) in self.monomers.iter().enumerate() {
                if idx_i == idx_j {
                    continue;
                }

                let m_j_atom_start = m_j.slice.atom_as_range().start;

                // For each shell pair (s in I, t in J)
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

                        let factor =
                            q_z_shell_all[idx_i][s_idx] * dq_shell_all[idx_j][t_idx] * dgamma_dr
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
        }

        gradient
    }

    /// Compute complete FMO2-xTB response gradient.
    ///
    /// Steps:
    /// 1. Build AO-level Lagrangian and qvo_ao for each monomer
    /// 2. Solve Z-vectors via matrix-free CG
    /// 3. Per-monomer response gradient (Z×F + Z×B + Z×G intra)
    /// 4. Inter-fragment Z×G
    /// 5. Apply -1.0 sign convention fix
    pub fn response_gradient_fmo_xtb(&mut self) -> Array1<f64> {
        let n_atoms_total = self.atoms.len();
        let n_grad = 3 * n_atoms_total;
        let atoms = &self.atoms[..];

        // Use global CN for response gradient
        let cn_numbers_global: ArrayView1<f64> = self.properties.cn().unwrap();
        let cn_grad_global: Array2<f64> =
            crate::gradients::hamiltonian::calculate_coordination_number_gradients(atoms);

        // let t0 = Instant::now();
        let (lagrangian_vec, qvo_ao_vec) = self.calculate_response_lagrangian_xtb();
        // let t_lag = t0.elapsed();

        // If all Lagrangians are zero (e.g., 2 monomers with 1 pair), skip response
        let all_zero = lagrangian_vec
            .iter()
            .all(|l| l.mapv(|x| x.abs()).sum() < 1e-30);
        if all_zero {
            return Array1::zeros(n_grad);
        }

        // let t0 = Instant::now();
        let z_vectors = self.solve_sczv_cg_xtb(&lagrangian_vec, &qvo_ao_vec);
        // let t_cg = t0.elapsed();

        // let t0 = Instant::now();
        let gradient_2d = self.response_gradient_onthefly_xtb(
            &z_vectors,
            &qvo_ao_vec,
            cn_numbers_global,
            cn_grad_global.view(),
        );
        // let t_grad_intra = t0.elapsed();

        // let t0 = Instant::now();
        let inter_grad = self.inter_fragment_response_gradient_xtb(&z_vectors);
        // let t_grad_inter = t0.elapsed();
        // eprintln!("  xTB Response gradient:");
        // eprintln!("    Lagrangian:   {:.3} s", t_lag.as_secs_f64());
        // eprintln!("    SCZV-CG:      {:.3} s", t_cg.as_secs_f64());
        // eprintln!("    Grad(intra):  {:.3} s", t_grad_intra.as_secs_f64());
        // eprintln!("    Grad(inter):  {:.3} s", t_grad_inter.as_secs_f64());

        let total_2d = gradient_2d + inter_grad;

        // Apply -1.0 sign convention fix
        let total_2d = -1.0 * total_2d;

        total_2d
            .into_shape([n_grad])
            .expect("Failed to reshape xTB response gradient")
    }
}
