use dialect_config::Configuration;
use dialect_utilities::mulliken::shell_to_ao_values;
use crate::fmo::gradients::monomer::grad_repulsive_energy_xtb;
use crate::fmo::monomer::XtbMonomer;
use crate::fmo::pair::XtbPair;
use crate::gradients::halogen_bonding::gradient_halogen_bonding_xtb;
use crate::gradients::hamiltonian::{
    calculate_coordination_number_gradients, calculate_h0_gradient_xtb1_new,
};
use crate::initialization::atom::XtbAtom;
use crate::integrals::calc_overlap_derivative_d_shells;
use crate::integrals::obara_saika_derivatives_all;
use crate::integrals::{
    calc_overlap_matrix_obs_derivs_new, calc_overlap_matrix_obs_derivs_parallel,
};
use crate::scc::gamma_matrix::gamma_gradient_xtb_new;
use dialect_base::constants::BOHR_TO_ANGS;
use dialect_base::defaults::PROXIMITY_CUTOFF;
use crate::{
        gradients::helpers::{coul_third_order_grad_contribution_xtb, gradient_disp3_xtb},
        initialization::system::XtbSystem,
        parameters::*,
        scc::hamiltonian::{
            calculate_coordination_numbers, calculate_pair_scaling_param,
            get_hueckel_constants_new, get_pi_term, get_self_energy_values_new,
        },
};
use nalgebra::Vector3;
use ndarray::prelude::*;
use ndarray_npy::write_npy;
use rayon::prelude::*;

/// Expand a per-AO potential vector into the symmetric AO matrix of pairwise
/// sums `M[i,j] = v[i] + v[j]` (the shift that multiplies the overlap).
pub fn aovec_to_aomat(esp_aowise: ArrayView1<f64>, n_orbs: usize) -> Array2<f64> {
    let esp_ao_column: Array2<f64> = esp_aowise.clone().to_owned().insert_axis(Axis(1));
    let esp_ao: Array2<f64> = &esp_ao_column.broadcast((n_orbs, n_orbs)).unwrap() + &esp_aowise;
    esp_ao
}

/// Helper function to get self-energy CN gradient coefficient
fn get_self_energy_cn_grad_coeff(z: u8, shell_idx: usize) -> f64 {
    let z_idx: usize = (z - 1) as usize;
    -HAMILTONIAN_KCN_VALUES[z_idx][shell_idx]
}

/// Get pi term gradient (moved from hamiltonian.rs for on-the-fly use)
fn get_pi_term_gradient_inline(
    r_vector: &Vector3<f64>,
    r_ab: f64,
    z_1: usize,
    z_2: usize,
    l_1: usize,
    l_2: usize,
) -> [f64; 3] {
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

impl XtbSystem {
    /// Ground-state xTB gradient: sums the H0, overlap (Pulay), gamma
    /// (shell-resolved Coulomb), third-order, repulsive, dispersion and
    /// halogen contributions.
    pub fn ground_state_gradient(&mut self) -> Array1<f64> {
        // call the low RAM version
        self.ground_state_gradient_onthefly()
    }

    /// Compute the ground state gradient using the original method with full matrix storage
    pub fn ground_state_gradient_old(&mut self) -> Array1<f64> {
        // take references/views to the necessary properties from the scc calculation
        let gamma: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        let dq_ao: ArrayView1<f64> = self.properties.dq_ao().unwrap();
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // calculate the gradient of the overlap matrix
        // Use parallel version for better performance on larger systems
        let grad_s: Array3<f64> = if self.n_atoms > 10 {
            calc_overlap_matrix_obs_derivs_parallel(&self.basis, self.n_atoms)
        } else {
            calc_overlap_matrix_obs_derivs_new(&self.basis, self.n_atoms)
        };
        // calculate the gradient of the H0 matrix
        // Note: Parallel version has high allocation overhead, only use for very large systems
        let grad_h0: Array3<f64> =
            calculate_h0_gradient_xtb1_new(self.n_orbs, &self.atoms, s, grad_s.view(), &self.basis);
        // and reshape them into a 2D array. the last two dimension (number of orbitals) are compressed
        // into one dimension to be able to just matrix-matrix products for the computation of the gradient
        let grad_s_2d: ArrayView2<f64> = grad_s
            .view()
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap();
        let grad_h0_2d: ArrayView2<f64> = grad_h0
            .view()
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap();

        // calculate the gradient of the gamma matrix
        let grad_gamma: Array3<f64> = gamma_gradient_xtb_new(
            &self.gammafunction,
            &self.atoms,
            &self.basis,
            self.n_atoms,
            self.n_orbs,
        );
        let grad_gamma_2d: ArrayView2<f64> = grad_gamma
            .view()
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap();

        // create array of hubbard derivatives
        let mut hubbard_derivatives: Array1<f64> = Array1::zeros(self.n_atoms);
        for (mut val, atom) in hubbard_derivatives.iter_mut().zip(self.atoms.iter()) {
            *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
        }

        // The density matrix in vector form.
        let p_flat: ArrayView1<f64> = p.into_shape([self.n_orbs * self.n_orbs]).unwrap();

        // compute the energy weighted density matrix
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let occupations: Array1<f64> = Array::from(self.properties.occupation().unwrap().to_vec());
        let weighted_orbe = &orbe * &occupations;
        let worbe_2d: Array2<f64> = Array2::from_diag(&weighted_orbe);
        let w_new: Array2<f64> = orbs.dot(&worbe_2d.dot(&orbs.t()));
        let w: Array1<f64> = w_new.into_shape([self.n_orbs * self.n_orbs]).unwrap();

        // calculate the gradient contribution of the third order energy
        // contribution of dq**2 and gamma third order
        let dq2_gamma: Array2<f64> =
            coul_third_order_grad_contribution_xtb(&self.basis, dq, hubbard_derivatives.view());
        // multiply with the density matrix
        let coulomb_p_third_order: Array1<f64> = 0.5
            * (&p * &dq2_gamma)
                .into_shape([self.n_orbs * self.n_orbs])
                .unwrap();

        let dq_column: ArrayView2<f64> = dq_ao.clone().insert_axis(Axis(1));
        let dq_x_dq: Array1<f64> = (&dq_column.broadcast((self.n_orbs, self.n_orbs)).unwrap()
            * &dq_ao)
            .into_shape([self.n_orbs * self.n_orbs])
            .unwrap();
        let coulomb_mat: Array2<f64> = aovec_to_aomat(gamma.dot(&dq_ao).view(), self.n_orbs) * 0.5;
        let coulomb_x_p: Array1<f64> = (&p * &coulomb_mat)
            .into_shape([self.n_orbs * self.n_orbs])
            .unwrap();

        // Separate the four overlap derivative contributions:
        // 1. dH0·P term (includes h0_val * P * dS contribution)
        let grad_h0_p: Array1<f64> = grad_h0_2d.dot(&p_flat);

        // 2. -dS·W term (energy-weighted density matrix)
        let grad_ds_w: Array1<f64> = -grad_s_2d.dot(&w);

        // 3. dS·(coulomb*P) term
        let grad_ds_coulomb: Array1<f64> = grad_s_2d.dot(&coulomb_x_p);

        // 4. -dS·(third_order*P) term
        let grad_ds_third: Array1<f64> = -grad_s_2d.dot(&coulomb_p_third_order);

        // Build gradient from separated contributions
        let mut gradient: Array1<f64> = &grad_h0_p + &grad_ds_w + &grad_ds_coulomb + &grad_ds_third;

        // 6th part: second order Coulomb gradient part 2 (gamma derivative)
        gradient += &(0.5 * grad_gamma_2d.dot(&dq_x_dq));

        // last part: dV_rep / dR
        let grad_rep_contrib = self.grad_repulsive_energy();
        gradient = gradient + &grad_rep_contrib;

        // dispersion
        let grad_disp_contrib = gradient_disp3_xtb(&self.atoms, &self.config);
        gradient = gradient + &grad_disp_contrib;

        // halogen bonding
        let grad_halogen_contrib = gradient_halogen_bonding_xtb(&self.atoms);
        gradient = gradient + &grad_halogen_contrib;

        if self.config.jobtype == "grad" {
            // save the gradient
            write_npy("gs_gradient.npy", &gradient).unwrap();
        }

        gradient
    }

    /// On-the-fly gradient calculation - avoids storing large 3D arrays
    /// Computes contributions directly to gradient vector for each orbital pair
    /// Uses parallel computation over shell pairs
    pub fn ground_state_gradient_onthefly(&mut self) -> Array1<f64> {
        // Get references to SCC results
        let gamma: ArrayView2<f64> = self.properties.gamma_shell().unwrap();
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        // let dq_ao: ArrayView1<f64> = self.properties.dq_ao().unwrap();
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();
        let dq_shell: ArrayView1<f64> = self.properties.dq_shell().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // Compute energy-weighted density matrix W
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let occupations: Array1<f64> = Array::from(self.properties.occupation().unwrap().to_vec());
        let weighted_orbe = &orbe * &occupations;
        let worbe_2d: Array2<f64> = Array2::from_diag(&weighted_orbe);
        let w: Array2<f64> = orbs.dot(&worbe_2d.dot(&orbs.t()));

        // Compute Coulomb potential terms
        let gamma_dq: Array1<f64> = gamma.dot(&dq_shell);
        let total_shift: Array1<f64> =
            shell_to_ao_values(&self.basis, self.n_orbs, gamma_dq.view());
        let coulomb_mat: Array2<f64> = aovec_to_aomat(total_shift.view(), self.n_orbs) * 0.5;

        // Third order terms
        let mut hubbard_derivatives: Array1<f64> = Array1::zeros(self.n_atoms);
        for (val, atom) in hubbard_derivatives.iter_mut().zip(self.atoms.iter()) {
            *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
        }
        let dq2_gamma: Array2<f64> =
            coul_third_order_grad_contribution_xtb(&self.basis, dq, hubbard_derivatives.view());

        // Effective matrix for dS·effective term: -W + coulomb*P - 0.5*third_order*P
        let effective_mat: Array2<f64> = &(-&w) + &(&coulomb_mat * &p) - &(0.5 * &dq2_gamma * &p);

        // Precompute coordination numbers and gradients
        let cn_numbers: Array1<f64> = calculate_coordination_numbers(&self.atoms);
        let cn_number_grads: Array2<f64> = calculate_coordination_number_gradients(&self.atoms);

        // Precompute shell-level dq sums for efficient gamma gradient
        // let n_shells = self.basis.shells.len();
        // let mut dq_shell: Vec<f64> = vec![0.0; n_shells];
        // for (shell_idx, shell) in self.basis.shells.iter().enumerate() {
        //     for idx in shell.sph_start..shell.sph_end {
        //         dq_shell[shell_idx] += dq_ao[idx];
        //     }
        // }

        // Parallel gradient computation - each thread gets its own gradient and cn_factors
        let n_atoms = self.n_atoms;
        let n_grad = 3 * n_atoms;
        let basis = &self.basis;
        let shells = &self.basis.shells;
        let basis_functions = &self.basis.basis_functions;
        let atoms = &self.atoms;
        let gammafunction = &self.gammafunction;

        // Parallel over shell_i with thread-local accumulators
        let (gradient, cn_factors): (Array1<f64>, Vec<f64>) = shells
            .par_iter()
            .enumerate()
            .map(|(shell_i_idx, shell_i)| {
                let mut local_gradient: Array1<f64> = Array1::zeros(n_grad);
                let mut local_cn_factors: Vec<f64> = vec![0.0; n_atoms];

                let atomi = &atoms[shell_i.atom_index];
                let at_i = shell_i.atom_index;
                let cn_1 = cn_numbers[at_i];

                for (shell_j_idx, shell_j) in shells.iter().enumerate() {
                    let atomj = &atoms[shell_j.atom_index];
                    let at_j = shell_j.atom_index;
                    let cn_2 = cn_numbers[at_j];

                    let r_vector: Vector3<f64> = atomi - atomj;
                    let distance: f64 = r_vector.norm();

                    if distance >= PROXIMITY_CUTOFF {
                        continue;
                    }

                    // Precompute self-energy term (used for both diagonal and off-diagonal)
                    let self_energy_term = get_self_energy_values_new(
                        atomi.number,
                        atomj.number,
                        cn_1,
                        cn_2,
                        shell_i.shell_index,
                        shell_j.shell_index,
                    );

                    // CN gradient coefficient for shell_i
                    let cn_coeff_i =
                        get_self_energy_cn_grad_coeff(atomi.number, shell_i.shell_index);
                    let cn_coeff_j =
                        get_self_energy_cn_grad_coeff(atomj.number, shell_j.shell_index);

                    // For off-diagonal orbital pairs, precompute H0 parameters
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

                    // Pi gradient (only for different atoms)
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

                    // Compute shell-level sums for CN gradient optimization
                    // This changes CN gradient from O(n_orbs^2 * n_atoms) to O(n_shells^2 * n_atoms)
                    let mut diag_sp_sum: f64 = 0.0; // sum of s[i,i] * p[i,i] for diagonal
                    let mut off_sp_sum: f64 = 0.0; // sum of s[i,j] * p[i,j] for off-diagonal

                    // Shell-level accumulator for overlap derivative contributions
                    // Accumulate dS * (h0_val * P + Eff) at shell level, apply to gradient once
                    let mut shell_ds_contrib: [f64; 3] = [0.0; 3];
                    let mut shell_pi_sp_sum: f64 = 0.0; // sum of s[i,j] * p[i,j] for pi gradient

                    // Loop over AO pairs within shell pair
                    for idx_i in shell_i.sph_start..shell_i.sph_end {
                        let idx_i_local = idx_i - shell_i.sph_start;

                        for idx_j in shell_j.sph_start..shell_j.sph_end {
                            let idx_j_local = idx_j - shell_j.sph_start;

                            let p_ij = p[[idx_i, idx_j]];
                            let s_ij = s[[idx_i, idx_j]];

                            if idx_i == idx_j {
                                // DIAGONAL: accumulate for shell-level CN gradient
                                diag_sp_sum += s_ij * p_ij;
                            } else {
                                // OFF-DIAGONAL: accumulate for shell-level CN gradient
                                off_sp_sum += s_ij * p_ij;

                                if at_i != at_j {
                                    // Different atoms: overlap derivative is non-zero
                                    if shell_i.angular_momentum < 2 && shell_j.angular_momentum < 2
                                    {
                                        let orbital1 =
                                            &basis_functions[shell_i.start + idx_i_local];
                                        let orbital2 =
                                            &basis_functions[shell_j.start + idx_j_local];
                                        let norm_prod =
                                            orbital1.contracted_norm * orbital2.contracted_norm;

                                        // Combined factor: h0_val * P + Eff (shell-level H0 factor times P, plus effective)
                                        let eff_ij = effective_mat[[idx_i, idx_j]];
                                        let combined_factor = h0_val * p_ij + eff_ij;

                                        // Compute all 3 overlap derivatives at once (optimized)
                                        let ds_all =
                                            obara_saika_derivatives_all(orbital1, orbital2);

                                        // Accumulate dS * combined_factor at shell level
                                        for dir in 0..3 {
                                            shell_ds_contrib[dir] +=
                                                ds_all[dir] * norm_prod * combined_factor;
                                        }

                                        // Accumulate S*P sum for pi gradient (factor applied at shell level)
                                        shell_pi_sp_sum += s_ij * p_ij;
                                    }
                                }
                            }
                        }
                    }

                    // D-orbital handling: compute overlap derivatives using calc_overlap_derivative_d_shells
                    // Process unique shell pairs (shell_i_idx < shell_j_idx) with factor 2 for symmetric storage.
                    // ds_d[0-2] gives derivative w.r.t. at_i, ds_d[3-5] gives derivative w.r.t. at_j.
                    // Both atoms receive their contributions with factor 2 to account for symmetric matrix storage.
                    let shell_i_has_d = shell_i.angular_momentum >= 2;
                    let shell_j_has_d = shell_j.angular_momentum >= 2;
                    let either_has_d = shell_i_has_d || shell_j_has_d;

                    if at_i != at_j && either_has_d && shell_i_idx < shell_j_idx {
                        let ds_d = calc_overlap_derivative_d_shells(&basis, shell_i, shell_j);
                        let sph_dim_i = shell_i.sph_end - shell_i.sph_start;
                        let sph_dim_j = shell_j.sph_end - shell_j.sph_start;

                        for sph_i in 0..sph_dim_i {
                            let idx_i = shell_i.sph_start + sph_i;
                            for sph_j in 0..sph_dim_j {
                                let idx_j = shell_j.sph_start + sph_j;

                                let p_ij = p[[idx_i, idx_j]];
                                let s_ij = s[[idx_i, idx_j]];
                                let eff_ij = effective_mat[[idx_i, idx_j]];

                                for dir in 0..3 {
                                    // Factor 2 accounts for symmetric matrix storage (both (i,j) and (j,i))
                                    // ds_d[0-2] is derivative w.r.t. at_i
                                    let ds_val_i = 2.0 * ds_d[[dir, sph_i, sph_j]];
                                    // ds_d[3-5] is derivative w.r.t. at_j
                                    let ds_val_j = 2.0 * ds_d[[3 + dir, sph_i, sph_j]];

                                    let combined_i = ds_val_i * (h0_val * p_ij + eff_ij);
                                    let combined_j = ds_val_j * (h0_val * p_ij + eff_ij);
                                    local_gradient[3 * at_i + dir] += combined_i;
                                    local_gradient[3 * at_j + dir] += combined_j;
                                }

                                // Accumulate S*P for pi gradient (d-orbital pairs)
                                shell_pi_sp_sum += s_ij * p_ij;
                            }
                        }
                    }

                    // Apply shell-level dS contribution to gradient (once per shell pair, not per AO pair)
                    if at_i != at_j {
                        for dir in 0..3 {
                            local_gradient[3 * at_i + dir] += shell_ds_contrib[dir];
                            local_gradient[3 * at_j + dir] -= shell_ds_contrib[dir];
                        }

                        // Apply shell-level pi gradient (factor of 2 for symmetric storage)
                        let pi_contrib = 2.0 * pi_factor * shell_pi_sp_sum;
                        for dir in 0..3 {
                            local_gradient[3 * at_i + dir] += pi_grad[dir] * pi_contrib;
                        }
                        // For d-orbital pairs processed via unique-pair approach (shell_i_idx < shell_j_idx),
                        // the swapped pair (j,i) is never processed, so at_j misses its pi contribution.
                        // Add it here with opposite sign (Newton's 3rd law).
                        if either_has_d && shell_i_idx < shell_j_idx {
                            for dir in 0..3 {
                                local_gradient[3 * at_j + dir] -= pi_grad[dir] * pi_contrib;
                            }
                        }
                    }

                    // Deferred CN gradient: accumulate per-atom factors
                    // Diagonal: gradient += cn_coeff_i * diag_sp_sum * cn_grad_i
                    if diag_sp_sum.abs() > 1e-15 {
                        local_cn_factors[at_i] += cn_coeff_i * diag_sp_sum;
                    }

                    // Off-diagonal: gradient += off_factor * (cn_coeff_i * cn_grad_i + cn_coeff_j * cn_grad_j)
                    if off_sp_sum.abs() > 1e-15 {
                        let off_factor = 0.5 * h_val_cn * off_sp_sum;
                        local_cn_factors[at_i] += off_factor * cn_coeff_i;
                        local_cn_factors[at_j] += off_factor * cn_coeff_j;
                    }
                }

                (local_gradient, local_cn_factors)
            })
            .reduce(
                || (Array1::zeros(n_grad), vec![0.0; n_atoms]),
                |(mut g1, mut cf1), (g2, cf2)| {
                    g1 += &g2;
                    for i in 0..n_atoms {
                        cf1[i] += cf2[i];
                    }
                    (g1, cf1)
                },
            );

        // Apply deferred CN gradient contributions: O(n_atoms^2) instead of O(n_shells^2 * n_atoms)
        let mut gradient = gradient; // make mutable
        for at in 0..self.n_atoms {
            if cn_factors[at].abs() > 1e-15 {
                let cn_grad_at: ArrayView1<f64> = cn_number_grads.slice(s![.., at]);
                for k in 0..(3 * self.n_atoms) {
                    gradient[k] += cn_factors[at] * cn_grad_at[k];
                }
            }
        }

        // Gamma gradient - separate loop without PROXIMITY_CUTOFF (parallel)
        // Gamma function decays as 1/r, so long-range contributions are important
        let gamma_grad: Array1<f64> = shells
            .par_iter()
            .enumerate()
            .map(|(shell_i_idx, shell_i)| {
                let mut local_gamma_grad: Array1<f64> = Array1::zeros(n_grad);
                let atomi = &atoms[shell_i.atom_index];
                let at_i = shell_i.atom_index;

                for (shell_j_idx, shell_j) in shells.iter().enumerate() {
                    let atomj = &atoms[shell_j.atom_index];
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
                            local_gamma_grad[3 * at_i + dir] += e_ij[dir] * shell_gamma_contrib;
                            local_gamma_grad[3 * at_j + dir] -= e_ij[dir] * shell_gamma_contrib;
                        }
                    }
                }
                local_gamma_grad
            })
            .reduce(
                || Array1::zeros(n_grad),
                |mut a, b| {
                    a += &b;
                    a
                },
            );
        gradient += &gamma_grad;

        // Repulsive energy gradient
        let grad_rep_contrib = self.grad_repulsive_energy();
        gradient += &grad_rep_contrib;

        // Dispersion gradient
        let grad_disp_contrib = gradient_disp3_xtb(&self.atoms, &self.config);
        gradient += &grad_disp_contrib;

        // halogen bonding
        let grad_halogen_contrib = gradient_halogen_bonding_xtb(&self.atoms);
        gradient = gradient + &grad_halogen_contrib;

        gradient
    }

    /// Serial version of on-the-fly gradient with all optimizations
    /// Use this for fair comparison with serial XTB or when parallelization overhead isn't worth it
    pub fn ground_state_gradient_onthefly_serial(&mut self) -> Array1<f64> {
        // Get references to SCC results
        let gamma: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        let dq_ao: ArrayView1<f64> = self.properties.dq_ao().unwrap();
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // Compute energy-weighted density matrix W
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let occupations: Array1<f64> = Array::from(self.properties.occupation().unwrap().to_vec());
        let weighted_orbe = &orbe * &occupations;
        let worbe_2d: Array2<f64> = Array2::from_diag(&weighted_orbe);
        let w: Array2<f64> = orbs.dot(&worbe_2d.dot(&orbs.t()));

        // Compute Coulomb potential terms
        let gamma_dq: Array1<f64> = gamma.dot(&dq_ao);
        let coulomb_mat: Array2<f64> = aovec_to_aomat(gamma_dq.view(), self.n_orbs) * 0.5;

        // Third order terms
        let mut hubbard_derivatives: Array1<f64> = Array1::zeros(self.n_atoms);
        for (val, atom) in hubbard_derivatives.iter_mut().zip(self.atoms.iter()) {
            *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
        }
        let dq2_gamma: Array2<f64> =
            coul_third_order_grad_contribution_xtb(&self.basis, dq, hubbard_derivatives.view());

        // Precompute effective matrix: -W + coulomb*P - third*P
        // This combines all overlap-dependent terms into a single matrix lookup
        let effective_mat: Array2<f64> = &(-&w) + &(&coulomb_mat * &p) - &(0.5 * &dq2_gamma * &p);

        // Precompute coordination numbers and gradients
        let cn_numbers: Array1<f64> = calculate_coordination_numbers(&self.atoms);
        let cn_number_grads: Array2<f64> = calculate_coordination_number_gradients(&self.atoms);

        // Precompute shell-level dq sums for efficient gamma gradient
        let n_shells = self.basis.shells.len();
        let mut dq_shell: Vec<f64> = vec![0.0; n_shells];
        for (shell_idx, shell) in self.basis.shells.iter().enumerate() {
            for idx in shell.sph_start..shell.sph_end {
                dq_shell[shell_idx] += dq_ao[idx];
            }
        }

        // Initialize gradient and CN factors
        let mut gradient: Array1<f64> = Array1::zeros(3 * self.n_atoms);
        let mut cn_factors: Vec<f64> = vec![0.0; self.n_atoms];

        // Serial loop over shell pairs
        for (shell_i_idx, shell_i) in self.basis.shells.iter().enumerate() {
            let atomi = &self.atoms[shell_i.atom_index];
            let at_i = shell_i.atom_index;
            let cn_1 = cn_numbers[at_i];

            for (shell_j_idx, shell_j) in self.basis.shells.iter().enumerate() {
                let atomj = &self.atoms[shell_j.atom_index];
                let at_j = shell_j.atom_index;
                let cn_2 = cn_numbers[at_j];

                let r_vector: Vector3<f64> = atomi - atomj;
                let distance: f64 = r_vector.norm();

                if distance >= PROXIMITY_CUTOFF {
                    continue;
                }

                // Precompute self-energy term (used for both diagonal and off-diagonal)
                let self_energy_term = get_self_energy_values_new(
                    atomi.number,
                    atomj.number,
                    cn_1,
                    cn_2,
                    shell_i.shell_index,
                    shell_j.shell_index,
                );

                // CN gradient coefficient for shell_i
                let cn_coeff_i = get_self_energy_cn_grad_coeff(atomi.number, shell_i.shell_index);
                let cn_coeff_j = get_self_energy_cn_grad_coeff(atomj.number, shell_j.shell_index);

                // For off-diagonal orbital pairs, precompute H0 parameters
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

                let h0_val =
                    scaling_constant * hueckel_const * self_energy_term * en_term * pi_term;
                let h_val_cn = scaling_constant * hueckel_const * en_term * pi_term;

                // Pi gradient (only for different atoms)
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

                // Unit vector for gamma derivative (only for different atoms)
                let e_ij: [f64; 3] = if at_i != at_j {
                    let inv_dist = 1.0 / distance;
                    [
                        r_vector.x * inv_dist,
                        r_vector.y * inv_dist,
                        r_vector.z * inv_dist,
                    ]
                } else {
                    [0.0, 0.0, 0.0]
                };

                // Compute shell-level sums
                let mut diag_sp_sum: f64 = 0.0;
                let mut off_sp_sum: f64 = 0.0;
                let mut shell_pi_sp_sum: f64 = 0.0;

                // Shell-level accumulator for overlap derivative contributions
                // Accumulates dS * (h0_val * P + Eff) at shell level, apply to gradient once
                let mut shell_ds_contrib: [f64; 3] = [0.0; 3];

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
                                        &self.basis.basis_functions[shell_i.start + idx_i_local];
                                    let orbital2 =
                                        &self.basis.basis_functions[shell_j.start + idx_j_local];
                                    let norm_prod =
                                        orbital1.contracted_norm * orbital2.contracted_norm;

                                    // Combined factor: h0_val * P + Eff (effective combines -W + coulomb*P - third*P)
                                    let eff_ij = effective_mat[[idx_i, idx_j]];
                                    let combined_factor = h0_val * p_ij + eff_ij;

                                    // Compute all 3 overlap derivatives at once
                                    let ds_all = obara_saika_derivatives_all(orbital1, orbital2);

                                    // Accumulate dS * combined_factor at shell level
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

                // D-orbital handling: compute overlap derivatives using calc_overlap_derivative_d_shells
                // Process unique shell pairs (shell_i_idx < shell_j_idx) with factor 2 for symmetric storage.
                // ds_d[0-2] gives derivative w.r.t. at_i, ds_d[3-5] gives derivative w.r.t. at_j.
                // Both atoms receive their contributions with factor 2 to account for symmetric matrix storage.
                let shell_i_has_d = shell_i.angular_momentum >= 2;
                let shell_j_has_d = shell_j.angular_momentum >= 2;
                let either_has_d = shell_i_has_d || shell_j_has_d;

                // D-orbital handling: process unique pairs (shell_i_idx < shell_j_idx) with factor 2
                if at_i != at_j && either_has_d && shell_i_idx < shell_j_idx {
                    let ds_d = calc_overlap_derivative_d_shells(&self.basis, shell_i, shell_j);
                    let sph_dim_i = shell_i.sph_end - shell_i.sph_start;
                    let sph_dim_j = shell_j.sph_end - shell_j.sph_start;

                    for sph_i in 0..sph_dim_i {
                        let idx_i = shell_i.sph_start + sph_i;
                        for sph_j in 0..sph_dim_j {
                            let idx_j = shell_j.sph_start + sph_j;

                            let p_ij = p[[idx_i, idx_j]];
                            let eff_ij = effective_mat[[idx_i, idx_j]];

                            for dir in 0..3 {
                                // Factor 2 accounts for symmetric matrix storage (both (i,j) and (j,i))
                                // ds_d[0-2] is derivative w.r.t. at_i
                                let ds_val_i = 2.0 * ds_d[[dir, sph_i, sph_j]];
                                // ds_d[3-5] is derivative w.r.t. at_j
                                let ds_val_j = 2.0 * ds_d[[3 + dir, sph_i, sph_j]];

                                // Combined contribution: h0*P + Eff (effective combines -W + coulomb*P - third*P)
                                let combined_factor = h0_val * p_ij + eff_ij;
                                gradient[3 * at_i + dir] += ds_val_i * combined_factor;
                                gradient[3 * at_j + dir] += ds_val_j * combined_factor;
                            }

                            // Accumulate S*P for pi gradient (d-orbital pairs)
                            shell_pi_sp_sum += s[[idx_i, idx_j]] * p_ij;
                        }
                    }
                }

                // Apply shell-level contributions to gradient (s/p orbitals)
                if at_i != at_j {
                    for dir in 0..3 {
                        gradient[3 * at_i + dir] += shell_ds_contrib[dir];
                        gradient[3 * at_j + dir] -= shell_ds_contrib[dir];
                    }

                    // Apply shell-level pi gradient (factor of 2 for symmetric storage)
                    let pi_contrib = 2.0 * pi_factor * shell_pi_sp_sum;
                    for dir in 0..3 {
                        gradient[3 * at_i + dir] += pi_grad[dir] * pi_contrib;
                    }
                    // For d-orbital pairs processed via unique-pair approach (shell_i_idx < shell_j_idx),
                    // the swapped pair (j,i) is never processed, so at_j misses its pi contribution.
                    // Add it here with opposite sign (Newton's 3rd law).
                    if either_has_d && shell_i_idx < shell_j_idx {
                        for dir in 0..3 {
                            gradient[3 * at_j + dir] -= pi_grad[dir] * pi_contrib;
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

        // Apply deferred CN gradient contributions
        for at in 0..self.n_atoms {
            if cn_factors[at].abs() > 1e-15 {
                let cn_grad_at: ArrayView1<f64> = cn_number_grads.slice(s![.., at]);
                for k in 0..(3 * self.n_atoms) {
                    gradient[k] += cn_factors[at] * cn_grad_at[k];
                }
            }
        }

        // Gamma gradient - separate loop without PROXIMITY_CUTOFF
        // Gamma function decays as 1/r, so long-range contributions are important
        for (shell_i_idx, shell_i) in self.basis.shells.iter().enumerate() {
            let atomi = &self.atoms[shell_i.atom_index];
            let at_i = shell_i.atom_index;

            for (shell_j_idx, shell_j) in self.basis.shells.iter().enumerate() {
                let atomj = &self.atoms[shell_j.atom_index];
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

                    let gamma_deriv = self.gammafunction.deriv(
                        distance,
                        atomi.number,
                        shell_i.angular_momentum as u8,
                        atomj.number,
                        shell_j.angular_momentum as u8,
                    );

                    let shell_dq_prod = dq_shell[shell_i_idx] * dq_shell[shell_j_idx];
                    let shell_gamma_contrib = 0.5 * gamma_deriv * shell_dq_prod;
                    for dir in 0..3 {
                        gradient[3 * at_i + dir] += e_ij[dir] * shell_gamma_contrib;
                        gradient[3 * at_j + dir] -= e_ij[dir] * shell_gamma_contrib;
                    }
                }
            }
        }

        // Repulsive energy gradient
        let grad_rep_contrib = self.grad_repulsive_energy();
        gradient += &grad_rep_contrib;

        // Dispersion gradient
        let grad_disp_contrib = gradient_disp3_xtb(&self.atoms, &self.config);
        gradient += &grad_disp_contrib;

        // halogen bonding
        let grad_halogen_contrib = gradient_halogen_bonding_xtb(&self.atoms);
        gradient = gradient + &grad_halogen_contrib;

        gradient
    }
}

#[macro_export]
macro_rules! impl_ground_state_gradient_on_the_fly {
    () => {
        /// Serial version of on-the-fly gradient with all optimizations
        pub fn ground_state_gradient_onthefly_serial(
            &mut self,
            config: &Configuration,
            atoms: &[XtbAtom],
        ) -> Array1<f64> {
            // Get references to SCC results
            let gamma: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
            let p: ArrayView2<f64> = self.properties.p().unwrap();
            let dq_ao: ArrayView1<f64> = self.properties.dq_ao().unwrap();
            let dq: ArrayView1<f64> = self.properties.dq().unwrap();
            let s: ArrayView2<f64> = self.properties.s().unwrap();
            // get the vesp contribution
            //let vesp_hamiltonian = self.properties.h_coul_x().unwrap();
            //let vesp_contribution: Array2<f64> = p.view().dot(&vesp_hamiltonian.dot(&p));

            // Compute energy-weighted density matrix W
            let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
            let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
            let occupations: Array1<f64> =
                Array::from(self.properties.occupation().unwrap().to_vec());
            let weighted_orbe = &orbe * &occupations;
            let worbe_2d: Array2<f64> = Array2::from_diag(&weighted_orbe);
            let w: Array2<f64> = orbs.dot(&worbe_2d.dot(&orbs.t()));
            // substract contribution of the vesp
            //let w: Array2<f64> = w - vesp_contribution;

            // Compute Coulomb potential terms
            let gamma_dq: Array1<f64> = gamma.dot(&dq_ao);
            let coulomb_mat: Array2<f64> = aovec_to_aomat(gamma_dq.view(), self.n_orbs) * 0.5;

            // Third order terms
            let mut hubbard_derivatives: Array1<f64> = Array1::zeros(self.n_atoms);
            for (val, atom) in hubbard_derivatives.iter_mut().zip(atoms.iter()) {
                *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
            }
            let dq2_gamma: Array2<f64> =
                coul_third_order_grad_contribution_xtb(&self.basis, dq, hubbard_derivatives.view());

            // Precompute effective matrix: -W + coulomb*P - third*P
            // This combines all overlap-dependent terms into a single matrix lookup
            let effective_mat: Array2<f64> =
                &(-&w) + &(&coulomb_mat * &p) - &(0.5 * &dq2_gamma * &p);

            // Precompute coordination numbers and gradients
            let cn_numbers: Array1<f64> = calculate_coordination_numbers(&atoms);
            let cn_number_grads: Array2<f64> = calculate_coordination_number_gradients(&atoms);

            // Precompute shell-level dq sums for efficient gamma gradient
            let n_shells = self.basis.shells.len();
            let mut dq_shell: Vec<f64> = vec![0.0; n_shells];
            for (shell_idx, shell) in self.basis.shells.iter().enumerate() {
                for idx in shell.sph_start..shell.sph_end {
                    dq_shell[shell_idx] += dq_ao[idx];
                }
            }

            // Initialize gradient and CN factors
            let mut gradient: Array1<f64> = Array1::zeros(3 * self.n_atoms);
            let mut cn_factors: Vec<f64> = vec![0.0; self.n_atoms];

            // Serial loop over shell pairs
            for (shell_i_idx, shell_i) in self.basis.shells.iter().enumerate() {
                let atomi = &atoms[shell_i.atom_index];
                let at_i = shell_i.atom_index;
                let cn_1 = cn_numbers[at_i];

                for (shell_j_idx, shell_j) in self.basis.shells.iter().enumerate() {
                    let atomj = &atoms[shell_j.atom_index];
                    let at_j = shell_j.atom_index;
                    let cn_2 = cn_numbers[at_j];

                    let r_vector: Vector3<f64> = atomi - atomj;
                    let distance: f64 = r_vector.norm();

                    if distance >= PROXIMITY_CUTOFF {
                        continue;
                    }

                    // Precompute self-energy term (used for both diagonal and off-diagonal)
                    let self_energy_term = get_self_energy_values_new(
                        atomi.number,
                        atomj.number,
                        cn_1,
                        cn_2,
                        shell_i.shell_index,
                        shell_j.shell_index,
                    );

                    // CN gradient coefficient for shell_i
                    let cn_coeff_i =
                        get_self_energy_cn_grad_coeff(atomi.number, shell_i.shell_index);
                    let cn_coeff_j =
                        get_self_energy_cn_grad_coeff(atomj.number, shell_j.shell_index);

                    // For off-diagonal orbital pairs, precompute H0 parameters
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

                    // Pi gradient (only for different atoms)
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

                    // Unit vector for gamma derivative (only for different atoms)
                    let e_ij: [f64; 3] = if at_i != at_j {
                        let inv_dist = 1.0 / distance;
                        [
                            r_vector.x * inv_dist,
                            r_vector.y * inv_dist,
                            r_vector.z * inv_dist,
                        ]
                    } else {
                        [0.0, 0.0, 0.0]
                    };

                    // Gamma derivative
                    let gamma_deriv = if at_i != at_j {
                        self.gammafunction.deriv(
                            distance,
                            atomi.number,
                            shell_i.angular_momentum as u8,
                            atomj.number,
                            shell_j.angular_momentum as u8,
                        )
                    } else {
                        0.0
                    };

                    // Compute shell-level sums
                    let mut diag_sp_sum: f64 = 0.0;
                    let mut off_sp_sum: f64 = 0.0;
                    let mut shell_pi_sp_sum: f64 = 0.0;

                    // Shell-level accumulator for overlap derivative contributions
                    // Accumulates dS * (h0_val * P + Eff) at shell level, apply to gradient once
                    let mut shell_ds_contrib: [f64; 3] = [0.0; 3];

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
                                    if shell_i.angular_momentum < 2 && shell_j.angular_momentum < 2
                                    {
                                        let orbital1 = &self.basis.basis_functions
                                            [shell_i.start + idx_i_local];
                                        let orbital2 = &self.basis.basis_functions
                                            [shell_j.start + idx_j_local];
                                        let norm_prod =
                                            orbital1.contracted_norm * orbital2.contracted_norm;

                                        // Combined factor: h0_val * P + Eff (effective combines -W + coulomb*P - third*P)
                                        let eff_ij = effective_mat[[idx_i, idx_j]];
                                        let combined_factor = h0_val * p_ij + eff_ij;

                                        // Compute all 3 overlap derivatives at once
                                        let ds_all =
                                            obara_saika_derivatives_all(orbital1, orbital2);

                                        // Accumulate dS * combined_factor at shell level
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

                    // D-orbital handling: compute overlap derivatives using calc_overlap_derivative_d_shells
                    // Process unique shell pairs (shell_i_idx < shell_j_idx) with factor 2 for symmetric storage.
                    // ds_d[0-2] gives derivative w.r.t. at_i, ds_d[3-5] gives derivative w.r.t. at_j.
                    // Both atoms receive their contributions with factor 2 to account for symmetric matrix storage.
                    let shell_i_has_d = shell_i.angular_momentum >= 2;
                    let shell_j_has_d = shell_j.angular_momentum >= 2;
                    let either_has_d = shell_i_has_d || shell_j_has_d;

                    // D-orbital handling: process unique pairs (shell_i_idx < shell_j_idx) with factor 2
                    if at_i != at_j && either_has_d && shell_i_idx < shell_j_idx {
                        let ds_d = calc_overlap_derivative_d_shells(&self.basis, shell_i, shell_j);
                        let sph_dim_i = shell_i.sph_end - shell_i.sph_start;
                        let sph_dim_j = shell_j.sph_end - shell_j.sph_start;

                        for sph_i in 0..sph_dim_i {
                            let idx_i = shell_i.sph_start + sph_i;
                            for sph_j in 0..sph_dim_j {
                                let idx_j = shell_j.sph_start + sph_j;

                                let p_ij = p[[idx_i, idx_j]];
                                let eff_ij = effective_mat[[idx_i, idx_j]];

                                for dir in 0..3 {
                                    // Factor 2 accounts for symmetric matrix storage (both (i,j) and (j,i))
                                    // ds_d[0-2] is derivative w.r.t. at_i
                                    let ds_val_i = 2.0 * ds_d[[dir, sph_i, sph_j]];
                                    // ds_d[3-5] is derivative w.r.t. at_j
                                    let ds_val_j = 2.0 * ds_d[[3 + dir, sph_i, sph_j]];

                                    // Combined contribution: h0*P + Eff (effective combines -W + coulomb*P - third*P)
                                    let combined_factor = h0_val * p_ij + eff_ij;
                                    gradient[3 * at_i + dir] += ds_val_i * combined_factor;
                                    gradient[3 * at_j + dir] += ds_val_j * combined_factor;
                                }

                                // Accumulate S*P for pi gradient (d-orbital pairs)
                                shell_pi_sp_sum += s[[idx_i, idx_j]] * p_ij;
                            }
                        }
                    }

                    // Apply shell-level contributions to gradient (s/p orbitals)
                    if at_i != at_j {
                        for dir in 0..3 {
                            gradient[3 * at_i + dir] += shell_ds_contrib[dir];
                            gradient[3 * at_j + dir] -= shell_ds_contrib[dir];
                        }

                        // Apply shell-level pi gradient (factor of 2 for symmetric storage)
                        let pi_contrib = 2.0 * pi_factor * shell_pi_sp_sum;
                        for dir in 0..3 {
                            gradient[3 * at_i + dir] += pi_grad[dir] * pi_contrib;
                        }
                        // For d-orbital pairs processed via unique-pair approach (shell_i_idx < shell_j_idx),
                        // the swapped pair (j,i) is never processed, so at_j misses its pi contribution.
                        // Add it here with opposite sign (Newton's 3rd law).
                        if either_has_d && shell_i_idx < shell_j_idx {
                            for dir in 0..3 {
                                gradient[3 * at_j + dir] -= pi_grad[dir] * pi_contrib;
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

            // Apply deferred CN gradient contributions
            for at in 0..self.n_atoms {
                if cn_factors[at].abs() > 1e-15 {
                    let cn_grad_at: ArrayView1<f64> = cn_number_grads.slice(s![.., at]);
                    for k in 0..(3 * self.n_atoms) {
                        gradient[k] += cn_factors[at] * cn_grad_at[k];
                    }
                }
            }

            // Gamma gradient - separate loop without PROXIMITY_CUTOFF
            // Gamma function decays as 1/r, so long-range contributions are important
            for (shell_i_idx, shell_i) in self.basis.shells.iter().enumerate() {
                let atomi = &atoms[shell_i.atom_index];
                let at_i = shell_i.atom_index;

                for (shell_j_idx, shell_j) in self.basis.shells.iter().enumerate() {
                    let atomj = &atoms[shell_j.atom_index];
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

                        let gamma_deriv = self.gammafunction.deriv(
                            distance,
                            atomi.number,
                            shell_i.angular_momentum as u8,
                            atomj.number,
                            shell_j.angular_momentum as u8,
                        );

                        let shell_dq_prod = dq_shell[shell_i_idx] * dq_shell[shell_j_idx];
                        let shell_gamma_contrib = 0.5 * gamma_deriv * shell_dq_prod;
                        for dir in 0..3 {
                            gradient[3 * at_i + dir] += e_ij[dir] * shell_gamma_contrib;
                            gradient[3 * at_j + dir] -= e_ij[dir] * shell_gamma_contrib;
                        }
                    }
                }
            }

            // Repulsive energy gradient
            let grad_rep_contrib = grad_repulsive_energy_xtb(atoms, self.n_atoms);
            gradient += &grad_rep_contrib;

            // Dispersion gradient
            let grad_disp_contrib = gradient_disp3_xtb(&atoms, &config);
            gradient += &grad_disp_contrib;

            gradient
        }
    };
}

impl XtbMonomer<'_> {
    impl_ground_state_gradient_on_the_fly!();
}

impl XtbPair<'_> {
    impl_ground_state_gradient_on_the_fly!();
}
