use crate::defaults::PROXIMITY_CUTOFF;
use crate::fmo::scc::helpers::atomvec_to_aomat;
use crate::fmo::{Monomer, Pair};
use crate::gradients::dispersion::gradient_disp;
use crate::gradients::helpers::{
    compute_lr_coefficients_onthefly, f_lr_par, f_v_par, gradient_v_rep,
};
use crate::initialization::*;
use crate::io::Configuration;
use crate::param::slako_transformations::{
    directional_cosines, slako_transformation_gradients_fast, SplineCache,
};
use crate::scc::construct_third_order_gradient_contribution;
use crate::scc::gamma_approximation::{
    gamma_gradients_ao_wise, gamma_gradients_ao_wise_shell_resolved, gamma_gradients_atomwise,
    gamma_third_order_derivative,
};
use crate::scc::h0_and_s::h0_and_s_gradients;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use ndarray_npy::write_npy;
use rayon::prelude::*;

impl System {
    /// Ground-state DFTB gradient. Dispatches to the atom-resolved or the
    /// shell-resolved on-the-fly implementation depending on the gamma mode,
    /// and writes `gs_gradient.npy` for `grad` jobs.
    pub fn ground_state_gradient(&mut self, _excited_gradients: bool) -> Array1<f64> {
        let gradient: Array1<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.ground_state_gradient_onthefly()
        } else {
            self.ground_state_gradient_shell_resolved_onthefly()
        };
        if self.config.jobtype == "grad" {
            // save the gradient
            write_npy("gs_gradient.npy", &gradient).unwrap();
        }
        gradient
    }

    /// Atom-resolved ground-state gradient: sums the H0, overlap (Pulay),
    /// gamma (Coulomb), repulsive and dispersion contributions.
    pub fn ground_state_gradient_atomwise(&mut self, excited_gradients: bool) -> Array1<f64> {
        // for the evaluation of the gradient it is necessary to compute the derivatives
        // of: - H0
        //     - S
        //     - Gamma
        //     - Repulsive Potential
        // the first three properties are calculated here at the beginning and the gradient that
        // originates from the repulsive potential is added at the end to total gradient

        // derivative of H0 and S
        let (grad_s, grad_h0) = h0_and_s_gradients(&self.atoms, self.n_orbs, &self.slako);

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

        // derivative of the gamma matrix and transform it in the same way to a 2D array
        let grad_gamma: Array2<f64> =
            gamma_gradients_atomwise(&self.gammafunction, &self.atoms, self.n_atoms)
                .into_shape([3 * self.n_atoms, self.n_atoms * self.n_atoms])
                .unwrap();

        // take references/views to the necessary properties from the scc calculation
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();

        // transform the expression Sum_c_in_X (gamma_AC + gamma_aC) * dq_C
        // into matrix of the dimension (norb, norb) to do an element wise multiplication with P
        let coulomb_mat: Array2<f64> =
            atomvec_to_aomat(gamma.dot(&dq).view(), self.n_orbs, &self.atoms) * 0.5;

        // The product of the Coulomb interaction matrix and the density matrix flattened as vector.
        let coulomb_x_p: Array1<f64> = (&p * &coulomb_mat)
            .into_shape([self.n_orbs * self.n_orbs])
            .unwrap();

        // The density matrix in vector form.
        let p_flat: ArrayView1<f64> = p.into_shape([self.n_orbs * self.n_orbs]).unwrap();

        // the gradient part which involves the gradient of the gamma matrix is given by:
        // 1/2 * dq . dGamma / dR . dq
        // the dq's are element wise multiplied into a 2D array and reshaped into a flat one, that
        // has the length of natoms^2. this allows to do only a single matrix vector product of
        // 'grad_gamma' with 'dq_x_dq' and avoids to reshape dGamma multiple times
        let dq_column: ArrayView2<f64> = dq.insert_axis(Axis(1));
        let dq_x_dq: Array1<f64> = (&dq_column.broadcast((self.n_atoms, self.n_atoms)).unwrap()
            * &dq)
            .into_shape([self.n_atoms * self.n_atoms])
            .unwrap();

        // compute the energy weighted density matrix: W = 1/2 * D . (H + H_Coul) . D
        let w: Array1<f64> = 0.5
            * (p.dot(&self.properties.h_coul_x().unwrap()).dot(&p))
                .into_shape([self.n_orbs * self.n_orbs])
                .unwrap();

        // calculation of the gradient
        // 1st part:  dH0 / dR . P
        let mut gradient: Array1<f64> = grad_h0_2d.dot(&p_flat);

        // 2nd part: dS / dR . W
        gradient -= &grad_s_2d.dot(&w);

        // 3rd part: 1/2 * dS / dR * sum_c_in_X (gamma_ac + gamma_bc) * dq_c
        gradient += &grad_s_2d.dot(&coulomb_x_p);

        // 4th part: 1/2 * dq . dGamma / dR . dq
        gradient += &(grad_gamma.dot(&dq_x_dq));

        // last part: dV_rep / dR
        gradient = gradient + gradient_v_rep(&self.atoms, &self.vrep);

        if self.config.dftb3.use_dftb3 {
            // get the third order gamma matrix
            let gamma_third_order: ArrayView2<f64> = self.properties.gamma_third_order().unwrap();

            // gradient += &contribution;
            let coulomb_mat_third_order = construct_third_order_gradient_contribution(
                self.n_orbs,
                &self.atoms,
                gamma_third_order,
                dq,
            ) * 0.5;
            // multiply with the density matrix
            let coulomb_p_third_order: Array1<f64> = (&p * &coulomb_mat_third_order)
                .into_shape([self.n_orbs * self.n_orbs])
                .unwrap();

            // add the contribution to the gradient by matrix multiplying with the
            // gradient of the overlap matrix
            gradient += &grad_s_2d.dot(&coulomb_p_third_order);

            // calculate the derivative of the third order gamma matrix
            let grad_gamma_third_order = gamma_third_order_derivative(
                &self.gammafunction,
                &self.atoms,
                self.n_atoms,
                &self.config.dftb3.hubbard_derivatives,
            );
            let mut contribution: Array1<f64> = Array1::zeros(gradient.raw_dim());
            let dq2: Array1<f64> = dq.map(|val| val.powi(2));

            for nc in 0..3 * self.n_atoms {
                let dgamma_slice: ArrayView2<f64> = grad_gamma_third_order.slice(s![nc, .., ..]);
                contribution[nc] = dq2.dot(&dgamma_slice.dot(&dq));
            }
            contribution /= 3.0;

            gradient += &contribution;
        }

        // dispersion
        if self.config.dispersion.use_dispersion {
            gradient = gradient + gradient_disp(&self.atoms, &self.config);
        }

        // long-range corrected part of the gradient
        if self.config.lc.long_range_correction {
            let (g1_lr, g1_lr_ao): (Array3<f64>, Array3<f64>) = gamma_gradients_ao_wise(
                self.gammafunction_lc.as_ref().unwrap(),
                &self.atoms,
                self.n_atoms,
                self.n_orbs,
            );

            let diff_p: Array2<f64> = &p - &self.properties.p_ref().unwrap();
            let flr_dmd0: Array3<f64> = f_lr_par(
                diff_p.view(),
                self.properties.s().unwrap(),
                grad_s.view(),
                self.properties.gamma_lr_ao().unwrap(),
                g1_lr_ao.view(),
                self.n_atoms,
                self.n_orbs,
            );
            gradient = gradient
                - 0.25
                    * flr_dmd0
                        .view()
                        .into_shape((3 * self.n_atoms, self.n_orbs * self.n_orbs))
                        .unwrap()
                        .dot(&diff_p.into_shape(self.n_orbs * self.n_orbs).unwrap());

            // save necessary properties for the excited gradient calculation with lr-correction
            if excited_gradients {
                self.properties.set_grad_gamma_lr(g1_lr);
                self.properties.set_grad_gamma_lr_ao(g1_lr_ao);
                self.properties.set_f_lr_dmd0(flr_dmd0);
            }
        }
        // save necessary properties for the excited gradient calculation
        if excited_gradients {
            self.properties.set_grad_s(grad_s);
            self.properties.set_grad_h0(grad_h0);
            self.properties.set_grad_gamma(
                grad_gamma
                    .into_shape([3 * self.n_atoms, self.n_atoms, self.n_atoms])
                    .unwrap(),
            );
        }

        gradient
    }

    /// Shell-resolved counterpart of [`Self::ground_state_gradient_atomwise`]
    /// (uses the shell-resolved gamma matrix).
    pub fn ground_state_gradient_shell_resolved(&mut self, excited_gradients: bool) -> Array1<f64> {
        // for the evaluation of the gradient it is necessary to compute the derivatives
        // of: - H0
        //     - S
        //     - Gamma
        //     - Repulsive Potential
        // the first three properties are calculated here at the beginning and the gradient that
        // originates from the repulsive potential is added at the end to total gradient

        // derivative of H0 and S
        let (grad_s, grad_h0) = h0_and_s_gradients(&self.atoms, self.n_orbs, &self.slako);

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

        // derivative of the gamma matrix and transform it in the same way to a 2D array
        let grad_gamma: Array3<f64> = gamma_gradients_ao_wise_shell_resolved(
            &self.gammafunction,
            &self.atoms,
            self.n_atoms,
            self.n_orbs,
        );
        // let grad_gamma_2d: ArrayView2<f64> = grad_gamma
        //     .view()
        //     .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
        //     .unwrap();

        // take references/views to the necessary properties from the scc calculation
        let gamma: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        let dp: Array2<f64> = &p - &self.properties.p_ref().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        let f_term: Array2<f64> = f_v_par(
            dp.view(),
            s,
            grad_s.view(),
            gamma,
            grad_gamma.view(),
            self.n_atoms,
            self.n_orbs,
        )
        .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
        .unwrap();

        // The density matrix in vector form.
        let p_flat: ArrayView1<f64> = p.into_shape([self.n_orbs * self.n_orbs]).unwrap();

        // compute the energy weighted density matrix: W = 1/2 * D . (H + H_Coul) . D
        let w: Array1<f64> = 0.5
            * (p.dot(&self.properties.h_coul_x().unwrap()).dot(&p))
                .into_shape([self.n_orbs * self.n_orbs])
                .unwrap();

        // calculation of the gradient
        // 1st part:  dH0 / dR . P
        let mut gradient: Array1<f64> = grad_h0_2d.dot(&p_flat);

        // 2nd part: dS / dR . W
        gradient -= &grad_s_2d.dot(&w);

        // 3rd part
        gradient = gradient + 0.5 * f_term.dot(&dp.into_shape(self.n_orbs * self.n_orbs).unwrap());

        // last part: dV_rep / dR
        gradient = gradient + gradient_v_rep(&self.atoms, &self.vrep);

        // dispersion
        if self.config.dispersion.use_dispersion {
            gradient = gradient + gradient_disp(&self.atoms, &self.config);
        }

        // long-range corrected part of the gradient
        if self.config.lc.long_range_correction {
            let g1_lr_ao: Array3<f64> = gamma_gradients_ao_wise_shell_resolved(
                self.gammafunction_lc.as_ref().unwrap(),
                &self.atoms,
                self.n_atoms,
                self.n_orbs,
            );

            let diff_p: Array2<f64> = &p - &self.properties.p_ref().unwrap();
            let flr_dmd0: Array3<f64> = f_lr_par(
                diff_p.view(),
                self.properties.s().unwrap(),
                grad_s.view(),
                self.properties.gamma_lr_ao().unwrap(),
                g1_lr_ao.view(),
                self.n_atoms,
                self.n_orbs,
            );
            gradient = gradient
                - 0.25
                    * flr_dmd0
                        .view()
                        .into_shape((3 * self.n_atoms, self.n_orbs * self.n_orbs))
                        .unwrap()
                        .dot(&diff_p.into_shape(self.n_orbs * self.n_orbs).unwrap());

            // save necessary properties for the excited gradient calculation with lr-correction
            if excited_gradients {
                self.properties.set_grad_gamma_lr_ao(g1_lr_ao);
                self.properties.set_f_lr_dmd0(flr_dmd0);
            }
        }
        // save necessary properties for the excited gradient calculation
        if excited_gradients {
            self.properties.set_grad_s(grad_s);
            self.properties.set_grad_h0(grad_h0);
            self.properties.set_grad_gamma_ao(grad_gamma);
        }

        gradient
    }

    /// On-the-fly gradient calculation - avoids storing huge 3D arrays
    /// Computes H0/S, gamma, and LC gradients all on-the-fly
    pub fn ground_state_gradient_onthefly(&mut self) -> Array1<f64> {
        // Get references to SCC results
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // Compute energy-weighted density matrix W = 0.5 * P * H_coul * P
        let h_coul: ArrayView2<f64> = self.properties.h_coul_x().unwrap();
        let w: Array2<f64> = 0.5 * p.dot(&h_coul).dot(&p);

        // Coulomb matrix: 0.5 * sum_C (gamma_AC + gamma_BC) * dq_C
        let coulomb_mat: Array2<f64> =
            atomvec_to_aomat(gamma.dot(&dq).view(), self.n_orbs, &self.atoms) * 0.5;

        // Effective matrix for dS term: -W + coulomb * P
        let effective_mat: Array2<f64> = &(-&w) + &(&coulomb_mat * &p);
        let n_grad = 3 * self.n_atoms;

        // Precompute LC coefficients if long-range correction is enabled
        let (coeff_s_lr, coeff_g_lr, diff_p): (
            Option<Array2<f64>>,
            Option<Array2<f64>>,
            Option<Array2<f64>>,
        ) = if self.config.lc.long_range_correction {
            let diff_p_val: Array2<f64> = &p - &self.properties.p_ref().unwrap();
            let gamma_lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
            let (cs, cg) = compute_lr_coefficients_onthefly(diff_p_val.view(), s, gamma_lr_ao);
            (Some(cs), Some(cg), Some(diff_p_val))
        } else {
            (None, None, None)
        };

        // Precompute DFTB3 coefficient matrix if enabled (for Term 1: overlap derivative contribution)
        let coeff_dftb3: Option<Array2<f64>> = if self.config.dftb3.use_dftb3 {
            let gamma_third_order: ArrayView2<f64> = self.properties.gamma_third_order().unwrap();
            let coulomb_mat_third = construct_third_order_gradient_contribution(
                self.n_orbs,
                &self.atoms,
                gamma_third_order,
                dq,
            ) * 0.5;
            Some(&p * &coulomb_mat_third) // coeff_dftb3[[mu,nu]] = p[[mu,nu]] * coulomb_mat[[mu,nu]]
        } else {
            None
        };

        // On-the-fly H0/S gradient computation (and LC overlap contribution)
        // Optimized: iterate over unique atom pairs (i < j) and compute both contributions at once
        // Pre-compute orbital index offsets for each atom
        let mut orbital_offsets: Vec<usize> = Vec::with_capacity(self.n_atoms + 1);
        orbital_offsets.push(0);
        for atom in &self.atoms {
            orbital_offsets.push(orbital_offsets.last().unwrap() + atom.n_orbs);
        }

        // Collect valid atom pairs (i < j, within cutoff)
        let mut atom_pairs: Vec<(usize, usize)> = Vec::new();
        for i in 0..self.n_atoms {
            for j in (i + 1)..self.n_atoms {
                let distance = (&self.atoms[i] - &self.atoms[j]).norm();
                if distance < PROXIMITY_CUTOFF {
                    atom_pairs.push((i, j));
                }
            }
        }

        // Parallel compute gradient contributions for each pair
        let pair_contributions: Vec<([f64; 3], [f64; 3], usize, usize)> = atom_pairs
            .par_iter()
            .map(|&(i, j)| {
                let atomi = &self.atoms[i];
                let atomj = &self.atoms[j];
                let mu_start = orbital_offsets[i];
                let nu_start = orbital_offsets[j];

                // Directional cosines from smaller atom type to larger (for consistent SK tables)
                // atomi <= atomj compares atom TYPES (atomic numbers), not indices
                let (r, x, y, z): (f64, f64, f64, f64) = if atomi <= atomj {
                    directional_cosines(&atomi.xyz, &atomj.xyz)
                } else {
                    directional_cosines(&atomj.xyz, &atomi.xyz)
                };

                // Pre-compute spline caches for this atom pair (once per pair)
                let skt = self.slako.get(atomi.kind, atomj.kind);
                let s_cache = SplineCache::new(r, &skt.s_spline);
                let h_cache = SplineCache::new(r, &skt.h_spline);

                let mut grad_i = [0.0_f64; 3];
                let mut grad_j = [0.0_f64; 3];

                let mut mu = mu_start;
                for orbi in atomi.valorbs.iter() {
                    let mut nu = nu_start;
                    for orbj in atomj.valorbs.iter() {
                        // Compute gradient with proper orbital and sign handling
                        // based on atom type ordering (not index ordering)
                        let (s_deriv_i, h0_deriv_i, s_deriv_j, h0_deriv_j): (
                            [f64; 3],
                            [f64; 3],
                            [f64; 3],
                            [f64; 3],
                        ) = if atomi <= atomj {
                            // atomi is smaller type: r points from i to j
                            // dS/dr_i = -dS/dr, dS/dr_j = +dS/dr
                            let s_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &s_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                            );
                            let h0_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &h_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                            );
                            (
                                [-s_grad[0], -s_grad[1], -s_grad[2]],
                                [-h0_grad[0], -h0_grad[1], -h0_grad[2]],
                                s_grad,
                                h0_grad,
                            )
                        } else {
                            // atomj is smaller type: r points from j to i
                            // dS/dr_i = +dS/dr, dS/dr_j = -dS/dr
                            let s_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &s_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                            );
                            let h0_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &h_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                            );
                            (
                                s_grad,
                                h0_grad,
                                [-s_grad[0], -s_grad[1], -s_grad[2]],
                                [-h0_grad[0], -h0_grad[1], -h0_grad[2]],
                            )
                        };

                        let p_mu_nu = p[[mu, nu]];
                        let eff_mu_nu = effective_mat[[mu, nu]];

                        // H0/S contribution for atom i
                        for dir in 0..3 {
                            grad_i[dir] +=
                                2.0 * (h0_deriv_i[dir] * p_mu_nu + s_deriv_i[dir] * eff_mu_nu);
                        }
                        // H0/S contribution for atom j (same matrix elements due to symmetry)
                        for dir in 0..3 {
                            grad_j[dir] +=
                                2.0 * (h0_deriv_j[dir] * p_mu_nu + s_deriv_j[dir] * eff_mu_nu);
                        }

                        // LC overlap contribution
                        if let Some(ref coeff_s) = coeff_s_lr {
                            let coeff_mu_nu = coeff_s[[mu, nu]];
                            for dir in 0..3 {
                                grad_i[dir] -= 0.0625 * s_deriv_i[dir] * coeff_mu_nu;
                                grad_j[dir] -= 0.0625 * s_deriv_j[dir] * coeff_mu_nu;
                            }
                        }

                        // DFTB3 overlap contribution (Term 1)
                        if let Some(ref coeff) = coeff_dftb3 {
                            let coeff_mu_nu = coeff[[mu, nu]];
                            for dir in 0..3 {
                                grad_i[dir] += 2.0 * s_deriv_i[dir] * coeff_mu_nu;
                                grad_j[dir] += 2.0 * s_deriv_j[dir] * coeff_mu_nu;
                            }
                        }

                        nu += 1;
                    }
                    mu += 1;
                }
                (grad_i, grad_j, i, j)
            })
            .collect();

        // Reduce contributions to gradient
        let mut gradient: Array1<f64> = Array1::zeros(n_grad);
        for (grad_i, grad_j, i, j) in pair_contributions {
            for dir in 0..3 {
                gradient[3 * i + dir] += grad_i[dir];
                gradient[3 * j + dir] += grad_j[dir];
            }
        }

        // On-the-fly gamma gradient (standard SCC contribution)
        for i in 0..self.n_atoms {
            let atomi = &self.atoms[i];
            for j in (i + 1)..self.n_atoms {
                let atomj = &self.atoms[j];
                let r_vec = atomi - atomj;
                let r_ij = r_vec.norm();
                let e_ij = r_vec / r_ij;
                let gamma_deriv = self.gammafunction.deriv(r_ij, atomi.number, atomj.number);
                let contrib = gamma_deriv * dq[i] * dq[j];
                gradient[3 * i + 0] += e_ij.x * contrib;
                gradient[3 * i + 1] += e_ij.y * contrib;
                gradient[3 * i + 2] += e_ij.z * contrib;
                gradient[3 * j + 0] -= e_ij.x * contrib;
                gradient[3 * j + 1] -= e_ij.y * contrib;
                gradient[3 * j + 2] -= e_ij.z * contrib;
            }
        }

        // On-the-fly LC gamma_lr_ao gradient contribution
        if let (Some(ref coeff_g), Some(_)) = (&coeff_g_lr, &diff_p) {
            let gamma_lc = self.gammafunction_lc.as_ref().unwrap();
            // Iterate over atom pairs and compute gamma_lr_ao derivative contributions
            for i in 0..self.n_atoms {
                let atomi = &self.atoms[i];
                let mu_start: usize = self.atoms[..i].iter().map(|a| a.n_orbs).sum();
                let n_orbs_i = atomi.n_orbs;

                for j in (i + 1)..self.n_atoms {
                    let atomj = &self.atoms[j];
                    let r_vec = atomi - atomj;
                    let r_ij = r_vec.norm();
                    let e_ij = r_vec / r_ij;

                    let nu_start: usize = self.atoms[..j].iter().map(|a| a.n_orbs).sum();
                    let n_orbs_j = atomj.n_orbs;

                    // gamma_lr derivative (scalar, same for all orbital pairs between these atoms)
                    let gamma_lr_deriv = gamma_lc.deriv(r_ij, atomi.number, atomj.number);

                    // Sum coefficients for all orbital pairs between atoms i and j
                    // Note: coeff_g is already symmetrized, so coeff_g[[mu,nu]] = coeff_g[[nu,mu]]
                    // We only sum coeff_g[[mu,nu]] once (not both directions) because the reference
                    // computation sums over both (mu,nu) and (nu,mu) entries in g1_lr_ao, which
                    // have the same value, effectively giving (coeff_g[mu,nu] + coeff_g[nu,mu]).
                    // Since we symmetrized coeff_g = coeff_g + coeff_g^T, using just coeff_g[[mu,nu]]
                    // already includes both contributions.
                    let mut coeff_sum = 0.0;
                    for mu_off in 0..n_orbs_i {
                        for nu_off in 0..n_orbs_j {
                            let mu = mu_start + mu_off;
                            let nu = nu_start + nu_off;
                            coeff_sum += coeff_g[[mu, nu]];
                        }
                    }

                    let contrib = -0.0625 * gamma_lr_deriv * coeff_sum;
                    gradient[3 * i + 0] += e_ij.x * contrib;
                    gradient[3 * i + 1] += e_ij.y * contrib;
                    gradient[3 * i + 2] += e_ij.z * contrib;
                    gradient[3 * j + 0] -= e_ij.x * contrib;
                    gradient[3 * j + 1] -= e_ij.y * contrib;
                    gradient[3 * j + 2] -= e_ij.z * contrib;
                }
            }
        }

        // Repulsive potential gradient
        gradient = gradient + gradient_v_rep(&self.atoms, &self.vrep);

        // DFTB3 Term 2: gamma derivative contribution (does NOT use grad_s)
        if self.config.dftb3.use_dftb3 {
            let grad_gamma_third_order = gamma_third_order_derivative(
                &self.gammafunction,
                &self.atoms,
                self.n_atoms,
                &self.config.dftb3.hubbard_derivatives,
            );
            let mut contribution: Array1<f64> = Array1::zeros(gradient.raw_dim());
            let dq2: Array1<f64> = dq.map(|val| val.powi(2));
            for nc in 0..3 * self.n_atoms {
                let dgamma_slice: ArrayView2<f64> = grad_gamma_third_order.slice(s![nc, .., ..]);
                contribution[nc] = dq2.dot(&dgamma_slice.dot(&dq));
            }
            contribution /= 3.0;
            gradient += &contribution;
        }

        // Dispersion gradient
        if self.config.dispersion.use_dispersion {
            gradient = gradient + gradient_disp(&self.atoms, &self.config);
        }

        gradient
    }

    /// On-the-fly shell-resolved gradient computation
    /// Uses shell-resolved gamma (AO-wise) instead of atomwise gamma
    /// True on-the-fly approach: no full 3D arrays stored
    ///
    /// The f_v term is computed as:
    /// f_v[nc,a,b] = 0.25 * (ds[nc,a,b] * (gsv[a] + gsv[b]) + s[a,b] * (dgsv[nc,a] + gdsv[nc,a] + dgsv[nc,b] + gdsv[nc,b]))
    ///
    /// After contraction with dp:
    /// Part A: 0.125 * sum_{a,b} ds[nc,a,b] * (gsv[a] + gsv[b]) * dp[a,b] - computed in H0/S loop
    /// Part B: 0.25 * sum_a (dgsv[nc,a] + gdsv[nc,a]) * sdp[a] - computed per gradient direction
    pub fn ground_state_gradient_shell_resolved_onthefly(&mut self) -> Array1<f64> {
        // Get references to SCC results
        let gamma_ao: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let dp: Array2<f64> = &p - &self.properties.p_ref().unwrap();

        // Compute energy-weighted density matrix W = 0.5 * P * H_coul * P
        let h_coul: ArrayView2<f64> = self.properties.h_coul_x().unwrap();
        let w: Array2<f64> = 0.5 * p.dot(&h_coul).dot(&p);

        let n_grad = 3 * self.n_atoms;

        // === Precompute quantities for f_v ===

        // vp = dp + dp^T (symmetrized difference density)
        let vp: Array2<f64> = &dp + &dp.t();

        // sv[a] = sum_b s[a,b] * vp[b]
        let sv: Array1<f64> = (&s * &vp).sum_axis(Axis(1));

        // gsv[a] = sum_b gamma_ao[a,b] * sv[b]
        let gsv: Array1<f64> = gamma_ao.dot(&sv);

        // sdp[a] = sum_b s[a,b] * dp[a,b]
        let sdp: Array1<f64> = (&s * &dp).sum_axis(Axis(1));

        // coeff_A[a,b] = (gsv[a] + gsv[b]) * dp[a,b]
        // For Part A: gradient += 0.125 * ds[a,b] * coeff_A[a,b]
        let mut coeff_a: Array2<f64> = Array2::zeros((self.n_orbs, self.n_orbs));
        for a in 0..self.n_orbs {
            for b in 0..self.n_orbs {
                coeff_a[[a, b]] = (gsv[a] + gsv[b]) * dp[[a, b]];
            }
        }

        // === Precompute LC coefficients if needed ===
        let (coeff_s_lr, coeff_g_lr): (Option<Array2<f64>>, Option<Array2<f64>>) =
            if self.config.lc.long_range_correction {
                let gamma_lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
                let (cs, cg) = compute_lr_coefficients_onthefly(dp.view(), s, gamma_lr_ao);
                (Some(cs), Some(cg))
            } else {
                (None, None)
            };

        // === H0/S gradient loop with Part A of f_v ===
        // Pre-compute orbital index offsets for each atom
        let mut orbital_offsets: Vec<usize> = Vec::with_capacity(self.n_atoms + 1);
        orbital_offsets.push(0);
        for atom in &self.atoms {
            orbital_offsets.push(orbital_offsets.last().unwrap() + atom.n_orbs);
        }

        // Collect valid atom pairs (i < j, within cutoff)
        let mut atom_pairs: Vec<(usize, usize)> = Vec::new();
        for i in 0..self.n_atoms {
            for j in (i + 1)..self.n_atoms {
                let distance = (&self.atoms[i] - &self.atoms[j]).norm();
                if distance < PROXIMITY_CUTOFF {
                    atom_pairs.push((i, j));
                }
            }
        }

        // Parallel compute gradient contributions for each pair (H0/S + Part A of f_v)
        let pair_contributions: Vec<([f64; 3], [f64; 3], usize, usize)> = atom_pairs
            .par_iter()
            .map(|&(i, j)| {
                let atomi = &self.atoms[i];
                let atomj = &self.atoms[j];
                let mu_start = orbital_offsets[i];
                let nu_start = orbital_offsets[j];

                // Directional cosines
                let (r, x, y, z): (f64, f64, f64, f64) = if atomi <= atomj {
                    directional_cosines(&atomi.xyz, &atomj.xyz)
                } else {
                    directional_cosines(&atomj.xyz, &atomi.xyz)
                };

                // Pre-compute spline caches
                let skt = self.slako.get(atomi.kind, atomj.kind);
                let s_cache = SplineCache::new(r, &skt.s_spline);
                let h_cache = SplineCache::new(r, &skt.h_spline);

                let mut grad_i = [0.0_f64; 3];
                let mut grad_j = [0.0_f64; 3];

                let mut mu = mu_start;
                for orbi in atomi.valorbs.iter() {
                    let mut nu = nu_start;
                    for orbj in atomj.valorbs.iter() {
                        let (s_deriv_i, h0_deriv_i, s_deriv_j, h0_deriv_j): (
                            [f64; 3],
                            [f64; 3],
                            [f64; 3],
                            [f64; 3],
                        ) = if atomi <= atomj {
                            let s_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &s_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                            );
                            let h0_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &h_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                            );
                            (
                                [-s_grad[0], -s_grad[1], -s_grad[2]],
                                [-h0_grad[0], -h0_grad[1], -h0_grad[2]],
                                s_grad,
                                h0_grad,
                            )
                        } else {
                            let s_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &s_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                            );
                            let h0_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &h_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                            );
                            (
                                s_grad,
                                h0_grad,
                                [-s_grad[0], -s_grad[1], -s_grad[2]],
                                [-h0_grad[0], -h0_grad[1], -h0_grad[2]],
                            )
                        };

                        let p_mu_nu = p[[mu, nu]];
                        let w_mu_nu = w[[mu, nu]];
                        let coeff_a_mu_nu = coeff_a[[mu, nu]];

                        // H0/S contribution: dH0*P - dS*W
                        for dir in 0..3 {
                            grad_i[dir] +=
                                2.0 * (h0_deriv_i[dir] * p_mu_nu - s_deriv_i[dir] * w_mu_nu);
                            grad_j[dir] +=
                                2.0 * (h0_deriv_j[dir] * p_mu_nu - s_deriv_j[dir] * w_mu_nu);
                        }

                        // f_v Part A: ds * (gsv[a] + gsv[b]) * dp contracted with 0.5
                        // f_v has factor 0.25, contracted with 0.5, symmetry factor 2 = 0.25
                        for dir in 0..3 {
                            grad_i[dir] += 0.25 * s_deriv_i[dir] * coeff_a_mu_nu;
                            grad_j[dir] += 0.25 * s_deriv_j[dir] * coeff_a_mu_nu;
                        }

                        // LC overlap contribution
                        if let Some(ref coeff_s) = coeff_s_lr {
                            let coeff_mu_nu = coeff_s[[mu, nu]];
                            for dir in 0..3 {
                                grad_i[dir] -= 0.0625 * s_deriv_i[dir] * coeff_mu_nu;
                                grad_j[dir] -= 0.0625 * s_deriv_j[dir] * coeff_mu_nu;
                            }
                        }

                        nu += 1;
                    }
                    mu += 1;
                }
                (grad_i, grad_j, i, j)
            })
            .collect();

        // Reduce H0/S + Part A contributions to gradient
        let mut gradient: Array1<f64> = Array1::zeros(n_grad);
        for (grad_i, grad_j, i, j) in pair_contributions {
            for dir in 0..3 {
                gradient[3 * i + dir] += grad_i[dir];
                gradient[3 * j + dir] += grad_j[dir];
            }
        }

        // === Part B of f_v: 0.25 * sum_a (dgsv + gdsv) * sdp ===
        // Parallelized over gradient directions (each nc is independent)
        // For each gradient direction nc = 3*K + d (moving atom K in direction d):
        // - dgsv[a] = sum_b dg[nc,a,b] * sv[b], where dg is the gamma_ao derivative
        // - gdsv[a] = sum_b gamma_ao[a,b] * dsv[b], where dsv[b] = sum_c ds[nc,b,c] * vp[c]

        let part_b_contributions: Vec<f64> = (0..n_grad)
            .into_par_iter()
            .map(|nc| {
                let k = nc / 3;
                let d = nc % 3;

                let atomk = &self.atoms[k];
                let mu_k_start = orbital_offsets[k];

                // Initialize dsv (will be sparse - only orbitals on K and neighbors within cutoff)
                let mut dsv: Array1<f64> = Array1::zeros(self.n_orbs);

                // Initialize dgsv (includes contributions from ALL atoms, not just cutoff neighbors)
                let mut dgsv: Array1<f64> = Array1::zeros(self.n_orbs);

                // === Compute dsv: only for pairs within PROXIMITY_CUTOFF ===
                for l in 0..self.n_atoms {
                    if k == l {
                        continue;
                    }

                    let atoml = &self.atoms[l];
                    let distance = (atomk - atoml).norm();
                    if distance >= PROXIMITY_CUTOFF {
                        continue;
                    }

                    let nu_l_start = orbital_offsets[l];

                    let (r, x, y, z): (f64, f64, f64, f64) = if atomk <= atoml {
                        directional_cosines(&atomk.xyz, &atoml.xyz)
                    } else {
                        directional_cosines(&atoml.xyz, &atomk.xyz)
                    };

                    let skt = self.slako.get(atomk.kind, atoml.kind);
                    let s_cache = SplineCache::new(r, &skt.s_spline);

                    let mut mu_k = mu_k_start;
                    for orbk in atomk.valorbs.iter() {
                        let mut nu_l = nu_l_start;
                        for orbl in atoml.valorbs.iter() {
                            let s_grad: [f64; 3] = if atomk <= atoml {
                                let g = slako_transformation_gradients_fast(
                                    r, x, y, z, &s_cache, orbk.l, orbk.m, orbl.l, orbl.m,
                                );
                                [-g[0], -g[1], -g[2]]
                            } else {
                                let g = slako_transformation_gradients_fast(
                                    r, x, y, z, &s_cache, orbl.l, orbl.m, orbk.l, orbk.m,
                                );
                                g
                            };

                            let ds_d = s_grad[d];
                            dsv[nu_l] += ds_d * vp[[mu_k, nu_l]];
                            dsv[mu_k] += ds_d * vp[[nu_l, mu_k]];

                            nu_l += 1;
                        }
                        mu_k += 1;
                    }
                }

                // === Compute dgsv: for ALL atom pairs (gamma has no cutoff) ===
                for l in 0..self.n_atoms {
                    if k == l {
                        continue;
                    }

                    let atoml = &self.atoms[l];
                    let nu_l_start = orbital_offsets[l];

                    let r_vec = atomk - atoml;
                    let r_kl = r_vec.norm();
                    let e_kl = r_vec / r_kl;
                    let e_d = match d {
                        0 => e_kl.x,
                        1 => e_kl.y,
                        _ => e_kl.z,
                    };

                    let mut mu_k = mu_k_start;
                    for orbk in atomk.valorbs.iter() {
                        let mut nu_l = nu_l_start;
                        for orbl in atoml.valorbs.iter() {
                            let gamma_deriv = self.gammafunction.deriv_shell_resolved(
                                r_kl,
                                atomk.number,
                                atoml.number,
                                orbk.l as u8,
                                orbl.l as u8,
                            );

                            let dg_d = e_d * gamma_deriv;
                            dgsv[mu_k] += dg_d * sv[nu_l];
                            dgsv[nu_l] += dg_d * sv[mu_k];

                            nu_l += 1;
                        }
                        mu_k += 1;
                    }
                }

                // Compute gdsv = gamma_ao · dsv
                let gdsv: Array1<f64> = gamma_ao.dot(&dsv);

                // Part B contribution
                0.25 * (&dgsv + &gdsv).dot(&sdp)
            })
            .collect();

        // Add Part B contributions to gradient
        for (nc, contrib) in part_b_contributions.into_iter().enumerate() {
            gradient[nc] += contrib;
        }

        // === LC gamma contribution (on-the-fly) ===
        if let Some(ref coeff_g) = coeff_g_lr {
            let gamma_lc = self.gammafunction_lc.as_ref().unwrap();

            for i in 0..self.n_atoms {
                let atomi = &self.atoms[i];
                let mu_start: usize = self.atoms[..i].iter().map(|a| a.n_orbs).sum();
                // let n_orbs_i = atomi.n_orbs;

                for j in (i + 1)..self.n_atoms {
                    let atomj = &self.atoms[j];
                    let r_vec = atomi - atomj;
                    let r_ij = r_vec.norm();
                    let e_ij = r_vec / r_ij;

                    let nu_start: usize = self.atoms[..j].iter().map(|a| a.n_orbs).sum();
                    // let n_orbs_j = atomj.n_orbs;

                    // For shell-resolved LC, we need orbital-specific gamma derivatives
                    let mut contrib_sum = 0.0;
                    let mut mu = mu_start;
                    for orbi in atomi.valorbs.iter() {
                        let mut nu = nu_start;
                        for orbj in atomj.valorbs.iter() {
                            let gamma_lr_deriv = gamma_lc.deriv_shell_resolved(
                                r_ij,
                                atomi.number,
                                atomj.number,
                                orbi.l as u8,
                                orbj.l as u8,
                            );
                            contrib_sum += gamma_lr_deriv * coeff_g[[mu, nu]];
                            nu += 1;
                        }
                        mu += 1;
                    }

                    let contrib = -0.0625 * contrib_sum;
                    gradient[3 * i + 0] += e_ij.x * contrib;
                    gradient[3 * i + 1] += e_ij.y * contrib;
                    gradient[3 * i + 2] += e_ij.z * contrib;
                    gradient[3 * j + 0] -= e_ij.x * contrib;
                    gradient[3 * j + 1] -= e_ij.y * contrib;
                    gradient[3 * j + 2] -= e_ij.z * contrib;
                }
            }
        }

        // Repulsive potential gradient
        gradient = gradient + gradient_v_rep(&self.atoms, &self.vrep);

        // Dispersion gradient
        if self.config.dispersion.use_dispersion {
            gradient = gradient + gradient_disp(&self.atoms, &self.config);
        }

        gradient
    }

    /// Serial on-the-fly gradient calculation for atomwise gamma
    /// Same as ground_state_gradient_onthefly but without parallel execution
    pub fn ground_state_gradient_onthefly_serial(&mut self) -> Array1<f64> {
        // Get references to SCC results
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // Compute energy-weighted density matrix W = 0.5 * P * H_coul * P
        let h_coul: ArrayView2<f64> = self.properties.h_coul_x().unwrap();
        let w: Array2<f64> = 0.5 * p.dot(&h_coul).dot(&p);

        // Coulomb matrix: 0.5 * sum_C (gamma_AC + gamma_BC) * dq_C
        let coulomb_mat: Array2<f64> =
            atomvec_to_aomat(gamma.dot(&dq).view(), self.n_orbs, &self.atoms) * 0.5;

        // Effective matrix for dS term: -W + coulomb * P
        let effective_mat: Array2<f64> = &(-&w) + &(&coulomb_mat * &p);

        let n_grad = 3 * self.n_atoms;

        // Precompute LC coefficients if long-range correction is enabled
        let (coeff_s_lr, coeff_g_lr, diff_p): (
            Option<Array2<f64>>,
            Option<Array2<f64>>,
            Option<Array2<f64>>,
        ) = if self.config.lc.long_range_correction {
            let diff_p_val: Array2<f64> = &p - &self.properties.p_ref().unwrap();
            let gamma_lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
            let (cs, cg) = compute_lr_coefficients_onthefly(diff_p_val.view(), s, gamma_lr_ao);
            (Some(cs), Some(cg), Some(diff_p_val))
        } else {
            (None, None, None)
        };

        // Pre-compute orbital index offsets for each atom
        let mut orbital_offsets: Vec<usize> = Vec::with_capacity(self.n_atoms + 1);
        orbital_offsets.push(0);
        for atom in &self.atoms {
            orbital_offsets.push(orbital_offsets.last().unwrap() + atom.n_orbs);
        }

        // Serial H0/S gradient computation
        let mut gradient: Array1<f64> = Array1::zeros(n_grad);

        for i in 0..self.n_atoms {
            let atomi = &self.atoms[i];
            let mu_start = orbital_offsets[i];

            for j in (i + 1)..self.n_atoms {
                let atomj = &self.atoms[j];
                let distance = (atomi - atomj).norm();
                if distance >= PROXIMITY_CUTOFF {
                    continue;
                }

                let nu_start = orbital_offsets[j];

                // Directional cosines
                let (r, x, y, z): (f64, f64, f64, f64) = if atomi <= atomj {
                    directional_cosines(&atomi.xyz, &atomj.xyz)
                } else {
                    directional_cosines(&atomj.xyz, &atomi.xyz)
                };

                // Pre-compute spline caches
                let skt = self.slako.get(atomi.kind, atomj.kind);
                let s_cache = SplineCache::new(r, &skt.s_spline);
                let h_cache = SplineCache::new(r, &skt.h_spline);

                let mut mu = mu_start;
                for orbi in atomi.valorbs.iter() {
                    let mut nu = nu_start;
                    for orbj in atomj.valorbs.iter() {
                        let (s_deriv_i, h0_deriv_i, s_deriv_j, h0_deriv_j): (
                            [f64; 3],
                            [f64; 3],
                            [f64; 3],
                            [f64; 3],
                        ) = if atomi <= atomj {
                            let s_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &s_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                            );
                            let h0_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &h_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                            );
                            (
                                [-s_grad[0], -s_grad[1], -s_grad[2]],
                                [-h0_grad[0], -h0_grad[1], -h0_grad[2]],
                                s_grad,
                                h0_grad,
                            )
                        } else {
                            let s_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &s_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                            );
                            let h0_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &h_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                            );
                            (
                                s_grad,
                                h0_grad,
                                [-s_grad[0], -s_grad[1], -s_grad[2]],
                                [-h0_grad[0], -h0_grad[1], -h0_grad[2]],
                            )
                        };

                        let p_mu_nu = p[[mu, nu]];
                        let eff_mu_nu = effective_mat[[mu, nu]];

                        // H0/S contribution
                        for dir in 0..3 {
                            gradient[3 * i + dir] +=
                                2.0 * (h0_deriv_i[dir] * p_mu_nu + s_deriv_i[dir] * eff_mu_nu);
                            gradient[3 * j + dir] +=
                                2.0 * (h0_deriv_j[dir] * p_mu_nu + s_deriv_j[dir] * eff_mu_nu);
                        }

                        // LC overlap contribution
                        if let Some(ref coeff_s) = coeff_s_lr {
                            let coeff_mu_nu = coeff_s[[mu, nu]];
                            for dir in 0..3 {
                                gradient[3 * i + dir] -= 0.0625 * s_deriv_i[dir] * coeff_mu_nu;
                                gradient[3 * j + dir] -= 0.0625 * s_deriv_j[dir] * coeff_mu_nu;
                            }
                        }

                        nu += 1;
                    }
                    mu += 1;
                }
            }
        }

        // On-the-fly gamma gradient (standard SCC contribution)
        for i in 0..self.n_atoms {
            let atomi = &self.atoms[i];
            for j in (i + 1)..self.n_atoms {
                let atomj = &self.atoms[j];
                let r_vec = atomi - atomj;
                let r_ij = r_vec.norm();
                let e_ij = r_vec / r_ij;
                let gamma_deriv = self.gammafunction.deriv(r_ij, atomi.number, atomj.number);
                let contrib = gamma_deriv * dq[i] * dq[j];
                gradient[3 * i + 0] += e_ij.x * contrib;
                gradient[3 * i + 1] += e_ij.y * contrib;
                gradient[3 * i + 2] += e_ij.z * contrib;
                gradient[3 * j + 0] -= e_ij.x * contrib;
                gradient[3 * j + 1] -= e_ij.y * contrib;
                gradient[3 * j + 2] -= e_ij.z * contrib;
            }
        }

        // On-the-fly LC gamma_lr_ao gradient contribution
        if let (Some(ref coeff_g), Some(_)) = (&coeff_g_lr, &diff_p) {
            let gamma_lc = self.gammafunction_lc.as_ref().unwrap();
            for i in 0..self.n_atoms {
                let atomi = &self.atoms[i];
                let mu_start = orbital_offsets[i];
                let n_orbs_i = atomi.n_orbs;

                for j in (i + 1)..self.n_atoms {
                    let atomj = &self.atoms[j];
                    let r_vec = atomi - atomj;
                    let r_ij = r_vec.norm();
                    let e_ij = r_vec / r_ij;

                    let nu_start = orbital_offsets[j];
                    let n_orbs_j = atomj.n_orbs;

                    let gamma_lr_deriv = gamma_lc.deriv(r_ij, atomi.number, atomj.number);

                    let mut coeff_sum = 0.0;
                    for mu_off in 0..n_orbs_i {
                        for nu_off in 0..n_orbs_j {
                            coeff_sum += coeff_g[[mu_start + mu_off, nu_start + nu_off]];
                        }
                    }

                    let contrib = -0.0625 * gamma_lr_deriv * coeff_sum;
                    gradient[3 * i + 0] += e_ij.x * contrib;
                    gradient[3 * i + 1] += e_ij.y * contrib;
                    gradient[3 * i + 2] += e_ij.z * contrib;
                    gradient[3 * j + 0] -= e_ij.x * contrib;
                    gradient[3 * j + 1] -= e_ij.y * contrib;
                    gradient[3 * j + 2] -= e_ij.z * contrib;
                }
            }
        }

        // Repulsive potential gradient
        gradient = gradient + gradient_v_rep(&self.atoms, &self.vrep);

        // DFTB3 contributions (if enabled)
        if self.config.dftb3.use_dftb3 {
            let grad_s_for_dftb3: Array3<f64> =
                h0_and_s_gradients(&self.atoms, self.n_orbs, &self.slako).0;
            let grad_s_2d: ArrayView2<f64> = grad_s_for_dftb3
                .view()
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap();

            let gamma_third_order: ArrayView2<f64> = self.properties.gamma_third_order().unwrap();
            let coulomb_mat_third_order = construct_third_order_gradient_contribution(
                self.n_orbs,
                &self.atoms,
                gamma_third_order,
                dq,
            ) * 0.5;
            let coulomb_p_third_order: Array1<f64> = (&p * &coulomb_mat_third_order)
                .into_shape([self.n_orbs * self.n_orbs])
                .unwrap();
            gradient += &grad_s_2d.dot(&coulomb_p_third_order);

            let grad_gamma_third_order = gamma_third_order_derivative(
                &self.gammafunction,
                &self.atoms,
                self.n_atoms,
                &self.config.dftb3.hubbard_derivatives,
            );
            let mut contribution: Array1<f64> = Array1::zeros(gradient.raw_dim());
            let dq2: Array1<f64> = dq.map(|val| val.powi(2));
            for nc in 0..3 * self.n_atoms {
                let dgamma_slice: ArrayView2<f64> = grad_gamma_third_order.slice(s![nc, .., ..]);
                contribution[nc] = dq2.dot(&dgamma_slice.dot(&dq));
            }
            contribution /= 3.0;
            gradient += &contribution;
        }

        // Dispersion gradient
        if self.config.dispersion.use_dispersion {
            gradient = gradient + gradient_disp(&self.atoms, &self.config);
        }

        gradient
    }

    /// Serial on-the-fly gradient calculation for shell-resolved gamma
    /// Same as ground_state_gradient_shell_resolved_onthefly but without parallel execution
    pub fn ground_state_gradient_shell_resolved_onthefly_serial(&mut self) -> Array1<f64> {
        // Get references to SCC results
        let gamma_ao: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let dp: Array2<f64> = &p - &self.properties.p_ref().unwrap();

        // Compute energy-weighted density matrix W = 0.5 * P * H_coul * P
        let h_coul: ArrayView2<f64> = self.properties.h_coul_x().unwrap();
        let w: Array2<f64> = 0.5 * p.dot(&h_coul).dot(&p);

        let n_grad = 3 * self.n_atoms;

        // === Precompute quantities for f_v ===
        // vp = dp + dp^T (symmetrized difference density)
        let vp: Array2<f64> = &dp + &dp.t();

        // sv[a] = sum_b s[a,b] * vp[b]
        let sv: Array1<f64> = (&s * &vp).sum_axis(Axis(1));

        // gsv[a] = sum_b gamma_ao[a,b] * sv[b]
        let gsv: Array1<f64> = gamma_ao.dot(&sv);

        // sdp[a] = sum_b s[a,b] * dp[a,b]
        let sdp: Array1<f64> = (&s * &dp).sum_axis(Axis(1));

        // coeff_A[a,b] = (gsv[a] + gsv[b]) * dp[a,b]
        let mut coeff_a: Array2<f64> = Array2::zeros((self.n_orbs, self.n_orbs));
        for a in 0..self.n_orbs {
            for b in 0..self.n_orbs {
                coeff_a[[a, b]] = (gsv[a] + gsv[b]) * dp[[a, b]];
            }
        }

        // === Precompute LC coefficients if needed ===
        let (coeff_s_lr, coeff_g_lr): (Option<Array2<f64>>, Option<Array2<f64>>) =
            if self.config.lc.long_range_correction {
                let gamma_lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
                let (cs, cg) = compute_lr_coefficients_onthefly(dp.view(), s, gamma_lr_ao);
                (Some(cs), Some(cg))
            } else {
                (None, None)
            };

        // Pre-compute orbital index offsets for each atom
        let mut orbital_offsets: Vec<usize> = Vec::with_capacity(self.n_atoms + 1);
        orbital_offsets.push(0);
        for atom in &self.atoms {
            orbital_offsets.push(orbital_offsets.last().unwrap() + atom.n_orbs);
        }

        // === Serial H0/S gradient loop with Part A of f_v ===
        let mut gradient: Array1<f64> = Array1::zeros(n_grad);

        for i in 0..self.n_atoms {
            let atomi = &self.atoms[i];
            let mu_start = orbital_offsets[i];

            for j in (i + 1)..self.n_atoms {
                let atomj = &self.atoms[j];
                let distance = (atomi - atomj).norm();
                if distance >= PROXIMITY_CUTOFF {
                    continue;
                }

                let nu_start = orbital_offsets[j];

                // Directional cosines
                let (r, x, y, z): (f64, f64, f64, f64) = if atomi <= atomj {
                    directional_cosines(&atomi.xyz, &atomj.xyz)
                } else {
                    directional_cosines(&atomj.xyz, &atomi.xyz)
                };

                // Pre-compute spline caches
                let skt = self.slako.get(atomi.kind, atomj.kind);
                let s_cache = SplineCache::new(r, &skt.s_spline);
                let h_cache = SplineCache::new(r, &skt.h_spline);

                let mut mu = mu_start;
                for orbi in atomi.valorbs.iter() {
                    let mut nu = nu_start;
                    for orbj in atomj.valorbs.iter() {
                        let (s_deriv_i, h0_deriv_i, s_deriv_j, h0_deriv_j): (
                            [f64; 3],
                            [f64; 3],
                            [f64; 3],
                            [f64; 3],
                        ) = if atomi <= atomj {
                            let s_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &s_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                            );
                            let h0_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &h_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                            );
                            (
                                [-s_grad[0], -s_grad[1], -s_grad[2]],
                                [-h0_grad[0], -h0_grad[1], -h0_grad[2]],
                                s_grad,
                                h0_grad,
                            )
                        } else {
                            let s_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &s_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                            );
                            let h0_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &h_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                            );
                            (
                                s_grad,
                                h0_grad,
                                [-s_grad[0], -s_grad[1], -s_grad[2]],
                                [-h0_grad[0], -h0_grad[1], -h0_grad[2]],
                            )
                        };

                        let p_mu_nu = p[[mu, nu]];
                        let w_mu_nu = w[[mu, nu]];
                        let coeff_a_mu_nu = coeff_a[[mu, nu]];

                        // H0/S contribution: dH0*P - dS*W
                        for dir in 0..3 {
                            gradient[3 * i + dir] +=
                                2.0 * (h0_deriv_i[dir] * p_mu_nu - s_deriv_i[dir] * w_mu_nu);
                            gradient[3 * j + dir] +=
                                2.0 * (h0_deriv_j[dir] * p_mu_nu - s_deriv_j[dir] * w_mu_nu);
                        }

                        // f_v Part A: ds * (gsv + gsv) * dp
                        for dir in 0..3 {
                            gradient[3 * i + dir] += 0.25 * s_deriv_i[dir] * coeff_a_mu_nu;
                            gradient[3 * j + dir] += 0.25 * s_deriv_j[dir] * coeff_a_mu_nu;
                        }

                        // LC overlap contribution
                        if let Some(ref coeff_s) = coeff_s_lr {
                            let coeff_mu_nu = coeff_s[[mu, nu]];
                            for dir in 0..3 {
                                gradient[3 * i + dir] -= 0.0625 * s_deriv_i[dir] * coeff_mu_nu;
                                gradient[3 * j + dir] -= 0.0625 * s_deriv_j[dir] * coeff_mu_nu;
                            }
                        }

                        nu += 1;
                    }
                    mu += 1;
                }
            }
        }

        // === Part B of f_v: serial computation ===
        for k in 0..self.n_atoms {
            let atomk = &self.atoms[k];
            let mu_k_start = orbital_offsets[k];

            for d in 0..3 {
                let nc = 3 * k + d;

                // Initialize dsv and dgsv
                let mut dsv: Array1<f64> = Array1::zeros(self.n_orbs);
                let mut dgsv: Array1<f64> = Array1::zeros(self.n_orbs);

                // === Compute dsv: only for pairs within PROXIMITY_CUTOFF ===
                for l in 0..self.n_atoms {
                    if k == l {
                        continue;
                    }

                    let atoml = &self.atoms[l];
                    let distance = (atomk - atoml).norm();
                    if distance >= PROXIMITY_CUTOFF {
                        continue;
                    }

                    let nu_l_start = orbital_offsets[l];

                    let (r, x, y, z): (f64, f64, f64, f64) = if atomk <= atoml {
                        directional_cosines(&atomk.xyz, &atoml.xyz)
                    } else {
                        directional_cosines(&atoml.xyz, &atomk.xyz)
                    };

                    let skt = self.slako.get(atomk.kind, atoml.kind);
                    let s_cache = SplineCache::new(r, &skt.s_spline);

                    let mut mu_k = mu_k_start;
                    for orbk in atomk.valorbs.iter() {
                        let mut nu_l = nu_l_start;
                        for orbl in atoml.valorbs.iter() {
                            let s_grad: [f64; 3] = if atomk <= atoml {
                                let g = slako_transformation_gradients_fast(
                                    r, x, y, z, &s_cache, orbk.l, orbk.m, orbl.l, orbl.m,
                                );
                                [-g[0], -g[1], -g[2]]
                            } else {
                                slako_transformation_gradients_fast(
                                    r, x, y, z, &s_cache, orbl.l, orbl.m, orbk.l, orbk.m,
                                )
                            };

                            let ds_d = s_grad[d];
                            dsv[nu_l] += ds_d * vp[[mu_k, nu_l]];
                            dsv[mu_k] += ds_d * vp[[nu_l, mu_k]];

                            nu_l += 1;
                        }
                        mu_k += 1;
                    }
                }

                // === Compute dgsv: for ALL atom pairs (gamma has no cutoff) ===
                for l in 0..self.n_atoms {
                    if k == l {
                        continue;
                    }

                    let atoml = &self.atoms[l];
                    let nu_l_start = orbital_offsets[l];

                    let r_vec = atomk - atoml;
                    let r_kl = r_vec.norm();
                    let e_kl = r_vec / r_kl;
                    let e_d = match d {
                        0 => e_kl.x,
                        1 => e_kl.y,
                        _ => e_kl.z,
                    };

                    let mut mu_k = mu_k_start;
                    for orbk in atomk.valorbs.iter() {
                        let mut nu_l = nu_l_start;
                        for orbl in atoml.valorbs.iter() {
                            let gamma_deriv = self.gammafunction.deriv_shell_resolved(
                                r_kl,
                                atomk.number,
                                atoml.number,
                                orbk.l as u8,
                                orbl.l as u8,
                            );

                            let dg_d = e_d * gamma_deriv;
                            dgsv[mu_k] += dg_d * sv[nu_l];
                            dgsv[nu_l] += dg_d * sv[mu_k];

                            nu_l += 1;
                        }
                        mu_k += 1;
                    }
                }

                // Compute gdsv = gamma_ao · dsv
                let gdsv: Array1<f64> = gamma_ao.dot(&dsv);

                // Part B contribution
                let part_b = 0.25 * (&dgsv + &gdsv).dot(&sdp);
                gradient[nc] += part_b;
            }
        }

        // === LC gamma contribution ===
        if let Some(ref coeff_g) = coeff_g_lr {
            let gamma_lc = self.gammafunction_lc.as_ref().unwrap();

            for i in 0..self.n_atoms {
                let atomi = &self.atoms[i];
                let mu_start = orbital_offsets[i];

                for j in (i + 1)..self.n_atoms {
                    let atomj = &self.atoms[j];
                    let r_vec = atomi - atomj;
                    let r_ij = r_vec.norm();
                    let e_ij = r_vec / r_ij;

                    let nu_start = orbital_offsets[j];

                    let mut contrib_sum = 0.0;
                    let mut mu = mu_start;
                    for orbi in atomi.valorbs.iter() {
                        let mut nu = nu_start;
                        for orbj in atomj.valorbs.iter() {
                            let gamma_lr_deriv = gamma_lc.deriv_shell_resolved(
                                r_ij,
                                atomi.number,
                                atomj.number,
                                orbi.l as u8,
                                orbj.l as u8,
                            );
                            contrib_sum += gamma_lr_deriv * coeff_g[[mu, nu]];
                            nu += 1;
                        }
                        mu += 1;
                    }

                    let contrib = -0.0625 * contrib_sum;
                    gradient[3 * i + 0] += e_ij.x * contrib;
                    gradient[3 * i + 1] += e_ij.y * contrib;
                    gradient[3 * i + 2] += e_ij.z * contrib;
                    gradient[3 * j + 0] -= e_ij.x * contrib;
                    gradient[3 * j + 1] -= e_ij.y * contrib;
                    gradient[3 * j + 2] -= e_ij.z * contrib;
                }
            }
        }

        // Repulsive potential gradient
        gradient = gradient + gradient_v_rep(&self.atoms, &self.vrep);

        // Dispersion gradient
        if self.config.dispersion.use_dispersion {
            gradient = gradient + gradient_disp(&self.atoms, &self.config);
        }

        gradient
    }
}

#[macro_export]
macro_rules! impl_dftb_ground_state_gradient_on_the_fly {
    () => {
        pub fn ground_state_gradient_onthefly(
            &mut self,
            atoms: &[Atom],
            config: &Configuration,
        ) -> Array1<f64> {
            // Get references to SCC results
            let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
            let p: ArrayView2<f64> = self.properties.p().unwrap();
            let dq: ArrayView1<f64> = self.properties.dq().unwrap();
            let s: ArrayView2<f64> = self.properties.s().unwrap();

            // Compute energy-weighted density matrix W = 0.5 * P * H_coul * P
            let h_coul: ArrayView2<f64> = self.properties.h_coul_x().unwrap();
            let w: Array2<f64> = 0.5 * p.dot(&h_coul).dot(&p);

            // Coulomb matrix: 0.5 * sum_C (gamma_AC + gamma_BC) * dq_C
            let coulomb_mat: Array2<f64> =
                atomvec_to_aomat(gamma.dot(&dq).view(), self.n_orbs, &atoms) * 0.5;

            // Effective matrix for dS term: -W + coulomb * P
            let effective_mat: Array2<f64> = &(-&w) + &(&coulomb_mat * &p);

            let n_grad = 3 * self.n_atoms;

            // Precompute LC coefficients if long-range correction is enabled
            let (coeff_s_lr, coeff_g_lr, diff_p): (
                Option<Array2<f64>>,
                Option<Array2<f64>>,
                Option<Array2<f64>>,
            ) = if config.lc.long_range_correction {
                let diff_p_val: Array2<f64> = &p - &self.properties.p_ref().unwrap();
                let gamma_lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
                let (cs, cg) = compute_lr_coefficients_onthefly(diff_p_val.view(), s, gamma_lr_ao);
                (Some(cs), Some(cg), Some(diff_p_val))
            } else {
                (None, None, None)
            };

            // Pre-compute orbital index offsets for each atom
            let mut orbital_offsets: Vec<usize> = Vec::with_capacity(self.n_atoms + 1);
            orbital_offsets.push(0);
            for atom in atoms.iter() {
                orbital_offsets.push(orbital_offsets.last().unwrap() + atom.n_orbs);
            }

            // Serial H0/S gradient computation
            let mut gradient: Array1<f64> = Array1::zeros(n_grad);

            for i in 0..self.n_atoms {
                let atomi = &atoms[i];
                let mu_start = orbital_offsets[i];

                for j in (i + 1)..self.n_atoms {
                    let atomj = &atoms[j];
                    let distance = (atomi - atomj).norm();
                    if distance >= PROXIMITY_CUTOFF {
                        continue;
                    }

                    let nu_start = orbital_offsets[j];

                    // Directional cosines
                    let (r, x, y, z): (f64, f64, f64, f64) = if atomi <= atomj {
                        directional_cosines(&atomi.xyz, &atomj.xyz)
                    } else {
                        directional_cosines(&atomj.xyz, &atomi.xyz)
                    };

                    // Pre-compute spline caches
                    let skt = self.slako.get(atomi.kind, atomj.kind);
                    let s_cache = SplineCache::new(r, &skt.s_spline);
                    let h_cache = SplineCache::new(r, &skt.h_spline);

                    let mut mu = mu_start;
                    for orbi in atomi.valorbs.iter() {
                        let mut nu = nu_start;
                        for orbj in atomj.valorbs.iter() {
                            let (s_deriv_i, h0_deriv_i, s_deriv_j, h0_deriv_j): (
                                [f64; 3],
                                [f64; 3],
                                [f64; 3],
                                [f64; 3],
                            ) = if atomi <= atomj {
                                let s_grad = slako_transformation_gradients_fast(
                                    r, x, y, z, &s_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                                );
                                let h0_grad = slako_transformation_gradients_fast(
                                    r, x, y, z, &h_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                                );
                                (
                                    [-s_grad[0], -s_grad[1], -s_grad[2]],
                                    [-h0_grad[0], -h0_grad[1], -h0_grad[2]],
                                    s_grad,
                                    h0_grad,
                                )
                            } else {
                                let s_grad = slako_transformation_gradients_fast(
                                    r, x, y, z, &s_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                                );
                                let h0_grad = slako_transformation_gradients_fast(
                                    r, x, y, z, &h_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                                );
                                (
                                    s_grad,
                                    h0_grad,
                                    [-s_grad[0], -s_grad[1], -s_grad[2]],
                                    [-h0_grad[0], -h0_grad[1], -h0_grad[2]],
                                )
                            };

                            let p_mu_nu = p[[mu, nu]];
                            let eff_mu_nu = effective_mat[[mu, nu]];

                            // H0/S contribution
                            for dir in 0..3 {
                                gradient[3 * i + dir] +=
                                    2.0 * (h0_deriv_i[dir] * p_mu_nu + s_deriv_i[dir] * eff_mu_nu);
                                gradient[3 * j + dir] +=
                                    2.0 * (h0_deriv_j[dir] * p_mu_nu + s_deriv_j[dir] * eff_mu_nu);
                            }

                            // LC overlap contribution
                            if let Some(ref coeff_s) = coeff_s_lr {
                                let coeff_mu_nu = coeff_s[[mu, nu]];
                                for dir in 0..3 {
                                    gradient[3 * i + dir] -= 0.0625 * s_deriv_i[dir] * coeff_mu_nu;
                                    gradient[3 * j + dir] -= 0.0625 * s_deriv_j[dir] * coeff_mu_nu;
                                }
                            }

                            nu += 1;
                        }
                        mu += 1;
                    }
                }
            }

            // On-the-fly gamma gradient (standard SCC contribution)
            for i in 0..self.n_atoms {
                let atomi = &atoms[i];
                for j in (i + 1)..self.n_atoms {
                    let atomj = &atoms[j];
                    let r_vec = atomi - atomj;
                    let r_ij = r_vec.norm();
                    let e_ij = r_vec / r_ij;
                    let gamma_deriv = self.gammafunction.deriv(r_ij, atomi.number, atomj.number);
                    let contrib = gamma_deriv * dq[i] * dq[j];
                    gradient[3 * i + 0] += e_ij.x * contrib;
                    gradient[3 * i + 1] += e_ij.y * contrib;
                    gradient[3 * i + 2] += e_ij.z * contrib;
                    gradient[3 * j + 0] -= e_ij.x * contrib;
                    gradient[3 * j + 1] -= e_ij.y * contrib;
                    gradient[3 * j + 2] -= e_ij.z * contrib;
                }
            }

            // On-the-fly LC gamma_lr_ao gradient contribution
            if let (Some(ref coeff_g), Some(_)) = (&coeff_g_lr, &diff_p) {
                let gamma_lc = self.gammafunction_lc.as_ref().unwrap();
                for i in 0..self.n_atoms {
                    let atomi = &atoms[i];
                    let mu_start = orbital_offsets[i];
                    let n_orbs_i = atomi.n_orbs;

                    for j in (i + 1)..self.n_atoms {
                        let atomj = &atoms[j];
                        let r_vec = atomi - atomj;
                        let r_ij = r_vec.norm();
                        let e_ij = r_vec / r_ij;

                        let nu_start = orbital_offsets[j];
                        let n_orbs_j = atomj.n_orbs;

                        let gamma_lr_deriv = gamma_lc.deriv(r_ij, atomi.number, atomj.number);

                        let mut coeff_sum = 0.0;
                        for mu_off in 0..n_orbs_i {
                            for nu_off in 0..n_orbs_j {
                                coeff_sum += coeff_g[[mu_start + mu_off, nu_start + nu_off]];
                            }
                        }

                        let contrib = -0.0625 * gamma_lr_deriv * coeff_sum;
                        gradient[3 * i + 0] += e_ij.x * contrib;
                        gradient[3 * i + 1] += e_ij.y * contrib;
                        gradient[3 * i + 2] += e_ij.z * contrib;
                        gradient[3 * j + 0] -= e_ij.x * contrib;
                        gradient[3 * j + 1] -= e_ij.y * contrib;
                        gradient[3 * j + 2] -= e_ij.z * contrib;
                    }
                }
            }

            // Repulsive potential gradient
            gradient = gradient + gradient_v_rep(&atoms, &self.vrep);

            // DFTB3 contributions (if enabled)
            if config.dftb3.use_dftb3 {
                let grad_s_for_dftb3: Array3<f64> =
                    h0_and_s_gradients(&atoms, self.n_orbs, &self.slako).0;
                let grad_s_2d: ArrayView2<f64> = grad_s_for_dftb3
                    .view()
                    .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                    .unwrap();

                let gamma_third_order: ArrayView2<f64> =
                    self.properties.gamma_third_order().unwrap();
                let coulomb_mat_third_order = construct_third_order_gradient_contribution(
                    self.n_orbs,
                    &atoms,
                    gamma_third_order,
                    dq,
                ) * 0.5;
                let coulomb_p_third_order: Array1<f64> = (&p * &coulomb_mat_third_order)
                    .into_shape([self.n_orbs * self.n_orbs])
                    .unwrap();
                gradient += &grad_s_2d.dot(&coulomb_p_third_order);

                let grad_gamma_third_order = gamma_third_order_derivative(
                    &self.gammafunction,
                    &atoms,
                    self.n_atoms,
                    &config.dftb3.hubbard_derivatives,
                );
                let mut contribution: Array1<f64> = Array1::zeros(gradient.raw_dim());
                let dq2: Array1<f64> = dq.map(|val| val.powi(2));
                for nc in 0..3 * self.n_atoms {
                    let dgamma_slice: ArrayView2<f64> =
                        grad_gamma_third_order.slice(s![nc, .., ..]);
                    contribution[nc] = dq2.dot(&dgamma_slice.dot(&dq));
                }
                contribution /= 3.0;
                gradient += &contribution;
            }

            // Dispersion gradient
            if config.dispersion.use_dispersion {
                gradient = gradient + gradient_disp(&atoms, &config);
            }

            gradient
        }
    };
}

impl Monomer<'_> {
    impl_dftb_ground_state_gradient_on_the_fly!();
}

impl Pair<'_> {
    impl_dftb_ground_state_gradient_on_the_fly!();
}

#[cfg(test)]
mod tests {
    use crate::initialization::System;
    use crate::properties::Properties;
    use crate::scc::scc_routine::RestrictedSCC;
    use crate::utils::tests::{get_molecule, get_molecule_no_lc, AVAILAIBLE_MOLECULES};
    use ndarray::prelude::*;

    pub const EPSILON: f64 = 1e-10;

    fn test_gs_gradient(molecule_and_properties: (&str, System, Properties), lc: bool) {
        let name = molecule_and_properties.0;
        let mut molecule = molecule_and_properties.1;
        let props = molecule_and_properties.2;

        // perform scc routine
        molecule.prepare_scc();
        molecule.run_scc().unwrap();
        let grad: Array1<f64> = molecule.ground_state_gradient(false);
        let grad_ref: Array1<f64> = if lc {
            props
                .get("gs_gradient_lc")
                .unwrap()
                .as_array1()
                .unwrap()
                .to_owned()
        } else {
            props
                .get("gs_gradient_no_lc")
                .unwrap()
                .as_array1()
                .unwrap()
                .to_owned()
        };
        assert!(
            grad.abs_diff_eq(&grad_ref, EPSILON),
            "Molecule: {}, Grad ref {:.15}, Grad calc: {:.15}",
            name,
            grad_ref,
            grad
        );
    }

    #[test]
    fn get_gs_gradient() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_gs_gradient(get_molecule(molecule), true);
        }
    }

    #[test]
    fn get_gs_gradient_no_lc() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_gs_gradient(get_molecule_no_lc(molecule), false);
        }
    }
}
