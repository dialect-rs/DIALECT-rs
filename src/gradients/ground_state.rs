use crate::fmo::scc::helpers::atomvec_to_aomat;
use crate::gradients::dispersion::gradient_disp;
use crate::gradients::helpers::{f_lr_par, f_v_par, gradient_v_rep};
use crate::initialization::*;
use crate::scc::construct_third_order_gradient_contribution;
use crate::scc::gamma_approximation::{
    gamma_gradients_ao_wise, gamma_gradients_ao_wise_shell_resolved, gamma_gradients_atomwise,
    gamma_third_order_derivative,
};
use crate::scc::h0_and_s::h0_and_s_gradients;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use ndarray_npy::write_npy;

impl System {
    pub fn ground_state_gradient(&mut self, excited_gradients: bool) -> Array1<f64> {
        let gradient: Array1<f64> = if !self.config.use_shell_resolved_gamma {
            self.ground_state_gradient_atomwise(excited_gradients)
        } else {
            self.ground_state_gradient_shell_resolved(excited_gradients)
        };
        if self.config.jobtype == "grad" {
            // save the gradient
            write_npy("gs_gradient.npy", &gradient).unwrap();
        }
        gradient
    }

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
        // let w: Array1<f64> = 0.5
        //     * (p.dot(&(&h0 + &(&coulomb_mat * &s))).dot(&p))
        //         .into_shape([self.n_orbs * self.n_orbs])
        //         .unwrap();
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
            // let mut contribution: Array1<f64> = Array1::zeros(gradient.raw_dim());
            // for nc in 0..3 {
            //     for (idx, atom_a) in self.atoms.iter().enumerate() {
            //         let grad_idx: usize = 3 * idx + nc;
            //         let s_contrib: ArrayView2<f64> = grad_s.slice(s![grad_idx, .., ..]);
            //         let gamma_contrib: Array2<f64> =
            //             construct_third_order_gradient_contribution_test(
            //                 self.n_orbs,
            //                 &self.atoms,
            //                 gamma_third_order,
            //                 dq,
            //                 atom_a,
            //                 idx,
            //             );
            //         let temp_grad: f64 = s_contrib.dot(&(&gamma_contrib * &p).t()).trace().unwrap();
            //         contribution[grad_idx] += temp_grad;
            //     }
            // }
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
            // for nc in 0..3 {
            //     for (idx, atom_a) in self.atoms.iter().enumerate() {
            //         let grad_idx: usize = 3 * idx + nc;
            //         for (idx2, atom_b) in self.atoms.iter().enumerate() {
            //             if idx2 != idx {
            //                 contribution[grad_idx] += dq[idx]
            //                     * dq[idx2]
            //                     * (grad_gamma_third_order[[grad_idx, idx2, idx]] * dq[idx2]
            //                         + grad_gamma_third_order[[grad_idx, idx, idx2]] * dq[idx]);
            //             }
            //         }
            //     }
            // }
            for nc in 0..3 * self.n_atoms {
                let dgamma_slice: ArrayView2<f64> = grad_gamma_third_order.slice(s![nc, .., ..]);
                contribution[nc] = dq2.dot(&dgamma_slice.dot(&dq));
            }
            contribution /= 3.0;

            gradient += &contribution;
        }

        // dispersion
        if self.config.dispersion.use_dispersion {
            gradient = gradient + gradient_disp(&self.atoms, &self.config.dispersion);
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
            gradient = gradient + gradient_disp(&self.atoms, &self.config.dispersion);
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
