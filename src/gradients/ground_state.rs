use crate::defaults;
use crate::fmo::scc::helpers::atomvec_to_aomat;
use crate::gradients::helpers::{f_lr, gradient_v_rep};
use crate::initialization::*;
use crate::scc::gamma_approximation::{gamma_gradients_ao_wise, gamma_gradients_atomwise};
use crate::scc::h0_and_s::h0_and_s_gradients;
use ndarray::{s, Array, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use ndarray_einsum_beta::tensordot;
use std::time::Instant;

impl System {
    pub fn ground_state_gradient(&mut self) -> Array1<f64> {
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
        let h0: ArrayView2<f64> = self.properties.h0().unwrap();
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // transform the expression Sum_c_in_X (gamma_AC + gamma_aC) * dq_C
        // into matrix of the dimension (norb, norb) to do an element wise multiplication with P
        let mut coulomb_mat: Array2<f64> =
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
        let dq_column: ArrayView2<f64> = dq.clone().insert_axis(Axis(1));
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

        // long-range corrected part of the gradient
        if self.config.lc.long_range_correction {
            let (g1_lr, g1_lr_ao): (Array3<f64>, Array3<f64>) = gamma_gradients_ao_wise(
                self.gammafunction_lc.as_ref().unwrap(),
                &self.atoms,
                self.n_atoms,
                self.n_orbs,
            );

            let diff_p: Array2<f64> = &p - &self.properties.p_ref().unwrap();
            let flr_dmd0: Array3<f64> = f_lr(
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
        }

        return gradient;
    }
}
