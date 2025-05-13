use crate::xtb::gradients::hamiltonian::calculate_h0_gradient_xtb1_new;
use crate::xtb::integrals::calc_overlap_matrix_obs_derivs_new;
use crate::xtb::scc::gamma_matrix::gamma_gradient_xtb_new;
use crate::{
    fmo::scc::helpers::aovec_to_aomat,
    xtb::{
        gradients::helpers::{coul_third_order_grad_contribution_xtb, gradient_disp3_xtb},
        initialization::system::XtbSystem,
        parameters::COUL_THIRD_ORDER_ATOM,
    },
};
use ndarray::prelude::*;
use ndarray_npy::write_npy;

impl XtbSystem {
    pub fn ground_state_gradient(&mut self) -> Array1<f64> {
        // take references/views to the necessary properties from the scc calculation
        let gamma: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        let dq_ao: ArrayView1<f64> = self.properties.dq_ao().unwrap();
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // calculate the gradient of the overlap matrix
        let grad_s: Array3<f64> = calc_overlap_matrix_obs_derivs_new(&self.basis, self.n_atoms);
        // calculate the gradient of the H0 matrix
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

        // calculation of the gradient
        // 1st part:  dH0 / dR . P
        let mut gradient: Array1<f64> = grad_h0_2d.dot(&p_flat);

        // 2nd part: dS / dR . W
        gradient -= &grad_s_2d.dot(&w);

        // 3rd part: second order Coulomb gradient part 1
        gradient -= &grad_s_2d.dot(&coulomb_x_p);

        // 4th part: second order Coulomb gradient part 2
        gradient += &(0.5 * grad_gamma_2d.dot(&dq_x_dq));

        // 5th part: third order Coulomb gradient
        gradient -= &grad_s_2d.dot(&coulomb_p_third_order);

        // last part: dV_rep / dR
        gradient = gradient + self.grad_repulsive_energy();

        // dispersion
        gradient = gradient + gradient_disp3_xtb(&self.atoms, &self.config.dispersion);

        if self.config.jobtype == "grad" {
            // save the gradient
            write_npy("gs_gradient.npy", &gradient).unwrap();
        }

        gradient
    }
}

// // For alternative way to calculate 3rd and 4th grad contributions
// let f_term: Array2<f64> = f_v_par(
//     dp.view(),
//     s,
//     grad_s.view(),
//     gamma,
//     grad_gamma.view(),
//     self.n_atoms,
//     self.n_orbs,
// )
// .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
// .unwrap();

// // Alternative way to calculate 3rd part + 4th part
// gradient = gradient + 0.5 * f_term.dot(&dp.into_shape(self.n_orbs * self.n_orbs).unwrap());

// // Other way to calculate third order gradient
// let mut grad_third_order: Array1<f64> = Array1::zeros(3 * self.n_atoms);
// // third order part
// for idx in 0..self.n_atoms {
//     for nc in 0..3 {
//         let grad_idx: usize = 3 * idx + nc;
//         // iterate over other atoms than idx
//         for atom_a in 0..self.n_atoms {
//             let gamma_a: f64 = hubbard_derivatives[atom_a] * dq[atom_a].powi(2);
//             if atom_a != idx {
//                 for shell in self.basis.shells.iter() {
//                     if shell.atom_index == atom_a {
//                         for shell_idx in (shell.sph_start..shell.sph_end) {
//                             for shell2 in self.basis.shells.iter() {
//                                 if shell2.atom_index == idx {
//                                     for shell_idx_2 in (shell2.sph_start..shell2.sph_end) {
//                                         grad_third_order[grad_idx] += gamma_a
//                                             * p[[shell_idx, shell_idx_2]]
//                                             * grad_s[[grad_idx, shell_idx, shell_idx_2]];
//                                     }
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//
//         let gamma_k: f64 = hubbard_derivatives[idx] * dq[idx].powi(2);
//         for shell in self.basis.shells.iter() {
//             if shell.atom_index == idx {
//                 for shell_idx in (shell.sph_start..shell.sph_end) {
//                     for shell2 in self.basis.shells.iter() {
//                         if shell2.atom_index != idx {
//                             for shell_idx_2 in (shell2.sph_start..shell2.sph_end) {
//                                 grad_third_order[grad_idx] += gamma_k
//                                     * p[[shell_idx, shell_idx_2]]
//                                     * grad_s[[grad_idx, shell_idx, shell_idx_2]];
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }
