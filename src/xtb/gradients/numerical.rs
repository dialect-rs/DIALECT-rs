use super::hamiltonian::{calculate_coordination_number_gradients, calculate_h0_gradient_xtb1_new};
use crate::fmo::scc::helpers::aovec_to_aomat;
use crate::xtb::gradients::helpers::coul_third_order_grad_contribution_xtb;
use crate::xtb::integrals::calc_overlap_matrix_obs_derivs_new;
use crate::xtb::scc::gamma_matrix::gamma_gradient_xtb_new;
use crate::{
    gradients::numerical::assert_deriv,
    scc::scc_routine::RestrictedSCC,
    xtb::{
        initialization::system::XtbSystem, integrals::calc_overlap_matrix_obs_derivs,
        parameters::COUL_THIRD_ORDER_ATOM, scc::hamiltonian::calculate_coordination_numbers,
    },
};
use ndarray::prelude::*;

impl XtbSystem {
    pub fn test_gs_gradient(&mut self) {
        let xyz = self.get_xyz();
        assert_deriv(
            self,
            XtbSystem::gs_energy_wrapper,
            XtbSystem::gs_gradient_wrapper,
            self.get_xyz(),
            0.001,
            1e-6,
        );
        self.update_xyz(xyz.view());
    }

    pub fn gs_gradient_wrapper(&mut self) -> Array1<f64> {
        self.properties.reset_reduced();
        self.prepare_scc();
        let _ = self.run_scc().unwrap();
        self.ground_state_gradient()
    }

    pub fn gs_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset_reduced();
        self.update_xyz(geometry.view());
        self.prepare_scc();
        self.run_scc().unwrap()
    }

    pub fn test_gs_band_energy_gradient(&mut self) {
        assert_deriv(
            self,
            XtbSystem::gs_band_energy_wrapper,
            XtbSystem::gs_gradient_wrapper,
            self.get_xyz(),
            0.001,
            1e-6,
        );
    }

    pub fn gs_band_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        self.update_xyz(geometry.view());
        self.prepare_scc();
        self.run_scc().unwrap();
        let h0: ArrayView2<f64> = self.properties.h0().unwrap();
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        (&p * &h0).sum()
    }

    pub fn test_gs_coul_energy_gradient(&mut self) {
        assert_deriv(
            self,
            XtbSystem::gs_coul_energy_wrapper,
            XtbSystem::gs_coul_energy_wrapper_test,
            self.get_xyz(),
            0.001,
            1e-6,
        );
    }

    pub fn gs_coul_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset_reduced();
        self.update_xyz(geometry.view());
        self.prepare_scc();
        self.run_scc().unwrap();
        let gamma: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let dq_ao: ArrayView1<f64> = self.properties.dq_ao().unwrap();
        0.5 * &dq_ao.dot(&gamma.dot(&dq_ao))
    }

    pub fn gs_coul_energy_wrapper_test(&mut self) -> Array1<f64> {
        self.properties.reset_reduced();
        self.prepare_scc();
        self.run_scc().unwrap();
        // take references/views to the necessary properties from the scc calculation
        let gamma: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        let dq_ao: ArrayView1<f64> = self.properties.dq_ao().unwrap();

        // calculate the gradient of the overlap matrix
        let grad_s: Array3<f64> = calc_overlap_matrix_obs_derivs_new(&self.basis, self.n_atoms);
        let grad_s_2d: ArrayView2<f64> = grad_s
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
        let dq_column: ArrayView2<f64> = dq_ao.clone().insert_axis(Axis(1));
        let dq_x_dq: Array1<f64> = (&dq_column.broadcast((self.n_orbs, self.n_orbs)).unwrap()
            * &dq_ao)
            .into_shape([self.n_orbs * self.n_orbs])
            .unwrap();
        let coulomb_mat: Array2<f64> = aovec_to_aomat(gamma.dot(&dq_ao).view(), self.n_orbs) * 0.5;
        let coulomb_x_p: Array1<f64> = (&p * &coulomb_mat)
            .into_shape([self.n_orbs * self.n_orbs])
            .unwrap();

        // 3rd part: 1/2 * dS / dR * sum_c_in_X (gamma_ac + gamma_bc) * dq_c
        let mut gradient = -grad_s_2d.dot(&coulomb_x_p);

        // 4th part: 1/2 * dq . dGamma / dR . dq
        gradient += &(0.5 * grad_gamma_2d.dot(&dq_x_dq));
        gradient
    }

    pub fn test_gs_coul_third_energy_gradient(&mut self) {
        assert_deriv(
            self,
            XtbSystem::gs_coul_third_energy_wrapper,
            XtbSystem::gs_coul_third_energy_wrapper_test,
            self.get_xyz(),
            0.001,
            1e-6,
        );
    }

    pub fn gs_coul_third_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        self.update_xyz(geometry.view());
        self.prepare_scc();
        self.run_scc().unwrap();
        let mut hubbard_derivatives: Array1<f64> = Array1::zeros(self.n_atoms);
        for (mut val, atom) in hubbard_derivatives.iter_mut().zip(self.atoms.iter()) {
            *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
        }

        let dq: ArrayView1<f64> = self.properties.dq().unwrap();
        1.0 / 3.0 * dq.map(|val| val.powi(3)).dot(&hubbard_derivatives)
    }

    pub fn gs_coul_third_energy_wrapper_test(&mut self) -> Array1<f64> {
        self.properties.reset();
        self.prepare_scc();
        self.run_scc().unwrap();
        let mut hubbard_derivatives: Array1<f64> = Array1::zeros(self.n_atoms);
        for (mut val, atom) in hubbard_derivatives.iter_mut().zip(self.atoms.iter()) {
            *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
        }
        // take references/views to the necessary properties from the scc calculation
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();

        // calculate the gradient of the overlap matrix
        let grad_s: Array3<f64> = calc_overlap_matrix_obs_derivs_new(&self.basis, self.n_atoms);
        let grad_s_2d: ArrayView2<f64> = grad_s
            .view()
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap();

        // calculate the gradient contribution of the third order energy
        // contribution of dq**2 and gamma third order
        let dq2_gamma: Array2<f64> =
            coul_third_order_grad_contribution_xtb(&self.basis, dq, hubbard_derivatives.view());
        // multiply with the density matrix
        let coulomb_p_third_order: Array1<f64> = 0.5
            * (&p * &dq2_gamma)
                .into_shape([self.n_orbs * self.n_orbs])
                .unwrap();

        grad_s_2d.dot(&coulomb_p_third_order)
    }

    pub fn test_repulsive_gradient(&mut self) {
        assert_deriv(
            self,
            XtbSystem::rep_energy_wrapper,
            XtbSystem::rep_gradient_wrapper,
            self.get_xyz(),
            0.001,
            1e-6,
        );
    }

    pub fn rep_gradient_wrapper(&mut self) -> Array1<f64> {
        self.properties.reset();
        self.grad_repulsive_energy()
    }

    pub fn rep_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        self.update_xyz(geometry.view());
        self.calculate_repulsive_energy()
    }

    pub fn test_cn_numbers_gradient(&mut self) {
        self.properties.reset();
        let analytical: Array2<f64> = self.analytical_cn_numbers_gradient();
        let numerical: Array2<f64> = self.numerical_cn_numbers_gradient();
        println!("Analytical cn numbers: \n{:.7}", analytical);
        println!("Numerical cn numbers: \n{:.7}", numerical);
    }

    fn analytical_cn_numbers_gradient(&mut self) -> Array2<f64> {
        self.properties.reset();
        let grad: Array2<f64> = calculate_coordination_number_gradients(&self.atoms);
        grad
    }

    fn numerical_cn_numbers_gradient(&mut self) -> Array2<f64> {
        self.properties.reset();
        let coords: Array1<f64> = self.get_xyz();
        let mut cn_derivs: Array2<f64> = Array2::zeros([3 * self.n_atoms, self.n_atoms]);

        let stepsize: f64 = 1.0e-4;
        for index in 0..(3 * self.n_atoms) {
            let mut step: Array1<f64> = Array1::zeros([3 * self.n_atoms]);
            step[index] = 1.0;
            let geom_1: Array1<f64> = coords.clone() + stepsize * &step;
            let geom_2: Array1<f64> = coords.clone() - stepsize * &step;

            self.properties.reset();
            self.update_xyz(geom_1.view());
            let coordination_numbers1: Array1<f64> = calculate_coordination_numbers(&self.atoms);

            self.properties.reset();
            self.update_xyz(geom_2.view());
            let coordination_numbers2: Array1<f64> = calculate_coordination_numbers(&self.atoms);

            let numerical_deriv: Array1<f64> =
                (&coordination_numbers1 - &coordination_numbers2) / (2.0 * stepsize);
            cn_derivs.slice_mut(s![index, ..]).assign(&numerical_deriv);
        }
        cn_derivs
    }

    pub fn test_h0_gradient(&mut self) {
        self.properties.reset();
        self.prepare_scc();
        // let _ = self.run_scc().unwrap();
        // let p: Array2<f64> = self.properties.p().unwrap().to_owned();

        let analytical: Array3<f64> = self.test_analytical_h0_gradient();
        let numerical: Array3<f64> = self.numerical_h0_gradient_wrapper();
        let diff: Array3<f64> = &analytical - &numerical;
        let sum: f64 = diff.map(|val| val.abs()).sum();

        println!("Analytical: \n {:.6}", analytical.slice(s![2, .., ..]));
        println!("Numerical: \n {:.6}", numerical.slice(s![2, .., ..]));
        println!("Difference: \n {:.7}", diff);
        println!("Sum of absolute difference: {:.10}", sum);

        // // The density matrix in vector form.
        // let p_flat: Array1<f64> = p.into_shape([self.n_orbs * self.n_orbs]).unwrap();
        // let grad_h0_2d: ArrayView2<f64> = numerical
        //     .view()
        //     .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
        //     .unwrap();
        // let test: Array1<f64> = grad_h0_2d.dot(&p_flat);
        // println!("Test arr: {:.6}", test);
    }

    fn test_analytical_h0_gradient(&mut self) -> Array3<f64> {
        self.properties.reset();
        self.prepare_scc();
        let grad_s: Array3<f64> = calc_overlap_matrix_obs_derivs_new(&self.basis, self.n_atoms);
        let grad_h0: Array3<f64> = calculate_h0_gradient_xtb1_new(
            self.n_orbs,
            &self.atoms,
            self.properties.s().unwrap(),
            grad_s.view(),
            &self.basis,
        );
        grad_h0
    }

    fn numerical_h0_gradient_wrapper(&mut self) -> Array3<f64> {
        self.properties.reset();
        self.prepare_scc();
        let coords: Array1<f64> = self.get_xyz();
        let mut h_deriv: Array3<f64> = Array3::zeros([3 * self.n_atoms, self.n_orbs, self.n_orbs]);

        let stepsize: f64 = 1.0e-4;
        for index in 0..(3 * self.n_atoms) {
            let mut step: Array1<f64> = Array1::zeros([3 * self.n_atoms]);
            step[index] = 1.0;
            let geom_1: Array1<f64> = coords.clone() + stepsize * &step;
            let geom_2: Array1<f64> = coords.clone() - stepsize * &step;

            self.properties.reset();
            self.update_xyz(geom_1.view());
            self.prepare_scc();
            let h_1: Array2<f64> = self.properties.h0().unwrap().to_owned();

            self.properties.reset();
            self.update_xyz(geom_2.view());
            self.prepare_scc();

            let h_2: ArrayView2<f64> = self.properties.h0().unwrap();
            let numerical_deriv: Array2<f64> = (&h_1 - &h_2) / (2.0 * stepsize);

            h_deriv
                .slice_mut(s![index, .., ..])
                .assign(&numerical_deriv);
        }
        h_deriv
    }

    pub fn test_overlap_gradient(&mut self) {
        self.properties.reset();
        self.prepare_scc();
        // let _ = self.run_scc().unwrap();
        // println!("Overlap matrix:\n{:.8}", self.properties.s().unwrap());
        // let p: Array2<f64> = self.properties.p().unwrap().to_owned();
        // let w: Array1<f64> = 0.5
        //     * (p.dot(&self.properties.h_coul_x().unwrap()).dot(&p))
        //         .into_shape([self.n_orbs * self.n_orbs])
        //         .unwrap();

        let analytical: Array3<f64> = self.test_analytical_overlap_gradient();
        let numerical: Array3<f64> = self.numerical_overlap_wrapper();
        let diff: Array3<f64> = &analytical - &numerical;
        let sum: f64 = diff.map(|val| val.abs()).sum();

        println!("Analytical: \n {:.5}", analytical);
        println!("Numerical: \n {:.5}", numerical);
        println!("Difference: \n {:.7}", diff);
        println!("Sum of absolute difference: {:.10}", sum);

        let grad_s_2d: ArrayView2<f64> = numerical
            .view()
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap();
        // let grad_contribution: Array1<f64> = grad_s_2d.dot(&w);
        // println!("Test grad: {:.6}", grad_contribution);
    }

    fn test_analytical_overlap_gradient(&mut self) -> Array3<f64> {
        self.properties.reset();
        self.prepare_scc();
        let grad_s: Array3<f64> = calc_overlap_matrix_obs_derivs_new(&self.basis, self.n_atoms);
        grad_s
    }

    fn numerical_overlap_wrapper(&mut self) -> Array3<f64> {
        self.properties.reset();
        self.prepare_scc();
        let coords: Array1<f64> = self.get_xyz();
        let mut s_deriv: Array3<f64> = Array3::zeros([3 * self.n_atoms, self.n_orbs, self.n_orbs]);

        let stepsize: f64 = 1.0e-4;
        for index in 0..(3 * self.n_atoms) {
            let mut step: Array1<f64> = Array1::zeros([3 * self.n_atoms]);
            step[index] = 1.0;
            let geom_1: Array1<f64> = coords.clone() + stepsize * &step;
            let geom_2: Array1<f64> = coords.clone() - stepsize * &step;

            self.properties.reset();
            self.update_xyz(geom_1.view());
            self.prepare_scc();
            let s_1: Array2<f64> = self.properties.s().unwrap().to_owned();

            self.properties.reset();
            self.update_xyz(geom_2.view());
            self.prepare_scc();

            let s_2: ArrayView2<f64> = self.properties.s().unwrap();
            let numerical_deriv: Array2<f64> = (&s_1 - &s_2) / (2.0 * stepsize);

            s_deriv
                .slice_mut(s![index, .., ..])
                .assign(&numerical_deriv);
        }
        s_deriv
    }
}
