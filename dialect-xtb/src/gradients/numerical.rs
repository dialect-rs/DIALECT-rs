use super::hamiltonian::{
    calculate_coordination_number_gradients, calculate_h0_gradient_xtb1_atom_specific,
    calculate_h0_gradient_xtb1_new,
};
use dialect_utilities::scc_helpers::aovec_to_aomat;
use dialect_utilities::numerical::{assert_deriv_5point, assert_deriv_7point};
use crate::gradients::helpers::coul_third_order_grad_contribution_xtb;
use crate::integrals::{
    calc_overlap_derivative_matrix_iterative_over_atoms, calc_overlap_matrix_obs_derivs_new,
};
use crate::scc::gamma_matrix::{gamma_gradient_xtb_atom_specific, gamma_gradient_xtb_new};
use crate::scc::scc_helpers::calculate_repulsive_energy_xtb;
use dialect_utilities::numerical::assert_deriv;
use dialect_utilities::scc_interface::RestrictedSCC;
use crate::{
        initialization::system::XtbSystem, parameters::COUL_THIRD_ORDER_ATOM,
        scc::hamiltonian::calculate_coordination_numbers,
};
use ndarray::prelude::*;

impl XtbSystem {
    pub fn test_gs_gradient(&mut self) {
        let xyz = self.get_xyz();
        assert_deriv_7point(
            self,
            XtbSystem::gs_energy_wrapper,
            XtbSystem::gs_gradient_wrapper,
            self.get_xyz(),
            0.01,
            1e-6,
        );
        self.update_xyz(xyz.view());
    }

    pub fn gs_gradient_wrapper(&mut self) -> Array1<f64> {
        self.properties.reset_reduced();
        self.prepare_scc();
        let _ = self.run_scc().unwrap();
        self.ground_state_gradient_onthefly()
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
        calculate_repulsive_energy_xtb(&self.atoms)
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

    pub fn test_compare_gamma_gradients(&mut self) {
        self.properties.reset();
        self.prepare_scc();

        let analytical_1: Array3<f64> = gamma_gradient_xtb_new(
            &self.gammafunction,
            &self.atoms,
            &self.basis,
            self.n_atoms,
            self.n_orbs,
        );
        let mut analytical2: Array3<f64> = Array3::zeros(analytical_1.raw_dim());
        for atom_idx in 0..self.n_atoms {
            let dgamma: Array3<f64> = gamma_gradient_xtb_atom_specific(
                &self.gammafunction,
                &self.atoms,
                &self.basis,
                self.n_orbs,
                atom_idx,
            );
            analytical2
                .slice_mut(s![3 * atom_idx..3 * atom_idx + 3, .., ..])
                .assign(&dgamma);
        }

        println!(
            "Analytical gamma: \n{:.7}",
            analytical_1.slice(s![..2, .., ..])
        );
        println!(
            "Analytical gamma2: \n{:.7}",
            analytical2.slice(s![..2, .., ..])
        );
        let diff: Array3<f64> = &analytical_1 - &analytical2;
        let sum: f64 = diff.map(|val| val.abs()).sum();
        println!("Sum of absolute difference: {:.10}", sum);
    }

    pub fn test_h0_gradient(&mut self) {
        self.properties.reset();
        self.prepare_scc();
        // let _ = self.run_scc().unwrap();
        // let p: Array2<f64> = self.properties.p().unwrap().to_owned();

        let analytical: Array3<f64> = self.test_analytical_h0_gradient();
        let analytical2: Array3<f64> = self.test_analytical_h0_gradient2();
        let numerical: Array3<f64> = self.numerical_h0_gradient_wrapper();
        let diff: Array3<f64> = &analytical - &numerical;
        let sum: f64 = diff.map(|val| val.abs()).sum();
        let diff2: Array3<f64> = &analytical - &analytical2;
        let sum2: f64 = diff2.map(|val| val.abs()).sum();

        println!("Analytical: \n {:.6}", analytical.slice(s![2, .., ..]));
        println!("Numerical: \n {:.6}", numerical.slice(s![2, .., ..]));
        println!("Analytical2: \n {:.6}", analytical2.slice(s![2, .., ..]));
        println!("Difference: \n {:.7}", diff);
        println!("Sum of absolute difference: {:.10}", sum);
        println!("Sum of absolute difference2: {:.10}", sum2);

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

    fn test_analytical_h0_gradient2(&mut self) -> Array3<f64> {
        self.properties.reset();
        self.prepare_scc();
        let grad_s: Array3<f64> = calc_overlap_matrix_obs_derivs_new(&self.basis, self.n_atoms);
        let mut grad_h0: Array3<f64> = Array3::zeros([3 * self.n_atoms, self.n_orbs, self.n_orbs]);
        for idx in 0..self.n_atoms {
            let h0_grad = calculate_h0_gradient_xtb1_atom_specific(
                self.n_orbs,
                &self.atoms,
                self.properties.s().unwrap(),
                grad_s.slice(s![3 * idx..3 * idx + 3, .., ..]),
                &self.basis,
                idx,
            );
            grad_h0
                .slice_mut(s![3 * idx..3 * idx + 3, .., ..])
                .assign(&h0_grad);
        }

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

        let analytical: Array3<f64> = self.test_analytical_overlap_gradient();
        let numerical: Array3<f64> = self.numerical_overlap_wrapper();
        let diff: Array3<f64> = &analytical - &numerical;
        let sum: f64 = diff.map(|val| val.abs()).sum();

        println!(
            "Sum of absolute difference with second approach: {:.10}",
            sum
        );
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

#[cfg(test)]
mod gradient_accuracy_tests {
    use crate::initialization::system::XtbSystem;
    use dialect_config::Configuration;
    use ndarray::prelude::*;

    /// Largest tolerated deviation between the analytical and numerical
    /// GFN1-xTB ground-state gradient (Hartree/Bohr).
    const GRADIENT_TOLERANCE: f64 = 1.0e-6;
    /// Finite-difference step size (Bohr) for the 5-point stencil.
    const STEP: f64 = 1.0e-2;

    /// GFN1-xTB configuration with a very tightly converged SCC, required for
    /// the numerical gradient to be meaningful.
    fn xtb_config() -> Configuration {
        let mut config: Configuration = toml::from_str("").unwrap();
        config.tight_binding.use_dftb = false;
        config.tight_binding.use_xtb1 = true;
        config.scf.electronic_temperature = 300.0;
        config.scf.scf_charge_conv = 1.0e-11;
        config.scf.scf_energy_conv = 1.0e-11;
        config
    }

    /// Numerical ground-state gradient via the 5-point central-difference
    /// stencil: f'(x) = [-f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h)] / (12 h).
    fn numerical_gradient(system: &mut XtbSystem, origin: &Array1<f64>) -> Array1<f64> {
        let mut grad: Array1<f64> = Array1::zeros(origin.len());
        for i in 0..origin.len() {
            let mut energy_at = |delta: f64| -> f64 {
                let mut geom: Array1<f64> = origin.clone();
                geom[i] += delta;
                system.gs_energy_wrapper(geom)
            };
            let f_p2: f64 = energy_at(2.0 * STEP);
            let f_p1: f64 = energy_at(STEP);
            let f_m1: f64 = energy_at(-STEP);
            let f_m2: f64 = energy_at(-2.0 * STEP);
            grad[i] = (-f_p2 + 8.0 * f_p1 - 8.0 * f_m1 + f_m2) / (12.0 * STEP);
        }
        grad
    }

    /// Compare the analytical and 5-point numerical gradient for one molecule.
    fn assert_gradient_accuracy(name: &str) {
        let path: String = format!(
            "{}/../tests/data/{}/{}.xyz",
            env!("CARGO_MANIFEST_DIR"),
            name,
            name
        );
        let mut system: XtbSystem = XtbSystem::from((path.as_str(), xtb_config()));
        let origin: Array1<f64> = system.get_xyz();

        let analytical: Array1<f64> = system.gs_gradient_wrapper();
        let numerical: Array1<f64> = numerical_gradient(&mut system, &origin);
        system.update_xyz(origin.view());

        let max_dev: f64 = analytical
            .iter()
            .zip(numerical.iter())
            .map(|(a, n)| (a - n).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_dev < GRADIENT_TOLERANCE,
            "Molecule: {}, max |analytical - numerical| gradient deviation {:.3e} exceeds {:.1e}",
            name,
            max_dev,
            GRADIENT_TOLERANCE
        );
    }

    /// The analytical GFN1-xTB ground-state gradient must agree with the
    /// 5-point numerical gradient for water (O-H), ammonia (N-H, non-trivial
    /// element-pair scaling) and PCl3 (d-d overlap, where the spherical d
    /// transform matters).
    #[test]
    fn gfn1_xtb_gradient_accuracy() {
        assert_gradient_accuracy("h2o");
        assert_gradient_accuracy("ammonia");
        assert_gradient_accuracy("pcl3");
    }
}
