#![allow(dead_code)]

use crate::scc::gamma_approximation::{gamma_third_order, gamma_third_order_derivative};
use crate::scc::scc_routine::RestrictedSCC;
use crate::{initialization::System, scc::gamma_approximation::gamma_gradients_atomwise};
use ndarray::prelude::*;

impl System {
    pub fn gamma_grad(&mut self) -> Array1<f64> {
        self.properties.reset();
        self.prepare_scc();
        let _ = self.run_scc().unwrap();

        let grad_gamma: Array3<f64> = gamma_gradients_atomwise(
            self.gammafunction_lc.as_ref().unwrap(),
            &self.atoms,
            self.n_atoms,
        );

        grad_gamma.slice(s![.., 0, 1]).to_owned()
    }

    pub fn gamma_grad_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        self.update_xyz(geometry.view());
        self.prepare_scc();
        let gamma: ArrayView2<f64> = self.properties.gamma_lr().unwrap();
        gamma[[0, 1]]
    }

    pub fn test_gamma_gradient(&mut self) {
        assert_deriv(
            self,
            System::gamma_grad_wrapper,
            System::gamma_grad,
            self.get_xyz(),
            0.001,
            1e-6,
        );
    }

    fn gamma_third_order_grad(&mut self) -> Array3<f64> {
        self.properties.reset();
        self.prepare_scc();
        let _ = self.run_scc().unwrap();

        let grad_gamma: Array3<f64> = gamma_third_order_derivative(
            &self.gammafunction,
            &self.atoms,
            self.n_atoms,
            &self.config.dftb3.hubbard_derivatives,
        );

        grad_gamma
    }

    fn gamma_grad_third_order_wrapper(&mut self) -> Array3<f64> {
        self.properties.reset();
        self.prepare_scc();
        let _ = self.run_scc().unwrap();
        let coords: Array1<f64> = self.get_xyz();
        let mut gamma_deriv: Array3<f64> =
            Array3::zeros([3 * self.n_atoms, self.n_atoms, self.n_atoms]);

        let stepsize: f64 = 1.0e-4;
        for index in 0..(3 * self.n_atoms) {
            let mut step: Array1<f64> = Array1::zeros([3 * self.n_atoms]);
            step[index] = 1.0;
            let geom_1: Array1<f64> = coords.clone() + stepsize * &step;
            let geom_2: Array1<f64> = coords.clone() - stepsize * &step;

            self.properties.reset();
            self.update_xyz(geom_1.view());
            self.prepare_scc();
            let _ = self.run_scc().unwrap();
            let gamma_1: Array2<f64> = gamma_third_order(
                &self.gammafunction,
                &self.atoms,
                self.n_atoms,
                &self.config.dftb3.hubbard_derivatives,
            );
            self.properties.reset();
            self.update_xyz(geom_2.view());
            self.prepare_scc();
            let _ = self.run_scc().unwrap();
            let gamma_2: Array2<f64> = gamma_third_order(
                &self.gammafunction,
                &self.atoms,
                self.n_atoms,
                &self.config.dftb3.hubbard_derivatives,
            );
            let numerical_deriv: Array2<f64> = (gamma_1 - gamma_2) / (2.0 * stepsize);

            gamma_deriv
                .slice_mut(s![index, .., ..])
                .assign(&numerical_deriv);
        }
        gamma_deriv
    }

    pub fn test_gamma_gradient_third_order(&mut self) {
        let analytical: Array3<f64> = self.gamma_third_order_grad();
        let numerical: Array3<f64> = self.gamma_grad_third_order_wrapper();
        let diff: Array3<f64> = &analytical - &numerical;

        println!("Analytical: \n {:.4}", analytical);
        println!("Numerical: \n {:.4}", numerical);
        println!("Difference: \n {:.7}", diff);
        assert!(analytical.abs_diff_eq(&numerical, 1.0e-5));
    }

    pub fn gs_grad(&mut self) -> Array1<f64> {
        self.properties.reset();
        self.prepare_scc();
        let _ = self.run_scc().unwrap();

        self.ground_state_gradient(false)
    }

    pub fn gs_gradient_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        self.update_xyz(geometry.view());
        self.prepare_scc();

        self.run_scc().unwrap()
    }

    pub fn test_gs_gradient(&mut self) {
        assert_deriv(
            self,
            System::gs_gradient_wrapper,
            System::gs_grad,
            self.get_xyz(),
            0.001,
            1e-5,
        );
    }

    pub fn excited_gradient_wrapper(&mut self) -> Array1<f64> {
        self.properties.reset();
        self.prepare_scc();
        let _ = self.run_scc().unwrap();
        self.ground_state_gradient(true);

        self.calculate_excited_states(true);

        self.calculate_excited_state_gradient(0)
    }

    pub fn numerical_excited_grad(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        self.update_xyz(geometry.view());
        self.prepare_scc();
        self.run_scc().unwrap();

        self.calculate_excited_states(false);

        self.properties.ci_eigenvalue(0).unwrap()
    }

    pub fn test_excited_gradient(&mut self) {
        assert_deriv(
            self,
            System::numerical_excited_grad,
            System::excited_gradient_wrapper,
            self.get_xyz(),
            0.001,
            1e-6,
        );
    }
}

/// Returns the derivative of a function `function` at an Array of points `origin` by Ridder's method.
/// The value `stepsize` is an initial stepsize, it need to be small, but should be an increment
/// over which the `function` changes substantially. An estimate of the error in the derivative is
/// returned. The method was developed by C.J.F Ridders in 1982 (see the original article
/// ["Accurate computation of F′(x) and F′(x) F″(x)"](https://doi.org/10.1016/S0141-1195(82)80057-0))
/// The implementation is based on the one described in the Book Numerical Recipes by
/// W. H. Press and S. A. Teukolsky, the section is available as an article in
/// [Computers in Physics](https://aip.scitation.org/doi/pdf/10.1063/1.4822971). Also the Python
/// implementation derivcheckby T. Verstraelen
/// influenced the implementation and the idea to create an `assert_deriv` function was adopted.


/// Test the gradient of a function.
/// * function: The function whose derivatives must be tested, takes one argument
/// * gradient: Computes the gradient of the function, to be tested.
/// * origin: The point at which the derivatives are computed.
/// * stepsize: The initial (maximal) step size for the finite difference method.
/// * tol: The allowed relative error on the derivative.
///   The idea of this function comes from the derivcheck
///   Python package by T. Verstraelen.
pub use dialect_utilities::numerical::*;
