use crate::initialization::System;
use crate::scc::scc_routine::RestrictedSCC;
use ndarray::prelude::*;
use rayon::prelude::*;

impl System {
    pub fn calculate_num_hessian(&self) -> Array2<f64> {
        // calculate the numerical hessian of the system using finite differences of the gs gradient
        let hess: Array2<f64> = derivative_gradient_fd(self, self.get_xyz(), 1.0e-5);
        // symmetrize the hessian
        let hessian: Array2<f64> = 0.5 * (&hess + &hess.t());

        hessian
    }

    pub fn gs_gradient_wrapper_hessian(&mut self, geometry: Array1<f64>) -> Array1<f64> {
        self.properties.reset();
        self.update_xyz(geometry.view());
        self.prepare_scc();
        self.run_scc().unwrap();

        self.ground_state_gradient(false)
    }
}

fn derivative_gradient_fd(system: &System, origin: Array1<f64>, stepsize: f64) -> Array2<f64> {
    assert!(
        stepsize > 0.0,
        "The stepsize has to be > 0.0, but it is {}",
        stepsize
    );
    let dim = origin.len();
    let mut derivatives: Array2<f64> = Array2::zeros([dim, dim]);

    // PARALLEL
    let deriv_vec: Vec<Array1<f64>> = (0..origin.len())
        .into_par_iter()
        .map(|i| {
            let mut system_clone = system.clone();
            // compute the numerical derivative of this function and an error estimate using
            // finite difference
            let numerical_deriv: Array1<f64> =
                finite_difference_1d(&mut system_clone, origin.clone(), i, stepsize);

            numerical_deriv
        })
        .collect();

    for (idx, array) in deriv_vec.iter().enumerate() {
        derivatives.slice_mut(s![idx, ..]).assign(array);
    }

    derivatives
}

fn finite_difference_1d<D>(
    system: &mut System,
    origin: ArrayBase<D, Ix1>, // Origin of coordinates, that are used for the function
    index: usize,              // Index for which the derivative is computed
    stepsize: f64,             // Initial step size
) -> Array1<f64>
where
    D: ndarray::Data<Elem = f64>,
{
    // make the stepsize mutable
    let stepsize: f64 = stepsize;
    let mut step: Array1<f64> = Array1::zeros([origin.len()]);
    step[index] = 1.0;

    (system.gs_gradient_wrapper_hessian(&origin + &(&step * stepsize))
        - system.gs_gradient_wrapper_hessian(&origin - &(&step * stepsize)))
        / (2.0 * stepsize)
}
