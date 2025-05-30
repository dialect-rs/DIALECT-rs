pub use ndarray::prelude::*;
use ndarray_linalg::c64;

/// Trait that provides an interface for a quantum chemistry programm.
/// The trait implements the function compute data,
/// which returns the energies, the gradient, the nonadiabatic couplings and the
/// dipoles of the molecular system
pub trait QCInterface {
    fn compute_data(
        &mut self,
        coordinates: ArrayView2<f64>,
        velocities: ArrayView2<f64>,
        state: usize,
        dt: f64,
        state_coupling: bool,
        use_nacv_couplings: bool,
        gs_dynamic: bool,
        step: usize,
        nstates: usize,
    ) -> (
        Array1<f64>,
        Array2<f64>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
        Option<Vec<Array1<f64>>>,
    );

    fn recompute_gradient(&mut self, coordinates: ArrayView2<f64>, state: usize) -> Array2<f64>;

    /// Returns the excitonic-coupling matrix and the gradient
    fn compute_ehrenfest(
        &mut self,
        coordinates: ArrayView2<f64>,
        velocities: ArrayView2<f64>,
        state_coefficients: ArrayView1<c64>,
        thresh: f64,
        dt: f64,
        step: usize,
        use_state_couplings: bool,
        use_nacv_couplings: bool,
    ) -> (f64, Array2<f64>, Array2<f64>, Array2<f64>);

    fn compute_ehrenfest_tab(
        &mut self,
        coordinates: ArrayView2<f64>,
        velocities: ArrayView2<f64>,
        state_coefficients: ArrayView1<c64>,
        thresh: f64,
        tab_grad_thresh: f64,
        dt: f64,
        step: usize,
        use_state_couplings: bool,
        use_nacv_couplings: bool,
    ) -> (f64, Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>);
}
