pub use ndarray::prelude::*;

/// Trait that provides an interface for a quantum chemistry programm.
/// The trait implements the function compute data,
/// which returns the energies, the gradient, the nonadiabatic couplings and the
/// dipoles of the molecular system
pub trait QCInterface {
    fn compute_data(
        &mut self,
        coordinates: ArrayView2<f64>,
        state: usize,
        dt: f64,
        state_coupling: bool,
        gs_dynamic: bool,
    ) -> (
        Array1<f64>,
        Array2<f64>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
    );
}
