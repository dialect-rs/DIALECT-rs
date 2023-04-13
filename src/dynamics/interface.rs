use crate::initialization::System;
use dialect_dynamics::interface::QCInterface;
use ndarray::prelude::*;

impl QCInterface for System {
    // Return enegies, forces, non-adiabtic coupling and the transition dipole
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
    ) {
        // update the coordinates of the system
        self.update_xyz(coordinates.into_shape(3 * self.n_atoms).unwrap());
        // calculate the energy and the gradient of the state
        let (energies, gradient): (Array1<f64>, Array1<f64>) =
            self.calculate_energies_and_gradient(state, state_coupling, gs_dynamic);
        let gradient: Array2<f64> = gradient.into_shape([self.n_atoms, 3]).unwrap();

        // calculate the scalar couplings
        let (couplings, olap): (Option<Array2<f64>>, Option<Array2<f64>>) =
            if state_coupling && gs_dynamic == false {
                let (couplings, olap): (Array2<f64>, Array2<f64>) = self.get_scalar_coupling(dt);
                (Some(couplings), Some(olap))
            } else {
                (None, None)
            };

        return (energies, gradient, couplings, olap);
    }
}
