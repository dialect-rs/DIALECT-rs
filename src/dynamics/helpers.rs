use crate::initialization::System;
use crate::scc::scc_routine::RestrictedSCC;
use ndarray::prelude::*;

impl System {
    pub fn calculate_energies_and_gradient(
        &mut self,
        state: usize,
        state_coupling: bool,
        gs_dynamic: bool,
    ) -> (Array1<f64>, Array1<f64>) {
        let mut gradient: Array1<f64> = Array::zeros(3 * self.n_atoms);
        let mut energies: Array1<f64> = Array1::zeros(self.config.excited.nstates + 1);

        if state == 0 {
            // ground state energy
            self.prepare_scc();
            let gs_energy: f64 = self.run_scc().unwrap();
            energies[0] = gs_energy;

            gradient = self.ground_state_gradient(false);

            if state_coupling && gs_dynamic == false {
                // calculate excited states
                self.calculate_excited_states(false);
            }
        } else {
            // ground state energy
            let excited_state: usize = state - 1;
            self.prepare_scc();
            let gs_energy: f64 = self.run_scc().unwrap();
            energies[0] = gs_energy;

            // calculate excited states
            self.calculate_excited_states(false);

            let ci_energies: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
            energies
                .slice_mut(s![1..])
                .assign(&(gs_energy + &ci_energies));

            gradient = self.ground_state_gradient(true);
            gradient = gradient + self.calculate_excited_state_gradient(excited_state);
        }

        return (energies, gradient);
    }
}
