use crate::initialization::Simulation;
use ndarray::prelude::*;
use ndarray_linalg::c64;
use rand::distributions::Standard;
use rand::prelude::*;

impl Simulation {
    /// Calculate the new electronic state of the molecular system.
    pub fn get_new_state(&mut self, old_coefficients: ArrayView1<c64>) {
        let nstates: usize = self.config.nstates;
        let mut occupations: Array1<f64> = Array1::zeros(nstates);
        let mut derivatives: Array1<f64> = Array1::zeros(nstates);

        for state in 0..nstates {
            occupations[state] =
                self.coefficients[state].re.powi(2) + self.coefficients[state].im.powi(2);
            derivatives[state] = (occupations[state]
                - (old_coefficients[state].re.powi(2) + old_coefficients[state].im.powi(2)))
                / self.stepsize;
        }
        let mut new_state: usize = self.state;
        if derivatives[self.state] < 0.0 {
            let mut hopping_probabilities: Array1<f64> = Array1::zeros(nstates);
            let mut probability: f64 = 0.0;
            let mut states_to_hopp: Vec<usize> = Vec::new();

            for k in 0..nstates {
                if derivatives[k] > 0.0 {
                    states_to_hopp.push(k);
                    probability += derivatives[k];
                }
            }
            for state in states_to_hopp {
                let tmp: f64 = old_coefficients[self.state].re.powi(2)
                    + old_coefficients[self.state].im.powi(2);
                hopping_probabilities[state] =
                    -1.0 * (derivatives[self.state] / tmp) * derivatives[state] * self.stepsize
                        / probability;
            }
            assert!(
                hopping_probabilities.sum() <= 1.0,
                "Total hopping probability bigger than 1.0!"
            );

            let random_number: f64 = StdRng::from_entropy().sample(Standard);

            let mut sum: f64 = 0.0;
            for state in 0..nstates {
                let prob: f64 = hopping_probabilities[state];
                if prob > 0.0 {
                    sum += prob;
                    if random_number < sum {
                        new_state = state;
                        break;
                    }
                }
            }
        }
        //  If the energy gap between the first excited state and the ground state
        //  approaches zero, because the trajectory has hit a conical intersection to
        //  the ground state, TD-DFT will break down. In this case, a transition
        //  to the ground state is forced.
        let threshold: f64 = 0.1 / 27.211;
        if new_state > 0 && self.config.hopping_config.force_switch_to_gs {
            let gap: f64 = self.energies[new_state] - self.energies[0];
            if gap < threshold {
                println!("Conical intersection to ground state reached.");
                println!("The trajectory will continue on the ground state.");
                new_state = 0;
                // if a conical intersection to the ground state is encountered
                // force the dynamic to stay in the ground state
                self.config.gs_dynamic = true;
            }
        }
        self.state = new_state;
    }

    /// decoherence correction according to eqn. (17) in
    /// G. Granucci, M. Persico,
    /// "Critical appraisal of the fewest switches algorithm for surface hopping",
    /// J. Chem. Phys. 126, 134114 (2007)
    /// If the trajectory is in the current state K, the coefficients of the other
    /// states J != K are made to decay exponentially, C'_J = exp(-dt/tau_JK) C_J.
    /// The decay time is proportional to the inverse of the energy gap |E_J-E_K|,
    /// so that the coherences C_J*C_K decay very quickly if the energy gap between
    /// the two states is large. The electronic transitions become irreversible.
    pub fn get_decoherence_correction(&self, decoherence_constant: f64) -> Array1<c64> {
        let mut sm: f64 = 0.0;
        let mut new_coefficients: Array1<c64> = self.coefficients.clone();
        for state in 0..self.config.nstates {
            if state != self.state {
                let tauij: f64 = 1.0 / (self.energies[state] - self.energies[self.state]).abs()
                    * (1.0 + decoherence_constant / self.kinetic_energy);
                new_coefficients[state] *= (-self.stepsize / tauij).exp();
                sm += new_coefficients[state].re.powi(2) + new_coefficients[state].im.powi(2);
            }
        }
        let tmp: f64 =
            new_coefficients[self.state].re.powi(2) + new_coefficients[self.state].im.powi(2);
        new_coefficients[self.state] = new_coefficients[self.state] * (1.0 - sm).sqrt() / tmp;

        new_coefficients
    }

    /// Uniform rescaling of the velocities after the surface hopping procedures
    pub fn uniformly_rescaled_velocities(&self, old_state: usize) -> (Array2<f64>, usize) {
        // hop is rejected when kinetic energy is too low
        let mut state: usize = self.state;
        let mut new_velocities: Array2<f64> = self.velocities.clone();
        if self.state > old_state
            && (self.energies[self.state] - self.energies[old_state]) > self.kinetic_energy
        {
            state = old_state;
        } else if self.kinetic_energy > 0.0 {
            let vel_scale: f64 = ((self.kinetic_energy
                + (self.energies[old_state] - self.energies[self.state]))
                / self.kinetic_energy)
                .sqrt();
            new_velocities *= vel_scale;
        }
        (new_velocities, state)
    }

    /// Rescaling of the velocities when using the energy conservation approach
    pub fn scale_velocities_const_energy(
        &self,
        old_state: usize,
        old_kinetic_energy: f64,
        old_potential_energy: f64,
    ) -> Array2<f64> {
        // The velocities are rescaled so that energy conservation
        // between two time-steps is fulfilled exactly.
        let mut new_velocities: Array2<f64> = Array2::zeros(self.velocities.raw_dim());
        if self.state != old_state {
            let scaling_factor: f64 = ((old_kinetic_energy
                + (old_potential_energy - self.energies[self.state]))
                / self.kinetic_energy)
                .sqrt();
            assert!(
                (scaling_factor - 1.0).abs() < 1.0e-1,
                "Total energy is not conserved!"
            );

            new_velocities = scaling_factor * &self.velocities;
        }
        new_velocities
    }
}

/// Normalize the state coefficients of the system
pub fn normalize_coefficients(coefficients: ArrayView1<c64>) -> Array1<c64> {
    let mut norm: f64 = 0.0;
    let nstates: usize = coefficients.len();

    for state in 0..nstates {
        norm += coefficients[state].re.powi(2) + coefficients[state].im.powi(2);
    }
    let new_coefficients: Array1<c64> = coefficients.to_owned() / norm.sqrt();
    new_coefficients
}
