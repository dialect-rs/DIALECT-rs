use crate::initialization::Simulation;
use ndarray::prelude::*;
use ndarray_linalg::c64;
use rand::distributions::Standard;
use rand::prelude::*;
use std::ops::DivAssign;

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
            // const const_n: usize = 32;
            // let random_number: f64 = StdRng::from_seed([0; const_n]).sample(Standard);
            // let random_number: f64 = StdRng::seed_from_u64(1).sample(Standard);
            // println!("random number:{}", random_number);
            let random_number: f64 = self.rng.sample(Standard);
            // println!("random number: {}", random_number);
            // let random_number: f64 = StdRng::from_entropy().sample(Standard);

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
        //  the ground state, TD-DFTB will break down. In this case, a transition
        //  to the ground state is forced.
        let threshold: f64 = self.config.hopping_config.force_switch_s0s1_threshold / 27.2114;
        if new_state > 0 && self.config.hopping_config.force_switch_to_gs {
            let gap: f64 = self.energies[new_state] - self.energies[0];
            if gap < threshold {
                println!("-------------------------------------------------------------------------------------");
                println!("Conical intersection to ground state reached.");
                println!("The trajectory will continue on the ground state.");
                println!("-------------------------------------------------------------------------------------");
                new_state = 0;
                let mut new_coeffs: Array1<c64> = Array1::zeros(self.config.nstates);
                new_coeffs[0] = c64::from(1.0);
                self.coefficients = new_coeffs;
                // if a conical intersection to the ground state is encountered
                // force the dynamic to stay in the ground state
                self.config.gs_dynamic = false;
                self.config.use_surface_hopping = false;
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
        new_coefficients[self.state] *= ((1.0 - sm) / tmp).sqrt();

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

    pub fn rescaled_velocities(&self, old_state: usize) -> (Array2<f64>, usize) {
        let state_after_rescale: usize;
        let mut vel_after_rescale = self.velocities.clone();

        if self.state > old_state
            && (self.energies[self.state] - self.energies[old_state]) > self.kinetic_energy
        {
            state_after_rescale = old_state;
        } else {
            let mut factor: f64 = 1.0;
            // get the nac vector for the state pair
            let new_state: usize = self.state;
            let mut index_1: usize = new_state;
            let mut index_2: usize = old_state;

            if new_state > old_state {
                factor = -1.0;
                index_1 = old_state;
                index_2 = new_state;
            }
            // get the correct nac vector from the array
            let mut count: usize = 0;
            let mut nac_idx: Option<usize> = None;
            for idx_1 in 0..self.config.nstates {
                for idx_2 in 0..self.config.nstates {
                    if idx_1 < idx_2 {
                        if index_1 == idx_1 && index_2 == idx_2 {
                            nac_idx = Some(count);
                        }
                        count += 1;
                    }
                }
            }
            let nac_idx: usize = nac_idx.unwrap();
            // get the nac vector
            let nac_vector: Array1<f64> = self.nonadiabatic_vectors[nac_idx].clone();
            // reshape the nac vector
            let nac_vector: Array2<f64> = nac_vector.into_shape([self.n_atoms, 3]).unwrap();

            let mut state: usize = new_state;
            // get the energy difference
            let delta_e: f64 = self.energies[old_state] - self.energies[new_state];

            // get the mass weighted nac vector
            let mut mass_weigh_nad: Array2<f64> = factor * &nac_vector;
            for i in 0..self.n_atoms {
                mass_weigh_nad
                    .slice_mut(s![i, ..])
                    .div_assign(self.masses[i]);
            }

            // calculate the rescaling factors
            let mut a: f64 = 0.0;
            for i in 0..self.n_atoms {
                a += nac_vector
                    .slice(s![i, ..])
                    .dot(&nac_vector.slice(s![i, ..]))
                    / self.masses[i];
            }
            a *= 0.5;
            let mut b: f64 = 0.0;
            for i in 0..self.n_atoms {
                b += self
                    .velocities
                    .slice(s![i, ..])
                    .dot(&(factor * &nac_vector.slice(s![i, ..])));
            }
            let val: f64 = b.powi(2) + 4.0 * a * delta_e;

            let gamma: f64;
            let new_velocities: Array2<f64>;
            // check frustrated hop
            if val < 0.0 {
                println!("Frustrated hop occured!");
                state = old_state;

                if self.config.hopping_config.use_rescaling_at_frustrated_hop {
                    gamma = b / a;
                    let nac_1d: Array1<f64> =
                        factor * nac_vector.clone().into_shape([3 * self.n_atoms]).unwrap();
                    let forces_1d: Array1<f64> =
                        self.forces.clone().into_shape([3 * self.n_atoms]).unwrap();
                    let mut momentum: Array2<f64> = self.velocities.clone();
                    for i in 0..self.n_atoms {
                        momentum.slice_mut(s![i, ..]).div_assign(self.masses[i])
                    }
                    let velocities_1d: Array1<f64> =
                        momentum.into_shape([3 * self.n_atoms]).unwrap();

                    let p_h: f64 = velocities_1d.dot(&nac_1d);
                    let f_h: f64 = forces_1d.dot(&nac_1d);
                    let sign: f64 = p_h * f_h;
                    // criterium for rescaling
                    if sign < 0.0 {
                        println!("Rescaled after frustrated hop!");
                        new_velocities = &self.velocities - &(gamma * mass_weigh_nad);
                    } else {
                        new_velocities = self.velocities.clone();
                    }
                } else {
                    new_velocities = self.velocities.clone();
                }
            } else {
                println!("Hop occurs, the pot. diff is {:.5} eV!", delta_e * 27.2114);
                // let compare_val_1: f64 = (-b + val.sqrt()).abs();
                // let compare_val_2: f64 = (-b - val.sqrt()).abs();
                // if compare_val_1 < compare_val_2 {
                //     gamma = (-b + val.sqrt()) / (2.0 * a);
                // } else {
                //     gamma = (-b - val.sqrt()) / (2.0 * a);
                // }
                // get the new velocities
                // new_velocities = &self.velocities + &(gamma * mass_weigh_nad);
                if b < 0.0 {
                    gamma = (b + val.sqrt()) / (2.0 * a);
                } else {
                    gamma = (b - val.sqrt()) / (2.0 * a);
                }
                // get the new velocities
                new_velocities = &self.velocities - &(gamma * mass_weigh_nad);
            }
            vel_after_rescale = new_velocities;
            state_after_rescale = state;
        }

        (vel_after_rescale, state_after_rescale)
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
