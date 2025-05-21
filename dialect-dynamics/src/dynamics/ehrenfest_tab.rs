use std::time::Instant;

use crate::initialization::Simulation;
use crate::interface::QCInterface;
use crate::output::helper::{
    print_footer_dynamics, print_footer_electronic_structure, print_header_dynamics_step,
    print_init_electronic_structure,
};
use ndarray::prelude::*;
use ndarray_linalg::c64;
use rand::Rng;
use rand_distr::Standard;
use rayon::prelude::*;

impl Simulation {
    ///Ehrenfest dynamics routine of the struct Simulation
    pub fn ehrenfest_dynamics_tab(&mut self, interface: &mut dyn QCInterface) {
        let initial_step: usize = self.initialize_ehrenfest(interface);
        for step in 1..self.config.nstep {
            print_header_dynamics_step();
            let timer: Instant = Instant::now();
            self.ehrenfest_tab_step(interface, step + initial_step);
            print_footer_dynamics(timer.elapsed().as_secs_f64());
        }
    }

    pub fn ehrenfest_tab_step(&mut self, interface: &mut dyn QCInterface, step: usize) {
        let old_forces: Array2<f64> = self.forces.clone();
        let old_force_matrix: Array2<f64> = self.force_array.clone();
        // calculate the gradient and the excitonic couplings
        let excitonic_couplings: Array2<f64> = self.get_ehrenfest_data_tab(interface, step);

        // ehrenfest integration
        self.choose_ehrenfest_integration(excitonic_couplings.view());

        // Calculate new coordinates from velocity-verlet
        self.velocities = self.get_velocities_verlet(old_forces.view());
        // remove tranlation and rotation from the velocities
        self.velocities = self.eliminate_translation_rotation_from_velocity();

        // calculate the kinetic energy
        self.kinetic_energy = self.get_kinetic_energy();

        // Do the TAB procedure
        let new_coeffs: Array1<c64> = self.collapse_to_a_block(old_force_matrix.view());
        // recompute the forces from the new coefficients and rescale the velocities
        // if the energy difference to the old wavefunction is smaller than the kinetic energy
        self.coefficients = self.tab_velocity_rescaling_and_forces(new_coeffs.view());

        // scale velocities
        self.velocities = self
            .thermostat
            .scale_velocities(self.velocities.view(), self.kinetic_energy);

        // Print settings
        self.print_ehrenfest_data(false, step);

        // Calculate new coordinates from velocity-verlet
        self.coordinates = self.get_coord_verlet();

        // Shift coordinates to center of mass
        self.coordinates = self.shift_to_center_of_mass();
    }

    pub fn tab_velocity_rescaling_and_forces(
        &mut self,
        new_coeffs: ArrayView1<c64>,
    ) -> Array1<c64> {
        // get the squared norm of the coefficients
        let coeff_abs: Array1<f64> = self.coefficients.map(|val| val.norm_sqr());
        let coeff_abs_new: Array1<f64> = new_coeffs.map(|val| val.norm_sqr());

        // calculate the old and the new ehrenfest energies
        let mut ehrenfest_energy_new: f64 = 0.0;
        let mut ehrenfest_energy_old: f64 = 0.0;
        for ((idx, coeff_old), coeff_new) in coeff_abs.iter().enumerate().zip(coeff_abs_new.iter())
        {
            ehrenfest_energy_old += self.energies[idx] * coeff_old;
            ehrenfest_energy_new += self.energies[idx] * coeff_new;
        }
        // get the energy difference
        let energy_difference: f64 = ehrenfest_energy_old - ehrenfest_energy_new;
        let energy_difference_2: f64 = -energy_difference;

        // masses array
        let mut masses_grad_dim: Array1<f64> = Array1::zeros(3 * self.n_atoms);
        for idx in 0..self.n_atoms {
            for nc_idx in 0..3 {
                let grad_idx: usize = idx * 3 + nc_idx;
                let mass: f64 = self.masses[idx];
                masses_grad_dim[grad_idx] = mass;
            }
        }

        let mut new_coefficients: Array1<c64> = new_coeffs.to_owned();
        // check if the kinetic energy is higher than the potential energy difference
        if energy_difference_2 > self.kinetic_energy {
            // no change in the coefficients if the kinetic energy is insufficient
            new_coefficients = self.coefficients.clone();
        } else {
            // calculate the new forces
            let mut new_forces: Array1<f64> = Array1::zeros(3 * self.n_atoms);
            for (idx, coeff) in coeff_abs_new.iter().enumerate() {
                new_forces = new_forces
                    + *coeff * &(&self.force_array.slice(s![.., idx]) / &masses_grad_dim);
            }
            // reshape the forces
            let new_forces: Array2<f64> = new_forces.into_shape([self.n_atoms, 3]).unwrap();
            // update the forces
            self.forces = new_forces;
            if self.config.ehrenfest_config.use_restraint {
                self.apply_harmonic_restraint();
            }

            // rescale the velocities
            let vel_scale: f64 =
                ((self.kinetic_energy + energy_difference) / self.kinetic_energy).sqrt();
            let new_velocities: Array2<f64> = &self.velocities * vel_scale;
            // update the velocities
            self.velocities = new_velocities;
        }
        new_coefficients
    }

    pub fn collapse_to_a_block(&mut self, old_force_array: ArrayView2<f64>) -> Array1<c64> {
        // TAB procedure of Esch and Levine
        // get the number of states
        let nstates: usize = self.coefficients.len();
        // calculate the average force for the previous and the current geometry
        let avg_forces: Array2<f64> = 0.5 * (&self.force_array + &old_force_array);

        // unwrap the alpha values
        let alpha_arr = self.alpha_values.as_ref().unwrap();

        // 1st: Calculate the density matrix from the coefficients
        // and the state pairwise decoherence time
        let mut index_vec: Vec<(usize, usize)> = Vec::new();
        // get pair indices
        for idx_i in 0..nstates {
            for idx_j in 0..nstates {
                index_vec.push((idx_i, idx_j));
            }
        }

        let (vec_c, vec_d): (Vec<c64>, Vec<c64>) = index_vec
            .par_iter()
            .map(|indices| {
                let idx_i = indices.0;
                let idx_j = indices.1;

                // get the coefficients
                let c_i = self.coefficients[idx_i];
                let c_j = self.coefficients[idx_j];
                // get the product of the coefficients
                let c_val: c64 = c_i * c_j.conj();
                // matrix element of the rho_d matrix
                let mut d_val: c64 = c_val;

                // calculate the off_diagonal factor if abs of c_val is above the threshold
                if c_val.norm_sqr() > 1.0e-10 && idx_i != idx_j {
                    // get the pair statewise decoherence time
                    // calculate the force difference
                    let force_diff_squared: Array1<f64> = (&avg_forces.slice(s![.., idx_i])
                        - &avg_forces.slice(s![.., idx_j]))
                        .map(|val| val.powi(2));
                    let mut force_diff_squared: Array2<f64> =
                        force_diff_squared.into_shape([self.n_atoms, 3]).unwrap();
                    for idx in 0..3 {
                        let tmp_force: Array1<f64> =
                            &force_diff_squared.slice(s![.., idx]) / alpha_arr;
                        force_diff_squared.slice_mut(s![.., idx]).assign(&tmp_force);
                    }
                    // divide the force difference by alpha and sum over the gradient dimension
                    let tau_ij_inv_2: f64 = force_diff_squared.sum();

                    d_val *= (-self.stepsize * tau_ij_inv_2).exp();
                }

                (c_val, d_val)
            })
            .collect();
        // reshape to rho_c and rho_d
        let rho_c: Array2<c64> = Array1::from(vec_c).into_shape([nstates, nstates]).unwrap();
        let rho_d: Array2<c64> = Array1::from(vec_d).into_shape([nstates, nstates]).unwrap();

        // create matrices for the loop
        let mut tmp_rho: Array2<c64> = rho_d.clone();
        let rho_c_ref: Array2<c64> = rho_c;

        // rho block vector
        let mut block_vector: Vec<Array2<c64>> = Vec::new();
        let mut weight_vector: Vec<f64> = Vec::new();

        // do the TAB loop
        'tab_loop: for _idx in 0..500 {
            if _idx == 499 {
                println!("TAB iteration limit reached: {}", _idx);
            }
            // println!("Rho tmp:\n{:.3}", tmp_rho);
            // find the states with populations below the treshold
            let mut excluded_states: Vec<usize> = Vec::new();
            for (idx, val) in tmp_rho.diag().iter().enumerate() {
                if val.re < 1.0e-10 {
                    excluded_states.push(idx);
                }
            }

            // get the b matrix
            // let b_mat: Array2<f64> = (&tmp_rho / &rho_c_ref).map(|val| val.re);
            let mut b_mat: Array2<f64> = Array2::zeros(rho_c_ref.raw_dim());
            for state_i in 0..nstates {
                if !excluded_states.contains(&state_i) {
                    for state_j in 0..nstates {
                        if !excluded_states.contains(&state_j) {
                            b_mat[[state_i, state_j]] =
                                (tmp_rho[[state_i, state_j]] / rho_c_ref[[state_i, state_j]]).re;
                        }
                    }
                }
            }

            // get the smallest value of the b matrix, which is above 1.0e-10
            let mut indices: (usize, usize) = (0, 0);
            let mut smallest_value: f64 = 10.0;
            let threshold: f64 = 1.0e-10;
            let mut b_counter: usize = 0;

            for (idx_1, b_i) in b_mat.outer_iter().enumerate() {
                for (idx_2, b_ij) in b_i.iter().enumerate() {
                    let b_val: f64 = *b_ij;
                    if b_val < smallest_value && b_val > threshold {
                        smallest_value = b_val;
                        indices = (idx_1, idx_2);
                    }
                    if b_val < threshold {
                        b_counter += 1;
                    }
                }
            }
            let k: usize = indices.0;
            let l: usize = indices.1;

            // check if every element of b is below the threshold of 1.0e-10
            if b_counter == nstates * nstates {
                break 'tab_loop;
            }

            // create beta vector
            let mut beta: Vec<usize> = if k != l { vec![k, l] } else { vec![k] };
            for (idx_1, b_i) in b_mat.outer_iter().enumerate() {
                if idx_1 != k && idx_1 != l {
                    let b_val: f64 = b_i[k];
                    if b_val > threshold {
                        beta.push(idx_1);
                    }
                    let b_val: f64 = b_i[l];
                    if b_val > threshold {
                        beta.push(idx_1);
                    }
                }
            }
            let mut excluded_states: Vec<usize> = Vec::new();
            // loop over beta
            for idx1 in beta.iter() {
                if *idx1 != k && *idx1 != l && !excluded_states.contains(idx1) {
                    for idx2 in beta.iter() {
                        if *idx2 != k && *idx2 != l && !excluded_states.contains(idx2) {
                            let b_val: f64 = b_mat[[*idx1, *idx2]];
                            if b_val <= 1.0e-10 {
                                // randomly choose between 1 and 2 and exclude it from beta
                                let random_number: f64 = self.rng.sample(Standard);
                                if random_number < 0.5 {
                                    // exclude idx1
                                    excluded_states.push(*idx1);
                                } else {
                                    excluded_states.push(*idx2);
                                }
                            }
                        }
                    }
                }
            }
            // remove excluded states from beta
            let mut beta_new: Vec<usize> = Vec::new();
            for state in beta.iter() {
                if !excluded_states.contains(state) && !beta_new.contains(state) {
                    beta_new.push(*state);
                }
            }
            beta = beta_new;
            // for excluded_state in excluded_states.iter() {
            //     if beta.contains(excluded_state) {
            //         beta.remove(*excluded_state);
            //     }
            // }

            // create the density matrix that represents a coherent superposition
            // of the states in beta
            let mut rho_block: Array2<c64> = Array2::zeros(rho_c_ref.raw_dim());
            // get the sum of the diagonal elements of rho_c
            let mut rho_c_diag_sum: c64 = c64::new(0.0, 0.0);
            for idx in beta.iter() {
                rho_c_diag_sum += rho_c_ref[[*idx, *idx]];
            }
            for state1 in beta.iter() {
                for state2 in beta.iter() {
                    rho_block[[*state1, *state2]] = rho_c_ref[[*state1, *state2]] / rho_c_diag_sum;
                }
            }
            // get the weight of the block matrix
            let p_block: f64 = (tmp_rho[[k, l]] / rho_block[[k, l]]).re;
            // update tmp_rho
            tmp_rho = tmp_rho - &rho_block * p_block;

            // update the weight and block vectors
            weight_vector.push(p_block);
            block_vector.push(rho_block);
        }

        // generate random number
        let random_number: f64 = self.rng.sample(Standard);
        // get the new block matrix from the random number
        let mut sum_val: f64 = 0.0;
        let mut new_rho: Array2<c64> = Array2::zeros(tmp_rho.raw_dim());
        'new_block: for (idx, val) in weight_vector.iter().enumerate() {
            let next_val: f64 = val + sum_val;
            if random_number >= sum_val && random_number < next_val {
                new_rho = block_vector[idx].clone();
                break 'new_block;
            }
            sum_val += val;
        }

        // get the coefficients from the new rho matrix
        // first, calculate the squared norm of the coefficients
        let coeff_abs: Array1<f64> = self.coefficients.map(|val| val.norm_sqr());
        let coeff_norm: Array1<f64> = coeff_abs.map(|val| val.sqrt());

        // calculate the phases of the complex values
        let phases: Array1<c64> = Array::from(
            self.coefficients
                .iter()
                .enumerate()
                .map(|(idx, val)| {
                    let ret_val: c64 = if coeff_norm[idx] > 0.0 {
                        val / coeff_norm[idx]
                    } else {
                        c64::new(0.0, 0.0)
                    };
                    ret_val
                })
                .collect::<Vec<c64>>(),
        );

        // take the new populations from the diagonal of the rho matrix
        let rho_diagonal: Array1<f64> = new_rho.diag().map(|val| val.re);
        // calculate the new coefficients from the old phases and the new absolute values
        let new_coefficients: Array1<c64> = Array::from(
            rho_diagonal
                .iter()
                .zip(phases.iter())
                .map(|(abs, phase)| phase * abs.sqrt())
                .collect::<Vec<c64>>(),
        );

        // return the new coefficients
        new_coefficients
    }

    pub fn get_ehrenfest_data_tab(
        &mut self,
        interface: &mut dyn QCInterface,
        step: usize,
    ) -> Array2<f64> {
        // header
        print_init_electronic_structure();
        // timer
        let timer: Instant = Instant::now();

        // let abs_coefficients: Array1<f64> = self.coefficients.map(|val| val.norm_sqr());
        let tmp: (f64, Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) = interface
            .compute_ehrenfest_tab(
                self.coordinates.view(),
                self.velocities.view(),
                self.coefficients.view(),
                // abs_coefficients.view(),
                self.config.ehrenfest_config.state_threshold,
                self.config.ehrenfest_config.tab_grad_threshold,
                self.config.stepsize,
                step,
                self.config.ehrenfest_config.use_state_coupling,
                self.config.nonadibatic_config.use_nacv_couplings,
            );
        self.energies[0] = tmp.0;
        self.energies
            .slice_mut(s![1..])
            .assign(&tmp.2.diag().slice(s![1..]));

        let forces: Array2<f64> = tmp.1;
        for (idx, mass) in self.masses.iter().enumerate() {
            self.forces
                .slice_mut(s![idx, ..])
                .assign(&(-1.0 * &forces.slice(s![idx, ..]) / mass.to_owned()));
        }
        let excitonic_couplings: Array2<f64> = tmp.2;
        // update the nonadiabatic coupling
        self.nonadiabatic_scalar = tmp.3;

        // the gradients of all adiabatic states
        let gradients: Array2<f64> = tmp.4;
        let gradients_3d: Array3<f64> = gradients
            .into_shape([self.n_atoms, 3, self.config.nstates])
            .unwrap();
        let mut force_array: Array3<f64> = Array3::zeros(gradients_3d.raw_dim());
        for (idx, _mass) in self.masses.iter().enumerate() {
            let grad_slice: ArrayView2<f64> = gradients_3d.slice(s![idx, .., ..]);
            force_array
                .slice_mut(s![idx, .., ..])
                .assign(&(-1.0 * &grad_slice));
        }
        // reshape
        let force_array: Array2<f64> = force_array
            .into_shape([3 * self.n_atoms, self.config.nstates])
            .unwrap();
        self.force_array = force_array;
        if self.config.ehrenfest_config.use_restraint {
            self.apply_harmonic_restraint();
        }

        // footer
        print_footer_electronic_structure(timer.elapsed().as_secs_f64());

        // diabatic_couplings
        excitonic_couplings
    }
}
