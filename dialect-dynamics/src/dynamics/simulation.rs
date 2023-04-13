use crate::initialization::restart::read_restart_parameters;
use crate::initialization::Simulation;
use crate::interface::QCInterface;
use ndarray::prelude::*;
use ndarray_linalg::c64;

impl Simulation {
    /// Velocity-verlet dynamic routine of the struct Simulation.
    pub fn verlet_dynamics(&mut self, interface: &mut dyn QCInterface) {
        self.initialize_verlet(interface);

        for _step in 0..self.config.nstep {
            self.verlet_step(interface);
        }
    }

    /// Langevin dynamics routine of the struct Simulation
    pub fn langevin_dynamics(&mut self, interface: &mut dyn QCInterface) {
        self.initialize_langevin(interface);

        for _step in 0..self.config.nstep {
            self.langevin_step(interface);
        }
    }

    /// Hopping and field coupling procedure. Includes the rescaling of the velocities.
    pub fn surface_hopping_step(&mut self, old_state: usize) {
        if self.config.gs_dynamic {
            // skip hopping procedure if the ground state is forced
        } else {
            let old_coeff: Array1<c64> = self.coefficients.clone();
            // integration of the schroedinger equation
            if !self.config.hopping_config.use_rk_integration {
                self.coefficients = self.matrix_exponential_integration();
            }
            else {
                // the coefficients are integrated using a 4th order runge kutta scheme
                self.coefficients = self.rk_integration();
            }

            // calculate the state of the simulation after the hopping procedure
            self.get_new_state(old_coeff.view());

            if self.config.hopping_config.decoherence_correction {
                self.coefficients = self.get_decoherence_correction(0.1);
            }
        }
        // Rescale the velocities after a population transfer
        if self.state != old_state {
            let tmp: (Array2<f64>, usize) = self.uniformly_rescaled_velocities(self.state);
            self.state = tmp.1;
            self.velocities = tmp.0;
        }
    }

    /// Initialize the velocity-verlet dynamic routine and print the first output of the dynamics simulation
    pub fn initialize_verlet(&mut self, interface: &mut dyn QCInterface) {
        if self.config.restart_flag {
            self.restart_trajectory(interface);
        } else {
            self.initiate_trajectory(interface);
        }

        // Print output
        self.print_data(None,true);

        // Calculate new coordinates from velocity-verlet
        self.coordinates = self.get_coord_verlet();
        // Shift coordinates to center of mass
        self.coordinates = self.shift_to_center_of_mass();
    }

    /// Calculate a single step of the velocity-verlet dynamics utilizing the [QCInterface]
    /// for the calculation of the required properties
    pub fn verlet_step(&mut self, interface: &mut dyn QCInterface) {
        let old_forces: Array2<f64> = self.forces.clone();
        let old_energy: f64 = self.energies[self.state] + self.kinetic_energy;
        let old_kinetic: f64 = self.kinetic_energy;
        let old_potential_energy: f64 = self.energies[self.state];
        let _last_energies: Array1<f64> = self.energies.clone();
        let old_state: usize = self.state;

        // calculate energies, forces, nonadiabatic_scalar
        // for the new geometry
        self.get_quantum_chem_data(interface);

        // surface hopping procedure
        if self.config.hopping_config.use_state_coupling {
            self.surface_hopping_step(old_state);
        }

        // Calculate new coordinates from velocity-verlet
        self.velocities = self.get_velocities_verlet(old_forces.view());
        // remove tranlation and rotation from the velocities
        self.velocities = self.eliminate_translation_rotation_from_velocity();

        // calculate the kinetic energy
        self.kinetic_energy = self.get_kinetic_energy();

        // scale velocities
        self.velocities = self
            .thermostat
            .scale_velocities(self.velocities.view(), self.kinetic_energy);

        if self.config.artificial_energy_conservation {
            self.velocities =
                self.scale_velocities_const_energy(old_state, old_kinetic, old_potential_energy);
        }

        // Print settings
        self.print_data(Some(old_energy),false);

        // Calculate new coordinates from velocity-verlet
        self.coordinates = self.get_coord_verlet();

        // Shift coordinates to center of mass
        self.coordinates = self.shift_to_center_of_mass();

        // update the actual time
        self.actual_time += self.stepsize;
    }

    /// Initialize the langevin dynamics and write the first output of the simulation
    pub fn initialize_langevin(&mut self, interface: &mut dyn QCInterface) {
        if self.config.restart_flag {
            self.restart_trajectory(interface);
        } else {
            self.initiate_trajectory(interface);
        }

        // Print settings
        self.print_data(None,true);

        // Langevin routine
        let (_vrand, prand): (Array2<f64>, Array2<f64>) = self.get_random_terms();
        self.saved_p_rand = prand;
        let efactor = self.get_e_factor_langevin();
        self.saved_efactor = efactor;

        // get new coordinates
        self.coordinates = self.get_coordinates_langevin();
        // Shift coordinates to center of mass
        self.coordinates = self.shift_to_center_of_mass();
    }

    /// Calculate a single step of the langevin dynamics
    pub fn langevin_step(&mut self, interface: &mut dyn QCInterface) {
        let old_forces: Array2<f64> = self.forces.clone();
        let old_energy: f64 = self.energies[self.state] + self.kinetic_energy;
        let _last_energies: Array1<f64> = self.energies.clone();
        let old_state: usize = self.state;

        // calculate energies, forces, nonadiabatic_scalar
        // for the new geometry
        self.get_quantum_chem_data(interface);

        if self.config.hopping_config.use_state_coupling {
            self.surface_hopping_step(old_state);
        }

        let (vrand, prand): (Array2<f64>, Array2<f64>) = self.get_random_terms();
        self.saved_p_rand = prand;
        let efactor = self.get_e_factor_langevin();
        self.saved_efactor = efactor;
        // calculate new velocities
        self.velocities = self.get_velocities_langevin(old_forces.view(), vrand.view());
        // remove translation and rotation from velocities
        self.velocities = self.eliminate_translation_rotation_from_velocity();
        // calculate kinetic energy
        self.kinetic_energy = self.get_kinetic_energy();

        // Print settings
        self.print_data(Some(old_energy),false);

        // calculate new coordinates
        self.coordinates = self.get_coordinates_langevin();

        // Shift coordinates to center of mass
        self.coordinates = self.shift_to_center_of_mass();

        self.actual_time += self.stepsize;
    }

    /// Calculate the energies, gradient, nonadiabatic couplings and the dipoles
    /// using the [QCInterface]
    pub fn get_quantum_chem_data(&mut self, interface: &mut dyn QCInterface) {
        // calculate energy, forces, etc for new coords
        let tmp: (
            Array1<f64>,
            Array2<f64>,
            Option<Array2<f64>>,
            Option<Array2<f64>>,
        ) = interface.compute_data(
            self.coordinates.view(),
            self.state,
            self.stepsize,
            self.config.hopping_config.use_state_coupling,
            self.config.gs_dynamic,
        );

        self.energies
            .slice_mut(s![..])
            .assign(&tmp.0.slice(s![..self.config.nstates]));
        let forces: Array2<f64> = tmp.1;
        for (idx, mass) in self.masses.iter().enumerate() {
            self.forces
                .slice_mut(s![idx, ..])
                .assign(&(-1.0 * &forces.slice(s![idx, ..]) / mass.to_owned()));
        }

        if self.config.hopping_config.use_state_coupling && self.config.gs_dynamic == false {
            self.nonadiabatic_scalar = tmp.2.unwrap();
            self.s_mat = tmp.3.unwrap();
        }
    }

    /// Initiate the trajectory
    pub fn initiate_trajectory(&mut self, interface: &mut dyn QCInterface) {
        self.coordinates = self.shift_to_center_of_mass();
        self.velocities = self.eliminate_translation_rotation_from_velocity();

        self.get_quantum_chem_data(interface);

        self.kinetic_energy = self.get_kinetic_energy();
    }

    /// Restart the trajectory
    pub fn restart_trajectory(&mut self, interface: &mut dyn QCInterface) {
        let temp: (Array2<f64>, Array2<f64>, Array2<f64>, Array1<c64>) = read_restart_parameters();
        self.coordinates = temp.0;
        self.velocities = temp.1;
        self.nonadiabatic_scalar = temp.2;
        self.coefficients = temp.3;

        // calculate quantum chemical data
        self.get_quantum_chem_data(interface);
        self.kinetic_energy = self.get_kinetic_energy();
    }
}
