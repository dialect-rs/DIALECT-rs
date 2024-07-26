use crate::initialization::restart::read_restart_parameters;
use crate::initialization::Simulation;
use crate::interface::QCInterface;
use ndarray::prelude::*;
use ndarray_linalg::c64;
use std::ops::AddAssign;

impl Simulation {
    ///Ehrenfest dynamics routine of the struct Simulation
    pub fn ehrenfest_dynamics(&mut self, interface: &mut dyn QCInterface) {
        self.initialize_ehrenfest(interface, 0);
        for step in 1..self.config.nstep {
            self.ehrenfest_step(interface, step);
        }
    }

    pub fn ehrenfest_step(&mut self, interface: &mut dyn QCInterface, step: usize) {
        let old_forces: Array2<f64> = self.forces.clone();
        let old_energy: f64 = self.energies[self.state] + self.kinetic_energy;
        // calculate the gradient and the excitonic couplings
        let excitonic_couplings: Array2<f64> = self.get_ehrenfest_data(interface, step);

        // ehrenfest procedure
        if self.config.ehrenfest_config.use_state_couplings {
            self.coefficients = self.ehrenfest_matrix_exponential_nacme(excitonic_couplings.view());
        } else if self.config.ehrenfest_config.use_rk_integration {
            self.coefficients = self.ehrenfest_rk(excitonic_couplings.view());
        } else {
            self.coefficients = self.ehrenfest_matrix_exponential_2(excitonic_couplings.view());
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

        // Print settings
        self.print_ehrenfest_data(Some(old_energy), false, step);

        // Calculate new coordinates from velocity-verlet
        self.coordinates = self.get_coord_verlet();

        // Shift coordinates to center of mass
        self.coordinates = self.shift_to_center_of_mass();
    }

    pub fn initialize_ehrenfest(&mut self, interface: &mut dyn QCInterface, step: usize) {
        if self.config.restart_flag {
            self.restart_trajectory_ehrenfest(interface, step);
            // Print settings
            self.print_ehrenfest_data(None, false, 0);
        } else {
            self.initiate_ehrenfest_trajectory(interface, step);
            // Print settings
            self.print_ehrenfest_data(None, true, 0);
        }

        // Calculate new coordinates from velocity-verlet
        self.coordinates = self.get_coord_verlet();
        // Shift coordinates to center of mass
        self.coordinates = self.shift_to_center_of_mass();
    }

    /// Initiate the trajectory
    pub fn initiate_ehrenfest_trajectory(&mut self, interface: &mut dyn QCInterface, step: usize) {
        // remove COM from coordinates
        self.coordinates = self.shift_to_center_of_mass();
        self.initial_coordinates = self.coordinates.clone();
        // remove tranlation and rotation
        self.velocities = self.eliminate_translation_rotation_from_velocity();
        // do the first calculation using the QuantumChemistryInterface
        self.get_ehrenfest_data(interface, step);

        // calculate the kinetic energy
        self.kinetic_energy = self.get_kinetic_energy();
    }

    /// Restart the trajectory
    pub fn restart_trajectory_ehrenfest(&mut self, interface: &mut dyn QCInterface, step: usize) {
        let temp: (Array2<f64>, Array2<f64>, Array2<f64>, Array1<c64>) = read_restart_parameters();
        self.coordinates = temp.0;
        self.initial_coordinates = self.coordinates.clone();
        self.velocities = temp.1;
        self.nonadiabatic_scalar = temp.2;
        self.coefficients = temp.3;

        // calculate quantum chemical data
        self.get_ehrenfest_data(interface, step);
        self.kinetic_energy = self.get_kinetic_energy();
    }

    pub fn get_ehrenfest_data(
        &mut self,
        interface: &mut dyn QCInterface,
        step: usize,
    ) -> Array2<f64> {
        let abs_coefficients: Array1<f64> = self.coefficients.map(|val| val.norm_sqr());
        let tmp: (f64, Array2<f64>, Array2<f64>, Array2<f64>) = interface.compute_ehrenfest(
            self.coordinates.view(),
            self.velocities.view(),
            abs_coefficients.view(),
            self.config.ehrenfest_config.state_threshold,
            self.config.stepsize,
            step,
            self.config.ehrenfest_config.use_state_couplings,
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

        if self.config.ehrenfest_config.use_restraint {
            self.apply_harmonic_restraint();
        }
        // diabatic_couplings
        excitonic_couplings
    }

    pub fn apply_harmonic_restraint(&mut self) {
        // calculate the force constant in au. The value of the config is in kcal/mol
        let force_constant: f64 = self.config.ehrenfest_config.force_constant * 0.00159362;
        // calculate the forces for the deviation from the initial coordinates
        let forces: Array2<f64> = -force_constant * (&self.coordinates - &self.initial_coordinates);

        for ((idx, force), _z_at) in forces
            .outer_iter()
            .enumerate()
            .zip(self.atomic_numbers.iter())
        {
            self.forces.slice_mut(s![idx, ..]).add_assign(&force);
        }
    }
}
