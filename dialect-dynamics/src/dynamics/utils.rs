use crate::initialization::Simulation;
use crate::output::*;
use ndarray::Array1;
use ndarray_npy::write_npy;

impl Simulation {
    pub fn print_data(&mut self, old_energy: Option<f64>, first_call: bool) {
        if self.config.print_config.print_restart {
            // Write Output in each step
            let restart: RestartOutput = RestartOutput::new(
                self.n_atoms,
                self.coordinates.view(),
                self.velocities.view(),
                self.nonadiabatic_scalar.view(),
                self.coefficients.view(),
            );
            write_restart(&restart);
        }

        if self.config.print_config.print_coordinates {
            let xyz_output: XyzOutput = XyzOutput::new(
                self.n_atoms,
                self.coordinates.view(),
                self.atomic_numbers.clone(),
            );
            write_xyz_custom(&xyz_output, first_call);
        }

        if self.config.print_config.print_standard {
            let total_energy: f64 = self.kinetic_energy + self.energies[self.state];
            let old_energy: f64 =
                if let Some(old_energy) = old_energy {
                    old_energy
                } else {
                    0.0
                };
            let energy_diff: f64 = total_energy - old_energy;

            let full: StandardOutput = StandardOutput::new(
                self.actual_time,
                self.coordinates.view(),
                self.velocities.view(),
                self.kinetic_energy,
                self.energies[self.state],
                total_energy,
                energy_diff,
                self.forces.view(),
                self.state,
            );
            write_full_custom(&full, self.masses.view(), first_call);
        }
        if self.config.print_config.print_energies {
            write_energies(self.energies.view(), first_call);
        }
        if self.config.print_config.print_state {
            write_state(self.state, first_call);
        }
        if self.config.print_config.print_temperature {
            let temperature: f64 = self.thermostat.get_temperature(self.kinetic_energy);
            write_temperature(temperature, first_call);
        }

        if self.config.hopping_config.use_state_coupling && self.config.print_config.print_hopping {
            let hopping_out: HoppingOutput = HoppingOutput::new(
                self.actual_time,
                self.coefficients.view(),
                self.nonadiabatic_scalar.view(),
            );
            write_hopping(&hopping_out, first_call);
        }
    }

    pub fn print_ehrenfest_data(&mut self, old_energy: Option<f64>, first_call: bool, step: usize) {
        if self.config.print_config.print_restart {
            // Write Output in each step
            let restart: RestartOutput = RestartOutput::new(
                self.n_atoms,
                self.coordinates.view(),
                self.velocities.view(),
                self.nonadiabatic_scalar.view(),
                self.coefficients.view(),
            );
            write_restart(&restart);
        }

        if self.config.print_config.print_coordinates {
            let xyz_output: XyzOutput = XyzOutput::new(
                self.n_atoms,
                self.coordinates.view(),
                self.atomic_numbers.clone(),
            );
            write_xyz_custom(&xyz_output, first_call);
        }

        if self.config.print_config.print_standard {
            let total_energy: f64 = self.kinetic_energy + self.energies[self.state];
            let old_energy: f64 =
                if let Some(old_energy) = old_energy {
                    old_energy
                } else {
                    0.0
                };
            let energy_diff: f64 = total_energy - old_energy;

            let full: StandardOutput = StandardOutput::new(
                self.actual_time,
                self.coordinates.view(),
                self.velocities.view(),
                self.kinetic_energy,
                self.energies[self.state],
                total_energy,
                energy_diff,
                self.forces.view(),
                self.state,
            );
            write_full_custom(&full, self.masses.view(), first_call);
        }
        if self.config.print_config.print_energies {
            write_energies(self.energies.view(), first_call);
        }
        if self.config.print_config.print_temperature {
            let temperature: f64 = self.thermostat.get_temperature(self.kinetic_energy);
            write_temperature(temperature, first_call);
        }

        if self.config.ehrenfest_config.use_ehrenfest {
            let out: EhrenfestOutput = EhrenfestOutput::new(self.coefficients.view());
            self.coeff_writer
                .add_array(step.to_string(), &out.coefficients_abs)
                .unwrap();
            if self.config.ehrenfest_config.print_coefficients {
                let abs_coefficients: Array1<f64> = self.coefficients.map(|val| val.norm_sqr());
                let mut string: String = String::from("coefficients_");
                string = string + &step.to_string();
                string = string + &String::from(".npy");
                write_npy(string, &abs_coefficients).unwrap();
            }
            // write_ehrenfest(&out, first_call);
        }
    }
}
