use crate::constants;
use crate::dynamics::thermostat::BerendsenThermostat;
use crate::dynamics::thermostat::NullThermostat;
use crate::dynamics::thermostat::Thermostat;
use crate::initialization::system::SystemData;
use crate::initialization::velocities::*;
use crate::initialization::DynamicConfiguration;
use ndarray::prelude::*;
use ndarray_linalg::c64;

/// Struct that holds the [DynamicConfiguration] and the other necessary
/// arguments, which are required for the molecular dynamics
pub struct Simulation {
    pub stepsize: f64,
    pub actual_time: f64,
    pub total_mass: f64,
    pub config: DynamicConfiguration,
    pub coefficients: Array1<c64>,
    pub coordinates: Array2<f64>,
    pub initial_coordinates: Array2<f64>,
    pub masses: Array1<f64>,
    pub velocities: Array2<f64>,
    pub kinetic_energy: f64,
    pub n_atoms: usize,
    pub atomic_numbers: Vec<u8>,
    pub last_forces: Array3<f64>,
    pub friction: Array1<f64>,
    pub forces: Array2<f64>,
    pub energies: Array1<f64>,
    pub nonadiabatic_scalar: Array2<f64>,
    pub s_mat: Array2<f64>,
    pub state: usize,
    pub saved_p_rand: Array2<f64>,
    pub saved_efactor: Array1<f64>,
    pub t_tot_last: Option<Array2<f64>>,
    pub hdiab: Array2<f64>,
    pub thermostat: Box<dyn Thermostat>,
}

impl Simulation {
    /// Initialize the struct [Simulation] from the [SystemData]
    /// Create all required arrays and initialize the velocities
    pub fn new(system: &SystemData) -> Simulation {
        let mut config: DynamicConfiguration = system.config.clone();
        let stepsize_au: f64 = config.stepsize * constants::FS_TO_AU;

        // initialize coefficients
        let mut coefficients: Array1<c64> = Array1::zeros(config.nstates);
        coefficients[config.initial_state] = c64::from(1.0);
        // calculate total mass of the system
        let total_mass: f64 = system.masses.sum();

        // force gs dynamics if initial state is 0
        if config.initial_state == 0 {
            config.gs_dynamic = true;
        }

        // initiate parameters
        let last_forces: Array3<f64> = Array3::zeros((3, system.n_atoms, 3));
        let forces: Array2<f64> = Array2::zeros((system.n_atoms, 3));
        let energies: Array1<f64> = Array1::zeros(config.nstates);
        let nonad_scalar: Array2<f64> = Array2::zeros((config.nstates, config.nstates));
        let s_mat: Array2<f64> = Array2::zeros((config.nstates, config.nstates));
        let efactor: Array1<f64> = Array1::zeros(system.n_atoms);
        let saved_p_rand: Array2<f64> = Array2::zeros((system.n_atoms, 3));
        let hdiab = Array2::zeros((config.nstates, config.nstates));

        // set friction
        let mut friction: Array1<f64> = Array1::ones(system.n_atoms);
        friction *= config.langevin_config.friction;

        // initialize velocities from boltzmann distribution
        let velocities = initialize_velocities(system, config.thermostat_config.temperature);

        let thermostat: Box<dyn Thermostat> = if !config.thermostat_config.use_thermostat {
            Box::new(NullThermostat::new(system.n_atoms))
        } else {
            Box::new(BerendsenThermostat::new(
                config.thermostat_config.time_coupling * constants::FS_TO_AU,
                stepsize_au,
                system.n_atoms,
                config.thermostat_config.temperature,
            ))
        };

        Simulation {
            state: config.initial_state,
            actual_time: 0.0,
            stepsize: stepsize_au,
            total_mass,
            config,
            coefficients,
            coordinates: system.coordinates.clone(),
            initial_coordinates: Array2::zeros(system.coordinates.raw_dim()),
            masses: system.masses.clone(),
            velocities,
            kinetic_energy: 0.0,
            n_atoms: system.n_atoms,
            atomic_numbers: system.atomic_numbers.clone(),
            last_forces,
            friction,
            forces,
            energies,
            nonadiabatic_scalar: nonad_scalar,
            s_mat,
            saved_efactor: efactor,
            saved_p_rand,
            t_tot_last: None,
            hdiab: hdiab,
            thermostat,
        }
    }
}
