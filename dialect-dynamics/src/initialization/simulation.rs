use crate::constants;
use crate::dynamics::thermostat::{BerendsenThermostat, NullThermostat, Thermostat};
use crate::initialization::system::SystemData;
use crate::initialization::velocities::*;
use crate::initialization::DynamicConfiguration;
use ndarray::prelude::*;
use ndarray_linalg::c64;
use ndarray_npy::read_npy;
use ndarray_npy::NpzWriter;
use rand::prelude::*;
use std::fs::File;

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
    pub force_array: Array2<f64>,
    pub energies: Array1<f64>,
    pub nonadiabatic_scalar: Array2<f64>,
    pub nonadiabatic_vectors: Option<Vec<Array1<f64>>>,
    pub s_mat: Array2<f64>,
    pub state: usize,
    pub saved_p_rand: Array2<f64>,
    pub saved_efactor: Array1<f64>,
    pub t_tot_last: Option<Array2<f64>>,
    pub hdiab: Array2<f64>,
    pub thermostat: Box<dyn Thermostat>,
    pub coeff_writer: NpzWriter<File>,
    pub alpha_values: Option<Array1<f64>>,
    pub rng: StdRng,
}

impl Simulation {
    /// Initialize the struct [Simulation] from the [SystemData]
    /// Create all required arrays and initialize the velocities
    pub fn new(system: &SystemData) -> Simulation {
        let mut config: DynamicConfiguration = system.config.clone();
        let stepsize_au: f64 = config.stepsize * constants::FS_TO_AU;

        // initialize coefficients
        let mut coefficients: Array1<c64> = Array1::zeros(config.nstates);
        if config.ehrenfest_config.load_adiabatic_coefficients {
            // Start the Ehrenfest wavefunction from a saved FMO-LC-TDDFTB
            // adiabatic eigenvector. The `.npy` file stores the (signed)
            // eigenvector coefficients — the amplitudes of the adiabatic
            // state over the diabatic basis — which become the initial
            // electronic amplitudes directly (preserving relative phases).
            let file: &str = &config.ehrenfest_config.adiabatic_coefficients_file;
            let raw: Array1<f64> = read_npy(file).unwrap_or_else(|e| {
                panic!("Failed to read adiabatic coefficients from '{file}': {e}")
            });
            assert_eq!(
                raw.len(),
                config.nstates,
                "adiabatic coefficient file '{}' has length {} but nstates = {}",
                file,
                raw.len(),
                config.nstates
            );
            for (i, &c) in raw.iter().enumerate() {
                coefficients[i] = c64::from(c);
            }
            // Renormalize defensively (Davidson eigenvectors converge to a
            // finite residual, so Σc² may be ~1 ± ε).
            let norm: f64 = coefficients.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            if norm > 0.0 {
                coefficients.mapv_inplace(|c| c / norm);
            }
        } else {
            for val in config.initial_state.iter() {
                let initial_states_length: f64 = config.initial_state.len() as f64;
                coefficients[*val] = c64::from(1.0 / initial_states_length.sqrt());
            }
        }
        // calculate total mass of the system
        let total_mass: f64 = system.masses.sum();

        // force gs dynamics if initial state is 0 — but not when the
        // electronic coefficients are loaded from a saved adiabatic
        // eigenvector (the populated state comes from the file, not
        // `initial_state`).
        if config.initial_state[0] == 0 && !config.ehrenfest_config.load_adiabatic_coefficients {
            config.gs_dynamic = true;
        }

        // initiate parameters
        let last_forces: Array3<f64> = Array3::zeros((3, system.n_atoms, 3));
        let forces: Array2<f64> = Array2::zeros((system.n_atoms, 3));
        let energies: Array1<f64> = Array1::zeros(config.nstates);
        let force_array: Array2<f64> = Array2::zeros([3 * system.n_atoms, config.nstates]);
        let nonad_scalar: Array2<f64> = Array2::zeros((config.nstates, config.nstates));
        let s_mat: Array2<f64> = Array2::zeros((config.nstates, config.nstates));
        let efactor: Array1<f64> = Array1::zeros(system.n_atoms);
        let saved_p_rand: Array2<f64> = Array2::zeros((system.n_atoms, 3));
        let hdiab = Array2::zeros((config.nstates, config.nstates));

        // set friction
        let mut friction: Array1<f64> = Array1::ones(system.n_atoms);
        friction *= config.langevin_config.friction;

        // initialize velocities from boltzmann distribution
        let velocities = if config.load_velocities_from_file {
            read_npy("velocities.npy").unwrap()
        } else if config.use_boltzmann_velocities {
            initialize_velocities(system, config.thermostat_config.temperature)
        } else {
            Array2::zeros((system.n_atoms, 3))
        };

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
        // create Npz writer
        let npz_writer = NpzWriter::new_compressed(File::create("coeff_abs.npz").unwrap());

        // get the standard rng
        // let rng: StdRng = StdRng::seed_from_u64(1);
        let rng: StdRng = StdRng::from_entropy();

        // create the alpha value array
        // Gaussian widths for the calculation of the decoherence time constant for TAB
        // Source: https://doi.org/10.1016/j.chemphys.2010.03.020
        let alpha_array: Option<Array1<f64>> = if config.ehrenfest_config.use_tab_decoherence {
            let mut alphas: Array1<f64> = Array1::zeros(system.n_atoms);
            for (idx, atom) in system.atomic_numbers.iter().enumerate() {
                // get the alpha value
                let atom_tmp: usize = *atom as usize;
                for (alpha_idx, val) in config.ehrenfest_decoherence.alpha_atoms.iter().enumerate()
                {
                    if *val == atom_tmp {
                        alphas[idx] = config.ehrenfest_decoherence.alpha_values[alpha_idx];
                    }
                }
                //alphas[idx] = constants::ALPHA_GAUSSIAN_WIDTHS[atom];
            }
            Some(alphas)
        } else {
            None
        };

        Simulation {
            state: config.initial_state[0],
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
            force_array,
            nonadiabatic_scalar: nonad_scalar,
            nonadiabatic_vectors: None,
            s_mat,
            saved_efactor: efactor,
            saved_p_rand,
            t_tot_last: None,
            hdiab,
            thermostat,
            coeff_writer: npz_writer,
            rng,
            alpha_values: alpha_array,
        }
    }
}
