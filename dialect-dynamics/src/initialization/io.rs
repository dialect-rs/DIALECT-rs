#![allow(dead_code)]
#![allow(warnings)]
#[macro_use]
use clap::crate_version;
use crate::constants;
use crate::defaults::*;
use chemfiles::{Frame, Trajectory};
use clap::App;
use log::{debug, error, info, trace, warn};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::Path;
use std::ptr::eq;
use std::{env, fs};

fn default_nstep() -> usize {
    NSTEP
}
fn default_stepsize() -> f64 {
    STEPSIZE
}
fn default_temperature() -> f64 {
    TEMPERATURE
}
fn default_friction() -> f64 {
    FRICTION
}
fn default_restart_flag() -> bool {
    RESTARTFLAG
}
fn default_print_coupling() -> bool {
    PRINT_COUPLING
}
fn default_initial_state() -> Vec<usize> {
    vec![INITIAL_STATE]
}
fn default_nstates() -> usize {
    NSTATES
}
fn default_use_state_coupling() -> bool {
    USE_STATE_COUPLING
}
fn default_force_switch_to_gs() -> bool {
    FORCE_SWITCH_TO_GS
}
fn default_artificial_energy_conservation() -> bool {
    ARTIFICIAL_ENERGY_CONSERVATION
}
fn default_use_boltzmann_velocities() -> bool {
    USE_BOLTZMANN_VELOCITIES
}
fn default_gs_dynamic() -> bool {
    GS_DYNAMIC
}
fn default_decoherence_correction() -> bool {
    DECOHERENCE_CORRECTION
}
fn default_time_coupling() -> f64 {
    TIME_COUPLING
}
fn default_rk_integration() -> bool {
    RK_INTEGRATION
}
fn default_integration_steps() -> usize {
    INTEGRATION_STEPS
}
fn default_hopping_config() -> HoppingConfiguration {
    let hopping_config: HoppingConfiguration = toml::from_str("").unwrap();
    return hopping_config;
}
fn default_use_thermostat() -> bool {
    USE_THERMOSTAT
}
fn default_thermostat_config() -> ThermostatConfiguration {
    let thermostat_config: ThermostatConfiguration = toml::from_str("").unwrap();
    thermostat_config
}
fn default_use_langevin() -> bool {
    USE_LANGEVIN
}
fn default_langevin_config() -> LangevinConfiguration {
    let langevin: LangevinConfiguration = toml::from_str("").unwrap();
    langevin
}
fn default_print_restart() -> bool {
    PRINT_RESTART
}
fn default_print_coordinates() -> bool {
    PRINT_COORDINATES
}
fn default_print_energies() -> bool {
    PRINT_ENERGIES
}
fn default_print_temperature() -> bool {
    PRINT_TEMPERATURES
}
fn default_print_standard() -> bool {
    PRINT_STANDARD
}
fn default_print_hopping() -> bool {
    PRINT_HOPPING
}
fn default_print_state() -> bool {
    PRINT_STATE
}
fn default_print_configuration() -> PrintConfiguration {
    let config: PrintConfiguration = toml::from_str("").unwrap();
    config
}
fn default_use_ehrenfest() -> bool {
    USE_EHRENFEST
}
fn default_use_state_couplings() -> bool {
    USE_EHRENFEST
}
fn default_use_nacv_couplings() -> bool {
    true
}
fn default_use_nact_couplings() -> bool {
    USE_EHRENFEST
}
fn default_state_threshold() -> f64 {
    STATE_THRESHOLD
}
fn default_use_restraint() -> bool {
    USE_RESTRAINT
}
fn default_force_constant() -> f64 {
    FORCE_CONSTANT
}
fn default_use_rk_integration() -> bool {
    USE_RK_INTEGRATION
}
fn default_print_coefficients() -> bool {
    PRINT_COEFFICIENTS
}
fn default_ehrenfest_configuration() -> EhrenfestConfiguration {
    let config: EhrenfestConfiguration = toml::from_str("").unwrap();
    config
}
fn default_nonadiabatic_configuration() -> NonadiabaticConfiguration {
    let config: NonadiabaticConfiguration = toml::from_str("").unwrap();
    config
}

/// Struct that loads the configuration of the dynamics from the file "fish.toml"
/// It holds the structs [HoppingConfiguration] and  [PulseConfigration]
#[derive(Serialize, Deserialize, Clone)]
pub struct DynamicConfiguration {
    #[serde(default = "default_nstep")]
    pub nstep: usize,
    #[serde(default = "default_stepsize")]
    pub stepsize: f64,
    #[serde(default = "default_restart_flag")]
    pub restart_flag: bool,
    #[serde(default = "default_initial_state")]
    pub initial_state: Vec<usize>,
    #[serde(default = "default_nstates")]
    pub nstates: usize,
    #[serde(default = "default_gs_dynamic")]
    pub gs_dynamic: bool,
    #[serde(default = "default_use_boltzmann_velocities")]
    pub use_boltzmann_velocities: bool,
    #[serde(default = "default_artificial_energy_conservation")]
    pub artificial_energy_conservation: bool,
    #[serde(default = "default_ehrenfest_configuration")]
    pub ehrenfest_config: EhrenfestConfiguration,
    #[serde(default = "default_hopping_config")]
    pub hopping_config: HoppingConfiguration,
    #[serde(default = "default_nonadiabatic_configuration")]
    pub nonadibatic_config: NonadiabaticConfiguration,
    #[serde(default = "default_thermostat_config")]
    pub thermostat_config: ThermostatConfiguration,
    #[serde(default = "default_langevin_config")]
    pub langevin_config: LangevinConfiguration,
    #[serde(default = "default_print_configuration")]
    pub print_config: PrintConfiguration,
}

impl DynamicConfiguration {
    pub fn new() -> Self {
        // read the configuration file, if it does not exist in the directory
        // the program initializes the default settings and writes an configuration file
        // to the directory
        let config_file_path: &Path = Path::new(CONFIG_FILE_NAME);
        let mut config_string: String = if config_file_path.exists() {
            fs::read_to_string(config_file_path).expect("Unable to read config file")
        } else {
            String::from("")
        };
        // load the configration settings
        let config: Self = toml::from_str(&config_string).unwrap();
        // save the configuration file if it does not exist already
        if config_file_path.exists() == false {
            config_string = toml::to_string(&config).unwrap();
            fs::write(config_file_path, config_string).expect("Unable to write config file");
        }
        return config;
    }
}

/// Structs that holds the parameters for the Ehrenfest routine
#[derive(Serialize, Deserialize, Clone)]
pub struct EhrenfestConfiguration {
    #[serde(default = "default_use_ehrenfest")]
    pub use_ehrenfest: bool,
    #[serde(default = "default_use_state_couplings")]
    pub use_state_couplings: bool,
    #[serde(default = "default_state_threshold")]
    pub state_threshold: f64,
    #[serde(default = "default_use_restraint")]
    pub use_restraint: bool,
    #[serde(default = "default_force_constant")]
    pub force_constant: f64,
    #[serde(default = "default_use_rk_integration")]
    pub use_rk_integration: bool,
    #[serde(default = "default_integration_steps")]
    pub integration_steps: usize,
    #[serde(default = "default_print_coefficients")]
    pub print_coefficients: bool,
}

/// Structs that holds the parameters for the Nonadiabatic couplings
#[derive(Serialize, Deserialize, Clone)]
pub struct NonadiabaticConfiguration {
    #[serde(default = "default_use_nacv_couplings")]
    pub use_nacv_couplings: bool,
    #[serde(default = "default_use_nact_couplings")]
    pub use_nact_couplings: bool,
}

/// Structs that holds the parameters for the surface hopping routines
#[derive(Serialize, Deserialize, Clone)]
pub struct HoppingConfiguration {
    #[serde(default = "default_use_state_coupling")]
    pub use_state_coupling: bool,
    #[serde(default = "default_force_switch_to_gs")]
    pub force_switch_to_gs: bool,
    #[serde(default = "default_decoherence_correction")]
    pub decoherence_correction: bool,
    #[serde(default = "default_rk_integration")]
    pub use_rk_integration: bool,
    #[serde(default = "default_integration_steps")]
    pub integration_steps: usize,
}

/// Struct that holds the parameters for the Thermostat
#[derive(Serialize, Deserialize, Clone)]
pub struct ThermostatConfiguration {
    #[serde(default = "default_use_thermostat")]
    pub use_thermostat: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_time_coupling")]
    pub time_coupling: f64,
}

/// Struct that holds the parameters for the Langevin dynamics
#[derive(Serialize, Deserialize, Clone)]
pub struct LangevinConfiguration {
    #[serde(default = "default_use_langevin")]
    pub use_langevin: bool,
    #[serde(default = "default_friction")]
    pub friction: f64,
}

/// Struct that controls the output of the simulation
#[derive(Serialize, Deserialize, Clone)]
pub struct PrintConfiguration {
    #[serde(default = "default_print_restart")]
    pub print_restart: bool,
    #[serde(default = "default_print_coordinates")]
    pub print_coordinates: bool,
    #[serde(default = "default_print_energies")]
    pub print_energies: bool,
    #[serde(default = "default_print_temperature")]
    pub print_temperature: bool,
    #[serde(default = "default_print_standard")]
    pub print_standard: bool,
    #[serde(default = "default_print_hopping")]
    pub print_hopping: bool,
    #[serde(default = "default_print_state")]
    pub print_state: bool,
}

/// Read a xyz-geometry file like .xyz or .pdb and returns a [Frame](chemfiles::Frame)
pub fn read_file_to_frame(filename: &str) -> Frame {
    // read the geometry file
    let mut trajectory = Trajectory::open(filename, 'r').unwrap();
    let mut frame = Frame::new();
    // if multiple geometries are contained in the file, we will only use the first one
    trajectory.read(&mut frame).unwrap();
    return frame;
}

/// Extract the atomic numbers and positions (in bohr) from a [Frame](chemfiles::frame)
pub fn frame_to_coordinates(frame: Frame) -> (Vec<u8>, Array2<f64>) {
    let mut positions: Array2<f64> =
        Array2::from_shape_vec(
            (frame.size() as usize, 3),
            frame
                .positions()
                .iter()
                .flat_map(|array| array.iter())
                .cloned()
                .collect(),
        )
        .unwrap();
    // transform the coordinates from angstrom to bohr
    positions = positions / constants::BOHR_TO_ANGS;
    // read the atomic number of each coordinate
    let atomic_numbers: Vec<u8> = (0..frame.size() as u64)
        .map(|i| frame.atom(i as usize).atomic_number() as u8)
        .collect();

    return (atomic_numbers, positions);
}
