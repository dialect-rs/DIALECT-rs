#![allow(dead_code)]
#![allow(warnings)]
#[macro_use]
use crate::constants;
use crate::defaults::*;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

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
fn default_load_velocities_from_file() -> bool {
    false
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
fn default_force_switch_s0s1_threshold() -> f64 {
    0.1
}
fn default_use_ehrenfest() -> bool {
    USE_EHRENFEST
}
fn default_use_surface_hopping() -> bool {
    false
}
fn default_use_nacv_couplings() -> bool {
    true
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
fn default_use_local_diabatisation() -> bool {
    true
}
fn default_use_rescaling_at_frustrated_hop() -> bool {
    false
}
fn default_print_coefficients() -> bool {
    PRINT_COEFFICIENTS
}
fn default_use_tab_decoherence() -> bool {
    false
}
fn default_tab_grad_threshold() -> f64 {
    1.0e-5
}
fn default_alpha_values() -> Vec<f64> {
    let vec: Vec<f64> = vec![4.7, 22.7, 19.8, 12.2];
    vec
}

fn default_alpha_atoms() -> Vec<usize> {
    let vec: Vec<usize> = vec![1, 6, 7, 8];
    vec
}

fn default_load_adiabatic_coefficients() -> bool {
    false
}
fn default_adiabatic_coefficients_file() -> String {
    String::from("adiabatic_coefficients_state_0.npy")
}
fn default_ehrenfest_configuration() -> EhrenfestConfiguration {
    let config: EhrenfestConfiguration = toml::from_str("").unwrap();
    config
}
fn default_nonadiabatic_configuration() -> NonadiabaticConfiguration {
    let config: NonadiabaticConfiguration = toml::from_str("").unwrap();
    config
}
fn default_ehrenfest_decoherence() -> EhrenfestDecoherence {
    let config: EhrenfestDecoherence = toml::from_str("").unwrap();
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
    #[serde(default = "default_use_surface_hopping")]
    pub use_surface_hopping: bool,
    #[serde(default = "default_use_ehrenfest")]
    pub use_ehrenfest: bool,
    #[serde(default = "default_load_velocities_from_file")]
    pub load_velocities_from_file: bool,
    #[serde(default = "default_use_boltzmann_velocities")]
    pub use_boltzmann_velocities: bool,
    #[serde(default = "default_artificial_energy_conservation")]
    pub artificial_energy_conservation: bool,
    #[serde(default = "default_ehrenfest_configuration")]
    pub ehrenfest_config: EhrenfestConfiguration,
    #[serde(default = "default_ehrenfest_decoherence")]
    pub ehrenfest_decoherence: EhrenfestDecoherence,
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
        // Read the user-facing configuration file (writing the commented
        // default template first if it does not exist) and convert it to
        // the internal representation.
        crate::initialization::user_config::load_dynamics_config(Path::new(CONFIG_FILE_NAME))
    }
}

/// Structs that holds the parameters for the Ehrenfest routine
#[derive(Serialize, Deserialize, Clone)]
pub struct EhrenfestConfiguration {
    #[serde(default = "default_use_state_coupling")]
    pub use_state_coupling: bool,
    #[serde(default = "default_state_threshold")]
    pub state_threshold: f64,
    #[serde(default = "default_use_tab_decoherence")]
    pub use_tab_decoherence: bool,
    #[serde(default = "default_tab_grad_threshold")]
    pub tab_grad_threshold: f64,
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
    /// If true, initialize the electronic (diabatic-basis) coefficients of
    /// the Ehrenfest wavefunction from a saved FMO-LC-TDDFTB adiabatic
    /// eigenvector instead of from `initial_state`. The `.npy` file holds
    /// the (signed) eigenvector coefficients (as written by
    /// `save_adiabatic_coefficients`), which become the initial electronic
    /// amplitudes directly, so the trajectory starts in the chosen
    /// adiabatic state with the correct relative phases.
    #[serde(default = "default_load_adiabatic_coefficients")]
    pub load_adiabatic_coefficients: bool,
    /// Path to the saved adiabatic-coefficients `.npy` file used when
    /// `load_adiabatic_coefficients` is true.
    #[serde(default = "default_adiabatic_coefficients_file")]
    pub adiabatic_coefficients_file: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct EhrenfestDecoherence {
    #[serde(default = "default_alpha_atoms")]
    pub alpha_atoms: Vec<usize>,
    #[serde(default = "default_alpha_values")]
    pub alpha_values: Vec<f64>,
}

/// Structs that holds the parameters for the Nonadiabatic couplings
#[derive(Serialize, Deserialize, Clone)]
pub struct NonadiabaticConfiguration {
    #[serde(default = "default_use_nacv_couplings")]
    pub use_nacv_couplings: bool,
}

/// Structs that holds the parameters for the surface hopping routines
#[derive(Serialize, Deserialize, Clone)]
pub struct HoppingConfiguration {
    #[serde(default = "default_force_switch_to_gs")]
    pub force_switch_to_gs: bool,
    #[serde(default = "default_force_switch_s0s1_threshold")]
    pub force_switch_s0s1_threshold: f64,
    #[serde(default = "default_decoherence_correction")]
    pub decoherence_correction: bool,
    #[serde(default = "default_rk_integration")]
    pub use_rk_integration: bool,
    #[serde(default = "default_use_local_diabatisation")]
    pub use_local_diabatisation: bool,
    #[serde(default = "default_use_rescaling_at_frustrated_hop")]
    pub use_rescaling_at_frustrated_hop: bool,
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
    #[serde(default = "default_print_state")]
    pub print_state: bool,
}

#[cfg(test)]
mod ehrenfest_load_tests {
    use super::EhrenfestConfiguration;

    #[test]
    fn adiabatic_load_defaults_off() {
        let cfg: EhrenfestConfiguration = toml::from_str("").unwrap();
        assert!(!cfg.load_adiabatic_coefficients);
        assert_eq!(cfg.adiabatic_coefficients_file, "adiabatic_coefficients_state_0.npy");
    }

    #[test]
    fn adiabatic_load_parsing() {
        let cfg: EhrenfestConfiguration = toml::from_str(
            "load_adiabatic_coefficients = true\n\
             adiabatic_coefficients_file = \"adiabatic_coefficients_state_3.npy\"\n",
        )
        .unwrap();
        assert!(cfg.load_adiabatic_coefficients);
        assert_eq!(cfg.adiabatic_coefficients_file, "adiabatic_coefficients_state_3.npy");
    }
}
