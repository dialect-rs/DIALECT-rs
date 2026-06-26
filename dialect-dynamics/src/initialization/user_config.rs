//! User-facing dynamics configuration (`dynamics.toml`).
//!
//! This module owns the *file format* of the dynamics configuration: a
//! grouped, enum-based layout whose allowed values are documented in the
//! commented template written when no `dynamics.toml` exists. After parsing,
//! [`DynamicsUserConfig`] is converted into the internal
//! [`DynamicConfiguration`], so the dynamics code itself is independent of
//! the file format.

use crate::defaults::*;
use crate::initialization::io::{
    DynamicConfiguration, EhrenfestConfiguration, EhrenfestDecoherence, HoppingConfiguration,
    LangevinConfiguration, NonadiabaticConfiguration, PrintConfiguration, ThermostatConfiguration,
};
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// The default `dynamics.toml` written when no configuration file exists.
/// Every enum key lists its allowed values in the comments; the values in
/// this template are the defaults.
pub const DYNAMICS_TOML_TEMPLATE: &str = r#"# =====================================================================
#  DIALECT dynamics configuration
#  The allowed values of every option are listed in the comments.
# =====================================================================

# Type of the molecular dynamics simulation.
# Allowed: "ground_state"    - Born-Oppenheimer dynamics on S0
#          "excited_state"   - Born-Oppenheimer dynamics on the initial
#                              state, without hopping
#          "surface_hopping" - Tully surface-hopping dynamics
#          "ehrenfest"       - Ehrenfest (mean-field) dynamics
dynamics_type = "ground_state"

# number of nuclear steps
nstep = 1000
# nuclear time step in fs
stepsize = 0.1
# restart the trajectory from dynamics_restart.out
restart = false
# number of electronic states in the dynamics (including the ground
# state). With overlap couplings this must equal the number of computed
# excited states + 1; with NACV couplings it can be smaller (nstates <
# number of computed roots is recommended there).
nstates = 1

# coupling type between the electronic states in nonadiabatic dynamics
# Allowed: "nacv"    - nonadiabatic coupling vectors
#          "overlap" - scalar couplings from wavefunction overlaps
coupling = "nacv"

[initial_conditions]
# initial electronic state(s); 0 = ground state. More than one entry
# starts the trajectory in an equal superposition of the given states.
states = [0]
# initial velocities
# Allowed: "zero", "boltzmann" (sampled at `temperature`), "file"
velocities = "boltzmann"
# sampling temperature in K (only used for velocities = "boltzmann";
# with an active thermostat its target temperature is used instead)
temperature = 300.0

# ----- integration of the electronic Schroedinger equation ----------
# (used by surface hopping and Ehrenfest dynamics)
[electronic_integration]
# Allowed: "local_diabatisation" - propagation in the locally diabatic
#                                  basis (surface hopping only; Ehrenfest
#                                  falls back to "matrix_exponential")
#          "matrix_exponential"  - exponential of the electronic Hamiltonian
#          "rk4"                 - 4th-order Runge-Kutta
method = "local_diabatisation"
# number of electronic substeps per nuclear step for the rk4 integrator
substeps = 1000

# ----- only read for dynamics_type = "surface_hopping" --------------
[surface_hopping]
# decoherence correction (eqn. 17, JCP 126, 134114)
decoherence_correction = false
# force a switch to S0 if the S0/S1 gap falls below the threshold (Hartree)
force_switch_to_gs = true
force_switch_s0s1_threshold = 0.1
# rescale the velocities at frustrated hops
rescale_at_frustrated_hop = false
# enforce energy conservation by velocity rescaling after each step
artificial_energy_conservation = false

# ----- only read for dynamics_type = "ehrenfest" --------------------
[ehrenfest]
# include the state couplings (NACMEs) in the electronic Hamiltonian
# (always propagated with the matrix exponential)
use_state_coupling = false
state_threshold = 0.01
# decoherence correction
# Allowed: "none", "tab"
decoherence = "none"
tab_grad_threshold = 1.0e-5
# element-specific TAB alpha parameters (atomic numbers / values)
alpha_atoms = [1, 6, 7, 8]
alpha_values = [4.7, 22.7, 19.8, 12.2]
# harmonic restraint on the nuclei
use_restraint = false
force_constant = 1.0
# write the electronic coefficients to a file
print_coefficients = false
# initialize the electronic coefficients from a saved adiabatic eigenvector
load_adiabatic_coefficients = false
adiabatic_coefficients_file = "adiabatic_coefficients_state_0.npy"

[thermostat]
# Allowed: "none"      - NVE, velocity Verlet
#          "berendsen" - Berendsen velocity rescaling
#          "langevin"  - Langevin dynamics with stochastic friction
kind = "none"
# target temperature in K
temperature = 300.0
# coupling time of the Berendsen thermostat in fs
time_coupling = 50.0
# friction coefficient for the Langevin dynamics in a.u.
friction = 0.015585

[output]
restart = true
coordinates = true
energies = true
temperature = false
state = false
"#;

#[derive(Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DynamicsType {
    GroundState,
    ExcitedState,
    SurfaceHopping,
    Ehrenfest,
}

#[derive(Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CouplingType {
    Nacv,
    Overlap,
}

#[derive(Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum VelocityInit {
    Zero,
    Boltzmann,
    File,
}

#[derive(Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum IntegrationMethod {
    LocalDiabatisation,
    MatrixExponential,
    Rk4,
}

#[derive(Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ThermostatKind {
    None,
    Berendsen,
    Langevin,
}

#[derive(Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DecoherenceKind {
    None,
    Tab,
}

fn default_dynamics_type() -> DynamicsType {
    DynamicsType::GroundState
}
fn default_coupling() -> CouplingType {
    CouplingType::Nacv
}
fn default_velocities() -> VelocityInit {
    VelocityInit::Boltzmann
}
fn default_method() -> IntegrationMethod {
    IntegrationMethod::LocalDiabatisation
}
fn default_thermostat_kind() -> ThermostatKind {
    ThermostatKind::None
}
fn default_decoherence() -> DecoherenceKind {
    DecoherenceKind::None
}
fn default_nstep() -> usize {
    NSTEP
}
fn default_stepsize() -> f64 {
    STEPSIZE
}
fn default_nstates() -> usize {
    NSTATES
}
fn default_states() -> Vec<usize> {
    vec![INITIAL_STATE]
}
fn default_temperature() -> f64 {
    TEMPERATURE
}
fn default_substeps() -> usize {
    INTEGRATION_STEPS
}
fn default_force_switch_to_gs() -> bool {
    FORCE_SWITCH_TO_GS
}
fn default_force_switch_s0s1_threshold() -> f64 {
    0.1
}
fn default_state_threshold() -> f64 {
    STATE_THRESHOLD
}
fn default_tab_grad_threshold() -> f64 {
    1.0e-5
}
fn default_alpha_atoms() -> Vec<usize> {
    vec![1, 6, 7, 8]
}
fn default_alpha_values() -> Vec<f64> {
    vec![4.7, 22.7, 19.8, 12.2]
}
fn default_force_constant() -> f64 {
    FORCE_CONSTANT
}
fn default_adiabatic_coefficients_file() -> String {
    String::from("adiabatic_coefficients_state_0.npy")
}
fn default_time_coupling() -> f64 {
    TIME_COUPLING
}
fn default_friction() -> f64 {
    FRICTION
}
fn default_true() -> bool {
    true
}
fn default_initial_conditions() -> InitialConditions {
    toml::from_str("").unwrap()
}
fn default_electronic_integration() -> ElectronicIntegration {
    toml::from_str("").unwrap()
}
fn default_surface_hopping() -> SurfaceHoppingUserConfig {
    toml::from_str("").unwrap()
}
fn default_ehrenfest() -> EhrenfestUserConfig {
    toml::from_str("").unwrap()
}
fn default_thermostat() -> ThermostatUserConfig {
    toml::from_str("").unwrap()
}
fn default_output() -> OutputUserConfig {
    toml::from_str("").unwrap()
}

/// The user-facing layout of `dynamics.toml`. Unknown keys are rejected so
/// that old-format files (and typos) fail at startup instead of being
/// silently ignored.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DynamicsUserConfig {
    #[serde(default = "default_dynamics_type")]
    pub dynamics_type: DynamicsType,
    #[serde(default = "default_nstep")]
    pub nstep: usize,
    #[serde(default = "default_stepsize")]
    pub stepsize: f64,
    #[serde(default)]
    pub restart: bool,
    #[serde(default = "default_nstates")]
    pub nstates: usize,
    #[serde(default = "default_coupling")]
    pub coupling: CouplingType,
    #[serde(default = "default_initial_conditions")]
    pub initial_conditions: InitialConditions,
    #[serde(default = "default_electronic_integration")]
    pub electronic_integration: ElectronicIntegration,
    #[serde(default = "default_surface_hopping")]
    pub surface_hopping: SurfaceHoppingUserConfig,
    #[serde(default = "default_ehrenfest")]
    pub ehrenfest: EhrenfestUserConfig,
    #[serde(default = "default_thermostat")]
    pub thermostat: ThermostatUserConfig,
    #[serde(default = "default_output")]
    pub output: OutputUserConfig,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct InitialConditions {
    #[serde(default = "default_states")]
    pub states: Vec<usize>,
    #[serde(default = "default_velocities")]
    pub velocities: VelocityInit,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ElectronicIntegration {
    #[serde(default = "default_method")]
    pub method: IntegrationMethod,
    #[serde(default = "default_substeps")]
    pub substeps: usize,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct SurfaceHoppingUserConfig {
    #[serde(default)]
    pub decoherence_correction: bool,
    #[serde(default = "default_force_switch_to_gs")]
    pub force_switch_to_gs: bool,
    #[serde(default = "default_force_switch_s0s1_threshold")]
    pub force_switch_s0s1_threshold: f64,
    #[serde(default)]
    pub rescale_at_frustrated_hop: bool,
    #[serde(default)]
    pub artificial_energy_conservation: bool,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct EhrenfestUserConfig {
    #[serde(default)]
    pub use_state_coupling: bool,
    #[serde(default = "default_state_threshold")]
    pub state_threshold: f64,
    #[serde(default = "default_decoherence")]
    pub decoherence: DecoherenceKind,
    #[serde(default = "default_tab_grad_threshold")]
    pub tab_grad_threshold: f64,
    #[serde(default = "default_alpha_atoms")]
    pub alpha_atoms: Vec<usize>,
    #[serde(default = "default_alpha_values")]
    pub alpha_values: Vec<f64>,
    #[serde(default)]
    pub use_restraint: bool,
    #[serde(default = "default_force_constant")]
    pub force_constant: f64,
    #[serde(default)]
    pub print_coefficients: bool,
    #[serde(default)]
    pub load_adiabatic_coefficients: bool,
    #[serde(default = "default_adiabatic_coefficients_file")]
    pub adiabatic_coefficients_file: String,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ThermostatUserConfig {
    #[serde(default = "default_thermostat_kind")]
    pub kind: ThermostatKind,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_time_coupling")]
    pub time_coupling: f64,
    #[serde(default = "default_friction")]
    pub friction: f64,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct OutputUserConfig {
    #[serde(default = "default_true")]
    pub restart: bool,
    #[serde(default = "default_true")]
    pub coordinates: bool,
    #[serde(default = "default_true")]
    pub energies: bool,
    #[serde(default)]
    pub temperature: bool,
    #[serde(default)]
    pub state: bool,
}

impl From<DynamicsUserConfig> for DynamicConfiguration {
    fn from(user: DynamicsUserConfig) -> Self {
        let method = user.electronic_integration.method;
        DynamicConfiguration {
            nstep: user.nstep,
            stepsize: user.stepsize,
            restart_flag: user.restart,
            initial_state: user.initial_conditions.states.clone(),
            nstates: user.nstates,
            gs_dynamic: user.dynamics_type == DynamicsType::GroundState,
            use_surface_hopping: user.dynamics_type == DynamicsType::SurfaceHopping,
            use_ehrenfest: user.dynamics_type == DynamicsType::Ehrenfest,
            load_velocities_from_file: user.initial_conditions.velocities == VelocityInit::File,
            use_boltzmann_velocities: user.initial_conditions.velocities
                == VelocityInit::Boltzmann,
            artificial_energy_conservation: user.surface_hopping.artificial_energy_conservation,
            ehrenfest_config: EhrenfestConfiguration {
                use_state_coupling: user.ehrenfest.use_state_coupling,
                state_threshold: user.ehrenfest.state_threshold,
                use_tab_decoherence: user.ehrenfest.decoherence == DecoherenceKind::Tab,
                tab_grad_threshold: user.ehrenfest.tab_grad_threshold,
                use_restraint: user.ehrenfest.use_restraint,
                force_constant: user.ehrenfest.force_constant,
                use_rk_integration: method == IntegrationMethod::Rk4,
                integration_steps: user.electronic_integration.substeps,
                print_coefficients: user.ehrenfest.print_coefficients,
                load_adiabatic_coefficients: user.ehrenfest.load_adiabatic_coefficients,
                adiabatic_coefficients_file: user.ehrenfest.adiabatic_coefficients_file,
            },
            ehrenfest_decoherence: EhrenfestDecoherence {
                alpha_atoms: user.ehrenfest.alpha_atoms,
                alpha_values: user.ehrenfest.alpha_values,
            },
            hopping_config: HoppingConfiguration {
                force_switch_to_gs: user.surface_hopping.force_switch_to_gs,
                force_switch_s0s1_threshold: user.surface_hopping.force_switch_s0s1_threshold,
                decoherence_correction: user.surface_hopping.decoherence_correction,
                use_rk_integration: method == IntegrationMethod::Rk4,
                use_local_diabatisation: method == IntegrationMethod::LocalDiabatisation,
                use_rescaling_at_frustrated_hop: user.surface_hopping.rescale_at_frustrated_hop,
                integration_steps: user.electronic_integration.substeps,
            },
            nonadibatic_config: NonadiabaticConfiguration {
                use_nacv_couplings: user.coupling == CouplingType::Nacv,
            },
            thermostat_config: ThermostatConfiguration {
                use_thermostat: user.thermostat.kind == ThermostatKind::Berendsen,
                // The initial Boltzmann sampling reads this field, so the
                // sampling temperature is taken from [initial_conditions]
                // unless a thermostat with its own target is active.
                temperature: if user.thermostat.kind == ThermostatKind::None {
                    user.initial_conditions.temperature
                } else {
                    user.thermostat.temperature
                },
                time_coupling: user.thermostat.time_coupling,
            },
            langevin_config: LangevinConfiguration {
                use_langevin: user.thermostat.kind == ThermostatKind::Langevin,
                friction: user.thermostat.friction,
            },
            print_config: PrintConfiguration {
                print_restart: user.output.restart,
                print_coordinates: user.output.coordinates,
                print_energies: user.output.energies,
                print_temperature: user.output.temperature,
                print_state: user.output.state,
            },
        }
    }
}

/// Keys/sections that only exist in the pre-1.2 `dynamics.toml` layout.
fn looks_like_old_format(content: &str) -> bool {
    [
        "gs_dynamic",
        "use_surface_hopping",
        "use_ehrenfest",
        "restart_flag",
        "initial_state",
        "use_boltzmann_velocities",
        "load_velocities_from_file",
        "[hopping_config]",
        "[nonadibatic_config]",
        "[thermostat_config]",
        "[langevin_config]",
        "[print_config]",
        "[ehrenfest_config]",
        "[ehrenfest_decoherence]",
    ]
    .iter()
    .any(|key| content.contains(key))
}

/// Parse the contents of a `dynamics.toml` in the user-facing format and
/// convert it to the internal [`DynamicConfiguration`]. Panics with a
/// readable message on invalid input; old-format files get a migration hint.
pub fn parse_dynamics_config(content: &str) -> DynamicConfiguration {
    match toml::from_str::<DynamicsUserConfig>(content) {
        Ok(user) => user.into(),
        Err(err) => {
            if looks_like_old_format(content) {
                panic!(
                    "dynamics.toml uses the old configuration format. Rename or \
                     delete the file and DIALECT will write a new commented \
                     template on the next run. (parse error: {})",
                    err
                );
            }
            panic!("invalid dynamics.toml: {}", err);
        }
    }
}

/// Read `dynamics.toml` from `path` (writing the commented default template
/// first if the file does not exist) and return the internal configuration.
pub fn load_dynamics_config(path: &Path) -> DynamicConfiguration {
    if !path.exists() {
        fs::write(path, DYNAMICS_TOML_TEMPLATE).expect("Unable to write config file");
    }
    let content: String = fs::read_to_string(path).expect("Unable to read config file");
    parse_dynamics_config(&content)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The template, an empty user config and the internal defaults must all
    /// describe the same configuration.
    #[test]
    fn template_matches_internal_defaults() {
        let internal_default: DynamicConfiguration = toml::from_str("").unwrap();
        let from_empty: DynamicConfiguration =
            toml::from_str::<DynamicsUserConfig>("").unwrap().into();
        let from_template: DynamicConfiguration = parse_dynamics_config(DYNAMICS_TOML_TEMPLATE);

        let reference = toml::to_string(&internal_default).unwrap();
        assert_eq!(toml::to_string(&from_empty).unwrap(), reference);
        assert_eq!(toml::to_string(&from_template).unwrap(), reference);
    }

    #[test]
    fn enum_mapping_surface_hopping_overlap_rk4() {
        let content = r#"
            dynamics_type = "surface_hopping"
            nstates = 3
            coupling = "overlap"
            [initial_conditions]
            states = [2]
            velocities = "zero"
            [electronic_integration]
            method = "rk4"
            substeps = 50
            [thermostat]
            kind = "langevin"
            friction = 0.5
        "#;
        let config = parse_dynamics_config(content);
        assert!(config.use_surface_hopping);
        assert!(!config.gs_dynamic);
        assert!(!config.use_ehrenfest);
        assert!(!config.nonadibatic_config.use_nacv_couplings);
        assert_eq!(config.initial_state, vec![2]);
        assert!(!config.use_boltzmann_velocities);
        assert!(!config.load_velocities_from_file);
        assert!(config.hopping_config.use_rk_integration);
        assert!(!config.hopping_config.use_local_diabatisation);
        assert!(config.ehrenfest_config.use_rk_integration);
        assert_eq!(config.hopping_config.integration_steps, 50);
        assert!(config.langevin_config.use_langevin);
        assert!(!config.thermostat_config.use_thermostat);
        assert_eq!(config.langevin_config.friction, 0.5);
    }

    #[test]
    fn excited_state_maps_to_no_flags() {
        let config = parse_dynamics_config("dynamics_type = \"excited_state\"");
        assert!(!config.gs_dynamic);
        assert!(!config.use_surface_hopping);
        assert!(!config.use_ehrenfest);
    }

    #[test]
    fn sampling_temperature_follows_thermostat() {
        // without a thermostat the sampling temperature comes from
        // [initial_conditions] ...
        let config = parse_dynamics_config(
            "[initial_conditions]\ntemperature = 500.0\n[thermostat]\nkind = \"none\"",
        );
        assert_eq!(config.thermostat_config.temperature, 500.0);
        // ... with one, from the thermostat target.
        let config = parse_dynamics_config(
            "[thermostat]\nkind = \"berendsen\"\ntemperature = 350.0",
        );
        assert!(config.thermostat_config.use_thermostat);
        assert_eq!(config.thermostat_config.temperature, 350.0);
    }

    #[test]
    fn ehrenfest_tab_decoherence_mapping() {
        let content = r#"
            dynamics_type = "ehrenfest"
            [ehrenfest]
            decoherence = "tab"
            alpha_atoms = [1, 6]
            alpha_values = [4.7, 22.7]
        "#;
        let config = parse_dynamics_config(content);
        assert!(config.use_ehrenfest);
        assert!(config.ehrenfest_config.use_tab_decoherence);
        assert_eq!(config.ehrenfest_decoherence.alpha_atoms, vec![1, 6]);
    }

    #[test]
    #[should_panic(expected = "old configuration format")]
    fn old_format_gets_migration_hint() {
        parse_dynamics_config("gs_dynamic = true\nnstep = 10");
    }

    #[test]
    #[should_panic(expected = "invalid dynamics.toml")]
    fn unknown_enum_value_is_rejected() {
        parse_dynamics_config("dynamics_type = \"surface_hoping\"");
    }
}
