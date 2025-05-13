#![allow(dead_code)]
#![allow(warnings)]
#[macro_use]
use crate::defaults::*;
use crate::scc::mixer::anderson::*;
use crate::scc::mixer::{AAType, AndersonAccel, AndersonAccelBuilder};
use anyhow::{Context, Result};
use ndarray::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

fn default_charge() -> i8 {
    CHARGE
}
fn default_use_gaussian_gamma() -> bool {
    true
}
fn default_use_xtb1() -> bool {
    false
}
fn default_use_shell_resolved_gamma() -> bool {
    false
}
fn default_use_gamma_damping() -> bool {
    false
}
fn default_multiplicity() -> u8 {
    MULTIPLICITY
}
fn default_jobtype() -> String {
    String::from(JOBTYPE)
}
fn default_use_fmo() -> bool {
    false
}
fn default_vdw_scaling() -> f64 {
    2.0
}
fn default_long_range_correction() -> bool {
    LONG_RANGE_CORRECTION
}
fn default_long_range_radius() -> f64 {
    LONG_RANGE_RADIUS
}
fn default_verbose() -> i8 {
    0
}
fn default_scf_max_cycles() -> usize {
    MAX_ITER
}
fn default_scf_charge_conv() -> f64 {
    SCF_CHARGE_CONV
}
fn default_scf_energy_conv() -> f64 {
    SCF_ENERGY_CONV
}
fn default_nr_active_occ() -> usize {
    ACTIVE_ORBITALS.0
}
fn default_nr_active_virt() -> usize {
    ACTIVE_ORBITALS.1
}
fn default_nstates() -> usize {
    10
}
fn default_davidson_iterations() -> usize {
    100
}
fn default_davidson_subspace_multiplier() -> usize {
    10
}
fn default_davidson_convergence() -> f64 {
    1.0e-5
}
fn default_geom_opt_max_cycles() -> usize {
    GEOM_OPT_MAX_CYCLES
}
fn default_geom_opt_tol_displacement() -> f64 {
    GEOM_OPT_TOL_DISPLACEMENT
}
fn default_geom_opt_tol_gradient() -> f64 {
    GEOM_OPT_TOL_GRADIENT
}
fn default_geom_opt_tol_energy() -> f64 {
    GEOM_OPT_TOL_ENERGY
}
fn default_state_to_optimize() -> usize {
    GEOM_OPT_STATE
}
fn default_use_bfgs() -> bool {
    true
}
fn default_use_line_search() -> bool {
    true
}
fn default_n_le() -> usize {
    NUM_LE_STATES
}
fn default_n_ct() -> usize {
    NUM_HOLES
}
fn default_use_external_skf() -> bool {
    USE_EXTERNAL
}
fn default_use_external_path() -> bool {
    USE_EXTERNAL
}
fn default_skf_directory() -> String {
    String::from(EXTERNAL_DIR)
}
fn default_number_of_cores() -> usize {
    1
}
fn default_active_space_threshold_le() -> f64 {
    1.0e-4
}
fn default_active_space_threshold_ct() -> f64 {
    1.0e-4
}
fn default_calculate_all_states() -> bool {
    false
}
fn default_calculate_excited_states() -> bool {
    false
}
fn default_calculate_ntos() -> bool {
    false
}
fn default_calculate_transition_densities() -> bool {
    false
}
fn default_save_transition_densities() -> bool {
    false
}
fn default_calculate_particle_hole_densities() -> bool {
    false
}
fn default_restrict_active_space() -> bool {
    true
}
fn default_use_casida() -> bool {
    false
}
fn default_save_natural_transition_orbitals() -> bool {
    false
}
fn default_restrict_active_orbitals() -> bool {
    false
}
fn default_active_orbitals_threshold() -> f64 {
    0.2
}
fn default_states_to_analyse() -> Vec<usize> {
    vec![0, 1]
}
fn default_points_per_bohr() -> f64 {
    2.0
}
fn default_path_to_density() -> String {
    String::from(" ")
}
fn default_threshold() -> f64 {
    1.0e-4
}
fn default_use_block_implementation() -> bool {
    true
}
fn default_n_blocks() -> usize {
    1
}
fn default_calculate_nth_step() -> usize {
    10
}
fn default_total_steps() -> usize {
    1000
}
fn default_store_tdm() -> bool {
    false
}
fn default_calc_cube() -> bool {
    true
}
fn default_use_parallelization() -> bool {
    true
}
fn default_use_dispersion() -> bool {
    USE_DISPERSION
}
fn default_s6() -> f64 {
    S6_DISP_PARAM_OB2
}
fn default_s8() -> f64 {
    S8_DISP_PARAM_OB2
}
fn default_a1() -> f64 {
    A1_DISP_PARAM_OB2
}
fn default_a2() -> f64 {
    A2_DISP_PARAM_OB2
}
fn default_atom_coordinates() -> Vec<Vec<f64>> {
    vec![vec![0.0, 0.0, 0.0]]
}
fn default_get_all_states() -> bool {
    false
}
fn default_calc_exact_s_sqrt_inv() -> bool {
    false
}
fn default_mol_config() -> MoleculeConfig {
    let mol_config: MoleculeConfig = toml::from_str("").unwrap();
    return mol_config;
}
fn default_scc_config() -> SccConfig {
    let scc_config: SccConfig = toml::from_str("").unwrap();
    return scc_config;
}
fn default_lc_config() -> LCConfig {
    let lc_config: LCConfig = toml::from_str("").unwrap();
    return lc_config;
}
fn default_excited_state_config() -> ExcitedStatesConfig {
    let excited_config: ExcitedStatesConfig = toml::from_str("").unwrap();
    return excited_config;
}
fn default_slater_koster_config() -> SlaterKosterConfig {
    let slako_config: SlaterKosterConfig = toml::from_str("").unwrap();
    return slako_config;
}
fn default_parallelization_config() -> ParallelizationConfig {
    let parallelization_config: ParallelizationConfig = toml::from_str("").unwrap();
    return parallelization_config;
}
fn default_density_config() -> DensityConfig {
    let density_config: DensityConfig = toml::from_str("").unwrap();
    return density_config;
}
fn default_lcmo_config() -> LcmoConfig {
    let config: LcmoConfig = toml::from_str("").unwrap();
    return config;
}
fn default_tda_dftb_config() -> TdaDftbConfig {
    let config: TdaDftbConfig = toml::from_str("").unwrap();
    return config;
}
fn default_opt_config() -> OptConfig {
    let opt_config: OptConfig = toml::from_str("").unwrap();
    return opt_config;
}
fn default_dispersion_config() -> DispersionConfig {
    let disp_config: DispersionConfig = toml::from_str("").unwrap();
    return disp_config;
}
fn default_tdm_calculation_config() -> TdmCalculation {
    let tdm_config: TdmCalculation = toml::from_str("").unwrap();
    return tdm_config;
}
fn default_identification_config() -> IdentificationConfig {
    let ident_config: IdentificationConfig = toml::from_str("").unwrap();
    return ident_config;
}
fn default_polariton_config() -> PolaritonConfig {
    let polariton_config: PolaritonConfig = toml::from_str("").unwrap();
    return polariton_config;
}
fn default_e() -> Vec<Vec<f64>> {
    vec![vec![1.0, 1.0, 1.0]]
}
fn default_p() -> Vec<Vec<f64>> {
    vec![vec![0.0, 0.0, 0.0]]
}
fn default_photon_energy() -> Vec<f64> {
    vec![0.0]
}
fn default_quantized_volume() -> Vec<f64> {
    vec![1.0]
}
fn default_temperature() -> f64 {
    50.0
}
fn default_n_samples() -> usize {
    50
}
fn default_use_dftb3() -> bool {
    false
}
fn default_hubbard_derivatives() -> Vec<f64> {
    vec![1.0, 1.0]
}
fn default_save_in_other_path() -> bool {
    USE_EXTERNAL
}
fn default_wigner_path() -> String {
    String::from(EXTERNAL_DIR)
}
fn default_n_cut() -> usize {
    6
}
fn default_write_velocities() -> bool {
    true
}
fn default_wigner_config() -> WignerConfig {
    let wigner_config: WignerConfig = toml::from_str("").unwrap();
    wigner_config
}
fn default_dftb3_config() -> Dftb3Config {
    let dftb_config: Dftb3Config = toml::from_str("").unwrap();
    dftb_config
}
fn default_parameterization_config() -> ParameterizationConfig {
    let config: ParameterizationConfig = toml::from_str("").unwrap();
    config
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Configuration {
    #[serde(default = "default_jobtype")]
    pub jobtype: String,
    #[serde(default = "default_use_xtb1")]
    pub use_xtb1: bool,
    #[serde(default = "default_use_gaussian_gamma")]
    pub use_gaussian_gamma: bool,
    #[serde(default = "default_use_shell_resolved_gamma")]
    pub use_shell_resolved_gamma: bool,
    #[serde(default = "default_use_fmo")]
    pub fmo: bool,
    #[serde(default = "default_vdw_scaling")]
    pub vdw_scaling: f64,
    #[serde(default = "default_verbose")]
    pub verbose: i8,
    #[serde(default = "default_mol_config")]
    pub mol: MoleculeConfig,
    #[serde(default = "default_scc_config")]
    pub scf: SccConfig,
    #[serde(default = "default_lc_config")]
    pub lc: LCConfig,
    #[serde(default = "default_dftb3_config")]
    pub dftb3: Dftb3Config,
    #[serde(default = "default_opt_config")]
    pub opt: OptConfig,
    #[serde(default = "default_excited_state_config")]
    pub excited: ExcitedStatesConfig,
    #[serde(default = "default_tda_dftb_config")]
    pub tddftb: TdaDftbConfig,
    #[serde(default = "default_slater_koster_config")]
    pub slater_koster: SlaterKosterConfig,
    #[serde(default = "default_parallelization_config")]
    pub parallelization: ParallelizationConfig,
    #[serde(default = "default_lcmo_config")]
    pub fmo_lc_tddftb: LcmoConfig,
    #[serde(default = "default_dispersion_config")]
    pub dispersion: DispersionConfig,
    #[serde(default = "default_density_config")]
    pub density: DensityConfig,
    #[serde(default = "default_tdm_calculation_config")]
    pub tdm_config: TdmCalculation,
    #[serde(default = "default_identification_config")]
    pub identification_config: IdentificationConfig,
    #[serde(default = "default_wigner_config")]
    pub wigner_config: WignerConfig,
    #[serde(default = "default_polariton_config")]
    pub polariton: PolaritonConfig,
    #[serde(default = "default_parameterization_config")]
    pub parameterization: ParameterizationConfig,
    #[serde(default)]
    pub mix_config: MixConfig,
}

impl Configuration {
    pub fn new() -> Self {
        // read tincr configuration file, if it does not exist in the directory
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

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct MoleculeConfig {
    #[serde(default = "default_charge")]
    pub charge: i8,
    #[serde(default = "default_multiplicity")]
    pub multiplicity: u8,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct SccConfig {
    #[serde(default = "default_scf_max_cycles")]
    pub scf_max_cycles: usize,
    #[serde(default = "default_scf_charge_conv")]
    pub scf_charge_conv: f64,
    #[serde(default = "default_scf_energy_conv")]
    pub scf_energy_conv: f64,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct LCConfig {
    #[serde(default = "default_long_range_correction")]
    pub long_range_correction: bool,
    #[serde(default = "default_long_range_radius")]
    pub long_range_radius: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Dftb3Config {
    #[serde(default = "default_use_dftb3")]
    pub use_dftb3: bool,
    #[serde(default = "default_use_gamma_damping")]
    pub use_gamma_damping: bool,
    #[serde(default = "default_hubbard_derivatives")]
    pub hubbard_derivatives: Vec<f64>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct ExcitedStatesConfig {
    #[serde(default = "default_calculate_excited_states")]
    pub calculate_excited_states: bool,
    #[serde(default = "default_nstates")]
    pub nstates: usize,
    #[serde(default = "default_davidson_iterations")]
    pub davidson_iterations: usize,
    #[serde(default = "default_davidson_subspace_multiplier")]
    pub davidson_subspace_multiplier: usize,
    #[serde(default = "default_davidson_convergence")]
    pub davidson_convergence: f64,
    #[serde(default = "default_use_casida")]
    pub use_casida: bool,
    #[serde(default = "default_get_all_states")]
    pub get_all_states: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TdaDftbConfig {
    #[serde(default = "default_restrict_active_orbitals")]
    pub restrict_active_orbitals: bool,
    #[serde(default = "default_active_orbitals_threshold")]
    pub active_orbital_threshold: f64,
    #[serde(default = "default_save_transition_densities")]
    pub save_transition_densities: bool,
    #[serde(default = "default_save_natural_transition_orbitals")]
    pub save_natural_transition_orbitals: bool,
    #[serde(default = "default_states_to_analyse")]
    pub states_to_analyse: Vec<usize>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SlaterKosterConfig {
    #[serde(default = "default_use_external_skf")]
    pub use_external_skf: bool,
    #[serde(default = "default_skf_directory")]
    pub skf_directory: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LcmoConfig {
    #[serde(default = "default_restrict_active_space")]
    pub restrict_active_space: bool,
    #[serde(default = "default_active_space_threshold_le")]
    pub active_space_threshold_le: f64,
    #[serde(default = "default_active_space_threshold_ct")]
    pub active_space_threshold_ct: f64,
    #[serde(default = "default_n_le")]
    pub n_le: usize,
    #[serde(default = "default_n_ct")]
    pub n_ct: usize,
    #[serde(default = "default_calculate_all_states")]
    pub calculate_all_states: bool,
    #[serde(default = "default_calculate_ntos")]
    pub calculate_ntos: bool,
    #[serde(default = "default_calculate_transition_densities")]
    pub calculate_transition_densities: bool,
    #[serde(default = "default_calculate_particle_hole_densities")]
    pub calculate_particle_hole_densities: bool,
    #[serde(default = "default_states_to_analyse")]
    pub states_to_analyse: Vec<usize>,
    #[serde(default = "default_calc_exact_s_sqrt_inv")]
    pub calc_exact_s_sqrt_inv: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ParallelizationConfig {
    #[serde(default = "default_number_of_cores")]
    pub number_of_cores: usize,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct OptConfig {
    #[serde(default = "default_state_to_optimize")]
    pub state_to_optimize: usize,
    #[serde(default = "default_geom_opt_max_cycles")]
    pub geom_opt_max_cycles: usize,
    #[serde(default = "default_geom_opt_tol_displacement")]
    pub geom_opt_tol_displacement: f64,
    #[serde(default = "default_geom_opt_tol_gradient")]
    pub geom_opt_tol_gradient: f64,
    #[serde(default = "default_geom_opt_tol_energy")]
    pub geom_opt_tol_energy: f64,
    #[serde(default = "default_use_bfgs")]
    pub use_bfgs: bool,
    #[serde(default = "default_use_line_search")]
    pub use_line_search: bool,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct DispersionConfig {
    #[serde(default = "default_use_dispersion")]
    pub use_dispersion: bool,
    #[serde(default = "default_s6")]
    pub s6: f64,
    #[serde(default = "default_s8")]
    pub s8: f64,
    #[serde(default = "default_a1")]
    pub a1: f64,
    #[serde(default = "default_a2")]
    pub a2: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TdmCalculation {
    #[serde(default = "default_calculate_nth_step")]
    pub calculate_nth_step: usize,
    #[serde(default = "default_total_steps")]
    pub total_steps: usize,
    #[serde(default = "default_store_tdm")]
    pub store_tdm: bool,
    #[serde(default = "default_store_tdm")]
    pub store_hole_particle: bool,
    #[serde(default = "default_calc_cube")]
    pub calc_cube: bool,
    #[serde(default = "default_store_tdm")]
    pub calc_tdm_cube: bool,
    #[serde(default = "default_use_parallelization")]
    pub use_parallelization: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DensityConfig {
    #[serde(default = "default_path_to_density")]
    pub path_to_density: String,
    #[serde(default = "default_points_per_bohr")]
    pub points_per_bohr: f64,
    #[serde(default = "default_threshold")]
    pub threshold: f64,
    #[serde(default = "default_use_block_implementation")]
    pub use_block_implementation: bool,
    #[serde(default = "default_n_blocks")]
    pub n_blocks: usize,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
#[serde(default)]
pub struct MixConfig {
    pub use_aa: bool,
    pub memory: usize,
    pub aa_type: AAType,
    pub regularization: f64,
    pub tol_safe: f64,
    pub max_norm: f64,
}

impl MixConfig {
    /// Initialize an instance of the Anderson Accelerator. The dimension `dim` specifies the
    /// length of the vector that should be mixed. Further details are given in the Ac2O3 crate.
    pub fn build_mixer(&self, dim: usize) -> Result<AndersonAccel> {
        // In case that AA should not be used linear mixing/vanilla iterations will be used. This
        // can be enabled by setting the memory of AndersonAccel to zero.
        let memory = match self.use_aa {
            true => self.memory,
            false => 0,
        };

        AndersonAccelBuilder::default()
            .dim(dim)
            .memory(memory)
            .aa_type(self.aa_type)
            .regularization(self.regularization)
            .safeguard_factor(self.tol_safe)
            .max_weight_norm(self.max_norm)
            .build()
            .context("Could not intialize Anderson Acceleration instance")
    }
}

impl Default for MixConfig {
    fn default() -> Self {
        Self {
            use_aa: USE_AA,
            memory: AA_MEMORY,
            aa_type: AA_TYPE,
            regularization: AA_REGULARIZATION,
            tol_safe: TOL_SAFEGUARD,
            max_norm: AA_MAX_NORM,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IdentificationConfig {
    #[serde(default = "default_atom_coordinates")]
    pub atom_coordinates: Vec<Vec<f64>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PolaritonConfig {
    #[serde(default = "default_e")]
    pub e: Vec<Vec<f64>>,
    #[serde(default = "default_p")]
    pub p: Vec<Vec<f64>>,
    #[serde(default = "default_photon_energy")]
    pub photon_energy: Vec<f64>,
    #[serde(default = "default_quantized_volume")]
    pub quantized_volume: Vec<f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WignerConfig {
    #[serde(default = "default_n_samples")]
    pub n_samples: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_n_cut")]
    pub n_cut: usize,
    #[serde(default = "default_save_in_other_path")]
    pub save_in_other_path: bool,
    #[serde(default = "default_wigner_path")]
    pub wigner_path: String,
    #[serde(default = "default_write_velocities")]
    pub write_velocities: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ParameterizationConfig {
    #[serde(default = "default_use_external_path")]
    pub use_external_path: bool,
    #[serde(default = "default_skf_directory")]
    pub skf_directory: String,
}
