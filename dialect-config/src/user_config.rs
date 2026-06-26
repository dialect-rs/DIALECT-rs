//! User-facing dialect configuration (`dialect.toml`).
//!
//! This module owns the *file format* of the main configuration: a grouped,
//! enum-based layout whose allowed values are documented in the commented
//! template written when no `dialect.toml` exists. After parsing,
//! [`DialectUserConfig`] is converted into the internal [`Configuration`],
//! so the rest of the code is independent of the file format.

use crate::settings::{
    AAType, BroydenConfig, Configuration, DensityConfig, Dftb3Config, DispersionConfig,
    ExcitedStatesConfig, FMOConfig, IdentificationConfig, LCConfig, LcmoConfig, MixConfig,
    MoleculeConfig, OptConfig, ParallelizationConfig, ParameterizationConfig, PolaritonConfig,
    SccConfig, SlaterKosterConfig, TdaDftbConfig, TdmCalculation, TightBindingConfig,
    WignerConfig,
};
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// The default `dialect.toml` written when no configuration file exists.
/// Every enum key lists its allowed values in the comments; the values in
/// this template are the defaults.
pub const DIALECT_TOML_TEMPLATE: &str = r#"# =====================================================================
#  DIALECT configuration
#  The allowed values of every option are listed in the comments.
# =====================================================================

# Type of the calculation.
# Allowed: "sp", "grad", "hessian", "opt", "dynamics", "density",
#          "tdm_ehrenfest", "monomer_identification", "initial_conditions",
#          "wigner_geometries", "polariton"
jobtype = "sp"

# Electronic-structure method.
# Allowed: "dftb", "xtb"
method = "dftb"

# print level
verbose = 0
# number of threads
n_cores = 1

[molecule]
charge = 0
multiplicity = 1

[scc]
max_cycles = 250
charge_conv = 1.0e-5
energy_conv = 1.0e-5
# electronic temperature in K; default 0 K for DFTB, 300 K for xTB
# electronic_temperature = 0.0
unrestricted = false

# Anderson acceleration of the SCC iterations
[scc.anderson]
use_aa = true
memory = 8
# Anderson type: 1 = type I, 2 = type II
aa_type = 1
regularization = 1.0e-4
tol_safe = 2.0
max_norm = 1.0e10

# Broyden charge mixer
[scc.broyden]
alpha = 0.4
omega0 = 0.01
memory = 20
# reset Broyden if the residual grows by this factor (0.0 disables)
safeguard_factor = 0.0

# ----- DFTB-specific options -----------------------------------------
[dftb]
gaussian_gamma = true
shell_resolved_gamma = false

# long-range correction
[dftb.lc]
enabled = true
radius = 3.03

[dftb.dftb3]
enabled = false
gamma_damping = false
hubbard_derivatives = [1.0, 1.0]

[dftb.slater_koster]
use_external_skf = false
skf_directory = " "

# ----- excited states (all methods) ----------------------------------
[excited]
enabled = false
nstates = 10
use_casida = false
get_all_states = false
restrict_active_orbitals = false
active_orbital_threshold = 0.2

[excited.davidson]
iterations = 100
convergence = 1.0e-5
subspace_multiplier = 10

# output/analysis of the excited states
[excited.analysis]
save_transition_densities = false
save_natural_transition_orbitals = false
tdm_fragment_analysis = false
states_to_analyse = [0, 1]

[opt]
# state to optimize (0 = ground state)
state = 0
max_cycles = 500
# Allowed: "damped_bfgs", "gdiis", "bfgs", "steepest_descent"
algorithm = "damped_bfgs"
# Allowed: "loose", "normal", "tight", "verytight"
convergence = "normal"
# line search (only used by the "bfgs" algorithm)
line_search = true

[fmo]
enabled = false
vdw_scaling = 2.0
trimer_vdw_scaling = 2.0
use_three_body = false
covalent_fragmentation = false
# Allowed: "automatic"  - graph-based fragmentation
#          "atom_count" - fragments of fragment_atom_count atoms each
#          "indices"    - fragments from fragment_index_vector
fragmentation = "automatic"
fragments_per_monomer = 1
fragment_atom_count = 0
number_of_fragments = 0
fragment_index_vector = []

[fmo.lc_tddftb]
restrict_active_space = true
active_space_threshold_le = 1.0e-4
active_space_threshold_ct = 1.0e-4
n_le = 2
n_ct = 1
calculate_all_states = false
calculate_ntos = false
calculate_transition_densities = false
calculate_particle_hole_densities = false
states_to_analyse = [0, 1]
calc_exact_s_sqrt_inv = false
save_adiabatic_coefficients = false
selected_coefficients = [0]

[dispersion]
enabled = false
s6 = 1.0
s8 = 0.01
a1 = 0.497
a2 = 3.622

# ----- job-specific sections (uncomment when needed) -----------------
# [density]          jobtype = "density"; points_per_bohr also for cubes/NTOs
#   keys: path_to_density, points_per_bohr, threshold,
#         use_block_implementation, n_blocks
# [tdm]              jobtype = "tdm_ehrenfest"
#   keys: calculate_nth_step, total_steps, store_tdm, store_hole_particle,
#         calc_cube, calc_tdm_cube, use_parallelization, use_average_trajectory
# [identification]   jobtype = "monomer_identification"
#   keys: atom_coordinates
# [wigner]           jobtype = "wigner_geometries"
#   keys: n_samples, temperature, n_cut, save_in_other_path, wigner_path,
#         write_velocities
# [polariton]        jobtype = "polariton"
#   keys: e, p, photon_energy, quantized_volume
"#;

#[derive(Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Jobtype {
    Sp,
    Grad,
    Hessian,
    Opt,
    Dynamics,
    Density,
    TdmEhrenfest,
    MonomerIdentification,
    InitialConditions,
    WignerGeometries,
    Polariton,
}

impl Jobtype {
    fn as_str(&self) -> &'static str {
        match self {
            Jobtype::Sp => "sp",
            Jobtype::Grad => "grad",
            Jobtype::Hessian => "hessian",
            Jobtype::Opt => "opt",
            Jobtype::Dynamics => "dynamics",
            Jobtype::Density => "density",
            Jobtype::TdmEhrenfest => "tdm_ehrenfest",
            Jobtype::MonomerIdentification => "monomer_identification",
            Jobtype::InitialConditions => "initial_conditions",
            Jobtype::WignerGeometries => "wigner_geometries",
            Jobtype::Polariton => "polariton",
        }
    }
}

#[derive(Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    Dftb,
    Xtb,
}

#[derive(Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OptAlgorithm {
    DampedBfgs,
    Gdiis,
    Bfgs,
    SteepestDescent,
}

#[derive(Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OptConvergence {
    Loose,
    Normal,
    Tight,
    Verytight,
}

impl OptConvergence {
    fn as_str(&self) -> &'static str {
        match self {
            OptConvergence::Loose => "loose",
            OptConvergence::Normal => "normal",
            OptConvergence::Tight => "tight",
            OptConvergence::Verytight => "verytight",
        }
    }
}

#[derive(Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Fragmentation {
    Automatic,
    AtomCount,
    Indices,
}

fn default_jobtype() -> Jobtype {
    Jobtype::Sp
}
fn default_method() -> Method {
    Method::Dftb
}
fn default_n_cores() -> usize {
    1
}
fn default_algorithm() -> OptAlgorithm {
    OptAlgorithm::DampedBfgs
}
fn default_convergence() -> OptConvergence {
    OptConvergence::Normal
}
fn default_fragmentation() -> Fragmentation {
    Fragmentation::Automatic
}
fn default_true() -> bool {
    true
}
fn empty_section<T: serde::de::DeserializeOwned>() -> T {
    toml::from_str("").unwrap()
}

/// The user-facing layout of `dialect.toml`. Unknown keys in the new
/// sections are rejected so that old-format files (and typos) fail at
/// startup instead of being silently ignored.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DialectUserConfig {
    #[serde(default = "default_jobtype")]
    pub jobtype: Jobtype,
    #[serde(default = "default_method")]
    pub method: Method,
    #[serde(default)]
    pub verbose: i8,
    #[serde(default = "default_n_cores")]
    pub n_cores: usize,
    #[serde(default = "empty_section")]
    pub molecule: MoleculeConfig,
    #[serde(default = "empty_section")]
    pub scc: SccUserConfig,
    #[serde(default = "empty_section")]
    pub dftb: DftbUserConfig,
    #[serde(default = "empty_section")]
    pub excited: ExcitedUserConfig,
    #[serde(default = "empty_section")]
    pub opt: OptUserConfig,
    #[serde(default = "empty_section")]
    pub fmo: FmoUserConfig,
    #[serde(default = "empty_section")]
    pub dispersion: DispersionUserConfig,
    // job-specific sections; the internal structs are reused directly
    #[serde(default = "empty_section")]
    pub density: DensityConfig,
    #[serde(default = "empty_section")]
    pub tdm: TdmCalculation,
    #[serde(default = "empty_section")]
    pub identification: IdentificationConfig,
    #[serde(default = "empty_section")]
    pub wigner: WignerConfig,
    #[serde(default = "empty_section")]
    pub polariton: PolaritonConfig,
    #[serde(default = "empty_section")]
    pub parameterization: ParameterizationConfig,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct SccUserConfig {
    #[serde(default = "default_scc_max_cycles")]
    pub max_cycles: usize,
    #[serde(default = "default_scc_conv")]
    pub charge_conv: f64,
    #[serde(default = "default_scc_conv")]
    pub energy_conv: f64,
    /// Electronic temperature in Kelvin. When omitted, a method-dependent
    /// default is used (0 K for DFTB, 300 K for xTB).
    #[serde(default)]
    pub electronic_temperature: Option<f64>,
    #[serde(default)]
    pub unrestricted: bool,
    #[serde(default = "empty_section")]
    pub anderson: MixConfig,
    #[serde(default = "empty_section")]
    pub broyden: BroydenConfig,
}
fn default_scc_max_cycles() -> usize {
    dialect_base::defaults::MAX_ITER
}
fn default_scc_conv() -> f64 {
    dialect_base::defaults::SCF_CHARGE_CONV
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DftbUserConfig {
    #[serde(default = "default_true")]
    pub gaussian_gamma: bool,
    #[serde(default)]
    pub shell_resolved_gamma: bool,
    #[serde(default = "empty_section")]
    pub lc: LcUserConfig,
    #[serde(default = "empty_section")]
    pub dftb3: Dftb3UserConfig,
    #[serde(default = "empty_section")]
    pub slater_koster: SlaterKosterConfig,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct LcUserConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_lc_radius")]
    pub radius: f64,
}
fn default_lc_radius() -> f64 {
    dialect_base::defaults::LONG_RANGE_RADIUS
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Dftb3UserConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub gamma_damping: bool,
    #[serde(default = "default_hubbard_derivatives")]
    pub hubbard_derivatives: Vec<f64>,
}
fn default_hubbard_derivatives() -> Vec<f64> {
    vec![1.0, 1.0]
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ExcitedUserConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_nstates")]
    pub nstates: usize,
    #[serde(default)]
    pub use_casida: bool,
    #[serde(default)]
    pub get_all_states: bool,
    #[serde(default)]
    pub restrict_active_orbitals: bool,
    #[serde(default = "default_active_orbital_threshold")]
    pub active_orbital_threshold: f64,
    #[serde(default = "empty_section")]
    pub davidson: DavidsonUserConfig,
    #[serde(default = "empty_section")]
    pub analysis: AnalysisUserConfig,
}
fn default_nstates() -> usize {
    10
}
fn default_active_orbital_threshold() -> f64 {
    0.2
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DavidsonUserConfig {
    #[serde(default = "default_davidson_iterations")]
    pub iterations: usize,
    #[serde(default = "default_davidson_convergence")]
    pub convergence: f64,
    #[serde(default = "default_davidson_subspace_multiplier")]
    pub subspace_multiplier: usize,
}
fn default_davidson_iterations() -> usize {
    100
}
fn default_davidson_convergence() -> f64 {
    1.0e-5
}
fn default_davidson_subspace_multiplier() -> usize {
    10
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct AnalysisUserConfig {
    #[serde(default)]
    pub save_transition_densities: bool,
    #[serde(default)]
    pub save_natural_transition_orbitals: bool,
    #[serde(default)]
    pub tdm_fragment_analysis: bool,
    #[serde(default = "default_states_to_analyse")]
    pub states_to_analyse: Vec<usize>,
}
fn default_states_to_analyse() -> Vec<usize> {
    vec![0, 1]
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct OptUserConfig {
    #[serde(default)]
    pub state: usize,
    #[serde(default = "default_opt_max_cycles")]
    pub max_cycles: usize,
    #[serde(default = "default_algorithm")]
    pub algorithm: OptAlgorithm,
    #[serde(default = "default_convergence")]
    pub convergence: OptConvergence,
    #[serde(default = "default_true")]
    pub line_search: bool,
}
fn default_opt_max_cycles() -> usize {
    dialect_base::defaults::GEOM_OPT_MAX_CYCLES
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct FmoUserConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_vdw_scaling")]
    pub vdw_scaling: f64,
    #[serde(default = "default_vdw_scaling")]
    pub trimer_vdw_scaling: f64,
    #[serde(default)]
    pub use_three_body: bool,
    #[serde(default)]
    pub covalent_fragmentation: bool,
    #[serde(default = "default_fragmentation")]
    pub fragmentation: Fragmentation,
    #[serde(default = "default_fragments_per_monomer")]
    pub fragments_per_monomer: usize,
    #[serde(default)]
    pub fragment_atom_count: usize,
    #[serde(default)]
    pub number_of_fragments: usize,
    #[serde(default)]
    pub fragment_index_vector: Vec<Vec<usize>>,
    #[serde(default = "empty_section")]
    pub lc_tddftb: LcmoConfig,
}
fn default_vdw_scaling() -> f64 {
    2.0
}
fn default_fragments_per_monomer() -> usize {
    1
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DispersionUserConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_s6")]
    pub s6: f64,
    #[serde(default = "default_s8")]
    pub s8: f64,
    #[serde(default = "default_a1")]
    pub a1: f64,
    #[serde(default = "default_a2")]
    pub a2: f64,
}
fn default_s6() -> f64 {
    dialect_base::defaults::S6_DISP_PARAM_OB2
}
fn default_s8() -> f64 {
    dialect_base::defaults::S8_DISP_PARAM_OB2
}
fn default_a1() -> f64 {
    dialect_base::defaults::A1_DISP_PARAM_OB2
}
fn default_a2() -> f64 {
    dialect_base::defaults::A2_DISP_PARAM_OB2
}

impl From<DialectUserConfig> for Configuration {
    fn from(user: DialectUserConfig) -> Self {
        // method selection: one enum -> the internal flag set
        let mut tight_binding = TightBindingConfig {
            use_dftb: false,
            use_xtb1: false,
            use_gaussian_gamma: user.dftb.gaussian_gamma,
            use_shell_resolved_gamma: user.dftb.shell_resolved_gamma,
        };
        match user.method {
            Method::Dftb => tight_binding.use_dftb = true,
            Method::Xtb => tight_binding.use_xtb1 = true,
        }

        Configuration {
            jobtype: user.jobtype.as_str().to_string(),
            verbose: user.verbose,
            tight_binding,
            fmo: FMOConfig {
                use_fmo: user.fmo.enabled,
                vdw_scaling: user.fmo.vdw_scaling,
                trimer_vdw_scaling: user.fmo.trimer_vdw_scaling,
                use_three_body: user.fmo.use_three_body,
                manual_fragmentation: user.fmo.fragmentation == Fragmentation::AtomCount,
                fragment_atom_count: user.fmo.fragment_atom_count,
                number_of_fragments: user.fmo.number_of_fragments,
                fragments_per_monomer: user.fmo.fragments_per_monomer,
                covalent_fragmentation: user.fmo.covalent_fragmentation,
                advanced_manual_fragmentation: user.fmo.fragmentation == Fragmentation::Indices,
                fragment_index_vector: user.fmo.fragment_index_vector,
            },
            mol: user.molecule,
            scf: SccConfig {
                unrestricted: user.scc.unrestricted,
                scf_max_cycles: user.scc.max_cycles,
                scf_charge_conv: user.scc.charge_conv,
                scf_energy_conv: user.scc.energy_conv,
                // Method-dependent default electronic temperature when the user
                // does not set one: xTB conventionally runs at 300 K (which
                // also stabilizes open-shell SCF), DFTB at 0 K.
                electronic_temperature: user.scc.electronic_temperature.unwrap_or(
                    if user.method == Method::Xtb {
                        300.0
                    } else {
                        0.0
                    },
                ),
            },
            lc: LCConfig {
                long_range_correction: user.dftb.lc.enabled,
                long_range_radius: user.dftb.lc.radius,
            },
            dftb3: Dftb3Config {
                use_dftb3: user.dftb.dftb3.enabled,
                use_gamma_damping: user.dftb.dftb3.gamma_damping,
                hubbard_derivatives: user.dftb.dftb3.hubbard_derivatives,
            },
            opt: OptConfig {
                state_to_optimize: user.opt.state,
                geom_opt_max_cycles: user.opt.max_cycles,
                use_bfgs: user.opt.algorithm == OptAlgorithm::Bfgs
                    || user.opt.algorithm == OptAlgorithm::DampedBfgs
                    || user.opt.algorithm == OptAlgorithm::Gdiis,
                use_line_search: user.opt.line_search,
                use_advanced_optimizer: user.opt.algorithm == OptAlgorithm::DampedBfgs
                    || user.opt.algorithm == OptAlgorithm::Gdiis,
                convergence_level: user.opt.convergence.as_str().to_string(),
                optimizer_version: if user.opt.algorithm == OptAlgorithm::Gdiis {
                    3
                } else {
                    2
                },
            },
            excited: ExcitedStatesConfig {
                calculate_excited_states: user.excited.enabled,
                nstates: user.excited.nstates,
                davidson_iterations: user.excited.davidson.iterations,
                davidson_subspace_multiplier: user.excited.davidson.subspace_multiplier,
                davidson_convergence: user.excited.davidson.convergence,
                use_casida: user.excited.use_casida,
                get_all_states: user.excited.get_all_states,
            },
            tddftb: TdaDftbConfig {
                restrict_active_orbitals: user.excited.restrict_active_orbitals,
                active_orbital_threshold: user.excited.active_orbital_threshold,
                save_transition_densities: user.excited.analysis.save_transition_densities,
                save_natural_transition_orbitals: user
                    .excited
                    .analysis
                    .save_natural_transition_orbitals,
                tdm_fragment_analysis: user.excited.analysis.tdm_fragment_analysis,
                states_to_analyse: user.excited.analysis.states_to_analyse,
            },
            slater_koster: user.dftb.slater_koster,
            parallelization: ParallelizationConfig {
                number_of_cores: user.n_cores,
            },
            fmo_lc_tddftb: user.fmo.lc_tddftb,
            dispersion: DispersionConfig {
                use_dispersion: user.dispersion.enabled,
                s6: user.dispersion.s6,
                s8: user.dispersion.s8,
                a1: user.dispersion.a1,
                a2: user.dispersion.a2,
            },
            density: user.density,
            tdm_config: user.tdm,
            identification_config: user.identification,
            wigner_config: user.wigner,
            polariton: user.polariton,
            parameterization: user.parameterization,
            mix_config: user.scc.anderson,
            broyden: user.scc.broyden,
        }
    }
}

/// Keys/sections that only exist in the pre-1.2 `dialect.toml` layout.
fn looks_like_old_format(content: &str) -> bool {
    [
        "[tight_binding]",
        "use_dftb",
        "use_xtb1",
        "use_am1",
        "use_om",
        "nddo_method",
        "om_method",
        "[mol]",
        "[scf]",
        "scf_max_cycles",
        "scf_charge_conv",
        "scf_energy_conv",
        "[lc]",
        "long_range_correction",
        "long_range_radius",
        "[dftb3]",
        "use_dftb3",
        "[slater_koster]",
        "[tddftb]",
        "[mix_config]",
        "[broyden]",
        "[parallelization]",
        "number_of_cores",
        "[fmo_lc_tddftb]",
        "calculate_excited_states",
        "spin_multiplicity",
        "davidson_iterations",
        "davidson_convergence",
        "davidson_subspace_multiplier",
        "use_fmo",
        "manual_fragmentation",
        "state_to_optimize",
        "geom_opt_max_cycles",
        "use_advanced_optimizer",
        "optimizer_version",
        "convergence_level",
        "use_dispersion",
        "[tdm_config]",
        "[identification_config]",
        "[wigner_config]",
    ]
    .iter()
    .any(|key| content.contains(key))
}

/// Parse the contents of a `dialect.toml` in the user-facing format and
/// convert it to the internal [`Configuration`]. Panics with a readable
/// message on invalid input; old-format files get a migration hint.
pub fn parse_dialect_config(content: &str) -> Configuration {
    match toml::from_str::<DialectUserConfig>(content) {
        Ok(user) => user.into(),
        Err(err) => {
            if looks_like_old_format(content) {
                panic!(
                    "dialect.toml uses the old configuration format. Rename or \
                     delete the file and DIALECT will write a new commented \
                     template on the next run. (parse error: {})",
                    err
                );
            }
            panic!("invalid dialect.toml: {}", err);
        }
    }
}

/// Read `dialect.toml` from `path` (writing the commented default template
/// first if the file does not exist) and return the internal configuration.
pub fn load_dialect_config(path: &Path) -> Configuration {
    if !path.exists() {
        fs::write(path, DIALECT_TOML_TEMPLATE).expect("Unable to write config file");
    }
    let content: String = fs::read_to_string(path).expect("Unable to read config file");
    parse_dialect_config(&content)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The template, an empty user config and the internal defaults must all
    /// describe the same configuration.
    #[test]
    fn template_matches_internal_defaults() {
        let internal_default: Configuration = toml::from_str("").unwrap();
        let from_empty: Configuration =
            toml::from_str::<DialectUserConfig>("").unwrap().into();
        let from_template: Configuration = parse_dialect_config(DIALECT_TOML_TEMPLATE);

        let reference = toml::to_string(&internal_default).unwrap();
        assert_eq!(toml::to_string(&from_empty).unwrap(), reference);
        assert_eq!(toml::to_string(&from_template).unwrap(), reference);
    }

    #[test]
    fn method_mapping() {
        let cases: [(&str, fn(&Configuration) -> bool); 2] = [
            ("dftb", |c| c.tight_binding.use_dftb),
            ("xtb", |c| c.tight_binding.use_xtb1),
        ];
        for (name, check) in cases {
            let config = parse_dialect_config(&format!("method = \"{}\"", name));
            assert!(check(&config), "method mapping failed for {}", name);
            // exactly one method flag set
            let flags = [config.tight_binding.use_dftb, config.tight_binding.use_xtb1];
            assert_eq!(flags.iter().filter(|&&f| f).count(), 1, "method {}", name);
        }
    }

    #[test]
    fn optimizer_mapping() {
        let config = parse_dialect_config("[opt]\nalgorithm = \"gdiis\"");
        assert!(config.opt.use_advanced_optimizer);
        assert_eq!(config.opt.optimizer_version, 3);

        let config = parse_dialect_config("[opt]\nalgorithm = \"bfgs\"");
        assert!(!config.opt.use_advanced_optimizer);
        assert!(config.opt.use_bfgs);

        let config =
            parse_dialect_config("[opt]\nalgorithm = \"steepest_descent\"\nconvergence = \"tight\"");
        assert!(!config.opt.use_advanced_optimizer);
        assert!(!config.opt.use_bfgs);
        assert_eq!(config.opt.convergence_level, "tight");
    }

    #[test]
    fn fmo_fragmentation_mapping() {
        let config = parse_dialect_config("[fmo]\nfragmentation = \"atom_count\"");
        assert!(config.fmo.manual_fragmentation);
        assert!(!config.fmo.advanced_manual_fragmentation);

        let config = parse_dialect_config("[fmo]\nfragmentation = \"indices\"");
        assert!(!config.fmo.manual_fragmentation);
        assert!(config.fmo.advanced_manual_fragmentation);
    }

    #[test]
    fn excited_and_dftb_sections() {
        let content = r#"
            method = "dftb"
            [dftb.lc]
            enabled = false
            [excited]
            enabled = true
            nstates = 30
            [excited.davidson]
            convergence = 1.0e-6
            [excited.analysis]
            save_natural_transition_orbitals = true
            states_to_analyse = [0, 3]
        "#;
        let config = parse_dialect_config(content);
        assert!(!config.lc.long_range_correction);
        assert!(config.excited.calculate_excited_states);
        assert_eq!(config.excited.nstates, 30);
        assert_eq!(config.excited.davidson_convergence, 1.0e-6);
        assert!(config.tddftb.save_natural_transition_orbitals);
        assert_eq!(config.tddftb.states_to_analyse, vec![0, 3]);
    }

    #[test]
    #[should_panic(expected = "old configuration format")]
    fn old_format_gets_migration_hint() {
        parse_dialect_config("[tight_binding]\nuse_dftb = true");
    }

    #[test]
    #[should_panic(expected = "invalid dialect.toml")]
    fn unknown_method_is_rejected() {
        parse_dialect_config("method = \"dft\"");
    }
}
