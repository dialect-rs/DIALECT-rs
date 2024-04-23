// config file
pub const CONFIG_FILE_NAME: &str = "dynamics.toml";
// restart file
pub const RESTART_FILE_NAME: &str = "dynamics_restart.out";
// spin multiplicity 2S + 1
pub const MULTIPLICITY: u8 = 1;
// print level
pub const VERBOSE: i8 = 0;
// number of nuclear steps
pub const NSTEP: usize = 1000;
// nuclear stepsize in fs
pub const STEPSIZE: f64 = 0.1;
pub const USE_LANGEVIN: bool = false;
// temperature (K)
pub const TEMPERATURE: f64 = 300.0;
// friction coefficient for ethylene glycol (in a.u.^-1)
pub const FRICTION: f64 = 0.015585;
// new trajectory: "new"  or restart dynamics: "restart"
pub const RESTARTFLAG: bool = false;
// write nonadiabatic couplings and transition dipole moments if available
pub const PRINT_COUPLING: bool = false;
// initial electronic state
pub const INITIAL_STATE: usize = 0;
// last state for nonad coup
pub const NSTATES: usize = 1;
pub const USE_STATE_COUPLING: bool = false;
// If something goes wrong in the excited state calculation, a switch to the ground state is forced
// is this flag is set to True. Otherwise the trajectory continues with the ground state gradients
// until the excited state calculation starts working again.
pub const FORCE_SWITCH_TO_GS: bool = true;
// Energy conservation can be enforced artificially by rescaling the velocities after each
// time step. This avoids the drift of the total energy observed for long dynamics simulations.
// this option only takes effect if dyn_mode="E"
pub const ARTIFICIAL_ENERGY_CONSERVATION: bool = false;
// If set to 1, the decoherence correction according to
// eqn. (17) in JCP 126, 134114 (2007) is turned on.
pub const DECOHERENCE_CORRECTION: bool = false;
pub const RK_INTEGRATION: bool = false;
pub const INTEGRATION_STEPS: usize = 1000;

// constant in hartree
// use the recommended value for C in eqn. (17) of JCP 126, 134114 (2007)
pub const DECOHERENCE_CONSTANT: f64 = 0.1;
// do only ground state dynamic
pub const GS_DYNAMIC: bool = true;
// After a surface hop the velocity is rescaled so that the energy lost or gained
// during the hop is transfered to or from the kinetic energy. Velocity rescaling
// becomes active only for time t > start_econst
pub const START_ECONST: f64 = 0.0;
// Time coupling for the thermostat
pub const USE_THERMOSTAT: bool = false;
pub const TIME_COUPLING: f64 = 50.0;
pub const USE_BOLTZMANN_VELOCITIES: bool = true;
pub const PRINT_RESTART: bool = true;
pub const PRINT_COORDINATES: bool = true;
pub const PRINT_ENERGIES: bool = true;
pub const PRINT_TEMPERATURES: bool = false;
pub const PRINT_STANDARD: bool = false;
pub const PRINT_HOPPING: bool = false;
pub const PRINT_STATE: bool = false;
pub const USE_EHRENFEST: bool = false;
pub const STATE_THRESHOLD: f64 = 0.01;
pub const USE_RESTRAINT: bool = false;
pub const FORCE_CONSTANT: f64 = 1.0;
pub const USE_RK_INTEGRATION: bool = false;
pub const PRINT_COEFFICIENTS: bool = false;
