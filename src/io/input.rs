use crate::defaults::{CONFIG_FILE_NAME, DYNAMIC_CONFIG_FILE_NAME};
use crate::initialization::Atom;
use crate::io::{read_file_to_frame, Configuration};
use crate::xtb::initialization::atom::XtbAtom;
use chemfiles::Frame;
use dialect_dynamics::initialization::{DynamicConfiguration, SystemData};
use ndarray::prelude::*;
use std::fs;
use std::path::Path;

pub fn read_input(geom_file: &str) -> (Frame, Configuration) {
    // The file containing the cartesian coordinates is the only mandatory file to
    // start a calculation.
    let frame: Frame = read_file_to_frame(geom_file);

    // The configuration file is read, if it does not exist in the directory
    // the program initializes the default settings and writes a configuration file
    // to the directory. At the moment the filename of the configuration file
    // is hard set in the defaults.rs file to "tincr.toml"
    let config_file_path: &Path = Path::new(CONFIG_FILE_NAME);
    let mut config_string: String = if config_file_path.exists() {
        fs::read_to_string(config_file_path).expect("Unable to read config file")
    } else {
        String::from("")
    };
    // Load the configuration.
    let config: Configuration = toml::from_str(&config_string).unwrap();
    // The configuration file is saved if it does not exist already so that the user can see
    // all the used options.
    if !config_file_path.exists() {
        config_string = toml::to_string(&config).unwrap();
        fs::write(config_file_path, config_string).expect("Unable to write config file");
    }
    (frame, config)
}

pub fn read_dynamic_input(dialect_config: &Configuration) -> DynamicConfiguration {
    let config_file_path: &Path = Path::new(DYNAMIC_CONFIG_FILE_NAME);
    let mut config_string: String = if config_file_path.exists() {
        fs::read_to_string(config_file_path).expect("Unable to read config file")
    } else {
        String::from("")
    };
    // load the configuration
    let mut config: DynamicConfiguration = toml::from_str(&config_string).unwrap();
    // save the configuration file if it does not exist already so that the user can see
    // all the used options
    if !config_file_path.exists() {
        config_string = toml::to_string(&config).unwrap();
        fs::write(config_file_path, config_string).expect("Unable to write config file");
    }
    // check the number of states
    let n_states: usize = config.nstates;
    let initial_state: usize = config.initial_state[0];
    if initial_state != 0 && n_states > (dialect_config.excited.nstates + 1) {
        config.nstates = dialect_config.excited.nstates + 1;
    }

    config
}

pub fn read_dynamic_input_old(dialect_config: &Configuration) -> DynamicConfiguration {
    let config_file_path: &Path = Path::new(DYNAMIC_CONFIG_FILE_NAME);
    let mut config_string: String = if config_file_path.exists() {
        fs::read_to_string(config_file_path).expect("Unable to read config file")
    } else {
        String::from("")
    };
    // load the configuration
    let mut config: DynamicConfiguration = toml::from_str(&config_string).unwrap();
    // save the configuration file if it does not exist already so that the user can see
    // all the used options
    if !config_file_path.exists() {
        config_string = toml::to_string(&config).unwrap();
        fs::write(config_file_path, config_string).expect("Unable to write config file");
    }

    if config.initial_state[0] != 0 {
        // Number of excited states
        let n_states: usize = dialect_config.excited.nstates;
        // change nstates of config
        config.nstates = n_states + 1;
    }

    config
}

pub fn create_dynamics_data(atoms: &[Atom], dynamics_config: DynamicConfiguration) -> SystemData {
    let n_atoms: usize = atoms.len();
    let mut coordinates: Array2<f64> = Array2::zeros((n_atoms, 3));
    let mut atomic_numbers: Vec<u8> = Vec::new();

    for (idx, atom) in atoms.iter().enumerate() {
        atomic_numbers.push(atom.number);
        let array = Array::from_iter(atom.xyz.iter().cloned());
        coordinates.slice_mut(s![idx, ..]).assign(&array);
    }

    let data_system: SystemData = SystemData::from((atomic_numbers, coordinates, dynamics_config));
    data_system
}

pub fn create_dynamics_data_xtb(
    atoms: &[XtbAtom],
    dynamics_config: DynamicConfiguration,
) -> SystemData {
    let n_atoms: usize = atoms.len();
    let mut coordinates: Array2<f64> = Array2::zeros((n_atoms, 3));
    let mut atomic_numbers: Vec<u8> = Vec::new();

    for (idx, atom) in atoms.iter().enumerate() {
        atomic_numbers.push(atom.number);
        let array = Array::from_iter(atom.xyz.iter().cloned());
        coordinates.slice_mut(s![idx, ..]).assign(&array);
    }

    let data_system: SystemData = SystemData::from((atomic_numbers, coordinates, dynamics_config));
    data_system
}

pub fn read_dynamic_input_ehrenfest(
    dialect_config: &Configuration,
    n_mol: usize,
) -> DynamicConfiguration {
    let config_file_path: &Path = Path::new(DYNAMIC_CONFIG_FILE_NAME);
    let mut config_string: String = if config_file_path.exists() {
        fs::read_to_string(config_file_path).expect("Unable to read config file")
    } else {
        String::from("")
    };
    // load the configuration
    let mut config: DynamicConfiguration = toml::from_str(&config_string).unwrap();
    // save the configuration file if it does not exist already so that the user can see
    // all the used options
    if !config_file_path.exists() {
        config_string = toml::to_string(&config).unwrap();
        fs::write(config_file_path, config_string).expect("Unable to write config file");
    }

    // Number of LE states per monomer.
    let n_le: usize = dialect_config.fmo_lc_tddftb.n_le;
    // Number of CT states.
    let n_ct: usize = dialect_config.fmo_lc_tddftb.n_ct;
    // The total number of states is given by: Sum_I n_LE_I + Sum_I Sum_J nocc_I * nvirt_J
    let n_states: usize = n_le * n_mol + n_ct * n_mol * (n_mol - 1);
    // change nstates of config
    config.nstates = n_states + 1;

    config
}
