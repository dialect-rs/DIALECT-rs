use crate::defaults::{CONFIG_FILE_NAME, DYNAMIC_CONFIG_FILE_NAME};
use crate::initialization::Atom;
use crate::io::Configuration;
use dialect_xtb::initialization::atom::XtbAtom;
use dialect_dynamics::initialization::{load_dynamics_config, DynamicConfiguration, SystemData};
use ndarray::prelude::*;
use std::path::Path;
use xyz_parser::{parse_xyz_file, XyzFrame};


pub fn read_input_new(geom_file: &str) -> (XyzFrame, Configuration) {
    // The file containing the cartesian coordinates is the only mandatory file to
    // start a calculation.
    let frame: XyzFrame = parse_xyz_file(geom_file).unwrap();

    // Read the user-facing configuration file; the commented default
    // template is written first if no file exists.
    let config: Configuration =
        dialect_config::load_dialect_config(Path::new(CONFIG_FILE_NAME));
    (frame, config)
}

pub fn read_dynamic_input(dialect_config: &Configuration) -> DynamicConfiguration {
    // Read the user-facing configuration file; the commented default
    // template is written first if no file exists.
    let mut config: DynamicConfiguration =
        load_dynamics_config(Path::new(DYNAMIC_CONFIG_FILE_NAME));
    // check the number of states
    let n_states: usize = config.nstates;
    let initial_state: usize = config.initial_state[0];
    if initial_state != 0 && n_states > (dialect_config.excited.nstates + 1) {
        config.nstates = dialect_config.excited.nstates + 1;
    }

    config
}

pub fn read_dynamic_input_old(dialect_config: &Configuration) -> DynamicConfiguration {
    // Read the user-facing configuration file; the commented default
    // template is written first if no file exists.
    let mut config: DynamicConfiguration =
        load_dynamics_config(Path::new(DYNAMIC_CONFIG_FILE_NAME));

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
    // Read the user-facing configuration file; the commented default
    // template is written first if no file exists.
    let mut config: DynamicConfiguration =
        load_dynamics_config(Path::new(DYNAMIC_CONFIG_FILE_NAME));

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
