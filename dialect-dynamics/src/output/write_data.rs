use crate::constants;
use ndarray::prelude::*;
use ndarray_linalg::c64;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Struct that holds the geometric data of the system
#[derive(Serialize, Deserialize, Clone)]
pub struct XyzOutput {
    pub n_atoms: usize,
    pub coordinates: Array2<f64>,
    pub atomic_numbers: Vec<u8>,
}

impl XyzOutput {
    pub fn new(n_atoms: usize, coordinates: ArrayView2<f64>, atomic_numbers: Vec<u8>) -> XyzOutput {
        XyzOutput {
            n_atoms,
            coordinates: coordinates.to_owned() * constants::BOHR_TO_ANGS,
            atomic_numbers,
        }
    }
}

/// Struct that stores the parameters, which are necessary to restart the dynamics simulation
#[derive(Serialize, Deserialize, Clone)]
pub struct RestartOutput {
    pub n_atoms: usize,
    pub coordinates: Array2<f64>,
    pub velocities: Array2<f64>,
    pub nonadiabatic_scalar: Array2<f64>,
    pub coefficients: Array1<c64>,
    pub state: usize,
    pub step: usize,
}

impl RestartOutput {
    pub fn new(
        n_atoms: usize,
        coordinates: ArrayView2<f64>,
        velocities: ArrayView2<f64>,
        nonadiabatic_scalar: ArrayView2<f64>,
        coefficients: ArrayView1<c64>,
        state: usize,
        step: usize,
    ) -> RestartOutput {
        RestartOutput {
            n_atoms,
            coordinates: coordinates.to_owned(),
            velocities: velocities.to_owned(),
            nonadiabatic_scalar: nonadiabatic_scalar.to_owned(),
            coefficients: coefficients.to_owned(),
            state,
            step,
        }
    }
}

/// Write the geometric data of the System from the struct [XyzOutput] to the file
/// "dynamics.xyz" in the yaml file format.
pub fn write_xyz(xyz: &XyzOutput) {
    let file_path: &Path = Path::new("dynamics.xyz");
    let xyz: String = serde_yaml::to_string(xyz).unwrap();
    if file_path.exists() {
        let file = OpenOptions::new().append(true).open(file_path).unwrap();
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", xyz)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, xyz).expect("Unable to write to dynamics.xyz file");
    }
}

/// Print the geometric data of the system from the struct [XyzOuput] to the file
/// "dynamics.xyz" in a custom file format.
pub fn write_xyz_custom(xyz: &XyzOutput, first_call: bool) {
    let file_path: &Path = Path::new("dynamics.xyz");
    let mut string: String = xyz.n_atoms.to_string();
    string.push('\n');
    string.push('\n');
    for atom in 0..xyz.n_atoms {
        let str: String = constants::ATOM_NAMES[xyz.atomic_numbers[atom] as usize].to_string();
        string.push_str(&str);
        string.push('\t');
        for item in 0..3 {
            let str: String = xyz.coordinates.slice(s![atom, item]).to_string();
            string.push_str(&str);
            string.push('\t');
        }
        string.push('\n');
    }

    if file_path.exists() {
        let file = if first_call {
            OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(file_path)
                .unwrap()
        } else {
            OpenOptions::new().append(true).open(file_path).unwrap()
        };
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", string)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, string).expect("Unable to write to dynamics.xyz file");
    }
}

/// Print the restart parameters of the dynamics from the struct [RestartOutput] to the file "dynamics_restart.out"
/// in the yaml file format.
pub fn write_restart(restart: &RestartOutput) {
    let file_path: &Path = Path::new("dynamics_restart.out");
    let restart: String = serde_yaml::to_string(restart).unwrap();
    fs::write(file_path, restart).expect("Unable to write restart file");
}

/// Write the restart parametersd of the dynamics from the struct [RestartOutput] to the file "dynamics_restart.out"
/// in a custom file format.
pub fn write_restart_custom(restart: &RestartOutput) {
    let mut string: String = restart.n_atoms.to_string();
    string.push('\n');
    string.push('\n');
    for atom in 0..restart.n_atoms {
        for item in 0..3 {
            let str: String = restart.coordinates.slice(s![atom, item]).to_string();
            string.push_str(&str);
            string.push('\t');
        }
        string.push('\n');
    }
    string.push('\n');
    for atom in 0..restart.n_atoms {
        for item in 0..3 {
            let str: String = restart.velocities.slice(s![atom, item]).to_string();
            string.push_str(&str);
            string.push('\t');
        }
        string.push('\n');
    }
    let file_path: &Path = Path::new("dynamics_restart.out");
    fs::write(file_path, string).expect("Unable to write restart file");
}

/// Print the energies of the system to the file "energies.dat"
pub fn write_energies(energies: ArrayView1<f64>, first_call: bool) {
    let file_path: &Path = Path::new("energies.dat");
    let mut string: String = String::from("");
    for (ind, energy) in energies.iter().enumerate() {
        if ind == 0 {
            string.push_str(&energy.to_string());
            string.push_str(&String::from("\t"));
        } else {
            string.push_str(&(energies[0] - energy).abs().to_string());
            string.push_str(&String::from("\t"));
        }
    }
    string.push('\n');

    if file_path.exists() {
        let file = if first_call {
            OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(file_path)
                .unwrap()
        } else {
            OpenOptions::new().append(true).open(file_path).unwrap()
        };
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", string)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, string).expect("Unable to write to energies.dat file");
    }
}

/// Print the energies of the system to the file "energies.dat"
pub fn write_kinetic_and_total_energy(kinetic: f64, total: f64, first_call: bool) {
    let file_path: &Path = Path::new("kinetic_and_total_energies.dat");
    let mut string: String = String::from("");
    // fill string
    string.push_str(&kinetic.to_string());
    string.push_str(&String::from("\t"));
    string.push_str(&total.to_string());
    string.push_str(&String::from("\t"));
    string.push('\n');

    if file_path.exists() {
        let file = if first_call {
            OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(file_path)
                .unwrap()
        } else {
            OpenOptions::new().append(true).open(file_path).unwrap()
        };
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", string)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, string).expect("Unable to write to energies.dat file");
    }
}

/// Print the electronic state of the molecular system to the file "state.dat"
pub fn write_state(electronic_state: usize, first_call: bool) {
    let file_path: &Path = Path::new("state.dat");
    let mut string: String = electronic_state.to_string();
    string.push_str(&String::from("\n"));

    if file_path.exists() {
        let file = if first_call {
            OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(file_path)
                .unwrap()
        } else {
            OpenOptions::new().append(true).open(file_path).unwrap()
        };
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", string)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, string).expect("Unable to write to state.dat file");
    }
}

/// Print the temperature of the system to the file "temperature.dat"
pub fn write_temperature(temperature: f64, first_call: bool) {
    let file_path: &Path = Path::new("temperature.dat");
    let mut string: String = temperature.to_string();
    string.push_str(&String::from("\n"));

    if file_path.exists() {
        let file = if first_call {
            OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(file_path)
                .unwrap()
        } else {
            OpenOptions::new().append(true).open(file_path).unwrap()
        };
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", string)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, string).expect("Unable to write to temperature.dat file");
    }
}
