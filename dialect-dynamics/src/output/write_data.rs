use crate::constants;
use ndarray::prelude::*;
use ndarray::{Array2, ArrayView2};
use ndarray_linalg::c64;
use serde::{Deserialize, Serialize};
use serde_yaml;
use std::fs;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use toml;

/// Struct that stores the standardized output of the dynamics simulation.
#[derive(Serialize, Deserialize, Clone)]
pub struct StandardOutput {
    pub time: f64,
    pub coordinates: Array2<f64>,
    pub velocities: Array2<f64>,
    pub kinetic_energy: f64,
    pub electronic_energy: f64,
    pub total_energy: f64,
    pub energy_difference: f64,
    pub forces: Array2<f64>,
    pub state: usize,
}

impl StandardOutput {
    pub fn new(
        time: f64,
        coordinates: ArrayView2<f64>,
        velocities: ArrayView2<f64>,
        kinetic_energy: f64,
        electronic_energy: f64,
        total_energy: f64,
        energy_difference: f64,
        forces: ArrayView2<f64>,
        state: usize,
    ) -> StandardOutput {
        let time: f64 = time / constants::FS_TO_AU;
        StandardOutput {
            time,
            coordinates: coordinates.to_owned() * constants::BOHR_TO_ANGS,
            velocities: velocities.to_owned(),
            kinetic_energy,
            electronic_energy,
            total_energy,
            energy_difference,
            forces: forces.to_owned(),
            state,
        }
    }
}

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

/// Struct that stores the output of the surface hopping routines
#[derive(Serialize, Deserialize, Clone)]
pub struct HoppingOutput {
    pub time: f64,
    pub coefficients_real: Array1<f64>,
    pub coefficients_imag: Array1<f64>,
    pub nonadiabatic_scalar: Array2<f64>,
}

impl HoppingOutput {
    pub fn new(
        time: f64,
        coefficients: ArrayView1<c64>,
        nonadiabatic_scalar: ArrayView2<f64>,
    ) -> HoppingOutput {
        let time: f64 = time / constants::FS_TO_AU;
        let coefficients_real: Vec<f64> = coefficients.iter().map(|val| val.re).collect();
        let coefficients_imag: Vec<f64> = coefficients.iter().map(|val| val.im).collect();
        HoppingOutput {
            time,
            coefficients_real: Array::from(coefficients_real),
            coefficients_imag: Array::from(coefficients_imag),
            nonadiabatic_scalar: nonadiabatic_scalar.to_owned(),
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
}

impl RestartOutput {
    pub fn new(
        n_atoms: usize,
        coordinates: ArrayView2<f64>,
        velocities: ArrayView2<f64>,
        nonadiabatic_scalar: ArrayView2<f64>,
        coefficients: ArrayView1<c64>,
    ) -> RestartOutput {
        RestartOutput {
            n_atoms,
            coordinates: coordinates.to_owned(),
            velocities: velocities.to_owned(),
            nonadiabatic_scalar: nonadiabatic_scalar.to_owned(),
            coefficients: coefficients.to_owned(),
        }
    }
}

/// Write the output of the struct [StandardOutput] to the file "dynamics.out" in the
/// yaml file format.
pub fn write_full(standard: &StandardOutput) {
    let file_path: &Path = Path::new("dynamics.out");
    let full: String = serde_yaml::to_string(standard).unwrap();
    if file_path.exists() {
        let file = OpenOptions::new().append(true).open(file_path).unwrap();
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", full)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, full).expect("Unable to write to dynamics.out file");
    }
}

/// Print the output from the struct [StandardOutput] to the file "dynamics.out" in
/// a custom format.
pub fn write_full_custom(standard: &StandardOutput, masses: ArrayView1<f64>, first_call: bool) {
    let mut string: String = String::from("######################################\n");
    string.push_str(&String::from("time: "));
    string.push_str(&standard.time.to_string());
    string.push_str(&String::from("\n[coordinates]\n"));
    let n_atoms: usize = standard.coordinates.dim().0;
    for atom in 0..n_atoms {
        for item in 0..3 {
            let str: String = standard.coordinates.slice(s![atom, item]).to_string();
            string.push_str(&str);
            string.push('\t');
        }
        string.push('\n');
    }
    string.push_str(&String::from("[velocities]\n"));
    for atom in 0..n_atoms {
        for item in 0..3 {
            let str: String = standard.velocities.slice(s![atom, item]).to_string();
            string.push_str(&str);
            string.push('\t');
        }
        string.push('\n');
    }
    string.push_str("Kinetic Energy: ");
    string.push_str(&standard.kinetic_energy.to_string());
    string.push('\n');
    string.push_str("Electronic energy: ");
    string.push_str(&standard.electronic_energy.to_string());
    string.push('\n');
    string.push_str("Total energy: ");
    string.push_str(&standard.total_energy.to_string());
    string.push('\n');
    string.push_str("Energy difference ");
    string.push_str(&standard.energy_difference.to_string());
    string.push('\n');
    string.push_str("Electronic state: ");
    string.push_str(&standard.state.to_string());
    string.push('\n');
    string.push_str("[forces]\n");
    for atom in 0..n_atoms {
        for item in 0..3 {
            let str: String = (standard.forces[[atom, item]] * masses[atom]).to_string();
            string.push_str(&str);
            string.push('\t');
        }
        string.push('\n');
    }

    let file_path: &Path = Path::new("dynamics.out");
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
        // write!(&mut file,full);
    } else {
        fs::write(file_path, string).expect("Unable to write to dynamics.out file");
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

/// Print the parameters of the struct [HoppingOutput] to the file "hopping.dat".
pub fn write_hopping(hopping_out: &HoppingOutput, first_call: bool) {
    let file_path: &Path = Path::new("hopping.dat");
    let mut hopp: String = String::from("#############################\n");
    hopp.push_str(&toml::to_string(hopping_out).unwrap());
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
        stream.write_fmt(format_args!("{}", hopp)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, hopp).expect("Unable to write to hopping.dat file");
    }
}

/// Print the energies of the system to the file "energies.dat"
pub fn write_energies(energies: ArrayView1<f64>, first_call: bool) {
    let file_path: &Path = Path::new("energies.dat");
    let mut string: String = String::from("");
    for (ind, energy) in energies.iter().enumerate() {
        // string.push_str(&energy.to_string());
        // string.push_str(&String::from("\t"));
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

/// Struct that stores the output of the ehrenfest routines
#[derive(Serialize, Deserialize, Clone)]
pub struct EhrenfestOutput {
    pub coefficients_real: Array1<f64>,
    pub coefficients_imag: Array1<f64>,
    pub coefficients_abs: Array1<f64>,
}

impl EhrenfestOutput {
    pub fn new(coefficients: ArrayView1<c64>) -> EhrenfestOutput {
        let coefficients_real: Array1<f64> = coefficients.map(|val| val.re);
        let coefficients_imag: Array1<f64> = coefficients.map(|val| val.im);
        let coefficients_abs: Array1<f64> = coefficients.map(|val| val.norm_sqr());

        EhrenfestOutput {
            coefficients_real,
            coefficients_imag,
            coefficients_abs,
        }
    }
}

/// Print the parameters of the struct [EhrenfestOutput] to the file "ehrenfest.dat".
pub fn write_ehrenfest(ehrenfest_out: &EhrenfestOutput, first_call: bool) {
    let file_path: &Path = Path::new("ehrenfest.dat");
    let mut string: String = String::from("#############################\n");
    string.push_str(&toml::to_string(ehrenfest_out).unwrap());
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
