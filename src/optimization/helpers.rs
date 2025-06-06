use crate::fmo::SuperSystem;
use crate::initialization::System;
use crate::xtb::initialization::system::XtbSystem;
use ndarray::prelude::*;
use ndarray_linalg::{into_col, into_row};
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
// References
// ----------
// [1] J. Nocedal, S. Wright, 'Numerical Optimization', Springer, 2006

pub fn bfgs_update(
    inv_hk: ArrayView2<f64>,
    sk: ArrayView1<f64>,
    yk: ArrayView1<f64>,
    k: usize,
) -> Array2<f64> {
    // update the inverse Hessian invH_(k+1) based on Algorithm 6.1 in Ref.[1]
    let n: usize = sk.len();
    let id: Array2<f64> = Array::eye(n);

    assert!(k >= 1);
    let inv_hkp1: Array2<f64> = if k == 1 {
        yk.dot(&sk) / yk.dot(&yk) * &id
    } else {
        let rk: f64 = 1.0 / yk.dot(&sk);
        let u: Array2<f64> = &id - &(rk * into_col(sk).dot(&into_row(yk)));
        let v: Array2<f64> = &id - &(rk * into_col(yk).dot(&into_row(sk)));
        let w: Array2<f64> = rk * into_col(sk).dot(&into_row(sk));

        u.dot(&inv_hk.dot(&v)) + w
    };
    inv_hkp1
}

#[macro_export]
macro_rules! impl_line_search {
    () => {
        pub fn line_search(
            &mut self,
            xk: ArrayView1<f64>,
            fk: f64,
            grad_fk: ArrayView1<f64>,
            pk: ArrayView1<f64>,
            state: usize,
        ) -> Array1<f64> {
            // set defaults
            let mut a: f64 = 1.0;
            let rho: f64 = 0.5;
            let c: f64 = 0.0001;
            let lmax: usize = 100;

            // directional derivative
            let df: f64 = grad_fk.dot(&pk);

            assert!(df <= 0.0, "pk = {} not a descent direction", &pk);
            let mut x_interp: Array1<f64> = Array::zeros(xk.len());

            for _i in 0..lmax {
                x_interp = &xk + &(a * &pk);

                // update coordinates
                self.update_xyz(x_interp.view());
                // calculate energy
                let energy: f64 = self.calculate_energy_line_search(state);

                if energy <= (fk + c * a * df) {
                    break;
                } else {
                    a *= rho;
                }
            }
            return x_interp;
        }
    };
}

impl System {
    impl_line_search!();
}

impl XtbSystem {
    impl_line_search!();
}

impl SuperSystem<'_> {
    impl_line_search!();
}

#[derive(Serialize, Deserialize, Clone)]
pub struct XYZOutput {
    pub atoms: Vec<String>,
    pub coordinates: Array2<f64>,
}

impl XYZOutput {
    pub fn new(atoms: Vec<String>, coordinates: Array2<f64>) -> XYZOutput {
        XYZOutput { atoms, coordinates }
    }
}

pub fn write_xyz_wigner(xyz: &XYZOutput, filename: String) {
    let file_path: &Path = Path::new(&filename);
    let n_atoms: usize = xyz.atoms.len();
    let mut string: String = n_atoms.to_string();
    string.push('\n');
    string.push('\n');
    for atom in 0..n_atoms {
        let str: String = xyz.atoms[atom].to_string();
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
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(file_path)
            .unwrap();
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", string)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, string).expect("Unable to write to geom.xyz for wigner sampling file");
    }
}

pub fn write_xyz_custom(xyz: &XYZOutput, first_call: bool) {
    let file_path: &Path = Path::new("optimization.xyz");
    let n_atoms: usize = xyz.atoms.len();
    let mut string: String = n_atoms.to_string();
    string.push('\n');
    string.push('\n');
    for atom in 0..n_atoms {
        let str: String = xyz.atoms[atom].to_string();
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
        fs::write(file_path, string).expect("Unable to write to optimization.xyz file");
    }
}

pub fn write_last_geom(xyz: &XYZOutput) {
    let file_path: &Path = Path::new("opt_geom.xyz");
    let n_atoms: usize = xyz.atoms.len();
    let mut string: String = n_atoms.to_string();
    string.push('\n');
    string.push('\n');
    for atom in 0..n_atoms {
        let str: String = xyz.atoms[atom].to_string();
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
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(file_path)
            .unwrap();
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", string)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, string).expect("Unable to write to opt_geom.xyz file");
    }
}

pub fn write_error_geom(xyz: &XYZOutput) {
    let file_path: &Path = Path::new("error_geom.xyz");
    let n_atoms: usize = xyz.atoms.len();
    let mut string: String = n_atoms.to_string();
    string.push('\n');
    string.push('\n');
    for atom in 0..n_atoms {
        let str: String = xyz.atoms[atom].to_string();
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
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(file_path)
            .unwrap();
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", string)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, string).expect("Unable to write to opt_geom.xyz file");
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct OptEnergyOutput {
    pub step: usize,
    pub energy: f64,
}

impl OptEnergyOutput {
    pub fn new(step: usize, energy: f64) -> OptEnergyOutput {
        OptEnergyOutput { step, energy }
    }
}

pub fn write_opt_energy(energy_out: &OptEnergyOutput, first_call: bool) {
    let file_path: &Path = Path::new("opt_energies.txt");
    let mut string: String = energy_out.step.to_string();
    string.push('\t');
    string.push_str(&energy_out.energy.to_string());
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
        fs::write(file_path, string).expect("Unable to write to dynamics.xyz file");
    }
}
