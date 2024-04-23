use crate::cubes::helpers::{spherical_harmonics_yreal, spline_radial_wavefunction_v2};
use crate::initialization::parameters::PseudoAtom;
use crate::initialization::Atom;
use hashbrown::HashMap;
use nalgebra::Vector3;
use ndarray::prelude::*;
use ndarray_npy::write_npy;
use splines::Spline;

#[derive(Debug, Clone)]
pub struct AtomicBasisFunction {
    pub z: u8,
    pub center: Vector3<f64>,
    pub n: i8,
    pub l: i8,
    pub m: i8,
    pub atom_index: usize,
    pub rmin: f64,
    pub rmax: f64,
    pub spline: Spline<f64, f64>,
}

impl AtomicBasisFunction {
    pub fn eval(&self, x: f64, y: f64, z: f64) -> f64 {
        let x0 = self.center[0];
        let y0 = self.center[1];
        let z0 = self.center[2];

        // difference between center and grid points
        // add some offset to avoid division by zero
        let dx = x - x0 + 1.0e-15;
        let dy = y - y0 + 1.0e-15;
        let dz = z - z0 + 1.0e-15;

        let r: f64 = (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt();
        let mut splined_value: f64 = 0.0;
        if r < self.rmax {
            splined_value = self.spline.sample(r).unwrap() / r;
            // splined_value = splev_uniform(&self.tck, &self.c, self.k, r) / r;
        }

        let val = (-1.0_f64).powi(self.l as i32)
            * splined_value
            * spherical_harmonics_yreal(self.l, self.m, (dx, dy, dz));
        return val;
    }

    pub fn eval_full_spline(&self) {
        let r_arr: Array1<f64> = Array::linspace(self.rmin + 1.0e-5, self.rmax - 0.05, 1000);
        let mut spline: Array1<f64> = Array1::zeros(r_arr.len());
        for (r, spl) in r_arr.iter().zip(spline.iter_mut()) {
            *spl = self.spline.sample(*r).unwrap();
        }
        write_npy(format!("spline_{}_{}.npy", self.z, self.l), &spline);
        write_npy(format!("radii_{}_{}.npy", self.z, self.l), &r_arr);
    }
}

#[derive(Debug, Clone)]
pub struct AtomicBasisSet {
    pub basisfunctions: Array1<AtomicBasisFunction>,
}

impl AtomicBasisSet {
    pub fn new(atoms: &[Atom]) -> Self {
        let (q_numbers, radial_wavefunctions, radial_values): (
            HashMap<u8, Vec<(i8, i8, i8)>>,
            HashMap<u8, Vec<Vec<f64>>>,
            HashMap<u8, Vec<f64>>,
        ) = load_pseudo_atoms(atoms, true);
        let mut basis_functions: Vec<AtomicBasisFunction> = Vec::new();

        for (idx, atom) in atoms.iter().enumerate() {
            let q_number: &Vec<(i8, i8, i8)> = q_numbers.get(&atom.number).unwrap();
            let radial_functions: &Vec<Vec<f64>> = radial_wavefunctions.get(&atom.number).unwrap();
            let radial: &Vec<f64> = radial_values.get(&atom.number).unwrap();
            for (numbers, radial_func) in q_number.iter().zip(radial_functions.iter()) {
                let spline_2 =
                    spline_radial_wavefunction_v2(radial.to_owned(), radial_func.to_owned());

                basis_functions.push(AtomicBasisFunction {
                    z: atom.number,
                    n: numbers.0,
                    l: numbers.1,
                    m: numbers.2,
                    center: atom.xyz,
                    atom_index: idx,
                    rmin: spline_2.1,
                    rmax: spline_2.2,
                    spline: spline_2.0,
                })
            }
        }
        Self {
            basisfunctions: Array::from(basis_functions),
        }
    }

    pub fn eval_full_spline(&self) {
        for basis in self.basisfunctions.iter() {
            basis.eval_full_spline();
        }
    }
}

pub fn load_pseudo_atoms(
    atoms: &[Atom],
    confined: bool,
) -> (
    HashMap<u8, Vec<(i8, i8, i8)>>,
    HashMap<u8, Vec<Vec<f64>>>,
    HashMap<u8, Vec<f64>>,
) {
    let mut unique_atoms: Vec<Atom> = Vec::new();
    for atom in atoms {
        if !unique_atoms.contains(atom) {
            unique_atoms.push(atom.clone());
        }
    }
    let mut qnumbers: HashMap<u8, Vec<(i8, i8, i8)>> = HashMap::new();
    let mut radial_wavefunctions: HashMap<u8, Vec<Vec<f64>>> = HashMap::new();
    let mut radial_values: HashMap<u8, Vec<f64>> = HashMap::new();
    for atom in unique_atoms {
        // load pseudo atoms
        let pseudo_atom = import_pseudo_atom(&atom, confined);
        // indices of the valence orbitals
        let orbital_indices = pseudo_atom.valence_orbitals;

        // vector of quantum numbers
        let mut qnumber: Vec<(i8, i8, i8)> = Vec::new();
        let mut radial_wavefunction: Vec<Vec<f64>> = Vec::new();
        for orb in orbital_indices {
            // get the quantum numbers
            let n = pseudo_atom.nshell[orb as usize];
            let l = pseudo_atom.angular_momenta[orb as usize];

            for m in -l..l + 1 {
                qnumber.push((n - 1, l, m));
                radial_wavefunction.push(pseudo_atom.radial_wavefunctions[orb as usize].clone());
            }
            // let txt_r = format!("r_wavef_arr_{}_{}.npy", atom.number, orb);
            // write_npy(
            //     txt_r,
            //     &Array::from(pseudo_atom.radial_wavefunctions[orb as usize].clone()),
            // );
        }
        // let txt_r = format!("r_arr_{}.npy", atom.number);
        // write_npy(txt_r, &Array::from(pseudo_atom.r.clone()));

        qnumbers.insert(atom.number, qnumber);
        radial_wavefunctions.insert(atom.number, radial_wavefunction);
        radial_values.insert(atom.number, pseudo_atom.r);
    }
    (qnumbers, radial_wavefunctions, radial_values)
}

pub fn import_pseudo_atom(atom: &Atom, confined: bool) -> PseudoAtom {
    if confined {
        return PseudoAtom::confined_atom(atom.name);
    } else {
        return PseudoAtom::free_atom(atom.name);
    }
}
