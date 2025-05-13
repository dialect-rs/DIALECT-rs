use crate::cubes::helpers::{spherical_harmonics_yreal, spline_radial_wavefunction_v2};
use crate::initialization::parameters::PseudoAtom;
use crate::initialization::Atom;
use crate::io::settings::ParameterizationConfig;
use crate::xtb::initialization::atom::XtbAtom;
use crate::xtb::initialization::basis::{create_basis_set_cubes, Basis, ContractedBasisfunction};
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

        (-1.0_f64).powi(self.l as i32)
            * splined_value
            * spherical_harmonics_yreal(self.l, self.m, (dx, dy, dz))
    }

    pub fn eval_full_spline(&self) {
        let r_arr: Array1<f64> = Array::linspace(self.rmin + 1.0e-5, self.rmax - 0.05, 1000);
        let mut spline: Array1<f64> = Array1::zeros(r_arr.len());
        for (r, spl) in r_arr.iter().zip(spline.iter_mut()) {
            *spl = self.spline.sample(*r).unwrap();
        }
        write_npy(format!("spline_{}_{}.npy", self.z, self.l), &spline).unwrap();
        write_npy(format!("radii_{}_{}.npy", self.z, self.l), &r_arr).unwrap();
    }
}

#[derive(Debug, Clone)]
pub struct AtomicBasisSet {
    pub basisfunctions: Array1<AtomicBasisFunction>,
}

impl AtomicBasisSet {
    pub fn new(atoms: &[Atom], config: &ParameterizationConfig) -> Self {
        let (q_numbers, radial_wavefunctions, radial_values): (
            HashMap<u8, Vec<(i8, i8, i8)>>,
            HashMap<u8, Vec<Vec<f64>>>,
            HashMap<u8, Vec<f64>>,
        ) = load_pseudo_atoms(atoms, true, config);
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
    config: &ParameterizationConfig,
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
        let pseudo_atom = import_pseudo_atom(&atom, confined, config);
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

pub fn import_pseudo_atom(
    atom: &Atom,
    confined: bool,
    config: &ParameterizationConfig,
) -> PseudoAtom {
    if confined {
        PseudoAtom::confined_atom(atom.name, config)
    } else {
        PseudoAtom::free_atom(atom.name, config)
    }
}

pub fn create_xtb_basis_from_atoms(atoms: &[Atom]) -> Basis {
    // create xtb atoms
    let mut xtbatom_vec: Vec<XtbAtom> = Vec::new();
    for atom in atoms.iter() {
        let mut xtb_atom = XtbAtom::from(atom.number);
        xtb_atom.xyz = atom.xyz;
        xtbatom_vec.push(xtb_atom);
    }
    // create xtb basis
    let basis: Basis = create_basis_set_cubes(&xtbatom_vec);
    basis
}

pub fn evaluate_xtb_func_on_grid(func: &ContractedBasisfunction, x: f64, y: f64, z: f64) -> f64 {
    // define the x, y and z coordinate of the centers
    let x0: f64 = func.center.0;
    let y0: f64 = func.center.1;
    let z0: f64 = func.center.2;
    // difference between center and grid points
    // add some offset to avoid division by zero
    let dx = x - x0 + 1.0e-15;
    let dy = y - y0 + 1.0e-15;
    let dz = z - z0 + 1.0e-15;

    // get the distance from the center
    let r: f64 = ((x - x0).powi(2) + (y - y0).powi(2) + (z - z0).powi(2)).sqrt();

    // Cartesian part
    let mut sum: f64 = 0.0;
    for prim in func.primitive_functions.iter() {
        let coeff: f64 = prim.coeff;
        let exp: f64 = prim.exponent;
        let i: i32 = prim.angular_momenta.0 as i32;
        let j: i32 = prim.angular_momenta.1 as i32;
        let k: i32 = prim.angular_momenta.2 as i32;

        sum += coeff * dx.powi(i) * dy.powi(j) * dz.powi(k) * (-exp * r.powi(2)).exp();
    }
    sum
}

// fn get_m_from_func(ijk: (i8, i8, i8)) -> i8 {
//     match ijk {
//         (0, 0, 0) => 0,
//         (1, 0, 0) => 1,
//         (0, 0, 1) => 0,
//         (0, 1, 0) => -1,
//         (1, 1, 0) => -2,
//         (1, 0, 1) => 1,
//         (0, 1, 1) => -1,
//         _ => 0,
//     }
// }
// fn double_factorial(i: usize) -> usize {
//     if i == 1 || i == 0 {
//         1
//     } else {
//         i * double_factorial(i - 2)
//     }
// }
