use crate::xtb::initialization::atom::XtbAtom;
use crate::xtb::{
    initialization::basis::Basis,
    parameters::{COUL_CHEMICAL_HARDNESS, COUL_SHELL_HARDNESS},
};
use hashbrown::HashMap;
use nalgebra::Vector3;
use ndarray::prelude::*;
use std::cmp::Ordering;

pub fn init_avg_hubbard_u(
    atoms: &[XtbAtom],
    basis: &Basis,
    unique_length: usize,
) -> HashMap<(u8, u8), f64> {
    let mut sigmas: HashMap<(u8, u8), f64> = HashMap::with_capacity(unique_length);

    for func in basis.basis_functions.iter() {
        let atom: &XtbAtom = &atoms[func.atom_index];
        let l: usize = func.angular_momentum;
        let z: usize = atom.number as usize - 1;

        if !sigmas.contains_key(&(atom.number, l as u8)) {
            sigmas.insert(
                (atom.number, l as u8),
                1.0 / ((1.0 + COUL_SHELL_HARDNESS[z][l]) * COUL_CHEMICAL_HARDNESS[z]),
            );
        }
    }
    sigmas
}

#[derive(Clone, Debug)]
pub struct XtbGammaFunction {
    pub c: HashMap<(u8, u8), f64>,
    pub eta: HashMap<((u8, u8), (u8, u8)), f64>,
}

impl XtbGammaFunction {
    pub fn initialize(&mut self) {
        // construct avg hubbard U matrix
        for key_r in self.c.keys() {
            for key_s in self.c.keys() {
                self.eta
                    .insert((*key_r, *key_s), 2.0 / (self.c[key_r] + self.c[key_s]));
            }
        }
    }

    fn eval(&self, r: f64, z_a: u8, l_a: u8, z_b: u8, l_b: u8) -> f64 {
        1.0 / (r.powi(2) + 1.0 / self.eta[&((z_a, l_a), (z_b, l_b))].powi(2)).sqrt()
    }

    fn deriv(&self, r: f64, z_a: u8, l_a: u8, z_b: u8, l_b: u8) -> f64 {
        -r / (r.powi(2) + 1.0 / self.eta[&((z_a, l_a), (z_b, l_b))].powi(2)).powf(1.5)
    }
}

pub fn gamma_matrix_xtb(
    gamma_func: &XtbGammaFunction,
    atoms: &[XtbAtom],
    basis: &Basis,
    n_orbs: usize,
) -> Array2<f64> {
    let mut g0_a0: Array2<f64> = Array2::zeros((n_orbs, n_orbs));

    for (i, funci) in basis.basis_functions.iter().enumerate() {
        let atomi: &XtbAtom = &atoms[funci.atom_index];
        let l_i: usize = funci.angular_momentum;

        for (j, funcj) in basis.basis_functions.iter().enumerate() {
            let atomj: &XtbAtom = &atoms[funcj.atom_index];
            let l_j: usize = funcj.angular_momentum;

            if i <= j {
                let g_val: f64 = gamma_func.eval(
                    (atomi.xyz - atomj.xyz).norm(),
                    atomi.number,
                    l_i as u8,
                    atomj.number,
                    l_j as u8,
                );
                g0_a0[[i, j]] = g_val;
                g0_a0[[j, i]] = g_val;
            }
        }
    }
    g0_a0
}

pub fn gamma_matrix_xtb_new(
    gamma_func: &XtbGammaFunction,
    atoms: &[XtbAtom],
    basis: &Basis,
) -> Array2<f64> {
    let mut g0_a0: Array2<f64> = Array2::zeros((basis.nbas, basis.nbas));

    for shell_i in basis.shells.iter() {
        let atomi: &XtbAtom = &atoms[shell_i.atom_index];
        let l_i: usize = shell_i.angular_momentum;
        // iterate over angular components
        for idx_i in (shell_i.sph_start..shell_i.sph_end) {
            // iteratve over second shells
            for shell_j in basis.shells.iter() {
                let atomj: &XtbAtom = &atoms[shell_j.atom_index];
                let l_j: usize = shell_j.angular_momentum;
                // iteratve over angular components
                for idx_j in (shell_j.sph_start..shell_j.sph_end) {
                    // compare idx_i and idx_j
                    if idx_i <= idx_j {
                        let g_val: f64 = gamma_func.eval(
                            (atomi.xyz - atomj.xyz).norm(),
                            atomi.number,
                            l_i as u8,
                            atomj.number,
                            l_j as u8,
                        );
                        g0_a0[[idx_i, idx_j]] = g_val;
                        g0_a0[[idx_j, idx_i]] = g_val;
                    }
                }
            }
        }
    }
    g0_a0
}

pub fn gamma_gradient_xtb_new(
    gamma_func: &XtbGammaFunction,
    atoms: &[XtbAtom],
    basis: &Basis,
    n_atoms: usize,
    n_orbs: usize,
) -> Array3<f64> {
    let mut g1: Array3<f64> = Array3::zeros((3 * n_atoms, n_orbs, n_orbs));
    let mut g1_val: Array2<f64> = Array2::zeros((n_orbs, n_orbs));

    for shell_i in basis.shells.iter() {
        let atomi: &XtbAtom = &atoms[shell_i.atom_index];
        let l_i: usize = shell_i.angular_momentum;
        let at_i: usize = shell_i.atom_index;
        // iterate over angular components
        for idx_i in (shell_i.sph_start..shell_i.sph_end) {
            // iterate over the next shells
            for shell_j in basis.shells.iter() {
                let atomj: &XtbAtom = &atoms[shell_j.atom_index];
                let l_j: usize = shell_j.angular_momentum;
                let at_j: usize = shell_j.atom_index;
                // iteratve over angular components
                for idx_j in (shell_j.sph_start..shell_j.sph_end) {
                    // compare the atomic indices
                    if at_i < at_j {
                        let r = atomi.xyz - atomj.xyz;
                        let r_ij: f64 = r.norm();
                        let e_ij: Vector3<f64> = r / r_ij;

                        g1_val[[idx_i, idx_j]] = gamma_func.deriv(
                            r_ij,
                            atomi.number,
                            l_i as u8,
                            atomj.number,
                            l_j as u8,
                        );
                        g1.slice_mut(s![(3 * at_i)..(3 * at_i + 3), idx_i, idx_j])
                            .assign(&Array1::from_iter(
                                (e_ij * g1_val[[idx_i, idx_j]]).iter().cloned(),
                            ));
                        g1.slice_mut(s![(3 * at_i)..(3 * at_i + 3), idx_j, idx_i])
                            .assign(&Array1::from_iter(
                                (e_ij * g1_val[[idx_i, idx_j]]).iter().cloned(),
                            ));
                    } else if at_i > at_j {
                        let r = atomi.xyz - atomj.xyz;
                        let e_ij: Vector3<f64> = r / r.norm();
                        g1.slice_mut(s![(3 * at_i)..(3 * at_i + 3), idx_i, idx_j])
                            .assign(&Array::from_iter(
                                (e_ij * g1_val[[idx_j, idx_i]]).iter().cloned(),
                            ));
                        g1.slice_mut(s![(3 * at_i)..(3 * at_i + 3), idx_j, idx_i])
                            .assign(&Array::from_iter(
                                (e_ij * g1_val[[idx_j, idx_i]]).iter().cloned(),
                            ));
                    }
                }
            }
        }
    }

    g1
}

pub fn gamma_gradient_xtb(
    gamma_func: &XtbGammaFunction,
    atoms: &[XtbAtom],
    basis: &Basis,
    n_atoms: usize,
    n_orbs: usize,
) -> Array3<f64> {
    let mut g1: Array3<f64> = Array3::zeros((3 * n_atoms, n_orbs, n_orbs));
    let mut g1_val: Array2<f64> = Array2::zeros((n_orbs, n_orbs));

    for (i, funci) in basis.basis_functions.iter().enumerate() {
        let atomi: &XtbAtom = &atoms[funci.atom_index];
        let at_i: usize = funci.atom_index;
        let l_i: usize = funci.angular_momentum;

        for (j, funcj) in basis.basis_functions.iter().enumerate() {
            let atomj: &XtbAtom = &atoms[funcj.atom_index];
            let at_j: usize = funcj.atom_index;
            let l_j: usize = funcj.angular_momentum;

            if at_i < at_j {
                let r = atomi.xyz - atomj.xyz;
                let r_ij: f64 = r.norm();
                let e_ij: Vector3<f64> = r / r_ij;

                g1_val[[i, j]] =
                    gamma_func.deriv(r_ij, atomi.number, l_i as u8, atomj.number, l_j as u8);
                g1.slice_mut(s![(3 * at_i)..(3 * at_i + 3), i, j])
                    .assign(&Array1::from_iter((e_ij * g1_val[[i, j]]).iter().cloned()));
                g1.slice_mut(s![(3 * at_i)..(3 * at_i + 3), j, i])
                    .assign(&Array1::from_iter((e_ij * g1_val[[i, j]]).iter().cloned()));
            } else if at_i > at_j {
                let r = atomi.xyz - atomj.xyz;
                let e_ij: Vector3<f64> = r / r.norm();
                g1.slice_mut(s![(3 * at_i)..(3 * at_i + 3), i, j])
                    .assign(&Array::from_iter((e_ij * g1_val[[j, i]]).iter().cloned()));
                g1.slice_mut(s![(3 * at_i)..(3 * at_i + 3), j, i])
                    .assign(&Array::from_iter((e_ij * g1_val[[j, i]]).iter().cloned()));
            }
        }
    }

    g1
}
