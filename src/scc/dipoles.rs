use crate::defaults::PROXIMITY_CUTOFF;
use crate::initialization::parameters::*;
use crate::initialization::{get_xyz_2d, Atom};
use crate::param::slako_transformations::*;
use crate::System;
use nalgebra::Vector3;
use ndarray::prelude::*;
use std::collections::HashMap;

impl System {
    pub fn calculate_dipole_moment(&self) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        // get the density matrix
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        // get the coordinate
        let xyz: Array2<f64> = get_xyz_2d(&self.atoms);
        // get nuclear charges
        let mut nuclear_charges: Array1<f64> = Array1::zeros(self.atoms.len());
        for (charge, atom) in nuclear_charges.iter_mut().zip(self.atoms.iter()) {
            *charge = atom.n_elec as f64;
        }

        // get mulliken dipole moment
        // first, calculate mulliken atomic charges
        let mut atomic_charges: Array1<f64> = Array1::zeros(self.atoms.len());
        // get overlap matrix
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // calculate the charges
        let mut mu: usize = 0;
        for (atom_a, z_a) in self.atoms.iter().enumerate() {
            for _ in 0..z_a.n_orbs {
                for nu in 0..self.n_orbs {
                    atomic_charges[atom_a] += p[[mu, nu]] * s[[mu, nu]];
                }
                mu += 1;
            }
        }
        // calculate the mulliken dipole moment
        let mulliken_dip: Array1<f64> = (&nuclear_charges - &atomic_charges).dot(&xyz);

        // calculate the dipole matrix
        let dipole_matrix: Array3<f64> = dipole_matrix(self.n_orbs, &self.atoms, &self.slako);
        // calculate the dipole moment
        let electronic_dipole_moment: Array1<f64> =
            p.into_shape(self.n_orbs * self.n_orbs).unwrap().dot(
                &dipole_matrix
                    .into_shape([self.n_orbs * self.n_orbs, 3])
                    .unwrap(),
            );
        let dipole_moment = &nuclear_charges.dot(&xyz) - &electronic_dipole_moment;

        (mulliken_dip, dipole_moment, atomic_charges)
    }
}

/// Computes the dipole matrix elements for a single molecule.
/// The code is taken from A. Humeniuks python implementation in the DFTBaby program
pub fn dipole_matrix(n_orbs: usize, atoms: &[Atom], skt: &SlaterKoster) -> Array3<f64> {
    let mut dipole_matrix: Array3<f64> = Array3::zeros((n_orbs, n_orbs, 3));
    // iterate over atoms
    let mut mu: usize = 0;
    for (_i, atomi) in atoms.iter().enumerate() {
        // iterate over orbitals on center i
        for orbi in atomi.valorbs.iter() {
            // iterate over atoms
            let mut nu: usize = 0;
            for (_j, atomj) in atoms.iter().enumerate() {
                // iterate over orbitals on center j
                for orbj in atomj.valorbs.iter() {
                    //if geometry.proximities.as_ref().unwrap()[[i, j]] {
                    if (atomi - atomj).norm() < PROXIMITY_CUTOFF {
                        if mu <= nu {
                            if atomi <= atomj {
                                let (r, x, y, z): (f64, f64, f64, f64) =
                                    directional_cosines(&atomi.xyz, &atomj.xyz);

                                dipole_matrix.slice_mut(s![mu, nu, ..]).assign(&get_dipoles(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &skt.get(atomi.kind, atomj.kind).dipole_spline,
                                    &skt.get(atomi.kind, atomj.kind).s_spline,
                                    orbi.l,
                                    orbi.m,
                                    orbj.l,
                                    orbj.m,
                                    &atomj.xyz,
                                ));
                            } else {
                                let (r, x, y, z): (f64, f64, f64, f64) =
                                    directional_cosines(&atomj.xyz, &atomi.xyz);

                                dipole_matrix.slice_mut(s![mu, nu, ..]).assign(&get_dipoles(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &skt.get(atomj.kind, atomi.kind).dipole_spline,
                                    &skt.get(atomj.kind, atomi.kind).s_spline,
                                    orbj.l,
                                    orbj.m,
                                    orbi.l,
                                    orbi.m,
                                    &atomi.xyz,
                                ));
                            }
                        } else {
                            dipole_matrix[[mu, nu, 0]] = dipole_matrix[[nu, mu, 0]];
                            dipole_matrix[[mu, nu, 1]] = dipole_matrix[[nu, mu, 1]];
                            dipole_matrix[[mu, nu, 2]] = dipole_matrix[[nu, mu, 2]];
                        }
                    }
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return dipole_matrix;
}

/// Computes the dipole matrix elements for a given AO combination
/// The code is taken from A. Humeniuks python implementation in the DFTBaby program
pub fn get_dipoles(
    r: f64,
    x: f64,
    y: f64,
    z: f64,
    dipole: &HashMap<u8, (Vec<f64>, Vec<f64>, usize)>,
    s_spline: &HashMap<u8, (Vec<f64>, Vec<f64>, usize)>,
    l1: i8,
    m1: i8,
    l2: i8,
    m2: i8,
    pos2: &Vector3<f64>,
) -> Array1<f64> {
    if l1 > 2 || l2 > 2 {
        Array1::zeros(3)
    } else {
        // HACK
        // The Slater-Koster rules for dipole matrix elements between d-orbitals
        // contain divisions by (x**2+y**2), which may raise a division by zero error.
        // To avoid this, x and y are shifted slightly away from 0
        let mut new_x: f64 = x;
        let mut new_y: f64 = y;
        if l1 > 1 || l2 > 1 {
            if (x.powi(2) + y.powi(2)) < 1.0e-14 {
                new_x = new_x + 1.0e-14;
                new_y = new_y + 1.0e-14;
            }
        }
        // calculate dipole slako transformations for each direction
        let mut dx = slako_transformations_dipole(r, new_x, new_y, z, dipole, l1, m1, l2, m2, 1, 1);
        let mut dy =
            slako_transformations_dipole(r, new_x, new_y, z, dipole, l1, m1, l2, m2, 1, -1);
        let mut dz = slako_transformations_dipole(r, new_x, new_y, z, dipole, l1, m1, l2, m2, 1, 0);

        // <nu(r-R1)|r|mu(r-R2)> = <nu(r)|r|mu(r-(R2-R1))> + R1*<nu(r)|mu(r-(R2-R1))> = D + R1*S
        let overlap = slako_transformation(r, new_x, new_y, z, s_spline, l1, m1, l2, m2);

        // apparently there is some confusion in the formula above,
        // if pos1 and pos2 are exchanged we get the expected result
        // for the dipole matrix elements for h-h
        dx += pos2[0] * overlap;
        dy += pos2[1] * overlap;
        dz += pos2[2] * overlap;

        array![dx, dy, dz]
    }
}
