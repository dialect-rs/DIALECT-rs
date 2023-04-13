use crate::fmo::Monomer;
use crate::initialization::Atom;
use ndarray::prelude::*;
use ndarray::{Array2, Array3};
use std::ops::AddAssign;

impl Monomer<'_> {
    pub fn compute_q_ov(&mut self, atoms: &[Atom], s_c: Option<ArrayView2<f64>>) -> Array2<f64> {
        // Reference to the overlap matrix in AO-basis.
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        // Reference to the MO coefficients. row Index: AO, column Index: MO
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        // Reference to the indices of occupied orbitals
        let occ_indices: &[usize] = self.properties.occ_indices().unwrap();
        // Reference to the indices of the virtual orbitals
        let virt_indices: &[usize] = self.properties.virt_indices().unwrap();
        // Number of occupied orbitals
        let n_occ: usize = occ_indices.len();
        // Number of virtual orbitals
        let n_virt: usize = virt_indices.len();

        // If the product was given as an argument it does not need to be computed again
        let s_c = match s_c {
            Some(s_c) => s_c.to_owned(),
            None => s.dot(&orbs),
        };

        let mut q_trans: Array3<f64> = Array3::zeros([self.n_atoms, n_occ, n_virt]);

        let mut mu: usize = 0;
        for (n, atom) in atoms.iter().enumerate() {
            for _ in 0..(atom.n_orbs) {
                // occupied - virtuals
                for (i, occi) in occ_indices.iter().enumerate() {
                    for (a, virta) in virt_indices.iter().enumerate() {
                        q_trans.slice_mut(s![n, i, a]).add_assign(
                            0.5 * (orbs[[mu, *occi]] * s_c[[mu, *virta]]
                                + orbs[[mu, *virta]] * s_c[[mu, *occi]]),
                        );
                    }
                }
                mu += 1;
            }
        }

        q_trans.into_shape([self.n_atoms, n_occ * n_virt]).unwrap()
    }

    pub fn compute_q_vo(&mut self, atoms: &[Atom], s_c: Option<ArrayView2<f64>>) -> Array2<f64> {
        // Reference to the overlap matrix in AO-basis.
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        // Reference to the MO coefficients. row Index: AO, column Index: MO
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        // Reference to the indices of occupied orbitals
        let occ_indices: &[usize] = self.properties.occ_indices().unwrap();
        // Reference to the indices of the virtual orbitals
        let virt_indices: &[usize] = self.properties.virt_indices().unwrap();
        // Number of occupied orbitals
        let n_occ: usize = occ_indices.len();
        // Number of virtual orbitals
        let n_virt: usize = virt_indices.len();

        // If the product was given as an argument it does not need to be computed again
        let s_c = match s_c {
            Some(s_c) => s_c.to_owned(),
            None => s.dot(&orbs),
        };

        let mut q_trans: Array3<f64> = Array3::zeros([self.n_atoms, n_virt, n_occ]);

        let mut mu: usize = 0;
        for (n, atom) in atoms.iter().enumerate() {
            for _ in 0..(atom.n_orbs) {
                // virtual - occupied
                for (a, virta) in virt_indices.iter().enumerate() {
                    for (i, occi) in occ_indices.iter().enumerate() {
                        q_trans.slice_mut(s![n, a, i]).add_assign(
                            0.5 * (orbs[[mu, *virta]] * s_c[[mu, *occi]]
                                + orbs[[mu, *occi]] * s_c[[mu, *virta]]),
                        );
                    }
                }
                mu += 1;
            }
        }

        q_trans.into_shape([self.n_atoms, n_occ * n_virt]).unwrap()
    }

    pub fn compute_q_oo(&mut self) {}

    pub fn compute_q_vv(&mut self) {}
}
