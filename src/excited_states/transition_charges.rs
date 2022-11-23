use crate::initialization::Atom;
use crate::System;
use ndarray::prelude::*;
use std::ops::AddAssign;

/// Computes the Mulliken transition charges between occupied-occupied,
/// occupied-virtual and virtual-virtual molecular orbitals.
/// Point charge approximation of transition densities according to formula (14)
/// in Heringer, Niehaus  J Comput Chem 28: 2589-2601 (2007)
pub fn trans_charges(
    n_atoms: usize,
    atoms: &[Atom],
    orbs: ArrayView2<f64>,
    s: ArrayView2<f64>,
    occ_indices: &[usize],
    virt_indices: &[usize],
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    // Number of occupied orbitals.
    let dim_o: usize = occ_indices.len();
    // Number of virtual orbitals.
    let dim_v: usize = virt_indices.len();
    // Initial index of occupied orbitals.
    let i_o: usize = occ_indices[0];
    // Final index of occupied orbitals.
    let f_o: usize = occ_indices[occ_indices.len() - 1] + 1;
    // Initial index of virtual orbitals.
    let i_v: usize = virt_indices[0];
    // Final index of virtual orbitals.
    let f_v: usize = virt_indices[virt_indices.len() - 1] + 1;
    // The transition charges between occupied and virtual orbitals are initialized.
    let mut q_trans_ov: Array2<f64> = Array2::zeros([n_atoms, dim_o * dim_v]);
    // The transition charges between occupied and occupied orbitals are initialized.
    let mut q_trans_oo: Array2<f64> = Array2::zeros([n_atoms, dim_o * dim_o]);
    // The transition charges between virtual and virtual orbitals are initialized.
    let mut q_trans_vv: Array2<f64> = Array2::zeros([n_atoms, dim_v * dim_v]);
    // Matrix product of overlap matrix with the MO coefficients.
    let s_c: Array2<f64> = s.dot(&orbs);

    let mut mu: usize = 0;
    for (n, atom) in atoms.iter().enumerate() {
        // Iteration over the atomic orbitals Mu.
        for _ in 0..atom.n_orbs {
            // Iteration over occupied orbital I.
            for (i, (orb_mu_i, s_c_mu_i)) in orbs
                .slice(s![mu, i_o..f_o])
                .iter()
                .zip(s_c.slice(s![mu, i_o..f_o]).iter())
                .enumerate()
            {
                // Iteration over virtual orbital A.
                for (a, (orb_mu_a, s_c_mu_a)) in orbs
                    .slice(s![mu, i_v..f_v])
                    .iter()
                    .zip(s_c.slice(s![mu, i_v..f_v]).iter())
                    .enumerate()
                {
                    // The index to determine the pair of MOs is computed.
                    let idx: usize = (i * dim_v) + a;
                    // The occupied - virtual transition charge is computed.
                    q_trans_ov[[n, idx]] += 0.5 * (orb_mu_i * s_c_mu_a + orb_mu_a * s_c_mu_i);
                }
                // Iteration over occupied orbital J.
                for (j, (orb_mu_j, s_c_mu_j)) in orbs
                    .slice(s![mu, i_o..f_o])
                    .iter()
                    .zip(s_c.slice(s![mu, i_o..f_o]).iter())
                    .enumerate()
                {
                    // The index is computed.
                    let idx: usize = (i * dim_o) + j;
                    // The occupied - occupied transition charge is computed.
                    q_trans_oo[[n, idx]] += 0.5 * (orb_mu_i * s_c_mu_j + orb_mu_j * s_c_mu_i);
                }
            }
            // Iteration over virtual orbital A.
            for (a, (orb_mu_a, s_c_mu_a)) in orbs
                .slice(s![mu, i_v..f_v])
                .iter()
                .zip(s_c.slice(s![mu, i_v..f_v]).iter())
                .enumerate()
            {
                // Iteration over virtual orbital B.
                for (b, (orb_mu_b, s_c_mu_b)) in orbs
                    .slice(s![mu, i_v..f_v])
                    .iter()
                    .zip(s_c.slice(s![mu, i_v..f_v]).iter())
                    .enumerate()
                {
                    // The index is computed.
                    let idx: usize = (a * dim_v) + b;
                    // The virtual - virtual transition charge is computed.
                    q_trans_vv[[n, idx]] += 0.5 * (orb_mu_a * s_c_mu_b + orb_mu_b * s_c_mu_a);
                }
            }
            mu += 1;
        }
    }

    (q_trans_ov, q_trans_oo, q_trans_vv)
}

pub fn trans_charges_restricted(
    n_atoms: usize,
    atoms: &[Atom],
    orbs: ArrayView2<f64>,
    s: ArrayView2<f64>,
    occ_indices: &[usize],
    virt_indices: &[usize],
    threshold: f64,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    // Number of occupied orbitals.
    let mut dim_o: usize = occ_indices.len();
    // Number of virtual orbitals.
    let mut dim_v: usize = virt_indices.len();

    let mut active_occupied_orbs = occ_indices.clone();
    let mut active_virtual_orbs = virt_indices.clone();

    dim_o = (dim_o as f64 * threshold) as usize;
    dim_v = (dim_v as f64 * threshold) as usize;

    active_occupied_orbs = &active_occupied_orbs[occ_indices.len() - dim_o..occ_indices.len()];
    active_virtual_orbs = &active_virtual_orbs[0..dim_v];

    // transition charges between occupied and virutal orbitals
    let mut q_trans_ov: Array3<f64> = Array3::zeros([n_atoms, dim_o, dim_v]);
    // transition charges between occupied and occupied orbitals
    let mut q_trans_oo: Array3<f64> = Array3::zeros([n_atoms, dim_o, dim_o]);
    // transition charges between virtual and virtual orbitals
    let mut q_trans_vv: Array3<f64> = Array3::zeros([n_atoms, dim_v, dim_v]);

    let s_c: Array2<f64> = s.dot(&orbs);

    let mut mu: usize = 0;
    for (atom_a, z_a) in atoms.iter().enumerate() {
        for _ in 0..z_a.n_orbs {
            // occupied - virtuals
            for (i, occi) in active_occupied_orbs.iter().enumerate() {
                for (a, virta) in active_virtual_orbs.iter().enumerate() {
                    q_trans_ov.slice_mut(s![atom_a, i, a]).add_assign(
                        0.5 * (orbs[[mu, *occi]] * s_c[[mu, *virta]]
                            + orbs[[mu, *virta]] * s_c[[mu, *occi]]),
                    );
                }
            }
            // occupied - occupied
            for (i, occi) in active_occupied_orbs.iter().enumerate() {
                for (j, occj) in active_occupied_orbs.iter().enumerate() {
                    q_trans_oo.slice_mut(s![atom_a, i, j]).add_assign(
                        0.5 * (orbs[[mu, *occi]] * s_c[[mu, *occj]]
                            + orbs[[mu, *occj]] * s_c[[mu, *occi]]),
                    );
                }
            }
            // virtual - virtual
            for (a, virta) in active_virtual_orbs.iter().enumerate() {
                for (b, virtb) in active_virtual_orbs.iter().enumerate() {
                    q_trans_vv.slice_mut(s![atom_a, a, b]).add_assign(
                        0.5 * (orbs[[mu, *virta]] * s_c[[mu, *virtb]]
                            + orbs[[mu, *virtb]] * s_c[[mu, *virta]]),
                    );
                }
            }
            mu += 1;
        }
    }
    let q_ov = q_trans_ov.into_shape([n_atoms, dim_o * dim_v]).unwrap();
    let q_oo = q_trans_oo.into_shape([n_atoms, dim_o * dim_o]).unwrap();
    let q_vv = q_trans_vv.into_shape([n_atoms, dim_v * dim_v]).unwrap();
    return (q_ov, q_oo, q_vv);
}

impl System {
    pub fn mulliken_atomic_transition_charges(&self, state: usize) -> Array1<f64> {
        // get CIS coefficients from properties
        let cis_coefficients: ArrayView2<f64> = self.properties.tdm(state).unwrap();

        // get the number of occupied orbitals
        let n_occ = self.properties.occ_indices().unwrap().len();

        // get the orbitals of the system
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let orbs_occ: ArrayView2<f64> = orbs.slice(s![.., ..n_occ]);
        let orbs_virt: ArrayView2<f64> = orbs.slice(s![.., n_occ..]);

        // calculate the transition density matrix in AO basis
        let tdm: Array2<f64> = orbs_occ.dot(&cis_coefficients.dot(&orbs_virt.t()));
        let tdm_diag: ArrayView1<f64> = tdm.diag();
        let tdm_symmetrical: Array2<f64> = (&tdm + &tdm.t()) / 2.0;

        // get the overlap matrix
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        // calculate the transition population of an AO
        let mut mulliken_transition_population: Array1<f64> = Array1::zeros(tdm_diag.len());
        for (idx_i, (s_i, tdm_i)) in s.outer_iter().zip(tdm_symmetrical.outer_iter()).enumerate() {
            for (idx_j, (s_ij, tdm_ij)) in s_i.iter().zip(tdm_i.iter()).enumerate() {
                if idx_j != idx_i {
                    mulliken_transition_population[idx_i] += s_ij * tdm_ij;
                }
            }
        }
        mulliken_transition_population = mulliken_transition_population + tdm_diag;

        // calculate the transition population of an AO version 2
        let arr: Array2<f64> = s.dot(&tdm_symmetrical);
        let mulliken_transition_population_2: ArrayView1<f64> = arr.diag();

        assert!(
            mulliken_transition_population.abs_diff_eq(&mulliken_transition_population_2, 1.0e-12),
            "Populations are NOT equal!"
        );

        // get the number of atoms of the system
        let n_atoms: usize = self.atoms.len();

        // calculate the atom specific charges
        let mut charges: Array1<f64> = Array1::zeros(n_atoms);

        let mut mu: usize = 0;
        for (idx_i, atom_i) in self.atoms.iter().enumerate() {
            for _ in 0..(atom_i.n_orbs) {
                charges[idx_i] -= mulliken_transition_population[mu];
                mu += 1;
            }
        }

        return charges;
    }
}
