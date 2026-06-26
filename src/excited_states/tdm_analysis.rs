use crate::fmo::{build_graph, fragmentation, Graph};
use crate::initialization::System;
use ndarray::prelude::*;
use ndarray_linalg::{SymmetricSqrt, UPLO};
use ndarray_npy::write_npy;

impl System {
    pub fn tdm_fragment_analysis(&self) {
        // perform the analysis of the transition density matrix after an excited state calculation
        // get the necessary matrices, S, TDM, MO coefficients
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        // get the squareroot of S
        let s_squared: Array2<f64> = s.ssqrt(UPLO::Lower).unwrap();
        // get the occupied and virtual MO coefficients
        let nocc: usize = self.occ_indices.len();
        // let nvirt: usize = self.virt_indices.len();
        let occ_orbs: ArrayView2<f64> = orbs.slice(s![.., ..nocc]);
        let virt_orbs: ArrayView2<f64> = orbs.slice(s![.., nocc..]);

        // detect the fragments
        // Build a connectivity graph to distinguish the individual monomers from each other
        let graph: Graph = build_graph(self.atoms.len(), &self.atoms);
        // Here does the fragmentation happens
        let monomer_indices: Vec<Vec<usize>> = fragmentation(&graph);
        // get the orbitals
        let mut prev_orbs: usize = 0;
        let norbs_vec: Vec<(usize, usize)> = monomer_indices
            .iter()
            .map(|indices| {
                let mut norbs: usize = 0;
                for idx in indices {
                    norbs += self.atoms[*idx].n_orbs;
                }
                let norbs_final = (prev_orbs, prev_orbs + norbs);
                prev_orbs += norbs;
                norbs_final
            })
            .collect();

        // loop over the states, which should be analyzed
        for state in self.config.tddftb.states_to_analyse.iter() {
            // get the TDM
            let tdm: ArrayView2<f64> = self.properties.tdm(*state).unwrap();
            // transform to AO basis
            let tdm_ao: Array2<f64> = 2.0 * occ_orbs.dot(&tdm.dot(&virt_orbs.t()));
            // calculate S_squared dot TDM_AO dot S_squared
            let s_p_s_squared = (s_squared.dot(&tdm_ao.dot(&s_squared))).map(|val| val.powi(2));
            let denominator: f64 = s_p_s_squared.sum();
            // initialize the F matrix
            let mut f_mat: Array2<f64> =
                Array2::zeros((monomer_indices.len(), monomer_indices.len()));

            // calculate the F^5_XY matrix
            for (idx, norbs) in norbs_vec.iter().enumerate() {
                // slice the s_p_s matrix
                let s_p_s_sum_1 = s_p_s_squared
                    .slice(s![norbs.0..norbs.1, norbs.0..norbs.1])
                    .sum();
                f_mat[[idx, idx]] = s_p_s_sum_1 / denominator;

                for (idx2, norbs_2) in norbs_vec.iter().enumerate() {
                    if idx != idx2 {
                        let s_p_s_sum_2 = s_p_s_squared
                            .slice(s![norbs.0..norbs.1, norbs_2.0..norbs_2.1])
                            .sum();
                        f_mat[[idx, idx2]] = s_p_s_sum_2 / denominator;
                    }
                }
            }

            // write the f matrix to a numpy file
            let string: String = format!("tdm_analysis_state_{}.npy", state);
            write_npy(string, &f_mat).unwrap();
        }
    }
}
