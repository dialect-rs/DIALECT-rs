use crate::fmo::*;
use crate::scc::gamma_approximation::gamma_gradients_atomwise_2d;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::ops::{AddAssign, SubAssign};

impl SuperSystem<'_> {
    /// Computes and returns the gradient of the embedding energy.
    pub fn embedding_gradient(&mut self) -> Array1<f64> {
        // The gradient of the embedding energy is initialized as an array with zeros.
        let mut gradient_array: Array2<f64> =
            Array2::zeros([3 * self.atoms.len(), self.pairs.len()]);

        // A reference to the charge differences and gamma matrix for all atoms is created.
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();

        // The charge differences are broadcast into the shape the gradients.
        let dq_f = dq
            .broadcast([3, self.atoms.len()])
            .unwrap()
            .reversed_axes()
            .as_standard_layout()
            .into_shape([3 * self.atoms.len()])
            .unwrap()
            .to_owned();

        // Reference to the derivative of the charges.
        let grad_dq: ArrayView1<f64> = self.properties.grad_dq_diag().unwrap();

        let grad_gamma_sparse: Array2<f64> =
            gamma_gradients_atomwise_2d(&self.gammafunction, &self.atoms, self.atoms.len());
        let grad_gamma_dot_dq: Array1<f64> = grad_gamma_sparse.dot(&dq);

        // Begin of the loop to compute the gradient of the embedding energy for each pair.
        self.pairs
            .par_iter()
            .zip(gradient_array.axis_iter_mut(Axis(1)).into_par_iter())
            .for_each(|(pair, mut gradient)| {
                // References to the corresponding monomers.
                let m_i: &Monomer = &self.monomers[pair.i];
                let m_j: &Monomer = &self.monomers[pair.j];

                // If the derivative is w.r.t to an atom that is within this pair:
                // The first part of the equation reads:
                // dDeltaE_IJ^V/dR_a x = DDq_a^IJ sum_(K!=I,J)^(N) sum_(C in K) Dq_C^K dgamma_(a C)/dR_(a x)

                // Reference to the DDq_a^IJ (difference of charge difference between pair and monomer)
                let delta_dq: ArrayView1<f64> = pair.properties.delta_dq().unwrap();

                // DDq is broadcasted into the shape of the gradients.
                let delta_dq_f = delta_dq
                    .broadcast([3, pair.n_atoms])
                    .unwrap()
                    .reversed_axes()
                    .as_standard_layout()
                    .into_shape([3 * pair.n_atoms])
                    .unwrap()
                    .to_owned();

                let self_interaction_i: Array1<f64> = &grad_gamma_sparse
                    .slice(s![m_i.slice.grad, m_i.slice.atom])
                    .dot(&dq.slice(s![m_i.slice.atom]))
                    + &grad_gamma_sparse
                        .slice(s![m_i.slice.grad, m_j.slice.atom])
                        .dot(&dq.slice(s![m_j.slice.atom]));

                let self_interaction_j: Array1<f64> = &grad_gamma_sparse
                    .slice(s![m_j.slice.grad, m_i.slice.atom])
                    .dot(&dq.slice(s![m_i.slice.atom]))
                    + &grad_gamma_sparse
                        .slice(s![m_j.slice.grad, m_j.slice.atom])
                        .dot(&dq.slice(s![m_j.slice.atom]));

                // The gradient for a in I is computed and assigned.
                gradient.slice_mut(s![m_i.slice.grad]).add_assign(
                    &(&delta_dq_f.slice(s![..3 * m_i.n_atoms])
                        * &(&grad_gamma_dot_dq.slice(s![m_i.slice.grad]) - &self_interaction_i)),
                );

                // The gradient for a in J is computed and assigned.
                gradient.slice_mut(s![m_j.slice.grad]).add_assign(
                    &(&delta_dq_f.slice(s![3 * m_i.n_atoms..])
                        * &(&grad_gamma_dot_dq.slice(s![m_j.slice.grad]) - &self_interaction_j)),
                );

                // Right hand side of Eq. 24, but a is still in I,J
                // The difference between the derivative of the charge differences between the monomer
                // and the dimer is computed.
                let grad_delta_dq: Array1<f64> = get_grad_delta_dq(pair, m_i, m_j);

                // The electrostatic potential (ESP) is collected from the corresponding monomers.
                let mut esp_ij: Array1<f64> = Array1::zeros([pair.n_atoms]);
                esp_ij.slice_mut(s![..m_i.n_atoms]).assign(
                    &(&m_i.properties.esp_q().unwrap()
                        - &gamma
                            .slice(s![m_i.slice.atom, m_j.slice.atom])
                            .dot(&dq.slice(s![m_j.slice.atom]))),
                );
                esp_ij.slice_mut(s![m_i.n_atoms..]).assign(
                    &(&m_j.properties.esp_q().unwrap()
                        - &gamma
                            .slice(s![m_j.slice.atom, m_i.slice.atom])
                            .dot(&dq.slice(s![m_i.slice.atom]))),
                );

                // The ESP is transformed into the shape of the gradient.
                let esp_ij = esp_ij
                    .broadcast([3, pair.n_atoms])
                    .unwrap()
                    .reversed_axes()
                    .as_standard_layout()
                    .into_shape([3 * pair.n_atoms])
                    .unwrap()
                    .to_owned();

                // The (elementwise) product of the ESP with the derivative of the pair charge
                // differences is computed.
                let gddq_esp: Array1<f64> = &grad_delta_dq * &esp_ij;

                // The gradient of the rhs for a in I is assigned.
                gradient
                    .slice_mut(s![m_i.slice.grad])
                    .add_assign(&gddq_esp.slice(s![..3 * m_i.n_atoms]));

                // The gradient of the rhs for a in J is assigned.
                gradient
                    .slice_mut(s![m_j.slice.grad])
                    .add_assign(&gddq_esp.slice(s![3 * m_i.n_atoms..]));

                // Start of the computation if the derivative is w.r.t to an atom that is not in
                // this pair. So that a in K where K != I,J.

                // The matrix vector product of the gamma matrix derivative with DDq is computed for a in I
                let mut dg_ddq: Array1<f64> = grad_gamma_sparse
                    .slice(s![0.., m_i.slice.atom])
                    .dot(&delta_dq.slice(s![..m_i.n_atoms]));

                // and for a in J.
                dg_ddq += &grad_gamma_sparse
                    .slice(s![0.., m_j.slice.atom])
                    .dot(&delta_dq.slice(s![m_i.n_atoms..]));

                // Since K != I,J => the elements where K = I,J are set to zero.
                dg_ddq
                    .slice_mut(s![m_i.slice.grad])
                    .assign(&Array1::zeros([3 * m_i.n_atoms]));
                dg_ddq
                    .slice_mut(s![m_j.slice.grad])
                    .assign(&Array1::zeros([3 * m_j.n_atoms]));

                // The (elementwise) product with the charge differences is computed and assigned.
                gradient += &(&dg_ddq * &dq_f);

                // Start of the computation of the right hand side of Eq. 25.
                // A in monomer I
                let mut ddq_gamma: Array1<f64> = delta_dq
                    .slice(s![..m_i.n_atoms])
                    .dot(&gamma.slice(s![m_i.slice.atom, 0..]));

                // A in monomer J
                ddq_gamma += &delta_dq
                    .slice(s![m_i.n_atoms..])
                    .dot(&gamma.slice(s![m_j.slice.atom, 0..]));

                // Since K != I,J => the elements were K = I,J are set to zero.
                ddq_gamma
                    .slice_mut(s![m_i.slice.atom])
                    .assign(&Array1::zeros([m_i.n_atoms]));
                ddq_gamma
                    .slice_mut(s![m_j.slice.atom])
                    .assign(&Array1::zeros([m_j.n_atoms]));

                // transform the Array into the shape of the gradients and multiply it with the derivative
                // of the charge (differences)
                gradient += &(&grad_dq
                    * &ddq_gamma
                        .broadcast([3, self.atoms.len()])
                        .unwrap()
                        .reversed_axes()
                        .as_standard_layout()
                        .into_shape([3 * self.atoms.len()])
                        .unwrap());
            });

        gradient_array.sum_axis(Axis(1))
    }

    /// Low-memory version of embedding_gradient that avoids storing the full [3*n_atoms, n_atoms] grad_gamma_sparse matrix.
    /// Instead, computes pair-local gamma gradient matrices of size [3*(n_i+n_j), (n_i+n_j)].
    /// Uses parallelization over atoms for grad_gamma_dot_dq and over pairs for the main loop.
    pub fn embedding_gradient_low_memory(&mut self) -> Array1<f64> {
        let n_atoms = self.atoms.len();

        // A reference to the charge differences and gamma matrix for all atoms is created.
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();

        // Reference to the derivative of the charges.
        let grad_dq: ArrayView1<f64> = self.properties.grad_dq_diag().unwrap();

        // Pre-compute grad_gamma_dot_dq without storing the full [3N, N] matrix
        // For atom i: grad_gamma_dot_dq[i*3+c] = sum_j e_ij[c] * deriv(r_ij) * dq[j]
        // Parallelize over atoms
        let grad_gamma_dot_dq: Array1<f64> = {
            let contributions: Vec<[f64; 3]> = (0..n_atoms)
                .into_par_iter()
                .map(|i| {
                    let atom_i = &self.atoms[i];
                    let mut contrib = [0.0f64; 3];
                    for (j, atom_j) in self.atoms.iter().enumerate() {
                        if i != j {
                            let r = atom_i - atom_j;
                            let r_ij = r.norm();
                            let deriv =
                                self.gammafunction.deriv(r_ij, atom_i.number, atom_j.number);
                            let e_ij = r / r_ij;
                            let dq_j = dq[j];
                            for c in 0..3 {
                                contrib[c] += e_ij[c] * deriv * dq_j;
                            }
                        }
                    }
                    contrib
                })
                .collect();

            let mut result = Array1::zeros([3 * n_atoms]);
            for (i, contrib) in contributions.iter().enumerate() {
                for c in 0..3 {
                    result[i * 3 + c] = contrib[c];
                }
            }
            result
        };

        // Create gradient array for each pair, then sum at the end
        let mut gradient_array: Array2<f64> = Array2::zeros([3 * n_atoms, self.pairs.len()]);

        // Process pairs in parallel
        self.pairs
            .par_iter()
            .zip(gradient_array.axis_iter_mut(Axis(1)).into_par_iter())
            .for_each(|(pair, mut gradient)| {
                // References to the corresponding monomers.
                let m_i: &Monomer = &self.monomers[pair.i];
                let m_j: &Monomer = &self.monomers[pair.j];

                let i_start = m_i.slice.atom_as_range().start;
                let j_start = m_j.slice.atom_as_range().start;

                // Reference to the DDq_a^IJ (difference of charge difference between pair and monomer)
                let delta_dq: ArrayView1<f64> = pair.properties.delta_dq().unwrap();
                let delta_dq_i: ArrayView1<f64> = delta_dq.slice(s![..m_i.n_atoms]);
                let delta_dq_j: ArrayView1<f64> = delta_dq.slice(s![m_i.n_atoms..]);

                let dq_i: ArrayView1<f64> = dq.slice(s![m_i.slice.atom]);
                let dq_j: ArrayView1<f64> = dq.slice(s![m_j.slice.atom]);

                // Combine dq for pair atoms
                let mut dq_pair: Array1<f64> = Array1::zeros(pair.n_atoms);
                dq_pair.slice_mut(s![..m_i.n_atoms]).assign(&dq_i);
                dq_pair.slice_mut(s![m_i.n_atoms..]).assign(&dq_j);

                // === FIRST TERM: a in I,J (Eq. 24 LHS) ===
                // DDq_a * (grad_gamma·dq - self_interaction)
                // self_interaction = grad_gamma[pair, pair]·dq[pair]

                // Compute LOCAL gamma gradient matrix for just the pair atoms
                // Size: [3*pair.n_atoms, pair.n_atoms] instead of [3*N, N]
                let mut grad_gamma_pair: Array2<f64> =
                    Array2::zeros([3 * pair.n_atoms, pair.n_atoms]);

                // Fill the local gamma gradient matrix
                for local_a in 0..pair.n_atoms {
                    let global_a = if local_a < m_i.n_atoms {
                        i_start + local_a
                    } else {
                        j_start + (local_a - m_i.n_atoms)
                    };
                    let atom_a = &self.atoms[global_a];

                    for local_b in 0..pair.n_atoms {
                        if local_a == local_b {
                            continue;
                        }
                        let global_b = if local_b < m_i.n_atoms {
                            i_start + local_b
                        } else {
                            j_start + (local_b - m_i.n_atoms)
                        };
                        let atom_b = &self.atoms[global_b];

                        let r = atom_a - atom_b;
                        let r_ab = r.norm();
                        let deriv = self.gammafunction.deriv(r_ab, atom_a.number, atom_b.number);
                        let e_ab = r / r_ab;

                        for c in 0..3 {
                            grad_gamma_pair[[local_a * 3 + c, local_b]] = e_ab[c] * deriv;
                        }
                    }
                }

                // self_interaction for atoms in I: grad_gamma_pair[I_local, :].dot(&dq_pair)
                let self_interaction_i: Array1<f64> = grad_gamma_pair
                    .slice(s![..3 * m_i.n_atoms, ..])
                    .dot(&dq_pair);

                // self_interaction for atoms in J: grad_gamma_pair[J_local, :].dot(&dq_pair)
                let self_interaction_j: Array1<f64> = grad_gamma_pair
                    .slice(s![3 * m_i.n_atoms.., ..])
                    .dot(&dq_pair);

                // DDq_f is broadcasted into the shape of the gradients.
                let delta_dq_f_i: Array1<f64> = delta_dq_i
                    .broadcast([3, m_i.n_atoms])
                    .unwrap()
                    .reversed_axes()
                    .as_standard_layout()
                    .into_shape([3 * m_i.n_atoms])
                    .unwrap()
                    .to_owned();

                let delta_dq_f_j: Array1<f64> = delta_dq_j
                    .broadcast([3, m_j.n_atoms])
                    .unwrap()
                    .reversed_axes()
                    .as_standard_layout()
                    .into_shape([3 * m_j.n_atoms])
                    .unwrap()
                    .to_owned();

                // The gradient for a in I: DDq_i * (grad_gamma_dot_dq[I] - self_interaction_i)
                gradient.slice_mut(s![m_i.slice.grad]).add_assign(
                    &(&delta_dq_f_i
                        * &(&grad_gamma_dot_dq.slice(s![m_i.slice.grad]) - &self_interaction_i)),
                );

                // The gradient for a in J: DDq_j * (grad_gamma_dot_dq[J] - self_interaction_j)
                gradient.slice_mut(s![m_j.slice.grad]).add_assign(
                    &(&delta_dq_f_j
                        * &(&grad_gamma_dot_dq.slice(s![m_j.slice.grad]) - &self_interaction_j)),
                );

                // === SECOND TERM: a in I,J (Eq. 24 RHS) ===
                // grad_delta_dq * ESP_ij
                let grad_delta_dq: Array1<f64> = get_grad_delta_dq(pair, m_i, m_j);

                // The electrostatic potential (ESP) is collected from the corresponding monomers.
                let mut esp_ij: Array1<f64> = Array1::zeros([pair.n_atoms]);
                esp_ij.slice_mut(s![..m_i.n_atoms]).assign(
                    &(&m_i.properties.esp_q().unwrap()
                        - &gamma.slice(s![m_i.slice.atom, m_j.slice.atom]).dot(&dq_j)),
                );
                esp_ij.slice_mut(s![m_i.n_atoms..]).assign(
                    &(&m_j.properties.esp_q().unwrap()
                        - &gamma.slice(s![m_j.slice.atom, m_i.slice.atom]).dot(&dq_i)),
                );

                // The ESP is transformed into the shape of the gradient.
                let esp_ij_f: Array1<f64> = esp_ij
                    .broadcast([3, pair.n_atoms])
                    .unwrap()
                    .reversed_axes()
                    .as_standard_layout()
                    .into_shape([3 * pair.n_atoms])
                    .unwrap()
                    .to_owned();

                // The (elementwise) product of the ESP with the derivative of the pair charge differences
                let gddq_esp: Array1<f64> = &grad_delta_dq * &esp_ij_f;

                gradient
                    .slice_mut(s![m_i.slice.grad])
                    .add_assign(&gddq_esp.slice(s![..3 * m_i.n_atoms]));
                gradient
                    .slice_mut(s![m_j.slice.grad])
                    .add_assign(&gddq_esp.slice(s![3 * m_i.n_atoms..]));

                // === THIRD TERM: a NOT in I,J (Eq. 25) ===
                // For atoms in K (K != I,J):
                // dq_a^K * sum_b_in_IJ dgamma[a,b] * DDq_b  +  grad_dq_a^K * sum_b_in_IJ gamma[a,b] * DDq_b

                for (k_idx, m_k) in self.monomers.iter().enumerate() {
                    if k_idx == m_i.index || k_idx == m_j.index {
                        continue;
                    }

                    let k_start = m_k.slice.atom_as_range().start;
                    let dq_k: ArrayView1<f64> = dq.slice(s![m_k.slice.atom]);

                    // Compute local gamma gradient matrix [3*n_k, pair.n_atoms] for K-pair interactions
                    let mut grad_gamma_k_pair: Array2<f64> =
                        Array2::zeros([3 * m_k.n_atoms, pair.n_atoms]);

                    for local_k in 0..m_k.n_atoms {
                        let global_k = k_start + local_k;
                        let atom_k = &self.atoms[global_k];

                        // Atoms in I (local indices 0..n_i in pair)
                        for local_b in 0..m_i.n_atoms {
                            let global_b = i_start + local_b;
                            let atom_b = &self.atoms[global_b];

                            let r = atom_k - atom_b;
                            let r_kb = r.norm();
                            let deriv =
                                self.gammafunction.deriv(r_kb, atom_k.number, atom_b.number);
                            let e_kb = r / r_kb;

                            for c in 0..3 {
                                grad_gamma_k_pair[[local_k * 3 + c, local_b]] = e_kb[c] * deriv;
                            }
                        }

                        // Atoms in J (local indices n_i..pair.n_atoms in pair)
                        for local_b in 0..m_j.n_atoms {
                            let global_b = j_start + local_b;
                            let atom_b = &self.atoms[global_b];

                            let r = atom_k - atom_b;
                            let r_kb = r.norm();
                            let deriv =
                                self.gammafunction.deriv(r_kb, atom_k.number, atom_b.number);
                            let e_kb = r / r_kb;

                            for c in 0..3 {
                                grad_gamma_k_pair[[local_k * 3 + c, m_i.n_atoms + local_b]] =
                                    e_kb[c] * deriv;
                            }
                        }
                    }

                    // First part: dq_k * (grad_gamma_k_pair · delta_dq)
                    let grad_gamma_dot_delta_dq: Array1<f64> = grad_gamma_k_pair.dot(&delta_dq);
                    let dq_k_broadcast: Array1<f64> = dq_k
                        .broadcast([3, m_k.n_atoms])
                        .unwrap()
                        .reversed_axes()
                        .as_standard_layout()
                        .into_shape([3 * m_k.n_atoms])
                        .unwrap()
                        .to_owned();
                    gradient
                        .slice_mut(s![m_k.slice.grad])
                        .add_assign(&(&dq_k_broadcast * &grad_gamma_dot_delta_dq));

                    // Second part: grad_dq_a^K * (DDq · gamma[IJ, K])
                    // DDq_i · gamma[I, K] + DDq_j · gamma[J, K]
                    let ddq_gamma_k: Array1<f64> = &delta_dq_i
                        .dot(&gamma.slice(s![m_i.slice.atom, m_k.slice.atom]))
                        + &delta_dq_j.dot(&gamma.slice(s![m_j.slice.atom, m_k.slice.atom]));

                    let ddq_gamma_k_broadcast: Array1<f64> = ddq_gamma_k
                        .broadcast([3, m_k.n_atoms])
                        .unwrap()
                        .reversed_axes()
                        .as_standard_layout()
                        .into_shape([3 * m_k.n_atoms])
                        .unwrap()
                        .to_owned();

                    gradient
                        .slice_mut(s![m_k.slice.grad])
                        .add_assign(&(&grad_dq.slice(s![m_k.slice.grad]) * &ddq_gamma_k_broadcast));
                }
            });

        gradient_array.sum_axis(Axis(1))
    }

    // pub fn embedding_gradient_2(&mut self) -> Array1<f64> {
    //     // The gradient of the embedding energy is initialized as an array with zeros.
    //     let mut gradient_array: Array2<f64> =
    //         Array2::zeros([3 * self.atoms.len(), self.pairs.len()]);
    //
    //     // A reference to the charge differences and gamma matrix for all atoms is created.
    //     let dq: ArrayView1<f64> = self.properties.dq().unwrap();
    //     let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
    //
    //     // TODO: it is not neccessary to calculate the derivative of gamma two times. this should be
    //     // improved! it is already computed in the gradient of the monomer/pair
    //     let grad_gamma_sparse: Array2<f64> =
    //         gamma_gradients_atomwise_2d(&self.gammafunction, &self.atoms, self.atoms.len());
    //     let grad_gamma_dot_dq: Array1<f64> = grad_gamma_sparse.dot(&dq);
    //
    //     // Begin of the loop to compute the gradient of the embedding energy for each pair.
    //     self.pairs
    //         .iter()
    //         .zip(gradient_array.axis_iter_mut(Axis(1)))
    //         .for_each(|(pair, mut gradient)| {
    //             // References to the corresponding monomers.
    //             let m_i: &Monomer = &self.monomers[pair.i];
    //             let m_j: &Monomer = &self.monomers[pair.j];
    //             // Reference to the DDq_a^IJ (difference of charge difference between pair and monomer)
    //             let delta_dq: ArrayView1<f64> = pair.properties.delta_dq().unwrap();
    //
    //             // grad dqs
    //             // let grad_dq_i: ArrayView2<f64> = m_i.properties.grad_dq().unwrap();
    //             // let grad_dq_j: ArrayView2<f64> = m_j.properties.grad_dq().unwrap();
    //             let grad_delta_dq: Array2<f64> = get_grad_delta_dq_2(pair, m_i, m_j);
    //
    //             let gamma_i: Array1<f64> = gamma
    //                 .slice(s![m_i.slice.atom, m_j.slice.atom])
    //                 .dot(&dq.slice(s![m_j.slice.atom]));
    //             let gamma_j: Array1<f64> = gamma
    //                 .slice(s![m_j.slice.atom, m_i.slice.atom])
    //                 .dot(&dq.slice(s![m_i.slice.atom]));
    //             let dq_i: ArrayView1<f64> = dq.slice(s![m_i.slice.atom]);
    //             let dq_j: ArrayView1<f64> = dq.slice(s![m_j.slice.atom]);
    //
    //             // self interactions
    //             let self_interaction_i: Array1<f64> = &grad_gamma_sparse
    //                 .slice(s![m_i.slice.grad, m_i.slice.atom])
    //                 .dot(&dq.slice(s![m_i.slice.atom]))
    //                 + &grad_gamma_sparse
    //                     .slice(s![m_i.slice.grad, m_j.slice.atom])
    //                     .dot(&dq.slice(s![m_j.slice.atom]));
    //
    //             let self_interaction_j: Array1<f64> = &grad_gamma_sparse
    //                 .slice(s![m_j.slice.grad, m_i.slice.atom])
    //                 .dot(&dq.slice(s![m_i.slice.atom]))
    //                 + &grad_gamma_sparse
    //                     .slice(s![m_j.slice.grad, m_j.slice.atom])
    //                     .dot(&dq.slice(s![m_j.slice.atom]));
    //
    //             let grad_gamma_dq_i: Array1<f64> =
    //                 &grad_gamma_dot_dq.slice(s![m_i.slice.grad]) - &self_interaction_i;
    //             let grad_gamma_dq_j: Array1<f64> =
    //                 &grad_gamma_dot_dq.slice(s![m_j.slice.grad]) - &self_interaction_j;
    //
    //             // The electrostatic potential (ESP) is collected from the corresponding monomers.
    //             let mut esp_ij: Array1<f64> = Array1::zeros([pair.n_atoms]);
    //             esp_ij.slice_mut(s![..m_i.n_atoms]).assign(
    //                 &(&m_i.properties.esp_q().unwrap()
    //                     - &gamma
    //                         .slice(s![m_i.slice.atom, m_j.slice.atom])
    //                         .dot(&dq.slice(s![m_j.slice.atom]))),
    //             );
    //             esp_ij.slice_mut(s![m_i.n_atoms..]).assign(
    //                 &(&m_j.properties.esp_q().unwrap()
    //                     - &gamma
    //                         .slice(s![m_j.slice.atom, m_i.slice.atom])
    //                         .dot(&dq.slice(s![m_i.slice.atom]))),
    //             );
    //
    //             // a in IJ
    //             for nc in 0..3 {
    //                 for na in 0..pair.n_atoms {
    //                     let grad_idx: usize = na * 3 + nc;
    //                     // a in I
    //                     if na < m_i.n_atoms {
    //                         let mut gradient_slice: ArrayViewMut1<f64> =
    //                             gradient.slice_mut(s![m_i.slice.grad]);
    //                         // first term
    //                         let tmp_1: f64 = delta_dq[na] * grad_gamma_dq_i[[grad_idx]];
    //                         // second term
    //                         let tmp_2: f64 = grad_delta_dq[[grad_idx, na]] * esp_ij[na];
    //                         // add to gradient
    //                         gradient_slice[grad_idx] += &(tmp_1 + tmp_2);
    //                     }
    //                     // a in J
    //                     else {
    //                         let mut gradient_slice: ArrayViewMut1<f64> =
    //                             gradient.slice_mut(s![m_j.slice.grad]);
    //                         let nat: usize = na - m_i.n_atoms;
    //                         let grad_idx_2: usize = grad_idx - 3 * m_i.n_atoms;
    //                         // first term
    //                         let tmp_1: f64 = delta_dq[na] * grad_gamma_dq_j[[grad_idx_2]];
    //                         // second term
    //                         let tmp_2: f64 = grad_delta_dq[[grad_idx, na]] * esp_ij[na];
    //                         // add to gradient
    //                         gradient_slice[grad_idx_2] += &(tmp_1 + tmp_2);
    //                     }
    //                 }
    //             }
    //
    //             let mut dg_ddq: Array1<f64> = grad_gamma_sparse
    //                 .slice(s![0.., m_i.slice.atom])
    //                 .dot(&delta_dq.slice(s![..m_i.n_atoms]));
    //
    //             // and for a in J.
    //             dg_ddq += &grad_gamma_sparse
    //                 .slice(s![0.., m_j.slice.atom])
    //                 .dot(&delta_dq.slice(s![m_i.n_atoms..]));
    //
    //             // Since K != I,J => the elements where K = I,J are set to zero.
    //             dg_ddq
    //                 .slice_mut(s![m_i.slice.grad])
    //                 .assign(&Array1::zeros([3 * m_i.n_atoms]));
    //             dg_ddq
    //                 .slice_mut(s![m_j.slice.grad])
    //                 .assign(&Array1::zeros([3 * m_j.n_atoms]));
    //
    //             let mut ddq_gamma: Array1<f64> = delta_dq
    //                 .slice(s![..m_i.n_atoms])
    //                 .dot(&gamma.slice(s![m_i.slice.atom, 0..]));
    //
    //             // A in monomer J
    //             ddq_gamma += &delta_dq
    //                 .slice(s![m_i.n_atoms..])
    //                 .dot(&gamma.slice(s![m_j.slice.atom, 0..]));
    //
    //             // Since K != I,J => the elements were K = I,J are set to zero.
    //             ddq_gamma
    //                 .slice_mut(s![m_i.slice.atom])
    //                 .assign(&Array1::zeros([m_i.n_atoms]));
    //             ddq_gamma
    //                 .slice_mut(s![m_j.slice.atom])
    //                 .assign(&Array1::zeros([m_j.n_atoms]));
    //
    //             // a in K
    //             for (idx, mol) in self.monomers.iter().enumerate() {
    //                 if idx != m_i.index && idx != m_j.index {
    //                     let mut gradient_slice: ArrayViewMut1<f64> =
    //                         gradient.slice_mut(s![mol.slice.grad]);
    //                     let dq_mol: ArrayView1<f64> = dq.slice(s![mol.slice.atom]);
    //                     let grad_gamma_ddq: ArrayView1<f64> = dg_ddq.slice(s![mol.slice.grad]);
    //                     let grad_dq: ArrayView2<f64> = mol.properties.grad_dq().unwrap();
    //                     let ddq_gamma_slice: ArrayView1<f64> = ddq_gamma.slice(s![mol.slice.atom]);
    //
    //                     for nc in 0..3 {
    //                         for na in 0..mol.n_atoms {
    //                             let grad_idx: usize = na * 3 + nc;
    //                             // first term
    //                             let tmp_1: f64 = dq_mol[na] * grad_gamma_ddq[[grad_idx]];
    //                             // second term
    //                             let tmp_2: f64 = grad_dq[[grad_idx, na]] * ddq_gamma_slice[na];
    //                             gradient_slice[grad_idx] += &(tmp_1 + tmp_2);
    //                         }
    //                     }
    //                 }
    //             }
    //         });
    //
    //     return gradient_array.sum_axis(Axis(1));
    // }
}

fn get_grad_delta_dq(pair: &Pair, m_i: &Monomer, m_j: &Monomer) -> Array1<f64> {
    // get the derivatives of the charge differences w.r.t to the each degree of freedom
    let grad_dq: ArrayView2<f64> = pair.properties.grad_dq().unwrap();
    let grad_dq_i: ArrayView2<f64> = m_i.properties.grad_dq().unwrap();
    let grad_dq_j: ArrayView2<f64> = m_j.properties.grad_dq().unwrap();

    // compute the difference between dimers and monomers and take the diagonal values
    let mut grad_delta_dq_2d: Array2<f64> = grad_dq.to_owned();

    //difference for monomer i
    grad_delta_dq_2d
        .slice_mut(s![..(3 * m_i.n_atoms), ..m_i.n_atoms])
        .sub_assign(&grad_dq_i);

    // difference for monomer j
    grad_delta_dq_2d
        .slice_mut(s![(3 * m_i.n_atoms).., m_i.n_atoms..])
        .sub_assign(&grad_dq_j);

    let grad_delta_dq_3d: Array3<f64> = grad_delta_dq_2d
        .into_shape([3, pair.n_atoms, pair.n_atoms])
        .unwrap();

    diag_of_last_dimensions(grad_delta_dq_3d)
}

fn get_grad_delta_dq_2(pair: &Pair, m_i: &Monomer, m_j: &Monomer) -> Array2<f64> {
    // get the derivatives of the charge differences w.r.t to the each degree of freedom
    let grad_dq: ArrayView2<f64> = pair.properties.grad_dq().unwrap();
    let grad_dq_i: ArrayView2<f64> = m_i.properties.grad_dq().unwrap();
    let grad_dq_j: ArrayView2<f64> = m_j.properties.grad_dq().unwrap();

    // compute the difference between dimers and monomers and take the diagonal values
    let mut grad_delta_dq_2d: Array2<f64> = grad_dq.to_owned();

    //difference for monomer i
    grad_delta_dq_2d
        .slice_mut(s![..(3 * m_i.n_atoms), ..m_i.n_atoms])
        .sub_assign(&grad_dq_i);

    // difference for monomer j
    grad_delta_dq_2d
        .slice_mut(s![(3 * m_i.n_atoms).., m_i.n_atoms..])
        .sub_assign(&grad_dq_j);

    grad_delta_dq_2d
}

pub fn diag_of_last_dimensions<S>(data: ArrayBase<S, Ix3>) -> Array1<f64>
where
    S: ndarray::Data<Elem = f64>,
{
    let (a, b, c): (usize, usize, usize) = data.dim();
    assert_eq!(b, c, "The last two dimension should have the same length");

    // A temporary array to store the values is created.
    let mut grad_charge: Array2<f64> = Array2::zeros([a, b]);

    // The diagonal of each of the three dimensions is saved.
    for i in 0..a {
        grad_charge
            .slice_mut(s![i, ..])
            .assign(&data.slice(s![i, .., ..]).diag());
    }
    // The gradient of the charges are reshaped into a one dimensional array.
    grad_charge.into_shape([a * b]).unwrap()
}
