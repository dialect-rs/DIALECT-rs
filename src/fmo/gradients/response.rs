#![allow(warnings)]

use crate::excited_states::trans_charges;
use crate::fmo::helpers::get_pair_slice;
use crate::fmo::monomer::Monomer;
use crate::fmo::pair::{Pair, PairType};
use crate::fmo::scc::helpers::atomvec_to_aomat;
use crate::fmo::supersystem::SuperSystem;
use crate::gradients::helpers::{f_lr, f_v};
use crate::initialization::Atom;
use crate::scc::gamma_approximation::{gamma_ao_wise, gamma_gradients_ao_wise};
use crate::scc::h0_and_s::s_gradient;
use ndarray::prelude::*;
use ndarray_linalg::{into_col, into_row, Solve, SVD};
use rayon::prelude::*;
use std::ops::AddAssign;

impl SuperSystem<'_> {
    pub fn solve_response_gradient(&mut self) -> Array1<f64> {
        // solve the z_vector equation
        let z_vectors: Vec<Array1<f64>> = self.get_response_z_vectors();
        let z_vectors_2: Vec<Array1<f64>> = self.get_response_z_vectors_iterative();

        for (z_vec_1, z_vec_2) in z_vectors.iter().zip(z_vectors_2.iter()) {
            println!("Z_vec 1:\n {:.6}", z_vec_1);
            println!("Z_vec 2:\n {:.6}", z_vec_2);
            println!("\n");
        }

        // empty array for gradient
        let mut grad_array: Array1<f64> = Array1::zeros(3 * self.atoms.len());

        // loop over the monomers
        for ((idx, m_i), z_vec) in self.monomers.iter().enumerate().zip(z_vectors_2.iter()) {
            // get the B matrix for this monomer
            let bmat: Array2<f64> = self.get_response_b_matrix(idx);
            // matmul between z_vec and bmat
            let tmp = z_vec.dot(&bmat.t());

            // add to gradient
            grad_array = grad_array + tmp;
        }
        println!("response gradient: {:.6}", grad_array);

        return grad_array;
    }

    pub fn solve_response_gradient_no_z_storage(&mut self) -> Array1<f64> {
        // prepare the calculation
        self.prepare_response_calculation();

        // get the lagrangian
        let lagrangian: Vec<Array1<f64>> = self.get_response_lagrangian();

        // empty array for gradient
        // let mut grad_array: Array1<f64> = Array1::zeros(3 * self.atoms.len());

        // loop over the monomers
        let grad_array: Array1<f64> = self
            .monomers
            .par_iter()
            .enumerate()
            .map(|(idx, m_i)| {
                // get the z_vector
                let z_vector: Array1<f64> = self.get_response_z_vector(m_i.index, &lagrangian);

                // get the product of the b matrix with the z_vector
                let bz_product: Array1<f64> =
                    self.get_product_b_matrix_z_vector(m_i.index, z_vector.view());

                // add to gradient
                bz_product
            })
            .reduce(|| Array1::zeros(3 * self.atoms.len()), |a, b| a + b);
        // for (idx, m_i) in self.monomers.iter().enumerate() {
        //     // get the z_vector
        //     let z_vector:Array1<f64> = self.get_response_z_vector(m_i.index, &lagrangian);

        //     // get the product of the b matrix with the z_vector
        //     let bz_product:Array1<f64> = self.get_product_b_matrix_z_vector(m_i.index, z_vector.view());

        //     // add to gradient
        //     grad_array = grad_array + bz_product;
        // }

        return grad_array;
    }

    pub fn get_response_z_vectors_iterative(&mut self) -> Vec<Array1<f64>> {
        // prepare the calculation
        self.prepare_response_calculation();

        // get the lagrangian
        let lagrangian: Vec<Array1<f64>> = self.get_response_lagrangian();

        // vector for z_vectors
        let z_vec_inital: Vec<Array1<f64>> = self.get_initial_z_vector(&lagrangian);
        let z_vec_inital_2: Vec<Array1<f64>> = self.get_initial_z_vector_2(&lagrangian);
        let z_vec_inital_3: Vec<Array1<f64>> = self.get_initial_z_vector_3(&lagrangian);
        let z_vec_inital_4: Vec<Array1<f64>> = self.get_initial_z_vector_4(&lagrangian);

        println!("Initial z vector 1: {:.10}", z_vec_inital[0]);
        println!("Initial z vector 2: {:.10}", z_vec_inital_2[0]);
        println!("Initial z vector 3: {:.10}", z_vec_inital_3[0]);
        println!("Initial z vector 4: {:.10}", z_vec_inital_4[0]);

        let mut z_new: Vec<Array1<f64>> = z_vec_inital.clone();
        let mut z_old: Vec<Array1<f64>> = z_vec_inital.clone();

        // set maxiter for z_vector routine
        let max_iter: usize = 50;
        // convergence for z_vector
        let conv: f64 = 1.0e-8;

        'zvec_loop: for iter in (0..max_iter) {
            // get the next z_vector
            println!("Calculate new z_vectors!");
            let z_vec_new = self.get_next_z_vector_3(&lagrangian, &z_old);

            // get convergence vector
            let mut convergence: Vec<bool> = Vec::new();

            for ((idx, z_i_new), z_i_old) in z_vec_new.iter().enumerate().zip(z_old.iter()) {
                let diff = z_i_new - z_i_old;
                let diff_sqr = diff.map(|val| val.powi(2));
                let rmsq: f64 = diff_sqr.mean().unwrap().sqrt();

                if rmsq < conv {
                    convergence.push(true);
                } else {
                    convergence.push(false);
                }
            }

            // store new z_vectors
            z_old = z_new;
            z_new = z_vec_new;

            // check convergence
            if !convergence.contains(&false) {
                break 'zvec_loop;
            }
        }

        z_new
    }

    pub fn get_initial_z_vector(&self, lagrangian: &Vec<Array1<f64>>) -> Vec<Array1<f64>> {
        // vector for z_vectors
        let mut z_vec: Vec<Array1<f64>> = Vec::with_capacity(lagrangian.len());

        // loop over the monomers
        for (idx_i, m_i) in self.monomers.iter().enumerate() {
            // get the lagrangian
            let l_k: ArrayView1<f64> = lagrangian[idx_i].view();
            self.response_z_vec_a_diag(idx_i);

            // get the A matrix for the single fragment
            let a_matrix: Array2<f64> = -1.0 * self.get_response_a_matrix_monomer(idx_i);

            // solve the Z vector eq
            let vec: Array1<f64> = self.solve_z_vec_lagrangian_eq(l_k, idx_i, a_matrix.view());

            z_vec.push(vec);
        }

        z_vec
    }

    pub fn get_next_z_vector(
        &self,
        lagrangian: &Vec<Array1<f64>>,
        z_vec: &Vec<Array1<f64>>,
    ) -> Vec<Array1<f64>> {
        // get the interaction lagrangian
        let new_lagrangian: Vec<Array1<f64>> = self.get_interaction_lagrangian(lagrangian, z_vec);

        // storage for new z_vectors
        let mut new_z_vectors: Vec<Array1<f64>> = Vec::with_capacity(z_vec.len());

        for (idx_i, m_i) in self.monomers.iter().enumerate() {
            // get the lagrangian
            let l_k: ArrayView1<f64> = new_lagrangian[idx_i].view();

            // get the A matrix for the single fragment
            let a_matrix: Array2<f64> = -1.0 * self.get_response_a_matrix_monomer(idx_i);

            // solve the Z vector eq
            let vec: Array1<f64> = self.solve_z_vec_lagrangian_eq(l_k, idx_i, a_matrix.view());

            new_z_vectors.push(vec);
        }

        new_z_vectors
    }

    pub fn get_interaction_lagrangian(
        &self,
        lagrangian: &Vec<Array1<f64>>,
        z_vec: &Vec<Array1<f64>>,
    ) -> Vec<Array1<f64>> {
        // new lagrangian vec
        let mut lagrangian_vec: Vec<Array1<f64>> = Vec::with_capacity(lagrangian.len());

        for (m_idx, m_x) in self.monomers.iter().enumerate() {
            // new lagrangian mat
            let mut new_lagrangian: Array1<f64> = lagrangian[m_idx].clone();

            // iterate over all monomers
            for (idx_k, m_k) in self.monomers.iter().enumerate() {
                if idx_k != m_idx {
                    // get the a matrix for 2 monomers
                    let a_matrix: Array2<f64> = -1.0 * self.get_response_a_matrix_xk(idx_k, m_idx);

                    // matrix product between A and Z
                    let a_z: Array1<f64> = a_matrix.dot(&z_vec[idx_k]);

                    // subtract the contribution from the lagrangian
                    new_lagrangian = new_lagrangian - a_z;
                }
            }
            lagrangian_vec.push(new_lagrangian);
        }

        lagrangian_vec
    }

    pub fn solve_z_vec_lagrangian_eq(
        &self,
        lagrangian: ArrayView1<f64>,
        m_idx: usize,
        a_matrix: ArrayView2<f64>,
    ) -> Array1<f64> {
        // do the svd
        let (u, s, vt) = a_matrix.svd(true, true).unwrap();
        let u: Array2<f64> = u.unwrap();
        let vt: Array2<f64> = vt.unwrap();
        let s: Array1<f64> = s;

        // threshold
        let thresh_val: f64 = f64::EPSILON * a_matrix.dim().0 as f64;

        let s_inv: Vec<f64> = s
            .iter()
            .enumerate()
            .map(|(idx, s_val)| {
                if s_val > &thresh_val {
                    1.0 / s_val
                } else {
                    0.0
                }
            })
            .collect();
        // convert to array
        let s_inv: Array1<f64> = Array::from(s_inv);
        // initialize array for inverted singular values
        let mut arr: Array2<f64> = Array2::zeros(a_matrix.raw_dim());
        // get diagonal array from inverted singular values
        let s_inv_2d: Array2<f64> = Array::from_diag(&s_inv);
        arr.slice_mut(s![..s_inv.len(), ..s_inv.len()])
            .assign(&s_inv_2d);

        let tmp: Array2<f64> = vt.t().dot(&arr.t().dot(&u.t()));
        let x: Array1<f64> = tmp.dot(&lagrangian);

        x
    }

    pub fn get_response_z_vectors(&mut self) -> Vec<Array1<f64>> {
        // prepare the calculation
        self.prepare_response_calculation();

        // get the lagrangian
        let lagrangian: Vec<Array1<f64>> = self.get_response_lagrangian();

        // vector for z_vectors
        let mut z_vec: Vec<Array1<f64>> = Vec::new();

        // loop over the monomers
        for (idx_i, m_i) in self.monomers.iter().enumerate() {
            // get nocc and nvirt of monomer X
            let nocc: usize = m_i.properties.occ_indices().unwrap().len();
            let nvirt: usize = m_i.properties.virt_indices().unwrap().len();

            // prepare the z_vector matrix
            let mut z_mat: Array1<f64> = Array1::zeros([nvirt * nocc]);

            // loop over all monomers
            for (idx_k, m_k) in self.monomers.iter().enumerate() {
                // get the a matrix
                // let a_matrix_full: Array2<f64> =
                //     -1.0 * self.get_full_response_a_matrix(idx_i, idx_k);
                let a_matrix: Array2<f64> = -1.0 * self.get_response_a_matrix(idx_i, idx_k);
                // get the lagrangian
                let l_k: ArrayView1<f64> = lagrangian[idx_k].view();
                // solve the linear equations A z = L
                // let mat: Array1<f64> = a_matrix.least_squares(&l_k).unwrap().solution;

                // // solve the full equation A z = L
                // println!("Test solve");
                // println!("Test1");
                // let x: Array1<f64> = a_matrix_full.t().solve(&l_k).unwrap();
                // println!("Test2");
                // let x: Array2<f64> = x.into_shape([nocc + nvirt, nocc + nvirt]).unwrap();
                // println!("Test3");
                // let x_2: Array1<f64> = x
                //     .slice(s![nocc.., ..nocc])
                //     .to_owned()
                //     .into_shape(nvirt * nocc)
                //     .unwrap();

                // do the svd
                let (u, s, vt) = a_matrix.svd(true, true).unwrap();
                let u: Array2<f64> = u.unwrap();
                let vt: Array2<f64> = vt.unwrap();
                let s: Array1<f64> = s;

                // threshold
                let thresh_val: f64 = f64::EPSILON * a_matrix.dim().0 as f64;

                let s_inv: Vec<f64> = s
                    .iter()
                    .enumerate()
                    .map(|(idx, s_val)| {
                        if s_val > &thresh_val {
                            1.0 / s_val
                        } else {
                            0.0
                        }
                    })
                    .collect();
                // convert to array
                let s_inv: Array1<f64> = Array::from(s_inv);
                // initialize array for inverted singular values
                let mut arr: Array2<f64> = Array2::zeros(a_matrix.raw_dim());
                // get diagonal array from inverted singular values
                let s_inv_2d: Array2<f64> = Array::from_diag(&s_inv);
                arr.slice_mut(s![..s_inv.len(), ..s_inv.len()])
                    .assign(&s_inv_2d);

                let tmp: Array2<f64> = vt.t().dot(&arr.t().dot(&u.t()));
                let x: Array1<f64> = tmp.dot(&l_k);

                // println!("X_svd {:.7}", x);
                // println!("X_solve {:.7}", x_2);

                // add to z_mat
                z_mat = z_mat + x;
            }
            // append z_vec
            z_vec.push(z_mat);
        }
        z_vec
    }

    pub fn get_response_z_vector(&self, m_idx: usize, lagrangian: &[Array1<f64>]) -> Array1<f64> {
        // get nocc and nvirt of monomer X
        let nocc: usize = self.monomers[m_idx].properties.occ_indices().unwrap().len();
        let nvirt: usize = self.monomers[m_idx].properties.occ_indices().unwrap().len();

        // prepare the z_vector matrix
        let mut z_mat: Array1<f64> = Array1::zeros([nvirt * nocc]);

        // loop over all monomers
        for (idx_k, m_k) in self.monomers.iter().enumerate() {
            // get the a matrix
            let a_matrix: Array2<f64> = -1.0 * self.get_response_a_matrix(m_idx, idx_k);
            // get the lagrangian
            let l_k: ArrayView1<f64> = lagrangian[idx_k].view();

            // solve the linear equations A z = L
            let mat: Array1<f64> = a_matrix.t().solve(&l_k).unwrap();

            // add to z_mat
            z_mat = z_mat + mat;
        }

        z_mat
    }

    pub fn get_product_b_matrix_z_vector(
        &self,
        m_idx: usize,
        z_vector: ArrayView1<f64>,
    ) -> Array1<f64> {
        // get the monomer
        let m_x: &Monomer = &self.monomers[m_idx];

        // get grad_s and grad_h from properties
        let grad_s: ArrayView3<f64> = m_x.properties.grad_s().unwrap();
        let grad_h0: ArrayView3<f64> = m_x.properties.grad_h0().unwrap();

        // get necessary arrays from properties
        let diff_p: Array2<f64> = &m_x.properties.p().unwrap() - &m_x.properties.p_ref().unwrap();
        let g0_ao: ArrayView2<f64> = m_x.properties.gamma_ao().unwrap();
        let g0: ArrayView2<f64> = m_x.properties.gamma().unwrap();
        let g0_lr: ArrayView2<f64> = m_x.properties.gamma_lr().unwrap();
        let g1_ao: ArrayView3<f64> = m_x.properties.grad_gamma_ao().unwrap();
        let s: ArrayView2<f64> = m_x.properties.s().unwrap();
        let orbs: ArrayView2<f64> = m_x.properties.orbs().unwrap();
        let orbe: ArrayView1<f64> = m_x.properties.orbe().unwrap();

        // calculate grad_Hxc
        let f_dmd0: Array3<f64> = f_v(
            diff_p.view(),
            s,
            grad_s.view(),
            g0_ao,
            g1_ao,
            m_x.n_atoms,
            m_x.n_orbs,
        );
        // calulcate gradH
        let mut grad_h: Array3<f64> = &grad_h0 + &f_dmd0;

        // add the lc-gradient of the hamiltonian
        if self.gammafunction_lc.is_some() {
            let g1lr_ao: ArrayView3<f64> = self.properties.grad_gamma_lr_ao().unwrap();
            let g0lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();

            let flr_dmd0: Array3<f64> = f_lr(
                diff_p.view(),
                s,
                grad_s.view(),
                g0lr_ao,
                g1lr_ao,
                m_x.n_atoms,
                m_x.n_orbs,
            );
            grad_h = grad_h - 0.5 * &flr_dmd0;
        }

        // get MO coefficient for the occupied or virtual orbital
        let occ_indices: &[usize] = m_x.properties.occ_indices().unwrap();
        let virt_indices: &[usize] = m_x.properties.virt_indices().unwrap();
        let nocc: usize = occ_indices.len();
        let nvirt: usize = virt_indices.len();
        let orbs_occ: ArrayView2<f64> = orbs.slice(s![.., ..nocc]);

        // get the transition charges
        let qov: ArrayView2<f64> = m_x.properties.q_ov().unwrap();
        let qoo: ArrayView2<f64> = m_x.properties.q_oo().unwrap();
        let qvv: ArrayView2<f64> = m_x.properties.q_vv().unwrap();
        // virtual-occupied transition charges
        let qvo: Array2<f64> = qov
            .clone()
            .into_shape([m_x.n_atoms, nocc, nvirt])
            .unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([m_x.n_atoms, nvirt * nocc])
            .unwrap();

        // initialize b_mat
        let mut b_mat: Array3<f64> = Array3::zeros([3 * m_x.n_atoms, nvirt, nocc]);
        let mut grad_arr: Array1<f64> = Array1::zeros(self.atoms.len());

        // Calculate integrals partwise before iteration over gradient
        // integral (ij|kl) - (ik|jl), i = nvirt, j = nocc
        let mut integral_vo_2d: Array2<f64> = (2.0 * qvo.t().dot(&g0.dot(&qoo))
            - qvo
                .t()
                .dot(&g0_lr.dot(&qoo))
                .into_shape([nvirt, nocc, nocc, nocc])
                .unwrap()
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned()
                .into_shape([nvirt * nocc, nocc * nocc])
                .unwrap());

        // gamma matrix of the full system
        let gamma_full: ArrayView2<f64> = self.properties.gamma().unwrap();

        // loop over all monomers
        // add contribution of other monomers to the integral
        for (idx_k, m_k) in self.monomers.iter().enumerate() {
            if m_k.index != m_x.index {
                // get qoo for k
                let qoo_k: ArrayView2<f64> = m_k.properties.q_oo().unwrap();
                // get the gamma_xk matrix
                let gamma_xk: ArrayView2<f64> =
                    gamma_full.slice(s![m_x.slice.atom, m_k.slice.atom]);

                let integral_vo_2d_k: Array2<f64> = 2.0 * qvo.t().dot(&gamma_xk.dot(&qoo_k));

                // contribution of K to the H-derivative
                // get properties of K
                let diff_p_k: Array2<f64> =
                    &m_k.properties.p().unwrap() - &m_k.properties.p_ref().unwrap();

                // check the pair type
                let type_pair: PairType = self.properties.type_of_pair(m_x.index, m_k.index);

                // get grads of k
                let grads_k = m_k.properties.grad_s().unwrap();
                // get orbs of k
                let orbs_k: ArrayView2<f64> = m_k.properties.orbs().unwrap();
                let nocc_k: usize = m_k.properties.occ_indices().unwrap().len();
                let occ_orbs_k: ArrayView2<f64> = orbs_k.slice(s![.., ..nocc_k]);

                if type_pair == PairType::Pair {
                    // get pair index
                    let index = self.properties.index_of_pair(m_x.index, m_k.index);
                    // get the pair
                    let pair: &Pair = &self.pairs[index];
                    // get the atoms
                    let pair_atoms: Vec<Atom> = get_pair_slice(
                        &self.atoms,
                        m_x.slice.atom_as_range(),
                        m_k.slice.atom_as_range(),
                    );
                    // get the gamma derivative
                    let (g1_pair, g1_ao_pair): (Array3<f64>, Array3<f64>) = gamma_gradients_ao_wise(
                        &pair.gammafunction,
                        &pair_atoms,
                        pair.n_atoms,
                        pair.n_orbs,
                    );
                    // slice the gamma matrix
                    let g1_ao_slice: ArrayView3<f64> =
                        g1_ao_pair.slice(s![.., ..m_x.n_orbs, m_x.n_orbs..]);

                    // calculate gamma ao
                    let (g0_pair, g0_ao_pair) =
                        gamma_ao_wise(&pair.gammafunction, &pair_atoms, pair.n_atoms, pair.n_orbs);
                    // slice gamma ao
                    let g0_ao_slice: ArrayView2<f64> =
                        g0_ao_pair.slice(s![..m_x.n_orbs, m_x.n_orbs..]);

                    // get grad s
                    let grad_s_xk: Array3<f64> = s_gradient(&pair_atoms, pair.n_orbs, pair.slako);

                    // get the contribution to the hamiltonian
                    let h_xk: Array3<f64> = f_v_xk(
                        diff_p_k.view(),
                        m_x.properties.s().unwrap(),
                        m_k.properties.s().unwrap(),
                        grad_s_xk.slice(s![.., ..m_x.n_orbs, ..m_x.n_orbs]),
                        grad_s_xk.slice(s![.., m_x.n_orbs.., m_x.n_orbs..]),
                        g0_ao_slice,
                        g1_ao_slice,
                        pair.n_atoms,
                        m_x.n_orbs,
                    );

                    // get the part of the b matrix
                    let b_slice_x = h_xk.slice(s![..3 * m_x.n_atoms, nocc.., ..nocc]);
                    let b_slice_k = h_xk.slice(s![3 * m_x.n_atoms.., nocc.., ..nocc]);
                    // and reshape them
                    let b_slice_x_2d = b_slice_x
                        .into_shape([3 * m_x.n_atoms, nvirt * nocc])
                        .unwrap();
                    let b_slice_k_2d = b_slice_k
                        .into_shape([3 * m_k.n_atoms, nvirt * nocc])
                        .unwrap();

                    // get the product with the z_vector
                    let bz_prod_x: Array1<f64> = z_vector.dot(&b_slice_x_2d.t());
                    let bz_prod_k: Array1<f64> = z_vector.dot(&b_slice_k_2d.t());

                    // add to the gradient
                    grad_arr
                        .slice_mut(s![m_x.slice.grad])
                        .add_assign(&bz_prod_x);
                    grad_arr
                        .slice_mut(s![m_k.slice.grad])
                        .add_assign(&bz_prod_k);
                }
                let mut b_mat_tmp: Array3<f64> = Array3::zeros([3 * m_k.n_atoms, nvirt, nocc]);

                // Calculate the B matrix
                for nc in 0..3 * m_k.n_atoms {
                    let ds_mo: Array1<f64> = occ_orbs_k
                        .t()
                        .dot(&grads_k.slice(s![nc, .., ..]).dot(&occ_orbs_k))
                        .into_shape([nocc * nocc])
                        .unwrap();

                    // integral (ij|kl) - (ik|jl), i = nvirt, j = nocc
                    let integral_vo: Array2<f64> = integral_vo_2d
                        .dot(&ds_mo)
                        .into_shape([nvirt, nocc])
                        .unwrap();

                    // loop version
                    for i in 0..nvirt {
                        for j in 0..nocc {
                            b_mat_tmp[[nc, nocc + i, j]] += -integral_vo[[i, j]];
                        }
                    }
                }

                // reshape the bmat
                let b_mat_tmp: Array2<f64> = b_mat_tmp
                    .into_shape([3 * m_k.n_atoms, nvirt * nocc])
                    .unwrap();
                // product with the z_vector
                let bz_prod_k: Array1<f64> = z_vector.dot(&b_mat_tmp.t());
                // add to the gradient
                grad_arr
                    .slice_mut(s![m_k.slice.grad])
                    .add_assign(&bz_prod_k);
            }
        }

        // Calculate the B matrix
        for nc in 0..3 * m_x.n_atoms {
            let ds_mo: Array1<f64> = orbs_occ
                .t()
                .dot(&grad_s.slice(s![nc, .., ..]).dot(&orbs_occ))
                .into_shape([nocc * nocc])
                .unwrap();

            // integral (ij|kl) - (ik|jl), i = nvirt, j = nocc
            let integral_vo: Array2<f64> = integral_vo_2d
                .dot(&ds_mo)
                .into_shape([nvirt, nocc])
                .unwrap();

            let gradh_mo: Array2<f64> = orbs.t().dot(&grad_h.slice(s![nc, .., ..]).dot(&orbs));
            let grads_mo: Array2<f64> = orbs.t().dot(&grad_s.slice(s![nc, .., ..]).dot(&orbs));

            // loop version
            for i in 0..nvirt {
                for j in 0..nocc {
                    b_mat[[nc, nocc + i, j]] = gradh_mo[[nocc + i, j]]
                        - grads_mo[[nocc + i, j]] * orbe[j]
                        - integral_vo[[i, j]];
                }
            }
        }

        // reshape b matrix
        let b_mat: Array2<f64> = b_mat.into_shape([3 * m_x.n_atoms, nvirt * nocc]).unwrap();
        // dot product between the b matrix and the z_vector
        let bz_product: Array1<f64> = z_vector.dot(&b_mat.t());

        // add the product to the gradient
        grad_arr
            .slice_mut(s![m_x.slice.grad])
            .add_assign(&bz_product);

        // return gradient
        grad_arr
    }

    pub fn get_response_b_matrix(&self, m_idx: usize) -> Array2<f64> {
        // get the monomer
        let m_x: &Monomer = &self.monomers[m_idx];

        // get grad_s and grad_h from properties
        let grad_s: ArrayView3<f64> = m_x.properties.grad_s().unwrap();
        let grad_h0: ArrayView3<f64> = m_x.properties.grad_h0().unwrap();

        // get necessary arrays from properties
        let diff_p: Array2<f64> = &m_x.properties.p().unwrap() - &m_x.properties.p_ref().unwrap();
        let g0_ao: ArrayView2<f64> = m_x.properties.gamma_ao().unwrap();
        let g0: ArrayView2<f64> = m_x.properties.gamma().unwrap();
        let g0_lr: ArrayView2<f64> = m_x.properties.gamma_lr().unwrap();
        let g1_ao: ArrayView3<f64> = m_x.properties.grad_gamma_ao().unwrap();
        let s: ArrayView2<f64> = m_x.properties.s().unwrap();
        let orbs: ArrayView2<f64> = m_x.properties.orbs().unwrap();
        let orbe: ArrayView1<f64> = m_x.properties.orbe().unwrap();

        // calculate grad_Hxc
        let f_dmd0: Array3<f64> = f_v(
            diff_p.view(),
            s,
            grad_s.view(),
            g0_ao,
            g1_ao,
            m_x.n_atoms,
            m_x.n_orbs,
        );
        // calulcate gradH
        let mut grad_h: Array3<f64> = &grad_h0 + &f_dmd0;

        // add the lc-gradient of the hamiltonian
        if m_x.gammafunction_lc.is_some() {
            let g1lr_ao: ArrayView3<f64> = m_x.properties.grad_gamma_lr_ao().unwrap();
            let g0lr_ao: ArrayView2<f64> = m_x.properties.gamma_lr_ao().unwrap();

            let flr_dmd0: Array3<f64> = f_lr(
                diff_p.view(),
                s,
                grad_s.view(),
                g0lr_ao,
                g1lr_ao,
                m_x.n_atoms,
                m_x.n_orbs,
            );
            grad_h = grad_h - 0.5 * &flr_dmd0;
        }

        // get MO coefficient for the occupied or virtual orbital
        let occ_indices: &[usize] = m_x.properties.occ_indices().unwrap();
        let virt_indices: &[usize] = m_x.properties.virt_indices().unwrap();
        let nocc: usize = occ_indices.len();
        let nvirt: usize = virt_indices.len();
        let orbs_occ: ArrayView2<f64> = orbs.slice(s![.., ..nocc]);

        // get the transition charges
        let qov: ArrayView2<f64> = m_x.properties.q_ov().unwrap();
        let qoo: ArrayView2<f64> = m_x.properties.q_oo().unwrap();
        let qvv: ArrayView2<f64> = m_x.properties.q_vv().unwrap();
        // virtual-occupied transition charges
        let qvo: Array2<f64> = qov
            .clone()
            .into_shape([m_x.n_atoms, nocc, nvirt])
            .unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([m_x.n_atoms, nvirt * nocc])
            .unwrap();

        // initialize b_mat
        let mut b_mat: Array3<f64> = Array3::zeros([3 * m_x.n_atoms, nvirt, nocc]);
        let mut b_mat_full: Array3<f64> = Array3::zeros([3 * self.atoms.len(), nvirt, nocc]);

        // Calculate integrals partwise before iteration over gradient
        // integral (ij|kl) - 0.5 * [(ik|lj) + (il|kj)], i = nvirt, j = nocc, k = nocc, l = nocc
        let mut integral_vo_2d: Array2<f64> = 2.0 * qvo.t().dot(&g0.dot(&qoo))
            - 0.5
                * (qvo
                    .t()
                    .dot(&g0_lr.dot(&qoo))
                    .into_shape([nvirt, nocc, nocc, nocc])
                    .unwrap()
                    .permuted_axes([0, 3, 1, 2])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nocc, nocc * nocc])
                    .unwrap())
            - 0.5
                * (qvo
                    .t()
                    .dot(&g0_lr.dot(&qoo))
                    .into_shape([nvirt, nocc, nocc, nocc])
                    .unwrap()
                    .permuted_axes([0, 3, 2, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nocc, nocc * nocc])
                    .unwrap());

        // gamma matrix of the full system
        let gamma_full: ArrayView2<f64> = self.properties.gamma().unwrap();

        // loop over all monomers
        // add contribution of other monomers to the integral
        for (idx_k, m_k) in self.monomers.iter().enumerate() {
            if m_k.index != m_x.index {
                // get qoo for k
                let qoo_k: ArrayView2<f64> = m_k.properties.q_oo().unwrap();
                // get the gamma_xk matrix
                let gamma_xk: ArrayView2<f64> =
                    gamma_full.slice(s![m_x.slice.atom, m_k.slice.atom]);

                let integral_vo_2d_k: Array2<f64> = 2.0 * qvo.t().dot(&gamma_xk.dot(&qoo_k));

                // contribution of K to the H-derivative
                // get properties of K
                let diff_p_k: Array2<f64> =
                    &m_k.properties.p().unwrap() - &m_k.properties.p_ref().unwrap();

                // check the pair type
                let type_pair: PairType = self.properties.type_of_pair(m_x.index, m_k.index);

                // get grads of k
                let grads_k = m_k.properties.grad_s().unwrap();
                // get orbs of k
                let orbs_k: ArrayView2<f64> = m_k.properties.orbs().unwrap();
                let nocc_k: usize = m_k.properties.occ_indices().unwrap().len();
                let occ_orbs_k: ArrayView2<f64> = orbs_k.slice(s![.., ..nocc_k]);

                if type_pair == PairType::Pair {
                    // get pair index
                    let index = self.properties.index_of_pair(m_x.index, m_k.index);
                    // get the pair
                    let pair: &Pair = &self.pairs[index];
                    // get the atoms
                    let pair_atoms: Vec<Atom> = get_pair_slice(
                        &self.atoms,
                        m_x.slice.atom_as_range(),
                        m_k.slice.atom_as_range(),
                    );
                    // get the gamma derivative
                    let (g1_pair, g1_ao_pair): (Array3<f64>, Array3<f64>) = gamma_gradients_ao_wise(
                        &pair.gammafunction,
                        &pair_atoms,
                        pair.n_atoms,
                        pair.n_orbs,
                    );
                    // slice the gamma matrix
                    let g1_ao_slice: ArrayView3<f64> =
                        g1_ao_pair.slice(s![.., ..m_x.n_orbs, m_x.n_orbs..]);

                    // calculate gamma ao
                    let (g0_pair, g0_ao_pair) =
                        gamma_ao_wise(&pair.gammafunction, &pair_atoms, pair.n_atoms, pair.n_orbs);
                    // slice gamma ao
                    let g0_ao_slice: ArrayView2<f64> =
                        g0_ao_pair.slice(s![..m_x.n_orbs, m_x.n_orbs..]);

                    // get grad s
                    let grad_s_xk: Array3<f64> = s_gradient(&pair_atoms, pair.n_orbs, pair.slako);

                    // get the contribution to the hamiltonian
                    let h_xk: Array3<f64> = f_v_xk(
                        diff_p_k.view(),
                        m_x.properties.s().unwrap(),
                        m_k.properties.s().unwrap(),
                        grad_s_xk.slice(s![.., ..m_x.n_orbs, ..m_x.n_orbs]),
                        grad_s_xk.slice(s![.., m_x.n_orbs.., m_x.n_orbs..]),
                        g0_ao_slice,
                        g1_ao_slice,
                        pair.n_atoms,
                        m_x.n_orbs,
                    );

                    // add to b matrix
                    b_mat_full
                        .slice_mut(s![m_x.slice.grad, .., ..])
                        .add_assign(&h_xk.slice(s![..3 * m_x.n_atoms, nocc.., ..nocc]));
                    b_mat_full
                        .slice_mut(s![m_k.slice.grad, .., ..])
                        .add_assign(&h_xk.slice(s![3 * m_x.n_atoms.., nocc.., ..nocc]));
                }
                let mut b_mat_slice = b_mat_full.slice_mut(s![m_k.slice.grad, .., ..]);

                // Calculate the B matrix
                for nc in 0..3 * m_k.n_atoms {
                    let ds_mo: Array1<f64> = occ_orbs_k
                        .t()
                        .dot(&grads_k.slice(s![nc, .., ..]).dot(&occ_orbs_k))
                        .into_shape([nocc * nocc])
                        .unwrap();

                    // integral (ij|kl) - (ik|jl), i = nvirt, j = nocc
                    let integral_vo: Array2<f64> = integral_vo_2d_k
                        .dot(&ds_mo)
                        .into_shape([nvirt, nocc])
                        .unwrap();

                    // loop version
                    for i in 0..nvirt {
                        for j in 0..nocc {
                            b_mat_slice[[nc, i, j]] += -integral_vo[[i, j]];
                        }
                    }
                }
            }
        }

        // Calculate the B matrix
        for nc in 0..3 * m_x.n_atoms {
            let ds_mo: Array1<f64> = orbs_occ
                .t()
                .dot(&grad_s.slice(s![nc, .., ..]).dot(&orbs_occ))
                .into_shape([nocc * nocc])
                .unwrap();

            // integral (ij|kl) - (ik|jl), i = nvirt, j = nocc
            let integral_vo: Array2<f64> = integral_vo_2d
                .dot(&ds_mo)
                .into_shape([nvirt, nocc])
                .unwrap();

            let gradh_mo: Array2<f64> = orbs.t().dot(&grad_h.slice(s![nc, .., ..]).dot(&orbs));
            let grads_mo: Array2<f64> = orbs.t().dot(&grad_s.slice(s![nc, .., ..]).dot(&orbs));

            // loop version
            for i in 0..nvirt {
                for j in 0..nocc {
                    b_mat[[nc, i, j]] = gradh_mo[[nocc + i, j]]
                        - grads_mo[[nocc + i, j]] * orbe[j]
                        - integral_vo[[i, j]];
                }
            }
        }

        // add b matrices
        b_mat_full
            .slice_mut(s![m_x.slice.grad, .., ..])
            .add_assign(&b_mat);

        let b_mat_full: Array2<f64> = b_mat_full
            .into_shape([3 * self.atoms.len(), nvirt * nocc])
            .unwrap();

        // return bmatrix
        b_mat_full
    }

    pub fn get_response_lagrangian(&self) -> Vec<Array1<f64>> {
        // lagrangian vector
        let mut lagrangian_vector: Vec<Array1<f64>> = Vec::new();

        // loop over all monomers
        for (idx, m_k) in self.monomers.iter().enumerate() {
            // slice the complete gamma matrix
            let gamma_full: ArrayView2<f64> = self.properties.gamma().unwrap();
            let gamma_slice: ArrayView2<f64> = gamma_full.slice(s![m_k.slice.atom, ..]);

            // The atoms are in general a non-contiguous range of the atoms
            let atoms: &[Atom] = &self.atoms[m_k.slice.atom_as_range()];

            // get the MO coefficients of monomer k
            let orbs: ArrayView2<f64> = m_k.properties.orbs().unwrap();

            // loop over all pairs
            let mat: Array2<f64> = self
                .pairs
                .par_iter()
                .enumerate()
                .map(|(idx_p, pair)| {
                    // get the monomers of the pair
                    let m_i: &Monomer = &self.monomers[pair.i];
                    let m_j: &Monomer = &self.monomers[pair.i];

                    if m_i.index != m_k.index && m_j.index != m_k.index {
                        // get the gamma slice
                        let mut g_arr: Array2<f64> =
                            Array2::zeros([m_k.n_atoms, m_i.n_atoms + m_j.n_atoms]);
                        g_arr
                            .slice_mut(s![.., ..m_i.n_atoms])
                            .assign(&gamma_slice.slice(s![.., m_i.slice.atom]));
                        g_arr
                            .slice_mut(s![.., m_i.n_atoms..])
                            .assign(&gamma_slice.slice(s![.., m_j.slice.atom]));

                        // get the difference charges
                        let ddq: ArrayView1<f64> = pair.properties.delta_dq().unwrap();
                        // get the esp interaction
                        let esp: Array1<f64> = g_arr.dot(&ddq);

                        // and convert it into a matrix in AO basis
                        let omega: Array2<f64> = atomvec_to_aomat(esp.view(), m_k.n_orbs, atoms);

                        // multiply with the overlap matrix
                        let esp_mat: Array2<f64> = 0.5 * omega * &m_k.properties.s().unwrap();

                        // transform to MO basis
                        let esp_mo: Array2<f64> = orbs.t().dot(&esp_mat.dot(&orbs));

                        esp_mo
                    } else {
                        Array2::zeros([m_k.n_orbs, m_k.n_orbs])
                    }
                })
                .reduce(|| Array2::zeros((m_k.n_orbs, m_k.n_orbs)), |a, b| a + b);

            lagrangian_vector.push(mat.into_shape([m_k.n_orbs * m_k.n_orbs]).unwrap());
        }

        lagrangian_vector
    }

    pub fn prepare_response_calculation(&mut self) {
        // iterate over all monomers
        for (idx, mut m_x) in self.monomers.iter_mut().enumerate() {
            // get the monomer atoms
            let x_atoms: &[Atom] = &self.atoms[m_x.slice.atom_as_range()];

            // calculate transition charges
            let (qov, qoo, qvv) = trans_charges(
                m_x.n_atoms,
                x_atoms,
                m_x.properties.orbs().unwrap(),
                m_x.properties.s().unwrap(),
                m_x.properties.occ_indices().unwrap(),
                m_x.properties.virt_indices().unwrap(),
            );

            m_x.properties.set_q_ov(qov);
            m_x.properties.set_q_oo(qoo);
            m_x.properties.set_q_vv(qvv);
        }
    }

    pub fn get_response_a_matrix(&self, x_idx: usize, k_idx: usize) -> Array2<f64> {
        // get the monomers x and k
        let m_x: &Monomer = &self.monomers[x_idx];
        let m_k: &Monomer = &self.monomers[k_idx];

        // The atoms are in general a non-contiguous range of the atoms
        let x_atoms: &[Atom] = &self.atoms[m_x.slice.atom_as_range()];
        let k_atoms: &[Atom] = &self.atoms[m_k.slice.atom_as_range()];

        // get nocc and nvirt of monomer X
        let nocc: usize = m_x.properties.occ_indices().unwrap().len();
        let nvirt: usize = m_x.properties.virt_indices().unwrap().len();
        let nocc_k: usize = m_k.properties.occ_indices().unwrap().len();
        let nvirt_k: usize = m_k.properties.virt_indices().unwrap().len();

        // get the gamma matrices of X
        let g0: ArrayView2<f64> = m_x.properties.gamma().unwrap();
        let g0_lr: ArrayView2<f64> = m_x.properties.gamma_lr().unwrap();

        // get the gamma matrix between X and K
        let g_full: ArrayView2<f64> = self.properties.gamma().unwrap();
        let g_xk: ArrayView2<f64> = g_full.slice(s![m_x.slice.atom, m_k.slice.atom]);

        // get the orbital energies of x
        let orbe = m_x.properties.orbe().unwrap();

        // get the transition charges
        let qov: ArrayView2<f64> = m_x.properties.q_ov().unwrap();
        let qoo: ArrayView2<f64> = m_x.properties.q_oo().unwrap();
        let qvv: ArrayView2<f64> = m_x.properties.q_vv().unwrap();

        // virtual-occupied transition charges
        let qvo: Array2<f64> = qov
            .clone()
            .into_shape([m_x.n_atoms, nocc, nvirt])
            .unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([m_x.n_atoms, nvirt * nocc])
            .unwrap();

        let qov_k: ArrayView2<f64> = m_k.properties.q_ov().unwrap();
        let qoo_k: ArrayView2<f64> = m_k.properties.q_oo().unwrap();
        let qvv_k: ArrayView2<f64> = m_k.properties.q_vv().unwrap();

        // virtual-occupied transition charges
        let qvo_k: Array2<f64> = qov_k
            .clone()
            .into_shape([m_k.n_atoms, nocc_k, nvirt_k])
            .unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([m_k.n_atoms, nvirt_k * nocc_k])
            .unwrap();

        // calculate A_matrix ij,kl i = nvirt, j = nocc, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_vo: Array2<f64> = Array2::zeros([nvirt_k * nocc_k, nvirt * nocc]);
        // integral (ij|kl)
        a_mat_vo = 4.0 * qvo_k.t().dot(&g_xk.t().dot(&qvo));

        if m_x.index == m_k.index {
            // integral (ik|jl)
            a_mat_vo = a_mat_vo
                - qvv
                    .t()
                    .dot(&g0_lr.dot(&qoo))
                    .into_shape([nvirt, nvirt, nocc, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nocc, nvirt * nocc])
                    .unwrap();
            // integral (il|jk)
            a_mat_vo = a_mat_vo
                - qvo
                    .t()
                    .dot(&g0_lr.dot(&qov))
                    .into_shape([nvirt, nocc, nocc, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nocc, nvirt * nocc])
                    .unwrap();
        }

        // calculate A_matrix ij,kl i = nocc, j = nocc, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_oo: Array2<f64> = Array2::zeros([nocc_k * nocc_k, nvirt * nocc]);
        // integral (ij|kl)
        a_mat_oo = 4.0 * qoo_k.t().dot(&g_xk.t().dot(&qvo));

        if m_x.index == m_k.index {
            // integral (ik|jl)
            a_mat_oo = a_mat_oo
                - qov
                    .t()
                    .dot(&g0_lr.dot(&qoo))
                    .into_shape([nocc, nvirt, nocc, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nocc, nvirt * nocc])
                    .unwrap();
            // integral (il|jk)
            a_mat_oo = a_mat_oo
                - qoo
                    .t()
                    .dot(&g0_lr.dot(&qov))
                    .into_shape([nocc, nocc, nocc, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nocc, nvirt * nocc])
                    .unwrap();
        }

        // calculate A_matrix ij,kl i = nocc, j = nvirt, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_ov: Array2<f64> = Array2::zeros([nocc_k * nvirt_k, nvirt * nocc]);
        // integral (ij|kl)
        a_mat_ov = 4.0 * qov_k.t().dot(&g_xk.t().dot(&qvo));

        if m_x.index == m_k.index {
            // integral (ik|jl)
            a_mat_ov = a_mat_ov
                - qov
                    .t()
                    .dot(&g0_lr.dot(&qvo))
                    .into_shape([nocc, nvirt, nvirt, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nvirt, nvirt * nocc])
                    .unwrap();
            // integral (il|jk)
            a_mat_ov = a_mat_ov
                - qoo
                    .t()
                    .dot(&g0_lr.dot(&qvv))
                    .into_shape([nocc, nocc, nvirt, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nvirt, nvirt * nocc])
                    .unwrap();
        }

        // calculate A_matrix ij,kl i = nvirt, j = nvirt, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_vv: Array2<f64> = Array2::zeros([nvirt_k * nvirt_k, nvirt * nocc]);
        // integral (ij|kl)
        a_mat_vv = 4.0 * qvv_k.t().dot(&g_xk.t().dot(&qvo));

        if m_x.index == m_k.index {
            // integral (ik|jl)
            a_mat_vv = a_mat_vv
                - qvv
                    .t()
                    .dot(&g0_lr.dot(&qvo))
                    .into_shape([nvirt, nvirt, nvirt, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nvirt, nvirt * nocc])
                    .unwrap();
            // integral (il|jk)
            a_mat_vv = a_mat_vv
                - qvo
                    .t()
                    .dot(&g0_lr.dot(&qvv))
                    .into_shape([nvirt, nocc, nvirt, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nvirt, nvirt * nocc])
                    .unwrap();
        }

        let mut a_matrix: Array4<f64> = Array4::zeros([m_k.n_orbs, m_k.n_orbs, nvirt, nocc]);
        a_matrix
            .slice_mut(s![..nocc_k, ..nocc_k, .., ..])
            .assign(&a_mat_oo.into_shape([nocc_k, nocc_k, nvirt, nocc]).unwrap());
        a_matrix
            .slice_mut(s![..nocc_k, nocc_k.., .., ..])
            .assign(&a_mat_ov.into_shape([nocc_k, nvirt_k, nvirt, nocc]).unwrap());
        a_matrix
            .slice_mut(s![nocc_k.., ..nocc_k, .., ..])
            .assign(&a_mat_vo.into_shape([nvirt_k, nocc_k, nvirt, nocc]).unwrap());
        a_matrix.slice_mut(s![nocc_k.., nocc_k.., .., ..]).assign(
            &a_mat_vv
                .into_shape([nvirt_k, nvirt_k, nvirt, nocc])
                .unwrap(),
        );

        // add the orbital energy contribution
        if m_x.index == m_k.index {
            for (idx, i) in (nocc..nocc + nvirt).into_iter().enumerate() {
                for j in 0..nocc {
                    for k in 0..m_k.n_orbs {
                        if i == k {
                            for l in 0..m_k.n_orbs {
                                if j == l {
                                    a_matrix[[k, l, idx, j]] = orbe[i] - orbe[j];
                                }
                            }
                        }
                    }
                }
            }
        }

        let a_mat: Array2<f64> = a_matrix
            .into_shape([m_k.n_orbs * m_k.n_orbs, nvirt * nocc])
            .unwrap();

        a_mat
    }

    pub fn get_response_a_matrix_monomer(&self, x_idx: usize) -> Array2<f64> {
        // get the monomers x and k
        let m_x: &Monomer = &self.monomers[x_idx];

        // The atoms are in general a non-contiguous range of the atoms
        let x_atoms: &[Atom] = &self.atoms[m_x.slice.atom_as_range()];

        // get nocc and nvirt of monomer X
        let nocc: usize = m_x.properties.occ_indices().unwrap().len();
        let nvirt: usize = m_x.properties.virt_indices().unwrap().len();

        // get the gamma matrices of X
        let g0: ArrayView2<f64> = m_x.properties.gamma().unwrap();
        let g0_lr: ArrayView2<f64> = m_x.properties.gamma_lr().unwrap();

        // get the orbital energies of x
        let orbe = m_x.properties.orbe().unwrap();

        // get the transition charges
        let qov: ArrayView2<f64> = m_x.properties.q_ov().unwrap();
        let qoo: ArrayView2<f64> = m_x.properties.q_oo().unwrap();
        let qvv: ArrayView2<f64> = m_x.properties.q_vv().unwrap();

        // virtual-occupied transition charges
        let qvo: Array2<f64> = qov
            .clone()
            .into_shape([m_x.n_atoms, nocc, nvirt])
            .unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([m_x.n_atoms, nvirt * nocc])
            .unwrap();

        // calculate A_matrix ij,kl i = nvirt, j = nocc, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_vo: Array2<f64> = Array2::zeros([nvirt * nocc, nvirt * nocc]);
        // integral (ij|kl)
        a_mat_vo = 4.0 * qvo.t().dot(&g0.t().dot(&qvo));

        // integral (ik|jl)
        a_mat_vo = a_mat_vo
            - qvv
                .t()
                .dot(&g0_lr.dot(&qoo))
                .into_shape([nvirt, nvirt, nocc, nocc])
                .unwrap()
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned()
                .into_shape([nvirt * nocc, nvirt * nocc])
                .unwrap();
        // integral (il|jk)
        a_mat_vo = a_mat_vo
            - qvo
                .t()
                .dot(&g0_lr.dot(&qov))
                .into_shape([nvirt, nocc, nocc, nvirt])
                .unwrap()
                .permuted_axes([0, 2, 3, 1])
                .as_standard_layout()
                .to_owned()
                .into_shape([nvirt * nocc, nvirt * nocc])
                .unwrap();

        // calculate A_matrix ij,kl i = nocc, j = nocc, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_oo: Array2<f64> = Array2::zeros([nocc * nocc, nvirt * nocc]);
        // integral (ij|kl)
        a_mat_oo = 4.0 * qoo.t().dot(&g0.t().dot(&qvo));

        // integral (ik|jl)
        a_mat_oo = a_mat_oo
            - qov
                .t()
                .dot(&g0_lr.dot(&qoo))
                .into_shape([nocc, nvirt, nocc, nocc])
                .unwrap()
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned()
                .into_shape([nocc * nocc, nvirt * nocc])
                .unwrap();
        // integral (il|jk)
        a_mat_oo = a_mat_oo
            - qoo
                .t()
                .dot(&g0_lr.dot(&qov))
                .into_shape([nocc, nocc, nocc, nvirt])
                .unwrap()
                .permuted_axes([0, 2, 3, 1])
                .as_standard_layout()
                .to_owned()
                .into_shape([nocc * nocc, nvirt * nocc])
                .unwrap();

        // calculate A_matrix ij,kl i = nocc, j = nvirt, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_ov: Array2<f64> = Array2::zeros([nocc * nvirt, nvirt * nocc]);
        // integral (ij|kl)
        a_mat_ov = 4.0 * qov.t().dot(&g0.t().dot(&qvo));

        // integral (ik|jl)
        a_mat_ov = a_mat_ov
            - qov
                .t()
                .dot(&g0_lr.dot(&qvo))
                .into_shape([nocc, nvirt, nvirt, nocc])
                .unwrap()
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned()
                .into_shape([nocc * nvirt, nvirt * nocc])
                .unwrap();
        // integral (il|jk)
        a_mat_ov = a_mat_ov
            - qoo
                .t()
                .dot(&g0_lr.dot(&qvv))
                .into_shape([nocc, nocc, nvirt, nvirt])
                .unwrap()
                .permuted_axes([0, 2, 3, 1])
                .as_standard_layout()
                .to_owned()
                .into_shape([nocc * nvirt, nvirt * nocc])
                .unwrap();

        // calculate A_matrix ij,kl i = nvirt, j = nvirt, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_vv: Array2<f64> = Array2::zeros([nvirt * nvirt, nvirt * nocc]);
        // integral (ij|kl)
        a_mat_vv = 4.0 * qvv.t().dot(&g0.t().dot(&qvo));

        // integral (ik|jl)
        a_mat_vv = a_mat_vv
            - qvv
                .t()
                .dot(&g0_lr.dot(&qvo))
                .into_shape([nvirt, nvirt, nvirt, nocc])
                .unwrap()
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned()
                .into_shape([nvirt * nvirt, nvirt * nocc])
                .unwrap();
        // integral (il|jk)
        a_mat_vv = a_mat_vv
            - qvo
                .t()
                .dot(&g0_lr.dot(&qvv))
                .into_shape([nvirt, nocc, nvirt, nvirt])
                .unwrap()
                .permuted_axes([0, 2, 3, 1])
                .as_standard_layout()
                .to_owned()
                .into_shape([nvirt * nvirt, nvirt * nocc])
                .unwrap();

        let mut a_matrix: Array4<f64> = Array4::zeros([m_x.n_orbs, m_x.n_orbs, nvirt, nocc]);
        a_matrix
            .slice_mut(s![..nocc, ..nocc, .., ..])
            .assign(&a_mat_oo.into_shape([nocc, nocc, nvirt, nocc]).unwrap());
        a_matrix
            .slice_mut(s![..nocc, nocc.., .., ..])
            .assign(&a_mat_ov.into_shape([nocc, nvirt, nvirt, nocc]).unwrap());
        a_matrix
            .slice_mut(s![nocc.., ..nocc, .., ..])
            .assign(&a_mat_vo.into_shape([nvirt, nocc, nvirt, nocc]).unwrap());
        a_matrix
            .slice_mut(s![nocc.., nocc.., .., ..])
            .assign(&a_mat_vv.into_shape([nvirt, nvirt, nvirt, nocc]).unwrap());

        // add the orbital energy contribution
        for (idx, i) in (nocc..nocc + nvirt).into_iter().enumerate() {
            for j in 0..nocc {
                for k in 0..m_x.n_orbs {
                    if i == k {
                        for l in 0..m_x.n_orbs {
                            if j == l {
                                a_matrix[[k, l, idx, j]] = -1.0 * (orbe[j] - orbe[i]);
                            }
                        }
                    }
                }
            }
        }

        let a_mat: Array2<f64> = a_matrix
            .into_shape([m_x.n_orbs * m_x.n_orbs, nvirt * nocc])
            .unwrap();

        a_mat
    }

    pub fn get_response_a_matrix_monomer_vo(&self, x_idx: usize) -> Array2<f64> {
        // get the monomers x and k
        let m_x: &Monomer = &self.monomers[x_idx];

        // The atoms are in general a non-contiguous range of the atoms
        let x_atoms: &[Atom] = &self.atoms[m_x.slice.atom_as_range()];

        // get nocc and nvirt of monomer X
        let nocc: usize = m_x.properties.occ_indices().unwrap().len();
        let nvirt: usize = m_x.properties.virt_indices().unwrap().len();

        // get the gamma matrices of X
        let g0: ArrayView2<f64> = m_x.properties.gamma().unwrap();
        let g0_lr: ArrayView2<f64> = m_x.properties.gamma_lr().unwrap();

        // get the orbital energies of x
        let orbe = m_x.properties.orbe().unwrap();

        // get the transition charges
        let qov: ArrayView2<f64> = m_x.properties.q_ov().unwrap();
        let qoo: ArrayView2<f64> = m_x.properties.q_oo().unwrap();
        let qvv: ArrayView2<f64> = m_x.properties.q_vv().unwrap();

        // virtual-occupied transition charges
        let qvo: Array2<f64> = qov
            .clone()
            .into_shape([m_x.n_atoms, nocc, nvirt])
            .unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([m_x.n_atoms, nvirt * nocc])
            .unwrap();

        // calculate A_matrix ij,kl i = nvirt, j = nocc, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_vo: Array2<f64> = Array2::zeros([nvirt * nocc, nvirt * nocc]);
        // integral (ij|kl)
        a_mat_vo = 4.0 * qvo.t().dot(&g0.t().dot(&qvo));

        // integral (ik|jl)
        a_mat_vo = a_mat_vo
            - qvv
                .t()
                .dot(&g0_lr.dot(&qoo))
                .into_shape([nvirt, nvirt, nocc, nocc])
                .unwrap()
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned()
                .into_shape([nvirt * nocc, nvirt * nocc])
                .unwrap();
        // integral (il|jk)
        a_mat_vo = a_mat_vo
            - qvo
                .t()
                .dot(&g0_lr.dot(&qov))
                .into_shape([nvirt, nocc, nocc, nvirt])
                .unwrap()
                .permuted_axes([0, 2, 3, 1])
                .as_standard_layout()
                .to_owned()
                .into_shape([nvirt * nocc, nvirt * nocc])
                .unwrap();

        // reshape to 4d array
        let mut a_mat_vo: Array4<f64> =
            -1.0 * a_mat_vo.into_shape([nvirt, nocc, nvirt, nocc]).unwrap();

        // add the orbital energy contribution
        for (idx, i) in (nocc..nocc + nvirt).into_iter().enumerate() {
            for j in 0..nocc {
                for (idx_k, k) in (nocc..nocc + nvirt).into_iter().enumerate() {
                    if i == k {
                        for l in 0..nocc {
                            if j == l {
                                a_mat_vo[[idx_k, l, idx, j]] = orbe[j] - orbe[i];
                            }
                        }
                    }
                }
            }
        }

        let a_mat: Array2<f64> = a_mat_vo.into_shape([nvirt * nocc, nvirt * nocc]).unwrap();

        a_mat
    }

    pub fn get_response_a_matrix_xk(&self, x_idx: usize, k_idx: usize) -> Array2<f64> {
        // get the monomers x and k
        let m_x: &Monomer = &self.monomers[x_idx];
        let m_k: &Monomer = &self.monomers[k_idx];

        // The atoms are in general a non-contiguous range of the atoms
        let x_atoms: &[Atom] = &self.atoms[m_x.slice.atom_as_range()];
        let k_atoms: &[Atom] = &self.atoms[m_k.slice.atom_as_range()];

        // get nocc and nvirt of monomer X
        let nocc: usize = m_x.properties.occ_indices().unwrap().len();
        let nvirt: usize = m_x.properties.virt_indices().unwrap().len();
        let nocc_k: usize = m_k.properties.occ_indices().unwrap().len();
        let nvirt_k: usize = m_k.properties.virt_indices().unwrap().len();

        // get the gamma matrices of X
        let g0: ArrayView2<f64> = m_x.properties.gamma().unwrap();
        let g0_lr: ArrayView2<f64> = m_x.properties.gamma_lr().unwrap();

        // get the gamma matrix between X and K
        let g_full: ArrayView2<f64> = self.properties.gamma().unwrap();
        let g_xk: ArrayView2<f64> = g_full.slice(s![m_x.slice.atom, m_k.slice.atom]);

        // get the orbital energies of x
        let orbe = m_x.properties.orbe().unwrap();

        // get the transition charges
        let qov: ArrayView2<f64> = m_x.properties.q_ov().unwrap();
        let qoo: ArrayView2<f64> = m_x.properties.q_oo().unwrap();
        let qvv: ArrayView2<f64> = m_x.properties.q_vv().unwrap();

        // virtual-occupied transition charges
        let qvo: Array2<f64> = qov
            .clone()
            .into_shape([m_x.n_atoms, nocc, nvirt])
            .unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([m_x.n_atoms, nvirt * nocc])
            .unwrap();

        let qov_k: ArrayView2<f64> = m_k.properties.q_ov().unwrap();
        let qoo_k: ArrayView2<f64> = m_k.properties.q_oo().unwrap();
        let qvv_k: ArrayView2<f64> = m_k.properties.q_vv().unwrap();

        // virtual-occupied transition charges
        let qvo_k: Array2<f64> = qov_k
            .clone()
            .into_shape([m_k.n_atoms, nocc_k, nvirt_k])
            .unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([m_k.n_atoms, nvirt_k * nocc_k])
            .unwrap();

        // calculate A_matrix ij,kl i = nvirt, j = nocc, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_vo: Array2<f64> = Array2::zeros([nvirt_k * nocc_k, nvirt * nocc]);
        // integral (ij|kl)
        a_mat_vo = 4.0 * qvo_k.t().dot(&g_xk.t().dot(&qvo));

        // calculate A_matrix ij,kl i = nocc, j = nocc, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_oo: Array2<f64> = Array2::zeros([nocc_k * nocc_k, nvirt * nocc]);
        // integral (ij|kl)
        a_mat_oo = 4.0 * qoo_k.t().dot(&g_xk.t().dot(&qvo));

        // calculate A_matrix ij,kl i = nocc, j = nvirt, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_ov: Array2<f64> = Array2::zeros([nocc_k * nvirt_k, nvirt * nocc]);
        // integral (ij|kl)
        a_mat_ov = 4.0 * qov_k.t().dot(&g_xk.t().dot(&qvo));

        // calculate A_matrix ij,kl i = nvirt, j = nvirt, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_vv: Array2<f64> = Array2::zeros([nvirt_k * nvirt_k, nvirt * nocc]);
        // integral (ij|kl)
        a_mat_vv = 4.0 * qvv_k.t().dot(&g_xk.t().dot(&qvo));

        let mut a_matrix: Array4<f64> = Array4::zeros([m_k.n_orbs, m_k.n_orbs, nvirt, nocc]);
        a_matrix
            .slice_mut(s![..nocc_k, ..nocc_k, .., ..])
            .assign(&a_mat_oo.into_shape([nocc_k, nocc_k, nvirt, nocc]).unwrap());
        a_matrix
            .slice_mut(s![..nocc_k, nocc_k.., .., ..])
            .assign(&a_mat_ov.into_shape([nocc_k, nvirt_k, nvirt, nocc]).unwrap());
        a_matrix
            .slice_mut(s![nocc_k.., ..nocc_k, .., ..])
            .assign(&a_mat_vo.into_shape([nvirt_k, nocc_k, nvirt, nocc]).unwrap());
        a_matrix.slice_mut(s![nocc_k.., nocc_k.., .., ..]).assign(
            &a_mat_vv
                .into_shape([nvirt_k, nvirt_k, nvirt, nocc])
                .unwrap(),
        );

        let a_mat: Array2<f64> = a_matrix
            .into_shape([m_k.n_orbs * m_k.n_orbs, nvirt * nocc])
            .unwrap();

        a_mat
    }

    pub fn get_full_response_a_matrix(&self, x_idx: usize, k_idx: usize) -> Array2<f64> {
        // get the monomers x and k
        let m_x: &Monomer = &self.monomers[x_idx];
        let m_k: &Monomer = &self.monomers[k_idx];

        // The atoms are in general a non-contiguous range of the atoms
        let x_atoms: &[Atom] = &self.atoms[m_x.slice.atom_as_range()];
        let k_atoms: &[Atom] = &self.atoms[m_k.slice.atom_as_range()];

        // get nocc and nvirt of monomer X
        let nocc: usize = m_x.properties.occ_indices().unwrap().len();
        let nvirt: usize = m_x.properties.virt_indices().unwrap().len();
        let nocc_k: usize = m_k.properties.occ_indices().unwrap().len();
        let nvirt_k: usize = m_k.properties.virt_indices().unwrap().len();

        // get the gamma matrices of X
        let g0: ArrayView2<f64> = m_x.properties.gamma().unwrap();
        let g0_lr: ArrayView2<f64> = m_x.properties.gamma_lr().unwrap();

        // get the gamma matrix between X and K
        let g_full: ArrayView2<f64> = self.properties.gamma().unwrap();
        let g_xk: ArrayView2<f64> = g_full.slice(s![m_x.slice.atom, m_k.slice.atom]);

        // get the orbital energies of x
        let orbe = m_x.properties.orbe().unwrap();

        // get the transition charges
        let qov: ArrayView2<f64> = m_x.properties.q_ov().unwrap();
        let qoo: ArrayView2<f64> = m_x.properties.q_oo().unwrap();
        let qvv: ArrayView2<f64> = m_x.properties.q_vv().unwrap();

        // virtual-occupied transition charges
        let qvo: Array2<f64> = qov
            .clone()
            .into_shape([m_x.n_atoms, nocc, nvirt])
            .unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([m_x.n_atoms, nvirt * nocc])
            .unwrap();

        let qov_k: ArrayView2<f64> = m_k.properties.q_ov().unwrap();
        let qoo_k: ArrayView2<f64> = m_k.properties.q_oo().unwrap();
        let qvv_k: ArrayView2<f64> = m_k.properties.q_vv().unwrap();

        // virtual-occupied transition charges
        let qvo_k: Array2<f64> = qov_k
            .clone()
            .into_shape([m_k.n_atoms, nocc_k, nvirt_k])
            .unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([m_k.n_atoms, nvirt_k * nocc_k])
            .unwrap();

        // calculate A_matrix ij,kl i = nvirt, j = nocc, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_vo: Array2<f64> = Array2::zeros([nvirt_k * nocc_k, nvirt * nocc]);
        let mut a_mat_vo_oo: Array2<f64> = Array2::zeros([nvirt_k * nocc_k, nocc * nocc]);
        let mut a_mat_vo_ov: Array2<f64> = Array2::zeros([nvirt_k * nocc_k, nvirt * nocc]);
        let mut a_mat_vo_vv: Array2<f64> = Array2::zeros([nvirt_k * nocc_k, nvirt * nvirt]);

        // integral (ij|kl)
        a_mat_vo = 4.0 * qvo_k.t().dot(&g_xk.t().dot(&qvo));
        a_mat_vo_oo = 4.0 * qvo_k.t().dot(&g_xk.t().dot(&qoo));
        a_mat_vo_ov = 4.0 * qvo_k.t().dot(&g_xk.t().dot(&qov));
        a_mat_vo_vv = 4.0 * qvo_k.t().dot(&g_xk.t().dot(&qvv));

        if m_x.index == m_k.index {
            // integral (ik|jl) | (vv|oo)
            a_mat_vo = a_mat_vo
                - qvv
                    .t()
                    .dot(&g0_lr.dot(&qoo))
                    .into_shape([nvirt, nvirt, nocc, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nocc, nvirt * nocc])
                    .unwrap();
            // integral (il|jk) (vo|ov)
            a_mat_vo = a_mat_vo
                - qvo
                    .t()
                    .dot(&g0_lr.dot(&qov))
                    .into_shape([nvirt, nocc, nocc, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nocc, nvirt * nocc])
                    .unwrap();

            // integral (ik|jl) = (vo|oo), where (ij|kl) = (vo|oo)
            a_mat_vo_oo = a_mat_vo_oo
                - qvo
                    .t()
                    .dot(&g0_lr.dot(&qoo))
                    .into_shape([nvirt, nocc, nocc, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nocc, nocc * nocc])
                    .unwrap();
            // integral (il|jk) = (vo|oo), where (ij|kl) = (vo|oo)
            a_mat_vo_oo = a_mat_vo_oo
                - qvo
                    .t()
                    .dot(&g0_lr.dot(&qoo))
                    .into_shape([nvirt, nocc, nocc, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nocc, nocc * nocc])
                    .unwrap();

            // integral (ik|jl) = (vo|ov), where (ij|kl) = (vo|ov)
            a_mat_vo_ov = a_mat_vo_ov
                - qvo
                    .t()
                    .dot(&g0_lr.dot(&qov))
                    .into_shape([nvirt, nocc, nocc, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nocc, nocc * nvirt])
                    .unwrap();
            // integral (il|jk) = (vv|oo), where (ij|kl) = (vo|ov)
            a_mat_vo_ov = a_mat_vo_ov
                - qvv
                    .t()
                    .dot(&g0_lr.dot(&qoo))
                    .into_shape([nvirt, nvirt, nocc, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nocc, nvirt * nocc])
                    .unwrap();

            // integral (ik|jl) = (vv|ov), where (ij|kl) = (vo|vv)
            a_mat_vo_vv = a_mat_vo_vv
                - qvv
                    .t()
                    .dot(&g0_lr.dot(&qov))
                    .into_shape([nvirt, nvirt, nocc, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nocc, nvirt * nvirt])
                    .unwrap();
            // integral (il|jk) = (vv|ov), where (ij|kl) = (vo|vv)
            a_mat_vo_vv = a_mat_vo_vv
                - qvv
                    .t()
                    .dot(&g0_lr.dot(&qov))
                    .into_shape([nvirt, nvirt, nocc, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nocc, nvirt * nvirt])
                    .unwrap();
        }

        // calculate A_matrix ij,kl i = nocc, j = nocc, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_oo: Array2<f64> = Array2::zeros([nocc_k * nocc_k, nvirt * nocc]);
        let mut a_mat_oo_oo: Array2<f64> = Array2::zeros([nocc_k * nocc_k, nocc * nocc]);
        let mut a_mat_oo_ov: Array2<f64> = Array2::zeros([nocc_k * nocc_k, nvirt * nocc]);
        let mut a_mat_oo_vv: Array2<f64> = Array2::zeros([nocc_k * nocc_k, nvirt * nvirt]);

        // integral (ij|kl)
        a_mat_oo = 4.0 * qoo_k.t().dot(&g_xk.t().dot(&qvo));
        a_mat_oo_oo = 4.0 * qoo_k.t().dot(&g_xk.t().dot(&qoo));
        a_mat_oo_ov = 4.0 * qoo_k.t().dot(&g_xk.t().dot(&qov));
        a_mat_oo_vv = 4.0 * qoo_k.t().dot(&g_xk.t().dot(&qvv));

        if m_x.index == m_k.index {
            // integral (ik|jl)
            a_mat_oo = a_mat_oo
                - qov
                    .t()
                    .dot(&g0_lr.dot(&qoo))
                    .into_shape([nocc, nvirt, nocc, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nocc, nvirt * nocc])
                    .unwrap();
            // integral (il|jk)
            a_mat_oo = a_mat_oo
                - qoo
                    .t()
                    .dot(&g0_lr.dot(&qov))
                    .into_shape([nocc, nocc, nocc, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nocc, nvirt * nocc])
                    .unwrap();

            // integral (ik|jl) = (oo|oo), where (ij|kl) = (oo|oo)
            a_mat_oo_oo = a_mat_oo_oo
                - qoo
                    .t()
                    .dot(&g0_lr.dot(&qoo))
                    .into_shape([nocc, nocc, nocc, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nocc, nocc * nocc])
                    .unwrap();

            // integral (il|jk) = (oo|oo), where (ij|kl) = (oo|oo)
            a_mat_oo_oo = a_mat_oo_oo
                - qoo
                    .t()
                    .dot(&g0_lr.dot(&qoo))
                    .into_shape([nocc, nocc, nocc, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nocc, nocc * nocc])
                    .unwrap();

            // integral (ik|jl) = (oo|ov), where (ij|kl) = (oo|ov)
            a_mat_oo_ov = a_mat_oo_ov
                - qoo
                    .t()
                    .dot(&g0_lr.dot(&qov))
                    .into_shape([nocc, nocc, nocc, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nocc, nocc * nvirt])
                    .unwrap();

            // integral (il|jk) = (ov|oo), where (ij|kl) = (oo|ov)
            a_mat_oo_ov = a_mat_oo_ov
                - qov
                    .t()
                    .dot(&g0_lr.dot(&qoo))
                    .into_shape([nocc, nvirt, nocc, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nocc, nocc * nvirt])
                    .unwrap();

            // integral (ik|jl) = (ov|ov), where (ij|kl) = (oo|vv)
            a_mat_oo_vv = a_mat_oo_vv
                - qov
                    .t()
                    .dot(&g0_lr.dot(&qov))
                    .into_shape([nocc, nvirt, nocc, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nocc, nvirt * nvirt])
                    .unwrap();

            // integral (il|jk) = (ov|ov), where (ij|kl) = (oo|vv)
            a_mat_oo_vv = a_mat_oo_vv
                - qov
                    .t()
                    .dot(&g0_lr.dot(&qov))
                    .into_shape([nocc, nvirt, nocc, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nocc, nvirt * nvirt])
                    .unwrap();
        }

        // calculate A_matrix ij,kl i = nocc, j = nvirt, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_ov: Array2<f64> = Array2::zeros([nocc_k * nvirt_k, nvirt * nocc]);
        let mut a_mat_ov_oo: Array2<f64> = Array2::zeros([nocc_k * nvirt_k, nocc * nocc]);
        let mut a_mat_ov_ov: Array2<f64> = Array2::zeros([nocc_k * nvirt_k, nvirt * nocc]);
        let mut a_mat_ov_vv: Array2<f64> = Array2::zeros([nocc_k * nvirt_k, nvirt * nvirt]);

        // integral (ij|kl)
        a_mat_ov = 4.0 * qov_k.t().dot(&g_xk.t().dot(&qvo));
        a_mat_ov_oo = 4.0 * qov_k.t().dot(&g_xk.t().dot(&qoo));
        a_mat_ov_ov = 4.0 * qov_k.t().dot(&g_xk.t().dot(&qov));
        a_mat_ov_vv = 4.0 * qov_k.t().dot(&g_xk.t().dot(&qvv));

        if m_x.index == m_k.index {
            // integral (ik|jl)
            a_mat_ov = a_mat_ov
                - qov
                    .t()
                    .dot(&g0_lr.dot(&qvo))
                    .into_shape([nocc, nvirt, nvirt, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nvirt, nvirt * nocc])
                    .unwrap();
            // integral (il|jk)
            a_mat_ov = a_mat_ov
                - qoo
                    .t()
                    .dot(&g0_lr.dot(&qvv))
                    .into_shape([nocc, nocc, nvirt, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nvirt, nvirt * nocc])
                    .unwrap();

            // integral (ik|jl) = (oo|vo) , where (ij|kl) = (ov|oo)
            a_mat_ov_oo = a_mat_ov_oo
                - qoo
                    .t()
                    .dot(&g0_lr.dot(&qvo))
                    .into_shape([nocc, nocc, nvirt, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nvirt, nocc * nocc])
                    .unwrap();
            // integral (il|jk) = (oo|vo), where (ij|kl) = (ov|oo)
            a_mat_ov_oo = a_mat_ov_oo
                - qoo
                    .t()
                    .dot(&g0_lr.dot(&qvo))
                    .into_shape([nocc, nocc, nvirt, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nvirt, nocc * nocc])
                    .unwrap();

            // integral (ik|jl) = (oo|vv) , where (ij|kl) = (ov|ov)
            a_mat_ov_ov = a_mat_ov_ov
                - qoo
                    .t()
                    .dot(&g0_lr.dot(&qvv))
                    .into_shape([nocc, nocc, nvirt, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nvirt, nocc * nvirt])
                    .unwrap();
            // integral (il|jk) = (ov|vo), where (ij|kl) = (ov|ov)
            a_mat_ov_ov = a_mat_ov_ov
                - qov
                    .t()
                    .dot(&g0_lr.dot(&qvo))
                    .into_shape([nocc, nvirt, nvirt, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nvirt, nocc * nvirt])
                    .unwrap();

            // integral (ik|jl) = (ov|vv) , where (ij|kl) = (ov|vv)
            a_mat_ov_vv = a_mat_ov_vv
                - qov
                    .t()
                    .dot(&g0_lr.dot(&qvv))
                    .into_shape([nocc, nvirt, nvirt, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nvirt, nvirt * nvirt])
                    .unwrap();
            // integral (il|jk) = (ov|vv), where (ij|kl) = (ov|vv)
            a_mat_ov_vv = a_mat_ov_vv
                - qov
                    .t()
                    .dot(&g0_lr.dot(&qvv))
                    .into_shape([nocc, nvirt, nvirt, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nocc * nvirt, nvirt * nvirt])
                    .unwrap();
        }

        // calculate A_matrix ij,kl i = nvirt, j = nvirt, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_vv: Array2<f64> = Array2::zeros([nvirt_k * nvirt_k, nvirt * nocc]);
        let mut a_mat_vv_oo: Array2<f64> = Array2::zeros([nvirt_k * nvirt_k, nocc * nocc]);
        let mut a_mat_vv_ov: Array2<f64> = Array2::zeros([nvirt_k * nvirt_k, nvirt * nocc]);
        let mut a_mat_vv_vv: Array2<f64> = Array2::zeros([nvirt_k * nvirt_k, nvirt * nvirt]);

        // integral (ij|kl)
        a_mat_vv = 4.0 * qvv_k.t().dot(&g_xk.t().dot(&qvo));
        a_mat_vv_oo = 4.0 * qvv_k.t().dot(&g_xk.t().dot(&qoo));
        a_mat_vv_ov = 4.0 * qvv_k.t().dot(&g_xk.t().dot(&qov));
        a_mat_vv_vv = 4.0 * qvv_k.t().dot(&g_xk.t().dot(&qvv));

        if m_x.index == m_k.index {
            // integral (ik|jl)
            a_mat_vv = a_mat_vv
                - qvv
                    .t()
                    .dot(&g0_lr.dot(&qvo))
                    .into_shape([nvirt, nvirt, nvirt, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nvirt, nvirt * nocc])
                    .unwrap();
            // integral (il|jk)
            a_mat_vv = a_mat_vv
                - qvo
                    .t()
                    .dot(&g0_lr.dot(&qvv))
                    .into_shape([nvirt, nocc, nvirt, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nvirt, nvirt * nocc])
                    .unwrap();

            // integral (ik|jl) = (vo,vo), where (ij|kl) = (vv|oo)
            a_mat_vv_oo = a_mat_vv_oo
                - qvo
                    .t()
                    .dot(&g0_lr.dot(&qvo))
                    .into_shape([nvirt, nocc, nvirt, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nvirt, nocc * nocc])
                    .unwrap();
            // integral (il|jk) = (vo|vo), where (ij|kl) = (vv|oo)
            a_mat_vv_oo = a_mat_vv_oo
                - qvo
                    .t()
                    .dot(&g0_lr.dot(&qvo))
                    .into_shape([nvirt, nocc, nvirt, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nvirt, nocc * nocc])
                    .unwrap();

            // integral (ik|jl) = (vo,vv), where (ij|kl) = (vv|ov)
            a_mat_vv_ov = a_mat_vv_ov
                - qvo
                    .t()
                    .dot(&g0_lr.dot(&qvv))
                    .into_shape([nvirt, nocc, nvirt, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nvirt, nocc * nvirt])
                    .unwrap();
            // integral (il|jk) = (vv|vo), where (ij|kl) = (vv|ov)
            a_mat_vv_ov = a_mat_vv_ov
                - qvv
                    .t()
                    .dot(&g0_lr.dot(&qvo))
                    .into_shape([nvirt, nvirt, nvirt, nocc])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nvirt, nocc * nvirt])
                    .unwrap();

            // integral (ik|jl) = (vv,vv), where (ij|kl) = (vv|vv)
            a_mat_vv_vv = a_mat_vv_vv
                - qvv
                    .t()
                    .dot(&g0_lr.dot(&qvv))
                    .into_shape([nvirt, nvirt, nvirt, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 1, 3])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nvirt, nvirt * nvirt])
                    .unwrap();
            // integral (il|jk) = (vv|vv), where (ij|kl) = (vv|vv)
            a_mat_vv_vv = a_mat_vv_vv
                - qvv
                    .t()
                    .dot(&g0_lr.dot(&qvv))
                    .into_shape([nvirt, nvirt, nvirt, nvirt])
                    .unwrap()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned()
                    .into_shape([nvirt * nvirt, nvirt * nvirt])
                    .unwrap();
        }

        let mut a_matrix: Array4<f64> =
            Array4::zeros([m_k.n_orbs, m_k.n_orbs, m_x.n_orbs, m_x.n_orbs]);
        // vo array
        a_matrix
            .slice_mut(s![..nocc_k, ..nocc_k, nocc.., ..nocc])
            .assign(&a_mat_oo.into_shape([nocc_k, nocc_k, nvirt, nocc]).unwrap());
        a_matrix
            .slice_mut(s![..nocc_k, nocc_k.., nocc.., ..nocc])
            .assign(&a_mat_ov.into_shape([nocc_k, nvirt_k, nvirt, nocc]).unwrap());
        a_matrix
            .slice_mut(s![nocc_k.., ..nocc_k, nocc.., ..nocc])
            .assign(&a_mat_vo.into_shape([nvirt_k, nocc_k, nvirt, nocc]).unwrap());
        a_matrix
            .slice_mut(s![nocc_k.., nocc_k.., nocc.., ..nocc])
            .assign(
                &a_mat_vv
                    .into_shape([nvirt_k, nvirt_k, nvirt, nocc])
                    .unwrap(),
            );

        // oo array
        a_matrix
            .slice_mut(s![..nocc_k, ..nocc_k, ..nocc, ..nocc])
            .assign(
                &a_mat_oo_oo
                    .into_shape([nocc_k, nocc_k, nocc, nocc])
                    .unwrap(),
            );
        a_matrix
            .slice_mut(s![..nocc_k, nocc_k.., ..nocc, ..nocc])
            .assign(
                &a_mat_ov_oo
                    .into_shape([nocc_k, nvirt_k, nocc, nocc])
                    .unwrap(),
            );
        a_matrix
            .slice_mut(s![nocc_k.., ..nocc_k, ..nocc, ..nocc])
            .assign(
                &a_mat_vo_oo
                    .into_shape([nvirt_k, nocc_k, nocc, nocc])
                    .unwrap(),
            );
        a_matrix
            .slice_mut(s![nocc_k.., nocc_k.., ..nocc, ..nocc])
            .assign(
                &a_mat_vv_oo
                    .into_shape([nvirt_k, nvirt_k, nocc, nocc])
                    .unwrap(),
            );

        // ov array
        a_matrix
            .slice_mut(s![..nocc_k, ..nocc_k, ..nocc, nocc..])
            .assign(
                &a_mat_oo_ov
                    .into_shape([nocc_k, nocc_k, nocc, nvirt])
                    .unwrap(),
            );
        a_matrix
            .slice_mut(s![..nocc_k, nocc_k.., ..nocc, nocc..])
            .assign(
                &a_mat_ov_ov
                    .into_shape([nocc_k, nvirt_k, nocc, nvirt])
                    .unwrap(),
            );
        a_matrix
            .slice_mut(s![nocc_k.., ..nocc_k, ..nocc, nocc..])
            .assign(
                &a_mat_vo_ov
                    .into_shape([nvirt_k, nocc_k, nocc, nvirt])
                    .unwrap(),
            );
        a_matrix
            .slice_mut(s![nocc_k.., nocc_k.., ..nocc, nocc..])
            .assign(
                &a_mat_vv_ov
                    .into_shape([nvirt_k, nvirt_k, nocc, nvirt])
                    .unwrap(),
            );

        // vv array
        a_matrix
            .slice_mut(s![..nocc_k, ..nocc_k, nocc.., nocc..])
            .assign(
                &a_mat_oo_vv
                    .into_shape([nocc_k, nocc_k, nvirt, nvirt])
                    .unwrap(),
            );
        a_matrix
            .slice_mut(s![..nocc_k, nocc_k.., nocc.., nocc..])
            .assign(
                &a_mat_ov_vv
                    .into_shape([nocc_k, nvirt_k, nvirt, nvirt])
                    .unwrap(),
            );
        a_matrix
            .slice_mut(s![nocc_k.., ..nocc_k, nocc.., nocc..])
            .assign(
                &a_mat_vo_vv
                    .into_shape([nvirt_k, nocc_k, nvirt, nvirt])
                    .unwrap(),
            );
        a_matrix
            .slice_mut(s![nocc_k.., nocc_k.., nocc.., nocc..])
            .assign(
                &a_mat_vv_vv
                    .into_shape([nvirt_k, nvirt_k, nvirt, nvirt])
                    .unwrap(),
            );

        // add the orbital energy contribution
        if m_x.index == m_k.index {
            for i in 0..m_x.n_orbs {
                for j in 0..m_x.n_orbs {
                    for k in 0..m_k.n_orbs {
                        if i == k {
                            for l in 0..m_k.n_orbs {
                                if j == l {
                                    a_matrix[[k, l, i, j]] = orbe[i] - orbe[j];
                                }
                            }
                        }
                    }
                }
            }
        }

        let a_mat: Array2<f64> = a_matrix
            .into_shape([m_k.n_orbs * m_k.n_orbs, m_x.n_orbs * m_x.n_orbs])
            .unwrap();

        a_mat
    }

    pub fn response_z_vec_a_diag(&self, m_idx: usize) -> Array2<f64> {
        // get the monomers x and k
        let m_x: &Monomer = &self.monomers[m_idx];

        // get nocc and nvirt of monomer X
        let nocc: usize = m_x.properties.occ_indices().unwrap().len();
        let nvirt: usize = m_x.properties.virt_indices().unwrap().len();

        // get the orbital energies of x
        let orbe = m_x.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = orbe.slice(s![..nocc]).to_owned();
        let orbe_virt: Array1<f64> = orbe.slice(s![nocc..]).to_owned();

        let omega_input: Array2<f64> = into_col(Array::ones(orbe_virt.len()))
            .dot(&into_row(orbe_occ.clone()))
            - into_col(orbe_virt.clone()).dot(&into_row(Array::ones(orbe_occ.len())));

        omega_input
    }

    pub fn get_initial_z_vector_2(&self, lagrangian: &Vec<Array1<f64>>) -> Vec<Array1<f64>> {
        // vector for z_vectors
        let mut z_vec: Vec<Array1<f64>> = Vec::with_capacity(lagrangian.len());

        // loop over the monomers
        for (idx_i, m_i) in self.monomers.iter().enumerate() {
            // get nocc and nvirt of monomer
            let nocc: usize = m_i.properties.occ_indices().unwrap().len();
            let nvirt: usize = m_i.properties.virt_indices().unwrap().len();
            let norbs: usize = m_i.n_orbs;

            // get the lagrangian
            let l_k: ArrayView1<f64> = lagrangian[idx_i].view();
            // reshape
            let l_k_2d: ArrayView2<f64> = l_k.into_shape([norbs, norbs]).unwrap();
            // slice
            let l_k_vo: Array2<f64> = l_k_2d.slice(s![nocc.., ..nocc]).to_owned();

            // get the A matrix for the single fragment
            let a_diag: Array2<f64> = self.response_z_vec_a_diag(idx_i);

            // solve the Z vector eq
            let vec: Array1<f64> =
                self.solve_response_z_vec_cg(idx_i, a_diag.view(), l_k_vo.view());

            z_vec.push(vec);
        }

        z_vec
    }

    pub fn get_initial_z_vector_3(&self, lagrangian: &Vec<Array1<f64>>) -> Vec<Array1<f64>> {
        // vector for z_vectors
        let mut z_vec: Vec<Array1<f64>> = Vec::with_capacity(lagrangian.len());

        // loop over the monomers
        for (idx_i, m_i) in self.monomers.iter().enumerate() {
            // get nocc and nvirt of monomer
            let nocc: usize = m_i.properties.occ_indices().unwrap().len();
            let nvirt: usize = m_i.properties.virt_indices().unwrap().len();
            let norbs: usize = m_i.n_orbs;

            // get the lagrangian
            let l_k: ArrayView1<f64> = lagrangian[idx_i].view();
            // reshape
            let l_k_2d: ArrayView2<f64> = l_k.into_shape([norbs, norbs]).unwrap();
            // slice
            let l_k_vo: Array2<f64> = l_k_2d.slice(s![nocc.., ..nocc]).to_owned();
            let l_k_vo_1d: Array1<f64> = l_k_vo.into_shape([nvirt * nocc]).unwrap();

            // get the A matrix for the single fragment
            let a_matrix: Array2<f64> = self.get_response_a_matrix_monomer_vo(idx_i);

            // solve the system of linear equations
            let vec: Array1<f64> = a_matrix.solve(&l_k_vo_1d).unwrap();

            z_vec.push(vec);
        }

        z_vec
    }

    pub fn get_initial_z_vector_4(&self, lagrangian: &Vec<Array1<f64>>) -> Vec<Array1<f64>> {
        // vector for z_vectors
        let mut z_vec: Vec<Array1<f64>> = Vec::with_capacity(lagrangian.len());

        // loop over the monomers
        for (idx_i, m_i) in self.monomers.iter().enumerate() {
            // get nocc and nvirt of monomer
            let nocc: usize = m_i.properties.occ_indices().unwrap().len();
            let nvirt: usize = m_i.properties.virt_indices().unwrap().len();
            let norbs: usize = m_i.n_orbs;

            // get the lagrangian
            let l_k: ArrayView1<f64> = lagrangian[idx_i].view();
            // reshape
            let l_k_2d: ArrayView2<f64> = l_k.into_shape([norbs, norbs]).unwrap();
            // slice
            let l_k_vo: Array2<f64> = l_k_2d.slice(s![nocc.., ..nocc]).to_owned();
            let l_k_vo_1d: Array1<f64> = l_k_vo.into_shape([nvirt * nocc]).unwrap();

            // get the A matrix for the single fragment
            let a_diag: Array2<f64> = self.response_z_vec_a_diag(idx_i);
            let a_diag: Array1<f64> = a_diag.into_shape(nvirt * nocc).unwrap();

            // get the A matrix for the single fragment
            let a_matrix: Array2<f64> = self.get_response_a_matrix_monomer_vo(idx_i);

            // solve the Z vector eq
            let vec: Array1<f64> = self.solve_response_z_vec_cg_2(
                idx_i,
                a_diag.view(),
                l_k_vo_1d.view(),
                a_matrix.view(),
            );

            z_vec.push(vec);
        }

        z_vec
    }

    pub fn get_next_z_vector_2(
        &self,
        lagrangian: &Vec<Array1<f64>>,
        z_vec: &Vec<Array1<f64>>,
    ) -> Vec<Array1<f64>> {
        // get the interaction lagrangian
        let new_lagrangian: Vec<Array1<f64>> = self.get_interaction_lagrangian(lagrangian, z_vec);

        // storage for new z_vectors
        let mut new_z_vectors: Vec<Array1<f64>> = Vec::with_capacity(z_vec.len());

        for (idx_i, m_i) in self.monomers.iter().enumerate() {
            // get nocc and nvirt of monomer
            let nocc: usize = m_i.properties.occ_indices().unwrap().len();
            let nvirt: usize = m_i.properties.virt_indices().unwrap().len();
            let norbs: usize = m_i.n_orbs;

            // get the lagrangian
            let l_k: ArrayView1<f64> = new_lagrangian[idx_i].view();
            // reshape
            let l_k_2d: ArrayView2<f64> = l_k.into_shape([norbs, norbs]).unwrap();
            // slice
            let l_k_vo: Array2<f64> = l_k_2d.slice(s![nocc.., ..nocc]).to_owned();

            // get the A matrix for the single fragment
            let a_diag: Array2<f64> = self.response_z_vec_a_diag(idx_i);

            // solve the Z vector eq
            let vec: Array1<f64> =
                self.solve_response_z_vec_cg(idx_i, a_diag.view(), l_k_vo.view());

            new_z_vectors.push(vec);
        }

        new_z_vectors
    }

    pub fn get_next_z_vector_3(
        &self,
        lagrangian: &Vec<Array1<f64>>,
        z_vec: &Vec<Array1<f64>>,
    ) -> Vec<Array1<f64>> {
        // get the interaction lagrangian
        let new_lagrangian: Vec<Array1<f64>> = self.get_interaction_lagrangian(lagrangian, z_vec);

        // storage for new z_vectors
        let mut new_z_vectors: Vec<Array1<f64>> = Vec::with_capacity(z_vec.len());

        for (idx_i, m_i) in self.monomers.iter().enumerate() {
            // get nocc and nvirt of monomer
            let nocc: usize = m_i.properties.occ_indices().unwrap().len();
            let nvirt: usize = m_i.properties.virt_indices().unwrap().len();
            let norbs: usize = m_i.n_orbs;

            // get the lagrangian
            let l_k: ArrayView1<f64> = new_lagrangian[idx_i].view();
            // reshape
            let l_k_2d: ArrayView2<f64> = l_k.into_shape([norbs, norbs]).unwrap();
            // slice
            let l_k_vo: Array2<f64> = l_k_2d.slice(s![nocc.., ..nocc]).to_owned();
            let l_k_vo_1d: Array1<f64> = l_k_vo.into_shape([nvirt * nocc]).unwrap();

            // get the A matrix for the single fragment
            let a_diag: Array2<f64> = self.response_z_vec_a_diag(idx_i);
            let a_diag: Array1<f64> = a_diag.into_shape(nvirt * nocc).unwrap();

            // get the A matrix for the single fragment
            let a_matrix: Array2<f64> = self.get_response_a_matrix_monomer_vo(idx_i);

            // solve the Z vector eq
            let vec: Array1<f64> = self.solve_response_z_vec_cg_2(
                idx_i,
                a_diag.view(),
                l_k_vo_1d.view(),
                a_matrix.view(),
            );

            new_z_vectors.push(vec);
        }

        new_z_vectors
    }

    pub fn solve_response_z_vec_cg(
        &self,
        m_idx: usize,
        a_diag: ArrayView2<f64>,
        r_mat: ArrayView2<f64>,
    ) -> Array1<f64> {
        // params
        let maxiter: usize = 100;
        let conv: f64 = 1.0e-8;

        // get the monomers x and k
        let m_x: &Monomer = &self.monomers[m_idx];
        // get nocc and nvirt of monomer X
        let nocc: usize = m_x.properties.occ_indices().unwrap().len();
        let nvirt: usize = m_x.properties.virt_indices().unwrap().len();
        let natoms: usize = m_x.n_atoms;

        // bs are expansion vectors
        let a_inv: Array2<f64> = 1.0 / &a_diag.to_owned();
        let bs: Array2<f64> = &a_inv * &r_mat;

        // nvirt * nocc shape
        let length: usize = nvirt * nocc;

        // arrays for cg procedure
        let mut rhs_2: Array1<f64> = Array::zeros(length);
        let mut rkm1: Array1<f64> = Array::zeros(length);
        let mut pkm1: Array1<f64> = Array::zeros(length);
        let rhs: Array1<f64> = r_mat.into_shape(length).unwrap().to_owned();

        // get the transition charges
        let qov: ArrayView2<f64> = m_x.properties.q_ov().unwrap();
        let qoo: ArrayView2<f64> = m_x.properties.q_oo().unwrap();
        let qvv: ArrayView2<f64> = m_x.properties.q_vv().unwrap();

        // get the gamma matrices of X
        let g0: ArrayView2<f64> = m_x.properties.gamma().unwrap();
        let g0_lr: ArrayView2<f64> = m_x.properties.gamma_lr().unwrap();

        // virtual-occupied transition charges
        let qvo: Array2<f64> = qov
            .clone()
            .into_shape([m_x.n_atoms, nocc, nvirt])
            .unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([m_x.n_atoms, nvirt * nocc])
            .unwrap();

        let apbv: Array2<f64> = mult_a_v_response(
            g0,
            g0_lr,
            a_diag,
            bs.view(),
            qvo.view(),
            qov,
            qvv,
            qoo,
            nocc,
            nvirt,
            natoms,
        );

        // updates arrays
        rkm1 = apbv.into_shape(length).unwrap();
        rhs_2 = bs.into_shape(length).unwrap();
        rkm1 = rhs - rkm1;
        pkm1 = rkm1.clone();

        for _it in 0..maxiter {
            let pkm1_2d: Array2<f64> = pkm1.clone().into_shape([nvirt, nocc]).unwrap();
            let apbv: Array2<f64> = mult_a_v_response(
                g0,
                g0_lr,
                a_diag,
                pkm1_2d.view(),
                qvo.view(),
                qov,
                qvv,
                qoo,
                nocc,
                nvirt,
                natoms,
            );
            let apk: Array1<f64> = apbv.into_shape(length).unwrap();

            let tmp1: f64 = rkm1.dot(&rkm1);
            let tmp2: f64 = pkm1.dot(&apk);

            rhs_2 = rhs_2 + (tmp1 / tmp2) * &pkm1;
            rkm1 = rkm1 - (tmp1 / tmp2) * &apk;

            let tmp2: f64 = rkm1.dot(&rkm1);

            if tmp2 <= conv {
                println!("Stop zvec cg! Iter: {}", _it);
                break;
            }
            pkm1 = (tmp2 / tmp1) * &pkm1 + &rkm1;
        }

        rhs_2
    }

    pub fn solve_response_z_vec_cg_2(
        &self,
        m_idx: usize,
        a_diag: ArrayView1<f64>,
        r_mat: ArrayView1<f64>,
        a_matrix: ArrayView2<f64>,
    ) -> Array1<f64> {
        // params
        let maxiter: usize = 100;
        let conv: f64 = 1.0e-13;

        // get the monomers x and k
        let m_x: &Monomer = &self.monomers[m_idx];
        // get nocc and nvirt of monomer X
        let nocc: usize = m_x.properties.occ_indices().unwrap().len();
        let nvirt: usize = m_x.properties.virt_indices().unwrap().len();
        let natoms: usize = m_x.n_atoms;

        // bs are expansion vectors
        let a_inv: Array1<f64> = 1.0 / &a_diag.to_owned();
        let bs: Array1<f64> = &a_inv * &r_mat;

        // nvirt * nocc shape
        let length: usize = nvirt * nocc;

        // arrays for cg procedure
        let mut rhs_2: Array1<f64> = Array::zeros(length);
        let mut rkm1: Array1<f64> = Array::zeros(length);
        let mut pkm1: Array1<f64> = Array::zeros(length);
        let rhs: Array1<f64> = r_mat.to_owned();

        let apbv: Array1<f64> = a_matrix.dot(&bs);

        // updates arrays
        rkm1 = apbv.into_shape(length).unwrap();
        rhs_2 = bs.into_shape(length).unwrap();
        rkm1 = rhs - rkm1;
        pkm1 = rkm1.clone();

        for it in 0..maxiter {
            let apk: Array1<f64> = a_matrix.dot(&pkm1);

            let tmp1: f64 = rkm1.dot(&rkm1);
            let tmp2: f64 = pkm1.dot(&apk);

            rhs_2 = rhs_2 + (tmp1 / tmp2) * &pkm1;
            rkm1 = rkm1 - (tmp1 / tmp2) * &apk;

            let tmp2: f64 = rkm1.dot(&rkm1);

            if tmp2 <= conv {
                println!("Stop zvec cg! Iter: {}", it);
                break;
            }
            pkm1 = (tmp2 / tmp1) * &pkm1 + &rkm1;
        }

        rhs_2
    }
}

fn mult_a_v_response(
    g0: ArrayView2<f64>,
    g0_lr: ArrayView2<f64>,
    a_diag: ArrayView2<f64>,
    vs: ArrayView2<f64>,
    qvo: ArrayView2<f64>,
    qov: ArrayView2<f64>,
    qvv: ArrayView2<f64>,
    qoo: ArrayView2<f64>,
    nocc: usize,
    nvirt: usize,
    natoms: usize,
) -> Array2<f64> {
    // calculate A_ij,kl dot vec_kl,
    // where  ij and kl are nvirt * nocc
    // A_ij,kl = 4 (ij|kl) - (ik|jl) - (il|jk)
    // or      = 4 (vo|vo) - (vv|oo) - (vo|ov)
    //
    // calculate the first term
    let vs_1d: ArrayView1<f64> = vs.into_shape([nvirt * nocc]).unwrap();
    let g_vo_vs: Array1<f64> = g0.dot(&qvo.dot(&vs_1d));
    let qvo_gvo_vs: Array1<f64> = qvo.t().dot(&g_vo_vs);
    let term_1: Array2<f64> = qvo_gvo_vs.into_shape([nvirt, nocc]).unwrap();

    // calculate the second term (ik|jl) vec_kl
    // contract over k first
    let qvv_3d: ArrayView3<f64> = qvv.into_shape([natoms, nvirt, nvirt]).unwrap();
    let qvv_2d: ArrayView2<f64> = qvv_3d.into_shape([natoms * nvirt, nvirt]).unwrap();
    // contract over k
    let qvv_vs: Array2<f64> = qvv_2d.dot(&vs); // array [natoms*nvirt,nocc] or (A|il)
    let q_vv_vs_3d: Array3<f64> = qvv_vs.into_shape([natoms, nvirt, nocc]).unwrap();
    let q_vv_vs_2d: Array2<f64> = q_vv_vs_3d.into_shape([natoms, nvirt * nocc]).unwrap();
    let g_vv_vs: Array2<f64> = q_vv_vs_2d.t().dot(&g0_lr); // (il|B)
                                                           // contract over B
    let tmp2: Array2<f64> = g_vv_vs.dot(&qoo); // dim [nvirt*nocc, nocc * nocc] or [il,jl]
                                               // reshape to [i,l,j,l]
    let tmp2_4d: Array4<f64> = tmp2.into_shape([nvirt, nocc, nocc, nocc]).unwrap();

    // prepare array
    let mut term2: Array2<f64> = Array::zeros([nvirt, nocc]);
    // sum over index l
    for idx in 0..nocc {
        term2
            .slice_mut(s![.., ..])
            .add_assign(&tmp2_4d.slice(s![.., idx, .., idx]));
    }

    // calculate the third term (il|jk) vec_kl
    // reshape qov and contract over k -> (B|jk) dot (kl) -> (B|jl)
    let qov_3d: ArrayView3<f64> = qvo.into_shape([natoms, nocc, nvirt]).unwrap();
    let qov_2d: ArrayView2<f64> = qov_3d.into_shape([natoms * nocc, nvirt]).unwrap();
    let qov_v: Array2<f64> = qov_2d.dot(&vs).into_shape([natoms, nocc * nocc]).unwrap(); // (B|jl)
                                                                                         // dot with g_lr
    let g_qov_v: Array2<f64> = g0_lr.dot(&qov_v); // (A|jl)
                                                  // dot with qvo
    let tmp3: Array2<f64> = qvo.t().dot(&g_qov_v); // (il|jl)
                                                   // reshape
    let tmp3_4d: Array4<f64> = tmp3.into_shape([nvirt, nocc, nocc, nocc]).unwrap();

    let mut term3: Array2<f64> = Array::zeros([nvirt, nocc]);
    // sum over index l
    for idx in 0..nocc {
        term3
            .slice_mut(s![.., ..])
            .add_assign(&tmp3_4d.slice(s![.., idx, .., idx]));
    }

    -1.0 * (4.0 * term_1 - term2 - term3)
}

fn f_v_xk(
    v: ArrayView2<f64>,
    s_x: ArrayView2<f64>,
    s_k: ArrayView2<f64>,
    grad_s_x: ArrayView3<f64>,
    grad_s_k: ArrayView3<f64>,
    g0_ao_xk: ArrayView2<f64>,
    g1_ao_xk: ArrayView3<f64>,
    n_atoms: usize,
    n_orb: usize,
) -> Array3<f64> {
    let vp: Array2<f64> = &v + &(v.t());
    let sv: Array1<f64> = (&s_k * &vp).sum_axis(Axis(0));
    let gsv: Array1<f64> = g0_ao_xk.dot(&sv);

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb, n_orb));

    for nc in 0..3 * n_atoms {
        let ds_k: ArrayView2<f64> = grad_s_k.slice(s![nc, .., ..]);
        let ds_x: ArrayView2<f64> = grad_s_x.slice(s![nc, .., ..]);
        let dg_xk: ArrayView2<f64> = g1_ao_xk.slice(s![nc, .., ..]);

        let gdsv: Array1<f64> = g0_ao_xk.dot(&(&ds_k * &vp).sum_axis(Axis(0)));
        let dgsv: Array1<f64> = dg_xk.dot(&sv);
        let mut d_f: Array2<f64> = Array2::zeros((n_orb, n_orb));

        for b in 0..n_orb {
            for a in 0..n_orb {
                d_f[[a, b]] = ds_x[[a, b]] * (gsv[a] + gsv[b])
                    + s_x[[a, b]] * (dgsv[a] + gdsv[a] + dgsv[b] + gdsv[b]);
            }
        }
        d_f = d_f * 0.25;
        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }

    return f_return;
}
