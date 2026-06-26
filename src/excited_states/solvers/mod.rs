use crate::excited_states::ProductCache;
use crate::fmo::{ChargeTransferPreparation, Monomer};
use crate::initialization::System;
// use crate::utils::array_helper::parallel_matrix_multiply;
use ndarray::prelude::*;
use rayon::prelude::*;
pub use dialect_solvers::traits::DavidsonEngine;
pub use dialect_solvers::utils::*;
pub use dialect_solvers::{casida_davidson, casida_for_tda, ct_workspace, davidson};

pub use ct_workspace::CTDavidsonWorkspace;

impl DavidsonEngine for Monomer<'_> {
    /// The products of the TDA/CIS-Hamiltonian with the subspace vectors is computed.
    fn compute_products(&mut self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        // Mutable reference to the product cache.
        let mut cache: ProductCache = self.properties.take_cache().unwrap();
        // Transition charges between occupied-virtual orbitals, of shape: [n_atoms, n_occ * n_virt]
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();
        // The gamma matrix of the shape: [n_atoms, n_atoms]
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        // The energy differences between virtual and occupied orbitals, shape: [n_occ * n_virt]
        let omega: ArrayView1<f64> = self.properties.omega().unwrap();
        // The number of products that need to be computed in the current iteration.
        let n_prod: usize = x.ncols();
        // The number of products that are already computed.
        let n_old: usize = cache.count("TDA");
        // Only the new vectors are computed.
        let compute_vectors: ArrayView2<f64> = if n_prod <= n_old {
            // If the subspace vectors space was collapsed, the cache needs to be cleared.
            cache.reset();
            // All vectors have to be computed.
            x
        } else {
            // Otherwise only the new products have to be computed.
            x.slice_move(s![.., n_old..])
        };
        // The number of vectors that needs to be computed in this iteration.
        let n_comp: usize = compute_vectors.ncols();

        // The product of the Fock matrix elements with the subspace vectors is computed.
        let fock: Array2<f64> =
            &omega.broadcast((n_comp, omega.len())).unwrap().t() * &compute_vectors;

        // The product of the Coulomb matrix elements with the subspace vectors is computed.
        let mut two_el: Array2<f64> = 2.0 * q_ov.t().dot(&gamma.dot(&q_ov.dot(&compute_vectors)));

        // If long-range correction is requested the exchange part needs to be computed.
        if self.gammafunction_lc.is_some() {
            // Reference to the transition charges between occupied-occupied orbitals.
            let q_oo: ArrayView2<f64> = self.properties.q_oo().unwrap();
            // Number of occupied orbitals.
            let n_occ: usize = (q_oo.dim().1 as f64).sqrt() as usize;
            // Reference to the transition charges between virtual-virtual orbitals.
            let q_vv: ArrayView2<f64> = self.properties.q_vv().unwrap();
            // Number of virtual orbitals.
            let n_virt: usize = (q_vv.dim().1 as f64).sqrt() as usize;
            // Reference to the screened Gamma matrix.
            let gamma_lr: ArrayView2<f64> = self.properties.gamma_lr().unwrap();
            // The contraction with the subpspace vectors is more complex than in the case
            // of the Coulomb part.
            // Contraction of the Gamma matrix with the o-o transition charges.
            let gamma_oo: Array2<f64> = gamma_lr
                .dot(&q_oo)
                .into_shape([self.n_atoms * n_occ, n_occ])
                .unwrap();
            // Initialization of the product of the exchange part with the subspace part.
            let mut k_x: Array2<f64> = Array::zeros(two_el.raw_dim());
            // Iteration over the subspace vectors.
            for (mut k, xi) in k_x
                .axis_iter_mut(Axis(1))
                .zip(compute_vectors.axis_iter(Axis(1)))
            {
                // The current vector reshaped into the form of n_occ, n_virt
                let xi = xi.as_standard_layout().into_shape((n_occ, n_virt)).unwrap();
                // The v-v transition have to be reshaped as well.
                let q_vv_r = q_vv.into_shape((self.n_atoms * n_virt, n_virt)).unwrap();
                // Contraction of the v-v transition charges with the subspace vector and the
                // and the product of the Gamma matrix wit the o-o transition charges.
                k.assign(
                    // nocc, natoms*nocc
                    &gamma_oo.t().dot(
                        &xi.dot(&q_vv_r.t()) //xi: nocc, nvirt | qvvrT: nvirt, natoms*nvirt
                            .into_shape((n_occ, self.n_atoms, n_virt))
                            .unwrap()
                            .permuted_axes([1, 0, 2]) // natoms, nocc, nvirt
                            .as_standard_layout()
                            .into_shape((self.n_atoms * n_occ, n_virt))
                            .unwrap(),
                    ).into_shape(n_occ*n_virt).unwrap(),
                );
            }
            // The product of the Exchange part with the subspace vector is added to the Coulomb part.
            two_el = &two_el - &k_x;
        }

        //let new: Array2<f64> = fock + two_el;
        // The new products are saved in the cache.
        let ax: Array2<f64> = cache.add("TDA", fock + two_el).to_owned();
        self.properties.set_cache(cache);
        // // The product of the CIS-Hamiltonian with the subspace vectors is returned.
        ax
    }

    fn compute_products_ao(&mut self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        // Mutable reference to the product cache.
        let mut cache: ProductCache = self.properties.take_cache().unwrap();
        // Transition charges between occupied-virtual orbitals, of shape: [n_atoms, n_occ * n_virt]
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();
        // The gamma matrix of the shape: [n_atoms, n_atoms]
        let gamma: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        // The energy differences between virtual and occupied orbitals, shape: [n_occ * n_virt]
        let omega: ArrayView1<f64> = self.properties.omega().unwrap();
        // The number of products that need to be computed in the current iteration.
        let n_prod: usize = x.ncols();
        // The number of products that are already computed.
        let n_old: usize = cache.count("TDA");
        // Only the new vectors are computed.
        let compute_vectors: ArrayView2<f64> = if n_prod <= n_old {
            // If the subspace vectors space was collapsed, the cache needs to be cleared.
            cache.reset();
            // All vectors have to be computed.
            x
        } else {
            // Otherwise only the new products have to be computed.
            x.slice_move(s![.., n_old..])
        };
        // The number of vectors that needs to be computed in this iteration.
        let n_comp: usize = compute_vectors.ncols();

        // The product of the Fock matrix elements with the subspace vectors is computed.
        let fock: Array2<f64> =
            &omega.broadcast((n_comp, omega.len())).unwrap().t() * &compute_vectors;

        // The product of the Coulomb matrix elements with the subspace vectors is computed.
        let mut two_el: Array2<f64> = 2.0 * q_ov.t().dot(&gamma.dot(&q_ov.dot(&compute_vectors)));

        // If long-range correction is requested the exchange part needs to be computed.
        if self.gammafunction_lc.is_some() {
            // Reference to the transition charges between occupied-occupied orbitals.
            // Contraction of the Gamma matrix with the o-o transition charges.
            // Constant for the whole Davidson run -> computed once and
            // cached across iterations.
            if self.properties.gamma_lr_qoo_ao().is_none() {
                let gamma_oo: Array2<f64> = {
                    let q_oo: ArrayView2<f64> = self.properties.q_oo().unwrap();
                    let n_occ: usize = (q_oo.dim().1 as f64).sqrt() as usize;
                    let gamma_lr: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
                    gamma_lr
                        .dot(&q_oo)
                        .into_shape([self.n_orbs * n_occ, n_occ])
                        .unwrap()
                };
                self.properties.set_gamma_lr_qoo_ao(gamma_oo);
            }
            let q_oo: ArrayView2<f64> = self.properties.q_oo().unwrap();
            // Number of occupied orbitals.
            let n_occ: usize = (q_oo.dim().1 as f64).sqrt() as usize;
            // Reference to the transition charges between virtual-virtual orbitals.
            let q_vv: ArrayView2<f64> = self.properties.q_vv().unwrap();
            // Number of virtual orbitals.
            let n_virt: usize = (q_vv.dim().1 as f64).sqrt() as usize;
            let gamma_oo: ArrayView2<f64> = self.properties.gamma_lr_qoo_ao().unwrap();
            // Initialization of the product of the exchange part with the subspace part.
            let mut k_x: Array2<f64> = Array::zeros(two_el.raw_dim());
            // Iteration over the subspace vectors (in parallel).
            let q_vv_r = q_vv.into_shape((self.n_orbs * n_virt, n_virt)).unwrap();
            let arr_vec: Vec<Array1<f64>> = compute_vectors
                .axis_iter(Axis(1))
                .into_par_iter()
                .map(|xi| {
                    // The current vector reshaped into the form of n_occ, n_virt
                    let xi = xi.as_standard_layout().into_shape((n_occ, n_virt)).unwrap();
                    // Contraction of the v-v transition charges with the subspace
                    // vector and the product of the Gamma matrix with the o-o
                    // transition charges.
                    gamma_oo
                        .t()
                        .dot(
                            &xi.dot(&q_vv_r.t()) //xi: nocc, nvirt | qvvrT: nvirt, natoms*nvirt
                                .into_shape((n_occ, self.n_orbs, n_virt))
                                .unwrap()
                                .permuted_axes([1, 0, 2]) // natoms, nocc, nvirt
                                .as_standard_layout()
                                .into_shape((self.n_orbs * n_occ, n_virt))
                                .unwrap(),
                        )
                        .into_shape(n_occ * n_virt)
                        .unwrap()
                })
                .collect();
            for (idx, arr) in arr_vec.iter().enumerate() {
                k_x.slice_mut(s![.., idx]).assign(arr);
            }
            // The product of the Exchange part with the subspace vector is added to the Coulomb part.
            two_el = &two_el - &k_x;
        }

        //let new: Array2<f64> = fock + two_el;
        // The new products are saved in the cache.
        let ax: Array2<f64> = cache.add("TDA", fock + two_el).to_owned();
        self.properties.set_cache(cache);
        // // The product of the CIS-Hamiltonian with the subspace vectors is returned.
        ax
    }

    /// The preconditioner and a shift are applied to the residual vectors.
    /// The energy difference of the virtual and occupied orbitals is used as a preconditioner.
    fn precondition(&self, r_k: ArrayView1<f64>, w_k: f64) -> Array1<f64> {
        // The denominator is build from the orbital energy differences and the shift value.
        let mut denom: Array1<f64> =
            &(Array1::from_elem(self.get_size(), w_k)) - &self.properties.omega().unwrap();
        // Values smaller than 0.0001 are replaced by 1.0.
        denom.mapv_inplace(|x| if x.abs() < 0.0001 { 1.0 } else { x });
        &r_k / &denom
    }

    fn get_size(&self) -> usize {
        self.properties.omega().unwrap().len()
    }
}

impl DavidsonEngine for ChargeTransferPreparation<'_> {
    fn compute_products(&mut self, x: ArrayView2<f64>) -> Array2<f64> {
        // Mutable reference to the product cache.
        let mut cache: ProductCache = self.properties.take_cache().unwrap();
        // Transition charges between occupied-virtual orbitals, of shape: [n_atoms, n_occ * n_virt]
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();
        // The gamma matrix of the shape: [n_atoms, n_atoms]
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        // set the number of atoms
        let _n_atoms: usize = gamma.dim().0;
        let natoms_h: usize = self.m_h.n_atoms;
        let _natoms_l: usize = self.m_l.n_atoms;
        // The energy differences between virtual and occupied orbitals, shape: [n_occ * n_virt]
        let omega: ArrayView1<f64> = self.properties.omega().unwrap();
        // The number of products that need to be computed in the current iteration.
        let n_prod: usize = x.ncols();
        // The number of products that are already computed.
        let n_old: usize = cache.count("TDA");
        // Only the new vectors are computed.
        let compute_vectors: ArrayView2<f64> = if n_prod <= n_old {
            // If the subspace vectors space was collapsed, the cache needs to be cleared.
            cache.reset();
            // All vectors have to be computed.
            x
        } else {
            // Otherwise only the new products have to be computed.
            x.slice_move(s![.., n_old..])
        };
        // The number of vectors that needs to be computed in this iteration.
        let n_comp: usize = compute_vectors.ncols();

        // The product of the Fock matrix elements with the subspace vectors is computed.
        let fock: Array2<f64> =
            &omega.broadcast((n_comp, omega.len())).unwrap().t() * &compute_vectors;

        // The product of the Coulomb matrix elements with the subspace vectors is computed.
        let mut two_el: Array2<f64> = 2.0 * q_ov.t().dot(&gamma.dot(&q_ov.dot(&compute_vectors)));

        // If long-range correction is requested the exchange part needs to be computed.
        if self.properties.gamma_lr().is_some() {
            // Reference to the transition charges between occupied-occupied orbitals.
            let q_oo: ArrayView2<f64> = self.properties.q_oo().unwrap();
            // Number of occupied orbitals.
            let n_occ: usize = (q_oo.dim().1 as f64).sqrt() as usize;
            // Reference to the transition charges between virtual-virtual orbitals.
            let q_vv: ArrayView2<f64> = self.properties.q_vv().unwrap();
            // Number of virtual orbitals.
            let n_virt: usize = (q_vv.dim().1 as f64).sqrt() as usize;

            // Initialization of the product of the exchange part with the subspace part.
            let mut k_x: Array2<f64> = Array::zeros(two_el.raw_dim());

            // Use BLAS-optimized path if workspace is available
            if let Some(ref mut workspace) = self.davidson_workspace {
                // BLAS-optimized exchange computation using pre-allocated buffers
                for (idx, xi) in compute_vectors.axis_iter(Axis(1)).enumerate() {
                    // Get flat slice of trial vector
                    let xi_flat: Vec<f64> = if let Some(slice) = xi.as_slice() {
                        slice.to_vec()
                    } else {
                        xi.iter().cloned().collect()
                    };
                    // Compute exchange using workspace
                    let exchange_result = workspace.compute_exchange_blas(&xi_flat);
                    k_x.slice_mut(s![.., idx])
                        .assign(&ArrayView1::from(exchange_result));
                }
            } else {
                // Fallback to ndarray-based computation
                // Use cached gamma_lr^T . q_vv if available, otherwise compute it
                let gamma_a_ab: Array3<f64> = if let Some(cached) = self.properties.gamma_lr_qvv() {
                    cached.to_owned()
                } else {
                    // Reference to the screened Gamma matrix.
                    let gamma_lr: ArrayView2<f64> = self.properties.gamma_lr().unwrap();
                    // Pre-compute gamma_lr.t().dot(q_vv) reshaped for efficient contraction
                    gamma_lr
                        .dot(&q_vv)
                        .into_shape([natoms_h, n_virt, n_virt])
                        .unwrap()
                };
                // Reshape q_oo for efficient contraction
                let q_oo_r: ArrayView2<f64> = q_oo.into_shape([natoms_h * n_occ, n_occ]).unwrap();

                // Use sequential iteration to avoid nested parallelism contention
                // (outer CT pair loop is already parallel)
                for (idx, xi) in compute_vectors.axis_iter(Axis(1)).enumerate() {
                    // The current vector reshaped into the form of n_occ, n_virt
                    let xi = xi.as_standard_layout().into_shape((n_occ, n_virt)).unwrap();
                    // Compute v_a_ib = q_oo_r . xi
                    let v_a_ib: Array2<f64> = q_oo_r.dot(&xi);
                    // Reshape and contract with gamma_a_ab
                    let arr: Array2<f64> = v_a_ib
                        .into_shape([natoms_h, n_occ, n_virt])
                        .unwrap()
                        .permuted_axes([1, 2, 0])
                        .as_standard_layout()
                        .into_shape([n_occ, n_virt * natoms_h])
                        .unwrap()
                        .dot(
                            &gamma_a_ab
                                .view()
                                .permuted_axes([1, 2, 0])
                                .as_standard_layout()
                                .into_shape([n_virt, n_virt * natoms_h])
                                .unwrap()
                                .t(),
                        );
                    k_x.slice_mut(s![.., idx])
                        .assign(&arr.into_shape(n_occ * n_virt).unwrap());
                }
            }
            // The product of the Exchange part with the subspace vector is added to the Coulomb part.
            two_el = &two_el - &k_x;
        }

        //let new: Array2<f64> = fock + two_el;
        // The new products are saved in the cache.
        let ax: Array2<f64> = cache.add("TDA", fock + two_el).to_owned();
        self.properties.set_cache(cache);
        // The product of the CIS-Hamiltonian with the subspace vectors is returned.
        ax
    }

    fn compute_products_ao(&mut self, x: ArrayView2<f64>) -> Array2<f64> {
        // Mutable reference to the product cache.
        let mut cache: ProductCache = self.properties.take_cache().unwrap();
        // Transition charges between occupied-virtual orbitals, of shape: [n_atoms, n_occ * n_virt]
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();
        // The gamma matrix of the shape: [n_atoms, n_atoms]
        let gamma: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        // set the number of atoms
        let norbs_l: usize = self.m_l.n_orbs;
        // The energy differences between virtual and occupied orbitals, shape: [n_occ * n_virt]
        let omega: ArrayView1<f64> = self.properties.omega().unwrap();
        // The number of products that need to be computed in the current iteration.
        let n_prod: usize = x.ncols();
        // The number of products that are already computed.
        let n_old: usize = cache.count("TDA");
        // Only the new vectors are computed.
        let compute_vectors: ArrayView2<f64> = if n_prod <= n_old {
            // If the subspace vectors space was collapsed, the cache needs to be cleared.
            cache.reset();
            // All vectors have to be computed.
            x
        } else {
            // Otherwise only the new products have to be computed.
            x.slice_move(s![.., n_old..])
        };
        // The number of vectors that needs to be computed in this iteration.
        let n_comp: usize = compute_vectors.ncols();

        // The product of the Fock matrix elements with the subspace vectors is computed.
        let fock: Array2<f64> =
            &omega.broadcast((n_comp, omega.len())).unwrap().t() * &compute_vectors;

        // The product of the Coulomb matrix elements with the subspace vectors is computed.
        let mut two_el: Array2<f64> = 2.0 * q_ov.t().dot(&gamma.dot(&q_ov.dot(&compute_vectors)));

        if self.properties.gamma_lr_ao().is_some() {
            // If long-range correction is requested the exchange part needs to be computed.
            // Reference to the transition charges between occupied-occupied orbitals.
            let q_oo: ArrayView2<f64> = self.properties.q_oo().unwrap();
            // Number of occupied orbitals.
            let n_occ: usize = (q_oo.dim().1 as f64).sqrt() as usize;
            // Reference to the transition charges between virtual-virtual orbitals.
            let q_vv: ArrayView2<f64> = self.properties.q_vv().unwrap();
            // Number of virtual orbitals.
            let n_virt: usize = (q_vv.dim().1 as f64).sqrt() as usize;
            // Use cached gamma_lr^T . q_vv if available, otherwise compute it
            let gamma_a_ab: Array3<f64> = if let Some(cached) = self.properties.gamma_lr_qvv() {
                cached.to_owned()
            } else {
                // Reference to the screened Gamma matrix.
                let gamma_lr: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
                // Pre-compute gamma_lr.t().dot(q_vv) reshaped for efficient contraction
                gamma_lr
                    .dot(&q_vv)
                    .into_shape([norbs_l, n_virt, n_virt])
                    .unwrap()
            };
            // Reshape q_oo for efficient contraction
            let q_oo_r: ArrayView2<f64> = q_oo.into_shape([norbs_l * n_occ, n_occ]).unwrap();

            // Initialization of the product of the exchange part with the subspace part.
            let mut k_x: Array2<f64> = Array::zeros(two_el.raw_dim());

            // Use sequential iteration to avoid nested parallelism contention
            for (idx, xi) in compute_vectors.axis_iter(Axis(1)).enumerate() {
                // The current vector reshaped into the form of n_occ, n_virt
                let xi = xi.as_standard_layout().into_shape((n_occ, n_virt)).unwrap();
                // Compute v_a_ib = q_oo_r . xi
                let v_a_ib: Array2<f64> = q_oo_r.dot(&xi);
                // Reshape and contract with gamma_a_ab
                let arr: Array2<f64> = v_a_ib
                    .into_shape([norbs_l, n_occ, n_virt])
                    .unwrap()
                    .permuted_axes([1, 2, 0])
                    .as_standard_layout()
                    .into_shape([n_occ, n_virt * norbs_l])
                    .unwrap()
                    .dot(
                        &gamma_a_ab
                            .view()
                            .permuted_axes([1, 2, 0])
                            .as_standard_layout()
                            .into_shape([n_virt, n_virt * norbs_l])
                            .unwrap()
                            .t(),
                    );
                k_x.slice_mut(s![.., idx])
                    .assign(&arr.into_shape(n_occ * n_virt).unwrap());
            }
            // The product of the Exchange part with the subspace vector is added to the Coulomb part.
            two_el = &two_el - &k_x;
        }

        // The new products are saved in the cache.
        let ax: Array2<f64> = cache.add("TDA", fock + two_el).to_owned();
        self.properties.set_cache(cache);
        // The product of the CIS-Hamiltonian with the subspace vectors is returned.
        ax
    }

    fn precondition(&self, r_k: ArrayView1<f64>, w_k: f64) -> Array1<f64> {
        // The denominator is build from the orbital energy differences and the shift value.
        let mut denom: Array1<f64> =
            &(Array1::from_elem(self.get_size(), w_k)) - &self.properties.omega().unwrap();
        // Values smaller than 0.0001 are replaced by 1.0.
        denom.mapv_inplace(|x| if x.abs() < 0.0001 { 1.0 } else { x });
        &r_k / &denom
    }

    fn get_size(&self) -> usize {
        self.properties.omega().unwrap().len()
    }
}

impl DavidsonEngine for System {
    /// The products of the TDA/CIS-Hamiltonian with the subspace vectors is computed.
    fn compute_products(&mut self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        // Mutable reference to the product cache.
        let mut cache: ProductCache = self.properties.take_cache().unwrap();
        // Transition charges between occupied-virtual orbitals, of shape: [n_atoms, n_occ * n_virt]
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();
        // The gamma matrix of the shape: [n_atoms, n_atoms]
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        // The energy differences between virtual and occupied orbitals, shape: [n_occ * n_virt]
        let omega: ArrayView1<f64> = self.properties.omega().unwrap();
        // The number of products that need to be computed in the current iteration.
        let n_prod: usize = x.ncols();
        // The number of products that are already computed.
        let n_old: usize = cache.count("TDA");
        // Only the new vectors are computed.
        let compute_vectors: ArrayView2<f64> = if n_prod <= n_old {
            // If the subspace vectors space was collapsed, the cache needs to be cleared.
            cache.reset();
            // All vectors have to be computed.
            x
        } else {
            // Otherwise only the new products have to be computed.
            x.slice_move(s![.., n_old..])
        };
        // The number of vectors that needs to be computed in this iteration.
        let n_comp: usize = compute_vectors.ncols();

        // The product of the Fock matrix elements with the subspace vectors is computed.
        let fock: Array2<f64> =
            &omega.broadcast((n_comp, omega.len())).unwrap().t() * &compute_vectors;

        // The product of the Coulomb matrix elements with the subspace vectors is computed.
        let mut two_el: Array2<f64> = 2.0 * q_ov.t().dot(&gamma.dot(&q_ov.dot(&compute_vectors)));
        // let two_el_part_1: Array2<f64> = parallel_matrix_multiply(q_ov, compute_vectors, 6);
        // let mut two_el: Array2<f64> = 2.0 * q_ov.t().dot(&gamma.dot(&two_el_part_1));

        // If long-range correction is requested the exchange part needs to be computed.
        if self.gammafunction_lc.is_some() {
            // (gamma_lr . q_vv), permuted to the (n_virt, n_virt * n_atoms)
            // layout needed below. This is constant for the whole Davidson
            // run, so it is computed once and cached across iterations
            // (the per-step property reset clears it when the geometry
            // changes).
            if self.properties.gamma_lr_qvv_perm().is_none() {
                let gamma_a_ab_perm: Array3<f64> = {
                    let q_vv: ArrayView2<f64> = self.properties.q_vv().unwrap();
                    let n_virt: usize = (q_vv.dim().1 as f64).sqrt() as usize;
                    let gamma_lr: ArrayView2<f64> = self.properties.gamma_lr().unwrap();
                    gamma_lr
                        .dot(&q_vv)
                        .into_shape([self.n_atoms, n_virt, n_virt])
                        .unwrap()
                        .permuted_axes([1, 2, 0])
                        .as_standard_layout()
                        .to_owned()
                };
                self.properties.set_gamma_lr_qvv_perm(gamma_a_ab_perm);
            }

            // Reference to the transition charges between occupied-occupied orbitals.
            let q_oo: ArrayView2<f64> = self.properties.q_oo().unwrap();
            // Number of occupied orbitals.
            let n_occ: usize = (q_oo.dim().1 as f64).sqrt() as usize;
            // Number of virtual orbitals.
            let n_virt: usize = (self.properties.q_vv().unwrap().dim().1 as f64).sqrt() as usize;
            let gamma_a_ab_2d: ArrayView2<f64> = self
                .properties
                .gamma_lr_qvv_perm()
                .unwrap()
                .into_shape([n_virt, n_virt * self.n_atoms])
                .unwrap();
            // Initialization of the product of the exchange part with the subspace part.
            let mut k_x: Array2<f64> = Array::zeros(two_el.raw_dim());
            // Iteration over the subspace vectors.
            // for (mut k, xi) in k_x
            //     .axis_iter_mut(Axis(1))
            //     .zip(compute_vectors.axis_iter(Axis(1)))
            // {
            //     // The current vector reshaped into the form of n_occ, n_virt
            //     let xi = xi.as_standard_layout().into_shape((n_occ, n_virt)).unwrap();
            //     // The v-v transition have to be reshaped as well.
            //     let q_vv_r = q_vv.into_shape((self.n_atoms * n_virt, n_virt)).unwrap();
            //     // Contraction of the v-v transition charges with the subspace vector and the
            //     // and the product of the Gamma matrix wit the o-o transition charges.
            //     k.assign(
            //         // nocc, natoms*nocc
            //         &gamma_oo.t().dot(
            //             &xi.dot(&q_vv_r.t()) //xi: nocc, nvirt | qvvrT: nvirt, natoms*nvirt
            //                 .into_shape((n_occ, self.n_atoms, n_virt))
            //                 .unwrap()
            //                 .permuted_axes([1, 0, 2]) // natoms, nocc, nvirt
            //                 .as_standard_layout()
            //                 .into_shape((self.n_atoms * n_occ, n_virt))
            //                 .unwrap(),
            //         ).into_shape(n_occ*n_virt).unwrap(),
            //     );
            // }
            let arr_vec: Vec<Array1<f64>> = compute_vectors
                .axis_iter(Axis(1))
                .into_par_iter()
                .map(|xi| {
                    // The current vector reshaped into the form of n_occ, n_virt
                    let xi = xi.as_standard_layout().into_shape((n_occ, n_virt)).unwrap();
                    // The v-v transition have to be reshaped as well.
                    let q_oo_r = q_oo.into_shape([self.n_atoms * n_occ, n_occ]).unwrap();
                    let v_a_ib: Array2<f64> = q_oo_r.dot(&xi);
                    let arr: Array2<f64> = v_a_ib
                        .into_shape([self.n_atoms, n_occ, n_virt])
                        .unwrap()
                        .permuted_axes([1, 2, 0])
                        .as_standard_layout()
                        .into_shape([n_occ, n_virt * self.n_atoms])
                        .unwrap()
                        .dot(&gamma_a_ab_2d.t());
                    arr.into_shape(n_occ * n_virt).unwrap()
                })
                .collect();
            for (idx, arr) in arr_vec.iter().enumerate() {
                k_x.slice_mut(s![.., idx]).assign(arr);
            }
            // The product of the Exchange part with the subspace vector is added to the Coulomb part.
            two_el = &two_el - &k_x;
        }

        //let new: Array2<f64> = fock + two_el;
        // The new products are saved in the cache.
        let ax: Array2<f64> = cache.add("TDA", fock + two_el).to_owned();
        self.properties.set_cache(cache);
        // // The product of the CIS-Hamiltonian with the subspace vectors is returned.
        ax
    }

    fn compute_products_ao(&mut self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        // Mutable reference to the product cache.
        let mut cache: ProductCache = self.properties.take_cache().unwrap();
        // Transition charges between occupied-virtual orbitals, of shape: [n_atoms, n_occ * n_virt]
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();
        // The gamma matrix of the shape: [n_atoms, n_atoms]
        let gamma: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        // The energy differences between virtual and occupied orbitals, shape: [n_occ * n_virt]
        let omega: ArrayView1<f64> = self.properties.omega().unwrap();
        // The number of products that need to be computed in the current iteration.
        let n_prod: usize = x.ncols();
        // The number of products that are already computed.
        let n_old: usize = cache.count("TDA");
        // Only the new vectors are computed.
        let compute_vectors: ArrayView2<f64> = if n_prod <= n_old {
            // If the subspace vectors space was collapsed, the cache needs to be cleared.
            cache.reset();
            // All vectors have to be computed.
            x
        } else {
            // Otherwise only the new products have to be computed.
            x.slice_move(s![.., n_old..])
        };
        // The number of vectors that needs to be computed in this iteration.
        let n_comp: usize = compute_vectors.ncols();

        // The product of the Fock matrix elements with the subspace vectors is computed.
        let fock: Array2<f64> =
            &omega.broadcast((n_comp, omega.len())).unwrap().t() * &compute_vectors;

        // The product of the Coulomb matrix elements with the subspace vectors is computed.
        let mut two_el: Array2<f64> = 2.0 * q_ov.t().dot(&gamma.dot(&q_ov.dot(&compute_vectors)));

        // If long-range correction is requested the exchange part needs to be computed.
        if self.gammafunction_lc.is_some() {
            // Reference to the transition charges between occupied-occupied orbitals.
            let q_oo: ArrayView2<f64> = self.properties.q_oo().unwrap();
            // Number of occupied orbitals.
            let n_occ: usize = (q_oo.dim().1 as f64).sqrt() as usize;
            // Reference to the transition charges between virtual-virtual orbitals.
            let q_vv: ArrayView2<f64> = self.properties.q_vv().unwrap();
            // Number of virtual orbitals.
            let n_virt: usize = (q_vv.dim().1 as f64).sqrt() as usize;
            // Reference to the screened Gamma matrix.
            let gamma_lr: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
            // The contraction with the subpspace vectors is more complex than in the case
            // of the Coulomb part.
            // Contraction of the Gamma matrix with the o-o transition charges.
            let gamma_oo: Array2<f64> = gamma_lr
                .dot(&q_oo)
                .into_shape([self.n_orbs * n_occ, n_occ])
                .unwrap();
            // Initialization of the product of the exchange part with the subspace part.
            let mut k_x: Array2<f64> = Array::zeros(two_el.raw_dim());
            // Iteration over the subspace vectors.
            for (mut k, xi) in k_x
                .axis_iter_mut(Axis(1))
                .zip(compute_vectors.axis_iter(Axis(1)))
            {
                // The current vector reshaped into the form of n_occ, n_virt
                let xi = xi.as_standard_layout().into_shape((n_occ, n_virt)).unwrap();
                // The v-v transition have to be reshaped as well.
                let q_vv_r = q_vv.into_shape((self.n_orbs * n_virt, n_virt)).unwrap();
                // Contraction of the v-v transition charges with the subspace vector and the
                // and the product of the Gamma matrix wit the o-o transition charges.
                k.assign(
                    // nocc, natoms*nocc
                    &gamma_oo.t().dot(
                        &xi.dot(&q_vv_r.t()) //xi: nocc, nvirt | qvvrT: nvirt, natoms*nvirt
                            .into_shape((n_occ, self.n_orbs, n_virt))
                            .unwrap()
                            .permuted_axes([1, 0, 2]) // natoms, nocc, nvirt
                            .as_standard_layout()
                            .into_shape((self.n_orbs * n_occ, n_virt))
                            .unwrap(),
                    ).into_shape(n_occ*n_virt).unwrap(),
                );
            }
            // The product of the Exchange part with the subspace vector is added to the Coulomb part.
            two_el = &two_el - &k_x;
        }

        //let new: Array2<f64> = fock + two_el;
        // The new products are saved in the cache.
        let ax: Array2<f64> = cache.add("TDA", fock + two_el).to_owned();
        self.properties.set_cache(cache);
        // // The product of the CIS-Hamiltonian with the subspace vectors is returned.
        ax
    }

    /// The preconditioner and a shift are applied to the residual vectors.
    /// The energy difference of the virtual and occupied orbitals is used as a preconditioner.
    fn precondition(&self, r_k: ArrayView1<f64>, w_k: f64) -> Array1<f64> {
        // The denominator is build from the orbital energy differences and the shift value.
        let mut denom: Array1<f64> =
            &(Array1::from_elem(self.get_size(), w_k)) - &self.properties.omega().unwrap();
        // Values smaller than 0.0001 are replaced by 1.0.
        denom.mapv_inplace(|x| if x.abs() < 0.0001 { 1.0 } else { x });
        &r_k / &denom
    }

    fn get_size(&self) -> usize {
        self.properties.omega().unwrap().len()
    }
}
