use crate::excited_states::ProductCache;
use crate::fmo::{ChargeTransferPreparation, Monomer};
use crate::initialization::System;
use ndarray::prelude::*;

pub trait CasidaEngine {
    /// Compute Matrix * trial vector products
    /// Expected output:
    ///  The product`A x X_{i}` and `B x X_{i}` for each `X_{i}` in `X`, in that order.
    fn compute_products(&mut self, x: ArrayView2<f64>) -> (Array2<f64>, Array2<f64>);
}

impl CasidaEngine for System {
    /// The products of the TDA/CIS-Hamiltonian with the subspace vectors is computed.
    fn compute_products<'a>(&mut self, x: ArrayView2<'a, f64>) -> (Array2<f64>, Array2<f64>) {
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
        let n_old: usize = cache.count("Casida_A");
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
        let coulomb: Array2<f64> = 2.0 * q_ov.t().dot(&gamma.dot(&q_ov.dot(&compute_vectors)));
        // initialize arrays for the A and B matrix
        let mut amat: Array2<f64> = fock + coulomb.clone();
        let mut bmat: Array2<f64> = coulomb;

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
            // Contraction of the Gamma matrix with the o-v transition charges.
            let gamma_ov: Array2<f64> = gamma_lr
                .dot(&q_ov)
                .into_shape([self.n_atoms, n_occ, n_virt])
                .unwrap()
                .permuted_axes([0, 2, 1])
                .as_standard_layout()
                .to_owned()
                .into_shape([self.n_atoms * n_virt, n_occ])
                .unwrap();

            // The v-v transition have to be reshaped as well.
            let q_vv_r: ArrayView2<f64> = q_vv.into_shape((self.n_atoms * n_virt, n_virt)).unwrap();
            // reshaped q_ov
            let q_ov_r: Array2<f64> = q_ov
                .into_shape([self.n_atoms, n_occ, n_virt])
                .unwrap()
                .permuted_axes([0, 2, 1])
                .as_standard_layout()
                .to_owned()
                .into_shape([self.n_atoms * n_virt, n_occ])
                .unwrap();

            // Initialization of the product of the exchange part with the subspace part.
            let mut k_a: Array2<f64> = Array::zeros(amat.raw_dim());
            // Initialization of the product of the exchange part with the subspace part.
            let mut k_b: Array2<f64> = Array::zeros(bmat.raw_dim());

            // Iteration over the subspace vectors.
            for (_i, (mut k_a_i, (mut k_b_i, xi))) in k_a
                .axis_iter_mut(Axis(1))
                .zip(
                    k_b.axis_iter_mut(Axis(1))
                        .zip(compute_vectors.axis_iter(Axis(1))),
                )
                .enumerate()
            {
                // The current vector reshaped into the form of n_occ, n_virt
                let xi = xi.as_standard_layout().into_shape((n_occ, n_virt)).unwrap();

                // Contraction of the v-v transition charges with the subspace vector and the
                // and the product of the Gamma matrix wit the o-o transition charges.
                k_a_i.assign(
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

                k_b_i.assign(
                    &gamma_ov
                        .t()
                        .dot(
                            &q_ov_r
                                .dot(&xi)
                                .into_shape([self.n_atoms, n_virt, n_virt])
                                .unwrap()
                                .permuted_axes([0, 2, 1])
                                .as_standard_layout()
                                .to_owned()
                                .into_shape([self.n_atoms * n_virt, n_virt])
                                .unwrap(),
                        )
                        .into_shape(n_occ * n_virt)
                        .unwrap(),
                );
            }
            // The product of the Exchange part with the subspace vector is added to the Coulomb part.
            amat = &amat - &k_a;
            bmat = &bmat - &k_b;
        }
        let amat: Array2<f64> = cache.add("Casida_A", amat).to_owned();
        let bmat: Array2<f64> = cache.add("Casida_B", bmat).to_owned();
        self.properties.set_cache(cache);

        (amat, bmat)
    }
}

impl CasidaEngine for Monomer<'_> {
    /// The products of the TDA/CIS-Hamiltonian with the subspace vectors is computed.
    fn compute_products<'a>(&mut self, x: ArrayView2<'a, f64>) -> (Array2<f64>, Array2<f64>) {
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
        let n_old: usize = cache.count("Casida_A");
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
        let coulomb: Array2<f64> = 2.0 * q_ov.t().dot(&gamma.dot(&q_ov.dot(&compute_vectors)));
        // initialize arrays for the A and B matrix
        let mut amat: Array2<f64> = fock + coulomb.clone();
        let mut bmat: Array2<f64> = coulomb;

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
            // Contraction of the Gamma matrix with the o-v transition charges.
            let gamma_ov: Array2<f64> = gamma_lr
                .dot(&q_ov)
                .into_shape([self.n_atoms, n_occ, n_virt])
                .unwrap()
                .permuted_axes([0, 2, 1])
                .as_standard_layout()
                .to_owned()
                .into_shape([self.n_atoms * n_virt, n_occ])
                .unwrap();

            // The v-v transition have to be reshaped as well.
            let q_vv_r: ArrayView2<f64> = q_vv.into_shape((self.n_atoms * n_virt, n_virt)).unwrap();
            // reshaped q_ov
            let q_ov_r: Array2<f64> = q_ov
                .into_shape([self.n_atoms, n_occ, n_virt])
                .unwrap()
                .permuted_axes([0, 2, 1])
                .as_standard_layout()
                .to_owned()
                .into_shape([self.n_atoms * n_virt, n_occ])
                .unwrap();

            // Initialization of the product of the exchange part with the subspace part.
            let mut k_a: Array2<f64> = Array::zeros(amat.raw_dim());
            // Initialization of the product of the exchange part with the subspace part.
            let mut k_b: Array2<f64> = Array::zeros(bmat.raw_dim());

            // Iteration over the subspace vectors.
            for (_i, (mut k_a_i, (mut k_b_i, xi))) in k_a
                .axis_iter_mut(Axis(1))
                .zip(
                    k_b.axis_iter_mut(Axis(1))
                        .zip(compute_vectors.axis_iter(Axis(1))),
                )
                .enumerate()
            {
                // The current vector reshaped into the form of n_occ, n_virt
                let xi = xi.as_standard_layout().into_shape((n_occ, n_virt)).unwrap();

                // Contraction of the v-v transition charges with the subspace vector and the
                // and the product of the Gamma matrix wit the o-o transition charges.
                k_a_i.assign(
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

                k_b_i.assign(
                    &gamma_ov
                        .t()
                        .dot(
                            &q_ov_r
                                .dot(&xi)
                                .into_shape([self.n_atoms, n_virt, n_virt])
                                .unwrap()
                                .permuted_axes([0, 2, 1])
                                .as_standard_layout()
                                .to_owned()
                                .into_shape([self.n_atoms * n_virt, n_virt])
                                .unwrap(),
                        )
                        .into_shape(n_occ * n_virt)
                        .unwrap(),
                );
            }
            // The product of the Exchange part with the subspace vector is added to the Coulomb part.
            amat = &amat - &k_a;
            bmat = &bmat - &k_b;
        }
        let amat: Array2<f64> = cache.add("Casida_A", amat).to_owned();
        let bmat: Array2<f64> = cache.add("Casida_B", bmat).to_owned();
        self.properties.set_cache(cache);

        (amat, bmat)
    }
}

impl CasidaEngine for ChargeTransferPreparation<'_> {
    fn compute_products(&mut self, x: ArrayView2<f64>) -> (Array2<f64>, Array2<f64>) {
        // Mutable reference to the product cache.
        let mut cache: ProductCache = self.properties.take_cache().unwrap();
        // Transition charges between occupied-virtual orbitals, of shape: [n_atoms, n_occ * n_virt]
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();
        // The gamma matrix of the shape: [n_atoms, n_atoms]
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        // get the complete gamma lr matrix
        let gamma_lr_full: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
        // set the number of atoms
        let natoms_h: usize = self.m_h.n_atoms;
        let natoms_l: usize = self.m_l.n_atoms;
        let natoms_pair: usize = natoms_h + natoms_l;
        // The energy differences between virtual and occupied orbitals, shape: [n_occ * n_virt]
        let omega: ArrayView1<f64> = self.properties.omega().unwrap();
        // The number of products that need to be computed in the current iteration.
        let n_prod: usize = x.ncols();
        // The number of products that are already computed.
        let n_old: usize = cache.count("Casida_A");
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
        let coulomb: Array2<f64> = 2.0 * q_ov.t().dot(&gamma.dot(&q_ov.dot(&compute_vectors)));
        // initialize arrays for the A and B matrix
        let mut amat: Array2<f64> = fock + coulomb.clone();
        let mut bmat: Array2<f64> = coulomb;

        // If long-range correction is requested the exchange part needs to be computed.
        // if self.pair_type == PairType::Pair {
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
            .t()
            .dot(&q_oo)
            .into_shape([natoms_l * n_occ, n_occ])
            .unwrap();
        // Contraction of the Gamma matrix with the o-v transition charges.
        let gamma_ov: Array2<f64> = gamma_lr_full
            .dot(&q_ov)
            .into_shape([natoms_pair, n_occ, n_virt])
            .unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([natoms_pair * n_virt, n_occ])
            .unwrap();

        // The v-v transition have to be reshaped as well.
        let q_vv_r: ArrayView2<f64> = q_vv.into_shape((natoms_l * n_virt, n_virt)).unwrap();
        // reshaped q_ov
        let q_ov_r: Array2<f64> = q_ov
            .into_shape([natoms_pair, n_occ, n_virt])
            .unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([natoms_pair * n_virt, n_occ])
            .unwrap();

        // Initialization of the product of the exchange part with the subspace part.
        let mut k_a: Array2<f64> = Array::zeros(amat.raw_dim());
        // Initialization of the product of the exchange part with the subspace part.
        let mut k_b: Array2<f64> = Array::zeros(bmat.raw_dim());

        // Iteration over the subspace vectors.
        for (_i, (mut k_a_i, (mut k_b_i, xi))) in k_a
            .axis_iter_mut(Axis(1))
            .zip(
                k_b.axis_iter_mut(Axis(1))
                    .zip(compute_vectors.axis_iter(Axis(1))),
            )
            .enumerate()
        {
            // The current vector reshaped into the form of n_occ, n_virt
            let xi = xi.as_standard_layout().into_shape((n_occ, n_virt)).unwrap();

            // Contraction of the v-v transition charges with the subspace vector and the
            // and the product of the Gamma matrix wit the o-o transition charges.
            k_a_i.assign(
                // nocc, natoms*nocc
                &gamma_oo
                    .t()
                    .dot(
                        &xi.dot(&q_vv_r.t()) //xi: nocc, nvirt | qvvrT: nvirt, natoms*nvirt
                            .into_shape((n_occ, natoms_l, n_virt))
                            .unwrap()
                            .permuted_axes([1, 0, 2]) // natoms, nocc, nvirt
                            .as_standard_layout()
                            .into_shape((natoms_l * n_occ, n_virt))
                            .unwrap(),
                    )
                    .into_shape(n_occ * n_virt)
                    .unwrap(),
            );

            k_b_i.assign(
                &gamma_ov
                    .t()
                    .dot(
                        &q_ov_r
                            .dot(&xi)
                            .into_shape([natoms_pair, n_virt, n_virt])
                            .unwrap()
                            .permuted_axes([0, 2, 1])
                            .as_standard_layout()
                            .to_owned()
                            .into_shape([natoms_pair * n_virt, n_virt])
                            .unwrap(),
                    )
                    .into_shape(n_occ * n_virt)
                    .unwrap(),
            );
        }
        // The product of the Exchange part with the subspace vector is added to the Coulomb part.
        amat = &amat - &k_a;
        bmat = &bmat - &k_b;

        let amat: Array2<f64> = cache.add("Casida_A", amat).to_owned();
        let bmat: Array2<f64> = cache.add("Casida_B", bmat).to_owned();
        self.properties.set_cache(cache);

        (amat, bmat)
    }
}
