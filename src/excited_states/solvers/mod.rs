use crate::excited_states::ProductCache;
use crate::fmo::{ChargeTransferPreparation, Monomer};
use crate::initialization::System;
use ndarray::prelude::*;
use ndarray::Data;
pub use utils::*;

pub mod b_solver;
pub mod casida_davidson;
pub(crate) mod davidson;
pub mod utils;

/// Abstract Trait defining the API required by solver engines.
///
/// Engines implement the correct product functions for iterative solvers that
/// do not require the target matrix be stored directly.
/// Classes intended to be used as an `engine` for `Davidson` or
/// `Hamiltonian` should implement this Trait to ensure
/// that the required methods are defined.
pub trait DavidsonEngine {
    /// Compute a Matrix * trial vector products
    /// Expected output:
    ///  The product`A x X_{i}` for each `X_{i}` in `X`, in that order.
    ///   Where `A` is the hermitian matrix to be diagonalized.
    fn compute_products(&mut self, x: ArrayView2<f64>) -> Array2<f64>;

    /// Apply the preconditioner to a Residual vector.
    /// The preconditioner is usually defined as :math:`(w_k - D_{i})^-1` where
    /// `D` is an approximation of the diagonal of the matrix that is being diagonalized.
    fn precondition(&self, r_k: ArrayView1<f64>, w_k: f64) -> Array1<f64>;

    /// Return the size of the matrix problem.
    fn get_size(&self) -> usize;
}

impl<S> DavidsonEngine for ArrayBase<S, Ix2>
where
    S: Data<Elem = f64>,
{
    fn compute_products(&mut self, x: ArrayView2<f64>) -> Array2<f64> {
        self.dot(&x)
    }

    fn precondition(&self, r_k: ArrayView1<f64>, w_k: f64) -> Array1<f64> {
        &r_k / &(Array1::from_elem(self.nrows(), w_k) - self.diag())
    }

    fn get_size(&self) -> usize {
        self.nrows()
    }
}

/// Implementation of Davidson engine for `System`, `Monomer`, `Pair`.
/// In principle this is also an implementation for `SuperSystem` but that cannot work!
/// This should be properly handled.
/// THERE SHOULD BE CHECKS IF TRANSITION CHARGES; GAMMA; OMEGA are compputed
impl DavidsonEngine for Monomer<'_> {
    /// The products of the TDA/CIS-Hamiltonian with the subspace vectors is computed.
    fn compute_products<'a>(&mut self, x: ArrayView2<'a, f64>) -> Array2<f64> {
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
            for (_i, (mut k, xi)) in k_x
                .axis_iter_mut(Axis(1))
                .zip(compute_vectors.axis_iter(Axis(1)))
                .enumerate()
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
        let _natoms_h: usize = self.m_h.n_atoms;
        let natoms_l: usize = self.m_l.n_atoms;
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
        // Initialization of the product of the exchange part with the subspace part.
        let mut k_x: Array2<f64> = Array::zeros(two_el.raw_dim());
        // Iteration over the subspace vectors.
        for (_i, (mut k, xi)) in k_x
            .axis_iter_mut(Axis(1))
            .zip(compute_vectors.axis_iter(Axis(1)))
            .enumerate()
        {
            // The current vector reshaped into the form of n_occ, n_virt
            let xi = xi.as_standard_layout().into_shape((n_occ, n_virt)).unwrap();
            // The v-v transition have to be reshaped as well.
            let q_vv_r = q_vv.into_shape((natoms_l * n_virt, n_virt)).unwrap();
            // Contraction of the v-v transition charges with the subspace vector and the
            // and the product of the Gamma matrix wit the o-o transition charges.
            k.assign(
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
        }
        // The product of the Exchange part with the subspace vector is added to the Coulomb part.
        two_el = &two_el - &k_x;
        // }

        //let new: Array2<f64> = fock + two_el;
        // The new products are saved in the cache.
        let ax: Array2<f64> = cache.add("TDA", fock + two_el).to_owned();
        self.properties.set_cache(cache);
        // // The product of the CIS-Hamiltonian with the subspace vectors is returned.
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
    fn compute_products<'a>(&mut self, x: ArrayView2<'a, f64>) -> Array2<f64> {
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
            for (_i, (mut k, xi)) in k_x
                .axis_iter_mut(Axis(1))
                .zip(compute_vectors.axis_iter(Axis(1)))
                .enumerate()
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
