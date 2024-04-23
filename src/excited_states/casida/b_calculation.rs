use crate::excited_states::moments::{mulliken_dipoles, oscillator_strength};
use crate::excited_states::solvers::b_solver::Bsolver;
use crate::excited_states::tda::new_mod::ExcitedStates;
use crate::excited_states::{initial_subspace, ProductCache};
use crate::fmo::{ChargeTransferPreparation, Monomer};
use crate::initialization::{Atom, System};
use ndarray::prelude::*;

impl Monomer<'_> {
    pub fn casida_b_solver(
        &mut self,
        atoms: &[Atom],
        n_roots: usize,
        max_iter: usize,
        tolerance: f64,
        subspace_multiplier: usize,
    ) {
        // Set an empty product cache.
        self.properties.set_cache(ProductCache::new());

        // Reference to the energy differences between virtuals and occupied orbitals.
        let omega: ArrayView1<f64> = self.properties.omega().unwrap();

        // The initial guess for the subspace is created.
        let guess: Array2<f64> = initial_subspace(omega.view(), n_roots);

        // Davidson iteration.
        let davidson: Bsolver = Bsolver::new(
            self,
            guess,
            n_roots,
            tolerance,
            max_iter,
            false,
            subspace_multiplier,
        )
        .unwrap();

        // Reference to the o-v transition charges.
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();

        // The transition charges for all excited states are computed.
        let q_trans: Array2<f64> = q_ov.dot(&davidson.eigenvectors);

        // The Mulliken transition dipole moments are computed.
        let tr_dipoles: Array2<f64> = mulliken_dipoles(q_trans.view(), atoms);

        // The eigenvalues are the excitation energies and the eigenvectors are the CI coefficients.
        self.properties.set_ci_eigenvalues_b(davidson.eigenvalues);
        self.properties.set_ci_coefficients_b(davidson.eigenvectors);
        self.properties.set_q_trans_b(q_trans);
    }
}

impl ChargeTransferPreparation<'_> {
    // Do the TDA-LC-TD-DFTB calculation
    pub fn casida_b_solver(
        &mut self,
        atoms: &[Atom],
        n_roots: usize,
        max_iter: usize,
        tolerance: f64,
        subspace_multiplier: usize,
    ) {
        // Set an empty product cache.
        self.properties.set_cache(ProductCache::new());

        // Reference to the energy differences between virtuals and occupied orbitals.
        let omega: ArrayView1<f64> = self.properties.omega().unwrap();

        // The initial guess for the subspace is created.
        let guess: Array2<f64> = initial_subspace(omega.view(), n_roots);

        // Davidson iteration.
        let davidson: Bsolver = Bsolver::new(
            self,
            guess,
            n_roots,
            tolerance,
            max_iter,
            false,
            subspace_multiplier,
        )
        .unwrap();

        // Reference to the o-v transition charges.
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();

        // The transition charges for all excited states are computed.
        let q_trans: Array2<f64> = q_ov.dot(&davidson.eigenvectors);

        // The eigenvalues are the excitation energies and the eigenvectors are the CI coefficients.
        self.properties.set_ci_eigenvalues_b(davidson.eigenvalues);
        self.properties.set_ci_coefficients_b(davidson.eigenvectors);
        self.properties.set_q_trans_b(q_trans);
    }
}
