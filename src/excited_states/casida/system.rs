use crate::excited_states::casida_davidson::CasidaSolver;
use crate::excited_states::moments::{mulliken_dipoles, oscillator_strength};
use crate::excited_states::new_mod::ExcitedStates;
use crate::excited_states::{initial_subspace, ProductCache};
use crate::fmo::helpers::get_pair_slice;
use crate::fmo::{ChargeTransferPreparation, Monomer};
use crate::initialization::{Atom, System};
use ndarray::prelude::*;
use ndarray::Data;
use ndarray_linalg::{Eigh, FactorizeHInto, Norm, SolveH, SymmetricSqrt, UPLO};
use ndarray_npy::write_npy;

impl Monomer<'_> {
    pub fn run_casida(
        &mut self,
        atoms: &[Atom],
        n_roots: usize,
        max_iter: usize,
        tolerance: f64,
        subspace_multiplier: usize,
        print_states: bool,
    ) {
        // Set an empty product cache.
        self.properties.set_cache(ProductCache::new());

        // Reference to the energy differences between virtuals and occupied orbitals.
        let omega: Array1<f64> = self.properties.omega().unwrap().to_owned();

        // The initial guess for the subspace is created.
        let guess: Array2<f64> = initial_subspace(omega.view(), n_roots);

        // Davidson iteration.
        let result: CasidaSolver = CasidaSolver::new(
            self,
            guess,
            omega.view(),
            n_roots,
            tolerance,
            max_iter,
            subspace_multiplier,
        )
        .unwrap();

        // Reference to the o-v transition charges.
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();

        // The transition charges for all excited states are computed.
        let q_trans: Array2<f64> = q_ov.dot(&result.eigenvectors);

        // The Mulliken transition dipole moments are computed.
        let tr_dipoles: Array2<f64> = mulliken_dipoles(q_trans.view(), atoms);

        // The oscillator strengths are computed.
        let f: Array1<f64> = oscillator_strength(result.eigenvalues.view(), tr_dipoles.view());

        let n_occ: usize = self.properties.occ_indices().unwrap().len();
        let n_virt: usize = self.properties.virt_indices().unwrap().len();
        let tdm: Array3<f64> = result
            .eigenvectors
            .clone()
            .into_shape([n_occ, n_virt, f.len()])
            .unwrap();

        let states: ExcitedStates = ExcitedStates {
            total_energy: self.properties.last_energy().unwrap(),
            energies: result.eigenvalues.clone(),
            tdm: tdm,
            f: f.clone(),
            tr_dip: tr_dipoles.clone(),
            orbs: self.properties.orbs().unwrap().to_owned(),
        };

        // take the X+Y and X-Y vectors from the result of the Casida solver and reshape them
        // to [nroots, nocc, nvirt]
        let xpy: Array3<f64> = result
            .xpy
            .clone()
            .into_shape([n_occ, n_virt, f.len()])
            .unwrap()
            .permuted_axes([2, 0, 1])
            .as_standard_layout()
            .to_owned();
        let xmy: Array3<f64> = result
            .xmy
            .clone()
            .into_shape([n_occ, n_virt, f.len()])
            .unwrap()
            .permuted_axes([2, 0, 1])
            .as_standard_layout()
            .to_owned();

        // The eigenvalues are the excitation energies and the eigenvectors are the CI coefficients.
        self.properties.set_ci_eigenvalues(result.eigenvalues);
        self.properties.set_ci_coefficients(result.eigenvectors);
        self.properties.set_xpy(xpy);
        self.properties.set_xmy(xmy);
        self.properties.set_q_trans(q_trans);
        self.properties.set_tr_dipoles(tr_dipoles);
        self.properties.set_oscillator_strengths(f);

        if print_states {
            println!("{}", states);
        }
    }
}

impl System {
    pub fn run_casida_restricted(
        &mut self,
        n_roots: usize,
        max_iter: usize,
        tolerance: f64,
        subspace_multiplier: usize,
        print_states: bool,
    ) {
        // Set an empty product cache.
        self.properties.set_cache(ProductCache::new());

        // Reference to the energy differences between virtuals and occupied orbitals.
        let omega: Array1<f64> = self.properties.omega().unwrap().to_owned();

        // The initial guess for the subspace is created.
        let guess: Array2<f64> = initial_subspace(omega.view(), n_roots);

        // Davidson iteration.
        let result: CasidaSolver = CasidaSolver::new(
            self,
            guess,
            omega.view(),
            n_roots,
            tolerance,
            max_iter,
            subspace_multiplier,
        )
        .unwrap();

        // check if the tda routine yields realistic energies
        let energy_vector = result.eigenvalues.clone().to_vec();
        for energy in energy_vector.iter() {
            let energy_ev: f64 = energy * 27.2114;

            // check for unrealistic energy values
            if energy_ev < 0.001 {
                panic!("Davidson routine convergence error! An unrealistic energy value of < 0.001 eV was obtained!");
            }
        }

        // Reference to the o-v transition charges.
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();

        // The transition charges for all excited states are computed.
        let q_trans: Array2<f64> = q_ov.dot(&result.eigenvectors);

        // The Mulliken transition dipole moments are computed.
        let tr_dipoles: Array2<f64> = mulliken_dipoles(q_trans.view(), &self.atoms);

        // The oscillator strengths are computed.
        let f: Array1<f64> = oscillator_strength(result.eigenvalues.view(), tr_dipoles.view());

        let mut n_occ: usize = self.properties.occ_indices().unwrap().len();
        let mut n_virt: usize = self.properties.virt_indices().unwrap().len();

        n_occ =
            (self.occ_indices.len() as f64 * self.config.tddftb.active_orbital_threshold) as usize;
        n_virt =
            (self.virt_indices.len() as f64 * self.config.tddftb.active_orbital_threshold) as usize;

        let tdm: Array3<f64> = result
            .eigenvectors
            .clone()
            .as_standard_layout()
            .to_owned()
            .into_shape([n_occ, n_virt, f.len()])
            .unwrap();

        let states: ExcitedStates = ExcitedStates {
            total_energy: self.properties.last_energy().unwrap(),
            energies: result.eigenvalues.clone(),
            tdm: tdm,
            f: f.clone(),
            tr_dip: tr_dipoles.clone(),
            orbs: self.properties.orbs().unwrap().to_owned(),
        };

        write_npy("full_energies.npy", &result.eigenvalues.view());
        write_npy("oscillator_strengths.npy", &f);

        // take the X+Y and X-Y vectors from the result of the Casida solver and reshape them
        // to [nroots, nocc, nvirt]
        let xpy: Array3<f64> = result
            .xpy
            .clone()
            .into_shape([n_occ, n_virt, f.len()])
            .unwrap()
            .permuted_axes([2, 0, 1])
            .as_standard_layout()
            .to_owned();
        let xmy: Array3<f64> = result
            .xmy
            .clone()
            .into_shape([n_occ, n_virt, f.len()])
            .unwrap()
            .permuted_axes([2, 0, 1])
            .as_standard_layout()
            .to_owned();

        // The eigenvalues are the excitation energies and the eigenvectors are the CI coefficients.
        self.properties.set_ci_eigenvalues(result.eigenvalues);
        self.properties.set_ci_coefficients(result.eigenvectors);
        self.properties.set_xpy(xpy);
        self.properties.set_xmy(xmy);
        self.properties.set_q_trans(q_trans);
        self.properties.set_tr_dipoles(tr_dipoles);
        self.properties.set_oscillator_strengths(f);

        if print_states {
            println!("{}", states);
        }
    }

    pub fn run_casida(
        &mut self,
        n_roots: usize,
        max_iter: usize,
        tolerance: f64,
        subspace_multiplier: usize,
        print_states: bool,
    ) {
        // Set an empty product cache.
        self.properties.set_cache(ProductCache::new());

        // Reference to the energy differences between virtuals and occupied orbitals.
        let omega: Array1<f64> = self.properties.omega().unwrap().to_owned();

        // The initial guess for the subspace is created.
        let guess: Array2<f64> = initial_subspace(omega.view(), n_roots);

        // Davidson iteration.
        let result: CasidaSolver = CasidaSolver::new(
            self,
            guess,
            omega.view(),
            n_roots,
            tolerance,
            max_iter,
            subspace_multiplier,
        )
        .unwrap();

        // check if the tda routine yields realistic energies
        let energy_vector = result.eigenvalues.clone().to_vec();
        for energy in energy_vector.iter() {
            let energy_ev: f64 = energy * 27.2114;

            // check for unrealistic energy values
            if energy_ev < 0.001 {
                panic!("Davidson routine convergence error! An unrealistic energy value of < 0.001 eV was obtained!");
            }
        }

        // Reference to the o-v transition charges.
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();

        // The transition charges for all excited states are computed.
        let q_trans: Array2<f64> = q_ov.dot(&result.eigenvectors);

        // The Mulliken transition dipole moments are computed.
        let tr_dipoles: Array2<f64> = mulliken_dipoles(q_trans.view(), &self.atoms);

        // The oscillator strengths are computed.
        let f: Array1<f64> = oscillator_strength(result.eigenvalues.view(), tr_dipoles.view());

        let n_occ: usize = self.properties.occ_indices().unwrap().len();
        let n_virt: usize = self.properties.virt_indices().unwrap().len();
        let tdm: Array3<f64> = result
            .eigenvectors
            .clone()
            .into_shape([n_occ, n_virt, f.len()])
            .unwrap();

        let states: ExcitedStates = ExcitedStates {
            total_energy: self.properties.last_energy().unwrap(),
            energies: result.eigenvalues.clone(),
            tdm: tdm,
            f: f.clone(),
            tr_dip: tr_dipoles.clone(),
            orbs: self.properties.orbs().unwrap().to_owned(),
        };

        write_npy("full_energies.npy", &result.eigenvalues.view());
        write_npy("oscillator_strengths.npy", &f);

        // take the X+Y and X-Y vectors from the result of the Casida solver and reshape them
        // to [nroots, nocc, nvirt]
        let xpy: Array3<f64> = result
            .xpy
            .clone()
            .into_shape([n_occ, n_virt, f.len()])
            .unwrap()
            .permuted_axes([2, 0, 1])
            .as_standard_layout()
            .to_owned();
        let xmy: Array3<f64> = result
            .xmy
            .clone()
            .into_shape([n_occ, n_virt, f.len()])
            .unwrap()
            .permuted_axes([2, 0, 1])
            .as_standard_layout()
            .to_owned();

        // The eigenvalues are the excitation energies and the eigenvectors are the CI coefficients.
        self.properties.set_ci_eigenvalues(result.eigenvalues);
        self.properties.set_ci_coefficients(result.eigenvectors);
        self.properties.set_xpy(xpy);
        self.properties.set_xmy(xmy);
        self.properties.set_q_trans(q_trans);
        self.properties.set_tr_dipoles(tr_dipoles);
        self.properties.set_oscillator_strengths(f);

        if print_states {
            println!("{}", states);
        }
    }

    pub fn casida_full_matrix(&mut self) {
        // get the number of occupied and virtual orbitals
        let occ_indices = self.properties.occ_indices().unwrap();
        let virt_indices = self.properties.virt_indices().unwrap();
        let n_occ: usize = occ_indices.len();
        let n_virt: usize = virt_indices.len();

        // build a and b matrix
        let a_mat: Array2<f64> = a_mat_fock_and_coulomb(&self) - a_mat_exchange(&self);
        let b_mat: Array2<f64> = b_mat_coulomb(&self) - b_mat_exchange(&self);

        //check whether A - B is diagonal
        let a_m_b: Array2<f64> = &a_mat - &b_mat;
        let a_p_b: Array2<f64> = &a_mat + &b_mat;

        let mut sq_a_m_b: Array2<f64> = Array2::zeros((n_occ * n_virt, n_occ * n_virt));
        let offdiag: f64 = (Array2::from_diag(&a_m_b.diag()) - &a_m_b).norm();
        if offdiag < 1.0e-10 {
            // calculate the sqareroot of the diagonal and transform to 2d matrix
            sq_a_m_b = Array2::from_diag(&a_m_b.diag().mapv(f64::sqrt));
        } else {
            // calculate matrix squareroot
            sq_a_m_b = a_m_b.ssqrt(UPLO::Upper).unwrap();
        }

        // construct hermitian eigenvalue problem
        // (A-B)^(1/2) (A+B) (A-B)^(1/2) F = Omega^2 F
        let r_mat: Array2<f64> = sq_a_m_b.dot(&a_p_b.dot(&sq_a_m_b));
        let (omega2, eigenvectors): (Array1<f64>, Array2<f64>) = r_mat.eigh(UPLO::Lower).unwrap();
        let eigenvalues: Array1<f64> = omega2.mapv(f64::sqrt);

        // Reference to the o-v transition charges.
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();

        // The transition charges for all excited states are computed.
        let q_trans: Array2<f64> = q_ov.dot(&eigenvectors);

        // The Mulliken transition dipole moments are computed.
        let tr_dipoles: Array2<f64> = mulliken_dipoles(q_trans.view(), &self.atoms);

        // The oscillator strengths are computed.
        let f: Array1<f64> = oscillator_strength(eigenvalues.view(), tr_dipoles.view());

        let n_occ: usize = self.properties.occ_indices().unwrap().len();
        let n_virt: usize = self.properties.virt_indices().unwrap().len();
        let tdm: Array3<f64> = eigenvectors
            .clone()
            .as_standard_layout()
            .to_owned()
            .into_shape([n_occ, n_virt, f.len()])
            .unwrap();

        let states: ExcitedStates = ExcitedStates {
            total_energy: self.properties.last_energy().unwrap(),
            energies: eigenvalues.clone(),
            tdm: tdm,
            f: f.clone(),
            tr_dip: tr_dipoles.clone(),
            orbs: self.properties.orbs().unwrap().to_owned(),
        };

        write_npy("full_energies.npy", &eigenvalues.view());
        write_npy("oscillator_strengths.npy", &f);

        // The eigenvalues are the excitation energies and the eigenvectors are the CI coefficients.
        self.properties.set_ci_eigenvalues(eigenvalues);
        self.properties.set_ci_coefficients(eigenvectors);
        self.properties.set_q_trans(q_trans);
        self.properties.set_tr_dipoles(tr_dipoles);
        self.properties.set_oscillator_strengths(f);

        println!("{}", states);
    }

    pub fn solve_casida(&self) -> (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) {
        // get the number of occupied and virtual orbitals
        let occ_indices = self.properties.occ_indices().unwrap();
        let virt_indices = self.properties.virt_indices().unwrap();
        let n_occ: usize = occ_indices.len();
        let n_virt: usize = virt_indices.len();

        // build a and b matrix
        let a_mat: Array2<f64> = a_mat_fock_and_coulomb(&self) - a_mat_exchange(&self);
        let b_mat: Array2<f64> = b_mat_coulomb(&self) - b_mat_exchange(&self);

        //check whether A - B is diagonal
        let a_m_b: Array2<f64> = &a_mat - &b_mat;
        let a_p_b: Array2<f64> = &a_mat + &b_mat;

        let mut sq_a_m_b: Array2<f64> = Array2::zeros((n_occ * n_virt, n_occ * n_virt));
        let offdiag: f64 = (Array2::from_diag(&a_m_b.diag()) - &a_m_b).norm();
        if offdiag < 1.0e-10 {
            // calculate the sqareroot of the diagonal and transform to 2d matrix
            sq_a_m_b = Array2::from_diag(&a_m_b.diag().mapv(f64::sqrt));
        } else {
            // calculate matrix squareroot
            sq_a_m_b = a_m_b.ssqrt(UPLO::Upper).unwrap();
        }

        // construct hermitian eigenvalue problem
        // (A-B)^(1/2) (A+B) (A-B)^(1/2) F = Omega^2 F
        let r_mat: Array2<f64> = sq_a_m_b.dot(&a_p_b.dot(&sq_a_m_b));
        let (omega2, f_mat): (Array1<f64>, Array2<f64>) = r_mat.eigh(UPLO::Lower).unwrap();
        let omega: Array1<f64> = omega2.mapv(f64::sqrt);

        // compute X-Y and X+Y
        // X+Y = 1/sqrt(Omega) * (A-B)^(1/2).F
        // X-Y = 1/Omega * (A+B).(X+Y)
        let x_p_y: Array2<f64> = &sq_a_m_b.dot(&f_mat) / &omega.mapv(f64::sqrt);
        let x_m_y: Array2<f64> = &a_p_b.dot(&x_p_y) / &omega;

        //C = (A-B)^(-1/2).((X+Y) * sqrt(Omega))
        // so that C^T.C = (X+Y)^T.(A-B)^(-1).(X+Y) * Omega
        //               = (X+Y)^T.(X-Y)
        // since (A-B).(X-Y) = Omega * (X+Y)
        let temp = &x_p_y * &omega.mapv(f64::sqrt);
        let mut c_matrix: Array2<f64> = Array2::zeros((omega.len(), omega.len()));
        let f = sq_a_m_b.factorizeh_into().unwrap();
        for i in 0..(omega.len()) {
            c_matrix
                .slice_mut(s![i, ..])
                .assign(&f.solveh_into(temp.slice(s![.., i]).to_owned()).unwrap());
        }

        c_matrix = c_matrix.reversed_axes();
        assert!(
            (((&c_matrix.slice(s![.., 0]).to_owned() * &c_matrix.slice(s![.., 0])).to_owned())
                .sum()
                .abs()
                - 1.0)
                < 1.0e-10
        );

        let x_m_y_final: Array3<f64> = x_m_y
            .to_owned_f()
            .t()
            .into_shape((n_occ * n_virt, n_occ, n_virt))
            .unwrap()
            .to_owned();
        let x_p_y_final: Array3<f64> = x_p_y
            .to_owned_f()
            .t()
            .into_shape((n_occ * n_virt, n_occ, n_virt))
            .unwrap()
            .to_owned();

        let c_matrix_transformed: Array3<f64> = c_matrix
            .reversed_axes()
            .into_shape((n_occ * n_virt, n_occ, n_virt))
            .unwrap();

        return (omega, c_matrix_transformed, x_m_y_final, x_p_y_final);
    }

    pub fn solve_casida_no_lc(&self) {
        // get the number of occupied and virtual orbitals
        let occ_indices = self.properties.occ_indices().unwrap();
        let virt_indices = self.properties.virt_indices().unwrap();
        let n_occ: usize = occ_indices.len();
        let n_virt: usize = virt_indices.len();

        // build a and b matrix
        let a_mat: Array2<f64> = a_mat_fock_and_coulomb(&self);
        let b_mat: Array2<f64> = b_mat_coulomb(&self);

        //check whether A - B is diagonal
        let a_m_b: Array2<f64> = &a_mat - &b_mat;
        let a_p_b: Array2<f64> = &a_mat + &b_mat;

        let mut sq_a_m_b: Array2<f64> = Array2::zeros((n_occ * n_virt, n_occ * n_virt));
        let offdiag: f64 = (Array2::from_diag(&a_m_b.diag()) - &a_m_b).norm();
        if offdiag < 1.0e-10 {
            // calculate the sqareroot of the diagonal and transform to 2d matrix
            sq_a_m_b = Array2::from_diag(&a_m_b.diag().mapv(f64::sqrt));
        } else {
            // calculate matrix squareroot
            sq_a_m_b = a_m_b.ssqrt(UPLO::Upper).unwrap();
        }

        // construct hermitian eigenvalue problem
        // (A-B)^(1/2) (A+B) (A-B)^(1/2) F = Omega^2 F
        let r_mat: Array2<f64> = sq_a_m_b.dot(&a_p_b.dot(&sq_a_m_b));
        let (omega2, f_mat): (Array1<f64>, Array2<f64>) = r_mat.eigh(UPLO::Lower).unwrap();
        let omega: Array1<f64> = omega2.mapv(f64::sqrt);

        // compute X-Y and X+Y
        // X+Y = 1/sqrt(Omega) * (A-B)^(1/2).F
        // X-Y = 1/Omega * (A+B).(X+Y)
        let x_p_y: Array2<f64> = &sq_a_m_b.dot(&f_mat) / &omega.mapv(f64::sqrt);
        let x_m_y: Array2<f64> = &a_p_b.dot(&x_p_y) / &omega;

        //C = (A-B)^(-1/2).((X+Y) * sqrt(Omega))
        // so that C^T.C = (X+Y)^T.(A-B)^(-1).(X+Y) * Omega
        //               = (X+Y)^T.(X-Y)
        // since (A-B).(X-Y) = Omega * (X+Y)
        let temp = &x_p_y * &omega.mapv(f64::sqrt);
        let mut c_matrix: Array2<f64> = Array2::zeros((omega.len(), omega.len()));
        let f = sq_a_m_b.factorizeh_into().unwrap();
        for i in 0..(omega.len()) {
            c_matrix
                .slice_mut(s![i, ..])
                .assign(&f.solveh_into(temp.slice(s![.., i]).to_owned()).unwrap());
        }
        println!("Energies: {}", omega * 27.2114);
    }
}

impl ChargeTransferPreparation<'_> {
    pub fn run_casida(
        &mut self,
        atoms: &[Atom],
        n_roots: usize,
        max_iter: usize,
        tolerance: f64,
        subspace_multiplier: usize,
        print_states: bool,
    ) {
        // Set an empty product cache.
        self.properties.set_cache(ProductCache::new());

        // Reference to the energy differences between virtuals and occupied orbitals.
        let omega: Array1<f64> = self.properties.omega().unwrap().to_owned();

        // The initial guess for the subspace is created.
        let guess: Array2<f64> = initial_subspace(omega.view(), n_roots);

        // Davidson iteration.
        let result: CasidaSolver = CasidaSolver::new(
            self,
            guess,
            omega.view(),
            n_roots,
            tolerance,
            max_iter,
            subspace_multiplier,
        )
        .unwrap();

        // Reference to the o-v transition charges.
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();

        // The transition charges for all excited states are computed.
        let q_trans: Array2<f64> = q_ov.dot(&result.eigenvectors);
        // get the atoms
        let pair_atoms: Vec<Atom> = get_pair_slice(
            atoms,
            self.m_h.slice.atom_as_range(),
            self.m_l.slice.atom_as_range(),
        );

        // The Mulliken transition dipole moments are computed.
        let tr_dipoles: Array2<f64> = mulliken_dipoles(q_trans.view(), &pair_atoms);

        // The oscillator strengths are computed.
        let f: Array1<f64> = oscillator_strength(result.eigenvalues.view(), tr_dipoles.view());

        let n_occ: usize = self.properties.occ_indices().unwrap().len();
        let n_virt: usize = self.properties.virt_indices().unwrap().len();

        // take the X+Y and X-Y vectors from the result of the Casida solver and reshape them
        // to [nroots, nocc, nvirt]
        let xpy: Array3<f64> = result
            .xpy
            .clone()
            .into_shape([n_occ, n_virt, f.len()])
            .unwrap()
            .permuted_axes([2, 0, 1])
            .as_standard_layout()
            .to_owned();
        let xmy: Array3<f64> = result
            .xmy
            .clone()
            .into_shape([n_occ, n_virt, f.len()])
            .unwrap()
            .permuted_axes([2, 0, 1])
            .as_standard_layout()
            .to_owned();

        // The eigenvalues are the excitation energies and the eigenvectors are the CI coefficients.
        self.properties.set_ci_eigenvalues(result.eigenvalues);
        self.properties.set_ci_coefficients(result.eigenvectors);
        self.properties.set_xpy(xpy);
        self.properties.set_xmy(xmy);
        self.properties.set_q_trans(q_trans);
        self.properties.set_tr_dipoles(tr_dipoles);
        self.properties.set_oscillator_strengths(f);
    }
}

pub fn a_mat_exchange(molecule: &System) -> Array2<f64> {
    // Number of occupied orbitals.
    let n_occ: usize = molecule.occ_indices.len();
    // Number of virtual orbitals.
    let n_virt: usize = molecule.virt_indices.len();
    // Reference to the o-o transition charges.
    let qoo: ArrayView2<f64> = molecule.properties.q_oo().unwrap();
    // Reference to the v-v transition charges.
    let qvv: ArrayView2<f64> = molecule.properties.q_vv().unwrap();
    // Reference to the screened Gamma matrix.
    let gamma_lr: ArrayView2<f64> = molecule.properties.gamma_lr().unwrap();
    // The exchange part to the CIS Hamiltonian is computed.
    let result = qoo
        .t()
        .dot(&gamma_lr.dot(&qvv))
        .into_shape((n_occ, n_occ, n_virt, n_virt))
        .unwrap()
        .permuted_axes([0, 2, 1, 3])
        .as_standard_layout()
        .into_shape([n_occ * n_virt, n_occ * n_virt])
        .unwrap()
        .to_owned();
    result
}

// The one-electron and Coulomb contribution to the CIS Hamiltonian is computed.
pub fn a_mat_fock_and_coulomb(molecule: &System) -> Array2<f64> {
    // Reference to the o-v transition charges.
    let qov: ArrayView2<f64> = molecule.properties.q_ov().unwrap();
    // Reference to the unscreened Gamma matrix.
    let gamma: ArrayView2<f64> = molecule.properties.gamma().unwrap();
    // Reference to the energy differences of the orbital energies.
    let omega: ArrayView1<f64> = molecule.properties.omega().unwrap();
    // The sum of one-electron part and Coulomb part is computed and retzurned.
    Array2::from_diag(&omega) + 2.0 * qov.t().dot(&gamma.dot(&qov))
}

// The one-electron and Coulomb contribution to the CIS Hamiltonian is computed.
pub fn b_mat_coulomb(molecule: &System) -> Array2<f64> {
    // Reference to the o-v transition charges.
    let qov: ArrayView2<f64> = molecule.properties.q_ov().unwrap();
    // Reference to the unscreened Gamma matrix.
    let gamma: ArrayView2<f64> = molecule.properties.gamma().unwrap();
    // The sum of one-electron part and Coulomb part is computed and retzurned.
    2.0 * qov.t().dot(&gamma.dot(&qov))
}

pub fn b_mat_exchange(molecule: &System) -> Array2<f64> {
    // Number of occupied orbitals.
    let n_occ: usize = molecule.occ_indices.len();
    // Number of virtual orbitals.
    let n_virt: usize = molecule.virt_indices.len();
    // Reference to the o-v transition charges.
    let qov: ArrayView2<f64> = molecule.properties.q_ov().unwrap();
    // Reference to the screened Gamma matrix.
    let gamma: ArrayView2<f64> = molecule.properties.gamma_lr().unwrap();
    // The sum of one-electron part and Coulomb part is computed and retzurned.
    let mat: Array4<f64> = qov
        .t()
        .dot(&gamma.dot(&qov))
        .into_shape([n_occ, n_virt, n_occ, n_virt])
        .unwrap();
    mat.permuted_axes([0, 3, 2, 1])
        .as_standard_layout()
        .to_owned()
        .into_shape([n_occ * n_virt, n_occ * n_virt])
        .unwrap()
}

pub trait ToOwnedF<A, D> {
    fn to_owned_f(&self) -> Array<A, D>;
}
impl<A, S, D> ToOwnedF<A, D> for ArrayBase<S, D>
where
    A: Copy + Clone,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn to_owned_f(&self) -> Array<A, D> {
        let mut tmp = unsafe { Array::uninitialized(self.dim().f()) };
        tmp.assign(self);
        tmp
    }
}
