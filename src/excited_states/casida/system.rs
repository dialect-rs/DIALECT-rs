#![allow(warnings)]
use crate::constants::HARTREE_TO_EV;
use crate::excited_states::casida_davidson::CasidaSolver;
use crate::excited_states::moments::{
    mulliken_dipoles, mulliken_dipoles_from_ao, oscillator_strength,
};
use crate::excited_states::new_mod::ExcitedStates;
use crate::excited_states::{initial_subspace, ProductCache};
use crate::initialization::System;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray_linalg::{Eigh, FactorizeHInto, Norm, SolveH, SymmetricSqrt, UPLO};
use ndarray_npy::write_npy;

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
            self.config.use_shell_resolved_gamma,
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

        let n_occ =
            (self.occ_indices.len() as f64 * self.config.tddftb.active_orbital_threshold) as usize;
        let n_virt =
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

        write_npy(
            "excited_energies.npy",
            &(&result.eigenvalues.view() * HARTREE_TO_EV),
        )
        .unwrap();
        write_npy("oscillator_strengths.npy", &f).unwrap();

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
            self.config.use_shell_resolved_gamma,
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
        let tr_dipoles: Array2<f64> = if !self.config.use_shell_resolved_gamma {
            mulliken_dipoles(q_trans.view(), &self.atoms)
        } else {
            mulliken_dipoles_from_ao(q_trans.view(), &self.atoms)
        };

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

        write_npy(
            "excited_energies.npy",
            &(&result.eigenvalues.view() * HARTREE_TO_EV),
        )
        .unwrap();
        write_npy("oscillator_strengths.npy", &f).unwrap();

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
        // build a and b matrix
        let a_mat: Array2<f64> = if self.config.lc.long_range_correction {
            a_mat_fock_and_coulomb(&self) - a_mat_exchange(&self)
        } else {
            a_mat_fock_and_coulomb(&self)
        };
        let b_mat: Array2<f64> = if self.config.lc.long_range_correction {
            b_mat_coulomb(&self) - b_mat_exchange(&self)
        } else {
            b_mat_coulomb(&self)
        };

        //check whether A - B is diagonal
        let a_m_b: Array2<f64> = &a_mat - &b_mat;
        let a_p_b: Array2<f64> = &a_mat + &b_mat;

        let sq_a_m_b: Array2<f64>;
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
        let tr_dipoles: Array2<f64> = if !self.config.use_shell_resolved_gamma {
            mulliken_dipoles(q_trans.view(), &self.atoms)
        } else {
            mulliken_dipoles_from_ao(q_trans.view(), &self.atoms)
        };

        // The oscillator strengths are computed.
        let f: Array1<f64> = oscillator_strength(eigenvalues.view(), tr_dipoles.view());

        // Number of occupied orbitals.
        let n_occ: usize = if self.config.tddftb.restrict_active_orbitals {
            let dim_o: usize = (self.occ_indices.len() as f64
                * self.config.tddftb.active_orbital_threshold)
                as usize;
            dim_o
        } else {
            self.occ_indices.len()
        };
        // Number of virtual orbitals.
        let n_virt: usize = if self.config.tddftb.restrict_active_orbitals {
            let dim_v: usize = (self.virt_indices.len() as f64
                * self.config.tddftb.active_orbital_threshold)
                as usize;
            dim_v
        } else {
            self.virt_indices.len()
        };
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

        write_npy(
            "excited_energies.npy",
            &(&eigenvalues.view() * HARTREE_TO_EV),
        )
        .unwrap();
        write_npy("oscillator_strengths.npy", &f).unwrap();

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
        let n_occ: usize = if self.config.tddftb.restrict_active_orbitals {
            let dim_o: usize = (self.occ_indices.len() as f64
                * self.config.tddftb.active_orbital_threshold)
                as usize;
            dim_o
        } else {
            self.occ_indices.len()
        };
        // Number of virtual orbitals.
        let n_virt: usize = if self.config.tddftb.restrict_active_orbitals {
            let dim_v: usize = (self.virt_indices.len() as f64
                * self.config.tddftb.active_orbital_threshold)
                as usize;
            dim_v
        } else {
            self.virt_indices.len()
        };

        // build a and b matrix
        let a_mat: Array2<f64> = a_mat_fock_and_coulomb(&self) - a_mat_exchange(&self);
        let b_mat: Array2<f64> = b_mat_coulomb(&self) - b_mat_exchange(&self);

        //check whether A - B is diagonal
        let a_m_b: Array2<f64> = &a_mat - &b_mat;
        let a_p_b: Array2<f64> = &a_mat + &b_mat;

        let sq_a_m_b: Array2<f64>;
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
        // build a and b matrix
        let a_mat: Array2<f64> = a_mat_fock_and_coulomb(&self);
        let b_mat: Array2<f64> = b_mat_coulomb(&self);

        //check whether A - B is diagonal
        let a_m_b: Array2<f64> = &a_mat - &b_mat;
        let a_p_b: Array2<f64> = &a_mat + &b_mat;

        let sq_a_m_b: Array2<f64>;
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
        // let x_m_y: Array2<f64> = &a_p_b.dot(&x_p_y) / &omega;

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

pub fn a_mat_exchange(molecule: &System) -> Array2<f64> {
    // Number of occupied orbitals.
    let n_occ: usize = if molecule.config.tddftb.restrict_active_orbitals {
        let dim_o: usize = (molecule.occ_indices.len() as f64
            * molecule.config.tddftb.active_orbital_threshold) as usize;
        dim_o
    } else {
        molecule.occ_indices.len()
    };
    // Number of virtual orbitals.
    let n_virt: usize = if molecule.config.tddftb.restrict_active_orbitals {
        let dim_v: usize = (molecule.virt_indices.len() as f64
            * molecule.config.tddftb.active_orbital_threshold) as usize;
        dim_v
    } else {
        molecule.virt_indices.len()
    };

    // Reference to the o-o transition charges.
    let qoo: ArrayView2<f64> = molecule.properties.q_oo().unwrap();
    // Reference to the v-v transition charges.
    let qvv: ArrayView2<f64> = molecule.properties.q_vv().unwrap();
    // Reference to the screened Gamma matrix.
    let gamma_lr: ArrayView2<f64> = if molecule.config.use_shell_resolved_gamma {
        molecule.properties.gamma_lr_ao().unwrap()
    } else {
        molecule.properties.gamma_lr().unwrap()
    };
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
    let gamma: ArrayView2<f64> = if molecule.config.use_shell_resolved_gamma {
        molecule.properties.gamma_ao().unwrap()
    } else {
        molecule.properties.gamma().unwrap()
    };
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
    let gamma: ArrayView2<f64> = if molecule.config.use_shell_resolved_gamma {
        molecule.properties.gamma_ao().unwrap()
    } else {
        molecule.properties.gamma().unwrap()
    };
    // The sum of one-electron part and Coulomb part is computed and retzurned.
    2.0 * qov.t().dot(&gamma.dot(&qov))
}

pub fn b_mat_exchange(molecule: &System) -> Array2<f64> {
    // Number of occupied orbitals.
    let n_occ: usize = if molecule.config.tddftb.restrict_active_orbitals {
        let dim_o: usize = (molecule.occ_indices.len() as f64
            * molecule.config.tddftb.active_orbital_threshold) as usize;
        dim_o
    } else {
        molecule.occ_indices.len()
    };
    // Number of virtual orbitals.
    let n_virt: usize = if molecule.config.tddftb.restrict_active_orbitals {
        let dim_v: usize = (molecule.virt_indices.len() as f64
            * molecule.config.tddftb.active_orbital_threshold) as usize;
        dim_v
    } else {
        molecule.virt_indices.len()
    };

    let qov: ArrayView2<f64> = molecule.properties.q_ov().unwrap();
    // Reference to the screened Gamma matrix.
    let gamma: ArrayView2<f64> = if molecule.config.use_shell_resolved_gamma {
        molecule.properties.gamma_lr_ao().unwrap()
    } else {
        molecule.properties.gamma_lr().unwrap()
    };
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

#[cfg(test)]
mod tests {
    use crate::constants::HARTREE_TO_EV;
    use crate::excited_states::casida_davidson::CasidaSolver;
    use crate::excited_states::{initial_subspace, ProductCache};
    use crate::initialization::System;
    use crate::properties::Properties;
    use crate::scc::scc_routine::RestrictedSCC;
    use crate::utils::tests::{get_molecule, get_molecule_no_lc, AVAILAIBLE_MOLECULES};
    use approx::AbsDiffEq;
    use ndarray::{Array1, Array2};

    pub const EPSILON: f64 = 1e-8;

    fn test_tddftb_calculation(molecule_and_properties: (&str, System, Properties), lc: bool) {
        let name = molecule_and_properties.0;
        let mut molecule = molecule_and_properties.1;
        let props = molecule_and_properties.2;

        // perform scc routine
        molecule.prepare_scc();
        molecule.run_scc().unwrap();

        // perform the tda calculation
        molecule.prepare_tda();
        // Set an empty product cache.
        molecule.properties.set_cache(ProductCache::new());

        // Reference to the energy differences between virtuals and occupied orbitals.
        let omega: Array1<f64> = molecule.properties.omega().unwrap().to_owned();

        // The initial guess for the subspace is created.
        let guess: Array2<f64> = initial_subspace(omega.view(), 4);

        // Davidson iteration.
        let davidson: CasidaSolver = CasidaSolver::new(
            &mut molecule,
            guess,
            omega.view(),
            4,
            1.0e-6,
            100,
            10,
            false,
        )
        .unwrap();
        // get the energies
        let energies: Array1<f64> = davidson.eigenvalues * HARTREE_TO_EV;

        let energies_ref: Array1<f64> = if lc {
            props
                .get("tddftb_energies")
                .unwrap()
                .as_array1()
                .unwrap()
                .to_owned()
        } else {
            props
                .get("tddftb_energies_no_lc")
                .unwrap()
                .as_array1()
                .unwrap()
                .to_owned()
        };
        assert!(
            energies.abs_diff_eq(&energies_ref, EPSILON),
            "Molecule: {}, Excited ref {:.15}, Excited calc: {:.15}",
            name,
            energies_ref,
            energies
        );
    }

    #[test]
    fn get_td_dftb_energy() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_tddftb_calculation(get_molecule(molecule), true);
        }
    }

    #[test]
    fn get_td_dftb_energy_no_lc() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_tddftb_calculation(get_molecule_no_lc(molecule), false);
        }
    }
}
