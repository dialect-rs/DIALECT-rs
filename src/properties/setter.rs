use crate::excited_states::ProductCache;
use crate::fmo::old_supersystem::OldSupersystem;
use crate::fmo::{PairType, ReducedBasisState};
use crate::initialization::old_system::OldSystem;
use crate::initialization::Atom;
use crate::properties::property::Property;
use crate::properties::Properties;
use crate::scc::mixer::{AndersonAccel, BroydenMixer};
use hashbrown::HashMap;
use ndarray::prelude::*;

impl Properties {
    pub fn set_old_atoms(&mut self, atoms: Vec<Atom>) {
        self.set("old_atoms", Property::VecAtom(atoms))
    }

    pub fn set_old_orbs(&mut self, orbs: Array2<f64>) {
        self.set("old_orbs", Property::Array2(orbs))
    }

    pub fn set_old_ci_coeffs(&mut self, ci_coeffs: Array3<f64>) {
        self.set("old_ci_coeffs", Property::Array3(ci_coeffs))
    }

    /// Set the HashMap that maps to monomers to the type of pair they form.
    pub fn set_pair_types(&mut self, map: HashMap<(usize, usize), PairType>) {
        self.set("pair_types", Property::PairMap(map))
    }

    /// Set the HashMap that maps the monomers to the index of the pair they form.
    pub fn set_pair_indices(&mut self, map: HashMap<(usize, usize), usize>) {
        self.set("pair_indices", Property::PairIndexMap(map))
    }

    /// Set the HashMap that maps the monomers to the index of the ESD pair they form.
    pub fn set_esd_pair_indices(&mut self, map: HashMap<(usize, usize), usize>) {
        self.set("esd_pair_indices", Property::PairIndexMap(map))
    }

    /// Set the energy of the last scc iteration
    pub fn set_occupation(&mut self, f: Vec<f64>) {
        self.set("occupation", Property::VecF64(f))
    }

    /// Set the index of the HOMO.
    pub fn set_homo(&mut self, idx: usize) {
        self.set("homo_index", Property::Usize(idx))
    }

    /// Set the index of the LUMO.
    pub fn set_lumo(&mut self, idx: usize) {
        self.set("lumo_index", Property::Usize(idx))
    }

    /// Set the number of occupied orbitals.
    pub fn set_n_occ(&mut self, idx: usize) {
        self.set("n_occ", Property::Usize(idx))
    }

    /// Set the number of unoccupied orbitals.
    pub fn set_n_virt(&mut self, idx: usize) {
        self.set("n_virt", Property::Usize(idx))
    }

    /// Set the energy of the last scc iteration
    pub fn set_last_energy(&mut self, energy: f64) {
        self.set("last_energy", Property::Double(energy))
    }

    /// Set the total ground state energy.
    pub fn set_total_energy(&mut self, energy: f64) {
        self.set("total_energy", Property::Double(energy))
    }

    /// Set the excitation energies.
    pub fn set_ci_eigenvalues(&mut self, eigenvalues: Array1<f64>) {
        self.set("ci_eigenvalues", Property::Array1(eigenvalues))
    }

    /// Set the excitation energies.
    pub fn set_ci_eigenvalues_b(&mut self, eigenvalues: Array1<f64>) {
        self.set("ci_eigenvalues_b", Property::Array1(eigenvalues))
    }

    /// Set the tddft excitation energies.
    pub fn set_tddft_eigenvalues(&mut self, eigenvalues: Array1<f64>) {
        self.set("tddft_eigenvalues", Property::Array1(eigenvalues))
    }

    /// Set the scc mixer
    pub fn set_mixer(&mut self, mixer: BroydenMixer) {
        self.set("mixer", Property::from(mixer))
    }

    /// Set the scc mixer
    pub fn set_accel(&mut self, accel: AndersonAccel) {
        self.set("accel", Property::from(accel))
    }

    /// Set the atomic numbers
    pub fn set_atomic_numbers(&mut self, atomic_numbers: Vec<u8>) {
        self.set("atomic_numbers", Property::from(atomic_numbers))
    }

    /// Set the reference density matrix
    pub fn set_p_ref(&mut self, ref_p: Array2<f64>) {
        self.set("ref_density_matrix", Property::from(ref_p))
    }

    /// Set the CI coefficients of all excited states, that were computed.
    pub fn set_ci_coefficients(&mut self, coeffs: Array2<f64>) {
        self.set("ci_coefficients", Property::from(coeffs))
    }

    /// Set the CI coefficients of all excited states, that were computed.
    pub fn set_ci_coefficients_b(&mut self, coeffs: Array2<f64>) {
        self.set("ci_coefficients_b", Property::from(coeffs))
    }

    /// Set the the difference of the density matrix between the pair and the corresponding monomers
    pub fn set_delta_p(&mut self, delta_p: Array2<f64>) {
        self.set("diff_density_matrix", Property::from(delta_p))
    }

    /// Set the LCMO Fock matrix in MO basis.
    pub fn set_lcmo_fock(&mut self, fock: Array2<f64>) {
        self.set("lcmo_fock", Property::from(fock));
    }

    /// Set the H0 matrix in AO basis.
    pub fn set_h0(&mut self, h0: Array2<f64>) {
        self.set("H0", Property::from(h0));
    }

    /// Set the overlap matrix in AO basis.
    pub fn set_s(&mut self, s: Array2<f64>) {
        self.set("S", Property::from(s));
    }

    /// Set the MO coefficients from the SCC calculation.
    pub fn set_orbs(&mut self, orbs: Array2<f64>) {
        self.set("orbs", Property::from(orbs));
    }

    /// Set the MO energies from the SCC calculation.
    pub fn set_orbe(&mut self, orbe: Array1<f64>) {
        self.set("orbe", Property::from(orbe));
    }

    /// Set the S^-1/2 in AO basis.
    pub fn set_x(&mut self, x: Array2<f64>) {
        self.set("X", Property::from(x));
    }

    /// Set the gradient of the H0 matrix in AO basis.
    pub fn set_grad_h0(&mut self, grad_h0: Array3<f64>) {
        self.set("gradH0", Property::from(grad_h0));
    }

    /// Set the gradient of the overlap matrix in AO basis.
    pub fn set_grad_s(&mut self, grad_s: Array3<f64>) {
        self.set("gradS", Property::from(grad_s));
    }

    /// Set the charge differences per atom.
    pub fn set_dq(&mut self, dq: Array1<f64>) {
        self.set("dq", Property::from(dq));
    }

    /// Set the charge differences per atom.
    pub fn set_dq_alpha(&mut self, dq: Array1<f64>) {
        self.set("dq_alpha", Property::from(dq));
    }

    /// Set the charge differences per atom.
    pub fn set_dq_beta(&mut self, dq: Array1<f64>) {
        self.set("dq_beta", Property::from(dq));
    }

    /// Set the derivative of the charge (differences) w.r.t. to degrees of freedom per atom
    pub fn set_grad_dq(&mut self, grad_dq: Array2<f64>) {
        self.set("grad_dq", Property::from(grad_dq))
    }

    /// Set the diagonal of the derivative of the charge (differences) w.r.t. to degrees of freedom per atom
    pub fn set_grad_dq_diag(&mut self, grad_dq_diag: Array1<f64>) {
        self.set("grad_dq_diag", Property::from(grad_dq_diag))
    }

    /// Set the difference of charge differences between the pair dq and the dq's from the
    /// corresponding monomers per atom.
    pub fn set_delta_dq(&mut self, delta_dq: Array1<f64>) {
        self.set("delta_dq", Property::from(delta_dq));
    }

    /// Set the transition charges between of alle excited states.
    pub fn set_q_trans(&mut self, q_trans: Array2<f64>) {
        self.set("q_trans", Property::from(q_trans));
    }

    /// Set the transition charges between of alle excited states.
    pub fn set_q_trans_b(&mut self, q_trans: Array2<f64>) {
        self.set("q_trans_b", Property::from(q_trans));
    }

    /// Set the transition dipole moments for all excited states.
    pub fn set_tr_dipoles(&mut self, tr_dipoles: Array2<f64>) {
        self.set("tr_dipoles", Property::from(tr_dipoles));
    }

    /// Set the oscillator strengtrhs for all excited states.
    pub fn set_oscillator_strengths(&mut self, f: Array1<f64>) {
        self.set("oscillator_strengths", Property::from(f));
    }

    pub fn set_q_ao(&mut self, q_ao: Array1<f64>) {
        self.set("q_ao", Property::from(q_ao));
    }

    /// Set the transition charges between occupied and virtual orbitals
    pub fn set_q_ov(&mut self, q_ov: Array2<f64>) {
        self.set("q_ov", Property::from(q_ov));
    }

    /// Set the transition charges between virtual and occupied orbitals
    pub fn set_q_vo(&mut self, q_vo: Array2<f64>) {
        self.set("q_vo", Property::from(q_vo));
    }

    /// Set the transition charges between occupied orbitals
    pub fn set_q_oo(&mut self, q_oo: Array2<f64>) {
        self.set("q_oo", Property::from(q_oo));
    }

    /// Set the transition charges between occupied orbitals
    pub fn set_q_oo_restricted(&mut self, q_oo: Array2<f64>) {
        self.set("q_oo_restricted", Property::from(q_oo));
    }

    /// Set the transition charges between virtual orbitals
    pub fn set_q_vv(&mut self, q_vv: Array2<f64>) {
        self.set("q_vv", Property::from(q_vv));
    }

    /// Set the transition charges between virtual orbitals
    pub fn set_q_vv_restricted(&mut self, q_vv: Array2<f64>) {
        self.set("q_vv_restricted", Property::from(q_vv));
    }

    /// Set the orbital energy differences between virtual and occupied orbitals.
    pub fn set_omega(&mut self, omega: Array1<f64>) {
        self.set("omega", Property::from(omega));
    }

    /// Set the `ProductCache`.
    pub fn set_cache(&mut self, cache: ProductCache) {
        self.set("cache", Property::from(cache));
    }

    /// Set the converged Z-vector from the FMO gradient response term.
    pub fn set_z_vector(&mut self, z_vector: Array1<f64>) {
        self.set("z_vector", Property::from(z_vector));
    }

    /// Set the indices of the occupied orbitals, starting at 0.
    pub fn set_occ_indices(&mut self, occ_indices: Vec<usize>) {
        self.set("occ_indices", Property::from(occ_indices));
    }

    /// Set the indices of the virtual orbitals.
    pub fn set_virt_indices(&mut self, virt_indices: Vec<usize>) {
        self.set("virt_indices", Property::from(virt_indices));
    }

    /// Set the esp charges per atom
    pub fn set_esp_q(&mut self, esp_q: Array1<f64>) {
        self.set("esp_charges", Property::from(esp_q));
    }

    /// Set the density matrix in AO basis.
    pub fn set_p(&mut self, p: Array2<f64>) {
        self.set("P", Property::from(p));
    }

    /// Set the density matrix of the alpha electrons in AO basis.
    pub fn set_p_alpha(&mut self, p: Array2<f64>) {
        self.set("P_alpha", Property::from(p));
    }

    /// Set the density matrix of the beta electrons in AO basis.
    pub fn set_p_beta(&mut self, p: Array2<f64>) {
        self.set("P_beta", Property::from(p));
    }

    /// Set the electrostatic potential in AO basis.
    pub fn set_v(&mut self, v: Array2<f64>) {
        self.set("V", Property::from(v));
    }

    /// Set the gamma matrix in atomic basis.
    pub fn set_gamma(&mut self, gamma: Array2<f64>) {
        self.set("gamma_atom_wise", Property::from(gamma));
    }

    /// Set the gamma matrix in atomic basis for the third order interactions.
    pub fn set_gamma_third_order(&mut self, gamma: Array2<f64>) {
        self.set("gamma_third_order", Property::from(gamma));
    }

    /// Set the gamma matrix in AO basis.
    pub fn set_gamma_ao(&mut self, gamma_ao: Array2<f64>) {
        self.set("gamma_ao_wise", Property::from(gamma_ao));
    }

    /// Set the long-range corrected gamma matrix in atomic basis.
    pub fn set_gamma_lr(&mut self, gamma_lr: Array2<f64>) {
        self.set("gamma_lr_atom_wise", Property::from(gamma_lr));
    }

    /// Set the long-range corrected gamma matrix in AO basis.
    pub fn set_gamma_lr_ao(&mut self, gamma_lr_ao: Array2<f64>) {
        self.set("gamma_lr_ao_wise", Property::from(gamma_lr_ao));
    }

    /// Set the gradient of the gamma matrix in atomic basis.
    pub fn set_grad_gamma(&mut self, grad_gamma: Array3<f64>) {
        self.set("gamma_atom_wise_gradient", Property::from(grad_gamma));
    }

    /// Set the gradient of the gamma matrix in AO basis.
    pub fn set_grad_gamma_ao(&mut self, grad_gamma_ao: Array3<f64>) {
        self.set("gamma_ao_wise_gradient", Property::from(grad_gamma_ao));
    }

    /// Set the gradient of the long-range corrected gamma matrix in atomic basis.
    pub fn set_grad_gamma_lr(&mut self, grad_gamma_lr: Array3<f64>) {
        self.set("gamma_lr_atom_wise_gradient", Property::from(grad_gamma_lr));
    }

    /// Set the gradient of the long-range corrected gamma matrix in AO basis.
    pub fn set_grad_gamma_lr_ao(&mut self, grad_gamma_lr_ao: Array3<f64>) {
        self.set(
            "gamma_lr_ao_wise_gradient",
            Property::from(grad_gamma_lr_ao),
        );
    }

    /// Set the coul/lc-Hamiltonian, which is required for the ground state gradient
    pub fn set_h_coul_x(&mut self, h: Array2<f64>) {
        self.set("h_coul_x", Property::from(h))
    }

    /// Set the transformed Hamiltonian, which is required for the ground state gradient
    pub fn set_h_coul_transformed(&mut self, h: Array2<f64>) {
        self.set("h_coul_transformed", Property::from(h))
    }

    /// set the sum of the excitation vectors x and y
    pub fn set_xpy(&mut self, xpy: Array3<f64>) {
        self.set("xpy", Property::from(xpy))
    }

    /// set the difference of the excitation vectors x and y
    pub fn set_xmy(&mut self, xmy: Array3<f64>) {
        self.set("xmy", Property::from(xmy))
    }

    /// Set the f matrix for the long-range contribution of the exc gradient
    pub fn set_f_lr_dmd0(&mut self, f_lr_dmd0: Array3<f64>) {
        self.set("f_lr_dmd0", Property::from(f_lr_dmd0))
    }

    /// Overlap between orbitals of monomer I and dimer IJ.
    pub fn set_overlap_i_ij(&mut self, s_i_ij: Array2<f64>) {
        self.set("s_i_ij", Property::from(s_i_ij))
    }

    /// Overlap between orbitals of monomer J and dimer IJ.
    pub fn set_overlap_j_ij(&mut self, s_j_ij: Array2<f64>) {
        self.set("s_j_ij", Property::from(s_j_ij))
    }

    pub fn set_last_scalar_coupling(&mut self, sci: Array2<f64>) {
        self.set("last_scalar_coupling", Property::from(sci))
    }

    pub fn set_aligned_pair(&mut self, boolean: bool) {
        self.set("aligned_pair", Property::from(boolean))
    }

    pub fn set_coupling_signs(&mut self, arr: Array1<f64>) {
        self.set("coupling_signs", Property::from(arr))
    }

    pub fn set_u_matrix(&mut self, u_matrix: Array3<f64>) {
        self.set("u_matrix", Property::from(u_matrix))
    }

    pub fn set_orbs_derivative(&mut self, orbs_derivative: Array3<f64>) {
        self.set("orbs_derivative", Property::from(orbs_derivative))
    }

    pub fn set_basis_states(&mut self, basis: Vec<ReducedBasisState>) {
        self.set("basis_states", Property::from(basis))
    }

    pub fn set_old_supersystem(&mut self, supersystem: OldSupersystem) {
        self.set("old_supersystem", Property::from(supersystem))
    }

    pub fn set_ref_supersystem(&mut self, supersystem: OldSupersystem) {
        self.set("ref_supersystem", Property::from(supersystem))
    }

    pub fn set_old_system(&mut self, system: OldSystem) {
        self.set("old_system", Property::from(system))
    }

    pub fn set_state_order(&mut self, state_order: Vec<usize>) {
        self.set("state_order", Property::from(state_order));
    }
}
