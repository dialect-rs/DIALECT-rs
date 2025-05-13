use crate::excited_states::ProductCache;
use crate::fmo::old_supersystem::OldSupersystem;
use crate::fmo::{PairType, ReducedBasisState};
use crate::initialization::old_system::OldSystem;
use crate::initialization::Atom;
use crate::properties::Properties;
use hashbrown::HashMap;
use nalgebra::Vector3;
use ndarray::prelude::*;
use ndarray::Slice;
use std::ops::AddAssign;

impl Properties {
    pub fn old_atoms(&self) -> Option<&[Atom]> {
        match self.get("old_atoms") {
            Some(value) => Some(value.as_vec_atom().unwrap()),
            _ => None,
        }
    }

    pub fn old_orbs(&self) -> Option<ArrayView2<f64>> {
        self.get("old_orbs")
            .map(|value| value.as_array2().unwrap().view())
    }

    pub fn old_ci_coeffs(&self) -> Option<ArrayView3<f64>> {
        self.get("old_ci_coeffs")
            .map(|value| value.as_array3().unwrap().view())
    }

    /// Returns a reference the atomic numbers
    pub fn atomic_numbers(&self) -> Option<&[u8]> {
        match self.get("atomic_numbers") {
            Some(value) => Some(value.as_vec_u8().unwrap()),
            _ => None,
        }
    }

    /// Returns the index of the HOMO.
    pub fn homo(&self) -> Option<usize> {
        self.get("homo_index")
            .map(|value| *value.as_usize().unwrap())
    }

    /// Returns the number of occupied orbitals.
    pub fn n_occ(&self) -> Option<usize> {
        self.get("n_occ").map(|value| *value.as_usize().unwrap())
    }

    /// Returns the number of unoccupied orbitals.
    pub fn n_virt(&self) -> Option<usize> {
        self.get("n_virt").map(|value| *value.as_usize().unwrap())
    }

    /// Returns the CI coefficients (in MO basis) for a specific excited state.
    pub fn q_trans(&self) -> Option<ArrayView2<f64>> {
        self.get("q_trans")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns the Mulliken charge for every atomic orbital
    pub fn q_ao(&self) -> Option<ArrayView1<f64>> {
        self.get("q_ao")
            .map(|value| value.as_array1().unwrap().view())
    }

    /// Returns the transition dipole moments for all excited states.
    pub fn tr_dipoles(&self) -> Option<ArrayView2<f64>> {
        self.get("tr_dipoles")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns the transition dipole moment for a specific excited state.
    pub fn tr_dipole(&self, idx: usize) -> Option<Vector3<f64>> {
        match self.get("tr_dipoles") {
            Some(value) => {
                let dip = value.as_array2().unwrap().column(idx);
                Some(Vector3::new(dip[0], dip[1], dip[2]))
            }
            _ => None,
        }
    }

    /// Returns the oscillator strengths for all excited states.
    pub fn oscillator_strengths(&self) -> Option<ArrayView1<f64>> {
        self.get("oscillator_strengths")
            .map(|value| value.as_array1().unwrap().view())
    }

    /// Returns the CI coefficients (in MO basis) for a specific excited state.
    /// The first excited state has the index 0.
    pub fn ci_coefficient(&self, idx: usize) -> Option<ArrayView1<f64>> {
        self.get("ci_coefficients")
            .map(|value| value.as_array2().unwrap().column(idx))
    }

    /// Returns the CI coefficients (in MO basis) for a specific excited state.
    /// The first excited state has the index 0.
    pub fn ci_coefficient_b(&self, idx: usize) -> Option<ArrayView1<f64>> {
        self.get("ci_coefficients_b")
            .map(|value| value.as_array2().unwrap().column(idx))
    }

    /// Returns the CI coefficients (in MO basis) for all computed states.
    pub fn ci_coefficients(&self) -> Option<ArrayView2<f64>> {
        self.get("ci_coefficients")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns the CI coefficients (in MO basis) for all computed states.
    pub fn ci_coefficients_b(&self) -> Option<ArrayView2<f64>> {
        self.get("ci_coefficients_b")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns the 1-particle transition density matrix (in MO basis) for an excited state.
    /// The first excited state has the index 0.
    pub fn tdm(&self, idx: usize) -> Option<ArrayView2<f64>> {
        let n_occ: usize = self.occ_indices().unwrap().len();
        let n_virt: usize = self.virt_indices().unwrap().len();
        let n_states: usize = self.ci_eigenvalues().unwrap().len();
        self.get("ci_coefficients").map(|value| {
            value
                .as_array2()
                .unwrap()
                .view()
                .into_shape([n_occ, n_virt, n_states])
                .unwrap()
                .slice_move(s![.., .., idx])
        })
    }

    /// Returns the 1-particle transition density matrix (in MO basis) for an excited state.
    /// The first excited state has the index 0.
    pub fn tdm_restricted(
        &self,
        idx: usize,
        n_occ: usize,
        n_virt: usize,
    ) -> Option<ArrayView2<f64>> {
        let n_states: usize = self.ci_eigenvalues().unwrap().len();
        self.get("ci_coefficients").map(|value| {
            value
                .as_array2()
                .unwrap()
                .view()
                .into_shape([n_occ, n_virt, n_states])
                .unwrap()
                .slice_move(s![.., .., idx])
        })
    }

    /// Returns the 1-particle transition density matrix (in MO basis) for an excited state.
    /// The first excited state has the index 0.
    pub fn tdm_b(&self, idx: usize) -> Option<ArrayView2<f64>> {
        let n_occ: usize = self.occ_indices().unwrap().len();
        let n_virt: usize = self.virt_indices().unwrap().len();
        let n_states: usize = self.ci_eigenvalues_b().unwrap().len();
        self.get("ci_coefficients_b").map(|value| {
            value
                .as_array2()
                .unwrap()
                .view()
                .into_shape([n_occ, n_virt, n_states])
                .unwrap()
                .slice_move(s![.., .., idx])
        })
    }

    pub fn gw_screend_eris(&self, idx: usize) -> Option<ArrayView2<f64>> {
        let n_occ: usize = self.occ_indices().unwrap().len();
        let n_virt: usize = self.virt_indices().unwrap().len();
        let n_states: usize = n_occ * n_virt;
        let n_orbs: usize = n_occ + n_virt;
        self.get("gw_eris").map(|value| {
            value
                .as_array3()
                .unwrap()
                .view()
                .into_shape([n_orbs, n_orbs, n_states])
                .unwrap()
                .slice_move(s![.., .., idx])
        })
    }

    /// Returns the excitation energy of an excited state.
    /// The first excited state has the index 0.
    pub fn ci_eigenvalue(&self, idx: usize) -> Option<f64> {
        self.get("ci_eigenvalues")
            .map(|value| value.as_array1().unwrap()[idx])
    }

    /// Returns the excitation energy of an excited state.
    /// The first excited state has the index 0.
    pub fn ci_eigenvalue_b(&self, idx: usize) -> Option<f64> {
        self.get("ci_eigenvalues_b")
            .map(|value| value.as_array1().unwrap()[idx])
    }

    /// Returns a reference to the excitation energies for all excited states.
    pub fn ci_eigenvalues(&self) -> Option<ArrayView1<f64>> {
        self.get("ci_eigenvalues")
            .map(|value| value.as_array1().unwrap().view())
    }

    /// Returns a reference to the excitation energies for all excited states.
    pub fn ci_eigenvalues_b(&self) -> Option<ArrayView1<f64>> {
        self.get("ci_eigenvalues_b")
            .map(|value| value.as_array1().unwrap().view())
    }

    /// Returns the excitation energy of a tddft excited state.
    /// The first excited state has the index 0.
    pub fn tddft_eigenvalue(&self, idx: usize) -> Option<f64> {
        self.get("tddft_eigenvalues")
            .map(|value| value.as_array1().unwrap()[idx])
    }

    /// Get the tddft excitation energies.
    pub fn tddft_eigenvalues(&self) -> Option<ArrayView1<f64>> {
        self.get("tddft_eigenvalues")
            .map(|value| value.as_array1().unwrap().view())
    }

    /// Returns the state energies for all states.
    pub fn state_energies(&self) -> Option<Array1<f64>> {
        // Reference to the excitation energies.
        let ci_eig: ArrayView1<f64> = self
            .get("ci_eigenvalues")
            .unwrap()
            .as_array1()
            .unwrap()
            .view();
        // The total ground state energy.
        let total_energy: f64 = self.total_energy().unwrap();
        // An array is created with the total ground state energy for each state.
        let mut energies: Array1<f64> = Array1::from_elem(ci_eig.len() + 1, total_energy);
        // and the excitation energies are added to the states.
        energies.slice_mut(s![1..]).add_assign(&ci_eig);

        Some(energies)
    }

    pub fn total_energy(&self) -> Option<f64> {
        self.get("total_energy")
            .map(|value| *value.as_double().unwrap())
    }

    /// Returns the index of the LUMO.
    pub fn lumo(&self) -> Option<usize> {
        self.get("lumo_index")
            .map(|value| *value.as_usize().unwrap())
    }

    /// Returns the energy of the last scc iteration
    pub fn last_energy(&self) -> Option<f64> {
        match self.get("last_energy") {
            Some(value) => Some(*value.as_double().unwrap()),
            _ => Some(0.0),
        }
    }

    /// Returns the energy of the last scc iteration
    pub fn occupation(&self) -> Option<&[f64]> {
        match self.get("occupation") {
            Some(value) => Some(value.as_vec_f64().unwrap()),
            _ => None,
        }
    }

    /// Returns a reference to the reference density matrix
    pub fn p_ref(&self) -> Option<ArrayView2<f64>> {
        self.get("ref_density_matrix")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to LCMO Fock matrix.
    pub fn lcmo_fock(&self) -> Option<ArrayView2<f64>> {
        self.get("lcmo_fock")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the H0 matrix in AO basis.
    pub fn h0(&self) -> Option<ArrayView2<f64>> {
        self.get("H0")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the MO coefficients.
    pub fn orbs(&self) -> Option<ArrayView2<f64>> {
        self.get("orbs")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the MO coefficients.
    pub fn orbs_slice(&self, start: usize, end: Option<usize>) -> Option<ArrayView2<f64>> {
        match self.get("orbs") {
            Some(value) => match end {
                Some(num) => Some(value.as_array2().unwrap().slice(s![.., start..num])),
                None => Some(value.as_array2().unwrap().slice(s![.., start..])),
            },
            _ => None,
        }
    }

    /// Returns a reference to the MO coefficients of a single MO.
    pub fn mo_coeff(&self, mo_idx: usize) -> Option<ArrayView1<f64>> {
        self.get("orbs")
            .map(|value| value.as_array2().unwrap().column(mo_idx))
    }

    /// Returns a reference to the MO energies.
    pub fn orbe(&self) -> Option<ArrayView1<f64>> {
        self.get("orbe")
            .map(|value| value.as_array1().unwrap().view())
    }

    /// Returns a reference to the MO energies.
    pub fn gw_orbe(&self) -> Option<ArrayView1<f64>> {
        self.get("gw_orbe")
            .map(|value| value.as_array1().unwrap().view())
    }

    pub fn gw_rpa_energies(&self) -> Option<ArrayView1<f64>> {
        self.get("gw_rpa_energies")
            .map(|value| value.as_array1().unwrap().view())
    }

    /// Returns a reference to the overlap matrix in AO basis.
    pub fn s(&self) -> Option<ArrayView2<f64>> {
        self.get("S").map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a slice of the overlap matrix in AO basis.
    pub fn s_slice(&self, rows: Slice, cols: Slice) -> Option<ArrayView2<f64>> {
        self.get("S")
            .map(|value| value.as_array2().unwrap().slice(s![rows, cols]))
    }

    /// Returns a reference to S^-1/2 in AO basis.
    pub fn x(&self) -> Option<ArrayView2<f64>> {
        self.get("X").map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the gradient of the H0 matrix in AO basis.
    pub fn grad_h0(&self) -> Option<ArrayView3<f64>> {
        self.get("gradH0")
            .map(|value| value.as_array3().unwrap().view())
    }

    /// Returns a reference to the gradient of the overlap matrix in AO basis.
    pub fn grad_s(&self) -> Option<ArrayView3<f64>> {
        self.get("gradS")
            .map(|value| value.as_array3().unwrap().view())
    }

    /// Returns a reference to the charge differences per atom.
    pub fn dq(&self) -> Option<ArrayView1<f64>> {
        self.get("dq")
            .map(|value| value.as_array1().unwrap().view())
    }

    /// Returns a reference to the charge differences per AO.
    pub fn dq_ao(&self) -> Option<ArrayView1<f64>> {
        self.get("dq_ao")
            .map(|value| value.as_array1().unwrap().view())
    }

    /// Returns a reference to the charge differences per atom.
    pub fn dq_alpha(&self) -> Option<ArrayView1<f64>> {
        self.get("dq_alpha")
            .map(|value| value.as_array1().unwrap().view())
    }

    /// Returns a reference to the charge differences per atom.
    pub fn dq_beta(&self) -> Option<ArrayView1<f64>> {
        self.get("dq_beta")
            .map(|value| value.as_array1().unwrap().view())
    }

    /// Returns a reference to the derivative of the charge (differences) w.r.t. to the degrees
    /// of freedom per atom. The first dimension is dof and the second one is the atom where the charge
    /// resides.
    pub fn grad_dq(&self) -> Option<ArrayView2<f64>> {
        self.get("grad_dq")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the diagonal of the derivative of charge (differences) w.r.t. to the
    /// degrees of freedom per atom. The first dimension is dof and the second one is the atom
    /// where the charge resides.
    pub fn grad_dq_diag(&self) -> Option<ArrayView1<f64>> {
        self.get("grad_dq_diag")
            .map(|value| value.as_array1().unwrap().view())
    }

    /// Returns a reference to the differences of charges differences between the pair and the
    /// corresponding monomers per atom.
    pub fn delta_dq(&self) -> Option<ArrayView1<f64>> {
        self.get("delta_dq")
            .map(|value| value.as_array1().unwrap().view())
    }

    /// Returns a reference to the transition charges between occupied and virtual orbitaös
    pub fn q_ov(&self) -> Option<ArrayView2<f64>> {
        self.get("q_ov")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the transition charges between virtual and occupied orbitals
    pub fn q_vo(&self) -> Option<ArrayView2<f64>> {
        self.get("q_vo")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the transition charges between occupied orbitals
    pub fn q_oo(&self) -> Option<ArrayView2<f64>> {
        self.get("q_oo")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the transition charges between occupied orbitals
    pub fn q_oo_restricted(&self) -> Option<ArrayView2<f64>> {
        self.get("q_oo_restricted")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the transition charges between virtual orbitals
    pub fn q_vv(&self) -> Option<ArrayView2<f64>> {
        self.get("q_vv")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the transition charges between virtual orbitals
    pub fn q_vv_restricted(&self) -> Option<ArrayView2<f64>> {
        self.get("q_vv_restricted")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the orbital energy differences between virtual and occupied orbitals
    pub fn omega(&self) -> Option<ArrayView1<f64>> {
        self.get("omega")
            .map(|value| value.as_array1().unwrap().view())
    }

    /// Returns a mutable reference to the `ProductCache`.
    pub fn cache(&self) -> Option<&ProductCache> {
        match self.get("cache") {
            Some(value) => Some(value.as_cache().unwrap()),
            _ => None,
        }
    }

    /// Returns a reference to the converged Z-vector from the FMO grad. response term.
    pub fn z_vector(&self) -> Option<ArrayView1<f64>> {
        self.get("z_vector")
            .map(|value| value.as_array1().unwrap().view())
    }

    /// Returns a reference to the esp charges per atom
    pub fn esp_q(&self) -> Option<ArrayView1<f64>> {
        self.get("esp_charges")
            .map(|value| value.as_array1().unwrap().view())
    }

    /// Returns a reference to the indices of the occupied orbitals.
    pub fn occ_indices(&self) -> Option<&[usize]> {
        match self.get("occ_indices") {
            Some(value) => Some(value.as_vec_usize().unwrap()),
            _ => None,
        }
    }

    /// Returns a reference to the indices of the virtual orbitals.
    pub fn virt_indices(&self) -> Option<&[usize]> {
        match self.get("virt_indices") {
            Some(value) => Some(value.as_vec_usize().unwrap()),
            _ => None,
        }
    }

    /// Returns a reference to the electrostatic potential in AO basis.
    pub fn v(&self) -> Option<ArrayView2<f64>> {
        self.get("V").map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the density matrix in AO basis.
    pub fn p(&self) -> Option<ArrayView2<f64>> {
        self.get("P").map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the density matrix of the alpha electrons in AO basis.
    pub fn p_alpha(&self) -> Option<ArrayView2<f64>> {
        self.get("P_alpha")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the density matrix of the beta electrons in AO basis.
    pub fn p_beta(&self) -> Option<ArrayView2<f64>> {
        self.get("P_beta")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the difference density matrix in AO basis.
    pub fn delta_p(&self) -> Option<ArrayView2<f64>> {
        self.get("diff_density_matrix")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the gamma matrix in atomic basis.
    pub fn gamma(&self) -> Option<ArrayView2<f64>> {
        self.get("gamma_atom_wise")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the gamma matrix in atomic basis for the third order interactions.
    pub fn gamma_third_order(&self) -> Option<ArrayView2<f64>> {
        self.get("gamma_third_order")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a slice of the gamma matrix in atomic basis.
    pub fn gamma_slice(&self, rows: Slice, cols: Slice) -> Option<ArrayView2<f64>> {
        self.get("gamma_atom_wise")
            .map(|value| value.as_array2().unwrap().slice(s![rows, cols]))
    }

    /// Returns a reference to the gamma matrix in AO basis.
    pub fn gamma_ao(&self) -> Option<ArrayView2<f64>> {
        self.get("gamma_ao_wise")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a slice of the gamma matrix in AO basis.
    pub fn gamma_ao_slice(&self, rows: Slice, cols: Slice) -> Option<ArrayView2<f64>> {
        self.get("gamma_ao_wise")
            .map(|value| value.as_array2().unwrap().slice(s![rows, cols]))
    }

    /// Returns a reference to the long-range corrected gamma matrix in atomic basis.
    pub fn gamma_lr(&self) -> Option<ArrayView2<f64>> {
        self.get("gamma_lr_atom_wise")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a slice of the gamma matrix in atomic basis.
    pub fn gamma_lr_slice(&self, rows: Slice, cols: Slice) -> Option<ArrayView2<f64>> {
        self.get("gamma_lr_atom_wise")
            .map(|value| value.as_array2().unwrap().slice(s![rows, cols]))
    }

    /// Returns a slice of the gamma matrix in AO basis.
    pub fn gamma_lr_ao_slice(&self, rows: Slice, cols: Slice) -> Option<ArrayView2<f64>> {
        self.get("gamma_lr_ao_wise")
            .map(|value| value.as_array2().unwrap().slice(s![rows, cols]))
    }

    /// Returns a reference to the long-range corrected gamma matrix in AO basis.
    pub fn gamma_lr_ao(&self) -> Option<ArrayView2<f64>> {
        self.get("gamma_lr_ao_wise")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the gradient of the gamma matrix in atomic basis.
    pub fn grad_gamma(&self) -> Option<ArrayView3<f64>> {
        self.get("gamma_atom_wise_gradient")
            .map(|value| value.as_array3().unwrap().view())
    }

    /// Returns a reference to the gradient of the gamma matrix in AO basis.
    pub fn grad_gamma_ao(&self) -> Option<ArrayView3<f64>> {
        self.get("gamma_ao_wise_gradient")
            .map(|value| value.as_array3().unwrap().view())
    }

    /// Returns a reference to the gradient of the long-range corrected gamma matrix in atomic basis.
    pub fn grad_gamma_lr(&self) -> Option<ArrayView3<f64>> {
        self.get("gamma_lr_atom_wise_gradient")
            .map(|value| value.as_array3().unwrap().view())
    }

    /// Returns a reference to the gradient of the long-range corrected gamma matrix in AO basis.
    pub fn grad_gamma_lr_ao(&self) -> Option<ArrayView3<f64>> {
        self.get("gamma_lr_ao_wise_gradient")
            .map(|value| value.as_array3().unwrap().view())
    }

    /// Get type of a monomer pair.
    pub fn type_of_pair(&self, i: usize, j: usize) -> PairType {
        if i == j {
            PairType::None
        } else {
            let map: &HashMap<(usize, usize), PairType> =
                self.get("pair_types").unwrap().as_pair_map().unwrap();
            map.get(&(i, j))
                .unwrap_or_else(|| map.get(&(j, i)).unwrap())
                .to_owned()
        }
    }

    /// Get the index of a monomer pair.
    pub fn index_of_pair(&self, i: usize, j: usize) -> usize {
        if i == j {
            panic!("Choose different monomers!");
        } else {
            let map: &HashMap<(usize, usize), usize> = self
                .get("pair_indices")
                .unwrap()
                .as_pair_index_map()
                .unwrap();
            map.get(&(i, j))
                .unwrap_or_else(|| map.get(&(j, i)).unwrap())
                .to_owned()
        }
    }

    /// Get the index of a ESD monomer pair.
    pub fn index_of_esd_pair(&self, i: usize, j: usize) -> usize {
        if i == j {
            panic!("Choose differnt monomers!");
        } else {
            let map: &HashMap<(usize, usize), usize> = self
                .get("esd_pair_indices")
                .unwrap()
                .as_pair_index_map()
                .unwrap();
            map.get(&(i, j))
                .unwrap_or_else(|| map.get(&(j, i)).unwrap())
                .to_owned()
        }
    }

    /// Get the coul/lc-Hamiltonian, which is required for the ground state gradient
    pub fn h_coul_x(&self) -> Option<ArrayView2<f64>> {
        self.get("h_coul_x")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Get the transformed-Hamiltonian, which is required for the ground state gradient
    pub fn h_coul_transformed(&self) -> Option<ArrayView2<f64>> {
        self.get("h_coul_transformed")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the difference of the excitation vectors
    pub fn xmy(&self) -> Option<ArrayView3<f64>> {
        self.get("xmy")
            .map(|value| value.as_array3().unwrap().view())
    }

    /// Returns a reference to the sum of the excitation vectors
    pub fn xpy(&self) -> Option<ArrayView3<f64>> {
        self.get("xpy")
            .map(|value| value.as_array3().unwrap().view())
    }

    /// Returns a reference to the flr matrix of the long-range corrected contribution to the gradient
    pub fn f_lr_dmd0(&self) -> Option<ArrayView3<f64>> {
        self.get("f_lr_dmd0")
            .map(|value| value.as_array3().unwrap().view())
    }

    /// Returns a reference to the overlap between orbitals of monomer I and dimer IJ.
    pub fn s_i_ij(&self) -> Option<ArrayView2<f64>> {
        self.get("s_i_ij")
            .map(|value| value.as_array2().unwrap().view())
    }

    /// Returns a reference to the overlap between orbitals of monomer J and dimer IJ.
    pub fn s_j_ij(&self) -> Option<ArrayView2<f64>> {
        self.get("s_j_ij")
            .map(|value| value.as_array2().unwrap().view())
    }

    pub fn last_scalar_coupling(&self) -> Option<ArrayView2<f64>> {
        self.get("last_scalar_coupling")
            .map(|value| value.as_array2().unwrap().view())
    }

    pub fn aligned_pair(&self) -> bool {
        match self.get("aligned_pair") {
            Some(value) => *value.as_bool().unwrap(),
            _ => false,
        }
    }

    pub fn coupling_signs(&self) -> Option<ArrayView1<f64>> {
        self.get("coupling_signs")
            .map(|value| value.as_array1().unwrap().view())
    }

    pub fn u_matrix(&self) -> Option<ArrayView3<f64>> {
        self.get("u_matrix")
            .map(|value| value.as_array3().unwrap().view())
    }

    pub fn orbs_derivative(&self) -> Option<ArrayView3<f64>> {
        self.get("orbs_derivative")
            .map(|value| value.as_array3().unwrap().view())
    }

    pub fn basis_states(&self) -> Option<&[ReducedBasisState]> {
        match self.get("basis_states") {
            Some(value) => Some(value.as_vec_basis().unwrap()),
            _ => None,
        }
    }

    pub fn old_supersystem(&self) -> Option<&OldSupersystem> {
        match self.get("old_supersystem") {
            Some(value) => Some(value.as_super_system().unwrap()),
            _ => None,
        }
    }
    pub fn ref_supersystem(&self) -> Option<&OldSupersystem> {
        match self.get("ref_supersystem") {
            Some(value) => Some(value.as_super_system().unwrap()),
            _ => None,
        }
    }
    pub fn old_system(&self) -> Option<&OldSystem> {
        match self.get("old_system") {
            Some(value) => Some(value.as_old_system().unwrap()),
            _ => None,
        }
    }

    pub fn state_order(&self) -> Option<&[usize]> {
        match self.get("state_order") {
            Some(value) => Some(value.as_vec_usize().unwrap()),
            _ => None,
        }
    }
}
