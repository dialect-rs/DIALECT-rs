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
        match self.get("old_orbs") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    pub fn old_ci_coeffs(&self) -> Option<ArrayView3<f64>> {
        match self.get("old_ci_coeffs") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
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
        match self.get("homo_index") {
            Some(value) => Some(*value.as_usize().unwrap()),
            _ => None,
        }
    }

    /// Returns the number of occupied orbitals.
    pub fn n_occ(&self) -> Option<usize> {
        match self.get("n_occ") {
            Some(value) => Some(*value.as_usize().unwrap()),
            _ => None,
        }
    }

    /// Returns the number of unoccupied orbitals.
    pub fn n_virt(&self) -> Option<usize> {
        match self.get("n_virt") {
            Some(value) => Some(*value.as_usize().unwrap()),
            _ => None,
        }
    }

    /// Returns the CI coefficients (in MO basis) for a specific excited state.
    pub fn q_trans(&self) -> Option<ArrayView2<f64>> {
        match self.get("q_trans") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns the Mulliken charge for every atomic orbital
    pub fn q_ao(&self) -> Option<ArrayView1<f64>> {
        match self.get("q_ao") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
    }

    /// Returns the transition dipole moments for all excited states.
    pub fn tr_dipoles(&self) -> Option<ArrayView2<f64>> {
        match self.get("tr_dipoles") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
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
        match self.get("oscillator_strengths") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
    }

    /// Returns the CI coefficients (in MO basis) for a specific excited state.
    /// The first excited state has the index 0.
    pub fn ci_coefficient(&self, idx: usize) -> Option<ArrayView1<f64>> {
        match self.get("ci_coefficients") {
            Some(value) => Some(value.as_array2().unwrap().column(idx)),
            _ => None,
        }
    }

    /// Returns the CI coefficients (in MO basis) for a specific excited state.
    /// The first excited state has the index 0.
    pub fn ci_coefficient_b(&self, idx: usize) -> Option<ArrayView1<f64>> {
        match self.get("ci_coefficients_b") {
            Some(value) => Some(value.as_array2().unwrap().column(idx)),
            _ => None,
        }
    }

    /// Returns the CI coefficients (in MO basis) for all computed states.
    pub fn ci_coefficients(&self) -> Option<ArrayView2<f64>> {
        match self.get("ci_coefficients") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns the CI coefficients (in MO basis) for all computed states.
    pub fn ci_coefficients_b(&self) -> Option<ArrayView2<f64>> {
        match self.get("ci_coefficients_b") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns the 1-particle transition density matrix (in MO basis) for an excited state.
    /// The first excited state has the index 0.
    pub fn tdm(&self, idx: usize) -> Option<ArrayView2<f64>> {
        let n_occ: usize = self.occ_indices().unwrap().len();
        let n_virt: usize = self.virt_indices().unwrap().len();
        let n_states: usize = self.ci_eigenvalues().unwrap().len();
        match self.get("ci_coefficients") {
            Some(value) => Some(
                value
                    .as_array2()
                    .unwrap()
                    .view()
                    .into_shape([n_occ, n_virt, n_states])
                    .unwrap()
                    .slice_move(s![.., .., idx]),
            ),
            _ => None,
        }
    }

    /// Returns the 1-particle transition density matrix (in MO basis) for an excited state.
    /// The first excited state has the index 0.
    pub fn tdm_b(&self, idx: usize) -> Option<ArrayView2<f64>> {
        let n_occ: usize = self.occ_indices().unwrap().len();
        let n_virt: usize = self.virt_indices().unwrap().len();
        let n_states: usize = self.ci_eigenvalues_b().unwrap().len();
        match self.get("ci_coefficients_b") {
            Some(value) => Some(
                value
                    .as_array2()
                    .unwrap()
                    .view()
                    .into_shape([n_occ, n_virt, n_states])
                    .unwrap()
                    .slice_move(s![.., .., idx]),
            ),
            _ => None,
        }
    }

    /// Returns the excitation energy of an excited state.
    /// The first excited state has the index 0.
    pub fn ci_eigenvalue(&self, idx: usize) -> Option<f64> {
        match self.get("ci_eigenvalues") {
            Some(value) => Some(value.as_array1().unwrap()[idx]),
            _ => None,
        }
    }

    /// Returns the excitation energy of an excited state.
    /// The first excited state has the index 0.
    pub fn ci_eigenvalue_b(&self, idx: usize) -> Option<f64> {
        match self.get("ci_eigenvalues_b") {
            Some(value) => Some(value.as_array1().unwrap()[idx]),
            _ => None,
        }
    }

    /// Returns a reference to the excitation energies for all excited states.
    pub fn ci_eigenvalues(&self) -> Option<ArrayView1<f64>> {
        match self.get("ci_eigenvalues") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the excitation energies for all excited states.
    pub fn ci_eigenvalues_b(&self) -> Option<ArrayView1<f64>> {
        match self.get("ci_eigenvalues_b") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
    }

    /// Returns the excitation energy of a tddft excited state.
    /// The first excited state has the index 0.
    pub fn tddft_eigenvalue(&self, idx: usize) -> Option<f64> {
        match self.get("tddft_eigenvalues") {
            Some(value) => Some(value.as_array1().unwrap()[idx]),
            _ => None,
        }
    }

    /// Get the tddft excitation energies.
    pub fn tddft_eigenvalues(&self) -> Option<ArrayView1<f64>> {
        match self.get("tddft_eigenvalues") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
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
        match self.get("total_energy") {
            Some(value) => Some(*value.as_double().unwrap()),
            _ => None,
        }
    }

    /// Returns the index of the LUMO.
    pub fn lumo(&self) -> Option<usize> {
        match self.get("lumo_index") {
            Some(value) => Some(*value.as_usize().unwrap()),
            _ => None,
        }
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
        match self.get("ref_density_matrix") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to LCMO Fock matrix.
    pub fn lcmo_fock(&self) -> Option<ArrayView2<f64>> {
        match self.get("lcmo_fock") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the H0 matrix in AO basis.
    pub fn h0(&self) -> Option<ArrayView2<f64>> {
        match self.get("H0") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the MO coefficients.
    pub fn orbs(&self) -> Option<ArrayView2<f64>> {
        match self.get("orbs") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
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
        match self.get("orbs") {
            Some(value) => Some(value.as_array2().unwrap().column(mo_idx)),
            _ => None,
        }
    }

    /// Returns a reference to the MO energies.
    pub fn orbe(&self) -> Option<ArrayView1<f64>> {
        match self.get("orbe") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the overlap matrix in AO basis.
    pub fn s(&self) -> Option<ArrayView2<f64>> {
        match self.get("S") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a slice of the overlap matrix in AO basis.
    pub fn s_slice(&self, rows: Slice, cols: Slice) -> Option<ArrayView2<f64>> {
        match self.get("S") {
            Some(value) => Some(value.as_array2().unwrap().slice(s![rows, cols])),
            _ => None,
        }
    }

    /// Returns a reference to S^-1/2 in AO basis.
    pub fn x(&self) -> Option<ArrayView2<f64>> {
        match self.get("X") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the gradient of the H0 matrix in AO basis.
    pub fn grad_h0(&self) -> Option<ArrayView3<f64>> {
        match self.get("gradH0") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the gradient of the overlap matrix in AO basis.
    pub fn grad_s(&self) -> Option<ArrayView3<f64>> {
        match self.get("gradS") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the charge differences per atom.
    pub fn dq(&self) -> Option<ArrayView1<f64>> {
        match self.get("dq") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the charge differences per atom.
    pub fn dq_alpha(&self) -> Option<ArrayView1<f64>> {
        match self.get("dq_alpha") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the charge differences per atom.
    pub fn dq_beta(&self) -> Option<ArrayView1<f64>> {
        match self.get("dq_beta") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the derivative of the charge (differences) w.r.t. to the degrees
    /// of freedom per atom. The first dimension is dof and the second one is the atom where the charge
    /// resides.
    pub fn grad_dq(&self) -> Option<ArrayView2<f64>> {
        match self.get("grad_dq") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the diagonal of the derivative of charge (differences) w.r.t. to the
    /// degrees of freedom per atom. The first dimension is dof and the second one is the atom
    /// where the charge resides.
    pub fn grad_dq_diag(&self) -> Option<ArrayView1<f64>> {
        match self.get("grad_dq_diag") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the differences of charges differences between the pair and the
    /// corresponding monomers per atom.
    pub fn delta_dq(&self) -> Option<ArrayView1<f64>> {
        match self.get("delta_dq") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the transition charges between occupied and virtual orbitaös
    pub fn q_ov(&self) -> Option<ArrayView2<f64>> {
        match self.get("q_ov") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the transition charges between virtual and occupied orbitals
    pub fn q_vo(&self) -> Option<ArrayView2<f64>> {
        match self.get("q_vo") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the transition charges between occupied orbitals
    pub fn q_oo(&self) -> Option<ArrayView2<f64>> {
        match self.get("q_oo") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the transition charges between occupied orbitals
    pub fn q_oo_restricted(&self) -> Option<ArrayView2<f64>> {
        match self.get("q_oo_restricted") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the transition charges between virtual orbitals
    pub fn q_vv(&self) -> Option<ArrayView2<f64>> {
        match self.get("q_vv") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the transition charges between virtual orbitals
    pub fn q_vv_restricted(&self) -> Option<ArrayView2<f64>> {
        match self.get("q_vv_restricted") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the orbital energy differences between virtual and occupied orbitals
    pub fn omega(&self) -> Option<ArrayView1<f64>> {
        match self.get("omega") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
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
        match self.get("z_vector") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the esp charges per atom
    pub fn esp_q(&self) -> Option<ArrayView1<f64>> {
        match self.get("esp_charges") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
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
        match self.get("V") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the density matrix in AO basis.
    pub fn p(&self) -> Option<ArrayView2<f64>> {
        match self.get("P") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the density matrix of the alpha electrons in AO basis.
    pub fn p_alpha(&self) -> Option<ArrayView2<f64>> {
        match self.get("P_alpha") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the density matrix of the beta electrons in AO basis.
    pub fn p_beta(&self) -> Option<ArrayView2<f64>> {
        match self.get("P_beta") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the difference density matrix in AO basis.
    pub fn delta_p(&self) -> Option<ArrayView2<f64>> {
        match self.get("diff_density_matrix") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the gamma matrix in atomic basis.
    pub fn gamma(&self) -> Option<ArrayView2<f64>> {
        match self.get("gamma_atom_wise") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the gamma matrix in atomic basis for the third order interactions.
    pub fn gamma_third_order(&self) -> Option<ArrayView2<f64>> {
        match self.get("gamma_third_order") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a slice of the gamma matrix in atomic basis.
    pub fn gamma_slice(&self, rows: Slice, cols: Slice) -> Option<ArrayView2<f64>> {
        match self.get("gamma_atom_wise") {
            Some(value) => Some(value.as_array2().unwrap().slice(s![rows, cols])),
            _ => None,
        }
    }

    /// Returns a reference to the gamma matrix in AO basis.
    pub fn gamma_ao(&self) -> Option<ArrayView2<f64>> {
        match self.get("gamma_ao_wise") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the long-range corrected gamma matrix in atomic basis.
    pub fn gamma_lr(&self) -> Option<ArrayView2<f64>> {
        match self.get("gamma_lr_atom_wise") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a slice of the gamma matrix in atomic basis.
    pub fn gamma_lr_slice(&self, rows: Slice, cols: Slice) -> Option<ArrayView2<f64>> {
        match self.get("gamma_lr_atom_wise") {
            Some(value) => Some(value.as_array2().unwrap().slice(s![rows, cols])),
            _ => None,
        }
    }

    /// Returns a reference to the long-range corrected gamma matrix in AO basis.
    pub fn gamma_lr_ao(&self) -> Option<ArrayView2<f64>> {
        match self.get("gamma_lr_ao_wise") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the gradient of the gamma matrix in atomic basis.
    pub fn grad_gamma(&self) -> Option<ArrayView3<f64>> {
        match self.get("gamma_atom_wise_gradient") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the gradient of the gamma matrix in AO basis.
    pub fn grad_gamma_ao(&self) -> Option<ArrayView3<f64>> {
        match self.get("gamma_ao_wise_gradient") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the gradient of the long-range corrected gamma matrix in atomic basis.
    pub fn grad_gamma_lr(&self) -> Option<ArrayView3<f64>> {
        match self.get("gamma_lr_atom_wise_gradient") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the gradient of the long-range corrected gamma matrix in AO basis.
    pub fn grad_gamma_lr_ao(&self) -> Option<ArrayView3<f64>> {
        match self.get("gamma_lr_ao_wise_gradient") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
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
            panic!("Choose differnt monomers!");
            0
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
            0
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
        match self.get("h_coul_x") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Get the transformed-Hamiltonian, which is required for the ground state gradient
    pub fn h_coul_transformed(&self) -> Option<ArrayView2<f64>> {
        match self.get("h_coul_transformed") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the difference of the excitation vectors
    pub fn xmy(&self) -> Option<ArrayView3<f64>> {
        match self.get("xmy") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the sum of the excitation vectors
    pub fn xpy(&self) -> Option<ArrayView3<f64>> {
        match self.get("xpy") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the flr matrix of the long-range corrected contribution to the gradient
    pub fn f_lr_dmd0(&self) -> Option<ArrayView3<f64>> {
        match self.get("f_lr_dmd0") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the overlap between orbitals of monomer I and dimer IJ.
    pub fn s_i_ij(&self) -> Option<ArrayView2<f64>> {
        match self.get("s_i_ij") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the overlap between orbitals of monomer J and dimer IJ.
    pub fn s_j_ij(&self) -> Option<ArrayView2<f64>> {
        match self.get("s_j_ij") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    pub fn last_scalar_coupling(&self) -> Option<ArrayView2<f64>> {
        match self.get("last_scalar_coupling") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    pub fn aligned_pair(&self) -> bool {
        match self.get("aligned_pair") {
            Some(value) => *value.as_bool().unwrap(),
            _ => false,
        }
    }

    pub fn coupling_signs(&self) -> Option<ArrayView1<f64>> {
        match self.get("coupling_signs") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
    }

    pub fn u_matrix(&self) -> Option<ArrayView3<f64>> {
        match self.get("u_matrix") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
    }

    pub fn orbs_derivative(&self) -> Option<ArrayView3<f64>> {
        match self.get("orbs_derivative") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
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
