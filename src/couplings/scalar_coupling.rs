use crate::couplings::overlaps::get_sign_of_array;
use crate::defaults;
use crate::fmo::old_supersystem::{OldMonomer, OldSupersystem};
use crate::fmo::{ChargeTransferPair, Monomer, ReducedBasisState, ReducedLE, SuperSystem};
use crate::initialization::old_system::OldSystem;
use crate::initialization::parameters::SlaterKoster;
use crate::initialization::System;
use crate::param::slako_transformations::{directional_cosines, slako_transformation};
use ndarray::prelude::*;
use ndarray_linalg::Determinant;
use rayon::prelude::*;
use std::time::Instant;

impl System {
    pub fn get_scalar_coupling(&mut self, dt: f64, step: usize) -> (Array2<f64>, Array2<f64>) {
        let n_states: usize = self.config.excited.nstates + 1;

        let old_system = if !self.properties.old_system().is_none() {
            self.properties.old_system().unwrap().clone()
        } else {
            println!("Create old system at first step!");
            OldSystem::new(&self, None, None)
        };

        // scalar coupling matrix
        let s_ci: Array2<f64> =
            self.ci_overlap_system(
                &old_system.atoms,
                old_system.orbs.view(),
                old_system.ci_coefficients.view(),
                n_states,
                step,
            );
        // align phases
        // The eigenvalue solver produces vectors with arbitrary global phases
        // (+1 or -1). The orbitals of the ground state can also change their signs.
        // Eigen states from neighbouring geometries should change continuously.
        let diag = s_ci.diag();
        // get signs of the diagonal
        let sign: Array1<f64> = get_sign_of_array(diag);

        let p: Array2<f64> = Array::from_diag(&sign);
        // align the new CI coefficients with the old coefficients
        let p_exclude_gs: ArrayView2<f64> = p.slice(s![1.., 1..]);

        // save the aligned coefficients
        let aligned_coeff: Array2<f64> = self
            .properties
            .ci_coefficients()
            .unwrap()
            .dot(&p_exclude_gs);

        self.properties.set_ci_coefficients(aligned_coeff);

        // align overlap matrix
        let mut s_ci = s_ci.dot(&p);

        // The relative signs for the overlap between the ground and excited states at different geometries
        // cannot be deduced from the diagonal elements of Sci. The phases are chosen such that the coupling
        // between S0 and S1-SN changes smoothly for most of the states.
        if old_system.old_scalar_couplings.is_some() {
            let old_s_ci: Array2<f64> = old_system.old_scalar_couplings.unwrap();
            let s: Array1<f64> =
                get_sign_of_array((&old_s_ci.slice(s![0, 1..]) / &s_ci.slice(s![0, 1..])).view());
            let w: Array1<f64> =
                (&s_ci.slice(s![0, 1..]) - &old_s_ci.slice(s![0, 1..])).map(|val| val.abs());
            let mut mean_sign: f64 = ((&w * &s).sum() / w.sum()).signum();
            if mean_sign.is_nan() {
                mean_sign = 1.0
            }
            for i in 1..n_states {
                s_ci[[0, i]] *= mean_sign;
                s_ci[[i, 0]] *= mean_sign;
            }
        }

        // coupl[A,B] = <Psi_A(t)|Psi_B(t+dt)> - delta_AB
        //            ~ <Psi_A(t)|d/dR Psi_B(t)>*dR/dt dt
        // The scalar coupling matrix should be more or less anti-symmetric
        // provided the time-step is small enough
        // set diagonal elements of coupl to zero
        let mut coupling: Array2<f64> = s_ci.clone();
        coupling = coupling - Array::from_diag(&s_ci.diag());

        // Because of the finite time-step it will not be completely antisymmetric,
        coupling = 0.5 * (&coupling - &coupling.t());

        let old_system: OldSystem = OldSystem::new(&self, Some(coupling.clone()), None);
        self.properties.set_old_system(old_system);

        // coupl = <Psi_A|d/dR Psi_B>*dR/dt * dt
        coupling = coupling / dt;

        // coupling
        return (coupling, s_ci);
    }
}

impl SuperSystem<'_> {
    pub fn nonadiabatic_scalar_coupling(
        &mut self,
        excitonic_coupling: ArrayView2<f64>,
        dt: f64,
    ) -> (
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
        Array1<f64>,
        Array1<f64>,
    ) {
        // get old supersystem from properties
        let old_system = self.properties.old_supersystem().unwrap().to_owned();

        // calculate the overlap of the wavefunctions
        let (sci_overlap, s_ao): (Array2<f64>, Array2<f64>) =
            self.scalar_coupling_ci_overlaps(&old_system);
        let dim: usize = sci_overlap.dim().0;

        // align phases
        // The eigenvalue solver produces vectors with arbitrary global phases
        // (+1 or -1). The orbitals of the ground state can also change their signs.
        // Eigen states from neighbouring geometries should change continuously.
        let diag = sci_overlap.diag();
        // get signs of the diagonal
        let sign: Array1<f64> = get_sign_of_array(diag);

        // create 2D matrix from the sign array
        let p: Array2<f64> = Array::from_diag(&sign);
        // align the new CI coefficients with the old coefficients
        let p_exclude_gs: ArrayView2<f64> = p.slice(s![1.., 1..]);
        // align the excitonic coupling matrix using the p matrix
        let excitonic_coupling: Array2<f64> =
            p_exclude_gs.dot(&excitonic_coupling).dot(&p_exclude_gs);

        // align overlap matrix
        let mut s_ci = sci_overlap.dot(&p);

        // set diagonal elements of coupl to zero
        let mut coupling: Array2<f64> = s_ci.clone();
        coupling = coupling - Array::from_diag(&s_ci.diag());
        // coupl[A,B] = <Psi_A(t)|Psi_B(t+dt)> - delta_AB
        //            ~ <Psi_A(t)|d/dR Psi_B(t)>*dR/dt dt
        // Because of the finite time-step it will not be completely antisymmetric,
        // so antisymmetrize it
        coupling = 0.5 * (&coupling - &coupling.t());

        // save the last coupling matrix
        let last_coupling: Array2<f64> = coupling.clone();
        self.properties.set_last_scalar_coupling(last_coupling);

        // coupl = <Psi_A|d/dR Psi_B>*dR/dt * dt
        coupling = coupling / dt;

        // align the CI coefficients
        self.scalar_coupling_align_coefficients(sign.view());

        // create the OldSupersystem and store it
        let old_system = OldSupersystem::new(&self);
        self.properties.set_old_supersystem(old_system);

        return (coupling, excitonic_coupling, s_ao, diag.to_owned(), sign);
    }

    pub fn align_signs_diabatic_hamiltonian(
        &mut self,
        excitonic_coupling: ArrayView2<f64>,
    ) -> (Array2<f64>) {
        let timer: Instant = Instant::now();
        // get old supersystem from properties
        let old_supersystem = self.properties.old_supersystem();
        // get the old supersystem
        let old_system: OldSupersystem = if old_supersystem.is_some() {
            old_supersystem.unwrap().to_owned()
        }
        // if the dynamic is at it's first step, calculate the coupling between the
        // starting geometry
        else {
            OldSupersystem::new(&self)
        };

        // calculate the overlap of the wavefunctions
        let sci_overlap_diag: Array1<f64> = self.scalar_coupling_ci_overlaps_diagonal(&old_system);

        // align phases
        // The eigenvalue solver produces vectors with arbitrary global phases
        // (+1 or -1). The orbitals of the ground state can also change their signs.
        // Eigen states from neighbouring geometries should change continuously.
        let sign: Array1<f64> = get_sign_of_array(sci_overlap_diag.view());

        // create 2D matrix from the sign array
        let p: Array2<f64> = Array::from_diag(&sign);
        // align the excitonic coupling matrix using the p matrix
        let excitonic_coupling: Array2<f64> = p.dot(&excitonic_coupling).dot(&p);

        let mut signs: Array1<f64> = Array1::zeros(sign.len() + 1);
        signs.slice_mut(s![1..]).assign(&sign);

        // align the CI coefficients
        self.scalar_coupling_align_coefficients(signs.view());

        // create the OldSupersystem and store it
        let old_system = OldSupersystem::new(&self);
        self.properties.set_old_supersystem(old_system);

        excitonic_coupling
    }

    pub fn scalar_coupling_align_coefficients(&mut self, signs: ArrayView1<f64>) {
        let mut basis_states = self.properties.basis_states().unwrap().to_vec();

        for (idx, state) in basis_states.iter_mut().enumerate() {
            match state {
                ReducedBasisState::LE(ref state_a) => {
                    let mol: &mut Monomer = &mut self.monomers[state_a.monomer_index];
                    let mut ci_full: Array2<f64> =
                        mol.properties.ci_coefficients().unwrap().to_owned();
                    let ci: ArrayView1<f64> =
                        mol.properties.ci_coefficient(state_a.state_index).unwrap();
                    let ci_aligned: Array1<f64> = signs[idx + 1] * &ci;
                    ci_full
                        .slice_mut(s![.., state_a.state_index])
                        .assign(&ci_aligned);
                    mol.properties.set_ci_coefficients(ci_full);
                }
                ReducedBasisState::CT(ref mut state_a) => {
                    // change the sign of the CI vectors
                    let ci_coeff: ArrayView2<f64> = state_a.eigenvectors.view();
                    let ci_aligned: Array2<f64> = &ci_coeff * signs[idx + 1];
                    state_a.eigenvectors = ci_aligned;
                }
            }
        }
        self.properties.set_basis_states(basis_states);
    }

    pub fn scalar_coupling_ci_overlaps(
        &self,
        other: &OldSupersystem,
    ) -> (Array2<f64>, Array2<f64>) {
        let basis_states = self.properties.basis_states().unwrap();
        let old_basis = &other.basis_states;

        // get the slater koster parameters
        let slako = &self.monomers[0].slako;
        // calculate the overlap matrix between the timesteps
        let s: Array2<f64> = self.supersystem_overlap_between_timesteps(other, slako);

        // empty coupling array
        let mut coupling: Array2<f64> =
            Array2::zeros([basis_states.len() + 1, basis_states.len() + 1]);

        let coupling_vec: Vec<Array1<f64>> = basis_states
            .par_iter()
            .map(|state_i| {
                let mut arr: Array1<f64> = Array1::zeros(basis_states.len());
                for (idx_j, state_j) in old_basis.iter().enumerate() {
                    // coupling between the diabatic states
                    arr[idx_j] =
                        self.scalar_coupling_diabatic_states(other, state_i, state_j, s.view())
                }
                arr
            })
            .collect();

        // slice the coupling matrix elements
        for (idx, arr) in coupling_vec.iter().enumerate() {
            coupling.slice_mut(s![idx + 1, 1..]).assign(arr);
        }

        (coupling, s)
    }

    pub fn scalar_coupling_ci_overlaps_diagonal(&self, other: &OldSupersystem) -> Array1<f64> {
        let basis_states = self.properties.basis_states().unwrap();
        let old_basis = &other.basis_states;

        // get the slater koster parameters
        let slako = &self.monomers[0].slako;
        // calculate the overlap matrix between the timesteps
        let s: Array2<f64> = self.supersystem_overlap_between_timesteps(other, slako);

        let coupling_vec: Vec<f64> =
            basis_states
                .par_iter()
                .zip(old_basis.par_iter())
                .map(|(state_i, state_j)| {
                    self.scalar_coupling_diabatic_states(other, state_i, state_j, s.view())
                })
                .collect();

        Array::from(coupling_vec)
    }

    pub fn scalar_coupling_diabatic_gs(
        &self,
        other: &OldSupersystem,
        state: &ReducedBasisState,
        overlap: ArrayView2<f64>,
        gs_old: bool,
    ) -> f64 {
        match state {
            ReducedBasisState::LE(ref a) => self.scalar_coupling_le_gs(other, a, overlap, gs_old),
            ReducedBasisState::CT(ref a) => self.scalar_coupling_ct_gs(other, a, overlap, gs_old),
        }
    }

    pub fn scalar_coupling_le_gs(
        &self,
        other: &OldSupersystem,
        state: &ReducedLE,
        overlap: ArrayView2<f64>,
        gs_old: bool,
    ) -> f64 {
        // get the monomers of the LE state of the new and old Supersystem
        let m_new: &Monomer = &self.monomers[state.monomer_index];
        let m_old: &OldMonomer = &other.monomers[state.monomer_index];

        // slice the overlap matrix to get the AO overlap of the LE
        let s_ao: ArrayView2<f64> = overlap.slice(s![m_new.slice.orb, m_old.slice.orb]);
        // get the MO coefficients of the old and the new geometry
        let orbs_old: ArrayView2<f64> = m_old.orbs.view();
        let orbs_new: ArrayView2<f64> = m_new.properties.orbs().unwrap();
        // transform the AO overlap to the MO basis
        let s_mo: Array2<f64> = orbs_new.t().dot(&s_ao.dot(&orbs_old));

        // get the CI matrix of the LE state
        let nocc: usize = m_new.properties.occ_indices().unwrap().len();
        let nvirt: usize = m_new.properties.virt_indices().unwrap().len();
        let mut ci: Array2<f64> = Array2::zeros([nocc, nvirt]);
        if gs_old {
            ci = m_new.properties.tdm(state.state_index).unwrap().to_owned();
        } else {
            let ci_vector: ArrayView1<f64> = m_old.tdm.slice(s![.., state.state_index]);
            ci = ci_vector.to_owned().into_shape([nocc, nvirt]).unwrap();
        }
        // get the occupied MO overlap matrix
        let s_mo_occ: ArrayView2<f64> = s_mo.slice(s![..nocc, ..nocc]);

        // calculate the ci_overlap
        self.ci_overlap_state_gs(s_mo.view(), s_mo_occ, ci.view(), gs_old)
    }

    pub fn ci_overlap_state_gs(
        &self,
        s_mo: ArrayView2<f64>,
        s_mo_occ: ArrayView2<f64>,
        ci: ArrayView2<f64>,
        gs_old: bool,
    ) -> f64 {
        // Excitations i->a with coefficients |C_ia| < threshold will be neglected
        let threshold: f64 = 0.0005;
        // get nocc and nvirt
        let nocc: usize = ci.dim().0;
        let nvirt: usize = ci.dim().1;

        // calc the ground state part of the ci overlap
        let det_ij: f64 = s_mo_occ.det().unwrap();
        let norb: usize = nocc + nvirt;

        // scalar coupling value
        let mut s_ci: f64 = 0.0;

        for i in 0..nocc {
            for (a_idx, a) in (nocc..norb).into_iter().enumerate() {
                // slice the CI coefficients of the diabatic state J at the indicies i and a
                let coeff = ci[[i, a_idx]];
                if coeff.abs() > threshold {
                    if gs_old {
                        let mut s_aj: Array2<f64> = s_mo_occ.to_owned();
                        // occupied orbitals in the configuration state function |Psi_ia>
                        // overlap <1,...,a,...|1,...,j,...>
                        s_aj.slice_mut(s![i, ..nocc])
                            .assign(&s_mo.slice(s![a, ..nocc]));
                        let det_aj: f64 = s_aj.det().unwrap();

                        // get the determinant between the LE of I and the ground state of J
                        s_ci += coeff * 2.0_f64.sqrt() * det_ij * det_aj;
                    } else {
                        let mut s_aj: Array2<f64> = s_mo_occ.to_owned();
                        // occupied orbitals in the configuration state function |Psi_ia>
                        // overlap <1,...,j,...|1,...,a,...>
                        s_aj.slice_mut(s![..nocc, i])
                            .assign(&s_mo.slice(s![..nocc, a]));
                        let det_aj: f64 = s_aj.det().unwrap();

                        // get the determinant between the LE of I and the ground state of J
                        s_ci += coeff * 2.0_f64.sqrt() * det_ij * det_aj;
                    }
                }
            }
        }
        s_ci
    }

    pub fn scalar_coupling_ct_gs(
        &self,
        other: &OldSupersystem,
        state: &ChargeTransferPair,
        overlap: ArrayView2<f64>,
        gs_old: bool,
    ) -> f64 {
        // get the references to the monomers
        let m_new_hole: &Monomer = &self.monomers[state.m_h];
        let m_new_elec: &Monomer = &self.monomers[state.m_l];
        let m_old_hole: &OldMonomer = &other.monomers[state.m_h];
        let m_old_elec: &OldMonomer = &other.monomers[state.m_l];

        // dimension of the MO overlap matrix
        let dim: usize = m_new_hole.n_orbs + m_new_elec.n_orbs;
        // get the MO coefficients of the monomers
        let orbs_j = m_new_hole.properties.orbs().unwrap();
        let orbs_i = m_new_elec.properties.orbs().unwrap();
        let orbs_l = m_old_hole.orbs.view();
        let orbs_k = m_old_elec.orbs.view();

        // prepare the MO overlap matrix
        let mut s_mo: Array2<f64> = Array2::zeros([dim, dim]);
        // fill the MO overlap matrix
        s_mo.slice_mut(s![..m_new_hole.n_orbs, ..m_old_hole.n_orbs])
            .assign(
                &orbs_j.t().dot(
                    &overlap
                        .slice(s![m_new_hole.slice.orb, m_new_hole.slice.orb])
                        .dot(&orbs_l),
                ),
            );
        s_mo.slice_mut(s![m_new_hole.n_orbs.., m_old_hole.n_orbs..])
            .assign(
                &orbs_i.t().dot(
                    &overlap
                        .slice(s![m_new_elec.slice.orb, m_new_elec.slice.orb])
                        .dot(&orbs_k),
                ),
            );
        s_mo.slice_mut(s![..m_new_hole.n_orbs, m_old_hole.n_orbs..])
            .assign(
                &orbs_j.t().dot(
                    &overlap
                        .slice(s![m_new_hole.slice.orb, m_new_elec.slice.orb])
                        .dot(&orbs_k),
                ),
            );
        s_mo.slice_mut(s![m_new_hole.n_orbs.., ..m_old_hole.n_orbs])
            .assign(
                &orbs_i.t().dot(
                    &overlap
                        .slice(s![m_new_elec.slice.orb, m_new_hole.slice.orb])
                        .dot(&orbs_l),
                ),
            );

        // occupied and virtual indices
        let nocc_j: usize = m_new_hole.properties.occ_indices().unwrap().len();
        let nocc_i: usize = m_new_elec.properties.occ_indices().unwrap().len();
        let nvirt_j: usize = m_new_hole.properties.virt_indices().unwrap().len();
        let nvirt_i: usize = m_new_elec.properties.virt_indices().unwrap().len();
        // number of orbitals
        let norb_j: usize = nocc_j + nvirt_j;
        let nocc: usize = nocc_i + nocc_j;
        // slice the MO overlap matrix
        let mut s_mo_occ: Array2<f64> = Array2::zeros((nocc, nocc));
        s_mo_occ
            .slice_mut(s![..nocc_j, ..nocc_j])
            .assign(&s_mo.slice(s![..nocc_j, ..nocc_j]));
        s_mo_occ
            .slice_mut(s![nocc_j.., nocc_j..])
            .assign(&s_mo.slice(s![norb_j..norb_j + nocc_i, norb_j..norb_j + nocc_i]));
        s_mo_occ
            .slice_mut(s![..nocc_j, nocc_j..])
            .assign(&s_mo.slice(s![..nocc_j, norb_j..norb_j + nocc_i]));
        s_mo_occ
            .slice_mut(s![nocc_j.., ..nocc_j])
            .assign(&s_mo.slice(s![norb_j..norb_j + nocc_i, ..nocc_j]));

        // get the CI coefficients of the CT states
        let cis: ArrayView2<f64> = state.eigenvectors.view();

        self.ci_overlap_gs_ct(
            s_mo.view(),
            s_mo_occ.view(),
            cis,
            nocc_j,
            nvirt_j,
            nocc_i,
            nvirt_i,
            gs_old,
        )
    }

    pub fn ci_overlap_gs_ct(
        &self,
        s_mo: ArrayView2<f64>,
        s_mo_occ: ArrayView2<f64>,
        cis: ArrayView2<f64>,
        nocc_hole: usize,
        nvirt_hole: usize,
        nocc_elec: usize,
        nvirt_elec: usize,
        gs_old: bool,
    ) -> f64 {
        // Excitations i->a with coefficients |C_ia| < threshold will be neglected
        let threshold: f64 = 0.0005;

        // calc the ground state part of the ci overlap
        let det_ij: f64 = s_mo_occ.det().unwrap();

        // get indics for iterating over the overlap matrix
        let norb_hole: usize = nocc_hole + nvirt_hole;
        let norb_elec: usize = nocc_elec + nvirt_elec;
        let norb: usize = norb_hole + norb_elec;
        let nocc: usize = nocc_hole + nocc_elec;
        let start_virt_elec = nocc + nvirt_hole;

        // scalar coupling value
        let mut s_ci: f64 = 0.0;

        for i in 0..nocc_hole {
            for (a_idx, a) in (start_virt_elec..norb).into_iter().enumerate() {
                // slice the CI coefficients of the diabatic state J at the indicies i and a
                let coeff = cis[[i, a_idx]];

                if coeff.abs() > threshold {
                    if gs_old {
                        let mut s_aj: Array2<f64> = s_mo_occ.to_owned();
                        // occupied orbitals in the configuration state function |Psi_ia>
                        // overlap <1,...,a,...|1,...,j,...>
                        s_aj.slice_mut(s![i, ..nocc_hole])
                            .assign(&s_mo.slice(s![a, ..nocc_hole]));
                        s_aj.slice_mut(s![i, nocc_hole..nocc])
                            .assign(&s_mo.slice(s![a, norb_hole..norb_hole + nocc_elec]));
                        let det_aj: f64 = s_aj.det().unwrap();

                        s_ci += coeff * 2.0_f64.sqrt() * det_ij * det_aj;
                    } else {
                        let mut s_ia: Array2<f64> = s_mo_occ.to_owned();
                        // occupied orbitals in the configuration state function |Psi_ia>
                        // overlap <1,...,a,...|1,...,j,...>
                        s_ia.slice_mut(s![..nocc_hole, i])
                            .assign(&s_mo.slice(s![..nocc_hole, a]));
                        s_ia.slice_mut(s![nocc_hole..nocc, i])
                            .assign(&s_mo.slice(s![norb_hole..norb_hole + nocc_elec, a]));
                        let det_ia: f64 = s_ia.det().unwrap();

                        s_ci += coeff * 2.0_f64.sqrt() * det_ij * det_ia;
                    }
                }
            }
        }

        s_ci
    }

    pub fn scalar_coupling_diabatic_states(
        &self,
        other: &OldSupersystem,
        lhs: &ReducedBasisState,
        rhs: &ReducedBasisState,
        overlap: ArrayView2<f64>,
    ) -> f64 {
        match (lhs, rhs) {
            // coupling between two LE states.
            (ReducedBasisState::LE(ref a), ReducedBasisState::LE(ref b)) => {
                if a.monomer_index == b.monomer_index {
                    let val = self.scalar_coupling_le_le(other, a, b, overlap);

                    val
                } else {
                    0.0
                }
            }
            // coupling between LE and CT state.
            (ReducedBasisState::LE(ref a), ReducedBasisState::CT(ref b)) => 0.0,
            // coupling between CT and LE state.
            (ReducedBasisState::CT(ref a), ReducedBasisState::LE(ref b)) => 0.0,
            // coupling between CT and CT
            (ReducedBasisState::CT(ref a), ReducedBasisState::CT(ref b)) => {
                if a.m_h == b.m_h && a.m_l == b.m_l {
                    let sci = self.scalar_coupling_ct_ct(other, a, b, overlap);
                    sci
                } else {
                    0.0
                }
            }
        }
    }

    pub fn scalar_coupling_le_le(
        &self,
        other: &OldSupersystem,
        state_new: &ReducedLE,
        state_old: &ReducedLE,
        overlap: ArrayView2<f64>,
    ) -> f64 {
        // get the monomers of the LE state of the new and old Supersystem
        let m_new: &Monomer = &self.monomers[state_new.monomer_index];
        let m_old: &OldMonomer = &other.monomers[state_old.monomer_index];

        // slice the overlap matrix to get the AO overlap of the LE
        let s_ao: ArrayView2<f64> = overlap.slice(s![m_new.slice.orb, m_old.slice.orb]);
        let s_ao: ArrayView2<f64> = s_ao.t();
        // get the MO coefficients of the old and the new geometry
        let orbs_old: ArrayView2<f64> = m_old.orbs.view();
        let orbs_new: ArrayView2<f64> = m_new.properties.orbs().unwrap();
        // transform the AO overlap to the MO basis
        let s_mo: Array2<f64> = orbs_old.t().dot(&s_ao.dot(&orbs_new));

        // get the number of occupied MOs
        let nocc: usize = m_new.properties.occ_indices().unwrap().len();
        // get the occupied MO overlap matrix
        let s_mo_occ: ArrayView2<f64> = s_mo.slice(s![..nocc, ..nocc]);

        // get the CI matrices of the LE states
        let ci_new: ArrayView2<f64> = m_new.properties.tdm(state_new.state_index).unwrap();
        let ci_old: ArrayView1<f64> = m_old.tdm.slice(s![.., state_old.state_index]);
        let n_occ: usize = m_old.occ_indices.len();
        let n_nvirt: usize = m_old.virt_indices.len();
        let ci_old: Array2<f64> = ci_old.to_owned().into_shape([n_occ, n_nvirt]).unwrap();

        // calculate the ci_overlap
        self.ci_overlap_same_fragment(s_mo.view(), s_mo_occ, ci_new, ci_old.view())
    }

    pub fn ci_overlap_same_fragment(
        &self,
        s_mo: ArrayView2<f64>,
        s_mo_occ: ArrayView2<f64>,
        ci_new: ArrayView2<f64>,
        ci_old: ArrayView2<f64>,
    ) -> f64 {
        // Excitations i->a with coefficients |C_ia| < threshold will be neglected
        let threshold: f64 = 0.0005;
        // get nocc and nvirt
        let nocc: usize = ci_new.dim().0;
        let nvirt: usize = ci_new.dim().1;

        // calc the ground state part of the ci overlap
        let det_ij: f64 = s_mo_occ.det().unwrap();
        let norb: usize = nocc + nvirt;

        // scalar coupling value
        let mut s_ci: f64 = 0.0;

        for i in 0..nocc {
            for (a_idx, a) in (nocc..norb).into_iter().enumerate() {
                // slice the CI coefficients of the diabatic state J at the indicies i and a
                let coeff_i = ci_old[[i, a_idx]];

                if coeff_i.powi(2) > threshold {
                    let mut s_aj: Array2<f64> = s_mo_occ.to_owned();
                    // occupied orbitals in the configuration state function |Psi_ia>
                    // overlap <1,...,a,...|1,...,j,...>
                    s_aj.slice_mut(s![i, ..nocc])
                        .assign(&s_mo.slice(s![a, ..nocc]));
                    let det_aj: f64 = s_aj.det().unwrap();

                    for j in 0..nocc {
                        for (b_idx, b) in (nocc..norb).into_iter().enumerate() {
                            let coeff_j = ci_new[[j, b_idx]];

                            if coeff_j.powi(2) > threshold {
                                let mut s_ab: Array2<f64> = s_mo_occ.to_owned();
                                // select part of overlap matrix for orbitals
                                // in |Psi_ia> and |Psi_jb>
                                // <1,...,a,...|1,...,b,...>
                                s_ab.slice_mut(s![i, ..]).assign(&s_mo.slice(s![a, ..nocc]));
                                s_ab.slice_mut(s![.., j]).assign(&s_mo.slice(s![..nocc, b]));
                                s_ab[[i, j]] = s_mo[[a, b]];
                                let det_ab: f64 = s_ab.det().unwrap();

                                let mut s_ib: Array2<f64> = s_mo_occ.to_owned();
                                // <1,...,i,...|1,...,b,...>
                                s_ib.slice_mut(s![.., j]).assign(&s_mo.slice(s![..nocc, b]));
                                let det_ib: f64 = s_ib.det().unwrap();

                                // calculate the ci overlap
                                s_ci += coeff_j * coeff_i * (det_ab * det_ij + det_aj * det_ib);
                            }
                        }
                    }
                }
            }
        }

        s_ci
    }

    pub fn scalar_coupling_ct_ct(
        &self,
        other: &OldSupersystem,
        state_new: &ChargeTransferPair,
        state_old: &ChargeTransferPair,
        overlap: ArrayView2<f64>,
    ) -> f64 {
        // get the references to the monomers
        let m_new_hole: &Monomer = &self.monomers[state_new.m_h];
        let m_new_elec: &Monomer = &self.monomers[state_new.m_l];
        let m_old_hole: &OldMonomer = &other.monomers[state_old.m_h];
        let m_old_elec: &OldMonomer = &other.monomers[state_old.m_l];

        // dimension of the MO overlap matrix
        let dim: usize = m_new_hole.n_orbs + m_new_elec.n_orbs;
        // get the MO coefficients of the monomers
        let orbs_j = m_new_hole.properties.orbs().unwrap();
        let orbs_i = m_new_elec.properties.orbs().unwrap();
        let orbs_l = m_old_hole.orbs.view();
        let orbs_k = m_old_elec.orbs.view();

        // prepare the MO overlap matrix
        let mut s_mo: Array2<f64> = Array2::zeros([dim, dim]);
        // fill the MO overlap matrix
        s_mo.slice_mut(s![..m_new_hole.n_orbs, ..m_old_hole.n_orbs])
            .assign(
                &orbs_j.t().dot(
                    &overlap
                        .slice(s![m_new_hole.slice.orb, m_new_hole.slice.orb])
                        .dot(&orbs_l),
                ),
            );
        s_mo.slice_mut(s![m_new_hole.n_orbs.., m_old_hole.n_orbs..])
            .assign(
                &orbs_i.t().dot(
                    &overlap
                        .slice(s![m_new_elec.slice.orb, m_new_elec.slice.orb])
                        .dot(&orbs_k),
                ),
            );
        s_mo.slice_mut(s![..m_new_hole.n_orbs, m_old_hole.n_orbs..])
            .assign(
                &orbs_j.t().dot(
                    &overlap
                        .slice(s![m_new_hole.slice.orb, m_new_elec.slice.orb])
                        .dot(&orbs_k),
                ),
            );
        s_mo.slice_mut(s![m_new_hole.n_orbs.., ..m_old_hole.n_orbs])
            .assign(
                &orbs_i.t().dot(
                    &overlap
                        .slice(s![m_new_elec.slice.orb, m_new_hole.slice.orb])
                        .dot(&orbs_l),
                ),
            );

        // occupied and virtual indices
        let nocc_j: usize = m_new_hole.properties.occ_indices().unwrap().len();
        let nocc_i: usize = m_new_elec.properties.occ_indices().unwrap().len();
        let nvirt_j: usize = m_new_hole.properties.virt_indices().unwrap().len();
        let nvirt_i: usize = m_new_elec.properties.virt_indices().unwrap().len();
        // number of orbitals
        let norb_i: usize = nocc_i + nvirt_i;
        let norb_j: usize = nocc_j + nvirt_j;
        let nocc: usize = nocc_i + nocc_j;
        let nvirt: usize = nvirt_i + nvirt_j;

        // slice the MO overlap matrix
        let mut s_mo_occ: Array2<f64> = Array2::zeros((nocc, nocc));
        s_mo_occ
            .slice_mut(s![..nocc_j, ..nocc_j])
            .assign(&s_mo.slice(s![..nocc_j, ..nocc_j]));
        s_mo_occ
            .slice_mut(s![nocc_j.., nocc_j..])
            .assign(&s_mo.slice(s![norb_j..norb_j + nocc_i, norb_j..norb_j + nocc_i]));
        s_mo_occ
            .slice_mut(s![..nocc_j, nocc_j..])
            .assign(&s_mo.slice(s![..nocc_j, norb_j..norb_j + nocc_i]));
        s_mo_occ
            .slice_mut(s![nocc_j.., ..nocc_j])
            .assign(&s_mo.slice(s![norb_j..norb_j + nocc_i, ..nocc_j]));

        // get the CI coefficients of the CT states
        let cis_new: ArrayView2<f64> = state_new.eigenvectors.view();
        let cis_old: ArrayView2<f64> = state_old.eigenvectors.view();

        self.ci_overlap_ct_ct(
            s_mo.view(),
            s_mo_occ.view(),
            cis_new,
            cis_old,
            nocc_j,
            nvirt_j,
            nocc_i,
            nvirt_i,
        )
    }

    pub fn ci_overlap_ct_ct(
        &self,
        s_mo: ArrayView2<f64>,
        s_mo_occ: ArrayView2<f64>,
        ci_new: ArrayView2<f64>,
        ci_old: ArrayView2<f64>,
        nocc_hole: usize,
        nvirt_hole: usize,
        nocc_elec: usize,
        nvirt_elec: usize,
    ) -> f64 {
        // Excitations i->a with coefficients |C_ia| < threshold will be neglected
        let threshold: f64 = 0.0005;

        // calc the ground state part of the ci overlap
        let det_ij: f64 = s_mo_occ.det().unwrap();

        // get indics for iterating over the overlap matrix
        let norb_hole: usize = nocc_hole + nvirt_hole;
        let norb_elec: usize = nocc_elec + nvirt_elec;
        let norb: usize = norb_hole + norb_elec;
        let nocc: usize = nocc_hole + nocc_elec;
        let start_virt_elec = nocc + nvirt_hole;

        // scalar coupling value
        let mut s_ci: f64 = 0.0;

        for i in 0..nocc_hole {
            for (a_idx, a) in (start_virt_elec..norb).into_iter().enumerate() {
                // slice the CI coefficients of the diabatic state J at the indicies i and a
                let coeff_i = ci_new[[i, a_idx]];

                if coeff_i.powi(2) > threshold {
                    let mut s_aj: Array2<f64> = s_mo_occ.to_owned();
                    // occupied orbitals in the configuration state function |Psi_ia>
                    // overlap <1,...,a,...|1,...,j,...>
                    s_aj.slice_mut(s![i, ..nocc_hole])
                        .assign(&s_mo.slice(s![a, ..nocc_hole]));
                    s_aj.slice_mut(s![i, nocc_hole..nocc])
                        .assign(&s_mo.slice(s![a, norb_hole..norb_hole + nocc_elec]));
                    let det_aj: f64 = s_aj.det().unwrap();

                    for j in 0..nocc_hole {
                        for (b_idx, b) in (start_virt_elec..norb).into_iter().enumerate() {
                            let coeff_j = ci_old[[j, b_idx]];

                            if coeff_j.powi(2) > threshold {
                                let mut s_ab: Array2<f64> = s_mo_occ.to_owned();
                                // select part of overlap matrix for orbitals
                                // in |Psi_ia> and |Psi_jb>
                                // <1,...,a,...|1,...,b,...>
                                s_ab.slice_mut(s![i, ..nocc_hole])
                                    .assign(&s_mo.slice(s![a, ..nocc_hole]));
                                s_ab.slice_mut(s![i, nocc_hole..nocc])
                                    .assign(&s_mo.slice(s![a, norb_hole..norb_hole + nocc_elec]));

                                s_ab.slice_mut(s![..nocc_hole, j])
                                    .assign(&s_mo.slice(s![..nocc_hole, b]));
                                s_ab.slice_mut(s![nocc_hole..nocc, j])
                                    .assign(&s_mo.slice(s![norb_hole..norb_hole + nocc_elec, b]));
                                s_ab[[i, j]] = s_mo[[a, b]];
                                let det_ab: f64 = s_ab.det().unwrap();

                                let mut s_ib: Array2<f64> = s_mo_occ.to_owned();
                                // <1,...,i,...|1,...,b,...>
                                s_ib.slice_mut(s![..nocc_hole, j])
                                    .assign(&s_mo.slice(s![..nocc_hole, b]));
                                s_ib.slice_mut(s![nocc_hole..nocc, j])
                                    .assign(&s_mo.slice(s![norb_hole..norb_hole + nocc_elec, b]));
                                let det_ib: f64 = s_ib.det().unwrap();

                                // calculate the ci overlap
                                s_ci += coeff_j * coeff_i * (det_ab * det_ij + det_aj * det_ib);
                            }
                        }
                    }
                }
            }
        }

        s_ci
    }

    pub fn supersystem_overlap_between_timesteps(
        &self,
        other: &OldSupersystem,
        skt: &SlaterKoster,
    ) -> Array2<f64> {
        // get the atoms of the old Supersystem
        let old_atoms = &other.atoms;
        // get the atoms of the Supersystem at the new geometry
        let atoms = &self.atoms;
        // get the number of the orbitals of the Supersystem
        let n_orbs: usize = self.properties.n_occ().unwrap() + self.properties.n_virt().unwrap();
        // empty matrix for the overlap
        let mut s: Array2<f64> = Array2::zeros([n_orbs, n_orbs]);

        let mut mu: usize = 0;
        // iterate over the atoms of the system
        for (idx_i, atom_i) in atoms.iter().enumerate() {
            // iterate over the orbitals on atom I
            for orbi in atom_i.valorbs.iter() {
                // iterate over the atoms of the old geometry
                let mut nu: usize = 0;
                for (j, atom_j) in old_atoms.iter().enumerate() {
                    // iterate over the orbitals on atom J
                    for orbj in atom_j.valorbs.iter() {
                        if (atom_i - atom_j).norm() < defaults::PROXIMITY_CUTOFF {
                            if atom_i <= atom_j {
                                let (r, x, y, z): (f64, f64, f64, f64) =
                                    directional_cosines(&atom_i.xyz, &atom_j.xyz);
                                s[[mu, nu]] = slako_transformation(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &skt.get(atom_i.kind, atom_j.kind).s_spline,
                                    orbi.l,
                                    orbi.m,
                                    orbj.l,
                                    orbj.m,
                                );
                            } else {
                                let (r, x, y, z): (f64, f64, f64, f64) =
                                    directional_cosines(&atom_j.xyz, &atom_i.xyz);
                                s[[mu, nu]] = slako_transformation(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &skt.get(atom_j.kind, atom_i.kind).s_spline,
                                    orbj.l,
                                    orbj.m,
                                    orbi.l,
                                    orbi.m,
                                );
                            }
                        }
                        nu += 1;
                    }
                }
                mu += 1;
            }
        }
        s
    }
}
