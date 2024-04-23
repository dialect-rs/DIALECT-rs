use crate::excited_states::tda::moments::{mulliken_dipoles, oscillator_strength};
use crate::excited_states::{
    orbe_differences, trans_oo_restricted, trans_vv_restricted, ProductCache,
};
use crate::fmo::helpers::get_pair_slice;
use crate::fmo::{ChargeTransferPreparation, PairType};
use crate::initialization::Atom;
use crate::io::Configuration;
use crate::{initial_subspace, Davidson};
use ndarray::prelude::*;

impl ChargeTransferPreparation<'_> {
    // Prepare the TDA-LC-TD-DFTB calculation.
    pub fn prepare_ct_tda(
        &mut self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        s_full: ArrayView2<f64>,
        atoms: &[Atom],
        config: &Configuration,
    ) {
        // indices of the occupied and virtual orbitals of the CT state
        let occ_indices: &[usize] = self.m_h.properties.occ_indices().unwrap();
        let virt_indices: &[usize] = self.m_l.properties.virt_indices().unwrap();

        // number of atoms
        let natoms_h: usize = self.m_h.n_atoms;
        let natoms_l: usize = self.m_l.n_atoms;
        let n_atoms: usize = natoms_h + natoms_l;

        // set the gamma matrix
        let mut gamma: Array2<f64> = Array2::zeros([n_atoms, n_atoms]);
        gamma
            .slice_mut(s![..natoms_h, ..natoms_h])
            .assign(&self.m_h.properties.gamma().unwrap());
        gamma
            .slice_mut(s![natoms_h.., natoms_h..])
            .assign(&self.m_l.properties.gamma().unwrap());
        let gamma_ab: ArrayView2<f64> = g0.slice(s![self.m_h.slice.atom, self.m_l.slice.atom]);
        gamma
            .slice_mut(s![..natoms_h, natoms_h..])
            .assign(&gamma_ab);
        gamma
            .slice_mut(s![natoms_h.., ..natoms_h])
            .assign(&gamma_ab.t());

        // get the overlap matrix
        let s: ArrayView2<f64> = s_full.slice(s![self.m_h.slice.orb, self.m_l.slice.orb]);
        self.properties.set_s(s.to_owned());
        self.properties.set_gamma(gamma);

        // set the gamma lr matrix
        let mut gamma_lr_full: Array2<f64> = Array2::zeros([n_atoms, n_atoms]);
        gamma_lr_full
            .slice_mut(s![..natoms_h, ..natoms_h])
            .assign(&self.m_h.properties.gamma_lr().unwrap());
        gamma_lr_full
            .slice_mut(s![natoms_h.., natoms_h..])
            .assign(&self.m_l.properties.gamma_lr().unwrap());
        let gamma_ab: ArrayView2<f64> = g0_lr.slice(s![self.m_h.slice.atom, self.m_l.slice.atom]);
        gamma_lr_full
            .slice_mut(s![..natoms_h, natoms_h..])
            .assign(&gamma_ab);
        gamma_lr_full
            .slice_mut(s![natoms_h.., ..natoms_h])
            .assign(&gamma_ab.t());
        let gamma_lr: ArrayView2<f64> = g0_lr.slice(s![self.m_h.slice.atom, self.m_l.slice.atom]);
        self.properties.set_gamma_lr(gamma_lr.to_owned());
        self.properties.set_gamma_lr_ao(gamma_lr_full);

        // The index of the HOMO (zero based).
        let homo: usize = occ_indices[occ_indices.len() - 1];

        // The index of the LUMO (zero based).
        let lumo: usize = virt_indices[0];

        let nocc: usize = occ_indices.len();
        let nvirt: usize = virt_indices.len();

        // Energies of the occupied orbitals.
        let orbe_h: ArrayView1<f64> = self.m_h.properties.orbe().unwrap();
        let mut orbe_occ: Array1<f64> = orbe_h.slice(s![0..homo + 1]).to_owned();

        // Energies of the virtual orbitals.
        let orbe_l: ArrayView1<f64> = self.m_l.properties.orbe().unwrap();
        let mut orbe_virt: Array1<f64> = orbe_l.slice(s![lumo..]).to_owned();

        if config.tddftb.restrict_active_orbitals {
            let dim_o: usize = (nocc as f64 * config.tddftb.active_orbital_threshold) as usize;
            let dim_v: usize = (nvirt as f64 * config.tddftb.active_orbital_threshold) as usize;

            orbe_occ = orbe_occ.slice(s![homo + 1 - dim_o..homo + 1]).to_owned();
            orbe_virt = orbe_virt.slice(s![..dim_v]).to_owned();
        }
        // Energy differences between virtual and occupied orbitals.
        let omega: Array1<f64> = orbe_differences(orbe_occ.view(), orbe_virt.view());

        // Energy differences are stored
        self.properties.set_omega(omega);
        self.properties.set_homo(homo);
        self.properties.set_lumo(lumo);

        // get the atoms of the fragments
        let atoms_h: &[Atom] = &atoms[self.m_h.slice.atom_as_range()];
        let atoms_l: &[Atom] = &atoms[self.m_l.slice.atom_as_range()];

        // calculate the transition charges q_ov
        let q_ov: Array2<f64> = self.calculate_q_ov(s, atoms_h, atoms_l, config);
        // store the transition charges
        self.properties.set_q_ov(q_ov);
        if config.tddftb.restrict_active_orbitals {
            self.properties
                .set_q_oo(self.m_h.properties.q_oo_restricted().unwrap().to_owned());
            self.properties
                .set_q_vv(self.m_h.properties.q_vv_restricted().unwrap().to_owned());
        } else {
            self.properties
                .set_q_oo(self.m_h.properties.q_oo().unwrap().to_owned());
            self.properties
                .set_q_vv(self.m_l.properties.q_vv().unwrap().to_owned());
        }
        self.properties.set_occ_indices(occ_indices.to_vec());
        self.properties.set_virt_indices(virt_indices.to_vec());
    }

    // Calculate the transition charges between the orbitals of both monomers of the CT state
    pub fn calculate_q_ov(
        &self,
        s: ArrayView2<f64>,
        atoms_h: &[Atom],
        atoms_l: &[Atom],
        config: &Configuration,
    ) -> Array2<f64> {
        let homo = self.properties.homo().unwrap();
        let mut occs = self
            .m_h
            .properties
            .orbs_slice(0, Some(homo + 1))
            .unwrap()
            .to_owned();
        let lumo = self.properties.lumo().unwrap();
        let mut virts =
            self.m_l
                .properties
                .orbs_slice(lumo, None)
                .unwrap()
                .to_owned();

        if config.tddftb.restrict_active_orbitals {
            let nocc: usize = occs.dim().1;
            let nvirt: usize = virts.dim().1;

            let dim_o: usize = (nocc as f64 * config.tddftb.active_orbital_threshold) as usize;
            let dim_v: usize = (nvirt as f64 * config.tddftb.active_orbital_threshold) as usize;
            let diff_occ = nocc - dim_o;

            occs = occs.slice(s![.., diff_occ..]).to_owned();
            virts = virts.slice(s![.., ..dim_v]).to_owned();
        }

        // Matrix product of overlap matrix with the orbitals on L.
        let s_c_l: Array2<f64> = s.dot(&virts);
        // Matrix product of overlap matrix with the orbitals on H.
        let s_c_h: Array2<f64> = s.t().dot(&occs);
        // Number of molecular orbitals on monomer I.
        let dim_h: usize = occs.ncols();
        // Number of molecular orbitals on monomer J.
        let dim_l: usize = virts.ncols();
        // get the number of atoms
        let natoms_h: usize = atoms_h.len();
        let natoms_l: usize = atoms_l.len();
        let n_atoms: usize = natoms_h + natoms_l;
        // The transition charges between the two sets of MOs  are initialized.
        let mut q_trans: Array3<f64> = Array3::zeros([n_atoms, dim_h, dim_l]);

        let mut mu: usize = 0;
        for (atom_h, mut q_n) in atoms_h.iter().zip(
            q_trans
                .slice_mut(s![0..natoms_h, .., ..])
                .axis_iter_mut(Axis(0)),
        ) {
            for _ in 0..atom_h.n_orbs {
                for (orb_h, mut q_h) in occs.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                    for (sc, q) in s_c_l.row(mu).iter().zip(q_h.iter_mut()) {
                        *q += orb_h * sc;
                    }
                }
                mu += 1;
            }
        }
        mu = 0;
        for (atom_l, mut q_n) in atoms_l.iter().zip(
            q_trans
                .slice_mut(s![natoms_h.., .., ..])
                .axis_iter_mut(Axis(0)),
        ) {
            for _ in 0..atom_l.n_orbs {
                for (sc, mut q_l) in s_c_h.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                    for (orb_l, q) in virts.row(mu).iter().zip(q_l.iter_mut()) {
                        *q += orb_l * sc;
                    }
                }
                mu += 1;
            }
        }
        q_trans = 0.5 * q_trans;
        q_trans.into_shape([n_atoms, dim_h * dim_l]).unwrap()
    }

    // Do the TDA-LC-TD-DFTB calculation
    pub fn run_ct_tda(
        &mut self,
        atoms: &[Atom],
        n_roots: usize,
        max_iter: usize,
        tolerance: f64,
        subspace_multiplier: usize,
        config: &Configuration,
    ) {
        // Set an empty product cache.
        self.properties.set_cache(ProductCache::new());

        // Reference to the energy differences between virtuals and occupied orbitals.
        let omega: ArrayView1<f64> = self.properties.omega().unwrap();

        // The initial guess for the subspace is created.
        let guess: Array2<f64> = initial_subspace(omega.view(), n_roots);

        // Davidson iteration.
        let davidson: Davidson = Davidson::new(
            self,
            guess,
            n_roots,
            tolerance,
            max_iter,
            false,
            subspace_multiplier,
        )
        .unwrap();

        // check if the tda routine yields realistic energies
        let energy_vector = davidson.eigenvalues.clone().to_vec();
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
        let q_trans: Array2<f64> = q_ov.dot(&davidson.eigenvectors);

        let pair_atoms: Vec<Atom> = get_pair_slice(
            atoms,
            self.m_h.slice.atom_as_range(),
            self.m_l.slice.atom_as_range(),
        );

        // The Mulliken transition dipole moments are computed.
        let tr_dipoles: Array2<f64> = mulliken_dipoles(q_trans.view(), &pair_atoms);

        // The oscillator strengths are computed.
        let f: Array1<f64> = oscillator_strength(davidson.eigenvalues.view(), tr_dipoles.view());

        if config.tddftb.restrict_active_orbitals {
            // indices of the occupied and virtual orbitals of the CT state
            let occ_indices: &[usize] = self.m_h.properties.occ_indices().unwrap();
            let virt_indices: &[usize] = self.m_l.properties.virt_indices().unwrap();
            let nocc_full = occ_indices.len();
            let nvirt_full = virt_indices.len();

            let n_occ = (nocc_full as f64 * config.tddftb.active_orbital_threshold) as usize;
            let n_virt = (nvirt_full as f64 * config.tddftb.active_orbital_threshold) as usize;

            let mut tdm: Array3<f64> = davidson
                .eigenvectors
                .clone()
                .into_shape([n_occ, n_virt, f.len()])
                .unwrap();
            let mut tdm_new: Array3<f64> = Array3::zeros((nocc_full, nvirt_full, f.len()));
            tdm_new
                .slice_mut(s![nocc_full - n_occ.., ..n_virt, ..])
                .assign(&tdm);
            tdm = tdm_new;

            // transform tdm back to 2d
            let eigvecs: Array2<f64> = tdm.into_shape([nocc_full * nvirt_full, f.len()]).unwrap();
            self.properties.set_ci_coefficients(eigvecs);

            // calculate the transition charges q_ov
            let mut config = config.clone();
            config.tddftb.restrict_active_orbitals = false;
            // get the atoms of the fragments
            let atoms_h: &[Atom] = &atoms[self.m_h.slice.atom_as_range()];
            let atoms_l: &[Atom] = &atoms[self.m_l.slice.atom_as_range()];
            let s = self.properties.take_s().unwrap();

            let q_ov: Array2<f64> = self.calculate_q_ov(s.view(), atoms_h, atoms_l, &config);
            // store the transition charges
            self.properties.set_q_ov(q_ov);
        } else {
            self.properties.set_ci_coefficients(davidson.eigenvectors);
        }

        // The eigenvalues are the excitation energies and the eigenvectors are the CI coefficients.
        self.properties.set_ci_eigenvalues(davidson.eigenvalues);
        self.properties.set_q_trans(q_trans);
        self.properties.set_tr_dipoles(tr_dipoles);
        self.properties.set_oscillator_strengths(f);
    }
}
