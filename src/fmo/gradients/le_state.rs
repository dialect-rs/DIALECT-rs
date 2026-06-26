use crate::defaults::PROXIMITY_CUTOFF;
use crate::excited_states::trans_charges;
use crate::fmo::{BasisState, LocallyExcited, Monomer, SuperSystem};
use crate::gradients::helpers::{
    compute_flr_dg_contributions, compute_flr_s_coefficients, compute_fv_coefficients_onthefly,
    compute_fv_dg_contributions, f_lr, f_v, h_a_nolr, tda_zvector_no_lc, zvector_lc,
    zvector_lc_optimized, Hav, Hplus, HplusType,
};
use crate::initialization::Atom;
use crate::param::slako_transformations::{
    directional_cosines, slako_transformation_gradients_fast, SplineCache,
};
use crate::scc::gamma_approximation::gamma_ao_wise_from_gamma_atomwise;
// use crate::scc::h0_and_s::h0_and_s_gradients;
use crate::utils::ToOwnedF;
use ndarray::{s, Array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use ndarray_linalg::{into_col, into_row, IntoTriangular, UPLO};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

impl SuperSystem<'_> {
    pub fn exciton_le_energy(&mut self, monomer_index: usize, state: usize) -> f64 {
        let lcmo_config = self.config.fmo_lc_tddftb.clone();
        let threshold_le: f64 = lcmo_config.active_space_threshold_le;
        // Number of LE states per monomer.
        let n_le: usize = lcmo_config.n_le;

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let n_roots: usize = n_le + 2;

        // get the monomer
        let mol = &mut self.monomers[monomer_index];
        // Compute the excited states for the monomer.
        mol.prepare_tda(&atoms[mol.slice.atom_as_range()], &self.config);
        mol.run_tda(
            &atoms[mol.slice.atom_as_range()],
            n_roots,
            self.config.excited.davidson_iterations,
            self.config.excited.davidson_convergence,
            self.config.excited.davidson_subspace_multiplier,
            false,
            &self.config,
        );

        // switch to immutable borrow for the monomer
        let mol = &self.monomers[monomer_index];

        // Calculate transition charges
        let homo: usize = mol.properties.homo().unwrap();
        let q_ov: ArrayView2<f64> = mol.properties.q_ov().unwrap();

        // Create the LE state
        let tdm: ArrayView1<f64> = mol.properties.ci_coefficient(state).unwrap();
        let tdm_dim2: ArrayView2<f64> = mol.properties.tdm(state).unwrap();

        // determine the relevant orbital indices
        let mut occ_indices: Vec<usize> = Vec::new();
        let mut virt_indices: Vec<usize> = Vec::new();
        for (idx_i, val_i) in tdm_dim2.outer_iter().enumerate() {
            for (idx_j, val_j) in val_i.iter().enumerate() {
                let abs_c_sqr: f64 = val_j.abs().powi(2);
                if abs_c_sqr > threshold_le {
                    if !occ_indices.contains(&idx_i) {
                        occ_indices.push(idx_i);
                    }
                    if !virt_indices.contains(&idx_j) {
                        virt_indices.push(idx_j);
                    }
                }
            }
        }

        let le_state: BasisState = BasisState::LE(LocallyExcited {
            monomer: mol,
            n: state,
            atoms: &atoms[mol.slice.atom_as_range()],
            q_trans: q_ov.dot(&tdm),
            occs: mol.properties.orbs_slice(0, Some(homo + 1)).unwrap(),
            virts: mol.properties.orbs_slice(homo + 1, None).unwrap(),
            tdm,
            tr_dipole: mol.properties.tr_dipole(state).unwrap(),
            occ_indices,
            virt_indices,
        });

        let val: f64 = self.exciton_coupling(&le_state, &le_state);
        val
    }

    pub fn exciton_le_gradient(&mut self, monomer_index: usize, state: usize) -> Array1<f64> {
        let lcmo_config = self.config.fmo_lc_tddftb.clone();
        // Number of LE states per monomer.
        let n_le: usize = lcmo_config.n_le;

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let n_roots: usize = n_le + 2;

        // get the monomer
        let mol = &mut self.monomers[monomer_index];
        // Compute the excited states for the monomer.
        mol.prepare_tda(&atoms[mol.slice.atom_as_range()], &self.config);
        mol.run_tda(
            &atoms[mol.slice.atom_as_range()],
            n_roots,
            self.config.excited.davidson_iterations,
            self.config.excited.davidson_convergence,
            self.config.excited.davidson_subspace_multiplier,
            false,
            &self.config,
        );

        // calculate the gradient
        mol.prepare_excited_gradient(&atoms[mol.slice.atom_as_range()]);

        mol.tda_gradient_lc_accumulation(&atoms[mol.slice.atom_as_range()], state)
    }
}

impl Monomer<'_> {
    pub fn prepare_excited_gradient(&mut self, atoms: &[Atom]) {
        // check if occ and virt indices exist
        let mut occ_indices: Vec<usize> = Vec::new();
        let mut virt_indices: Vec<usize> = Vec::new();
        if (!self.properties.contains_key("occ_indices"))
            || (self.properties.contains_key("virt_indices"))
        {
            // calculate the number of electrons
            let n_elec: usize = atoms.iter().fold(0, |n, atom| n + atom.n_elec);
            // get the indices of the occupied and virtual orbitals
            (0..self.n_orbs).for_each(|index| {
                if index < (n_elec / 2) {
                    occ_indices.push(index)
                } else {
                    virt_indices.push(index)
                }
            });

            self.properties.set_occ_indices(occ_indices.clone());
            self.properties.set_virt_indices(virt_indices.clone());
        } else {
            occ_indices = self.properties.occ_indices().unwrap().to_vec();
            virt_indices = self.properties.virt_indices().unwrap().to_vec();
        }
        // calculate transition charges if they don't exist
        if !self.properties.contains_key("q_ov") {
            let tmp: (Array2<f64>, Array2<f64>, Array2<f64>) = trans_charges(
                self.n_atoms,
                atoms,
                self.properties.orbs().unwrap(),
                self.properties.s().unwrap(),
                &occ_indices,
                &virt_indices,
            );

            self.properties.set_q_ov(tmp.0);
            self.properties.set_q_oo(tmp.1);
            self.properties.set_q_vv(tmp.2);
        }

        // // prepare the grad gamma_lr ao matrix
        // if self.gammafunction_lc.is_some() {
        //     // calculate the gamma gradient matrix in AO basis
        //     let (_g1_lr, g1_lr_ao): (Array3<f64>, Array3<f64>) = gamma_gradients_ao_wise(
        //         self.gammafunction_lc.as_ref().unwrap(),
        //         atoms,
        //         self.n_atoms,
        //         self.n_orbs,
        //     );
        //     self.properties.set_grad_gamma_lr_ao(g1_lr_ao);
        // }
        // prepare gamma and grad gamma AO matrix
        if self.properties.gamma_ao().is_none() {
            let g0_ao: Array2<f64> = gamma_ao_wise_from_gamma_atomwise(
                self.properties.gamma().unwrap(),
                atoms,
                self.n_orbs,
            );
            self.properties.set_gamma_ao(g0_ao);
        }
        // if self.properties.grad_gamma().is_none() {
        //     let (g1, g1_ao): (Array3<f64>, Array3<f64>) =
        //         gamma_gradients_ao_wise(&self.gammafunction, atoms, self.n_atoms, self.n_orbs);
        //     self.properties.set_grad_gamma(g1);
        //     self.properties.set_grad_gamma_ao(g1_ao);
        // }

        // // derivative of H0 and S
        // if self.properties.grad_s().is_none() || self.properties.grad_h0().is_none() {
        //     let (grad_s, grad_h0) = h0_and_s_gradients(atoms, self.n_orbs, self.slako);
        //     self.properties.set_grad_s(grad_s);
        //     self.properties.set_grad_h0(grad_h0);
        // }
    }

    fn tda_gradient_nolc(&self, state: usize) -> Array1<f64> {
        // get occ and virt indices from properties
        let occ_indices: &[usize] = self.properties.occ_indices().unwrap();
        let virt_indices: &[usize] = self.properties.virt_indices().unwrap();

        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = occ_indices.iter().map(|&occ| orbe[occ]).collect();
        let orbe_virt: Array1<f64> = virt_indices.iter().map(|&virt| orbe[virt]).collect();

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

        let n_occ: usize = orbe_occ.len();
        let n_virt: usize = orbe_virt.len();

        // excitation energy of the state
        let omega_state: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
        let n_states: usize = omega_state.len();
        let omega_state: f64 = omega_state[state];
        // take state specific values from the excitation vectors
        let x_state: ArrayView2<f64> = self.properties.ci_coefficients().unwrap();
        let x_state: Array3<f64> = x_state
            .t()
            .as_standard_layout()
            .to_owned()
            .into_shape([n_states, n_occ, n_virt])
            .unwrap();
        let x_state: ArrayView2<f64> = x_state.slice(s![state, .., ..]);

        // calculate the vectors u, v and t
        // vectors U, V and T
        let u_ab: Array2<f64> = 2.0 * x_state.t().dot(&x_state);
        let u_ij: Array2<f64> = 2.0 * x_state.dot(&x_state.t());
        let v_ab: Array2<f64> = 2.0 * ei.dot(&x_state).t().dot(&x_state);
        let v_ij: Array2<f64> = 2.0 * x_state.dot(&ea).dot(&x_state.t());
        let t_ab: Array2<f64> = x_state.t().dot(&x_state);
        let t_ij: Array2<f64> = x_state.dot(&x_state.t());

        // get the transition charges
        let qtrans_ov: ArrayView3<f64> = self
            .properties
            .q_ov()
            .unwrap()
            .into_shape((self.n_atoms, n_occ, n_virt))
            .unwrap();
        let qtrans_oo: ArrayView3<f64> = self
            .properties
            .q_oo()
            .unwrap()
            .into_shape((self.n_atoms, n_occ, n_occ))
            .unwrap();
        let qtrans_vv: ArrayView3<f64> = self
            .properties
            .q_vv()
            .unwrap()
            .into_shape((self.n_atoms, n_virt, n_virt))
            .unwrap();

        // set gamma matrix
        let g0: ArrayView2<f64> = self.properties.gamma().unwrap();

        // compute hplus of tab and tij
        let hplus_tab: Array2<f64> = h_a_nolr(g0, qtrans_oo, qtrans_vv, t_ab.view());
        let hplus_tij: Array2<f64> = h_a_nolr(g0, qtrans_oo, qtrans_oo, t_ij.view());

        // calculate q_ij
        let g_ij: Array2<f64> = hplus_tab - hplus_tij;
        let q_ij: Array2<f64> = omega_state * u_ij - v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = omega_state * u_ab + v_ab;

        // calculate q_ia
        let mut q_ia: Array2<f64> = x_state.dot(&h_a_nolr(g0, qtrans_vv, qtrans_ov, x_state).t());
        q_ia = q_ia + h_a_nolr(g0, qtrans_ov, qtrans_vv, t_ab.view());
        q_ia = q_ia - h_a_nolr(g0, qtrans_ov, qtrans_oo, t_ij.view());

        // calculate q_ai
        let q_ai: Array2<f64> = x_state
            .t()
            .dot(&h_a_nolr(g0, qtrans_oo, qtrans_ov, x_state));

        // calculate right hand side of the z-vector equation
        let r_ia: Array2<f64> = &q_ai.t() - &q_ia;

        // input for zvector routine
        let omega_input: Array2<f64> = into_col(Array::ones(orbe_occ.len()))
            .dot(&into_row(orbe_virt.clone()))
            - into_col(orbe_occ.clone()).dot(&into_row(Array::ones(orbe_virt.len())));
        let r_ia_flat: Array1<f64> = r_ia.t().to_owned_f().into_shape(n_occ * n_virt).unwrap();
        let r_matrix: Array2<f64> = r_ia_flat.into_shape((n_occ, n_virt)).unwrap();

        // calculate the z-vector
        let z_ia: Array2<f64> =
            tda_zvector_no_lc(omega_input.view(), r_matrix.view(), g0, qtrans_ov);

        // calculate w_ij
        let mut w_ij: Array2<f64> = q_ij + h_a_nolr(g0, qtrans_oo, qtrans_ov, z_ia.view());
        for i in 0..w_ij.dim().0 {
            w_ij[[i, i]] /= 2.0;
        }
        // w_ia
        let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia); //+ h_a_nolr(g0, qtrans_ov, qtrans_ov, z_ia.view());

        // w_ai
        let w_ai: Array2<f64> = &q_ai + &ei.dot(&z_ia).t();

        // w_ab
        let mut w_ab: Array2<f64> = q_ab;
        for i in 0..w_ab.dim().0 {
            w_ab[[i, i]] /= 2.0;
        }

        // build w matrix: combine w_ij, w_ia, w_ai and w_ab
        let length: usize = n_occ + n_virt;
        let mut w_matrix: Array2<f64> = Array::zeros((length, length));
        for i in 0..w_ij.dim().0 {
            w_matrix
                .slice_mut(s![i, ..w_ij.dim().1])
                .assign(&w_ij.slice(s![i, ..]));
            w_matrix
                .slice_mut(s![i, w_ij.dim().1..])
                .assign(&w_ia.slice(s![i, ..]));
        }
        for i in 0..w_ai.dim().0 {
            w_matrix
                .slice_mut(s![w_ij.dim().0 + i, ..w_ai.dim().1])
                .assign(&w_ai.slice(s![i, ..]));
            w_matrix
                .slice_mut(s![w_ij.dim().0 + i, w_ai.dim().1..])
                .assign(&w_ab.slice(s![i, ..]));
        }

        // get arrays from properties
        let diff_p: Array2<f64> = &self.properties.p().unwrap() - &self.properties.p_ref().unwrap();
        let g0_ao: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let g1_ao: ArrayView3<f64> = self.properties.grad_gamma_ao().unwrap();
        let grad_h: ArrayView3<f64> = self.properties.grad_h0().unwrap();
        let grad_s: ArrayView3<f64> = self.properties.grad_s().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // calculate gradH: gradH0 + gradHexc
        let f_dmd0: Array3<f64> = f_v(
            diff_p.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let grad_h: Array3<f64> = &grad_h + &f_dmd0;

        // set the occupied and virtuals orbital coefficients
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let mut orbs_occ: Array2<f64> = Array::zeros((self.n_orbs, n_occ));
        let mut orbs_virt: Array2<f64> = Array::zeros((self.n_orbs, n_virt));
        for (i, index) in occ_indices.iter().enumerate() {
            orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }
        for (i, index) in virt_indices.iter().enumerate() {
            orbs_virt.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }

        // transform t and z vectors to AO basis
        let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
        let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
        let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));
        let x_ao: Array2<f64> = orbs_occ.dot(&x_state.dot(&orbs_virt.t()));

        // calculate contributions to the excited gradient
        let f: Array3<f64> = f_v(
            x_ao.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );

        // assemble the excited gradient
        let mut grad_exc: Array1<f64> = Array::zeros(3 * self.n_atoms);
        // gradH * (T + Z)
        grad_exc = grad_exc
            + grad_h
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(
                    &(t_vv - t_oo + z_ao)
                        .into_shape(self.n_orbs * self.n_orbs)
                        .unwrap(),
                );
        // - gradS * W
        grad_exc = grad_exc
            - grad_s
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // 2.0 * sum (X+Y) F (X+Y)
        grad_exc = grad_exc
            + 2.0
                * f.into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                    .unwrap()
                    .dot(&x_ao.view().into_shape(self.n_orbs * self.n_orbs).unwrap());

        grad_exc
    }

    pub fn tda_gradient_lc(&self, state: usize) -> Array1<f64> {
        // get occ and virt indices from properties
        let occ_indices: &[usize] = self.properties.occ_indices().unwrap();
        let virt_indices: &[usize] = self.properties.virt_indices().unwrap();

        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = occ_indices.iter().map(|&occ| orbe[occ]).collect();
        let orbe_virt: Array1<f64> = virt_indices.iter().map(|&virt| orbe[virt]).collect();

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

        let n_occ: usize = orbe_occ.len();
        let n_virt: usize = orbe_virt.len();

        // excitation energy of the state
        let omega_state: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
        let n_states: usize = omega_state.len();
        let omega_state: f64 = omega_state[state];
        // take state specific values from the excitation vectors
        let x_state: ArrayView2<f64> = self.properties.ci_coefficients().unwrap();
        let x_state: Array3<f64> = x_state
            .t()
            .as_standard_layout()
            .to_owned()
            .into_shape([n_states, n_occ, n_virt])
            .unwrap();
        let x_state: ArrayView2<f64> = x_state.slice(s![state, .., ..]);

        // calculate the vectors u, v and t
        // vectors U, V and T
        let u_ab: Array2<f64> = 2.0 * x_state.t().dot(&x_state);
        let u_ij: Array2<f64> = 2.0 * x_state.dot(&x_state.t());
        let v_ab: Array2<f64> = 2.0 * ei.dot(&x_state).t().dot(&x_state);
        let v_ij: Array2<f64> = 2.0 * x_state.dot(&ea).dot(&x_state.t());
        let t_ab: Array2<f64> = x_state.t().dot(&x_state);
        let t_ij: Array2<f64> = x_state.dot(&x_state.t());

        // get the transition charges
        let qtrans_ov: ArrayView3<f64> = self
            .properties
            .q_ov()
            .unwrap()
            .into_shape((self.n_atoms, n_occ, n_virt))
            .unwrap();
        let qtrans_oo: ArrayView3<f64> = self
            .properties
            .q_oo()
            .unwrap()
            .into_shape((self.n_atoms, n_occ, n_occ))
            .unwrap();
        let qtrans_vv: ArrayView3<f64> = self
            .properties
            .q_vv()
            .unwrap()
            .into_shape((self.n_atoms, n_virt, n_virt))
            .unwrap();
        let qtrans_vo: Array3<f64> = qtrans_ov
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();

        // create struct hplus
        let hav: Hav = Hav::new(qtrans_ov, qtrans_vv, qtrans_oo, qtrans_vo.view());
        let hplus: Hplus = Hplus::new(qtrans_ov, qtrans_vv, qtrans_oo, qtrans_vo.view());

        // set gamma matrix
        let g0: ArrayView2<f64> = self.properties.gamma().unwrap();
        let g0_lr: ArrayView2<f64> = self.properties.gamma_lr().unwrap();

        // compute hplus of tab and tij
        let hplus_tab: Array2<f64> = hav.compute(g0, g0_lr, t_ab.view(), HplusType::Tab);
        let hplus_tij: Array2<f64> = hav.compute(g0, g0_lr, t_ij.view(), HplusType::Tij);

        // calculate q_ij
        let g_ij: Array2<f64> = hplus_tab - hplus_tij;
        let q_ij: Array2<f64> = omega_state * u_ij - v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = omega_state * u_ab + v_ab;

        // calculate q_ia
        let mut q_ia: Array2<f64> =
            x_state.dot(&hav.compute(g0, g0_lr, x_state, HplusType::QiaXpy).t());
        q_ia = q_ia + hav.compute(g0, g0_lr, t_ab.view(), HplusType::QiaTab);
        q_ia = q_ia - hav.compute(g0, g0_lr, t_ij.view(), HplusType::QiaTij);

        // calculate q_ai
        let q_ai: Array2<f64> = x_state
            .t()
            .dot(&hav.compute(g0, g0_lr, x_state, HplusType::Qai));

        // calculate right hand side of the z-vector equation
        let r_ia: Array2<f64> = &q_ai.t() - &q_ia;

        // input for zvector routine
        let omega_input: Array2<f64> = into_col(Array::ones(orbe_occ.len()))
            .dot(&into_row(orbe_virt.clone()))
            - into_col(orbe_occ.clone()).dot(&into_row(Array::ones(orbe_virt.len())));
        let r_ia_flat: Array1<f64> = r_ia.t().to_owned_f().into_shape(n_occ * n_virt).unwrap();
        let r_matrix: Array2<f64> = r_ia_flat.into_shape((n_occ, n_virt)).unwrap();

        // calculate the z-vector
        let z_ia: Array2<f64> = zvector_lc(
            omega_input.view(),
            r_matrix.view(),
            g0,
            g0_lr,
            qtrans_oo,
            qtrans_vv,
            qtrans_ov,
        );

        // calculate w_ij
        let mut w_ij: Array2<f64> = q_ij + hplus.compute(g0, g0_lr, z_ia.view(), HplusType::Wij);
        for i in 0..w_ij.dim().0 {
            w_ij[[i, i]] /= 2.0;
        }
        // w_ia
        let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia); //+ h_a_nolr(g0, qtrans_ov, qtrans_ov, z_ia.view());

        // w_ai
        let w_ai: Array2<f64> = &q_ai + &ei.dot(&z_ia).t();

        // w_ab
        let mut w_ab: Array2<f64> = q_ab;
        for i in 0..w_ab.dim().0 {
            w_ab[[i, i]] /= 2.0;
        }

        // build w matrix: combine w_ij, w_ia, w_ai and w_ab
        let length: usize = n_occ + n_virt;
        let mut w_matrix: Array2<f64> = Array::zeros((length, length));
        for i in 0..w_ij.dim().0 {
            w_matrix
                .slice_mut(s![i, ..w_ij.dim().1])
                .assign(&w_ij.slice(s![i, ..]));
            w_matrix
                .slice_mut(s![i, w_ij.dim().1..])
                .assign(&w_ia.slice(s![i, ..]));
        }
        for i in 0..w_ai.dim().0 {
            w_matrix
                .slice_mut(s![w_ij.dim().0 + i, ..w_ai.dim().1])
                .assign(&w_ai.slice(s![i, ..]));
            w_matrix
                .slice_mut(s![w_ij.dim().0 + i, w_ai.dim().1..])
                .assign(&w_ab.slice(s![i, ..]));
        }

        // get arrays from properties
        let diff_p: Array2<f64> = &self.properties.p().unwrap() - &self.properties.p_ref().unwrap();
        let g0_ao: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let g1_ao: ArrayView3<f64> = self.properties.grad_gamma_ao().unwrap();
        let g1lr_ao: ArrayView3<f64> = self.properties.grad_gamma_lr_ao().unwrap();
        let g0lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
        let grad_h: ArrayView3<f64> = self.properties.grad_h0().unwrap();
        let grad_s: ArrayView3<f64> = self.properties.grad_s().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // calculate gradH: gradH0 + gradHexc
        let f_dmd0: Array3<f64> = f_v(
            diff_p.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_dmd0: Array3<f64> = f_lr(
            diff_p.view(),
            s,
            grad_s,
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );

        let grad_h: Array3<f64> = &grad_h + &f_dmd0 - 0.5 * &flr_dmd0;

        // set the occupied and virtuals orbital coefficients
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let mut orbs_occ: Array2<f64> = Array::zeros((self.n_orbs, n_occ));
        let mut orbs_virt: Array2<f64> = Array::zeros((self.n_orbs, n_virt));
        for (i, index) in occ_indices.iter().enumerate() {
            orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }
        for (i, index) in virt_indices.iter().enumerate() {
            orbs_virt.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }

        // transform t and z vectors to AO basis
        let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
        let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
        let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));
        let x_ao: Array2<f64> = orbs_occ.dot(&x_state.dot(&orbs_virt.t()));

        // calculate contributions to the excited gradient
        let f: Array3<f64> = f_v(
            x_ao.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_p = f_lr(
            x_ao.t(),
            s,
            grad_s,
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );

        // assemble the excited gradient
        let mut grad_exc: Array1<f64> = Array::zeros(3 * self.n_atoms);
        // gradH * (T + Z)
        grad_exc = grad_exc
            + grad_h
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(
                    &(t_vv - t_oo + z_ao)
                        .into_shape(self.n_orbs * self.n_orbs)
                        .unwrap(),
                );
        // - gradS * W
        grad_exc = grad_exc
            - grad_s
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // 2.0 * sum (X+Y) F (X+Y)
        grad_exc = grad_exc
            + 2.0
                * f.into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                    .unwrap()
                    .dot(&x_ao.view().into_shape(self.n_orbs * self.n_orbs).unwrap());
        // - sum (X) F_lr (X)(X)
        grad_exc = grad_exc
            - flr_p
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&x_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        grad_exc
    }

    /// On-the-fly accumulation version of tda_gradient_lc for FMO Monomers.
    /// Computes H0/S/gamma gradients on-the-fly instead of storing/loading large 3D arrays
    /// (grad_s, grad_h0, grad_gamma_ao, grad_gamma_lr_ao).
    /// This significantly reduces memory usage for large monomers.
    pub fn tda_gradient_lc_accumulation(&self, atoms: &[Atom], state: usize) -> Array1<f64> {
        // get occ and virt indices from properties
        let occ_indices: &[usize] = self.properties.occ_indices().unwrap();
        let virt_indices: &[usize] = self.properties.virt_indices().unwrap();

        let n_occ: usize = occ_indices.len();
        let n_virt: usize = virt_indices.len();

        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = occ_indices.iter().map(|&occ| orbe[occ]).collect();
        let orbe_virt: Array1<f64> = virt_indices.iter().map(|&virt| orbe[virt]).collect();

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

        // excitation energy of the state
        let omega_state: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
        let n_states: usize = omega_state.len();
        let omega_state: f64 = omega_state[state];
        // take state specific values from the excitation vectors
        let x_state: ArrayView2<f64> = self.properties.ci_coefficients().unwrap();
        let x_state: Array3<f64> = x_state
            .t()
            .as_standard_layout()
            .to_owned()
            .into_shape([n_states, n_occ, n_virt])
            .unwrap();
        let x_state: ArrayView2<f64> = x_state.slice(s![state, .., ..]);

        // calculate the vectors u, v and t
        let u_ab: Array2<f64> = 2.0 * x_state.t().dot(&x_state);
        let u_ij: Array2<f64> = 2.0 * x_state.dot(&x_state.t());
        let v_ab: Array2<f64> = 2.0 * ei.dot(&x_state).t().dot(&x_state);
        let v_ij: Array2<f64> = 2.0 * x_state.dot(&ea).dot(&x_state.t());
        let t_ab: Array2<f64> = x_state.t().dot(&x_state);
        let t_ij: Array2<f64> = x_state.dot(&x_state.t());

        // get the transition charges (atomwise only for FMO monomers)
        let qtrans_ov: ArrayView3<f64> = self
            .properties
            .q_ov()
            .unwrap()
            .into_shape((self.n_atoms, n_occ, n_virt))
            .unwrap();
        let qtrans_oo: ArrayView3<f64> = self
            .properties
            .q_oo()
            .unwrap()
            .into_shape((self.n_atoms, n_occ, n_occ))
            .unwrap();
        let qtrans_vv: ArrayView3<f64> = self
            .properties
            .q_vv()
            .unwrap()
            .into_shape((self.n_atoms, n_virt, n_virt))
            .unwrap();
        let qtrans_vo: Array3<f64> = qtrans_ov
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();

        // create struct hplus
        let hav: Hav = Hav::new(qtrans_ov, qtrans_vv, qtrans_oo, qtrans_vo.view());
        let hplus: Hplus = Hplus::new(qtrans_ov, qtrans_vv, qtrans_oo, qtrans_vo.view());

        // set gamma matrix
        let g0: ArrayView2<f64> = self.properties.gamma().unwrap();
        let g0_lr: ArrayView2<f64> = self.properties.gamma_lr().unwrap();

        // compute hplus of tab and tij
        let hplus_tab: Array2<f64> = hav.compute(g0, g0_lr, t_ab.view(), HplusType::Tab);
        let hplus_tij: Array2<f64> = hav.compute(g0, g0_lr, t_ij.view(), HplusType::Tij);

        // calculate q_ij
        let g_ij: Array2<f64> = hplus_tab - hplus_tij;
        let q_ij: Array2<f64> = omega_state * u_ij - v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = omega_state * u_ab + v_ab;

        // calculate q_ia
        let mut q_ia: Array2<f64> =
            x_state.dot(&hav.compute(g0, g0_lr, x_state, HplusType::QiaXpy).t());
        q_ia = q_ia + hav.compute(g0, g0_lr, t_ab.view(), HplusType::QiaTab);
        q_ia = q_ia - hav.compute(g0, g0_lr, t_ij.view(), HplusType::QiaTij);

        // calculate q_ai
        let q_ai: Array2<f64> = x_state
            .t()
            .dot(&hav.compute(g0, g0_lr, x_state, HplusType::Qai));

        // calculate right hand side of the z-vector equation
        let r_ia: Array2<f64> = &q_ai.t() - &q_ia;

        // input for zvector routine
        let omega_input: Array2<f64> = into_col(Array::ones(orbe_occ.len()))
            .dot(&into_row(orbe_virt.clone()))
            - into_col(orbe_occ.clone()).dot(&into_row(Array::ones(orbe_virt.len())));

        // calculate the z-vector
        let z_ia: Array2<f64> = zvector_lc_optimized(
            omega_input.view(),
            r_ia.view(),
            g0,
            g0_lr,
            qtrans_oo,
            qtrans_vv,
            qtrans_ov,
        );

        // calculate w_ij
        let mut w_ij: Array2<f64> = q_ij + hplus.compute(g0, g0_lr, z_ia.view(), HplusType::Wij);
        for i in 0..w_ij.dim().0 {
            w_ij[[i, i]] /= 2.0;
        }
        // w_ia
        let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia);

        // w_ai
        let w_ai: Array2<f64> = &q_ai + &ei.dot(&z_ia).t();

        // w_ab
        let mut w_ab: Array2<f64> = q_ab;
        for i in 0..w_ab.dim().0 {
            w_ab[[i, i]] /= 2.0;
        }

        // build w matrix: combine w_ij, w_ia, w_ai and w_ab
        let length: usize = n_occ + n_virt;
        let mut w_matrix: Array2<f64> = Array::zeros((length, length));
        for i in 0..w_ij.dim().0 {
            w_matrix
                .slice_mut(s![i, ..w_ij.dim().1])
                .assign(&w_ij.slice(s![i, ..]));
            w_matrix
                .slice_mut(s![i, w_ij.dim().1..])
                .assign(&w_ia.slice(s![i, ..]));
        }
        for i in 0..w_ai.dim().0 {
            w_matrix
                .slice_mut(s![w_ij.dim().0 + i, ..w_ai.dim().1])
                .assign(&w_ai.slice(s![i, ..]));
            w_matrix
                .slice_mut(s![w_ij.dim().0 + i, w_ai.dim().1..])
                .assign(&w_ab.slice(s![i, ..]));
        }

        // set the occupied and virtuals orbital coefficients
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let mut orbs_occ: Array2<f64> = Array::zeros((self.n_orbs, n_occ));
        let mut orbs_virt: Array2<f64> = Array::zeros((self.n_orbs, n_virt));
        for (i, index) in occ_indices.iter().enumerate() {
            orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }
        for (i, index) in virt_indices.iter().enumerate() {
            orbs_virt.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }

        // transform t and z vectors to AO basis
        let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
        let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
        let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));
        let t_plus_z: Array2<f64> = &t_vv - &t_oo + &z_ao;

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));
        let x_ao: Array2<f64> = orbs_occ.dot(&x_state.dot(&orbs_virt.t()));

        // ======= On-the-fly gradient accumulation =======
        // This version eliminates loading of grad_s, grad_gamma_ao, and grad_gamma_lr_ao
        // by computing dS and dG contributions on-the-fly using precomputed coefficient matrices.

        // Get arrays needed for coefficient precomputation (NO grad_s, grad_gamma_ao, grad_gamma_lr_ao!)
        let diff_p: Array2<f64> = &self.properties.p().unwrap() - &self.properties.p_ref().unwrap();
        let g0_ao: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let g0lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();

        // Pre-compute orbital index offsets for each atom
        let mut orbital_offsets: Vec<usize> = Vec::with_capacity(self.n_atoms + 1);
        orbital_offsets.push(0);
        for atom in atoms {
            orbital_offsets.push(orbital_offsets.last().unwrap() + atom.n_orbs);
        }

        // ======= Precompute coefficient matrices for dS-dependent terms =======
        // These coefficients allow us to compute dS contributions on-the-fly in the atom-pair loop

        // 1. f_v(diff_p) contracted with t_plus_z (coefficient for f_dmd0)
        let coeff_fv_dmd0 =
            compute_fv_coefficients_onthefly(diff_p.view(), s, g0_ao, t_plus_z.view());

        // 2. f_lr(diff_p) contracted with t_plus_z (scaled by -0.5)
        let coeff_flr_dmd0 = compute_flr_s_coefficients(diff_p.view(), s, g0lr_ao, t_plus_z.view());

        // 3. f_v(x_ao) contracted with x_ao (scaled by 2.0)
        let coeff_fv_xao = compute_fv_coefficients_onthefly(x_ao.view(), s, g0_ao, x_ao.view());

        // 4. f_lr(x_ao.t()) contracted with x_ao (scaled by -1.0)
        let coeff_flr_xao = compute_flr_s_coefficients(x_ao.t(), s, g0lr_ao, x_ao.view());

        // Combined dS coefficient: (1.0)*coeff_fv_dmd0 + (-0.5)*coeff_flr_dmd0 + (2.0)*coeff_fv_xao + (-1.0)*coeff_flr_xao
        let coeff_ds_total: Array2<f64> =
            &coeff_fv_dmd0 - 0.5 * &coeff_flr_dmd0 + 2.0 * &coeff_fv_xao - &coeff_flr_xao;

        // Collect valid atom pairs (i < j, within cutoff)
        let mut atom_pairs: Vec<(usize, usize)> = Vec::new();
        for i in 0..self.n_atoms {
            for j in (i + 1)..self.n_atoms {
                let distance = (&atoms[i] - &atoms[j]).norm();
                if distance < PROXIMITY_CUTOFF {
                    atom_pairs.push((i, j));
                }
            }
        }

        // Parallel compute H0/S/dS gradient contributions for each atom pair
        let pair_contributions: Vec<([f64; 3], [f64; 3], usize, usize)> = atom_pairs
            .par_iter()
            .map(|&(i, j)| {
                let atomi = &atoms[i];
                let atomj = &atoms[j];
                let mu_start = orbital_offsets[i];
                let nu_start = orbital_offsets[j];

                // Directional cosines
                let (r, x, y, z): (f64, f64, f64, f64) = if atomi <= atomj {
                    directional_cosines(&atomi.xyz, &atomj.xyz)
                } else {
                    directional_cosines(&atomj.xyz, &atomi.xyz)
                };

                // Pre-compute spline caches
                let skt = self.slako.get(atomi.kind, atomj.kind);
                let s_cache = SplineCache::new(r, &skt.s_spline);
                let h_cache = SplineCache::new(r, &skt.h_spline);

                let mut grad_i = [0.0_f64; 3];
                let mut grad_j = [0.0_f64; 3];

                let mut mu = mu_start;
                for orbi in atomi.valorbs.iter() {
                    let mut nu = nu_start;
                    for orbj in atomj.valorbs.iter() {
                        let (s_deriv_i, h0_deriv_i, s_deriv_j, h0_deriv_j): (
                            [f64; 3],
                            [f64; 3],
                            [f64; 3],
                            [f64; 3],
                        ) = if atomi <= atomj {
                            let s_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &s_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                            );
                            let h0_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &h_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                            );
                            (
                                [-s_grad[0], -s_grad[1], -s_grad[2]],
                                [-h0_grad[0], -h0_grad[1], -h0_grad[2]],
                                s_grad,
                                h0_grad,
                            )
                        } else {
                            let s_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &s_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                            );
                            let h0_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &h_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                            );
                            (
                                s_grad,
                                h0_grad,
                                [-s_grad[0], -s_grad[1], -s_grad[2]],
                                [-h0_grad[0], -h0_grad[1], -h0_grad[2]],
                            )
                        };

                        // Coefficients for contraction
                        let t_plus_z_mu_nu = t_plus_z[[mu, nu]];
                        let t_plus_z_nu_mu = t_plus_z[[nu, mu]];
                        let w_ao_mu_nu = w_ao[[mu, nu]];
                        let w_ao_nu_mu = w_ao[[nu, mu]];

                        // Get dS coefficient for f_v/f_lr contribution
                        let coeff_mu_nu = coeff_ds_total[[mu, nu]];
                        let coeff_nu_mu = coeff_ds_total[[nu, mu]];

                        // grad_H0 * (T + Z) contribution
                        for dir in 0..3 {
                            // H0 contribution: dH0[mu,nu] * t_plus_z[mu,nu] + dH0[nu,mu] * t_plus_z[nu,mu]
                            grad_i[dir] +=
                                h0_deriv_i[dir] * t_plus_z_mu_nu + h0_deriv_i[dir] * t_plus_z_nu_mu;
                            grad_j[dir] +=
                                h0_deriv_j[dir] * t_plus_z_mu_nu + h0_deriv_j[dir] * t_plus_z_nu_mu;

                            // -grad_S * W contribution: -dS[mu,nu] * w_ao[mu,nu] - dS[nu,mu] * w_ao[nu,mu]
                            grad_i[dir] -=
                                s_deriv_i[dir] * w_ao_mu_nu + s_deriv_i[dir] * w_ao_nu_mu;
                            grad_j[dir] -=
                                s_deriv_j[dir] * w_ao_mu_nu + s_deriv_j[dir] * w_ao_nu_mu;

                            // f_v/f_lr dS contribution: dS[mu,nu] * coeff_ds_total[mu,nu]
                            grad_i[dir] += s_deriv_i[dir] * coeff_mu_nu;
                            grad_j[dir] += s_deriv_j[dir] * coeff_nu_mu;
                        }

                        nu += 1;
                    }
                    mu += 1;
                }
                (grad_i, grad_j, i, j)
            })
            .collect();

        // Reduce H0/S/dS contributions to gradient
        let mut grad_h0_s_contrib: Array1<f64> = Array1::zeros(3 * self.n_atoms);
        for (grad_i, grad_j, i, j) in pair_contributions {
            for dir in 0..3 {
                grad_h0_s_contrib[3 * i + dir] += grad_i[dir];
                grad_h0_s_contrib[3 * j + dir] += grad_j[dir];
            }
        }

        // ======= dG (gamma gradient) contributions computed on-the-fly =======
        // Instead of loading grad_gamma_ao and grad_gamma_lr_ao (3D arrays),
        // we compute the gamma gradient contributions using atomwise gamma derivatives

        // Get gamma functions for on-the-fly gamma gradient computation
        let gammafunction = &self.gammafunction;
        let gammafunction_lr = self.gammafunction_lc.as_ref().unwrap();

        // 1. f_v(diff_p) dG contribution contracted with t_plus_z
        let fv_dg_dmd0 = compute_fv_dg_contributions(
            gammafunction,
            atoms,
            &orbital_offsets,
            s,
            diff_p.view(),
            t_plus_z.view(),
            self.n_atoms,
            self.n_orbs,
        );

        // 2. f_lr(diff_p) dG contribution contracted with t_plus_z (scaled by -0.5)
        let flr_dg_dmd0 = compute_flr_dg_contributions(
            gammafunction_lr,
            atoms,
            &orbital_offsets,
            s,
            diff_p.view(),
            t_plus_z.view(),
            self.n_atoms,
            self.n_orbs,
        );

        // 3. f_v(x_ao) dG contribution contracted with x_ao (scaled by 2.0)
        let fv_dg_xao = compute_fv_dg_contributions(
            gammafunction,
            atoms,
            &orbital_offsets,
            s,
            x_ao.view(),
            x_ao.view(),
            self.n_atoms,
            self.n_orbs,
        );

        // 4. f_lr(x_ao.t()) dG contribution contracted with x_ao (scaled by -1.0)
        let flr_dg_xao = compute_flr_dg_contributions(
            gammafunction_lr,
            atoms,
            &orbital_offsets,
            s,
            x_ao.t().to_owned().view(),
            x_ao.view(),
            self.n_atoms,
            self.n_orbs,
        );

        // Combine dG contributions with the same scaling factors
        let grad_dg_contrib: Array1<f64> =
            &fv_dg_dmd0 - 0.5 * &flr_dg_dmd0 + 2.0 * &fv_dg_xao - &flr_dg_xao;

        // Assemble the excited gradient:
        // grad_H0 * (T + Z) - grad_S * W + dS contribution to f_v/f_lr (computed on-the-fly above)
        // + dG contribution to f_v/f_lr (computed on-the-fly above)
        let grad_exc: Array1<f64> = &grad_h0_s_contrib + &grad_dg_contrib;

        grad_exc
    }
}
