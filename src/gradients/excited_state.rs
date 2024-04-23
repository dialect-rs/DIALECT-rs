use crate::excited_states::trans_charges;
use crate::gradients::helpers::*;
use crate::initialization::*;
use crate::scc::gamma_approximation::{
    gamma_ao_wise_from_gamma_atomwise, gamma_gradients_ao_wise,
    gamma_gradients_ao_wise_from_atomwise,
};
use crate::utils::ToOwnedF;
use ndarray::{s, Array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use ndarray_linalg::{into_col, into_row, IntoTriangular, UPLO};

impl System {
    pub fn calculate_excited_state_gradient(&mut self, state: usize) -> Array1<f64> {
        self.prepare_excited_grad();

        let gradient: Array1<f64> = if self.config.lc.long_range_correction {
            if self.config.excited.use_casida {
                self.tddft_gradient_lc(state)
            } else {
                self.tda_gradient_lc(state)
            }
        } else {
            if self.config.excited.use_casida {
                self.tddft_gradient_no_lc(state)
            } else {
                self.tda_gradient_nolc(state)
            }
        };
        gradient
    }

    pub fn prepare_excited_grad(&mut self) {
        // calculate transition charges if they don't exist
        if self.properties.contains_key("q_ov") == false {
            let tmp: (Array2<f64>, Array2<f64>, Array2<f64>) = trans_charges(
                self.n_atoms,
                &self.atoms,
                self.properties.orbs().unwrap(),
                self.properties.s().unwrap(),
                &self.occ_indices,
                &self.virt_indices,
            );

            self.properties.set_q_ov(tmp.0);
            self.properties.set_q_oo(tmp.1);
            self.properties.set_q_vv(tmp.2);
        }
        // prepare gamma and grad gamma AO matrix
        let g0_ao: Array2<f64> = gamma_ao_wise_from_gamma_atomwise(
            self.properties.gamma().unwrap(),
            &self.atoms,
            self.n_orbs,
        );
        self.properties.set_gamma_ao(g0_ao);

        if self.properties.contains_key("grad_gamma") == false {
            let (g1, g1_ao): (Array3<f64>, Array3<f64>) = gamma_gradients_ao_wise(
                &self.gammafunction,
                &self.atoms,
                self.n_atoms,
                self.n_orbs,
            );
            self.properties.set_grad_gamma(g1);
            self.properties.set_grad_gamma_ao(g1_ao);
        } else {
            let g1_ao: Array3<f64> = gamma_gradients_ao_wise_from_atomwise(
                self.properties.grad_gamma().unwrap(),
                &self.atoms,
                self.n_atoms,
                self.n_orbs,
            );
            self.properties.set_grad_gamma_ao(g1_ao);
        }
    }

    pub fn tda_gradient_nolc(&mut self, state: usize) -> Array1<f64> {
        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = self.occ_indices.iter().map(|&occ| orbe[occ]).collect();
        let orbe_virt: Array1<f64> = self.virt_indices.iter().map(|&virt| orbe[virt]).collect();

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

        let n_occ: usize = orbe_occ.len();
        let n_virt: usize = orbe_virt.len();

        // excitation energy of the state
        let n_states: usize = self.config.excited.nstates;
        let omega_state: f64 = self.properties.ci_eigenvalues().unwrap()[state];
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
            w_ij[[i, i]] = w_ij[[i, i]] / 2.0;
        }
        // w_ia
        let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia); //+ h_a_nolr(g0, qtrans_ov, qtrans_ov, z_ia.view());

        // w_ai
        let w_ai: Array2<f64> = &q_ai + &ei.dot(&z_ia).t();

        // w_ab
        let mut w_ab: Array2<f64> = q_ab;
        for i in 0..w_ab.dim().0 {
            w_ab[[i, i]] = w_ab[[i, i]] / 2.0;
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
        let f_dmd0: Array3<f64> =
            f_v(
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
        for (i, index) in self.occ_indices.iter().enumerate() {
            orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }
        for (i, index) in self.virt_indices.iter().enumerate() {
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
        let f: Array3<f64> =
            f_v(
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

        return grad_exc;
    }

    pub fn tda_gradient_lc(&mut self, state: usize) -> Array1<f64> {
        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = self.occ_indices.iter().map(|&occ| orbe[occ]).collect();
        let orbe_virt: Array1<f64> = self.virt_indices.iter().map(|&virt| orbe[virt]).collect();

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

        let n_occ: usize = orbe_occ.len();
        let n_virt: usize = orbe_virt.len();

        // excitation energy of the state
        let n_states: usize = self.config.excited.nstates;
        let omega_state: f64 = self.properties.ci_eigenvalues().unwrap()[state];
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

        // calculate the z-vector
        let z_ia: Array2<f64> = zvector_lc(
            omega_input.view(),
            r_ia.view(),
            g0,
            g0_lr,
            qtrans_oo,
            qtrans_vv,
            qtrans_ov,
        );

        // calculate w_ij
        let mut w_ij: Array2<f64> = q_ij + hplus.compute(g0, g0_lr, z_ia.view(), HplusType::Wij); //+hav.compute(g0, g0_lr, z_ia.view(), HplusType::Wij);
        for i in 0..w_ij.dim().0 {
            w_ij[[i, i]] = w_ij[[i, i]] / 2.0;
        }
        // w_ia
        let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia); //+ h_a_nolr(g0, qtrans_ov, qtrans_ov, z_ia.view());

        // w_ai
        let w_ai: Array2<f64> = &q_ai + &ei.dot(&z_ia).t();

        // w_ab
        let mut w_ab: Array2<f64> = q_ab;
        for i in 0..w_ab.dim().0 {
            w_ab[[i, i]] = w_ab[[i, i]] / 2.0;
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
        let flr_dmd0: ArrayView3<f64> = self.properties.f_lr_dmd0().unwrap();
        let grad_h: ArrayView3<f64> = self.properties.grad_h0().unwrap();
        let grad_s: ArrayView3<f64> = self.properties.grad_s().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // calculate gradH: gradH0 + gradHexc
        let f_dmd0: Array3<f64> =
            f_v(
                diff_p.view(),
                s,
                grad_s,
                g0_ao,
                g1_ao,
                self.n_atoms,
                self.n_orbs,
            );
        let grad_h: Array3<f64> = &grad_h + &f_dmd0 - 0.5 * &flr_dmd0;

        // set the occupied and virtuals orbital coefficients
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let mut orbs_occ: Array2<f64> = Array::zeros((self.n_orbs, n_occ));
        let mut orbs_virt: Array2<f64> = Array::zeros((self.n_orbs, n_virt));
        for (i, index) in self.occ_indices.iter().enumerate() {
            orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }
        for (i, index) in self.virt_indices.iter().enumerate() {
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

        // set g0lr_ao and g1lr_ao
        let g0lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
        let g1lr_ao: ArrayView3<f64> = self.properties.grad_gamma_lr_ao().unwrap();

        // calculate contributions to the excited gradient
        let f: Array3<f64> =
            f_v(
                x_ao.view(),
                s,
                grad_s,
                g0_ao,
                g1_ao,
                self.n_atoms,
                self.n_orbs,
            );
        let flr_p =
            f_lr(
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
        return grad_exc;
    }

    pub fn tddft_gradient_lc(&mut self, state: usize) -> Array1<f64> {
        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = self.occ_indices.iter().map(|&occ| orbe[occ]).collect();
        let orbe_virt: Array1<f64> = self.virt_indices.iter().map(|&virt| orbe[virt]).collect();

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

        let n_occ: usize = orbe_occ.len();
        let n_virt: usize = orbe_virt.len();

        // take state specific values from the excitation vectors
        let xmy_state: ArrayView3<f64> = self.properties.xmy().unwrap();
        let xpy_state: ArrayView3<f64> = self.properties.xpy().unwrap();
        let xmy_state: ArrayView2<f64> = xmy_state.slice(s![state, .., ..]);
        let xpy_state: ArrayView2<f64> = xpy_state.slice(s![state, .., ..]);
        // excitation energy of the state
        let omega_state: f64 = self.properties.ci_eigenvalue(state).unwrap();

        // calculate the vectors u, v and t
        let u_ab: Array2<f64> = xpy_state.t().dot(&xmy_state) + xmy_state.t().dot(&xpy_state);
        let u_ij: Array2<f64> = xpy_state.dot(&xmy_state.t()) + xmy_state.dot(&xpy_state.t());

        let v_ab: Array2<f64> =
            ei.dot(&xpy_state).t().dot(&xpy_state) + ei.dot(&xmy_state).t().dot(&xmy_state);
        let v_ij: Array2<f64> =
            xpy_state.dot(&ea).dot(&xpy_state.t()) + xmy_state.dot(&ea).dot(&xmy_state.t());

        let t_ab: Array2<f64> =
            0.5 * (xpy_state.t().dot(&xpy_state) + xmy_state.t().dot(&xmy_state));
        let t_ij: Array2<f64> =
            0.5 * (xpy_state.dot(&xpy_state.t()) + xmy_state.dot(&xmy_state.t()));

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
        let hplus: Hplus = Hplus::new(qtrans_ov, qtrans_vv, qtrans_oo, qtrans_vo.view());

        // set gamma matrices
        let g0: ArrayView2<f64> = self.properties.gamma().unwrap();
        let g0_lr: ArrayView2<f64> = self.properties.gamma_lr().unwrap();

        // compute hplus of tab and tij
        let hplus_tab: Array2<f64> = hplus.compute(g0, g0_lr, t_ab.view(), HplusType::Tab);
        let hplus_tij: Array2<f64> = hplus.compute(g0, g0_lr, t_ij.view(), HplusType::Tij);

        // calculate q_ij
        let g_ij: Array2<f64> = hplus_tab - hplus_tij;
        let q_ij: Array2<f64> = omega_state * u_ij - v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = omega_state * u_ab + v_ab;

        // calculate q_ia
        let mut q_ia: Array2<f64> =
            xpy_state.dot(&hplus.compute(g0, g0_lr, xpy_state, HplusType::QiaXpy).t());
        q_ia = q_ia
            + xmy_state.dot(
                &h_minus(
                    g0_lr,
                    qtrans_vv,
                    qtrans_vo.view(),
                    qtrans_vo.view(),
                    qtrans_vv,
                    xmy_state,
                )
                .t(),
            );
        q_ia = q_ia + hplus.compute(g0, g0_lr, t_ab.view(), HplusType::QiaTab);
        q_ia = q_ia - hplus.compute(g0, g0_lr, t_ij.view(), HplusType::QiaTij);

        // calculate q_ai
        let mut q_ai: Array2<f64> =
            xpy_state
                .t()
                .dot(&hplus.compute(g0, g0_lr, xpy_state, HplusType::Qai));
        q_ai = q_ai
            + xmy_state
                .t()
                .dot(&h_minus(g0_lr, qtrans_ov, qtrans_oo, qtrans_oo, qtrans_ov, xmy_state));

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
            zvector_lc(
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
            w_ij[[i, i]] = w_ij[[i, i]] / 2.0;
        }
        // w_ia
        let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia);

        // w_ai
        let w_ai: Array2<f64> = w_ia.clone().reversed_axes();

        // w_ab
        let mut w_ab: Array2<f64> = q_ab;
        for i in 0..w_ab.dim().0 {
            w_ab[[i, i]] = w_ab[[i, i]] / 2.0;
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
        let flr_dmd0: ArrayView3<f64> = self.properties.f_lr_dmd0().unwrap();
        let grad_h: ArrayView3<f64> = self.properties.grad_h0().unwrap();
        let grad_s: ArrayView3<f64> = self.properties.grad_s().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // calculate gradH: gradH0 + gradHexc
        let f_dmd0: Array3<f64> =
            f_v(
                diff_p.view(),
                s,
                grad_s,
                g0_ao,
                g1_ao,
                self.n_atoms,
                self.n_orbs,
            );
        let grad_h: Array3<f64> = &grad_h + &f_dmd0 - 0.5 * &flr_dmd0;

        // set the occupied and virtuals orbital coefficients
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let mut orbs_occ: Array2<f64> = Array::zeros((self.n_orbs, n_occ));
        let mut orbs_virt: Array2<f64> = Array::zeros((self.n_orbs, n_virt));
        for (i, index) in self.occ_indices.iter().enumerate() {
            orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }
        for (i, index) in self.virt_indices.iter().enumerate() {
            orbs_virt.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }

        // transform t and z vectors to AO basis
        let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
        let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
        let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));
        let xpy_ao: Array2<f64> = orbs_occ.dot(&xpy_state.dot(&orbs_virt.t()));
        let xmy_ao: Array2<f64> = orbs_occ.dot(&xmy_state.dot(&orbs_virt.t()));

        // set g0lr_ao and g1lr_ao
        let g0lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
        let g1lr_ao: ArrayView3<f64> = self.properties.grad_gamma_lr_ao().unwrap();

        // calculate contributions to the excited gradient
        let f: Array3<f64> =
            f_v(
                xpy_ao.view(),
                s,
                grad_s,
                g0_ao,
                g1_ao,
                self.n_atoms,
                self.n_orbs,
            );
        let flr_p = f_lr(
            (&xpy_ao + &xpy_ao.t()).view(),
            s,
            grad_s,
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_m = -f_lr(
            (&xmy_ao - &xmy_ao.t()).view(),
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
                    .dot(&xpy_ao.view().into_shape(self.n_orbs * self.n_orbs).unwrap());
        // - 0.5 * sum (X+Y) F_lr (X+Y)(X+Y)
        grad_exc = grad_exc
            - 0.5
                * flr_p
                    .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                    .unwrap()
                    .dot(&xpy_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // - 0.5 * sum (X-Y) F_lr (X-Y)(X-Y)
        grad_exc = grad_exc
            - 0.5
                * flr_m
                    .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                    .unwrap()
                    .dot(&xmy_ao.view().into_shape(self.n_orbs * self.n_orbs).unwrap());

        return grad_exc;
    }

    pub fn tddft_gradient_no_lc(&mut self, state: usize) -> Array1<f64> {
        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = self.occ_indices.iter().map(|&occ| orbe[occ]).collect();
        let orbe_virt: Array1<f64> = self.virt_indices.iter().map(|&virt| orbe[virt]).collect();

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

        let n_occ: usize = orbe_occ.len();
        let n_virt: usize = orbe_virt.len();

        // take state specific values from the excitation vectors
        let xmy_state: ArrayView3<f64> = self.properties.xmy().unwrap();
        let xpy_state: ArrayView3<f64> = self.properties.xpy().unwrap();
        let xmy_state: ArrayView2<f64> = xmy_state.slice(s![state, .., ..]);
        let xpy_state: ArrayView2<f64> = xpy_state.slice(s![state, .., ..]);
        // excitation energy of the state
        let omega_state: f64 = self.properties.ci_eigenvalue(state).unwrap();

        // calculate the vectors u, v and t
        let u_ab: Array2<f64> = xpy_state.t().dot(&xmy_state) + xmy_state.t().dot(&xpy_state);
        let u_ij: Array2<f64> = xpy_state.dot(&xmy_state.t()) + xmy_state.dot(&xpy_state.t());

        let v_ab: Array2<f64> =
            ei.dot(&xpy_state).t().dot(&xpy_state) + ei.dot(&xmy_state).t().dot(&xmy_state);
        let v_ij: Array2<f64> =
            xpy_state.dot(&ea).dot(&xpy_state.t()) + xmy_state.dot(&ea).dot(&xmy_state.t());

        let t_ab: Array2<f64> =
            0.5 * (xpy_state.t().dot(&xpy_state) + xmy_state.t().dot(&xmy_state));
        let t_ij: Array2<f64> =
            0.5 * (xpy_state.dot(&xpy_state.t()) + xmy_state.dot(&xmy_state.t()));

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
        let hplus_tab: Array2<f64> = h_plus_no_lr(g0, qtrans_oo, qtrans_vv, t_ab.view());
        let hplus_tij: Array2<f64> = h_plus_no_lr(g0, qtrans_oo, qtrans_oo, t_ij.view());

        // calculate q_ij
        let g_ij: Array2<f64> = hplus_tab - hplus_tij;
        let q_ij: Array2<f64> = omega_state * u_ij - v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = omega_state * u_ab + v_ab;

        // calculate q_ia
        let mut q_ia: Array2<f64> =
            xpy_state.dot(&h_plus_no_lr(g0, qtrans_vv, qtrans_ov, xpy_state).t());
        q_ia = q_ia + h_plus_no_lr(g0, qtrans_ov, qtrans_vv, t_ab.view());
        q_ia = q_ia - h_plus_no_lr(g0, qtrans_ov, qtrans_oo, t_ij.view());

        // calculate q_ai
        let q_ai: Array2<f64> = xpy_state
            .t()
            .dot(&h_plus_no_lr(g0, qtrans_oo, qtrans_ov, xpy_state));

        // calculate right hand side of the z-vector equation
        let r_ia: Array2<f64> = &q_ai.t() - &q_ia;

        // input for zvector routine
        let omega_input: Array2<f64> = into_col(Array::ones(orbe_occ.len()))
            .dot(&into_row(orbe_virt.clone()))
            - into_col(orbe_occ.clone()).dot(&into_row(Array::ones(orbe_virt.len())));
        let r_ia_flat: Array1<f64> = r_ia.t().to_owned_f().into_shape(n_occ * n_virt).unwrap();
        let r_matrix: Array2<f64> = r_ia_flat.into_shape((n_occ, n_virt)).unwrap();

        // calculate the z-vector
        let z_ia: Array2<f64> = zvector_no_lc(omega_input.view(), r_matrix.view(), g0, qtrans_ov);

        // calculate w_ij
        let mut w_ij: Array2<f64> = q_ij + h_plus_no_lr(g0, qtrans_oo, qtrans_ov, z_ia.view());
        for i in 0..w_ij.dim().0 {
            w_ij[[i, i]] = w_ij[[i, i]] / 2.0;
        }
        // w_ia
        let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia);

        // w_ai
        let w_ai: Array2<f64> = w_ia.clone().reversed_axes();

        // w_ab
        let mut w_ab: Array2<f64> = q_ab;
        for i in 0..w_ab.dim().0 {
            w_ab[[i, i]] = w_ab[[i, i]] / 2.0;
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
        let f_dmd0: Array3<f64> =
            f_v(
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
        for (i, index) in self.occ_indices.iter().enumerate() {
            orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }
        for (i, index) in self.virt_indices.iter().enumerate() {
            orbs_virt.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }

        // transform t and z vectors to AO basis
        let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
        let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
        let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));
        let xpy_ao: Array2<f64> = orbs_occ.dot(&xpy_state.dot(&orbs_virt.t()));

        // calculate contributions to the excited gradient
        let f: Array3<f64> =
            f_v(
                xpy_ao.view(),
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
                    .dot(&xpy_ao.view().into_shape(self.n_orbs * self.n_orbs).unwrap());

        return grad_exc;
    }
}
