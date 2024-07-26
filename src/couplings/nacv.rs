use crate::excited_states::trans_charges;
use crate::fmo::helpers::get_pair_slice;
use crate::fmo::old_supersystem::OldSupersystem;
use crate::fmo::{
    ChargeTransferPair, ESDPair, Monomer, Pair, PairType, ReducedBasisState, SuperSystem,
};
use crate::gradients::helpers::*;
use crate::initialization::*;
use crate::scc::gamma_approximation::{
    gamma_ao_wise_from_gamma_atomwise, gamma_gradients_ao_wise,
    gamma_gradients_ao_wise_from_atomwise,
};
use crate::utils::ToOwnedF;
use hashbrown::HashMap;
use ndarray::{s, Array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use ndarray_linalg::{into_col, into_row, IntoTriangular, Norm, UPLO};
use rayon::prelude::*;

impl System {
    pub fn nac_tddft_gs(&mut self, state: usize) -> Array1<f64> {
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

        // get matrix P_ia
        let p_ia: Array2<f64> = (1.0 / omega_state) * &xpy_state;

        // calculate w_ij
        let mut w_ij: Array2<f64> = hplus.compute(g0, g0_lr, p_ia.view(), HplusType::Wij);
        for i in 0..w_ij.dim().0 {
            w_ij[[i, i]] = w_ij[[i, i]] / 2.0;
        }
        // w_ia
        let w_ia: Array2<f64> = 0.5 * &xmy_state + &ei.dot(&p_ia);

        // w_ai
        let w_ai: Array2<f64> = w_ia.clone().reversed_axes();

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
        let f_dmd0: Array3<f64> = f_v(
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

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));
        let xpy_ao: Array2<f64> = orbs_occ.dot(&xpy_state.dot(&orbs_virt.t()));
        let xmy_ao: Array2<f64> = orbs_occ.dot(&xmy_state.dot(&orbs_virt.t()));
        let p_ao: Array2<f64> = orbs_occ.dot(&p_ia.dot(&orbs_virt.t()));

        // assemble the excited gradient
        let mut nac: Array1<f64> = Array::zeros(3 * self.n_atoms);
        // gradH * (T + Z)
        nac = nac
            + grad_h
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&(p_ao).into_shape(self.n_orbs * self.n_orbs).unwrap());
        // - gradS * W
        nac = nac
            - grad_s
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());

        return nac;
    }

    pub fn nac_tddft_gs_no_lc(&mut self, state: usize) -> Array1<f64> {
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

        // set gamma matrices
        let g0: ArrayView2<f64> = self.properties.gamma().unwrap();

        // get matrix P_ia
        let p_ia: Array2<f64> = (1.0 / omega_state) * &xpy_state;

        // calculate w_ij
        let mut w_ij: Array2<f64> = h_plus_no_lr(g0, qtrans_oo, qtrans_ov, p_ia.view());
        for i in 0..w_ij.dim().0 {
            w_ij[[i, i]] = w_ij[[i, i]] / 2.0;
        }
        // w_ia
        let w_ia: Array2<f64> = 0.5 * &xmy_state + &ei.dot(&p_ia);

        // w_ai
        let w_ai: Array2<f64> = w_ia.clone().reversed_axes();

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
        for (i, index) in self.occ_indices.iter().enumerate() {
            orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }
        for (i, index) in self.virt_indices.iter().enumerate() {
            orbs_virt.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));
        let xpy_ao: Array2<f64> = orbs_occ.dot(&xpy_state.dot(&orbs_virt.t()));
        let xmy_ao: Array2<f64> = orbs_occ.dot(&xmy_state.dot(&orbs_virt.t()));
        let p_ao: Array2<f64> = orbs_occ.dot(&p_ia.dot(&orbs_virt.t()));

        // assemble the excited gradient
        let mut nac: Array1<f64> = Array::zeros(3 * self.n_atoms);
        // gradH * (T + Z)
        nac = nac
            + grad_h
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&(p_ao).into_shape(self.n_orbs * self.n_orbs).unwrap());
        // - gradS * W
        nac = nac
            - grad_s
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());

        return nac;
    }

    pub fn nac_tddft_excited(&self, state_1: usize, state_2: usize) -> Array1<f64> {
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
        let xmy_state_1: ArrayView2<f64> = xmy_state.slice(s![state_1, .., ..]);
        let xpy_state_1: ArrayView2<f64> = xpy_state.slice(s![state_1, .., ..]);
        // excitation energy of the state
        let omega_state_1: f64 = self.properties.ci_eigenvalue(state_1).unwrap();

        // take state specific values from the excitation vectors
        let xmy_state_2: ArrayView2<f64> = xmy_state.slice(s![state_2, .., ..]);
        let xpy_state_2: ArrayView2<f64> = xpy_state.slice(s![state_2, .., ..]);
        // excitation energy of the state
        let omega_state_2: f64 = self.properties.ci_eigenvalue(state_2).unwrap();
        let omega_avg: f64 = (omega_state_1 + omega_state_2) / 2.0;
        let omega_diff: f64 = (omega_state_1 - omega_state_2);

        // calculate the vectors u, v and t
        let u_ab: Array2<f64> = xpy_state_1.t().dot(&xmy_state_2)
            + xmy_state_1.t().dot(&xpy_state_2)
            + xpy_state_2.t().dot(&xmy_state_1)
            + xmy_state_2.t().dot(&xpy_state_1);
        let u_ij: Array2<f64> = xpy_state_1.dot(&xmy_state_2.t())
            + xmy_state_1.dot(&xpy_state_2.t())
            + xpy_state_2.dot(&xmy_state_1.t())
            + xmy_state_2.dot(&xpy_state_1.t());

        let v_ab: Array2<f64> = ei.dot(&xpy_state_1).t().dot(&xpy_state_2)
            + ei.dot(&xmy_state_1).t().dot(&xmy_state_2)
            + ei.dot(&xpy_state_2).t().dot(&xpy_state_1)
            + ei.dot(&xmy_state_2).t().dot(&xmy_state_1);
        let v_ij: Array2<f64> = xpy_state_1.dot(&ea).dot(&xpy_state_2.t())
            + xmy_state_1.dot(&ea).dot(&xmy_state_2.t())
            + xpy_state_2.dot(&ea).dot(&xpy_state_1.t())
            + xmy_state_2.dot(&ea).dot(&xmy_state_1.t());

        let t_ab: Array2<f64> = 0.25
            * (xpy_state_1.t().dot(&xpy_state_2)
                + xpy_state_2.t().dot(&xpy_state_1)
                + xmy_state_1.t().dot(&xmy_state_2)
                + xmy_state_2.t().dot(&xmy_state_1));
        let t_ij: Array2<f64> = 0.25
            * (xpy_state_1.dot(&xpy_state_2.t())
                + xpy_state_2.dot(&xpy_state_1.t())
                + xmy_state_1.dot(&xmy_state_2.t())
                + xmy_state_2.dot(&xmy_state_1.t()));

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
        let q_ij: Array2<f64> = 0.5 * omega_avg * u_ij - 0.5 * v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = 0.5 * omega_avg * u_ab + 0.5 * v_ab;

        // calculate q_ia
        // first term
        let mut q_ia: Array2<f64> =
            0.5 * xpy_state_1.dot(&hplus.compute(g0, g0_lr, xpy_state_2, HplusType::QiaXpy).t());
        // second term
        q_ia = q_ia
            + 0.5 * xpy_state_2.dot(&hplus.compute(g0, g0_lr, xpy_state_1, HplusType::QiaXpy).t());
        // first term
        q_ia = q_ia
            + 0.5
                * xmy_state_1.dot(
                    &h_minus(
                        g0_lr,
                        qtrans_vv,
                        qtrans_vo.view(),
                        qtrans_vo.view(),
                        qtrans_vv,
                        xmy_state_2,
                    )
                    .t(),
                );
        // second term
        q_ia = q_ia
            + 0.5
                * xmy_state_2.dot(
                    &h_minus(
                        g0_lr,
                        qtrans_vv,
                        qtrans_vo.view(),
                        qtrans_vo.view(),
                        qtrans_vv,
                        xmy_state_1,
                    )
                    .t(),
                );
        q_ia = q_ia + hplus.compute(g0, g0_lr, t_ab.view(), HplusType::QiaTab);
        q_ia = q_ia - hplus.compute(g0, g0_lr, t_ij.view(), HplusType::QiaTij);

        // calculate q_ai
        let mut q_ai: Array2<f64> = 0.5
            * xpy_state_1
                .t()
                .dot(&hplus.compute(g0, g0_lr, xpy_state_2, HplusType::Qai));
        q_ai = q_ai
            + 0.5
                * xpy_state_2
                    .t()
                    .dot(&hplus.compute(g0, g0_lr, xpy_state_1, HplusType::Qai));
        q_ai = q_ai
            + 0.5
                * xmy_state_1.t().dot(&h_minus(
                    g0_lr,
                    qtrans_ov,
                    qtrans_oo,
                    qtrans_oo,
                    qtrans_ov,
                    xmy_state_2,
                ));
        q_ai = q_ai
            + 0.5
                * xmy_state_2.t().dot(&h_minus(
                    g0_lr,
                    qtrans_ov,
                    qtrans_oo,
                    qtrans_oo,
                    qtrans_ov,
                    xmy_state_1,
                ));

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
        let f_dmd0: Array3<f64> = f_v(
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
        let xpy_ao_1: Array2<f64> = orbs_occ.dot(&xpy_state_1.dot(&orbs_virt.t()));
        let xpy_ao_2: Array2<f64> = orbs_occ.dot(&xpy_state_2.dot(&orbs_virt.t()));
        let xmy_ao_1: Array2<f64> = orbs_occ.dot(&xmy_state_1.dot(&orbs_virt.t()));
        let xmy_ao_2: Array2<f64> = orbs_occ.dot(&xmy_state_2.dot(&orbs_virt.t()));

        // set g0lr_ao and g1lr_ao
        let g0lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
        let g1lr_ao: ArrayView3<f64> = self.properties.grad_gamma_lr_ao().unwrap();

        // calculate contributions to the excited gradient
        let f: Array3<f64> = f_v(
            xpy_ao_2.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_p = f_lr(
            (&xpy_ao_2 + &xpy_ao_2.t()).view(),
            s,
            grad_s,
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_m = -f_lr(
            (&xmy_ao_2 - &xmy_ao_2.t()).view(),
            s,
            grad_s,
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );

        // assemble the excited gradient
        let mut nac: Array1<f64> = Array::zeros(3 * self.n_atoms);
        // gradH * (T + Z)
        nac = nac
            + grad_h
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(
                    &(t_vv - t_oo + z_ao)
                        .into_shape(self.n_orbs * self.n_orbs)
                        .unwrap(),
                );
        // - gradS * W
        nac = nac
            - grad_s
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // 2.0 * sum (X+Y) F (X+Y)
        nac = nac
            + 2.0
                * f.into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                    .unwrap()
                    .dot(
                        &xpy_ao_1
                            .view()
                            .into_shape(self.n_orbs * self.n_orbs)
                            .unwrap(),
                    );
        // - 0.5 * sum (X+Y) F_lr (X+Y)(X+Y)
        nac = nac
            - 0.5
                * flr_p
                    .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                    .unwrap()
                    .dot(&xpy_ao_1.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // - 0.5 * sum (X-Y) F_lr (X-Y)(X-Y)
        nac = nac
            - 0.5
                * flr_m
                    .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                    .unwrap()
                    .dot(
                        &xmy_ao_1
                            .view()
                            .into_shape(self.n_orbs * self.n_orbs)
                            .unwrap(),
                    );
        if state_1 == state_2 {
            return nac;
        } else {
            return (1.0 / omega_diff) * nac;
        }
    }

    pub fn nac_tddft_excited_no_lc(&self, state_1: usize, state_2: usize) -> Array1<f64> {
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
        let xmy_state_1: ArrayView2<f64> = xmy_state.slice(s![state_1, .., ..]);
        let xpy_state_1: ArrayView2<f64> = xpy_state.slice(s![state_1, .., ..]);
        // excitation energy of the state
        let omega_state_1: f64 = self.properties.ci_eigenvalue(state_1).unwrap();

        // take state specific values from the excitation vectors
        let xmy_state_2: ArrayView2<f64> = xmy_state.slice(s![state_2, .., ..]);
        let xpy_state_2: ArrayView2<f64> = xpy_state.slice(s![state_2, .., ..]);
        // excitation energy of the state
        let omega_state_2: f64 = self.properties.ci_eigenvalue(state_2).unwrap();
        let omega_avg: f64 = (omega_state_1 + omega_state_2) / 2.0;
        let omega_diff: f64 = (omega_state_1 - omega_state_2);

        // calculate the vectors u, v and t
        let u_ab: Array2<f64> = xpy_state_1.t().dot(&xmy_state_2)
            + xmy_state_1.t().dot(&xpy_state_2)
            + xpy_state_2.t().dot(&xmy_state_1)
            + xmy_state_2.t().dot(&xpy_state_1);
        let u_ij: Array2<f64> = xpy_state_1.dot(&xmy_state_2.t())
            + xmy_state_1.dot(&xpy_state_2.t())
            + xpy_state_2.dot(&xmy_state_1.t())
            + xmy_state_2.dot(&xpy_state_1.t());

        let v_ab: Array2<f64> = ei.dot(&xpy_state_1).t().dot(&xpy_state_2)
            + ei.dot(&xmy_state_1).t().dot(&xmy_state_2)
            + ei.dot(&xpy_state_2).t().dot(&xpy_state_1)
            + ei.dot(&xmy_state_2).t().dot(&xmy_state_1);
        let v_ij: Array2<f64> = xpy_state_1.dot(&ea).dot(&xpy_state_2.t())
            + xmy_state_1.dot(&ea).dot(&xmy_state_2.t())
            + xpy_state_2.dot(&ea).dot(&xpy_state_1.t())
            + xmy_state_2.dot(&ea).dot(&xmy_state_1.t());

        let t_ab: Array2<f64> = 0.25
            * (xpy_state_1.t().dot(&xpy_state_2)
                + xpy_state_2.t().dot(&xpy_state_1)
                + xmy_state_1.t().dot(&xmy_state_2)
                + xmy_state_2.t().dot(&xmy_state_1));
        let t_ij: Array2<f64> = 0.25
            * (xpy_state_1.dot(&xpy_state_2.t())
                + xpy_state_2.dot(&xpy_state_1.t())
                + xmy_state_1.dot(&xmy_state_2.t())
                + xmy_state_2.dot(&xmy_state_1.t()));

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
        let q_ij: Array2<f64> = 0.5 * omega_avg * u_ij - 0.5 * v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = 0.5 * omega_avg * u_ab + 0.5 * v_ab;

        // calculate q_ia
        let mut q_ia: Array2<f64> = 0.5
            * (xpy_state_1.dot(&h_plus_no_lr(g0, qtrans_vv, qtrans_ov, xpy_state_2).t())
                + xpy_state_2.dot(&h_plus_no_lr(g0, qtrans_vv, qtrans_ov, xpy_state_1).t()));
        q_ia = q_ia + h_plus_no_lr(g0, qtrans_ov, qtrans_vv, t_ab.view());
        q_ia = q_ia - h_plus_no_lr(g0, qtrans_ov, qtrans_oo, t_ij.view());

        // calculate q_ai
        let q_ai: Array2<f64> = 0.5
            * (xpy_state_1
                .t()
                .dot(&h_plus_no_lr(g0, qtrans_oo, qtrans_ov, xpy_state_2))
                + xpy_state_2
                    .t()
                    .dot(&h_plus_no_lr(g0, qtrans_oo, qtrans_ov, xpy_state_1)));

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
        let xpy_ao_1: Array2<f64> = orbs_occ.dot(&xpy_state_1.dot(&orbs_virt.t()));
        let xpy_ao_2: Array2<f64> = orbs_occ.dot(&xpy_state_2.dot(&orbs_virt.t()));

        // calculate contributions to the excited gradient
        let f: Array3<f64> = f_v(
            xpy_ao_2.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );

        // assemble the excited gradient
        let mut nac: Array1<f64> = Array::zeros(3 * self.n_atoms);
        // gradH * (T + Z)
        nac = nac
            + grad_h
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(
                    &(t_vv - t_oo + z_ao)
                        .into_shape(self.n_orbs * self.n_orbs)
                        .unwrap(),
                );
        // - gradS * W
        nac = nac
            - grad_s
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // 2.0 * sum (X+Y) F (X+Y)
        nac = nac
            + 2.0
                * f.into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                    .unwrap()
                    .dot(
                        &xpy_ao_1
                            .view()
                            .into_shape(self.n_orbs * self.n_orbs)
                            .unwrap(),
                    );

        if state_1 == state_2 {
            return nac;
        } else {
            return (1.0 / omega_diff) * nac;
        }
    }

    pub fn nac_tda_dft_excited(&self, state_1: usize, state_2: usize) -> Array1<f64> {
        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = self.occ_indices.iter().map(|&occ| orbe[occ]).collect();
        let orbe_virt: Array1<f64> = self.virt_indices.iter().map(|&virt| orbe[virt]).collect();

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

        let n_occ: usize = orbe_occ.len();
        let n_virt: usize = orbe_virt.len();

        // get the ci eigenvalues
        let omega_state: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
        let n_states: usize = omega_state.len();

        // take state specific values from the excitation vectors
        let x_state: ArrayView2<f64> = self.properties.ci_coefficients().unwrap();
        let x_state: Array3<f64> = x_state
            .t()
            .as_standard_layout()
            .to_owned()
            .into_shape([n_states, n_occ, n_virt])
            .unwrap();
        let x_state_1: ArrayView2<f64> = x_state.slice(s![state_1, .., ..]);
        let x_state_2: ArrayView2<f64> = x_state.slice(s![state_2, .., ..]);
        // excitation energy of the state
        let omega_state_1: f64 = self.properties.ci_eigenvalue(state_1).unwrap();

        // excitation energy of the state
        let omega_state_2: f64 = self.properties.ci_eigenvalue(state_2).unwrap();
        let omega_avg: f64 = (omega_state_1 + omega_state_2) / 2.0;
        let omega_diff: f64 = (omega_state_1 - omega_state_2);

        // calculate the vectors u, v and t
        let u_ab: Array2<f64> =
            2.0 * (x_state_1.t().dot(&x_state_2) + x_state_2.t().dot(&x_state_1));
        let u_ij: Array2<f64> =
            2.0 * (x_state_1.dot(&x_state_2.t()) + x_state_2.dot(&x_state_1.t()));
        let v_ab: Array2<f64> =
            2.0 * (ei.dot(&x_state_1).t().dot(&x_state_2) + ei.dot(&x_state_2).t().dot(&x_state_1));
        let v_ij: Array2<f64> =
            2.0 * (x_state_1.dot(&ea).dot(&x_state_2.t()) + x_state_2.dot(&ea).dot(&x_state_1.t()));

        let t_ab: Array2<f64> =
            0.5 * (x_state_1.t().dot(&x_state_2) + x_state_2.t().dot(&x_state_1));
        let t_ij: Array2<f64> =
            0.5 * (x_state_1.dot(&x_state_2.t()) + x_state_2.dot(&x_state_1.t()));

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
        let q_ij: Array2<f64> = 0.5 * omega_avg * u_ij - 0.5 * v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = 0.5 * omega_avg * u_ab + 0.5 * v_ab;

        // calculate q_ia
        // first term
        let mut q_ia: Array2<f64> =
            0.5 * x_state_1.dot(&hav.compute(g0, g0_lr, x_state_2, HplusType::QiaXpy).t());
        // second term
        q_ia =
            q_ia + 0.5 * x_state_2.dot(&hav.compute(g0, g0_lr, x_state_1, HplusType::QiaXpy).t());
        q_ia = q_ia + hav.compute(g0, g0_lr, t_ab.view(), HplusType::QiaTab);
        q_ia = q_ia - hav.compute(g0, g0_lr, t_ij.view(), HplusType::QiaTij);

        // calculate q_ai
        let mut q_ai: Array2<f64> = 0.5
            * x_state_1
                .t()
                .dot(&hav.compute(g0, g0_lr, x_state_2, HplusType::Qai));
        q_ai = q_ai
            + 0.5
                * x_state_2
                    .t()
                    .dot(&hav.compute(g0, g0_lr, x_state_1, HplusType::Qai));

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
        let f_dmd0: Array3<f64> = f_v(
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
        let x_ao_1: Array2<f64> = orbs_occ.dot(&x_state_1.dot(&orbs_virt.t()));
        let x_ao_2: Array2<f64> = orbs_occ.dot(&x_state_2.dot(&orbs_virt.t()));

        // set g0lr_ao and g1lr_ao
        let g0lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
        let g1lr_ao: ArrayView3<f64> = self.properties.grad_gamma_lr_ao().unwrap();

        // calculate contributions to the excited gradient
        let f: Array3<f64> = f_v(
            x_ao_2.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_p = f_lr(
            x_ao_2.t(),
            s,
            grad_s,
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );

        // assemble the excited gradient
        let mut nac: Array1<f64> = Array::zeros(3 * self.n_atoms);
        // gradH * (T + Z)
        nac = nac
            + grad_h
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(
                    &(t_vv - t_oo + z_ao)
                        .into_shape(self.n_orbs * self.n_orbs)
                        .unwrap(),
                );
        // - gradS * W
        nac = nac
            - grad_s
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // 2.0 * sum (X+Y) F (X+Y)
        nac = nac
            + 2.0
                * f.into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                    .unwrap()
                    .dot(&x_ao_1.view().into_shape(self.n_orbs * self.n_orbs).unwrap());
        // - 0.5 * sum (X+Y) F_lr (X+Y)(X+Y)
        nac = nac
            - flr_p
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&x_ao_1.into_shape(self.n_orbs * self.n_orbs).unwrap());
        if state_1 == state_2 {
            return nac;
        } else {
            return (1.0 / omega_diff) * nac;
        }
    }

    pub fn nac_tda_dft_excited_no_lc(&self, state_1: usize, state_2: usize) -> Array1<f64> {
        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = self.occ_indices.iter().map(|&occ| orbe[occ]).collect();
        let orbe_virt: Array1<f64> = self.virt_indices.iter().map(|&virt| orbe[virt]).collect();

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

        let n_occ: usize = orbe_occ.len();
        let n_virt: usize = orbe_virt.len();

        // get the ci eigenvalues
        let omega_state: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
        let n_states: usize = omega_state.len();

        // take state specific values from the excitation vectors
        let x_state: ArrayView2<f64> = self.properties.ci_coefficients().unwrap();
        let x_state: Array3<f64> = x_state
            .t()
            .as_standard_layout()
            .to_owned()
            .into_shape([n_states, n_occ, n_virt])
            .unwrap();
        let x_state_1: ArrayView2<f64> = x_state.slice(s![state_1, .., ..]);
        let x_state_2: ArrayView2<f64> = x_state.slice(s![state_2, .., ..]);
        // excitation energy of the state
        let omega_state_1: f64 = self.properties.ci_eigenvalue(state_1).unwrap();

        // excitation energy of the state
        let omega_state_2: f64 = self.properties.ci_eigenvalue(state_2).unwrap();
        let omega_avg: f64 = (omega_state_1 + omega_state_2) / 2.0;
        let omega_diff: f64 = (omega_state_1 - omega_state_2);

        // calculate the vectors u, v and t
        let u_ab: Array2<f64> =
            2.0 * (x_state_1.t().dot(&x_state_2) + x_state_2.t().dot(&x_state_1));
        let u_ij: Array2<f64> =
            2.0 * (x_state_1.dot(&x_state_2.t()) + x_state_2.dot(&x_state_1.t()));
        let v_ab: Array2<f64> =
            2.0 * (ei.dot(&x_state_1).t().dot(&x_state_2) + ei.dot(&x_state_2).t().dot(&x_state_1));
        let v_ij: Array2<f64> =
            2.0 * (x_state_1.dot(&ea).dot(&x_state_2.t()) + x_state_2.dot(&ea).dot(&x_state_1.t()));

        let t_ab: Array2<f64> =
            0.5 * (x_state_1.t().dot(&x_state_2) + x_state_2.t().dot(&x_state_1));
        let t_ij: Array2<f64> =
            0.5 * (x_state_1.dot(&x_state_2.t()) + x_state_2.dot(&x_state_1.t()));

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
        let q_ij: Array2<f64> = 0.5 * omega_avg * u_ij - 0.5 * v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = 0.5 * omega_avg * u_ab + 0.5 * v_ab;

        // calculate q_ia
        // first term
        let mut q_ia: Array2<f64> =
            0.5 * x_state_1.dot(&h_a_nolr(g0, qtrans_vv, qtrans_ov, x_state_2).t());
        // second term
        q_ia = q_ia + 0.5 * x_state_2.dot(&h_a_nolr(g0, qtrans_vv, qtrans_ov, x_state_1).t());
        q_ia = q_ia + h_a_nolr(g0, qtrans_ov, qtrans_vv, t_ab.view());
        q_ia = q_ia - h_a_nolr(g0, qtrans_ov, qtrans_oo, t_ij.view());

        // calculate q_ai
        let mut q_ai: Array2<f64> = 0.5
            * x_state_1
                .t()
                .dot(&h_a_nolr(g0, qtrans_oo, qtrans_ov, x_state_2));
        q_ai = q_ai
            + 0.5
                * x_state_2
                    .t()
                    .dot(&h_a_nolr(g0, qtrans_oo, qtrans_ov, x_state_1));

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
        let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia);

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
        let x_ao_1: Array2<f64> = orbs_occ.dot(&x_state_1.dot(&orbs_virt.t()));
        let x_ao_2: Array2<f64> = orbs_occ.dot(&x_state_2.dot(&orbs_virt.t()));

        // calculate contributions to the excited gradient
        let f: Array3<f64> = f_v(
            x_ao_2.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );

        // assemble the excited gradient
        let mut nac: Array1<f64> = Array::zeros(3 * self.n_atoms);
        // gradH * (T + Z)
        nac = nac
            + grad_h
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(
                    &(t_vv - t_oo + z_ao)
                        .into_shape(self.n_orbs * self.n_orbs)
                        .unwrap(),
                );
        // - gradS * W
        nac = nac
            - grad_s
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // 2.0 * sum (X+Y) F (X+Y)
        nac = nac
            + 2.0
                * f.into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                    .unwrap()
                    .dot(&x_ao_1.view().into_shape(self.n_orbs * self.n_orbs).unwrap());

        if state_1 == state_2 {
            return nac;
        } else {
            return (1.0 / omega_diff) * nac;
        }
    }

    pub fn get_nonadiabatic_coupling_vectors(&self, state_1: usize, state_2: usize) -> Array1<f64> {
        let vector: Array1<f64> = if self.config.excited.use_casida {
            if self.config.lc.long_range_correction {
                self.nac_tddft_excited(state_1, state_2)
            } else {
                self.nac_tddft_excited_no_lc(state_1, state_2)
            }
        } else {
            if self.config.lc.long_range_correction {
                self.nac_tda_dft_excited(state_1, state_2)
            } else {
                self.nac_tda_dft_excited_no_lc(state_1, state_2)
            }
        };
        vector
    }

    pub fn get_nonadiabatic_vector_coupling(
        &self,
        velocities: ArrayView2<f64>,
    ) -> (Array2<f64>, Vec<Array1<f64>>) {
        // transform the velocities to 1d
        let velocities_1d: ArrayView1<f64> = velocities.into_shape(3 * self.n_atoms).unwrap();
        // get number of excited states
        let nstates: usize = self.config.excited.nstates;

        // init coupling array
        let mut coupling: Array2<f64> = Array2::zeros([nstates, nstates]);
        // let mut coupling_vectors: Array3<f64> = Array3::zeros([nstates, nstates, 3 * self.n_atoms]);
        let mut coupling_vectors: Vec<Array1<f64>> = Vec::new();

        // get the old system
        let old_system = self.properties.old_system();
        if old_system.is_some() {
            let old_sys = old_system.unwrap();
            let old_vectors: Vec<Array1<f64>> = old_sys.old_nacv.clone().unwrap();
            let mut count: usize = 0;

            for i in (0..nstates) {
                for j in (0..nstates) {
                    if i < j {
                        let vector: Array1<f64> = self.get_nonadiabatic_coupling_vectors(i, j);
                        let normed_vector: Array1<f64> = &vector / (vector.norm());
                        // let old_vec: ArrayView1<f64> = old_vectors.slice(s![i, j, ..]);
                        let old_vec: ArrayView1<f64> = old_vectors[count].view();
                        let normed_old_vec: Array1<f64> = &old_vec / (old_vec.norm());
                        let sign: f64 = normed_old_vec.dot(&normed_vector);

                        // check for positive or negative sign
                        if sign > 0.0 {
                            // coupling_vectors.slice_mut(s![i, j, ..]).assign(&vector);
                            let val: f64 = vector.dot(&velocities_1d);
                            coupling_vectors.push(vector);

                            coupling[[i, j]] = val;
                            coupling[[j, i]] = -1.0 * val;
                        } else {
                            let vec_changed_sign: Array1<f64> = -1.0 * &vector;
                            // coupling_vectors
                            //     .slice_mut(s![i, j, ..])
                            //     .assign(&vec_changed_sign);
                            let val: f64 = vec_changed_sign.dot(&velocities_1d);
                            coupling_vectors.push(vec_changed_sign);

                            coupling[[i, j]] = val;
                            coupling[[j, i]] = -1.0 * val;
                        }
                    }
                    count += 1;
                }
            }
        } else {
            for i in (0..nstates) {
                for j in (0..nstates) {
                    if i < j {
                        let vector: Array1<f64> = self.get_nonadiabatic_coupling_vectors(i, j);
                        // coupling_vectors.slice_mut(s![i, j, ..]).assign(&vector);
                        let val: f64 = vector.dot(&velocities_1d);
                        coupling_vectors.push(vector);

                        coupling[[i, j]] = val;
                        coupling[[j, i]] = -1.0 * val;
                    }
                }
            }
        }

        return (coupling, coupling_vectors);
    }
}

impl Monomer<'_> {
    pub fn nac_tda_dft_excited(&self, state_1: usize, state_2: usize) -> Array1<f64> {
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

        // get the ci eigenvalues
        let omega_state: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
        let n_states: usize = omega_state.len();

        // take state specific values from the excitation vectors
        let x_state: ArrayView2<f64> = self.properties.ci_coefficients().unwrap();
        let x_state: Array3<f64> = x_state
            .t()
            .as_standard_layout()
            .to_owned()
            .into_shape([n_states, n_occ, n_virt])
            .unwrap();
        let x_state_1: ArrayView2<f64> = x_state.slice(s![state_1, .., ..]);
        let x_state_2: ArrayView2<f64> = x_state.slice(s![state_2, .., ..]);
        // excitation energy of the state
        let omega_state_1: f64 = self.properties.ci_eigenvalue(state_1).unwrap();

        // excitation energy of the state
        let omega_state_2: f64 = self.properties.ci_eigenvalue(state_2).unwrap();
        let omega_avg: f64 = (omega_state_1 + omega_state_2) / 2.0;
        let omega_diff: f64 = (omega_state_1 - omega_state_2);

        // calculate the vectors u, v and t
        let u_ab: Array2<f64> =
            2.0 * (x_state_1.t().dot(&x_state_2) + x_state_2.t().dot(&x_state_1));
        let u_ij: Array2<f64> =
            2.0 * (x_state_1.dot(&x_state_2.t()) + x_state_2.dot(&x_state_1.t()));
        let v_ab: Array2<f64> =
            2.0 * (ei.dot(&x_state_1).t().dot(&x_state_2) + ei.dot(&x_state_2).t().dot(&x_state_1));
        let v_ij: Array2<f64> =
            2.0 * (x_state_1.dot(&ea).dot(&x_state_2.t()) + x_state_2.dot(&ea).dot(&x_state_1.t()));

        let t_ab: Array2<f64> =
            0.5 * (x_state_1.t().dot(&x_state_2) + x_state_2.t().dot(&x_state_1));
        let t_ij: Array2<f64> =
            0.5 * (x_state_1.dot(&x_state_2.t()) + x_state_2.dot(&x_state_1.t()));

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
        let q_ij: Array2<f64> = 0.5 * omega_avg * u_ij - 0.5 * v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = 0.5 * omega_avg * u_ab + 0.5 * v_ab;

        // calculate q_ia
        // first term
        let mut q_ia: Array2<f64> =
            0.5 * x_state_1.dot(&hav.compute(g0, g0_lr, x_state_2, HplusType::QiaXpy).t());
        // second term
        q_ia =
            q_ia + 0.5 * x_state_2.dot(&hav.compute(g0, g0_lr, x_state_1, HplusType::QiaXpy).t());
        q_ia = q_ia + hav.compute(g0, g0_lr, t_ab.view(), HplusType::QiaTab);
        q_ia = q_ia - hav.compute(g0, g0_lr, t_ij.view(), HplusType::QiaTij);

        // calculate q_ai
        let mut q_ai: Array2<f64> = 0.5
            * x_state_1
                .t()
                .dot(&hav.compute(g0, g0_lr, x_state_2, HplusType::Qai));
        q_ai = q_ai
            + 0.5
                * x_state_2
                    .t()
                    .dot(&hav.compute(g0, g0_lr, x_state_1, HplusType::Qai));

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
        let x_ao_1: Array2<f64> = orbs_occ.dot(&x_state_1.dot(&orbs_virt.t()));
        let x_ao_2: Array2<f64> = orbs_occ.dot(&x_state_2.dot(&orbs_virt.t()));

        // calculate contributions to the excited gradient
        let f: Array3<f64> = f_v(
            x_ao_2.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_p = f_lr(
            x_ao_2.t(),
            s,
            grad_s,
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );

        // assemble the excited gradient
        let mut nac: Array1<f64> = Array::zeros(3 * self.n_atoms);
        // gradH * (T + Z)
        nac = nac
            + grad_h
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(
                    &(t_vv - t_oo + z_ao)
                        .into_shape(self.n_orbs * self.n_orbs)
                        .unwrap(),
                );
        // - gradS * W
        nac = nac
            - grad_s
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // 2.0 * sum (X+Y) F (X+Y)
        nac = nac
            + 2.0
                * f.into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                    .unwrap()
                    .dot(&x_ao_1.view().into_shape(self.n_orbs * self.n_orbs).unwrap());
        // - 0.5 * sum (X+Y) F_lr (X+Y)(X+Y)
        nac = nac
            - flr_p
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&x_ao_1.into_shape(self.n_orbs * self.n_orbs).unwrap());
        if state_1 == state_2 {
            return nac;
        } else {
            return (1.0 / omega_diff) * nac;
        }
    }

    pub fn nac_tddft_excited(&mut self, state_1: usize, state_2: usize) -> Array1<f64> {
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

        // take state specific values from the excitation vectors
        let xmy_state: ArrayView3<f64> = self.properties.xmy().unwrap();
        let xpy_state: ArrayView3<f64> = self.properties.xpy().unwrap();
        let xmy_state_1: ArrayView2<f64> = xmy_state.slice(s![state_1, .., ..]);
        let xpy_state_1: ArrayView2<f64> = xpy_state.slice(s![state_1, .., ..]);
        // excitation energy of the state
        let omega_state_1: f64 = self.properties.ci_eigenvalue(state_1).unwrap();

        // take state specific values from the excitation vectors
        let xmy_state_2: ArrayView2<f64> = xmy_state.slice(s![state_2, .., ..]);
        let xpy_state_2: ArrayView2<f64> = xpy_state.slice(s![state_2, .., ..]);
        // excitation energy of the state
        let omega_state_2: f64 = self.properties.ci_eigenvalue(state_2).unwrap();
        let omega_avg: f64 = (omega_state_1 + omega_state_2) / 2.0;
        let omega_diff: f64 = (omega_state_1 - omega_state_2);

        // calculate the vectors u, v and t
        let u_ab: Array2<f64> = xpy_state_1.t().dot(&xmy_state_2)
            + xmy_state_1.t().dot(&xpy_state_2)
            + xpy_state_2.t().dot(&xmy_state_1)
            + xmy_state_2.t().dot(&xpy_state_1);
        let u_ij: Array2<f64> = xpy_state_1.dot(&xmy_state_2.t())
            + xmy_state_1.dot(&xpy_state_2.t())
            + xpy_state_2.dot(&xmy_state_1.t())
            + xmy_state_2.dot(&xpy_state_1.t());

        let v_ab: Array2<f64> = ei.dot(&xpy_state_1).t().dot(&xpy_state_2)
            + ei.dot(&xmy_state_1).t().dot(&xmy_state_2)
            + ei.dot(&xpy_state_2).t().dot(&xpy_state_1)
            + ei.dot(&xmy_state_2).t().dot(&xmy_state_1);
        let v_ij: Array2<f64> = xpy_state_1.dot(&ea).dot(&xpy_state_2.t())
            + xmy_state_1.dot(&ea).dot(&xmy_state_2.t())
            + xpy_state_2.dot(&ea).dot(&xpy_state_1.t())
            + xmy_state_2.dot(&ea).dot(&xmy_state_1.t());

        let t_ab: Array2<f64> = 0.25
            * (xpy_state_1.t().dot(&xpy_state_2)
                + xpy_state_2.t().dot(&xpy_state_1)
                + xmy_state_1.t().dot(&xmy_state_2)
                + xmy_state_2.t().dot(&xmy_state_1));
        let t_ij: Array2<f64> = 0.25
            * (xpy_state_1.dot(&xpy_state_2.t())
                + xpy_state_2.dot(&xpy_state_1.t())
                + xmy_state_1.dot(&xmy_state_2.t())
                + xmy_state_2.dot(&xmy_state_1.t()));

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
        let q_ij: Array2<f64> = 0.5 * omega_avg * u_ij - 0.5 * v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = 0.5 * omega_avg * u_ab + 0.5 * v_ab;

        // calculate q_ia
        // first term
        let mut q_ia: Array2<f64> =
            0.5 * xpy_state_1.dot(&hplus.compute(g0, g0_lr, xpy_state_2, HplusType::QiaXpy).t());
        // second term
        q_ia = q_ia
            + 0.5 * xpy_state_2.dot(&hplus.compute(g0, g0_lr, xpy_state_1, HplusType::QiaXpy).t());
        // first term
        q_ia = q_ia
            + 0.5
                * xmy_state_1.dot(
                    &h_minus(
                        g0_lr,
                        qtrans_vv,
                        qtrans_vo.view(),
                        qtrans_vo.view(),
                        qtrans_vv,
                        xmy_state_2,
                    )
                    .t(),
                );
        // second term
        q_ia = q_ia
            + 0.5
                * xmy_state_2.dot(
                    &h_minus(
                        g0_lr,
                        qtrans_vv,
                        qtrans_vo.view(),
                        qtrans_vo.view(),
                        qtrans_vv,
                        xmy_state_1,
                    )
                    .t(),
                );
        q_ia = q_ia + hplus.compute(g0, g0_lr, t_ab.view(), HplusType::QiaTab);
        q_ia = q_ia - hplus.compute(g0, g0_lr, t_ij.view(), HplusType::QiaTij);

        // calculate q_ai
        let mut q_ai: Array2<f64> = 0.5
            * xpy_state_1
                .t()
                .dot(&hplus.compute(g0, g0_lr, xpy_state_2, HplusType::Qai));
        q_ai = q_ai
            + 0.5
                * xpy_state_2
                    .t()
                    .dot(&hplus.compute(g0, g0_lr, xpy_state_1, HplusType::Qai));
        q_ai = q_ai
            + 0.5
                * xmy_state_1.t().dot(&h_minus(
                    g0_lr,
                    qtrans_ov,
                    qtrans_oo,
                    qtrans_oo,
                    qtrans_ov,
                    xmy_state_2,
                ));
        q_ai = q_ai
            + 0.5
                * xmy_state_2.t().dot(&h_minus(
                    g0_lr,
                    qtrans_ov,
                    qtrans_oo,
                    qtrans_oo,
                    qtrans_ov,
                    xmy_state_1,
                ));

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
        let f_dmd0: Array3<f64> = f_v(
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
        let xpy_ao_1: Array2<f64> = orbs_occ.dot(&xpy_state_1.dot(&orbs_virt.t()));
        let xpy_ao_2: Array2<f64> = orbs_occ.dot(&xpy_state_2.dot(&orbs_virt.t()));
        let xmy_ao_1: Array2<f64> = orbs_occ.dot(&xmy_state_1.dot(&orbs_virt.t()));
        let xmy_ao_2: Array2<f64> = orbs_occ.dot(&xmy_state_2.dot(&orbs_virt.t()));

        // set g0lr_ao and g1lr_ao
        let g0lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
        let g1lr_ao: ArrayView3<f64> = self.properties.grad_gamma_lr_ao().unwrap();

        // calculate contributions to the excited gradient
        let f: Array3<f64> = f_v(
            xpy_ao_2.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_p = f_lr(
            (&xpy_ao_2 + &xpy_ao_2.t()).view(),
            s,
            grad_s,
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_m = -f_lr(
            (&xmy_ao_2 - &xmy_ao_2.t()).view(),
            s,
            grad_s,
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );

        // assemble the excited gradient
        let mut nac: Array1<f64> = Array::zeros(3 * self.n_atoms);
        // gradH * (T + Z)
        nac = nac
            + grad_h
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(
                    &(t_vv - t_oo + z_ao)
                        .into_shape(self.n_orbs * self.n_orbs)
                        .unwrap(),
                );
        // - gradS * W
        nac = nac
            - grad_s
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // 2.0 * sum (X+Y) F (X+Y)
        nac = nac
            + 2.0
                * f.into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                    .unwrap()
                    .dot(
                        &xpy_ao_1
                            .view()
                            .into_shape(self.n_orbs * self.n_orbs)
                            .unwrap(),
                    );
        // - 0.5 * sum (X+Y) F_lr (X+Y)(X+Y)
        nac = nac
            - 0.5
                * flr_p
                    .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                    .unwrap()
                    .dot(&xpy_ao_1.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // - 0.5 * sum (X-Y) F_lr (X-Y)(X-Y)
        nac = nac
            - 0.5
                * flr_m
                    .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                    .unwrap()
                    .dot(
                        &xmy_ao_1
                            .view()
                            .into_shape(self.n_orbs * self.n_orbs)
                            .unwrap(),
                    );
        if state_1 == state_2 {
            return nac;
        } else {
            return (1.0 / omega_diff) * nac;
        }
    }
}

impl Pair<'_> {
    pub fn prepare_charge_transfer_nacvs(
        &mut self,
        atoms: &[Atom],
        m_i: &Monomer,
        m_j: &Monomer,
        ct_state: &ChargeTransferPair,
        ct_state_2: &ChargeTransferPair,
    ) {
        // check if occ and virt indices exist
        let mut occ_indices: Vec<usize> = Vec::new();
        let mut virt_indices: Vec<usize> = Vec::new();
        if (self.properties.contains_key("occ_indices") == false)
            || (self.properties.contains_key("virt_indices") == true)
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
        let nocc: usize = occ_indices.len();
        let nvirt: usize = virt_indices.len();
        let nocc_i: usize = m_i.properties.occ_indices().unwrap().len();
        let nocc_j: usize = m_j.properties.occ_indices().unwrap().len();

        // set ct_energy
        let cis_energy: Array1<f64> =
            Array::from(vec![ct_state.state_energy, ct_state_2.state_energy]);
        // save in properties
        self.properties.set_ci_eigenvalues(cis_energy);

        // get the overlap matrices between the monomers and the dimer
        let s_i_ij: ArrayView2<f64> = self.properties.s_i_ij().unwrap();
        let s_j_ij: ArrayView2<f64> = self.properties.s_j_ij().unwrap();

        // if the hole is on fragment I
        if m_i.index == ct_state.m_h {
            // reduce overlap matrices to occupied and virtual blocks
            let s_i_ij_occ: ArrayView2<f64> = s_i_ij.slice(s![..nocc_i, ..nocc]);
            let s_j_ij_virt: ArrayView2<f64> = s_j_ij.slice(s![nocc_j.., nocc..]);

            // transform the CT matrix using the reduced overlap matrices between the monomers
            // and the dimer
            let transformed_ct_coeff: Array2<f64> =
                s_i_ij_occ.t().dot(&ct_state.eigenvectors.dot(&s_j_ij_virt));
            let transformed_ct_coeff_2: Array2<f64> = s_i_ij_occ
                .t()
                .dot(&ct_state_2.eigenvectors.dot(&s_j_ij_virt));

            // set cis coefficient for the CT transition
            let mut cis_coeff: Array3<f64> = Array3::zeros([2, nocc, nvirt]);
            cis_coeff
                .slice_mut(s![0, .., ..])
                .assign(&transformed_ct_coeff);
            cis_coeff
                .slice_mut(s![1, .., ..])
                .assign(&transformed_ct_coeff_2);

            // save in properties
            self.properties
                .set_ci_coefficients(cis_coeff.into_shape([1, nocc * nvirt]).unwrap());
        }
        // if the hole is on fragment J
        else {
            // reduce overlap matrices to occupied and virtual blocks
            let s_i_ij_virt: ArrayView2<f64> = s_i_ij.slice(s![nocc_i.., nocc..]);
            let s_j_ij_occ: ArrayView2<f64> = s_j_ij.slice(s![..nocc_j, ..nocc]);

            // transform the CT matrix using the reduced overlap matrices between the monomers
            // and the dimer
            let transformed_ct_coeff: Array2<f64> =
                s_j_ij_occ.t().dot(&ct_state.eigenvectors.dot(&s_i_ij_virt));
            let transformed_ct_coeff_2: Array2<f64> = s_j_ij_occ
                .t()
                .dot(&ct_state_2.eigenvectors.dot(&s_i_ij_virt));

            // set cis coefficient for the CT transition
            let mut cis_coeff: Array3<f64> = Array3::zeros([2, nocc, nvirt]);
            cis_coeff
                .slice_mut(s![0, .., ..])
                .assign(&transformed_ct_coeff);
            cis_coeff
                .slice_mut(s![1, .., ..])
                .assign(&transformed_ct_coeff_2);

            // save in properties
            self.properties
                .set_ci_coefficients(cis_coeff.into_shape([1, nocc * nvirt]).unwrap());
        }
    }

    pub fn nac_tda_dft_excited(&self, state_1: usize, state_2: usize) -> Array1<f64> {
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

        // get the ci eigenvalues
        let omega_state: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
        let n_states: usize = omega_state.len();

        // take state specific values from the excitation vectors
        let x_state: ArrayView2<f64> = self.properties.ci_coefficients().unwrap();
        let x_state: Array3<f64> = x_state
            .t()
            .as_standard_layout()
            .to_owned()
            .into_shape([n_states, n_occ, n_virt])
            .unwrap();
        let x_state_1: ArrayView2<f64> = x_state.slice(s![state_1, .., ..]);
        let x_state_2: ArrayView2<f64> = x_state.slice(s![state_2, .., ..]);
        // excitation energy of the state
        let omega_state_1: f64 = self.properties.ci_eigenvalue(state_1).unwrap();

        // excitation energy of the state
        let omega_state_2: f64 = self.properties.ci_eigenvalue(state_2).unwrap();
        let omega_avg: f64 = (omega_state_1 + omega_state_2) / 2.0;
        let omega_diff: f64 = (omega_state_1 - omega_state_2);

        // calculate the vectors u, v and t
        let u_ab: Array2<f64> =
            2.0 * (x_state_1.t().dot(&x_state_2) + x_state_2.t().dot(&x_state_1));
        let u_ij: Array2<f64> =
            2.0 * (x_state_1.dot(&x_state_2.t()) + x_state_2.dot(&x_state_1.t()));
        let v_ab: Array2<f64> =
            2.0 * (ei.dot(&x_state_1).t().dot(&x_state_2) + ei.dot(&x_state_2).t().dot(&x_state_1));
        let v_ij: Array2<f64> =
            2.0 * (x_state_1.dot(&ea).dot(&x_state_2.t()) + x_state_2.dot(&ea).dot(&x_state_1.t()));

        let t_ab: Array2<f64> =
            0.5 * (x_state_1.t().dot(&x_state_2) + x_state_2.t().dot(&x_state_1));
        let t_ij: Array2<f64> =
            0.5 * (x_state_1.dot(&x_state_2.t()) + x_state_2.dot(&x_state_1.t()));

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
        let q_ij: Array2<f64> = 0.5 * omega_avg * u_ij - 0.5 * v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = 0.5 * omega_avg * u_ab + 0.5 * v_ab;

        // calculate q_ia
        // first term
        let mut q_ia: Array2<f64> =
            0.5 * x_state_1.dot(&hav.compute(g0, g0_lr, x_state_2, HplusType::QiaXpy).t());
        // second term
        q_ia =
            q_ia + 0.5 * x_state_2.dot(&hav.compute(g0, g0_lr, x_state_1, HplusType::QiaXpy).t());
        q_ia = q_ia + hav.compute(g0, g0_lr, t_ab.view(), HplusType::QiaTab);
        q_ia = q_ia - hav.compute(g0, g0_lr, t_ij.view(), HplusType::QiaTij);

        // calculate q_ai
        let mut q_ai: Array2<f64> = 0.5
            * x_state_1
                .t()
                .dot(&hav.compute(g0, g0_lr, x_state_2, HplusType::Qai));
        q_ai = q_ai
            + 0.5
                * x_state_2
                    .t()
                    .dot(&hav.compute(g0, g0_lr, x_state_1, HplusType::Qai));

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
        let g0lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
        let g1lr_ao: ArrayView3<f64> = self.properties.grad_gamma_lr_ao().unwrap();

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
        let x_ao_1: Array2<f64> = orbs_occ.dot(&x_state_1.dot(&orbs_virt.t()));
        let x_ao_2: Array2<f64> = orbs_occ.dot(&x_state_2.dot(&orbs_virt.t()));

        // calculate contributions to the excited gradient
        let f: Array3<f64> = f_v(
            x_ao_2.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_p = f_lr(
            x_ao_2.t(),
            s,
            grad_s,
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );

        // assemble the excited gradient
        let mut nac: Array1<f64> = Array::zeros(3 * self.n_atoms);
        // gradH * (T + Z)
        nac = nac
            + grad_h
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(
                    &(t_vv - t_oo + z_ao)
                        .into_shape(self.n_orbs * self.n_orbs)
                        .unwrap(),
                );
        // - gradS * W
        nac = nac
            - grad_s
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // 2.0 * sum (X+Y) F (X+Y)
        nac = nac
            + 2.0
                * f.into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                    .unwrap()
                    .dot(&x_ao_1.view().into_shape(self.n_orbs * self.n_orbs).unwrap());
        // - 0.5 * sum (X+Y) F_lr (X+Y)(X+Y)
        nac = nac
            - flr_p
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&x_ao_1.into_shape(self.n_orbs * self.n_orbs).unwrap());
        if state_1 == state_2 {
            return nac;
        } else {
            return (1.0 / omega_diff) * nac;
        }
    }
}

impl ESDPair<'_> {
    pub fn prepare_charge_transfer_nacvs(
        &mut self,
        atoms: &[Atom],
        m_i: &Monomer,
        m_j: &Monomer,
        ct_state: &ChargeTransferPair,
        ct_state_2: &ChargeTransferPair,
    ) {
        // check if occ and virt indices exist
        let mut occ_indices: Vec<usize> = Vec::new();
        let mut virt_indices: Vec<usize> = Vec::new();
        if (self.properties.contains_key("occ_indices") == false)
            || (self.properties.contains_key("virt_indices") == true)
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
        let nocc: usize = occ_indices.len();
        let nvirt: usize = virt_indices.len();
        let nocc_i: usize = m_i.properties.occ_indices().unwrap().len();
        let nocc_j: usize = m_j.properties.occ_indices().unwrap().len();

        // set ct_energy
        let cis_energy: Array1<f64> =
            Array::from(vec![ct_state.state_energy, ct_state_2.state_energy]);
        // save in properties
        self.properties.set_ci_eigenvalues(cis_energy);

        // get overlap matrix of the esd pair from the properties
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        // Reference to the MO coefficients of monomer I.
        let orbs_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
        // Reference to the MO coefficients of monomer J.
        let orbs_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();
        // Reference to the MO coefficients of the pair IJ.
        let orbs_ij: ArrayView2<f64> = self.properties.orbs().unwrap();
        // Overlap between orbitals of monomer I and dimer IJ.
        let s_i_ij: Array2<f64> = (orbs_i.t().dot(&s.slice(s![0..m_i.n_orbs, ..]))).dot(&orbs_ij);
        // Overlap between orbitals of monomer J and dimer IJ.
        let s_j_ij: Array2<f64> = (orbs_j.t().dot(&s.slice(s![m_i.n_orbs.., ..]))).dot(&orbs_ij);

        // if the hole is on fragment I
        if m_i.index == ct_state.m_h {
            // reduce overlap matrices to occupied and virtual blocks
            let s_i_ij_occ: ArrayView2<f64> = s_i_ij.slice(s![..nocc_i, ..nocc]);
            let s_j_ij_virt: ArrayView2<f64> = s_j_ij.slice(s![nocc_j.., nocc..]);

            // transform the CT matrix using the reduced overlap matrices between the monomers
            // and the dimer
            let transformed_ct_coeff: Array2<f64> =
                s_i_ij_occ.t().dot(&ct_state.eigenvectors.dot(&s_j_ij_virt));

            // set cis coefficient for the CT transition
            let mut cis_coeff: Array3<f64> = Array3::zeros([1, nocc, nvirt]);
            cis_coeff
                .slice_mut(s![0, .., ..])
                .assign(&transformed_ct_coeff);
            // save in properties
            self.properties
                .set_ci_coefficients(cis_coeff.into_shape([1, nocc * nvirt]).unwrap());
        }
        // if the hole is on fragment J
        else {
            // reduce overlap matrices to occupied and virtual blocks
            let s_i_ij_virt: ArrayView2<f64> = s_i_ij.slice(s![nocc_i.., nocc..]);
            let s_j_ij_occ: ArrayView2<f64> = s_j_ij.slice(s![..nocc_j, ..nocc]);

            // transform the CT matrix using the reduced overlap matrices between the monomers
            // and the dimer
            let transformed_ct_coeff: Array2<f64> =
                s_j_ij_occ.t().dot(&ct_state.eigenvectors.dot(&s_i_ij_virt));

            // set cis coefficient for the CT transition
            let mut cis_coeff: Array3<f64> = Array3::zeros([1, nocc, nvirt]);
            cis_coeff
                .slice_mut(s![0, .., ..])
                .assign(&transformed_ct_coeff);
            // save in properties
            self.properties
                .set_ci_coefficients(cis_coeff.into_shape([1, nocc * nvirt]).unwrap());
        }
    }

    pub fn nac_tda_dft_excited(&self, state_1: usize, state_2: usize) -> Array1<f64> {
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

        // get the ci eigenvalues
        let omega_state: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
        let n_states: usize = omega_state.len();

        // take state specific values from the excitation vectors
        let x_state: ArrayView2<f64> = self.properties.ci_coefficients().unwrap();
        let x_state: Array3<f64> = x_state
            .t()
            .as_standard_layout()
            .to_owned()
            .into_shape([n_states, n_occ, n_virt])
            .unwrap();
        let x_state_1: ArrayView2<f64> = x_state.slice(s![state_1, .., ..]);
        let x_state_2: ArrayView2<f64> = x_state.slice(s![state_2, .., ..]);
        // excitation energy of the state
        let omega_state_1: f64 = self.properties.ci_eigenvalue(state_1).unwrap();

        // excitation energy of the state
        let omega_state_2: f64 = self.properties.ci_eigenvalue(state_2).unwrap();
        let omega_avg: f64 = (omega_state_1 + omega_state_2) / 2.0;
        let omega_diff: f64 = (omega_state_1 - omega_state_2);

        // calculate the vectors u, v and t
        let u_ab: Array2<f64> =
            2.0 * (x_state_1.t().dot(&x_state_2) + x_state_2.t().dot(&x_state_1));
        let u_ij: Array2<f64> =
            2.0 * (x_state_1.dot(&x_state_2.t()) + x_state_2.dot(&x_state_1.t()));
        let v_ab: Array2<f64> =
            2.0 * (ei.dot(&x_state_1).t().dot(&x_state_2) + ei.dot(&x_state_2).t().dot(&x_state_1));
        let v_ij: Array2<f64> =
            2.0 * (x_state_1.dot(&ea).dot(&x_state_2.t()) + x_state_2.dot(&ea).dot(&x_state_1.t()));

        let t_ab: Array2<f64> =
            0.5 * (x_state_1.t().dot(&x_state_2) + x_state_2.t().dot(&x_state_1));
        let t_ij: Array2<f64> =
            0.5 * (x_state_1.dot(&x_state_2.t()) + x_state_2.dot(&x_state_1.t()));

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
        let q_ij: Array2<f64> = 0.5 * omega_avg * u_ij - 0.5 * v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = 0.5 * omega_avg * u_ab + 0.5 * v_ab;

        // calculate q_ia
        // first term
        let mut q_ia: Array2<f64> =
            0.5 * x_state_1.dot(&hav.compute(g0, g0_lr, x_state_2, HplusType::QiaXpy).t());
        // second term
        q_ia =
            q_ia + 0.5 * x_state_2.dot(&hav.compute(g0, g0_lr, x_state_1, HplusType::QiaXpy).t());
        q_ia = q_ia + hav.compute(g0, g0_lr, t_ab.view(), HplusType::QiaTab);
        q_ia = q_ia - hav.compute(g0, g0_lr, t_ij.view(), HplusType::QiaTij);

        // calculate q_ai
        let mut q_ai: Array2<f64> = 0.5
            * x_state_1
                .t()
                .dot(&hav.compute(g0, g0_lr, x_state_2, HplusType::Qai));
        q_ai = q_ai
            + 0.5
                * x_state_2
                    .t()
                    .dot(&hav.compute(g0, g0_lr, x_state_1, HplusType::Qai));

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
        let x_ao_1: Array2<f64> = orbs_occ.dot(&x_state_1.dot(&orbs_virt.t()));
        let x_ao_2: Array2<f64> = orbs_occ.dot(&x_state_2.dot(&orbs_virt.t()));

        // calculate contributions to the excited gradient
        let f: Array3<f64> = f_v(
            x_ao_2.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_p = f_lr(
            x_ao_2.t(),
            s,
            grad_s,
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );

        // assemble the excited gradient
        let mut nac: Array1<f64> = Array::zeros(3 * self.n_atoms);
        // gradH * (T + Z)
        nac = nac
            + grad_h
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(
                    &(t_vv - t_oo + z_ao)
                        .into_shape(self.n_orbs * self.n_orbs)
                        .unwrap(),
                );
        // - gradS * W
        nac = nac
            - grad_s
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // 2.0 * sum (X+Y) F (X+Y)
        nac = nac
            + 2.0
                * f.into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                    .unwrap()
                    .dot(&x_ao_1.view().into_shape(self.n_orbs * self.n_orbs).unwrap());
        // - 0.5 * sum (X+Y) F_lr (X+Y)(X+Y)
        nac = nac
            - flr_p
                .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
                .unwrap()
                .dot(&x_ao_1.into_shape(self.n_orbs * self.n_orbs).unwrap());
        if state_1 == state_2 {
            return nac;
        } else {
            return (1.0 / omega_diff) * nac;
        }
    }
}

impl SuperSystem<'_> {
    pub fn get_nonadiabatic_vector_coupling(
        &self,
        velocities: ArrayView2<f64>,
        state_coefficients: ArrayView1<f64>,
        threshold: f64,
    ) -> (Array2<f64>, HashMap<(usize, usize), Array1<f64>>) {
        // transform the velocities to 1d
        let velocities_1d: ArrayView1<f64> = velocities.into_shape(3 * self.atoms.len()).unwrap();
        // get the basis states
        let basis_states = self.properties.basis_states().unwrap();

        let old_system = if !self.properties.old_supersystem().is_none() {
            self.properties.old_supersystem().unwrap().clone()
        } else {
            println!("Create old system at first step!");
            OldSupersystem::new(self)
        };

        // storage for nac vectors
        let mut nacv_storage: HashMap<(usize, usize), Array1<f64>> = HashMap::new();
        let old_nacv: &HashMap<(usize, usize), Array1<f64>> = &old_system.nacv_storage;

        // empty coupling array
        let mut coupling: Array2<f64> =
            Array2::zeros([basis_states.len() + 1, basis_states.len() + 1]);

        for (i, state_i) in basis_states.iter().enumerate() {
            for (j, state_j) in basis_states.iter().enumerate() {
                if i < j {
                    let coefficient_i = state_coefficients[i + 1];
                    let coefficient_j = state_coefficients[j + 1];
                    if coefficient_i > threshold || coefficient_j > threshold {
                        // get the old nacv
                        let nacv_entry: Option<&Array1<f64>> = old_nacv.get(&(i, j));

                        let tmp = self.nonadiabatic_vector_coupling_state(
                            state_i,
                            state_j,
                            velocities_1d,
                            nacv_entry,
                        );
                        coupling[[i + 1, j + 1]] = tmp.0;
                        coupling[[j + 1, i + 1]] = -tmp.0;

                        if tmp.1.is_some() {
                            nacv_storage.insert((i, j), tmp.1.unwrap());
                        }
                    }
                }
            }
        }

        return (coupling, nacv_storage);
    }

    pub fn nonadiabatic_vector_coupling_state(
        &self,
        lhs: &ReducedBasisState,
        rhs: &ReducedBasisState,
        velocities: ArrayView1<f64>,
        old_nacv: Option<&Array1<f64>>,
    ) -> (f64, Option<Array1<f64>>) {
        match (lhs, rhs) {
            // coupling between two LE states.
            (ReducedBasisState::LE(ref a), ReducedBasisState::LE(ref b)) => {
                if a.monomer_index == b.monomer_index {
                    // get the monomer
                    let m_i: &Monomer = &self.monomers[a.monomer_index];

                    // get the velocities of the monomer
                    let velocities: ArrayView1<f64> = velocities.slice(s![m_i.slice.grad]);

                    // get the nonadibatic coupling vectors
                    let nacv: Array1<f64> = if old_nacv.is_some() {
                        // old nacv
                        let old_vec = old_nacv.unwrap();
                        let vector = m_i.nac_tda_dft_excited(a.state_index, b.state_index);

                        let normed_vector: Array1<f64> = &vector / (vector.norm());
                        let normed_old_vec: Array1<f64> = old_vec / (old_vec.norm());
                        let sign: f64 = normed_old_vec.dot(&normed_vector);

                        if sign > 0.0 {
                            vector
                        } else {
                            -1.0 * &vector
                        }
                    } else {
                        // calculate the vector coupling
                        m_i.nac_tda_dft_excited(a.state_index, b.state_index)
                    };

                    // dot product between nac and velocities
                    let nac_val: f64 = nacv.dot(&velocities);

                    (nac_val, Some(nacv))
                } else {
                    (0.0, None)
                }
            }
            // coupling between LE and CT state.
            (ReducedBasisState::LE(ref a), ReducedBasisState::CT(ref b)) => (0.0, None),
            // coupling between CT and LE state.
            (ReducedBasisState::CT(ref a), ReducedBasisState::LE(ref b)) => (0.0, None),
            // coupling between CT and CT
            (ReducedBasisState::CT(ref a), ReducedBasisState::CT(ref b)) => {
                if a.m_h == b.m_h && a.m_l == b.m_l {
                    // get monomers
                    let m_i: &Monomer = &self.monomers[a.m_h];
                    let m_j: &Monomer = &self.monomers[a.m_l];

                    // get the velocities of the monomers
                    let velocities_i: ArrayView1<f64> = velocities.slice(s![m_i.slice.grad]);
                    let velocities_j: ArrayView1<f64> = velocities.slice(s![m_j.slice.grad]);

                    let mut velocities: Array1<f64> =
                        Array1::zeros(velocities_i.len() + velocities_j.len());
                    // append the velocities
                    if m_i.index < m_j.index {
                        velocities
                            .slice_mut(s![..velocities_i.len()])
                            .assign(&velocities_i);
                        velocities
                            .slice_mut(s![velocities_i.len()..])
                            .assign(&velocities_j);
                    } else {
                        velocities
                            .slice_mut(s![..velocities_j.len()])
                            .assign(&velocities_j);
                        velocities
                            .slice_mut(s![velocities_j.len()..])
                            .assign(&velocities_i);
                    }

                    // get the nonadiabtic coupling vectors
                    let nacv: Array1<f64> = if old_nacv.is_some() {
                        // old nacv
                        let old_vec = old_nacv.unwrap();
                        let vector = self.charge_transfer_nacvs(a, b);

                        let normed_vector: Array1<f64> = &vector / (vector.norm());
                        let normed_old_vec: Array1<f64> = old_vec / (old_vec.norm());
                        let sign: f64 = normed_old_vec.dot(&normed_vector);

                        if sign > 0.0 {
                            vector
                        } else {
                            -1.0 * &vector
                        }
                    } else {
                        self.charge_transfer_nacvs(a, b)
                    };

                    // get the nonadibatic coupling value
                    let nac_val: f64 = nacv.dot(&velocities);

                    (nac_val, Some(nacv))
                } else {
                    (0.0, None)
                }
            }
        }
    }

    pub fn charge_transfer_nacvs(
        &self,
        ct_1: &ChargeTransferPair,
        ct_2: &ChargeTransferPair,
    ) -> Array1<f64> {
        // get pair type
        let pair_type: PairType = self.properties.type_of_pair(ct_1.m_h, ct_1.m_l);

        let ct_gradient: Array1<f64> = if pair_type == PairType::Pair {
            // get pair index
            let pair_index: usize = self.properties.index_of_pair(ct_1.m_h, ct_1.m_l);
            // get correct pair from pairs vector
            let mut pair_ij: Pair = self.pairs[pair_index].clone();
            // get monomers
            let m_i: &Monomer = &self.monomers[pair_ij.i];
            let m_j: &Monomer = &self.monomers[pair_ij.j];

            // get pair atoms
            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );

            pair_ij.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);
            pair_ij.prepare_charge_transfer_nacvs(&pair_atoms, m_i, m_j, ct_1, ct_2);
            pair_ij.nac_tda_dft_excited(0, 1)

            // reset gradient specific properties
            // pair_ij.properties.reset_gradient();
        } else {
            // Do something for ESD pairs
            // get pair index
            let pair_index: usize = self.properties.index_of_esd_pair(ct_1.m_h, ct_1.m_l);
            // get correct pair from pairs vector
            let mut pair_ij: ESDPair = self.esd_pairs[pair_index].clone();
            // get monomers
            let m_i: &Monomer = &self.monomers[pair_ij.i];
            let m_j: &Monomer = &self.monomers[pair_ij.j];
            // get pair atoms
            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );
            // do a scc calculation of the ESD pair
            pair_ij.prepare_scc(&pair_atoms, m_i, m_j);
            pair_ij.run_scc_lc(&pair_atoms, self.config.scf);

            pair_ij.prepare_ct_lcmo_gradient(&pair_atoms);
            pair_ij.prepare_charge_transfer_nacvs(&pair_atoms, m_i, m_j, ct_1, ct_2);
            pair_ij.nac_tda_dft_excited(0, 1)

            // reset gradient specific properties
            // pair_ij.properties.reset_gradient();
        };

        return ct_gradient;
    }
}
