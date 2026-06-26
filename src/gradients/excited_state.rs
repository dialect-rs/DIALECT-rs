use crate::defaults::PROXIMITY_CUTOFF;
use crate::excited_states::{trans_charges, trans_charges_ao};
use crate::gradients::helpers::*;
use crate::initialization::*;
use crate::param::slako_transformations::{
    directional_cosines, slako_transformation_gradients_fast, SplineCache,
};
use crate::scc::gamma_approximation::gamma_ao_wise_from_gamma_atomwise;
use crate::utils::ToOwnedF;
use ndarray::{s, Array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use ndarray_linalg::{into_col, into_row, IntoTriangular, UPLO};
use rayon::prelude::*;

impl System {
    /// Excited-state gradient for the given state. Dispatches to the TDA or
    /// full-TDDFT (Casida) implementation, with or without long-range
    /// correction, according to the config.
    pub fn calculate_excited_state_gradient(&mut self, state: usize) -> Array1<f64> {
        self.prepare_excited_grad();

        let gradient: Array1<f64> = if self.config.lc.long_range_correction {
            if self.config.excited.use_casida {
                self.tddft_gradient_lc_accumulation(state)
            } else {
                self.tda_gradient_lc_accumulation(state)
            }
        } else if self.config.excited.use_casida {
            self.tddft_gradient_no_lc_accumulation(state)
        } else {
            self.tda_gradient_nolc_accumulation(state)
        };
        gradient
    }

    /// Prepare the properties needed for the excited-state gradient: restore
    /// the full active space and (re)compute the transition charges
    /// (q_ov/q_oo/q_vv) when they are missing or were restricted.
    pub fn prepare_excited_grad(&mut self) {
        // if active space is restricted, calculate full transition charges again
        if self.config.tddftb.restrict_active_orbitals
            && !self.config.tight_binding.use_shell_resolved_gamma
        {
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

            // get the tddftb vectors and fill them completely
            let eigenvectors = self.properties.ci_coefficients().unwrap();
            let nocc: usize = self.occ_indices.len();
            let nvirt: usize = self.virt_indices.len();
            let nstates: usize = if self.config.excited.nstates < nocc * nvirt {
                self.config.excited.nstates
            } else {
                nocc * nvirt
            };
            let n_occ_reduced = (self.occ_indices.len() as f64
                * self.config.tddftb.active_orbital_threshold)
                as usize;
            let n_virt_reduced = (self.virt_indices.len() as f64
                * self.config.tddftb.active_orbital_threshold)
                as usize;

            // reshape vectors
            let eigenvectors = eigenvectors
                .to_owned()
                .into_shape([n_occ_reduced, n_virt_reduced, nstates])
                .unwrap();
            // create complete orbital space for vectors
            let mut new_vectors: Array3<f64> = Array3::zeros((nocc, nvirt, nstates));
            new_vectors
                .slice_mut(s![nocc - n_occ_reduced.., 0..n_virt_reduced, ..])
                .assign(&eigenvectors);
            // reshape the vectors
            let new_vectors = new_vectors.into_shape([nocc * nvirt, nstates]).unwrap();
            // set the eigenvectors
            self.properties.set_ci_coefficients(new_vectors);
        }
        // calculate transition charges if they don't exist
        if !self.properties.contains_key("q_ov") {
            if !self.config.tight_binding.use_shell_resolved_gamma {
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
            } else {
                let (qov, qoo, qvv): (Array2<f64>, Array2<f64>, Array2<f64>) = trans_charges_ao(
                    self.n_orbs,
                    self.properties.orbs().unwrap(),
                    self.properties.s().unwrap(),
                    &self.occ_indices,
                    &self.virt_indices,
                );
                // And stored in the properties HashMap.
                self.properties.set_q_oo(qoo);
                self.properties.set_q_ov(qov);
                self.properties.set_q_vv(qvv);
            }
        }
        if !self.properties.contains_key("gamma_ao_wise")
            && !self.config.tight_binding.use_shell_resolved_gamma
        {
            // prepare gamma and grad gamma AO matrix
            let g0_ao: Array2<f64> = gamma_ao_wise_from_gamma_atomwise(
                self.properties.gamma().unwrap(),
                &self.atoms,
                self.n_orbs,
            );
            self.properties.set_gamma_ao(g0_ao);
        }
    }

    /// TDA excited-state gradient without long-range correction.
    pub fn tda_gradient_nolc(&mut self, state: usize) -> Array1<f64> {
        // The index of the HOMO (zero based).
        let homo: usize = self.occ_indices[self.occ_indices.len() - 1];

        // The index of the LUMO (zero based).
        let lumo: usize = self.virt_indices[0];

        let n_occ: usize = self.occ_indices.len();
        let n_virt: usize = self.virt_indices.len();

        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        // Energies of the occupied orbitals.
        let orbe_occ = orbe.slice(s![homo + 1 - n_occ..homo + 1]);
        // Energies of the virtual orbitals.
        let orbe_virt = orbe.slice(s![lumo..lumo + n_virt]);

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

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
        let qtrans_ov: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_ov()
                .unwrap()
                .into_shape((self.n_atoms, n_occ, n_virt))
                .unwrap()
        } else {
            self.properties
                .q_ov()
                .unwrap()
                .into_shape((self.n_orbs, n_occ, n_virt))
                .unwrap()
        };
        let qtrans_oo: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_oo()
                .unwrap()
                .into_shape((self.n_atoms, n_occ, n_occ))
                .unwrap()
        } else {
            self.properties
                .q_oo()
                .unwrap()
                .into_shape((self.n_orbs, n_occ, n_occ))
                .unwrap()
        };
        let qtrans_vv: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_vv()
                .unwrap()
                .into_shape((self.n_atoms, n_virt, n_virt))
                .unwrap()
        } else {
            self.properties
                .q_vv()
                .unwrap()
                .into_shape((self.n_orbs, n_virt, n_virt))
                .unwrap()
        };

        // set gamma matrix
        let g0: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma().unwrap()
        } else {
            self.properties.gamma_ao().unwrap()
        };

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
        let g0_ao: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma_ao().unwrap()
        } else {
            g0.view()
        };
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
        let orbs_occ: ArrayView2<f64> = orbs.slice(s![.., homo + 1 - n_occ..homo + 1]);
        let orbs_virt: ArrayView2<f64> = orbs.slice(s![.., lumo..lumo + n_virt]);

        // let mut orbs_occ: Array2<f64> = Array::zeros((self.n_orbs, n_occ));
        // let mut orbs_virt: Array2<f64> = Array::zeros((self.n_orbs, n_virt));
        // for (i, index) in self.occ_indices.iter().enumerate() {
        //     orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        // }
        // for (i, index) in self.virt_indices.iter().enumerate() {
        //     orbs_virt.slice_mut(s![.., i]).assign(&orbs.column(*index));
        // }

        // transform t and z vectors to AO basis
        let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
        let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
        let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = if self.config.tddftb.restrict_active_orbitals {
            let orbs_reduced: ArrayView2<f64> = orbs.slice(s![.., homo + 1 - n_occ..lumo + n_virt]);
            orbs_reduced.dot(&w_triangular.dot(&orbs_reduced.t()))
        } else {
            orbs.dot(&w_triangular.dot(&orbs.t()))
        };
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

    /// TDA excited-state gradient with long-range correction.
    pub fn tda_gradient_lc(&mut self, state: usize) -> Array1<f64> {
        // The index of the HOMO (zero based).
        let homo: usize = self.occ_indices[self.occ_indices.len() - 1];

        // The index of the LUMO (zero based).
        let lumo: usize = self.virt_indices[0];
        let n_occ: usize = self.occ_indices.len();
        let n_virt: usize = self.virt_indices.len();

        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = self.occ_indices.iter().map(|&occ| orbe[occ]).collect();
        let orbe_virt: Array1<f64> = self.virt_indices.iter().map(|&virt| orbe[virt]).collect();

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

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
        let qtrans_ov: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_ov()
                .unwrap()
                .into_shape((self.n_atoms, n_occ, n_virt))
                .unwrap()
        } else {
            self.properties
                .q_ov()
                .unwrap()
                .into_shape((self.n_orbs, n_occ, n_virt))
                .unwrap()
        };
        let qtrans_oo: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_oo()
                .unwrap()
                .into_shape((self.n_atoms, n_occ, n_occ))
                .unwrap()
        } else {
            self.properties
                .q_oo()
                .unwrap()
                .into_shape((self.n_orbs, n_occ, n_occ))
                .unwrap()
        };
        let qtrans_vv: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_vv()
                .unwrap()
                .into_shape((self.n_atoms, n_virt, n_virt))
                .unwrap()
        } else {
            self.properties
                .q_vv()
                .unwrap()
                .into_shape((self.n_orbs, n_virt, n_virt))
                .unwrap()
        };
        let qtrans_vo: Array3<f64> = qtrans_ov
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();

        // create struct hplus
        let hav: Hav = Hav::new(qtrans_ov, qtrans_vv, qtrans_oo, qtrans_vo.view());
        let hplus: Hplus = Hplus::new(qtrans_ov, qtrans_vv, qtrans_oo, qtrans_vo.view());

        // set gamma matrix
        let g0: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma().unwrap()
        } else {
            self.properties.gamma_ao().unwrap()
        };
        let g0_lr: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma_lr().unwrap()
        } else {
            self.properties.gamma_lr_ao().unwrap()
        };

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
        let g0_ao: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma_ao().unwrap()
        } else {
            g0.view()
        };
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
        let orbs_occ: ArrayView2<f64> = orbs.slice(s![.., homo + 1 - n_occ..homo + 1]);
        let orbs_virt: ArrayView2<f64> = orbs.slice(s![.., lumo..lumo + n_virt]);
        // for (i, index) in self.occ_indices.iter().enumerate() {
        //     orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        // }
        // for (i, index) in self.virt_indices.iter().enumerate() {
        //     orbs_virt.slice_mut(s![.., i]).assign(&orbs.column(*index));
        // }

        // transform t and z vectors to AO basis
        let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
        let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
        let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));
        // let w_ao: Array2<f64> = if self.config.tddftb.restrict_active_orbitals {
        //     let orbs_reduced: ArrayView2<f64> = orbs.slice(s![.., homo + 1 - n_occ..lumo + n_virt]);
        //     orbs_reduced.dot(&w_triangular.dot(&orbs_reduced.t()))
        // } else {
        //     orbs.dot(&w_triangular.dot(&orbs.t()))
        // };
        let x_ao: Array2<f64> = orbs_occ.dot(&x_state.dot(&orbs_virt.t()));

        // set g0lr_ao and g1lr_ao
        let g0lr_ao: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma_lr_ao().unwrap()
        } else {
            g0_lr.view()
        };
        let g1lr_ao: ArrayView3<f64> = self.properties.grad_gamma_lr_ao().unwrap();

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

    /// On-the-fly accumulation version of tda_gradient_lc.
    /// Computes grad_h0 and grad_s contributions on-the-fly without loading 3D arrays from properties.
    /// This reduces memory usage and can be faster for large systems.
    pub fn tda_gradient_lc_accumulation(&mut self, state: usize) -> Array1<f64> {
        // The index of the HOMO (zero based).
        let homo: usize = self.occ_indices[self.occ_indices.len() - 1];

        // The index of the LUMO (zero based).
        let lumo: usize = self.virt_indices[0];

        let n_occ: usize = self.occ_indices.len();
        let n_virt: usize = self.virt_indices.len();

        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = self.occ_indices.iter().map(|&occ| orbe[occ]).collect();
        let orbe_virt: Array1<f64> = self.virt_indices.iter().map(|&virt| orbe[virt]).collect();

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

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
        let u_ab: Array2<f64> = 2.0 * x_state.t().dot(&x_state);
        let u_ij: Array2<f64> = 2.0 * x_state.dot(&x_state.t());
        let v_ab: Array2<f64> = 2.0 * ei.dot(&x_state).t().dot(&x_state);
        let v_ij: Array2<f64> = 2.0 * x_state.dot(&ea).dot(&x_state.t());
        let t_ab: Array2<f64> = x_state.t().dot(&x_state);
        let t_ij: Array2<f64> = x_state.dot(&x_state.t());

        // get the transition charges
        let qtrans_ov: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_ov()
                .unwrap()
                .into_shape((self.n_atoms, n_occ, n_virt))
                .unwrap()
        } else {
            self.properties
                .q_ov()
                .unwrap()
                .into_shape((self.n_orbs, n_occ, n_virt))
                .unwrap()
        };
        let qtrans_oo: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_oo()
                .unwrap()
                .into_shape((self.n_atoms, n_occ, n_occ))
                .unwrap()
        } else {
            self.properties
                .q_oo()
                .unwrap()
                .into_shape((self.n_orbs, n_occ, n_occ))
                .unwrap()
        };
        let qtrans_vv: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_vv()
                .unwrap()
                .into_shape((self.n_atoms, n_virt, n_virt))
                .unwrap()
        } else {
            self.properties
                .q_vv()
                .unwrap()
                .into_shape((self.n_orbs, n_virt, n_virt))
                .unwrap()
        };
        let qtrans_vo: Array3<f64> = qtrans_ov
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();

        // create Hav and Hplus structs
        let hav: Hav = Hav::new(qtrans_ov, qtrans_vv, qtrans_oo, qtrans_vo.view());
        let hplus: Hplus = Hplus::new(qtrans_ov, qtrans_vv, qtrans_oo, qtrans_vo.view());

        // set gamma matrix
        let g0: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma().unwrap()
        } else {
            self.properties.gamma_ao().unwrap()
        };
        let g0_lr: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma_lr().unwrap()
        } else {
            self.properties.gamma_lr_ao().unwrap()
        };

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
        let hplus_wij_result = hplus.compute(g0, g0_lr, z_ia.view(), HplusType::Wij);
        let mut w_ij: Array2<f64> = q_ij + hplus_wij_result;
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
        let orbs_occ: ArrayView2<f64> = orbs.slice(s![.., homo + 1 - n_occ..homo + 1]);
        let orbs_virt: ArrayView2<f64> = orbs.slice(s![.., lumo..lumo + n_virt]);

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
        let g0_ao: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma_ao().unwrap()
        } else {
            g0.view()
        };
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let g0lr_ao: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma_lr_ao().unwrap()
        } else {
            g0_lr.view()
        };

        // Pre-compute orbital index offsets for each atom
        let mut orbital_offsets: Vec<usize> = Vec::with_capacity(self.n_atoms + 1);
        orbital_offsets.push(0);
        for atom in &self.atoms {
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
                let distance = (&self.atoms[i] - &self.atoms[j]).norm();
                if distance < PROXIMITY_CUTOFF {
                    atom_pairs.push((i, j));
                }
            }
        }

        // Parallel compute H0/S/dS gradient contributions for each atom pair
        // Each pair contributes:
        // - dH0[mu,nu] * t_plus_z[mu,nu] (H0 gradient)
        // - dS[mu,nu] * w_ao[mu,nu] (S gradient for W term)
        // - dS[mu,nu] * coeff_ds_total[mu,nu] (dS contribution to f_v and f_lr)
        let pair_contributions: Vec<([f64; 3], [f64; 3], usize, usize)> = atom_pairs
            .par_iter()
            .map(|&(i, j)| {
                let atomi = &self.atoms[i];
                let atomj = &self.atoms[j];
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
                        // The coefficient is already symmetrized, so coeff[mu,nu] includes contribution from both (mu,nu) and (nu,mu)
                        let coeff_mu_nu = coeff_ds_total[[mu, nu]];
                        let coeff_nu_mu = coeff_ds_total[[nu, mu]];

                        // grad_H0 * (T + Z) contribution
                        // Need to account for both (mu, nu) and (nu, mu) since matrices are not symmetric
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
                            // Note: grad_s for nc=3*i has dS[mu,nu] = s_deriv_i for mu in atom i
                            //       grad_s for nc=3*j has dS[mu,nu] = s_deriv_j for nu in atom j
                            // The coefficient is symmetric, so we use the same coeff for both directions
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
            &self.atoms,
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
            &self.atoms,
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
            &self.atoms,
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
            &self.atoms,
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

    /// On-the-fly accumulation version of tda_gradient_nolc.
    /// Computes grad_h0 and grad_s contributions on-the-fly without loading 3D arrays from properties.
    /// This reduces memory usage and can be faster for large systems.
    pub fn tda_gradient_nolc_accumulation(&mut self, state: usize) -> Array1<f64> {
        // The index of the HOMO (zero based).
        let homo: usize = self.occ_indices[self.occ_indices.len() - 1];

        // The index of the LUMO (zero based).
        let lumo: usize = self.virt_indices[0];

        let n_occ: usize = self.occ_indices.len();
        let n_virt: usize = self.virt_indices.len();

        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ = orbe.slice(s![homo + 1 - n_occ..homo + 1]);
        let orbe_virt = orbe.slice(s![lumo..lumo + n_virt]);

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

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
        let u_ab: Array2<f64> = 2.0 * x_state.t().dot(&x_state);
        let u_ij: Array2<f64> = 2.0 * x_state.dot(&x_state.t());
        let v_ab: Array2<f64> = 2.0 * ei.dot(&x_state).t().dot(&x_state);
        let v_ij: Array2<f64> = 2.0 * x_state.dot(&ea).dot(&x_state.t());
        let t_ab: Array2<f64> = x_state.t().dot(&x_state);
        let t_ij: Array2<f64> = x_state.dot(&x_state.t());

        // get the transition charges
        let qtrans_ov: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_ov()
                .unwrap()
                .into_shape((self.n_atoms, n_occ, n_virt))
                .unwrap()
        } else {
            self.properties
                .q_ov()
                .unwrap()
                .into_shape((self.n_orbs, n_occ, n_virt))
                .unwrap()
        };
        let qtrans_oo: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_oo()
                .unwrap()
                .into_shape((self.n_atoms, n_occ, n_occ))
                .unwrap()
        } else {
            self.properties
                .q_oo()
                .unwrap()
                .into_shape((self.n_orbs, n_occ, n_occ))
                .unwrap()
        };
        let qtrans_vv: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_vv()
                .unwrap()
                .into_shape((self.n_atoms, n_virt, n_virt))
                .unwrap()
        } else {
            self.properties
                .q_vv()
                .unwrap()
                .into_shape((self.n_orbs, n_virt, n_virt))
                .unwrap()
        };

        // set gamma matrix
        let g0: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma().unwrap()
        } else {
            self.properties.gamma_ao().unwrap()
        };

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
            .dot(&into_row(orbe_virt.to_owned()))
            - into_col(orbe_occ.to_owned()).dot(&into_row(Array::ones(orbe_virt.len())));
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
        let orbs_occ: ArrayView2<f64> = orbs.slice(s![.., homo + 1 - n_occ..homo + 1]);
        let orbs_virt: ArrayView2<f64> = orbs.slice(s![.., lumo..lumo + n_virt]);

        // transform t and z vectors to AO basis
        let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
        let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
        let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));
        let t_plus_z: Array2<f64> = &t_vv - &t_oo + &z_ao;

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = if self.config.tddftb.restrict_active_orbitals {
            let orbs_reduced: ArrayView2<f64> = orbs.slice(s![.., homo + 1 - n_occ..lumo + n_virt]);
            orbs_reduced.dot(&w_triangular.dot(&orbs_reduced.t()))
        } else {
            orbs.dot(&w_triangular.dot(&orbs.t()))
        };
        let x_ao: Array2<f64> = orbs_occ.dot(&x_state.dot(&orbs_virt.t()));

        // ======= On-the-fly gradient accumulation =======
        // This version eliminates loading of grad_s and grad_gamma_ao
        // by computing dS and dG contributions on-the-fly using precomputed coefficient matrices.

        // Get arrays needed for coefficient precomputation (NO grad_s, grad_gamma_ao!)
        let diff_p: Array2<f64> = &self.properties.p().unwrap() - &self.properties.p_ref().unwrap();
        let g0_ao: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma_ao().unwrap()
        } else {
            g0.view()
        };
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // Pre-compute orbital index offsets for each atom
        let mut orbital_offsets: Vec<usize> = Vec::with_capacity(self.n_atoms + 1);
        orbital_offsets.push(0);
        for atom in &self.atoms {
            orbital_offsets.push(orbital_offsets.last().unwrap() + atom.n_orbs);
        }

        // ======= Precompute coefficient matrices for dS-dependent terms =======
        // For nolc, we only have f_v (no f_lr)

        // 1. f_v(diff_p) contracted with t_plus_z (coefficient for f_dmd0)
        let coeff_fv_dmd0 =
            compute_fv_coefficients_onthefly(diff_p.view(), s, g0_ao, t_plus_z.view());

        // 2. f_v(x_ao) contracted with x_ao (scaled by 2.0)
        let coeff_fv_xao = compute_fv_coefficients_onthefly(x_ao.view(), s, g0_ao, x_ao.view());

        // Combined dS coefficient: (1.0)*coeff_fv_dmd0 + (2.0)*coeff_fv_xao
        let coeff_ds_total: Array2<f64> = &coeff_fv_dmd0 + 2.0 * &coeff_fv_xao;

        // Collect valid atom pairs (i < j, within cutoff)
        let mut atom_pairs: Vec<(usize, usize)> = Vec::new();
        for i in 0..self.n_atoms {
            for j in (i + 1)..self.n_atoms {
                let distance = (&self.atoms[i] - &self.atoms[j]).norm();
                if distance < PROXIMITY_CUTOFF {
                    atom_pairs.push((i, j));
                }
            }
        }

        // Parallel compute H0/S/dS gradient contributions for each atom pair
        let pair_contributions: Vec<([f64; 3], [f64; 3], usize, usize)> = atom_pairs
            .par_iter()
            .map(|&(i, j)| {
                let atomi = &self.atoms[i];
                let atomj = &self.atoms[j];
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

                        // Get dS coefficient for f_v contribution
                        let coeff_mu_nu = coeff_ds_total[[mu, nu]];
                        let coeff_nu_mu = coeff_ds_total[[nu, mu]];

                        for dir in 0..3 {
                            // H0 contribution: dH0[mu,nu] * t_plus_z[mu,nu] + dH0[nu,mu] * t_plus_z[nu,mu]
                            grad_i[dir] +=
                                h0_deriv_i[dir] * t_plus_z_mu_nu + h0_deriv_i[dir] * t_plus_z_nu_mu;
                            grad_j[dir] +=
                                h0_deriv_j[dir] * t_plus_z_mu_nu + h0_deriv_j[dir] * t_plus_z_nu_mu;

                            // -grad_S * W contribution
                            grad_i[dir] -=
                                s_deriv_i[dir] * w_ao_mu_nu + s_deriv_i[dir] * w_ao_nu_mu;
                            grad_j[dir] -=
                                s_deriv_j[dir] * w_ao_mu_nu + s_deriv_j[dir] * w_ao_nu_mu;

                            // f_v dS contribution
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
        // For nolc, we only have f_v contributions (no f_lr)

        let gammafunction = &self.gammafunction;

        // 1. f_v(diff_p) dG contribution contracted with t_plus_z
        let fv_dg_dmd0 = compute_fv_dg_contributions(
            gammafunction,
            &self.atoms,
            &orbital_offsets,
            s,
            diff_p.view(),
            t_plus_z.view(),
            self.n_atoms,
            self.n_orbs,
        );

        // 2. f_v(x_ao) dG contribution contracted with x_ao (scaled by 2.0)
        let fv_dg_xao = compute_fv_dg_contributions(
            gammafunction,
            &self.atoms,
            &orbital_offsets,
            s,
            x_ao.view(),
            x_ao.view(),
            self.n_atoms,
            self.n_orbs,
        );

        // Combine dG contributions
        let grad_dg_contrib: Array1<f64> = &fv_dg_dmd0 + 2.0 * &fv_dg_xao;

        // Assemble the excited gradient
        let grad_exc: Array1<f64> = &grad_h0_s_contrib + &grad_dg_contrib;

        grad_exc
    }

    /// Full-TDDFT (Casida) excited-state gradient with long-range correction.
    pub fn tddft_gradient_lc(&mut self, state: usize) -> Array1<f64> {
        // The index of the HOMO (zero based).
        let homo: usize = self.occ_indices[self.occ_indices.len() - 1];
        // The index of the LUMO (zero based).
        let lumo: usize = self.virt_indices[0];
        let n_occ: usize = self.occ_indices.len();
        let n_virt: usize = self.virt_indices.len();

        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        // Energies of the occupied orbitals.
        let orbe_occ = orbe.slice(s![homo + 1 - n_occ..homo + 1]);
        // Energies of the virtual orbitals.
        let orbe_virt = orbe.slice(s![lumo..lumo + n_virt]);

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

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
        let qtrans_ov: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_ov()
                .unwrap()
                .into_shape((self.n_atoms, n_occ, n_virt))
                .unwrap()
        } else {
            self.properties
                .q_ov()
                .unwrap()
                .into_shape((self.n_orbs, n_occ, n_virt))
                .unwrap()
        };
        let qtrans_oo: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_oo()
                .unwrap()
                .into_shape((self.n_atoms, n_occ, n_occ))
                .unwrap()
        } else {
            self.properties
                .q_oo()
                .unwrap()
                .into_shape((self.n_orbs, n_occ, n_occ))
                .unwrap()
        };
        let qtrans_vv: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_vv()
                .unwrap()
                .into_shape((self.n_atoms, n_virt, n_virt))
                .unwrap()
        } else {
            self.properties
                .q_vv()
                .unwrap()
                .into_shape((self.n_orbs, n_virt, n_virt))
                .unwrap()
        };
        let qtrans_vo: Array3<f64> = qtrans_ov
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();

        // create struct hplus
        let hplus: Hplus = Hplus::new(qtrans_ov, qtrans_vv, qtrans_oo, qtrans_vo.view());

        // set gamma matrix
        let g0: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma().unwrap()
        } else {
            self.properties.gamma_ao().unwrap()
        };
        let g0_lr: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma_lr().unwrap()
        } else {
            self.properties.gamma_lr_ao().unwrap()
        };

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
            + xmy_state.t().dot(&h_minus(
                g0_lr, qtrans_ov, qtrans_oo, qtrans_oo, qtrans_ov, xmy_state,
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
            w_ij[[i, i]] /= 2.0;
        }
        // w_ia
        let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia);

        // w_ai
        let w_ai: Array2<f64> = w_ia.clone().reversed_axes();

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
        let g0_ao: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma_ao().unwrap()
        } else {
            g0.view()
        };
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
        let orbs_occ: ArrayView2<f64> = orbs.slice(s![.., homo + 1 - n_occ..homo + 1]);
        let orbs_virt: ArrayView2<f64> = orbs.slice(s![.., lumo..lumo + n_virt]);

        // transform t and z vectors to AO basis
        let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
        let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
        let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = if self.config.tddftb.restrict_active_orbitals {
            let orbs_reduced: ArrayView2<f64> = orbs.slice(s![.., homo + 1 - n_occ..lumo + n_virt]);
            orbs_reduced.dot(&w_triangular.dot(&orbs_reduced.t()))
        } else {
            orbs.dot(&w_triangular.dot(&orbs.t()))
        };
        let xpy_ao: Array2<f64> = orbs_occ.dot(&xpy_state.dot(&orbs_virt.t()));
        let xmy_ao: Array2<f64> = orbs_occ.dot(&xmy_state.dot(&orbs_virt.t()));

        // set g0lr_ao and g1lr_ao
        let g0lr_ao: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma_lr_ao().unwrap()
        } else {
            g0_lr.view()
        };
        let g1lr_ao: ArrayView3<f64> = self.properties.grad_gamma_lr_ao().unwrap();

        // calculate contributions to the excited gradient
        let f: Array3<f64> = f_v(
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

        grad_exc
    }

    /// Full-TDDFT (Casida) excited-state gradient without long-range correction.
    pub fn tddft_gradient_no_lc(&mut self, state: usize) -> Array1<f64> {
        // The index of the HOMO (zero based).
        let homo: usize = self.occ_indices[self.occ_indices.len() - 1];
        // The index of the LUMO (zero based).
        let lumo: usize = self.virt_indices[0];
        let n_occ: usize = self.occ_indices.len();
        let n_virt: usize = self.virt_indices.len();

        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        // Energies of the occupied orbitals.
        let orbe_occ = orbe.slice(s![homo + 1 - n_occ..homo + 1]);
        // Energies of the virtual orbitals.
        let orbe_virt = orbe.slice(s![lumo..lumo + n_virt]);

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

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
        let qtrans_ov: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_ov()
                .unwrap()
                .into_shape((self.n_atoms, n_occ, n_virt))
                .unwrap()
        } else {
            self.properties
                .q_ov()
                .unwrap()
                .into_shape((self.n_orbs, n_occ, n_virt))
                .unwrap()
        };
        let qtrans_oo: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_oo()
                .unwrap()
                .into_shape((self.n_atoms, n_occ, n_occ))
                .unwrap()
        } else {
            self.properties
                .q_oo()
                .unwrap()
                .into_shape((self.n_orbs, n_occ, n_occ))
                .unwrap()
        };
        let qtrans_vv: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_vv()
                .unwrap()
                .into_shape((self.n_atoms, n_virt, n_virt))
                .unwrap()
        } else {
            self.properties
                .q_vv()
                .unwrap()
                .into_shape((self.n_orbs, n_virt, n_virt))
                .unwrap()
        };

        // set gamma matrix
        let g0: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma().unwrap()
        } else {
            self.properties.gamma_ao().unwrap()
        };

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
            w_ij[[i, i]] /= 2.0;
        }
        // w_ia
        let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia);

        // w_ai
        let w_ai: Array2<f64> = w_ia.clone().reversed_axes();

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
        let g0_ao: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma_ao().unwrap()
        } else {
            g0.view()
        };
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
        let orbs_occ: ArrayView2<f64> = orbs.slice(s![.., homo + 1 - n_occ..homo + 1]);
        let orbs_virt: ArrayView2<f64> = orbs.slice(s![.., lumo..lumo + n_virt]);

        // transform t and z vectors to AO basis
        let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
        let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
        let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = if self.config.tddftb.restrict_active_orbitals {
            let orbs_reduced: ArrayView2<f64> = orbs.slice(s![.., homo + 1 - n_occ..lumo + n_virt]);
            orbs_reduced.dot(&w_triangular.dot(&orbs_reduced.t()))
        } else {
            orbs.dot(&w_triangular.dot(&orbs.t()))
        };
        let xpy_ao: Array2<f64> = orbs_occ.dot(&xpy_state.dot(&orbs_virt.t()));

        // calculate contributions to the excited gradient
        let f: Array3<f64> = f_v(
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

        grad_exc
    }

    /// On-the-fly accumulation version of tddft_gradient_no_lc.
    /// Computes grad_h0 and grad_s contributions on-the-fly without loading 3D arrays from properties.
    pub fn tddft_gradient_no_lc_accumulation(&mut self, state: usize) -> Array1<f64> {
        // The index of the HOMO (zero based).
        let homo: usize = self.occ_indices[self.occ_indices.len() - 1];

        // The index of the LUMO (zero based).
        let lumo: usize = self.virt_indices[0];

        let n_occ: usize = self.occ_indices.len();
        let n_virt: usize = self.virt_indices.len();

        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ = orbe.slice(s![homo + 1 - n_occ..homo + 1]);
        let orbe_virt = orbe.slice(s![lumo..lumo + n_virt]);

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

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
        let qtrans_ov: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_ov()
                .unwrap()
                .into_shape((self.n_atoms, n_occ, n_virt))
                .unwrap()
        } else {
            self.properties
                .q_ov()
                .unwrap()
                .into_shape((self.n_orbs, n_occ, n_virt))
                .unwrap()
        };
        let qtrans_oo: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_oo()
                .unwrap()
                .into_shape((self.n_atoms, n_occ, n_occ))
                .unwrap()
        } else {
            self.properties
                .q_oo()
                .unwrap()
                .into_shape((self.n_orbs, n_occ, n_occ))
                .unwrap()
        };
        let qtrans_vv: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_vv()
                .unwrap()
                .into_shape((self.n_atoms, n_virt, n_virt))
                .unwrap()
        } else {
            self.properties
                .q_vv()
                .unwrap()
                .into_shape((self.n_orbs, n_virt, n_virt))
                .unwrap()
        };

        // set gamma matrix
        let g0: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma().unwrap()
        } else {
            self.properties.gamma_ao().unwrap()
        };

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
            .dot(&into_row(orbe_virt.to_owned()))
            - into_col(orbe_occ.to_owned()).dot(&into_row(Array::ones(orbe_virt.len())));
        let r_ia_flat: Array1<f64> = r_ia.t().to_owned_f().into_shape(n_occ * n_virt).unwrap();
        let r_matrix: Array2<f64> = r_ia_flat.into_shape((n_occ, n_virt)).unwrap();

        // calculate the z-vector
        let z_ia: Array2<f64> = zvector_no_lc(omega_input.view(), r_matrix.view(), g0, qtrans_ov);

        // calculate w_ij
        let mut w_ij: Array2<f64> = q_ij + h_plus_no_lr(g0, qtrans_oo, qtrans_ov, z_ia.view());
        for i in 0..w_ij.dim().0 {
            w_ij[[i, i]] /= 2.0;
        }
        // w_ia
        let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia);

        // w_ai
        let w_ai: Array2<f64> = w_ia.clone().reversed_axes();

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
        let orbs_occ: ArrayView2<f64> = orbs.slice(s![.., homo + 1 - n_occ..homo + 1]);
        let orbs_virt: ArrayView2<f64> = orbs.slice(s![.., lumo..lumo + n_virt]);

        // transform t and z vectors to AO basis
        let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
        let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
        let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));
        let t_plus_z: Array2<f64> = &t_vv - &t_oo + &z_ao;

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = if self.config.tddftb.restrict_active_orbitals {
            let orbs_reduced: ArrayView2<f64> = orbs.slice(s![.., homo + 1 - n_occ..lumo + n_virt]);
            orbs_reduced.dot(&w_triangular.dot(&orbs_reduced.t()))
        } else {
            orbs.dot(&w_triangular.dot(&orbs.t()))
        };
        let xpy_ao: Array2<f64> = orbs_occ.dot(&xpy_state.dot(&orbs_virt.t()));

        // ======= On-the-fly gradient accumulation =======
        let diff_p: Array2<f64> = &self.properties.p().unwrap() - &self.properties.p_ref().unwrap();
        let g0_ao: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma_ao().unwrap()
        } else {
            g0.view()
        };
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // Pre-compute orbital index offsets for each atom
        let mut orbital_offsets: Vec<usize> = Vec::with_capacity(self.n_atoms + 1);
        orbital_offsets.push(0);
        for atom in &self.atoms {
            orbital_offsets.push(orbital_offsets.last().unwrap() + atom.n_orbs);
        }

        // ======= Precompute coefficient matrices for dS-dependent terms =======
        // 1. f_v(diff_p) contracted with t_plus_z
        let coeff_fv_dmd0 =
            compute_fv_coefficients_onthefly(diff_p.view(), s, g0_ao, t_plus_z.view());

        // 2. f_v(xpy_ao) contracted with xpy_ao (scaled by 2.0)
        let coeff_fv_xpy = compute_fv_coefficients_onthefly(xpy_ao.view(), s, g0_ao, xpy_ao.view());

        // Combined dS coefficient
        let coeff_ds_total: Array2<f64> = &coeff_fv_dmd0 + 2.0 * &coeff_fv_xpy;

        // Collect valid atom pairs
        let mut atom_pairs: Vec<(usize, usize)> = Vec::new();
        for i in 0..self.n_atoms {
            for j in (i + 1)..self.n_atoms {
                let distance = (&self.atoms[i] - &self.atoms[j]).norm();
                if distance < PROXIMITY_CUTOFF {
                    atom_pairs.push((i, j));
                }
            }
        }

        // Parallel compute H0/S/dS gradient contributions
        let pair_contributions: Vec<([f64; 3], [f64; 3], usize, usize)> = atom_pairs
            .par_iter()
            .map(|&(i, j)| {
                let atomi = &self.atoms[i];
                let atomj = &self.atoms[j];
                let mu_start = orbital_offsets[i];
                let nu_start = orbital_offsets[j];

                let (r, x, y, z): (f64, f64, f64, f64) = if atomi <= atomj {
                    directional_cosines(&atomi.xyz, &atomj.xyz)
                } else {
                    directional_cosines(&atomj.xyz, &atomi.xyz)
                };

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

                        let t_plus_z_mu_nu = t_plus_z[[mu, nu]];
                        let t_plus_z_nu_mu = t_plus_z[[nu, mu]];
                        let w_ao_mu_nu = w_ao[[mu, nu]];
                        let w_ao_nu_mu = w_ao[[nu, mu]];
                        let coeff_mu_nu = coeff_ds_total[[mu, nu]];
                        let coeff_nu_mu = coeff_ds_total[[nu, mu]];

                        for dir in 0..3 {
                            grad_i[dir] +=
                                h0_deriv_i[dir] * t_plus_z_mu_nu + h0_deriv_i[dir] * t_plus_z_nu_mu;
                            grad_j[dir] +=
                                h0_deriv_j[dir] * t_plus_z_mu_nu + h0_deriv_j[dir] * t_plus_z_nu_mu;
                            grad_i[dir] -=
                                s_deriv_i[dir] * w_ao_mu_nu + s_deriv_i[dir] * w_ao_nu_mu;
                            grad_j[dir] -=
                                s_deriv_j[dir] * w_ao_mu_nu + s_deriv_j[dir] * w_ao_nu_mu;
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

        let mut grad_h0_s_contrib: Array1<f64> = Array1::zeros(3 * self.n_atoms);
        for (grad_i, grad_j, i, j) in pair_contributions {
            for dir in 0..3 {
                grad_h0_s_contrib[3 * i + dir] += grad_i[dir];
                grad_h0_s_contrib[3 * j + dir] += grad_j[dir];
            }
        }

        // ======= dG (gamma gradient) contributions computed on-the-fly =======
        let gammafunction = &self.gammafunction;

        let fv_dg_dmd0 = compute_fv_dg_contributions(
            gammafunction,
            &self.atoms,
            &orbital_offsets,
            s,
            diff_p.view(),
            t_plus_z.view(),
            self.n_atoms,
            self.n_orbs,
        );

        let fv_dg_xpy = compute_fv_dg_contributions(
            gammafunction,
            &self.atoms,
            &orbital_offsets,
            s,
            xpy_ao.view(),
            xpy_ao.view(),
            self.n_atoms,
            self.n_orbs,
        );

        let grad_dg_contrib: Array1<f64> = &fv_dg_dmd0 + 2.0 * &fv_dg_xpy;
        let grad_exc: Array1<f64> = &grad_h0_s_contrib + &grad_dg_contrib;

        grad_exc
    }

    /// On-the-fly accumulation version of tddft_gradient_lc.
    /// Computes grad_h0 and grad_s contributions on-the-fly without loading 3D arrays from properties.
    pub fn tddft_gradient_lc_accumulation(&mut self, state: usize) -> Array1<f64> {
        // The index of the HOMO (zero based).
        let homo: usize = self.occ_indices[self.occ_indices.len() - 1];

        // The index of the LUMO (zero based).
        let lumo: usize = self.virt_indices[0];

        let n_occ: usize = self.occ_indices.len();
        let n_virt: usize = self.virt_indices.len();

        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ = orbe.slice(s![homo + 1 - n_occ..homo + 1]);
        let orbe_virt = orbe.slice(s![lumo..lumo + n_virt]);

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

        // take state specific values from the excitation vectors
        let xmy_state: ArrayView3<f64> = self.properties.xmy().unwrap();
        let xpy_state: ArrayView3<f64> = self.properties.xpy().unwrap();
        let xmy_state: ArrayView2<f64> = xmy_state.slice(s![state, .., ..]);
        let xpy_state: ArrayView2<f64> = xpy_state.slice(s![state, .., ..]);
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
        let qtrans_ov: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_ov()
                .unwrap()
                .into_shape((self.n_atoms, n_occ, n_virt))
                .unwrap()
        } else {
            self.properties
                .q_ov()
                .unwrap()
                .into_shape((self.n_orbs, n_occ, n_virt))
                .unwrap()
        };
        let qtrans_oo: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_oo()
                .unwrap()
                .into_shape((self.n_atoms, n_occ, n_occ))
                .unwrap()
        } else {
            self.properties
                .q_oo()
                .unwrap()
                .into_shape((self.n_orbs, n_occ, n_occ))
                .unwrap()
        };
        let qtrans_vv: ArrayView3<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties
                .q_vv()
                .unwrap()
                .into_shape((self.n_atoms, n_virt, n_virt))
                .unwrap()
        } else {
            self.properties
                .q_vv()
                .unwrap()
                .into_shape((self.n_orbs, n_virt, n_virt))
                .unwrap()
        };
        let qtrans_vo: Array3<f64> = qtrans_ov
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();

        // create struct hplus
        let hplus: Hplus = Hplus::new(qtrans_ov, qtrans_vv, qtrans_oo, qtrans_vo.view());

        // set gamma matrix
        let g0: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma().unwrap()
        } else {
            self.properties.gamma_ao().unwrap()
        };
        let g0_lr: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma_lr().unwrap()
        } else {
            self.properties.gamma_lr_ao().unwrap()
        };

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
            + xmy_state.t().dot(&h_minus(
                g0_lr, qtrans_ov, qtrans_oo, qtrans_oo, qtrans_ov, xmy_state,
            ));

        // calculate right hand side of the z-vector equation
        let r_ia: Array2<f64> = &q_ai.t() - &q_ia;

        // input for zvector routine
        let omega_input: Array2<f64> = into_col(Array::ones(orbe_occ.len()))
            .dot(&into_row(orbe_virt.to_owned()))
            - into_col(orbe_occ.to_owned()).dot(&into_row(Array::ones(orbe_virt.len())));
        let r_ia_flat: Array1<f64> = r_ia.t().to_owned_f().into_shape(n_occ * n_virt).unwrap();
        let r_matrix: Array2<f64> = r_ia_flat.into_shape((n_occ, n_virt)).unwrap();

        // calculate the z-vector
        let z_ia: Array2<f64> = zvector_lc_optimized(
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
        let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia);

        // w_ai
        let w_ai: Array2<f64> = w_ia.clone().reversed_axes();

        // w_ab
        let mut w_ab: Array2<f64> = q_ab;
        for i in 0..w_ab.dim().0 {
            w_ab[[i, i]] /= 2.0;
        }

        // build w matrix
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
        let orbs_occ: ArrayView2<f64> = orbs.slice(s![.., homo + 1 - n_occ..homo + 1]);
        let orbs_virt: ArrayView2<f64> = orbs.slice(s![.., lumo..lumo + n_virt]);

        // transform t and z vectors to AO basis
        let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
        let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
        let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));
        let t_plus_z: Array2<f64> = &t_vv - &t_oo + &z_ao;

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = if self.config.tddftb.restrict_active_orbitals {
            let orbs_reduced: ArrayView2<f64> = orbs.slice(s![.., homo + 1 - n_occ..lumo + n_virt]);
            orbs_reduced.dot(&w_triangular.dot(&orbs_reduced.t()))
        } else {
            orbs.dot(&w_triangular.dot(&orbs.t()))
        };
        let xpy_ao: Array2<f64> = orbs_occ.dot(&xpy_state.dot(&orbs_virt.t()));
        let xmy_ao: Array2<f64> = orbs_occ.dot(&xmy_state.dot(&orbs_virt.t()));

        // ======= On-the-fly gradient accumulation =======
        let diff_p: Array2<f64> = &self.properties.p().unwrap() - &self.properties.p_ref().unwrap();
        let g0_ao: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma_ao().unwrap()
        } else {
            g0.view()
        };
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let g0lr_ao: ArrayView2<f64> = if !self.config.tight_binding.use_shell_resolved_gamma {
            self.properties.gamma_lr_ao().unwrap()
        } else {
            g0_lr.view()
        };

        // Pre-compute orbital index offsets for each atom
        let mut orbital_offsets: Vec<usize> = Vec::with_capacity(self.n_atoms + 1);
        orbital_offsets.push(0);
        for atom in &self.atoms {
            orbital_offsets.push(orbital_offsets.last().unwrap() + atom.n_orbs);
        }

        // ======= Precompute coefficient matrices for dS-dependent terms =======
        // TDDFT LC has: f_v(diff_p), f_lr(diff_p), f_v(xpy_ao), f_lr(xpy+xpy^T), f_lr(xmy-xmy^T)

        // 1. f_v(diff_p) contracted with t_plus_z
        let coeff_fv_dmd0 =
            compute_fv_coefficients_onthefly(diff_p.view(), s, g0_ao, t_plus_z.view());

        // 2. f_lr(diff_p) contracted with t_plus_z (scaled by -0.5)
        let coeff_flr_dmd0 = compute_flr_s_coefficients(diff_p.view(), s, g0lr_ao, t_plus_z.view());

        // 3. f_v(xpy_ao) contracted with xpy_ao (scaled by 2.0)
        let coeff_fv_xpy = compute_fv_coefficients_onthefly(xpy_ao.view(), s, g0_ao, xpy_ao.view());

        // 4. f_lr(xpy_ao + xpy_ao^T) contracted with xpy_ao (scaled by -0.5)
        let xpy_sym: Array2<f64> = &xpy_ao + &xpy_ao.t();
        let coeff_flr_xpy = compute_flr_s_coefficients(xpy_sym.view(), s, g0lr_ao, xpy_ao.view());

        // 5. f_lr(xmy_ao - xmy_ao^T) contracted with xmy_ao (scaled by +0.5, since -f_lr(-) = +f_lr)
        let xmy_antisym: Array2<f64> = &xmy_ao - &xmy_ao.t();
        let coeff_flr_xmy =
            compute_flr_s_coefficients(xmy_antisym.view(), s, g0lr_ao, xmy_ao.view());

        // Combined dS coefficient:
        // (1.0)*coeff_fv_dmd0 + (-0.5)*coeff_flr_dmd0 + (2.0)*coeff_fv_xpy + (-0.5)*coeff_flr_xpy + (0.5)*coeff_flr_xmy
        let coeff_ds_total: Array2<f64> =
            &coeff_fv_dmd0 - 0.5 * &coeff_flr_dmd0 + 2.0 * &coeff_fv_xpy - 0.5 * &coeff_flr_xpy
                + 0.5 * &coeff_flr_xmy;

        // Collect valid atom pairs
        let mut atom_pairs: Vec<(usize, usize)> = Vec::new();
        for i in 0..self.n_atoms {
            for j in (i + 1)..self.n_atoms {
                let distance = (&self.atoms[i] - &self.atoms[j]).norm();
                if distance < PROXIMITY_CUTOFF {
                    atom_pairs.push((i, j));
                }
            }
        }

        // Parallel compute H0/S/dS gradient contributions
        let pair_contributions: Vec<([f64; 3], [f64; 3], usize, usize)> = atom_pairs
            .par_iter()
            .map(|&(i, j)| {
                let atomi = &self.atoms[i];
                let atomj = &self.atoms[j];
                let mu_start = orbital_offsets[i];
                let nu_start = orbital_offsets[j];

                let (r, x, y, z): (f64, f64, f64, f64) = if atomi <= atomj {
                    directional_cosines(&atomi.xyz, &atomj.xyz)
                } else {
                    directional_cosines(&atomj.xyz, &atomi.xyz)
                };

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

                        let t_plus_z_mu_nu = t_plus_z[[mu, nu]];
                        let t_plus_z_nu_mu = t_plus_z[[nu, mu]];
                        let w_ao_mu_nu = w_ao[[mu, nu]];
                        let w_ao_nu_mu = w_ao[[nu, mu]];
                        let coeff_mu_nu = coeff_ds_total[[mu, nu]];
                        let coeff_nu_mu = coeff_ds_total[[nu, mu]];

                        for dir in 0..3 {
                            grad_i[dir] +=
                                h0_deriv_i[dir] * t_plus_z_mu_nu + h0_deriv_i[dir] * t_plus_z_nu_mu;
                            grad_j[dir] +=
                                h0_deriv_j[dir] * t_plus_z_mu_nu + h0_deriv_j[dir] * t_plus_z_nu_mu;
                            grad_i[dir] -=
                                s_deriv_i[dir] * w_ao_mu_nu + s_deriv_i[dir] * w_ao_nu_mu;
                            grad_j[dir] -=
                                s_deriv_j[dir] * w_ao_mu_nu + s_deriv_j[dir] * w_ao_nu_mu;
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

        let mut grad_h0_s_contrib: Array1<f64> = Array1::zeros(3 * self.n_atoms);
        for (grad_i, grad_j, i, j) in pair_contributions {
            for dir in 0..3 {
                grad_h0_s_contrib[3 * i + dir] += grad_i[dir];
                grad_h0_s_contrib[3 * j + dir] += grad_j[dir];
            }
        }

        // ======= dG (gamma gradient) contributions computed on-the-fly =======
        let gammafunction = &self.gammafunction;
        let gammafunction_lr = self.gammafunction_lc.as_ref().unwrap();

        // 1. f_v(diff_p) dG contribution contracted with t_plus_z
        let fv_dg_dmd0 = compute_fv_dg_contributions(
            gammafunction,
            &self.atoms,
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
            &self.atoms,
            &orbital_offsets,
            s,
            diff_p.view(),
            t_plus_z.view(),
            self.n_atoms,
            self.n_orbs,
        );

        // 3. f_v(xpy_ao) dG contribution contracted with xpy_ao (scaled by 2.0)
        let fv_dg_xpy = compute_fv_dg_contributions(
            gammafunction,
            &self.atoms,
            &orbital_offsets,
            s,
            xpy_ao.view(),
            xpy_ao.view(),
            self.n_atoms,
            self.n_orbs,
        );

        // 4. f_lr(xpy_ao + xpy_ao^T) dG contribution contracted with xpy_ao (scaled by -0.5)
        let flr_dg_xpy = compute_flr_dg_contributions(
            gammafunction_lr,
            &self.atoms,
            &orbital_offsets,
            s,
            xpy_sym.view(),
            xpy_ao.view(),
            self.n_atoms,
            self.n_orbs,
        );

        // 5. f_lr(xmy_ao - xmy_ao^T) dG contribution contracted with xmy_ao (scaled by +0.5)
        let flr_dg_xmy = compute_flr_dg_contributions(
            gammafunction_lr,
            &self.atoms,
            &orbital_offsets,
            s,
            xmy_antisym.view(),
            xmy_ao.view(),
            self.n_atoms,
            self.n_orbs,
        );

        // Combine dG contributions
        let grad_dg_contrib: Array1<f64> = &fv_dg_dmd0 - 0.5 * &flr_dg_dmd0 + 2.0 * &fv_dg_xpy
            - 0.5 * &flr_dg_xpy
            + 0.5 * &flr_dg_xmy;

        let grad_exc: Array1<f64> = &grad_h0_s_contrib + &grad_dg_contrib;

        grad_exc
    }
}
