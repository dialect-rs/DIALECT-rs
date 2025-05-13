use crate::fmo::gradients::*;
use crate::fmo::scc::helpers::*;
use crate::fmo::Pair;
use crate::gradients::helpers::f_lr;
use crate::initialization::Atom;
use crate::scc::gamma_approximation::{gamma_gradients_ao_wise, gamma_gradients_atomwise};
use crate::scc::h0_and_s::h0_and_s_gradients;
use std::ops::AddAssign;

impl GroundStateGradient for Pair<'_> {
    fn scc_gradient(&mut self, atoms: &[Atom]) -> Array1<f64> {
        // for the evaluation of the gradient it is necessary to compute the derivatives
        // of: - H0
        //     - S
        //     - Gamma
        //     - Repulsive Potential
        // the first three properties are calculated here at the beginning and the gradient that
        // originates from the repulsive potential is added at the end to total gradient

        // derivative of H0 and S
        let (grad_s, grad_h0) = h0_and_s_gradients(atoms, self.n_orbs, self.slako);
        // println!("grad s, h0: {:4}",timer.elapsed().as_secs_f32());
        // Reference to the difference of the density matrix of the pair and the corresponding monomers.
        let _dp: ArrayView2<f64> = self.properties.delta_p().unwrap();
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        // the derivatives of the charge (difference)s are computed at this point, since they depend
        // on the derivative of S and this is available here at no additional cost.
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let grad_dq: Array2<f64> = self.get_grad_dq(atoms, s.view(), grad_s.view(), p.view());
        self.properties.set_grad_dq(grad_dq);

        // and reshape them into a 2D array. the last two dimension (number of orbitals) are compressed
        // into one dimension to be able to just matrix-matrix products for the computation of the gradient
        let grad_s: Array2<f64> = grad_s
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap();
        let grad_h0: Array2<f64> = grad_h0
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap();

        // derivative of the gamma matrix and transform it in the same way to a 2D array
        let grad_gamma: Array2<f64> =
            gamma_gradients_atomwise(&self.gammafunction, atoms, self.n_atoms)
                .into_shape([3 * self.n_atoms, self.n_atoms * self.n_atoms])
                .unwrap();

        // take references/views to the necessary properties from the scc calculation
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        let _h0: ArrayView2<f64> = self.properties.h0().unwrap();
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();
        let _s: ArrayView2<f64> = self.properties.s().unwrap();

        // transform the expression Sum_c_in_X (gamma_AC + gamma_aC) * dq_C
        // into matrix of the dimension (norb, norb) to do an element wise multiplication with P
        let esp_mat: Array2<f64> =
            atomvec_to_aomat(gamma.dot(&dq).view(), self.n_orbs, atoms) * 0.5;
        let esp_x_p: Array1<f64> = (&p * &esp_mat)
            .into_shape([self.n_orbs * self.n_orbs])
            .unwrap();
        let p_flat: ArrayView1<f64> = p.into_shape([self.n_orbs * self.n_orbs]).unwrap();

        // the gradient part which involves the gradient of the gamma matrix is given by:
        // 1/2 * dq . dGamma / dR . dq
        // the dq's are element wise multiplied into a 2D array and reshaped into a flat one, that
        // has the length of natoms^2. this allows to do only a single matrix vector product of
        // 'grad_gamma' with 'dq_x_dq' and avoids to reshape dGamma multiple times
        let dq_column: ArrayView2<f64> = dq.insert_axis(Axis(1));
        let dq_x_dq: Array1<f64> = (&dq_column.broadcast((self.n_atoms, self.n_atoms)).unwrap()
            * &dq)
            .into_shape([self.n_atoms * self.n_atoms])
            .unwrap();

        // compute the energy weighted density matrix: W = 1/2 * D . (H + H_Coul) . D
        let w: Array1<f64> = 0.5
            * (p.dot(&self.properties.h_coul_x().unwrap()).dot(&p))
                .into_shape([self.n_orbs * self.n_orbs])
                .unwrap();

        // calculation of the gradient
        // 1st part:  dH0 / dR . P
        let mut gradient: Array1<f64> = grad_h0.dot(&p_flat);

        // 2nd part: dS / dR . W
        gradient -= &grad_s.dot(&w);

        // 3rd part: 1/2 * dS / dR * sum_c_in_X (gamma_ac + gamma_bc) * dq
        gradient += &grad_s.dot(&esp_x_p);

        // 4th part: 1/2 * dq . dGamma / dR . dq
        gradient += &(grad_gamma.dot(&dq_x_dq));

        // last part: dV_rep / dR
        gradient = gradient + gradient_v_rep(atoms, self.vrep);
        // println!("build gradient: {:4}",timer.elapsed().as_secs_f32());

        // long-range contribution to the gradient
        if self.gammafunction_lc.is_some() {
            // reshape gradS
            let grad_s: Array3<f64> = grad_s
                .into_shape([3 * self.n_atoms, self.n_orbs, self.n_orbs])
                .unwrap();
            // calculate the gamma gradient matrix in AO basis
            let (_g1_lr, g1_lr_ao): (Array3<f64>, Array3<f64>) = gamma_gradients_ao_wise(
                self.gammafunction_lc.as_ref().unwrap(),
                atoms,
                self.n_atoms,
                self.n_orbs,
            );
            // calculate the difference density matrix
            let diff_p: Array2<f64> = &p - &self.properties.p_ref().unwrap();
            // calculate the matrix F_lr[diff_p]
            let flr_dmd0: Array3<f64> = f_lr(
                diff_p.view(),
                self.properties.s().unwrap(),
                grad_s.view(),
                self.properties.gamma_lr_ao().unwrap(),
                g1_lr_ao.view(),
                self.n_atoms,
                self.n_orbs,
            );
            // -0.25 * F_lr[diff_p] * diff_p
            gradient = gradient
                - 0.25
                    * flr_dmd0
                        .view()
                        .into_shape((3 * self.n_atoms, self.n_orbs * self.n_orbs))
                        .unwrap()
                        .dot(&diff_p.into_shape(self.n_orbs * self.n_orbs).unwrap());
        }

        gradient
    }

    /// Compute the derivative of the partial charges according to equation 24 and 26 in Ref. [1]
    /// The result will be the derivative of the charge w.r.t. to all degree of freedoms of a single
    /// monomer. This means that the first dimension of the Array is the degree of freedom and the
    /// second dimension is the atom on which the charge resides.
    /// [1]: [J. Chem. Theory Comput. 2014, 10, 4801−4812](https://pubs.acs.org/doi/pdf/10.1021/ct500489d)
    fn get_grad_dq(
        &self,
        atoms: &[Atom],
        s: ArrayView2<f64>,
        grad_s: ArrayView3<f64>,
        p: ArrayView2<f64>,
    ) -> Array2<f64> {
        // get the shape of the derivative of S, it should be [f, n_orb, n_orb], where f = 3 * n_atoms
        let (f, n_orb, _): (usize, usize, usize) = grad_s.dim();

        // reshape S' so that the last dimension can be contracted with the density matrix
        let grad_s_2d: ArrayView2<f64> = grad_s.into_shape([f * n_orb, n_orb]).unwrap();

        // compute W according to eq. 26 in the reference stated above in matrix fashion
        // W_(mu nu)^a = -1/2 sum_(rho sigma) P_(mu rho) dS_(rho sigma) / dR_(a x) P_(sigma nu)
        // Implementation:
        // S'[f * rho, sigma] . P[sigma, nu] -> X1[f * rho, nu]
        // X1.T[nu, f * rho]       --reshape--> X2[nu * f, rho]
        // X2[nu * f, rho]    . P[mu, rho]   -> X3[nu * f, mu];  since P is symmetric -> P = P.T
        // X3[nu * f, mu]          --reshape--> X3[nu, f * mu]
        // W.T[f * mu, nu]    . S[mu, nu|    -> WS[f * mu, mu] since S is symmetric -> S = S.T
        let w_s: Array2<f64> = -0.5
            * grad_s_2d
                .dot(&p)
                .reversed_axes()
                .as_standard_layout()
                .into_shape([n_orb * f, n_orb])
                .unwrap()
                .dot(&p)
                .into_shape([n_orb, f * n_orb])
                .unwrap()
                .reversed_axes()
                .as_standard_layout()
                .dot(&s);

        // compute P . S' and contract their last dimension
        let d_grad_s: Array2<f64> = grad_s_2d.dot(&p);

        // do the sum of both terms
        let w_plus_ps: Array3<f64> = (&w_s + &d_grad_s).into_shape([f, n_orb, n_orb]).unwrap();

        // sum over mu where mu is on atom a
        let mut grad_dq: Array2<f64> = Array2::zeros([f, self.n_atoms]);
        let mut mu: usize = 0;
        for (idx, atom) in atoms.iter().enumerate() {
            for _ in atom.valorbs.iter() {
                grad_dq
                    .slice_mut(s![.., idx])
                    .add_assign(&w_plus_ps.slice(s![.., mu, mu]));
                mu += 1;
            }
        }

        // Shape of returned Array: [f, n_atoms], f = 3 * n_atoms
        grad_dq
    }
}
