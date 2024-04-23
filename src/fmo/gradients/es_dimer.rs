use crate::fmo::helpers::get_pair_slice;
use crate::fmo::*;
use crate::initialization::Atom;
use crate::scc::gamma_approximation::{gamma_gradients_atomwise, gamma_gradients_atomwise_2d};
use ndarray::prelude::*;

impl SuperSystem<'_> {
    /// Computes and returns the gradient of the embedding energy.
    pub fn es_dimer_gradient(&mut self) -> Array1<f64> {
        let mut gradient: Array1<f64> = Array1::zeros([3 * self.atoms.len()]);

        // A reference to the charge differences and gamma matrix for all atoms is created.
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();
        let grad_dq: ArrayView1<f64> = self.properties.grad_dq_diag().unwrap();

        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();

        // The charge differences are broadcast into the shape the gradients.
        let dq_f: Array1<f64> = dq
            .broadcast([3, self.atoms.len()])
            .unwrap()
            .reversed_axes()
            .as_standard_layout()
            .into_shape([3 * self.atoms.len()])
            .unwrap()
            .to_owned();

        // TODO: it is not neccessary to calculate the derivative of gamma two times. this should be
        // improved! it is already computed in the gradient of the monomer/pair
        let grad_gamma_sparse: Array2<f64> =
            gamma_gradients_atomwise_2d(&self.gammafunction, &self.atoms, self.atoms.len());

        for pair in self.esd_pairs.iter() {
            // References to the corresponding monomers
            let m_i: &Monomer = &self.monomers[pair.i];
            let m_j: &Monomer = &self.monomers[pair.j];

            // a in I
            let mut gradient_slice: ArrayViewMut1<f64> = gradient.slice_mut(s![m_i.slice.grad]);

            // lhs. of Eq. 29 in Ref. [1]
            gradient_slice += &(&dq_f.slice(s![m_i.slice.grad])
                * &grad_gamma_sparse
                    .slice(s![m_i.slice.grad, m_j.slice.atom])
                    .dot(&dq.slice(s![m_j.slice.atom])));

            // rhs of Eq. 29 in Ref. [1]
            gradient_slice += &(&grad_dq.slice(s![m_i.slice.grad])
                * &gamma
                    .slice(s![m_i.slice.atom, m_j.slice.atom])
                    .dot(&dq.slice(s![m_j.slice.atom]))
                    .broadcast([3, m_i.n_atoms])
                    .unwrap()
                    .reversed_axes()
                    .as_standard_layout()
                    .into_shape([3 * m_i.n_atoms])
                    .unwrap());

            // a in J
            let mut gradient_slice: ArrayViewMut1<f64> = gradient.slice_mut(s![m_j.slice.grad]);

            // lhs of Eq. 30 in Ref. [1]
            gradient_slice += &(&dq_f.slice(s![m_j.slice.grad])
                * &grad_gamma_sparse
                    .slice(s![m_j.slice.grad, m_i.slice.atom])
                    .dot(&dq.slice(s![m_i.slice.atom])));

            // rhs of Eq. 30 in Ref [1]
            gradient_slice += &(&grad_dq.slice(s![m_j.slice.grad])
                * &gamma
                    .slice(s![m_j.slice.atom, m_i.slice.atom])
                    .dot(&dq.slice(s![m_i.slice.atom]))
                    .broadcast([3, m_j.n_atoms])
                    .unwrap()
                    .reversed_axes()
                    .as_standard_layout()
                    .into_shape([3 * m_j.n_atoms])
                    .unwrap());
        }

        gradient
    }
}
