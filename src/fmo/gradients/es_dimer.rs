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

    pub fn es_dimer_gradient_test(&mut self) -> Array1<f64> {
        let mut gradient: Array1<f64> = Array1::zeros([3 * self.atoms.len()]);

        // A reference to the charge differences and gamma matrix for all atoms is created.
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();

        for pair in self.esd_pairs.iter() {
            // References to the corresponding monomers
            let m_i: &Monomer = &self.monomers[pair.i];
            let m_j: &Monomer = &self.monomers[pair.j];

            // grad dqs
            let grad_dq_i: ArrayView2<f64> = m_i.properties.grad_dq().unwrap();
            let grad_dq_j: ArrayView2<f64> = m_j.properties.grad_dq().unwrap();
            let gamma_i: Array1<f64> = gamma
                .slice(s![m_i.slice.atom, m_j.slice.atom])
                .dot(&dq.slice(s![m_j.slice.atom]));
            let gamma_j: Array1<f64> = gamma
                .slice(s![m_j.slice.atom, m_i.slice.atom])
                .dot(&dq.slice(s![m_i.slice.atom]));
            let dq_i: ArrayView1<f64> = dq.slice(s![m_i.slice.atom]);
            let dq_j: ArrayView1<f64> = dq.slice(s![m_j.slice.atom]);

            // get pair atoms
            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );
            let grad_gamma_ij: Array3<f64> =
                gamma_gradients_atomwise(&pair.gammafunction, &pair_atoms, pair_atoms.len());
            let grad_gamma_ab: Array3<f64> = grad_gamma_ij
                .slice(s![.., ..m_i.n_atoms, m_i.n_atoms..])
                .to_owned();
            let grad_gamma_ba: Array3<f64> = grad_gamma_ij
                .slice(s![.., m_i.n_atoms.., ..m_i.n_atoms])
                .to_owned();
            let grad_gamma_ab_dq: Array2<f64> = grad_gamma_ab
                .into_shape([3 * pair.n_atoms * m_i.n_atoms, m_j.n_atoms])
                .unwrap()
                .dot(&dq_j)
                .into_shape([3 * pair.n_atoms, m_i.n_atoms])
                .unwrap();
            let grad_gamma_ba_dq: Array2<f64> = grad_gamma_ba
                .into_shape([3 * pair.n_atoms * m_j.n_atoms, m_i.n_atoms])
                .unwrap()
                .dot(&dq_i)
                .into_shape([3 * pair.n_atoms, m_j.n_atoms])
                .unwrap();

            for nc in 0..3 {
                for na in 0..pair.n_atoms {
                    let grad_idx: usize = na * 3 + nc;
                    // a in I
                    if na < m_i.n_atoms {
                        let mut gradient_slice: ArrayViewMut1<f64> =
                            gradient.slice_mut(s![m_i.slice.grad]);
                        // first term
                        let tmp_1: f64 = dq_i[na] * grad_gamma_ab_dq[[grad_idx, na]];
                        // second term
                        let tmp_2: f64 = grad_dq_i[[grad_idx, na]] * gamma_i[na];
                        // add to gradient
                        gradient_slice[grad_idx] += &(tmp_1 + tmp_2);
                    }
                    // a in J
                    else {
                        let mut gradient_slice: ArrayViewMut1<f64> =
                            gradient.slice_mut(s![m_j.slice.grad]);
                        let nat: usize = na - m_i.n_atoms;
                        let grad_idx_2: usize = grad_idx - 3 * m_i.n_atoms;
                        // first term
                        let tmp_1: f64 = dq_j[nat] * grad_gamma_ba_dq[[grad_idx, nat]];
                        // second term
                        let tmp_2: f64 = grad_dq_j[[grad_idx_2, nat]] * gamma_j[nat];
                        // add to gradient
                        gradient_slice[grad_idx_2] += &(tmp_1 + tmp_2);
                    }
                }
            }
        }

        gradient
    }
}
