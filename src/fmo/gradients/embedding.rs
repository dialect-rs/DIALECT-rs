use crate::fmo::*;
use crate::initialization::*;
use crate::scc::gamma_approximation::gamma_gradients_atomwise_2d;
use crate::scc::*;
use crate::utils::Timer;
use nalgebra::Vector3;
use ndarray::prelude::*;
use ndarray::RawData;
use std::iter::FromIterator;
use std::ops::{AddAssign, SubAssign};
//use rayon::iter::ParallelIterator;
//use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;

impl SuperSystem<'_> {
    /// Computes and returns the gradient of the embedding energy.
    pub fn embedding_gradient(&mut self) -> Array1<f64> {
        // The gradient of the embedding energy is initialized as an array with zeros.
        let mut gradient_array: Array2<f64> =
            Array2::zeros([3 * self.atoms.len(), self.pairs.len()]);

        // A reference to the charge differences and gamma matrix for all atoms is created.
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();

        // The charge differences are broadcast into the shape the gradients.
        let dq_f = dq
            .broadcast([3, self.atoms.len()])
            .unwrap()
            .reversed_axes()
            .as_standard_layout()
            .into_shape([3 * self.atoms.len()])
            .unwrap()
            .to_owned();

        // Reference to the derivative of the charges.
        let mut grad_dq: ArrayView1<f64> = self.properties.grad_dq_diag().unwrap();

        // TODO: it is not neccessary to calculate the derivative of gamma two times. this should be
        // improved! it is already computed in the gradient of the monomer/pair
        let grad_gamma_sparse: Array2<f64> =
            gamma_gradients_atomwise_2d(&self.gammafunction, &self.atoms, self.atoms.len());
        let mut grad_gamma_dot_dq: Array1<f64> = grad_gamma_sparse.dot(&dq);

        // Begin of the loop to compute the gradient of the embedding energy for each pair.
        self.pairs
            .par_iter()
            .zip(gradient_array.axis_iter_mut(Axis(1)).into_par_iter())
            .for_each(|(pair, mut gradient)| {
                // References to the corresponding monomers.
                let m_i: &Monomer = &self.monomers[pair.i];
                let m_j: &Monomer = &self.monomers[pair.j];

                // If the derivative is w.r.t to an atom that is within this pair:
                // The first part of the equation reads:
                // dDeltaE_IJ^V/dR_a x = DDq_a^IJ sum_(K!=I,J)^(N) sum_(C in K) Dq_C^K dgamma_(a C)/dR_(a x)

                // Reference to the DDq_a^IJ (difference of charge difference between pair and monomer)
                let delta_dq: ArrayView1<f64> = pair.properties.delta_dq().unwrap();

                // DDq is broadcasted into the shape of the gradients.
                let delta_dq_f = delta_dq
                    .broadcast([3, pair.n_atoms])
                    .unwrap()
                    .reversed_axes()
                    .as_standard_layout()
                    .into_shape([3 * pair.n_atoms])
                    .unwrap()
                    .to_owned();

                let self_interaction_i: Array1<f64> = &grad_gamma_sparse
                    .slice(s![m_i.slice.grad, m_i.slice.atom])
                    .dot(&dq.slice(s![m_i.slice.atom]))
                    + &grad_gamma_sparse
                        .slice(s![m_i.slice.grad, m_j.slice.atom])
                        .dot(&dq.slice(s![m_j.slice.atom]));

                let self_interaction_j: Array1<f64> = &grad_gamma_sparse
                    .slice(s![m_j.slice.grad, m_i.slice.atom])
                    .dot(&dq.slice(s![m_i.slice.atom]))
                    + &grad_gamma_sparse
                        .slice(s![m_j.slice.grad, m_j.slice.atom])
                        .dot(&dq.slice(s![m_j.slice.atom]));

                // The gradient for a in I is computed and assigned.
                gradient.slice_mut(s![m_i.slice.grad]).add_assign(
                    &(&delta_dq_f.slice(s![..3 * m_i.n_atoms])
                        * &(&grad_gamma_dot_dq.slice(s![m_i.slice.grad]) - &self_interaction_i)),
                );

                // The gradient for a in J is computed and assigned.
                gradient.slice_mut(s![m_j.slice.grad]).add_assign(
                    &(&delta_dq_f.slice(s![3 * m_i.n_atoms..])
                        * &(&grad_gamma_dot_dq.slice(s![m_j.slice.grad]) - &self_interaction_j)),
                );

                // Right hand side of Eq. 24, but a is still in I,J
                // The difference between the derivative of the charge differences between the monomer
                // and the dimer is computed.
                let grad_delta_dq: Array1<f64> = get_grad_delta_dq(pair, m_i, m_j);

                // The electrostatic potential (ESP) is collected from the corresponding monomers.
                let mut esp_ij: Array1<f64> = Array1::zeros([pair.n_atoms]);
                esp_ij.slice_mut(s![..m_i.n_atoms]).assign(
                    &(&m_i.properties.esp_q().unwrap()
                        - &gamma
                            .slice(s![m_i.slice.atom, m_j.slice.atom])
                            .dot(&dq.slice(s![m_j.slice.atom]))),
                );
                esp_ij.slice_mut(s![m_i.n_atoms..]).assign(
                    &(&m_j.properties.esp_q().unwrap()
                        - &gamma
                            .slice(s![m_j.slice.atom, m_i.slice.atom])
                            .dot(&dq.slice(s![m_i.slice.atom]))),
                );

                // The ESP is transformed into the shape of the gradient.
                let esp_ij = esp_ij
                    .broadcast([3, pair.n_atoms])
                    .unwrap()
                    .reversed_axes()
                    .as_standard_layout()
                    .into_shape([3 * pair.n_atoms])
                    .unwrap()
                    .to_owned();

                // The (elementwise) product of the ESP with the derivative of the pair charge
                // differences is computed.
                let gddq_esp: Array1<f64> = &grad_delta_dq * &esp_ij;

                // The gradient of the rhs for a in I is assigned.
                gradient
                    .slice_mut(s![m_i.slice.grad])
                    .add_assign(&gddq_esp.slice(s![..3 * m_i.n_atoms]));

                // The gradient of the rhs for a in J is assigned.
                gradient
                    .slice_mut(s![m_j.slice.grad])
                    .add_assign(&gddq_esp.slice(s![3 * m_i.n_atoms..]));

                // Start of the computation if the derivative is w.r.t to an atom that is not in
                // this pair. So that a in K where K != I,J.

                // The matrix vector product of the gamma matrix derivative with DDq is computed for a in I
                let mut dg_ddq: Array1<f64> = grad_gamma_sparse
                    .slice(s![0.., m_i.slice.atom])
                    .dot(&delta_dq.slice(s![..m_i.n_atoms]));

                // and for a in J.
                dg_ddq += &grad_gamma_sparse
                    .slice(s![0.., m_j.slice.atom])
                    .dot(&delta_dq.slice(s![m_i.n_atoms..]));

                // Since K != I,J => the elements where K = I,J are set to zero.
                dg_ddq
                    .slice_mut(s![m_i.slice.grad])
                    .assign(&Array1::zeros([3 * m_i.n_atoms]));
                dg_ddq
                    .slice_mut(s![m_j.slice.grad])
                    .assign(&Array1::zeros([3 * m_j.n_atoms]));

                // The (elementwise) product with the charge differences is computed and assigned.
                gradient += &(&dg_ddq * &dq_f);

                // Start of the computation of the right hand side of Eq. 25.
                // A in monomer I
                let mut ddq_gamma: Array1<f64> = delta_dq
                    .slice(s![..m_i.n_atoms])
                    .dot(&gamma.slice(s![m_i.slice.atom, 0..]));

                // A in monomer J
                ddq_gamma += &delta_dq
                    .slice(s![m_i.n_atoms..])
                    .dot(&gamma.slice(s![m_j.slice.atom, 0..]));

                // Since K != I,J => the elements were K = I,J are set to zero.
                ddq_gamma
                    .slice_mut(s![m_i.slice.atom])
                    .assign(&Array1::zeros([m_i.n_atoms]));
                ddq_gamma
                    .slice_mut(s![m_j.slice.atom])
                    .assign(&Array1::zeros([m_j.n_atoms]));

                // transform the Array into the shape of the gradients and multiply it with the derivative
                // of the charge (differences)
                gradient += &(&grad_dq
                    * &ddq_gamma
                        .broadcast([3, self.atoms.len()])
                        .unwrap()
                        .reversed_axes()
                        .as_standard_layout()
                        .into_shape([3 * self.atoms.len()])
                        .unwrap());
            });

        return gradient_array.sum_axis(Axis(1));
    }
}

fn get_grad_delta_dq(pair: &Pair, m_i: &Monomer, m_j: &Monomer) -> Array1<f64> {
    // get the derivatives of the charge differences w.r.t to the each degree of freedom
    let grad_dq: ArrayView2<f64> = pair.properties.grad_dq().unwrap();
    let grad_dq_i: ArrayView2<f64> = m_i.properties.grad_dq().unwrap();
    let grad_dq_j: ArrayView2<f64> = m_j.properties.grad_dq().unwrap();

    // compute the difference between dimers and monomers and take the diagonal values
    let mut grad_delta_dq_2d: Array2<f64> = grad_dq.to_owned();

    //difference for monomer i
    grad_delta_dq_2d
        .slice_mut(s![..(3 * m_i.n_atoms), ..m_i.n_atoms])
        .sub_assign(&grad_dq_i);

    // difference for monomer j
    grad_delta_dq_2d
        .slice_mut(s![(3 * m_i.n_atoms).., m_i.n_atoms..])
        .sub_assign(&grad_dq_j);

    let grad_delta_dq_3d: Array3<f64> = grad_delta_dq_2d
        .into_shape([3, pair.n_atoms, pair.n_atoms])
        .unwrap();

    diag_of_last_dimensions(grad_delta_dq_3d)
}

pub fn diag_of_last_dimensions<S>(data: ArrayBase<S, Ix3>) -> Array1<f64>
where
    S: ndarray::Data<Elem = f64>,
{
    let (a, b, c): (usize, usize, usize) = data.dim();
    assert_eq!(b, c, "The last two dimension should have the same length");

    // A temporary array to store the values is created.
    let mut grad_charge: Array2<f64> = Array2::zeros([a, b]);

    // The diagonal of each of the three dimensions is saved.
    for i in 0..a {
        grad_charge
            .slice_mut(s![i, ..])
            .assign(&data.slice(s![i, .., ..]).diag());
    }
    // The gradient of the charges are reshaped into a one dimensional array.
    grad_charge.into_shape([a * b]).unwrap()
}
