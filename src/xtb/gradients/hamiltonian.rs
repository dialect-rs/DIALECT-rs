use crate::{
    constants::BOHR_TO_ANGS,
    defaults::PROXIMITY_CUTOFF,
    xtb::{
        initialization::{atom::XtbAtom, basis::Basis},
        parameters::*,
        scc::hamiltonian::{
            calculate_coordination_numbers, calculate_pair_scaling_param,
            get_hueckel_constants_new, get_pi_term, get_self_energy_values_new,
        },
    },
};
use nalgebra::Vector3;
use ndarray::prelude::*;
use std::ops::AddAssign;

// pub fn calculate_h0_gradient_xtb1(
//     n_orbs: usize,
//     atoms: &[XtbAtom],
//     s: ArrayView2<f64>,
//     grad_s: ArrayView3<f64>,
//     basis: &Basis,
// ) -> Array3<f64> {
//     // get number of atoms
//     let n_atoms: usize = atoms.len();
//     // init empty array
//     let mut h0_grad: Array3<f64> = Array3::zeros([3 * n_atoms, n_orbs, n_orbs]);
//     // calculate the coordination numbers
//     let cn_numbers: Array1<f64> = calculate_coordination_numbers(atoms);
//     // get the gradients of the coordination numbers
//     let cn_number_grads: Array2<f64> = calculate_coordination_number_gradients(atoms);
//
//     // iterate over the basis functions
//     for (i, funci) in basis.basis_functions.iter().enumerate() {
//         let atomi: &XtbAtom = &atoms[funci.atom_index];
//         let cn_1: f64 = cn_numbers[funci.atom_index];
//         let at_i: usize = funci.atom_index;
//         let cn_grad_i: ArrayView1<f64> = cn_number_grads.slice(s![.., at_i]);
//
//         for (j, funcj) in basis.basis_functions.iter().enumerate() {
//             let atomj: &XtbAtom = &atoms[funcj.atom_index];
//             let cn_2: f64 = cn_numbers[funcj.atom_index];
//             let at_j: usize = funcj.atom_index;
//             let cn_grad_j: ArrayView1<f64> = cn_number_grads.slice(s![.., at_j]);
//
//             let r_vector: Vector3<f64> = atomi - atomj;
//             let distance: f64 = (atomi - atomj).norm();
//             if distance < PROXIMITY_CUTOFF {
//                 if i == j {
//                     // self energies
//                     let self_energy_term: f64 = get_self_energy_values(
//                         funci,
//                         funcj,
//                         atomi.number,
//                         atomj.number,
//                         cn_1,
//                         cn_2,
//                     );
//                     // assign value to the gtadient
//                     let val: Array1<f64> = self_energy_term * &grad_s.slice(s![.., i, j]);
//                     h0_grad.slice_mut(s![.., i, j]).add_assign(&val);
//                     // get the contribution to the gradient of the coordination number gradients
//                     let s_val: f64 = s[[i, j]];
//                     let grad_val: Array1<f64> = get_self_energy_value_grad(
//                         funci.angular_momentum,
//                         funci.polarization,
//                         atomi.number,
//                         cn_grad_i.view(),
//                     ) * s_val;
//                     h0_grad.slice_mut(s![.., i, j]).add_assign(&grad_val);
//                 } else if i != j {
//                     let scaling_constant: f64 = calculate_pair_scaling_param(
//                         atomi.number,
//                         atomj.number,
//                         funci.angular_momentum,
//                         funcj.angular_momentum,
//                     );
//                     // calculate the square of the electronegativity difference
//                     let pauling_diff: f64 = (PAULING_EN[atomi.number as usize - 1]
//                         - PAULING_EN[atomj.number as usize - 1])
//                         .powi(2);
//                     // term of H0: 1.0 + k_EN * \Delta EN_AB^2
//                     let en_term: f64 = if funci.polarization == false && funcj.polarization == false
//                     {
//                         (1.0 + EN_SHELL_PARAM * pauling_diff)
//                     } else {
//                         1.0
//                     };
//                     // get the Hueckel constants term
//                     let hueckel_const: f64 =
//                         get_hueckel_constants(funci, funcj, atomi.number, atomj.number);
//                     // self energies
//                     let self_energy_term: f64 = get_self_energy_values(
//                         funci,
//                         funcj,
//                         atomi.number,
//                         atomj.number,
//                         cn_1,
//                         cn_2,
//                     );
//                     // calculate the pi term
//                     let pi_term: f64 = get_pi_term(
//                         distance,
//                         atomi.number as usize,
//                         atomj.number as usize,
//                         funci.angular_momentum,
//                         funcj.angular_momentum,
//                     );
//
//                     // combine the parts of the equation
//                     let h0_val: f64 =
//                         scaling_constant * hueckel_const * self_energy_term * en_term * pi_term;
//                     // get gradient contribution of the overlap matrix
//                     let grad_contribution: Array1<f64> = h0_val * &grad_s.slice(s![.., i, j]);
//                     h0_grad
//                         .slice_mut(s![.., i, j])
//                         .add_assign(&grad_contribution);
//
//                     if funci.atom_index != funcj.atom_index {
//                         // gradient contribution of the pi function
//                         let pi_grad: Array1<f64> = get_pi_term_gradient(
//                             &r_vector,
//                             distance,
//                             atomi.number as usize,
//                             atomj.number as usize,
//                             funci.angular_momentum,
//                             funcj.angular_momentum,
//                         );
//                         let pi_h0_grad_val: Array1<f64> = scaling_constant
//                             * hueckel_const
//                             * self_energy_term
//                             * s[[i, j]]
//                             * en_term
//                             * &pi_grad;
//                         h0_grad
//                             .slice_mut(s![at_i * 3..at_i * 3 + 3, i, j])
//                             .add_assign(&pi_h0_grad_val);
//                         h0_grad
//                             .slice_mut(s![at_i * 3..at_i * 3 + 3, j, i])
//                             .add_assign(&pi_h0_grad_val);
//                     }
//
//                     // get cn number gradient contribution
//                     let cn_grad_val_i: Array1<f64> = get_self_energy_value_grad(
//                         funci.angular_momentum,
//                         funci.polarization,
//                         atomi.number,
//                         cn_grad_i.view(),
//                     );
//                     let cn_grad_val_j: Array1<f64> = get_self_energy_value_grad(
//                         atomj.number,
//                         cn_grad_j.view(),
//                         shell_j.
//                     );
//                     let h_val: f64 =
//                         scaling_constant * hueckel_const * en_term * pi_term * s[[i, j]];
//
//                     let grad_contribution: Array1<f64> =
//                         0.5 * h_val * &(&cn_grad_val_i + &cn_grad_val_j);
//                     h0_grad
//                         .slice_mut(s![.., i, j])
//                         .add_assign(&grad_contribution);
//                 }
//             }
//         }
//     }
//     h0_grad
// }

pub fn calculate_h0_gradient_xtb1_new(
    n_orbs: usize,
    atoms: &[XtbAtom],
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    basis: &Basis,
) -> Array3<f64> {
    // get number of atoms
    let n_atoms: usize = atoms.len();
    // init empty array
    let mut h0_grad: Array3<f64> = Array3::zeros([3 * n_atoms, n_orbs, n_orbs]);
    // calculate the coordination numbers
    let cn_numbers: Array1<f64> = calculate_coordination_numbers(atoms);
    // get the gradients of the coordination numbers
    let cn_number_grads: Array2<f64> = calculate_coordination_number_gradients(atoms);

    // iterate over shells
    for shell_i in basis.shells.iter() {
        // loop over number of angular components
        for idx_i in (shell_i.sph_start..shell_i.sph_end) {
            // get the atom and the cn numbers
            let atomi: &XtbAtom = &atoms[shell_i.atom_index];
            let cn_1: f64 = cn_numbers[shell_i.atom_index];
            let at_i: usize = shell_i.atom_index;
            let cn_grad_i: ArrayView1<f64> = cn_number_grads.slice(s![.., at_i]);

            // iterate over the next shells
            for shell_j in basis.shells.iter() {
                let atomj: &XtbAtom = &atoms[shell_j.atom_index];
                let cn_2: f64 = cn_numbers[shell_j.atom_index];
                let at_j: usize = shell_j.atom_index;
                let cn_grad_j: ArrayView1<f64> = cn_number_grads.slice(s![.., at_j]);

                let r_vector: Vector3<f64> = atomi - atomj;
                let distance: f64 = (atomi - atomj).norm();
                if distance < PROXIMITY_CUTOFF {
                    // loop over number of angular components
                    for idx_j in (shell_j.sph_start..shell_j.sph_end) {
                        if idx_i == idx_j {
                            // self energies
                            let self_energy_term: f64 = get_self_energy_values_new(
                                atomi.number,
                                atomj.number,
                                cn_1,
                                cn_2,
                                shell_i.shell_index,
                                shell_j.shell_index,
                            );
                            // assign value to the gtadient
                            let val: Array1<f64> =
                                self_energy_term * &grad_s.slice(s![.., idx_i, idx_j]);
                            h0_grad.slice_mut(s![.., idx_i, idx_j]).add_assign(&val);
                            // get the contribution to the gradient of the coordination number gradients
                            let s_val: f64 = s[[idx_i, idx_j]];
                            let grad_val: Array1<f64> = get_self_energy_value_grad(
                                atomi.number,
                                cn_grad_i.view(),
                                shell_i.shell_index,
                            ) * s_val;
                            h0_grad
                                .slice_mut(s![.., idx_i, idx_j])
                                .add_assign(&grad_val);
                        } else if idx_i != idx_j {
                            let scaling_constant: f64 = calculate_pair_scaling_param(
                                atomi.number,
                                atomj.number,
                                shell_i.angular_momentum,
                                shell_j.angular_momentum,
                                shell_i.shell_index,
                                shell_j.shell_index,
                            );
                            // calculate the square of the electronegativity difference
                            let pauling_diff: f64 = (PAULING_EN[atomi.number as usize - 1]
                                - PAULING_EN[atomj.number as usize - 1])
                                .powi(2);
                            // term of H0: 1.0 + k_EN * \Delta EN_AB^2
                            let en_term: f64 =
                                if shell_i.polarization == false && shell_j.polarization == false {
                                    (1.0 + EN_SHELL_PARAM * pauling_diff)
                                } else {
                                    1.0
                                };
                            // get the Hueckel constants term
                            let hueckel_const: f64 = get_hueckel_constants_new(
                                atomi.number,
                                atomj.number,
                                shell_i.angular_momentum,
                                shell_j.angular_momentum,
                                shell_i.polarization,
                                shell_j.polarization,
                            );
                            // self energies
                            let self_energy_term: f64 = get_self_energy_values_new(
                                atomi.number,
                                atomj.number,
                                cn_1,
                                cn_2,
                                shell_i.shell_index,
                                shell_j.shell_index,
                            );
                            // calculate the pi term
                            let pi_term: f64 = get_pi_term(
                                distance,
                                atomi.number as usize,
                                atomj.number as usize,
                                shell_i.angular_momentum,
                                shell_j.angular_momentum,
                            );
                            // combine the parts of the equation
                            let h0_val: f64 = scaling_constant
                                * hueckel_const
                                * self_energy_term
                                * en_term
                                * pi_term;
                            // get gradient contribution of the overlap matrix
                            let grad_contribution: Array1<f64> =
                                h0_val * &grad_s.slice(s![.., idx_i, idx_j]);
                            h0_grad
                                .slice_mut(s![.., idx_i, idx_j])
                                .add_assign(&grad_contribution);

                            if shell_i.atom_index != shell_j.atom_index {
                                // gradient contribution of the pi function
                                let pi_grad: Array1<f64> = get_pi_term_gradient(
                                    &r_vector,
                                    distance,
                                    atomi.number as usize,
                                    atomj.number as usize,
                                    shell_i.angular_momentum,
                                    shell_j.angular_momentum,
                                );
                                let pi_h0_grad_val: Array1<f64> = scaling_constant
                                    * hueckel_const
                                    * self_energy_term
                                    * s[[idx_i, idx_j]]
                                    * en_term
                                    * &pi_grad;
                                h0_grad
                                    .slice_mut(s![at_i * 3..at_i * 3 + 3, idx_i, idx_j])
                                    .add_assign(&pi_h0_grad_val);
                                h0_grad
                                    .slice_mut(s![at_i * 3..at_i * 3 + 3, idx_j, idx_i])
                                    .add_assign(&pi_h0_grad_val);
                            }

                            // get cn number gradient contribution
                            let cn_grad_val_i: Array1<f64> = get_self_energy_value_grad(
                                atomi.number,
                                cn_grad_i.view(),
                                shell_i.shell_index,
                            );
                            let cn_grad_val_j: Array1<f64> = get_self_energy_value_grad(
                                atomj.number,
                                cn_grad_j.view(),
                                shell_j.shell_index,
                            );
                            let h_val: f64 = scaling_constant
                                * hueckel_const
                                * en_term
                                * pi_term
                                * s[[idx_i, idx_j]];

                            let grad_contribution: Array1<f64> =
                                0.5 * h_val * &(&cn_grad_val_i + &cn_grad_val_j);
                            h0_grad
                                .slice_mut(s![.., idx_i, idx_j])
                                .add_assign(&grad_contribution);
                        }
                    }
                }
            }
        }
    }

    h0_grad
}

fn get_self_energy_value_grad(
    // l: usize,
    // polarization: bool,
    z: u8,
    cn_grad: ArrayView1<f64>,
    shell_idx: usize,
) -> Array1<f64> {
    let z_idx: usize = (z - 1) as usize;
    // // idx for self energies
    // let en_idx: usize = if !polarization { l } else { l + 1 };

    // let k_cn_params: [f64; 3] = [0.006, -0.003, -0.005];
    // let en_grad: Array1<f64> = (HAMILTONIAN_SELF_ENERGY[z_idx][shell_idx] / HARTREE_TO_EV)
    //     * HAMILTONIAN_KCN_VALUES[z_idx][shell_idx]
    //     * &cn_grad;
    let en_grad: Array1<f64> = -HAMILTONIAN_KCN_VALUES[z_idx][shell_idx] * &cn_grad;
    // let en_grad: Array1<f64> =
    //     (HAMILTONIAN_SELF_ENERGY[z_idx][en_idx] / HARTREE_TO_EV) * k_cn_params[l] * &cn_grad;

    en_grad
}

fn get_pi_term_gradient(
    r_vector: &Vector3<f64>,
    r_ab: f64,
    z_1: usize,
    z_2: usize,
    l_1: usize,
    l_2: usize,
) -> Array1<f64> {
    let z_idx_1: usize = z_1 - 1;
    let z_idx_2: usize = z_2 - 1;

    // get the polynomial values for both atoms
    let k_poly_1: f64 = HAMILTONIAN_SHELL_POLYNOMIALS[z_idx_1][l_1] * 0.01;
    let k_poly_2: f64 = HAMILTONIAN_SHELL_POLYNOMIALS[z_idx_2][l_2] * 0.01;

    // get the covalence radii
    let cov_1: f64 = COV_RADII[z_idx_1] / BOHR_TO_ANGS;
    let cov_2: f64 = COV_RADII[z_idx_2] / BOHR_TO_ANGS;
    let cov_sum: f64 = cov_1 + cov_2;
    let distance_term: f64 = (r_ab / cov_sum).sqrt();
    // calculate pi term
    let deriv_val: f64 = (1.0 + k_poly_1 * distance_term) * k_poly_2
        / (2.0 * cov_sum * distance_term)
        + (1.0 + k_poly_2 * distance_term) * k_poly_1 / (2.0 * distance_term * cov_sum);

    let mut r: Vector3<f64> = r_vector.clone();
    r /= r_ab;
    r *= deriv_val;
    let v: Array1<f64> = array![r.x, r.y, r.z];

    v
}

pub fn calculate_coordination_number_gradients(atoms: &[XtbAtom]) -> Array2<f64> {
    let mut grad_cn_numbers: Array2<f64> = Array2::zeros([3 * atoms.len(), atoms.len()]);

    for (i, atomi) in atoms.iter().enumerate() {
        let mut grad_i: Array1<f64> = Array::zeros([3]);
        for (j, atomj) in atoms.iter().enumerate() {
            if i != j {
                let cov_i: f64 = COV_RADII_CN[atomi.number as usize - 1] / BOHR_TO_ANGS;
                let cov_j: f64 = COV_RADII_CN[atomj.number as usize - 1] / BOHR_TO_ANGS;

                let mut r_vector: Vector3<f64> = atomi.xyz - atomj.xyz;
                let distance: f64 = ((atomi.xyz.x - atomj.xyz.x).powi(2)
                    + (atomi.xyz.y - atomj.xyz.y).powi(2)
                    + (atomi.xyz.z - atomj.xyz.z).powi(2))
                .sqrt();
                r_vector /= distance;

                // get the derivative value
                let exp_val: f64 = (-16.0 * (4.0 / 3.0 * (cov_i + cov_j) / distance - 1.0)).exp();
                let deriv_val: f64 = -64.0 * (cov_i + cov_j) * exp_val
                    / (3.0 * distance.powi(2) * (exp_val + 1.0).powi(2));
                r_vector *= deriv_val;
                let v = array![r_vector.x, r_vector.y, r_vector.z];
                grad_i = &grad_i + &v;

                grad_cn_numbers
                    .slice_mut(s![i * 3..i * 3 + 3, j])
                    .assign(&v);
            }
        }
        grad_cn_numbers
            .slice_mut(s![i * 3..i * 3 + 3, i])
            .assign(&grad_i);
    }

    grad_cn_numbers
}
