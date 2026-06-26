use dialect_utilities::scc_helpers::aovec_to_aomat;
use crate::fmo::monomer::XtbMonomer;
use crate::gradients::hamiltonian::calculate_h0_gradient_xtb1_new;
use crate::gradients::helpers::coul_third_order_grad_contribution_xtb;
use crate::initialization::atom::XtbAtom;
use crate::initialization::basis::Basis;
use crate::integrals::{
    calc_overlap_derivative_d_shells, calc_overlap_matrix_obs_derivs_new,
    obara_saika_derivatives_all,
};
use crate::parameters::{COUL_THIRD_ORDER_ATOM, REP_ALPHA_PARAMS, REP_Z_EFF_PARAMS};
use crate::scc::gamma_matrix::gamma_gradient_xtb_new;
use nalgebra::Vector3;
use ndarray::prelude::*;
use ndarray::Zip;

impl XtbMonomer<'_> {
    /// Ground-state xTB gradient of a single FMO monomer (in the embedding
    /// field of the other fragments), used as a building block of the FMO
    /// gradient.
    pub fn ground_state_gradient(&mut self, atoms: &[XtbAtom]) -> Array1<f64> {
        // take references/views to the necessary properties from the scc calculation
        let gamma: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        let dq_ao: ArrayView1<f64> = self.properties.dq_ao().unwrap();
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // calculate the gradient of the overlap matrix
        let grad_s: Array3<f64> = calc_overlap_matrix_obs_derivs_new(&self.basis, self.n_atoms);
        // calculate the gradient of the charge differences
        let grad_dq: Array2<f64> = get_grad_dq_xtb(s, grad_s.view(), p, self.n_atoms, self.n_orbs);
        // calculate the gradient of the H0 matrix
        let grad_h0: Array3<f64> =
            calculate_h0_gradient_xtb1_new(self.n_orbs, &atoms, s, grad_s.view(), &self.basis);
        // and reshape them into a 2D array. the last two dimension (number of orbitals) are compressed
        // into one dimension to be able to just matrix-matrix products for the computation of the gradient
        let grad_s_2d: ArrayView2<f64> = grad_s
            .view()
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap();
        let grad_h0_2d: ArrayView2<f64> = grad_h0
            .view()
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap();

        // calculate the gradient of the gamma matrix
        let grad_gamma: Array3<f64> = gamma_gradient_xtb_new(
            &self.gammafunction,
            &atoms,
            &self.basis,
            self.n_atoms,
            self.n_orbs,
        );
        let grad_gamma_2d: ArrayView2<f64> = grad_gamma
            .view()
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap();

        // create array of hubbard derivatives
        let mut hubbard_derivatives: Array1<f64> = Array1::zeros(self.n_atoms);
        for (mut val, atom) in hubbard_derivatives.iter_mut().zip(atoms.iter()) {
            *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
        }
        // get the vesp contribution
        let vesp_hamiltonian = self.properties.h_coul_x().unwrap();
        let vesp_contribution = p
            .view()
            .dot(&vesp_hamiltonian.dot(&p))
            .into_shape([self.n_orbs * self.n_orbs])
            .unwrap();

        // The density matrix in vector form.
        let p_flat: ArrayView1<f64> = p.into_shape([self.n_orbs * self.n_orbs]).unwrap();

        // compute the energy weighted density matrix
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let occupations: Array1<f64> = Array::from(self.properties.occupation().unwrap().to_vec());
        let weighted_orbe = &orbe * &occupations;
        let worbe_2d: Array2<f64> = Array2::from_diag(&weighted_orbe);
        let w_new: Array2<f64> = orbs.dot(&worbe_2d.dot(&orbs.t()));
        let w: Array1<f64> = w_new.into_shape([self.n_orbs * self.n_orbs]).unwrap();
        // substract contribution of the vesp
        let w: Array1<f64> = w - vesp_contribution;

        // calculate the gradient contribution of the third order energy
        // contribution of dq**2 and gamma third order
        let dq2_gamma: Array2<f64> =
            coul_third_order_grad_contribution_xtb(&self.basis, dq, hubbard_derivatives.view());
        // multiply with the density matrix
        let coulomb_p_third_order: Array1<f64> = 0.5
            * (&p * &dq2_gamma)
                .into_shape([self.n_orbs * self.n_orbs])
                .unwrap();

        let dq_column: ArrayView2<f64> = dq_ao.clone().insert_axis(Axis(1));
        let dq_x_dq: Array1<f64> = (&dq_column.broadcast((self.n_orbs, self.n_orbs)).unwrap()
            * &dq_ao)
            .into_shape([self.n_orbs * self.n_orbs])
            .unwrap();
        let coulomb_mat: Array2<f64> = aovec_to_aomat(gamma.dot(&dq_ao).view(), self.n_orbs) * 0.5;
        let coulomb_x_p: Array1<f64> = (&p * &coulomb_mat)
            .into_shape([self.n_orbs * self.n_orbs])
            .unwrap();

        // calculation of the gradient
        // 1st part:  dH0 / dR . P
        let mut gradient: Array1<f64> = grad_h0_2d.dot(&p_flat);

        // 2nd part: dS / dR . W
        gradient -= &grad_s_2d.dot(&w);

        // 3rd part: second order Coulomb gradient part 1
        gradient += &grad_s_2d.dot(&coulomb_x_p);

        // 4th part: second order Coulomb gradient part 2
        gradient += &(0.5 * grad_gamma_2d.dot(&dq_x_dq));

        // 5th part: third order Coulomb gradient
        gradient -= &grad_s_2d.dot(&coulomb_p_third_order);

        // last part: dV_rep / dR
        gradient = gradient + grad_repulsive_energy_xtb(atoms, self.n_atoms);

        // save grad dq
        self.properties.set_grad_dq(grad_dq);

        gradient
    }
}

pub fn get_grad_dq_xtb(
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    p: ArrayView2<f64>,
    n_atoms: usize,
    n_orbs: usize,
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
    // this array now has the shape [gradient_dim, mu, mu]
    // to get the gradient of the charge difference, we take the diagonal of the last 2 dimensions
    let mut grad_dq: Array2<f64> = Array2::zeros([f, n_orbs]);
    for nc in 0..f {
        let slice = w_plus_ps.slice(s![nc, .., ..]);
        let diag: ArrayView1<f64> = slice.diag();
        // fill the gradient array
        grad_dq.slice_mut(s![nc, ..]).assign(&diag);
    }

    grad_dq
}

/// Compute gradient of charge difference on-the-fly without storing 3D overlap derivative array.
///
/// Computes: grad_dq[a, mu] = (dS[a] @ P)[mu, mu] - 0.5 * (P @ dS[a] @ P @ S)[mu, mu]
///
/// This avoids O(n³) memory for grad_s by computing overlap derivatives on-the-fly.
pub fn get_grad_dq_xtb_onthefly(
    s: ArrayView2<f64>,
    p: ArrayView2<f64>,
    basis: &Basis,
    n_atoms: usize,
    n_orbs: usize,
) -> Array2<f64> {
    let n_grad = 3 * n_atoms;
    let mut grad_dq = Array2::zeros([n_grad, n_orbs]);

    // Precompute PS = P @ S (used in term 2)
    let ps: Array2<f64> = p.dot(&s);

    for (shell_i_idx, shell_i) in basis.shells.iter().enumerate() {
        let at_i = shell_i.atom_index;

        for (shell_j_idx, shell_j) in basis.shells.iter().enumerate() {
            let at_j = shell_j.atom_index;

            // No overlap derivative for same-atom pairs
            if at_i == at_j {
                continue;
            }

            // Only process unique pairs (shell_i_idx < shell_j_idx)
            // We'll add contributions for both (rho, sigma) and (sigma, rho)
            if shell_i_idx > shell_j_idx {
                continue;
            }

            let n_i = shell_i.sph_end - shell_i.sph_start;
            let n_j = shell_j.sph_end - shell_j.sph_start;

            // Check if d-orbitals are involved
            let either_has_d = shell_i.angular_momentum >= 2 || shell_j.angular_momentum >= 2;

            if !either_has_d {
                // s/p orbital case: use obara_saika_derivatives_all
                for rho_local in 0..n_i {
                    let rho = shell_i.sph_start + rho_local;
                    let orbital_rho = &basis.basis_functions[shell_i.start + rho_local];

                    for sigma_local in 0..n_j {
                        let sigma = shell_j.sph_start + sigma_local;
                        let orbital_sigma = &basis.basis_functions[shell_j.start + sigma_local];

                        // Compute overlap derivatives for all 3 directions at once
                        let ds_all = obara_saika_derivatives_all(orbital_rho, orbital_sigma);
                        let norm_prod = orbital_rho.contracted_norm * orbital_sigma.contracted_norm;

                        for dir in 0..3 {
                            let ds_i = ds_all[dir] * norm_prod;
                            let a_i = 3 * at_i + dir;
                            let a_j = 3 * at_j + dir;

                            // Term 1: diag(dS @ P)
                            // From (rho, sigma): grad_dq[a, rho] += ds * P[sigma, rho]
                            grad_dq[[a_i, rho]] += ds_i * p[[sigma, rho]];
                            grad_dq[[a_j, rho]] -= ds_i * p[[sigma, rho]];

                            // From (sigma, rho) by symmetry of dS
                            grad_dq[[a_i, sigma]] += ds_i * p[[rho, sigma]];
                            grad_dq[[a_j, sigma]] -= ds_i * p[[rho, sigma]];

                            // Term 2: -0.5 * diag(P @ dS @ PS)
                            let factor = -0.5 * ds_i;

                            // From (rho, sigma)
                            Zip::from(grad_dq.row_mut(a_i))
                                .and(p.column(rho))
                                .and(ps.row(sigma))
                                .for_each(|g, &p_val, &ps_val| {
                                    *g += factor * p_val * ps_val;
                                });
                            Zip::from(grad_dq.row_mut(a_j))
                                .and(p.column(rho))
                                .and(ps.row(sigma))
                                .for_each(|g, &p_val, &ps_val| {
                                    *g -= factor * p_val * ps_val;
                                });

                            // From (sigma, rho) by symmetry
                            Zip::from(grad_dq.row_mut(a_i))
                                .and(p.column(sigma))
                                .and(ps.row(rho))
                                .for_each(|g, &p_val, &ps_val| {
                                    *g += factor * p_val * ps_val;
                                });
                            Zip::from(grad_dq.row_mut(a_j))
                                .and(p.column(sigma))
                                .and(ps.row(rho))
                                .for_each(|g, &p_val, &ps_val| {
                                    *g -= factor * p_val * ps_val;
                                });
                        }
                    }
                }
            } else {
                // d-orbital case: use calc_overlap_derivative_d_shells
                // No factor of 2 here because we handle symmetric contributions explicitly
                let ds_d = calc_overlap_derivative_d_shells(basis, shell_i, shell_j);

                for rho_local in 0..n_i {
                    let rho = shell_i.sph_start + rho_local;

                    for sigma_local in 0..n_j {
                        let sigma = shell_j.sph_start + sigma_local;

                        for dir in 0..3 {
                            // ds_d[0-2] is d/dR_{at_i}, ds_d[3-5] is d/dR_{at_j}
                            let ds_i = ds_d[[dir, rho_local, sigma_local]];
                            let ds_j = ds_d[[3 + dir, rho_local, sigma_local]];
                            let a_i = 3 * at_i + dir;
                            let a_j = 3 * at_j + dir;

                            // Term 1: from (rho, sigma)
                            grad_dq[[a_i, rho]] += ds_i * p[[sigma, rho]];
                            grad_dq[[a_j, rho]] += ds_j * p[[sigma, rho]];
                            // Term 1: from (sigma, rho) by symmetry
                            grad_dq[[a_i, sigma]] += ds_i * p[[rho, sigma]];
                            grad_dq[[a_j, sigma]] += ds_j * p[[rho, sigma]];

                            // Term 2: from (rho, sigma)
                            let factor_i = -0.5 * ds_i;
                            let factor_j = -0.5 * ds_j;

                            Zip::from(grad_dq.row_mut(a_i))
                                .and(p.column(rho))
                                .and(ps.row(sigma))
                                .for_each(|g, &p_val, &ps_val| *g += factor_i * p_val * ps_val);
                            Zip::from(grad_dq.row_mut(a_j))
                                .and(p.column(rho))
                                .and(ps.row(sigma))
                                .for_each(|g, &p_val, &ps_val| *g += factor_j * p_val * ps_val);

                            // Term 2: from (sigma, rho) by symmetry
                            Zip::from(grad_dq.row_mut(a_i))
                                .and(p.column(sigma))
                                .and(ps.row(rho))
                                .for_each(|g, &p_val, &ps_val| *g += factor_i * p_val * ps_val);
                            Zip::from(grad_dq.row_mut(a_j))
                                .and(p.column(sigma))
                                .and(ps.row(rho))
                                .for_each(|g, &p_val, &ps_val| *g += factor_j * p_val * ps_val);
                        }
                    }
                }
            }
        }
    }

    grad_dq
}

pub fn grad_repulsive_energy_xtb(atoms: &[XtbAtom], n_atoms: usize) -> Array1<f64> {
    let mut grad: Array1<f64> = Array1::zeros([3 * n_atoms]);

    // two loops over the atoms
    for (i, atomi) in atoms.iter().enumerate() {
        // get the z_eff and alpha values
        let z_eff_i: f64 = REP_Z_EFF_PARAMS[atomi.kind.number_usize() - 1];
        let alpha_i: f64 = REP_ALPHA_PARAMS[atomi.kind.number_usize() - 1];
        let mut grad_i: Array1<f64> = Array1::zeros([3]);

        for (j, atomj) in atoms.iter().enumerate() {
            let z_eff_j: f64 = REP_Z_EFF_PARAMS[atomj.kind.number_usize() - 1];
            let alpha_j: f64 = REP_ALPHA_PARAMS[atomj.kind.number_usize() - 1];

            if i != j {
                let mut r: Vector3<f64> = atomi - atomj;
                let diff_vec = atomi - atomj;
                let distance: f64 =
                    (diff_vec.x.powi(2) + diff_vec.y.powi(2) + diff_vec.z.powi(2)).sqrt();
                r /= distance;

                let exponential: f64 = (-(alpha_i * alpha_j).sqrt() * distance.powf(1.5)).exp();
                let part1: f64 = exponential * z_eff_i * z_eff_j / distance.powi(2);
                let part2: f64 = 3.0 * (alpha_i * alpha_j).sqrt() * z_eff_i * z_eff_j * exponential
                    / (2.0 * distance.sqrt());

                let deriv_val: f64 = -part1 - part2;
                r *= deriv_val;

                let v = Array1::from_iter(r.iter());
                grad_i = &grad_i + &v;
            }
        }
        grad.slice_mut(s![i * 3..i * 3 + 3]).assign(&grad_i);
    }

    grad
}
