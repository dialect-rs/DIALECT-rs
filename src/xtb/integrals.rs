use crate::constants::OVERLAP_THRESHOLD;
use crate::xtb::initialization::basis::{Basis, BasisShell, ContractedBasisfunction};
use ndarray::prelude::*;
use std::f64::consts::PI;

/// Calculation of the overlap matrix in cartesian basis
pub fn calc_overlap_matrix_obs(basis: &Basis) -> Array2<f64> {
    let mut s: Array2<f64> =
        Array2::zeros((basis.basis_functions.len(), basis.basis_functions.len()));
    for (i, orbital1) in basis.basis_functions.iter().enumerate() {
        for (j, orbital2) in basis.basis_functions.iter().enumerate() {
            if i <= j {
                let distance: f64 = ((orbital1.center.0 - orbital2.center.0).powi(2)
                    + (orbital1.center.1 - orbital2.center.1).powi(2)
                    + (orbital1.center.2 - orbital2.center.2).powi(2))
                .sqrt();

                if distance < OVERLAP_THRESHOLD {
                    s[[i, j]] = overlap_obs_new(orbital1, orbital2)
                        * orbital1.contracted_norm
                        * orbital2.contracted_norm;
                }
            } else {
                s[[i, j]] = s[[j, i]]
            }
        }
    }
    s
}

pub fn calc_overlap_matrix_obs_new(basis: &Basis) -> Array2<f64> {
    let mut s: Array2<f64> = Array2::zeros((basis.nbas, basis.nbas));

    // iterate over shells
    for (idx_1, shell_i) in basis.shells.iter().enumerate() {
        for (idx_2, shell_j) in basis.shells.iter().enumerate() {
            // check if angular momentum is below 2
            if shell_i.angular_momentum < 2 && shell_j.angular_momentum < 2 {
                // iterate over ao shell indices
                for (idx_i, i) in (shell_i.start..shell_i.end).enumerate() {
                    // get the contracted basis function
                    let orbital1 = &basis.basis_functions[i];
                    for (idx_j, j) in (shell_j.start..shell_j.end).enumerate() {
                        // get the contracted basis function
                        let orbital2 = &basis.basis_functions[j];

                        if i <= j {
                            let distance: f64 = ((orbital1.center.0 - orbital2.center.0).powi(2)
                                + (orbital1.center.1 - orbital2.center.1).powi(2)
                                + (orbital1.center.2 - orbital2.center.2).powi(2))
                            .sqrt();

                            if distance < OVERLAP_THRESHOLD {
                                s[[shell_i.sph_start + idx_i, shell_j.sph_start + idx_j]] =
                                    overlap_obs_new(orbital1, orbital2)
                                        * orbital1.contracted_norm
                                        * orbital2.contracted_norm;
                            }
                        } else {
                            s[[shell_i.sph_start + idx_i, shell_j.sph_start + idx_j]] =
                                s[[shell_j.sph_start + idx_j, shell_i.sph_start + idx_i]];
                        }
                    }
                }
            } else {
                if idx_1 <= idx_2 {
                    // calc d overlap shells
                    let tmp_arr: Array2<f64> = calc_overlap_d_shells(basis, shell_i, shell_j);
                    s.slice_mut(s![
                        shell_i.sph_start..shell_i.sph_end,
                        shell_j.sph_start..shell_j.sph_end
                    ])
                    .assign(&tmp_arr);
                } else {
                    // slice the matrix
                    let s_temp: Array2<f64> = s
                        .slice(s![
                            shell_j.sph_start..shell_j.sph_end,
                            shell_i.sph_start..shell_i.sph_end,
                        ])
                        .to_owned();
                    s.slice_mut(s![
                        shell_i.sph_start..shell_i.sph_end,
                        shell_j.sph_start..shell_j.sph_end
                    ])
                    .assign(&s_temp.t());
                }
            }
        }
    }
    s
}

fn calc_overlap_d_shells(basis: &Basis, shell_i: &BasisShell, shell_j: &BasisShell) -> Array2<f64> {
    // get the angular momenta of the shells
    let l_1: i8 = shell_i.angular_momentum as i8;
    let l_2: i8 = shell_j.angular_momentum as i8;

    // get the possible dimension of the array
    let dim: (usize, usize) = match (l_1 as usize, l_2 as usize) {
        (0, 2) => (1, 6),
        (2, 0) => (6, 1),
        (1, 2) => (3, 6),
        (2, 1) => (6, 3),
        (2, 2) => (6, 6),
        _ => (1, 1),
    };
    // create temporary array
    let mut array: Array2<f64> = Array2::zeros([dim.0, dim.1]);

    // iterate over ao shell indices
    for (idx_i, i) in (shell_i.start..shell_i.end).enumerate() {
        // get the contracted basis function
        let orbital1 = &basis.basis_functions[i];
        for (idx_j, j) in (shell_j.start..shell_j.end).enumerate() {
            // get the contracted basis function
            let orbital2 = &basis.basis_functions[j];

            let distance: f64 = ((orbital1.center.0 - orbital2.center.0).powi(2)
                + (orbital1.center.1 - orbital2.center.1).powi(2)
                + (orbital1.center.2 - orbital2.center.2).powi(2))
            .sqrt();

            if distance < OVERLAP_THRESHOLD {
                array[[idx_i, idx_j]] = overlap_obs_new(orbital1, orbital2)
                    * orbital1.contracted_norm
                    * orbital2.contracted_norm;
            }
        }
    }

    // combine the cartesian results to obtain the overlap in spherical basis
    let dim_1: usize = 2 * l_1 as usize + 1;
    let dim_2: usize = 2 * l_2 as usize + 1;
    let mut spherical_array: Array2<f64> = Array2::zeros([dim_1, dim_2]);

    // iterate over m
    for (idx_i, m_i) in (-l_1..l_1 + 1).enumerate() {
        // get the prefactor for the m_value
        let prefac_i: f64 = get_spherical_norm_factor(l_1, m_i);
        for (idx_j, m_j) in (-l_2..l_2 + 1).enumerate() {
            // get the prefactor for the m_value
            let prefac: f64 = get_spherical_norm_factor(l_2, m_j) * prefac_i;
            // fill array
            spherical_array[[idx_i, idx_j]] =
                prefac * get_spherical_transform_values(l_1, m_i, l_2, m_j, array.view());
        }
    }

    spherical_array
}

fn get_spherical_norm_factor(l: i8, m: i8) -> f64 {
    match (l, m) {
        // (2, -2) => 0.5 * 3.0_f64.sqrt(),
        // (2, -1) => 0.5 * 3.0_f64.sqrt(),
        (2, -2) => 0.5,
        (2, -1) => 0.5,
        (2, 0) => 1.0,
        // (2, 1) => 0.5 * 3.0_f64.sqrt(),
        (2, 1) => 0.5,
        (2, 2) => 0.5 * 3.0_f64.sqrt(),
        _ => 1.0,
    }
}

fn get_spherical_transform_indices(l: i8, m: i8) -> Vec<usize> {
    match (l, m) {
        // (1, -1) => vec![1],
        // (1, 0) => vec![2],
        // (1, 1) => vec![0],
        (1, -1) => vec![0],
        (1, 0) => vec![1],
        (1, 1) => vec![2],
        (2, -2) => vec![1],
        (2, -1) => vec![4],
        (2, 0) => vec![5, 0, 3],
        (2, 1) => vec![2],
        (2, 2) => vec![0, 3],
        _ => vec![0],
    }
}

fn get_spherical_transform_values(l1: i8, m1: i8, l2: i8, m2: i8, arr: ArrayView2<f64>) -> f64 {
    let indices_1: Vec<usize> = get_spherical_transform_indices(l1, m1);
    let indices_2: Vec<usize> = get_spherical_transform_indices(l2, m2);

    match (l1, m1, l2, m2) {
        // d-d integrals
        (2, -2, 2, -2) => 4.0 * arr[[indices_1[0], indices_2[0]]],
        (2, -2, 2, -1) => 4.0 * arr[[indices_1[0], indices_2[0]]],
        (2, -2, 2, 0) => {
            2.0 * (arr[[indices_1[0], indices_2[0]]]
                - 0.5 * arr[[indices_1[0], indices_2[1]]]
                - 0.5 * arr[[indices_1[0], indices_2[2]]])
        }
        (2, -2, 2, 1) => 4.0 * arr[[indices_1[0], indices_2[0]]],
        (2, -2, 2, 2) => {
            2.0 * (arr[[indices_1[0], indices_2[0]]] - arr[[indices_1[0], indices_2[1]]])
        }
        (2, -1, 2, -2) => 4.0 * arr[[indices_1[0], indices_2[0]]],
        (2, -1, 2, -1) => 4.0 * arr[[indices_1[0], indices_2[0]]],
        (2, -1, 2, 0) => {
            2.0 * (arr[[indices_1[0], indices_2[0]]]
                - 0.5 * arr[[indices_1[0], indices_2[1]]]
                - 0.5 * arr[[indices_1[0], indices_2[2]]])
        }
        (2, -1, 2, 1) => 4.0 * arr[[indices_1[0], indices_2[0]]],
        (2, -1, 2, 2) => {
            2.0 * (arr[[indices_1[0], indices_2[0]]] - arr[[indices_1[0], indices_2[1]]])
        }
        (2, 0, 2, -2) => {
            2.0 * (arr[[indices_1[0], indices_2[0]]]
                - 0.5 * arr[[indices_1[1], indices_2[0]]]
                - 0.5 * arr[[indices_1[2], indices_2[0]]])
        }
        (2, 0, 2, -1) => {
            2.0 * (arr[[indices_1[0], indices_2[0]]]
                - 0.5 * arr[[indices_1[1], indices_2[0]]]
                - 0.5 * arr[[indices_1[2], indices_2[0]]])
        }
        (2, 0, 2, 0) => {
            (arr[[indices_1[0], indices_2[0]]]
                - 0.5 * arr[[indices_1[1], indices_2[0]]]
                - 0.5 * arr[[indices_1[2], indices_2[0]]])
                - 0.5
                    * (arr[[indices_1[0], indices_2[1]]]
                        - 0.5 * arr[[indices_1[1], indices_2[1]]]
                        - 0.5 * arr[[indices_1[2], indices_2[1]]])
                - 0.5
                    * (arr[[indices_1[0], indices_2[2]]]
                        - 0.5 * arr[[indices_1[1], indices_2[2]]]
                        - 0.5 * arr[[indices_1[2], indices_2[2]]])
        }
        (2, 0, 2, 1) => {
            2.0 * (arr[[indices_1[0], indices_2[0]]]
                - 0.5 * arr[[indices_1[1], indices_2[0]]]
                - 0.5 * arr[[indices_1[2], indices_2[0]]])
        }
        (2, 0, 2, 2) => {
            (arr[[indices_1[0], indices_2[0]]]
                - 0.5 * arr[[indices_1[1], indices_2[0]]]
                - 0.5 * arr[[indices_1[2], indices_2[0]]])
                - (arr[[indices_1[0], indices_2[1]]]
                    - 0.5 * arr[[indices_1[1], indices_2[1]]]
                    - 0.5 * arr[[indices_1[2], indices_2[1]]])
        }
        (2, 1, 2, -2) => 4.0 * arr[[indices_1[0], indices_2[0]]],
        (2, 1, 2, -1) => 4.0 * arr[[indices_1[0], indices_2[0]]],
        (2, 1, 2, 0) => {
            2.0 * (arr[[indices_1[0], indices_2[0]]]
                - 0.5 * arr[[indices_1[0], indices_2[1]]]
                - 0.5 * arr[[indices_1[0], indices_2[2]]])
        }
        (2, 1, 2, 1) => 4.0 * arr[[indices_1[0], indices_2[0]]],
        (2, 1, 2, 2) => {
            2.0 * (arr[[indices_1[0], indices_2[0]]] - arr[[indices_1[0], indices_2[1]]])
        }
        (2, 2, 2, -2) => {
            2.0 * (arr[[indices_1[0], indices_2[0]]] - arr[[indices_1[1], indices_2[0]]])
        }
        (2, 2, 2, -1) => {
            2.0 * (arr[[indices_1[0], indices_2[0]]] - arr[[indices_1[1], indices_2[0]]])
        }
        (2, 2, 2, 0) => 0.0,
        (2, 2, 2, 1) => {
            2.0 * (arr[[indices_1[0], indices_2[0]]] - arr[[indices_1[1], indices_2[0]]])
        }
        (2, 2, 2, 2) => {
            (arr[[indices_1[0], indices_2[0]]] - arr[[indices_1[1], indices_2[0]]])
                - (arr[[indices_1[0], indices_2[1]]] - arr[[indices_1[1], indices_2[1]]])
        }
        // s and p orbitals
        (_, _, 2, -2) => 2.0 * arr[[indices_1[0], indices_2[0]]],
        (_, _, 2, -1) => 2.0 * arr[[indices_1[0], indices_2[0]]],
        (_, _, 2, 0) => {
            arr[[indices_1[0], indices_2[0]]]
                - 0.5 * arr[[indices_1[0], indices_2[1]]]
                - 0.5 * arr[[indices_1[0], indices_2[2]]]
        }
        (_, _, 2, 1) => 2.0 * arr[[indices_1[0], indices_2[0]]],
        (_, _, 2, 2) => arr[[indices_1[0], indices_2[0]]] - arr[[indices_1[0], indices_2[1]]],
        (2, -2, _, _) => 2.0 * arr[[indices_1[0], indices_2[0]]],
        (2, -1, _, _) => 2.0 * arr[[indices_1[0], indices_2[0]]],
        (2, 0, _, _) => {
            arr[[indices_1[0], indices_2[0]]]
                - 0.5 * arr[[indices_1[1], indices_2[0]]]
                - 0.5 * arr[[indices_1[2], indices_2[0]]]
        }
        (2, 1, _, _) => 2.0 * arr[[indices_1[0], indices_2[0]]],
        (2, 2, _, _) => arr[[indices_1[0], indices_2[0]]] - arr[[indices_1[1], indices_2[0]]],
        _ => 0.0,
    }
}

/// Calculation of the overlap between two contracted cartesian basis functions.
fn overlap_obs_new(ao1: &ContractedBasisfunction, ao2: &ContractedBasisfunction) -> f64 {
    let mut num_overlap: f64 = 0.0;
    for basis1 in ao1.primitive_functions.iter() {
        for basis2 in ao2.primitive_functions.iter() {
            // define p and u
            let p: f64 = basis1.exponent + basis2.exponent;
            let u: f64 = (basis1.exponent * basis2.exponent) / p;
            let ab_x: f64 = ao1.center.0 - ao2.center.0;
            let ab_y: f64 = ao1.center.1 - ao2.center.1;
            let ab_z: f64 = ao1.center.2 - ao2.center.2;

            let a: f64 = obara_saika_helper(
                basis1.angular_momenta.0,
                basis2.angular_momenta.0,
                basis1.exponent,
                basis2.exponent,
                ao1.center.0,
                ao2.center.0,
                p,
            );
            let b: f64 = obara_saika_helper(
                basis1.angular_momenta.1,
                basis2.angular_momenta.1,
                basis1.exponent,
                basis2.exponent,
                ao1.center.1,
                ao2.center.1,
                p,
            );
            let c: f64 = obara_saika_helper(
                basis1.angular_momenta.2,
                basis2.angular_momenta.2,
                basis1.exponent,
                basis2.exponent,
                ao1.center.2,
                ao2.center.2,
                p,
            );
            num_overlap = num_overlap
                + basis1.coeff
                    * basis2.coeff
                    * a
                    * b
                    * c
                    * (PI / p).powf(1.5)
                    * (-u * (ab_x.powi(2) + ab_y.powi(2) + ab_z.powi(2))).exp();
        }
    }
    num_overlap
}

/// Calculation of the overlap matrix in cartesian basis
pub fn calc_overlap_matrix_obs_derivs(basis: &Basis, n_atoms: usize) -> Array3<f64> {
    let mut ds: Array3<f64> = Array3::zeros((
        3 * n_atoms,
        basis.basis_functions.len(),
        basis.basis_functions.len(),
    ));
    for (i, orbital1) in basis.basis_functions.iter().enumerate() {
        for (j, orbital2) in basis.basis_functions.iter().enumerate() {
            let at_idx: usize = orbital1.atom_index;
            let at_idx2: usize = orbital2.atom_index;

            let distance: f64 = ((orbital1.center.0 - orbital2.center.0).powi(2)
                + (orbital1.center.1 - orbital2.center.1).powi(2)
                + (orbital1.center.2 - orbital2.center.2).powi(2))
            .sqrt();

            if distance < OVERLAP_THRESHOLD && i < j && at_idx != at_idx2 {
                ds[[3 * at_idx, i, j]] = obara_saika_derivatives(orbital1, orbital2, 0)
                    * orbital1.contracted_norm
                    * orbital2.contracted_norm;
                ds[[3 * at_idx, j, i]] = 1.0 * ds[[3 * at_idx, i, j]];
                ds[[3 * at_idx2, i, j]] = -1.0 * ds[[3 * at_idx, i, j]];
                ds[[3 * at_idx2, j, i]] = -1.0 * ds[[3 * at_idx, i, j]];

                ds[[3 * at_idx + 1, i, j]] = obara_saika_derivatives(orbital1, orbital2, 1)
                    * orbital1.contracted_norm
                    * orbital2.contracted_norm;
                ds[[3 * at_idx + 1, j, i]] = 1.0 * ds[[3 * at_idx + 1, i, j]];
                ds[[3 * at_idx2 + 1, i, j]] = -1.0 * ds[[3 * at_idx + 1, i, j]];
                ds[[3 * at_idx2 + 1, j, i]] = -1.0 * ds[[3 * at_idx + 1, i, j]];

                ds[[3 * at_idx + 2, i, j]] = obara_saika_derivatives(orbital1, orbital2, 2)
                    * orbital1.contracted_norm
                    * orbital2.contracted_norm;
                ds[[3 * at_idx + 2, j, i]] = 1.0 * ds[[3 * at_idx + 2, i, j]];
                ds[[3 * at_idx2 + 2, i, j]] = -1.0 * ds[[3 * at_idx + 2, i, j]];
                ds[[3 * at_idx2 + 2, j, i]] = -1.0 * ds[[3 * at_idx + 2, i, j]];
            }
        }
    }
    ds
}

pub fn calc_overlap_matrix_obs_derivs_new(basis: &Basis, n_atoms: usize) -> Array3<f64> {
    let mut ds: Array3<f64> = Array3::zeros((3 * n_atoms, basis.nbas, basis.nbas));

    // iterate over shells
    for (idx_1, shell_i) in basis.shells.iter().enumerate() {
        for (idx_2, shell_j) in basis.shells.iter().enumerate() {
            let at_idx: usize = shell_i.atom_index;
            let at_idx2: usize = shell_j.atom_index;
            // check if angular momentum is below 2
            if shell_i.angular_momentum < 2 && shell_j.angular_momentum < 2 {
                // iterate over ao shell indices
                for (idx_i, i) in (shell_i.start..shell_i.end).enumerate() {
                    // get the contracted basis function
                    let orbital1 = &basis.basis_functions[i];
                    for (idx_j, j) in (shell_j.start..shell_j.end).enumerate() {
                        // get the contracted basis function
                        let orbital2 = &basis.basis_functions[j];

                        let distance: f64 = ((orbital1.center.0 - orbital2.center.0).powi(2)
                            + (orbital1.center.1 - orbital2.center.1).powi(2)
                            + (orbital1.center.2 - orbital2.center.2).powi(2))
                        .sqrt();
                        let index_i: usize = shell_i.sph_start + idx_i;
                        let index_j: usize = shell_j.sph_start + idx_j;

                        if distance < OVERLAP_THRESHOLD && i < j && at_idx != at_idx2 {
                            ds[[3 * at_idx, index_i, index_j]] =
                                obara_saika_derivatives(orbital1, orbital2, 0)
                                    * orbital1.contracted_norm
                                    * orbital2.contracted_norm;
                            ds[[3 * at_idx, index_j, index_i]] =
                                1.0 * ds[[3 * at_idx, index_i, index_j]];
                            ds[[3 * at_idx2, index_i, index_j]] =
                                -1.0 * ds[[3 * at_idx, index_i, index_j]];
                            ds[[3 * at_idx2, index_j, index_i]] =
                                -1.0 * ds[[3 * at_idx, index_i, index_j]];

                            ds[[3 * at_idx + 1, index_i, index_j]] =
                                obara_saika_derivatives(orbital1, orbital2, 1)
                                    * orbital1.contracted_norm
                                    * orbital2.contracted_norm;
                            ds[[3 * at_idx + 1, index_j, index_i]] =
                                1.0 * ds[[3 * at_idx + 1, index_i, index_j]];
                            ds[[3 * at_idx2 + 1, index_i, index_j]] =
                                -1.0 * ds[[3 * at_idx + 1, index_i, index_j]];
                            ds[[3 * at_idx2 + 1, index_j, index_i]] =
                                -1.0 * ds[[3 * at_idx + 1, index_i, index_j]];

                            ds[[3 * at_idx + 2, index_i, index_j]] =
                                obara_saika_derivatives(orbital1, orbital2, 2)
                                    * orbital1.contracted_norm
                                    * orbital2.contracted_norm;
                            ds[[3 * at_idx + 2, index_j, index_i]] =
                                1.0 * ds[[3 * at_idx + 2, index_i, index_j]];
                            ds[[3 * at_idx2 + 2, index_i, index_j]] =
                                -1.0 * ds[[3 * at_idx + 2, index_i, index_j]];
                            ds[[3 * at_idx2 + 2, index_j, index_i]] =
                                -1.0 * ds[[3 * at_idx + 2, index_i, index_j]];
                        }
                    }
                }
            } else {
                if idx_1 < idx_2 {
                    // derivatives involving d orbitals
                    let tmp_arr: Array3<f64> =
                        calc_overlap_derivative_d_shells(basis, shell_i, shell_j);
                    // slice the matrix
                    ds.slice_mut(s![
                        3 * at_idx,
                        shell_i.sph_start..shell_i.sph_end,
                        shell_j.sph_start..shell_j.sph_end
                    ])
                    .assign(&tmp_arr.slice(s![0, .., ..]));
                    ds.slice_mut(s![
                        3 * at_idx + 1,
                        shell_i.sph_start..shell_i.sph_end,
                        shell_j.sph_start..shell_j.sph_end
                    ])
                    .assign(&tmp_arr.slice(s![1, .., ..]));
                    ds.slice_mut(s![
                        3 * at_idx + 2,
                        shell_i.sph_start..shell_i.sph_end,
                        shell_j.sph_start..shell_j.sph_end
                    ])
                    .assign(&tmp_arr.slice(s![2, .., ..]));
                    ds.slice_mut(s![
                        3 * at_idx2,
                        shell_i.sph_start..shell_i.sph_end,
                        shell_j.sph_start..shell_j.sph_end
                    ])
                    .assign(&tmp_arr.slice(s![3, .., ..]));
                    ds.slice_mut(s![
                        3 * at_idx2 + 1,
                        shell_i.sph_start..shell_i.sph_end,
                        shell_j.sph_start..shell_j.sph_end
                    ])
                    .assign(&tmp_arr.slice(s![4, .., ..]));
                    ds.slice_mut(s![
                        3 * at_idx2 + 2,
                        shell_i.sph_start..shell_i.sph_end,
                        shell_j.sph_start..shell_j.sph_end
                    ])
                    .assign(&tmp_arr.slice(s![5, .., ..]));
                } else {
                    // slice the matrix
                    let tmp_arr: Array2<f64> = ds
                        .slice(s![
                            3 * at_idx,
                            shell_j.sph_start..shell_j.sph_end,
                            shell_i.sph_start..shell_i.sph_end
                        ])
                        .to_owned();
                    // insert the transpose
                    ds.slice_mut(s![
                        3 * at_idx,
                        shell_i.sph_start..shell_i.sph_end,
                        shell_j.sph_start..shell_j.sph_end
                    ])
                    .assign(&tmp_arr.t());
                    // slice the matrix
                    let tmp_arr: Array2<f64> = ds
                        .slice(s![
                            3 * at_idx + 1,
                            shell_j.sph_start..shell_j.sph_end,
                            shell_i.sph_start..shell_i.sph_end
                        ])
                        .to_owned();
                    // insert the transpose
                    ds.slice_mut(s![
                        3 * at_idx + 1,
                        shell_i.sph_start..shell_i.sph_end,
                        shell_j.sph_start..shell_j.sph_end
                    ])
                    .assign(&tmp_arr.t());
                    // slice the matrix
                    let tmp_arr: Array2<f64> = ds
                        .slice(s![
                            3 * at_idx + 2,
                            shell_j.sph_start..shell_j.sph_end,
                            shell_i.sph_start..shell_i.sph_end
                        ])
                        .to_owned();
                    // insert the transpose
                    ds.slice_mut(s![
                        3 * at_idx + 2,
                        shell_i.sph_start..shell_i.sph_end,
                        shell_j.sph_start..shell_j.sph_end
                    ])
                    .assign(&tmp_arr.t());
                    // do the same for atom2
                    // slice the matrix
                    let tmp_arr: Array2<f64> = ds
                        .slice(s![
                            3 * at_idx2,
                            shell_j.sph_start..shell_j.sph_end,
                            shell_i.sph_start..shell_i.sph_end
                        ])
                        .to_owned();
                    // insert the transpose
                    ds.slice_mut(s![
                        3 * at_idx2,
                        shell_i.sph_start..shell_i.sph_end,
                        shell_j.sph_start..shell_j.sph_end
                    ])
                    .assign(&tmp_arr.t());
                    // slice the matrix
                    let tmp_arr: Array2<f64> = ds
                        .slice(s![
                            3 * at_idx2 + 1,
                            shell_j.sph_start..shell_j.sph_end,
                            shell_i.sph_start..shell_i.sph_end
                        ])
                        .to_owned();
                    // insert the transpose
                    ds.slice_mut(s![
                        3 * at_idx2 + 1,
                        shell_i.sph_start..shell_i.sph_end,
                        shell_j.sph_start..shell_j.sph_end
                    ])
                    .assign(&tmp_arr.t());
                    // slice the matrix
                    let tmp_arr: Array2<f64> = ds
                        .slice(s![
                            3 * at_idx2 + 2,
                            shell_j.sph_start..shell_j.sph_end,
                            shell_i.sph_start..shell_i.sph_end
                        ])
                        .to_owned();
                    // insert the transpose
                    ds.slice_mut(s![
                        3 * at_idx2 + 2,
                        shell_i.sph_start..shell_i.sph_end,
                        shell_j.sph_start..shell_j.sph_end
                    ])
                    .assign(&tmp_arr.t());
                }
            }
        }
    }

    ds
}

fn calc_overlap_derivative_d_shells(
    basis: &Basis,
    shell_i: &BasisShell,
    shell_j: &BasisShell,
) -> Array3<f64> {
    // get the angular momenta of the shells
    let l_1: i8 = shell_i.angular_momentum as i8;
    let l_2: i8 = shell_j.angular_momentum as i8;
    // atoms
    let at_idx: usize = shell_i.atom_index;
    let at_idx2: usize = shell_j.atom_index;

    // get the possible dimension of the array
    let dim: (usize, usize) = match (l_1 as usize, l_2 as usize) {
        (0, 2) => (1, 6),
        (2, 0) => (6, 1),
        (1, 2) => (3, 6),
        (2, 1) => (6, 3),
        (2, 2) => (6, 6),
        _ => (1, 1),
    };
    // create temporary array
    let mut array: Array3<f64> = Array3::zeros([6, dim.0, dim.1]);

    // iterate over ao shell indices
    for (idx_i, i) in (shell_i.start..shell_i.end).enumerate() {
        // get the contracted basis function
        let orbital1 = &basis.basis_functions[i];
        for (idx_j, j) in (shell_j.start..shell_j.end).enumerate() {
            // get the contracted basis function
            let orbital2 = &basis.basis_functions[j];

            let distance: f64 = ((orbital1.center.0 - orbital2.center.0).powi(2)
                + (orbital1.center.1 - orbital2.center.1).powi(2)
                + (orbital1.center.2 - orbital2.center.2).powi(2))
            .sqrt();

            if distance < OVERLAP_THRESHOLD && at_idx != at_idx2 {
                // derivative in x direction
                array[[0, idx_i, idx_j]] = obara_saika_derivatives(orbital1, orbital2, 0)
                    * orbital1.contracted_norm
                    * orbital2.contracted_norm;
                array[[3, idx_i, idx_j]] = -1.0 * array[[0, idx_i, idx_j]];

                // derivative in y direction
                array[[1, idx_i, idx_j]] = obara_saika_derivatives(orbital1, orbital2, 1)
                    * orbital1.contracted_norm
                    * orbital2.contracted_norm;
                array[[4, idx_i, idx_j]] = -1.0 * array[[1, idx_i, idx_j]];

                // derivative in z direction
                array[[2, idx_i, idx_j]] = obara_saika_derivatives(orbital1, orbital2, 2)
                    * orbital1.contracted_norm
                    * orbital2.contracted_norm;
                array[[5, idx_i, idx_j]] = -1.0 * array[[2, idx_i, idx_j]];
            }
        }
    }
    // combine the cartesian results to obtain the overlap in spherical basis
    let dim_1: usize = 2 * l_1 as usize + 1;
    let dim_2: usize = 2 * l_2 as usize + 1;
    let mut spherical_array: Array3<f64> = Array3::zeros([6, dim_1, dim_2]);

    // iterate over m
    for (idx_i, m_i) in (-l_1..l_1 + 1).enumerate() {
        // get the prefactor for the m_value
        let prefac_i: f64 = get_spherical_norm_factor(l_1, m_i);
        for (idx_j, m_j) in (-l_2..l_2 + 1).enumerate() {
            // get the prefactor for the m_value
            let prefac: f64 = get_spherical_norm_factor(l_2, m_j) * prefac_i;
            // fill array
            spherical_array[[0, idx_i, idx_j]] = prefac
                * get_spherical_transform_values(l_1, m_i, l_2, m_j, array.slice(s![0, .., ..]));
            spherical_array[[1, idx_i, idx_j]] = prefac
                * get_spherical_transform_values(l_1, m_i, l_2, m_j, array.slice(s![1, .., ..]));
            spherical_array[[2, idx_i, idx_j]] = prefac
                * get_spherical_transform_values(l_1, m_i, l_2, m_j, array.slice(s![2, .., ..]));
            spherical_array[[3, idx_i, idx_j]] = prefac
                * get_spherical_transform_values(l_1, m_i, l_2, m_j, array.slice(s![3, .., ..]));
            spherical_array[[4, idx_i, idx_j]] = prefac
                * get_spherical_transform_values(l_1, m_i, l_2, m_j, array.slice(s![4, .., ..]));
            spherical_array[[5, idx_i, idx_j]] = prefac
                * get_spherical_transform_values(l_1, m_i, l_2, m_j, array.slice(s![5, .., ..]));
        }
    }

    spherical_array
}

pub fn obara_saika_helper(i: i8, j: i8, a: f64, b: f64, ax: f64, bx: f64, p: f64) -> f64 {
    let px: f64 = (a * ax + b * bx) / p;
    let dist_pa: f64 = px - ax;
    let dist_pb: f64 = px - bx;

    let val: f64 = if i < 3 && j < 3 {
        obs_cases(i, j, p, dist_pa, dist_pb)
    } else if i == 3 || j == 3 {
        obs_cases_l3(i, j, p, dist_pa, dist_pb)
    } else {
        obara_saika_recursion_new(i, j, p, dist_pa, dist_pb)
    };

    val
}

fn obara_saika_derivatives(
    ao1: &ContractedBasisfunction,
    ao2: &ContractedBasisfunction,
    direction: usize,
) -> f64 {
    let mut num_overlap: f64 = 0.0;
    for basis1 in ao1.primitive_functions.iter() {
        for basis2 in ao2.primitive_functions.iter() {
            // define p and u
            let p: f64 = basis1.exponent + basis2.exponent;
            let u: f64 = (basis1.exponent * basis2.exponent) / p;
            let ab_x: f64 = ao1.center.0 - ao2.center.0;
            let ab_y: f64 = ao1.center.1 - ao2.center.1;
            let ab_z: f64 = ao1.center.2 - ao2.center.2;

            if direction == 0 {
                let a: f64 = 2.0
                    * basis1.exponent
                    * obara_saika_helper(
                        basis1.angular_momenta.0 + 1,
                        basis2.angular_momenta.0,
                        basis1.exponent,
                        basis2.exponent,
                        ao1.center.0,
                        ao2.center.0,
                        p,
                    )
                    - (basis1.angular_momenta.0 as f64)
                        * obara_saika_helper(
                            basis1.angular_momenta.0 - 1,
                            basis2.angular_momenta.0,
                            basis1.exponent,
                            basis2.exponent,
                            ao1.center.0,
                            ao2.center.0,
                            p,
                        );
                let b: f64 = obara_saika_helper(
                    basis1.angular_momenta.1,
                    basis2.angular_momenta.1,
                    basis1.exponent,
                    basis2.exponent,
                    ao1.center.1,
                    ao2.center.1,
                    p,
                );
                let c: f64 = obara_saika_helper(
                    basis1.angular_momenta.2,
                    basis2.angular_momenta.2,
                    basis1.exponent,
                    basis2.exponent,
                    ao1.center.2,
                    ao2.center.2,
                    p,
                );

                num_overlap += basis1.coeff
                    * basis2.coeff
                    * a
                    * b
                    * c
                    * (PI / p).powf(1.5)
                    * (-u * (ab_x.powi(2) + ab_y.powi(2) + ab_z.powi(2))).exp();
            } else if direction == 1 {
                let a: f64 = obara_saika_helper(
                    basis1.angular_momenta.0,
                    basis2.angular_momenta.0,
                    basis1.exponent,
                    basis2.exponent,
                    ao1.center.0,
                    ao2.center.0,
                    p,
                );
                let b: f64 = 2.0
                    * basis1.exponent
                    * obara_saika_helper(
                        basis1.angular_momenta.1 + 1,
                        basis2.angular_momenta.1,
                        basis1.exponent,
                        basis2.exponent,
                        ao1.center.1,
                        ao2.center.1,
                        p,
                    )
                    - (basis1.angular_momenta.1 as f64)
                        * obara_saika_helper(
                            basis1.angular_momenta.1 - 1,
                            basis2.angular_momenta.1,
                            basis1.exponent,
                            basis2.exponent,
                            ao1.center.1,
                            ao2.center.1,
                            p,
                        );
                let c: f64 = obara_saika_helper(
                    basis1.angular_momenta.2,
                    basis2.angular_momenta.2,
                    basis1.exponent,
                    basis2.exponent,
                    ao1.center.2,
                    ao2.center.2,
                    p,
                );

                num_overlap += basis1.coeff
                    * basis2.coeff
                    * a
                    * b
                    * c
                    * (PI / p).powf(1.5)
                    * (-u * (ab_x.powi(2) + ab_y.powi(2) + ab_z.powi(2))).exp();
            } else {
                let a: f64 = obara_saika_helper(
                    basis1.angular_momenta.0,
                    basis2.angular_momenta.0,
                    basis1.exponent,
                    basis2.exponent,
                    ao1.center.0,
                    ao2.center.0,
                    p,
                );
                let b: f64 = obara_saika_helper(
                    basis1.angular_momenta.1,
                    basis2.angular_momenta.1,
                    basis1.exponent,
                    basis2.exponent,
                    ao1.center.1,
                    ao2.center.1,
                    p,
                );
                let c: f64 = 2.0
                    * basis1.exponent
                    * obara_saika_helper(
                        basis1.angular_momenta.2 + 1,
                        basis2.angular_momenta.2,
                        basis1.exponent,
                        basis2.exponent,
                        ao1.center.2,
                        ao2.center.2,
                        p,
                    )
                    - (basis1.angular_momenta.2 as f64)
                        * obara_saika_helper(
                            basis1.angular_momenta.2 - 1,
                            basis2.angular_momenta.2,
                            basis1.exponent,
                            basis2.exponent,
                            ao1.center.2,
                            ao2.center.2,
                            p,
                        );

                num_overlap += basis1.coeff
                    * basis2.coeff
                    * a
                    * b
                    * c
                    * (PI / p).powf(1.5)
                    * (-u * (ab_x.powi(2) + ab_y.powi(2) + ab_z.powi(2))).exp();
            }
        }
    }
    num_overlap
}

fn obara_saika_recursion_new(i: i8, j: i8, p: f64, dist_pa: f64, dist_pb: f64) -> f64 {
    if i == 0 && j == 0 {
        1.0 //(PI / p).sqrt() * (-mu * dist_ab.powi(2)).exp()
    } else if i < 0 || j < 0 {
        0.0
    } else if i > 0 && j >= 0 {
        dist_pa * obara_saika_recursion_new(i - 1, j, p, dist_pa, dist_pb)
            + 1. / (2. * p)
                * (((i - 1) as f64) * obara_saika_recursion_new(i - 2, j, p, dist_pa, dist_pb)
                    + ((j as f64) * obara_saika_recursion_new(i - 1, j - 1, p, dist_pa, dist_pb)))
    } else if i == 0 && j > 0 {
        dist_pb * obara_saika_recursion_new(i, j - 1, p, dist_pa, dist_pb)
            + 1. / (2. * p)
                * ((i as f64) * obara_saika_recursion_new(i - 1, j - 1, p, dist_pa, dist_pb)
                    + (((j - 1) as f64) * obara_saika_recursion_new(i, j - 2, p, dist_pa, dist_pb)))
    } else {
        panic!("obara_saika failed");
    }
}

#[inline(always)]
fn obs_cases(i: i8, j: i8, p: f64, dist_pa: f64, dist_pb: f64) -> f64 {
    match (i, j) {
        (0, 0) => return 1.0,
        (0, 1) => return dist_pb,
        (1, 0) => return dist_pa,
        (1, 1) => return dist_pa * dist_pb + 0.5 / p,
        (0, 2) => return dist_pb.powi(2) + 0.5 / p,
        (2, 0) => return dist_pa.powi(2) + 0.5 / p,
        (1, 2) => return dist_pa * (dist_pb.powi(2) + 0.5 / p) + dist_pb / p,
        (2, 1) => return dist_pa * (dist_pa * dist_pb + 0.5 / p) + 0.5 * (dist_pa + dist_pb) / p,
        (2, 2) => {
            return dist_pa * (dist_pa * (dist_pb.powi(2) + 0.5 / p) + dist_pb / p)
                + 0.5 * (2.0 * dist_pa * dist_pb + dist_pb.powi(2) + 1.5 / p) / p
        }
        _ => return 0.0,
    }
}

#[inline(always)]
fn obs_cases_l3(i: i8, j: i8, p: f64, dist_pa: f64, dist_pb: f64) -> f64 {
    match (i, j) {
        (0, 3) => return dist_pb * (dist_pb.powi(2) + 0.5 / p) + dist_pb / p,
        (3, 0) => return dist_pa * (dist_pa.powi(2) + 0.5 / p) + dist_pa / p,
        (1, 3) => {
            return dist_pa * (dist_pb * (dist_pb.powi(2) + 0.5 / p) + dist_pb / p)
                + 0.5 * (3.0 * dist_pb.powi(2) + 1.5 / p) / p
        }
        (3, 1) => {
            return dist_pa
                * (dist_pa * (dist_pa * dist_pb + 0.5 / p) + 0.5 * (dist_pa + dist_pb) / p)
                + 0.5 * (dist_pa.powi(2) + 2.0 * dist_pa * dist_pb + 1.5 / p) / p
        }
        (2, 3) => {
            return dist_pa
                * (dist_pa * (dist_pb * (dist_pb.powi(2) + 0.5 / p) + dist_pb / p)
                    + 0.5 * (3.0 * dist_pb.powi(2) + 1.5 / p) / p)
                + 0.5
                    * (3_f64 * dist_pa * (dist_pb.powi(2) + 0.5 / p)
                        + dist_pb * (dist_pb.powi(2) + 0.5 / p)
                        + 4.0 * dist_pb / p)
                    / p
        }
        (3, 2) => {
            return dist_pa
                * (dist_pa * (dist_pa * (dist_pb.powi(2) + 0.5 / p) + dist_pb / p)
                    + 0.5 * (2.0 * dist_pa * dist_pb + dist_pb.powi(2) + 1.5 / p) / p)
                + 0.5
                    * (2_f64 * dist_pa * (dist_pb.powi(2) + 0.5 / p)
                        + 2_f64 * dist_pa * (dist_pa * dist_pb + 0.5 / p)
                        + 2.0 * dist_pb / p
                        + (dist_pa + dist_pb) / p)
                    / p
        }
        (3, 3) => {
            return dist_pa
                * (dist_pa
                    * (dist_pa * (dist_pb * (dist_pb.powi(2) + 0.5 / p) + dist_pb / p)
                        + 0.5 * (3.0 * dist_pb.powi(2) + 1.5 / p) / p)
                    + 0.5
                        * (3_f64 * dist_pa * (dist_pb.powi(2) + 0.5 / p)
                            + dist_pb * (dist_pb.powi(2) + 0.5 / p)
                            + 4.0 * dist_pb / p)
                        / p)
                + 0.5
                    * (3_f64 * dist_pa * (dist_pa * (dist_pb.powi(2) + 0.5 / p) + dist_pb / p)
                        + 2_f64 * dist_pa * (dist_pb * (dist_pb.powi(2) + 0.5 / p) + dist_pb / p)
                        + (3.0 * dist_pb.powi(2) + 1.5 / p) / p
                        + (3_f64 / 2.0) * (2.0 * dist_pa * dist_pb + dist_pb.powi(2) + 1.5 / p)
                            / p)
                    / p
        }
        _ => return 0.0,
    }
}
