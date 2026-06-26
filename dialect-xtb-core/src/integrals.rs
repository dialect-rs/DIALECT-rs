use dialect_base::constants::OVERLAP_THRESHOLD;
use crate::initialization::atom::XtbAtom;
use crate::initialization::basis::{Basis, BasisShell, ContractedBasisfunction};
use ndarray::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::ops::AddAssign;

/// Precomputed constant: sqrt(PI)^3 = PI^1.5 ≈ 5.568327996831708
/// This avoids computing (PI/p)^1.5 with powf in the inner loop
const SQRTPI3: f64 = 5.568327996831707978;

/// Precomputed squared threshold to avoid sqrt in distance comparisons
const OVERLAP_THRESHOLD_SQ: f64 = OVERLAP_THRESHOLD * OVERLAP_THRESHOLD;

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

    // iterate over shells - only upper triangle (idx_1 <= idx_2)
    for (idx_1, shell_i) in basis.shells.iter().enumerate() {
        // Get shell center from first basis function (all have same center)
        let center_i = basis.basis_functions[shell_i.start].center;

        for (idx_2, shell_j) in basis.shells.iter().enumerate().skip(idx_1) {
            // Get shell center and compute distance once per shell pair
            let center_j = basis.basis_functions[shell_j.start].center;
            let dist_sq: f64 = (center_i.0 - center_j.0).powi(2)
                + (center_i.1 - center_j.1).powi(2)
                + (center_i.2 - center_j.2).powi(2);

            // Skip shell pair if beyond threshold (using squared comparison)
            if dist_sq >= OVERLAP_THRESHOLD_SQ {
                continue;
            }

            // check if angular momentum is below 2
            if shell_i.angular_momentum < 2 && shell_j.angular_momentum < 2 {
                // iterate over ao shell indices
                for (idx_i, i) in (shell_i.start..shell_i.end).enumerate() {
                    let orbital1 = &basis.basis_functions[i];
                    let norm1 = orbital1.contracted_norm;

                    // For diagonal shell blocks, only compute upper triangle
                    let j_start = if idx_1 == idx_2 { i } else { shell_j.start };

                    for (idx_j, j) in (j_start..shell_j.end).enumerate() {
                        let actual_idx_j = if idx_1 == idx_2 { idx_i + idx_j } else { idx_j };
                        let orbital2 = &basis.basis_functions[j];

                        let val =
                            overlap_obs_new(orbital1, orbital2) * norm1 * orbital2.contracted_norm;

                        let si = shell_i.sph_start + idx_i;
                        let sj = shell_j.sph_start + actual_idx_j;
                        s[[si, sj]] = val;
                        if si != sj {
                            s[[sj, si]] = val;
                        }
                    }
                }
            } else {
                // calc d overlap shells
                let tmp_arr: Array2<f64> = calc_overlap_d_shells(basis, shell_i, shell_j);
                s.slice_mut(s![
                    shell_i.sph_start..shell_i.sph_end,
                    shell_j.sph_start..shell_j.sph_end
                ])
                .assign(&tmp_arr);

                // Fill symmetric part for off-diagonal blocks
                if idx_1 != idx_2 {
                    s.slice_mut(s![
                        shell_j.sph_start..shell_j.sph_end,
                        shell_i.sph_start..shell_i.sph_end
                    ])
                    .assign(&tmp_arr.t());
                }
            }
        }
    }
    s
}

/// Parallel version of overlap matrix calculation using rayon
/// Computes shell pair overlaps in parallel with direct matrix writes
pub fn calc_overlap_matrix_parallel(basis: &Basis) -> Array2<f64> {
    let n_shells = basis.shells.len();
    let nbas = basis.nbas;

    // Precompute shell centers for faster access
    let shell_centers: Vec<(f64, f64, f64)> = basis
        .shells
        .iter()
        .map(|s| basis.basis_functions[s.start].center)
        .collect();

    // Generate all shell pair indices (upper triangle including diagonal)
    let shell_pairs: Vec<(usize, usize)> = (0..n_shells)
        .flat_map(|i| (i..n_shells).map(move |j| (i, j)))
        .collect();

    // Use a flat vector for the matrix to enable parallel writes
    let mut s_flat: Vec<f64> = vec![0.0; nbas * nbas];

    // Compute overlaps for each shell pair in parallel
    // Each shell pair writes to a unique set of matrix elements, so no data races
    shell_pairs.par_iter().for_each(|&(idx_1, idx_2)| {
        let shell_i = &basis.shells[idx_1];
        let shell_j = &basis.shells[idx_2];
        let center_i = shell_centers[idx_1];
        let center_j = shell_centers[idx_2];

        // Distance check
        let dist_sq: f64 = (center_i.0 - center_j.0).powi(2)
            + (center_i.1 - center_j.1).powi(2)
            + (center_i.2 - center_j.2).powi(2);

        if dist_sq >= OVERLAP_THRESHOLD_SQ {
            return;
        }

        let is_diagonal = idx_1 == idx_2;

        if shell_i.angular_momentum < 2 && shell_j.angular_momentum < 2 {
            // s/p shells - compute directly and write to flat array
            for (idx_i, i) in (shell_i.start..shell_i.end).enumerate() {
                let orbital1 = &basis.basis_functions[i];
                let norm1 = orbital1.contracted_norm;
                let si = shell_i.sph_start + idx_i;

                let j_start_offset = if is_diagonal { idx_i } else { 0 };
                for (rel_j, j) in (shell_j.start + j_start_offset..shell_j.end).enumerate() {
                    let idx_j = rel_j + j_start_offset;
                    let orbital2 = &basis.basis_functions[j];
                    let val =
                        overlap_obs_new(orbital1, orbital2) * norm1 * orbital2.contracted_norm;
                    let sj = shell_j.sph_start + idx_j;

                    // Write to flat array using unsafe (we know indices don't overlap between shell pairs)
                    unsafe {
                        let ptr = s_flat.as_ptr() as *mut f64;
                        *ptr.add(si * nbas + sj) = val;
                        if si != sj {
                            *ptr.add(sj * nbas + si) = val;
                        }
                    }
                }
            }
        } else {
            // d shells - compute block then write
            let block = calc_overlap_d_shells(basis, shell_i, shell_j);
            let dim_i = shell_i.sph_end - shell_i.sph_start;
            let dim_j = shell_j.sph_end - shell_j.sph_start;

            unsafe {
                let ptr = s_flat.as_ptr() as *mut f64;
                for bi in 0..dim_i {
                    let si = shell_i.sph_start + bi;
                    for bj in 0..dim_j {
                        let sj = shell_j.sph_start + bj;
                        let val = block[[bi, bj]];
                        *ptr.add(si * nbas + sj) = val;
                        if si != sj {
                            *ptr.add(sj * nbas + si) = val;
                        }
                    }
                }
            }
        }
    });

    // Convert flat vector to Array2
    Array2::from_shape_vec((nbas, nbas), s_flat).unwrap()
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
    // Note: distance check already done at shell level before calling this function
    for (idx_i, i) in (shell_i.start..shell_i.end).enumerate() {
        let orbital1 = &basis.basis_functions[i];
        let norm1 = orbital1.contracted_norm;
        for (idx_j, j) in (shell_j.start..shell_j.end).enumerate() {
            let orbital2 = &basis.basis_functions[j];
            array[[idx_i, idx_j]] =
                overlap_obs_new(orbital1, orbital2) * norm1 * orbital2.contracted_norm;
        }
    }

    // Transform the cartesian overlap block to the spherical basis via
    // S_sph = C_i * S_cart * C_j^T. The explicit matrix form is robust for
    // d-d blocks where a per-(m,m') hand expansion is error prone.
    let c_i: Array2<f64> = cart_to_spherical_matrix(l_1 as usize);
    let c_j: Array2<f64> = cart_to_spherical_matrix(l_2 as usize);
    c_i.dot(&array).dot(&c_j.t())
}

/// Cartesian-to-spherical transformation matrix C with shape
/// `(2l+1) x n_cart`, mapping the individually normalised cartesian gaussians
/// (ordered as produced by `permuts_2`, i.e. l=2: [xx, xy, xz, yy, yz, zz])
/// to the real spherical harmonics ordered by m = -l..=l. With each cartesian
/// component normalised to unit self-overlap, these coefficients reproduce
/// orthonormal spherical functions.
fn cart_to_spherical_matrix(l: usize) -> Array2<f64> {
    match l {
        0 => array![[1.0]],
        1 => Array2::eye(3),
        2 => {
            let s3: f64 = 3.0_f64.sqrt();
            // rows: m = -2,-1,0,1,2 ; cols: xx, xy, xz, yy, yz, zz
            array![
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],            // d_xy
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],            // d_yz
                [-0.5, 0.0, 0.0, -0.5, 0.0, 1.0],          // d_z2
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],            // d_xz
                [0.5 * s3, 0.0, 0.0, -0.5 * s3, 0.0, 0.0]  // d_x2-y2
            ]
        }
        _ => panic!("cart_to_spherical_matrix: unsupported l = {}", l),
    }
}

/// Calculation of the overlap between two contracted cartesian basis functions.
#[inline]
fn overlap_obs_new(ao1: &ContractedBasisfunction, ao2: &ContractedBasisfunction) -> f64 {
    let mut num_overlap: f64 = 0.0;

    // Precompute center differences (constant for all primitive pairs)
    let ab_x: f64 = ao1.center.0 - ao2.center.0;
    let ab_y: f64 = ao1.center.1 - ao2.center.1;
    let ab_z: f64 = ao1.center.2 - ao2.center.2;
    let ab2: f64 = ab_x * ab_x + ab_y * ab_y + ab_z * ab_z;

    // Cache centers to avoid repeated struct field access
    let c1 = ao1.center;
    let c2 = ao2.center;

    for basis1 in ao1.primitive_functions.iter() {
        let a_exp = basis1.exponent;
        let coeff1 = basis1.coeff;
        let l1 = basis1.angular_momenta;

        // Precompute a * center for product center calculation
        let a_cx = a_exp * c1.0;
        let a_cy = a_exp * c1.1;
        let a_cz = a_exp * c1.2;

        for basis2 in ao2.primitive_functions.iter() {
            let b_exp = basis2.exponent;

            // Precompute common terms
            let p: f64 = a_exp + b_exp;
            let inv_p: f64 = 1.0 / p;
            let inv_2p: f64 = 0.5 * inv_p;
            let u: f64 = a_exp * b_exp * inv_p;

            // Early screening based on exponential decay
            let est = u * ab2;
            if est > 40.0 {
                continue;
            }

            // Gaussian product prefactor: (PI/p)^1.5 * exp(-u*ab2)
            let sqrt_inv_p: f64 = inv_p.sqrt();
            let prefactor: f64 = SQRTPI3 * inv_p * sqrt_inv_p * (-est).exp();

            // Product center P = (a*A + b*B) / p
            let px: f64 = (a_cx + b_exp * c2.0) * inv_p;
            let py: f64 = (a_cy + b_exp * c2.1) * inv_p;
            let pz: f64 = (a_cz + b_exp * c2.2) * inv_p;

            // PA and PB distances
            let pa_x: f64 = px - c1.0;
            let pa_y: f64 = py - c1.1;
            let pa_z: f64 = pz - c1.2;
            let pb_x: f64 = px - c2.0;
            let pb_y: f64 = py - c2.1;
            let pb_z: f64 = pz - c2.2;

            let l2 = basis2.angular_momenta;
            let sx: f64 = obs_optimized(l1.0, l2.0, inv_2p, pa_x, pb_x);
            let sy: f64 = obs_optimized(l1.1, l2.1, inv_2p, pa_y, pb_y);
            let sz: f64 = obs_optimized(l1.2, l2.2, inv_2p, pa_z, pb_z);

            num_overlap += coeff1 * basis2.coeff * sx * sy * sz * prefactor;
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

pub fn calc_overlap_derivative_matrix_iterative_over_atoms(
    basis: &Basis,
    atoms: &[XtbAtom],
) -> Array3<f64> {
    let mut ds: Array3<f64> = Array3::zeros((3 * atoms.len(), basis.nbas, basis.nbas));

    for (idx, _atom) in atoms.iter().enumerate() {
        let ds_atom_specific = calc_overlap_matrix_obs_derivs_atom_specific(basis, idx);
        ds.slice_mut(s![3 * idx..3 * idx + 3, .., ..])
            .assign(&ds_atom_specific);
    }

    ds
}

/// Parallel version of overlap derivative calculation - computes derivatives for each atom in parallel
/// This version collects atom-specific derivatives and combines them, avoiding mutex contention
pub fn calc_overlap_matrix_obs_derivs_parallel(basis: &Basis, n_atoms: usize) -> Array3<f64> {
    // Compute derivatives for each atom in parallel
    let atom_derivs: Vec<(usize, Array3<f64>)> = (0..n_atoms)
        .into_par_iter()
        .map(|idx| {
            let ds_atom = calc_overlap_matrix_obs_derivs_atom_specific(basis, idx);
            (idx, ds_atom)
        })
        .collect();

    // Combine results into final array
    let mut ds: Array3<f64> = Array3::zeros((3 * n_atoms, basis.nbas, basis.nbas));
    for (idx, ds_atom) in atom_derivs {
        ds.slice_mut(s![3 * idx..3 * idx + 3, .., ..])
            .assign(&ds_atom);
    }

    ds
}

pub fn calc_overlap_matrix_obs_derivs_atom_specific(basis: &Basis, atom_idx: usize) -> Array3<f64> {
    let mut ds: Array3<f64> = Array3::zeros((3, basis.nbas, basis.nbas));

    // iterate over shells
    for (idx_1, shell_i) in basis.shells.iter().enumerate() {
        let at_idx: usize = shell_i.atom_index;
        if at_idx == atom_idx {
            for (idx_2, shell_j) in basis.shells.iter().enumerate() {
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

                            if distance < OVERLAP_THRESHOLD && at_idx != at_idx2 {
                                ds[[0, index_i, index_j]] =
                                    obara_saika_derivatives(orbital1, orbital2, 0)
                                        * orbital1.contracted_norm
                                        * orbital2.contracted_norm;
                                ds[[0, index_j, index_i]] = 1.0 * ds[[0, index_i, index_j]];

                                ds[[1, index_i, index_j]] =
                                    obara_saika_derivatives(orbital1, orbital2, 1)
                                        * orbital1.contracted_norm
                                        * orbital2.contracted_norm;
                                ds[[1, index_j, index_i]] = 1.0 * ds[[1, index_i, index_j]];

                                ds[[2, index_i, index_j]] =
                                    obara_saika_derivatives(orbital1, orbital2, 2)
                                        * orbital1.contracted_norm
                                        * orbital2.contracted_norm;
                                ds[[2, index_j, index_i]] = 1.0 * ds[[2, index_i, index_j]];
                            }
                        }
                    }
                } else {
                    // derivatives involving d orbitals
                    // IMPORTANT: Must use canonical ordering (smaller shell index first) to get
                    // correct spherical transformation. Then extract the appropriate derivative
                    // (indices 0-2 for first atom, 3-5 for second atom).
                    let at_idx2: usize = shell_j.atom_index;
                    let (shell_lo, shell_hi, query_is_first) = if idx_1 < idx_2 {
                        (shell_i, shell_j, true) // shell_i is first, it's the query atom
                    } else {
                        (shell_j, shell_i, false) // shell_j is first, query atom is second
                    };

                    // Only process unique pairs (idx_1 < idx_2) or when query atom is second
                    // This avoids double-processing while ensuring all pairs are covered
                    if idx_1 < idx_2 || !query_is_first {
                        let tmp_arr: Array3<f64> =
                            calc_overlap_derivative_d_shells(basis, shell_lo, shell_hi);

                        // Select derivative indices based on which atom we're querying
                        // shell_lo's atom gets indices 0-2, shell_hi's atom gets indices 3-5
                        let deriv_offset = if query_is_first { 0 } else { 3 };

                        // For idx_1 < idx_2: shell_i is first (lo), so matrix positions are (shell_i, shell_j)
                        // For idx_1 >= idx_2: shell_j is first (lo), so we need transposed positions
                        if query_is_first {
                            // Query atom's shell is shell_lo, store at (shell_lo, shell_hi) and transpose
                            ds.slice_mut(s![
                                0,
                                shell_lo.sph_start..shell_lo.sph_end,
                                shell_hi.sph_start..shell_hi.sph_end
                            ])
                            .assign(&tmp_arr.slice(s![
                                deriv_offset,
                                ..,
                                ..
                            ]));
                            ds.slice_mut(s![
                                0,
                                shell_hi.sph_start..shell_hi.sph_end,
                                shell_lo.sph_start..shell_lo.sph_end
                            ])
                            .assign(&tmp_arr.slice(s![deriv_offset, .., ..]).t());
                            ds.slice_mut(s![
                                1,
                                shell_lo.sph_start..shell_lo.sph_end,
                                shell_hi.sph_start..shell_hi.sph_end
                            ])
                            .assign(&tmp_arr.slice(s![
                                deriv_offset + 1,
                                ..,
                                ..
                            ]));
                            ds.slice_mut(s![
                                1,
                                shell_hi.sph_start..shell_hi.sph_end,
                                shell_lo.sph_start..shell_lo.sph_end
                            ])
                            .assign(&tmp_arr.slice(s![deriv_offset + 1, .., ..]).t());
                            ds.slice_mut(s![
                                2,
                                shell_lo.sph_start..shell_lo.sph_end,
                                shell_hi.sph_start..shell_hi.sph_end
                            ])
                            .assign(&tmp_arr.slice(s![
                                deriv_offset + 2,
                                ..,
                                ..
                            ]));
                            ds.slice_mut(s![
                                2,
                                shell_hi.sph_start..shell_hi.sph_end,
                                shell_lo.sph_start..shell_lo.sph_end
                            ])
                            .assign(&tmp_arr.slice(s![deriv_offset + 2, .., ..]).t());
                        } else {
                            // Query atom's shell is shell_hi, derivative is at indices 3-5
                            // Store at (shell_lo, shell_hi) and transpose to (shell_hi, shell_lo)
                            ds.slice_mut(s![
                                0,
                                shell_lo.sph_start..shell_lo.sph_end,
                                shell_hi.sph_start..shell_hi.sph_end
                            ])
                            .assign(&tmp_arr.slice(s![
                                deriv_offset,
                                ..,
                                ..
                            ]));
                            ds.slice_mut(s![
                                0,
                                shell_hi.sph_start..shell_hi.sph_end,
                                shell_lo.sph_start..shell_lo.sph_end
                            ])
                            .assign(&tmp_arr.slice(s![deriv_offset, .., ..]).t());
                            ds.slice_mut(s![
                                1,
                                shell_lo.sph_start..shell_lo.sph_end,
                                shell_hi.sph_start..shell_hi.sph_end
                            ])
                            .assign(&tmp_arr.slice(s![
                                deriv_offset + 1,
                                ..,
                                ..
                            ]));
                            ds.slice_mut(s![
                                1,
                                shell_hi.sph_start..shell_hi.sph_end,
                                shell_lo.sph_start..shell_lo.sph_end
                            ])
                            .assign(&tmp_arr.slice(s![deriv_offset + 1, .., ..]).t());
                            ds.slice_mut(s![
                                2,
                                shell_lo.sph_start..shell_lo.sph_end,
                                shell_hi.sph_start..shell_hi.sph_end
                            ])
                            .assign(&tmp_arr.slice(s![
                                deriv_offset + 2,
                                ..,
                                ..
                            ]));
                            ds.slice_mut(s![
                                2,
                                shell_hi.sph_start..shell_hi.sph_end,
                                shell_lo.sph_start..shell_lo.sph_end
                            ])
                            .assign(&tmp_arr.slice(s![deriv_offset + 2, .., ..]).t());
                        }
                    }
                }
            }
        }
    }

    ds
}

pub fn calc_overlap_derivative_d_shells(
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
    // Transform each cartesian derivative component to the spherical basis via
    // S_sph = C_i * S_cart * C_j^T (same convention as the overlap itself).
    let dim_1: usize = 2 * l_1 as usize + 1;
    let dim_2: usize = 2 * l_2 as usize + 1;
    let c_i: Array2<f64> = cart_to_spherical_matrix(l_1 as usize);
    let c_j: Array2<f64> = cart_to_spherical_matrix(l_2 as usize);
    let c_j_t: Array2<f64> = c_j.t().to_owned();
    let mut spherical_array: Array3<f64> = Array3::zeros([6, dim_1, dim_2]);
    for k in 0..6 {
        let cart_k: Array2<f64> = array.slice(s![k, .., ..]).to_owned();
        spherical_array
            .slice_mut(s![k, .., ..])
            .assign(&c_i.dot(&cart_k).dot(&c_j_t));
    }

    spherical_array
}

/// Optimized Obara-Saika computation using precomputed inv_2p
/// Direct formulas for l <= 3 to avoid recursion overhead
#[inline(always)]
fn obs_optimized(i: i8, j: i8, inv_2p: f64, dist_pa: f64, dist_pb: f64) -> f64 {
    match (i, j) {
        // Base cases (l=0)
        (0, 0) => 1.0,
        (0, 1) => dist_pb,
        (1, 0) => dist_pa,
        // l=1 cases
        (1, 1) => dist_pa * dist_pb + inv_2p,
        // l=2 cases (one side)
        (0, 2) => {
            let pb2 = dist_pb * dist_pb;
            pb2 + inv_2p
        }
        (2, 0) => {
            let pa2 = dist_pa * dist_pa;
            pa2 + inv_2p
        }
        // l=2 mixed cases
        (1, 2) => {
            let pb2 = dist_pb * dist_pb;
            dist_pa * (pb2 + inv_2p) + 2.0 * inv_2p * dist_pb
        }
        (2, 1) => {
            let pa2 = dist_pa * dist_pa;
            dist_pb * (pa2 + inv_2p) + 2.0 * inv_2p * dist_pa
        }
        // l=2,2 case
        // S(2,2) = PA*S(1,2) + h*(S(0,2) + 2*S(1,1))
        //        = PA²PB² + PA²h + PB²h + 4PAPBh + 3h²
        (2, 2) => {
            let pa2 = dist_pa * dist_pa;
            let pb2 = dist_pb * dist_pb;
            let inv_2p_2 = inv_2p * inv_2p;
            (pa2 + inv_2p) * (pb2 + inv_2p)
                + 4.0 * inv_2p * dist_pa * dist_pb
                + 2.0 * inv_2p_2
        }
        // l=3 cases (for gradients)
        (3, 0) => {
            let pa2 = dist_pa * dist_pa;
            dist_pa * (pa2 + 3.0 * inv_2p)
        }
        (0, 3) => {
            let pb2 = dist_pb * dist_pb;
            dist_pb * (pb2 + 3.0 * inv_2p)
        }
        (3, 1) => {
            let pa2 = dist_pa * dist_pa;
            let s30 = dist_pa * (pa2 + 3.0 * inv_2p);
            let s20 = pa2 + inv_2p;
            dist_pb * s30 + 3.0 * inv_2p * s20
        }
        (1, 3) => {
            let pb2 = dist_pb * dist_pb;
            let s03 = dist_pb * (pb2 + 3.0 * inv_2p);
            let s02 = pb2 + inv_2p;
            dist_pa * s03 + 3.0 * inv_2p * s02
        }
        // S(3,2) = PA*S(2,2) + h*(2*S(1,2) + 2*S(2,1))
        (3, 2) => {
            let pa2 = dist_pa * dist_pa;
            let pb2 = dist_pb * dist_pb;
            let inv_2p_2 = inv_2p * inv_2p;
            let s20 = pa2 + inv_2p;
            let s02 = pb2 + inv_2p;
            let s22 = s20 * s02 + 4.0 * inv_2p * dist_pa * dist_pb + 2.0 * inv_2p_2;
            let s12 = dist_pa * s02 + 2.0 * inv_2p * dist_pb;
            let s21 = dist_pb * s20 + 2.0 * inv_2p * dist_pa;
            dist_pa * s22 + 2.0 * inv_2p * (s12 + s21)
        }
        // S(2,3) = PB*S(2,2) + h*(2*S(2,1) + 2*S(1,2))
        (2, 3) => {
            let pa2 = dist_pa * dist_pa;
            let pb2 = dist_pb * dist_pb;
            let inv_2p_2 = inv_2p * inv_2p;
            let s20 = pa2 + inv_2p;
            let s02 = pb2 + inv_2p;
            let s22 = s20 * s02 + 4.0 * inv_2p * dist_pa * dist_pb + 2.0 * inv_2p_2;
            let s21 = dist_pb * s20 + 2.0 * inv_2p * dist_pa;
            let s12 = dist_pa * s02 + 2.0 * inv_2p * dist_pb;
            dist_pb * s22 + 2.0 * inv_2p * (s21 + s12)
        }
        // S(3,3) = PA*S(2,3) + h*(2*S(1,3) + 3*S(2,2))
        (3, 3) => {
            let pa2 = dist_pa * dist_pa;
            let pb2 = dist_pb * dist_pb;
            let inv_2p_2 = inv_2p * inv_2p;
            let s20 = pa2 + inv_2p;
            let s02 = pb2 + inv_2p;
            let s22 = s20 * s02 + 4.0 * inv_2p * dist_pa * dist_pb + 2.0 * inv_2p_2;
            let s21 = dist_pb * s20 + 2.0 * inv_2p * dist_pa;
            let s12 = dist_pa * s02 + 2.0 * inv_2p * dist_pb;
            let s23 = dist_pb * s22 + 2.0 * inv_2p * (s21 + s12);
            let s03 = dist_pb * (pb2 + 3.0 * inv_2p);
            let s13 = dist_pa * s03 + 3.0 * inv_2p * s02;
            dist_pa * s23 + inv_2p * (2.0 * s13 + 3.0 * s22)
        }
        // Negative indices return 0
        (-1, _) | (_, -1) => 0.0,
        // Fallback to recursion for higher angular momenta (not reached for the s/p/d valence basis)
        _ => {
            let p = 0.5 / inv_2p;
            obara_saika_recursion_new(i, j, p, dist_pa, dist_pb)
        }
    }
}

pub fn obara_saika_helper(i: i8, j: i8, a: f64, b: f64, ax: f64, bx: f64, p: f64) -> f64 {
    let px: f64 = (a * ax + b * bx) / p;
    let dist_pa: f64 = px - ax;
    let dist_pb: f64 = px - bx;
    let inv_2p: f64 = 0.5 / p;

    obs_optimized(i, j, inv_2p, dist_pa, dist_pb)
}

pub fn obara_saika_derivatives(
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

/// Compute all 3 overlap derivatives at once, sharing common computations
/// Returns [dS/dx, dS/dy, dS/dz]
pub fn obara_saika_derivatives_all(
    ao1: &ContractedBasisfunction,
    ao2: &ContractedBasisfunction,
) -> [f64; 3] {
    let mut result = [0.0f64; 3];

    for basis1 in ao1.primitive_functions.iter() {
        for basis2 in ao2.primitive_functions.iter() {
            // Shared computations
            let p: f64 = basis1.exponent + basis2.exponent;
            let u: f64 = (basis1.exponent * basis2.exponent) / p;
            let ab_x: f64 = ao1.center.0 - ao2.center.0;
            let ab_y: f64 = ao1.center.1 - ao2.center.1;
            let ab_z: f64 = ao1.center.2 - ao2.center.2;

            // Shared prefactor: coeff * coeff * (PI/p)^1.5 * exp(...)
            let prefactor = basis1.coeff
                * basis2.coeff
                * (PI / p).powf(1.5)
                * (-u * (ab_x * ab_x + ab_y * ab_y + ab_z * ab_z)).exp();

            let two_alpha = 2.0 * basis1.exponent;

            // Compute the "normal" (non-derivative) 1D overlaps - shared across directions
            let a_normal = obara_saika_helper(
                basis1.angular_momenta.0,
                basis2.angular_momenta.0,
                basis1.exponent,
                basis2.exponent,
                ao1.center.0,
                ao2.center.0,
                p,
            );
            let b_normal = obara_saika_helper(
                basis1.angular_momenta.1,
                basis2.angular_momenta.1,
                basis1.exponent,
                basis2.exponent,
                ao1.center.1,
                ao2.center.1,
                p,
            );
            let c_normal = obara_saika_helper(
                basis1.angular_momenta.2,
                basis2.angular_momenta.2,
                basis1.exponent,
                basis2.exponent,
                ao1.center.2,
                ao2.center.2,
                p,
            );

            // Direction 0 (x): derivative term for a, normal for b and c
            let a_deriv_x = two_alpha
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
            result[0] += prefactor * a_deriv_x * b_normal * c_normal;

            // Direction 1 (y): normal for a, derivative term for b, normal for c
            let b_deriv_y = two_alpha
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
            result[1] += prefactor * a_normal * b_deriv_y * c_normal;

            // Direction 2 (z): normal for a and b, derivative term for c
            let c_deriv_z = two_alpha
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
            result[2] += prefactor * a_normal * b_normal * c_deriv_z;
        }
    }
    result
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
