/*!

## Auxiliar functions to manipulate arrays

 */

use crate::utils::array_helper::argsort32;
use approx::relative_eq;
use log::info;
use ndarray::prelude::*;
use ndarray_linalg::generate::random;
use std::cmp::Ordering;
use std::time::Instant;
// use std::{println as info, println as warn};

/// Generate the initial subspace vectors. These correspond to the `dim` lowest
/// diagonal elements of the matrix that will be diagonalized.
pub fn initial_subspace(diag: ArrayView1<f64>, dim: usize) -> Array2<f64> {
    let order: Vec<usize> = argsort(diag.view());
    let mut mtx: Array2<f64> = Array2::zeros([diag.len(), dim]);
    for (idx, i) in order.into_iter().enumerate() {
        if idx < dim {
            mtx[[i, idx]] = 1.0;
        }
    }
    mtx
}

/// Generate a random highly diagonal symmetric matrix
pub fn generate_diagonal_dominant(dim: usize, sparsity: f64) -> Array2<f64> {
    let diag: Array1<f64> = 10.0 * random([dim]);
    let off_diag: Array2<f64> = random((dim, dim));
    let arr = &off_diag + &off_diag.t();
    let mut arr = &arr * sparsity;
    arr.diag_mut().assign(&diag);
    arr
}

/// Random symmetric matrix
pub fn generate_random_symmetric(dim: usize, magnitude: f64) -> Array2<f64> {
    let arr: Array2<f64> = random((dim, dim)) * magnitude;
    arr.dot(&arr.t())
}

pub fn sort_vector<T: PartialOrd>(vs: &mut Vec<T>, ascending: bool) {
    if ascending {
        vs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    } else {
        vs.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
    }
}

pub fn test_eigenpairs(
    ref_eigenpair: (Array1<f64>, Array2<f64>),
    eigenpair: (Array1<f64>, Array2<f64>),
    number: usize,
) {
    let (dav_eigenvalues, dav_eigenvectors) = eigenpair;
    let (ref_eigenvalues, ref_eigenvectors) = ref_eigenpair;
    for i in 0..number {
        // Test Eigenvalues
        assert!(relative_eq!(
            ref_eigenvalues[i],
            dav_eigenvalues[i],
            epsilon = 1e-6
        ));
        // Test Eigenvectors
        let x = ref_eigenvectors.column(i);
        let y = dav_eigenvectors.column(i);
        // The autovectors may different in their sign
        // They should be either parallel or antiparallel
        let dot = x.dot(&y).abs();
        assert!(relative_eq!(dot, 1.0, epsilon = 1e-6));
    }
}

fn argsort(v: ArrayView1<f64>) -> Vec<usize> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(Ordering::Equal));
    idx
}

pub fn print_davidson_init(max_iter: usize, nroots: usize, tolerance: f64) {
    info!("{:^80}", "");
    info!("{: ^80}", "Iterative Davidson Routine");
    info!("{:-^80}", "");
    info!(
        "{: <25} {:4.2e}",
        "Energy is converged when residual is below:", tolerance
    );
    info!("{: <25} {}", "Maximum number of iterations:", max_iter);
    if nroots == 1 {
        info!("{: >4} {: <25}", nroots, " Root will be computed.");
    } else {
        info!("{: >4} {: <25}", nroots, " Roots will be computed.");
    }
    info!("{:-^75} ", "");
    info!(
        "{: <5}{: >14}{: >14}{: >14}{: >14}{: >14}",
        "Iter.", "Roots conv.", "Roots left", "#subsp. Vec.", "Total dev.", "Max dev."
    );
    info!("{:-^75} ", "");
}

pub fn print_davidson_iteration(
    iter: usize,
    roots_cvd: usize,
    roots_lft: usize,
    nvec: usize,
    t_dev: f64,
    max_dev: f64,
) {
    info!(
        "{: >5}{:>14}{:>14}{:>14}{:>14.8}{:>14.8}",
        iter + 1,
        roots_cvd,
        roots_lft,
        nvec,
        t_dev,
        max_dev
    );
}

pub fn print_davidson_end(result_is_ok: bool, time: Instant) {
    info!("{:-^75} ", "");
    if result_is_ok {
        info!("Davidson routine converged")
    } else {
        info!("Davidson routine did not converge!")
    }
    info!(
        "{:>68} {:>8.2} s",
        "elapsed time:",
        time.elapsed().as_secs_f32()
    );
    info!("{:-^80}", "");
    info!("{:^80}", "");
}

pub fn print_davidson32_init(max_iter: usize, nroots: usize, tolerance: f32) {
    info!("{:^80}", "");
    info!("{: ^80}", "Iterative Davidson Routine");
    info!("{:-^80}", "");
    info!(
        "{: <25} {:4.2e}",
        "Energy is converged when residual is below:", tolerance
    );
    info!("{: <25} {}", "Maximum number of iterations:", max_iter);
    if nroots == 1 {
        info!("{: >4} {: <25}", nroots, " Root will be computed.");
    } else {
        info!("{: >4} {: <25}", nroots, " Roots will be computed.");
    }
    info!("{:-^75} ", "");
    info!(
        "{: <5}{: >14}{: >14}{: >14}{: >14}{: >14}",
        "Iter.", "Roots conv.", "Roots left", "#subsp. Vec.", "Total dev.", "Max dev."
    );
    info!("{:-^75} ", "");
}

#[cfg(test)]
mod test {
    use ndarray::prelude::*;

    #[test]
    fn test_random_symmetric() {
        let matrix = super::generate_random_symmetric(10, 2.5);
        test_symmetric(matrix);
    }
    #[test]
    fn test_diagonal_dominant() {
        let matrix = super::generate_diagonal_dominant(10, 0.005);
        test_symmetric(matrix);
    }

    fn test_symmetric(matrix: Array2<f64>) {
        let rs = &matrix - &matrix.t();
        assert!(rs.sum() < f64::EPSILON);
    }
}
