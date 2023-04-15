use crate::excited_states::casida::engine::CasidaEngine;
use crate::excited_states::solvers::utils;
use ndarray::prelude::*;
use ndarray_linalg::*;
use ndarray_stats::QuantileExt;
use std::error;
use std::fmt;
use std::time::Instant;

#[derive(Debug, PartialEq)]
pub struct CasidaError;

impl fmt::Display for CasidaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Casida solver did not converge!")
    }
}

impl error::Error for CasidaError {}

/// Structure with the configuration data
pub struct CasidaSolver {
    pub eigenvalues: Array1<f64>,
    pub eigenvectors: Array2<f64>,
    pub xpy: Array2<f64>,
    pub xmy: Array2<f64>,
}

impl CasidaSolver {
    /// Compute the lowest eigenvalues of a symmetric, diagonal dominant matrix.
    /// * `engine` an object that implements the `CasidaEngine` trait.
    /// * `guess` the initial guess for the eigenvectors.
    /// * `omega` - An array containing the MO energies.
    /// * `n_roots` the number of (lowest) eigenvalues/eigenvectors to compute.
    /// * `tolerance` numerical tolerance for convergence.
    /// * `max_iter` the maximal number of iterations.
    /// * `subspace_multiplier` Determines the maximum size of the number of expansion vectors
    pub fn new<D: CasidaEngine>(
        engine: &mut D,
        guess: Array2<f64>,
        omega: ArrayView1<f64>,
        n_roots: usize,
        tolerance: f64,
        max_iter: usize,
        subspace_multiplier: usize,
    ) -> Result<Self, CasidaError> {
        // Timer to measure the time within the Davidson routine.
        let timer: Instant = Instant::now();

        // The initial guess needs to be mutable.
        let mut guess: Array2<f64> = guess;

        // Dimension of the subspace.
        let dim_sub_origin: usize = guess.ncols();
        let mut dim_sub: usize = dim_sub_origin;

        // The maximal possible subspace, before it will be collapsed.
        let max_space: usize = subspace_multiplier * n_roots;

        // The initial information of the Davidson routine are printed.
        utils::print_davidson_init(max_iter, n_roots, tolerance);

        // Initialization of the result.
        let mut result = Err(CasidaError);

        // Outer loop block Davidson schema.
        for i in 0..max_iter {
            // 1. The initial subspace is formed by projecting into the new guess vectors.
            // Matrix-vector product of A with the trial vectors.
            let (ax, bx): (Array2<f64>, Array2<f64>) = engine.compute_products(guess.view());

            // build a+b and a-b
            let a_m_b: Array2<f64> = &ax - &bx;
            let a_p_b: Array2<f64> = &ax + &bx;

            // 1.1 Initialization of the subspace Hamiltonian.
            let apb_proj: Array2<f64> = guess.t().dot(&a_p_b);
            let amb_proj: Array2<f64> = guess.t().dot(&a_m_b);

            // matrix squareroot of a-b
            let sq_a_m_b: Array2<f64> = amb_proj.ssqrt(UPLO::Upper).unwrap();
            // build matrix
            let h: Array2<f64> = sq_a_m_b.dot(&apb_proj.dot(&sq_a_m_b));

            // 2. Solve the eigenvalue problem for the subspace Hamiltonian.
            // The eigenvalues (u) and eigenvectors (v) are already sorted in ascending order.
            let (mut u, mut v): (Array1<f64>, Array2<f64>) = h.eigh(UPLO::Upper).unwrap();

            // Only the first n_roots eigenvalues and eigenvectors are used.
            u = u.slice_move(s![0..n_roots]);
            v = v.slice_move(s![.., 0..n_roots]);

            // 3. Convergence checks are made.
            // 3.1 Compute the Ritz vectors.
            let ritz: Array2<f64> = guess.dot(&v);

            // take the square root of the eigenvalues
            let w: Array1<f64> = u.mapv(f64::sqrt);
            let wsq: Array1<f64> = w.mapv(f64::sqrt);

            // 3.2 approximate right R = (X+Y) and left L = (X-Y) eigenvectors
            // in the basis bs
            // (X+Y) = (A-B)^(1/2).T / sqrt(w)
            let rb: Array2<f64> = sq_a_m_b.dot(&v) / &wsq;
            // L = (X-Y) = 1/w * (A+B).(X+Y)
            let lb: Array2<f64> = apb_proj.dot(&rb) / &w;

            // 3.3 transform to the canonical basis Lb -> L, Rb -> R
            let l_vector: Array2<f64> = guess.dot(&lb);
            let r_vector: Array2<f64> = guess.dot(&rb);

            // 3.4 Calculate the residual vectors
            let wl: Array2<f64> = a_p_b.dot(&rb) - &l_vector * &w;
            let wr: Array2<f64> = a_m_b.dot(&lb) - &r_vector * &w;

            // 3.5 Calculate the norms of the residual vectors
            let mut norms: Array1<f64> = Array::zeros(n_roots);
            let mut norms_l: Array1<f64> = Array::zeros(n_roots);
            let mut norms_r: Array1<f64> = Array::zeros(n_roots);
            for i in 0..n_roots {
                norms_l[i] = wl.slice(s![.., i]).dot(&wl.slice(s![.., i]));
                norms_r[i] = wr.slice(s![.., i]).dot(&wr.slice(s![.., i]));
                norms[i] = norms_l[i] + norms_r[i];
            }
            // 4. Check for convergence
            let roots_cvd: usize = norms
                .iter()
                .fold(0, |n, &x| if x < tolerance { n + 1 } else { n });

            // number of not converged roots
            let roots_lft: usize = n_roots - roots_cvd;
            // sum of all errors
            let error: f64 = norms.sum();
            // the maximum value of the errors
            let max_error: f64 = *norms.max().unwrap();

            // If all eigenvalues are converged, the Davidson routine finished successfully.
            if roots_lft == 0 && i > 0 {
                result = Ok(Self::create_results(
                    u.view(),
                    ritz.view(),
                    n_roots,
                    r_vector.view(),
                    l_vector.view(),
                ));
                utils::print_davidson_iteration(
                    i,
                    roots_cvd,
                    n_roots - roots_cvd,
                    dim_sub,
                    error,
                    max_error,
                );
                break;
            }
            // The information of the current iteration is printed to the console.
            utils::print_davidson_iteration(
                i,
                roots_cvd,
                n_roots - roots_cvd,
                dim_sub,
                error,
                max_error,
            );

            // 5.  If the eigenvalues are not yet converged, the subspace basis is updated.
            // 5.1 Correction vectors are added to the current subspace basis, if the new
            //     dimension is lower than the maximal subspace size.
            if dim_sub + roots_lft <= max_space {
                let dk: usize = roots_lft * 2;
                let mut q_s: Array2<f64> = Array::zeros((guess.dim().0, dk));
                let mut nb: usize = 0;
                // select new expansion vectors among the non-converged left residual vectors
                for i in 0..n_roots {
                    if nb == dk {
                        //got enough new expansion vectors
                        break;
                    }
                    let mut w_precon: Array1<f64> = w[i] - &omega;
                    w_precon.mapv_inplace(|x| if x.abs() < 0.0001 { 1.0 } else { x });

                    if norms_l[i] > tolerance {
                        q_s.slice_mut(s![.., nb])
                            .assign(&((1.0 / &w_precon) * wl.slice(s![.., i])));
                        nb += 1;
                    }
                    if nb == dk {
                        //got enough new expansion vectors
                        break;
                    }
                    if norms_r[i] > tolerance {
                        q_s.slice_mut(s![.., nb])
                            .assign(&((1.0 / &w_precon) * wr.slice(s![.., i])));
                        nb += 1;
                    }
                }
                let mut new_dim:usize = 0;
                // The new subspace vectors are orthonormalized and added to the existing basis.
                for vec in q_s.axis_iter(Axis(1)) {
                    let orth_v: Array1<f64> = &vec - &guess.dot(&guess.t().dot(&vec));
                    let norm: f64 = orth_v.norm();
                    if norm > 1.0e-7 {
                        guess.push_column((&orth_v / norm).view());
                        new_dim += 1;
                    }
                }
                // if the norm of all new expansion vectors is below 1.0e-7, reset the subspace
                if new_dim == 0{
                    // The dimension of the subspace is reset to the initial value.
                    dim_sub = dim_sub_origin;
                    guess = ritz.slice(s![.., 0..dim_sub]).to_owned();
                }

                dim_sub = dim_sub + new_dim;
            }
            // 5.1 If the dimension is larger than the maximal subspace size, the subspace is
            //     collapsed.
            else {
                // The dimension of the subspace is reset to the initial value.
                dim_sub = dim_sub_origin;
                guess = ritz.slice(s![.., 0..dim_sub]).to_owned();
            }
        }
        // The end of the Davidson routine is noted in the console together with information
        // about the used wall time.
        utils::print_davidson_end(result.is_ok(), timer);

        // The returned result contains either an Err if the iteration is not converged or
        // an instance of Davidson that contains the eigenvectors and eigenvalues.
        result
    }

    /// Extract the requested eigenvalues/eigenvectors pairs
    fn create_results(
        subspace_eigenvalues: ArrayView1<f64>,
        ritz_vectors: ArrayView2<f64>,
        nvalues: usize,
        r_vectors: ArrayView2<f64>,
        l_vectors: ArrayView2<f64>,
    ) -> CasidaSolver {
        let eigenvectors: Array2<f64> = ritz_vectors
            .slice(s![.., 0..nvalues])
            .as_standard_layout()
            .to_owned();
        let eigs_squared: ArrayView1<f64> = subspace_eigenvalues.slice(s![0..nvalues]);
        let eigenvalues: Array1<f64> = eigs_squared.map(|val| val.sqrt());

        let xpy: Array2<f64> = r_vectors
            .slice(s![.., 0..nvalues])
            .as_standard_layout()
            .to_owned();
        let xmy: Array2<f64> = l_vectors
            .slice(s![.., 0..nvalues])
            .as_standard_layout()
            .to_owned();

        CasidaSolver {
            eigenvalues,
            eigenvectors,
            xpy,
            xmy,
        }
    }
}
