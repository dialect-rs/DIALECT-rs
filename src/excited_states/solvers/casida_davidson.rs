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

            let w: Array1<f64> = u.mapv(f64::sqrt);
            let wsq: Array1<f64> = w.mapv(f64::sqrt);

            // approximate right R = (X+Y) and left L = (X-Y) eigenvectors
            // in the basis bs
            // (X+Y) = (A-B)^(1/2).T / sqrt(w)
            let rb: Array2<f64> = sq_a_m_b.dot(&v) / &wsq;
            // L = (X-Y) = 1/w * (A+B).(X+Y)
            let lb: Array2<f64> = apb_proj.dot(&rb) / &w;

            // transform to the canonical basis Lb -> L, Rb -> R
            let l_vector: Array2<f64> = guess.dot(&lb);
            let r_vector: Array2<f64> = guess.dot(&rb);

            // Calculate the residual vectors
            let wl: Array2<f64> = a_p_b.dot(&rb) - &l_vector * &w;
            let wr: Array2<f64> = a_m_b.dot(&lb) - &r_vector * &w;

            // norms
            let mut norms: Array1<f64> = Array::zeros(n_roots);
            let mut norms_l: Array1<f64> = Array::zeros(n_roots);
            let mut norms_r: Array1<f64> = Array::zeros(n_roots);
            for i in 0..n_roots {
                norms_l[i] = wl.slice(s![.., i]).dot(&wl.slice(s![.., i]));
                norms_r[i] = wr.slice(s![.., i]).dot(&wr.slice(s![.., i]));
                norms[i] = norms_l[i] + norms_r[i];
            }
            // check for convergence
            let eps: f64 = 0.1 * tolerance;
            let roots_cvd: usize = norms
                .iter()
                .fold(0, |n, &x| if x < tolerance { n + 1 } else { n });

            // number of not converged roots
            let roots_lft: usize = n_roots - roots_cvd;
            // number of norm values that are above the convergence threshold
            let nc_l: usize = norms_r
                .iter()
                .fold(0, |n, &x| if x > 0.1 * eps { n + 1 } else { n });
            let nc_r: usize = norms_l
                .iter()
                .fold(0, |n, &x| if x > 0.1 * eps { n + 1 } else { n });

            let error: f64 = norms.sum();
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
                // Half the new expansion vectors should come from the left residual vectors
                // the other half from the right residual vectors.
                let dk_r: usize = ((roots_lft as f64 / 2.0) as usize).min(nc_r);
                let dk_l: usize = (roots_lft - dk_r).min(nc_l);
                let mut dk: usize = (dk_r + dk_l) * 2;
                if dk == 0 {
                    dk += 2;
                }

                let mut Qs: Array2<f64> = Array::zeros((guess.dim().0, dk));
                let mut nb: usize = 0;
                // select new expansion vectors among the non-converged left residual vectors
                for i in 0..n_roots {
                    if nb == dk {
                        //got enough new expansion vectors
                        break;
                    }
                    let w_precon: Array1<f64> = w[i] - &omega;
                    let temp: Array1<f64> =
                        w_precon.map(|val| if val < &1.0e-6 { 1.0e-6 } else { 0.0 });
                    let temp_2: Array1<f64> =
                        w_precon.map(|&val| if val < 1.0e-6 { 0.0 } else { val });
                    let w_precon: Array1<f64> = &temp * &omega + temp_2;

                    if norms_l[i] > tolerance {
                        Qs.slice_mut(s![.., nb])
                            .assign(&((1.0 / &w_precon) * wl.slice(s![.., i])));
                        nb += 1;
                    }
                    if nb == dk {
                        //got enough new expansion vectors
                        break;
                    }
                    if norms_r[i] > tolerance {
                        Qs.slice_mut(s![.., nb])
                            .assign(&((1.0 / &w_precon) * wr.slice(s![.., i])));
                        nb += 1;
                    }
                }
                let mut new_guess: Array2<f64> = Array::zeros((guess.dim().0, dim_sub + dk));
                new_guess.slice_mut(s![.., ..dim_sub]).assign(&guess);
                new_guess.slice_mut(s![.., dim_sub..]).assign(&Qs);

                //QR decomposition
                let qr = new_guess.qr().unwrap();
                guess = qr.0;

                dim_sub = dim_sub + dk;
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
