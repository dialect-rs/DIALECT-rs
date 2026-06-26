/*!

# Davidson Diagonalization

The Davidson method is suitable for diagonal-dominant symmetric matrices,
that are quite common in certain scientific problems like [electronic
structure](https://en.wikipedia.org/wiki/Electronic_structure). The Davidson
method could be not practical for other kind of symmetric matrices.

The current implementation uses a general davidson algorithm, meaning
that it compute all the requested eigenvalues simultaneusly using a variable
size block approach. The family of Davidson algorithm only differ in the way
that the correction vector is computed.

*/

use crate::utils;
use crate::traits::DavidsonEngine;
// use crate::utils::array_helper::parallel_matrix_multiply;
use ndarray::prelude::*;
use ndarray_linalg::*;
use ndarray_stats::QuantileExt;
use rayon::prelude::*;
use std::error;
use std::fmt;
use std::time::Instant;

/// Threshold (in `m` — the small "output rows" dimension) below which
/// the parallel matmul wrappers fall back to a single `.dot()` call.
const PAR_MATMUL_MIN_M: usize = 16;

/// Compute `C = Gᵀ · A` with rayon-level row-split parallelism.
///
/// `G`: `(dim × m)`, `A`: `(dim × n)`, returns `(m × n)`. Splits along
/// `m` (columns of `G`): each chunk computes its own `C_chunk` via
/// `g_chunk.t().dot(a)`. Column slices of a column-major (Fortran)
/// `G` are contiguous, so chunking is BLAS-friendly even after
/// ndarray's `push_column` has flipped the layout (which it does
/// after iter 0 in this driver).
///
/// Falls back to a single `g.t().dot(a)` when only one rayon thread
/// is available or `m` is below `PAR_MATMUL_MIN_M`.
fn par_at_dot_b(g: ArrayView2<f64>, a: ArrayView2<f64>) -> Array2<f64> {
    let m = g.ncols();
    let n = a.ncols();
    let num_threads = rayon::current_num_threads();
    if num_threads <= 1 || m < PAR_MATMUL_MIN_M {
        return g.t().dot(&a);
    }
    let target_chunks = (3 * num_threads).min(m).max(1);
    let chunk_m = ((m + target_chunks - 1) / target_chunks).max(1);
    let ranges: Vec<(usize, usize)> = (0..m)
        .step_by(chunk_m)
        .map(|s| (s, (s + chunk_m).min(m)))
        .collect();
    let parts: Vec<Array2<f64>> = ranges
        .par_iter()
        .map(|&(s, e)| g.slice(s![.., s..e]).t().dot(&a))
        .collect();
    let mut out = Array2::<f64>::zeros((m, n));
    let mut row = 0;
    for part in parts {
        let nrows = part.nrows();
        out.slice_mut(s![row..row + nrows, ..]).assign(&part);
        row += nrows;
    }
    out
}

/// Compute `C = G · V` with rayon-level row-split parallelism.
///
/// `G`: `(dim × k)`, `V`: `(k × n)`, returns `(dim × n)`. Splits along
/// the `dim` direction — slicing rows of `G` and writing rows of `C`.
/// Falls back to a single `g.dot(v)` when serial.
fn par_dot(g: ArrayView2<f64>, v: ArrayView2<f64>) -> Array2<f64> {
    let dim = g.nrows();
    let n = v.ncols();
    let num_threads = rayon::current_num_threads();
    if num_threads <= 1 || dim < 128 {
        return g.dot(&v);
    }
    let target_chunks = (3 * num_threads).min(dim).max(1);
    let chunk_d = ((dim + target_chunks - 1) / target_chunks).max(1);
    let ranges: Vec<(usize, usize)> = (0..dim)
        .step_by(chunk_d)
        .map(|s| (s, (s + chunk_d).min(dim)))
        .collect();
    let parts: Vec<Array2<f64>> = ranges
        .par_iter()
        .map(|&(s, e)| g.slice(s![s..e, ..]).dot(&v))
        .collect();
    let mut out = Array2::<f64>::zeros((dim, n));
    let mut row = 0;
    for part in parts {
        let nrows = part.nrows();
        out.slice_mut(s![row..row + nrows, ..]).assign(&part);
        row += nrows;
    }
    out
}

#[derive(Debug, PartialEq)]
pub struct DavidsonError;

impl fmt::Display for DavidsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Davidson Algorithm did not converge!")
    }
}

impl error::Error for DavidsonError {}

/// Structure with the configuration data
pub struct Davidson {
    pub eigenvalues: Array1<f64>,
    pub eigenvectors: Array2<f64>,
}

impl Davidson {
    /// Compute the lowest eigenvalues of a symmetric, diagonal dominant matrix.
    /// * `engine` an object that implements the `DavidsonEngine` trait.
    /// * `guess` the initial guess for the eigenvectors.
    /// * `n_roots` the number of (lowest) eigenvalues/eigenvectors to compute.
    /// * `tolerance` numerical tolerance for convergence.
    /// * `max_iter` the maximal number of iterations.
    /// * `subspace_multiplier` Determines the maximum size of the number of expansion vectors
    /// * `verbose` whether to print iteration information
    pub fn new<D: DavidsonEngine>(
        engine: &mut D,
        guess: Array2<f64>,
        n_roots: usize,
        tolerance: f64,
        max_iter: usize,
        use_qr: bool,
        subspace_multiplier: usize,
        shell_resolved: bool,
    ) -> Result<Self, DavidsonError> {
        Self::new_with_verbose(engine, guess, n_roots, tolerance, max_iter, use_qr, subspace_multiplier, shell_resolved, true)
    }

    /// Same as `new` but with explicit verbose control
    pub fn new_with_verbose<D: DavidsonEngine>(
        engine: &mut D,
        guess: Array2<f64>,
        n_roots: usize,
        tolerance: f64,
        max_iter: usize,
        use_qr: bool,
        subspace_multiplier: usize,
        shell_resolved: bool,
        verbose: bool,
    ) -> Result<Self, DavidsonError> {
        // Timer to measure the time within the Davidson routine.
        let timer: Instant = Instant::now();

        // Per-iter step timing — opt-in via DAVIDSON_TIME=1.

        // Dimension of the original matrix problem.
        let dim: usize = engine.get_size();

        // The initial guess needs to be mutable.
        let mut guess: Array2<f64> = guess;

        // set original tolerance
        let tolerance: f64 = tolerance;

        // Dimension of the subspace.
        let dim_sub_origin: usize = guess.ncols();
        let mut dim_sub: usize = dim_sub_origin;

        // The maximal possible subspace, before it will be collapsed.
        let max_space: usize = subspace_multiplier * n_roots;

        // storage for print strings
        let mut print_str = String::from("");

        // The initial information of the Davidson routine are printed.
        print_str += &utils::print_davidson_init(max_iter, n_roots, tolerance);

        // Initialization of the result.
        let mut result = Err(DavidsonError);

        // Outer loop block Davidson schema.
        for i in 0..max_iter {
            // 1. The initial subspace is formed by projecting into the new guess vectors.
            // Matrix-vector product of A with the trial vectors.
            let ax: Array2<f64> = if shell_resolved {
                engine.compute_products_ao(guess.view())
            } else {
                engine.compute_products(guess.view())
            };

            // 1.1 Initialization of the subspace Hamiltonian.
            // `par_at_dot_b` splits the work along the `dim_sub` axis
            // and runs each chunk's DGEMM in a rayon worker. OpenBLAS
            // doesn't internally thread this gemm (K = dim is large but
            // M = dim_sub is small ~20-200), so explicit row-split is
            // the only way to get parallel scaling here.
            let a_proj: Array2<f64> = par_at_dot_b(guess.view(), ax.view());

            // 2. Solve the eigenvalue problem for the subspace Hamiltonian.
            // The eigenvalues (u) and eigenvectors (v) are already sorted in ascending order.
            let (mut u, mut v): (Array1<f64>, Array2<f64>) = a_proj.eigh(UPLO::Upper).unwrap();

            // Slice (u, v) to `dim_sub_origin`. This is the width we
            // keep through the iteration: the first `n_roots` drive
            // convergence checks; the full `dim_sub_origin` is what the
            // collapse path restarts from. A wider initial guess
            // (`dim_sub_origin > n_roots`) is useful for tight tolerances
            // or near-degenerate spectra — without this slicing the
            // collapse path would assume `dim_sub_origin == n_roots`.
            // The final output narrows to `n_roots` after the loop.
            let keep: usize = dim_sub_origin.min(u.len());
            u = u.slice_move(s![0..keep]);
            v = v.slice_move(s![.., 0..keep]);

            // check if omega contains zero
            //
            // This guard is here because the driver was originally built
            // for CIS, where subspace eigenvalues should be excitation
            // energies (positive, ~eV-scale). If two or more come in at
            // `< zero_tol` something has gone numerically wrong — singular
            // subspace, NaN, etc. — and Davidson should bail.
            let zero_tol: f64 = 1.0e-4;
            let mut zero_bool: bool = false;
            let mut zero_counter: usize = 0;
            // Only the FIRST n_roots eigenvalues drive the singular-
            // subspace guard. With `dim_sub_origin > n_roots` (wider
            // initial guess) the extra eigenpairs are valid Ritz states
            // we keep around for the collapse path; they shouldn't be
            // allowed to falsely fire the bail-out.
            for &val in u.iter().take(n_roots) {
                if val < zero_tol || val.is_nan() {
                    zero_counter += 1;
                }
            }
            if zero_counter > 1 {
                zero_bool = true;
            }
            if zero_bool {
                // stop davidson
                break;
            } else {
                // 3. Convergence checks are made.
                // 3.1 Compute the Ritz vectors. Row-split parallel
                // matmul along the `dim` axis (the dominant dimension).
                let ritz: Array2<f64> = par_dot(guess.view(), v.view());

                // 3.2 Compute the residue vectors. `ax · v` uses the
                // same row-split helper; the `- ritz · diag(u)` term is
                // just column scaling (cheap, kept serial).
                let ax_v = par_dot(ax.view(), v.view());
                let rk: Array2<f64> = ax_v - ritz.dot(&Array::from_diag(&u));

                // 3.3 Convergence check for each pair of eigenvalue and
                // eigenvector. Only the first n_roots residue columns
                // drive convergence — any extra columns from the wider
                // subspace (`dim_sub_origin > n_roots`) are kept for the
                // collapse path but not part of the user's target set.
                let errors: Array1<f64> = rk
                    .slice(s![.., 0..n_roots])
                    .axis_iter(Axis(1))
                    .map(|col| col.norm())
                    .collect();

                // The sum of all errors.
                let error: f64 = errors.sum();
                // The maximum value of the errors.
                let max_error: f64 = *errors.max().unwrap();

                // 4.3 Check how many eigenvalues are converged.
                let roots_cvd: usize = errors
                    .iter()
                    .fold(0, |n, &x| if x < tolerance { n + 1 } else { n });
                let roots_lft: usize = n_roots - roots_cvd;

                // If all eigenvalues are converged, the Davidson routine finished successfully.
                if roots_lft == 0 && i > 0 {
                    result = Ok(Self::create_results(u.view(), ritz.view(), n_roots));
                    print_str += &utils::print_davidson_iteration(
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
                print_str += &utils::print_davidson_iteration(
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
                    // For each (not converged) eigenvalue a new preconditioned subspace vector is
                    // added.
                    let mut add_space: Array2<f64> = Array::zeros([dim, roots_lft]);
                    for ((idx, _), mut space_k) in errors
                        .iter()
                        .enumerate()
                        .filter(|(_, &x)| x > tolerance)
                        .zip(add_space.axis_iter_mut(Axis(1)))
                    {
                        space_k.assign(&engine.precondition(rk.column(idx), u[idx]));
                    }
                    // The dimension of the subspace is updated.
                    dim_sub += roots_lft;

                    // The new subspace vectors are orthonormalised and
                    // added to the existing basis.
                    for vec in add_space.axis_iter(Axis(1)) {
                        let nrm0 = vec.norm();
                        if nrm0 <= 0.0 { continue; }
                        let vec_n: Array1<f64> = &vec / nrm0;
                        let mut orth_v: Array1<f64> = &vec_n - &guess.dot(&guess.t().dot(&vec_n));
                        let norm: f64 = orth_v.norm();
                        if norm > 1.0e-7 {
                            guess.push_column((&orth_v / norm).view()).unwrap();
                        }
                    }

                    if use_qr {
                        let tmp = guess.qr().unwrap();
                        guess = tmp.0;
                    }
                }
                // 5.1 If the dimension is larger than the maximal subspace size, the subspace is
                //     collapsed.
                else {
                    // The dimension of the subspace is reset to the initial value.
                    dim_sub = dim_sub_origin;
                    guess = ritz.slice(s![.., 0..dim_sub]).to_owned();
                    // **Double CGS** of the restarted basis.
                    for _ in 0..2 {
                        let n_cols = guess.ncols();
                        let mut renormed = Array2::<f64>::zeros((dim, 0));
                        for k in 0..n_cols {
                            let col = guess.column(k).to_owned();
                            let orth_v: Array1<f64> = if renormed.ncols() == 0 {
                                col
                            } else {
                                &col - &renormed.dot(&renormed.t().dot(&col))
                            };
                            let norm = orth_v.norm();
                            if norm > 1.0e-7 {
                                renormed.push_column((&orth_v / norm).view()).unwrap();
                            }
                        }
                        guess = renormed;
                    }
                    dim_sub = guess.ncols();
                }
            }
        }
        // The end of the Davidson routine is noted in the console together with information
        // about the used wall time.
        if verbose {
            utils::print_davidson_end(result.is_ok(), timer, print_str);
        }

        // The returned result contains either an Err if the iteration is not converged or
        // an instance of Davidson that contains the eigenvectors and eigenvalues.
        result
    }

    /// Extract the requested eigenvalues/eigenvectors pairs
    fn create_results(
        subspace_eigenvalues: ArrayView1<f64>,
        ritz_vectors: ArrayView2<f64>,
        nvalues: usize,
    ) -> Davidson {
        let eigenvectors: Array2<f64> = ritz_vectors
            .slice(s![.., 0..nvalues])
            .as_standard_layout()
            .to_owned();

        Davidson {
            eigenvalues: subspace_eigenvalues.slice(s![0..nvalues]).to_owned(),
            eigenvectors,
        }
    }
}
