//! Direct LAPACK eigenvalue solver for symmetric matrices
//!
//! This bypasses ndarray-linalg overhead by calling DSYEVD directly.

use ndarray::{Array1, Array2, ArrayView2};

extern "C" {
    /// LAPACK DSYEVD: Symmetric eigenvalue problem with divide-and-conquer
    fn dsyevd_(
        jobz: *const u8,
        uplo: *const u8,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        w: *mut f64,
        work: *mut f64,
        lwork: *const i32,
        iwork: *mut i32,
        liwork: *const i32,
        info: *mut i32,
    );
}

/// Solve symmetric eigenvalue problem A*v = lambda*v using DSYEVD
///
/// Returns (eigenvalues, eigenvectors) with eigenvalues in ascending order.
/// The input matrix is in column-major (Fortran) order.
pub fn dsyevd_eigh(a: ArrayView2<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = a.nrows() as i32;
    assert_eq!(a.nrows(), a.ncols(), "Matrix must be square");

    // Copy matrix to column-major order (LAPACK expects Fortran order)
    // ndarray is row-major by default, so we need to transpose
    let mut a_work: Vec<f64> = Vec::with_capacity((n * n) as usize);
    for j in 0..n as usize {
        for i in 0..n as usize {
            a_work.push(a[[i, j]]);
        }
    }

    let mut w = vec![0.0f64; n as usize];  // eigenvalues
    let mut info: i32 = 0;

    // Workspace query
    let mut work_query = vec![0.0f64; 1];
    let mut iwork_query = vec![0i32; 1];
    let lwork_query: i32 = -1;
    let liwork_query: i32 = -1;

    unsafe {
        dsyevd_(
            b"V".as_ptr(),  // compute eigenvalues and eigenvectors
            b"U".as_ptr(),  // upper triangle
            &n,
            a_work.as_mut_ptr(),
            &n,
            w.as_mut_ptr(),
            work_query.as_mut_ptr(),
            &lwork_query,
            iwork_query.as_mut_ptr(),
            &liwork_query,
            &mut info,
        );
    }

    if info != 0 {
        panic!("DSYEVD workspace query failed with info = {}", info);
    }

    // Allocate optimal workspace
    let lwork = work_query[0] as i32;
    let liwork = iwork_query[0];
    let mut work = vec![0.0f64; lwork as usize];
    let mut iwork = vec![0i32; liwork as usize];

    // Actual computation
    unsafe {
        dsyevd_(
            b"V".as_ptr(),
            b"U".as_ptr(),
            &n,
            a_work.as_mut_ptr(),
            &n,
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &liwork,
            &mut info,
        );
    }

    if info != 0 {
        panic!("DSYEVD failed with info = {}", info);
    }

    // Convert back to row-major ndarray
    let eigenvalues = Array1::from_vec(w);
    let mut eigenvectors = Array2::zeros((n as usize, n as usize));
    for j in 0..n as usize {
        for i in 0..n as usize {
            eigenvectors[[i, j]] = a_work[j * n as usize + i];
        }
    }

    (eigenvalues, eigenvectors)
}

/// Compute S^(-1/2) using direct LAPACK eigenvalue decomposition
/// S = V * D * V^T, so S^(-1/2) = V * D^(-1/2) * V^T
pub fn compute_s_inv_sqrt(s: ArrayView2<f64>) -> Array2<f64> {
    let (eigenvalues, eigenvectors) = dsyevd_eigh(s);
    let n = eigenvalues.len();

    // Compute D^(-1/2) and multiply into eigenvectors: X = V * D^(-1/2)
    // Then result = X * V^T
    let mut x = eigenvectors.clone();
    for j in 0..n {
        let d_inv_sqrt = 1.0 / eigenvalues[j].sqrt();
        for i in 0..n {
            x[[i, j]] *= d_inv_sqrt;
        }
    }

    // X * V^T = (V * D^(-1/2)) * V^T = S^(-1/2)
    x.dot(&eigenvectors.t())
}

/// Works directly with column-major data.
/// Assumes input is already in column-major format (transposed from row-major view)
pub fn dsyevd_eigh_inplace(a: &mut [f64], n: usize) -> Vec<f64> {
    let n_i32 = n as i32;
    let mut w = vec![0.0f64; n];
    let mut info: i32 = 0;

    // Workspace query
    let mut work_query = vec![0.0f64; 1];
    let mut iwork_query = vec![0i32; 1];
    let lwork_query: i32 = -1;
    let liwork_query: i32 = -1;

    unsafe {
        dsyevd_(
            b"V".as_ptr(),
            b"U".as_ptr(),
            &n_i32,
            a.as_mut_ptr(),
            &n_i32,
            w.as_mut_ptr(),
            work_query.as_mut_ptr(),
            &lwork_query,
            iwork_query.as_mut_ptr(),
            &liwork_query,
            &mut info,
        );
    }

    let lwork = work_query[0] as i32;
    let liwork = iwork_query[0];
    let mut work = vec![0.0f64; lwork as usize];
    let mut iwork = vec![0i32; liwork as usize];

    unsafe {
        dsyevd_(
            b"V".as_ptr(),
            b"U".as_ptr(),
            &n_i32,
            a.as_mut_ptr(),
            &n_i32,
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &liwork,
            &mut info,
        );
    }

    if info != 0 {
        panic!("DSYEVD failed with info = {}", info);
    }

    w
}

/// Same as [`dsyevd_eigh`] but returns the eigenvalues as a plain `Vec`.
pub fn dsyevd_eigh_vec(a: ArrayView2<f64>) -> (Vec<f64>, Array2<f64>) {
    let (w, u) = dsyevd_eigh(a);
    (w.to_vec(), u)
}
