//! Direct BLAS DGEMM for in-place matrix multiplication
//!
//! This bypasses ndarray allocation overhead by calling DGEMM directly.
//! Follows the same pattern as scc/lapack_eigh.rs for DSYEVD.

extern "C" {
    /// BLAS DGEMM: C = alpha * op(A) * op(B) + beta * C
    fn dgemm_(
        transa: *const u8, // 'N' or 'T'
        transb: *const u8, // 'N' or 'T'
        m: *const i32,     // rows of op(A) and C
        n: *const i32,     // cols of op(B) and C
        k: *const i32,     // cols of op(A), rows of op(B)
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *const f64,
        ldb: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
}

/// In-place matrix multiplication: C = alpha * A * B + beta * C
///
/// For row-major matrices (ndarray default), we use the identity:
/// (A * B)^T = B^T * A^T
///
/// Since DGEMM expects column-major (Fortran) order, but ndarray is row-major,
/// a row-major matrix looks like its transpose to DGEMM.
/// So for row-major: A_row * B_row = C_row means B^T * A^T = C^T in column-major view.
///
/// # Arguments
/// * `alpha` - Scalar multiplier for A*B
/// * `a` - Input matrix A [m x k] in row-major order
/// * `b` - Input matrix B [k x n] in row-major order
/// * `beta` - Scalar multiplier for C
/// * `c` - Output matrix C [m x n] in row-major order (modified in-place)
///
/// # Safety
/// Caller must ensure:
/// - All slices have correct sizes
/// - c is mutable and sized correctly
#[inline]
pub unsafe fn dgemm_row_major(
    alpha: f64,
    a: &[f64],
    m: usize,
    k: usize, // A is m x k
    b: &[f64],
    n: usize, // B is k x n
    beta: f64,
    c: &mut [f64], // C is m x n
) {
    debug_assert_eq!(a.len(), m * k, "A size mismatch");
    debug_assert_eq!(b.len(), k * n, "B size mismatch");
    debug_assert_eq!(c.len(), m * n, "C size mismatch");

    // For row-major: compute B^T * A^T = C^T
    // In DGEMM terms with column-major assumption:
    // B viewed as column-major is B^T (n x k)
    // A viewed as column-major is A^T (k x m)
    // C viewed as column-major is C^T (n x m)
    // So we call dgemm('N', 'N', n, m, k, alpha, b, n, a, k, beta, c, n)
    let m_i32 = m as i32;
    let n_i32 = n as i32;
    let k_i32 = k as i32;

    dgemm_(
        b"N".as_ptr(),
        b"N".as_ptr(),
        &n_i32, // rows of op(B^T) = n
        &m_i32, // cols of op(A^T) = m
        &k_i32, // inner dimension
        &alpha,
        b.as_ptr(),
        &n_i32, // leading dimension of B (viewed as column-major n x k)
        a.as_ptr(),
        &k_i32, // leading dimension of A (viewed as column-major k x m)
        &beta,
        c.as_mut_ptr(),
        &n_i32, // leading dimension of C (viewed as column-major n x m)
    );
}

/// In-place matrix multiplication with transposed A: C = alpha * A^T * B + beta * C
///
/// # Arguments
/// * `alpha` - Scalar multiplier for A^T*B
/// * `a` - Input matrix A [k x m] in row-major order (will be transposed)
/// * `b` - Input matrix B [k x n] in row-major order
/// * `beta` - Scalar multiplier for C
/// * `c` - Output matrix C [m x n] in row-major order (modified in-place)
#[inline]
pub unsafe fn dgemm_at_b(
    alpha: f64,
    a: &[f64],
    k: usize,
    m: usize, // A is k x m, A^T is m x k
    b: &[f64],
    n: usize, // B is k x n
    beta: f64,
    c: &mut [f64], // C is m x n
) {
    debug_assert_eq!(a.len(), k * m, "A size mismatch");
    debug_assert_eq!(b.len(), k * n, "B size mismatch");
    debug_assert_eq!(c.len(), m * n, "C size mismatch");

    // For row-major A^T * B = C:
    // Row-major A (k x m) viewed as column-major is A^T (m x k)
    // So A^T * B in row-major means: A (m x k) * B (k x n) = C (m x n)
    // In column-major view: B^T (n x k) * A^T (k x m) = C^T (n x m)
    // Call dgemm('N', 'T', n, m, k, alpha, b, n, a, m, beta, c, n)
    let m_i32 = m as i32;
    let n_i32 = n as i32;
    let k_i32 = k as i32;

    dgemm_(
        b"N".as_ptr(),
        b"T".as_ptr(), // Transpose A in column-major view
        &n_i32,
        &m_i32,
        &k_i32,
        &alpha,
        b.as_ptr(),
        &n_i32,
        a.as_ptr(),
        &m_i32, // lda for transposed operation
        &beta,
        c.as_mut_ptr(),
        &n_i32,
    );
}

/// In-place matrix multiplication with transposed B: C = alpha * A * B^T + beta * C
///
/// # Arguments
/// * `alpha` - Scalar multiplier for A*B^T
/// * `a` - Input matrix A [m x k] in row-major order
/// * `b` - Input matrix B [n x k] in row-major order (will be transposed to k x n)
/// * `beta` - Scalar multiplier for C
/// * `c` - Output matrix C [m x n] in row-major order (modified in-place)
#[inline]
pub unsafe fn dgemm_a_bt(
    alpha: f64,
    a: &[f64],
    m: usize,
    k: usize, // A is m x k
    b: &[f64],
    n: usize, // B is n x k, B^T is k x n
    beta: f64,
    c: &mut [f64], // C is m x n
) {
    debug_assert_eq!(a.len(), m * k, "A size mismatch");
    debug_assert_eq!(b.len(), n * k, "B size mismatch");
    debug_assert_eq!(c.len(), m * n, "C size mismatch");

    // For row-major A * B^T = C where B is n x k:
    // A * B^T in row-major = (B * A^T)^T in any order
    // Row-major A (m x k) as col-major = A^T (k x m)
    // Row-major B (n x k) as col-major = B^T (k x n), we want (B^T)^T = B
    // So: (B)^T * (A)^T = C^T => call dgemm('T', 'N', n, m, k, ...)
    let m_i32 = m as i32;
    let n_i32 = n as i32;
    let k_i32 = k as i32;

    dgemm_(
        b"T".as_ptr(), // Transpose B in column-major view
        b"N".as_ptr(),
        &n_i32,
        &m_i32,
        &k_i32,
        &alpha,
        b.as_ptr(),
        &k_i32, // ldb for transposed: original cols of B (row-major)
        a.as_ptr(),
        &k_i32,
        &beta,
        c.as_mut_ptr(),
        &n_i32,
    );
}

/// Cache-blocked batch transpose: [n_batch, m, n] -> [n_batch, n, m]
/// Each 2D slice is transposed independently using cache-friendly blocking.
#[inline]
pub fn batch_transpose_blocked(
    src: &[f64],
    dst: &mut [f64],
    n_batch: usize,
    m: usize, // rows in source (cols in dest)
    n: usize, // cols in source (rows in dest)
) {
    const BLOCK: usize = 32; // Tuned for typical L1 cache line

    debug_assert_eq!(src.len(), n_batch * m * n);
    debug_assert_eq!(dst.len(), n_batch * m * n);

    let slice_size = m * n;

    for batch in 0..n_batch {
        let src_base = batch * slice_size;
        let dst_base = batch * slice_size;

        // Process in blocks for cache efficiency
        let mut ii = 0;
        while ii < m {
            let i_end = (ii + BLOCK).min(m);
            let mut jj = 0;
            while jj < n {
                let j_end = (jj + BLOCK).min(n);

                // Tight inner loops - compiler can vectorize these
                for i in ii..i_end {
                    let src_row = src_base + i * n;
                    for j in jj..j_end {
                        // src[batch, i, j] -> dst[batch, j, i]
                        dst[dst_base + j * m + i] = src[src_row + j];
                    }
                }
                jj += BLOCK;
            }
            ii += BLOCK;
        }
    }
}

/// Optimized transpose for the common case where n_batch is large
/// Uses unsafe pointer arithmetic to eliminate bounds checking overhead
#[inline]
pub unsafe fn batch_transpose_blocked_unchecked(
    src: *const f64,
    dst: *mut f64,
    n_batch: usize,
    m: usize,
    n: usize,
) {
    const BLOCK: usize = 32;

    let slice_size = m * n;

    for batch in 0..n_batch {
        let src_base = src.add(batch * slice_size);
        let dst_base = dst.add(batch * slice_size);

        let mut ii = 0;
        while ii < m {
            let i_end = (ii + BLOCK).min(m);
            let mut jj = 0;
            while jj < n {
                let j_end = (jj + BLOCK).min(n);

                for i in ii..i_end {
                    let src_row = src_base.add(i * n);
                    for j in jj..j_end {
                        *dst_base.add(j * m + i) = *src_row.add(j);
                    }
                }
                jj += BLOCK;
            }
            ii += BLOCK;
        }
    }
}

/// Permute 3D array [A, B, C] -> [B, C, A] with cache-blocked access
///
/// This is used for CT state exchange integrals where we need to permute
/// [natoms_l, n_occ, n_virt] -> [n_occ, n_virt, natoms_l]
///
/// The result is flattened to [B, C * A] for subsequent BLAS operations.
///
/// # Arguments
/// * `src` - Source array [A * B * C] in row-major order (A is outermost)
/// * `dst` - Destination array [B * C * A] in row-major order (B is outermost)
/// * `dim_a` - Size of dimension A (e.g., natoms_l)
/// * `dim_b` - Size of dimension B (e.g., n_occ)
/// * `dim_c` - Size of dimension C (e.g., n_virt)
///
/// # Safety
/// Caller must ensure src and dst have correct sizes: A * B * C elements each
#[inline]
pub unsafe fn ct_permute_axes_120(
    src: *const f64,
    dst: *mut f64,
    dim_a: usize,
    dim_b: usize,
    dim_c: usize,
) {
    const BLOCK: usize = 32; // L1 cache optimization

    // Source layout: [A, B, C] -> index = a * (B * C) + b * C + c
    // Dest layout: [B, C, A] -> index = b * (C * A) + c * A + a

    // Process in blocks for cache efficiency
    let mut bb = 0;
    while bb < dim_b {
        let b_end = (bb + BLOCK).min(dim_b);
        let mut cc = 0;
        while cc < dim_c {
            let c_end = (cc + BLOCK).min(dim_c);
            let mut aa = 0;
            while aa < dim_a {
                let a_end = (aa + BLOCK).min(dim_a);

                // Tight inner loops - compiler can vectorize these
                for b in bb..b_end {
                    for c in cc..c_end {
                        let dst_base = b * (dim_c * dim_a) + c * dim_a;
                        for a in aa..a_end {
                            let src_idx = a * (dim_b * dim_c) + b * dim_c + c;
                            let dst_idx = dst_base + a;
                            *dst.add(dst_idx) = *src.add(src_idx);
                        }
                    }
                }
                aa += BLOCK;
            }
            cc += BLOCK;
        }
        bb += BLOCK;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_dgemm_row_major() {
        // A = [[1, 2], [3, 4]] (2x2)
        // B = [[5, 6], [7, 8]] (2x2)
        // A * B = [[19, 22], [43, 50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];

        unsafe {
            dgemm_row_major(1.0, &a, 2, 2, &b, 2, 0.0, &mut c);
        }

        assert!((c[0] - 19.0).abs() < 1e-10);
        assert!((c[1] - 22.0).abs() < 1e-10);
        assert!((c[2] - 43.0).abs() < 1e-10);
        assert!((c[3] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_dgemm_at_b() {
        // A = [[1, 3], [2, 4]] (2x2), A^T = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]] (2x2)
        // A^T * B = [[19, 22], [43, 50]]
        let a = vec![1.0, 3.0, 2.0, 4.0]; // A stored row-major
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];

        unsafe {
            dgemm_at_b(1.0, &a, 2, 2, &b, 2, 0.0, &mut c);
        }

        assert!((c[0] - 19.0).abs() < 1e-10);
        assert!((c[1] - 22.0).abs() < 1e-10);
        assert!((c[2] - 43.0).abs() < 1e-10);
        assert!((c[3] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_dgemm_with_beta() {
        // Test C = alpha * A * B + beta * C
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![1.0, 1.0, 1.0, 1.0];

        unsafe {
            // C = 1.0 * A * B + 2.0 * C
            // = [[19, 22], [43, 50]] + [[2, 2], [2, 2]]
            // = [[21, 24], [45, 52]]
            dgemm_row_major(1.0, &a, 2, 2, &b, 2, 2.0, &mut c);
        }

        assert!((c[0] - 21.0).abs() < 1e-10);
        assert!((c[1] - 24.0).abs() < 1e-10);
        assert!((c[2] - 45.0).abs() < 1e-10);
        assert!((c[3] - 52.0).abs() < 1e-10);
    }

    #[test]
    fn test_against_ndarray() {
        // Larger test comparing against ndarray::dot
        let m = 10;
        let k = 15;
        let n = 12;

        let a_nd: Array2<f64> = Array2::from_shape_fn((m, k), |(i, j)| (i * k + j) as f64);
        let b_nd: Array2<f64> = Array2::from_shape_fn((k, n), |(i, j)| (i * n + j) as f64 * 0.1);
        let c_expected = a_nd.dot(&b_nd);

        let a_vec: Vec<f64> = a_nd.as_slice().unwrap().to_vec();
        let b_vec: Vec<f64> = b_nd.as_slice().unwrap().to_vec();
        let mut c_vec = vec![0.0; m * n];

        unsafe {
            dgemm_row_major(1.0, &a_vec, m, k, &b_vec, n, 0.0, &mut c_vec);
        }

        for (i, (expected, actual)) in c_expected.iter().zip(c_vec.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 1e-10,
                "Mismatch at {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_ct_permute_axes_120() {
        // Test permutation [A, B, C] -> [B, C, A]
        // A=2, B=3, C=4
        let dim_a = 2;
        let dim_b = 3;
        let dim_c = 4;

        // Create source with known pattern: src[a,b,c] = a*100 + b*10 + c
        let mut src = vec![0.0; dim_a * dim_b * dim_c];
        for a in 0..dim_a {
            for b in 0..dim_b {
                for c in 0..dim_c {
                    let idx = a * (dim_b * dim_c) + b * dim_c + c;
                    src[idx] = (a * 100 + b * 10 + c) as f64;
                }
            }
        }

        let mut dst = vec![0.0; dim_a * dim_b * dim_c];

        unsafe {
            ct_permute_axes_120(src.as_ptr(), dst.as_mut_ptr(), dim_a, dim_b, dim_c);
        }

        // Verify: dst[b, c, a] should equal src[a, b, c]
        for a in 0..dim_a {
            for b in 0..dim_b {
                for c in 0..dim_c {
                    let src_val = (a * 100 + b * 10 + c) as f64;
                    let dst_idx = b * (dim_c * dim_a) + c * dim_a + a;
                    assert!(
                        (dst[dst_idx] - src_val).abs() < 1e-10,
                        "Mismatch at [{}, {}, {}]: expected {}, got {}",
                        a,
                        b,
                        c,
                        src_val,
                        dst[dst_idx]
                    );
                }
            }
        }
    }
}
