//! BLAS-optimized workspace for CT state exchange integrals in Davidson algorithm
//!
//! This workspace eliminates per-iteration allocations and expensive `as_standard_layout()`
//! copies by pre-computing intermediates and reusing workspace buffers.

use dialect_utilities::linalg::dgemm::{ct_permute_axes_120, dgemm_a_bt, dgemm_row_major};
use ndarray::prelude::*;

/// BLAS-optimized workspace for CT Davidson exchange computations
///
/// This workspace stores:
/// - Pre-computed intermediate: gamma_lr^T . q_vv permuted for efficient contraction
/// - Reusable buffers for DGEMM operations
///
/// Memory layout:
/// - gamma_lr_qvv_permuted: [n_virt, n_virt * natoms_l] - pre-computed once
/// - buf_v_a_ib: [natoms_l * n_occ * n_virt] - intermediate product
/// - buf_permuted: [n_occ * n_virt * natoms_l] - permuted intermediate
/// - buf_exchange_out: [n_occ * n_virt] - final exchange result
#[derive(Clone, Debug)]
pub struct CTDavidsonWorkspace {
    // Dimensions
    pub n_occ: usize,
    pub n_virt: usize,
    pub natoms_h: usize,

    // Pre-computed intermediate: gamma_lr . q_vv permuted to [n_virt, n_virt * natoms_h]
    // This avoids recomputing the permutation every iteration
    pub gamma_lr_qvv_permuted: Vec<f64>,

    // Flattened q_oo for direct BLAS access: [natoms_h * n_occ, n_occ]
    pub q_oo_flat: Vec<f64>,

    // BLAS workspace buffers (reused across iterations)
    pub buf_v_a_ib: Vec<f64>,       // [natoms_h * n_occ * n_virt]
    pub buf_permuted: Vec<f64>,     // [n_occ * n_virt * natoms_l]
    pub buf_exchange_out: Vec<f64>, // [n_occ * n_virt]
}

impl CTDavidsonWorkspace {
    /// Create a new workspace from pre-computed arrays
    ///
    /// # Arguments
    /// * `q_oo` - Transition charges [natoms_l, n_occ * n_occ] (from m_h)
    /// * `q_vv` - Transition charges [natoms_l, n_virt * n_virt] (from m_l)
    /// * `gamma_lr` - Long-range gamma matrix slice [natoms_h, natoms_l]
    /// * `n_occ` - Number of occupied orbitals
    /// * `n_virt` - Number of virtual orbitals
    pub fn new(
        q_oo: ArrayView2<f64>,
        q_vv: ArrayView2<f64>,
        gamma_lr: ArrayView2<f64>,
        n_occ: usize,
        n_virt: usize,
    ) -> Self {
        let _natoms_l = q_vv.dim().0;
        let natoms_h = q_oo.dim().0;

        // Pre-compute gamma_lr^T . q_vv and permute to [n_virt, n_virt * natoms_l]
        // Original: gamma_lr.t().dot(&q_vv) gives [natoms_l, n_virt * n_virt]
        // Reshape to [natoms_l, n_virt, n_virt], permute to [n_virt, n_virt, natoms_l]
        // Then flatten to [n_virt, n_virt * natoms_l]
        let gamma_lr_qvv: Array2<f64> = gamma_lr.dot(&q_vv);
        let gamma_lr_qvv_3d = gamma_lr_qvv.into_shape([natoms_h, n_virt, n_virt]).unwrap();

        // Permute [natoms_l, n_virt, n_virt] -> [n_virt, n_virt, natoms_l]
        // and flatten to [n_virt, n_virt * natoms_l]
        let gamma_lr_qvv_permuted =
            Self::permute_and_flatten_120(gamma_lr_qvv_3d.view(), natoms_h, n_virt, n_virt);

        // Flatten q_oo for direct BLAS access
        let q_oo_flat: Vec<f64> = if let Some(slice) = q_oo.as_slice() {
            slice.to_vec()
        } else {
            q_oo.as_standard_layout().iter().cloned().collect()
        };

        // Allocate workspace buffers
        let buf_v_a_ib = vec![0.0; natoms_h * n_occ * n_virt];
        let buf_permuted = vec![0.0; n_occ * n_virt * natoms_h];
        let buf_exchange_out = vec![0.0; n_occ * n_virt];

        Self {
            n_occ,
            n_virt,
            natoms_h,
            gamma_lr_qvv_permuted,
            q_oo_flat,
            buf_v_a_ib,
            buf_permuted,
            buf_exchange_out,
        }
    }

    /// Permute 3D array [A, B, C] -> [B, C, A] and flatten to [B, C * A]
    fn permute_and_flatten_120(
        arr: ArrayView3<f64>,
        dim_a: usize,
        dim_b: usize,
        dim_c: usize,
    ) -> Vec<f64> {
        let mut result = vec![0.0; dim_b * dim_c * dim_a];
        // Original: arr[a, b, c]
        // Target: result[b, c * dim_a + a] = result[b * (dim_c * dim_a) + c * dim_a + a]
        for a in 0..dim_a {
            for b in 0..dim_b {
                for c in 0..dim_c {
                    result[b * (dim_c * dim_a) + c * dim_a + a] = arr[[a, b, c]];
                }
            }
        }
        result
    }

    /// Compute exchange term using ndarray (reference implementation for verification)
    ///
    /// This is the original implementation that the BLAS version should match.
    pub fn compute_exchange_ndarray(
        &self,
        xi_flat: &[f64],
        q_oo: ArrayView2<f64>,
        gamma_lr_qvv: ArrayView3<f64>,
    ) -> Array1<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        let natoms_h = self.natoms_h;

        // Reshape xi to [n_occ, n_virt]
        let xi = ArrayView1::from(xi_flat)
            .into_shape((n_occ, n_virt))
            .unwrap();

        // Reshape q_oo for contraction: [natoms_l * n_occ, n_occ]
        let q_oo_r: ArrayView2<f64> = q_oo.into_shape([natoms_h * n_occ, n_occ]).unwrap();

        // Compute v_a_ib = q_oo_r . xi
        let v_a_ib: Array2<f64> = q_oo_r.dot(&xi);

        // Reshape and contract with gamma_lr_qvv
        let arr: Array2<f64> = v_a_ib
            .into_shape([natoms_h, n_occ, n_virt])
            .unwrap()
            .permuted_axes([1, 2, 0])
            .as_standard_layout()
            .into_shape([n_occ, n_virt * natoms_h])
            .unwrap()
            .dot(
                &gamma_lr_qvv
                    .view()
                    .permuted_axes([1, 2, 0])
                    .as_standard_layout()
                    .into_shape([n_virt, n_virt * natoms_h])
                    .unwrap()
                    .t(),
            );

        arr.into_shape(n_occ * n_virt).unwrap()
    }

    /// Compute exchange term for a single trial vector using direct BLAS calls
    ///
    /// This replaces the per-vector loop in compute_products that does:
    /// ```text
    /// let v_a_ib = q_oo_r.dot(&xi);
    /// let arr = v_a_ib
    ///     .into_shape([natoms_l, n_occ, n_virt]).unwrap()
    ///     .permuted_axes([1, 2, 0])
    ///     .as_standard_layout()
    ///     .into_shape([n_occ, n_virt * natoms_l]).unwrap()
    ///     .dot(&gamma_a_ab.view().permuted_axes([1, 2, 0])
    ///         .as_standard_layout()
    ///         .into_shape([n_virt, n_virt * natoms_l]).unwrap().t());
    /// ```
    ///
    /// # Arguments
    /// * `xi_flat` - Trial vector as flat slice [n_occ * n_virt] in row-major
    ///
    /// # Returns
    /// Reference to buf_exchange_out containing the exchange contribution [n_occ * n_virt]
    ///
    /// # Safety
    /// Uses unsafe BLAS (FFI) calls.
    pub fn compute_exchange_blas(&mut self, xi_flat: &[f64]) -> &[f64] {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        let natoms_h = self.natoms_h;

        unsafe {
            // Step 1: v_a_ib = q_oo_r . xi
            // q_oo_flat is [natoms_l * n_occ, n_occ]
            // xi is [n_occ, n_virt] (row-major, so [n_occ * n_virt] flat)
            // Result: [natoms_l * n_occ, n_virt]
            dgemm_row_major(
                1.0,
                &self.q_oo_flat,
                natoms_h * n_occ,
                n_occ,
                xi_flat,
                n_virt,
                0.0,
                &mut self.buf_v_a_ib,
            );

            // Step 2: Permute [natoms_l, n_occ, n_virt] -> [n_occ, n_virt, natoms_l]
            // buf_v_a_ib is [natoms_l * n_occ * n_virt] interpreted as [natoms_l, n_occ, n_virt]
            // buf_permuted will be [n_occ * n_virt * natoms_l] interpreted as [n_occ, n_virt * natoms_l]
            ct_permute_axes_120(
                self.buf_v_a_ib.as_ptr(),
                self.buf_permuted.as_mut_ptr(),
                natoms_h,
                n_occ,
                n_virt,
            );

            // Step 3: result = buf_permuted . gamma_lr_qvv_permuted^T
            // buf_permuted is [n_occ, n_virt * natoms_l]
            // gamma_lr_qvv_permuted is [n_virt, n_virt * natoms_l]
            // gamma_lr_qvv_permuted^T is [n_virt * natoms_l, n_virt]
            // Result: [n_occ, n_virt]
            dgemm_a_bt(
                1.0,
                &self.buf_permuted,
                n_occ,
                n_virt * natoms_h,
                &self.gamma_lr_qvv_permuted,
                n_virt, // B is [n_virt, n_virt * natoms_l], B^T is [n_virt * natoms_l, n_virt]
                0.0,
                &mut self.buf_exchange_out,
            );
        }

        &self.buf_exchange_out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};
    use std::time::Instant;

    #[test]
    fn test_permute_and_flatten_120() {
        // Create a simple 2x3x4 array with known values
        let dim_a = 2;
        let dim_b = 3;
        let dim_c = 4;
        let mut arr = Array3::<f64>::zeros((dim_a, dim_b, dim_c));
        for a in 0..dim_a {
            for b in 0..dim_b {
                for c in 0..dim_c {
                    arr[[a, b, c]] = (a * 100 + b * 10 + c) as f64;
                }
            }
        }

        let result = CTDavidsonWorkspace::permute_and_flatten_120(arr.view(), dim_a, dim_b, dim_c);

        // Verify: result[b, c * dim_a + a] should equal arr[a, b, c]
        for a in 0..dim_a {
            for b in 0..dim_b {
                for c in 0..dim_c {
                    let idx = b * (dim_c * dim_a) + c * dim_a + a;
                    assert!(
                        (result[idx] - arr[[a, b, c]]).abs() < 1e-10,
                        "Mismatch at [{}, {}, {}]: expected {}, got {}",
                        a,
                        b,
                        c,
                        arr[[a, b, c]],
                        result[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_blas_vs_ndarray_exchange() {
        // Test with realistic dimensions for a CT state
        // Typical fragment: ~85 atoms, ~170 occ, ~170 virt
        // Using smaller dimensions for faster test: 20 atoms, 30 occ, 40 virt
        let natoms_l = 20;
        let n_occ = 30;
        let n_virt = 40;

        // Create random test data with reproducible seed
        let seed = 42u64;
        let mut rng_state = seed;
        let next_rand = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*state as f64) / (u64::MAX as f64) - 0.5
        };

        // q_oo: [natoms_l, n_occ * n_occ]
        let mut q_oo_data = vec![0.0; natoms_l * n_occ * n_occ];
        for x in q_oo_data.iter_mut() {
            *x = next_rand(&mut rng_state);
        }
        let q_oo = Array2::from_shape_vec((natoms_l, n_occ * n_occ), q_oo_data).unwrap();

        // q_vv: [natoms_l, n_virt * n_virt]
        let mut q_vv_data = vec![0.0; natoms_l * n_virt * n_virt];
        for x in q_vv_data.iter_mut() {
            *x = next_rand(&mut rng_state);
        }
        let q_vv = Array2::from_shape_vec((natoms_l, n_virt * n_virt), q_vv_data).unwrap();

        // gamma_lr: [natoms_h, natoms_l] - using same for simplicity
        let mut gamma_lr_data = vec![0.0; natoms_l * natoms_l];
        for x in gamma_lr_data.iter_mut() {
            *x = next_rand(&mut rng_state);
        }
        let gamma_lr = Array2::from_shape_vec((natoms_l, natoms_l), gamma_lr_data).unwrap();

        // Pre-compute gamma_lr_qvv for ndarray reference
        let gamma_lr_qvv: Array3<f64> = gamma_lr
            .dot(&q_vv)
            .into_shape([natoms_l, n_virt, n_virt])
            .unwrap();

        // Create workspace
        let mut workspace =
            CTDavidsonWorkspace::new(q_oo.view(), q_vv.view(), gamma_lr.view(), n_occ, n_virt);

        // Create multiple test trial vectors
        let n_vectors = 50;
        let mut xi_vectors: Vec<Vec<f64>> = Vec::new();
        for _ in 0..n_vectors {
            let mut xi = vec![0.0; n_occ * n_virt];
            for x in xi.iter_mut() {
                *x = next_rand(&mut rng_state);
            }
            xi_vectors.push(xi);
        }

        // Test correctness: compare BLAS vs ndarray for each vector
        let mut max_diff = 0.0f64;
        for xi in &xi_vectors {
            // Copy BLAS result to release mutable borrow
            let blas_result: Vec<f64> = workspace.compute_exchange_blas(xi).to_vec();
            let ndarray_result =
                workspace.compute_exchange_ndarray(xi, q_oo.view(), gamma_lr_qvv.view());

            for (b, n) in blas_result.iter().zip(ndarray_result.iter()) {
                let diff = (b - n).abs();
                max_diff = max_diff.max(diff);
            }
        }

        println!("Max difference between BLAS and ndarray: {:.2e}", max_diff);
        assert!(
            max_diff < 1e-10,
            "BLAS and ndarray results differ by {:.2e}",
            max_diff
        );

        // Timing comparison
        let iterations = 100;

        // Time BLAS version
        let start_blas = Instant::now();
        for _ in 0..iterations {
            for xi in &xi_vectors {
                let _ = workspace.compute_exchange_blas(xi);
            }
        }
        let blas_time = start_blas.elapsed();

        // Time ndarray version
        let start_ndarray = Instant::now();
        for _ in 0..iterations {
            for xi in &xi_vectors {
                let _ = workspace.compute_exchange_ndarray(xi, q_oo.view(), gamma_lr_qvv.view());
            }
        }
        let ndarray_time = start_ndarray.elapsed();

        let total_ops = iterations * n_vectors;
        let blas_per_op = blas_time.as_nanos() as f64 / total_ops as f64;
        let ndarray_per_op = ndarray_time.as_nanos() as f64 / total_ops as f64;
        let speedup = ndarray_per_op / blas_per_op;

        println!(
            "\n=== Timing Comparison ({} iterations x {} vectors) ===",
            iterations, n_vectors
        );
        println!(
            "Dimensions: natoms_l={}, n_occ={}, n_virt={}",
            natoms_l, n_occ, n_virt
        );
        println!(
            "BLAS:   {:>8.2} us/op ({:>8.2} ms total)",
            blas_per_op / 1000.0,
            blas_time.as_millis()
        );
        println!(
            "ndarray: {:>8.2} us/op ({:>8.2} ms total)",
            ndarray_per_op / 1000.0,
            ndarray_time.as_millis()
        );
        println!("Speedup: {:.2}x", speedup);
    }

    #[test]
    fn test_blas_vs_ndarray_large() {
        // Test with larger dimensions closer to real-world CT states
        // LH2 fragment pair: ~170 atoms total, ~85 atoms per fragment
        let natoms_l = 85;
        let n_occ = 120;
        let n_virt = 141;

        // Create random test data with reproducible seed
        let seed = 12345u64;
        let mut rng_state = seed;
        let next_rand = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*state as f64) / (u64::MAX as f64) - 0.5
        };

        // q_oo: [natoms_l, n_occ * n_occ]
        let mut q_oo_data = vec![0.0; natoms_l * n_occ * n_occ];
        for x in q_oo_data.iter_mut() {
            *x = next_rand(&mut rng_state);
        }
        let q_oo = Array2::from_shape_vec((natoms_l, n_occ * n_occ), q_oo_data).unwrap();

        // q_vv: [natoms_l, n_virt * n_virt]
        let mut q_vv_data = vec![0.0; natoms_l * n_virt * n_virt];
        for x in q_vv_data.iter_mut() {
            *x = next_rand(&mut rng_state);
        }
        let q_vv = Array2::from_shape_vec((natoms_l, n_virt * n_virt), q_vv_data).unwrap();

        // gamma_lr: [natoms_h, natoms_l]
        let mut gamma_lr_data = vec![0.0; natoms_l * natoms_l];
        for x in gamma_lr_data.iter_mut() {
            *x = next_rand(&mut rng_state);
        }
        let gamma_lr = Array2::from_shape_vec((natoms_l, natoms_l), gamma_lr_data).unwrap();

        // Pre-compute gamma_lr_qvv for ndarray reference
        let gamma_lr_qvv: Array3<f64> = gamma_lr
            .dot(&q_vv)
            .into_shape([natoms_l, n_virt, n_virt])
            .unwrap();

        // Create workspace
        let mut workspace =
            CTDavidsonWorkspace::new(q_oo.view(), q_vv.view(), gamma_lr.view(), n_occ, n_virt);

        // Create test trial vectors (typical Davidson subspace size)
        let n_vectors = 20;
        let mut xi_vectors: Vec<Vec<f64>> = Vec::new();
        for _ in 0..n_vectors {
            let mut xi = vec![0.0; n_occ * n_virt];
            for x in xi.iter_mut() {
                *x = next_rand(&mut rng_state);
            }
            xi_vectors.push(xi);
        }

        // Test correctness
        let mut max_diff = 0.0f64;
        for xi in &xi_vectors {
            // Copy BLAS result to release mutable borrow
            let blas_result: Vec<f64> = workspace.compute_exchange_blas(xi).to_vec();
            let ndarray_result =
                workspace.compute_exchange_ndarray(xi, q_oo.view(), gamma_lr_qvv.view());

            for (b, n) in blas_result.iter().zip(ndarray_result.iter()) {
                let diff = (b - n).abs();
                max_diff = max_diff.max(diff);
            }
        }

        println!("\n=== Large Test (LH2-like dimensions) ===");
        println!(
            "Dimensions: natoms_l={}, n_occ={}, n_virt={}",
            natoms_l, n_occ, n_virt
        );
        println!("Max difference between BLAS and ndarray: {:.2e}", max_diff);
        assert!(
            max_diff < 1e-10,
            "BLAS and ndarray results differ by {:.2e}",
            max_diff
        );

        // Timing comparison
        let iterations = 10;

        // Time BLAS version
        let start_blas = Instant::now();
        for _ in 0..iterations {
            for xi in &xi_vectors {
                let _ = workspace.compute_exchange_blas(xi);
            }
        }
        let blas_time = start_blas.elapsed();

        // Time ndarray version
        let start_ndarray = Instant::now();
        for _ in 0..iterations {
            for xi in &xi_vectors {
                let _ = workspace.compute_exchange_ndarray(xi, q_oo.view(), gamma_lr_qvv.view());
            }
        }
        let ndarray_time = start_ndarray.elapsed();

        let total_ops = iterations * n_vectors;
        let blas_per_op = blas_time.as_nanos() as f64 / total_ops as f64;
        let ndarray_per_op = ndarray_time.as_nanos() as f64 / total_ops as f64;
        let speedup = ndarray_per_op / blas_per_op;

        println!(
            "\n=== Timing Comparison ({} iterations x {} vectors) ===",
            iterations, n_vectors
        );
        println!(
            "BLAS:    {:>10.2} us/op ({:>8.2} ms total)",
            blas_per_op / 1000.0,
            blas_time.as_millis()
        );
        println!(
            "ndarray: {:>10.2} us/op ({:>8.2} ms total)",
            ndarray_per_op / 1000.0,
            ndarray_time.as_millis()
        );
        println!("Speedup: {:.2}x", speedup);
    }
}
