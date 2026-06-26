//! BLAS-optimized Hav and Hplus implementations
//!
//! These use direct DGEMM calls and preallocated workspaces to avoid
//! the overhead of ndarray's reshape/permute/as_standard_layout operations.

use crate::linalg::dgemm::{batch_transpose_blocked, dgemm_a_bt, dgemm_row_major};
use crate::gradients::helpers::HplusType;
use ndarray::{Array2, ArrayView2, ArrayView3};

/// BLAS-optimized Hplus workspace with precomputed flattened arrays
pub struct HplusBlas {
    // Dimensions
    pub n_occ: usize,
    pub n_virt: usize,
    pub n_at: usize,

    // Flattened qtrans arrays [n_at, dim1 * dim2] in row-major
    pub qtrans_ov_flat: Vec<f64>, // [n_at, n_occ * n_virt]
    pub qtrans_vv_flat: Vec<f64>, // [n_at, n_virt * n_virt]
    pub qtrans_oo_flat: Vec<f64>, // [n_at, n_occ * n_occ]
    pub qtrans_vo_flat: Vec<f64>, // [n_at, n_virt * n_occ]

    // Transposed qtrans for exchange terms: [dim1, n_at * dim2]
    // qtrans_ov with axes [1,0,2] gives [n_occ, n_at, n_virt] -> [n_occ, n_at * n_virt]
    pub qtrans_ov_102: Vec<f64>, // [n_occ, n_at * n_virt]
    pub qtrans_oo_102: Vec<f64>, // [n_occ, n_at * n_occ]
    pub qtrans_vv_102: Vec<f64>, // [n_virt, n_at * n_virt]
    pub qtrans_vo_102: Vec<f64>, // [n_virt, n_at * n_occ]

    // Workspace buffers (reused across calls)
    pub buf_at: Vec<f64>,      // [n_at]
    pub buf_tmp1: Vec<f64>,    // [n_at, max_dim1 * max_dim2]
    pub buf_tmp2: Vec<f64>,    // [n_at, max_dim1 * max_dim2]
    pub buf_swapped: Vec<f64>, // [n_at, max_dim1 * max_dim2]
}

impl HplusBlas {
    pub fn new(
        qtrans_ov: ArrayView3<f64>,
        qtrans_vv: ArrayView3<f64>,
        qtrans_oo: ArrayView3<f64>,
        qtrans_vo: ArrayView3<f64>,
    ) -> Self {
        let n_at = qtrans_ov.dim().0;
        let n_occ = qtrans_ov.dim().1;
        let n_virt = qtrans_ov.dim().2;

        // Flatten qtrans arrays
        let qtrans_ov_flat = Self::flatten_3d(qtrans_ov);
        let qtrans_vv_flat = Self::flatten_3d(qtrans_vv);
        let qtrans_oo_flat = Self::flatten_3d(qtrans_oo);
        let qtrans_vo_flat = Self::flatten_3d(qtrans_vo);

        // Create transposed versions [dim1, n_at * dim2]
        let qtrans_ov_102 = Self::transpose_102(qtrans_ov, n_at, n_occ, n_virt);
        let qtrans_oo_102 = Self::transpose_102(qtrans_oo, n_at, n_occ, n_occ);
        let qtrans_vv_102 = Self::transpose_102(qtrans_vv, n_at, n_virt, n_virt);
        let qtrans_vo_102 = Self::transpose_102(qtrans_vo, n_at, n_virt, n_occ);

        // Allocate workspace buffers
        let max_dim = n_occ.max(n_virt);
        let buf_at = vec![0.0; n_at];
        let buf_tmp1 = vec![0.0; n_at * max_dim * max_dim];
        let buf_tmp2 = vec![0.0; n_at * max_dim * max_dim];
        let buf_swapped = vec![0.0; n_at * max_dim * max_dim];

        Self {
            n_occ,
            n_virt,
            n_at,
            qtrans_ov_flat,
            qtrans_vv_flat,
            qtrans_oo_flat,
            qtrans_vo_flat,
            qtrans_ov_102,
            qtrans_oo_102,
            qtrans_vv_102,
            qtrans_vo_102,
            buf_at,
            buf_tmp1,
            buf_tmp2,
            buf_swapped,
        }
    }

    pub fn flatten_3d(arr: ArrayView3<f64>) -> Vec<f64> {
        if let Some(slice) = arr.as_slice() {
            slice.to_vec()
        } else {
            arr.as_standard_layout().iter().cloned().collect()
        }
    }

    /// Transpose from [n_at, dim1, dim2] to [dim1, n_at * dim2]
    pub fn transpose_102(arr: ArrayView3<f64>, n_at: usize, dim1: usize, dim2: usize) -> Vec<f64> {
        let mut result = vec![0.0; dim1 * n_at * dim2];
        // Original: arr[at, i, j]
        // Target: result[i, at * dim2 + j] = result[i * (n_at * dim2) + at * dim2 + j]
        for at in 0..n_at {
            for i in 0..dim1 {
                for j in 0..dim2 {
                    result[i * (n_at * dim2) + at * dim2 + j] = arr[[at, i, j]];
                }
            }
        }
        result
    }

    pub fn compute(
        &mut self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
        hplus_type: HplusType,
    ) -> Array2<f64> {
        match hplus_type {
            HplusType::Tab => self.hplus_tab(g0, g0_lr, v),
            HplusType::Tij => self.hplus_tij(g0, g0_lr, v),
            HplusType::QiaXpy => self.hplus_qia_xpy(g0, g0_lr, v),
            HplusType::QiaTab => self.hplus_qia_tab(g0, g0_lr, v),
            HplusType::QiaTij => self.hplus_qia_tij(g0, g0_lr, v),
            HplusType::Qai | HplusType::Wij => self.hplus_qai_or_wij(g0, g0_lr, v),
        }
    }

    /// Compute Coulomb term: 4.0 * q_out^T · (g0 · (q_in · v_flat))
    /// where q_in is [n_at, in_size], q_out is [n_at, out_size], v_flat is [in_size]
    #[inline]
    fn coulomb_term(
        &self,
        g0: &[f64],
        q_in: &[f64],
        in_size: usize,
        q_out: &[f64],
        out_dim1: usize,
        out_dim2: usize,
        v_flat: &[f64],
        buf_at: &mut [f64],
    ) -> Array2<f64> {
        let n_at = self.n_at;
        let out_size = out_dim1 * out_dim2;

        // Step 1: tmp[at] = sum_k q_in[at, k] * v_flat[k]
        // This is q_in · v_flat where q_in is [n_at, in_size]
        unsafe {
            // tmp = q_in · v_flat  -> [n_at]
            dgemm_row_major(
                1.0, q_in, n_at, in_size, v_flat, 1, // v_flat as column vector
                0.0, buf_at,
            );
        }

        // Step 2: tmp2[at] = sum_at2 g0[at, at2] * tmp[at2]
        // This is g0 · tmp
        let mut tmp2 = vec![0.0; n_at];
        unsafe {
            dgemm_row_major(1.0, g0, n_at, n_at, buf_at, 1, 0.0, &mut tmp2);
        }

        // Step 3: result[k] = sum_at tmp2[at] * q_out[at, k]
        // This is tmp2^T · q_out = [1, n_at] · [n_at, out_size] = [1, out_size]
        let mut result_flat = vec![0.0; out_size];
        unsafe {
            // tmp2 as [1, n_at], q_out as [n_at, out_size]
            dgemm_row_major(4.0, &tmp2, 1, n_at, q_out, out_size, 0.0, &mut result_flat);
        }

        Array2::from_shape_vec((out_dim1, out_dim2), result_flat).unwrap()
    }

    /// Exchange term pattern:
    /// q · v^T -> [n_at, dim1, dim2] -> g0_lr · reshape -> [n_at, dim1, dim2]
    /// -> swap axes to [n_at, dim2, dim1] -> q_102 · swapped
    ///
    /// q_flat: [n_at, q_rows * q_cols] (will multiply with v transposed)
    /// Output: [out_dim1, out_dim2]
    #[inline]
    fn exchange_term(
        &mut self,
        g0_lr: &[f64],
        q_flat: &[f64],
        q_rows: usize,
        q_cols: usize,
        v: &[f64],
        v_rows: usize,
        v_cols: usize,
        transpose_v: bool,
        q_102: &[f64],
        out_dim1: usize,
        out_dim2: usize,
    ) -> Array2<f64> {
        let n_at = self.n_at;

        // The multiplication pattern is:
        // q_flat[n_at * q_rows, q_cols] · v^T[q_cols, v_rows] -> [n_at * q_rows, v_rows]
        // Then reshape to [n_at, q_rows * v_rows] for gamma multiplication

        let tmp_dim1 = q_rows;
        let tmp_dim2 = if transpose_v { v_rows } else { v_cols };
        let tmp_size = n_at * tmp_dim1 * tmp_dim2;

        // Ensure buffers are large enough
        let buf_tmp1 = &mut self.buf_tmp1[..tmp_size];
        let buf_tmp2 = &mut self.buf_tmp2[..tmp_size];
        let buf_swapped = &mut self.buf_swapped[..tmp_size];

        unsafe {
            // Step 1: buf_tmp1 = q_flat · v^T (or q_flat · v)
            // q_flat viewed as [n_at * q_rows, q_cols]
            // v is [v_rows, v_cols]
            if transpose_v {
                // [n_at * q_rows, q_cols] · [v_cols, v_rows]^T = [n_at * q_rows, v_rows]
                // which is [n_at * q_rows, q_cols] · [q_cols, v_rows] (since q_cols == v_cols for this to work)
                dgemm_a_bt(
                    1.0,
                    q_flat,
                    n_at * q_rows,
                    q_cols,
                    v,
                    v_rows, // B is [v_rows, v_cols], B^T is [v_cols, v_rows]
                    0.0,
                    buf_tmp1,
                );
            } else {
                // [n_at * q_rows, q_cols] · [v_rows, v_cols] = [n_at * q_rows, v_cols]
                dgemm_row_major(1.0, q_flat, n_at * q_rows, q_cols, v, v_cols, 0.0, buf_tmp1);
            }

            // Step 2: buf_tmp2 = g0_lr · buf_tmp1_reshaped
            // buf_tmp1 is [n_at, tmp_dim1 * tmp_dim2], g0_lr is [n_at, n_at]
            // Result: [n_at, tmp_dim1 * tmp_dim2]
            dgemm_row_major(
                1.0,
                g0_lr,
                n_at,
                n_at,
                buf_tmp1,
                tmp_dim1 * tmp_dim2,
                0.0,
                buf_tmp2,
            );

            // Step 3: Swap axes [n_at, tmp_dim1, tmp_dim2] -> [n_at, tmp_dim2, tmp_dim1]
            // This is batch transpose
            batch_transpose_blocked(buf_tmp2, buf_swapped, n_at, tmp_dim1, tmp_dim2);
        }

        // The exchange term computation is complex because dimensions vary per method.
        let mut result_flat = vec![0.0; out_dim1 * out_dim2];
        unsafe {
            // For now, use the simpler pattern where tmp_dim2 matches q_102's second grouping
            dgemm_row_major(
                1.0,
                q_102,
                out_dim1,
                n_at * tmp_dim2, // assuming q_102 cols = n_at * tmp_dim2
                buf_swapped,
                tmp_dim1, // swapped is [n_at * tmp_dim2, tmp_dim1]
                0.0,
                &mut result_flat,
            );
        }

        Array2::from_shape_vec((out_dim1, tmp_dim1), result_flat)
            .unwrap()
            .slice(ndarray::s![.., ..out_dim2])
            .to_owned()
    }

    fn hplus_tab(
        &mut self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        // let n_at = self.n_at;

        let g0_slice = g0.as_slice().unwrap();
        let g0_lr_slice = g0_lr.as_slice().unwrap();
        let v_flat: Vec<f64> = if let Some(s) = v.as_slice() {
            s.to_vec()
        } else {
            v.iter().cloned().collect()
        };

        // Term 1: Coulomb
        // q_vv · v_flat -> g0 -> q_oo
        let mut result = self.coulomb_term(
            g0_slice,
            &self.qtrans_vv_flat,
            n_virt * n_virt,
            &self.qtrans_oo_flat,
            n_occ,
            n_occ,
            &v_flat,
            &mut self.buf_at.clone(),
        );

        // Term 2: Exchange with v^T
        // q_ov · v^T -> [n_at, n_occ, n_virt] -> g0_lr -> swap -> q_ov_102 · swapped
        let term2 = self.exchange_term_tab_2(g0_lr_slice, &v_flat, n_virt, n_virt, true);
        result = result - term2;

        // Term 3: Exchange with v
        let term3 = self.exchange_term_tab_2(g0_lr_slice, &v_flat, n_virt, n_virt, false);
        result = result - term3;

        result
    }

    /// Specialized exchange term for hplus_tab terms 2 and 3
    fn exchange_term_tab_2(
        &mut self,
        g0_lr: &[f64],
        v: &[f64],
        v_rows: usize,
        v_cols: usize,
        transpose_v: bool,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        let n_at = self.n_at;

        // q_ov is [n_at, n_occ, n_virt] flattened to [n_at * n_occ, n_virt]
        // v is [n_virt, n_virt]
        // q_ov · v^T -> [n_at * n_occ, n_virt] but we interpret as [n_at, n_occ, n_virt]

        let tmp_size = n_at * n_occ * n_virt;
        let mut buf_tmp1 = vec![0.0; tmp_size];
        let mut buf_tmp2 = vec![0.0; tmp_size];
        let mut buf_swapped = vec![0.0; tmp_size];

        unsafe {
            // Step 1: q_ov[n_at*n_occ, n_virt] · v^T[n_virt, n_virt] or v[n_virt, n_virt]
            if transpose_v {
                dgemm_a_bt(
                    1.0,
                    &self.qtrans_ov_flat,
                    n_at * n_occ,
                    n_virt,
                    v,
                    v_rows,
                    0.0,
                    &mut buf_tmp1,
                );
            } else {
                dgemm_row_major(
                    1.0,
                    &self.qtrans_ov_flat,
                    n_at * n_occ,
                    n_virt,
                    v,
                    v_cols,
                    0.0,
                    &mut buf_tmp1,
                );
            }

            // Step 2: g0_lr[n_at, n_at] · buf_tmp1[n_at, n_occ*n_virt]
            dgemm_row_major(
                1.0,
                g0_lr,
                n_at,
                n_at,
                &buf_tmp1,
                n_occ * n_virt,
                0.0,
                &mut buf_tmp2,
            );

            // Step 3: Swap axes [n_at, n_occ, n_virt] -> [n_at, n_virt, n_occ]
            batch_transpose_blocked(&buf_tmp2, &mut buf_swapped, n_at, n_occ, n_virt);

            // Step 4: q_ov_102[n_occ, n_at*n_virt] · swapped[n_at*n_virt, n_occ]
            let mut result_flat = vec![0.0; n_occ * n_occ];
            dgemm_row_major(
                1.0,
                &self.qtrans_ov_102,
                n_occ,
                n_at * n_virt,
                &buf_swapped,
                n_occ,
                0.0,
                &mut result_flat,
            );

            Array2::from_shape_vec((n_occ, n_occ), result_flat).unwrap()
        }
    }

    fn hplus_tij(
        &mut self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        // let n_at = self.n_at;

        let g0_slice = g0.as_slice().unwrap();
        let g0_lr_slice = g0_lr.as_slice().unwrap();
        let v_flat: Vec<f64> = if let Some(s) = v.as_slice() {
            s.to_vec()
        } else {
            v.iter().cloned().collect()
        };

        // Term 1: Coulomb - q_oo · v_flat -> g0 -> q_oo
        let mut result = self.coulomb_term(
            g0_slice,
            &self.qtrans_oo_flat,
            n_occ * n_occ,
            &self.qtrans_oo_flat,
            n_occ,
            n_occ,
            &v_flat,
            &mut self.buf_at.clone(),
        );

        // Terms 2 and 3: Exchange
        let term2 = self.exchange_term_tij(g0_lr_slice, &v_flat, n_occ, n_occ, true);
        result = result - term2;

        let term3 = self.exchange_term_tij(g0_lr_slice, &v_flat, n_occ, n_occ, false);
        result = result - term3;

        result
    }

    fn exchange_term_tij(
        &mut self,
        g0_lr: &[f64],
        v: &[f64],
        v_rows: usize,
        v_cols: usize,
        transpose_v: bool,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_at = self.n_at;

        let tmp_size = n_at * n_occ * n_occ;
        let mut buf_tmp1 = vec![0.0; tmp_size];
        let mut buf_tmp2 = vec![0.0; tmp_size];
        let mut buf_swapped = vec![0.0; tmp_size];

        unsafe {
            // q_oo[n_at*n_occ, n_occ] · v^T or v
            if transpose_v {
                dgemm_a_bt(
                    1.0,
                    &self.qtrans_oo_flat,
                    n_at * n_occ,
                    n_occ,
                    v,
                    v_rows,
                    0.0,
                    &mut buf_tmp1,
                );
            } else {
                dgemm_row_major(
                    1.0,
                    &self.qtrans_oo_flat,
                    n_at * n_occ,
                    n_occ,
                    v,
                    v_cols,
                    0.0,
                    &mut buf_tmp1,
                );
            }

            // g0_lr · buf_tmp1
            dgemm_row_major(
                1.0,
                g0_lr,
                n_at,
                n_at,
                &buf_tmp1,
                n_occ * n_occ,
                0.0,
                &mut buf_tmp2,
            );

            // Swap [n_at, n_occ, n_occ] -> [n_at, n_occ, n_occ]
            batch_transpose_blocked(&buf_tmp2, &mut buf_swapped, n_at, n_occ, n_occ);

            // q_oo_102[n_occ, n_at*n_occ] · swapped[n_at*n_occ, n_occ]
            let mut result_flat = vec![0.0; n_occ * n_occ];
            dgemm_row_major(
                1.0,
                &self.qtrans_oo_102,
                n_occ,
                n_at * n_occ,
                &buf_swapped,
                n_occ,
                0.0,
                &mut result_flat,
            );

            Array2::from_shape_vec((n_occ, n_occ), result_flat).unwrap()
        }
    }

    fn hplus_qia_xpy(
        &mut self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        // let n_at = self.n_at;

        let g0_slice = g0.as_slice().unwrap();
        let g0_lr_slice = g0_lr.as_slice().unwrap();
        let v_flat: Vec<f64> = if let Some(s) = v.as_slice() {
            s.to_vec()
        } else {
            v.iter().cloned().collect()
        };

        // Term 1: Coulomb - q_ov · v_flat -> g0 -> q_vv
        let mut result = self.coulomb_term(
            g0_slice,
            &self.qtrans_ov_flat,
            n_occ * n_virt,
            &self.qtrans_vv_flat,
            n_virt,
            n_virt,
            &v_flat,
            &mut self.buf_at.clone(),
        );

        // Term 2: Exchange - q_vv · v^T, then q_vo_102
        let term2 = self.exchange_term_qia_xpy_2(g0_lr_slice, &v_flat, n_occ, n_virt, true);
        result = result - term2;

        // Term 3: Exchange - q_vo · v, then q_vv_102
        let term3 = self.exchange_term_qia_xpy_3(g0_lr_slice, &v_flat, n_occ, n_virt);
        result = result - term3;

        result
    }

    fn exchange_term_qia_xpy_2(
        &mut self,
        g0_lr: &[f64],
        v: &[f64],
        v_rows: usize,
        v_cols: usize,
        transpose_v: bool,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        let n_at = self.n_at;

        let tmp_size = n_at * n_virt * n_occ;
        let mut buf_tmp1 = vec![0.0; tmp_size];
        let mut buf_tmp2 = vec![0.0; tmp_size];
        let mut buf_swapped = vec![0.0; tmp_size];

        unsafe {
            // q_vv[n_at*n_virt, n_virt] · v^T[n_virt, n_occ] -> [n_at*n_virt, n_occ]
            if transpose_v {
                dgemm_a_bt(
                    1.0,
                    &self.qtrans_vv_flat,
                    n_at * n_virt,
                    n_virt,
                    v,
                    v_rows,
                    0.0,
                    &mut buf_tmp1,
                );
            } else {
                dgemm_row_major(
                    1.0,
                    &self.qtrans_vv_flat,
                    n_at * n_virt,
                    n_virt,
                    v,
                    v_cols,
                    0.0,
                    &mut buf_tmp1,
                );
            }

            // g0_lr · buf_tmp1 (reshaped as [n_at, n_virt*n_occ])
            dgemm_row_major(
                1.0,
                g0_lr,
                n_at,
                n_at,
                &buf_tmp1,
                n_virt * n_occ,
                0.0,
                &mut buf_tmp2,
            );

            // Swap [n_at, n_virt, n_occ] -> [n_at, n_occ, n_virt]
            batch_transpose_blocked(&buf_tmp2, &mut buf_swapped, n_at, n_virt, n_occ);

            // q_vo_102[n_virt, n_at*n_occ] · swapped[n_at*n_occ, n_virt]
            let mut result_flat = vec![0.0; n_virt * n_virt];
            dgemm_row_major(
                1.0,
                &self.qtrans_vo_102,
                n_virt,
                n_at * n_occ,
                &buf_swapped,
                n_virt,
                0.0,
                &mut result_flat,
            );

            Array2::from_shape_vec((n_virt, n_virt), result_flat).unwrap()
        }
    }

    fn exchange_term_qia_xpy_3(
        &mut self,
        g0_lr: &[f64],
        v: &[f64],
        _v_rows: usize,
        v_cols: usize,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        let n_at = self.n_at;

        let tmp_size = n_at * n_virt * n_virt;
        let mut buf_tmp1 = vec![0.0; tmp_size];
        let mut buf_tmp2 = vec![0.0; tmp_size];
        let mut buf_swapped = vec![0.0; tmp_size];

        unsafe {
            // q_vo[n_at*n_virt, n_occ] · v[n_occ, n_virt] -> [n_at*n_virt, n_virt]
            dgemm_row_major(
                1.0,
                &self.qtrans_vo_flat,
                n_at * n_virt,
                n_occ,
                v,
                v_cols,
                0.0,
                &mut buf_tmp1,
            );

            // g0_lr · buf_tmp1
            dgemm_row_major(
                1.0,
                g0_lr,
                n_at,
                n_at,
                &buf_tmp1,
                n_virt * n_virt,
                0.0,
                &mut buf_tmp2,
            );

            // Swap [n_at, n_virt, n_virt] -> [n_at, n_virt, n_virt]
            batch_transpose_blocked(&buf_tmp2, &mut buf_swapped, n_at, n_virt, n_virt);

            // q_vv_102[n_virt, n_at*n_virt] · swapped[n_at*n_virt, n_virt]
            let mut result_flat = vec![0.0; n_virt * n_virt];
            dgemm_row_major(
                1.0,
                &self.qtrans_vv_102,
                n_virt,
                n_at * n_virt,
                &buf_swapped,
                n_virt,
                0.0,
                &mut result_flat,
            );

            Array2::from_shape_vec((n_virt, n_virt), result_flat).unwrap()
        }
    }

    fn hplus_qia_tab(
        &mut self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        // let n_at = self.n_at;

        let g0_slice = g0.as_slice().unwrap();
        let g0_lr_slice = g0_lr.as_slice().unwrap();
        let v_flat: Vec<f64> = if let Some(s) = v.as_slice() {
            s.to_vec()
        } else {
            v.iter().cloned().collect()
        };

        // Term 1: Coulomb - q_vv · v_flat -> g0 -> q_ov
        let mut result = self.coulomb_term(
            g0_slice,
            &self.qtrans_vv_flat,
            n_virt * n_virt,
            &self.qtrans_ov_flat,
            n_occ,
            n_virt,
            &v_flat,
            &mut self.buf_at.clone(),
        );

        // Terms 2 and 3: Exchange
        let term2 = self.exchange_term_qia_tab(g0_lr_slice, &v_flat, n_virt, n_virt, true);
        result = result - term2;

        let term3 = self.exchange_term_qia_tab(g0_lr_slice, &v_flat, n_virt, n_virt, false);
        result = result - term3;

        result
    }

    fn exchange_term_qia_tab(
        &mut self,
        g0_lr: &[f64],
        v: &[f64],
        v_rows: usize,
        v_cols: usize,
        transpose_v: bool,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        let n_at = self.n_at;

        let tmp_size = n_at * n_virt * n_virt;
        let mut buf_tmp1 = vec![0.0; tmp_size];
        let mut buf_tmp2 = vec![0.0; tmp_size];
        let mut buf_swapped = vec![0.0; tmp_size];

        unsafe {
            // q_vv[n_at*n_virt, n_virt] · v^T or v
            if transpose_v {
                dgemm_a_bt(
                    1.0,
                    &self.qtrans_vv_flat,
                    n_at * n_virt,
                    n_virt,
                    v,
                    v_rows,
                    0.0,
                    &mut buf_tmp1,
                );
            } else {
                dgemm_row_major(
                    1.0,
                    &self.qtrans_vv_flat,
                    n_at * n_virt,
                    n_virt,
                    v,
                    v_cols,
                    0.0,
                    &mut buf_tmp1,
                );
            }

            // g0_lr · buf_tmp1
            dgemm_row_major(
                1.0,
                g0_lr,
                n_at,
                n_at,
                &buf_tmp1,
                n_virt * n_virt,
                0.0,
                &mut buf_tmp2,
            );

            // Swap [n_at, n_virt, n_virt] -> [n_at, n_virt, n_virt]
            batch_transpose_blocked(&buf_tmp2, &mut buf_swapped, n_at, n_virt, n_virt);

            // q_ov_102[n_occ, n_at*n_virt] · swapped[n_at*n_virt, n_virt]
            let mut result_flat = vec![0.0; n_occ * n_virt];
            dgemm_row_major(
                1.0,
                &self.qtrans_ov_102,
                n_occ,
                n_at * n_virt,
                &buf_swapped,
                n_virt,
                0.0,
                &mut result_flat,
            );

            Array2::from_shape_vec((n_occ, n_virt), result_flat).unwrap()
        }
    }

    fn hplus_qia_tij(
        &mut self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        // let n_at = self.n_at;

        let g0_slice = g0.as_slice().unwrap();
        let g0_lr_slice = g0_lr.as_slice().unwrap();
        let v_flat: Vec<f64> = if let Some(s) = v.as_slice() {
            s.to_vec()
        } else {
            v.iter().cloned().collect()
        };

        // Term 1: Coulomb - q_oo · v_flat -> g0 -> q_ov
        let mut result = self.coulomb_term(
            g0_slice,
            &self.qtrans_oo_flat,
            n_occ * n_occ,
            &self.qtrans_ov_flat,
            n_occ,
            n_virt,
            &v_flat,
            &mut self.buf_at.clone(),
        );

        // Terms 2 and 3: Exchange
        let term2 = self.exchange_term_qia_tij(g0_lr_slice, &v_flat, n_occ, n_occ, true);
        result = result - term2;

        let term3 = self.exchange_term_qia_tij(g0_lr_slice, &v_flat, n_occ, n_occ, false);
        result = result - term3;

        result
    }

    fn exchange_term_qia_tij(
        &mut self,
        g0_lr: &[f64],
        v: &[f64],
        v_rows: usize,
        v_cols: usize,
        transpose_v: bool,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        let n_at = self.n_at;

        let tmp_size = n_at * n_virt * n_occ;
        let mut buf_tmp1 = vec![0.0; tmp_size];
        let mut buf_tmp2 = vec![0.0; tmp_size];
        let mut buf_swapped = vec![0.0; tmp_size];

        unsafe {
            // q_vo[n_at*n_virt, n_occ] · v^T[n_occ, n_occ] or v[n_occ, n_occ]
            if transpose_v {
                dgemm_a_bt(
                    1.0,
                    &self.qtrans_vo_flat,
                    n_at * n_virt,
                    n_occ,
                    v,
                    v_rows,
                    0.0,
                    &mut buf_tmp1,
                );
            } else {
                dgemm_row_major(
                    1.0,
                    &self.qtrans_vo_flat,
                    n_at * n_virt,
                    n_occ,
                    v,
                    v_cols,
                    0.0,
                    &mut buf_tmp1,
                );
            }

            // g0_lr · buf_tmp1
            dgemm_row_major(
                1.0,
                g0_lr,
                n_at,
                n_at,
                &buf_tmp1,
                n_virt * n_occ,
                0.0,
                &mut buf_tmp2,
            );

            // Swap [n_at, n_virt, n_occ] -> [n_at, n_occ, n_virt]
            batch_transpose_blocked(&buf_tmp2, &mut buf_swapped, n_at, n_virt, n_occ);

            // q_oo_102[n_occ, n_at*n_occ] · swapped[n_at*n_occ, n_virt]
            let mut result_flat = vec![0.0; n_occ * n_virt];
            dgemm_row_major(
                1.0,
                &self.qtrans_oo_102,
                n_occ,
                n_at * n_occ,
                &buf_swapped,
                n_virt,
                0.0,
                &mut result_flat,
            );

            Array2::from_shape_vec((n_occ, n_virt), result_flat).unwrap()
        }
    }

    fn hplus_qai_or_wij(
        &mut self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        // let n_at = self.n_at;

        let g0_slice = g0.as_slice().unwrap();
        let g0_lr_slice = g0_lr.as_slice().unwrap();
        let v_flat: Vec<f64> = if let Some(s) = v.as_slice() {
            s.to_vec()
        } else {
            v.iter().cloned().collect()
        };

        // Term 1: Coulomb - q_ov · v_flat -> g0 -> q_oo
        let mut result = self.coulomb_term(
            g0_slice,
            &self.qtrans_ov_flat,
            n_occ * n_virt,
            &self.qtrans_oo_flat,
            n_occ,
            n_occ,
            &v_flat,
            &mut self.buf_at.clone(),
        );

        // Term 2: Exchange - q_ov · v^T
        let term2 = self.exchange_term_qai_wij_2(g0_lr_slice, &v_flat, n_occ, n_virt);
        result = result - term2;

        // Term 3: Exchange - q_oo · v
        let term3 = self.exchange_term_qai_wij_3(g0_lr_slice, &v_flat, n_occ, n_virt);
        result = result - term3;

        result
    }

    fn exchange_term_qai_wij_2(
        &mut self,
        g0_lr: &[f64],
        v: &[f64],
        v_rows: usize,
        _v_cols: usize,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        let n_at = self.n_at;

        let tmp_size = n_at * n_occ * n_occ;
        let mut buf_tmp1 = vec![0.0; tmp_size];
        let mut buf_tmp2 = vec![0.0; tmp_size];
        let mut buf_swapped = vec![0.0; tmp_size];

        unsafe {
            // q_ov[n_at*n_occ, n_virt] · v^T[n_virt, n_occ] -> [n_at*n_occ, n_occ]
            dgemm_a_bt(
                1.0,
                &self.qtrans_ov_flat,
                n_at * n_occ,
                n_virt,
                v,
                v_rows,
                0.0,
                &mut buf_tmp1,
            );

            // g0_lr · buf_tmp1
            dgemm_row_major(
                1.0,
                g0_lr,
                n_at,
                n_at,
                &buf_tmp1,
                n_occ * n_occ,
                0.0,
                &mut buf_tmp2,
            );

            // Swap [n_at, n_occ, n_occ] -> [n_at, n_occ, n_occ]
            batch_transpose_blocked(&buf_tmp2, &mut buf_swapped, n_at, n_occ, n_occ);

            // q_oo_102[n_occ, n_at*n_occ] · swapped[n_at*n_occ, n_occ]
            let mut result_flat = vec![0.0; n_occ * n_occ];
            dgemm_row_major(
                1.0,
                &self.qtrans_oo_102,
                n_occ,
                n_at * n_occ,
                &buf_swapped,
                n_occ,
                0.0,
                &mut result_flat,
            );

            Array2::from_shape_vec((n_occ, n_occ), result_flat).unwrap()
        }
    }

    fn exchange_term_qai_wij_3(
        &mut self,
        g0_lr: &[f64],
        v: &[f64],
        _v_rows: usize,
        v_cols: usize,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        let n_at = self.n_at;

        let tmp_size = n_at * n_occ * n_virt;
        let mut buf_tmp1 = vec![0.0; tmp_size];
        let mut buf_tmp2 = vec![0.0; tmp_size];
        let mut buf_swapped = vec![0.0; tmp_size];

        unsafe {
            // q_oo[n_at*n_occ, n_occ] · v[n_occ, n_virt] -> [n_at*n_occ, n_virt]
            dgemm_row_major(
                1.0,
                &self.qtrans_oo_flat,
                n_at * n_occ,
                n_occ,
                v,
                v_cols,
                0.0,
                &mut buf_tmp1,
            );

            // g0_lr · buf_tmp1
            dgemm_row_major(
                1.0,
                g0_lr,
                n_at,
                n_at,
                &buf_tmp1,
                n_occ * n_virt,
                0.0,
                &mut buf_tmp2,
            );

            // Swap [n_at, n_occ, n_virt] -> [n_at, n_virt, n_occ]
            batch_transpose_blocked(&buf_tmp2, &mut buf_swapped, n_at, n_occ, n_virt);

            // q_ov_102[n_occ, n_at*n_virt] · swapped[n_at*n_virt, n_occ]
            let mut result_flat = vec![0.0; n_occ * n_occ];
            dgemm_row_major(
                1.0,
                &self.qtrans_ov_102,
                n_occ,
                n_at * n_virt,
                &buf_swapped,
                n_occ,
                0.0,
                &mut result_flat,
            );

            Array2::from_shape_vec((n_occ, n_occ), result_flat).unwrap()
        }
    }
}

/// BLAS-optimized Hav with correct coefficients
/// Hav = 4.0*Coulomb - 2.0*Exchange (vs Hplus = 4.0*Coulomb - 1.0*Exchange)
pub struct HavBlas {
    // Dimensions
    pub n_occ: usize,
    pub n_virt: usize,
    pub n_at: usize,

    // Flattened qtrans arrays (shared with HplusBlas pattern)
    pub qtrans_ov_flat: Vec<f64>,
    pub qtrans_vv_flat: Vec<f64>,
    pub qtrans_oo_flat: Vec<f64>,
    pub qtrans_vo_flat: Vec<f64>,

    // Transposed qtrans for exchange terms
    pub qtrans_ov_102: Vec<f64>,
    pub qtrans_oo_102: Vec<f64>,
    pub qtrans_vv_102: Vec<f64>,
    pub qtrans_vo_102: Vec<f64>,

    // Workspace buffer
    pub buf_at: Vec<f64>,
}

impl HavBlas {
    pub fn new(
        qtrans_ov: ArrayView3<f64>,
        qtrans_vv: ArrayView3<f64>,
        qtrans_oo: ArrayView3<f64>,
        qtrans_vo: ArrayView3<f64>,
    ) -> Self {
        let n_at = qtrans_ov.dim().0;
        let n_occ = qtrans_ov.dim().1;
        let n_virt = qtrans_ov.dim().2;

        // Reuse the same helper functions from HplusBlas
        let qtrans_ov_flat = HplusBlas::flatten_3d(qtrans_ov);
        let qtrans_vv_flat = HplusBlas::flatten_3d(qtrans_vv);
        let qtrans_oo_flat = HplusBlas::flatten_3d(qtrans_oo);
        let qtrans_vo_flat = HplusBlas::flatten_3d(qtrans_vo);

        let qtrans_ov_102 = HplusBlas::transpose_102(qtrans_ov, n_at, n_occ, n_virt);
        let qtrans_oo_102 = HplusBlas::transpose_102(qtrans_oo, n_at, n_occ, n_occ);
        let qtrans_vv_102 = HplusBlas::transpose_102(qtrans_vv, n_at, n_virt, n_virt);
        let qtrans_vo_102 = HplusBlas::transpose_102(qtrans_vo, n_at, n_virt, n_occ);

        let buf_at = vec![0.0; n_at];

        Self {
            n_occ,
            n_virt,
            n_at,
            qtrans_ov_flat,
            qtrans_vv_flat,
            qtrans_oo_flat,
            qtrans_vo_flat,
            qtrans_ov_102,
            qtrans_oo_102,
            qtrans_vv_102,
            qtrans_vo_102,
            buf_at,
        }
    }

    pub fn compute(
        &mut self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
        hplus_type: HplusType,
    ) -> Array2<f64> {
        match hplus_type {
            HplusType::Tab => self.hav_tab(g0, g0_lr, v),
            HplusType::Tij => self.hav_tij(g0, g0_lr, v),
            HplusType::QiaXpy => self.hav_qia_xpy(g0, g0_lr, v),
            HplusType::QiaTab => self.hav_qia_tab(g0, g0_lr, v),
            HplusType::QiaTij => self.hav_qia_tij(g0, g0_lr, v),
            HplusType::Qai | HplusType::Wij => self.hav_qai_or_wij(g0, g0_lr, v),
        }
    }

    /// Coulomb term with factor 4.0 (same as Hplus)
    #[inline]
    fn coulomb_term(
        &self,
        g0: &[f64],
        q_in: &[f64],
        in_size: usize,
        q_out: &[f64],
        out_dim1: usize,
        out_dim2: usize,
        v_flat: &[f64],
        buf_at: &mut [f64],
    ) -> Array2<f64> {
        let n_at = self.n_at;
        let out_size = out_dim1 * out_dim2;

        unsafe {
            dgemm_row_major(1.0, q_in, n_at, in_size, v_flat, 1, 0.0, buf_at);
        }

        let mut tmp2 = vec![0.0; n_at];
        unsafe {
            dgemm_row_major(1.0, g0, n_at, n_at, buf_at, 1, 0.0, &mut tmp2);
        }

        let mut result_flat = vec![0.0; out_size];
        unsafe {
            dgemm_row_major(4.0, &tmp2, 1, n_at, q_out, out_size, 0.0, &mut result_flat);
        }

        Array2::from_shape_vec((out_dim1, out_dim2), result_flat).unwrap()
    }

    fn hav_tab(
        &mut self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        // let n_at = self.n_at;

        let g0_slice = g0.as_slice().unwrap();
        let g0_lr_slice = g0_lr.as_slice().unwrap();
        let v_flat: Vec<f64> = if let Some(s) = v.as_slice() {
            s.to_vec()
        } else {
            v.iter().cloned().collect()
        };

        // Coulomb term (factor 4.0, same as Hplus)
        let mut result = self.coulomb_term(
            g0_slice,
            &self.qtrans_vv_flat,
            n_virt * n_virt,
            &self.qtrans_oo_flat,
            n_occ,
            n_occ,
            &v_flat,
            &mut self.buf_at.clone(),
        );

        // Hav has ONLY ONE exchange term with v.t() (factor 2.0)
        // Unlike Hplus which has two terms (v.t() and v) each with factor 1.0
        let term2 = self.exchange_term_tab(g0_lr_slice, &v_flat, n_virt, true);
        result = result - 2.0 * &term2;

        result
    }

    fn exchange_term_tab(
        &self,
        g0_lr: &[f64],
        v: &[f64],
        v_dim: usize,
        transpose_v: bool,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        let n_at = self.n_at;

        let tmp_size = n_at * n_occ * n_virt;
        let mut buf_tmp1 = vec![0.0; tmp_size];
        let mut buf_tmp2 = vec![0.0; tmp_size];
        let mut buf_swapped = vec![0.0; tmp_size];

        unsafe {
            if transpose_v {
                dgemm_a_bt(
                    1.0,
                    &self.qtrans_ov_flat,
                    n_at * n_occ,
                    n_virt,
                    v,
                    v_dim,
                    0.0,
                    &mut buf_tmp1,
                );
            } else {
                dgemm_row_major(
                    1.0,
                    &self.qtrans_ov_flat,
                    n_at * n_occ,
                    n_virt,
                    v,
                    v_dim,
                    0.0,
                    &mut buf_tmp1,
                );
            }

            dgemm_row_major(
                1.0,
                g0_lr,
                n_at,
                n_at,
                &buf_tmp1,
                n_occ * n_virt,
                0.0,
                &mut buf_tmp2,
            );
            batch_transpose_blocked(&buf_tmp2, &mut buf_swapped, n_at, n_occ, n_virt);

            let mut result_flat = vec![0.0; n_occ * n_occ];
            dgemm_row_major(
                1.0,
                &self.qtrans_ov_102,
                n_occ,
                n_at * n_virt,
                &buf_swapped,
                n_occ,
                0.0,
                &mut result_flat,
            );

            Array2::from_shape_vec((n_occ, n_occ), result_flat).unwrap()
        }
    }

    fn hav_tij(
        &mut self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        // let n_at = self.n_at;

        let g0_slice = g0.as_slice().unwrap();
        let g0_lr_slice = g0_lr.as_slice().unwrap();
        let v_flat: Vec<f64> = if let Some(s) = v.as_slice() {
            s.to_vec()
        } else {
            v.iter().cloned().collect()
        };

        let mut result = self.coulomb_term(
            g0_slice,
            &self.qtrans_oo_flat,
            n_occ * n_occ,
            &self.qtrans_oo_flat,
            n_occ,
            n_occ,
            &v_flat,
            &mut self.buf_at.clone(),
        );

        // Hav has only ONE exchange term with v.t() (factor 2.0)
        let term2 = self.exchange_term_tij(g0_lr_slice, &v_flat, n_occ, true);
        result = result - 2.0 * &term2;

        result
    }

    fn exchange_term_tij(
        &self,
        g0_lr: &[f64],
        v: &[f64],
        v_dim: usize,
        transpose_v: bool,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_at = self.n_at;

        let tmp_size = n_at * n_occ * n_occ;
        let mut buf_tmp1 = vec![0.0; tmp_size];
        let mut buf_tmp2 = vec![0.0; tmp_size];
        let mut buf_swapped = vec![0.0; tmp_size];

        unsafe {
            if transpose_v {
                dgemm_a_bt(
                    1.0,
                    &self.qtrans_oo_flat,
                    n_at * n_occ,
                    n_occ,
                    v,
                    v_dim,
                    0.0,
                    &mut buf_tmp1,
                );
            } else {
                dgemm_row_major(
                    1.0,
                    &self.qtrans_oo_flat,
                    n_at * n_occ,
                    n_occ,
                    v,
                    v_dim,
                    0.0,
                    &mut buf_tmp1,
                );
            }

            dgemm_row_major(
                1.0,
                g0_lr,
                n_at,
                n_at,
                &buf_tmp1,
                n_occ * n_occ,
                0.0,
                &mut buf_tmp2,
            );
            batch_transpose_blocked(&buf_tmp2, &mut buf_swapped, n_at, n_occ, n_occ);

            let mut result_flat = vec![0.0; n_occ * n_occ];
            dgemm_row_major(
                1.0,
                &self.qtrans_oo_102,
                n_occ,
                n_at * n_occ,
                &buf_swapped,
                n_occ,
                0.0,
                &mut result_flat,
            );

            Array2::from_shape_vec((n_occ, n_occ), result_flat).unwrap()
        }
    }

    fn hav_qia_xpy(
        &mut self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        // let n_at = self.n_at;

        let g0_slice = g0.as_slice().unwrap();
        let g0_lr_slice = g0_lr.as_slice().unwrap();
        let v_flat: Vec<f64> = if let Some(s) = v.as_slice() {
            s.to_vec()
        } else {
            v.iter().cloned().collect()
        };

        let mut result = self.coulomb_term(
            g0_slice,
            &self.qtrans_ov_flat,
            n_occ * n_virt,
            &self.qtrans_vv_flat,
            n_virt,
            n_virt,
            &v_flat,
            &mut self.buf_at.clone(),
        );

        // Hav has only ONE exchange term (term2 with v.t())
        let term2 = self.exchange_term_qia_xpy_2(g0_lr_slice, &v_flat, n_occ, n_virt);
        result = result - 2.0 * &term2;

        result
    }

    fn exchange_term_qia_xpy_2(
        &self,
        g0_lr: &[f64],
        v: &[f64],
        v_rows: usize,
        _v_cols: usize,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        let n_at = self.n_at;

        let tmp_size = n_at * n_virt * n_occ;
        let mut buf_tmp1 = vec![0.0; tmp_size];
        let mut buf_tmp2 = vec![0.0; tmp_size];
        let mut buf_swapped = vec![0.0; tmp_size];

        unsafe {
            dgemm_a_bt(
                1.0,
                &self.qtrans_vv_flat,
                n_at * n_virt,
                n_virt,
                v,
                v_rows,
                0.0,
                &mut buf_tmp1,
            );
            dgemm_row_major(
                1.0,
                g0_lr,
                n_at,
                n_at,
                &buf_tmp1,
                n_virt * n_occ,
                0.0,
                &mut buf_tmp2,
            );
            batch_transpose_blocked(&buf_tmp2, &mut buf_swapped, n_at, n_virt, n_occ);

            let mut result_flat = vec![0.0; n_virt * n_virt];
            dgemm_row_major(
                1.0,
                &self.qtrans_vo_102,
                n_virt,
                n_at * n_occ,
                &buf_swapped,
                n_virt,
                0.0,
                &mut result_flat,
            );

            Array2::from_shape_vec((n_virt, n_virt), result_flat).unwrap()
        }
    }

    fn exchange_term_qia_xpy_3(
        &self,
        g0_lr: &[f64],
        v: &[f64],
        _v_rows: usize,
        v_cols: usize,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        let n_at = self.n_at;

        let tmp_size = n_at * n_virt * n_virt;
        let mut buf_tmp1 = vec![0.0; tmp_size];
        let mut buf_tmp2 = vec![0.0; tmp_size];
        let mut buf_swapped = vec![0.0; tmp_size];

        unsafe {
            dgemm_row_major(
                1.0,
                &self.qtrans_vo_flat,
                n_at * n_virt,
                n_occ,
                v,
                v_cols,
                0.0,
                &mut buf_tmp1,
            );
            dgemm_row_major(
                1.0,
                g0_lr,
                n_at,
                n_at,
                &buf_tmp1,
                n_virt * n_virt,
                0.0,
                &mut buf_tmp2,
            );
            batch_transpose_blocked(&buf_tmp2, &mut buf_swapped, n_at, n_virt, n_virt);

            let mut result_flat = vec![0.0; n_virt * n_virt];
            dgemm_row_major(
                1.0,
                &self.qtrans_vv_102,
                n_virt,
                n_at * n_virt,
                &buf_swapped,
                n_virt,
                0.0,
                &mut result_flat,
            );

            Array2::from_shape_vec((n_virt, n_virt), result_flat).unwrap()
        }
    }

    fn hav_qia_tab(
        &mut self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;

        let g0_slice = g0.as_slice().unwrap();
        let g0_lr_slice = g0_lr.as_slice().unwrap();
        let v_flat: Vec<f64> = if let Some(s) = v.as_slice() {
            s.to_vec()
        } else {
            v.iter().cloned().collect()
        };

        let mut result = self.coulomb_term(
            g0_slice,
            &self.qtrans_vv_flat,
            n_virt * n_virt,
            &self.qtrans_ov_flat,
            n_occ,
            n_virt,
            &v_flat,
            &mut self.buf_at.clone(),
        );

        // Hav has only ONE exchange term (term2 with v.t())
        let term2 = self.exchange_term_qia_tab(g0_lr_slice, &v_flat, n_virt, true);
        result = result - 2.0 * &term2;

        result
    }

    fn exchange_term_qia_tab(
        &self,
        g0_lr: &[f64],
        v: &[f64],
        v_dim: usize,
        transpose_v: bool,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        let n_at = self.n_at;

        let tmp_size = n_at * n_virt * n_virt;
        let mut buf_tmp1 = vec![0.0; tmp_size];
        let mut buf_tmp2 = vec![0.0; tmp_size];
        let mut buf_swapped = vec![0.0; tmp_size];

        unsafe {
            if transpose_v {
                dgemm_a_bt(
                    1.0,
                    &self.qtrans_vv_flat,
                    n_at * n_virt,
                    n_virt,
                    v,
                    v_dim,
                    0.0,
                    &mut buf_tmp1,
                );
            } else {
                dgemm_row_major(
                    1.0,
                    &self.qtrans_vv_flat,
                    n_at * n_virt,
                    n_virt,
                    v,
                    v_dim,
                    0.0,
                    &mut buf_tmp1,
                );
            }

            dgemm_row_major(
                1.0,
                g0_lr,
                n_at,
                n_at,
                &buf_tmp1,
                n_virt * n_virt,
                0.0,
                &mut buf_tmp2,
            );
            batch_transpose_blocked(&buf_tmp2, &mut buf_swapped, n_at, n_virt, n_virt);

            let mut result_flat = vec![0.0; n_occ * n_virt];
            dgemm_row_major(
                1.0,
                &self.qtrans_ov_102,
                n_occ,
                n_at * n_virt,
                &buf_swapped,
                n_virt,
                0.0,
                &mut result_flat,
            );

            Array2::from_shape_vec((n_occ, n_virt), result_flat).unwrap()
        }
    }

    fn hav_qia_tij(
        &mut self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;

        let g0_slice = g0.as_slice().unwrap();
        let g0_lr_slice = g0_lr.as_slice().unwrap();
        let v_flat: Vec<f64> = if let Some(s) = v.as_slice() {
            s.to_vec()
        } else {
            v.iter().cloned().collect()
        };

        let mut result = self.coulomb_term(
            g0_slice,
            &self.qtrans_oo_flat,
            n_occ * n_occ,
            &self.qtrans_ov_flat,
            n_occ,
            n_virt,
            &v_flat,
            &mut self.buf_at.clone(),
        );

        // Hav has only ONE exchange term (term2 with v.t())
        let term2 = self.exchange_term_qia_tij(g0_lr_slice, &v_flat, n_occ, true);
        result = result - 2.0 * &term2;

        result
    }

    fn exchange_term_qia_tij(
        &self,
        g0_lr: &[f64],
        v: &[f64],
        v_dim: usize,
        transpose_v: bool,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        let n_at = self.n_at;

        let tmp_size = n_at * n_virt * n_occ;
        let mut buf_tmp1 = vec![0.0; tmp_size];
        let mut buf_tmp2 = vec![0.0; tmp_size];
        let mut buf_swapped = vec![0.0; tmp_size];

        unsafe {
            if transpose_v {
                dgemm_a_bt(
                    1.0,
                    &self.qtrans_vo_flat,
                    n_at * n_virt,
                    n_occ,
                    v,
                    v_dim,
                    0.0,
                    &mut buf_tmp1,
                );
            } else {
                dgemm_row_major(
                    1.0,
                    &self.qtrans_vo_flat,
                    n_at * n_virt,
                    n_occ,
                    v,
                    v_dim,
                    0.0,
                    &mut buf_tmp1,
                );
            }

            dgemm_row_major(
                1.0,
                g0_lr,
                n_at,
                n_at,
                &buf_tmp1,
                n_virt * n_occ,
                0.0,
                &mut buf_tmp2,
            );
            batch_transpose_blocked(&buf_tmp2, &mut buf_swapped, n_at, n_virt, n_occ);

            let mut result_flat = vec![0.0; n_occ * n_virt];
            dgemm_row_major(
                1.0,
                &self.qtrans_oo_102,
                n_occ,
                n_at * n_occ,
                &buf_swapped,
                n_virt,
                0.0,
                &mut result_flat,
            );

            Array2::from_shape_vec((n_occ, n_virt), result_flat).unwrap()
        }
    }

    fn hav_qai_or_wij(
        &mut self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;

        let g0_slice = g0.as_slice().unwrap();
        let g0_lr_slice = g0_lr.as_slice().unwrap();
        let v_flat: Vec<f64> = if let Some(s) = v.as_slice() {
            s.to_vec()
        } else {
            v.iter().cloned().collect()
        };

        let mut result = self.coulomb_term(
            g0_slice,
            &self.qtrans_ov_flat,
            n_occ * n_virt,
            &self.qtrans_oo_flat,
            n_occ,
            n_occ,
            &v_flat,
            &mut self.buf_at.clone(),
        );

        // Hav has only ONE exchange term (term2 with v.t())
        let term2 = self.exchange_term_qai_wij_2(g0_lr_slice, &v_flat, n_occ, n_virt);
        result = result - 2.0 * &term2;

        result
    }

    fn exchange_term_qai_wij_2(
        &self,
        g0_lr: &[f64],
        v: &[f64],
        v_rows: usize,
        _v_cols: usize,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        let n_at = self.n_at;

        let tmp_size = n_at * n_occ * n_occ;
        let mut buf_tmp1 = vec![0.0; tmp_size];
        let mut buf_tmp2 = vec![0.0; tmp_size];
        let mut buf_swapped = vec![0.0; tmp_size];

        unsafe {
            dgemm_a_bt(
                1.0,
                &self.qtrans_ov_flat,
                n_at * n_occ,
                n_virt,
                v,
                v_rows,
                0.0,
                &mut buf_tmp1,
            );
            dgemm_row_major(
                1.0,
                g0_lr,
                n_at,
                n_at,
                &buf_tmp1,
                n_occ * n_occ,
                0.0,
                &mut buf_tmp2,
            );
            batch_transpose_blocked(&buf_tmp2, &mut buf_swapped, n_at, n_occ, n_occ);

            let mut result_flat = vec![0.0; n_occ * n_occ];
            dgemm_row_major(
                1.0,
                &self.qtrans_oo_102,
                n_occ,
                n_at * n_occ,
                &buf_swapped,
                n_occ,
                0.0,
                &mut result_flat,
            );

            Array2::from_shape_vec((n_occ, n_occ), result_flat).unwrap()
        }
    }

    fn exchange_term_qai_wij_3(
        &self,
        g0_lr: &[f64],
        v: &[f64],
        _v_rows: usize,
        v_cols: usize,
    ) -> Array2<f64> {
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;
        let n_at = self.n_at;

        let tmp_size = n_at * n_occ * n_virt;
        let mut buf_tmp1 = vec![0.0; tmp_size];
        let mut buf_tmp2 = vec![0.0; tmp_size];
        let mut buf_swapped = vec![0.0; tmp_size];

        unsafe {
            dgemm_row_major(
                1.0,
                &self.qtrans_oo_flat,
                n_at * n_occ,
                n_occ,
                v,
                v_cols,
                0.0,
                &mut buf_tmp1,
            );
            dgemm_row_major(
                1.0,
                g0_lr,
                n_at,
                n_at,
                &buf_tmp1,
                n_occ * n_virt,
                0.0,
                &mut buf_tmp2,
            );
            batch_transpose_blocked(&buf_tmp2, &mut buf_swapped, n_at, n_occ, n_virt);

            let mut result_flat = vec![0.0; n_occ * n_occ];
            dgemm_row_major(
                1.0,
                &self.qtrans_ov_102,
                n_occ,
                n_at * n_virt,
                &buf_swapped,
                n_occ,
                0.0,
                &mut result_flat,
            );

            Array2::from_shape_vec((n_occ, n_occ), result_flat).unwrap()
        }
    }
}
