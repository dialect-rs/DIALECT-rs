//! Löwdin decomposition and the analytic derivative of the determinant
//! overlap (sigma) — shared by the NDDO and OMx NACV implementations.

use crate::linalg::eigh::dsyevd_eigh_vec as dsyevd_eigh;
use ndarray::{Array1, Array2, ArrayView2};

pub fn lowdin_decompose(s: ArrayView2<f64>) -> (Array2<f64>, Array2<f64>, Vec<f64>, Array2<f64>) {
    let (eigvals, u) = dsyevd_eigh(s);
    let n = eigvals.len();
    let mut s_minus_half = Array2::<f64>::zeros((n, n));
    let mut s_plus_half = Array2::<f64>::zeros((n, n));
    for p in 0..n {
        let lam = eigvals[p];
        assert!(
            lam > 1e-12,
            "lowdin_decompose: overlap eigenvalue {} too small ({:.3e}) — linearly dependent basis?",
            p, lam
        );
        let sqrt_lam = lam.sqrt();
        let inv_sqrt_lam = 1.0 / sqrt_lam;
        for mu in 0..n {
            for nu in 0..n {
                s_minus_half[[mu, nu]] += u[[mu, p]] * inv_sqrt_lam * u[[nu, p]];
                s_plus_half[[mu, nu]] += u[[mu, p]] * sqrt_lam * u[[nu, p]];
            }
        }
    }
    (s_minus_half, s_plus_half, eigvals, u)
}

/// **Analytic** derivative `DSMI = ∂(S^(-1/2))/∂R` given the
/// pre-computed Löwdin eigendecomposition and the overlap derivative
/// `dS/dR` at the same geometry.
///
/// Uses the Lyapunov-equation identity:
/// ```text
///   X · S^(1/2) + S^(1/2) · X = Y    where Y = −S^(-1/2) · dS · S^(-1/2)
///   X_eig[p,q] = Y_eig[p,q] / (√λ_p + √λ_q)
///   DSMI = U · X_eig · U^T
/// ```
pub fn dsmi_analytic(
    s_minus_half: ArrayView2<f64>,
    eigvals: &[f64],
    eigvecs: ArrayView2<f64>,
    ds_dr: ArrayView2<f64>,
) -> Array2<f64> {
    let n = eigvals.len();
    // Y_AO = −S^(-1/2) · dS · S^(-1/2)  (symmetric).
    let temp = s_minus_half.dot(&ds_dr);
    let mut y_ao = temp.dot(&s_minus_half);
    y_ao.mapv_inplace(|v| -v);
    // Y_eig = U^T · Y_AO · U.
    let y_eig = eigvecs.t().dot(&y_ao).dot(&eigvecs);
    // X_eig[p,q] = Y_eig[p,q] / (√λ_p + √λ_q).
    let mut x_eig = Array2::<f64>::zeros((n, n));
    let sqrt_lams: Vec<f64> = eigvals.iter().map(|&v| v.sqrt()).collect();
    for p in 0..n {
        for q in 0..n {
            x_eig[[p, q]] = y_eig[[p, q]] / (sqrt_lams[p] + sqrt_lams[q]);
        }
    }
    // DSMI = U · X_eig · U^T.
    eigvecs.dot(&x_eig).dot(&eigvecs.t())
}
