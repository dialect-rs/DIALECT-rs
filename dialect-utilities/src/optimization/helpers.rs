use ndarray::prelude::*;
use ndarray_linalg::{into_col, into_row, Solve};
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
// References
// ----------
// [1] J. Nocedal, S. Wright, 'Numerical Optimization', Springer, 2006

// =============================================================================
// Advanced Optimization Components (v2)
// =============================================================================

// =============================================================================
// GDIIS (Geometry Direct Inversion in the Iterative Subspace)
// Reference: P. Csaszar, P. Pulay, J. Mol. Struct. 114, 31 (1984)
// =============================================================================

/// GDIIS accelerator for geometry optimization
/// Stores history of geometries and gradients, extrapolates to accelerate convergence
#[derive(Clone, Debug)]
pub struct GDIIS {
    /// Maximum number of vectors to store
    pub max_vectors: usize,
    /// History of coordinate vectors
    coords_history: Vec<Array1<f64>>,
    /// History of gradient (error) vectors
    grads_history: Vec<Array1<f64>>,
    /// Minimum number of vectors before extrapolation
    pub min_vectors: usize,
}

impl GDIIS {
    pub fn new(max_vectors: usize) -> Self {
        GDIIS {
            max_vectors,
            coords_history: Vec::with_capacity(max_vectors),
            grads_history: Vec::with_capacity(max_vectors),
            min_vectors: 3,
        }
    }

    /// Add a new geometry and gradient to the history
    pub fn add(&mut self, coords: &Array1<f64>, grad: &Array1<f64>) {
        if self.coords_history.len() >= self.max_vectors {
            // Remove oldest entry
            self.coords_history.remove(0);
            self.grads_history.remove(0);
        }
        self.coords_history.push(coords.clone());
        self.grads_history.push(grad.clone());
    }

    /// Clear the history (e.g., after a reset)
    pub fn clear(&mut self) {
        self.coords_history.clear();
        self.grads_history.clear();
    }

    /// Get the number of stored vectors
    pub fn len(&self) -> usize {
        self.coords_history.len()
    }

    /// Check if GDIIS can be applied (enough vectors)
    pub fn can_extrapolate(&self) -> bool {
        self.coords_history.len() >= self.min_vectors
    }

    /// Perform GDIIS extrapolation
    /// Returns extrapolated coordinates, or None if extrapolation fails
    pub fn extrapolate(&self) -> Option<Array1<f64>> {
        let n_vecs = self.coords_history.len();
        if n_vecs < self.min_vectors {
            return None;
        }

        // Build the error overlap matrix B_ij = e_i · e_j
        // With Lagrange multiplier constraint: sum(c_i) = 1
        // Matrix form: [B  1] [c]   [0]
        //              [1  0] [λ] = [1]
        let n = n_vecs + 1;
        let mut b_matrix: Array2<f64> = Array2::zeros((n, n));

        // Fill error overlap matrix
        for i in 0..n_vecs {
            for j in 0..n_vecs {
                b_matrix[[i, j]] = self.grads_history[i].dot(&self.grads_history[j]);
            }
        }

        // Add Lagrange multiplier row and column
        for i in 0..n_vecs {
            b_matrix[[i, n_vecs]] = 1.0;
            b_matrix[[n_vecs, i]] = 1.0;
        }
        b_matrix[[n_vecs, n_vecs]] = 0.0;

        // Right-hand side: [0, 0, ..., 0, 1]
        let mut rhs: Array1<f64> = Array1::zeros(n);
        rhs[n_vecs] = 1.0;

        // Solve for coefficients
        let coeffs = match b_matrix.solve(&rhs) {
            Ok(c) => c,
            Err(_) => return None,
        };

        // Check for reasonable coefficients (not too large)
        let max_coeff = coeffs
            .slice(s![..n_vecs])
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        if max_coeff > 10.0 {
            // Coefficients too large, extrapolation unreliable
            return None;
        }

        // Extrapolate coordinates: x_new = sum(c_i * x_i)
        let coord_len = self.coords_history[0].len();
        let mut x_extrap: Array1<f64> = Array1::zeros(coord_len);
        for i in 0..n_vecs {
            x_extrap = x_extrap + coeffs[i] * &self.coords_history[i];
        }

        Some(x_extrap)
    }
}

impl Default for GDIIS {
    fn default() -> Self {
        let mut g = GDIIS::new(6);
        g.min_vectors = 2; // Start extrapolating earlier
        g
    }
}

// =============================================================================
// Model Hessian for better initialization
// =============================================================================

/// Build a simple diagonal model Hessian based on atom types
/// Heavier atoms get smaller force constants (they move more slowly)
/// This provides better conditioning than a simple identity matrix
pub fn build_model_hessian_diagonal(atomic_numbers: &[u8]) -> Array2<f64> {
    let n_atoms = atomic_numbers.len();
    let n = n_atoms * 3;
    let mut h = Array2::zeros((n, n));

    for (i, &z) in atomic_numbers.iter().enumerate() {
        // Simple model: force constant inversely related to atomic mass
        // H: 0.5, C/N/O: 0.3, heavier: 0.2
        let k = match z {
            1 => 0.5,          // H
            6 | 7 | 8 => 0.35, // C, N, O
            _ => 0.25,         // Heavier elements
        };

        // Set diagonal elements for x, y, z of this atom
        h[[3 * i, 3 * i]] = k;
        h[[3 * i + 1, 3 * i + 1]] = k;
        h[[3 * i + 2, 3 * i + 2]] = k;
    }

    h
}

/// Build a full model Hessian in Cartesian coordinates using interatomic distances.
/// This captures the directional character of bonds: stiff along bond axes,
/// soft perpendicular to bonds. Much better conditioning than a diagonal model.
///
/// For each atom pair (i,j), the Hessian contribution from bond stretching is:
///   H_ii += k/r² · Δr⊗Δr,  H_jj += k/r² · Δr⊗Δr
///   H_ij -= k/r² · Δr⊗Δr
/// where k depends on whether the pair is bonded, close-contact, or far apart.
///
/// Reference: inspired by Schlegel's model Hessian (Theor. Chim. Acta 66, 333, 1984)
/// and Lindh's model (Chem. Phys. Lett. 241, 423, 1995).
pub fn build_model_hessian_full(coords: &Array1<f64>, atomic_numbers: &[u8]) -> Array2<f64> {
    let n_atoms = atomic_numbers.len();
    let n = n_atoms * 3;
    let mut h = Array2::zeros((n, n));

    // Covalent radii in Bohr (used for bond detection)
    let cov_radius = |z: u8| -> f64 {
        match z {
            1 => 0.60,   // H
            5 => 1.56,   // B
            6 => 1.46,   // C
            7 => 1.38,   // N
            8 => 1.29,   // O
            9 => 1.19,   // F
            14 => 2.10,  // Si
            15 => 2.02,  // P
            16 => 1.96,  // S
            17 => 1.91,  // Cl
            35 => 2.16,  // Br
            53 => 2.38,  // I
            _ => 1.50,   // default
        }
    };

    // Small diagonal shift for positive definiteness and to regularize
    // non-bonded degrees of freedom (translations, rotations of fragments)
    for i in 0..n {
        h[[i, i]] = 0.01;
    }

    // Add bond-stretching contributions for all atom pairs
    for i in 0..n_atoms {
        let xi = [coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]];
        let r_cov_i = cov_radius(atomic_numbers[i]);

        for j in (i + 1)..n_atoms {
            let xj = [coords[3 * j], coords[3 * j + 1], coords[3 * j + 2]];
            let r_cov_j = cov_radius(atomic_numbers[j]);

            // Interatomic distance
            let dx = xi[0] - xj[0];
            let dy = xi[1] - xj[1];
            let dz = xi[2] - xj[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();

            // Reference distance (sum of covalent radii)
            let r_ref = r_cov_i + r_cov_j;

            // Force constant based on distance regime
            let k = if r < r_ref + 0.6 {
                // Bonded: strong directional force constant
                // Gaussian decay for stretched bonds
                0.45 * (-1.0 * ((r / r_ref) - 1.0).powi(2)).exp()
            } else if r < 2.5 * r_ref {
                // Close non-bonded (hydrogen bonds, vdW contacts)
                0.05 * (-0.7 * ((r / r_ref) - 1.5).powi(2)).exp()
            } else if r < 4.0 * r_ref {
                // Medium range: weak coupling
                0.003 * (r_ref / r).powi(2)
            } else {
                // Far apart: negligible
                continue;
            };

            // Build 3×3 outer product block: k/r² * Δr⊗Δr
            let dr = [dx, dy, dz];
            for a in 0..3 {
                for b in 0..3 {
                    let val = k * dr[a] * dr[b] / r2;

                    // Diagonal blocks (i-i and j-j): positive contribution
                    h[[3 * i + a, 3 * i + b]] += val;
                    h[[3 * j + a, 3 * j + b]] += val;

                    // Off-diagonal blocks (i-j and j-i): negative contribution
                    h[[3 * i + a, 3 * j + b]] -= val;
                    h[[3 * j + a, 3 * i + b]] -= val;
                }
            }
        }
    }

    h
}

/// Build a full model Hessian with monomer-aware force constants for FMO.
/// Takes a monomer membership array: monomer_of[i] = monomer index for atom i.
/// Atom pairs within the same monomer get strong intramolecular force constants.
/// Atom pairs in different monomers get weaker intermolecular force constants
/// (hydrogen bond / vdW regime).
pub fn build_model_hessian_full_fmo(
    coords: &Array1<f64>,
    atomic_numbers: &[u8],
    monomer_of: &[usize],
) -> Array2<f64> {
    let n_atoms = atomic_numbers.len();
    let n = n_atoms * 3;
    let mut h = Array2::zeros((n, n));

    // Covalent radii in Bohr
    let cov_radius = |z: u8| -> f64 {
        match z {
            1 => 0.60,
            5 => 1.56,
            6 => 1.46,
            7 => 1.38,
            8 => 1.29,
            9 => 1.19,
            14 => 2.10,
            15 => 2.02,
            16 => 1.96,
            17 => 1.91,
            35 => 2.16,
            53 => 2.38,
            _ => 1.50,
        }
    };

    // Small diagonal shift for positive definiteness
    for i in 0..n {
        h[[i, i]] = 0.01;
    }

    for i in 0..n_atoms {
        let xi = [coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]];
        let r_cov_i = cov_radius(atomic_numbers[i]);
        let mon_i = monomer_of[i];

        for j in (i + 1)..n_atoms {
            let xj = [coords[3 * j], coords[3 * j + 1], coords[3 * j + 2]];
            let r_cov_j = cov_radius(atomic_numbers[j]);
            let mon_j = monomer_of[j];

            let dx = xi[0] - xj[0];
            let dy = xi[1] - xj[1];
            let dz = xi[2] - xj[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();
            let r_ref = r_cov_i + r_cov_j;

            let k = if mon_i == mon_j {
                // Same monomer: intramolecular force constants
                if r < r_ref + 0.6 {
                    // Bonded
                    0.45 * (-1.0 * ((r / r_ref) - 1.0).powi(2)).exp()
                } else if r < 2.5 * r_ref {
                    // 1-3, 1-4 interactions within monomer
                    0.15 * (-0.5 * ((r / r_ref) - 1.5).powi(2)).exp()
                } else {
                    // Distant within same monomer
                    0.01 * (r_ref / r).powi(2)
                }
            } else {
                // Different monomers: intermolecular force constants
                if r < r_ref + 1.0 {
                    // Very close intermolecular (e.g., short hydrogen bond)
                    0.08 * (-1.5 * ((r / r_ref) - 1.0).powi(2)).exp()
                } else if r < 3.0 * r_ref {
                    // Hydrogen bond / close vdW range
                    0.02 * (-0.5 * ((r / r_ref) - 2.0).powi(2)).exp()
                } else if r < 5.0 * r_ref {
                    // Medium range intermolecular
                    0.002 * (r_ref / r).powi(2)
                } else {
                    // Far: negligible
                    continue;
                }
            };

            let dr = [dx, dy, dz];
            for a in 0..3 {
                for b in 0..3 {
                    let val = k * dr[a] * dr[b] / r2;
                    h[[3 * i + a, 3 * i + b]] += val;
                    h[[3 * j + a, 3 * j + b]] += val;
                    h[[3 * i + a, 3 * j + b]] -= val;
                    h[[3 * j + a, 3 * i + b]] -= val;
                }
            }
        }
    }

    h
}

// =============================================================================
// End GDIIS and Model Hessian
// =============================================================================

// =============================================================================
// L-BFGS (Limited-memory BFGS)
// Reference: Nocedal & Wright, Algorithm 7.4
// =============================================================================

/// Limited-memory BFGS optimizer
/// Only stores the last m step/gradient difference pairs
/// Avoids accumulation of stale curvature information
pub struct LBFGS {
    /// Maximum number of stored pairs
    m: usize,
    /// Step history: s_i = x_{i+1} - x_i
    s_history: Vec<Array1<f64>>,
    /// Gradient difference history: y_i = g_{i+1} - g_i
    y_history: Vec<Array1<f64>>,
    /// Precomputed rho_i = 1 / (y_i^T * s_i)
    rho_history: Vec<f64>,
    /// Diagonal preconditioner (model Hessian diagonal values)
    h0_diag: Array1<f64>,
    /// Optional full model Hessian as preconditioner (much better conditioning
    /// than diagonal alone - captures bond directionality)
    h0_full: Option<Array2<f64>>,
}

impl LBFGS {
    /// Create new L-BFGS with m stored pairs and model Hessian diagonal
    pub fn new(m: usize, h0_diag: Array1<f64>) -> Self {
        LBFGS {
            m,
            s_history: Vec::with_capacity(m),
            y_history: Vec::with_capacity(m),
            rho_history: Vec::with_capacity(m),
            h0_diag,
            h0_full: None,
        }
    }

    /// Create L-BFGS with a full model Hessian as preconditioner.
    /// The full Hessian captures bond directionality (stiff along bonds,
    /// soft perpendicular), providing much better conditioning than diagonal.
    /// The diagonal is extracted for fallback.
    pub fn new_with_full_h0(m: usize, h0_full: Array2<f64>) -> Self {
        let n = h0_full.nrows();
        let mut h0_diag = Array1::zeros(n);
        for i in 0..n {
            h0_diag[i] = h0_full[[i, i]];
        }
        LBFGS {
            m,
            s_history: Vec::with_capacity(m),
            y_history: Vec::with_capacity(m),
            rho_history: Vec::with_capacity(m),
            h0_diag,
            h0_full: Some(h0_full),
        }
    }

    /// Add a new step/gradient-difference pair with Powell's damping.
    /// Modifies y to ensure positive curvature (s^T y_damped > 0),
    /// preventing corruption of the L-BFGS approximation from noisy gradients.
    pub fn update(&mut self, s: Array1<f64>, y: Array1<f64>) {
        let s_norm = s.dot(&s).sqrt();
        if s_norm < 1e-12 {
            return;
        }

        // Approximate B*s using preconditioner
        let bs = if let Some(ref h0) = self.h0_full {
            h0.dot(&s)
        } else {
            let mut bs = Array1::zeros(s.len());
            for i in 0..s.len() {
                bs[i] = self.h0_diag[i] * s[i];
            }
            bs
        };
        let sbs = s.dot(&bs);
        let sy = s.dot(&y);

        if sbs.abs() < 1e-14 {
            return;
        }

        // Powell's damping: ensure positive curvature
        let (y_damped, sy_damped) = if sy >= 0.2 * sbs {
            // Curvature is fine, use y as-is
            (y, sy)
        } else {
            // Damp y to guarantee positive curvature
            let theta = 0.8 * sbs / (sbs - sy);
            let r = theta * &y + (1.0 - theta) * &bs;
            let sr = s.dot(&r);
            (r, sr)
        };

        if sy_damped.abs() < 1e-14 {
            return;
        }

        if self.s_history.len() >= self.m {
            // Remove oldest
            self.s_history.remove(0);
            self.y_history.remove(0);
            self.rho_history.remove(0);
        }

        self.rho_history.push(1.0 / sy_damped);
        self.s_history.push(s);
        self.y_history.push(y_damped);
    }

    /// Compute search direction p = -H^{-1} * g using two-loop recursion
    /// Returns the search direction (already negated)
    pub fn search_direction(&self, grad: &Array1<f64>) -> Array1<f64> {
        let k = self.s_history.len();

        if k == 0 {
            // No history: use preconditioner only (H0^{-1} * (-grad))
            if let Some(ref h0) = self.h0_full {
                let neg_grad = -grad;
                if let Ok(p) = h0.solve(&neg_grad) {
                    return p;
                }
            }
            // Fallback to diagonal
            let mut p = Array1::zeros(grad.len());
            for i in 0..grad.len() {
                p[i] = -grad[i] / self.h0_diag[i];
            }
            return p;
        }

        let mut q = grad.clone();
        let mut alpha = vec![0.0; k];

        // First loop (backward)
        for i in (0..k).rev() {
            alpha[i] = self.rho_history[i] * self.s_history[i].dot(&q);
            q = q - alpha[i] * &self.y_history[i];
        }

        // Apply initial inverse Hessian H0^{-1} to q
        let mut r = if let Some(ref h0) = self.h0_full {
            // Full model Hessian: solve H0 * r = q
            // No gamma scaling - model Hessian already has proper scale
            match h0.solve(&q) {
                Ok(sol) => sol,
                Err(_) => {
                    // Fallback to diagonal
                    let mut r = Array1::zeros(grad.len());
                    for i in 0..grad.len() {
                        r[i] = q[i] / self.h0_diag[i];
                    }
                    r
                }
            }
        } else {
            // Diagonal preconditioner with gamma scaling
            let last = k - 1;
            let sy = self.s_history[last].dot(&self.y_history[last]);
            let yy = self.y_history[last].dot(&self.y_history[last]);
            let gamma = if yy > 1e-14 {
                (sy / yy).clamp(0.1, 10.0)
            } else {
                1.0
            };
            let mut r = Array1::zeros(grad.len());
            for i in 0..grad.len() {
                r[i] = gamma * q[i] / self.h0_diag[i];
            }
            r
        };

        // Second loop (forward)
        for i in 0..k {
            let beta = self.rho_history[i] * self.y_history[i].dot(&r);
            r = r + (alpha[i] - beta) * &self.s_history[i];
        }

        -r
    }

    /// Clear all history
    pub fn clear(&mut self) {
        self.s_history.clear();
        self.y_history.clear();
        self.rho_history.clear();
    }

    /// Get number of stored pairs
    pub fn len(&self) -> usize {
        self.s_history.len()
    }

    /// Update the full model Hessian preconditioner.
    /// Call this periodically as geometry changes to keep H0 accurate.
    pub fn set_h0_full(&mut self, h0_new: Array2<f64>) {
        let n = h0_new.nrows();
        for i in 0..n {
            self.h0_diag[i] = h0_new[[i, i]];
        }
        self.h0_full = Some(h0_new);
    }

    /// Compute predicted reduction for trust region quality assessment.
    /// Uses H0 as the Hessian approximation: pred = -g^T s - 0.5 s^T H0 s
    pub fn predicted_reduction(&self, grad: &Array1<f64>, step: &Array1<f64>) -> f64 {
        let bs = if let Some(ref h0) = self.h0_full {
            h0.dot(step)
        } else {
            let mut bs = Array1::zeros(step.len());
            for i in 0..step.len() {
                bs[i] = self.h0_diag[i] * step[i];
            }
            bs
        };
        -grad.dot(step) - 0.5 * step.dot(&bs)
    }
}

// =============================================================================
// End L-BFGS
// =============================================================================

/// Compute outer product of two vectors: a * b^T
fn outer(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let mut result = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

/// Damped BFGS update with Powell's modification
/// Guarantees positive definiteness of the Hessian
/// Reference: Nocedal & Wright, Algorithm 18.2
pub fn damped_bfgs_update(
    hk: ArrayView2<f64>, // Current Hessian (not inverse)
    sk: ArrayView1<f64>, // Step: s = x_new - x_old
    yk: ArrayView1<f64>, // Gradient diff: y = g_new - g_old
) -> Array2<f64> {
    // Skip update for negligible steps
    let s_norm = sk.dot(&sk).sqrt();
    if s_norm < 1e-10 {
        return hk.to_owned();
    }

    // Compute B*s
    let bs = hk.dot(&sk);
    let sbs = sk.dot(&bs);
    let sy = sk.dot(&yk);

    // Skip if sbs is too small (avoid division by zero)
    if sbs.abs() < 1e-14 {
        return hk.to_owned();
    }

    // Powell's damping: ensure positive curvature
    let theta = if sy >= 0.2 * sbs {
        1.0
    } else {
        0.8 * sbs / (sbs - sy)
    };

    // Damped y vector: r = theta*y + (1-theta)*B*s
    let r = theta * &yk + (1.0 - theta) * &bs;
    let sr = sk.dot(&r);

    // Skip if sr is too small
    if sr.abs() < 1e-14 {
        return hk.to_owned();
    }

    // BFGS update: B_new = B - (B*s*s'*B)/(s'*B*s) + (r*r')/(s'*r)
    let term1 = outer(&bs.view(), &bs.view()) / sbs;
    let term2 = outer(&r.view(), &r.view()) / sr;

    hk.to_owned() - term1 + term2
}

/// Check condition number of a matrix (approximate using diagonal ratio)
pub fn estimate_condition_number(h: &Array2<f64>) -> f64 {
    let n = h.nrows();
    let mut max_diag = 0.0_f64;
    let mut min_diag = f64::MAX;
    for i in 0..n {
        let val = h[[i, i]].abs();
        if val > max_diag {
            max_diag = val;
        }
        if val < min_diag && val > 1e-15 {
            min_diag = val;
        }
    }
    if min_diag < 1e-15 {
        return f64::MAX;
    }
    max_diag / min_diag
}

/// Trust region management with adaptive radius control.
/// Standard textbook implementation (Nocedal & Wright, Ch. 4).
#[derive(Clone, Debug)]
pub struct TrustRegion {
    pub radius: f64,         // Current trust radius (Bohr)
    pub min_radius: f64,     // Minimum radius (Bohr)
    pub max_radius: f64,     // Maximum radius (Bohr)
    initial_radius: f64,     // Initial radius for reset
}

impl TrustRegion {
    /// Create a new trust region with default parameters
    pub fn new(initial_radius: f64) -> Self {
        TrustRegion {
            radius: initial_radius,
            min_radius: 0.001,
            max_radius: 1.0,
            initial_radius,
        }
    }

    /// Update trust radius based on step quality ratio rho.
    /// rho = (actual reduction) / (predicted reduction)
    /// Standard thresholds: shrink at rho < 0.25, expand at rho > 0.75.
    pub fn update(&mut self, rho: f64, step_norm: f64) {
        if rho < 0.25 {
            // Poor agreement with model - shrink trust region
            self.radius = (0.25 * self.radius).max(self.min_radius);
        } else if rho > 0.75 && step_norm >= 0.8 * self.radius {
            // Good agreement and step at boundary - expand
            self.radius = (2.0 * self.radius).min(self.max_radius);
        }
        // Otherwise: rho in [0.25, 0.75] or step not at boundary - keep radius
    }

    /// Apply trust region constraint to a step
    /// Returns scaled step if ||p|| > radius
    pub fn apply(&self, p: &Array1<f64>) -> Array1<f64> {
        let p_norm = p.dot(p).sqrt();
        if p_norm > self.radius {
            (self.radius / p_norm) * p
        } else {
            p.clone()
        }
    }

    /// Reset trust region to initial radius
    pub fn reset(&mut self) {
        self.radius = self.initial_radius;
    }
}

impl Default for TrustRegion {
    fn default() -> Self {
        TrustRegion::new(0.3)
    }
}

/// Enhanced convergence criteria including RMS values
#[derive(Clone, Debug)]
pub struct ConvergenceCriteria {
    pub grad_max: f64, // Max gradient component (default: 4.5e-4)
    pub grad_rms: f64, // RMS gradient (default: 3.0e-4)
    pub disp_max: f64, // Max displacement component (default: 1.8e-3)
    pub disp_rms: f64, // RMS displacement (default: 1.2e-3)
    pub energy: f64,   // Energy change threshold (default: 1.0e-6)
}

impl ConvergenceCriteria {
    /// Create convergence criteria based on level string
    /// Levels: "loose", "normal", "tight", "verytight"
    pub fn from_level(level: &str) -> Self {
        match level.to_lowercase().as_str() {
            "loose" => ConvergenceCriteria {
                grad_max: 2.5e-3,
                grad_rms: 1.7e-3,
                disp_max: 1.0e-2,
                disp_rms: 6.7e-3,
                energy: 1.0e-4,
            },
            "tight" => ConvergenceCriteria {
                grad_max: 1.5e-5,
                grad_rms: 1.0e-5,
                disp_max: 6.0e-5,
                disp_rms: 4.0e-5,
                energy: 1.0e-8,
            },
            "verytight" => ConvergenceCriteria {
                grad_max: 2.0e-6,
                grad_rms: 1.0e-6,
                disp_max: 6.0e-6,
                disp_rms: 4.0e-6,
                energy: 1.0e-10,
            },
            _ => ConvergenceCriteria::default(), // "normal"
        }
    }

    /// Check if all convergence criteria are satisfied
    pub fn check(
        &self,
        grad: &Array1<f64>,
        step: &Array1<f64>,
        de: f64,
    ) -> (bool, ConvergenceStatus) {
        let n = grad.len() as f64;

        let grad_max = grad.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let grad_rms = (grad.dot(grad) / n).sqrt();
        let disp_max = step.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let disp_rms = (step.dot(step) / n).sqrt();
        let energy_change = de.abs();

        let status = ConvergenceStatus {
            grad_max,
            grad_rms,
            disp_max,
            disp_rms,
            energy_change,
            grad_max_converged: grad_max < self.grad_max,
            grad_rms_converged: grad_rms < self.grad_rms,
            disp_max_converged: disp_max < self.disp_max,
            disp_rms_converged: disp_rms < self.disp_rms,
            energy_converged: energy_change < self.energy,
        };

        let converged = status.grad_max_converged
            && status.grad_rms_converged
            && status.disp_max_converged
            && status.disp_rms_converged
            && status.energy_converged;

        (converged, status)
    }
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        // Default "normal" convergence criteria
        ConvergenceCriteria {
            grad_max: 3.0e-4,
            grad_rms: 2.0e-4,
            disp_max: 1.2e-3,
            disp_rms: 1.0e-3,
            energy: 1.0e-6,
        }
    }
}

/// Status of convergence for each criterion
#[derive(Clone, Debug)]
pub struct ConvergenceStatus {
    pub grad_max: f64,
    pub grad_rms: f64,
    pub disp_max: f64,
    pub disp_rms: f64,
    pub energy_change: f64,
    pub grad_max_converged: bool,
    pub grad_rms_converged: bool,
    pub disp_max_converged: bool,
    pub disp_rms_converged: bool,
    pub energy_converged: bool,
}

/// Solve Newton step: H * p = -g
/// Returns the step direction p
pub fn solve_newton_step(hessian: &Array2<f64>, gradient: &Array1<f64>) -> Array1<f64> {
    let neg_grad = -gradient;
    // Try to solve the linear system H * p = -g
    match hessian.solve(&neg_grad) {
        Ok(p) => p,
        Err(_) => {
            // If solve fails, fall back to steepest descent
            neg_grad
        }
    }
}

/// Wolfe line search parameters
pub struct WolfeParams {
    pub c1: f64,         // Sufficient decrease parameter (default: 1e-4)
    pub c2: f64,         // Curvature condition parameter (default: 0.9)
    pub alpha_max: f64,  // Maximum step size (default: 2.0)
    pub max_iter: usize, // Maximum iterations (default: 20)
}

impl Default for WolfeParams {
    fn default() -> Self {
        WolfeParams {
            c1: 1e-4,
            c2: 0.9,
            alpha_max: 2.0,
            max_iter: 20,
        }
    }
}

/// Cubic interpolation for line search
/// Given two points with function values and derivatives, find minimizer
pub fn cubic_interpolation(alpha_lo: f64, f_lo: f64, df_lo: f64, alpha_hi: f64, f_hi: f64) -> f64 {
    let d1 = df_lo + (f_hi - f_lo) / (alpha_hi - alpha_lo);
    let d2_sq = d1 * d1 - df_lo * (f_hi - f_lo) / (alpha_hi - alpha_lo);

    if d2_sq < 0.0 {
        // No real minimum, use bisection
        return 0.5 * (alpha_lo + alpha_hi);
    }

    let d2 = d2_sq.sqrt();
    let alpha_new = alpha_lo + (alpha_hi - alpha_lo) * (1.0 - (d1 + d2 - df_lo) / (2.0 * d2));

    // Ensure the result is within bounds
    let margin = 0.1 * (alpha_hi - alpha_lo);
    alpha_new.max(alpha_lo + margin).min(alpha_hi - margin)
}

/// Macro implementing Strong Wolfe line search with cubic interpolation
#[macro_export]
macro_rules! impl_wolfe_line_search {
    () => {
        /// Strong Wolfe line search with cubic interpolation
        /// Returns (alpha, f_new, grad_new) or None if search fails
        #[allow(unused_variables)]
        pub fn wolfe_line_search(
            &mut self,
            xk: ArrayView1<f64>,
            fk: f64,
            grad_fk: ArrayView1<f64>,
            pk: ArrayView1<f64>,
            state: usize,
            params: &$crate::optimization::helpers::WolfeParams,
        ) -> Option<(f64, f64, Array1<f64>)> {
            #[allow(unused_imports)]
            use $crate::optimization::helpers::cubic_interpolation;

            let c1 = params.c1;
            let c2 = params.c2;
            let alpha_max = params.alpha_max;
            let max_iter = params.max_iter;

            // Directional derivative at current point
            let df0 = grad_fk.dot(&pk);

            // pk must be a descent direction
            if df0 >= 0.0 {
                return None;
            }

            let mut alpha_prev = 0.0;
            let mut f_prev = fk;
            let mut alpha = 1.0; // Initial step size

            for i in 0..max_iter {
                // Evaluate function at trial point
                let x_trial = &xk + &(alpha * &pk);
                self.update_xyz(x_trial.view());
                let f_alpha = self.calculate_energy_line_search(state);

                // Check Armijo condition (sufficient decrease)
                if f_alpha > fk + c1 * alpha * df0 || (i > 0 && f_alpha >= f_prev) {
                    // Need to zoom into [alpha_prev, alpha]
                    return self.wolfe_zoom(
                        xk,
                        fk,
                        grad_fk,
                        pk,
                        state,
                        alpha_prev,
                        alpha,
                        f_prev,
                        f_alpha,
                        df0,
                        c1,
                        c2,
                        max_iter - i,
                    );
                }

                // Compute gradient at trial point
                let (_, grad_alpha) = self.opt_energy_and_gradient(state);
                let df_alpha = grad_alpha.dot(&pk);

                // Check strong Wolfe curvature condition
                if df_alpha.abs() <= -c2 * df0 {
                    // Found acceptable step
                    return Some((alpha, f_alpha, grad_alpha));
                }

                // If derivative is positive, zoom into [alpha, alpha_prev]
                if df_alpha >= 0.0 {
                    return self.wolfe_zoom(
                        xk,
                        fk,
                        grad_fk,
                        pk,
                        state,
                        alpha,
                        alpha_prev,
                        f_alpha,
                        f_prev,
                        df0,
                        c1,
                        c2,
                        max_iter - i,
                    );
                }

                // Update for next iteration
                alpha_prev = alpha;
                f_prev = f_alpha;
                alpha = (2.0 * alpha).min(alpha_max);

                if alpha >= alpha_max {
                    // Return what we have at alpha_max
                    let x_trial = &xk + &(alpha * &pk);
                    self.update_xyz(x_trial.view());
                    let (f_new, grad_new) = self.opt_energy_and_gradient(state);
                    return Some((alpha, f_new, grad_new));
                }
            }

            // Line search didn't converge, return last point
            let x_trial = &xk + &(alpha * &pk);
            self.update_xyz(x_trial.view());
            let (f_new, grad_new) = self.opt_energy_and_gradient(state);
            Some((alpha, f_new, grad_new))
        }

        /// Zoom phase of Wolfe line search using cubic interpolation
        #[allow(unused_variables)]
        fn wolfe_zoom(
            &mut self,
            xk: ArrayView1<f64>,
            fk: f64,
            grad_fk: ArrayView1<f64>,
            pk: ArrayView1<f64>,
            state: usize,
            mut alpha_lo: f64,
            mut alpha_hi: f64,
            mut f_lo: f64,
            mut f_hi: f64,
            df0: f64,
            c1: f64,
            c2: f64,
            max_iter: usize,
        ) -> Option<(f64, f64, Array1<f64>)> {
            #[allow(unused_imports)]
            use $crate::optimization::helpers::cubic_interpolation;

            // Get derivative at alpha_lo
            let x_lo = &xk + &(alpha_lo * &pk);
            self.update_xyz(x_lo.view());
            let (_, grad_lo) = self.opt_energy_and_gradient(state);
            let mut df_lo = grad_lo.dot(&pk);

            for _ in 0..max_iter {
                // Use cubic interpolation to find trial point
                let alpha_j = cubic_interpolation(alpha_lo, f_lo, df_lo, alpha_hi, f_hi);

                // Evaluate at trial point
                let x_trial = &xk + &(alpha_j * &pk);
                self.update_xyz(x_trial.view());
                let f_j = self.calculate_energy_line_search(state);

                if f_j > fk + c1 * alpha_j * df0 || f_j >= f_lo {
                    // Armijo not satisfied, shrink interval
                    alpha_hi = alpha_j;
                    f_hi = f_j;
                } else {
                    // Compute gradient
                    let (_, grad_j) = self.opt_energy_and_gradient(state);
                    let df_j = grad_j.dot(&pk);

                    // Check curvature condition
                    if df_j.abs() <= -c2 * df0 {
                        return Some((alpha_j, f_j, grad_j));
                    }

                    if df_j * (alpha_hi - alpha_lo) >= 0.0 {
                        alpha_hi = alpha_lo;
                        f_hi = f_lo;
                    }

                    alpha_lo = alpha_j;
                    f_lo = f_j;
                    df_lo = df_j;
                }

                // Check for convergence of interval
                if (alpha_hi - alpha_lo).abs() < 1e-10 {
                    let x_final = &xk + &(alpha_lo * &pk);
                    self.update_xyz(x_final.view());
                    let (f_final, grad_final) = self.opt_energy_and_gradient(state);
                    return Some((alpha_lo, f_final, grad_final));
                }
            }

            // Return best point found
            let x_final = &xk + &(alpha_lo * &pk);
            self.update_xyz(x_final.view());
            let (f_final, grad_final) = self.opt_energy_and_gradient(state);
            Some((alpha_lo, f_final, grad_final))
        }
    };
}

// =============================================================================
// End of Advanced Optimization Components (v2)
// =============================================================================

pub fn bfgs_update(
    inv_hk: ArrayView2<f64>,
    sk: ArrayView1<f64>,
    yk: ArrayView1<f64>,
    k: usize,
) -> Array2<f64> {
    // update the inverse Hessian invH_(k+1) based on Algorithm 6.1 in Ref.[1]
    let n: usize = sk.len();
    let id: Array2<f64> = Array::eye(n);

    assert!(k >= 1);
    let sy = yk.dot(&sk);
    let yy = yk.dot(&yk);

    // Safety check for division by zero
    if sy.abs() < 1e-10 {
        inv_hk.to_owned()
    } else {
        let h_curr: Array2<f64> = if k == 1 && yy > 1e-10 {
            let gamma = sy / yy;
            // Reset H to scaled identity
            gamma * &id
        } else {
            inv_hk.to_owned()
        };

        let rk: f64 = 1.0 / yk.dot(&sk);
        let u: Array2<f64> = &id - &(rk * into_col(sk).dot(&into_row(yk)));
        let v: Array2<f64> = &id - &(rk * into_col(yk).dot(&into_row(sk)));
        let w: Array2<f64> = rk * into_col(sk).dot(&into_row(sk));

        let h_new = u.dot(&h_curr.dot(&v)) + w;
        h_new
    }
}

#[macro_export]
macro_rules! impl_line_search {
    () => {
        pub fn line_search(
            &mut self,
            xk: ArrayView1<f64>,
            fk: f64,
            grad_fk: ArrayView1<f64>,
            pk: ArrayView1<f64>,
            state: usize,
        ) -> Array1<f64> {
            // set defaults
            let mut a: f64 = 1.0;
            let rho: f64 = 0.5;
            let c: f64 = 0.0001;
            let lmax: usize = 100;

            // directional derivative
            let df: f64 = grad_fk.dot(&pk);

            assert!(df <= 0.0, "pk = {} not a descent direction", &pk);
            let mut x_interp: Array1<f64> = Array::zeros(xk.len());

            for _i in 0..lmax {
                x_interp = &xk + &(a * &pk);

                // update coordinates
                self.update_xyz(x_interp.view());
                // calculate energy
                let energy: f64 = self.calculate_energy_line_search(state);

                if energy <= (fk + c * a * df) {
                    break;
                } else {
                    a *= rho;
                }
            }
            return x_interp;
        }
    };
}

#[derive(Serialize, Deserialize, Clone)]
pub struct XYZOutput {
    pub atoms: Vec<String>,
    pub coordinates: Array2<f64>,
}

impl XYZOutput {
    pub fn new(atoms: Vec<String>, coordinates: Array2<f64>) -> XYZOutput {
        XYZOutput { atoms, coordinates }
    }
}

pub fn write_xyz_wigner(xyz: &XYZOutput, filename: String) {
    let file_path: &Path = Path::new(&filename);
    let n_atoms: usize = xyz.atoms.len();
    let mut string: String = n_atoms.to_string();
    string.push('\n');
    string.push('\n');
    for atom in 0..n_atoms {
        let str: String = xyz.atoms[atom].to_string();
        string.push_str(&str);
        string.push('\t');
        for item in 0..3 {
            let str: String = xyz.coordinates.slice(s![atom, item]).to_string();
            string.push_str(&str);
            string.push('\t');
        }
        string.push('\n');
    }

    if file_path.exists() {
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(file_path)
            .unwrap();
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", string)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, string).expect("Unable to write to geom.xyz for wigner sampling file");
    }
}

pub fn write_xyz_custom(xyz: &XYZOutput, first_call: bool) {
    let file_path: &Path = Path::new("optimization.xyz");
    let n_atoms: usize = xyz.atoms.len();
    let mut string: String = n_atoms.to_string();
    string.push('\n');
    string.push('\n');
    for atom in 0..n_atoms {
        let str: String = xyz.atoms[atom].to_string();
        string.push_str(&str);
        string.push('\t');
        for item in 0..3 {
            let str: String = xyz.coordinates.slice(s![atom, item]).to_string();
            string.push_str(&str);
            string.push('\t');
        }
        string.push('\n');
    }

    if file_path.exists() {
        let file = if first_call {
            OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(file_path)
                .unwrap()
        } else {
            OpenOptions::new().append(true).open(file_path).unwrap()
        };
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", string)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, string).expect("Unable to write to optimization.xyz file");
    }
}

pub fn write_last_geom(xyz: &XYZOutput) {
    let file_path: &Path = Path::new("opt_geom.xyz");
    let n_atoms: usize = xyz.atoms.len();
    let mut string: String = n_atoms.to_string();
    string.push('\n');
    string.push('\n');
    for atom in 0..n_atoms {
        let str: String = xyz.atoms[atom].to_string();
        string.push_str(&str);
        string.push('\t');
        for item in 0..3 {
            let str: String = xyz.coordinates.slice(s![atom, item]).to_string();
            string.push_str(&str);
            string.push('\t');
        }
        string.push('\n');
    }

    if file_path.exists() {
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(file_path)
            .unwrap();
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", string)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, string).expect("Unable to write to opt_geom.xyz file");
    }
}

pub fn write_error_geom(xyz: &XYZOutput) {
    let file_path: &Path = Path::new("error_geom.xyz");
    let n_atoms: usize = xyz.atoms.len();
    let mut string: String = n_atoms.to_string();
    string.push('\n');
    string.push('\n');
    for atom in 0..n_atoms {
        let str: String = xyz.atoms[atom].to_string();
        string.push_str(&str);
        string.push('\t');
        for item in 0..3 {
            let str: String = xyz.coordinates.slice(s![atom, item]).to_string();
            string.push_str(&str);
            string.push('\t');
        }
        string.push('\n');
    }

    if file_path.exists() {
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(file_path)
            .unwrap();
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", string)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, string).expect("Unable to write to opt_geom.xyz file");
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct OptEnergyOutput {
    pub step: usize,
    pub energy: f64,
}

impl OptEnergyOutput {
    pub fn new(step: usize, energy: f64) -> OptEnergyOutput {
        OptEnergyOutput { step, energy }
    }
}

pub fn write_opt_energy(energy_out: &OptEnergyOutput, first_call: bool) {
    let file_path: &Path = Path::new("opt_energies.txt");
    let mut string: String = energy_out.step.to_string();
    string.push('\t');
    string.push_str(&energy_out.energy.to_string());
    string.push('\n');

    if file_path.exists() {
        let file = if first_call {
            OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(file_path)
                .unwrap()
        } else {
            OpenOptions::new().append(true).open(file_path).unwrap()
        };
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", string)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, string).expect("Unable to write to dynamics.xyz file");
    }
}
