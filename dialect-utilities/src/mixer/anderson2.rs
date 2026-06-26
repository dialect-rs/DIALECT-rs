use ndarray::prelude::*;
use ndarray_linalg::{Norm, Solve};

/// Anderson mixer following Eyert's formulation
/// Reference: J. Comput. Phys. 124, 271 (1996)
#[derive(Clone, Debug)]
pub struct AndersonMixer {
    /// Number of generations (history vectors) to keep
    memory: usize,
    /// Current iteration (1-indexed after first mix)
    iter: usize,
    /// Mixing parameter β
    beta: f64,
    /// Initial mixing parameter (for first iterations)
    beta_init: f64,
    /// Diagonal offset to prevent linear dependence
    diagonal_offset: f64,
    /// Use soft start (simple mixing for first `memory` iterations)
    soft_start: bool,
    /// History of input vectors: x_hist[0] is most recent
    x_hist: Array2<f64>,
    /// History of residuals F = x_out - x_in: f_hist[0] is most recent
    f_hist: Array2<f64>,
    /// Dimension of vectors
    ndim: usize,
}

impl AndersonMixer {
    pub fn new(ndim: usize) -> Self {
        let memory: usize = 20;
        Self {
            memory,
            iter: 0,
            beta: 0.2,
            beta_init: 0.01,
            diagonal_offset: 0.01,
            soft_start: false,
            // Store memory+1 vectors (current + history)
            x_hist: Array2::zeros((memory + 1, ndim)),
            f_hist: Array2::zeros((memory + 1, ndim)),
            ndim,
        }
    }

    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    pub fn with_beta_init(mut self, beta_init: f64) -> Self {
        self.beta_init = beta_init;
        self
    }

    pub fn with_diagonal_offset(mut self, offset: f64) -> Self {
        self.diagonal_offset = offset;
        self
    }

    pub fn with_soft_start(mut self, soft_start: bool) -> Self {
        self.soft_start = soft_start;
        self
    }

    pub fn reset(&mut self) {
        self.iter = 0;
        self.x_hist.fill(0.0);
        self.f_hist.fill(0.0);
    }

    /// Get the residual norm from the last iteration
    pub fn get_error(&self) -> f64 {
        if self.iter == 0 {
            return f64::INFINITY;
        }
        self.f_hist.row(0).norm_l2()
    }

    /// Perform Anderson mixing
    ///
    /// # Arguments
    /// * `x_in` - Input vector (e.g., input charges)
    /// * `x_out` - Output vector from the current iteration (e.g., output charges)
    ///
    /// # Returns
    /// Mixed vector to use as input for next iteration
    pub fn mix(&mut self, x_in: &Array1<f64>, x_out: &Array1<f64>) -> Array1<f64> {
        // Compute residual F = x_out - x_in
        let f: Array1<f64> = x_out - x_in;

        // Shift history arrays: roll everything down by one
        // After roll: index 0 is free for new data, index 1 has previous data, etc.
        self.roll_history();

        // Store current values at index 0
        self.x_hist.row_mut(0).assign(x_in);
        self.f_hist.row_mut(0).assign(&f);

        self.iter += 1;

        // Determine if we should use Anderson or simple mixing
        let use_anderson = if self.soft_start {
            self.iter > self.memory
        } else {
            self.iter > 1
        };

        if use_anderson {
            self.anderson_step()
        } else {
            // Simple mixing: x_new = x_in + beta_init * F
            x_in + &(&f * self.beta_init)
        }
    }

    /// Perform the Anderson mixing step
    fn anderson_step(&self) -> Array1<f64> {
        // Number of history vectors to use (excluding current)
        let n = (self.iter - 1).min(self.memory);

        // Build dF matrix: dF[j] = F[0] - F[j+1] for j = 0..n
        // Each row is a difference vector
        let mut df = Array2::<f64>::zeros((n, self.ndim));
        for j in 0..n {
            let diff = &self.f_hist.row(0) - &self.f_hist.row(j + 1);
            df.row_mut(j).assign(&diff);
        }

        // Build overlap matrix A: A[i,j] = <dF[i] | dF[j]>
        // and right-hand side b: b[i] = <dF[i] | F[0]>
        let mut a = Array2::<f64>::zeros((n, n));
        let mut b = Array1::<f64>::zeros(n);

        let f0 = self.f_hist.row(0);

        for i in 0..n {
            let df_i = df.row(i);
            b[i] = df_i.dot(&f0);
            for j in 0..=i {
                let df_j = df.row(j);
                let aij = df_i.dot(&df_j);
                a[[i, j]] = aij;
                a[[j, i]] = aij; // Symmetric
            }
        }

        // Regularize diagonal to prevent linear dependence (Eyert eq. 8.2)
        // A[i,i] *= (1 + diagonal_offset^2)
        if self.diagonal_offset > 0.0 {
            let factor = 1.0 + self.diagonal_offset.powi(2);
            for i in 0..n {
                a[[i, i]] *= factor;
            }
        }

        // Solve A * theta = b for coefficients
        let theta = match a.solve(&b) {
            Ok(t) => t,
            Err(_) => {
                // Fallback to simple mixing if solve fails
                eprintln!("Anderson: linear solve failed, using simple mixing");
                let f0 = self.f_hist.row(0);
                let x0 = self.x_hist.row(0);
                return x0.to_owned() + &(&f0 * self.beta);
            }
        };

        // Build averaged vectors (Eyert eq. 4.1, 4.2)
        // x_bar = x[0] + sum_j theta[j] * (x[j+1] - x[0])
        // f_bar = F[0] + sum_j theta[j] * (F[j+1] - F[0])
        //       = F[0] - sum_j theta[j] * dF[j]
        let x0 = self.x_hist.row(0);
        let f0 = self.f_hist.row(0);

        let mut x_bar = x0.to_owned();
        let mut f_bar = f0.to_owned();

        for j in 0..n {
            let dx = &self.x_hist.row(j + 1) - &x0;
            x_bar = &x_bar + &(&dx * theta[j]);
            // Note: dF[j] = F[0] - F[j+1], so F[j+1] - F[0] = -dF[j]
            f_bar = &f_bar - &(&df.row(j) * theta[j]);
        }

        // Final mixed vector (Eyert eq. 4.4)
        // x_new = x_bar + beta * f_bar
        &x_bar + &(&f_bar * self.beta)
    }

    /// Roll history arrays: shift all rows down by 1
    fn roll_history(&mut self) {
        // Shift rows: row[i] <- row[i-1] for i = memory..1
        for i in (1..=self.memory).rev() {
            let (head, mut tail) = self.x_hist.view_mut().split_at(Axis(0), i);
            tail.row_mut(0).assign(&head.row(i - 1));

            let (head, mut tail) = self.f_hist.view_mut().split_at(Axis(0), i);
            tail.row_mut(0).assign(&head.row(i - 1));
        }
    }
}
