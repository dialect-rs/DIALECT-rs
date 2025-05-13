use crate::defaults;
use crate::scc::mixer::Mixer;
use ndarray::*;
use ndarray_linalg::{Inverse, Norm};

/// Modified Broyden mixer
///
/// The algorithm is based on J. Chem. Phys. 152, 124101 (2020); https://doi.org/10.1063/1.5143190
#[derive(Debug, Clone)]
pub struct BroydenMixer {
    // current iteration
    iter: i32,
    maxiter: usize,
    omega0: f64,
    // mixing parameter
    alpha: f64,
    // minimal weight allowed
    min_weight: f64,
    // maximal weight allowed
    max_weight: f64,
    // numerator of the weight
    weight_factor: f64,
    weights: Array1<f64>,
    // charge difference in last iteration
    delta_q_old: Array1<f64>,
    // input charges in last iteration
    pub q_old: Array1<f64>,
    // storage for A matrix
    a_mat: Array2<f64>,
    // df vectors
    df: Array2<f64>,
    // uu vectors
    uu: Array2<f64>,
}

impl Mixer for BroydenMixer {
    fn new(dim: usize) -> BroydenMixer {
        BroydenMixer {
            iter: -1,
            maxiter: defaults::MAX_ITER,
            omega0: defaults::BROYDEN_OMEGA0,
            alpha: defaults::BROYDEN_MIXING_PARAMETER,
            min_weight: defaults::BROYDEN_MIN_WEIGHT,
            max_weight: defaults::BROYDEN_MAX_WEIGHT,
            weight_factor: defaults::BROYDEN_WEIGHT_FACTOR,
            weights: Array1::zeros([defaults::MAX_ITER - 1]),
            delta_q_old: Array1::zeros([dim]),
            q_old: Array1::zeros([dim]),
            a_mat: Array2::zeros([defaults::MAX_ITER - 1, defaults::MAX_ITER - 1]),
            df: Array2::zeros([dim, defaults::MAX_ITER - 1]),
            uu: Array2::zeros([dim, defaults::MAX_ITER - 1]),
        }
    }

    fn next(&mut self, q: Array1<f64>, delta_q: Array1<f64>) -> Array1<f64> {
        self.mix(q, delta_q)
    }

    fn reset(&mut self, dim: usize) {
        self.iter = 0;
        self.weights = Array1::zeros([self.maxiter - 1]);
        self.a_mat = Array2::zeros([self.maxiter - 1, self.maxiter - 1]);
        self.delta_q_old = Array1::zeros([dim]);
        self.q_old = Array1::zeros([dim]);
        self.a_mat = Array2::zeros([defaults::MAX_ITER - 1, defaults::MAX_ITER - 1]);
        self.df = Array2::zeros([dim, defaults::MAX_ITER - 1]);
        self.uu = Array2::zeros([dim, defaults::MAX_ITER - 1]);
    }

    /// Mixes dq from current diagonalization and the difference to the last iteration
    fn mix(&mut self, q: Array1<f64>, delta_q: Array1<f64>) -> Array1<f64> {
        let _q_in: Array1<f64> = q.clone();
        let mut q: Array1<f64> = q;

        let rel_change: f64 = delta_q.norm() / self.delta_q_old.norm();

        // it is sometimes beneficial to restart the Broyden mixer to prevent convergence problems
        if self.iter > 0 && rel_change > 1.0 {
            self.reset(q.len());
        }

        let q_out: Result<Array1<f64>, _> = match self.iter {
            -1 => {
                self.q_old = delta_q.clone();
                self.delta_q_old = delta_q.clone();
                Ok(delta_q)
            }
            // In the first iteration a linear damping scheme is used.
            // q = q + alpha * Delta q, where alpha is the Broyden mixing parameter.
            0 => {
                // The current q is stored for the next iteration.
                self.q_old = q.clone();
                // The same is done for the difference.
                self.delta_q_old = delta_q.clone();
                // Linear interpolation/damping.
                Ok(&q + &(&delta_q * self.alpha))
            }
            // For all other iterations the Broyden mixing is used.
            _ if (self.iter as usize) < self.maxiter - 1 => {
                let iter_usize: usize = self.iter as usize;
                // Index variable to access the matrix/vector element of the current iteration.
                let idx: usize = iter_usize - 1;

                // Create the weight factor of the current iteration.
                let mut weight: f64 = delta_q.dot(&delta_q).sqrt();
                if weight > self.weight_factor / self.max_weight {
                    weight = self.weight_factor / weight;
                } else {
                    weight = self.max_weight;
                }
                if weight < self.min_weight {
                    weight = self.min_weight;
                }
                // Store the current weight in the Struct.
                self.weights[idx] = weight;

                // Build |DF(idx)>.
                let mut df_idx: Array1<f64> = &delta_q - &self.delta_q_old;
                // Normalize it.
                let inv_norm: f64 = 1.0 / df_idx.dot(&df_idx).sqrt();
                df_idx = &df_idx * inv_norm;

                let mut c: Array1<f64> = Array1::zeros([iter_usize]);
                // Build a, beta, c, and gamma
                for i in 0..idx {
                    self.a_mat[[i, idx]] = self.df.slice(s![.., i]).dot(&df_idx);
                    self.a_mat[[idx, i]] = self.a_mat[[i, idx]];
                    c[i] = self.weights[i] * self.df.slice(s![.., i]).dot(&delta_q);
                }
                self.a_mat[[idx, idx]] = 1.0;
                c[idx] = self.weights[idx] * df_idx.dot(&delta_q);
                let mut beta: Array2<f64> = Array2::zeros([iter_usize, iter_usize]);
                for i in 0..iter_usize {
                    beta.slice_mut(s![i, 0..]).assign(
                        &(self.weights[i]
                            * &(&self.weights.slice(s![0..iter_usize])
                                * &self.a_mat.slice(s![0..iter_usize, i]))),
                    );
                    beta[[i, i]] += self.omega0.powi(2);
                }
                // The inverse of the matrix is computed.
                beta = beta.inv().unwrap();
                let gamma: Array1<f64> = c.dot(&beta);
                // Store |dF(m-1)>
                self.df.slice_mut(s![.., idx]).assign(&df_idx);

                // Create |u(m-1)>
                self.uu
                    .slice_mut(s![.., idx])
                    .assign(&(&(&df_idx * self.alpha) + &((&q - &self.q_old) * inv_norm)));
                // Save charge vectors before overwriting
                self.q_old = q.clone();
                self.delta_q_old = delta_q.clone();

                // Build new vector
                q = &q + &(self.alpha * &delta_q);
                for i in 0..iter_usize {
                    q -= &(&self.uu.slice(s![.., i]) * self.weights[i] * gamma[i]);
                }
                Ok(q)
            }
            _ => Err("SCC did not converge"),
        };
        self.iter += 1;
        q_out.unwrap()
    }
}
