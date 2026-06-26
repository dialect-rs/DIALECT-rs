use dialect_base::defaults;
use crate::mixer::Mixer;
use ndarray::*;
use ndarray_linalg::Solve;
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

#[derive(Debug, Clone)]
pub struct BroydenMixerNew {
    iter: usize,
    memory: usize,
    alpha: f64,
    omega0: f64,
    min_weight: f64,
    max_weight: f64,
    weight_factor: f64,
    // Safeguard parameters
    safeguard_factor: f64,
    max_gamma_norm: f64,
    dq_norm_last: f64,
    // Storage (using cyclic indexing)
    omega: Array1<f64>,
    df: Array2<f64>,
    u: Array2<f64>,
    a: Array2<f64>,
    dq_last: Array1<f64>,
    q_last: Array1<f64>,
}

impl BroydenMixerNew {
    pub fn new(ndim: usize) -> Self {
        let memory: usize = 20;
        BroydenMixerNew {
            iter: 0,
            memory: memory,
            alpha: 0.40,  // default mixing parameter
            omega0: 0.01, // default
            min_weight: 1.0,
            max_weight: 100000.0,
            weight_factor: 0.01,
            safeguard_factor: 0.0,   // disabled by default
            max_gamma_norm: 1e6,     // Maximum allowed gamma norm before fallback
            dq_norm_last: f64::MAX,
            omega: Array1::zeros(memory),
            df: Array2::zeros((ndim, memory)),
            u: Array2::zeros((ndim, memory)),
            a: Array2::zeros((memory, memory)),
            dq_last: Array1::zeros(ndim),
            q_last: Array1::zeros(ndim),
        }
    }

    /// Create a new Broyden mixer with parameters from the configuration
    pub fn from_config(ndim: usize, config: &dialect_config::settings::BroydenConfig) -> Self {
        let memory = config.memory;
        BroydenMixerNew {
            iter: 0,
            memory,
            alpha: config.alpha,
            omega0: config.omega0,
            min_weight: 1.0,
            max_weight: 100000.0,
            weight_factor: 0.01,
            safeguard_factor: config.safeguard_factor,
            max_gamma_norm: 1e6,
            dq_norm_last: f64::MAX,
            omega: Array1::zeros(memory),
            df: Array2::zeros((ndim, memory)),
            u: Array2::zeros((ndim, memory)),
            a: Array2::zeros((memory, memory)),
            dq_last: Array1::zeros(ndim),
            q_last: Array1::zeros(ndim),
        }
    }

    pub fn new_fmo_mixer(ndim: usize) -> Self {
        let memory: usize = 20;
        BroydenMixerNew {
            iter: 0,
            memory: memory,
            alpha: 0.25,  // Lower mixing for FMO stability
            omega0: 0.01, // default
            min_weight: 1.0,
            max_weight: 100000.0,
            weight_factor: 0.01,
            safeguard_factor: 1.5,   // Reset if residual increases by more than 50%
            max_gamma_norm: 1e6,     // Maximum allowed gamma norm before fallback
            dq_norm_last: f64::MAX,
            omega: Array1::zeros(memory),
            df: Array2::zeros((ndim, memory)),
            u: Array2::zeros((ndim, memory)),
            a: Array2::zeros((memory, memory)),
            dq_last: Array1::zeros(ndim),
            q_last: Array1::zeros(ndim),
        }
    }

    pub fn reset(&mut self) {
        self.iter = 0;
        self.dq_norm_last = f64::MAX;
        self.omega.fill(0.0);
        self.df.fill(0.0);
        self.u.fill(0.0);
        self.a.fill(0.0);
        self.dq_last.fill(0.0);
        self.q_last.fill(0.0);
    }

    /// Set the mixing parameter alpha (used for gap-dependent damping)
    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    /// Get the current mixing parameter alpha
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the omega (weight) from the last iteration
    /// Returns 0.0 for iteration 1 (simple mixing, no omega computed)
    pub fn last_omega(&self) -> f64 {
        if self.iter <= 1 {
            0.0
        } else {
            let it1 = (self.iter - 2) % self.memory;
            self.omega[it1]
        }
    }

    /// Simple linear mixing fallback
    fn simple_mix(&mut self, q: &Array1<f64>, dq: &Array1<f64>) -> Array1<f64> {
        self.dq_last.assign(dq);
        self.q_last.assign(q);
        self.dq_norm_last = dq.dot(dq).sqrt();
        q + &(self.alpha * dq)
    }

    /// Mix charges: q_in is current input charges, dq is (q_out - q_in) from this iteration
    pub fn next(&mut self, q: &Array1<f64>, dq: &Array1<f64>) -> Array1<f64> {
        self.iter += 1;

        // Compute current residual norm
        let dq_norm = dq.dot(dq).sqrt();

        // First iteration: simple linear mixing
        if self.iter == 1 {
            self.dq_last.assign(dq);
            self.q_last.assign(q);
            self.dq_norm_last = dq_norm;
            return q + &(self.alpha * dq);
        }

        // Safeguard: reset Broyden if residual increases beyond safeguard_factor.
        // Set safeguard_factor to 0.0 to disable (default).
        // For difficult molecules, enable with e.g. safeguard_factor = 1.5.
        if self.safeguard_factor > 0.0
            && self.iter > 2
            && dq_norm > self.safeguard_factor * self.dq_norm_last
        {
            self.reset();
            self.iter = 1;
            return self.simple_mix(q, dq);
        }

        let itn = self.iter - 1; // Number of previous Broyden steps (0-indexed history count)
        let it1 = (itn - 1) % self.memory; // Cyclic index for current storage slot
        let mut weight = if dq_norm > self.weight_factor / self.max_weight {
            self.weight_factor / dq_norm
        } else {
            self.max_weight
        };
        weight = weight.max(self.min_weight);
        self.omega[it1] = weight;

        // Build |dF(it1)> = (dq - dq_last) / ||dq - dq_last||
        let mut df_new: Array1<f64> = dq - &self.dq_last;
        let df_norm = df_new.dot(&df_new).sqrt().max(f64::EPSILON);
        let inv_norm = 1.0 / df_norm;
        df_new *= inv_norm;

        // Number of history vectors to use
        let n_hist = itn.min(self.memory);

        // Build A matrix elements and c vector
        let mut c: Array1<f64> = Array1::zeros(n_hist);

        for jj in 0..n_hist {
            // Map to cyclic index
            let j = if itn <= self.memory {
                jj
            } else {
                (itn - self.memory + jj) % self.memory
            };

            // A matrix: <df_j | df_it1>
            let df_j = self.df.column(j);
            if jj < n_hist - 1 || itn == 1 {
                // For previous iterations, use stored df
                self.a[[j, it1]] = df_j.dot(&df_new);
                self.a[[it1, j]] = self.a[[j, it1]];
            }

            // c vector: omega_j * <df_j | dq>
            if j == it1 {
                c[jj] = self.omega[j] * df_new.dot(dq);
            } else {
                c[jj] = self.omega[j] * df_j.dot(dq);
            }
        }
        self.a[[it1, it1]] = 1.0;

        // Build beta matrix: beta_ij = omega_i * omega_j * a_ij + omega0^2 * delta_ij
        let mut beta: Array2<f64> = Array2::zeros((n_hist, n_hist));
        for ii in 0..n_hist {
            let i = if itn <= self.memory {
                ii
            } else {
                (itn - self.memory + ii) % self.memory
            };
            for jj in 0..n_hist {
                let j = if itn <= self.memory {
                    jj
                } else {
                    (itn - self.memory + jj) % self.memory
                };
                beta[[ii, jj]] = self.omega[i] * self.omega[j] * self.a[[i, j]];
            }
            beta[[ii, ii]] += self.omega0.powi(2);
        }

        // Solve beta * gamma = c
        let gamma = match beta.solve(&c) {
            Ok(g) => g,
            Err(_) => {
                // Fallback to simple mixing if solve fails
                self.reset();
                self.iter = 1;
                return self.simple_mix(q, dq);
            }
        };

        // Divergence handling: if gamma norm is too large, reset and use simple mixing
        let gamma_norm = gamma.dot(&gamma).sqrt();
        if gamma_norm > self.max_gamma_norm {
            self.reset();
            self.iter = 1;
            return self.simple_mix(q, dq);
        }

        // Store |df(it1)>
        self.df.column_mut(it1).assign(&df_new);

        // Store |u(it1)> = alpha * df + inv_norm * (q - q_last)
        let u_new: Array1<f64> = &(&df_new * self.alpha) + &(&(q - &self.q_last) * inv_norm);
        self.u.column_mut(it1).assign(&u_new);

        // Save for next iteration
        self.dq_last.assign(dq);
        self.q_last.assign(q);
        self.dq_norm_last = dq_norm;

        // Compute new charges: q_new = q + alpha * dq - sum_j omega_j * gamma_j * u_j
        let mut q_new = q + &(self.alpha * dq);
        for jj in 0..n_hist {
            let j = if itn <= self.memory {
                jj
            } else {
                (itn - self.memory + jj) % self.memory
            };
            let correction = &self.u.column(j) * (self.omega[j] * gamma[jj]);
            q_new = &q_new - &correction;
        }

        q_new
    }
}
