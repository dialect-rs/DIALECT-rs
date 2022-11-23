use crate::defaults;
use ndarray::*;
use ndarray_linalg::{Norm, Solve};
use ndarray_stats::QuantileExt;
use std::cmp::min;
use std::iter::FromIterator;

pub struct Pulay80 {
    trial_vectors: Vec<Array2<f64>>,
    residual_vectors: Vec<Array1<f64>>,
    memory: usize,
    iter: usize,
    start: bool,
}

impl Pulay80 {
    pub fn new() -> Pulay80 {
        let t_v: Vec<Array2<f64>> = Vec::new();
        let r_v: Vec<Array1<f64>> = Vec::new();
        return Pulay80 {
            trial_vectors: t_v,
            residual_vectors: r_v,
            memory: defaults::DIIS_LIMIT,
            iter: 0,
            start: false,
        };
    }

    pub fn reset(&mut self) {
        self.residual_vectors = Vec::new();
        self.trial_vectors = Vec::new();
        self.iter = 0;
        self.start = false;
    }

    fn add_trial_vector(&mut self, p: Array2<f64>) {
        // add the vector to the set of iterative solutions
        self.trial_vectors.push(p);
        if self.trial_vectors.len() >= 2 {
            // guess error vectors from change
            // with respect to previous iteration
            self.residual_vectors.push(Array1::from_iter(
                (&self.trial_vectors[self.trial_vectors.len() - 1]
                    - &self.trial_vectors[self.trial_vectors.len() - 2])
                    .iter()
                    .cloned(),
            ))
        }
        if self.trial_vectors.len() > self.memory {
            self.trial_vectors.remove(0);
            self.residual_vectors.remove(0);
        }
    }

    fn get_approximation(&mut self) -> Array2<f64> {
        // determine the best linear combination of the previous
        // solution vectors.
        let diis_count: usize = self.residual_vectors.len();
        assert!(
            diis_count > 1,
            "There should be at least 2 residual vectors"
        );
        // build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
        let mut b: Array2<f64> = Array2::zeros((diis_count + 1, diis_count + 1));
        for (idx1, e1) in self.residual_vectors.iter().enumerate() {
            for (idx2, e2) in self.residual_vectors.iter().enumerate() {
                if idx2 <= idx1 {
                    let val: f64 = e1.dot(e2);
                    b[[idx1, idx2]] = val;
                    b[[idx2, idx1]] = val;
                }
            }
        }
        b.slice_mut(s![diis_count, ..]).fill(-1.0);
        b.slice_mut(s![.., diis_count]).fill(-1.0);
        b[[diis_count, diis_count]] = 0.0;

        // normalize
        // calculate the maximal element of the array slice
        let max: f64 = *b
            .slice(s![0..diis_count, 0..diis_count])
            .map(|x| x.abs())
            .max()
            .unwrap();
        b.slice_mut(s![0..diis_count, 0..diis_count])
            .map(|x| x / max);

        // build residual vector, [Pulay:1980:393], Eqn. 6, RHS
        let mut resid: Array1<f64> = Array1::zeros((diis_count + 1));
        resid[diis_count] = -1.0;

        // Solve Pulay equations, [Pulay:1980:393], Eqn. 6
        let ci: Array1<f64> = b.solve_into(resid).unwrap();

        // calculate new density matrix as linear combination of previous density matrices
        let mut p_next: Array2<f64> = Array2::zeros(self.trial_vectors[0].raw_dim());
        for (idx, coeff) in ci.slice(s![0..diis_count]).iter().enumerate() {
            p_next += &self.trial_vectors[idx].map(|x| x * *coeff);
        }
        let t_len: usize = self.trial_vectors.len();
        self.trial_vectors[t_len - 1] = p_next.clone();
        self.residual_vectors[diis_count - 1] = Array1::from_iter(
            (&self.trial_vectors[t_len - 1] - &self.trial_vectors[t_len - 2])
                .iter()
                .cloned(),
        );
        return p_next;
    }

    // add the trial vector p and replace it with the DIIS approximate
    pub fn next(&mut self, p: Array2<f64>) -> Array2<f64> {
        self.add_trial_vector(p.clone());
        let p_next: Array2<f64>;
        if self.start {
            p_next = self.get_approximation()
        } else {
            if self.trial_vectors.len() > 3 {
                if self.relative_change() < 0.5 {
                    self.start = true;
                }
            }
            p_next = p;
        }
        return p_next;
    }

    // compute the relative change in the last iteration
    // |p_(i+1) - p_i|/|p_i|
    pub fn relative_change(&self) -> f64 {
        let mut change: f64;
        if self.residual_vectors.len() == 0 {
            change = 0.0;
        } else {
            change = 0.0;
            let navg: usize = min(self.residual_vectors.len(), 2);
            for i in 1..navg + 1 {
                change += self.residual_vectors[self.residual_vectors.len() - i].norm()
                    / self.trial_vectors[self.trial_vectors.len() - i].norm()
            }
            change /= navg as f64;
        }
        return change;
    }
}

pub struct Pulay82 {
    trial_vectors: Vec<Array2<f64>>,
    error_vectors: Vec<Array1<f64>>,
    memory: usize,
    iter: usize,
    start: bool,
}

impl Pulay82 {
    pub fn new() -> Pulay82 {
        let t_v: Vec<Array2<f64>> = Vec::new();
        let r_v: Vec<Array1<f64>> = Vec::new();
        return Pulay82 {
            trial_vectors: t_v,
            error_vectors: r_v,
            memory: defaults::DIIS_LIMIT,
            iter: 0,
            start: false,
        };
    }

    pub fn reset(&mut self) {
        self.error_vectors = Vec::new();
        self.trial_vectors = Vec::new();
        self.iter = 0;
        self.start = false;
    }

    pub fn add_error_vector(&mut self, err: Array1<f64>) {
        self.error_vectors.push(err);
        if self.trial_vectors.len() > self.memory {
            self.trial_vectors.remove(0);
            self.error_vectors.remove(0);
        }
    }

    fn get_approximation(&mut self) -> Array2<f64> {
        // determine the best linear combination of the previous
        // solution vectors.
        let diis_count: usize = self.error_vectors.len();
        assert!(diis_count > 1, "There should be at least 2 error vectors");
        // build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
        let mut b: Array2<f64> = Array2::zeros((diis_count + 1, diis_count + 1));
        for (idx1, e1) in self.error_vectors[0..diis_count].iter().enumerate() {
            for (idx2, e2) in self.error_vectors[0..diis_count].iter().enumerate() {
                if idx2 <= idx1 {
                    let val: f64 = e1.dot(e2);
                    b[[idx1, idx2]] = val;
                    b[[idx2, idx1]] = val;
                }
            }
        }
        b.slice_mut(s![diis_count, ..]).fill(-1.0);
        b.slice_mut(s![.., diis_count]).fill(-1.0);
        b[[diis_count, diis_count]] = 0.0;

        // normalize
        // calculate the maximal element of the array slice
        // let max: f64 = *b
        //     .slice(s![0..diis_count, 0..diis_count])
        //     .map(|x| x.abs())
        //     .max()
        //     .unwrap();
        // b.slice_mut(s![0..diis_count, 0..diis_count])
        //     .map(|x| x / max);
        // build residual vector, [Pulay:1980:393], Eqn. 6, RHS
        let mut resid: Array1<f64> = Array1::zeros((diis_count + 1));
        resid[diis_count] = -1.0;

        // Solve Pulay equations, [Pulay:1980:393], Eqn. 6
        let ci: Array1<f64> = b.solve_into(resid).unwrap();
        println!("linear comb {}", ci);

        // calculate new density matrix as linear combination of previous density matrices
        let mut h_next: Array2<f64> = Array2::zeros(self.trial_vectors[0].raw_dim());
        for (idx, coeff) in ci.slice(s![0..diis_count]).iter().enumerate() {
            h_next += &self.trial_vectors[idx].map(|x| x * *coeff);
        }
        return h_next;
    }

    // add the trial vector p and replace it with the DIIS approximate
    pub fn next(&mut self, h: Array2<f64>) -> Array2<f64> {
        println!("NEXT {}", self.start);
        let h_next: Array2<f64>;
        if self.start == false {
            if self.trial_vectors.len() > 2 {
                println!("relative change {}", self.relative_change());
                if self.relative_change() < 0.1 {
                    println!("START DIIS");
                    self.start = true;
                }
            }
        }
        if self.start {
            h_next = self.get_approximation()
        } else {
            h_next = h.clone();
        }
        self.trial_vectors.push(h);
        return h_next;
    }

    // compute the relative change in the last iteration
    pub fn relative_change(&self) -> f64 {
        let mut change: f64;
        if self.trial_vectors.len() < 2 {
            change = 1.0;
        } else {
            let t_len: usize = self.trial_vectors.len();
            let diff: f64 =
                (&self.trial_vectors[t_len - 1] - &self.trial_vectors[t_len - 2]).norm();
            change = diff / self.trial_vectors[t_len - 1].norm();
        }
        println!("CHANGE {}", change);
        return change;
    }
}
