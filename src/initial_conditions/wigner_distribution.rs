use crate::constants::{ATOMIC_MASSES, HARTREE_TO_WAVENUMBERS};
use crate::initialization::System;
use libm::tanh;
use ndarray::prelude::*;
use ndarray_linalg::{c64, into_col, into_row, Eigh, Inverse, SVD, UPLO};
use ndarray_rand::rand_distr::{Distribution, Normal};
use ndarray_stats::QuantileExt;

pub struct WignerEnsemble<'a> {
    pub temperature: f64,
    pub nsample: usize,
    pub freqs: Array1<f64>,
    pub masses: ArrayView1<'a, f64>,
    pub coords: ArrayView1<'a, f64>,
    pub modes: ArrayView2<'a, f64>,
    pub n_atoms: usize,
    pub dim: usize,
    zerofreq: usize,
}

impl WignerEnsemble<'_> {
    pub fn new<'a>(
        system: &System,
        freqs: Array1<f64>,
        modes: ArrayView2<'a, f64>,
        nsample: usize,
        masses: ArrayView1<'a, f64>,
        coords: ArrayView1<'a, f64>,
    ) -> WignerEnsemble<'a> {
        let dim: usize = freqs.len();
        let wigner_config = &system.config.wigner_config;

        WignerEnsemble {
            temperature: wigner_config.temperature,
            nsample: wigner_config.n_samples,
            freqs,
            masses,
            coords,
            modes,
            dim,
            n_atoms: system.n_atoms,
            zerofreq: 6,
        }
    }

    pub fn cut_frequencies(&mut self) {
        // reshape the coordinates
        let coords_2d: Array2<f64> = self
            .coords
            .clone()
            .to_owned()
            .into_shape([self.n_atoms, 3])
            .unwrap();
        let c0 = coords_2d.mean_axis(Axis(0)).unwrap();
        // get the difference
        let diff: Array2<f64> = coords_2d - c0;
        // get the svd
        let tmp = diff.svd(false, false).unwrap();
        let s: Array1<f64> = tmp.1;

        let mut cut_number: usize = 6;
        // check for linearity
        if s[1] / s[0] < 0.01 {
            cut_number = 5;
            self.zerofreq = 5;
        }

        self.freqs
            .slice_mut(s![..cut_number])
            .assign(&Array1::zeros(cut_number));
        // give warning if frequency is below zero
        let min_freq: f64 = *self.freqs.min().unwrap();
        if min_freq < 0.0 {
            panic!("A frequency below zero exists! Cannot calculate the square root!");
        }
        // take the squareroot of the frequencies
        self.freqs = self.freqs.map(|val| val.sqrt());
    }

    pub fn sample_initial_conditions(&self, freq_idx: usize) -> (f64, f64) {
        // get the frequency of the mode
        let freq: f64 = self.freqs[freq_idx];

        // check the temperature
        let factor: f64 = if self.temperature <= 1.0e-14 {
            1.0
        } else {
            tanh(1.0 * freq / (2.0 * 3.1671e-6 * self.temperature))
        };
        // sample the position and momentum
        // get the standard deviations
        let dq: f64 = (1.0 / (2.0 * freq * factor)).sqrt();
        let dp: f64 = (1.0 * freq / (2.0 * factor)).sqrt();

        // create distributions
        let dist_q = Normal::new(0.0, dq).expect("Error with the distribution of the position!");
        let dist_p = Normal::new(0.0, dp).expect("Error with the distribution of the momentum!");

        // sample the distribution
        let q0: f64 = dist_q.sample(&mut rand::thread_rng());
        let p0: f64 = dist_p.sample(&mut rand::thread_rng());

        (q0, p0)
    }

    pub fn get_ensemble(&mut self) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
        // cut the frequencies
        self.cut_frequencies();

        // get the inverted mass matrix
        let inv_mass: Array2<f64> = Array2::from_diag(&self.masses);
        // set dim as nsample
        let dim: usize = self.dim;

        // initialize the vectors for the coordinates and velocities
        let mut coord_vec: Vec<Array1<f64>> = Vec::new();
        let mut velocity_vec: Vec<Array1<f64>> = Vec::new();

        // loop over number of samples
        for i in 0..self.nsample {
            // initialize the arrays for the q and p vals
            let mut q_arr: Array1<f64> = Array1::zeros(dim);
            let mut p_arr: Array1<f64> = Array1::zeros(dim);

            // loop over the dimension of the hessian
            for j in 0..self.dim {
                // initialize q0 and p0
                let mut q0: f64 = 0.0;
                let mut p0: f64 = 0.0;

                // check if j is above the zero frequencies
                if j >= self.zerofreq {
                    let tmp = self.sample_initial_conditions(j);
                    // set q0 and p0
                    q0 = tmp.0;
                    p0 = tmp.1;
                }
                q_arr[j] = q0;
                p_arr[j] = p0;
            }

            // calculate the coordinates and the velocities
            let coords: Array1<f64> = inv_mass.dot(&self.modes.dot(&q_arr)) + &self.coords;
            let velocities: Array1<f64> = inv_mass.dot(&self.modes.dot(&p_arr));

            // add the arrays to the respective vectors
            coord_vec.push(coords);
            velocity_vec.push(velocities);
        }

        (coord_vec, velocity_vec)
    }
}