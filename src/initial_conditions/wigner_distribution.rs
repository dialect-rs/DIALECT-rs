use crate::constants::HARTREE_TO_WAVENUMBERS;
use crate::initialization::System;
use libm::tanh;
use log::{log_enabled, warn, Level};
use ndarray::prelude::*;
use ndarray_linalg::SVD;
use ndarray_rand::rand_distr::{Distribution, Normal};
use ndarray_stats::QuantileExt;

pub struct WignerEnsemble<'a> {
    pub temperature: f64,
    pub nsample: usize,
    pub n_cut: usize,
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
        masses: ArrayView1<'a, f64>,
        coords: ArrayView1<'a, f64>,
    ) -> WignerEnsemble<'a> {
        let dim: usize = freqs.len();
        let wigner_config = &system.config.wigner_config;

        WignerEnsemble {
            temperature: wigner_config.temperature,
            nsample: wigner_config.n_samples,
            n_cut: wigner_config.n_cut,
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

        let mut cut_number: usize = self.n_cut;
        // check for linearity
        if s[1] / s[0] < 0.01 && cut_number <= 6 {
            cut_number = 5;
            self.zerofreq = 5;
        }
        self.zerofreq = cut_number;

        // cut frequencies
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
        // print the frequencies
        if log_enabled!(Level::Warn) {
            warn!("{:-^80}", "");
            warn!("Number of cut frequencies: {}", cut_number);
            warn!(
                "Lowest non-cut frequency: {:.6}",
                self.freqs[cut_number] * HARTREE_TO_WAVENUMBERS,
            );
            warn!("{:-^80}", "");
        }
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
        let inv_mass: Array2<f64> = Array2::from_diag(&(self.masses.map(|val| 1.0 / val.sqrt())));
        // set dim as nsample
        let dim: usize = self.dim;

        // initialize the vectors for the coordinates and velocities
        let mut coord_vec: Vec<Array1<f64>> = Vec::new();
        let mut velocity_vec: Vec<Array1<f64>> = Vec::new();

        // loop over number of samples
        for _i in 0..self.nsample {
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
            let coords: Array1<f64> = inv_mass.dot(&self.modes.dot(&q_arr)) + self.coords;
            let velocities: Array1<f64> = inv_mass.dot(&self.modes.dot(&p_arr));

            // add the arrays to the respective vectors
            coord_vec.push(coords);
            velocity_vec.push(velocities);
        }

        (coord_vec, velocity_vec)
    }
}

// impl System {
//     pub fn sample_wigner(&self,aw:Array2<f64>,bw:Array1<f64>,nsample:usize){
//         // get the covariance matrix
//         let cov:Array2<f64> = aw.inv().unwrap();
//         // mean
//         let mean:Array1<f64> = cov.dot(&bw);
//     }

//     pub fn wigner_distribution(
//         &self,
//         omega2: ArrayView1<f64>,
//         modes: ArrayView2<f64>,
//         masses: ArrayView1<f64>,
//     ) -> (Array2<f64>, Array1<f64>) {
//         // set the gradient dimension
//         let dim: usize = omega2.len();
//         // set the zero threshold for frequencies
//         let zero_threshold: f64 = 1.0e-8;

//         // set the modes that are zero within numerical accuracy
//         let mut zero_indices: Vec<usize> = Vec::new();
//         let mut vib_indices: Vec<usize> = Vec::new();
//         for (idx, val) in omega2.iter().enumerate() {
//             if *val < zero_threshold {
//                 zero_indices.push(idx);
//             } else {
//                 vib_indices.push(idx);
//             }
//         }

//         // calculate the outer product of the masses and take the square root
//         let masses_matrix: Array2<f64> = into_col(masses.clone())
//             .dot(&into_row(masses))
//             .map(|val| val.sqrt());
//         let masses_inv: Array2<f64> = 1.0 / &masses_matrix;
//         // get maximum values of both masses arrays
//         let max_msq: f64 = *masses_matrix.max().unwrap();
//         let max_minv: f64 = *masses_inv.max().unwrap();

//         // define empty arrays
//         let mut non_zero_freqs: Array1<f64> = Array::zeros(vib_indices.len());
//         let mut non_zero_modes: Array2<f64> = Array::zeros([dim, vib_indices.len()]);

//         // get the non zero freqs and modes
//         for (idx, idx_val) in vib_indices.iter().enumerate() {
//             non_zero_freqs[idx] = omega2[*idx_val].sqrt();
//             non_zero_modes
//                 .slice_mut(s![.., idx])
//                 .assign(&modes.slice(s![.., *idx_val]));
//         }

//         // get the wigner distribution
//         let freqs_2d: Array2<f64> = Array::from_diag(&non_zero_freqs);
//         let freqs_2d_inv: Array2<f64> = Array::from_diag(&(1.0 / &non_zero_freqs));
//         let mut aq: Array2<f64> =
//             2.0 * non_zero_modes.dot(&freqs_2d.dot(&non_zero_modes.t())) * &masses_matrix;
//         let mut ap: Array2<f64> =
//             2.0 * non_zero_modes.dot(&freqs_2d_inv.dot(&non_zero_modes.t())) * &masses_inv;

//         // constrain zero modes by delta-functions
//         let constraint: Array2<f64> = Array::eye(zero_indices.len()) * 1.0e6;
//         let mut zero_modes: Array2<f64> = Array::zeros([dim, vib_indices.len()]);
//         for (idx, idx_val) in zero_indices.iter().enumerate() {
//             zero_modes
//                 .slice_mut(s![.., idx])
//                 .assign(&modes.slice(s![.., *idx_val]));
//         }
//         aq = aq + zero_modes.dot(&constraint.dot(&zero_modes.t())) * &masses_matrix;
//         ap = ap
//             + zero_modes.dot(&constraint.dot(&zero_modes.t())) * &masses_inv * max_msq / max_minv;

//         // get the coordinates of the system
//         let x0: Array1<f64> = self.get_xyz();

//         let bq: Array1<f64> = x0.dot(&aq);
//         let bp: Array1<f64> = Array1::zeros(dim);
//         let z0: Array2<f64> = Array2::zeros([dim,dim]);

//         // stack matrices aq, ap, z0
//         let mut aw: Array2<f64> = Array2::zeros([2 * dim, 2 * dim]);
//         aw.slice_mut(s![..dim, ..dim]).assign(&aq);
//         aw.slice_mut(s![..dim, dim..]).assign(&z0);
//         aw.slice_mut(s![dim.., ..dim]).assign(&z0);
//         aw.slice_mut(s![dim.., dim..]).assign(&ap);

//         // stack matices bq, bp
//         let mut bw: Array1<f64> = Array1::zeros(2 * dim);
//         bw.slice_mut(s![..dim]).assign(&bq);
//         bw.slice_mut(s![dim..]).assign(&bp);

//         (aw, bw)
//     }
// }
