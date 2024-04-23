use crate::initialization::Simulation;
use faer::{Faer, IntoFaer, IntoNdarray};
use ndarray::prelude::*;
use ndarray_linalg::{c64, Eig, Eigh, Inverse, UPLO};

impl Simulation {
    pub fn ehrenfest_matrix_exponential(&self, exciton_couplings: ArrayView2<c64>) -> Array1<c64> {
        let mat: Array2<c64> =
            exciton_couplings.mapv(|val| -c64::new(0.0, 1.0) * val * self.stepsize);
        let (eig, eig_vec): (Array1<c64>, Array2<c64>) = mat.eig().unwrap();

        let diag: Array1<c64> = eig.mapv(|val| val.exp());
        let mat: Array2<c64> = eig_vec.dot(&Array::from_diag(&diag).dot(&eig_vec.inv().unwrap()));

        mat.dot(&self.coefficients)
    }

    pub fn ehrenfest_matrix_exponential_2(
        &self,
        exciton_couplings: ArrayView2<f64>,
    ) -> Array1<c64> {
        let (eig, eig_vec): (Array1<f64>, Array2<f64>) =
            exciton_couplings.eigh(UPLO::Lower).unwrap();
        let eig_vec_c: Array2<c64> = eig_vec.map(|val| val * c64::new(1.0, 0.0));
        // let (eig, eig_vec): (Array1<c64>, Array2<c64>) = mat.block(0.0001);
        let diag: Array1<c64> = eig.mapv(|val| (-c64::new(0.0, 1.0) * self.stepsize * val).exp());
        let mat: Array2<c64> = eig_vec_c.dot(&Array::from_diag(&diag).dot(&eig_vec_c.t()));

        mat.dot(&self.coefficients)
    }

    pub fn ehrenfest_matrix_exponential_faer(
        &self,
        exciton_couplings: ArrayView2<f64>,
    ) -> Array1<c64> {
        let new_mat = exciton_couplings.view().into_faer();

        let eigendecomp = new_mat.selfadjoint_eigendecomposition(faer::Side::Lower);
        let faer_vec = eigendecomp.u();
        let faer_val = eigendecomp.s_diagonal();

        let eigs = faer_val.into_ndarray();
        let eigvecs = faer_vec.into_ndarray();
        let eigvecs_c: Array2<c64> = eigvecs.map(|val| val * c64::new(1.0, 0.0));

        let eig: Array1<f64> = eigs.slice(s![.., 0]).to_owned();
        let diag: Array2<c64> =
            Array::from_diag(&eig.mapv(|val| (-c64::new(0.0, 1.0) * self.stepsize * val).exp()));
        let mat: Array2<c64> = eigvecs_c.dot(&diag.dot(&eigvecs_c.t()));

        mat.dot(&self.coefficients)
    }

    pub fn ehrenfest_matrix_exponential_nacme(
        &self,
        exciton_couplings: ArrayView2<f64>,
    ) -> Array1<c64> {
        let mat: Array2<c64> = &exciton_couplings
            .mapv(|val| -c64::new(0.0, 1.0) * val * self.stepsize)
            - &(&self.nonadiabatic_scalar * self.stepsize);
        //mat = mat * self.stepsize;
        let (eig, eig_vec): (Array1<c64>, Array2<c64>) = mat.eig().unwrap();
        let diag: Array1<c64> = eig.mapv(|val| val.exp());
        let mat: Array2<c64> = eig_vec.dot(&Array::from_diag(&diag).dot(&eig_vec.inv().unwrap()));

        mat.dot(&self.coefficients)
    }

    pub fn ehrenfest_rk(&self, excitonic_couplings: ArrayView2<f64>) -> Array1<c64> {
        // set the stepsize of the RK-integration
        let n_delta: usize = self.config.ehrenfest_config.integration_steps;
        // let delta_rk: f64 = self.stepsize / n_delta as f64;
        let delta_rk: f64 = self.stepsize / n_delta as f64;

        let excitonic_couplings: Array2<c64> =
            excitonic_couplings.mapv(|val| -c64::new(0.0, 1.0) * val);

        // start the Runge-Kutta integration
        let mut old_coefficients: Array1<c64> = self.coefficients.clone();
        for _i in 0..n_delta {
            // do one step of the integration
            old_coefficients = self.runge_kutta_ehrenfest(
                old_coefficients.view(),
                delta_rk,
                excitonic_couplings.view(),
            );
        }
        // calculate the new coefficients
        old_coefficients
    }

    pub fn runge_kutta_ehrenfest(
        &self,
        coefficients: ArrayView1<c64>,
        delta_rk: f64,
        excitonic_couplings: ArrayView2<c64>,
    ) -> Array1<c64> {
        let mut k_1: Array1<c64> = self.rk_ehrenfest_helper(coefficients, excitonic_couplings);
        k_1 = k_1 * delta_rk;
        let tmp: Array1<c64> = &coefficients + &(&k_1 * 0.5);

        let mut k_2: Array1<c64> = self.rk_ehrenfest_helper(tmp.view(), excitonic_couplings);
        k_2 = k_2 * delta_rk;
        let tmp: Array1<c64> = &coefficients + &(&k_2 * 0.5);

        let mut k_3: Array1<c64> = self.rk_ehrenfest_helper(tmp.view(), excitonic_couplings);
        k_3 = k_3 * delta_rk;
        let tmp: Array1<c64> = &coefficients + &k_3;

        let mut k_4: Array1<c64> = self.rk_ehrenfest_helper(tmp.view(), excitonic_couplings);
        k_4 = k_4 * delta_rk;

        let new_coefficients: Array1<c64> =
            &coefficients + &((k_1 + k_2 * 2.0 + k_3 * 2.0 + k_4) * 1.0 / 6.0);
        new_coefficients
    }

    fn rk_ehrenfest_helper(
        &self,
        coefficients: ArrayView1<c64>,
        exciton_couplings: ArrayView2<c64>,
    ) -> Array1<c64> {
        exciton_couplings.dot(&coefficients)
    }
}
