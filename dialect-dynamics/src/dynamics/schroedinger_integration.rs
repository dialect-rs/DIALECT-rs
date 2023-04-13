use crate::initialization::Simulation;
use ndarray::prelude::*;
use ndarray_linalg::{c64, Eig, Eigh, Inverse, UPLO};

impl Simulation {
    /// The coefficients of the electronic wavefunction are propagated
    /// in the local diabatic basis as explained in
    /// [1]  JCP 114, 10608 (2001) and
    /// [2]  JCP 137, 22A514 (2012)
    pub fn get_local_diabatization(&mut self, energy_last: ArrayView1<f64>) -> Array1<c64> {
        // Loewding orthogonalization of the S matrix
        // see eqns. (B5) and (B6) in [2]
        let s_ts: Array2<f64> = self.s_mat.t().dot(&self.s_mat);
        let (l, o): (Array1<f64>, Array2<f64>) = s_ts.eigh(UPLO::Upper).unwrap();
        let lm12: Array1<f64> = (1.0 / l).mapv(|val| val.sqrt());

        // unitary transformation matrix, see eqn. (B5) in [1]
        let t: Array2<f64> = self.s_mat.dot(&o.dot(&Array::from_diag(&lm12).dot(&o.t())));

        let t_inv: ArrayView2<f64> = t.t();
        // electronic coefficients c(t)
        let c_0: ArrayView1<c64> = self.coefficients.view();
        // adiabatic energies at the beginning of the time step, E(t)
        let e_0: ArrayView1<f64> = energy_last;
        // adiabatic energies at the end of the time step, E(t+dt)
        let e_1: ArrayView1<f64> = self.energies.view();

        // diabatic hamiltonian H(t+dt)
        let h: Array2<f64> = t.dot(&Array::from_diag(&e_1).dot(&t_inv));
        let mut h_interp: Array2<f64> = (Array::from_diag(&e_0) + h) / 2.0;

        // subtract lowest energy from diagonal
        let h_00_val: f64 = h_interp[[0, 0]];
        for ii in 0..h_interp.dim().0 {
            h_interp[[ii, ii]] -= h_00_val;
        }

        // propagator in diabatic basis, see eqn. (11) in [1]
        let u_1: Array2<c64> = h_interp.mapv(|val| -c64::new(0.0, 1.0) * val * self.stepsize);
        let (eig, eig_vec): (Array1<c64>, Array2<c64>) = u_1.eig().unwrap();
        let diag: Array1<c64> = eig.mapv(|val| val.exp());
        let u_mat: Array2<c64> = eig_vec.dot(&Array::from_diag(&diag).dot(&eig_vec.inv().unwrap()));

        // at the beginning of the time step the adiabatic and diabatic basis is assumed to coincide
        // new electronic coefficients c(t+dt) in the adiabatic basis
        let complex_t_inv: Array2<c64> = t_inv.mapv(|val| val * c64::new(1.0, 0.0));
        let c_1: Array1<c64> = complex_t_inv.t().dot(&u_mat.dot(&c_0));

        // norm of electronic wavefunction
        let norm_c: f64 = c_1.map(|val| val.re.powi(2) + val.im.powi(2)).sum();
        assert!(
            norm_c - 1.0 < 1.0e-3,
            "Norm of electronic coefficients not conserved! Norm = {}",
            norm_c
        );

        // save the diabatic hamiltonian along the trajectory
        let ttot_last: Array2<f64> = if self.t_tot_last.is_none() {
            Array::eye(t.dim().0)
        } else {
            self.t_tot_last.clone().unwrap()
        };

        // the transformations are concatenated to obtain the diabatic
        // Hamiltonian relative to the first time step
        let t_tot: Array2<f64> = t.dot(&ttot_last);

        let h_diab: Array2<f64> = t_tot.dot(&Array::from_diag(&e_1).dot(&t_tot.t()));
        self.t_tot_last = Some(t_tot);
        self.hdiab = h_diab;

        c_1
    }

    // solve the electronic schroedinger equation using the matrix exponential
    pub fn matrix_exponential_integration(&self) -> Array1<c64> {
        let mut mat: Array2<c64> =
            &Array::from_diag(&self.energies.map(|val| val * c64::new(0.0, -1.0)))
                - &self.nonadiabatic_scalar;
        mat = mat * self.stepsize;
        let (eig, eig_vec): (Array1<c64>, Array2<c64>) = mat.eig().unwrap();
        let diag: Array1<c64> = eig.mapv(|val| val.exp());
        let mat: Array2<c64> = eig_vec.dot(&Array::from_diag(&diag).dot(&eig_vec.inv().unwrap()));

        mat.dot(&self.coefficients)
    }

    // Obtain the new coefficients of the states by utilizing a Runge-Kutta 4th order scheme
    // Necessary for the interaction with an electric field
    pub fn rk_integration(&self) -> Array1<c64> {
        // get the nonadiabatic scalar couplings
        let nonadiabatic_scalar: Array2<f64> = -1.0 * &self.nonadiabatic_scalar;

        // set the stepsize of the RK-integration
        // let n_delta: usize = self.config.hopping_config.integration_steps;
        let n_delta: usize = self.config.hopping_config.integration_steps;
        let delta_rk: f64 = self.stepsize / n_delta as f64;

        // start the Runge-Kutta integration
        let mut old_coefficients: Array1<c64> = self.coefficients.clone();
        for i in 0..n_delta {
            let t_i: f64 = i as f64 * delta_rk;

            // do one step of the integration
            let new_coefficients: Array1<c64> = self.runge_kutta_integration(
                t_i,
                old_coefficients.view(),
                delta_rk,
                nonadiabatic_scalar.view(),
            );
            old_coefficients = new_coefficients;
        }
        // calculate the new coefficients
        let time: f64 = delta_rk * n_delta as f64;
        let energy_compl: Array1<c64> = self
            .energies
            .mapv(|val| (-c64::new(0.0, 1.0) * val * time).exp());
        let c_new: Array1<c64> = old_coefficients * energy_compl;

        c_new
    }

    /// Calculate one step of the 4th order Runge-Kutta method
    pub fn runge_kutta_integration(
        &self,
        time: f64,
        coefficients: ArrayView1<c64>,
        delta_rk: f64,
        nonadiabatic_scalar: ArrayView2<f64>,
    ) -> Array1<c64> {
        let mut k_1: Array1<c64> = self.runge_kutta_helper(time, coefficients, nonadiabatic_scalar);
        k_1 = k_1 * delta_rk;
        let tmp: Array1<c64> = &coefficients + &(&k_1 * 0.5);

        let mut k_2: Array1<c64> =
            self.runge_kutta_helper(time + 0.5 * delta_rk, tmp.view(), nonadiabatic_scalar);
        k_2 = k_2 * delta_rk;
        let tmp: Array1<c64> = &coefficients + &(&k_2 * 0.5);

        let mut k_3: Array1<c64> =
            self.runge_kutta_helper(time + 0.5 * delta_rk, tmp.view(), nonadiabatic_scalar);
        k_3 = k_3 * delta_rk;
        let tmp: Array1<c64> = &coefficients + &(&k_3 * 0.5);

        let mut k_4: Array1<c64> =
            self.runge_kutta_helper(time + delta_rk, tmp.view(), nonadiabatic_scalar);
        k_4 = k_4 * delta_rk;

        let new_coefficients: Array1<c64> =
            &coefficients + &((k_1 + k_2 * 2.0 + k_3 * 2.0 + k_4) * 1.0 / 6.0);
        new_coefficients
    }

    /// Calculate a coeffiecient k of the runge kutta method
    fn runge_kutta_helper(
        &self,
        time: f64,
        coefficients: ArrayView1<c64>,
        nonadiabatic_scalar: ArrayView2<f64>,
    ) -> Array1<c64> {
        let _f: Array1<c64> = Array1::zeros(coefficients.raw_dim());
        // let mut non_adiabatic: Array2<f64> = Array2::zeros(nonadiabatic_slope.raw_dim());
        let nstates: usize = self.config.nstates;

        // create energy difference array
        let energy_arr_tmp: Array2<f64> = self.energies.clone().insert_axis(Axis(1));
        let mesh_1: ArrayView2<f64> = energy_arr_tmp.broadcast((nstates, nstates)).unwrap();
        let energy_difference: Array2<f64> = &mesh_1.clone() - &mesh_1.t();

        // alternative way instead of iteration
        let de: Array2<c64> = energy_difference.mapv(|val| (c64::new(0.0, 1.0) * val * time).exp());
        let h: Array2<c64> = de * nonadiabatic_scalar;
        let f_new: Array1<c64> = h.dot(&coefficients);

        f_new
    }
}
