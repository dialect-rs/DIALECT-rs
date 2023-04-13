use crate::couplings::overlaps::get_sign_of_array;
use crate::initialization::old_system::OldSystem;
use crate::initialization::System;
use ndarray::prelude::*;

impl System {
    pub fn get_scalar_coupling(&mut self, dt: f64) -> (Array2<f64>, Array2<f64>) {
        let n_states: usize = self.config.excited.nstates + 1;

        let old_system = if !self.properties.old_system().is_none() {
            self.properties.old_system().unwrap().clone()
        } else {
            OldSystem::new(&self, None)
        };

        // scalar coupling matrix
        let s_ci: Array2<f64> = self.ci_overlap_system(
            &old_system.atoms,
            old_system.orbs.view(),
            old_system.ci_coefficients.view(),
            n_states,
        );
        // align phases
        // The eigenvalue solver produces vectors with arbitrary global phases
        // (+1 or -1). The orbitals of the ground state can also change their signs.
        // Eigen states from neighbouring geometries should change continuously.
        let diag = s_ci.diag();
        // get signs of the diagonal
        let sign: Array1<f64> = get_sign_of_array(diag);

        let p: Array2<f64> = Array::from_diag(&sign);
        // align the new CI coefficients with the old coefficients
        let p_exclude_gs: ArrayView2<f64> = p.slice(s![1.., 1..]);

        // save the aligned coefficients
        let aligned_coeff: Array2<f64> = self
            .properties
            .ci_coefficients()
            .unwrap()
            .dot(&p_exclude_gs);

        self.properties.set_ci_coefficients(aligned_coeff);

        // align overlap matrix
        let mut s_ci = s_ci.dot(&p);

        // The relative signs for the overlap between the ground and excited states at different geometries
        // cannot be deduced from the diagonal elements of Sci. The phases are chosen such that the coupling
        // between S0 and S1-SN changes smoothly for most of the states.
        if old_system.old_scalar_couplings.is_some() {
            let old_s_ci: Array2<f64> = old_system.old_scalar_couplings.unwrap();
            let s: Array1<f64> =
                get_sign_of_array((&old_s_ci.slice(s![0, 1..]) / &s_ci.slice(s![0, 1..])).view());
            let w: Array1<f64> =
                (&s_ci.slice(s![0, 1..]) - &old_s_ci.slice(s![0, 1..])).map(|val| val.abs());
            let mut mean_sign: f64 = ((&w * &s).sum() / w.sum()).signum();
            if mean_sign.is_nan() {
                mean_sign = 1.0
            }
            for i in 1..n_states {
                s_ci[[0, i]] *= mean_sign;
                s_ci[[i, 0]] *= mean_sign;
            }
        }

        // coupl[A,B] = <Psi_A(t)|Psi_B(t+dt)> - delta_AB
        //            ~ <Psi_A(t)|d/dR Psi_B(t)>*dR/dt dt
        // The scalar coupling matrix should be more or less anti-symmetric
        // provided the time-step is small enough
        // set diagonal elements of coupl to zero
        let mut coupling: Array2<f64> = s_ci.clone();
        coupling = coupling - Array::from_diag(&s_ci.diag());

        // Because of the finite time-step it will not be completely antisymmetric,
        coupling = 0.5 * (&coupling - &coupling.t());

        let old_system: OldSystem = OldSystem::new(&self, Some(coupling.clone()));
        self.properties.set_old_system(old_system);

        // coupl = <Psi_A|d/dR Psi_B>*dR/dt * dt
        coupling = coupling / dt;

        // coupling
        return (coupling, s_ci);
    }
}
