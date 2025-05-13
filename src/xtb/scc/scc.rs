use crate::scc::mixer::{BroydenMixer, Mixer};
use crate::{
    io::settings::MixConfig,
    scc::{
        density_matrix, fermi_occupation,
        logging::{print_energies_at_iteration, print_scc_end, print_scc_end_xtb, print_scc_init},
        mulliken::{mulliken_aowise, mulliken_atomwise_from_ao_xtb, mulliken_atomwise_xtb},
        outer_sum,
        scc_routine::SCCError,
    },
};
use crate::{
    scc::scc_routine::RestrictedSCC,
    utils::Timer,
    xtb::{
        initialization::system::XtbSystem,
        parameters::COUL_THIRD_ORDER_ATOM,
        scc::{
            gamma_matrix::{gamma_matrix_xtb, gamma_matrix_xtb_new},
            scc_helpers::{
                coul_third_order_hamiltonian, create_density_ref, get_dispersion_energy_xtb,
                get_electronic_energy_xtb, get_entropy_energy_contribution,
            },
        },
    },
};
use log::{log_enabled, Level};
use ndarray::prelude::*;
use ndarray_linalg::{Eigh, Inverse, SymmetricSqrt, UPLO};
use ndarray_stats::DeviationExt;

impl<'a> RestrictedSCC for XtbSystem {
    fn prepare_scc(&mut self) {
        // calculate s and h0
        self.get_overlap();
        self.get_h0();

        // calculate the gamma matrix
        let g0: Array2<f64> = gamma_matrix_xtb_new(&self.gammafunction, &self.atoms, &self.basis);
        self.properties.set_gamma_ao(g0);

        // if this is the first SCC calculation the charge differences will be initialized to zeros
        if !self.properties.contains_key("dq") {
            self.properties.set_dq(Array1::zeros(self.n_atoms));
        }
        if !self.properties.contains_key("dq_ao") {
            self.properties.set_dq_ao(Array1::zeros(self.n_orbs));
        }

        // this is also only needed in the first SCC calculation
        if !self.properties.contains_key("ref_density_matrix") {
            self.properties
                .set_p_ref(create_density_ref(&self.basis, &self.atoms));
        }

        // in the first SCC calculation the density matrix is set to the reference density matrix
        if !self.properties.contains_key("P") {
            self.properties
                .set_p(self.properties.p_ref().unwrap().to_owned());
        }
    }

    fn run_scc(&mut self) -> Result<f64, SCCError> {
        let timer: Timer = Timer::start();

        // SCC settings from the user input
        let max_iter: usize = self.config.scf.scf_max_cycles;
        let scf_charge_conv: f64 = self.config.scf.scf_charge_conv;
        let scf_energy_conv: f64 = self.config.scf.scf_energy_conv;
        let temperature: f64 = 300.0;

        // the properties that are changed during the SCC routine are taken
        // and will be inserted at the end of the SCC routine
        let mut p: Array2<f64> = self.properties.take_p().unwrap();
        let mut dq: Array1<f64> = self.properties.take_dq().unwrap();
        let mut dq_ao: Array1<f64> = self.properties.dq_ao().unwrap().to_owned();

        // initialize the charge mixer
        let mut broyden_mixer: BroydenMixer = BroydenMixer::new(self.n_orbs);

        // molecular properties, we take all properties that are needed from the Properties type
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let h0: ArrayView2<f64> = self.properties.h0().unwrap();
        let p0: ArrayView2<f64> = self.properties.p_ref().unwrap();

        // the orbital energies and coefficients can be safely reset, since the
        // Hamiltonian does not depends on the charge differences and not on the orbital coefficients
        let mut orbs: Array2<f64> = Array2::zeros([self.n_orbs, self.n_orbs]);
        let mut orbe: Array1<f64> = Array1::zeros([self.n_orbs]);
        // orbital occupation numbers
        let mut f: Vec<f64> = vec![0.0; self.n_orbs];

        // variables that are updated during the iterations
        let mut last_energy: f64 = 0.0;
        let mut total_energy: Result<f64, SCCError> = Ok(0.0);
        let mut scf_energy: f64 = 0.0;

        // get the repulsive energy
        let rep_energy: f64 = self.calculate_repulsive_energy();

        if log_enabled!(Level::Info) {
            print_scc_init(max_iter, temperature, rep_energy);
        }
        // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
        // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
        let x: Array2<f64> = s.ssqrt(UPLO::Upper).unwrap().inv().unwrap();
        // get the dispersion energy
        let e_disp: f64 = get_dispersion_energy_xtb(&self.atoms, &self.config.dispersion);

        // create array of hubbard derivatives
        let mut hubbard_derivatives: Array1<f64> = Array1::zeros(self.n_atoms);
        for (mut val, atom) in hubbard_derivatives.iter_mut().zip(self.atoms.iter()) {
            *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
        }
        // calculate the halogen correction
        let halogen_correction: f64 = self.get_halogen_correction();

        'scf_loop: for i in 0..max_iter {
            // calculate the coulomb part of the Hamiltonian
            let h_coul: Array2<f64> =
                outer_sum(self.properties.gamma_ao().unwrap().dot(&dq_ao).view()) * &s * 0.5;
            // calculate the third order coulomb Hamiltonian
            let h_coul_third_order: Array2<f64> =
                coul_third_order_hamiltonian(hubbard_derivatives.view(), dq.view(), &self.basis)
                    * &s
                    * 0.5;
            // add the parts of the Hamiltonian
            let mut h: Array2<f64> = &h0 - &h_coul - &h_coul_third_order;

            // H' = X^t.H.X
            h = x.t().dot(&h).dot(&x);
            let tmp: (Array1<f64>, Array2<f64>) = h.eigh(UPLO::Upper).unwrap();
            orbe = tmp.0;
            // C = X.C'
            orbs = x.dot(&tmp.1);

            // compute the fermi orbital occupation
            let tmp: (f64, Vec<f64>) = fermi_occupation(orbe.view(), self.n_elec, temperature);
            f = tmp.1;

            // calculate the density matrix
            p = density_matrix(orbs.view(), &f[..]);

            // Compute the difference density matrix. This will be mixed in case of long-range correction.
            let dp: Array2<f64> = &p0 - &p;

            // mulliken charges
            let dq1 = mulliken_aowise(dp.view(), s.view());
            // Charge difference to previous iteration
            let delta_dq: Array1<f64> = &dq1 - &dq_ao;
            // Broyden mixing of Mulliken charges per orbital.
            dq_ao = broyden_mixer.next(dq_ao, delta_dq);
            let dq_new_ao: Array1<f64> = dq_ao.clone();
            // let dq_new_ao: Array1<f64> = accel.apply(dq_ao.view(), dq1.view()).unwrap();
            let dq_new: Array1<f64> =
                mulliken_atomwise_from_ao_xtb(&self.basis, self.n_atoms, dq_new_ao.view());

            // compute electronic energy
            scf_energy = get_electronic_energy_xtb(
                p.view(),
                h0.view(),
                dq_new.view(),
                dq_new_ao.view(),
                self.properties.gamma_ao().unwrap(),
                hubbard_derivatives.view(),
            ) + get_entropy_energy_contribution(&f, temperature);

            let diff_dq_max: f64 = dq_new.root_mean_sq_err(&dq).unwrap();
            if log_enabled!(Level::Info) {
                print_energies_at_iteration(i, scf_energy, rep_energy, last_energy, diff_dq_max)
            }

            // check if charge difference to the previous iteration is lower than 1e-5
            let converged: bool = if (diff_dq_max < scf_charge_conv)
                && (last_energy - scf_energy).abs() < scf_energy_conv
            {
                true
            } else {
                false
            };
            // save the scf energy and charges from the current iteration
            last_energy = scf_energy;
            dq = dq_new;
            // dq_ao = dq_new_ao;

            if converged {
                total_energy = Ok(scf_energy + rep_energy + e_disp + halogen_correction);
                break 'scf_loop;
            }
            total_energy = Err(SCCError::new(i, last_energy - scf_energy, diff_dq_max));
        }
        if log_enabled!(Level::Info) {
            print_scc_end_xtb(
                timer,
                self.config.jobtype.as_str(),
                scf_energy,
                rep_energy,
                e_disp,
                halogen_correction,
                orbe.view(),
                &f,
            );
        }
        self.properties.set_orbs(orbs);
        self.properties.set_orbe(orbe);
        self.properties.set_occupation(f);
        self.properties.set_p(p);
        self.properties.set_dq(dq);
        self.properties.set_dq_ao(dq_ao);
        self.properties
            .set_last_energy(total_energy.clone().unwrap());
        return total_energy;
    }
}

// // get the new charges
// let dp_flat: ArrayView1<f64> = dp.view().into_shape(dim).unwrap();
// delta_p = match i {
//     0 => accel
//         .apply(Array1::zeros(dim).view(), dp_flat.view())
//         .unwrap()
//         .into_shape(p.raw_dim())
//         .unwrap(),
//     _ => {
//         let dp0_flat: ArrayView1<f64> = delta_p.view().into_shape(dim).unwrap();
//         accel
//             .apply(dp0_flat.view(), dp_flat.view())
//             .unwrap()
//             .into_shape(p.raw_dim())
//             .unwrap()
//     }
// };
// p = &delta_p + &p0;
// // mulliken charges
// // let dq_new: Array1<f64> =
// //     mulliken_atomwise_xtb(delta_p.view(), s.view(), &self.basis, self.n_atoms);
// let dq_new_ao: Array1<f64> = mulliken_aowise(delta_p.view(), s.view());
// let dq_new: Array1<f64> =
//     mulliken_atomwise_from_ao_xtb(&self.basis, self.n_atoms, dq_new_ao.view());
//
