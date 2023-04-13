use crate::fmo::Monomer;
use crate::initialization::Atom;
use crate::io::settings::MixConfig;
use crate::io::SccConfig;
use crate::scc::gamma_approximation::*;
use crate::scc::h0_and_s::*;
use crate::scc::mixer::{AndersonAccel, BroydenMixer, Mixer};
use crate::scc::mulliken::mulliken_atomwise;
use crate::scc::{
    calc_exchange, density_matrix, density_matrix_ref, get_electronic_energy_new, lc_exact_exchange,
};
use ndarray::prelude::*;
use ndarray_linalg::*;
use ndarray_stats::DeviationExt;

impl Monomer<'_> {
    pub fn prepare_scc(&mut self, atoms: &[Atom]) {
        // get H0 and S
        let (s, h0): (Array2<f64>, Array2<f64>) = h0_and_s(self.n_orbs, &atoms, &self.slako);
        // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
        // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
        let x: Array2<f64> = s.ssqrt(UPLO::Upper).unwrap().inv().unwrap();
        // and save it in the self properties
        self.properties.set_h0(h0);
        self.properties.set_s(s);
        self.properties.set_x(x);
        // save the atomic numbers since we need them multiple times
        let atomic_numbers: Vec<u8> = atoms.iter().map(|atom| atom.number).collect();
        self.properties.set_atomic_numbers(atomic_numbers);
        // get the gamma matrix

        let gamma: Array2<f64> = gamma_atomwise(&self.gammafunction, &atoms, self.n_atoms);
        let gamma_ao = gamma_ao_wise_from_gamma_atomwise(gamma.view(), &atoms, self.n_orbs);
        // and save it as a `Property`
        self.properties.set_gamma(gamma);
        self.properties.set_gamma_ao(gamma_ao);

        // calculate the number of electrons
        let n_elec: usize = atoms.iter().fold(0, |n, atom| n + atom.n_elec);

        // Set the indices of the occupied and virtual orbitals based on the number of electrons.
        self.set_mo_indices(n_elec);

        // occupation is determined by Aufbau principle and no electronic temperature is considered
        let f: Vec<f64> = (0..self.n_orbs)
            .map(|idx| if idx < n_elec / 2 { 2.0 } else { 0.0 })
            .collect();
        self.properties.set_occupation(f);

        // if the system contains a long-range corrected Gammafunction the gamma matrix will be computed
        if self.gammafunction_lc.is_some() {
            let (gamma_lr, gamma_lr_ao): (Array2<f64>, Array2<f64>) = gamma_ao_wise(
                self.gammafunction_lc.as_ref().unwrap(),
                &atoms,
                self.n_atoms,
                self.n_orbs,
            );
            self.properties.set_gamma_lr(gamma_lr);
            self.properties.set_gamma_lr_ao(gamma_lr_ao);
        }

        self.properties.set_mixer(BroydenMixer::new(self.n_orbs));

        // Anderson mixer
        let mix_config: MixConfig = MixConfig::default();
        let mut dim: usize = 0;
        if self.gammafunction_lc.is_some() {
            dim = self.n_orbs * self.n_orbs;
        } else {
            dim = self.n_atoms;
        }
        let accel = mix_config.build_mixer(dim).unwrap();
        self.properties.set_accel(accel);

        // if this is the first SCC calculation the charge differences will be initialized to zeros
        if !self.properties.contains_key("dq") {
            self.properties.set_dq(Array1::zeros(self.n_atoms));
            // self.properties.set_dq(Array1::zeros(self.n_orbs));
            self.properties.set_q_ao(Array1::zeros(self.n_orbs));
        }

        // this is also only needed in the first SCC calculation
        if !self.properties.contains_key("ref_density_matrix") {
            self.properties
                .set_p_ref(density_matrix_ref(self.n_orbs, &atoms));
        }

        // in the first SCC calculation the density matrix is set to the reference density matrix
        if !self.properties.contains_key("P") {
            self.properties
                .set_p(self.properties.p_ref().unwrap().to_owned());
        }
    }

    pub fn scc_step(&mut self, atoms: &[Atom], v_esp: Array2<f64>, config: SccConfig) -> bool {
        let scf_charge_conv: f64 = config.scf_charge_conv;
        let scf_energy_conv: f64 = config.scf_energy_conv;
        let dq: Array1<f64> = self.properties.take_dq().unwrap();
        let mut accel: AndersonAccel = self.properties.take_accel().unwrap();
        let mut p: Array2<f64> = self.properties.take_p().unwrap();
        let x: ArrayView2<f64> = self.properties.x().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let h0: ArrayView2<f64> = self.properties.h0().unwrap();
        let p0: ArrayView2<f64> = self.properties.p_ref().unwrap();
        let last_energy: f64 = self.properties.last_energy().unwrap();
        let f: &[f64] = self.properties.occupation().unwrap();
        // electrostatic interaction between the atoms of the same monomer and all the other atoms
        // the coulomb term and the electrostatic potential term are combined into one:
        // H_mu_nu = H0_mu_nu + HCoul_mu_nu + HESP_mu_nu
        // H_mu_nu = H0_mu_nu + 1/2 S_mu_nu sum_k sum_c_on_k (gamma_ac + gamma_bc) dq_c
        let h_coul: Array2<f64> = v_esp * &s * 0.5;
        let mut h: Array2<f64> = h_coul + h0;
        if self.gammafunction_lc.is_some() && self.properties.delta_p().is_some() {
            let h_x: Array2<f64> = lc_exact_exchange(
                s,
                self.properties.gamma_lr_ao().unwrap(),
                self.properties.delta_p().unwrap(),
            );
            h = h + h_x;
        }
        let h_save: Array2<f64> = h.clone();

        // H' = X^t.H.X
        h = x.t().dot(&h).dot(&x);
        let tmp: (Array1<f64>, Array2<f64>) = h.eigh(UPLO::Upper).unwrap();
        let orbe: Array1<f64> = tmp.0;
        // C = X.C'
        let orbs: Array2<f64> = x.dot(&tmp.1);

        // calculate the density matrix
        p = density_matrix(orbs.view(), &f[..]);

        // Compute the difference density matrix. This will be mixed in case of long-range correction.
        let dp: Array2<f64> = &p - &p0;

        let (dq_new, delta_p_temp): (Array1<f64>, Option<Array2<f64>>) =
            if self.gammafunction_lc.is_some() {
                let dim: usize = self.n_orbs * self.n_orbs;
                let dp_flat: ArrayView1<f64> = dp.view().into_shape(dim).unwrap();

                let delta_p: Array2<f64> = match &self.properties.delta_p() {
                    Some(dp0) => {
                        let dp0_flat: ArrayView1<f64> = dp0.into_shape(dim).unwrap();
                        accel
                            .apply(dp0_flat.view(), dp_flat.view())
                            .unwrap()
                            .into_shape(p.raw_dim())
                            .unwrap()
                    }
                    None => accel
                        .apply(Array1::zeros(dim).view(), dp_flat.view())
                        .unwrap()
                        .into_shape(p.raw_dim())
                        .unwrap(),
                };
                p = &delta_p + &p0;
                // mulliken charges
                let dq_temp = mulliken_atomwise(delta_p.view(), s.view(), atoms, self.n_atoms);
                (dq_temp, Some(delta_p))
            } else {
                // mulliken charges
                let dq1 = mulliken_atomwise(dp.view(), s.view(), atoms, self.n_atoms);
                let dq_temp = accel.apply(dq.view(), dq1.view()).unwrap();
                (dq_temp, None)
            };

        // compute electronic energy
        let mut scf_energy = get_electronic_energy_new(
            p.view(),
            h0.view(),
            dq_new.view(),
            self.properties.gamma().unwrap(),
        );
        if self.gammafunction_lc.is_some() {
            scf_energy += calc_exchange(
                s.view(),
                self.properties.gamma_lr_ao().unwrap(),
                delta_p_temp.clone().unwrap().view(),
            );
        }

        let diff_dq_max: f64 = dq_new.root_mean_sq_err(&dq).unwrap();

        // check if charge difference to the previous iteration is lower than threshold
        let conv_charge: bool = diff_dq_max < scf_charge_conv;
        // same check for the electronic energy
        let conv_energy: bool = (last_energy - scf_energy).abs() < scf_energy_conv;

        if self.gammafunction_lc.is_some() {
            self.properties.set_delta_p(delta_p_temp.unwrap());
        }
        self.properties.set_orbs(orbs);
        self.properties.set_orbe(orbe);
        self.properties.set_p(p);
        self.properties.set_dq(dq_new);
        self.properties.set_accel(accel);
        self.properties.set_last_energy(scf_energy);
        self.properties.set_h_coul_x(h_save);
        self.properties.set_h_coul_transformed(h);

        // scc (for one fragment) is converged if both criteria are passed
        conv_charge && conv_energy
    }
}
