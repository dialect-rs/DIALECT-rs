use crate::fmo::scc::helpers::*;
use crate::fmo::{ESDPair, Monomer, Pair};
use crate::initialization::Atom;
use crate::io::settings::MixConfig;
use crate::io::SccConfig;
use crate::scc::gamma_approximation::*;
use crate::scc::h0_and_s::*;
use crate::scc::mixer::Mixer;
use crate::scc::mulliken::mulliken_atomwise;
use crate::scc::{
    calc_exchange, density_matrix, density_matrix_ref, get_electronic_energy_new,
    get_repulsive_energy, lc_exact_exchange,
};
use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray_linalg::*;
use ndarray_npy::write_npy;
use ndarray_stats::DeviationExt;
use std::path::Path;

impl Pair<'_> {
    pub fn prepare_scc(&mut self, atoms: &[Atom], m1: &Monomer, m2: &Monomer) {
        // get H0 and S outer diagonal block
        let (s_ab, h0_ab): (Array2<f64>, Array2<f64>) = h0_and_s_ab(
            m1.n_orbs,
            m2.n_orbs,
            &atoms[0..m1.n_atoms],
            &atoms[m1.n_atoms..],
            &m1.slako,
        );
        let mut s: Array2<f64> = Array2::zeros([self.n_orbs, self.n_orbs]);
        let mut h0: Array2<f64> = s.clone();

        let mu: usize = m1.n_orbs;
        let a: usize = m1.n_atoms;

        s.slice_mut(s![0..mu, 0..mu])
            .assign(&m1.properties.s().unwrap());
        s.slice_mut(s![mu.., mu..])
            .assign(&m2.properties.s().unwrap());
        s.slice_mut(s![0..mu, mu..]).assign(&s_ab);
        s.slice_mut(s![mu.., 0..mu]).assign(&s_ab.t());

        h0.slice_mut(s![0..mu, 0..mu])
            .assign(&m1.properties.h0().unwrap());
        h0.slice_mut(s![mu.., mu..])
            .assign(&m2.properties.h0().unwrap());
        h0.slice_mut(s![0..mu, mu..]).assign(&h0_ab);
        h0.slice_mut(s![mu.., 0..mu]).assign(&h0_ab.t());

        // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
        // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
        let x: Array2<f64> = s.ssqrt(UPLO::Upper).unwrap().inv().unwrap();
        // get the gamma matrix
        let gamma_ab: Array2<f64> = gamma_atomwise_ab(
            &self.gammafunction,
            &atoms[0..m1.n_atoms],
            &atoms[m1.n_atoms..],
            m1.n_atoms,
            m2.n_atoms,
        );
        let mut gamma: Array2<f64> = Array2::zeros([self.n_atoms, self.n_atoms]);

        gamma
            .slice_mut(s![0..a, 0..a])
            .assign(&m1.properties.gamma().unwrap());
        gamma
            .slice_mut(s![a.., a..])
            .assign(&m2.properties.gamma().unwrap());
        gamma.slice_mut(s![0..a, a..]).assign(&gamma_ab);
        gamma.slice_mut(s![a.., 0..a]).assign(&gamma_ab.t());

        // get electrostatic potential that acts on the pair. This is based on Eq. 44 from the book
        // chapter "The FMO-DFTB Method" by Yoshio Nishimoto and Stephan Irle on page 474 in
        // Recent Advances of the Fragment Molecular Orbital Method
        // See: https://www.springer.com/gp/book/9789811592348
        let mut esp: Array1<f64> = Array1::zeros([self.n_atoms]);
        esp.slice_mut(s![0..a]).assign(
            &(&m1.properties.esp_q().unwrap() - &(gamma_ab.dot(&m2.properties.dq().unwrap()))),
        );
        esp.slice_mut(s![a..]).assign(
            &(&m2.properties.esp_q().unwrap() - &(gamma_ab.t().dot(&m1.properties.dq().unwrap()))),
        );
        // and convert it into a matrix in AO basis
        let omega: Array2<f64> = atomvec_to_aomat(esp.view(), self.n_orbs, &atoms);
        self.properties.set_v(omega * &s * 0.5);

        // and save it in the self properties
        self.properties.set_h0(h0);
        self.properties.set_s(s);
        self.properties.set_x(x);
        self.properties.set_gamma(gamma);

        // save the atomic numbers since we need them multiple times
        let atomic_numbers: Vec<u8> = atoms.iter().map(|atom| atom.number).collect();
        self.properties.set_atomic_numbers(atomic_numbers);

        // calculate the number of electrons
        let n_elec: usize = atoms.iter().fold(0, |n, atom| n + atom.n_elec);
        self.set_mo_indices(n_elec);

        // occupation is determined by Aufbau principle and no electronic temperature is considered
        let f: Vec<f64> = (0..self.n_orbs)
            .map(|idx| if idx < n_elec / 2 { 2.0 } else { 0.0 })
            .collect();
        self.properties.set_occupation(f);

        // if the system contains a long-range corrected Gamma function the gamma matrix will be computed
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

        // if this is the first SCC calculation the charge will be taken from the corresponding
        // monomers
        if !self.properties.contains_key("dq") {
            self.properties.set_dq(concatenate![
                Axis(0),
                m1.properties.dq().unwrap(),
                m2.properties.dq().unwrap()
            ]);
            self.properties.set_q_ao(concatenate![
                Axis(0),
                m1.properties.q_ao().unwrap(),
                m2.properties.q_ao().unwrap()
            ]);
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

    pub fn run_scc(&mut self, atoms: &[Atom], config: SccConfig) {
        let scf_charge_conv: f64 = config.scf_charge_conv;
        let scf_energy_conv: f64 = config.scf_energy_conv;
        let max_iter: usize = config.scf_max_cycles;
        let mut p: Array2<f64> = self.properties.take_p().unwrap();
        let mut delta_p: Array2<f64> = Array2::zeros(p.raw_dim());
        let mut dq: Array1<f64> = self.properties.dq().unwrap().to_owned();

        // Anderson mixer
        let mix_config: MixConfig = MixConfig::default();
        let mut dim: usize = 0;
        if self.gammafunction_lc.is_some() {
            dim = self.n_orbs * self.n_orbs;
        } else {
            dim = self.n_atoms;
        }
        let mut accel = mix_config.build_mixer(dim).unwrap();

        let x: ArrayView2<f64> = self.properties.x().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        let h0: ArrayView2<f64> = self.properties.h0().unwrap();
        let p0: ArrayView2<f64> = self.properties.p_ref().unwrap();
        let mut last_energy: f64 = self.properties.last_energy().unwrap();
        let f: &[f64] = self.properties.occupation().unwrap();
        let v: ArrayView2<f64> = self.properties.v().unwrap();
        let h_esp: Array2<f64> = &h0 + &v;
        let dq_saved: Array2<f64> = Array2::zeros((self.n_atoms, max_iter));
        let delta_dq_saved: Array2<f64> = Array2::zeros((self.n_orbs, max_iter));

        'scf_loop: for iter in 0..max_iter {
            let h_coul: Array2<f64> =
                atomvec_to_aomat(gamma.dot(&dq).view(), self.n_orbs, &atoms) * &s * 0.5;
            let mut h: Array2<f64> = h_coul + &h_esp;

            if self.gammafunction_lc.is_some() && iter > 0 {
                let h_x: Array2<f64> =
                    lc_exact_exchange(s, self.properties.gamma_lr_ao().unwrap(), delta_p.view());
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

            let dq_new: Array1<f64> = if self.gammafunction_lc.is_some() {
                let dim: usize = self.n_orbs * self.n_orbs;
                let dp_flat: ArrayView1<f64> = dp.view().into_shape(dim).unwrap();

                delta_p = match iter {
                    0 => accel
                        .apply(Array1::zeros(dim).view(), dp_flat.view())
                        .unwrap()
                        .into_shape(p.raw_dim())
                        .unwrap(),
                    _ => {
                        let dp0_flat: ArrayView1<f64> = delta_p.view().into_shape(dim).unwrap();
                        accel
                            .apply(dp0_flat.view(), dp_flat.view())
                            .unwrap()
                            .into_shape(p.raw_dim())
                            .unwrap()
                    }
                };
                p = &delta_p + &p0;

                // mulliken charges
                mulliken_atomwise(delta_p.view(), s.view(), atoms, self.n_atoms)
            } else {
                // mulliken charges
                let dq1 = mulliken_atomwise(dp.view(), s.view(), atoms, self.n_atoms);
                accel.apply(dq.view(), dq1.view()).unwrap()
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
                    delta_p.view(),
                );
            }

            let diff_dq_max: f64 = dq_new.root_mean_sq_err(&dq).unwrap();

            // check if charge difference to the previous iteration is lower than 1e-5
            let converged: bool = if (diff_dq_max < scf_charge_conv)
                && (last_energy - scf_energy).abs() < scf_energy_conv
            {
                true
            } else {
                false
            };
            last_energy = scf_energy;
            dq = dq_new;

            if converged {
                let e_rep: f64 = get_repulsive_energy(&atoms, self.n_atoms, &self.vrep);
                self.properties.set_last_energy(scf_energy + e_rep);
                self.properties.set_p(p);
                self.properties.set_orbs(orbs);
                self.properties.set_orbe(orbe);
                self.properties.set_h_coul_x(h_save);
                break 'scf_loop;
            }
            if !converged && iter == max_iter - 1 {
                println!("Iteration {}", iter);
                println!("Monomer indices: {},{}", self.i, self.j);
                let string: String = String::from("dq.npy");
                write_npy(Path::new(&string), &dq_saved.view());
                write_npy(
                    Path::new(&String::from("delta_dq_saved.npy")),
                    &delta_dq_saved.view(),
                );
                panic!("Pair scc routine does not converge!");
            }
        }
        // only remove the large arrays not the energy or charges
        //self.properties.reset();

        self.properties
            .set_delta_dq(&dq - &self.properties.dq().unwrap());
        self.properties.set_dq(dq);
    }
}

impl ESDPair<'_> {
    pub fn prepare_scc(&mut self, atoms: &[Atom], m1: &Monomer, m2: &Monomer) {
        // get H0 and S outer diagonal block
        let (s_ab, h0_ab): (Array2<f64>, Array2<f64>) = h0_and_s_ab(
            m1.n_orbs,
            m2.n_orbs,
            &atoms[0..m1.n_atoms],
            &atoms[m1.n_atoms..],
            &m1.slako,
        );
        let mut s: Array2<f64> = Array2::zeros([self.n_orbs, self.n_orbs]);
        let mut h0: Array2<f64> = s.clone();

        let mu: usize = m1.n_orbs;
        let a: usize = m1.n_atoms;

        s.slice_mut(s![0..mu, 0..mu])
            .assign(&m1.properties.s().unwrap());
        s.slice_mut(s![mu.., mu..])
            .assign(&m2.properties.s().unwrap());
        s.slice_mut(s![0..mu, mu..]).assign(&s_ab);
        s.slice_mut(s![mu.., 0..mu]).assign(&s_ab.t());

        h0.slice_mut(s![0..mu, 0..mu])
            .assign(&m1.properties.h0().unwrap());
        h0.slice_mut(s![mu.., mu..])
            .assign(&m2.properties.h0().unwrap());
        h0.slice_mut(s![0..mu, mu..]).assign(&h0_ab);
        h0.slice_mut(s![mu.., 0..mu]).assign(&h0_ab.t());

        // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
        // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
        let x: Array2<f64> = s.ssqrt(UPLO::Upper).unwrap().inv().unwrap();
        // get the gamma matrix
        let gamma_ab: Array2<f64> = gamma_atomwise_ab(
            &self.gammafunction,
            &atoms[0..m1.n_atoms],
            &atoms[m1.n_atoms..],
            m1.n_atoms,
            m2.n_atoms,
        );
        let mut gamma: Array2<f64> = Array2::zeros([self.n_atoms, self.n_atoms]);

        gamma
            .slice_mut(s![0..a, 0..a])
            .assign(&m1.properties.gamma().unwrap());
        gamma
            .slice_mut(s![a.., a..])
            .assign(&m2.properties.gamma().unwrap());
        gamma.slice_mut(s![0..a, a..]).assign(&gamma_ab);
        gamma.slice_mut(s![a.., 0..a]).assign(&gamma_ab.t());

        // get electrostatic potential that acts on the pair. This is based on Eq. 44 from the book
        // chapter "The FMO-DFTB Method" by Yoshio Nishimoto and Stephan Irle on page 474 in
        // Recent Advances of the Fragment Molecular Orbital Method
        // See: https://www.springer.com/gp/book/9789811592348
        let mut esp: Array1<f64> = Array1::zeros([self.n_atoms]);
        esp.slice_mut(s![0..a]).assign(
            &(&m1.properties.esp_q().unwrap() - &(gamma_ab.dot(&m2.properties.dq().unwrap()))),
        );
        esp.slice_mut(s![a..]).assign(
            &(&m2.properties.esp_q().unwrap() - &(gamma_ab.t().dot(&m1.properties.dq().unwrap()))),
        );
        // and convert it into a matrix in AO basis
        let omega: Array2<f64> = atomvec_to_aomat(esp.view(), self.n_orbs, &atoms);
        self.properties.set_v(omega * &s * 0.5);

        // and save it in the self properties
        self.properties.set_h0(h0);
        self.properties.set_s(s);
        self.properties.set_x(x);
        self.properties.set_gamma(gamma);

        // save the atomic numbers since we need them multiple times
        let atomic_numbers: Vec<u8> = atoms.iter().map(|atom| atom.number).collect();
        self.properties.set_atomic_numbers(atomic_numbers);

        // calculate the number of electrons
        let n_elec: usize = atoms.iter().fold(0, |n, atom| n + atom.n_elec);
        self.set_mo_indices(n_elec);

        // occupation is determined by Aufbau principle and no electronic temperature is considered
        let f: Vec<f64> = (0..self.n_orbs)
            .map(|idx| if idx < n_elec / 2 { 2.0 } else { 0.0 })
            .collect();
        self.properties.set_occupation(f);

        // if this is the first SCC calculation the charge will be taken from the corresponding
        // monomers
        if !self.properties.contains_key("dq") {
            self.properties.set_dq(concatenate![
                Axis(0),
                m1.properties.dq().unwrap(),
                m2.properties.dq().unwrap()
            ]);
            self.properties.set_q_ao(concatenate![
                Axis(0),
                m1.properties.q_ao().unwrap(),
                m2.properties.q_ao().unwrap()
            ]);
        }

        // if the system contains a long-range corrected Gamma function the gamma matrix will be computed
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

    pub fn run_scc(&mut self, atoms: &[Atom], config: SccConfig) {
        let scf_charge_conv: f64 = config.scf_charge_conv;
        let scf_energy_conv: f64 = config.scf_energy_conv;
        let max_iter: usize = config.scf_max_cycles;
        let mut p: Array2<f64> = self.properties.take_p().unwrap();
        let mut dq: Array1<f64> = self.properties.dq().unwrap().to_owned();

        // Anderson mixer
        let mix_config: MixConfig = MixConfig::default();
        let mut accel = mix_config.build_mixer(self.n_atoms).unwrap();

        let x: ArrayView2<f64> = self.properties.x().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        let h0: ArrayView2<f64> = self.properties.h0().unwrap();
        let p0: ArrayView2<f64> = self.properties.p_ref().unwrap();
        let mut last_energy: f64 = self.properties.last_energy().unwrap();
        let f: &[f64] = self.properties.occupation().unwrap();
        let v: ArrayView2<f64> = self.properties.v().unwrap();
        let h_esp: Array2<f64> = &h0 + &v;

        'scf_loop: for iter in 0..max_iter {
            let h_coul: Array2<f64> =
                atomvec_to_aomat(gamma.dot(&dq).view(), self.n_orbs, &atoms) * &s * 0.5;
            let mut h: Array2<f64> = h_coul + &h_esp;

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

            // mulliken charges
            let dq1 = mulliken_atomwise(dp.view(), s.view(), atoms, self.n_atoms);
            let dq_new = accel.apply(dq.view(), dq1.view()).unwrap();

            // compute electronic energy
            let scf_energy = get_electronic_energy_new(
                p.view(),
                h0.view(),
                dq_new.view(),
                self.properties.gamma().unwrap(),
            );

            let diff_dq_max: f64 = dq_new.root_mean_sq_err(&dq).unwrap();

            // check if charge difference to the previous iteration is lower than 1e-5
            let converged: bool = if (diff_dq_max < scf_charge_conv)
                && (last_energy - scf_energy).abs() < scf_energy_conv
            {
                true
            } else {
                false
            };
            last_energy = scf_energy;
            dq = dq_new;

            if converged {
                let e_rep: f64 = get_repulsive_energy(&atoms, self.n_atoms, &self.vrep);
                self.properties.set_last_energy(scf_energy + e_rep);
                self.properties.set_p(p);
                self.properties.set_orbs(orbs);
                self.properties.set_orbe(orbe);
                break 'scf_loop;
            }
            if !converged && iter == max_iter - 1 {
                println!("Iteration {}", iter);
                println!("Monomer indices: {},{}", self.i, self.j);
                panic!("ESD Pair scc routine does not converge!");
            }
        }
        // only remove the large arrays not the energy or charges
        //self.properties.reset();
        self.properties
            .set_delta_dq(&dq - &self.properties.dq().unwrap());
        self.properties.set_dq(dq);
    }

    pub fn run_scc_lc(&mut self, atoms: &[Atom], config: SccConfig) {
        let scf_charge_conv: f64 = config.scf_charge_conv;
        let scf_energy_conv: f64 = config.scf_energy_conv;
        let max_iter: usize = config.scf_max_cycles;
        let mut p: Array2<f64> = self.properties.take_p().unwrap();
        let mut delta_p: Array2<f64> = Array2::zeros(p.raw_dim());
        let mut dq: Array1<f64> = self.properties.dq().unwrap().to_owned();

        // Anderson mixer
        let mix_config: MixConfig = MixConfig::default();
        let mut dim: usize = 0;
        if self.gammafunction_lc.is_some() {
            dim = self.n_orbs * self.n_orbs;
        } else {
            dim = self.n_atoms;
        }
        let mut accel = mix_config.build_mixer(dim).unwrap();

        let x: ArrayView2<f64> = self.properties.x().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        let h0: ArrayView2<f64> = self.properties.h0().unwrap();
        let p0: ArrayView2<f64> = self.properties.p_ref().unwrap();
        let mut last_energy: f64 = self.properties.last_energy().unwrap();
        let f: &[f64] = self.properties.occupation().unwrap();
        let v: ArrayView2<f64> = self.properties.v().unwrap();
        let h_esp: Array2<f64> = &h0 + &v;
        let dq_saved: Array2<f64> = Array2::zeros((self.n_atoms, max_iter));
        let delta_dq_saved: Array2<f64> = Array2::zeros((self.n_orbs, max_iter));

        'scf_loop: for iter in 0..max_iter {
            let h_coul: Array2<f64> =
                atomvec_to_aomat(gamma.dot(&dq).view(), self.n_orbs, &atoms) * &s * 0.5;
            let mut h: Array2<f64> = h_coul + &h_esp;

            if self.gammafunction_lc.is_some() && iter > 0 {
                let h_x: Array2<f64> =
                    lc_exact_exchange(s, self.properties.gamma_lr_ao().unwrap(), delta_p.view());
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

            let dq_new: Array1<f64> = if self.gammafunction_lc.is_some() {
                let dim: usize = self.n_orbs * self.n_orbs;
                let dp_flat: ArrayView1<f64> = dp.view().into_shape(dim).unwrap();

                delta_p = match iter {
                    0 => accel
                        .apply(Array1::zeros(dim).view(), dp_flat.view())
                        .unwrap()
                        .into_shape(p.raw_dim())
                        .unwrap(),
                    _ => {
                        let dp0_flat: ArrayView1<f64> = delta_p.view().into_shape(dim).unwrap();
                        accel
                            .apply(dp0_flat.view(), dp_flat.view())
                            .unwrap()
                            .into_shape(p.raw_dim())
                            .unwrap()
                    }
                };
                p = &delta_p + &p0;

                // mulliken charges
                mulliken_atomwise(delta_p.view(), s.view(), atoms, self.n_atoms)
            } else {
                // mulliken charges
                let dq1 = mulliken_atomwise(dp.view(), s.view(), atoms, self.n_atoms);
                accel.apply(dq.view(), dq1.view()).unwrap()
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
                    delta_p.view(),
                );
            }

            let diff_dq_max: f64 = dq_new.root_mean_sq_err(&dq).unwrap();

            // check if charge difference to the previous iteration is lower than 1e-5
            let converged: bool = if (diff_dq_max < scf_charge_conv)
                && (last_energy - scf_energy).abs() < scf_energy_conv
            {
                true
            } else {
                false
            };
            last_energy = scf_energy;
            dq = dq_new;

            if converged {
                let e_rep: f64 = get_repulsive_energy(&atoms, self.n_atoms, &self.vrep);
                self.properties.set_last_energy(scf_energy + e_rep);
                self.properties.set_p(p);
                self.properties.set_orbs(orbs);
                self.properties.set_orbe(orbe);
                self.properties.set_h_coul_x(h_save);
                break 'scf_loop;
            }
            if !converged && iter == max_iter - 1 {
                println!("Iteration {}", iter);
                println!("Monomer indices: {},{}", self.i, self.j);
                let string: String = String::from("dq.npy");
                write_npy(Path::new(&string), &dq_saved.view());
                write_npy(
                    Path::new(&String::from("delta_dq_saved.npy")),
                    &delta_dq_saved.view(),
                );
                panic!("Pair scc routine does not converge!");
            }
        }
        // only remove the large arrays not the energy or charges
        //self.properties.reset();

        self.properties
            .set_delta_dq(&dq - &self.properties.dq().unwrap());
        self.properties.set_dq(dq);
    }
}
