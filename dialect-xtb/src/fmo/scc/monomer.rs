use dialect_config::Configuration;
use dialect_utilities::linalg::eigh::{compute_s_inv_sqrt, dsyevd_eigh};
use dialect_utilities::mixer::BroydenMixerNew;
use dialect_utilities::mulliken::{
    ao_to_shell_charges, mulliken_aowise, mulliken_atomwise_from_ao_xtb, shell_to_ao_values,
};
use dialect_utilities::fermi_occupation::fermi_occupation;
use dialect_utilities::scc_helpers::{density_matrix, outer_sum};
use crate::fmo::monomer::XtbMonomer;
use crate::initialization::atom::XtbAtom;
use crate::integrals::calc_overlap_matrix_parallel;
use crate::parameters::COUL_THIRD_ORDER_ATOM;
// use crate::scc::eeq::compute_eeq_charges;
use crate::scc::gamma_matrix::{gamma_matrix_shell, gamma_matrix_xtb_new};
use crate::scc::hamiltonian::h0_xtb1_with_cn;
use dialect_utilities::fermi_occupation::compute_total_entropy;
use crate::scc::scc_helpers::{
    coul_third_order_hamiltonian, create_density_ref, get_electronic_energy_xtb,
    get_electronic_energy_xtb_shell,
};
use ndarray::prelude::*;
// use ndarray_linalg::{Eigh, Inverse, SymmetricSqrt, UPLO};
use ndarray_stats::DeviationExt;

impl XtbMonomer<'_> {
    pub fn prepare_scc(&mut self, atoms: &[XtbAtom], cn: ArrayView1<f64>, broyden_config: &dialect_config::settings::BroydenConfig) {
        // Build extended atom list (real + ghost boundary atoms) and CN
        let ext_atoms: Vec<XtbAtom> = atoms
            .iter()
            .chain(self.ghost_atoms.iter())
            .cloned()
            .collect();
        let mut ext_cn: Array1<f64> = Array1::zeros(ext_atoms.len());
        ext_cn.slice_mut(s![..atoms.len()]).assign(&cn);
        // Ghost atoms get CN=0 (minimal coordination number)

        // get the S and H0 matrices (using extended atom list for ghost interactions)
        let s: Array2<f64> = calc_overlap_matrix_parallel(&self.basis);
        let h0: Array2<f64> = h0_xtb1_with_cn(&ext_atoms, s.view(), &self.basis, ext_cn.view());

        // calculate the gamma matrix (using extended atoms)
        // let g0: Array2<f64> = gamma_matrix_xtb_new(&self.gammafunction, atoms, &self.basis);
        let g0: Array2<f64> = gamma_matrix_shell(&self.gammafunction, &ext_atoms, &self.basis);

        // calculate X matrix, where X = S^(-1/2)
        let x: Array2<f64> = compute_s_inv_sqrt(s.view());
        // and save it in the self properties
        self.properties.set_h0(h0);
        self.properties.set_s(s);
        self.properties.set_x(x);
        self.properties.set_gamma_shell(g0);
        // self.properties.set_gamma_ao(g0);

        // n_elec includes ghost atom electrons (gives even count for covalent fragments)
        let n_elec: usize = self
            .properties
            .n_elec()
            .unwrap_or_else(|| ext_atoms.iter().fold(0, |n, atom| n + atom.n_elec));

        // Set the indices of the occupied and virtual orbitals based on the number of electrons.
        self.set_mo_indices(n_elec);

        // if this is the first SCC calculation the charge differences will be initialized to zeros
        if !self.properties.contains_key("dq") {
            self.properties.set_dq(Array1::zeros(self.n_atoms));
        }
        if !self.properties.contains_key("dq_ao") {
            self.properties.set_dq_ao(Array1::zeros(self.n_orbs));
        }
        if !self.properties.contains_key("dq_shell") {
            self.properties
                .set_dq_shell(Array1::zeros(self.basis.shells.len()));
        }
        // Reference density with ZREF approach:
        // 1. BDA atoms: reduce reference occupation by 1 (bonding electron removed)
        // 2. Ghost atoms: zero reference occupation (no electrons, just basis + potential)
        if !self.properties.contains_key("ref_density_matrix") {
            let mut p_ref = create_density_ref(&self.basis, &ext_atoms);
            // Reduce BDA atoms' reference density by 1 electron (ZREF -= 1)
            for &bda_local in &self.bda_local_indices {
                let bda_n_elec: f64 = ext_atoms[bda_local].n_elec as f64;
                if bda_n_elec > 0.0 {
                    let scale = (bda_n_elec - 1.0) / bda_n_elec;
                    for shell in self.basis.shells.iter() {
                        if shell.atom_index == bda_local {
                            for i in shell.sph_start..shell.sph_end {
                                p_ref[[i, i]] *= scale;
                            }
                        }
                    }
                }
            }
            // Zero ghost atom reference occupations (no electrons)
            if !self.ghost_atoms.is_empty() {
                for shell in self.basis.shells.iter() {
                    if shell.atom_index >= self.n_real_atoms {
                        for i in shell.sph_start..shell.sph_end {
                            p_ref[[i, i]] = 0.0;
                        }
                    }
                }
            }
            self.properties.set_p_ref(p_ref);
        }
        // in the first SCC calculation the density matrix is set to the reference density matrix
        if !self.properties.contains_key("P") {
            self.properties
                .set_p(self.properties.p_ref().unwrap().to_owned());
        }

        // // Number of shells for shell-level SCC
        // let n_shells = self.basis.shells.len();
        // // initialize the charge mixer (using BroydenMixerNew for better convergence)
        let broyden_mixer: BroydenMixerNew = BroydenMixerNew::from_config(self.n_orbs, broyden_config);
        self.properties.set_mixer_new(broyden_mixer);
    }

    pub fn scc_step(
        &mut self,
        atoms: &[XtbAtom],
        v_esp: Array2<f64>,
        config: &Configuration,
    ) -> bool {
        // Build extended atom list (real + ghost boundary atoms)
        let ext_atoms: Vec<XtbAtom> = atoms
            .iter()
            .chain(self.ghost_atoms.iter())
            .cloned()
            .collect();

        // get the mixer (using BroydenMixerNew for better convergence)
        let mut broyden_mixer: BroydenMixerNew = self.properties.take_mixer_new().unwrap();
        // convergence criteria
        let scf_charge_conv: f64 = config.scf.scf_charge_conv;
        let scf_energy_conv: f64 = config.scf.scf_energy_conv;
        // charges
        let dq: Array1<f64> = self.properties.take_dq().unwrap();
        let dq_ao: Array1<f64> = self.properties.dq_ao().unwrap().to_owned();
        // let dq_shell: Array1<f64> = ao_to_shell_charges(&self.basis, dq_ao.view());
        // necessary matrices, X, S, H0 and P0
        let x: ArrayView2<f64> = self.properties.x().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let h0: ArrayView2<f64> = self.properties.h0().unwrap();
        let p0: ArrayView2<f64> = self.properties.p_ref().unwrap();
        let gamma_shell: ArrayView2<f64> = self.properties.gamma_shell().unwrap();
        // get the last energy
        let last_energy: f64 = self.properties.last_energy().unwrap();
        // electronic temperature
        let temperature: f64 = config.scf.electronic_temperature;
        // n_elec includes ghost atom electrons (gives even count for covalent fragments)
        let n_elec: usize = self
            .properties
            .n_elec()
            .unwrap_or_else(|| ext_atoms.iter().fold(0, |n, atom| n + atom.n_elec));
        // create array of hubbard derivatives (using extended atoms)
        let mut hubbard_derivatives: Array1<f64> = Array1::zeros(self.n_atoms);
        for (mut val, atom) in hubbard_derivatives.iter_mut().zip(ext_atoms.iter()) {
            *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
        }

        // calculate the coulomb part of the Hamiltonian
        // let h_coul: Array2<f64> =
        //     outer_sum(self.properties.gamma_ao().unwrap().dot(&dq_ao).view()) * &s * 0.5;
        let h_coul_vesp: Array2<f64> = v_esp * &s * 0.5;
        // calculate the third order coulomb Hamiltonian
        let h_coul_third_order: Array2<f64> =
            coul_third_order_hamiltonian(hubbard_derivatives.view(), dq.view(), &self.basis)
                * &s
                * 0.5;
        // add the parts of the Hamiltonian
        let mut h: Array2<f64> = &h0 + &h_coul_vesp - &h_coul_third_order;
        // Add HOP projector if present (covalent fragmentation)
        if let Some(p_hop) = self.properties.hop_projector() {
            h += &p_hop;
        }
        // H' = X^t.H.X
        let h = x.t().dot(&h).dot(&x);
        let tmp: (Array1<f64>, Array2<f64>) = dsyevd_eigh(h.view());
        let orbe = tmp.0;
        // C = X.C'
        let orbs = x.dot(&tmp.1);

        // compute the fermi orbital occupation
        let tmp: (f64, Vec<f64>) = fermi_occupation(orbe.view(), n_elec, temperature);
        let f = tmp.1;

        // calculate the density matrix
        let p = density_matrix(orbs.view(), &f[..]);
        // Compute the difference density matrix. This will be mixed in case of long-range correction.
        let dp: Array2<f64> = &p - &p0;

        // mulliken charges
        let dq1 = mulliken_aowise(dp.view(), s.view());
        // Charge difference to previous iteration
        let delta_dq: Array1<f64> = &dq1 - &dq_ao;
        // Broyden mixing of Mulliken charges per orbital (BroydenMixerNew takes references)
        let dq_ao_new = broyden_mixer.next(&dq_ao, &delta_dq);
        // let dq_new_ao: Array1<f64> = accel.apply(dq_ao.view(), dq1.view()).unwrap();
        let dq_new: Array1<f64> =
            mulliken_atomwise_from_ao_xtb(&self.basis, self.n_atoms, dq_ao_new.view());

        // compute electronic energy
        let dq_shell_new = ao_to_shell_charges(&self.basis, dq_ao_new.view());
        let scf_energy = get_electronic_energy_xtb_shell(
            p.view(),
            h0.view(),
            dq_new.view(),
            // dq_ao_new.view(),
            dq_shell_new.view(),
            // self.properties.gamma_ao().unwrap(),
            self.properties.gamma_shell().unwrap(),
            hubbard_derivatives.view(),
        ) + compute_total_entropy(orbe.view(), n_elec, temperature);

        // convergence check
        let diff_dq_max: f64 = dq_new.root_mean_sq_err(&dq).unwrap();
        // check if charge difference to the previous iteration is lower than threshold
        let conv_charge: bool = diff_dq_max < scf_charge_conv;
        // same check for the electronic energy
        let conv_energy: bool = (last_energy - scf_energy).abs() < scf_energy_conv;
        // save to property
        self.properties.set_orbs(orbs);
        self.properties.set_orbe(orbe);
        self.properties.set_p(p);
        self.properties.set_dq(dq_new);
        self.properties.set_dq_shell(dq_shell_new);
        self.properties.set_dq_ao(dq_ao_new);
        self.properties.set_mixer_new(broyden_mixer);
        self.properties.set_last_energy(scf_energy);
        self.properties.set_occupation(f);

        // scc (for one fragment) is converged if both criteria are passed
        conv_charge && conv_energy
    }
}
