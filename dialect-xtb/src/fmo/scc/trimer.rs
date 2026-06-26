use dialect_utilities::scc_helpers::aovec_to_aomat;
use dialect_config::Configuration;
use dialect_utilities::fermi_occupation::compute_total_entropy;
use dialect_utilities::linalg::eigh::{compute_s_inv_sqrt, dsyevd_eigh};
use dialect_utilities::mixer::{BroydenMixerNew, Mixer};
use dialect_utilities::mulliken::{
    ao_to_shell_charges, mulliken_aowise_diff, mulliken_atomwise_from_ao_xtb, shell_to_ao_charges,
    shell_to_ao_values,
};
use dialect_utilities::fermi_occupation::fermi_occupation;
use dialect_utilities::scc_helpers::{density_matrix, outer_sum};
use crate::fmo::monomer::XtbMonomer;
use crate::fmo::trimer::XtbTrimer;
use crate::initialization::atom::XtbAtom;
use crate::initialization::basis::create_basis_set;
use crate::integrals::calc_overlap_matrix_parallel;
use crate::parameters::COUL_THIRD_ORDER_ATOM;
use crate::scc::gamma_matrix::gamma_matrix_shell;
use crate::scc::hamiltonian::h0_xtb1_with_cn;
use crate::scc::scc_helpers::{
    calculate_repulsive_energy_xtb, coul_third_order_hamiltonian, create_density_ref,
    get_electronic_energy_xtb_shell,
};
use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray_stats::DeviationExt;

impl XtbTrimer<'_> {
    pub fn prepare_scc(
        &mut self,
        atoms: &[XtbAtom],
        m1: &XtbMonomer,
        m2: &XtbMonomer,
        m3: &XtbMonomer,
        gamma_shell_full: ArrayView2<f64>,
        cn: ArrayView1<f64>,
    ) {
        let has_ghosts = self.n_real_atoms < self.n_atoms;

        // Set n_real_shells if not yet set (first time through without ghosts)
        if self.n_real_shells == 0 && !has_ghosts {
            self.n_real_shells = self.basis().shells.len();
        }

        // get the S and H0 matrices
        let s: Array2<f64> = calc_overlap_matrix_parallel(&self.basis());
        let h0: Array2<f64> = h0_xtb1_with_cn(atoms, s.view(), &self.basis(), cn);

        // build the trimer gamma_shell
        let ns1: usize = m1.n_real_shells;
        let ns2: usize = m2.n_real_shells;
        let ns12: usize = ns1 + ns2;
        let ns_trimer: usize = self.basis().shells.len();

        if has_ghosts {
            // Compute gamma from scratch (includes ghost-real interactions)
            let gamma_shell_tri: Array2<f64> =
                gamma_matrix_shell(&self.gammafunction, atoms, &self.basis());
            self.properties.set_gamma_shell(gamma_shell_tri);
        } else {
            // Assemble from supersystem gamma blocks (real shells only)
            let mut gamma_shell_tri: Array2<f64> = Array2::zeros([ns_trimer, ns_trimer]);

            let gamma_ab_shell: ArrayView2<f64> =
                gamma_shell_full.slice(s![m1.slice.shell, m2.slice.shell]);
            let gamma_ac_shell: ArrayView2<f64> =
                gamma_shell_full.slice(s![m1.slice.shell, m3.slice.shell]);
            let gamma_bc_shell: ArrayView2<f64> =
                gamma_shell_full.slice(s![m2.slice.shell, m3.slice.shell]);

            // off-diagonal blocks
            gamma_shell_tri
                .slice_mut(s![..ns1, ns1..ns12])
                .assign(&gamma_ab_shell);
            gamma_shell_tri
                .slice_mut(s![ns1..ns12, ..ns1])
                .assign(&gamma_ab_shell.t());
            gamma_shell_tri
                .slice_mut(s![..ns1, ns12..])
                .assign(&gamma_ac_shell);
            gamma_shell_tri
                .slice_mut(s![ns12.., ..ns1])
                .assign(&gamma_ac_shell.t());
            gamma_shell_tri
                .slice_mut(s![ns1..ns12, ns12..])
                .assign(&gamma_bc_shell);
            gamma_shell_tri
                .slice_mut(s![ns12.., ns1..ns12])
                .assign(&gamma_bc_shell.t());

            // diagonal blocks
            gamma_shell_tri
                .slice_mut(s![..ns1, ..ns1])
                .assign(&gamma_shell_full.slice(s![m1.slice.shell, m1.slice.shell]));
            gamma_shell_tri
                .slice_mut(s![ns1..ns12, ns1..ns12])
                .assign(&gamma_shell_full.slice(s![m2.slice.shell, m2.slice.shell]));
            gamma_shell_tri
                .slice_mut(s![ns12.., ns12..])
                .assign(&gamma_shell_full.slice(s![m3.slice.shell, m3.slice.shell]));
            self.properties.set_gamma_shell(gamma_shell_tri);
        }

        // build the Vesp contribution using shell-level gamma from supersystem
        // (ESP is from external monomers, uses real shell/orb counts)
        let dim1: usize = m1.n_real_orbs;
        let dim2: usize = m2.n_real_orbs;
        let dim3: usize = m3.n_real_orbs;
        let dim12: usize = dim1 + dim2;
        let m1_dq_shell: ArrayView1<f64> = m1.properties.dq_shell().unwrap();
        let m2_dq_shell: ArrayView1<f64> = m2.properties.dq_shell().unwrap();
        let m3_dq_shell: ArrayView1<f64> = m3.properties.dq_shell().unwrap();

        let gamma_ab_shell: ArrayView2<f64> =
            gamma_shell_full.slice(s![m1.slice.shell, m2.slice.shell]);
        let gamma_ac_shell: ArrayView2<f64> =
            gamma_shell_full.slice(s![m1.slice.shell, m3.slice.shell]);
        let gamma_bc_shell: ArrayView2<f64> =
            gamma_shell_full.slice(s![m2.slice.shell, m3.slice.shell]);

        // Subtract corrections at shell level, then expand to AO
        let esp_shell_1: Array1<f64> = &m1.properties.esp_q().unwrap()
            - &gamma_ab_shell.dot(&m2_dq_shell)
            - &gamma_ac_shell.dot(&m3_dq_shell);
        let esp_shell_2: Array1<f64> = &m2.properties.esp_q().unwrap()
            - &gamma_ab_shell.t().dot(&m1_dq_shell)
            - &gamma_bc_shell.dot(&m3_dq_shell);
        let esp_shell_3: Array1<f64> = &m3.properties.esp_q().unwrap()
            - &gamma_ac_shell.t().dot(&m1_dq_shell)
            - &gamma_bc_shell.t().dot(&m2_dq_shell);

        // Create temporary real-atom bases for shell→AO expansion
        let n_atoms_i = m1.n_real_atoms;
        let n_atoms_j = m2.n_real_atoms;
        let n_atoms_k = m3.n_real_atoms;
        let real_basis_i = create_basis_set(&atoms[..n_atoms_i]);
        let real_basis_j = create_basis_set(&atoms[n_atoms_i..n_atoms_i + n_atoms_j]);
        let real_basis_k =
            create_basis_set(&atoms[n_atoms_i + n_atoms_j..n_atoms_i + n_atoms_j + n_atoms_k]);

        let mut esp = Array1::zeros([self.n_orbs]);
        esp.slice_mut(s![0..dim1]).assign(&shell_to_ao_values(
            &real_basis_i,
            m1.n_real_orbs,
            esp_shell_1.view(),
        ));
        esp.slice_mut(s![dim1..dim12]).assign(&shell_to_ao_values(
            &real_basis_j,
            m2.n_real_orbs,
            esp_shell_2.view(),
        ));
        esp.slice_mut(s![dim12..dim12 + dim3]).assign(&shell_to_ao_values(
            &real_basis_k,
            m3.n_real_orbs,
            esp_shell_3.view(),
        ));
        // Ghost orbs at dim12+dim3.. remain zero (no external ESP on ghosts)
        // and convert it into a matrix in AO basis
        let omega: Array2<f64> = aovec_to_aomat(esp.view(), self.n_orbs);
        // and save in the self properties
        self.properties.set_v(omega * &s * 0.5);
        self.properties.set_h0(h0);
        self.properties.set_s(s);

        // Use stored n_elec (accounts for HOP adjustment) or compute from atoms
        let n_elec: usize = self
            .properties
            .n_elec()
            .unwrap_or_else(|| atoms.iter().fold(0, |n, atom| n + atom.n_elec));
        self.set_mo_indices(n_elec);

        // if this is the first SCC calculation the charge will be taken from the corresponding
        // monomers, padded with zeros for ghost atoms/orbs/shells
        if !self.properties.contains_key("dq") {
            let mut dq = concatenate![
                Axis(0),
                m1.properties.dq().unwrap(),
                m2.properties.dq().unwrap(),
                m3.properties.dq().unwrap()
            ];
            let mut dq_ao = concatenate![
                Axis(0),
                m1.properties.dq_ao().unwrap(),
                m2.properties.dq_ao().unwrap(),
                m3.properties.dq_ao().unwrap()
            ];
            let mut dq_shell = concatenate![
                Axis(0),
                m1.properties.dq_shell().unwrap(),
                m2.properties.dq_shell().unwrap(),
                m3.properties.dq_shell().unwrap()
            ];
            // Pad with zeros for ghost atoms/orbs/shells
            if has_ghosts {
                let mut dq_ext = Array1::zeros(self.n_atoms);
                dq_ext.slice_mut(s![..dq.len()]).assign(&dq);
                dq = dq_ext;
                let mut dq_ao_ext = Array1::zeros(self.n_orbs);
                dq_ao_ext.slice_mut(s![..dq_ao.len()]).assign(&dq_ao);
                dq_ao = dq_ao_ext;
                let mut dq_shell_ext = Array1::zeros(ns_trimer);
                dq_shell_ext.slice_mut(s![..dq_shell.len()]).assign(&dq_shell);
                dq_shell = dq_shell_ext;
            }
            self.properties.set_dq(dq);
            self.properties.set_dq_ao(dq_ao);
            self.properties.set_dq_shell(dq_shell);
        }
        // this is also only needed in the first SCC calculation
        if !self.properties.contains_key("ref_density_matrix") {
            self.properties
                .set_p_ref(create_density_ref(&self.basis(), atoms));
        }

        // in the first SCC calculation the density matrix is set to the reference density matrix
        if !self.properties.contains_key("P") {
            self.properties
                .set_p(self.properties.p_ref().unwrap().to_owned());
        }
    }

    pub fn run_scc(&mut self, atoms: &[XtbAtom], config: &Configuration) {
        // convergence criteria
        let scf_charge_conv: f64 = config.scf.scf_charge_conv;
        let scf_energy_conv: f64 = config.scf.scf_energy_conv;
        let max_iter: usize = config.scf.scf_max_cycles;
        let temperature: f64 = config.scf.electronic_temperature;

        // the properties that are changed during the SCC routine are taken
        // and will be inserted at the end of the SCC routine
        let mut p: Array2<f64> = self.properties.take_p().unwrap();
        let mut dq: Array1<f64> = self.properties.take_dq().unwrap();
        let mut dq_ao: Array1<f64> = self.properties.dq_ao().unwrap().to_owned();
        // Initialize shell-level charges from dq_ao
        let mut dq_shell: Array1<f64> = ao_to_shell_charges(&self.basis(), dq_ao.view());

        // Number of shells for shell-level SCC
        let n_shells = self.basis().shells.len();

        // Initialize shell-level charge mixer (smaller dimension, faster convergence)
        let mut broyden_mixer: BroydenMixerNew = BroydenMixerNew::from_config(n_shells, &config.broyden);

        // molecular properties, we take all properties that are needed from the Properties type
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let h0: ArrayView2<f64> = self.properties.h0().unwrap();
        let p0: ArrayView2<f64> = self.properties.p_ref().unwrap();
        let gamma_shell: ArrayView2<f64> = self.properties.gamma_shell().unwrap();

        // the orbital energies and coefficients can be safely reset, since the
        // Hamiltonian does not depends on the charge differences and not on the orbital coefficients
        let mut orbs: Array2<f64> = Array2::zeros([self.n_orbs, self.n_orbs]);
        let mut orbe: Array1<f64> = Array1::zeros([self.n_orbs]);
        // orbital occupation numbers
        let mut f: Vec<f64> = vec![0.0; self.n_orbs];

        // variables that are updated during the iterations
        let mut last_energy: f64 = 0.0;
        let mut scf_energy: f64 = 0.0;

        // get the repulsive energy (real atoms only, ghosts excluded)
        let rep_energy: f64 = calculate_repulsive_energy_xtb(&atoms[..self.n_real_atoms]);
        // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
        // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
        let x: Array2<f64> = compute_s_inv_sqrt(s);
        // create array of hubbard derivatives
        let mut hubbard_derivatives: Array1<f64> = Array1::zeros(self.n_atoms);
        for (mut val, atom) in hubbard_derivatives.iter_mut().zip(atoms.iter()) {
            *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
        }
        // Use stored n_elec (accounts for HOP adjustment) or compute from atoms
        let n_elec: usize = self
            .properties
            .n_elec()
            .unwrap_or_else(|| atoms.iter().fold(0, |n, atom| n + atom.n_elec));

        // get the Vesp Hamiltonian
        let v: ArrayView2<f64> = self.properties.v().unwrap();
        let mut h_esp: Array2<f64> = &h0 + &v;
        // Add HOP projector if present (covalent fragmentation)
        if let Some(p_hop) = self.properties.hop_projector() {
            h_esp += &p_hop;
        }

        'scf_loop: for i in 0..max_iter {
            // calculate the coulomb part of the Hamiltonian
            // let h_coul: Array2<f64> =
            //     outer_sum(self.properties.gamma_ao().unwrap().dot(&dq_ao).view()) * &s * 0.5;
            let v_shell_iter = gamma_shell.dot(&dq_shell);
            let v_ao_iter = shell_to_ao_values(&self.basis(), self.n_orbs, v_shell_iter.view());
            let h_coul: Array2<f64> = outer_sum(v_ao_iter.view()) * &s * 0.5;
            // calculate the third order coulomb Hamiltonian
            let h_coul_third_order: Array2<f64> =
                coul_third_order_hamiltonian(hubbard_derivatives.view(), dq.view(), &self.basis())
                    * &s
                    * 0.5;
            // add the parts of the Hamiltonian
            let h_full: Array2<f64> = &h_esp + &h_coul - &h_coul_third_order;
            // H' = X^t.H.X
            // Full Hamiltonian = H0 + H_pert
            let h_ortho = x.t().dot(&h_full).dot(&x);
            // Full eigenvalue decomposition in orthogonal basis (direct LAPACK call)
            let (evals, evecs) = dsyevd_eigh(h_ortho.view());
            orbe = evals;

            // C = X * C' (back-transformation to original basis)
            orbs = x.dot(&evecs);

            // compute the fermi orbital occupation
            let tmp: (f64, Vec<f64>) = fermi_occupation(orbe.view(), n_elec, temperature);
            f = tmp.1;

            // calculate the density matrix
            p = density_matrix(orbs.view(), &f[..]);

            // Compute mulliken charges directly from p - p0 (avoids creating dp array)
            let dq_ao_new = mulliken_aowise_diff(p.view(), p0, s);
            // Convert to shell-level charges for mixing
            let dq_shell_new = ao_to_shell_charges(&self.basis(), dq_ao_new.view());

            // Charge difference at shell level
            let delta_dq_shell: Array1<f64> = &dq_shell_new - &dq_shell;

            // Compute atomic charges from OUTPUT shell charges (before mixing) for energy
            let dq_ao_out = shell_to_ao_charges(&self.basis(), self.n_orbs, dq_shell_new.view());
            let dq_atom_out: Array1<f64> =
                mulliken_atomwise_from_ao_xtb(&self.basis(), self.n_atoms, dq_ao_out.view());

            // compute electronic energy using OUTPUT charges (before mixing)
            // Add entropy contribution computed separately for alpha/beta channels
            let entropy = compute_total_entropy(orbe.view(), n_elec, temperature);
            scf_energy = get_electronic_energy_xtb_shell(
                p.view(),
                h0.view(),
                dq_atom_out.view(),
                dq_shell_new.view(),
                gamma_shell,
                hubbard_derivatives.view(),
            ) + entropy;

            // Broyden mixing at shell level (smaller dimension, faster convergence)
            dq_shell = broyden_mixer.next(&dq_shell, &delta_dq_shell);

            // Convert mixed shell charges to AO level for next iteration
            dq_ao = shell_to_ao_charges(&self.basis(), self.n_orbs, dq_shell.view());
            let dq_new: Array1<f64> =
                mulliken_atomwise_from_ao_xtb(&self.basis(), self.n_atoms, dq_ao.view());

            // convergence
            let diff_dq_max: f64 = dq_new.root_mean_sq_err(&dq).unwrap();
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

            if converged {
                let total_energy = scf_energy + rep_energy;
                self.properties.set_last_energy(total_energy);
                // Only store heavy properties needed for gradient calculations
                if config.jobtype != "sp" {
                    self.properties.set_p(p);
                    self.properties.set_orbs(orbs);
                    self.properties.set_orbe(orbe);
                    self.properties.set_occupation(f);
                }
                break 'scf_loop;
            }
            if !converged && i == max_iter - 1 {
                log::debug!("Iteration {}", i);
                log::debug!("Monomer indices: {},{},{}", self.i, self.j, self.k);
                panic!("Trimer scc routine does not converge!");
            }
        }
        self.properties
            .set_delta_dq(&dq_ao - &self.properties.dq_ao().unwrap());
        self.properties
            .set_delta_dq_shell(&dq_shell - &self.properties.dq_shell().unwrap());
        self.properties.set_dq(dq);
        self.properties.set_dq_ao(dq_ao);
        self.properties.set_dq_shell(dq_shell);
    }
}
