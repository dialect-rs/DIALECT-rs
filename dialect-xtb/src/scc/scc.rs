use dialect_utilities::fermi_occupation::compute_channel_entropy;
use dialect_utilities::linalg::eigh::{compute_s_inv_sqrt, dsyevd_eigh};
use dialect_utilities::mixer::{BroydenMixerNew, Mixer};
use dialect_utilities::scc_interface::{RestrictedSCC, SCCError};
use crate::scc::eeq::compute_eeq_charges;
use crate::scc::halogen_correction::get_halogen_correction;
use crate::scc::scc_helpers::calculate_repulsive_energy_xtb;
use dialect_config::settings::MixConfig;
use dialect_utilities::fermi_occupation::fermi_occupation_single;
use dialect_utilities::mulliken::{
    ao_to_shell_charges, mulliken_aowise_diff, mulliken_atomwise_from_ao_xtb,
    shell_to_ao_charges, shell_to_ao_values,
};
use dialect_utilities::scc_helpers::density_matrix;
use dialect_utilities::scc_logging::{print_energies_at_iteration, print_scc_end, print_scc_end_xtb, print_scc_init};
use dialect_base::Timer;
use crate::{
        initialization::system::XtbSystem,
        parameters::{COUL_THIRD_ORDER_ATOM, REFERENCE_OCCUPATION},
        scc::{
            gamma_matrix::{gamma_matrix_shell, gamma_matrix_xtb, gamma_matrix_xtb_new},
            scc_helpers::{
                build_perturbation_hamiltonian, create_density_ref, get_dispersion_energy_xtb,
                get_electronic_energy_xtb_shell,
            },
        },
};
use log::{log_enabled, Level};
use ndarray::prelude::*;
use ndarray_linalg::{Eigh, Inverse};
use ndarray_stats::DeviationExt;
use std::fmt;
use std::time::Instant;

impl<'a> RestrictedSCC for XtbSystem {
    fn prepare_scc(&mut self) {
        // calculate s and h0
        self.get_overlap();
        self.get_h0();

        // Calculate shell-shell gamma matrix for shell-level SCC
        let jmat: Array2<f64> = gamma_matrix_shell(&self.gammafunction, &self.atoms, &self.basis);
        self.properties.set_gamma_shell(jmat);

        // this is also only needed in the first SCC calculation
        if !self.properties.contains_key("ref_density_matrix") {
            self.properties
                .set_p_ref(create_density_ref(&self.basis, &self.atoms));
        }

        // Compute and cache the Loewdin orthogonalization matrix X = S^(-1/2)
        // Using direct LAPACK eigenvalue decomposition (optimized)
        if !self.properties.contains_key("X") {
            let s = self.properties.s().unwrap();
            let x: Array2<f64> = compute_s_inv_sqrt(s);
            self.properties.set_x(x);
        }

        // Initialize charges using EEQ (electronegativity equilibration)
        if !self.properties.contains_key("dq") || !self.properties.contains_key("dq_ao") {
            let total_charge = self.config.mol.charge as f64;

            // Compute EEQ atomic charges
            // Negate to match our internal sign convention (positive = more electrons)
            // EEQ convention: positive = electron-deficient
            // Our convention: positive = more electrons than reference
            let eeq_raw: Array1<f64> = compute_eeq_charges(&self.atoms, total_charge);
            let dq_atom: Array1<f64> = -&eeq_raw;

            // Distribute atomic charges to shells/AOs using reference occupation weighting (like iniqshell)
            // Polarization shells get 0 charge - only valence shells participate
            let mut total_ref_occ_per_atom: Vec<f64> = vec![0.0; self.n_atoms];
            for shell in self.basis.shells.iter() {
                if shell.polarization {
                    continue; // Skip polarization shells
                }
                let atom_idx = shell.atom_index;
                let atom_z = self.atoms[atom_idx].number as usize - 1;
                let l = shell.angular_momentum;
                total_ref_occ_per_atom[atom_idx] += REFERENCE_OCCUPATION[atom_z][l];
            }

            let mut dq_ao_init: Array1<f64> = Array1::zeros(self.n_orbs);
            for shell in self.basis.shells.iter() {
                if shell.polarization {
                    continue; // Polarization shells get 0 charge
                }
                let atom_idx = shell.atom_index;
                let atom_z = self.atoms[atom_idx].number as usize - 1;
                let l = shell.angular_momentum;
                let n_orb_in_shell = shell.sph_end - shell.sph_start;

                let ref_occ = REFERENCE_OCCUPATION[atom_z][l];
                let total_ref_occ = total_ref_occ_per_atom[atom_idx];

                let shell_charge = if total_ref_occ > 1e-10 {
                    dq_atom[atom_idx] * ref_occ / total_ref_occ
                } else {
                    0.0
                };

                let charge_per_orb = shell_charge / n_orb_in_shell as f64;
                for orb_idx in shell.sph_start..shell.sph_end {
                    dq_ao_init[orb_idx] = charge_per_orb;
                }
            }

            self.properties.set_dq(dq_atom);
            self.properties.set_dq_ao(dq_ao_init);
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
        let temperature: f64 = self.config.scf.electronic_temperature;

        // the properties that are changed during the SCC routine are taken
        // and will be inserted at the end of the SCC routine
        let mut p: Array2<f64> = self.properties.take_p().unwrap();
        let mut dq: Array1<f64> = self.properties.take_dq().unwrap();
        let mut dq_ao: Array1<f64> = self.properties.dq_ao().unwrap().to_owned();

        // Number of shells for shell-level SCC
        let n_shells = self.basis.shells.len();

        // Initialize shell-level charge mixer (smaller dimension, faster convergence)
        let mut broyden_mixer: BroydenMixerNew =
            BroydenMixerNew::from_config(n_shells, &self.config.broyden);

        // Initialize shell-level charges from dq_ao
        let mut dq_shell: Array1<f64> = ao_to_shell_charges(&self.basis, dq_ao.view());

        // molecular properties, we take all properties that are needed from the Properties type
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let h0: ArrayView2<f64> = self.properties.h0().unwrap();
        let p0: ArrayView2<f64> = self.properties.p_ref().unwrap();
        // let gamma_ao: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let gamma_shell: ArrayView2<f64> = self.properties.gamma_shell().unwrap();

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
        let rep_energy: f64 = calculate_repulsive_energy_xtb(&self.atoms);

        if log_enabled!(Level::Info) {
            print_scc_init(max_iter, temperature, rep_energy);
        }
        // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
        // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
        // X matrix was computed and cached in prepare_scc
        let x: ArrayView2<f64> = self.properties.x().unwrap();
        // get the dispersion energy
        let e_disp: f64 = get_dispersion_energy_xtb(&self.atoms, &self.config);

        // create array of hubbard derivatives
        let mut hubbard_derivatives: Array1<f64> = Array1::zeros(self.n_atoms);
        for (mut val, atom) in hubbard_derivatives.iter_mut().zip(self.atoms.iter()) {
            *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
        }
        // calculate the halogen correction
        let halogen_correction: f64 = get_halogen_correction(&self.atoms);
        let mut n_iter: usize = 0;

        // Track dq_diff for convergence
        let mut last_dq_diff: f64 = 1.0;

        'scf_loop: for i in 0..max_iter {
            n_iter = i + 1;

            // Build the Hamiltonian in the orthogonal basis
            // Calculate shell-level electrostatic potential: v_shell = jmat @ qsh
            let v_shell = gamma_shell.dot(&dq_shell);

            // Expand to AO level: each orbital gets its shell's potential (no division)
            let gamma_dq = shell_to_ao_values(&self.basis, self.n_orbs, v_shell.view());
            // Build perturbation Hamiltonian: h_coul - h_coul_third_order
            let h_pert: Array2<f64> = build_perturbation_hamiltonian(
                gamma_dq.view(),
                hubbard_derivatives.view(),
                dq.view(),
                &self.basis,
                s,
            );
            // Full Hamiltonian = H0 + H_pert
            let h_full: Array2<f64> = &h0 + &h_pert;

            // Transform to orthogonal basis: H' = X^T * H * X
            let h_ortho = x.t().dot(&h_full).dot(&x);

            // Full eigenvalue decomposition in orthogonal basis (direct LAPACK call)
            let (evals, evecs) = dsyevd_eigh(h_ortho.view());
            orbe = evals;

            // C = X * C' (back-transformation to original basis)
            orbs = x.dot(&evecs);

            // Compute the (spin-restricted open-shell) orbital occupation.
            // `nopen` is the number of unpaired electrons: an explicitly
            // requested multiplicity fixes it, otherwise it defaults to the
            // minimal open shell (0 for an even electron count, 1 for an odd
            // one). The alpha and beta channels are filled separately over the
            // same spatial orbitals and summed
            // (this also supports forced higher multiplicities, e.g. triplets).
            let nopen: usize = if self.config.mol.multiplicity > 1 {
                (self.config.mol.multiplicity - 1) as usize
            } else {
                self.n_elec % 2
            };
            let n_alpha: usize = (self.n_elec + nopen) / 2;
            let n_beta: usize = (self.n_elec - nopen) / 2;
            let (_, focc_a) = fermi_occupation_single(orbe.view(), n_alpha, temperature);
            let (_, focc_b) = fermi_occupation_single(orbe.view(), n_beta, temperature);
            f = focc_a
                .iter()
                .zip(focc_b.iter())
                .map(|(a, b)| a + b)
                .collect();

            // Gap-dependent damping: reduce Broyden alpha when HOMO-LUMO gap is small.
            // Disabled: the gap condition ("iter == 0") is never true since the loop
            // starts at iter = 1, so alpha effectively stays at 0.40.
            // calculate the density matrix
            p = density_matrix(orbs.view(), &f[..]);

            // Compute mulliken charges directly from p - p0 (avoids creating dp array)
            let dq_ao_new = mulliken_aowise_diff(p.view(), p0, s);
            // Convert to shell-level charges for mixing
            let dq_shell_new = ao_to_shell_charges(&self.basis, dq_ao_new.view());

            // Charge difference at shell level
            let delta_dq_shell: Array1<f64> = &dq_shell_new - &dq_shell;

            // Compute atomic charges from OUTPUT shell charges (before mixing) for energy
            let dq_ao_out = shell_to_ao_charges(&self.basis, self.n_orbs, dq_shell_new.view());
            let dq_atom_out: Array1<f64> =
                mulliken_atomwise_from_ao_xtb(&self.basis, self.n_atoms, dq_ao_out.view());

            // compute electronic energy using OUTPUT charges (before mixing)
            // Add entropy contribution computed separately for alpha/beta channels
            let entropy = compute_channel_entropy(orbe.view(), n_alpha, temperature)
                + compute_channel_entropy(orbe.view(), n_beta, temperature);

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
            dq_ao = shell_to_ao_charges(&self.basis, self.n_orbs, dq_shell.view());
            let dq_new: Array1<f64> =
                mulliken_atomwise_from_ao_xtb(&self.basis, self.n_atoms, dq_ao.view());

            // RMSdq: sqrt(sum(delta_dq_shell^2) / n_atoms)
            // Note: sum over shells but divide by n_atoms (not n_shells)
            let diff_dq_max: f64 =
                (delta_dq_shell.mapv(|x| x * x).sum() / self.n_atoms as f64).sqrt();
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
            last_dq_diff = diff_dq_max;
            dq = dq_new;

            if converged {
                total_energy = Ok(scf_energy + rep_energy + e_disp + halogen_correction);
                break 'scf_loop;
            }
            total_energy = Err(SCCError::new(i, last_energy - scf_energy, diff_dq_max));
        }

        if log_enabled!(Level::Info) && total_energy.is_ok() {
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
        self.properties.set_dq_shell(dq_shell);
        self.properties.set_dq_ao(dq_ao.clone());

        // Debug: print Ti/O shell charges (shells 112-116 in 0-based indexing)
        self.properties
            .set_last_energy(total_energy.clone().unwrap());
        return total_energy;
    }
}

#[cfg(test)]
mod tests {
    use crate::initialization::system::XtbSystem;
    use dialect_config::Configuration;
    use dialect_utilities::scc_interface::RestrictedSCC;

    /// Tolerance for the regression comparison (Hartree).
    const EPSILON: f64 = 1.0e-8;

    /// GFN1-xTB total energies (Hartree) computed with dialect-xtb itself, used
    /// as regression references. They are produced with a tightly converged SCC
    /// (charge/energy convergence 1e-10) at an electronic temperature of 300 K.
    const REFERENCE_ENERGIES: [(&str, f64); 6] = [
        ("h2o", -5.76839158742582),
        ("benzene", -15.89422885016334),
        ("ammonia", -4.81771458535579),
        ("uracil", -26.41709783454642),
        // PCl3 exercises the d-d overlap (spherical d transform).
        ("pcl3", -15.51737349732025),
        // TiCl4: closed-shell transition-metal d0 with Ti-Cl d-d overlap.
        ("ticl4", -18.48388326941341),
    ];

    /// Build a GFN1-xTB configuration with a tightly converged SCC.
    fn xtb_config() -> Configuration {
        let mut config: Configuration = toml::from_str("").unwrap();
        config.tight_binding.use_dftb = false;
        config.tight_binding.use_xtb1 = true;
        config.scf.electronic_temperature = 300.0;
        config.scf.scf_charge_conv = 1.0e-10;
        config.scf.scf_energy_conv = 1.0e-10;
        config
    }

    /// Run a GFN1-xTB single point for the molecule with the given name, using
    /// the geometry from the workspace `tests/data/<name>/<name>.xyz` file.
    fn run_single_point(name: &str) -> f64 {
        let path: String = format!(
            "{}/../tests/data/{}/{}.xyz",
            env!("CARGO_MANIFEST_DIR"),
            name,
            name
        );
        let mut system: XtbSystem = XtbSystem::from((path.as_str(), xtb_config()));
        system.prepare_scc();
        system.run_scc().unwrap()
    }

    /// Regression test: the GFN1-xTB single-point energies of a few small
    /// molecules must match the reference values computed with dialect-xtb.
    #[test]
    fn gfn1_xtb_single_point_energies() {
        for (name, energy_ref) in REFERENCE_ENERGIES.iter() {
            let energy: f64 = run_single_point(name);
            assert!(
                (energy - energy_ref).abs() < EPSILON,
                "Molecule: {}, reference energy {:.14}, computed energy {:.14}",
                name,
                energy_ref,
                energy
            );
        }
    }
}
