use crate::excited_states::tda::*;
use crate::excited_states::{orbe_differences, trans_charges, trans_charges_restricted};
use crate::initialization::System;
use ndarray::prelude::*;

impl System {
    pub fn calculate_excited_states(&mut self, print_states: bool) {
        // restrict active orbitals in TDA-TD-DFTB calculation
        if self.config.tddftb.restrict_active_orbitals {
            self.prepare_tda_restricted();

            if self.config.excited.use_casida {
                self.run_casida_restricted(
                    self.config.excited.nstates,
                    self.config.excited.davidson_iterations,
                    self.config.excited.davidson_convergence,
                    self.config.excited.davidson_subspace_multiplier,
                    print_states,
                );
            } else {
                self.run_tda_restricted(
                    self.config.excited.nstates,
                    self.config.excited.davidson_iterations,
                    self.config.excited.davidson_convergence,
                    self.config.excited.davidson_subspace_multiplier,
                );
            }
        } else {
            // use the full active space
            self.prepare_tda();

            if self.config.excited.use_casida {
                self.run_casida(
                    self.config.excited.nstates,
                    self.config.excited.davidson_iterations,
                    self.config.excited.davidson_convergence,
                    self.config.excited.davidson_subspace_multiplier,
                    print_states,
                );
            } else {
                self.run_tda(
                    self.config.excited.nstates,
                    self.config.excited.davidson_iterations,
                    self.config.excited.davidson_convergence,
                    self.config.excited.davidson_subspace_multiplier,
                    print_states,
                );
            }
        }
        if self.config.tddftb.save_transition_densities {
            for state in self.config.tddftb.states_to_analyse.iter() {
                self.save_transition_density(*state);
            }
        }
    }

    pub fn prepare_tda(&mut self) {
        if self.properties.q_ov().is_none() {
            let (qov, qoo, qvv): (Array2<f64>, Array2<f64>, Array2<f64>) = trans_charges(
                self.n_atoms,
                &self.atoms,
                self.properties.orbs().unwrap(),
                self.properties.s().unwrap(),
                &self.occ_indices,
                &self.virt_indices,
            );
            // And stored in the properties HashMap.
            self.properties.set_q_oo(qoo);
            self.properties.set_q_ov(qov);
            self.properties.set_q_vv(qvv);
        }

        if self.properties.omega().is_none() {
            // Reference to the orbital energies.
            // Check if the orbital energy differences were already computed.
            let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();

            // The index of the HOMO (zero based).
            let homo: usize = self.occ_indices[self.occ_indices.len() - 1];

            // The index of the LUMO (zero based).
            let lumo: usize = self.virt_indices[0];

            // Energies of the occupied orbitals.
            let orbe_occ: ArrayView1<f64> = orbe.slice(s![0..homo + 1]);

            // Energies of the virtual orbitals.
            let orbe_virt: ArrayView1<f64> = orbe.slice(s![lumo..]);

            // Energy differences between virtual and occupied orbitals.
            let omega: Array1<f64> = orbe_differences(orbe_occ, orbe_virt);

            // Energy differences are stored in the molecule.
            self.properties.set_omega(omega);

            self.properties.set_homo(homo);
            self.properties.set_lumo(lumo);
        }

        // limit number of excited statess
        let nocc: usize = self.properties.occ_indices().unwrap().len();
        let nvirt: usize = self.properties.occ_indices().unwrap().len();
        let max_excited_states: usize = nocc * nvirt;
        if self.config.excited.nstates > max_excited_states {
            println!("Exceeded number of possible excited states! Number of excited states was set to the maximum limit.");
            self.config.excited.nstates = max_excited_states;
        }
    }

    pub fn prepare_tda_restricted(&mut self) {
        if self.properties.q_ov().is_none() {
            let (qov, qoo, qvv): (Array2<f64>, Array2<f64>, Array2<f64>) = trans_charges_restricted(
                self.n_atoms,
                &self.atoms,
                self.properties.orbs().unwrap(),
                self.properties.s().unwrap(),
                &self.occ_indices,
                &self.virt_indices,
                self.config.tddftb.active_orbital_threshold,
            );
            // And stored in the properties HashMap.
            self.properties.set_q_oo(qoo);
            self.properties.set_q_ov(qov);
            self.properties.set_q_vv(qvv);
        }

        if self.properties.omega().is_none() {
            // Reference to the orbital energies.
            // Check if the orbital energy differences were already computed.
            let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();

            // The index of the HOMO (zero based).
            let homo: usize = self.occ_indices[self.occ_indices.len() - 1];

            // The index of the LUMO (zero based).
            let lumo: usize = self.virt_indices[0];

            // Energies of the occupied orbitals.
            let mut orbe_occ: ArrayView1<f64> = orbe.slice(s![0..homo + 1]);

            // Energies of the virtual orbitals.
            let mut orbe_virt: ArrayView1<f64> = orbe.slice(s![lumo..]);

            let dim_o: usize = (self.occ_indices.len() as f64
                * self.config.tddftb.active_orbital_threshold)
                as usize;
            let dim_v: usize = (self.virt_indices.len() as f64
                * self.config.tddftb.active_orbital_threshold)
                as usize;

            orbe_occ = orbe.slice(s![homo + 1 - dim_o..homo + 1]);
            orbe_virt = orbe.slice(s![lumo..lumo + dim_v]);

            // limit number of excited statess
            let max_excited_states: usize = dim_o * dim_v;
            if self.config.excited.nstates > max_excited_states {
                println!("Exceeded number of possible excited states! Number of excited states was set to the maximum limit.");
                self.config.excited.nstates = max_excited_states;
            }

            // Energy differences between virtual and occupied orbitals.
            let omega: Array1<f64> = orbe_differences(orbe_occ, orbe_virt);

            // Energy differences are stored in the molecule.
            self.properties.set_omega(omega);

            self.properties.set_homo(homo);
            self.properties.set_lumo(lumo);
        }
    }

    pub fn save_transition_density(&self, state: usize) {
        // get occupied and virtual orbitals
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let n_occ: usize = self.properties.occ_indices().unwrap().len();
        let occ_orbs: ArrayView2<f64> = orbs.slice(s![.., ..n_occ]);
        let virt_orbs: ArrayView2<f64> = orbs.slice(s![.., n_occ..]);

        // calculate transition density in AO basis
        let density_mo: ArrayView2<f64> = self.properties.tdm(state).unwrap();
        let density_ao: Array2<f64> = occ_orbs.dot(&density_mo.dot(&virt_orbs.t()));
        // filename
        let txt: String = format!("transition_density_{}.npy", state);

        // write to numpy file
        write_npy(txt, &density_ao);
    }
}
