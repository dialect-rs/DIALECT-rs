use crate::excited_states::{
    orbe_differences, trans_charges, trans_charges_ao, trans_charges_restricted,
};
use crate::fmo::Monomer;
use crate::initialization::Atom;
use crate::Configuration;
use ndarray::prelude::*;

impl Monomer<'_> {
    pub fn prepare_tda(&mut self, atoms: &[Atom], config: &Configuration) {
        let occ_indices: &[usize] = self.properties.occ_indices().unwrap();
        let virt_indices: &[usize] = self.properties.virt_indices().unwrap();
        let nocc: usize = occ_indices.len();
        let nvirt: usize = virt_indices.len();

        // The index of the HOMO (zero based).
        let homo: usize = occ_indices[occ_indices.len() - 1];

        // The index of the LUMO (zero based).
        let lumo: usize = virt_indices[0];

        if config.tddftb.restrict_active_orbitals {
            if self.properties.q_ov().is_none() {
                let (qov, qoo, qvv): (Array2<f64>, Array2<f64>, Array2<f64>) =
                    trans_charges_restricted(
                        self.n_atoms,
                        atoms,
                        self.properties.orbs().unwrap(),
                        self.properties.s().unwrap(),
                        occ_indices,
                        virt_indices,
                        config.tddftb.active_orbital_threshold,
                    );
                // And stored in the properties HashMap.
                self.properties.set_q_oo_restricted(qoo.clone());
                self.properties.set_q_vv_restricted(qvv.clone());
                self.properties.set_q_oo(qoo);
                self.properties.set_q_ov(qov);
                self.properties.set_q_vv(qvv);
            }

            if self.properties.omega().is_none() {
                // Reference to the orbital energies.
                // Check if the orbital energy differences were already computed.
                let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
                let dim_o: usize = (nocc as f64 * config.tddftb.active_orbital_threshold) as usize;
                let dim_v: usize = (nvirt as f64 * config.tddftb.active_orbital_threshold) as usize;

                // Energies of the occupied orbitals.
                let orbe_occ = orbe.slice(s![homo + 1 - dim_o..homo + 1]);
                // Energies of the virtual orbitals.
                let orbe_virt = orbe.slice(s![lumo..lumo + dim_v]);

                // Energy differences between virtual and occupied orbitals.
                let omega: Array1<f64> = orbe_differences(orbe_occ, orbe_virt);

                // Energy differences are stored in the molecule.
                self.properties.set_omega(omega);

                self.properties.set_homo(homo);
                self.properties.set_lumo(lumo);
            }
        } else {
            if self.properties.q_ov().is_none() && self.properties.q_ov().is_none() {
                if config.use_shell_resolved_gamma {
                    let (qov, qoo, qvv): (Array2<f64>, Array2<f64>, Array2<f64>) = trans_charges_ao(
                        self.n_orbs,
                        self.properties.orbs().unwrap(),
                        self.properties.s().unwrap(),
                        occ_indices,
                        virt_indices,
                    );
                    // And stored in the properties HashMap.
                    self.properties.set_q_oo(qoo);
                    self.properties.set_q_ov(qov);
                    self.properties.set_q_vv(qvv);
                } else {
                    let (qov, qoo, qvv): (Array2<f64>, Array2<f64>, Array2<f64>) = trans_charges(
                        self.n_atoms,
                        atoms,
                        self.properties.orbs().unwrap(),
                        self.properties.s().unwrap(),
                        occ_indices,
                        virt_indices,
                    );
                    // And stored in the properties HashMap.
                    self.properties.set_q_oo(qoo);
                    self.properties.set_q_ov(qov);
                    self.properties.set_q_vv(qvv);
                }
            }

            if self.properties.omega().is_none() {
                // Reference to the orbital energies.
                // Check if the orbital energy differences were already computed.
                let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();

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
        }
    }
}
