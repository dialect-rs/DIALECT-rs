use crate::{initialization::System, scc::scc_routine::RestrictedSCC};
use ndarray::{Array3, ArrayView1, ArrayView2, Axis};
use ndarray_npy::{write_npy, NpzWriter};
use std::fs::File;

impl System {
    pub fn parameterization_output(&mut self) {
        self.prepare_scc();
        let gs_energy: f64 = self.run_scc().unwrap();
        let (mulliken, dipole_skf, mulliken_charges) = self.calculate_dipole_moment();
        self.calculate_excited_states(true);

        let exc_energies = self.properties.ci_eigenvalues().unwrap();
        let cis_coeff: ArrayView2<f64> = self.properties.ci_coefficients().unwrap();
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();

        write_npy("tdms.npy", &cis_coeff).unwrap();
        write_npy("mo_coeffs.npy", &orbs).unwrap();
        write_npy("excited_energies.npy", &exc_energies).unwrap();
        write_npy("mulliken_charge_differences.npy", &mulliken_charges).unwrap();
    }
}
