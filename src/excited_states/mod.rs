use std::fs::File;
use std::io::Write;
use std::ops::AddAssign;

use ndarray::prelude::*;
use ndarray_npy::{write_npy, WriteNpyError};

pub use solvers::*;
pub use tda::*;
pub use transition_charges::*;
pub use utils::*;

use crate::constants::HARTREE_TO_EV;
use crate::excited_states::ntos::natural_transition_orbitals;
use crate::initialization::Atom;
use crate::io::MoldenExporter;
use crate::MoldenExporterBuilder;

pub mod casida;
pub mod ntos;
mod solvers;
pub(crate) mod tda;
mod transition_charges;
mod utils;

/// General trait for all excited states struct, to implement basis functions.
pub trait ExcitedState {
    /// Returns the index of the LUMO.
    fn get_lumo(&self) -> usize;

    /// Returns the MO coefficients.
    fn get_mo_coefficients(&self) -> ArrayView2<f64>;

    /// Returns the reduced one particle transition density matrix in MO basis for a specific
    /// excited states. 0 => S1, 1 => S2, ...
    fn get_transition_density_matrix(&self, state: usize) -> Array2<f64>;

    /// Returns the relative excitation energy of all excited states.
    fn get_energies(&self) -> ArrayView1<f64>;

    /// Returns the oscilaltor strengths of all excited states.
    fn get_oscillator_strengths(&self) -> ArrayView1<f64>;

    /// Returns the number of excited states.
    fn get_num_states(&self) -> usize;

    /// Write the excitation energies (in eV) and oscillator strength to a .npy file
    fn spectrum_to_npy(&self, filename: &str) -> Result<(), WriteNpyError> {
        // Stack the energies and osc. strengths into a 2D Array (columnwise).
        let mut data: Array2<f64> = Array2::zeros([self.get_num_states(), 0]);

        let energies_ev: Array1<f64> = HARTREE_TO_EV * &self.get_energies();
        // Convert the excitation energy in eV.
        data.push(Axis(1), energies_ev.view());
        data.push(Axis(1), self.get_oscillator_strengths());

        // Write the npy file.
        write_npy(filename, &data)
    }

    /// Write the excitation energies (in eV) and the oscillator strength to a text file.
    fn spectrum_to_txt(&self, filename: &str) {
        // Two lines of header.
        let mut txt: String = "# Absorption spectrum\n".to_owned();
        txt += "# exc. energy / eV    osc. strength\n";

        // Each energy and oscillator strength is written together to a line.
        for (e, f) in self
            .get_energies()
            .iter()
            .zip(self.get_oscillator_strengths().iter())
        {
            txt += &format!("{:16.14}      {:16.14}\n", e * HARTREE_TO_EV, f);
        }

        // Try to create the output file.
        let mut f =
            File::create(filename).expect(&*format!("Unable to create file: {}", &filename));

        // and write the data.
        f.write_all(txt.as_bytes())
            .expect(&format!("Unable to write data at: {}", &filename));
    }

    /// Compute the Natural Transition Orbitals (NTOs) of an excited state and write these orbitals
    /// to a molden file.
    fn ntos_to_molden(&self, atoms: &[Atom], state: usize, filename: &str) {
        // Get NTOs.
        let (lambdas, ntos): (Array1<f64>, Array2<f64>) = natural_transition_orbitals(
            self.get_transition_density_matrix(state).view(),
            self.get_mo_coefficients(),
            self.get_lumo(),
        );

        // Dummy energies are used for the NTOs to sort the orbitals in the correct order.
        let half_dim: usize = lambdas.len() / 2;
        let mut dummy_energies: Array1<f64> = Array1::zeros([lambdas.len()]);
        for (idx, e) in dummy_energies.iter_mut().enumerate() {
            *e = -1.0 * (half_dim as f64) + idx as f64;
        }
        // +1 for all particle NTOs to shift the lowest unoccupied to an energy of +1.
        dummy_energies.slice_mut(s![half_dim..]).add_assign(1.0);

        // The singular values are used as occupation numbers.
        let molden_exporter: MoldenExporter = MoldenExporterBuilder::default()
            .atoms(&atoms)
            .orbs(ntos.view())
            .orbe(dummy_energies.view())
            .f(lambdas.to_vec())
            .build()
            .unwrap();

        // Write the NTOs to the file.
        molden_exporter.write_to(filename.as_ref())
    }
}
