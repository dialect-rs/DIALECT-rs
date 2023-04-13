pub(crate) mod helpers;
mod logging;
mod monomer;
mod pair;
mod supersystem;

use crate::fmo::scc::helpers::get_dispersion_energy;
use crate::fmo::{Monomer, SuperSystem};
use crate::initialization::Atom;
use crate::scc::scc_routine::{RestrictedSCC, SCCError};
use crate::utils::Timer;
use ndarray::prelude::*;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefMutIterator;

impl RestrictedSCC for SuperSystem<'_> {
    ///  To run the SCC calculation of the FMO [SuperSystem] the following properties need to be set:
    /// For each [Monomer]
    /// - H0
    /// - S: overlap matrix in AO basis
    /// - Gamma matrix (and long-range corrected Gamma matrix if we use LRC)
    /// - If there are no charge differences, `dq`, from a previous calculation
    ///  they are initialized to zeros
    /// - the density matrix and reference density matrix
    fn prepare_scc(&mut self) {
        // prepare all individual monomers
        let atoms: &[Atom] = &self.atoms;

        self.monomers.par_iter_mut().for_each(|mol: &mut Monomer| {
            mol.prepare_scc(&atoms[mol.slice.atom_as_range()]);
        });
    }

    fn run_scc(&mut self) -> Result<f64, SCCError> {
        let timer: Timer = Timer::start();

        // SCC settings from the user input
        let max_iter: usize = self.config.scf.scf_max_cycles;
        logging::fmo_scc_init(max_iter);

        let mut e_disp: f64 = 0.0;
        if self.config.dispersion.use_dispersion {
            e_disp = get_dispersion_energy(&self.atoms, &self.config.dispersion);
        }

        // Assembling of the energy following Eq. 11 in
        // https://pubs.acs.org/doi/pdf/10.1021/ct500489d
        // E = sum_I^N E_I^ + sum_(I>J)^N ( E_(IJ) - E_I - E_J ) + sum_(I>J)^(N) DeltaE_(IJ)^V

        // Do the self-consistent monomer calculations
        let (monomer_energies, dq): (f64, Array1<f64>) = self.monomer_scc(max_iter);

        // Do the SCC-calculation for each pair individually
        let pair_energies: f64 = self.pair_scc(dq.view());

        // Compute the embedding energy from all pairs
        let embedding: f64 = self.embedding_energy();

        // Compute the energy from pairs that are far apart. The electrostatic dimer approximation
        // is used in this case.
        let esd_pair_energies: f64 = self.esd_pair_energy();

        // Sum up all the individual energies
        let total_energy: f64 =
            monomer_energies + pair_energies + embedding + esd_pair_energies + e_disp;

        // Save the charge differences of all monomers in the SuperSystem
        self.properties.set_dq(dq);

        // Print information of the SCC-routine
        logging::fmo_scc_end(
            timer,
            monomer_energies,
            pair_energies,
            embedding,
            esd_pair_energies,
            e_disp,
        );

        self.properties.set_last_energy(total_energy);
        // Return the energy
        Ok(total_energy)
    }
}
