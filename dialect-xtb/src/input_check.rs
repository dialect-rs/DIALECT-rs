//! Input sanity checks for the xTB systems.

use crate::fmo::supersystem::XtbSuperSystem;
use crate::initialization::system::XtbSystem;
use dialect_config::Configuration;
use log::{debug, warn};

impl XtbSystem {
    pub fn input_check(&self) {
        // check if the config contains any errors
        let config: &Configuration = &self.config;

        debug!("{:^80}", "");
        debug!("{:-^80}", "");
        if config.excited.calculate_excited_states {
            debug!("The calculation of excited states is not supported with the xtb Hamiltonian!");
            panic!("Error occured in the input check!");
        }

        // Open-shell handling. The occupation is built from the Fermi
        // distribution over the spatial orbitals, which realizes the *minimal*
        // open shell for the electron count: a closed shell for an even count
        // and a single singly-occupied orbital (a doublet) for an odd count.
        // multiplicity = 1 is the default ("auto"): the occupation is then
        // determined by the electron count alone (even -> singlet, odd ->
        // doublet), so any electron count is accepted.
        if config.mol.multiplicity != 1 {
            // An explicitly requested multiplicity fixes the number of unpaired
            // electrons; it must have the same parity as the electron count.
            let n_unpaired: usize = (config.mol.multiplicity - 1) as usize;
            if self.n_elec % 2 != n_unpaired % 2 {
                debug!(
                    "The requested multiplicity ({}) implies {} unpaired electrons, which is inconsistent with the electron count ({}).",
                    config.mol.multiplicity, n_unpaired, self.n_elec
                );
                debug!("Please adjust [molecule] multiplicity / charge in the dialect.toml file accordingly.");
                panic!("Error occured in the input check!");
            }
            // The occupation fills n_alpha = (n_elec + n_unpaired)/2 and
            // n_beta = (n_elec - n_unpaired)/2 orbitals. Both must be physically
            // realizable: n_unpaired <= n_elec (otherwise n_beta < 0) and
            // n_alpha <= n_orbs (enough orbitals to hold the alpha electrons,
            // i.e. an "n_elec > 2*nao" capacity guard).
            if n_unpaired > self.n_elec || (self.n_elec + n_unpaired) / 2 > self.n_orbs {
                debug!(
                    "The requested multiplicity ({}) implies {} unpaired electrons, which cannot be accommodated by {} electrons in {} orbitals.",
                    config.mol.multiplicity, n_unpaired, self.n_elec, self.n_orbs
                );
                debug!("Please change the dialect.toml file accordingly.");
                panic!("Error occured in the input check!");
            }
        }

        // Open-shell SCF at zero electronic temperature tends to oscillate
        // (the singly-occupied frontier orbital sloshes between near-degenerate
        // states). Warn the user, since a finite electronic temperature is
        // usually required for convergence.
        if self.n_elec % 2 != 0 && config.scf.electronic_temperature == 0.0 {
            warn!(
                "Open-shell system (odd electron count {}) requested at an electronic temperature of 0 K; the SCC may fail to converge. Consider setting [scc] electronic_temperature (e.g. 300 K).",
                self.n_elec
            );
        }

        debug!("{: ^80}", "Finished input check. No problems occured!");
        debug!("{:-<80} ", "");
        debug!("{:^80} ", "");
    }
}

impl XtbSuperSystem<'_> {
    pub fn input_check(&self) {
        // check if the config contains any errors
        let config: &Configuration = &self.config;

        debug!("{:^80}", "");
        debug!("{:-^80}", "");

        // xTB has no excited-state implementation.
        if config.excited.calculate_excited_states {
            debug!("The calculation of excited states is not supported with the xtb Hamiltonian!");
            debug!("Please change the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }

        // FMO-xTB is closed-shell; the per-monomer occupied orbitals are filled
        // with n_elec / 2 and the global multiplicity is otherwise ignored.
        if config.mol.multiplicity != 1 {
            debug!(
                "Only closed-shell (multiplicity = 1) FMO-xTB calculations are supported, got multiplicity = {}.",
                config.mol.multiplicity
            );
            debug!("Please change the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }

        // Charged systems are not handled by the FMO-xTB fragmentation (the
        // monomer electron counts are built from the neutral atoms), so the
        // charge option would be silently ignored.
        if config.mol.charge != 0 {
            debug!(
                "Charged systems are not implemented for the FMO-xTB routines, got charge = {}.",
                config.mol.charge
            );
            debug!("Please change the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }

        debug!("{: ^80}", "Finished input check. No problems occured!");
        debug!("{:-<80} ", "");
        debug!("{:^80} ", "");
    }
}
