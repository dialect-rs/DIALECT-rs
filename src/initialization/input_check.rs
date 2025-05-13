use super::System;
use crate::io::Configuration;
use crate::xtb::initialization::system::XtbSystem;
use log::debug;

impl System {
    pub fn input_check(&self) {
        // check if the config contains any errors
        let config: &Configuration = &self.config;

        debug!("{:^80}", "");
        debug!("{:-^80}", "");

        // both long-range correction and dftb3 are not allowed
        if config.lc.long_range_correction && config.dftb3.use_dftb3 {
            debug!("The long-range correction cannot be used in conjunction with dftb3!");
            debug!("Please change the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }

        // check if damping of the gamma matrix and gaussian basis is requested
        if config.dftb3.use_gamma_damping && config.use_gaussian_gamma {
            debug!("The damping of the gamma matrix is not implemented for gaussian functions!");
            debug!("Please change the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }

        // check if damping of the gamma matrix and long-range correction is requested
        if config.dftb3.use_gamma_damping && config.lc.long_range_correction {
            debug!(
                "The damping of the gamma matrix is intended for use with long-range correction!"
            );
            debug!("Please change the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }

        if config.use_shell_resolved_gamma && config.dftb3.use_dftb3 {
            debug!("The shell resolved option for the gamma matrix has only been implemented for DFTB2!");
            debug!("Please change the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }

        if config.use_shell_resolved_gamma && config.tddftb.restrict_active_orbitals {
            debug!("The shell resolved option for the gamma matrix is not yet supported for restricted active orbitals!");
            debug!("Please change the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }

        // check number of excited states
        if config.excited.calculate_excited_states {
            // number of occupied and virtual orbitals
            let nocc: usize = self.occ_indices.len();
            let nvirt: usize = self.virt_indices.len();

            // maximum number of excited states
            let nstates_max: usize = nocc * nvirt;
            if config.excited.nstates > nstates_max {
                debug!("The requested number of excited states is higher than the maximum");
                debug!(
                    "possible number of excitations between the occupied and virtual orbitals!!"
                );
                debug!("Maximum number of excitations: {}", nstates_max);
                panic!(
                    "Error in the excited state configuration! \nPlease change the dialect.toml file accordingly."
                );
            }
        }
        debug!("{: ^80}", "Finished input check. No problems occured!");
        debug!("{:-<80} ", "");
        debug!("{:^80} ", "");
    }
}

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

        debug!("{: ^80}", "Finished input check. No problems occured!");
        debug!("{:-<80} ", "");
        debug!("{:^80} ", "");
    }
}
