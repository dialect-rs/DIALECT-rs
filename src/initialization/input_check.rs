use super::System;
use crate::io::Configuration;
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
        if config.dftb3.use_gamma_damping && config.tight_binding.use_gaussian_gamma {
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

        if config.tight_binding.use_shell_resolved_gamma && config.dftb3.use_dftb3 {
            debug!("The shell resolved option for the gamma matrix has only been implemented for DFTB2!");
            debug!("Please change the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }

        if config.tight_binding.use_shell_resolved_gamma && config.tddftb.restrict_active_orbitals {
            debug!("The shell resolved option for the gamma matrix is not yet supported for restricted active orbitals!");
            debug!("Please change the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }

        // DFTB is a closed-shell method: the occupied orbitals are filled with
        // n_elec / 2 (integer division), so an odd electron count would
        // silently drop the unpaired electron and yield a wrong energy.
        if self.n_elec % 2 != 0 {
            debug!(
                "DFTB requires a closed-shell system, but the electron count ({}) is odd.",
                self.n_elec
            );
            debug!("Please adjust the [molecule] charge in the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }

        // Only closed-shell (singlet) DFTB calculations are supported; the
        // multiplicity option is otherwise silently ignored.
        if config.mol.multiplicity != 1 {
            debug!(
                "Only closed-shell (multiplicity = 1) DFTB calculations are supported, got multiplicity = {}.",
                config.mol.multiplicity
            );
            debug!("Please change the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }

        // The radius of the long-range correction must be positive.
        if config.lc.long_range_correction && config.lc.long_range_radius <= 0.0 {
            debug!(
                "The long-range radius must be positive, got {}.",
                config.lc.long_range_radius
            );
            debug!("Please change the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }

        // The active-orbital threshold has to lie in (0, 1].
        if config.tddftb.restrict_active_orbitals
            && (config.tddftb.active_orbital_threshold <= 0.0
                || config.tddftb.active_orbital_threshold > 1.0)
        {
            debug!(
                "The active orbital threshold must lie in (0, 1], got {}.",
                config.tddftb.active_orbital_threshold
            );
            debug!("Please change the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }

        // Optimizing, evaluating the gradient of, or the Hessian of an excited
        // state (state_to_optimize >= 1 selects excited state state - 1)
        // requires the excited states to be calculated and the target state to
        // be within the requested number of states.
        if matches!(config.jobtype.as_str(), "opt" | "grad" | "hessian")
            && config.opt.state_to_optimize >= 1
        {
            if !config.excited.calculate_excited_states {
                debug!(
                    "state_to_optimize = {} selects an excited state, but excited states are disabled.",
                    config.opt.state_to_optimize
                );
                debug!("Please enable [excited] in the dialect.toml file accordingly.");
                panic!("Error occured in the input check!");
            }
            if config.opt.state_to_optimize > config.excited.nstates {
                debug!(
                    "state_to_optimize = {} exceeds the number of requested excited states ({}).",
                    config.opt.state_to_optimize, config.excited.nstates
                );
                debug!("Please change the dialect.toml file accordingly.");
                panic!("Error occured in the input check!");
            }
        }

        // check number of excited states
        if config.excited.calculate_excited_states {
            // a zero-state excited calculation is meaningless
            if config.excited.nstates == 0 {
                debug!("Excited states are enabled but the number of states is 0.");
                debug!("Please change the dialect.toml file accordingly.");
                panic!("Error occured in the input check!");
            }

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

