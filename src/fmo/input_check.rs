use super::SuperSystem;
use crate::io::Configuration;

impl SuperSystem<'_> {
    pub fn input_check(&self) {
        // check if the config contains any errors
        let config: &Configuration = &self.config;
        println!("{:^80}", "");
        println!("{:-^80}", "");

        if config.dftb3.use_dftb3 {
            println!("FMO-DFTB3 is not implemented as of yet.");
            println!("Please adjust the dialect.toml file accordingly!");
            panic!("Error occured in the input check!")
        }
        if config.dftb3.use_gamma_damping {
            println!("The damping of the gamma matrix for FMO-DFTB is not implemented as of yet.");
            println!("Please adjust the dialect.toml file accordingly!");
            panic!("Error occured in the input check!")
        }
        if config.jobtype == "dynamics" && !config.lc.long_range_correction {
            println!("The dynamcis module for FMO-DFTB without long-range correction is not implemented as of yet.");
            println!("Please adjust the dialect.toml file accordingly!");
            panic!("Error occured in the input check!")
        }
        if config.tight_binding.use_shell_resolved_gamma && config.tddftb.restrict_active_orbitals {
            println!("The shell resolved option for the gamma matrix is not yet supported for restricted active orbitals!");
            println!("Please change the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }
        if config.tight_binding.use_shell_resolved_gamma && config.jobtype != *"sp" {
            println!("The shell resolved option for FMO only supports single point calculations at the moment!");
            println!("Please change the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }
        // FMO geometry optimization is implemented for the ground state only;
        // state_to_optimize >= 1 would select an (unsupported) excited state.
        if config.jobtype == "opt" && config.opt.state_to_optimize != 0 {
            println!("FMO geometry optimization is only implemented for the ground state, but state_to_optimize = {} was requested.", config.opt.state_to_optimize);
            println!("Please set [opt] state = 0 in the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }
        if config.fmo.manual_fragmentation {
            if config.fmo.fragment_atom_count == 0 {
                println!("Manual fragmentation enabled but fragment_atom_count is 0.");
                println!("Please set fragment_atom_count in the [fmo] section of dialect.toml.");
                panic!("Error occured in the input check!");
            }
            if config.fmo.number_of_fragments == 0 {
                println!("Manual fragmentation enabled but number_of_fragments is 0.");
                println!("Please set number_of_fragments in the [fmo] section of dialect.toml.");
                panic!("Error occured in the input check!");
            }
        }
        // The `indices` fragmentation (advanced_manual_fragmentation) is
        // validated more thoroughly in `advanced_manual_fragmentation()` during
        // SuperSystem construction (empty vector, out-of-range / duplicate /
        // unassigned atom indices), which runs before this check.

        println!("{: ^80}", "Finished input check. No problems occured!");
        println!("{:-<80} ", "");
        println!("{:^80} ", "");
    }
}
