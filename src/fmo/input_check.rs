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
        if config.use_shell_resolved_gamma && config.tddftb.restrict_active_orbitals {
            println!("The shell resolved option for the gamma matrix is not yet supported for restricted active orbitals!");
            println!("Please change the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }
        if config.use_shell_resolved_gamma && config.jobtype != *"sp" {
            println!("The shell resolved option for FMO only supports single point calculations at the moment!");
            println!("Please change the dialect.toml file accordingly.");
            panic!("Error occured in the input check!");
        }

        println!("{: ^80}", "Finished input check. No problems occured!");
        println!("{:-<80} ", "");
        println!("{:^80} ", "");
    }
}
