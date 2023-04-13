#![allow(dead_code)]
#![allow(warnings)]

use std::env;
use std::io::Write;
use std::process;

use crate::excited_states::davidson::Davidson;
use crate::excited_states::initial_subspace;
use crate::fmo::SuperSystem;
use crate::initialization::parameter_handling::generate_parameters;
use crate::initialization::System;
use crate::io::{
    create_dynamics_data, read_dynamic_input, read_input, write_header, Configuration,
};
use crate::io::{write_footer, MoldenExporterBuilder};
use crate::scc::scc_routine::RestrictedSCC;
use crate::utils::Timer;
use chemfiles::Frame;
use clap::{App, Arg};
use dialect_dynamics::initialization::{DynamicConfiguration, Simulation, SystemData};
use env_logger::Builder;
use log::LevelFilter;

mod constants;
mod couplings;
mod cubes;
mod defaults;
mod dynamics;
mod excited_states;
mod fmo;
mod gradients;
mod initialization;
mod io;
mod optimization;
mod param;
mod properties;
mod scc;
mod utils;

#[macro_use]
extern crate clap;

fn main() {
    // Input.
    let matches = App::new(crate_name!())
        .version(crate_version!())
        .about("software package for tight-binding DFT calculations")
        .arg(
            Arg::new("xyz-File")
                .about("Sets the xyz file to use")
                .required(true)
                .index(1),
        )
        .get_matches();
    // The file containing the cartesian coordinates is the only mandatory file to
    // start a calculation.
    let geometry_file = matches.value_of("xyz-File").unwrap();
    let (frame, config): (Frame, Configuration) = read_input(geometry_file);

    // Multithreading.
    rayon::ThreadPoolBuilder::new()
        .num_threads(config.parallelization.number_of_cores)
        .build_global()
        .unwrap();

    // Logging.
    // The log level is set.
    let log_level: LevelFilter = match config.verbose {
        2 => LevelFilter::Trace,
        1 => LevelFilter::Debug,
        0 => LevelFilter::Info,
        -1 => LevelFilter::Warn,
        -2 => LevelFilter::Error,
        _ => LevelFilter::Info,
    };
    // and the logger is build.
    Builder::new()
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .filter(None, log_level)
        .init();

    // The program header is written to the command line.
    write_header();
    // and the total wall-time timer is started.
    let timer: Timer = Timer::start();

    // Computations.
    // ................................................................
    match config.jobtype.as_str() {
        "sp" => {
            // Normal DFTB calculation
            if !config.fmo {
                // Create system from frame and config
                let mut system = System::from((frame, config.clone()));

                // Prepare and run the SCC routine
                system.prepare_scc();
                system.run_scc();

                // Calculate the excited state energies
                if config.excited.calculate_excited_states {
                    system.calculate_excited_states(true);
                }
            // FMO DFTB calculation
            } else {
                // create Slater-Koster files and the atoms from frame and config
                let (slako, vrep, atoms, unique_atoms) =
                    generate_parameters(frame.clone(), config.clone());
                // Create the system from the Slater-Koster files, the config and the atoms
                let mut system =
                    SuperSystem::from((frame, config.clone(), &slako, &vrep, unique_atoms, atoms));

                // Prepare and run the FMO SCC routine
                system.prepare_scc();
                system.run_scc();

                // Calculate the excited state energies
                if config.excited.calculate_excited_states {
                    system.create_exciton_hamiltonian();
                }
            }
        }
        // Calculate the density on a grid and save it in a cube file
        "density" => {
            let system = System::from((frame, config.clone()));
            system.density_to_cube();
        }
        "dynamics" => {
            if !config.fmo {
                let mut system = System::from((frame, config.clone()));
                let dynamics_config: DynamicConfiguration = read_dynamic_input(&system.config);
                let dynamics_data: SystemData =
                    create_dynamics_data(&system.atoms, dynamics_config);

                let mut dynamics: Simulation = Simulation::new(&dynamics_data);
                if dynamics.config.langevin_config.use_langevin {
                    dynamics.langevin_dynamics(&mut system);
                } else {
                    dynamics.verlet_dynamics(&mut system);
                }
            } else {
                panic!("No implementation of molecular dynamics for the FMO system yet!");
            }
        }
        "opt" => {
            if !config.fmo {
                // Create system from frame and config
                let mut system = System::from((frame, config.clone()));
                // run the cartesian optimization
                system.optimize_cartesian(system.config.opt.state_to_optimize);
            } else {
                // create Slater-Koster files and the atoms from frame and config
                let (slako, vrep, atoms, unique_atoms) =
                    generate_parameters(frame.clone(), config.clone());
                // Create the system from the Slater-Koster files, the config and the atoms
                let mut system =
                    SuperSystem::from((frame, config.clone(), &slako, &vrep, unique_atoms, atoms));

                // run the cartesian optimization
                // at the moment, only a ground state optimization of the fmo system is implemented
                system.optimize_cartesian(system.config.opt.state_to_optimize);
            }
        }
        jtype => {
            println!("Jobtype: {} is not available.", jtype);
            println!("Choose one of the available types: sp, opt, dynamics, density");
        }
    }
    // ................................................................

    // Finished.
    // The total wall-time is printed together with the end statement.
    write_footer(timer);
    process::exit(1);
}
