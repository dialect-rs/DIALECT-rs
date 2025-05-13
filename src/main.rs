#![allow(dead_code)]

use crate::excited_states::davidson::Davidson;
use crate::excited_states::initial_subspace;
use crate::fmo::SuperSystem;
use crate::initialization::parameter_handling::generate_parameters;
use crate::initialization::System;
use crate::io::{
    create_dynamics_data, read_dynamic_input, read_dynamic_input_ehrenfest, read_input,
    write_header, Configuration,
};
use crate::io::{create_dynamics_data_xtb, write_footer};
use crate::scc::scc_routine::RestrictedSCC;
use crate::utils::Timer;
use chemfiles::Frame;
use clap::{App, Arg};
use dialect_dynamics::initialization::{DynamicConfiguration, Simulation, SystemData};
use env_logger::Builder;
use fmo::helpers::{monomer_identification, remove_duplicate_atoms};
use log::LevelFilter;
use std::env;
use std::io::Write;
use std::process;
use xtb::initialization::system::XtbSystem;

mod constants;
mod couplings;
mod cubes;
mod defaults;
mod dynamics;
mod excited_states;
mod fmo;
mod gradients;
mod hessian;
mod initial_conditions;
mod initialization;
mod io;
mod optimization;
mod param;
mod parameterization;
mod properties;
mod scc;
mod utils;
mod xtb;

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
            if !config.use_xtb1 {
                // Normal DFTB calculation
                if !config.fmo {
                    // Create system from frame and config
                    let mut system = System::from((frame, config.clone()));
                    system.input_check();

                    // Prepare and run the SCC routine
                    system.prepare_scc();
                    system.run_scc().unwrap();

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
                    let mut system = SuperSystem::from((
                        frame,
                        config.clone(),
                        &slako,
                        &vrep,
                        unique_atoms,
                        atoms,
                    ));
                    system.input_check();

                    // Prepare and run the FMO SCC routine
                    system.prepare_scc();
                    system.run_scc().unwrap();

                    // Calculate the excited state energies
                    if config.excited.calculate_excited_states {
                        system.create_exciton_hamiltonian();
                    }
                }
            } else {
                // xtb1 calculation
                // create the xtb system
                let mut system = XtbSystem::from((frame, config.clone()));
                system.input_check();

                // prepare and run the scc routine
                system.prepare_scc();
                system.run_scc().unwrap();
            }
        }
        // Calculate the density on a grid and save it in a cube file
        "density" => {
            let system = System::from((frame, config.clone()));
            system.input_check();
            system.density_to_cube();
        }
        "dynamics" => {
            if !config.use_xtb1 {
                if !config.fmo {
                    let mut system = System::from((frame, config.clone()));
                    system.input_check();
                    let dynamics_config: DynamicConfiguration = read_dynamic_input(&system.config);
                    let dynamics_data: SystemData =
                        create_dynamics_data(&system.atoms, dynamics_config);

                    let mut dynamics: Simulation = Simulation::new(&dynamics_data);
                    if dynamics.config.gs_dynamic || dynamics.config.use_surface_hopping {
                        if dynamics.config.langevin_config.use_langevin {
                            dynamics.langevin_dynamics(&mut system);
                        } else {
                            dynamics.verlet_dynamics(&mut system);
                        }
                    } else if dynamics.config.use_ehrenfest {
                        if dynamics.config.ehrenfest_config.use_tab_decoherence {
                            dynamics.ehrenfest_dynamics_tab(&mut system);
                        } else {
                            dynamics.ehrenfest_dynamics(&mut system);
                        }
                    } else {
                        dynamics.verlet_dynamics(&mut system);
                    }
                } else {
                    // create Slater-Koster files and the atoms from frame and config
                    let (slako, vrep, atoms, unique_atoms) =
                        generate_parameters(frame.clone(), config.clone());
                    // Create the system from the Slater-Koster files, the config and the atoms
                    let mut system = SuperSystem::from((
                        frame,
                        config.clone(),
                        &slako,
                        &vrep,
                        unique_atoms,
                        atoms,
                    ));
                    system.input_check();

                    let n_monomer: usize = system.monomers.len();
                    let mut dynamics_config: DynamicConfiguration =
                        read_dynamic_input_ehrenfest(&config, n_monomer);

                    if dynamics_config.use_ehrenfest {
                        let dynamics_data: SystemData =
                            create_dynamics_data(&system.atoms, dynamics_config);
                        let mut dynamics: Simulation = Simulation::new(&dynamics_data);

                        if dynamics.config.ehrenfest_config.use_tab_decoherence {
                            dynamics.ehrenfest_dynamics_tab(&mut system);
                        } else {
                            dynamics.ehrenfest_dynamics(&mut system);
                        }
                    } else {
                        // Only allow ground-state dynamics
                        dynamics_config.nstates = 1;
                        let dynamics_data: SystemData =
                            create_dynamics_data(&system.atoms, dynamics_config);
                        let mut dynamics: Simulation = Simulation::new(&dynamics_data);

                        dynamics.verlet_dynamics(&mut system);
                    }
                }
            } else {
                // create the xtb system
                let mut system = XtbSystem::from((frame, config.clone()));
                system.input_check();
                let dynamics_config: DynamicConfiguration = read_dynamic_input(&system.config);
                let dynamics_data: SystemData =
                    create_dynamics_data_xtb(&system.atoms, dynamics_config);
                let mut dynamics: Simulation = Simulation::new(&dynamics_data);
                // start the dynamics
                if dynamics.config.langevin_config.use_langevin {
                    dynamics.langevin_dynamics(&mut system);
                } else {
                    dynamics.verlet_dynamics(&mut system);
                }
            }
        }
        "opt" => {
            if !config.use_xtb1 {
                if !config.fmo {
                    // Create system from frame and config
                    let mut system = System::from((frame, config.clone()));
                    system.input_check();
                    // run the cartesian optimization
                    system.optimize_cartesian(system.config.opt.state_to_optimize, &config);
                } else {
                    // create Slater-Koster files and the atoms from frame and config
                    let (slako, vrep, atoms, unique_atoms) =
                        generate_parameters(frame.clone(), config.clone());
                    // Create the system from the Slater-Koster files, the config and the atoms
                    let mut system = SuperSystem::from((
                        frame,
                        config.clone(),
                        &slako,
                        &vrep,
                        unique_atoms,
                        atoms,
                    ));
                    system.input_check();

                    // run the cartesian optimization
                    // at the moment, only a ground state optimization of the fmo system is implemented
                    system.optimize_cartesian(system.config.opt.state_to_optimize, &config);
                }
            } else {
                // create the xtb system
                let mut system = XtbSystem::from((frame, config.clone()));
                system.input_check();

                // start the optimization of the ground state
                system.optimize_cartesian(0, &config);
            }
        }
        "tdm_ehrenfest" => {
            // create Slater-Koster files and the atoms from frame and config
            let (slako, vrep, atoms, unique_atoms) =
                generate_parameters(frame.clone(), config.clone());
            // Create the system from the Slater-Koster files, the config and the atoms
            let mut system =
                SuperSystem::from((frame, config.clone(), &slako, &vrep, unique_atoms, atoms));
            system.input_check();

            system.get_ehrenfest_densities();
        }
        "grad" => {
            if !config.use_xtb1 {
                // Normal DFTB calculation
                if !config.fmo {
                    // Create system from frame and config
                    let mut system = System::from((frame, config.clone()));
                    system.input_check();

                    // Prepare and run the SCC routine
                    system.prepare_scc();
                    system.run_scc().unwrap();

                    if system.config.excited.calculate_excited_states {
                        system.ground_state_gradient(true);
                        system.calculate_excited_states(true);
                        system.calculate_excited_state_gradient(0);
                    } else {
                        system.ground_state_gradient(false);
                    }
                } else {
                    // FMO DFTB calculation
                    // create Slater-Koster files and the atoms from frame and config
                    let (slako, vrep, atoms, unique_atoms) =
                        generate_parameters(frame.clone(), config.clone());
                    // Create the system from the Slater-Koster files, the config and the atoms
                    let mut system = SuperSystem::from((
                        frame,
                        config.clone(),
                        &slako,
                        &vrep,
                        unique_atoms,
                        atoms,
                    ));
                    system.input_check();

                    // Prepare and run the FMO SCC routine
                    system.prepare_scc();
                    system.run_scc().unwrap();
                    system.ground_state_gradient();
                }
            } else {
                // create the xtb system
                let mut system = XtbSystem::from((frame, config.clone()));

                // prepare and run the scc routine
                system.prepare_scc();
                system.run_scc().unwrap();
                system.ground_state_gradient();
            }
        }
        "monomer_identification" => {
            // create Slater-Koster files and the atoms from frame and config
            let (slako, vrep, atoms, unique_atoms) =
                generate_parameters(frame.clone(), config.clone());

            let new_atoms = remove_duplicate_atoms(&atoms);

            // Create the system from the Slater-Koster files, the config and the atoms
            let system = SuperSystem::from((
                frame,
                config.clone(),
                &slako,
                &vrep,
                unique_atoms,
                new_atoms,
            ));
            system.input_check();
            // get the number of monomers and pairs
            println!("Number of Monomers: {}", system.monomers.len());
            println!("Number of pairs: {}", system.pairs.len());

            let monomer_indices: Vec<usize> = monomer_identification(
                &config.identification_config,
                &system.atoms,
                &system.monomers,
            );
            println!("Monomer indices: {:?}", monomer_indices);
        }
        "initial_conditions" => {
            // Create system from frame and config
            let mut system = System::from((frame, config.clone()));
            system.input_check();
            // sample the wigner ensemble
            system.create_initial_conditions();
        }
        "polariton" => {
            // create Slater-Koster files and the atoms from frame and config
            let (slako, vrep, atoms, unique_atoms) =
                generate_parameters(frame.clone(), config.clone());
            // Create the system from the Slater-Koster files, the config and the atoms
            let mut system =
                SuperSystem::from((frame, config.clone(), &slako, &vrep, unique_atoms, atoms));
            system.input_check();

            // Prepare and run the FMO SCC routine
            system.prepare_scc();
            system.run_scc().unwrap();

            // Calculate the excited state energies
            system.create_exciton_polariton_hamiltonian();
        }
        jtype => {
            println!("Jobtype: {} is not available.", jtype);
            println!("Choose one of the available types: sp, opt, dynamics, density, tdm_ehrenfest, monomer_identification, initial_conditions, polariton");
        }
    }
    // ................................................................

    // Finished.
    // The total wall-time is printed together with the end statement.
    write_footer(timer);
    process::exit(1);
}
