#![allow(dead_code)]
#![allow(warnings)]

use std::io::Write;
use std::path::Path;
use std::process;
use std::time::{Duration, Instant};
use std::{env, fs};

use crate::defaults::CONFIG_FILE_NAME;
use crate::excited_states::ExcitedState;
use crate::initialization::System;
use crate::io::{read_file_to_frame, read_input, write_header, Configuration, MoldenExporter};
use crate::io::{write_footer, MoldenExporterBuilder};
use crate::scc::gamma_approximation::gamma_atomwise;
use crate::scc::scc_routine::RestrictedSCC;
use chemfiles::Frame;
use clap::{App, Arg};
use env_logger::Builder;
use log::info;
use log::LevelFilter;
use ndarray::prelude::*;
use petgraph::stable_graph::*;
use toml;

use crate::utils::Timer;
use ndarray::{Array1, Array2};

use crate::excited_states::davidson::Davidson;
use crate::excited_states::tda::*;
use crate::excited_states::{initial_subspace, orbe_differences, trans_charges, ProductCache};

use crate::fmo::SuperSystem;
use crate::initialization::parameter_handling::generate_parameters;
use ndarray_npy::{write_npy, NpzWriter};
use std::fs::File;

mod constants;
mod cubes;
mod defaults;
mod excited_states;
mod fmo;
mod gradients;
mod initialization;
mod io;
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
                if config.excited.calculate_excited_states{
                    system.calculate_excited_states();
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
                if config.excited.calculate_excited_states{
                    system.create_exciton_hamiltonian();
                }
            }
        }
        // Calculate the density on a grid and save it in a cube file
        "density" => {
            let mut system = System::from((frame, config.clone()));
            system.density_to_cube();
        }
        jtype => {
            println!("Jobtype: {} is not available.", jtype);
            println!("Choose one of the available types: sp, density");
        }
    }
    // ................................................................

    // Finished.
    // The total wall-time is printed together with the end statement.
    write_footer(timer);
    process::exit(1);
}
