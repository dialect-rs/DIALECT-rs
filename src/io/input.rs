use crate::defaults::CONFIG_FILE_NAME;
use crate::initialization::Atom;
use crate::io::{read_file_to_frame, Configuration};
use chemfiles::Frame;
use ndarray::prelude::*;
use std::fs;
use std::path::Path;

pub fn read_input(geom_file: &str) -> (Frame, Configuration) {
    // The file containing the cartesian coordinates is the only mandatory file to
    // start a calculation.
    let frame: Frame = read_file_to_frame(geom_file);

    // The configuration file is read, if it does not exist in the directory
    // the program initializes the default settings and writes a configuration file
    // to the directory. At the moment the filename of the configuration file
    // is hard set in the defaults.rs file to "tincr.toml"
    let config_file_path: &Path = Path::new(CONFIG_FILE_NAME);
    let mut config_string: String = if config_file_path.exists() {
        fs::read_to_string(config_file_path).expect("Unable to read config file")
    } else {
        String::from("")
    };
    // Load the configuration.
    let config: Configuration = toml::from_str(&config_string).unwrap();
    // The configuration file is saved if it does not exist already so that the user can see
    // all the used options.
    if config_file_path.exists() == false {
        config_string = toml::to_string(&config).unwrap();
        fs::write(config_file_path, config_string).expect("Unable to write config file");
    }
    (frame, config)
}
