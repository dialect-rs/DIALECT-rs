use crate::initialization::system::System;
use crate::io::settings::*;
use crate::properties::{Properties, Property};
use crate::utils::get_path_prefix;
use data_reader::reader::{load_txt_f64, Delimiter, ReaderParams};
use ndarray::prelude::*;

pub const AVAILAIBLE_MOLECULES: [&str; 4] = ["h2o", "benzene", "ammonia", "uracil"];

fn get_config() -> Configuration {
    let config_string: String = String::from("");
    let config: Configuration = toml::from_str(&config_string).unwrap();
    config
}

fn get_config_external_skf() -> Configuration {
    let config_string: String = String::from("");
    let path_prefix = get_path_prefix();
    let mut config: Configuration = toml::from_str(&config_string).unwrap();
    // edit config params
    config.slater_koster.use_external_skf = true;
    config.slater_koster.skf_directory = format!("{}/tests/data/slako/ob2-1-1-split", path_prefix);
    config.scf.scf_charge_conv = 1.0e-11;
    config.scf.scf_charge_conv = 1.0e-11;
    config.use_gaussian_gamma = false;
    config
}

/// Returns the absolute path to the the data directory of the tests. This function
/// requires that the `TINCR_SRC_DIR` environment variable is set, since the function
/// depends on the [get_path_prefix](crate::utils::get_path_prefix) function.
fn get_test_path_prefix() -> String {
    let path_prefix = get_path_prefix();
    format!("{}/tests/data", path_prefix,)
}

fn get_system(name: &str) -> System {
    let filename: String = format!("{}/{}/{}.xyz", get_test_path_prefix(), name, name);
    let mut config = get_config_external_skf();
    config.lc.long_range_correction = true;
    config.lc.long_range_radius = 1.0 / 0.3;
    System::from((filename.as_str(), config))
}

fn get_system_no_lc(name: &str) -> System {
    let filename: String = format!("{}/{}/{}.xyz", get_test_path_prefix(), name, name);
    let mut config = get_config_external_skf();
    config.lc.long_range_correction = false;
    System::from((filename.as_str(), config))
}

fn load_1d(filename: &str) -> Array1<f64> {
    let file = String::from(filename);
    let params = ReaderParams {
        comments: Some(b'%'),
        delimiter: Delimiter::WhiteSpace,
        skip_header: None,
        skip_footer: None,
        usecols: None,
        max_rows: None,
    };
    let results = load_txt_f64(&file, &params);
    Array1::from(results.unwrap().results)
}

fn load_2d(filename: &str) -> Array2<f64> {
    let file = String::from(filename);
    let params = ReaderParams {
        comments: Some(b'%'),
        delimiter: Delimiter::WhiteSpace,
        skip_header: None,
        skip_footer: None,
        usecols: None,
        max_rows: None,
    };
    let results = load_txt_f64(&file, &params).unwrap();
    let shape: (usize, usize) = (results.num_lines, results.num_fields);
    Array2::from_shape_vec(shape, results.results).unwrap()
}

// fn load_3d(filename: &str) -> Array3<f64> {
//     let file = String::from(filename);
//     let params = ReaderParams {
//         comments: Some(b'%'),
//         delimiter: Delimiter::WhiteSpace,
//         skip_header: None,
//         skip_footer: None,
//         usecols: None,
//         max_rows: None,
//     };
//     let results = load_txt_f64(&file, &params).unwrap();
//     let last_axis: usize = (results.num_fields as f64).sqrt() as usize;
//     let shape: (usize, usize, usize) = (results.num_lines, last_axis, last_axis);
//     return Array3::from_shape_vec(shape, results.results).unwrap();
// }

fn get_properties(mol: &str) -> Properties {
    let mut properties: Properties = Properties::new();
    let path: String = format!("{}/{}", get_test_path_prefix(), mol);
    let props1d = [
        "gs_gradient_lc",
        "gs_gradient_no_lc",
        "tdadftb_energies",
        "tdadftb_energies_no_lc",
        "tddftb_energies",
        "tddftb_energies_no_lc",
    ];
    let props2d = [
        "H0",
        "S",
        "gamma_ao_wise",
        "gamma_ao_wise_lc",
        "gamma_atomwise",
        "gamma_atomwise_lc",
    ];
    for property_name in props1d.iter() {
        let tmp: Property = Property::from(load_1d(&format!("{}/{}.dat", path, property_name)));
        properties.set(property_name, tmp);
    }
    for property_name in props2d.iter() {
        let tmp: Property = Property::from(load_2d(&format!("{}/{}.dat", path, property_name)));
        properties.set(property_name, tmp);
    }
    properties
}

pub fn get_molecule(molecule_name: &'static str) -> (&'static str, System, Properties) {
    let system: System = get_system(molecule_name);
    let properties: Properties = get_properties(molecule_name);
    (molecule_name, system, properties)
}

pub fn get_molecule_no_lc(molecule_name: &'static str) -> (&'static str, System, Properties) {
    let system: System = get_system_no_lc(molecule_name);
    let properties: Properties = get_properties(molecule_name);
    (molecule_name, system, properties)
}
