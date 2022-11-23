use crate::initialization::system::System;
use crate::io::settings::*;
use crate::properties::{Properties, Property};
use crate::utils::get_path_prefix;
use data_reader::reader::{load_txt_f64, Delimiter, ReaderParams};
use ndarray::prelude::*;

pub const AVAILAIBLE_MOLECULES: [&'static str; 4] = ["h2o", "benzene", "ammonia", "uracil"];

fn get_config() -> Configuration {
    let config_string: String = String::from("");
    let config: Configuration = toml::from_str(&config_string).unwrap();
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
    let mut config = get_config();
    config.lc.long_range_correction = true;
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
    return Array1::from(results.unwrap().results);
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
    return Array2::from_shape_vec(shape, results.results).unwrap();
}

fn load_3d(filename: &str) -> Array3<f64> {
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
    let last_axis: usize = (results.num_fields as f64).sqrt() as usize;
    let shape: (usize, usize, usize) = (results.num_lines, last_axis, last_axis);
    return Array3::from_shape_vec(shape, results.results).unwrap();
}

fn get_properties(mol: &str, _name: &str) -> Properties {
    let mut properties: Properties = Properties::new();
    let path: String = format!("{}/{}", get_test_path_prefix(), mol);
    let props3d = [
        "gamma_ao_wise_grad",
        "gamma_ao_wise_grad_lc",
        "gamma_atomwise_grad",
        "gamma_atomwise_grad_lc",
    ];
    let props2d = [
        "H0",
        "P0",
        "P_after_scc",
        "S",
        "g0_lr_ao",
        "lc_exact_exchange",
        "gamma_ao_wise",
        "gamma_ao_wise_lc",
        "distance_matrix",
        "gamma_atomwise",
        "gamma_atomwise_lc",
        "H1_after_scc",
        "orbs_after_scc",
    ];
    let props1d = [
        "orbs_per_atom",
        "q_after_scc",
        "dq_after_scc",
        "E_band_structure",
        "E_coulomb_after_scc",
        "E_rep",
        "occupation",
        "lc_exchange_energy",
    ];
    for property_name in props1d.iter() {
        // println!("{}", property_name);
        let tmp: Property = Property::from(load_1d(&format!("{}/{}.dat", path, property_name)));
        properties.set(property_name, tmp);
    }
    for property_name in props2d.iter() {
        // println!("{}", property_name);
        let tmp: Property = Property::from(load_2d(&format!("{}/{}.dat", path, property_name)));
        properties.set(property_name, tmp);
    }
    for property_name in props3d.iter() {
        // println!("{}", property_name);
        let tmp: Property = Property::from(load_3d(&format!("{}/{}.dat", path, property_name)));
        properties.set(property_name, tmp);
    }
    return properties;
}

pub fn get_molecule(
    molecule_name: &'static str,
    calculation_type: &str,
) -> (&'static str, System, Properties) {
    let system: System = get_system(molecule_name);
    let properties: Properties = get_properties(molecule_name, calculation_type);
    return (molecule_name, system, properties);
}
