//! Geometry-input helpers shared by the method driver crates, based on the
//! pure-Rust `xyz_parser` crate (no chemfiles dependency).

use dialect_base::constants::BOHR_TO_ANGS;
use ndarray::{Array1, Array2};
use xyz_parser::{parse_xyz_file, XyzFrame};

/// Extract the atomic numbers and positions (in bohr) from an [`XyzFrame`]
/// (coordinates in the file are Angstrom).
pub fn xyz_frame_to_coordinates(frame: XyzFrame) -> (Vec<u8>, Array2<f64>) {
    let mut positions: Array1<f64> = Array1::from_vec(frame.coordinates);
    // transform the coordinates from angstrom to bohr
    positions /= BOHR_TO_ANGS;
    let positions_2d: Array2<f64> = positions
        .into_shape([frame.atomic_numbers.len(), 3])
        .unwrap();
    (frame.atomic_numbers, positions_2d)
}

/// Read an xyz-geometry file and return the first frame.
pub fn read_xyz_frame(filename: &str) -> XyzFrame {
    parse_xyz_file(filename)
        .unwrap_or_else(|e| panic!("could not parse xyz file {}: {}", filename, e))
}
