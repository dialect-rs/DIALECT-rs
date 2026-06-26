use crate::constants;
use crate::initialization::DynamicConfiguration;
use ndarray::prelude::*;

/// Struct that hold the data of the molecular system:
/// the cartesian coordinates, atomic numbers and the masses
pub struct SystemData {
    // Type that holds all the input settings from the user.
    pub config: DynamicConfiguration,
    pub n_atoms: usize,
    pub atomic_numbers: Vec<u8>,
    pub coordinates: Array2<f64>,
    pub masses: Array1<f64>,
}

impl From<(Vec<u8>, Array2<f64>, DynamicConfiguration)> for SystemData {
    /// Creates the struct [SystemData] from a vector containing the atomic numbers,
    /// an array containing the cartesian coordinates and the global configuration [DynamicConfiguration]
    fn from(molecule: (Vec<u8>, Array2<f64>, DynamicConfiguration)) -> Self {
        let mut masses: Vec<f64> = Vec::new();
        // get the masses to the corresponding atomic numbers
        molecule.0.iter().for_each(|num| {
            masses.push(constants::ATOMIC_MASSES[num]);
        });
        let masses: Array1<f64> = Array::from(masses);

        Self {
            config: molecule.2,
            n_atoms: molecule.0.len(),
            atomic_numbers: molecule.0,
            coordinates: molecule.1,
            masses,
        }
    }
}

