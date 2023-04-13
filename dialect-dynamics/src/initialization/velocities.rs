use crate::constants;
use crate::initialization::SystemData;
use ndarray::Array2;
use rand_distr::{Distribution, Normal};

/// Struct that holds a Boltzmann distribution
pub struct BoltzmannVelocities {
    dist: Normal<f64>,
}

impl BoltzmannVelocities {
    /// Initialize the distribution from a given temperature
    pub fn new(temperature: f64) -> BoltzmannVelocities {
        let dist = Normal::new(0.0, f64::sqrt(constants::K_BOLTZMANN * temperature))
            .expect("Error regarding the distribution!");
        BoltzmannVelocities { dist }
    }
}

/// Initialize the velocities of the system using the struct [BoltzmannVelocities]
/// by sampling the distribution
pub fn initialize_velocities(system: &SystemData, temperature: f64) -> Array2<f64> {
    let boltzmann: BoltzmannVelocities = BoltzmannVelocities::new(temperature);
    let mut velocities: Array2<f64> = Array2::zeros(system.coordinates.raw_dim());

    for atom in 0..system.n_atoms {
        let mass_inv: f64 = 1.0 / system.masses[atom];
        velocities[[atom, 0]] =
            f64::sqrt(mass_inv) * boltzmann.dist.sample(&mut rand::thread_rng());
        velocities[[atom, 1]] =
            f64::sqrt(mass_inv) * boltzmann.dist.sample(&mut rand::thread_rng());
        velocities[[atom, 2]] =
            f64::sqrt(mass_inv) * boltzmann.dist.sample(&mut rand::thread_rng());
    }
    velocities
}
