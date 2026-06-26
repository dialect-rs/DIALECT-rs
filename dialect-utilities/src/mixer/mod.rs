use ndarray::Array1;

pub mod anderson;
pub mod anderson2;
pub mod broyden;

pub use anderson::*;
pub use broyden::BroydenMixerNew;

/// Trait that allows mixing of partial charge differences for the acceleration
/// of the SCC routine
pub trait Mixer {
    fn new(n_atoms: usize) -> Self;
    fn mix(&mut self, q_inp: Array1<f64>, q_diff: Array1<f64>) -> Array1<f64>;
    fn next(&mut self, q_inp: Array1<f64>, q_diff: Array1<f64>) -> Array1<f64>;
    fn reset(&mut self, n_atoms: usize);
}

use dialect_config::settings::MixConfig;
use anyhow::{Context, Result};

/// Builds the Anderson accelerator from a [`MixConfig`]. Lives here (not on
/// MixConfig itself) because the config types are defined in dialect-config,
/// below the mixer machinery.
pub trait BuildMixer {
    fn build_mixer(&self, dim: usize) -> Result<AndersonAccel>;
}

impl BuildMixer for MixConfig {
    /// Initialize an instance of the Anderson Accelerator. The dimension `dim` specifies the
    /// length of the vector that should be mixed. Further details are given in the Ac2O3 crate.
    fn build_mixer(&self, dim: usize) -> Result<AndersonAccel> {
        // In case that AA should not be used linear mixing/vanilla iterations will be used. This
        // can be enabled by setting the memory of AndersonAccel to zero.
        let memory = match self.use_aa {
            true => self.memory,
            false => 0,
        };

        AndersonAccelBuilder::default()
            .dim(dim)
            .memory(memory)
            .aa_type(self.aa_type)
            .regularization(self.regularization)
            .safeguard_factor(self.tol_safe)
            .max_weight_norm(self.max_norm)
            .build()
            .context("Could not intialize Anderson Acceleration instance")
    }
}
