//! The SCC driver interface: every system type (DFTB System/SuperSystem,
//! XtbSystem, ...) implements `RestrictedSCC`; `SCCError` reports
//! non-convergence.

use std::fmt;

pub trait RestrictedSCC {
    fn prepare_scc(&mut self);
    fn run_scc(&mut self) -> Result<f64, SCCError>;
}

#[derive(Debug, Clone)]
pub struct SCCError {
    pub message: String,
    iteration: usize,
    energy_diff: f64,
    charge_diff: f64,
}

impl SCCError {
    pub fn new(iter: usize, energy_diff: f64, charge_diff: f64) -> Self {
        let message: String = format! {"SCC-Routine failed in Iteration: {}. The charge\
         difference at the last iteration was {} and the energy\
         difference was {}",
        iter,
        charge_diff,
        charge_diff};
        Self {
            message,
            iteration: iter,
            energy_diff,
            charge_diff,
        }
    }
}

impl fmt::Display for SCCError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write! {f, "{}", self.message.as_str()}
    }
}

impl std::error::Error for SCCError {
    fn description(&self) -> &str {
        self.message.as_str()
    }
}
