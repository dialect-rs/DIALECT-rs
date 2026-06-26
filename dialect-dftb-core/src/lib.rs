//! DFTB foundations: the `Atom`/`AtomicOrbital` types, Slater-Koster and
//! repulsive-potential parameter tables (including SKF file handling), the
//! Slater-Koster transformations, the H0/overlap matrix builders and the
//! gamma-function machinery. All routines operate on plain atom slices and
//! parameter tables; SCF drivers and the `System` struct live in the main
//! crate.
#![allow(warnings)]

pub mod atom;
pub mod gamma_approximation;
pub mod geometry;
pub mod h0_and_s;
pub mod parameters;
pub mod slako_transformations;
