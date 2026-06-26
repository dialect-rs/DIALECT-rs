//! xTB (extended tight-binding) foundations: element/shell parameter tables,
//! the `XtbAtom` and contracted Gaussian `Basis` types, overlap integrals
//! (Obara-Saika), the H0 core Hamiltonian, the Klopman-Ohno gamma matrix and
//! EEQ initial charges. All routines operate on plain atom slices, basis
//! sets and arrays; SCF drivers and FMO machinery live in the main crate.
#![allow(warnings)]

pub mod initialization;
pub mod integrals;
pub mod parameters;
pub mod scc;
