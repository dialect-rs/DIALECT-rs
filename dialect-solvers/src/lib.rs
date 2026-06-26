//! Method-agnostic iterative eigensolvers for excited-state calculations:
//! the Davidson solver, the Casida-Davidson and Casida-for-TDA solvers and
//! the B-equation solver, all generic over the engine traits in [`traits`]
//! (`DavidsonEngine`, `CasidaEngine`, `AlternateCasidaEngine`,
//! `BsolverEngine`). The specialized engines — DFTB `System`/`Monomer`,
//! the NDDO/OMx CIS engines, MRCI — live with their methods and implement
//! these traits.
#![allow(warnings)]

pub use cache::ProductCache;
pub use traits::{AlternateCasidaEngine, BsolverEngine, CasidaEngine, DavidsonEngine};

pub mod cache;
pub mod casida_davidson;
pub mod casida_for_tda;
pub mod ct_workspace;
pub mod davidson;
pub mod traits;
pub mod utils;
