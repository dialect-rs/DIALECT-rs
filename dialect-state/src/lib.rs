//! Shared mutable state for the DFTB/FMO stack: the `Properties` store
//! (a typed property map attached to every System/Monomer/Pair), the
//! reduced LE/CT basis-state descriptors, fragment slice bookkeeping,
//! the `PairType` classification, and the Old* snapshot structs used to
//! carry information between dynamics steps.
#![allow(warnings)]

pub use basis_states::{ChargeTransferPair, ReducedBasisState, ReducedCT, ReducedLE};
pub use old_supersystem::{OldEsdPair, OldMonomer, OldPair, OldSupersystem};
pub use old_system::OldSystem;
pub use pair_type::PairType;
pub use properties::Properties;
pub use slices::{MolIncrements, MolIndices, MolecularSlice};

pub mod basis_states;
pub mod old_supersystem;
pub mod old_system;
pub mod pair_type;
pub mod properties;
pub mod slices;
