//! Shared utilities for dialect:
//!
//! * [`linalg`] — centralized raw BLAS/LAPACK FFI wrappers. Every direct
//!   `extern "C"` call into the linear-algebra backend lives here:
//!   DSYEVD eigendecomposition ([`linalg::eigh`]) and raw row-major DGEMM
//!   kernels plus blocked transpose/permute helpers ([`linalg::dgemm`]).
//!   (Exception: the DSYMV binding sits in `dialect-xtb-core` because this
//!   crate depends on it — see the note there.)
//! * [`cubes`] — Gaussian cube file generation: grid/box construction,
//!   basis evaluation on the grid (DFTB STO-3G splines, xTB Gaussians) and
//!   density/orbital cube writing via [`cubes::cube::DensityToCube`].
//!
//! Higher-level code should call these wrappers (or the `ndarray-linalg`
//! traits) instead of declaring its own `extern "C"` blocks.
#![allow(warnings)]

// Force the LAPACK/BLAS backend to be linked even though no Rust
// symbol from it is referenced directly (the extern "C" declarations
// in `linalg` resolve against it at link time).
extern crate lapack_src;

pub mod cubes;
pub mod io;
pub mod optimization;
pub mod output;
pub mod gradients;
pub mod fermi_occupation;
pub mod fmo_helpers;
pub mod fmo_logging;
pub mod fragmentation;
pub mod linalg;
pub mod mixer;
pub mod mulliken;
pub mod numerical;
pub mod scc_helpers;
pub mod scc_interface;
pub mod scc_logging;
pub mod wigner;
pub mod zbrent;
