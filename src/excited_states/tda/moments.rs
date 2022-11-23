use crate::fmo::Monomer;
use crate::initialization::{get_xyz_2d, Atom};
use ndarray::prelude::*;

/// The Mulliken transition dipole moments between the electronic ground state
/// and all excited states are computed and returned.
pub fn mulliken_dipoles(q_trans: ArrayView2<f64>, atoms: &[Atom]) -> Array2<f64> {
    // Get the xyz-coordinates.
    let xyz: Array2<f64> = get_xyz_2d(atoms);
    // and compute the transition dipole moments according to:
    // ->    ->
    // µ = ∑ r_i * q_i
    xyz.t().dot(&q_trans)
}

/// Computes the oscillator strength from the excitation energies, E, and the transition dipole
/// moments, µ, according to:
/// f = 2/3 * E * |µ|^2
pub fn oscillator_strength(
    exc_energies: ArrayView1<f64>,
    tr_dipoles: ArrayView2<f64>,
) -> Array1<f64> {
    2.0 / 3.0 * &exc_energies * &tr_dipoles.t().dot(&tr_dipoles).diag()
}
