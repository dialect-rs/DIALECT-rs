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

/// The Mulliken transition dipole moments between the electronic ground state
/// and all excited states are computed and returned.
pub fn mulliken_dipoles_from_ao(q_trans: ArrayView2<f64>, atoms: &[Atom]) -> Array2<f64> {
    // transform qtrans from ao to atom basis
    let mut qtrans_new: Array2<f64> = Array2::zeros([atoms.len(), q_trans.dim().1]);

    for (mut q_tr_at, qtr_ao) in qtrans_new
        .axis_iter_mut(Axis(1))
        .zip(q_trans.axis_iter(Axis(1)))
    {
        let mut mu: usize = 0;
        for (idx, atom) in atoms.iter().enumerate() {
            for _ in 0..atom.n_orbs {
                q_tr_at[idx] += qtr_ao[mu];

                mu += 1;
            }
        }
    }

    // Get the xyz-coordinates.
    let xyz: Array2<f64> = get_xyz_2d(atoms);
    // and compute the transition dipole moments according to:
    // ->    ->
    // µ = ∑ r_i * q_i
    xyz.t().dot(&qtrans_new)
}

/// Computes the oscillator strength from the excitation energies, E, and the transition dipole
/// moments, µ, according to:
/// f = 2/3 * E * |µ|^2
pub fn oscillator_strength(
    exc_energies: ArrayView1<f64>,
    tr_dipoles: ArrayView2<f64>,
) -> Array1<f64> {
    2.0 / 3.0 * &exc_energies * tr_dipoles.t().dot(&tr_dipoles).diag()
}
