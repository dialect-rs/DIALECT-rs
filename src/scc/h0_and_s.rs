use crate::defaults::PROXIMITY_CUTOFF;
use crate::initialization::parameters::*;
use crate::initialization::{Atom, Geometry};
use crate::param::slako_transformations::*;
use ndarray::prelude::*;
use std::collections::HashMap;

/// Computes the H0 and S outer diagonal block for two sets of atoms
pub fn h0_and_s_ab(
    n_orbs_a: usize,
    n_orbs_b: usize,
    atoms_a: &[Atom],
    atoms_b: &[Atom],
    skt: &SlaterKoster,
) -> (Array2<f64>, Array2<f64>) {
    let mut h0: Array2<f64> = Array2::zeros((n_orbs_a, n_orbs_b));
    let mut s: Array2<f64> = Array2::zeros((n_orbs_a, n_orbs_b));
    // iterate over atoms
    let mut mu: usize = 0;
    for (i, atomi) in atoms_a.iter().enumerate() {
        // iterate over orbitals on center i
        for orbi in atomi.valorbs.iter() {
            // iterate over atoms
            let mut nu: usize = 0;
            for (j, atomj) in atoms_b.iter().enumerate() {
                // iterate over orbitals on center j
                for orbj in atomj.valorbs.iter() {
                    //if geometry.proximities.as_ref().unwrap()[[i, j]] {
                    if (atomi - atomj).norm() < PROXIMITY_CUTOFF {
                        if atomi <= atomj {
                            let (r, x, y, z): (f64, f64, f64, f64) =
                                directional_cosines(&atomi.xyz, &atomj.xyz);
                            s[[mu, nu]] = slako_transformation(
                                r,
                                x,
                                y,
                                z,
                                &skt.get(atomi.kind, atomj.kind).s_spline,
                                orbi.l,
                                orbi.m,
                                orbj.l,
                                orbj.m,
                            );
                            h0[[mu, nu]] = slako_transformation(
                                r,
                                x,
                                y,
                                z,
                                &skt.get(atomi.kind, atomj.kind).h_spline,
                                orbi.l,
                                orbi.m,
                                orbj.l,
                                orbj.m,
                            );
                        } else {
                            let (r, x, y, z): (f64, f64, f64, f64) =
                                directional_cosines(&atomj.xyz, &atomi.xyz);
                            s[[mu, nu]] = slako_transformation(
                                r,
                                x,
                                y,
                                z,
                                &skt.get(atomj.kind, atomi.kind).s_spline,
                                orbj.l,
                                orbj.m,
                                orbi.l,
                                orbi.m,
                            );
                            h0[[mu, nu]] = slako_transformation(
                                r,
                                x,
                                y,
                                z,
                                &skt.get(atomj.kind, atomi.kind).h_spline,
                                orbj.l,
                                orbj.m,
                                orbi.l,
                                orbi.m,
                            );
                        }
                    }
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return (s, h0);
}

/// Computes the H0 and S matrix elements for a single molecule.
pub fn h0_and_s(n_orbs: usize, atoms: &[Atom], skt: &SlaterKoster) -> (Array2<f64>, Array2<f64>) {
    let mut h0: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
    let mut s: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
    // iterate over atoms
    let mut mu: usize = 0;
    for (i, atomi) in atoms.iter().enumerate() {
        // iterate over orbitals on center i
        for orbi in atomi.valorbs.iter() {
            // iterate over atoms
            let mut nu: usize = 0;
            for (j, atomj) in atoms.iter().enumerate() {
                // iterate over orbitals on center j
                for orbj in atomj.valorbs.iter() {
                    //if geometry.proximities.as_ref().unwrap()[[i, j]] {
                    if (atomi - atomj).norm() < PROXIMITY_CUTOFF {
                        if mu < nu {
                            if atomi <= atomj {
                                if i != j {
                                    let (r, x, y, z): (f64, f64, f64, f64) =
                                        directional_cosines(&atomi.xyz, &atomj.xyz);
                                    s[[mu, nu]] = slako_transformation(
                                        r,
                                        x,
                                        y,
                                        z,
                                        &skt.get(atomi.kind, atomj.kind).s_spline,
                                        orbi.l,
                                        orbi.m,
                                        orbj.l,
                                        orbj.m,
                                    );
                                    h0[[mu, nu]] = slako_transformation(
                                        r,
                                        x,
                                        y,
                                        z,
                                        &skt.get(atomi.kind, atomj.kind).h_spline,
                                        orbi.l,
                                        orbi.m,
                                        orbj.l,
                                        orbj.m,
                                    );
                                }
                            } else {
                                let (r, x, y, z): (f64, f64, f64, f64) =
                                    directional_cosines(&atomj.xyz, &atomi.xyz);
                                s[[mu, nu]] = slako_transformation(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &skt.get(atomj.kind, atomi.kind).s_spline,
                                    orbj.l,
                                    orbj.m,
                                    orbi.l,
                                    orbi.m,
                                );
                                h0[[mu, nu]] = slako_transformation(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &skt.get(atomj.kind, atomi.kind).h_spline,
                                    orbj.l,
                                    orbj.m,
                                    orbi.l,
                                    orbi.m,
                                );
                            }
                        } else if mu == nu {
                            assert_eq!(atomi.number, atomj.number);
                            h0[[mu, nu]] = orbi.energy;
                            s[[mu, nu]] = 1.0;
                        } else {
                            s[[mu, nu]] = s[[nu, mu]];
                            h0[[mu, nu]] = h0[[nu, mu]];
                        }
                    }
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return (s, h0);
}

// gradients of overlap matrix S and 0-order hamiltonian matrix H0
// using Slater-Koster Rules
//
// Parameters:
// ===========
// atomlist: list of tuples (Zi,[xi,yi,zi]) of atom types and positions
// valorbs: list of valence orbitals with quantum numbers (ni,li,mi)
// SKT: Slater Koster table
// Mproximity: M[i,j] == 1, if the atoms i and j are close enough
// so that the gradients for matrix elements
// between orbitals on i and j should be computed
pub fn h0_and_s_gradients(
    atoms: &[Atom],
    n_orbs: usize,
    skt: &SlaterKoster,
) -> (Array3<f64>, Array3<f64>) {
    let n_atoms: usize = atoms.len();
    let mut grad_h0: Array3<f64> = Array3::zeros((3 * n_atoms, n_orbs, n_orbs));
    let mut grad_s: Array3<f64> = Array3::zeros((3 * n_atoms, n_orbs, n_orbs));

    // iterate over atoms
    let mut mu: usize = 0;
    for (i, atomi) in atoms.iter().enumerate() {
        // iterate over orbitals on center i
        for orbi in atomi.valorbs.iter() {
            // iterate over atoms
            let mut nu: usize = 0;
            for (j, atomj) in atoms.iter().enumerate() {
                // iterate over orbitals on center j
                for orbj in atomj.valorbs.iter() {
                    if (atomi - atomj).norm() < PROXIMITY_CUTOFF && mu != nu {
                        let mut s_deriv: Array1<f64> = Array1::zeros([3]);
                        let mut h0_deriv: Array1<f64> = Array1::zeros([3]);
                        if atomi <= atomj {
                            if i != j {
                                // the hardcoded Slater-Koster rules compute the gradient
                                // with respect to r = posj - posi
                                // but we want the gradient with respect to posi, so an additional
                                // minus sign is introduced
                                let (r, x, y, z): (f64, f64, f64, f64) =
                                    directional_cosines(&atomi.xyz, &atomj.xyz);
                                s_deriv = -1.0
                                    * slako_transformation_gradients(
                                        r,
                                        x,
                                        y,
                                        z,
                                        &skt.get(atomi.kind, atomj.kind).s_spline,
                                        orbi.l,
                                        orbi.m,
                                        orbj.l,
                                        orbj.m,
                                    );
                                h0_deriv = -1.0
                                    * slako_transformation_gradients(
                                        r,
                                        x,
                                        y,
                                        z,
                                        &skt.get(atomi.kind, atomj.kind).h_spline,
                                        orbi.l,
                                        orbi.m,
                                        orbj.l,
                                        orbj.m,
                                    );
                            }
                        } else {
                            // swap atoms if Zj > Zi, since posi and posj are swapped, the gradient
                            // with respect to r = posi - posj equals the gradient with respect to
                            // posi, so no additional minus sign is needed.
                            let (r, x, y, z): (f64, f64, f64, f64) =
                                directional_cosines(&atomj.xyz, &atomi.xyz);
                            s_deriv = slako_transformation_gradients(
                                r,
                                x,
                                y,
                                z,
                                &skt.get(atomi.kind, atomj.kind).s_spline,
                                orbj.l,
                                orbj.m,
                                orbi.l,
                                orbi.m,
                            );
                            h0_deriv = slako_transformation_gradients(
                                r,
                                x,
                                y,
                                z,
                                &skt.get(atomi.kind, atomj.kind).h_spline,
                                orbj.l,
                                orbj.m,
                                orbi.l,
                                orbi.m,
                            );
                        }

                        grad_s
                            .slice_mut(s![(3 * i)..(3 * i + 3), mu, nu])
                            .assign(&s_deriv);
                        grad_h0
                            .slice_mut(s![(3 * i)..(3 * i + 3), mu, nu])
                            .assign(&h0_deriv);
                        // S and H0 are hermitian/symmetric
                        grad_s
                            .slice_mut(s![(3 * i)..(3 * i + 3), nu, mu])
                            .assign(&s_deriv);
                        grad_h0
                            .slice_mut(s![(3 * i)..(3 * i + 3), nu, mu])
                            .assign(&h0_deriv);
                    }
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return (grad_s, grad_h0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initialization::Properties;
    use crate::initialization::System;
    use crate::utils::*;
    use approx::AbsDiffEq;

    pub const EPSILON: f64 = 1e-15;

    fn test_h0_and_s(molecule_and_properties: (&str, System, Properties)) {
        let name = molecule_and_properties.0;
        let molecule = molecule_and_properties.1;
        let props = molecule_and_properties.2;
        let (s, h0): (Array2<f64>, Array2<f64>) =
            h0_and_s(molecule.n_orbs, &molecule.atoms, &molecule.slako);
        let s_ref: Array2<f64> = props.get("S").unwrap().as_array2().unwrap().to_owned();
        let h0_ref: Array2<f64> = props.get("H0").unwrap().as_array2().unwrap().to_owned();

        assert!(
            s_ref.abs_diff_eq(&s, EPSILON),
            "Molecule: {}, S (ref): {}  S: {}",
            name,
            s_ref,
            s
        );

        assert!(
            h0_ref.abs_diff_eq(&h0, EPSILON),
            "Molecule: {}, H0 (ref): {}  H0: {}",
            name,
            h0_ref,
            h0
        );
    }

    #[test]
    fn get_h0_and_s() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_h0_and_s(get_molecule(molecule, "no_lc_gs"));
        }
    }
}
