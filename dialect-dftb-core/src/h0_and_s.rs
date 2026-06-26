use dialect_base::defaults::PROXIMITY_CUTOFF;
use crate::parameters::*;
use crate::atom::Atom;
use crate::slako_transformations::*;
use ndarray::prelude::*;

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
    for atomi in atoms_a.iter() {
        // iterate over orbitals on center i
        for orbi in atomi.valorbs.iter() {
            // iterate over atoms
            let mut nu: usize = 0;
            for atomj in atoms_b.iter() {
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
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    (s, h0)
}

/// Computes the H0 and S matrix elements for a single molecule.
pub fn s_supersystem(n_orbs: usize, atoms: &[Atom], skt: &SlaterKoster) -> Array2<f64> {
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
                            }
                        } else if mu == nu {
                            assert_eq!(atomi.number, atomj.number);
                            s[[mu, nu]] = 1.0;
                        } else {
                            s[[mu, nu]] = s[[nu, mu]];
                        }
                    }
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    s
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
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    (s, h0)
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
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    (grad_s, grad_h0)
}

pub fn h0_and_s_gradients_atom_specific(
    atom_idx: usize,
    atomi: &Atom,
    atoms: &[Atom],
    n_orbs: usize,
    skt: &SlaterKoster,
) -> (Array3<f64>, Array3<f64>) {
    let mut grad_h0: Array3<f64> = Array3::zeros((3, n_orbs, n_orbs));
    let mut grad_s: Array3<f64> = Array3::zeros((3, n_orbs, n_orbs));

    let mut mu: usize = 0;
    // get the mu index of the atom
    for (idx, atomi) in atoms.iter().enumerate() {
        if idx < atom_idx {
            for _orb in atomi.valorbs.iter() {
                mu += 1;
            }
        }
    }

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
                        if atom_idx != j {
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

                    grad_s.slice_mut(s![0..3, mu, nu]).assign(&s_deriv);
                    grad_h0.slice_mut(s![0..3, mu, nu]).assign(&h0_deriv);
                    // S and H0 are hermitian/symmetric
                    grad_s.slice_mut(s![0..3, nu, mu]).assign(&s_deriv);
                    grad_h0.slice_mut(s![0..3, nu, mu]).assign(&h0_deriv);
                }
                nu += 1;
            }
        }
        mu += 1;
    }
    (grad_s, grad_h0)
}

pub fn h0_gradient(atoms: &[Atom], n_orbs: usize, skt: &SlaterKoster) -> Array3<f64> {
    let n_atoms: usize = atoms.len();
    let mut grad_h0: Array3<f64> = Array3::zeros((3 * n_atoms, n_orbs, n_orbs));

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
                        let mut h0_deriv: Array1<f64> = Array1::zeros([3]);
                        if atomi <= atomj {
                            if i != j {
                                // the hardcoded Slater-Koster rules compute the gradient
                                // with respect to r = posj - posi
                                // but we want the gradient with respect to posi, so an additional
                                // minus sign is introduced
                                let (r, x, y, z): (f64, f64, f64, f64) =
                                    directional_cosines(&atomi.xyz, &atomj.xyz);
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

                        grad_h0
                            .slice_mut(s![(3 * i)..(3 * i + 3), mu, nu])
                            .assign(&h0_deriv);
                        grad_h0
                            .slice_mut(s![(3 * i)..(3 * i + 3), nu, mu])
                            .assign(&h0_deriv);
                    }
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    grad_h0
}

pub fn s_gradient(atoms: &[Atom], n_orbs: usize, skt: &SlaterKoster) -> Array3<f64> {
    let n_atoms: usize = atoms.len();
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
                        }

                        grad_s
                            .slice_mut(s![(3 * i)..(3 * i + 3), mu, nu])
                            .assign(&s_deriv);
                        // S and H0 are hermitian/symmetric
                        grad_s
                            .slice_mut(s![(3 * i)..(3 * i + 3), nu, mu])
                            .assign(&s_deriv);
                    }
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    grad_s
}
