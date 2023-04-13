use crate::initialization::*;
use hashbrown::HashMap;
use libm;
use nalgebra::Vector3;
use ndarray::parallel::prelude::{IntoParallelIterator, IntoParallelRefIterator};
use ndarray::prelude::*;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use std::f64::consts::PI;

const PI_SQRT: f64 = 1.7724538509055159;

/// The decay constants for the gaussian charge fluctuations
/// are determined from the requirement d^2 E_atomic/d n^2 = U_H.
///
/// In the DFTB approximations with long-range correction one has
///
/// U_H = gamma_AA - 1/2 * 1/(2*l+1) gamma^lr_AA
///
/// where l is the angular momentum of the highest valence orbital
///
/// see "Implementation and benchmark of a long-range corrected functional
///      in the DFTB method" by V. Lutsker, B. Aradi and Th. Niehaus
///
/// Here, this equation is solved for sigmaA, the decay constant
/// of a gaussian.
pub fn gaussian_decay(unique_atoms: &[Atom]) -> HashMap<u8, f64> {
    let mut sigmas: HashMap<u8, f64> = HashMap::with_capacity(unique_atoms.len());
    for atom in unique_atoms.iter() {
        sigmas.insert(atom.number, 1.0 / (atom.hubbard * PI_SQRT));
    }
    return sigmas;
}

/// ## Gamma Function
/// gamma_AB = int F_A(r-RA) * 1/|RA-RB| * F_B(r-RB) d^3r
#[derive(Clone, Debug)]
pub enum GammaFunction {
    Slater {
        tau: HashMap<u8, f64>,
    },
    Gaussian {
        sigma: HashMap<u8, f64>,
        c: HashMap<(u8, u8), f64>,
        r_lr: f64,
    },
}

impl GammaFunction {
    pub(crate) fn initialize(&mut self) {
        match *self {
            GammaFunction::Gaussian {
                ref sigma,
                ref mut c,
                ref r_lr,
            } => {
                // Construct the C_AB matrix
                for z_a in sigma.keys() {
                    for z_b in sigma.keys() {
                        c.insert(
                            (*z_a, *z_b),
                            1.0 / (2.0
                                * (sigma[z_a].powi(2) + sigma[z_b].powi(2) + 0.5 * r_lr.powi(2)))
                            .sqrt(),
                        );
                    }
                }
            }
            _ => {}
        }
    }

    fn eval(&self, r: f64, z_a: u8, z_b: u8) -> f64 {
        let result: f64 = match *self {
            GammaFunction::Gaussian {
                sigma: _,
                ref c,
                r_lr: _,
            } => {
                assert!(r > 0.0);
                libm::erf(c[&(z_a, z_b)] * r) / r
            }
            GammaFunction::Slater { ref tau } => {
                let t_a: f64 = tau[&z_a];
                let t_b: f64 = tau[&z_b];
                if r.abs() < 1.0e-5 {
                    // R -> 0 limit
                    t_a * t_b * (t_a.powi(2) + 3.0 * t_a * t_b + t_b.powi(2))
                        / (2.0 * (t_a + t_b).powi(3))
                } else if (t_a - t_b).abs() < 1.0e-5 {
                    // t_A == t_b limit
                    let x: f64 = t_a * r;
                    (1.0 / r)
                        * (1.0
                            - (-t_a * r).exp() * (48.0 + 33.0 * x + 9.0 * x.powi(2) + x.powi(3))
                                / 48.0)
                } else {
                    // general case R != 0 and t_a != t_b
                    let denom_ab: f64 = t_b.powi(4)
                        * (t_b.powi(2) * (2.0 + t_a * r) - t_a.powi(2) * (6.0 + t_a * r));
                    let denom_ba: f64 = t_a.powi(4)
                        * (t_a.powi(2) * (2.0 + t_b * r) - t_b.powi(2) * (6.0 + t_b * r));
                    let num: f64 = 2.0 * (t_a.powi(2) - t_b.powi(2)).powi(3);
                    (1.0 / r)
                        * (1.0 + ((-t_a * r).exp() * denom_ab - (-t_b * r).exp() * denom_ba) / num)
                }
            }
        };
        return result;
    }

    fn eval_limit0(&self, z: u8) -> f64 {
        let result: f64 = match *self {
            GammaFunction::Gaussian {
                ref sigma,
                c: _,
                ref r_lr,
            } => 1.0 / (PI * (sigma[&z].powi(2) + 0.25 * r_lr.powi(2))).sqrt(),
            GammaFunction::Slater { ref tau } => (5.0 / 16.0) * tau[&z],
        };
        return result;
    }

    fn deriv(&self, r: f64, z_a: u8, z_b: u8) -> f64 {
        let result: f64 = match *self {
            GammaFunction::Gaussian {
                sigma: _,
                ref c,
                r_lr: _,
            } => {
                assert!(r > 0.0);
                let c_v: f64 = c[&(z_a, z_b)];
                2.0 * c_v / PI_SQRT * (-(c_v * r).powi(2)).exp() / r
                    - libm::erf(c_v * r) / r.powi(2)
            }
            GammaFunction::Slater { ref tau } => {
                let t_a: f64 = tau[&z_a];
                let t_b: f64 = tau[&z_b];
                if r.abs() < 1.0e-5 {
                    // R -> 0 limit
                    0.0
                } else if (t_a - t_b).abs() < 1.0e-5 {
                    // t_A == t_b limit
                    let x: f64 = t_a * r;
                    -1.0 / r.powi(2)
                        * (1.0
                            - (-x).exp()
                                * (1.0 + 1.0 / 48.0 * (x * (4.0 + x) * (12.0 + x * (3.0 + x)))))
                } else {
                    // general case R != 0 and t_a != t_b
                    let t_a_r: f64 = t_a * r;
                    let t_b_r: f64 = t_b * r;
                    let t_a2: f64 = t_a.powi(2);
                    let t_b2: f64 = t_b.powi(2);
                    let denom: f64 = 2.0 * (t_a2 - t_b2).powi(3);
                    let f_b: f64 =
                        (2.0 + t_b_r * (2.0 + t_b_r)) * t_a2 - (6.0 + t_b_r * (6.0 + t_b_r)) * t_b2;
                    let f_a: f64 =
                        (2.0 + t_a_r * (2.0 + t_a_r)) * t_b2 - (6.0 + t_a_r * (6.0 + t_a_r)) * t_a2;
                    -1.0 / r.powi(2)
                        * (1.0
                            - 1.0 / denom
                                * (t_a2.powi(2) * f_b * (-t_b_r).exp()
                                    - t_b2.powi(2) * f_a * (-t_a_r).exp()))
                }
            }
        };
        return result;
    }
}

/// Compute the atomwise Coulomb interaction between all atoms of one sets of atoms
pub fn gamma_atomwise(gamma_func: &GammaFunction, atoms: &[Atom], n_atoms: usize) -> Array2<f64> {
    let mut g0 = Array2::zeros((n_atoms, n_atoms));
    for (i, atomi) in atoms.iter().enumerate() {
        for (j, atomj) in atoms.iter().enumerate() {
            if i == j {
                g0[[i, j]] = gamma_func.eval_limit0(atomi.number);
            } else if i < j {
                g0[[i, j]] = gamma_func.eval((atomi - atomj).norm(), atomi.number, atomj.number);
            } else {
                g0[[i, j]] = g0[[j, i]];
            }
        }
    }
    return g0;
}

/// Compute the atomwise Coulomb interaction between two sets of atoms.
pub fn gamma_atomwise_ab(
    gamma_func: &GammaFunction,
    atoms_a: &[Atom],
    atoms_b: &[Atom],
    n_atoms_a: usize,
    n_atoms_b: usize,
) -> Array2<f64> {
    let mut g0 = Array2::zeros((n_atoms_a, n_atoms_b));
    for (i, atomi) in atoms_a.iter().enumerate() {
        for (j, atomj) in atoms_b.iter().enumerate() {
            g0[[i, j]] = gamma_func.eval((atomi - atomj).norm(), atomi.number, atomj.number);
        }
    }
    return g0;
}

pub fn gamma_gradients_atomwise(
    gamma_func: &GammaFunction,
    atoms: &[Atom],
    n_atoms: usize,
) -> Array3<f64> {
    let mut g1_val: Array2<f64> = Array2::zeros((n_atoms, n_atoms));
    let mut g1: Array3<f64> = Array3::zeros((3 * n_atoms, n_atoms, n_atoms));
    for (i, atomi) in atoms.iter().enumerate() {
        for (j, atomj) in atoms.iter().enumerate() {
            if i < j {
                let r = atomi - atomj;
                let r_ij: f64 = r.norm();
                let e_ij: Vector3<f64> = r / r_ij;
                g1_val[[i, j]] = gamma_func.deriv(r_ij, atomi.number, atomj.number);
                g1.slice_mut(s![(3 * i)..(3 * i + 3), i, j])
                    .assign(&Array1::from_iter((e_ij * g1_val[[i, j]]).iter().cloned()));
            } else if j < i {
                g1_val[[i, j]] = g1_val[[j, i]];
                let r = atomi - atomj;
                let e_ij: Vector3<f64> = r / r.norm();
                g1.slice_mut(s![(3 * i)..(3 * i + 3), i, j])
                    .assign(&Array::from_iter((e_ij * g1_val[[i, j]]).iter().cloned()));
            }
        }
    }
    return g1;
}

pub fn gamma_gradients_atomwise_2d(
    gamma_func: &GammaFunction,
    atoms: &[Atom],
    n_atoms: usize,
) -> Array2<f64> {
    let mut g1_val: Array2<f64> = Array2::zeros((n_atoms, n_atoms));
    let mut g1: Array2<f64> = Array2::zeros((3 * n_atoms, n_atoms));
    for (i, atomi) in atoms.iter().enumerate() {
        for (j, atomj) in atoms.iter().enumerate() {
            if i < j {
                let r = atomi - atomj;
                let r_ij: f64 = r.norm();
                let e_ij: Vector3<f64> = r / r_ij;
                g1_val[[i, j]] = gamma_func.deriv(r_ij, atomi.number, atomj.number);
                g1.slice_mut(s![(3 * i)..(3 * i + 3), j])
                    .assign(&Array1::from_iter((e_ij * g1_val[[i, j]]).iter().cloned()));
            } else if j < i {
                g1_val[[i, j]] = g1_val[[j, i]];
                let r = atomi - atomj;
                let e_ij: Vector3<f64> = r / r.norm();
                g1.slice_mut(s![(3 * i)..(3 * i + 3), j])
                    .assign(&Array::from_iter((e_ij * g1_val[[i, j]]).iter().cloned()));
            }
        }
    }
    return g1;
}

pub fn gamma_ao_wise(
    gamma_func: &GammaFunction,
    atoms: &[Atom],
    n_atoms: usize,
    n_orbs: usize,
) -> (Array2<f64>, Array2<f64>) {
    let g0: Array2<f64> = gamma_atomwise(gamma_func, &atoms, n_atoms);
    let mut g0_a0: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
    let mut mu: usize = 0;
    let mut nu: usize;
    for (i, atomi) in atoms.iter().enumerate() {
        for _ in 0..atomi.n_orbs {
            nu = 0;
            for (j, atomj) in atoms.iter().enumerate() {
                for _ in 0..atomj.n_orbs {
                    g0_a0[[mu, nu]] = g0[[i, j]];
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return (g0, g0_a0);
}

pub fn gamma_ao_wise_from_gamma_atomwise(
    gamma_atomwise: ArrayView2<f64>,
    atoms: &[Atom],
    n_orbs: usize,
) -> Array2<f64> {
    let mut g0_a0: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
    let mut mu: usize = 0;
    let mut nu: usize;
    for (atom_i, g0_i) in atoms.iter().zip(gamma_atomwise.outer_iter()) {
        for _ in 0..atom_i.n_orbs {
            nu = 0;
            for (atom_j, g0_ij) in atoms.iter().zip(g0_i.iter()) {
                for _ in 0..atom_j.n_orbs {
                    if mu <= nu {
                        g0_a0[[mu, nu]] = *g0_ij;
                        g0_a0[[nu, mu]] = *g0_ij;
                    }
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return g0_a0;
}

pub fn gamma_gradients_ao_wise(
    gamma_func: &GammaFunction,
    atoms: &[Atom],
    n_atoms: usize,
    n_orbs: usize,
) -> (Array3<f64>, Array3<f64>) {
    let g1: Array3<f64> = gamma_gradients_atomwise(gamma_func, &atoms, n_atoms);
    let mut g1_a0: Array3<f64> = Array3::zeros((3 * n_atoms, n_orbs, n_orbs));
    let mut mu: usize = 0;
    let mut nu: usize;
    for (i, atomi) in atoms.iter().enumerate() {
        for _ in 0..atomi.n_orbs {
            nu = 0;
            for (j, atomj) in atoms.iter().enumerate() {
                for _ in 0..atomj.n_orbs {
                    if i != j {
                        g1_a0
                            .slice_mut(s![(3 * i)..(3 * i + 3), mu, nu])
                            .assign(&g1.slice(s![(3 * i)..(3 * i + 3), i, j]));
                        g1_a0
                            .slice_mut(s![(3 * i)..(3 * i + 3), nu, mu])
                            .assign(&g1.slice(s![(3 * i)..(3 * i + 3), i, j]));
                    }
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return (g1, g1_a0);
}

pub fn gamma_gradients_ao_wise_from_atomwise(
    g1: ArrayView3<f64>,
    atoms: &[Atom],
    n_atoms: usize,
    n_orbs: usize,
) -> Array3<f64> {
    let mut g1_a0: Array3<f64> = Array3::zeros((3 * n_atoms, n_orbs, n_orbs));
    let mut mu: usize = 0;
    let mut nu: usize;
    for (i, atomi) in atoms.iter().enumerate() {
        for _ in 0..atomi.n_orbs {
            nu = 0;
            for (j, atomj) in atoms.iter().enumerate() {
                for _ in 0..atomj.n_orbs {
                    if i != j {
                        g1_a0
                            .slice_mut(s![(3 * i)..(3 * i + 3), mu, nu])
                            .assign(&g1.slice(s![(3 * i)..(3 * i + 3), i, j]));
                        g1_a0
                            .slice_mut(s![(3 * i)..(3 * i + 3), nu, mu])
                            .assign(&g1.slice(s![(3 * i)..(3 * i + 3), i, j]));
                    }
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return g1_a0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initialization::System;
    use crate::properties::Properties;
    use crate::utils::*;
    use approx::AbsDiffEq;

    pub const EPSILON: f64 = 1e-15;

    fn test_gamma_atomwise(molecule_and_properties: (&str, System, Properties)) {
        let name = molecule_and_properties.0;
        let molecule = molecule_and_properties.1;
        let props = molecule_and_properties.2;
        let atomic_numbers: Vec<u8> = molecule.atoms.iter().map(|atom| atom.number).collect();
        let gamma: Array2<f64> =
            gamma_atomwise(&molecule.gammafunction, &molecule.atoms, molecule.n_atoms);
        let gamma_ref: Array2<f64> = props
            .get("gamma_atomwise")
            .unwrap()
            .as_array2()
            .unwrap()
            .to_owned();

        assert!(
            gamma_ref.abs_diff_eq(&gamma, EPSILON),
            "Molecule: {}, Gamma (ref): {}  Gamma: {}",
            name,
            gamma_ref,
            gamma
        );
    }

    fn test_gamma_atomwise_lc(molecule_and_properties: (&str, System, Properties)) {
        let name = molecule_and_properties.0;
        let molecule = molecule_and_properties.1;
        let props = molecule_and_properties.2;
        let atomic_numbers: Vec<u8> = molecule.atoms.iter().map(|atom| atom.number).collect();
        let gamma: Array2<f64> = gamma_atomwise(
            &molecule.gammafunction_lc.unwrap(),
            &molecule.atoms,
            molecule.n_atoms,
        );
        let gamma_ref: Array2<f64> = props
            .get("gamma_atomwise_lc")
            .unwrap()
            .as_array2()
            .unwrap()
            .to_owned();

        assert!(
            gamma_ref.abs_diff_eq(&gamma, EPSILON),
            "Molecule: {}, Gamma-LC (ref): {}  Gamma-LC: {}",
            name,
            gamma_ref,
            gamma
        );
    }

    fn test_gamma_ao_wise(molecule_and_properties: (&str, System, Properties)) {
        let name = molecule_and_properties.0;
        let molecule = molecule_and_properties.1;
        let props = molecule_and_properties.2;
        let atomic_numbers: Vec<u8> = molecule.atoms.iter().map(|atom| atom.number).collect();
        let (g0, g0_ao): (Array2<f64>, Array2<f64>) = gamma_ao_wise(
            &molecule.gammafunction,
            &molecule.atoms,
            molecule.n_atoms,
            molecule.n_orbs,
        );
        let g0_ref: Array2<f64> = props
            .get("gamma_atomwise")
            .unwrap()
            .as_array2()
            .unwrap()
            .to_owned();
        let g0_ao_ref: Array2<f64> = props
            .get("gamma_ao_wise")
            .unwrap()
            .as_array2()
            .unwrap()
            .to_owned();
        assert!(
            g0_ref.abs_diff_eq(&g0, EPSILON),
            "Molecule: {}, Gamma-LC (ref): {}  Gamma-LC: {}",
            name,
            g0_ref,
            g0
        );
        assert!(
            g0_ao_ref.abs_diff_eq(&g0_ao, EPSILON),
            "Molecule: {}, Gamma-LC (ao basis) (ref): {}  Gamma-LC (ao basis): {}",
            name,
            g0_ao_ref,
            g0_ao
        );
    }

    fn test_gamma_ao_wise_lc(molecule_and_properties: (&str, System, Properties)) {
        let name = molecule_and_properties.0;
        let molecule = molecule_and_properties.1;
        let props = molecule_and_properties.2;
        let atomic_numbers: Vec<u8> = molecule.atoms.iter().map(|atom| atom.number).collect();
        let (g0, g0_ao): (Array2<f64>, Array2<f64>) = gamma_ao_wise(
            &molecule.gammafunction_lc.unwrap(),
            &molecule.atoms,
            molecule.n_atoms,
            molecule.n_orbs,
        );
        let g0_ref: Array2<f64> = props
            .get("gamma_atomwise_lc")
            .unwrap()
            .as_array2()
            .unwrap()
            .to_owned();
        let g0_ao_ref: Array2<f64> = props
            .get("gamma_ao_wise_lc")
            .unwrap()
            .as_array2()
            .unwrap()
            .to_owned();
        assert!(
            g0_ref.abs_diff_eq(&g0, EPSILON),
            "Molecule: {}, Gamma-LC (ref): {}  Gamma-LC: {}",
            name,
            g0_ref,
            g0
        );
        assert!(
            g0_ao_ref.abs_diff_eq(&g0_ao, EPSILON),
            "Molecule: {}, Gamma-LC (ao basis) (ref): {}  Gamma-LC (ao basis): {}",
            name,
            g0_ao_ref,
            g0_ao
        );
    }

    #[test]
    fn get_gamma_atomwise() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_gamma_atomwise(get_molecule(molecule, "no_lc_gs"));
        }
    }

    #[test]
    fn get_gamma_atomwise_lc() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_gamma_atomwise_lc(get_molecule(molecule, "no_lc_gs"));
        }
    }

    #[test]
    fn get_gamma_ao_wise() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_gamma_ao_wise(get_molecule(molecule, "no_lc_gs"));
        }
    }

    #[test]
    fn get_gamma_ao_wise_lc() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_gamma_ao_wise_lc(get_molecule(molecule, "no_lc_gs"));
        }
    }
}
