use crate::initialization::*;
use hashbrown::HashMap;
use libm;
use nalgebra::Vector3;
use ndarray::prelude::*;
use peroxide::fuga::PowOps;
use std::cmp::Ordering;
use std::f64::consts::PI;

const PI_SQRT: f64 = 1.7724538509055159;
const DAMPING_PARAM: f64 = 4.05;

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
pub fn gaussian_decay(unique_atoms: &[Atom]) -> (HashMap<u8, f64>, HashMap<u8, f64>) {
    let mut sigmas: HashMap<u8, f64> = HashMap::with_capacity(unique_atoms.len());
    let mut hubbards: HashMap<u8, f64> = HashMap::with_capacity(unique_atoms.len());
    for atom in unique_atoms.iter() {
        sigmas.insert(atom.number, 1.0 / (atom.hubbard[0] * PI_SQRT));
        hubbards.insert(atom.number, atom.hubbard[0]);
    }
    (sigmas, hubbards)
}

pub fn slater_decay(
    unique_atoms: &[Atom],
    use_damping: bool,
) -> (HashMap<u8, f64>, HashMap<u8, bool>) {
    let mut sigmas: HashMap<u8, f64> = HashMap::with_capacity(unique_atoms.len());
    let mut damping: HashMap<u8, bool> = HashMap::with_capacity(unique_atoms.len());
    for atom in unique_atoms.iter() {
        sigmas.insert(atom.number, 16.0 / 5.0 * atom.hubbard[0]);
        if atom.number == 1 && use_damping {
            damping.insert(atom.number, true);
        } else {
            damping.insert(atom.number, false);
        }
    }
    (sigmas, damping)
}

pub fn gaussian_decay_shell_resolved(unique_atoms: &[Atom]) -> HashMap<(u8, u8), f64> {
    let mut sigmas: HashMap<(u8, u8), f64> = HashMap::with_capacity(unique_atoms.len());
    for atom in unique_atoms.iter() {
        for orb in atom.valorbs.iter() {
            sigmas.insert(
                (atom.number, orb.l as u8),
                1.0 / (atom.hubbard[orb.l as usize] * PI_SQRT),
            );
        }
    }
    sigmas
}

pub fn slater_decay_shell_resolved(
    unique_atoms: &[Atom],
    use_damping: bool,
) -> (HashMap<(u8, u8), f64>, HashMap<u8, bool>) {
    let mut sigmas: HashMap<(u8, u8), f64> = HashMap::with_capacity(unique_atoms.len());
    let mut damping: HashMap<u8, bool> = HashMap::with_capacity(unique_atoms.len());
    for atom in unique_atoms.iter() {
        for orb in atom.valorbs.iter() {
            sigmas.insert(
                (atom.number, orb.l as u8),
                16.0 / 5.0 * atom.hubbard[orb.l as usize],
            );
        }
        if atom.number == 1 && use_damping {
            damping.insert(atom.number, true);
        } else {
            damping.insert(atom.number, false);
        }
    }
    (sigmas, damping)
}

/// ## Gamma Function
/// gamma_AB = int F_A(r-RA) * 1/|RA-RB| * F_B(r-RB) d^3r
#[derive(Clone, Debug)]
pub enum GammaFunction {
    Slater {
        tau: HashMap<u8, f64>,
        r_lr: f64,
        damping: HashMap<u8, bool>,
    },
    SlaterShellResolved {
        tau: HashMap<(u8, u8), f64>,
        r_lr: f64,
        damping: HashMap<u8, bool>,
    },
    Gaussian {
        hubbards: HashMap<u8, f64>,
        sigma: HashMap<u8, f64>,
        c: HashMap<(u8, u8), f64>,
        c_deriv: HashMap<(u8, u8), f64>,
        r_lr: f64,
    },
    GaussianShellResolved {
        sigma: HashMap<(u8, u8), f64>,
        c: HashMap<((u8, u8), (u8, u8)), f64>,
        r_lr: f64,
    },
}

impl GammaFunction {
    pub(crate) fn initialize(&mut self) {
        match *self {
            GammaFunction::Gaussian {
                ref hubbards,
                ref sigma,
                ref mut c,
                ref mut c_deriv,
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
                        c_deriv.insert(
                            (*z_a, *z_b),
                            -2_f64.sqrt()
                                / (PI.powf(1.5)
                                    * hubbards[z_a].powi(3)
                                    * (sigma[z_a].powi(2) + sigma[z_b].powi(2)).powf(1.5)),
                        );
                    }
                }
            }
            GammaFunction::GaussianShellResolved {
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
                hubbards: _,
                sigma: _,
                ref c,
                c_deriv: _,
                r_lr: _,
            } => {
                assert!(r > 0.0);
                libm::erf(c[&(z_a, z_b)] * r) / r
            }
            GammaFunction::Slater {
                ref tau,
                ref r_lr,
                ref damping,
            } => {
                if r_lr.abs() < 1.0e-5 {
                    let damping_a: bool = damping[&z_a];
                    let damping_b: bool = damping[&z_b];
                    let damping_val: f64 = if damping_a || damping_b {
                        self.damping(r, z_a, z_b)
                    } else {
                        1.0
                    };

                    let t_a: f64 = tau[&z_a];
                    let t_b: f64 = tau[&z_b];
                    if r.abs() < 1.0e-5 {
                        // R -> 0 limit
                        0.5 * ((t_a * t_b) / (t_a + t_b)
                            + (t_a * t_b).powi(2) / (t_a + t_b).powi(3))
                    } else if (t_a - t_b).abs() < 1.0e-5 {
                        // t_A == t_b limit
                        1.0 / r
                            - (-t_a * r).exp()
                                * (1.0 / r
                                    + 0.6875 * t_a
                                    + 0.1875 * r * t_a.powi(2)
                                    + r.powi(2) * t_a.powi(3) / 48.0)
                                * damping_val
                    } else {
                        // general case R != 0 and t_a != t_b
                        let part_1: f64 = (-t_a * r).exp()
                            * (0.5 * t_a * t_b.powi(4) / (t_a.powi(2) - t_b.powi(2)).powi(2)
                                - (t_b.powi(6) - 3.0 * t_b.powi(4) * t_a.powi(2))
                                    / (r * (t_a.powi(2) - t_b.powi(2)).powi(3)));
                        let part_2: f64 = (-t_b * r).exp()
                            * (0.5 * t_b * t_a.powi(4) / (t_b.powi(2) - t_a.powi(2)).powi(2)
                                - (t_a.powi(6) - 3.0 * t_a.powi(4) * t_b.powi(2))
                                    / (r * (t_b.powi(2) - t_a.powi(2)).powi(3)));

                        1.0 / r - (part_1 + part_2) * damping_val
                    }
                } else {
                    // factors
                    let t_a: f64 = tau[&z_a];
                    let t_b: f64 = tau[&z_b];
                    let w: f64 = 1.0 / r_lr;
                    let w_2: f64 = w.powi(2);
                    let t_a_2: f64 = t_a.powi(2);
                    let t_b_2: f64 = t_b.powi(2);
                    let t_a_4: f64 = t_a.powi(4);
                    let t_b_4: f64 = t_b.powi(4);

                    if (t_a - t_b).abs() < 1.0e-5 {
                        // t_A == t_b limit, but r!=0
                        let x: f64 = r * t_a;
                        let tmp_val: f64 = (-t_a * r).exp()
                            * (1.0 + 0.6875 * x + 0.1875 * x.powi(2) + x.powi(3) / 48.0);
                        let part_1: f64 = (1.0 / r) * (1.0 - tmp_val);
                        let part_2: f64 = -t_a_4.powi(2) / (t_a_2 - w_2).powi(4)
                            * ((tmp_val / r)
                                + (-t_a * r).exp()
                                    * (r.powi(2)
                                        * (3.0 * t_a_4 * w_2.powi(2)
                                            - 3.0 * t_a_2.powi(3) * w_2
                                            - t_a_2 * w_2.powi(3))
                                        + r * (15.0 * t_a.powi(3) * w_2.powi(2)
                                            - 21.0 * t_a.powi(5) * w_2
                                            - 3.0 * t_a * w_2.powi(3))
                                        + (15.0 * t_a_2 * w_2.powi(2)
                                            - 45.0 * t_a_4 * w_2
                                            - 3.0 * w_2.powi(3)))
                                    / (48.0 * t_a.powi(5)));
                        part_1
                            - (t_a_4.powi(2) / (t_a_2 - w_2).powi(4) * (-w * r).exp() / r + part_2)
                    } else {
                        // general case R != 0 and t_a != t_b
                        // part 1 \omega == 0
                        let tmp_1: f64 = (-t_a * r).exp()
                            * (0.5 * t_a * t_b.powi(4) / (t_a.powi(2) - t_b.powi(2)).powi(2)
                                - (t_b.powi(6) - 3.0 * t_b.powi(4) * t_a.powi(2))
                                    / (r * (t_a.powi(2) - t_b.powi(2)).powi(3)));
                        let tmp_2: f64 = (-t_b * r).exp()
                            * (0.5 * t_b * t_a.powi(4) / (t_b.powi(2) - t_a.powi(2)).powi(2)
                                - (t_a.powi(6) - 3.0 * t_a.powi(4) * t_b.powi(2))
                                    / (r * (t_b.powi(2) - t_a.powi(2)).powi(3)));
                        let part_1: f64 = 1.0 / r - (tmp_1 + tmp_2);

                        // part 2 \omega != 0
                        let prefac: f64 = (t_a_4 * t_b_4
                            / ((t_a_2 - w_2).powi(2) * (t_b_2 - w_2).powi(2)))
                            * (-w * r).exp()
                            / r;
                        // part with start index a
                        let val_1: f64 = t_a_2 - w_2;
                        let val_2: f64 = t_a_2 / val_1;
                        let val_3: f64 =
                            (t_b.powi(6) - 3.0 * t_a_2 * t_b_4 + 2.0 * w_2 * t_b_4) / r;
                        let val_4: f64 = val_3 * val_2.powi(2) / (t_a_2 - t_b_2).powi(3);
                        let val_5: f64 =
                            t_a * t_b_4 * 0.5 * val_2 / (t_b_2 - t_a_2).powi(2) - val_4;
                        let a_part: f64 = val_5 * (-t_a * r).exp();

                        let val_1: f64 = t_b_2 - w_2;
                        let val_2: f64 = t_b_2 / val_1;
                        let val_3: f64 =
                            (t_a.powi(6) - 3.0 * t_b_2 * t_a_4 + 2.0 * w_2 * t_a_4) / r;
                        let val_4: f64 = val_3 * val_2.powi(2) / (t_b_2 - t_a_2).powi(3);
                        let val_5: f64 =
                            t_b * t_a_4 * 0.5 * val_2 / (t_a_2 - t_b_2).powi(2) - val_4;
                        let b_part: f64 = val_5 * (-t_b * r).exp();

                        part_1 - prefac + a_part + b_part
                    }
                }
            }
            GammaFunction::GaussianShellResolved {
                sigma: _,
                ref c,
                r_lr: _,
            } => {
                assert!(r > 0.0);
                libm::erf(c[&((z_a, 0), (z_b, 0))] * r) / r
            }
            _ => 0.0,
        };
        result
    }

    fn eval_shell_resolved(&self, r: f64, z_a: u8, z_b: u8, l_a: u8, l_b: u8) -> f64 {
        let result: f64 = match *self {
            GammaFunction::GaussianShellResolved {
                sigma: _,
                ref c,
                r_lr: _,
            } => {
                assert!(r > 0.0);
                libm::erf(c[&((z_a, l_a), (z_b, l_b))] * r) / r
            }
            GammaFunction::SlaterShellResolved {
                ref tau,
                ref r_lr,
                ref damping,
            } => {
                if r_lr.abs() < 1.0e-5 {
                    let damping_a: bool = damping[&z_a];
                    let damping_b: bool = damping[&z_b];
                    let damping_val: f64 = if damping_a || damping_b {
                        self.damping(r, z_a, z_b)
                    } else {
                        1.0
                    };

                    let t_a: f64 = tau[&(z_a, l_a)];
                    let t_b: f64 = tau[&(z_b, l_b)];
                    if r.abs() < 1.0e-5 {
                        // R -> 0 limit
                        0.5 * ((t_a * t_b) / (t_a + t_b)
                            + (t_a * t_b).powi(2) / (t_a + t_b).powi(3))
                    } else if (t_a - t_b).abs() < 1.0e-5 {
                        // t_A == t_b limit
                        1.0 / r
                            - (-t_a * r).exp()
                                * (1.0 / r
                                    + 0.6875 * t_a
                                    + 0.1875 * r * t_a.powi(2)
                                    + r.powi(2) * t_a.powi(3) / 48.0)
                                * damping_val
                    } else {
                        // general case R != 0 and t_a != t_b
                        let part_1: f64 = (-t_a * r).exp()
                            * (0.5 * t_a * t_b.powi(4) / (t_a.powi(2) - t_b.powi(2)).powi(2)
                                - (t_b.powi(6) - 3.0 * t_b.powi(4) * t_a.powi(2))
                                    / (r * (t_a.powi(2) - t_b.powi(2)).powi(3)));
                        let part_2: f64 = (-t_b * r).exp()
                            * (0.5 * t_b * t_a.powi(4) / (t_b.powi(2) - t_a.powi(2)).powi(2)
                                - (t_a.powi(6) - 3.0 * t_a.powi(4) * t_b.powi(2))
                                    / (r * (t_b.powi(2) - t_a.powi(2)).powi(3)));

                        1.0 / r - (part_1 + part_2) * damping_val
                    }
                } else {
                    // factors
                    let t_a: f64 = tau[&(z_a, l_a)];
                    let t_b: f64 = tau[&(z_b, l_b)];
                    let w: f64 = 1.0 / r_lr;
                    let w_2: f64 = w.powi(2);
                    let t_a_2: f64 = t_a.powi(2);
                    let t_b_2: f64 = t_b.powi(2);
                    let t_a_4: f64 = t_a.powi(4);
                    let t_b_4: f64 = t_b.powi(4);

                    if (t_a - t_b).abs() < 1.0e-5 {
                        // t_A == t_b limit, but r!=0
                        let x: f64 = r * t_a;
                        let tmp_val: f64 = (-t_a * r).exp()
                            * (1.0 + 0.6875 * x + 0.1875 * x.powi(2) + x.powi(3) / 48.0);
                        let part_1: f64 = (1.0 / r) * (1.0 - tmp_val);
                        let part_2: f64 = -t_a_4.powi(2) / (t_a_2 - w_2).powi(4)
                            * ((tmp_val / r)
                                + (-t_a * r).exp()
                                    * (r.powi(2)
                                        * (3.0 * t_a_4 * w_2.powi(2)
                                            - 3.0 * t_a_2.powi(3) * w_2
                                            - t_a_2 * w_2.powi(3))
                                        + r * (15.0 * t_a.powi(3) * w_2.powi(2)
                                            - 21.0 * t_a.powi(5) * w_2
                                            - 3.0 * t_a * w_2.powi(3))
                                        + (15.0 * t_a_2 * w_2.powi(2)
                                            - 45.0 * t_a_4 * w_2
                                            - 3.0 * w_2.powi(3)))
                                    / (48.0 * t_a.powi(5)));
                        part_1
                            - (t_a_4.powi(2) / (t_a_2 - w_2).powi(4) * (-w * r).exp() / r + part_2)
                    } else {
                        // general case R != 0 and t_a != t_b
                        // part 1 \omega == 0
                        let tmp_1: f64 = (-t_a * r).exp()
                            * (0.5 * t_a * t_b.powi(4) / (t_a.powi(2) - t_b.powi(2)).powi(2)
                                - (t_b.powi(6) - 3.0 * t_b.powi(4) * t_a.powi(2))
                                    / (r * (t_a.powi(2) - t_b.powi(2)).powi(3)));
                        let tmp_2: f64 = (-t_b * r).exp()
                            * (0.5 * t_b * t_a.powi(4) / (t_b.powi(2) - t_a.powi(2)).powi(2)
                                - (t_a.powi(6) - 3.0 * t_a.powi(4) * t_b.powi(2))
                                    / (r * (t_b.powi(2) - t_a.powi(2)).powi(3)));
                        let part_1: f64 = 1.0 / r - (tmp_1 + tmp_2);

                        // part 2 \omega != 0
                        let prefac: f64 = (t_a_4 * t_b_4
                            / ((t_a_2 - w_2).powi(2) * (t_b_2 - w_2).powi(2)))
                            * (-w * r).exp()
                            / r;
                        // part with start index a
                        let val_1: f64 = t_a_2 - w_2;
                        let val_2: f64 = t_a_2 / val_1;
                        let val_3: f64 =
                            (t_b.powi(6) - 3.0 * t_a_2 * t_b_4 + 2.0 * w_2 * t_b_4) / r;
                        let val_4: f64 = val_3 * val_2.powi(2) / (t_a_2 - t_b_2).powi(3);
                        let val_5: f64 =
                            t_a * t_b_4 * 0.5 * val_2 / (t_b_2 - t_a_2).powi(2) - val_4;
                        let a_part: f64 = val_5 * (-t_a * r).exp();

                        let val_1: f64 = t_b_2 - w_2;
                        let val_2: f64 = t_b_2 / val_1;
                        let val_3: f64 =
                            (t_a.powi(6) - 3.0 * t_b_2 * t_a_4 + 2.0 * w_2 * t_a_4) / r;
                        let val_4: f64 = val_3 * val_2.powi(2) / (t_b_2 - t_a_2).powi(3);
                        let val_5: f64 =
                            t_b * t_a_4 * 0.5 * val_2 / (t_a_2 - t_b_2).powi(2) - val_4;
                        let b_part: f64 = val_5 * (-t_b * r).exp();

                        part_1 - prefac + a_part + b_part
                    }
                }
            }
            _ => 0.0,
        };
        result
    }

    fn eval_limit0(&self, z: u8) -> f64 {
        let result: f64 = match *self {
            GammaFunction::Gaussian {
                hubbards: _,
                ref sigma,
                c: _,
                c_deriv: _,
                ref r_lr,
            } => 1.0 / (PI * (sigma[&z].powi(2) + 0.25 * r_lr.powi(2))).sqrt(),
            GammaFunction::Slater {
                ref tau,
                ref r_lr,
                damping: _,
            } => {
                let tau_val: f64 = tau[&z];
                if r_lr.abs() < 1.0e-5 {
                    0.3125 * tau_val
                } else {
                    let w = 1.0 / r_lr;
                    let t_1: f64 = 0.3125 * tau_val;
                    let t_2: f64 = tau_val.powi(8) / (tau_val.powi(2) - w.powi(2)).powi(4)
                        * ((5.0 * tau_val.powi(6) + 15.0 * tau_val.powi(4) * w.powi(2)
                            - 5.0 * tau_val.powi(2) * w.powi(4)
                            + w.powi(6))
                            / (16.0 * tau_val.powi(5))
                            - w);
                    t_1 - t_2
                }
            }
            GammaFunction::GaussianShellResolved {
                ref sigma,
                c: _,
                ref r_lr,
            } => 1.0 / (PI * (sigma[&(z, 0)].powi(2) + 0.25 * r_lr.powi(2))).sqrt(),
            _ => 0.0,
        };
        result
    }

    fn eval_limit0_shell_resolved(&self, z: u8, l: u8) -> f64 {
        let result: f64 = match *self {
            GammaFunction::GaussianShellResolved {
                ref sigma,
                c: _,
                ref r_lr,
            } => 1.0 / (PI * (sigma[&(z, l)].powi(2) + 0.25 * r_lr.powi(2))).sqrt(),
            GammaFunction::SlaterShellResolved {
                ref tau,
                ref r_lr,
                damping: _,
            } => {
                let tau_val: f64 = tau[&(z, l)];
                if r_lr.abs() < 1.0e-5 {
                    0.3125 * tau_val
                } else {
                    let w = 1.0 / r_lr;
                    let t_1: f64 = 0.3125 * tau_val;
                    let t_2: f64 = tau_val.powi(8) / (tau_val.powi(2) - w.powi(2)).powi(4)
                        * ((5.0 * tau_val.powi(6) + 15.0 * tau_val.powi(4) * w.powi(2)
                            - 5.0 * tau_val.powi(2) * w.powi(4)
                            + w.powi(6))
                            / (16.0 * tau_val.powi(5))
                            - w);
                    t_1 - t_2
                }
            }
            _ => 0.0,
        };
        result
    }

    fn deriv(&self, r: f64, z_a: u8, z_b: u8) -> f64 {
        let result: f64 = match *self {
            GammaFunction::Gaussian {
                hubbards: _,
                sigma: _,
                ref c,
                c_deriv: _,
                r_lr: _,
            } => {
                assert!(r > 0.0);
                let c_v: f64 = c[&(z_a, z_b)];
                2.0 * c_v / PI_SQRT * (-(c_v * r).powi(2)).exp() / r
                    - libm::erf(c_v * r) / r.powi(2)
            }
            GammaFunction::GaussianShellResolved {
                sigma: _,
                ref c,
                r_lr: _,
            } => {
                assert!(r > 0.0);
                let c_v: f64 = c[&((z_a, 0), (z_b, 0))];
                2.0 * c_v / PI_SQRT * (-(c_v * r).powi(2)).exp() / r
                    - libm::erf(c_v * r) / r.powi(2)
            }
            GammaFunction::Slater {
                ref tau,
                ref r_lr,
                ref damping,
            } => {
                let t_a: f64 = tau[&z_a];
                let t_b: f64 = tau[&z_b];

                if r_lr.abs() < 1.0e-5 {
                    let damping_a: bool = damping[&z_a];
                    let damping_b: bool = damping[&z_b];
                    let mut damping_bool: bool = false;
                    if damping_a || damping_b {
                        damping_bool = true;
                    }
                    let damping_val: f64 = if damping_bool {
                        self.damping(r, z_a, z_b)
                    } else {
                        1.0
                    };

                    if r.abs() < 1.0e-5 {
                        // R -> 0 limit
                        0.0
                    } else if (t_a - t_b).abs() < 1.0e-5 {
                        // t_A == t_b limit
                        let x: f64 = t_a * r;
                        let part_1: f64 =
                            t_a * (1.0 + 0.6875 * x + 0.1875 * x.powi(2) + x.powi(3) / 48.0) / r;
                        let part_2: f64 =
                            -1.0 / r.powi(2) + 0.1875 * t_a.powi(2) + t_a.powi(3) / 24.0 * r;

                        let mut deriv: f64 =
                            -1.0 / r.powi(2) - ((-x).exp() * (part_2 - part_1)) * damping_val;
                        if damping_bool {
                            //calculate the damping derivative
                            let damping_deriv: f64 = self.damping_r_derivative(r, z_a, z_b);
                            let s_val: f64 = (-t_a * r).exp()
                                * (1.0 / r
                                    + 0.6875 * t_a
                                    + 0.1875 * r * t_a.powi(2)
                                    + r.powi(2) * t_a.powi(3) / 48.0);

                            deriv -= s_val * damping_deriv;
                        }
                        deriv
                    } else {
                        // general case R != 0 and t_a != t_b
                        let t_a2: f64 = t_a.powi(2);
                        let t_b2: f64 = t_b.powi(2);
                        let denom: f64 = (t_a2 - t_b2).powi(3);
                        let denom2: f64 = (t_b2 - t_a2).powi(3);

                        let f_a: f64 = 0.5 * t_a * t_b.powi(4) / (t_a2 - t_b2).powi(2)
                            - (t_b.powi(6) - 3.0 * t_b.powi(4) * t_a2) / (r * denom);
                        let f_b: f64 = 0.5 * t_b * t_a.powi(4) / (t_b2 - t_a2).powi(2)
                            - (t_a.powi(6) - 3.0 * t_a.powi(4) * t_b2) / (r * denom2);
                        let part_1: f64 =
                            t_a * (-t_a * r).exp() * f_a + t_b * (-t_b * r).exp() * f_b;

                        let part_2: f64 = (-t_a * r).exp()
                            * (t_b2.powi(3) - 3.0 * t_a2 * t_b2.powi(2))
                            / (r.powi(2) * denom);
                        let part_3: f64 = (-t_b * r).exp()
                            * (t_a2.powi(3) - 3.0 * t_b2 * t_a2.powi(2))
                            / (r.powi(2) * denom2);

                        let mut deriv: f64 =
                            -1.0 / r.powi(2) - (part_2 + part_3 - part_1) * damping_val;

                        if damping_bool {
                            //calculate the damping derivative
                            let damping_deriv: f64 = self.damping_r_derivative(r, z_a, z_b);
                            let s_1: f64 = (-t_a * r).exp()
                                * (0.5 * t_a * t_b.powi(4) / (t_a.powi(2) - t_b.powi(2)).powi(2)
                                    - (t_b.powi(6) - 3.0 * t_b.powi(4) * t_a.powi(2))
                                        / (r * (t_a.powi(2) - t_b.powi(2)).powi(3)));
                            let s_2: f64 = (-t_b * r).exp()
                                * (0.5 * t_b * t_a.powi(4) / (t_b.powi(2) - t_a.powi(2)).powi(2)
                                    - (t_a.powi(6) - 3.0 * t_a.powi(4) * t_b.powi(2))
                                        / (r * (t_b.powi(2) - t_a.powi(2)).powi(3)));

                            let s_val: f64 = s_1 + s_2;
                            deriv -= s_val * damping_deriv;
                        }

                        deriv
                    }
                } else if r.abs() < 1.0e-5 {
                    // R -> 0 limit
                    0.0
                } else if (t_a - t_b).abs() < 1.0e-5 {
                    // t_A == t_b limit
                    // factors
                    let t_a: f64 = tau[&z_a];
                    let w: f64 = 1.0 / r_lr;
                    let w_2: f64 = w.powi(2);
                    let w_4: f64 = w.powi(4);
                    let w_6: f64 = w.powi(6);
                    let t_a_2: f64 = t_a.powi(2);
                    let t_a_3: f64 = t_a.powi(3);
                    let t_a_4: f64 = t_a.powi(4);
                    let t_a_5: f64 = t_a.powi(5);
                    let t_a_6: f64 = t_a.powi(6);

                    let tmp: f64 = r.powi(2)
                        * (3.0 * t_a_4 * w_4 - 3.0 * t_a_6 * w_2 - t_a_2 * w_6)
                        + r * (15.0 * t_a_3 * w_4 - 21.0 * t_a_5 * w_2 - 3.0 * t_a * w_6)
                        + (15.0 * t_a_2 * w_4 - 45.0 * t_a_4 * w_2 - 3.0 * w_6);

                    let dtmp: f64 = 2.0 * r * (3.0 * t_a_4 * w_4 - 3.0 * t_a_6 * w_2 - t_a_2 * w_6)
                        + (15.0 * t_a_3 * w_4 - 21.0 * t_a_5 * w_2 - 3.0 * t_a * w_6);

                    let dtmp_1: f64 =
                        (dtmp * (-t_a * r).exp() - tmp * t_a * (-t_a * r).exp()) / (48.0 * t_a_5);

                    let dtmp_2: f64 = (2.0 * r * t_a_3 / 48.0 + 0.1875 * t_a_2 - 1.0 / r.powi(2))
                        * (-t_a * r).exp()
                        - (r.powi(2) * t_a_3 / 48.0 + 0.1875 * r * t_a_2 + 0.6875 * t_a + 1.0 / r)
                            * t_a
                            * (-t_a * r).exp();

                    let val: f64 = -1.0 / r.powi(2) - dtmp_2
                        + (t_a_4.powi(2) / (t_a_2 - w_2).powi(4))
                            * (dtmp_1
                                + dtmp_2
                                + w * (-w * r).exp() / r
                                + (-w * r).exp() / r.powi(2));

                    val
                } else {
                    // general case R != 0 and t_a != t_b
                    // factors
                    let t_a: f64 = tau[&z_a];
                    let t_b: f64 = tau[&z_b];
                    let w: f64 = 1.0 / r_lr;
                    let w_2: f64 = w.powi(2);
                    let t_a_2: f64 = t_a.powi(2);
                    let t_b_2: f64 = t_b.powi(2);
                    let t_a_4: f64 = t_a.powi(4);
                    let t_b_4: f64 = t_b.powi(4);
                    let t_a_6: f64 = t_a.powi(6);
                    let t_b_6: f64 = t_b.powi(6);

                    // parts with \omega == 0
                    let denom: f64 = (t_a_2 - t_b_2).powi(3);
                    let denom2: f64 = (t_b_2 - t_a_2).powi(3);
                    let f_a: f64 = (0.5 * t_a * t_b_4) / (t_b_2 - t_a_2).powi(2)
                        - (t_b_6 - 3.0 * t_b_4 * t_a_2) / (r * denom);
                    let f_b: f64 = (0.5 * t_b * t_a_4) / (t_a_2 - t_b_2).powi(2)
                        - (t_a_6 - 3.0 * t_a_4 * t_b_2) / (r * denom2);
                    let part_a: f64 = (t_b_6 - 3.0 * t_a_2 * t_b_4) / (r.powi(2) * denom);
                    let part_b: f64 = (t_a_6 - 3.0 * t_b_2 * t_a_4) / (r.powi(2) * denom2);
                    let no_omega: f64 = (-t_a * r).exp() * (part_a - t_a * f_a)
                        + (-t_b * r).exp() * (part_b - t_b * f_b);

                    // parts that contain omega
                    let prefac: f64 = t_a_2 / (t_a_2 - w_2);
                    let tmp_1: f64 = prefac.powi(2) / denom;
                    let dtmp: f64 =
                        tmp_1 * (t_b_6 - 3.0 * t_a_2 * t_b_4 + 2.0 * w_2 * t_b_4) / r.powi(2);
                    let tmp_2: f64 = tmp_1 * (t_b_6 - 3.0 * t_a_2 * t_b_4 + 2.0 * w_2 * t_b_4) / r;
                    let tmp_3: f64 = t_a * t_b_4 * 0.5 * prefac / (t_b_2 - t_a_2).powi(2) - tmp_2;
                    let w_part_a: f64 = (dtmp - tmp_3 * t_a) * (-t_a * r).exp();

                    // part 2 with b
                    let prefac: f64 = t_b_2 / (t_b_2 - w_2);
                    let tmp_1: f64 = prefac.powi(2) / denom2;
                    let dtmp: f64 =
                        tmp_1 * (t_a_6 - 3.0 * t_b_2 * t_a_4 + 2.0 * w_2 * t_a_4) / r.powi(2);
                    let tmp_2: f64 = tmp_1 * (t_a_6 - 3.0 * t_b_2 * t_a_4 + 2.0 * w_2 * t_a_4) / r;
                    let tmp_3: f64 = t_b * t_a_4 * 0.5 * prefac / (t_a_2 - t_b_2).powi(2) - tmp_2;
                    let w_part_b: f64 = (dtmp - tmp_3 * t_b) * (-t_b * r).exp();

                    let prefac: f64 = t_a_4 / (t_a_2 - w_2).powi(2) * t_b_4 / (t_b_2 - w_2).powi(2)
                        * (-w * (-w * r).exp() / r - (-w * r).exp() / r.powi(2));

                    // combine the different parts
                    let val: f64 = -1.0 / r.powi(2) - prefac + w_part_a + w_part_b - no_omega;
                    val
                }
            }
            _ => 0.0,
        };
        result
    }

    fn deriv_shell_resolved(&self, r: f64, z_a: u8, z_b: u8, l_a: u8, l_b: u8) -> f64 {
        let result: f64 = match *self {
            GammaFunction::GaussianShellResolved {
                sigma: _,
                ref c,
                r_lr: _,
            } => {
                assert!(r > 0.0);
                let c_v: f64 = c[&((z_a, l_a), (z_b, l_b))];
                2.0 * c_v / PI_SQRT * (-(c_v * r).powi(2)).exp() / r
                    - libm::erf(c_v * r) / r.powi(2)
            }
            GammaFunction::SlaterShellResolved {
                ref tau,
                ref r_lr,
                ref damping,
            } => {
                let t_a: f64 = tau[&(z_a, l_a)];
                let t_b: f64 = tau[&(z_b, l_b)];

                if r_lr.abs() < 1.0e-5 {
                    let damping_a: bool = damping[&z_a];
                    let damping_b: bool = damping[&z_b];
                    let mut damping_bool: bool = false;
                    if damping_a || damping_b {
                        damping_bool = true;
                    }
                    let damping_val: f64 = if damping_bool {
                        self.damping(r, z_a, z_b)
                    } else {
                        1.0
                    };

                    if r.abs() < 1.0e-5 {
                        // R -> 0 limit
                        0.0
                    } else if (t_a - t_b).abs() < 1.0e-5 {
                        // t_A == t_b limit
                        let x: f64 = t_a * r;
                        let part_1: f64 =
                            t_a * (1.0 + 0.6875 * x + 0.1875 * x.powi(2) + x.powi(3) / 48.0) / r;
                        let part_2: f64 =
                            -1.0 / r.powi(2) + 0.1875 * t_a.powi(2) + t_a.powi(3) / 24.0 * r;

                        let mut deriv: f64 =
                            -1.0 / r.powi(2) - ((-x).exp() * (part_2 - part_1)) * damping_val;
                        if damping_bool {
                            //calculate the damping derivative
                            let damping_deriv: f64 = self.damping_r_derivative(r, z_a, z_b);
                            let s_val: f64 = (-t_a * r).exp()
                                * (1.0 / r
                                    + 0.6875 * t_a
                                    + 0.1875 * r * t_a.powi(2)
                                    + r.powi(2) * t_a.powi(3) / 48.0);

                            deriv -= s_val * damping_deriv;
                        }
                        deriv
                    } else {
                        // general case R != 0 and t_a != t_b
                        let t_a2: f64 = t_a.powi(2);
                        let t_b2: f64 = t_b.powi(2);
                        let denom: f64 = (t_a2 - t_b2).powi(3);
                        let denom2: f64 = (t_b2 - t_a2).powi(3);

                        let f_a: f64 = 0.5 * t_a * t_b.powi(4) / (t_a2 - t_b2).powi(2)
                            - (t_b.powi(6) - 3.0 * t_b.powi(4) * t_a2) / (r * denom);
                        let f_b: f64 = 0.5 * t_b * t_a.powi(4) / (t_b2 - t_a2).powi(2)
                            - (t_a.powi(6) - 3.0 * t_a.powi(4) * t_b2) / (r * denom2);
                        let part_1: f64 =
                            t_a * (-t_a * r).exp() * f_a + t_b * (-t_b * r).exp() * f_b;

                        let part_2: f64 = (-t_a * r).exp()
                            * (t_b2.powi(3) - 3.0 * t_a2 * t_b2.powi(2))
                            / (r.powi(2) * denom);
                        let part_3: f64 = (-t_b * r).exp()
                            * (t_a2.powi(3) - 3.0 * t_b2 * t_a2.powi(2))
                            / (r.powi(2) * denom2);

                        let mut deriv: f64 =
                            -1.0 / r.powi(2) - (part_2 + part_3 - part_1) * damping_val;

                        if damping_bool {
                            //calculate the damping derivative
                            let damping_deriv: f64 = self.damping_r_derivative(r, z_a, z_b);
                            let s_1: f64 = (-t_a * r).exp()
                                * (0.5 * t_a * t_b.powi(4) / (t_a.powi(2) - t_b.powi(2)).powi(2)
                                    - (t_b.powi(6) - 3.0 * t_b.powi(4) * t_a.powi(2))
                                        / (r * (t_a.powi(2) - t_b.powi(2)).powi(3)));
                            let s_2: f64 = (-t_b * r).exp()
                                * (0.5 * t_b * t_a.powi(4) / (t_b.powi(2) - t_a.powi(2)).powi(2)
                                    - (t_a.powi(6) - 3.0 * t_a.powi(4) * t_b.powi(2))
                                        / (r * (t_b.powi(2) - t_a.powi(2)).powi(3)));

                            let s_val: f64 = s_1 + s_2;
                            deriv -= s_val * damping_deriv;
                        }

                        deriv
                    }
                } else if r.abs() < 1.0e-5 {
                    // R -> 0 limit
                    0.0
                } else if (t_a - t_b).abs() < 1.0e-5 {
                    // t_A == t_b limit
                    // factors
                    let w: f64 = 1.0 / r_lr;
                    let w_2: f64 = w.powi(2);
                    let w_4: f64 = w.powi(4);
                    let w_6: f64 = w.powi(6);
                    let t_a_2: f64 = t_a.powi(2);
                    let t_a_3: f64 = t_a.powi(3);
                    let t_a_4: f64 = t_a.powi(4);
                    let t_a_5: f64 = t_a.powi(5);
                    let t_a_6: f64 = t_a.powi(6);

                    let tmp: f64 = r.powi(2)
                        * (3.0 * t_a_4 * w_4 - 3.0 * t_a_6 * w_2 - t_a_2 * w_6)
                        + r * (15.0 * t_a_3 * w_4 - 21.0 * t_a_5 * w_2 - 3.0 * t_a * w_6)
                        + (15.0 * t_a_2 * w_4 - 45.0 * t_a_4 * w_2 - 3.0 * w_6);

                    let dtmp: f64 = 2.0 * r * (3.0 * t_a_4 * w_4 - 3.0 * t_a_6 * w_2 - t_a_2 * w_6)
                        + (15.0 * t_a_3 * w_4 - 21.0 * t_a_5 * w_2 - 3.0 * t_a * w_6);

                    let dtmp_1: f64 =
                        (dtmp * (-t_a * r).exp() - tmp * t_a * (-t_a * r).exp()) / (48.0 * t_a_5);

                    let dtmp_2: f64 = (2.0 * r * t_a_3 / 48.0 + 0.1875 * t_a_2 - 1.0 / r.powi(2))
                        * (-t_a * r).exp()
                        - (r.powi(2) * t_a_3 / 48.0 + 0.1875 * r * t_a_2 + 0.6875 * t_a + 1.0 / r)
                            * t_a
                            * (-t_a * r).exp();

                    let val: f64 = -1.0 / r.powi(2) - dtmp_2
                        + (t_a_4.powi(2) / (t_a_2 - w_2).powi(4))
                            * (dtmp_1
                                + dtmp_2
                                + w * (-w * r).exp() / r
                                + (-w * r).exp() / r.powi(2));

                    val
                } else {
                    // general case R != 0 and t_a != t_b
                    // factors
                    let w: f64 = 1.0 / r_lr;
                    let w_2: f64 = w.powi(2);
                    let t_a_2: f64 = t_a.powi(2);
                    let t_b_2: f64 = t_b.powi(2);
                    let t_a_4: f64 = t_a.powi(4);
                    let t_b_4: f64 = t_b.powi(4);
                    let t_a_6: f64 = t_a.powi(6);
                    let t_b_6: f64 = t_b.powi(6);

                    // parts with \omega == 0
                    let denom: f64 = (t_a_2 - t_b_2).powi(3);
                    let denom2: f64 = (t_b_2 - t_a_2).powi(3);
                    let f_a: f64 = (0.5 * t_a * t_b_4) / (t_b_2 - t_a_2).powi(2)
                        - (t_b_6 - 3.0 * t_b_4 * t_a_2) / (r * denom);
                    let f_b: f64 = (0.5 * t_b * t_a_4) / (t_a_2 - t_b_2).powi(2)
                        - (t_a_6 - 3.0 * t_a_4 * t_b_2) / (r * denom2);
                    let part_a: f64 = (t_b_6 - 3.0 * t_a_2 * t_b_4) / (r.powi(2) * denom);
                    let part_b: f64 = (t_a_6 - 3.0 * t_b_2 * t_a_4) / (r.powi(2) * denom2);
                    let no_omega: f64 = (-t_a * r).exp() * (part_a - t_a * f_a)
                        + (-t_b * r).exp() * (part_b - t_b * f_b);

                    // parts that contain omega
                    let prefac: f64 = t_a_2 / (t_a_2 - w_2);
                    let tmp_1: f64 = prefac.powi(2) / denom;
                    let dtmp: f64 =
                        tmp_1 * (t_b_6 - 3.0 * t_a_2 * t_b_4 + 2.0 * w_2 * t_b_4) / r.powi(2);
                    let tmp_2: f64 = tmp_1 * (t_b_6 - 3.0 * t_a_2 * t_b_4 + 2.0 * w_2 * t_b_4) / r;
                    let tmp_3: f64 = t_a * t_b_4 * 0.5 * prefac / (t_b_2 - t_a_2).powi(2) - tmp_2;
                    let w_part_a: f64 = (dtmp - tmp_3 * t_a) * (-t_a * r).exp();

                    // part 2 with b
                    let prefac: f64 = t_b_2 / (t_b_2 - w_2);
                    let tmp_1: f64 = prefac.powi(2) / denom2;
                    let dtmp: f64 =
                        tmp_1 * (t_a_6 - 3.0 * t_b_2 * t_a_4 + 2.0 * w_2 * t_a_4) / r.powi(2);
                    let tmp_2: f64 = tmp_1 * (t_a_6 - 3.0 * t_b_2 * t_a_4 + 2.0 * w_2 * t_a_4) / r;
                    let tmp_3: f64 = t_b * t_a_4 * 0.5 * prefac / (t_a_2 - t_b_2).powi(2) - tmp_2;
                    let w_part_b: f64 = (dtmp - tmp_3 * t_b) * (-t_b * r).exp();

                    let prefac: f64 = t_a_4 / (t_a_2 - w_2).powi(2) * t_b_4 / (t_b_2 - w_2).powi(2)
                        * (-w * (-w * r).exp() / r - (-w * r).exp() / r.powi(2));

                    // combine the different parts
                    let val: f64 = -1.0 / r.powi(2) - prefac + w_part_a + w_part_b - no_omega;
                    val
                }
            }
            _ => 0.0,
        };
        result
    }

    fn third_order(&self, r: f64, z_a: u8, z_b: u8) -> f64 {
        let result: f64 = match *self {
            GammaFunction::Slater {
                ref tau,
                ref r_lr,
                ref damping,
            } => {
                let t_a: f64 = tau[&z_a];
                let t_b: f64 = tau[&z_b];

                // check for long range correction
                let val: f64 = if r_lr.abs() < 1.0e-5 {
                    let damping_a: bool = damping[&z_a];
                    let damping_b: bool = damping[&z_b];
                    let mut damping_bool: bool = false;
                    if damping_a || damping_b {
                        damping_bool = true;
                    }
                    let damping_val: f64 = if damping_bool {
                        self.damping(r, z_a, z_b)
                    } else {
                        1.0
                    };

                    if r.abs() < 1.0e-5 {
                        // R -> 0 limit
                        0.3125
                    } else if (t_a - t_b).abs() < 1.0e-5 {
                        // t_A == t_b limit
                        let g: f64 = 1.0 / r
                            + 0.6875 * t_a
                            + 0.1875 * r * t_a.powi(2)
                            + r.powi(2) * t_a.powi(3) / 48.0;
                        let dg: f64 =
                            1.0 / 48.0 * (33.0 + 18.0 * t_a * r + 3.0 * t_a.powi(2) * r.powi(2));

                        let mut val: f64 =
                            (-(-t_a * r).exp() * dg + r * (-t_a * r).exp() * g) * damping_val;

                        if damping_bool {
                            let deriv: f64 = self.damping_u_derivative(r, z_a, z_b);
                            let s_val: f64 = (-t_a * r).exp()
                                * (1.0 / r
                                    + 0.6875 * t_a
                                    + 0.1875 * r * t_a.powi(2)
                                    + r.powi(2) * t_a.powi(3) / 48.0);
                            val -= 0.3125 * deriv * s_val;
                        }

                        val
                    } else {
                        // general case R != 0 and t_a != t_b
                        let t_a2: f64 = t_a.powi(2);
                        let t_b2: f64 = t_b.powi(2);
                        let f: f64 = 0.5 * t_a * t_b.powi(4) / (t_a.powi(2) - t_b.powi(2)).powi(2)
                            - (t_b.powi(6) - 3.0 * t_b.powi(4) * t_a.powi(2))
                                / (r * (t_a.powi(2) - t_b.powi(2)).powi(3));
                        let df_a: f64 = -0.5 * (t_b2.powi(3) + 3.0 * t_a2 * t_b2.powi(2))
                            / (t_a2 - t_b2).powi(3)
                            - 12.0 * t_a.powi(3) * t_b2.powi(2) / ((t_a2 - t_b2).powi(4) * r);
                        let df_b: f64 = 2.0 * t_b.powi(3) * t_a.powi(3) / (t_b2 - t_a2).powi(3)
                            + 12.0 * t_b2.powi(2) * t_a.powi(3) / ((t_b2 - t_a2).powi(4) * r);

                        let mut val: f64 = (-(-t_a * r).exp() * df_a + r * (-t_a * r).exp() * f
                            - (-t_b * r).exp() * df_b)
                            * damping_val;

                        if damping_bool {
                            let deriv: f64 = self.damping_u_derivative(r, z_a, z_b);
                            let s_1: f64 = (-t_a * r).exp()
                                * (0.5 * t_a * t_b.powi(4) / (t_a.powi(2) - t_b.powi(2)).powi(2)
                                    - (t_b.powi(6) - 3.0 * t_b.powi(4) * t_a.powi(2))
                                        / (r * (t_a.powi(2) - t_b.powi(2)).powi(3)));
                            let s_2: f64 = (-t_b * r).exp()
                                * (0.5 * t_b * t_a.powi(4) / (t_b.powi(2) - t_a.powi(2)).powi(2)
                                    - (t_a.powi(6) - 3.0 * t_a.powi(4) * t_b.powi(2))
                                        / (r * (t_b.powi(2) - t_a.powi(2)).powi(3)));
                            let s_val: f64 = s_1 + s_2;
                            val -= 0.3125 * deriv * s_val;
                        }

                        val
                    }
                } else {
                    0.0
                };
                val * 3.2
            }
            GammaFunction::Gaussian {
                hubbards: _,
                ref sigma,
                c: _,
                ref c_deriv,
                r_lr: _,
            } => {
                (-r.powi(2) / (2.0 * (sigma[&z_a].powi(2) + sigma[&z_b].powi(2)))).exp()
                    * c_deriv[&(z_a, z_b)]
            }
            _ => 0.0,
        };
        result
    }

    fn third_order_derivative(&self, r: f64, z_a: u8, z_b: u8) -> f64 {
        let result: f64 = match *self {
            GammaFunction::Slater {
                ref tau,
                ref r_lr,
                ref damping,
            } => {
                let t_a: f64 = tau[&z_a];
                let t_b: f64 = tau[&z_b];

                // check for long range correction
                let val: f64 = if r_lr.abs() < 1.0e-5 {
                    let damping_a: bool = damping[&z_a];
                    let damping_b: bool = damping[&z_b];
                    let mut damping_bool: bool = false;
                    if damping_a || damping_b {
                        damping_bool = true;
                    }
                    let damping_val: f64 = if damping_bool {
                        self.damping(r, z_a, z_b)
                    } else {
                        1.0
                    };

                    if r.abs() < 1.0e-5 {
                        // R -> 0 limit
                        0.0
                    } else if (t_a - t_b).abs() < 1.0e-5 {
                        let g: f64 = 1.0 / r
                            + 0.6875 * t_a
                            + 0.1875 * r * t_a.powi(2)
                            + r.powi(2) * t_a.powi(3) / 48.0;
                        let dg_alpha: f64 =
                            1.0 / 48.0 * (33.0 + 18.0 * t_a * r + 3.0 * t_a.powi(2) * r.powi(2));
                        let dg_r: f64 =
                            -1.0 / r.powi(2) + 0.1875 * t_a.powi(2) + t_a.powi(3) / 24.0 * r;
                        let d2g_alpha: f64 = 3.0 / 8.0 * t_a + 0.125 * t_a.powi(2) * r;
                        let term_1: f64 = (t_a * r - 1.0) * g;
                        let term_2: f64 = -t_a * dg_alpha - r * dg_r + d2g_alpha;

                        let mut result: f64 = (-t_a * r).exp() * (term_1 + term_2) * damping_val;

                        if damping_bool {
                            let damp_u: f64 = self.damping_u_derivative(r, z_a, z_b);
                            let damp_r: f64 = self.damping_r_derivative(r, z_a, z_b);
                            let damp_ur: f64 = self.damping_u_r_derivative(r, z_a, z_b);
                            let ds_du: f64 = (-t_a * r).exp() * (dg_alpha - r * g);
                            let ds_du_dh_dr: f64 = ds_du * damp_r * 3.2;
                            let alpha_g: f64 = t_a * g;
                            let ds_dr: f64 = (-t_a * r).exp() * (dg_r - alpha_g);
                            let ds_dr_dh_du: f64 = ds_dr * damp_u;
                            let s_val: f64 = (-t_a * r).exp()
                                * (1.0 / r
                                    + 0.6875 * t_a
                                    + 0.1875 * r * t_a.powi(2)
                                    + r.powi(2) * t_a.powi(3) / 48.0);
                            let s_dh_dudr: f64 = s_val * damp_ur;

                            result += 0.3125 * (ds_du_dh_dr + ds_dr_dh_du + s_dh_dudr);
                        }

                        result
                    } else {
                        let t_a2: f64 = t_a.powi(2);
                        let t_b2: f64 = t_b.powi(2);
                        let f: f64 = 0.5 * t_a * t_b.powi(4) / (t_a.powi(2) - t_b.powi(2)).powi(2)
                            - (t_b.powi(6) - 3.0 * t_b.powi(4) * t_a.powi(2))
                                / (r * (t_a.powi(2) - t_b.powi(2)).powi(3));
                        let df_a: f64 = -0.5 * (t_b2.powi(3) + 3.0 * t_a2 * t_b2.powi(2))
                            / (t_a2 - t_b2).powi(3)
                            - 12.0 * t_a.powi(3) * t_b2.powi(2) / ((t_a2 - t_b2).powi(4) * r);
                        let denom: f64 = (t_a2 - t_b2).powi(3);
                        let df_r: f64 =
                            (t_b2.powi(3) - 3.0 * t_a2 * t_b2.powi(2)) / (r.powi(2) * denom);
                        let d2f_dalpha_dr: f64 =
                            12.0 * t_a.powi(3) * t_b2.powi(2) / ((t_a2 - t_b2).powi(4) * r.powi(2));
                        let term_1: f64 = (-t_a * r).exp()
                            * ((t_a * r - 1.0) * f - t_a * df_a - r * df_r + d2f_dalpha_dr);

                        let d2fb_dalpha_dr: f64 = -12.0 * t_a.powi(3) * t_b2.powi(2)
                            / ((t_b2 - t_a2).powi(4) * r.powi(2));
                        let df_b: f64 = 2.0 * t_b.powi(3) * t_a.powi(3) / (t_b2 - t_a2).powi(3)
                            + 12.0 * t_b2.powi(2) * t_a.powi(3) / ((t_b2 - t_a2).powi(4) * r);
                        let term_2: f64 = (-t_b * r).exp() * (d2fb_dalpha_dr - t_b * df_b);

                        let mut result: f64 = (term_1 + term_2) * damping_val;

                        if damping_bool {
                            let damp_u: f64 = self.damping_u_derivative(r, z_a, z_b);
                            let damp_r: f64 = self.damping_r_derivative(r, z_a, z_b);
                            let damp_ur: f64 = self.damping_u_r_derivative(r, z_a, z_b);
                            let s_1: f64 = (-t_a * r).exp()
                                * (0.5 * t_a * t_b.powi(4) / (t_a.powi(2) - t_b.powi(2)).powi(2)
                                    - (t_b.powi(6) - 3.0 * t_b.powi(4) * t_a.powi(2))
                                        / (r * (t_a.powi(2) - t_b.powi(2)).powi(3)));
                            let s_2: f64 = (-t_b * r).exp()
                                * (0.5 * t_b * t_a.powi(4) / (t_b.powi(2) - t_a.powi(2)).powi(2)
                                    - (t_a.powi(6) - 3.0 * t_a.powi(4) * t_b.powi(2))
                                        / (r * (t_b.powi(2) - t_a.powi(2)).powi(3)));
                            let s_val: f64 = s_1 + s_2;
                            let s_dh_dudr: f64 = s_val * damp_ur;
                            let ds_du: f64 = (-t_a * r).exp() * df_a - r * (-t_a * r).exp() * f
                                + (-t_b * r).exp() * df_b;
                            let ds_du_dh_dr: f64 = ds_du * damp_r * 3.2;

                            let denom2: f64 = (t_b2 - t_a2).powi(3);
                            let f_b: f64 = 0.5 * t_b * t_a.powi(4) / (t_b2 - t_a2).powi(2)
                                - (t_a.powi(6) - 3.0 * t_a.powi(4) * t_b2) / (r * denom2);
                            let ds_part_1: f64 =
                                t_a * (-t_a * r).exp() * f + t_b * (-t_b * r).exp() * f_b;
                            let ds_part_2: f64 = (-t_a * r).exp() * df_r;
                            let ds_part_3: f64 = (-t_b * r).exp()
                                * (t_a2.powi(3) - 3.0 * t_b2 * t_a2.powi(2))
                                / (r.powi(2) * denom2);
                            let ds_dr: f64 = ds_part_2 + ds_part_3 - ds_part_1;
                            let ds_dr_dh_du: f64 = ds_dr * damp_u;

                            result += 0.3125 * (ds_du_dh_dr + ds_dr_dh_du + s_dh_dudr);
                        }

                        result
                    }
                } else {
                    0.0
                };
                val * (-3.2)
            }
            GammaFunction::Gaussian {
                hubbards: _,
                ref sigma,
                c: _,
                ref c_deriv,
                r_lr: _,
            } => {
                if r.abs() < 1.0e-5 {
                    0.0
                } else {
                    (-r.powi(2) / (2.0 * (sigma[&z_a].powi(2) + sigma[&z_b].powi(2)))).exp()
                        * c_deriv[&(z_a, z_b)]
                        * (-1.0)
                        * r
                        / (sigma[&z_a].powi(2) + sigma[&z_b].powi(2))
                }
            }
            _ => 0.0,
        };
        result
    }

    fn damping(&self, r: f64, z_a: u8, z_b: u8) -> f64 {
        let result: f64 = match *self {
            GammaFunction::Slater {
                ref tau,
                ref r_lr,
                damping: _,
            } => {
                let val: f64 = if r_lr.abs() < 1.0e-5 {
                    let t_a: f64 = tau[&z_a];
                    let t_b: f64 = tau[&z_b];
                    if r.abs() < 1.0e-5 {
                        // R -> 0 limit
                        1.0
                    } else if (t_a - t_b).abs() < 1.0e-5 {
                        (-(0.3125 * t_a).powf(DAMPING_PARAM) * r.powi(2)).exp()
                    } else {
                        (-(0.5 * 0.3125 * (t_a + t_b)).powf(DAMPING_PARAM) * r.powi(2)).exp()
                    }
                } else {
                    0.0
                };
                val
            }
            GammaFunction::Gaussian {
                hubbards: _,
                sigma: _,
                c: _,
                c_deriv: _,
                r_lr: _,
            } => 1.0,
            _ => 0.0,
        };
        result
    }

    fn damping_r_derivative(&self, r: f64, z_a: u8, z_b: u8) -> f64 {
        let result: f64 = match *self {
            GammaFunction::Slater {
                ref tau,
                ref r_lr,
                damping: _,
            } => {
                let val: f64 = if r_lr.abs() < 1.0e-5 {
                    let t_a: f64 = tau[&z_a];
                    let t_b: f64 = tau[&z_b];
                    if r.abs() < 1.0e-5 {
                        // R -> 0 limit
                        0.0
                    } else if (t_a - t_b).abs() < 1.0e-5 {
                        -2.0 * r * (0.3125 * t_a).powf(DAMPING_PARAM)
                    } else {
                        -2.0 * r * (0.5 * 0.3125 * (t_a + t_b)).powf(DAMPING_PARAM)
                    }
                } else {
                    0.0
                };
                val * self.damping(r, z_a, z_b)
            }
            GammaFunction::Gaussian {
                hubbards: _,
                sigma: _,
                c: _,
                c_deriv: _,
                r_lr: _,
            } => 1.0,
            _ => 0.0,
        };
        result
    }

    fn damping_u_derivative(&self, r: f64, z_a: u8, z_b: u8) -> f64 {
        let result: f64 = match *self {
            GammaFunction::Slater {
                ref tau,
                ref r_lr,
                damping: _,
            } => {
                let val: f64 = if r_lr.abs() < 1.0e-5 {
                    let t_a: f64 = tau[&z_a];
                    let t_b: f64 = tau[&z_b];
                    if r.abs() < 1.0e-5 {
                        // R -> 0 limit
                        1.0
                    } else if (t_a - t_b).abs() < 1.0e-5 {
                        -0.5 * (0.3125 * t_a).powf(DAMPING_PARAM - 1.0) * r.powi(2) * DAMPING_PARAM
                    } else {
                        -0.5 * (0.3125 * 0.5 * (t_a + t_b)).powf(DAMPING_PARAM - 1.0)
                            * r.powi(2)
                            * DAMPING_PARAM
                    }
                } else {
                    0.0
                };
                val * self.damping(r, z_a, z_b)
            }
            GammaFunction::Gaussian {
                hubbards: _,
                sigma: _,
                c: _,
                c_deriv: _,
                r_lr: _,
            } => 1.0,
            _ => 0.0,
        };
        result
    }

    fn damping_u_r_derivative(&self, r: f64, z_a: u8, z_b: u8) -> f64 {
        let result: f64 = match *self {
            GammaFunction::Slater {
                ref tau,
                ref r_lr,
                damping: _,
            } => {
                let val: f64 = if r_lr.abs() < 1.0e-5 {
                    let t_a: f64 = tau[&z_a];
                    let t_b: f64 = tau[&z_b];
                    if r.abs() < 1.0e-5 {
                        // R -> 0 limit
                        1.0
                    } else if (t_a - t_b).abs() < 1.0e-5 {
                        r * DAMPING_PARAM
                            * (0.3125 * t_a).powf(DAMPING_PARAM - 1.0)
                            * (r.powi(2) * (0.3125 * t_a).powf(DAMPING_PARAM) - 1.0)
                    } else {
                        r * DAMPING_PARAM
                            * (0.3125 * 0.5 * (t_a + t_b)).powf(DAMPING_PARAM - 1.0)
                            * (r.powi(2) * (0.3125 * 0.5 * (t_a + t_b)).powf(DAMPING_PARAM) - 1.0)
                    }
                } else {
                    0.0
                };
                val * self.damping(r, z_a, z_b)
            }
            GammaFunction::Gaussian {
                hubbards: _,
                sigma: _,
                c: _,
                c_deriv: _,
                r_lr: _,
            } => 1.0,
            _ => 0.0,
        };
        result
    }
}

/// Compute the atomwise Coulomb interaction between all atoms of one sets of atoms
pub fn gamma_atomwise(gamma_func: &GammaFunction, atoms: &[Atom], n_atoms: usize) -> Array2<f64> {
    let mut g0 = Array2::zeros((n_atoms, n_atoms));
    for (i, atomi) in atoms.iter().enumerate() {
        for (j, atomj) in atoms.iter().enumerate() {
            match i.cmp(&j) {
                Ordering::Equal => {
                    g0[[i, j]] = gamma_func.eval_limit0(atomi.number);
                }
                Ordering::Less => {
                    g0[[i, j]] =
                        gamma_func.eval((atomi - atomj).norm(), atomi.number, atomj.number);
                }
                Ordering::Greater => {
                    g0[[i, j]] = g0[[j, i]];
                }
            }
            // if i == j {
            //     g0[[i, j]] = gamma_func.eval_limit0(atomi.number);
            // } else if i < j {
            //     g0[[i, j]] = gamma_func.eval((atomi - atomj).norm(), atomi.number, atomj.number);
            // } else {
            //     g0[[i, j]] = g0[[j, i]];
            // }
        }
    }
    g0
}

pub fn gamma_third_order(
    gamma_func: &GammaFunction,
    atoms: &[Atom],
    n_atoms: usize,
    hubbard_derivs: &[f64],
) -> Array2<f64> {
    // get the derivatives of the hubbard U in respect to the charges
    let mut unique_atoms: Vec<usize> = Vec::new();
    for atomi in atoms.iter() {
        if !unique_atoms.contains(&(atomi.number as usize)) {
            unique_atoms.push(atomi.number as usize);
        }
    }
    unique_atoms.sort();
    let mut hubbard_q_derivs: HashMap<u8, f64> = HashMap::new();
    for (idx, number) in unique_atoms.iter().enumerate() {
        hubbard_q_derivs.insert(*number as u8, hubbard_derivs[idx]);
    }

    // calculate the \Gamma matrix for third order interactions
    let mut g_hubbard_d: Array2<f64> = Array2::zeros((n_atoms, n_atoms));
    for (i, atomi) in atoms.iter().enumerate() {
        for (j, atomj) in atoms.iter().enumerate() {
            match i.cmp(&j) {
                Ordering::Equal => {
                    g_hubbard_d[[i, j]] = 0.5
                        * gamma_func.third_order(0.0, atomi.number, atomi.number)
                        * hubbard_q_derivs[&atomi.number];
                }
                Ordering::Less => {
                    let dist: f64 = (atomi - atomj).norm();
                    g_hubbard_d[[i, j]] = gamma_func.third_order(dist, atomi.number, atomj.number)
                        * hubbard_q_derivs[&atomi.number];
                    g_hubbard_d[[j, i]] = gamma_func.third_order(dist, atomj.number, atomi.number)
                        * hubbard_q_derivs[&atomj.number];
                }
                Ordering::Greater => {}
            }
            // if i == j {
            //     g_hubbard_d[[i, j]] = 0.5
            //         * gamma_func.third_order(0.0, atomi.number, atomi.number)
            //         * hubbard_q_derivs[&atomi.number];
            // } else if i < j {
            //     let dist: f64 = (atomi - atomj).norm();
            //     g_hubbard_d[[i, j]] = gamma_func.third_order(dist, atomi.number, atomj.number)
            //         * hubbard_q_derivs[&atomi.number];
            //     g_hubbard_d[[j, i]] = gamma_func.third_order(dist, atomj.number, atomi.number)
            //         * hubbard_q_derivs[&atomj.number];
            // }
        }
    }
    g_hubbard_d
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
    g0
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
            match i.cmp(&j) {
                Ordering::Less => {
                    let r = atomi - atomj;
                    let r_ij: f64 = r.norm();
                    let e_ij: Vector3<f64> = r / r_ij;
                    g1_val[[i, j]] = gamma_func.deriv(r_ij, atomi.number, atomj.number);
                    g1.slice_mut(s![(3 * i)..(3 * i + 3), i, j])
                        .assign(&Array1::from_iter((e_ij * g1_val[[i, j]]).iter().cloned()));
                }
                Ordering::Greater => {
                    g1_val[[i, j]] = g1_val[[j, i]];
                    let r = atomi - atomj;
                    let e_ij: Vector3<f64> = r / r.norm();
                    g1.slice_mut(s![(3 * i)..(3 * i + 3), i, j])
                        .assign(&Array::from_iter((e_ij * g1_val[[i, j]]).iter().cloned()));
                }
                Ordering::Equal => {}
            }
            // if i < j {
            //     let r = atomi - atomj;
            //     let r_ij: f64 = r.norm();
            //     let e_ij: Vector3<f64> = r / r_ij;
            //     g1_val[[i, j]] = gamma_func.deriv(r_ij, atomi.number, atomj.number);
            //     g1.slice_mut(s![(3 * i)..(3 * i + 3), i, j])
            //         .assign(&Array1::from_iter((e_ij * g1_val[[i, j]]).iter().cloned()));
            //     // g1.slice_mut(s![(3 * i)..(3 * i + 3), j, i])
            //     //     .assign(&Array1::from_iter((e_ij * g1_val[[i, j]]).iter().cloned()));
            // } else if j < i {
            //     g1_val[[i, j]] = g1_val[[j, i]];
            //     let r = atomi - atomj;
            //     let e_ij: Vector3<f64> = r / r.norm();
            //     g1.slice_mut(s![(3 * i)..(3 * i + 3), i, j])
            //         .assign(&Array::from_iter((e_ij * g1_val[[i, j]]).iter().cloned()));
            //     // g1.slice_mut(s![(3 * i)..(3 * i + 3), j, i])
            //     //     .assign(&Array::from_iter((e_ij * g1_val[[i, j]]).iter().cloned()));
            // }
        }
    }
    g1
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
    g1
}

pub fn gamma_third_order_derivative(
    gamma_func: &GammaFunction,
    atoms: &[Atom],
    n_atoms: usize,
    hubbard_derivs: &[f64],
) -> Array3<f64> {
    // get the derivatives of the hubbard U in respect to the charges
    let mut unique_atoms: Vec<usize> = Vec::new();
    for atomi in atoms.iter() {
        if !unique_atoms.contains(&(atomi.number as usize)) {
            unique_atoms.push(atomi.number as usize);
        }
    }
    unique_atoms.sort();
    let mut hubbard_q_derivs: HashMap<u8, f64> = HashMap::new();
    for (idx, number) in unique_atoms.iter().enumerate() {
        hubbard_q_derivs.insert(*number as u8, hubbard_derivs[idx]);
    }

    let mut g1: Array3<f64> = Array3::zeros((3 * n_atoms, n_atoms, n_atoms));
    for (i, atomi) in atoms.iter().enumerate() {
        for (j, atomj) in atoms.iter().enumerate() {
            if i != j {
                let r = atomi - atomj;
                let r_ij: f64 = r.norm();
                let e_ij: Vector3<f64> = r / r_ij;
                let g1_val: f64 =
                    gamma_func.third_order_derivative(r_ij, atomi.number, atomj.number)
                        * hubbard_q_derivs[&atomi.number];
                let g2_val: f64 =
                    gamma_func.third_order_derivative(r_ij, atomj.number, atomi.number)
                        * hubbard_q_derivs[&atomj.number];
                g1.slice_mut(s![(3 * i)..(3 * i + 3), i, j])
                    .assign(&Array1::from_iter((e_ij * g1_val).iter().cloned()));
                g1.slice_mut(s![(3 * i)..(3 * i + 3), j, i])
                    .assign(&Array1::from_iter((e_ij * g2_val).iter().cloned()));
            }
        }
    }
    g1
}

pub fn gamma_ao_wise(
    gamma_func: &GammaFunction,
    atoms: &[Atom],
    n_atoms: usize,
    n_orbs: usize,
) -> (Array2<f64>, Array2<f64>) {
    let g0: Array2<f64> = gamma_atomwise(gamma_func, atoms, n_atoms);
    let mut g0_a0: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
    let mut mu: usize = 0;
    let mut nu: usize;
    for (i, atomi) in atoms.iter().enumerate() {
        for _ in 0..atomi.n_orbs {
            nu = 0;
            for (j, atomj) in atoms.iter().enumerate() {
                for _ in 0..atomj.n_orbs {
                    g0_a0[[mu, nu]] = g0[[i, j]];
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    (g0, g0_a0)
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
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    g0_a0
}

pub fn gamma_gradients_ao_wise(
    gamma_func: &GammaFunction,
    atoms: &[Atom],
    n_atoms: usize,
    n_orbs: usize,
) -> (Array3<f64>, Array3<f64>) {
    let g1: Array3<f64> = gamma_gradients_atomwise(gamma_func, atoms, n_atoms);
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
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    (g1, g1_a0)
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
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    g1_a0
}

pub fn gamma_ao_wise_shell_resolved(
    gamma_func: &GammaFunction,
    atoms: &[Atom],
    n_orbs: usize,
) -> Array2<f64> {
    let mut g0_a0: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
    let mut mu: usize = 0;
    let mut nu: usize;
    for (i, atomi) in atoms.iter().enumerate() {
        for orb_i in atomi.valorbs.iter() {
            nu = 0;
            for (j, atomj) in atoms.iter().enumerate() {
                for orb_j in atomj.valorbs.iter() {
                    g0_a0[[mu, nu]] = match i.cmp(&j) {
                        Ordering::Equal => {
                            gamma_func.eval_limit0_shell_resolved(atomi.number, orb_i.l as u8)
                        }
                        Ordering::Less => gamma_func.eval_shell_resolved(
                            (atomi.xyz - atomj.xyz).norm(),
                            atomi.number,
                            atomj.number,
                            orb_i.l as u8,
                            orb_j.l as u8,
                        ),
                        Ordering::Greater => g0_a0[[nu, mu]],
                    };
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    g0_a0
}

pub fn gamma_gradients_ao_wise_shell_resolved(
    gamma_func: &GammaFunction,
    atoms: &[Atom],
    n_atoms: usize,
    n_orbs: usize,
) -> Array3<f64> {
    let mut g1: Array3<f64> = Array3::zeros((3 * n_atoms, n_orbs, n_orbs));
    let mut g1_val: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
    let mut mu: usize = 0;
    let mut nu: usize;
    for (i, atomi) in atoms.iter().enumerate() {
        for orb_i in atomi.valorbs.iter() {
            nu = 0;
            for (j, atomj) in atoms.iter().enumerate() {
                for orb_j in atomj.valorbs.iter() {
                    match i.cmp(&j) {
                        Ordering::Equal => {}
                        Ordering::Greater => {
                            let r = atomi.xyz - atomj.xyz;
                            let e_ij: Vector3<f64> = r / r.norm();
                            g1.slice_mut(s![(3 * i)..(3 * i + 3), mu, nu]).assign(
                                &Array::from_iter((e_ij * g1_val[[nu, mu]]).iter().cloned()),
                            );
                            g1.slice_mut(s![(3 * i)..(3 * i + 3), nu, mu]).assign(
                                &Array::from_iter((e_ij * g1_val[[nu, mu]]).iter().cloned()),
                            );
                        }
                        Ordering::Less => {
                            let r = atomi.xyz - atomj.xyz;
                            let r_ij: f64 = r.norm();
                            let e_ij: Vector3<f64> = r / r_ij;
                            g1_val[[mu, nu]] = gamma_func.deriv_shell_resolved(
                                r_ij,
                                atomi.number,
                                atomj.number,
                                orb_i.l as u8,
                                orb_j.l as u8,
                            );
                            g1.slice_mut(s![(3 * i)..(3 * i + 3), mu, nu]).assign(
                                &Array1::from_iter((e_ij * g1_val[[mu, nu]]).iter().cloned()),
                            );
                            g1.slice_mut(s![(3 * i)..(3 * i + 3), nu, mu]).assign(
                                &Array1::from_iter((e_ij * g1_val[[mu, nu]]).iter().cloned()),
                            );
                        }
                    }
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    g1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initialization::System;
    use crate::properties::Properties;
    use crate::utils::tests::{get_molecule, AVAILAIBLE_MOLECULES};

    pub const EPSILON: f64 = 1e-15;

    fn test_gamma_atomwise(molecule_and_properties: (&str, System, Properties)) {
        let name = molecule_and_properties.0;
        let molecule = molecule_and_properties.1;
        let props = molecule_and_properties.2;
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
            test_gamma_atomwise(get_molecule(molecule));
        }
    }

    #[test]
    fn get_gamma_atomwise_lc() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_gamma_atomwise_lc(get_molecule(molecule));
        }
    }

    #[test]
    fn get_gamma_ao_wise() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_gamma_ao_wise(get_molecule(molecule));
        }
    }

    #[test]
    fn get_gamma_ao_wise_lc() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_gamma_ao_wise_lc(get_molecule(molecule));
        }
    }
}
