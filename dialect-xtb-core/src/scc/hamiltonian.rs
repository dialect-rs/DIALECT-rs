use dialect_base::constants::{BOHR_TO_ANGS, HARTREE_TO_EV};
use dialect_base::defaults::PROXIMITY_CUTOFF;
use crate::initialization::atom::XtbAtom;
use crate::initialization::basis::{Basis, ContractedBasisfunction};
use crate::parameters::*;
use ndarray::prelude::*;
use rayon::prelude::*;

/// Compute H0 matrix using pre-computed coordination numbers.
/// This is used in FMO where CN is computed at the supersystem level.
pub fn h0_xtb1_with_cn(
    atoms: &[XtbAtom],
    s: ArrayView2<f64>,
    basis: &Basis,
    cn_numbers: ArrayView1<f64>,
) -> Array2<f64> {
    let mut h0: Array2<f64> = Array2::zeros((basis.nbas, basis.nbas));

    for shell_i in basis.shells.iter() {
        for idx_i in shell_i.sph_start..shell_i.sph_end {
            let atomi: &XtbAtom = &atoms[shell_i.atom_index];
            let cn_1: f64 = cn_numbers[shell_i.atom_index];
            for shell_j in basis.shells.iter() {
                let atomj: &XtbAtom = &atoms[shell_j.atom_index];
                let cn_2: f64 = cn_numbers[shell_j.atom_index];

                let distance: f64 = (atomi - atomj).norm();
                if distance < PROXIMITY_CUTOFF {
                    for idx_j in shell_j.sph_start..shell_j.sph_end {
                        if idx_i == idx_j {
                            let self_energy_term: f64 = get_self_energy_values_new(
                                atomi.number,
                                atomj.number,
                                cn_1,
                                cn_2,
                                shell_i.shell_index,
                                shell_j.shell_index,
                            );
                            h0[[idx_i, idx_j]] = self_energy_term;
                        } else if idx_i <= idx_j {
                            // xTB applies the element-pair scaling only to
                            // valence-valence shell pairs; pairs involving a
                            // polarization shell use 1.0.
                            let scaling_constant: f64 =
                                if shell_i.polarization || shell_j.polarization {
                                    1.0
                                } else {
                                    calculate_pair_scaling_param(
                                        atomi.number,
                                        atomj.number,
                                        shell_i.angular_momentum,
                                        shell_j.angular_momentum,
                                        shell_i.shell_index,
                                        shell_j.shell_index,
                                    )
                                };
                            let pauling_diff: f64 = (PAULING_EN[atomi.number as usize - 1]
                                - PAULING_EN[atomj.number as usize - 1])
                                .powi(2);
                            let en_term: f64 =
                                if shell_i.polarization == false && shell_j.polarization == false {
                                    1.0 + EN_SHELL_PARAM * pauling_diff
                                } else {
                                    1.0
                                };
                            let hueckel_const: f64 = get_hueckel_constants_new(
                                atomi.number,
                                atomj.number,
                                shell_i.angular_momentum,
                                shell_j.angular_momentum,
                                shell_i.polarization,
                                shell_j.polarization,
                            );
                            let self_energy_term: f64 = get_self_energy_values_new(
                                atomi.number,
                                atomj.number,
                                cn_1,
                                cn_2,
                                shell_i.shell_index,
                                shell_j.shell_index,
                            );
                            let pi_term: f64 = get_pi_term(
                                distance,
                                atomi.number as usize,
                                atomj.number as usize,
                                shell_i.angular_momentum,
                                shell_j.angular_momentum,
                            );
                            let h0_val: f64 = scaling_constant
                                * hueckel_const
                                * self_energy_term
                                * s[[idx_i, idx_j]]
                                * en_term
                                * pi_term;

                            h0[[idx_i, idx_j]] = h0_val;
                            h0[[idx_j, idx_i]] = h0_val;
                        }
                    }
                }
            }
        }
    }
    h0
}

/// Build the GFN1-xTB core Hamiltonian H0. Diagonal elements are the
/// CN-dependent shell self-energies; off-diagonal elements are
/// `K_AB * 0.5*(H_i+H_j) * scaling * Π(R) * S_ij` (Hückel constant,
/// averaged self-energies, pair scaling, bond polynomial, overlap).
pub fn h0_xtb1_new(atoms: &[XtbAtom], s: ArrayView2<f64>, basis: &Basis) -> Array2<f64> {
    let mut h0: Array2<f64> = Array2::zeros((basis.nbas, basis.nbas));
    // calculate the coordination numbers
    let cn_numbers: Array1<f64> = calculate_coordination_numbers(atoms);

    // iterate over the basis functions
    for shell_i in basis.shells.iter() {
        // loop over number of angular components
        for idx_i in (shell_i.sph_start..shell_i.sph_end) {
            let atomi: &XtbAtom = &atoms[shell_i.atom_index];
            let cn_1: f64 = cn_numbers[shell_i.atom_index];
            for shell_j in basis.shells.iter() {
                let atomj: &XtbAtom = &atoms[shell_j.atom_index];
                let cn_2: f64 = cn_numbers[shell_j.atom_index];

                let distance: f64 = (atomi - atomj).norm();
                if distance < PROXIMITY_CUTOFF {
                    // loop over number of angular components
                    for idx_j in (shell_j.sph_start..shell_j.sph_end) {
                        if idx_i == idx_j {
                            // self energies
                            let self_energy_term: f64 = get_self_energy_values_new(
                                atomi.number,
                                atomj.number,
                                cn_1,
                                cn_2,
                                shell_i.shell_index,
                                shell_j.shell_index,
                            );
                            h0[[idx_i, idx_j]] = self_energy_term;
                        } else if idx_i <= idx_j {
                            // xTB applies the element-pair scaling only to
                            // valence-valence shell pairs; pairs involving a
                            // polarization shell use 1.0.
                            let scaling_constant: f64 =
                                if shell_i.polarization || shell_j.polarization {
                                    1.0
                                } else {
                                    calculate_pair_scaling_param(
                                        atomi.number,
                                        atomj.number,
                                        shell_i.angular_momentum,
                                        shell_j.angular_momentum,
                                        shell_i.shell_index,
                                        shell_j.shell_index,
                                    )
                                };
                            // calculate the square of the electronegativity difference
                            let pauling_diff: f64 = (PAULING_EN[atomi.number as usize - 1]
                                - PAULING_EN[atomj.number as usize - 1])
                                .powi(2);
                            // term of H0: 1.0 + k_EN * \Delta EN_AB^2
                            let en_term: f64 =
                                if shell_i.polarization == false && shell_j.polarization == false {
                                    (1.0 + EN_SHELL_PARAM * pauling_diff)
                                } else {
                                    1.0
                                };
                            // get the Hueckel constants term
                            let hueckel_const: f64 = get_hueckel_constants_new(
                                atomi.number,
                                atomj.number,
                                shell_i.angular_momentum,
                                shell_j.angular_momentum,
                                shell_i.polarization,
                                shell_j.polarization,
                            );
                            // self energies
                            let self_energy_term: f64 = get_self_energy_values_new(
                                atomi.number,
                                atomj.number,
                                cn_1,
                                cn_2,
                                shell_i.shell_index,
                                shell_j.shell_index,
                            );
                            // calculate the pi term
                            let pi_term: f64 = get_pi_term(
                                distance,
                                atomi.number as usize,
                                atomj.number as usize,
                                shell_i.angular_momentum,
                                shell_j.angular_momentum,
                            );
                            // combine the parts of the equation
                            let h0_val: f64 = scaling_constant
                                * hueckel_const
                                * self_energy_term
                                * s[[idx_i, idx_j]]
                                * en_term
                                * pi_term;

                            // set h0 values
                            h0[[idx_i, idx_j]] = h0_val;
                            h0[[idx_j, idx_i]] = h0_val;
                        }
                    }
                }
            }
        }
    }
    h0
}

/// Parallel version of H0 calculation using rayon
/// Computes shell pair H0 elements in parallel, then assembles the matrix
pub fn h0_xtb1_parallel(atoms: &[XtbAtom], s: ArrayView2<f64>, basis: &Basis) -> Array2<f64> {
    let n_shells = basis.shells.len();
    let nbas = basis.nbas;

    // Pre-calculate coordination numbers (this is fast, keep sequential)
    let cn_numbers: Array1<f64> = calculate_coordination_numbers(atoms);

    // Generate shell pairs (upper triangle including diagonal)
    let shell_pairs: Vec<(usize, usize)> = (0..n_shells)
        .flat_map(|i| (i..n_shells).map(move |j| (i, j)))
        .collect();

    // Compute H0 elements for each shell pair in parallel
    // Returns vector of (start_i, end_i, start_j, end_j, block, is_diagonal)
    let results: Vec<(usize, usize, usize, usize, Vec<(usize, usize, f64)>)> = shell_pairs
        .par_iter()
        .filter_map(|&(idx_1, idx_2)| {
            let shell_i = &basis.shells[idx_1];
            let shell_j = &basis.shells[idx_2];
            let atomi = &atoms[shell_i.atom_index];
            let atomj = &atoms[shell_j.atom_index];
            let cn_1 = cn_numbers[shell_i.atom_index];
            let cn_2 = cn_numbers[shell_j.atom_index];

            let distance = (atomi - atomj).norm();
            if distance >= PROXIMITY_CUTOFF {
                return None;
            }

            let mut entries: Vec<(usize, usize, f64)> = Vec::new();

            for idx_i in shell_i.sph_start..shell_i.sph_end {
                for idx_j in shell_j.sph_start..shell_j.sph_end {
                    if idx_i == idx_j {
                        // Self energies (diagonal)
                        let self_energy_term = get_self_energy_values_new(
                            atomi.number,
                            atomj.number,
                            cn_1,
                            cn_2,
                            shell_i.shell_index,
                            shell_j.shell_index,
                        );
                        entries.push((idx_i, idx_j, self_energy_term));
                    } else if idx_i < idx_j {
                        // xTB applies the element-pair scaling only to
                        // valence-valence shell pairs; pairs involving a
                        // polarization shell use 1.0.
                        let scaling_constant = if shell_i.polarization || shell_j.polarization {
                            1.0
                        } else {
                            calculate_pair_scaling_param(
                                atomi.number,
                                atomj.number,
                                shell_i.angular_momentum,
                                shell_j.angular_momentum,
                                shell_i.shell_index,
                                shell_j.shell_index,
                            )
                        };
                        let pauling_diff = (PAULING_EN[atomi.number as usize - 1]
                            - PAULING_EN[atomj.number as usize - 1])
                            .powi(2);
                        let en_term = if !shell_i.polarization && !shell_j.polarization {
                            1.0 + EN_SHELL_PARAM * pauling_diff
                        } else {
                            1.0
                        };
                        let hueckel_const = get_hueckel_constants_new(
                            atomi.number,
                            atomj.number,
                            shell_i.angular_momentum,
                            shell_j.angular_momentum,
                            shell_i.polarization,
                            shell_j.polarization,
                        );
                        let self_energy_term = get_self_energy_values_new(
                            atomi.number,
                            atomj.number,
                            cn_1,
                            cn_2,
                            shell_i.shell_index,
                            shell_j.shell_index,
                        );
                        let pi_term = get_pi_term(
                            distance,
                            atomi.number as usize,
                            atomj.number as usize,
                            shell_i.angular_momentum,
                            shell_j.angular_momentum,
                        );
                        let h0_val = scaling_constant
                            * hueckel_const
                            * self_energy_term
                            * s[[idx_i, idx_j]]
                            * en_term
                            * pi_term;
                        entries.push((idx_i, idx_j, h0_val));
                    }
                }
            }

            Some((
                shell_i.sph_start,
                shell_i.sph_end,
                shell_j.sph_start,
                shell_j.sph_end,
                entries,
            ))
        })
        .collect();

    // Assemble the matrix
    let mut h0: Array2<f64> = Array2::zeros((nbas, nbas));
    for (_, _, _, _, entries) in results {
        for (i, j, val) in entries {
            h0[[i, j]] = val;
            if i != j {
                h0[[j, i]] = val;
            }
        }
    }

    h0
}

pub fn calculate_coordination_numbers(atoms: &[XtbAtom]) -> Array1<f64> {
    let mut cn_numbers: Array1<f64> = Array1::zeros(atoms.len());

    for (i, atomi) in atoms.iter().enumerate() {
        for (j, atomj) in atoms.iter().enumerate() {
            if i != j {
                let cov_i: f64 = COV_RADII_CN[atomi.number as usize - 1] / BOHR_TO_ANGS;
                let cov_j: f64 = COV_RADII_CN[atomj.number as usize - 1] / BOHR_TO_ANGS;
                let distance: f64 = ((atomi.xyz.x - atomj.xyz.x).powi(2)
                    + (atomi.xyz.y - atomj.xyz.y).powi(2)
                    + (atomi.xyz.z - atomj.xyz.z).powi(2))
                .sqrt();
                cn_numbers[i] +=
                    1.0 / (1.0 + (-16.0 * (4.0 / 3.0 * (cov_i + cov_j) / distance - 1.0)).exp());
            }
        }
    }
    cn_numbers
}

/// GFN1 shell-pair Hückel scaling constant K_AB for the off-diagonal H0
/// element (s-p pairs use the special value 2.08; H polarization shells
/// use K_DIFF_PARAM).
pub fn get_hueckel_constants(
    funci: &ContractedBasisfunction,
    funcj: &ContractedBasisfunction,
    z_1: u8,
    z_2: u8,
) -> f64 {
    let l_1: usize = funci.angular_momentum;
    let l_2: usize = funcj.angular_momentum;

    let k_val: f64 = if (z_1 == 1 && funci.polarization) || (z_2 == 1 && funcj.polarization) {
        if z_1 == 1 && z_2 == 1 && funci.polarization && funcj.polarization {
            K_DIFF_PARAM
        } else if z_1 == 1 && funci.polarization {
            0.5 * (K_SHELL_PARAMS[l_2] + K_DIFF_PARAM)
        } else {
            0.5 * (K_SHELL_PARAMS[l_1] + K_DIFF_PARAM)
        }
    } else if (l_1 == 0 && l_2 == 1) || (l_1 == 1 && l_2 == 0) {
        2.08
    } else {
        0.5 * (K_SHELL_PARAMS[l_1] + K_SHELL_PARAMS[l_2])
    };

    k_val
}

/// Same as [`get_hueckel_constants`] but taking angular momenta and
/// polarization flags directly instead of the basis functions.
pub fn get_hueckel_constants_new(
    z_1: u8,
    z_2: u8,
    l_1: usize,
    l_2: usize,
    polarization_1: bool,
    polarization_2: bool,
) -> f64 {
    let k_val: f64 = if (z_1 == 1 && polarization_1) || (z_2 == 1 && polarization_2) {
        if z_1 == 1 && z_2 == 1 && polarization_1 && polarization_2 {
            K_DIFF_PARAM
        } else if z_1 == 1 && polarization_1 {
            0.5 * (K_SHELL_PARAMS[l_2] + K_DIFF_PARAM)
        } else {
            0.5 * (K_SHELL_PARAMS[l_1] + K_DIFF_PARAM)
        }
    } else if (l_1 == 0 && l_2 == 1) || (l_1 == 1 && l_2 == 0) {
        2.08
    } else {
        0.5 * (K_SHELL_PARAMS[l_1] + K_SHELL_PARAMS[l_2])
    };

    k_val
}

/// Coordination-number-dependent atomic shell self-energies (the H0
/// diagonal levels), averaged over the orbital pair. This variant uses the
/// additive kCN form `H_ll - kCN*CN`.
pub fn get_self_energy_values_new(
    z_1: u8,
    z_2: u8,
    cn_1: f64,
    cn_2: f64,
    shell_idx_1: usize,
    shell_idx_2: usize,
) -> f64 {
    let z_idx_1: usize = (z_1 - 1) as usize;
    let z_idx_2: usize = (z_2 - 1) as usize;

    let en_1: f64 = (HAMILTONIAN_SELF_ENERGY[z_idx_1][shell_idx_1] / HARTREE_TO_EV)
        - (HAMILTONIAN_KCN_VALUES[z_idx_1][shell_idx_1] * cn_1);
    let en_2: f64 = (HAMILTONIAN_SELF_ENERGY[z_idx_2][shell_idx_2] / HARTREE_TO_EV)
        - (HAMILTONIAN_KCN_VALUES[z_idx_2][shell_idx_2] * cn_2);

    0.5 * (en_1 + en_2)
}

/// Coordination-number-dependent atomic shell self-energies (the H0
/// diagonal levels), averaged over the orbital pair. Uses the
/// multiplicative kCN form `H_ll * (1 + kCN*CN)`.
pub fn get_self_energy_values(
    funci: &ContractedBasisfunction,
    funcj: &ContractedBasisfunction,
    z_1: u8,
    z_2: u8,
    cn_1: f64,
    cn_2: f64,
) -> f64 {
    let l_1: usize = funci.angular_momentum;
    let l_2: usize = funcj.angular_momentum;
    let z_idx_1: usize = (z_1 - 1) as usize;
    let z_idx_2: usize = (z_2 - 1) as usize;
    // idx for self energies
    let en_idx_1: usize = if !funci.polarization { l_1 } else { l_1 + 1 };
    let en_idx_2: usize = if !funcj.polarization { l_2 } else { l_2 + 1 };

    let k_cn_params: [f64; 3] = [0.006, -0.003, -0.005];
    let en_1: f64 = (HAMILTONIAN_SELF_ENERGY[z_idx_1][en_idx_1] / HARTREE_TO_EV)
        * (1.0 + k_cn_params[l_1] * cn_1);
    let en_2: f64 = (HAMILTONIAN_SELF_ENERGY[z_idx_2][en_idx_2] / HARTREE_TO_EV)
        * (1.0 + k_cn_params[l_2] * cn_2);

    0.5 * (en_1 + en_2)
}

/// Distance-dependent shell-polynomial bond correction Π(R) that enters the
/// off-diagonal H0 element, built from the per-shell polynomial coefficients
/// and the covalent-radii sum.
pub fn get_pi_term(r_ab: f64, z_1: usize, z_2: usize, l_1: usize, l_2: usize) -> f64 {
    let z_idx_1: usize = z_1 - 1;
    let z_idx_2: usize = z_2 - 1;

    // get the polynomial values for both atoms
    let k_poly_1: f64 = HAMILTONIAN_SHELL_POLYNOMIALS[z_idx_1][l_1] * 0.01;
    let k_poly_2: f64 = HAMILTONIAN_SHELL_POLYNOMIALS[z_idx_2][l_2] * 0.01;

    // get the covalence radii
    let cov_1: f64 = COV_RADII[z_idx_1] / BOHR_TO_ANGS;
    let cov_2: f64 = COV_RADII[z_idx_2] / BOHR_TO_ANGS;
    let cov_sum: f64 = cov_1 + cov_2;
    let distance_term: f64 = (r_ab / cov_sum).sqrt();

    // calculate pi term
    (1.0 + k_poly_1 * distance_term) * (1.0 + k_poly_2 * distance_term)
}

/// Element-pair-specific scaling factor for the off-diagonal H0 element.
/// Most pairs use 1.0; selected pairs (and d-d shell pairs) are reparameterised.
pub fn calculate_pair_scaling_param(
    z_1: u8,
    z_2: u8,
    l_1: usize,
    l_2: usize,
    shell_idx_1: usize,
    shell_idx_2: usize,
) -> f64 {
    let d_params: [f64; 3] = [1.1, 1.2, 1.2];
    let val: f64 = match (z_1, z_2) {
        (1, 1) => {
            if z_1 == 1 && z_2 == 1 && shell_idx_1 == 0 && shell_idx_2 == 0 {
                0.96
            } else {
                1.0
            }
        }
        (5, 1) => 0.95,
        (1, 5) => 0.95,
        (7, 1) => 1.04,
        (1, 7) => 1.04,
        (28, 1) => 0.9,
        (1, 28) => 0.9,
        (75, 1) => 0.8,
        (1, 75) => 0.8,
        (78, 1) => 0.8,
        (1, 78) => 0.8,
        (15, 5) => 0.96,
        (5, 15) => 0.97,
        (14, 7) => 1.01,
        (7, 14) => 1.01,
        _ => {
            let d1: usize = check_d_elements(z_1);
            let d2: usize = check_d_elements(z_2);
            if d1 > 0 && d2 > 0 && l_1 == 2 && l_2 == 2 {
                0.5 * (d_params[d1 - 1] + d_params[d2 - 1])
            } else {
                1.0
            }
        }
    };

    val
}

fn check_d_elements(z: u8) -> usize {
    if z > 20 && z < 30 {
        1
    } else if z > 38 && z < 48 {
        2
    } else if z > 56 && z < 80 {
        3
    } else {
        0
    }
}