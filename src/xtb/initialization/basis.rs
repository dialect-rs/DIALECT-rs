use crate::xtb::initialization::atom::XtbAtom;
use crate::xtb::integrals::obara_saika_helper;
use crate::xtb::parameters::{
    ANG_SHELL, BASIS_COEFFS_3_GTOS, BASIS_COEFFS_4_GTOS, BASIS_COEFFS_6_GTOS,
    BASIS_EXPONENTS_3_GTOS, BASIS_EXPONENTS_4_GTOS, BASIS_EXPONENTS_6_GTOS, BASIS_FACTORIALS,
    BASIS_NUMBER_PRIMITIVES, BASIS_SLATER_EXPONENT, N_SHELL, QUANTUM_NUMBER, VALENCE_SHELL,
};
use itertools::Itertools;
use ndarray::prelude::*;
use std::f64::consts::PI;

// structs for the definition of cartesian bases
#[derive(Clone)]
pub struct Basis {
    pub basis_functions: Vec<ContractedBasisfunction>,
    pub shells: Vec<BasisShell>,
    pub nbas: usize,
}

#[derive(Clone)]
pub struct PrimitiveBasisfunction {
    pub angular_momenta: (i8, i8, i8),
    pub exponent: f64,
    pub coeff: f64,
}

#[derive(Clone)]
pub struct ContractedBasisfunction {
    pub primitive_functions: Vec<PrimitiveBasisfunction>,
    pub center: (f64, f64, f64),
    pub atom_index: usize,
    pub angular_momentum: usize,
    pub contracted_norm: f64,
    pub polarization: bool,
    pub n: usize,
}

#[derive(Clone)]
pub struct BasisShell {
    pub angular_momentum: usize,
    pub atom_index: usize,
    pub start: usize,
    pub end: usize,
    pub sph_start: usize,
    pub sph_end: usize,
    pub polarization: bool,
    pub shell_index: usize,
}

pub fn create_basis_set(atoms: &[XtbAtom]) -> Basis {
    let mut basis = Basis {
        basis_functions: Vec::new(),
        shells: Vec::new(),
        nbas: 0,
    };

    let mut basis_count: usize = 0;
    let mut prev_count: usize = 0;
    let mut sph_count: usize = 0;
    let mut prev_sph: usize = 0;
    for (atom_idx, atom) in atoms.iter().enumerate() {
        // get the index from the atomic number
        let idx: usize = atom.number as usize - 1;
        // get the slater exponent
        let slater: [f64; 3] = BASIS_SLATER_EXPONENT[idx];
        // get the number of primitives
        let n_prim: [usize; 3] = BASIS_NUMBER_PRIMITIVES[idx];
        // get the number of shells
        let nshell: usize = N_SHELL[idx];
        // get the angular momenta
        let ang_momenta: [usize; 3] = ANG_SHELL[idx];
        // get the quantum numbers
        let quantum: [usize; 3] = QUANTUM_NUMBER[idx];
        // valence occupation
        let val_occ: [usize; 3] = VALENCE_SHELL[idx];

        // vector for basis functions
        let mut basis_functions: Vec<ContractedBasisfunction> = Vec::new();

        // store the previous angular momentum
        let mut prev_l: usize = 0;
        for (iter_idx, slater_val) in slater.iter().enumerate() {
            if *slater_val > 0.0 {
                // get the index for the exponents and coefficients
                let n: usize = quantum[iter_idx];
                let l: usize = ang_momenta[iter_idx];
                let idx_exp_coeff: usize = convert_quantum_ang_to_index(n, l);

                // number of primitives
                let nprim: usize = n_prim[iter_idx];
                let exp: Array1<f64>;
                let coeffs: Array1<f64>;

                if nprim == 3 {
                    let tmp_exp: Vec<f64> = BASIS_EXPONENTS_3_GTOS[idx_exp_coeff]
                        .iter()
                        .map(|val| *val)
                        .collect();
                    let tmp_coeff: Vec<f64> = BASIS_COEFFS_3_GTOS[idx_exp_coeff]
                        .iter()
                        .map(|val| *val)
                        .collect();
                    exp = Array::from(tmp_exp) * slater_val.powi(2);
                    coeffs = Array::from(tmp_coeff);
                } else if nprim == 4 {
                    let tmp_exp: Vec<f64> = BASIS_EXPONENTS_4_GTOS[idx_exp_coeff]
                        .iter()
                        .map(|val| *val)
                        .collect();
                    let tmp_coeff: Vec<f64> = BASIS_COEFFS_4_GTOS[idx_exp_coeff]
                        .iter()
                        .map(|val| *val)
                        .collect();
                    exp = Array::from(tmp_exp) * slater_val.powi(2);
                    coeffs = Array::from(tmp_coeff);
                } else {
                    let tmp_exp: Vec<f64> = BASIS_EXPONENTS_6_GTOS[idx_exp_coeff]
                        .iter()
                        .map(|val| *val)
                        .collect();
                    let tmp_coeff: Vec<f64> = BASIS_COEFFS_6_GTOS[idx_exp_coeff]
                        .iter()
                        .map(|val| *val)
                        .collect();
                    exp = Array::from(tmp_exp) * slater_val.powi(2);
                    coeffs = Array::from(tmp_coeff);
                }

                // for l_val in permuts(l as u32).iter().unique() {
                for l_val in permuts_2(l as u32).iter() {
                    let mut list_prim_func: Vec<PrimitiveBasisfunction> = Vec::new();
                    for (exp_val, coeff_val) in exp.iter().zip(coeffs.iter()) {
                        list_prim_func.push(PrimitiveBasisfunction {
                            angular_momenta: (l_val[0] as i8, l_val[1] as i8, l_val[2] as i8),
                            exponent: *exp_val,
                            coeff: *coeff_val,
                        });
                    }

                    let mut func = ContractedBasisfunction {
                        primitive_functions: list_prim_func,
                        center: (atom.xyz[0], atom.xyz[1], atom.xyz[2]),
                        atom_index: atom_idx,
                        angular_momentum: l,
                        contracted_norm: 1.0,
                        polarization: false,
                        n,
                    };
                    normalize_basis_function(&mut func);
                    if iter_idx > 0 && prev_l == l {
                        func.polarization = true;
                    }
                    basis_functions.push(func);

                    // increase the shell counter
                    basis_count += 1;
                }
                // check if the new primitive function have the same angular momentum
                // as the previous primitive functions
                // if yes, orthogonalize the functions
                if iter_idx > 0 && prev_l == l {
                    // get the overlap between the old and new primitives
                    let ovlp: f64 =
                        ovlp_basis_function(0, &basis_functions[0], &basis_functions[1]);
                    let basis_0 = basis_functions[0].clone();

                    for func in basis_0.primitive_functions.iter() {
                        let mut mut_func: PrimitiveBasisfunction = func.clone();
                        mut_func.coeff = -1.0 * func.coeff * ovlp;

                        basis_functions[1].primitive_functions.push(mut_func);
                    }
                    normalize_contracted_basis_function2(&mut basis_functions[1]);
                }
                // set spherical count
                sph_count += 2 * l + 1;
                // polarization
                let polarization: bool = if iter_idx > 0 && prev_l == l {
                    true
                } else {
                    false
                };
                //create the shell
                let shell: BasisShell = BasisShell {
                    angular_momentum: l,
                    atom_index: atom_idx,
                    start: prev_count,
                    end: basis_count,
                    sph_start: prev_sph,
                    sph_end: sph_count,
                    polarization,
                    shell_index: iter_idx,
                };
                basis.shells.push(shell);
                prev_count = basis_count;
                prev_sph = sph_count;
                prev_l = l;
            }
        }
        basis.basis_functions.append(&mut basis_functions);
    }

    // get the number of basis functions
    let mut nbas: usize = 0;
    for shell in basis.shells.iter() {
        nbas += 2 * shell.angular_momentum + 1;
    }
    basis.nbas = nbas;
    basis
}

pub fn basis_helper(atom: &XtbAtom, atom_idx: usize) -> Vec<ContractedBasisfunction> {
    // get the index from the atomic number
    let idx: usize = atom.number as usize - 1;
    // get the slater exponent
    let slater: [f64; 3] = BASIS_SLATER_EXPONENT[idx];
    // get the number of primitives
    let n_prim: [usize; 3] = BASIS_NUMBER_PRIMITIVES[idx];
    // get the number of shells
    let nshell: usize = N_SHELL[idx];
    // get the angular momenta
    let ang_momenta: [usize; 3] = ANG_SHELL[idx];
    // get the quantum numbers
    let quantum: [usize; 3] = QUANTUM_NUMBER[idx];
    // valence occupation
    let val_occ: [usize; 3] = VALENCE_SHELL[idx];

    // vector for basis functions
    let mut basis_functions: Vec<ContractedBasisfunction> = Vec::new();

    // store the previous angular momentum
    let mut prev_l: usize = 0;
    for (iter_idx, slater_val) in slater.iter().enumerate() {
        if *slater_val > 0.0 {
            // get the index for the exponents and coefficients
            let n: usize = quantum[iter_idx];
            let l: usize = ang_momenta[iter_idx];
            let idx_exp_coeff: usize = convert_quantum_ang_to_index(n, l);

            // number of primitives
            let nprim: usize = n_prim[iter_idx];
            let exp: Array1<f64>;
            let coeffs: Array1<f64>;

            if nprim == 3 {
                let tmp_exp: Vec<f64> = BASIS_EXPONENTS_3_GTOS[idx_exp_coeff]
                    .iter()
                    .map(|val| *val)
                    .collect();
                let tmp_coeff: Vec<f64> = BASIS_COEFFS_3_GTOS[idx_exp_coeff]
                    .iter()
                    .map(|val| *val)
                    .collect();
                exp = Array::from(tmp_exp) * slater_val.powi(2);
                coeffs = Array::from(tmp_coeff);
            } else if nprim == 4 {
                let tmp_exp: Vec<f64> = BASIS_EXPONENTS_4_GTOS[idx_exp_coeff]
                    .iter()
                    .map(|val| *val)
                    .collect();
                let tmp_coeff: Vec<f64> = BASIS_COEFFS_4_GTOS[idx_exp_coeff]
                    .iter()
                    .map(|val| *val)
                    .collect();
                exp = Array::from(tmp_exp) * slater_val.powi(2);
                coeffs = Array::from(tmp_coeff);
            } else {
                let tmp_exp: Vec<f64> = BASIS_EXPONENTS_6_GTOS[idx_exp_coeff]
                    .iter()
                    .map(|val| *val)
                    .collect();
                let tmp_coeff: Vec<f64> = BASIS_COEFFS_6_GTOS[idx_exp_coeff]
                    .iter()
                    .map(|val| *val)
                    .collect();
                exp = Array::from(tmp_exp) * slater_val.powi(2);
                coeffs = Array::from(tmp_coeff);
            }

            for l_val in permuts(l as u32).iter().unique() {
                let mut list_prim_func: Vec<PrimitiveBasisfunction> = Vec::new();
                for (exp_val, coeff_val) in exp.iter().zip(coeffs.iter()) {
                    list_prim_func.push(PrimitiveBasisfunction {
                        angular_momenta: (l_val[0] as i8, l_val[1] as i8, l_val[2] as i8),
                        exponent: *exp_val,
                        coeff: *coeff_val,
                    });
                }

                let mut func = ContractedBasisfunction {
                    primitive_functions: list_prim_func,
                    center: (atom.xyz[0], atom.xyz[1], atom.xyz[2]),
                    atom_index: atom_idx,
                    angular_momentum: l,
                    contracted_norm: 1.0,
                    polarization: false,
                    n,
                };
                normalize_basis_function(&mut func);
                if iter_idx > 0 && prev_l == l {
                    func.polarization = true;
                }
                basis_functions.push(func);
            }
            // check if the new primitive function have the same angular momentum
            // as the previous primitive functions
            // if yes, orthogonalize the functions
            if iter_idx > 0 && prev_l == l {
                // get the overlap between the old and new primitives
                let ovlp: f64 = ovlp_basis_function(0, &basis_functions[0], &basis_functions[1]);
                let basis_0 = basis_functions[0].clone();

                for func in basis_0.primitive_functions.iter() {
                    let mut mut_func: PrimitiveBasisfunction = func.clone();
                    mut_func.coeff = -1.0 * func.coeff * ovlp;

                    basis_functions[1].primitive_functions.push(mut_func);
                }
                normalize_contracted_basis_function2(&mut basis_functions[1]);
            }
            prev_l = l;
        }
    }
    basis_functions
}

pub fn basis_helper_2(atom: &XtbAtom, atom_idx: usize) -> Vec<ContractedBasisfunction> {
    // get the index from the atomic number
    let idx: usize = atom.number as usize - 1;
    // get the slater exponent
    let slater: [f64; 3] = BASIS_SLATER_EXPONENT[idx];
    // get the number of primitives
    let n_prim: [usize; 3] = BASIS_NUMBER_PRIMITIVES[idx];
    // get the number of shells
    let nshell: usize = N_SHELL[idx];
    // get the angular momenta
    let ang_momenta: [usize; 3] = ANG_SHELL[idx];
    // get the quantum numbers
    let quantum: [usize; 3] = QUANTUM_NUMBER[idx];

    // vector for basis functions
    let mut basis_functions: Vec<ContractedBasisfunction> = Vec::new();

    // store the previous angular momentum
    let mut prev_l: usize = 0;
    for (iter_idx, slater_val) in slater.iter().enumerate() {
        if *slater_val > 0.0 {
            // get the index for the exponents and coefficients
            let n: usize = quantum[iter_idx];
            let l: usize = ang_momenta[iter_idx];
            let idx_exp_coeff: usize = convert_quantum_ang_to_index(n, l);

            if (iter_idx > 0 && prev_l != l) || iter_idx == 0 {
                // number of primitives
                let nprim: usize = n_prim[iter_idx];
                let exp: Array1<f64>;
                let coeffs: Array1<f64>;

                if nprim == 3 {
                    let tmp_exp: Vec<f64> = BASIS_EXPONENTS_3_GTOS[idx_exp_coeff]
                        .iter()
                        .map(|val| *val)
                        .collect();
                    let tmp_coeff: Vec<f64> = BASIS_COEFFS_3_GTOS[idx_exp_coeff]
                        .iter()
                        .map(|val| *val)
                        .collect();
                    exp = Array::from(tmp_exp) * slater_val.powi(2);
                    coeffs = Array::from(tmp_coeff);
                } else if nprim == 4 {
                    let tmp_exp: Vec<f64> = BASIS_EXPONENTS_4_GTOS[idx_exp_coeff]
                        .iter()
                        .map(|val| *val)
                        .collect();
                    let tmp_coeff: Vec<f64> = BASIS_COEFFS_4_GTOS[idx_exp_coeff]
                        .iter()
                        .map(|val| *val)
                        .collect();
                    exp = Array::from(tmp_exp) * slater_val.powi(2);
                    coeffs = Array::from(tmp_coeff);
                } else {
                    let tmp_exp: Vec<f64> = BASIS_EXPONENTS_6_GTOS[idx_exp_coeff]
                        .iter()
                        .map(|val| *val)
                        .collect();
                    let tmp_coeff: Vec<f64> = BASIS_COEFFS_6_GTOS[idx_exp_coeff]
                        .iter()
                        .map(|val| *val)
                        .collect();
                    exp = Array::from(tmp_exp) * slater_val.powi(2);
                    coeffs = Array::from(tmp_coeff);
                }

                for l_val in permuts_2(l as u32).iter() {
                    let mut list_prim_func: Vec<PrimitiveBasisfunction> = Vec::new();
                    for (exp_val, coeff_val) in exp.iter().zip(coeffs.iter()) {
                        list_prim_func.push(PrimitiveBasisfunction {
                            angular_momenta: (l_val[0] as i8, l_val[1] as i8, l_val[2] as i8),
                            exponent: *exp_val,
                            coeff: *coeff_val,
                        });
                    }

                    let mut func = ContractedBasisfunction {
                        primitive_functions: list_prim_func,
                        center: (atom.xyz[0], atom.xyz[1], atom.xyz[2]),
                        atom_index: atom_idx,
                        angular_momentum: l,
                        contracted_norm: 1.0,
                        polarization: false,
                        n,
                    };
                    normalize_basis_function(&mut func);
                    if iter_idx > 0 && prev_l == l {
                        func.polarization = true;
                    }
                    basis_functions.push(func);
                }
                // check if the new primitive function have the same angular momentum
                // as the previous primitive functions
                // if yes, orthogonalize the functions
                if iter_idx > 0 && prev_l == l {
                    // get the overlap between the old and new primitives
                    let ovlp: f64 =
                        ovlp_basis_function(0, &basis_functions[0], &basis_functions[1]);
                    let basis_0 = basis_functions[0].clone();

                    for func in basis_0.primitive_functions.iter() {
                        let mut mut_func: PrimitiveBasisfunction = func.clone();
                        mut_func.coeff = -1.0 * func.coeff * ovlp;

                        basis_functions[1].primitive_functions.push(mut_func);
                    }
                    normalize_contracted_basis_function2(&mut basis_functions[1]);
                }
                prev_l = l;
            }
        }
    }
    basis_functions
}

fn convert_quantum_ang_to_index(n: usize, l: usize) -> usize {
    let index: usize = match (n, l) {
        (1, 0) => 0,
        (2, 0) => 1,
        (3, 0) => 2,
        (4, 0) => 3,
        (5, 0) => 4,
        (2, 1) => 5,
        (3, 1) => 6,
        (4, 1) => 7,
        (5, 1) => 8,
        (3, 2) => 9,
        (4, 2) => 10,
        (5, 2) => 11,
        (6, 0) => 12,
        (6, 1) => 13,
        _ => {
            panic!(
                "The combination of {} and {} as quantum numbers is not supported!",
                n, l
            )
        }
    };
    index
}

// pub fn create_basis_set(atoms: &[XtbAtom]) -> Basis {
//     let mut basis = Basis {
//         basis_functions: Vec::new(),
//     };
//
//     for (i, atom) in atoms.iter().enumerate() {
//         let mut functions: Vec<ContractedBasisfunction> = basis_helper(atom, i);
//         basis.basis_functions.append(&mut functions);
//     }
//
//     // for func in basis.basis_functions.iter_mut() {
//     //     normalize_basis_function(func);
//     //     normalize_contracted_basis_function(func);
//     // }
//
//     basis
// }

pub fn create_basis_set_cubes(atoms: &[XtbAtom]) -> Basis {
    let mut basis = Basis {
        basis_functions: Vec::new(),
        shells: Vec::new(),
        nbas: 0,
    };

    for (i, atom) in atoms.iter().enumerate() {
        let mut functions: Vec<ContractedBasisfunction> = basis_helper_2(atom, i);
        basis.basis_functions.append(&mut functions);
    }
    basis
}

/// Calculates the Norm of every single primtive basis function and scales every single
/// coefficient of primitive functions
pub fn normalize_basis_function(func: &mut ContractedBasisfunction) {
    for prim_func in func.primitive_functions.iter_mut() {
        let alpha = prim_func.exponent;
        let p: f64 = 2.0 * alpha;
        let u: f64 = alpha * alpha / p;

        let a: f64 = obara_saika_helper(
            prim_func.angular_momenta.0,
            prim_func.angular_momenta.0,
            alpha,
            alpha,
            func.center.0,
            func.center.0,
            p,
        );
        let b: f64 = obara_saika_helper(
            prim_func.angular_momenta.1,
            prim_func.angular_momenta.1,
            alpha,
            alpha,
            func.center.1,
            func.center.1,
            p,
        );
        let c: f64 = obara_saika_helper(
            prim_func.angular_momenta.2,
            prim_func.angular_momenta.2,
            alpha,
            alpha,
            func.center.2,
            func.center.2,
            p,
        );

        let norm: f64 = 1.0 / (a * b * c * (PI / p).powf(1.5)).sqrt();

        prim_func.coeff = prim_func.coeff * norm;
    }
}

pub fn ovlp_basis_function(
    l: usize,
    func: &ContractedBasisfunction,
    func2: &ContractedBasisfunction,
) -> f64 {
    let mut ovlp: f64 = 0.0;
    for prim_func in func.primitive_functions.iter() {
        for prim_func2 in func2.primitive_functions.iter() {
            let alpha = prim_func.exponent;
            let beta = prim_func2.exponent;
            let p: f64 = alpha + beta;

            let a: f64 = obara_saika_helper(
                prim_func.angular_momenta.0,
                prim_func2.angular_momenta.0,
                alpha,
                beta,
                func.center.0,
                func2.center.0,
                p,
            );
            let b: f64 = obara_saika_helper(
                prim_func.angular_momenta.1,
                prim_func2.angular_momenta.1,
                alpha,
                beta,
                func.center.1,
                func2.center.1,
                p,
            );
            let c: f64 = obara_saika_helper(
                prim_func.angular_momenta.2,
                prim_func2.angular_momenta.2,
                alpha,
                beta,
                func.center.2,
                func2.center.2,
                p,
            );

            ovlp += prim_func.coeff * prim_func2.coeff * a * b * c * (PI / p).powf(1.5);
        }
    }
    ovlp
}

pub fn normalize_contracted_basis_function(func: &mut ContractedBasisfunction) {
    let mut overlap: f64 = 0.0;

    for prim_func_a in func.primitive_functions.iter() {
        for prim_func_b in func.primitive_functions.iter() {
            let alpha = prim_func_a.exponent;
            let beta = prim_func_b.exponent;
            let p: f64 = alpha + beta;

            let a: f64 = obara_saika_helper(
                prim_func_a.angular_momenta.0,
                prim_func_b.angular_momenta.0,
                alpha,
                beta,
                func.center.0,
                func.center.0,
                p,
            );
            let b: f64 = obara_saika_helper(
                prim_func_a.angular_momenta.1,
                prim_func_b.angular_momenta.1,
                alpha,
                beta,
                func.center.1,
                func.center.1,
                p,
            );
            let c: f64 = obara_saika_helper(
                prim_func_a.angular_momenta.2,
                prim_func_b.angular_momenta.2,
                alpha,
                beta,
                func.center.2,
                func.center.2,
                p,
            );

            overlap += prim_func_a.coeff * prim_func_b.coeff * a * b * c * (PI / p).powf(1.5);
        }
    }
    let norm: f64 = 1.0 / overlap;
    func.contracted_norm = norm.sqrt();
}

pub fn normalize_contracted_basis_function2(func: &mut ContractedBasisfunction) {
    let mut overlap: f64 = 0.0;

    for prim_func_a in func.primitive_functions.iter() {
        for prim_func_b in func.primitive_functions.iter() {
            let alpha = prim_func_a.exponent;
            let beta = prim_func_b.exponent;
            let p: f64 = alpha + beta;

            overlap += prim_func_a.coeff * prim_func_b.coeff * (PI / p).powf(1.5);
        }
    }
    for prim_func_a in func.primitive_functions.iter_mut() {
        prim_func_a.coeff = prim_func_a.coeff / overlap.sqrt();
    }
}

// Calculates all permutations of l identical balls on 3 not identical urns. Returns a vector with
/// all possible permutations. Needed for the initialization of the cartesian basis functions. Urns
/// correspond to the three cartesian directions and the number of balls correspond the the total
/// angular momentum of the basis function.
pub fn permuts(l: u32) -> Vec<Array1<u8>> {
    let mut permut: Vec<Array1<u8>> = vec![Array1::zeros(3)];

    fn add(tmp: Vec<Array1<u8>>) -> Vec<Array1<u8>> {
        let mut result: Vec<Array1<u8>> = Vec::new();
        for (j, entry) in tmp.iter().enumerate() {
            for i in 0..3 {
                result.push(entry.to_owned());
                result[j * 3 + i][i] += 1;
            }
        }
        result
    }

    for _ball in 0..l {
        permut = add(permut);
    }
    permut
}

fn permuts_2(l: u32) -> Vec<Array1<u8>> {
    match l {
        0 => vec![array![0, 0, 0]],
        1 => vec![array![0, 1, 0], array![0, 0, 1], array![1, 0, 0]],
        2 => vec![
            array![2, 0, 0],
            array![1, 1, 0],
            array![1, 0, 1],
            array![0, 2, 0],
            array![0, 1, 1],
            array![0, 0, 2],
        ],
        _ => vec![array![0, 0, 0]],
    }
}
