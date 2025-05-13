use crate::io::settings::DispersionConfig;
use crate::xtb::initialization::{atom::XtbAtom, basis::Basis, system::XtbSystem};
use crate::xtb::parameters::{REP_ALPHA_PARAMS, REP_Z_EFF_PARAMS};
use nalgebra::Point3;
use nalgebra::Vector3;
use ndarray::prelude::*;
use ndarray::{s, Array1, Array2, ArrayView1};
use rusty_dftd_lib::model::*;
use rusty_dftd_lib::*;

pub fn gradient_disp3_xtb(atoms: &[XtbAtom], config: &DispersionConfig) -> Array1<f64> {
    let positions: Vec<Point3<f64>> = atoms
        .iter()
        .map(|atom| Point3::from(atom.xyz))
        .collect::<Vec<Point3<f64>>>();
    let atomic_numbers: Vec<u8> = atoms.iter().map(|atom| atom.number).collect::<Vec<u8>>();
    let pos_an = (&positions, &atomic_numbers);
    let mut disp_mol: Molecule = model::Molecule::from(pos_an);
    let disp: D3Model = D3Model::from_molecule(&disp_mol, None);
    let cutoff: RealspaceCutoff = RealspaceCutoffBuilder::new()
        .set_cn(CN_CUTOFF_D3_DEFAULT)
        .build();

    let d3param: D3Param = D3ParamBuilder::new()
        .set_s6(config.s6)
        .set_s8(config.s8)
        .set_s9(1.0)
        .set_a1(config.a1)
        .set_a2(config.a2)
        .build();
    let param: RationalDamping3Param = RationalDamping3Param::from((d3param, &disp_mol.num));
    let disp_result = get_dispersion(&mut disp_mol, &disp, &param, &cutoff, false, true);
    Array1::from_iter(disp_result.gradient.unwrap().into_iter())
}

// pub fn coul_third_order_grad_contribution_xtb(
//     basis: &Basis,
//     atoms: &[XtbAtom],
//     dq: ArrayView1<f64>,
//     gamma_third: ArrayView1<f64>,
// ) -> Array2<f64> {
//     // get the number of basis functions
//     let nbas: usize = basis.basis_functions.len();
//     // initialize array
//     let mut arr: Array2<f64> = Array2::zeros([nbas, nbas]);
//     // calculate dq**2 * gamma
//     let epot: Array1<f64> = dq.map(|val| val.powi(2)) * &gamma_third;
//
//     // loop over basis
//     for (i, funci) in basis.basis_functions.iter().enumerate() {
//         // get the atom index
//         let at_i: usize = funci.atom_index;
//
//         for (j, funcj) in basis.basis_functions.iter().enumerate() {
//             let at_j: usize = funcj.atom_index;
//
//             // calculate gradient contribution
//             arr[[i, j]] = epot[at_i] + epot[at_j];
//         }
//     }
//
//     arr
// }

pub fn coul_third_order_grad_contribution_xtb(
    basis: &Basis,
    dq: ArrayView1<f64>,
    gamma_third: ArrayView1<f64>,
) -> Array2<f64> {
    // get the number of basis functions
    let nbas: usize = basis.nbas;
    // initialize array
    let mut arr: Array2<f64> = Array2::zeros([nbas, nbas]);
    // calculate dq**2 * gamma
    let epot: Array1<f64> = dq.map(|val| val.powi(2)) * &gamma_third;

    // loop over the shells
    for shell_i in basis.shells.iter() {
        // get the atom index
        let at_i: usize = shell_i.atom_index;
        // iteratve over the shell indices
        for idx_i in (shell_i.sph_start..shell_i.sph_end) {
            // iterate over the next shells
            for shell_j in basis.shells.iter() {
                let at_j: usize = shell_j.atom_index;
                for idx_j in (shell_j.sph_start..shell_j.sph_end) {
                    // calculate gradient contribution
                    arr[[idx_i, idx_j]] = epot[at_i] + epot[at_j];
                }
            }
        }
    }

    arr
}

impl XtbSystem {
    pub fn grad_repulsive_energy(&self) -> Array1<f64> {
        let mut grad: Array1<f64> = Array1::zeros([3 * self.n_atoms]);

        // two loops over the atoms
        for (i, atomi) in self.atoms.iter().enumerate() {
            // get the z_eff and alpha values
            let z_eff_i: f64 = REP_Z_EFF_PARAMS[atomi.kind.number_usize() - 1];
            let alpha_i: f64 = REP_ALPHA_PARAMS[atomi.kind.number_usize() - 1];
            let mut grad_i: Array1<f64> = Array1::zeros([3]);

            for (j, atomj) in self.atoms.iter().enumerate() {
                let z_eff_j: f64 = REP_Z_EFF_PARAMS[atomj.kind.number_usize() - 1];
                let alpha_j: f64 = REP_ALPHA_PARAMS[atomj.kind.number_usize() - 1];

                if i != j {
                    let mut r: Vector3<f64> = atomi - atomj;
                    let diff_vec = atomi - atomj;
                    let distance: f64 =
                        (diff_vec.x.powi(2) + diff_vec.y.powi(2) + diff_vec.z.powi(2)).sqrt();
                    r /= distance;

                    let exponential: f64 = (-(alpha_i * alpha_j).sqrt() * distance.powf(1.5)).exp();
                    let part1: f64 = exponential * z_eff_i * z_eff_j / distance.powi(2);
                    let part2: f64 =
                        3.0 * (alpha_i * alpha_j).sqrt() * z_eff_i * z_eff_j * exponential
                            / (2.0 * distance.sqrt());

                    let deriv_val: f64 = -part1 - part2;
                    r *= deriv_val;

                    let v = Array1::from_iter(r.iter());
                    grad_i = &grad_i + &v;
                }
            }
            grad.slice_mut(s![i * 3..i * 3 + 3]).assign(&grad_i);
        }

        grad
    }
}
