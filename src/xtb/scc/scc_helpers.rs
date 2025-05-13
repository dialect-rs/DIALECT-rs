use crate::constants::K_BOLTZMANN;
use crate::io::settings::DispersionConfig;
use crate::xtb::initialization::basis::Basis;
use crate::xtb::parameters::{REFERENCE_OCCUPATION, REP_ALPHA_PARAMS, REP_Z_EFF_PARAMS};
use crate::xtb::scc::hamiltonian::h0_xtb1_new;
use crate::xtb::{
    initialization::atom::XtbAtom, initialization::system::XtbSystem,
    integrals::calc_overlap_matrix_obs, integrals::calc_overlap_matrix_obs_new,
};
use nalgebra::Point3;
use ndarray::prelude::*;
use rusty_dftd_lib::*;

impl XtbSystem {
    pub fn get_overlap(&mut self) {
        // calculate the overlap matrix
        // let s: Array2<f64> = calc_overlap_matrix_obs(&self.basis);
        let s: Array2<f64> = calc_overlap_matrix_obs_new(&self.basis);
        self.properties.set_s(s);
    }

    pub fn get_h0(&mut self) {
        // let h0: Array2<f64> = h0_xtb1(
        //     self.n_orbs,
        //     &self.atoms,
        //     self.properties.s().unwrap(),
        //     &self.basis,
        // );
        let h0: Array2<f64> = h0_xtb1_new(&self.atoms, self.properties.s().unwrap(), &self.basis);
        self.properties.set_h0(h0);
    }

    pub fn calculate_repulsive_energy(&self) -> f64 {
        let mut erep: f64 = 0.0;

        // two loops over the atoms
        for (i, atomi) in self.atoms.iter().enumerate() {
            // get the z_eff and alpha values
            let z_eff_i: f64 = REP_Z_EFF_PARAMS[atomi.kind.number_usize() - 1];
            let alpha_i: f64 = REP_ALPHA_PARAMS[atomi.kind.number_usize() - 1];

            for (j, atomj) in self.atoms.iter().enumerate() {
                let z_eff_j: f64 = REP_Z_EFF_PARAMS[atomj.kind.number_usize() - 1];
                let alpha_j: f64 = REP_ALPHA_PARAMS[atomj.kind.number_usize() - 1];

                if i < j {
                    // get the distance between the atoms
                    let diff_vec = atomi - atomj;
                    let distance: f64 =
                        (diff_vec.x.powi(2) + diff_vec.y.powi(2) + diff_vec.z.powi(2)).sqrt();
                    let energy_val: f64 = (-(alpha_i * alpha_j).sqrt() * distance.powf(1.5)).exp()
                        * z_eff_i
                        * z_eff_j
                        / distance;
                    erep += energy_val;
                }
            }
        }
        erep
    }
}

// pub fn coul_third_order_hamiltonian(
//     hubbards_derivs: ArrayView1<f64>,
//     dq: ArrayView1<f64>,
//     basis: &Basis,
// ) -> Array2<f64> {
//     // multiplication of dq**2 with Hubbards derivs
//     let arr: Array1<f64> = &dq * &dq * &hubbards_derivs;
//     // get the length of the basis
//     let n_orbs: usize = basis.basis_functions.len();
//     // loop over basis functions
//     let mut esp_ao_row: Array1<f64> = Array1::zeros(n_orbs);
//     for (i, funci) in basis.basis_functions.iter().enumerate() {
//         esp_ao_row[i] = arr[funci.atom_index];
//     }
//
//     let esp_ao_column: Array2<f64> = esp_ao_row.clone().insert_axis(Axis(1));
//     let esp_ao: Array2<f64> = &esp_ao_column.broadcast((n_orbs, n_orbs)).unwrap() + &esp_ao_row;
//     esp_ao
// }

pub fn coul_third_order_hamiltonian(
    hubbards_derivs: ArrayView1<f64>,
    dq: ArrayView1<f64>,
    basis: &Basis,
) -> Array2<f64> {
    // multiplication of dq**2 with Hubbards derivs
    let arr: Array1<f64> = &dq * &dq * &hubbards_derivs;
    // get the length of the basis
    let n_orbs: usize = basis.nbas;
    // loop over basis functions
    let mut esp_ao_row: Array1<f64> = Array1::zeros(n_orbs);
    for shell in basis.shells.iter() {
        for i in (shell.sph_start..shell.sph_end) {
            esp_ao_row[i] = arr[shell.atom_index];
        }
    }

    let esp_ao_column: Array2<f64> = esp_ao_row.clone().insert_axis(Axis(1));
    let esp_ao: Array2<f64> = &esp_ao_column.broadcast((n_orbs, n_orbs)).unwrap() + &esp_ao_row;
    esp_ao
}

pub fn get_electronic_energy_xtb(
    p: ArrayView2<f64>,
    h0: ArrayView2<f64>,
    dq: ArrayView1<f64>,
    dq_ao: ArrayView1<f64>,
    gamma: ArrayView2<f64>,
    hubbard_derivs: ArrayView1<f64>,
) -> f64 {
    // band structure energy
    let e_band_structure: f64 = (&p * &h0).sum();
    // Coulomb energy from monopoles
    let e_coulomb: f64 = 0.5 * &dq_ao.dot(&gamma.dot(&dq_ao));
    // Coulomb third order energy
    let e_coul_third: f64 = 1.0 / 3.0 * dq.map(|val| val.powi(3)).dot(&hubbard_derivs);
    let e_elec: f64 = e_band_structure + e_coulomb + e_coul_third;

    e_elec
}

pub fn get_entropy_energy_contribution(occupation: &[f64], t: f64) -> f64 {
    let mut energy: f64 = 0.0;
    let occ_half: Array1<f64> = 0.5 * &Array::from(occupation.to_vec());

    for (idx, val) in occ_half.iter().enumerate() {
        if *val < 1.0 && *val > 0.0 {
            energy += val * (val).ln() + (1.0 - val) * (1.0 - val).ln();
        }
        if *val == 1.0 {
            energy += val * (val).ln();
        }
        if *val == 0.0 {
            energy += (1.0 - val) * (1.0 - val).ln();
        }
    }
    energy *= 2.0 * t * K_BOLTZMANN;
    energy
}

pub fn get_dispersion_energy_xtb(atoms: &[XtbAtom], config: &DispersionConfig) -> f64 {
    let positions: Vec<Point3<f64>> = atoms
        .iter()
        .map(|atom| Point3::from(atom.xyz))
        .collect::<Vec<Point3<f64>>>();
    let atomic_numbers: Vec<u8> = atoms.iter().map(|atom| atom.number).collect::<Vec<u8>>();
    let pos_an = (&positions, &atomic_numbers);
    let mut disp_mol = model::Molecule::from(pos_an);
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
    let disp_result = get_dispersion(&mut disp_mol, &disp, &param, &cutoff, false, false);
    disp_result.energy
}

pub fn create_density_ref(basis: &Basis, atoms: &[XtbAtom]) -> Array2<f64> {
    // initialize empty density matrix
    let nbas: usize = basis.nbas;
    let mut p: Array2<f64> = Array2::zeros([nbas, nbas]);

    // iterate over basis shells
    for shell_i in basis.shells.iter() {
        let atom: &XtbAtom = &atoms[shell_i.atom_index];
        let l: usize = shell_i.angular_momentum;
        let ref_occ: f64 =
            REFERENCE_OCCUPATION[atom.number as usize - 1][l] / (2.0 * l as f64 + 1.0);
        // iterate over angular components
        for i in (shell_i.sph_start..shell_i.sph_end) {
            if shell_i.polarization {
                p[[i, i]] = 0.0;
            } else {
                p[[i, i]] = ref_occ;
            }
        }
    }
    p
}

// pub fn create_density_ref(basis: &Basis, atoms: &[XtbAtom]) -> Array2<f64> {
//     // initialize empty density matrix
//     let nbas: usize = basis.basis_functions.len();
//     let mut p: Array2<f64> = Array2::zeros([nbas, nbas]);
//
//     // iterate over basis functions
//     for (i, funci) in basis.basis_functions.iter().enumerate() {
//         let atom: &XtbAtom = &atoms[funci.atom_index];
//         // get the reference occupations
//         let l: usize = funci.angular_momentum;
//         let ref_occ: f64 =
//             REFERNECE_OCCUPATION[atom.number as usize - 1][l] / (2.0 * l as f64 + 1.0);
//         if funci.polarization {
//             p[[i, i]] = 0.0;
//         } else {
//             p[[i, i]] = ref_occ;
//         }
//     }
//
//     p
// }
