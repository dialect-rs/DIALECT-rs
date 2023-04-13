use crate::initialization::Atom;
use crate::io::settings::DispersionConfig;
use nalgebra::Point3;
use ndarray::prelude::*;
use rusty_dftd_lib::model::*;
use rusty_dftd_lib::*;

pub fn atomvec_to_aomat(
    esp_atomwise: ArrayView1<f64>,
    n_orbs: usize,
    atoms: &[Atom],
) -> Array2<f64> {
    let mut esp_ao_row: Array1<f64> = Array1::zeros(n_orbs);
    let mut mu: usize = 0;
    for (atom, esp_at) in atoms.iter().zip(esp_atomwise.iter()) {
        for _ in 0..atom.n_orbs {
            esp_ao_row[mu] = *esp_at;
            mu = mu + 1;
        }
    }
    let esp_ao_column: Array2<f64> = esp_ao_row.clone().insert_axis(Axis(1));
    let esp_ao: Array2<f64> = &esp_ao_column.broadcast((n_orbs, n_orbs)).unwrap() + &esp_ao_row;
    return esp_ao;
}

pub fn get_dispersion_energy(atoms: &[Atom], config: &DispersionConfig) -> f64 {
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
