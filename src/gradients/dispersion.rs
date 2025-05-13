use crate::initialization::Atom;
use crate::io::settings::DispersionConfig;
use nalgebra::Point3;
use ndarray::prelude::*;
use rusty_dftd_lib::model::*;
use rusty_dftd_lib::*;

pub fn gradient_disp(atoms: &[Atom], config: &DispersionConfig) -> Array1<f64> {
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
    Array1::from_iter(disp_result.gradient.unwrap())
}
