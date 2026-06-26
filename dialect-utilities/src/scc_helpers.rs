//! Small SCC building blocks shared by the DFTB and xTB drivers.

use dialect_dftb_core::atom::Atom;
use itertools::Itertools;
use ndarray::prelude::*;

pub fn density_matrix(orbs: ArrayView2<f64>, f: &[f64]) -> Array2<f64> {
    let occ_indx: Vec<usize> = f.iter().positions(|&x| x > 0.0).collect();
    let occ_orbs: Array2<f64> = orbs.select(Axis(1), &occ_indx);
    let f_occ: Vec<f64> = f.iter().filter(|&&x| x > 0.0).cloned().collect();
    // THIS IS NOT AN EFFICIENT WAY TO BUILD THE LEFT HAND SIDE
    let mut f_occ_mat: Vec<f64> = Vec::new();
    for _ in 0..occ_orbs.nrows() {
        for val in f_occ.iter() {
            f_occ_mat.push(*val);
        }
    }
    let f_occ_mat: Array2<f64> = Array2::from_shape_vec(occ_orbs.raw_dim(), f_occ_mat).unwrap();
    let p: Array2<f64> = (f_occ_mat * &occ_orbs).dot(&occ_orbs.t());
    p
}

pub fn density_matrix_ref(n_orbs: usize, atoms: &[Atom]) -> Array2<f64> {
    let mut p0: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
    // iterate over orbitals on center i
    let mut idx: usize = 0;
    for atomi in atoms.iter() {
        // how many electrons are put into the nl-shell
        for occ in atomi.valorbs_occupation.iter() {
            p0[[idx, idx]] = *occ;
            idx += 1;
        }
    }
    p0
}

pub fn outer_sum(vec: ArrayView1<f64>) -> Array2<f64> {
    let vec_column: Array2<f64> = vec.to_owned().insert_axis(Axis(1));
    let result: Array2<f64> = &vec_column.broadcast((vec.dim(), vec.dim())).unwrap() + &vec;
    result
}

pub fn aovec_to_aomat(esp_aowise: ArrayView1<f64>, n_orbs: usize) -> Array2<f64> {
    let esp_ao_column: Array2<f64> = esp_aowise.clone().to_owned().insert_axis(Axis(1));
    let esp_ao: Array2<f64> = &esp_ao_column.broadcast((n_orbs, n_orbs)).unwrap() + &esp_aowise;
    esp_ao
}
