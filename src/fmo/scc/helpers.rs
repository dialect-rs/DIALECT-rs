use crate::initialization::Atom;
use nalgebra::Point3;
use ndarray::prelude::*;

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
