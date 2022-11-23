use ndarray::prelude::*;
use ndarray::Order;
use std::collections::HashMap;

/// The differences between the virtual and occupied orbitals are computed. The quantity to be
/// computed can be either the energies of the orbitals sets or e.g. the occupation. The length
/// of the output Array will be the `len(occ_quant) x len(virt_quant)`.
pub fn orbe_differences(occ_quant: ArrayView1<f64>, virt_quant: ArrayView1<f64>) -> Array1<f64> {
    // Number of occupied orbitals.
    let n_occ: usize = occ_quant.len();
    // Number of virtual orbitals.
    let n_virt: usize = virt_quant.len();
    // Compute the distance matrix by broadcasting the energies of the occupied orbitals to
    // 2D array of the shape n_occ x n_virt and subtract the virtual energies.
    (&virt_quant
        - &occ_quant
            .to_owned()
            .insert_axis(Axis(1))
            .broadcast((n_occ, n_virt))
            .unwrap())
        .to_shape(((n_occ * n_virt), Order::RowMajor))
        .unwrap()
        .to_owned()
}

/// A cache for the products of the CIS-Hamiltonian with the subspace vectors.
/// The idea for this type of structure is inspired by Psi4.
#[derive(Clone, Debug)]
pub struct ProductCache {
    products: HashMap<&'static str, Array2<f64>>,
}

impl ProductCache {
    /// A new product cache with an empty dictionary is created.
    pub fn new() -> Self {
        Self {
            products: HashMap::new(),
        }
    }

    /// New product are added to the cache and all products with this key are returned.
    pub fn add(&mut self, key: &'static str, value: Array2<f64>) -> ArrayView2<f64> {
        // If the key is not yet in the HashMap the key-value pair is inserted. Otherwise
        // the array is stacked as new columns to the old values.
        self.products
            .entry(key)
            .or_insert(Array2::zeros([value.nrows(), 0]))
            .append(Axis(1), value.view());
        // Return a view of the products.
        self.products.get(key).unwrap().view()
    }

    /// The HashMap is cleared.
    pub fn reset(&mut self) -> () {
        self.products = HashMap::new();
    }

    /// The number of columns (vectors) is returned.
    pub fn count(&self, key: &'static str) -> usize {
        if self.products.contains_key(key) {
            self.products[key].ncols()
        } else {
            0
        }
    }
}
