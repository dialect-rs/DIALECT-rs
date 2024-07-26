use crate::defaults;
use crate::initialization::Atom;
use ndarray::prelude::*;
use ndarray_linalg::Norm;

pub fn get_xyz_2d(atoms: &[Atom]) -> Array2<f64> {
    let mut xyz: Array2<f64> = Array2::zeros([0, 3]);
    atoms.iter().for_each(|atom| {
        xyz.push(Axis(0), Array1::from_iter(atom.xyz.iter().cloned()).view())
            .unwrap()
    });
    xyz
}

/// Type that holds the atomic numbers and positions of a molecule.
/// The distance, directions and proximity matrices are optional informations that are stored
/// in this type.
#[derive(Debug, Clone)]
pub struct Geometry {
    pub coordinates: Array2<f64>,
    pub distances: Option<Array2<f64>>,
    pub directions: Option<Array3<f64>>,
    pub proximities: Option<Array2<bool>>,
}

impl From<Array2<f64>> for Geometry {
    /// Constructs a new [Geometry] from the coordinates (in atomic units).
    /// The distance, directions and proximity matrices will be computed and set.
    fn from(coordinates: Array2<f64>) -> Self {
        Geometry {
            coordinates: coordinates,
            distances: None,
            directions: None,
            proximities: None,
        }
    }
}

impl Geometry {
    pub fn new() -> Self {
        Self {
            coordinates: Array2::zeros([1, 1]),
            distances: None,
            directions: None,
            proximities: None,
        }
    }

    /// Update the coordinates and the geometric matrices. The following matrices will
    /// be set to the struct: coordinates, distances, directions, proximities.
    pub fn update(&mut self, coordinates: Array2<f64>) {
        let (dist, dir, prox): (Array2<f64>, Array3<f64>, Array2<bool>) =
            distance_matrices(coordinates.view(), None);
        self.set_coordinates(coordinates);
        self.set_distances_from_array(dist);
        self.set_directions_from_array(dir);
        self.set_proximities_from_array(prox);
    }

    pub fn set_matrices(&mut self) {
        let (dist, dir, prox): (Array2<f64>, Array3<f64>, Array2<bool>) =
            distance_matrices(self.coordinates.view(), None);
        self.distances = Some(dist);
        self.directions = Some(dir);
        self.proximities = Some(prox);
    }

    /// Set distances, directions and proximity matrices to None to free up some memory.
    pub fn reset(&mut self) {
        self.distances = None;
        self.directions = None;
        self.proximities = None;
    }

    pub fn set_coordinates(&mut self, coordinates: Array2<f64>) {
        self.coordinates = coordinates;
    }

    pub fn set_distances(&mut self, distances: Option<Array2<f64>>) {
        self.distances = distances;
    }

    pub fn set_directions(&mut self, directions: Option<Array3<f64>>) {
        self.directions = directions;
    }

    pub fn set_proximities(&mut self, proximities: Option<Array2<bool>>) {
        self.proximities = proximities;
    }

    pub fn set_distances_from_array(&mut self, distances: Array2<f64>) {
        self.distances = Some(distances);
    }

    pub fn set_directions_from_array(&mut self, directions: Array3<f64>) {
        self.directions = Some(directions);
    }

    pub fn set_proximities_from_array(&mut self, proximities: Array2<bool>) {
        self.proximities = Some(proximities);
    }
}
/// Compute the distance, directions and proximity matrix from the atomic coordinates. The cutoff
/// parameter is used for the estimation if two atoms are in proximity to each other. The H0 and
/// Overlap matrix elements will only be computed for atoms that within this cutoff.
pub fn distance_matrices(
    coordinates: ArrayView2<f64>,
    cutoff: Option<f64>,
) -> (Array2<f64>, Array3<f64>, Array2<bool>) {
    let cutoff: f64 = cutoff.unwrap_or(defaults::PROXIMITY_CUTOFF);
    let n_atoms: usize = coordinates.nrows();
    let mut dist_matrix: Array2<f64> = Array::zeros((n_atoms, n_atoms));
    let mut directions_matrix: Array3<f64> = Array::zeros((n_atoms, n_atoms, 3));
    let mut prox_matrix: Array2<bool> = Array::from_elem((n_atoms, n_atoms), false);
    for (i, posi) in coordinates.outer_iter().enumerate() {
        for (j0, posj) in coordinates.slice(s![i.., ..]).outer_iter().enumerate() {
            let j: usize = j0 + i;
            let r: Array1<f64> = &posi - &posj;
            let r_ij: f64 = r.norm();
            if i != j {
                dist_matrix[[i, j]] = r_ij;
                dist_matrix[[j, i]] = r_ij;
                let e_ij: Array1<f64> = r / r_ij;
                directions_matrix.slice_mut(s![i, j, ..]).assign(&e_ij);
                directions_matrix.slice_mut(s![j, i, ..]).assign(&-e_ij);
            }
            if r_ij <= cutoff {
                prox_matrix[[i, j]] = true;
                prox_matrix[[j, i]] = true;
            }
        }
    }
    return (dist_matrix, directions_matrix, prox_matrix);
}

/// Computes the distance, directions and proximity matrix between to sets of coordinates. Only the
/// off diagonal block between atom A (in coordinate set I) and atom B (in coordinate set J)
pub fn distance_matrix_pair(
    coordinates_i: ArrayView2<f64>,
    coordinates_j: ArrayView2<f64>,
    cutoff: Option<f64>,
) -> (Array2<f64>, Array3<f64>, Array2<bool>) {
    let cutoff: f64 = cutoff.unwrap_or(defaults::PROXIMITY_CUTOFF);
    let n_atoms_i: usize = coordinates_i.nrows();
    let n_atoms_j: usize = coordinates_j.nrows();
    let _n_atoms: usize = n_atoms_i + n_atoms_j;
    let mut dist_matrix: Array2<f64> = Array::zeros((n_atoms_i, n_atoms_j));
    let mut directions_matrix: Array3<f64> = Array::zeros((n_atoms_i, n_atoms_j, 3));
    let mut prox_matrix: Array2<bool> = Array::from_elem((n_atoms_i, n_atoms_j), false);
    for (i, pos_a) in coordinates_i.outer_iter().enumerate() {
        for (j, pos_b) in coordinates_j.outer_iter().enumerate() {
            let r: Array1<f64> = &pos_a - &pos_b;
            let r_ij: f64 = r.norm();
            if i != j {
                dist_matrix[[i, j]] = r_ij;
                dist_matrix[[j, i]] = r_ij;
                let e_ij: Array1<f64> = r / r_ij;
                directions_matrix.slice_mut(s![i, j, ..]).assign(&e_ij);
                directions_matrix.slice_mut(s![j, i, ..]).assign(&-e_ij);
            }
            if r_ij <= cutoff {
                prox_matrix[[i, j]] = true;
                prox_matrix[[j, i]] = true;
            }
        }
    }
    return (dist_matrix, directions_matrix, prox_matrix);
}
