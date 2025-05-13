use crate::initialization::geometry::*;
use crate::io::{frame_to_coordinates, read_file_to_frame, Configuration};
use crate::properties::Properties;
use crate::xtb::initialization::atom::XtbAtom;
use crate::xtb::initialization::basis::{create_basis_set, Basis};
use crate::xtb::initialization::helpers::{get_unique_atoms_xtb, init_gamma_func_xtb};
use crate::xtb::scc::gamma_matrix::XtbGammaFunction;
use chemfiles::Frame;
use hashbrown::HashMap;
use ndarray::prelude::*;

#[derive(Clone)]
pub struct XtbSystem {
    /// Type that holds all the input settings from the user.
    pub config: Configuration,
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of atomic orbitals
    pub n_orbs: usize,
    /// Number of valence electrons
    pub n_elec: usize,
    /// Charge of the system
    pub charge: i8,
    /// Indices of occupied orbitals starting from zero
    pub occ_indices: Vec<usize>,
    /// Indices of virtual orbitals
    pub virt_indices: Vec<usize>,
    /// Vector with the data of the individual atoms that are stored as [Atom] type.
    pub atoms: Vec<XtbAtom>,
    pub basis: Basis,
    /// Type that stores the  coordinates and matrices that depend on the position of the atoms
    pub geometry: Geometry,
    /// Type that holds the calculated properties e.g. gamma matrix, overlap matrix and so on.
    pub properties: Properties,
    pub gammafunction: XtbGammaFunction,
}

impl From<(Vec<u8>, Array2<f64>, Configuration)> for XtbSystem {
    // Creates a new [System] from a [Vec](alloc::vec) of atomic numbers, the coordinates as an [Array2](ndarray::Array2) and
    // the global configuration as [Configuration](crate::io::settings::Configuration).
    fn from(molecule: (Vec<u8>, Array2<f64>, Configuration)) -> Self {
        // get the unique [Atom]s and the HashMap with the mapping from the numbers to the [Atom]s
        let tmp: (Vec<XtbAtom>, HashMap<u8, XtbAtom>) = get_unique_atoms_xtb(&molecule.0);
        let unique_atoms = tmp.0;
        let num_to_atom = tmp.1;

        // get all the Atom's from the HashMap
        let mut atoms: Vec<XtbAtom> = Vec::with_capacity(molecule.0.len());
        molecule
            .0
            .iter()
            .for_each(|num| atoms.push((*num_to_atom.get(num).unwrap()).clone()));
        // set the positions for each atom
        molecule
            .1
            .outer_iter()
            .enumerate()
            .for_each(|(idx, position)| {
                atoms[idx].position_from_slice(position.as_slice().unwrap())
            });

        // create the basis set
        let basis: Basis = create_basis_set(&atoms);
        let n_orbs: usize = basis.nbas;

        // init gamma function
        let gamma_func: XtbGammaFunction = init_gamma_func_xtb(&atoms, &basis, unique_atoms.len());

        // set charge of the system
        let charge: i8 = molecule.2.mol.charge;
        // calculate the number of electrons
        let n_elec: usize =
            (atoms.iter().fold(0, |n, atom| n + atom.n_elec) as isize - charge as isize) as usize;

        // get the indices of the occupied and virtual orbitals
        let mut occ_indices: Vec<usize> = Vec::new();
        let mut virt_indices: Vec<usize> = Vec::new();
        (0..n_orbs).for_each(|index| {
            if index < (n_elec / 2) {
                occ_indices.push(index)
            } else {
                virt_indices.push(index)
            }
        });

        // Create the Geometry from the coordinates. At this point the coordinates have to be
        // transformed already in atomic units
        let mut geom: Geometry = Geometry::from(molecule.1);
        geom.set_matrices();
        // Create a new and empty Properties type
        let mut properties: Properties = Properties::new();
        properties.set_occ_indices(occ_indices.clone());
        properties.set_virt_indices(virt_indices.clone());

        // modify the dispersion parameters
        let mut config = molecule.2.clone();
        config.dispersion.a1 = 0.63;
        config.dispersion.a2 = 5.0;
        config.dispersion.s6 = 1.0;
        config.dispersion.s8 = 2.4;

        Self {
            config,
            n_atoms: molecule.0.len(),
            n_orbs,
            n_elec,
            charge,
            occ_indices,
            virt_indices,
            atoms,
            basis,
            geometry: geom,
            properties,
            gammafunction: gamma_func,
        }
    }
}

impl From<(Frame, Configuration)> for XtbSystem {
    // Creates a new [System] from a [Frame](chemfiles::Frame) and
    // the global configuration as [Configuration](crate::io::settings::Configuration).
    fn from(frame: (Frame, Configuration)) -> Self {
        let (numbers, coords) = frame_to_coordinates(frame.0);
        Self::from((numbers, coords, frame.1))
    }
}

impl From<(&str, Configuration)> for XtbSystem {
    fn from(filename_and_config: (&str, Configuration)) -> Self {
        let frame: Frame = read_file_to_frame(filename_and_config.0);
        let (numbers, coords) = frame_to_coordinates(frame);
        Self::from((numbers, coords, filename_and_config.1))
    }
}

impl XtbSystem {
    pub fn update_xyz(&mut self, coordinates: ArrayView1<f64>) {
        let coordinates: ArrayView2<f64> = coordinates.into_shape([self.atoms.len(), 3]).unwrap();
        // PARALLEL
        for (atom, xyz) in self.atoms.iter_mut().zip(coordinates.outer_iter()) {
            atom.position_from_ndarray(xyz.to_owned());
        }
        let basis: Basis = create_basis_set(&self.atoms);
        self.basis = basis;
    }

    pub fn get_xyz(&self) -> Array1<f64> {
        let xyz_list: Vec<Vec<f64>> = self
            .atoms
            .iter()
            .map(|atom| atom.xyz.iter().cloned().collect())
            .collect();
        Array1::from_shape_vec(3 * self.atoms.len(), itertools::concat(xyz_list)).unwrap()
    }
}
