use crate::fmo::Fragment;
use crate::initialization::atom::Atom;
use crate::initialization::geometry::*;
use crate::initialization::parameters::*;
use crate::initialization::{
    get_unique_atoms, get_unique_atoms_mio, initialize_gamma_function, initialize_unrestricted_elec,
};
use crate::io::{frame_to_coordinates, read_file_to_frame, Configuration};
use crate::param::Element;
use crate::properties::Properties;
use crate::scc::gamma_approximation;
use crate::scc::gamma_approximation::GammaFunction;
use chemfiles::Frame;
use hashbrown::HashMap;
use itertools::Itertools;
use ndarray::prelude::*;
use std::borrow::BorrowMut;

/// Type that holds a molecular system that contains all data for the quantum chemical routines.
/// This type is only used for non-FMO calculations. In the case of FMO based calculation
/// the `FMO`, `Molecule` and `Pair` types are used instead.
pub struct System {
    /// Type that holds all the input settings from the user.
    pub config: Configuration,
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of atomic orbitals
    pub n_orbs: usize,
    /// Number of valence electrons
    pub n_elec: usize,
    /// Number of unpaired electrons (singlet -> 0, doublet -> 1, triplet -> 2)
    pub n_unpaired: usize,
    /// Number of alpha electrons in an unrestricted calculation
    pub alpha_elec: f64,
    /// Number of beta electrons in an unrestricted calculation
    pub beta_elec: f64,
    /// Charge of the system
    pub charge: i8,
    /// Indices of occupied orbitals starting from zero
    pub occ_indices: Vec<usize>,
    /// Indices of virtual orbitals
    pub virt_indices: Vec<usize>,
    /// Vector with the data of the individual atoms that are stored as [Atom] type. This is not an
    /// efficient solution because there are only a few [Atom] types and they are copied for each
    /// atom in the molecule. However, to the best of the authors knowledge the self referential
    /// structs are not trivial to implement in Rust. There are ways to do this by using crates like
    /// `owning_ref`, `rental`, or `ouroboros` or by using the [Pin](std::marker::Pin) type. The other
    /// crates does not fit exactly to our purpose and the [Pin](std::marker::Pin) requires a lot of
    /// extra effort. The computational cost for the clones was measured to be on the order of
    /// ~ 0.003 seconds (on a mac mini with M1) and to consume about 9 MB of memory for 20.000 [Atoms]
    pub atoms: Vec<Atom>,
    /// Type that stores the  coordinates and matrices that depend on the position of the atoms
    pub geometry: Geometry,
    /// Type that holds the calculated properties e.g. gamma matrix, overlap matrix and so on.
    pub properties: Properties,
    /// Repulsive potential type. Type that contains the repulsion energy and its derivative
    /// w.r.t. d/dR for each unique pair of atoms as a spline.
    pub vrep: RepulsivePotential,
    /// Slater-Koster parameters for the H0 and overlap matrix elements. For each unique atom pair
    /// the matrix elements and their derivatives can be splined.
    pub slako: SlaterKoster,
    /// Type of Gamma function. This can be either `Gaussian` or `Slater` type.
    pub gammafunction: GammaFunction,
    /// Gamma function for the long-range correction. Only used if long-range correction is requested
    pub gammafunction_lc: Option<GammaFunction>,
}

impl Fragment for System {}

impl From<(Vec<u8>, Array2<f64>, Configuration)> for System {
    /// Creates a new [System] from a [Vec](alloc::vec) of atomic numbers, the coordinates as an [Array2](ndarray::Array2) and
    /// the global configuration as [Configuration](crate::io::settings::Configuration).
    fn from(molecule: (Vec<u8>, Array2<f64>, Configuration)) -> Self {
        // create mutable Vectors
        let mut unique_atoms: Vec<Atom> = Vec::new();
        let mut num_to_atom: HashMap<u8, Atom> = HashMap::new();
        let mut skf_handlers: Vec<SkfHandler> = Vec::new();

        if molecule.2.slater_koster.use_external_skf == true {
            // get the unique [Atom]s and the HashMap with the mapping from the numbers to the [Atom]s
            // if use_mio is true, create a vector of homonuclear SkfHandlers and a vector
            // of heteronuclear SkfHandlers
            let tmp: (Vec<Atom>, HashMap<u8, Atom>, Vec<SkfHandler>) =
                get_unique_atoms_mio(&molecule.0, &molecule.2);
            unique_atoms = tmp.0;
            num_to_atom = tmp.1;
            skf_handlers = tmp.2;
        } else {
            // get the unique [Atom]s and the HashMap with the mapping from the numbers to the [Atom]s
            let tmp: (Vec<Atom>, HashMap<u8, Atom>) = get_unique_atoms(&molecule.0);
            unique_atoms = tmp.0;
            num_to_atom = tmp.1;
        }

        // get all the Atom's from the HashMap
        let mut atoms: Vec<Atom> = Vec::with_capacity(molecule.0.len());
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
        // calculate the number of electrons
        let n_elec: usize = atoms.iter().fold(0, |n, atom| n + atom.n_elec);
        // get the number of unpaired electrons from the input option
        let n_unpaired: usize = match molecule.2.mol.multiplicity {
            1u8 => 0,
            2u8 => 1,
            3u8 => 2,
            _ => panic!("The specified multiplicity is not implemented"),
        };
        // set charge of the system
        let charge: i8 = molecule.2.mol.charge;
        // set alpha and beta electrons of the system
        let (alpha_elec, beta_elec): (f64, f64) =
            initialize_unrestricted_elec(charge, n_elec, molecule.2.mol.multiplicity);

        // calculate the number of atomic orbitals for the whole system as the sum of the atomic
        // orbitals per atom
        let n_orbs: usize = atoms.iter().fold(0, |n, atom| n + atom.n_orbs);
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
        let mut slako: SlaterKoster = SlaterKoster::new();
        let mut vrep: RepulsivePotential = RepulsivePotential::new();

        // add all unique element pairs
        if molecule.2.slater_koster.use_external_skf == true {
            for handler in skf_handlers.iter() {
                // in the heteronuclear case, the slako tables of the element combinations "AB"
                // and "BA" must be combined
                // if handler.element_a == handler.element_b {
                //     let repot_table: RepulsivePotentialTable =
                //         RepulsivePotentialTable::from(handler);
                //     let slako_table: SlaterKosterTable =
                //         SlaterKosterTable::from((handler, None, "ab"));
                //
                //     // insert the tables into the hashmaps
                //     slako
                //         .map
                //         .insert((handler.element_a, handler.element_b), slako_table);
                //     // slako
                //     //     .add_from_handler(handler.element_a, handler.element_b, handler.clone(),None,"ab");
                //     vrep.map
                //         .insert((handler.element_a, handler.element_b), repot_table);
                // } else {
                let repot_table: RepulsivePotentialTable = RepulsivePotentialTable::from(handler);
                let slako_table_ab: SlaterKosterTable =
                    SlaterKosterTable::from((handler, None, "ab"));
                let slako_handler_ba: SkfHandler = SkfHandler::new(
                    handler.element_b,
                    handler.element_a,
                    molecule.2.slater_koster.skf_directory.clone(),
                );
                let slako_table: SlaterKosterTable =
                    SlaterKosterTable::from((&slako_handler_ba, Some(slako_table_ab), "ba"));

                // insert the tables into the hashmaps
                slako
                    .map
                    .insert((handler.element_a, handler.element_b), slako_table);
                // slako
                //     .add_from_handler(handler.element_a, handler.element_b, slako_handler_ba,Some(slako_table_ab),"ba");
                vrep.map
                    .insert((handler.element_a, handler.element_b), repot_table);
            }
        } else {
            let element_iter = unique_atoms.iter().map(|atom| Element::from(atom.number));
            for (kind1, kind2) in element_iter.clone().cartesian_product(element_iter) {
                slako.add(kind1, kind2);
                vrep.add(kind1, kind2);
            }
        }
        // initialize the gamma function
        let gf: GammaFunction = initialize_gamma_function(&unique_atoms, 0.0);
        // initialize the gamma function for long-range correction if it is requested
        let gf_lc: Option<GammaFunction> = if molecule.2.lc.long_range_correction {
            Some(initialize_gamma_function(
                &unique_atoms,
                molecule.2.lc.long_range_radius,
            ))
        } else {
            None
        };

        Self {
            config: molecule.2,
            n_atoms: molecule.0.len(),
            n_orbs: n_orbs,
            n_elec: n_elec,
            n_unpaired: n_unpaired,
            alpha_elec: alpha_elec,
            beta_elec: beta_elec,
            charge: charge,
            occ_indices: occ_indices,
            virt_indices: virt_indices,
            atoms: atoms,
            geometry: geom,
            properties: properties,
            vrep: vrep,
            slako: slako,
            gammafunction: gf,
            gammafunction_lc: gf_lc,
        }
    }
}

impl From<(Frame, Configuration)> for System {
    /// Creates a new [System] from a [Frame](chemfiles::Frame) and
    /// the global configuration as [Configuration](crate::io::settings::Configuration).
    fn from(frame: (Frame, Configuration)) -> Self {
        let (numbers, coords) = frame_to_coordinates(frame.0);
        Self::from((numbers, coords, frame.1))
    }
}

impl From<(&str, Configuration)> for System {
    /// Creates a new [System] from a &str and
    /// the global configuration as [Configuration](crate::io::settings::Configuration).
    fn from(filename_and_config: (&str, Configuration)) -> Self {
        let frame: Frame = read_file_to_frame(filename_and_config.0);
        let (numbers, coords) = frame_to_coordinates(frame);
        Self::from((numbers, coords, filename_and_config.1))
    }
}

impl System {
    pub fn update_xyz(&mut self, coordinates: Array1<f64>) {
        let coordinates: Array2<f64> = coordinates.into_shape([self.atoms.len(), 3]).unwrap();
        // TODO: The IntoIterator trait was released for ndarray 0.15. The dependencies should be
        // updated, so that this can be used. At the moment of writing ndarray-linalg is not yet
        // compatible with ndarray 0.15x
        // PARALLEL
        for (atom, xyz) in self.atoms.iter_mut().zip(coordinates.outer_iter()) {
            atom.position_from_ndarray(xyz.to_owned());
        }
        //.for_each(|(atom, xyz)| atom.position_from_ndarray(xyz.to_owned()))
    }

    pub fn get_xyz(&self) -> Array1<f64> {
        let xyz_list: Vec<Vec<f64>> = self
            .atoms
            .iter()
            .map(|atom| atom.xyz.iter().cloned().collect())
            .collect();
        Array1::from_shape_vec((3 * self.atoms.len()), itertools::concat(xyz_list)).unwrap()
    }
}
