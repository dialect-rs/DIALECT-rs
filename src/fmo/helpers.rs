use super::Monomer;
use crate::constants;
use crate::{initialization::Atom, io::settings::IdentificationConfig};
use ndarray::Slice;
use std::fs;
use std::ops::Range;
use std::path::Path;

/// The atoms in the pair are two blocks in the Vec<Atom>, but these two blocks are in general not
/// connected. Since non-contiguous views of data structures are not trivial, the blocks are just
/// sliced individually and then appended afterwards.
pub fn get_pair_slice(atoms: &[Atom], mi_range: Range<usize>, mj_range: Range<usize>) -> Vec<Atom> {
    atoms[mi_range]
        .iter()
        .cloned()
        .chain(atoms[mj_range].iter().cloned())
        .collect()
}

#[derive(Copy, Clone)]
pub struct MolIndices {
    pub atom: usize,
    pub orbs: usize,
    pub occs: usize,
    pub virts: usize,
}

impl MolIndices {
    pub fn new() -> Self {
        Self {
            atom: 0,
            orbs: 0,
            occs: 0,
            virts: 0,
        }
    }

    pub fn add(&mut self, incr: MolIncrements) {
        self.atom += incr.atom;
        self.orbs += incr.orbs;
        self.occs += incr.occs;
        self.virts += incr.virts;
    }
}

#[derive(Copy, Clone)]
pub struct MolIncrements {
    pub atom: usize,
    pub orbs: usize,
    pub occs: usize,
    pub virts: usize,
}

/// Type that holds different Slices that are frequently used for indexing of molecular subunits
#[derive(Debug, Clone)]
pub struct MolecularSlice {
    /// [Slice](ndarray::prelude::Slice) for the atoms corresponding to the molecular unit
    pub atom: Slice,
    /// Similar to the atom slice, but as an Range. In contrast to the Slice the Range does not
    /// implement the Copy trait
    atom_range: Range<usize>,
    /// Gradient slice, this is the atom slice multiplied by the factor 3
    pub grad: Slice,
    /// [Slice](ndarray::prelude::Slice) for the orbitals corresponding to this molecular unit
    pub orb: Slice,
    /// [Slice](ndarray::prelude::Slice) for occupied orbitals corresponding to this molecular unit
    pub occ_orb: Slice,
    /// [Slice](ndarray::prelude::Slice) for virtual orbitals corresponding to this molecular unit
    pub virt_orb: Slice,
}

impl MolecularSlice {
    pub fn new(indices: MolIndices, incr: MolIncrements) -> Self {
        MolecularSlice {
            atom: Slice::from(indices.atom..(indices.atom + incr.atom)),
            atom_range: indices.atom..(indices.atom + incr.atom),
            grad: Slice::from((indices.atom * 3)..(indices.atom + incr.atom) * 3),
            orb: Slice::from(indices.orbs..(indices.orbs + incr.orbs)),
            occ_orb: Slice::from(indices.occs..(indices.occs + incr.occs)),
            virt_orb: Slice::from(indices.virts..(indices.virts + incr.virts)),
        }
    }

    /// Return the range of the atoms corresponding to this molecular unit
    pub fn atom_as_range(&self) -> Range<usize> {
        // since Range does not implement Copy trait, it need to be cloned every time it gets called
        self.atom_range.clone()
    }
}

pub fn remove_duplicate_atoms(atoms: &[Atom]) -> Vec<Atom> {
    let mut duplicates: Vec<usize> = Vec::new();
    // iterate over all atoms
    for (idx_i, atom_i) in atoms.iter().enumerate() {
        for (idx_j, atom_j) in atoms.iter().enumerate() {
            if idx_i != idx_j && idx_i < idx_j {
                let distance_xyz = atom_i.xyz - atom_j.xyz;
                let r = distance_xyz.norm();
                if r < 0.05 {
                    duplicates.push(idx_j);
                }
            }
        }
    }
    let mut new_atoms: Vec<Atom> = Vec::new();
    if duplicates.is_empty() {
        new_atoms = atoms.to_vec();
    } else {
        for (idx_i, atom_i) in atoms.iter().enumerate() {
            if !duplicates.contains(&idx_i) {
                new_atoms.push(atom_i.clone());
            }
        }
        // get string of all atoms
        let mut string: String = new_atoms.len().to_string();
        string += "\n\n";

        // iterate over the atoms
        for atom_i in new_atoms.iter() {
            string += &format!(
                "{}   {}   {}   {} \n",
                constants::ATOM_NAMES_UPPER[atom_i.number as usize],
                atom_i.xyz.x * constants::BOHR_TO_ANGS,
                atom_i.xyz.y * constants::BOHR_TO_ANGS,
                atom_i.xyz.z * constants::BOHR_TO_ANGS
            );
        }
        // write the string
        let file_path: &Path = Path::new("new_geom.xyz");
        fs::write(file_path, string).expect("Unable to write to new_geom.xyz file");
    }

    new_atoms
}

pub fn monomer_identification(
    config: &IdentificationConfig,
    atoms: &[Atom],
    monomers: &[Monomer],
) -> Vec<usize> {
    let mut monomer_indices: Vec<usize> = Vec::new();
    for coords in config.atom_coordinates.iter() {
        for mol in monomers.iter() {
            let monomer_atoms = &atoms[mol.slice.atom_as_range()];

            for atom in monomer_atoms.iter() {
                let x = coords[0] / constants::BOHR_TO_ANGS;
                let y = coords[1] / constants::BOHR_TO_ANGS;
                let z = coords[2] / constants::BOHR_TO_ANGS;

                let diff_norm: f64 = ((atom.xyz.x - x).powi(2)
                    + (atom.xyz.y - y).powi(2)
                    + (atom.xyz.z - z).powi(2))
                .sqrt();

                if diff_norm < 5.0e-3 {
                    println!("Monomer index: {}", mol.index);
                    monomer_indices.push(mol.index);
                }
            }
        }
    }
    monomer_indices
}
