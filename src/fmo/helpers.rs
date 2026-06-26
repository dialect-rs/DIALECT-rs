pub use dialect_utilities::fmo_helpers::get_pair_slice;
use super::Monomer;
use crate::constants;
use crate::{initialization::Atom, io::settings::IdentificationConfig};
use std::fs;
use std::path::Path;

/// The atoms in the pair are two blocks in the Vec<Atom>, but these two blocks are in general not
/// connected. Since non-contiguous views of data structures are not trivial, the blocks are just
/// sliced individually and then appended afterwards.



pub use dialect_state::slices::{MolIncrements, MolIndices, MolecularSlice};

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
