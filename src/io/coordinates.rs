use crate::constants::BOHR_TO_ANGS;
use crate::initialization::Atom;
use crate::io::settings::ParameterizationConfig;
use hashbrown::HashMap;
use xyz_parser::XyzFrame;


pub use dialect_utilities::io::xyz_frame_to_coordinates;

/// Extract the atoms and coordinates from an [XyzFrame](xyz_parser::XyzFrame). The unique atoms
/// will be stored as a HashMap and a Vec<> with all [Atom]s and their position will be returned.
/// The stored position in each [Atom] are in bohr.
pub fn xyz_frame_to_atoms(frame: &XyzFrame, config: &ParameterizationConfig) -> (Vec<Atom>, Vec<Atom>) {
    let mut unique_atoms_map: HashMap<u8, Atom> = HashMap::new();
    let mut unique_atoms: Vec<Atom> = Vec::new();
    let mut atoms: Vec<Atom> = Vec::with_capacity(frame.num_atoms());
    for i in 0..frame.num_atoms() {
        let number: u8 = frame.atomic_numbers[i];
        if !unique_atoms_map.contains_key(&number) {
            unique_atoms_map.insert(number, Atom::from((number, config)));
            unique_atoms.push(Atom::from((number, config)));
        }
        let mut atom: Atom = unique_atoms_map.get(&number).unwrap().clone();
        let (x, y, z) = frame.get_atom_coords(i);
        atom.position_from_slice(&[x, y, z]);
        // Convert angstrom to bohr. Assert that the coordinates are given in Angstrom
        atom.xyz /= BOHR_TO_ANGS;
        atoms.push(atom);
    }
    (atoms, unique_atoms)
}

