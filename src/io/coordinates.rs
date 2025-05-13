use crate::constants::BOHR_TO_ANGS;
use crate::initialization::Atom;
use crate::io::settings::ParameterizationConfig;
use chemfiles::{Frame, Trajectory};
use hashbrown::HashMap;
use ndarray::Array2;

/// Extract the atomic numbers and positions (in bohr) from a [Frame](chemfiles::frame)
pub fn frame_to_coordinates(frame: Frame) -> (Vec<u8>, Array2<f64>) {
    let mut positions: Array2<f64> = Array2::from_shape_vec(
        (frame.size(), 3),
        frame
            .positions()
            .iter()
            .flat_map(|array| array.iter())
            .cloned()
            .collect(),
    )
    .unwrap();
    // transform the coordinates from angstrom to bohr
    positions /= BOHR_TO_ANGS;
    // read the atomic number of each coordinate
    let atomic_numbers: Vec<u8> = (0..frame.size())
        .map(|i| frame.atom(i).atomic_number() as u8)
        .collect();

    // let mut smiles_repr: Trajectory = Trajectory::memory_writer("SMI").unwrap();
    // smiles_repr.write(&mut frame).unwrap();
    // let smiles: String = smiles_repr.memory_buffer().unwrap().replace('~', "").replace('\n', "");
    (atomic_numbers, positions)
}

/// Extract the atoms and coordinates from the [Frame](chemfiles::Frame). The unique atoms will
/// be stored as a HashMap and a Vec<> with all [Atom]s and their position will be returned. The
/// stored position in each [Atom] are in bohr.
pub fn frame_to_atoms(frame: Frame, config: &ParameterizationConfig) -> (Vec<Atom>, Vec<Atom>) {
    let mut unique_atoms_map: HashMap<u8, Atom> = HashMap::new();
    let mut unique_atoms: Vec<Atom> = Vec::new();
    let mut atoms: Vec<Atom> = Vec::with_capacity(frame.size());
    for i in 0..frame.size() {
        let number: u8 = frame.atom(i).atomic_number() as u8;
        if !unique_atoms_map.contains_key(&number) {
            unique_atoms_map.insert(number, Atom::from((number, config)));
            unique_atoms.push(Atom::from((number, config)));
        }
        let mut atom: Atom = unique_atoms_map.get(&number).unwrap().clone();
        atom.position_from_slice(&frame.positions()[i]);
        // Convert angstrom to bohr. Assert that the coordinates are given in Angstrom
        atom.xyz /= BOHR_TO_ANGS;
        atoms.push(atom);
    }
    (atoms, unique_atoms)
}

/// Read a xyz-geometry file like .xyz or .pdb and returns a [Frame](chemfiles::Frame)
pub fn read_file_to_frame(filename: &str) -> Frame {
    // read the geometry file
    let mut trajectory = Trajectory::open(filename, 'r').unwrap();
    let mut frame = Frame::new();
    // if multiple geometries are contained in the file, we will only use the first one
    trajectory.read(&mut frame).unwrap();
    frame
}
