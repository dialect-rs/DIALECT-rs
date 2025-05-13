use crate::xtb::initialization::atom::XtbAtom;
use crate::xtb::{
    initialization::basis::Basis,
    scc::gamma_matrix::{init_avg_hubbard_u, XtbGammaFunction},
};
use hashbrown::HashMap;

pub fn get_unique_atoms_xtb(atomic_numbers: &[u8]) -> (Vec<XtbAtom>, HashMap<u8, XtbAtom>) {
    let mut unique_numbers: Vec<u8> = atomic_numbers.to_owned();
    unique_numbers.sort_unstable(); // fast sort of atomic numbers
    unique_numbers.dedup(); // delete duplicates
                            // create the unique Atoms
    let unique_atoms: Vec<XtbAtom> = unique_numbers
        .iter()
        .map(|number| XtbAtom::from(*number))
        .collect();
    let mut num_to_atom: HashMap<u8, XtbAtom> = HashMap::with_capacity(unique_numbers.len());
    // insert the atomic numbers and the reference to atoms in the HashMap
    for (num, atom) in unique_numbers
        .into_iter()
        .zip(unique_atoms.clone().into_iter())
    {
        num_to_atom.insert(num, atom);
    }
    return (unique_atoms, num_to_atom);
}

pub fn init_gamma_func_xtb(
    atoms: &[XtbAtom],
    basis: &Basis,
    unique_len: usize,
) -> XtbGammaFunction {
    // init gamma function
    let map: HashMap<(u8, u8), f64> = init_avg_hubbard_u(&atoms, &basis, unique_len);
    let mut gamma_func: XtbGammaFunction = XtbGammaFunction {
        c: map,
        eta: HashMap::new(),
    };
    gamma_func.initialize();
    gamma_func
}
