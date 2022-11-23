use crate::initialization::parameters::SkfHandler;
use crate::initialization::Atom;
use crate::io::Configuration;
use crate::param::Element;
use crate::scc::gamma_approximation::{gaussian_decay, GammaFunction};
use hashbrown::HashMap;
use itertools::Itertools;

/// Finds the unique elements in a large list of elements/atoms that are specified by their atomic
/// numbers. For each of these unique elements a [Atom] is created and stored in a Vec<Atom>.
/// Furthermore, a HashMap<u8, Atom> is created that links an atomic number to an [Atom] so that
/// it can be cloned for every atom in the molecule.
pub fn get_unique_atoms(atomic_numbers: &[u8]) -> (Vec<Atom>, HashMap<u8, Atom>) {
    let mut unique_numbers: Vec<u8> = atomic_numbers.to_owned();
    unique_numbers.sort_unstable(); // fast sort of atomic numbers
    unique_numbers.dedup(); // delete duplicates
                            // create the unique Atoms
    let unique_atoms: Vec<Atom> = unique_numbers
        .iter()
        .map(|number| Atom::from(*number))
        .collect();
    let mut num_to_atom: HashMap<u8, Atom> = HashMap::with_capacity(unique_numbers.len());
    // insert the atomic numbers and the reference to atoms in the HashMap
    for (num, atom) in unique_numbers
        .into_iter()
        .zip(unique_atoms.clone().into_iter())
    {
        num_to_atom.insert(num, atom);
    }
    return (unique_atoms, num_to_atom);
}

pub fn get_unique_atoms_mio(
    atomic_numbers: &[u8],
    config: &Configuration,
) -> (Vec<Atom>, HashMap<u8, Atom>, Vec<SkfHandler>) {
    let mut unique_numbers: Vec<u8> = atomic_numbers.to_owned();
    unique_numbers.sort_unstable(); // fast sort of atomic numbers
    unique_numbers.dedup(); // delete duplicates

    // create SkfHandlers for homonuclear and heteronuclear combinations
    let mut homonuc_skf: Vec<SkfHandler> = Vec::new();
    let mut heteronuc_skf: Vec<SkfHandler> = Vec::new();
    let path_prefix: String = config.slater_koster.skf_directory.clone();

    let element_iter = unique_numbers.iter().map(|number| Element::from(*number));
    for (kind1, kind2) in element_iter.clone().cartesian_product(element_iter) {
        if kind1.number() > kind2.number() {
            continue;
        }
        if kind1 == kind2 {
            homonuc_skf.push(SkfHandler::new(kind1, kind2, path_prefix.clone()));
        } else {
            heteronuc_skf.push(SkfHandler::new(kind1, kind2, path_prefix.clone()));
        }
    }

    // create the unique Atoms
    let unique_atoms: Vec<Atom> = homonuc_skf
        .iter()
        .map(|handler| {
            let element: Element = handler.element_a.clone();
            Atom::from((element, handler))
        })
        .collect();
    let mut num_to_atom: HashMap<u8, Atom> = HashMap::with_capacity(unique_numbers.len());
    // insert the atomic numbers and the reference to atoms in the HashMap
    for (num, atom) in unique_numbers
        .into_iter()
        .zip(unique_atoms.clone().into_iter())
    {
        num_to_atom.insert(num, atom);
    }
    // combine homo- and heteronuclear skf_handlers
    let mut skf_handlers: Vec<SkfHandler> = Vec::new();
    skf_handlers.append(&mut homonuc_skf);
    skf_handlers.append(&mut heteronuc_skf);

    return (unique_atoms, num_to_atom, skf_handlers);
}

pub fn initialize_gamma_function(unique_atoms: &[Atom], r_lr: f64) -> GammaFunction {
    // initialize the gamma function
    let sigma: HashMap<u8, f64> = gaussian_decay(&unique_atoms);
    let c: HashMap<(u8, u8), f64> = HashMap::new();
    let mut gf = GammaFunction::Gaussian {
        sigma,
        c,
        r_lr: r_lr,
    };
    gf.initialize();
    gf
}

pub fn initialize_unrestricted_elec(charge: i8, n_elec: usize, multiplicity: u8) -> (f64, f64) {
    let mut alpha_electrons: f64 = 0.0;
    let mut beta_electrons: f64 = 0.0;

    if multiplicity == 1 && charge == 0 {
        alpha_electrons = (n_elec / 2) as f64;
        beta_electrons = (n_elec / 2) as f64;
    } else if multiplicity == 3 && charge == 0 {
        alpha_electrons = (n_elec / 2) as f64 + 0.5;
        beta_electrons = (n_elec / 2) as f64 - 0.5;
    } else if multiplicity == 2 {
        if charge == 1 {
            alpha_electrons = (n_elec / 2) as f64;
            beta_electrons = (n_elec / 2) as f64 - 1.0;
        } else if charge == -1 {
            alpha_electrons = (n_elec / 2) as f64 + 1.0;
            beta_electrons = (n_elec / 2) as f64;
        }
    }
    return (alpha_electrons, beta_electrons);
}
