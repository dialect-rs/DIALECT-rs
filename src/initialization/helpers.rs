use crate::initialization::parameters::SkfHandler;
use crate::initialization::Atom;
use crate::io::settings::ParameterizationConfig;
use crate::io::Configuration;
use crate::param::Element;
use crate::scc::gamma_approximation::{
    gaussian_decay, gaussian_decay_shell_resolved, slater_decay, slater_decay_shell_resolved,
    GammaFunction,
};
use hashbrown::HashMap;
use itertools::Itertools;

/// Finds the unique elements in a large list of elements/atoms that are specified by their atomic
/// numbers. For each of these unique elements a [Atom] is created and stored in a Vec<Atom>.
/// Furthermore, a HashMap<u8, Atom> is created that links an atomic number to an [Atom] so that
/// it can be cloned for every atom in the molecule.
pub fn get_unique_atoms(
    atomic_numbers: &[u8],
    config: &ParameterizationConfig,
) -> (Vec<Atom>, HashMap<u8, Atom>) {
    let mut unique_numbers: Vec<u8> = atomic_numbers.to_owned();
    unique_numbers.sort_unstable(); // fast sort of atomic numbers
    unique_numbers.dedup(); // delete duplicates
                            // create the unique Atoms
    let unique_atoms: Vec<Atom> = unique_numbers
        .iter()
        .map(|number| Atom::from((*number, config)))
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

pub fn get_unique_atoms_skf(
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
            let element: Element = handler.element_a;
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

    (unique_atoms, num_to_atom, skf_handlers)
}

pub fn initialize_gamma_function(
    unique_atoms: &[Atom],
    r_lr: f64,
    use_gaussian: bool,
    use_shell_resolved: bool,
    use_damping: bool,
) -> GammaFunction {
    let gammafunc: GammaFunction = if use_gaussian {
        if use_shell_resolved {
            // initialize the gamma function
            let sigma = gaussian_decay_shell_resolved(unique_atoms);
            let c: HashMap<((u8, u8), (u8, u8)), f64> = HashMap::new();
            let mut gf = GammaFunction::GaussianShellResolved { sigma, c, r_lr };
            gf.initialize();
            gf
        } else {
            // initialize the gamma function
            let tmp = gaussian_decay(unique_atoms);
            let sigma: HashMap<u8, f64> = tmp.0;
            let hubbards: HashMap<u8, f64> = tmp.1;
            let c: HashMap<(u8, u8), f64> = HashMap::new();
            let c_deriv: HashMap<(u8, u8), f64> = HashMap::new();
            let mut gf = GammaFunction::Gaussian {
                sigma,
                hubbards,
                c,
                c_deriv,
                r_lr,
            };
            gf.initialize();
            gf
        }
    } else {
        // initialize the slater function
        if use_shell_resolved {
            let tmp: (HashMap<(u8, u8), f64>, HashMap<u8, bool>) =
                slater_decay_shell_resolved(unique_atoms, use_damping);
            let tau = tmp.0;
            let damping = tmp.1;
            let gf: GammaFunction = GammaFunction::SlaterShellResolved { tau, r_lr, damping };
            gf
        } else {
            let tmp: (HashMap<u8, f64>, HashMap<u8, bool>) =
                slater_decay(unique_atoms, use_damping);
            let tau = tmp.0;
            let damping = tmp.1;
            let gf: GammaFunction = GammaFunction::Slater { tau, r_lr, damping };
            gf
        }
    };
    gammafunc
}
