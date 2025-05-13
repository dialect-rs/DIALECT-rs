use crate::constants::BOHR_TO_ANGS;
use crate::utils::array_helper::argsort_usize;
use crate::xtb::initialization::atom::XtbAtom;
use crate::xtb::initialization::system::XtbSystem;
use crate::xtb::parameters::{COV_RADII, HALOGEN_BOND_STRENGH, HALOGEN_DAMPING, HALOGEN_RAD_SCALE};

impl XtbSystem {
    pub fn get_halogen_correction(&self) -> f64 {
        // identify the halogens
        let halogens: Vec<usize> = check_halogen_donor(&self.atoms);
        // identify the halogens
        let acceptors: Vec<usize> = check_acceptor_atoms(&self.atoms);

        // return zero if no halogen is present
        if halogens.is_empty() || acceptors.is_empty() {
            0.0
        } else {
            // identify bond indices
            let bond_vec: Vec<(usize, usize, usize)> =
                get_halogen_bond_indices(&self.atoms, &halogens, &acceptors);
            let energy: f64 = calculate_halogen_bond_energy(&self.atoms, &bond_vec);

            energy
        }
    }
}

fn calculate_halogen_bond_energy(atoms: &[XtbAtom], bond_vec: &Vec<(usize, usize, usize)>) -> f64 {
    // energy variable
    let mut energy: f64 = 0.0;
    // iterate over the bond vectors
    for (halogen_idx, acceptor_idx, neighbor_idx) in bond_vec.iter() {
        // get the atoms
        let h_atom: &XtbAtom = &atoms[*halogen_idx];
        let a_atom: &XtbAtom = &atoms[*acceptor_idx];
        let n_atom: &XtbAtom = &atoms[*neighbor_idx];
        let h_number: usize = h_atom.number as usize - 1;
        let a_number: usize = a_atom.number as usize - 1;

        // covalent radii
        let cov_h: f64 = COV_RADII[h_number] / BOHR_TO_ANGS;
        let cov_a: f64 = COV_RADII[a_number] / BOHR_TO_ANGS;

        // get the bond strength
        let bond_strength: f64 = HALOGEN_BOND_STRENGH[h_number];
        // get the square of the distance norms
        let diff_h_a: f64 = (h_atom.xyz.x - a_atom.xyz.x).powi(2)
            + (h_atom.xyz.y - a_atom.xyz.y).powi(2)
            + (h_atom.xyz.z - a_atom.xyz.z).powi(2);
        let diff_h_n: f64 = (h_atom.xyz.x - n_atom.xyz.x).powi(2)
            + (h_atom.xyz.y - n_atom.xyz.y).powi(2)
            + (h_atom.xyz.z - n_atom.xyz.z).powi(2);
        let diff_n_a: f64 = (n_atom.xyz.x - a_atom.xyz.x).powi(2)
            + (n_atom.xyz.y - a_atom.xyz.y).powi(2)
            + (n_atom.xyz.z - a_atom.xyz.z).powi(2);

        // get the distance between halogen and acceptor
        let dist_ha: f64 = diff_h_a.sqrt();
        // get the ratio between covalent radii and the real distance
        let ratio: f64 = (cov_h + cov_a) / dist_ha;

        // get the damping damping value
        let damping: f64 =
            (0.5 - 0.25 * (diff_h_a + diff_h_n - diff_n_a) / (diff_h_n * diff_h_a).sqrt()).powi(6);
        // calculate term with HALOGEN_RAD_SCALE and ratio
        let ratio_term: f64 = HALOGEN_RAD_SCALE * ratio;

        // calculate the energy term
        energy +=
            damping * bond_strength * (ratio_term.powi(12) - HALOGEN_DAMPING * ratio_term.powi(6))
                / (ratio_term.powi(12) + 1.0);
    }

    energy
}

fn check_halogen_donor(atoms: &[XtbAtom]) -> Vec<usize> {
    // get the halogen donor indices
    let mut donor_indices: Vec<usize> = Vec::new();
    let halogen_donors: Vec<u8> = vec![17, 35, 53, 85];

    for (idx, atom) in atoms.iter().enumerate() {
        if halogen_donors.contains(&atom.number) {
            donor_indices.push(idx);
        }
    }
    donor_indices
}

fn check_acceptor_atoms(atoms: &[XtbAtom]) -> Vec<usize> {
    // get the halogen acceptor indices
    let mut acceptor_indices: Vec<usize> = Vec::new();
    let halogen_acceptors: Vec<u8> = vec![7, 8, 15, 16];

    for (idx, atom) in atoms.iter().enumerate() {
        if halogen_acceptors.contains(&atom.number) {
            acceptor_indices.push(idx);
        }
    }
    acceptor_indices
}

fn get_halogen_bond_indices(
    atoms: &[XtbAtom],
    halogens: &[usize],
    acceptors: &[usize],
) -> Vec<(usize, usize, usize)> {
    // prepare bond vector
    let mut bond_vec: Vec<(usize, usize, usize)> = Vec::new();

    // iterate over halogens
    for idx in halogens.iter() {
        // get the atom
        let halogen: &XtbAtom = &atoms[*idx];
        // distance variable
        let mut dist: f64 = 50.0;
        let mut neighbor_idx: usize = 0;
        let mut add_indices_neighbor: bool = false;
        let mut add_indices_acc: bool = false;

        // iterate over all atoms to get the neighbor
        for (atom_idx, atom) in atoms.iter().enumerate() {
            if atom_idx != *idx {
                // get the distance
                let r: f64 = ((halogen.xyz.x - atom.xyz.x).powi(2)
                    + (halogen.xyz.y - atom.xyz.y).powi(2)
                    + (halogen.xyz.z - atom.xyz.z).powi(2))
                .sqrt();
                // check if radius is smaller than the previous smallest radius and the cutoff
                if r < dist && r < 20.0 {
                    neighbor_idx = atom_idx;
                    dist = r;
                    add_indices_neighbor = true;
                }
            }
        }
        // temporary vec for all acceptors
        let mut acceptor_vec: Vec<usize> = Vec::new();

        // iterate over the acceptors
        for a_idx in acceptors.iter() {
            // get the atom
            let acceptor: &XtbAtom = &atoms[*a_idx];
            // get the radius
            let r: f64 = ((halogen.xyz.x - acceptor.xyz.x).powi(2)
                + (halogen.xyz.y - acceptor.xyz.y).powi(2)
                + (halogen.xyz.z - acceptor.xyz.z).powi(2))
            .sqrt();
            // check if the radius is smaller than the cutoff
            if r < 20.0 {
                acceptor_vec.push(*a_idx);
                add_indices_acc = true;
            }
        }
        // add indices to bond vectors
        if add_indices_neighbor && add_indices_acc {
            for acceptor_idx in acceptor_vec.iter() {
                bond_vec.push((*idx, *acceptor_idx, neighbor_idx));
            }
        }
    }
    bond_vec
}
