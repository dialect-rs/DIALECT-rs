use dialect_base::constants::VDW_SUM;
use dialect_state::PairType;
use dialect_state::Properties;
use crate::fmo::monomer::XtbMonomer;
use crate::initialization::atom::XtbAtom;
use crate::initialization::basis::{create_basis_set, Basis};
use crate::scc::gamma_matrix::XtbGammaFunction;

/// A close FMO fragment pair treated with a full xTB SCC (the FMO2 pair
/// correction).
#[derive(Debug, Clone)]
pub struct XtbPair<'a> {
    /// Index of the first monomer contained in the pair
    pub i: usize,
    /// Index of the second monomer contained in the pair
    pub j: usize,
    /// Number of atoms (including ghost boundary atoms)
    pub n_atoms: usize,
    /// Number of atomic orbitals (including ghost basis functions)
    pub n_orbs: usize,
    pub gammafunction: &'a XtbGammaFunction,
    pub basis: Basis,
    /// Type that holds the calculated properties e.g. gamma matrix, overlap matrix and so on.
    pub properties: Properties,
    /// Number of real atoms (without ghost boundary atoms).
    pub n_real_atoms: usize,
    /// Number of real AOs (without ghost basis functions).
    pub n_real_orbs: usize,
    /// Number of real shells (without ghost shells).
    pub n_real_shells: usize,
}

impl PartialEq for XtbPair<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.i == other.i && self.j == other.j
    }
}

impl<'a> XtbPair<'a> {
    pub fn new(
        i: usize,
        j: usize,
        monomer1: &XtbMonomer,
        monomer2: &XtbMonomer,
        gammafunction: &'a XtbGammaFunction,
    ) -> Self {
        let n_atoms = monomer1.n_real_atoms + monomer2.n_real_atoms;
        let n_orbs = monomer1.n_real_orbs + monomer2.n_real_orbs;
        Self {
            i,
            j,
            n_atoms,
            n_orbs,
            properties: Properties::new(),
            gammafunction,
            basis: Basis {
                basis_functions: Vec::new(),
                shells: Vec::new(),
                nbas: 0,
            },
            n_real_atoms: n_atoms,
            n_real_orbs: n_orbs,
            n_real_shells: 0,
        }
    }

    /// Build the basis set from atoms. Called before SCC computation.
    pub fn init_basis(&mut self, atoms: &[XtbAtom]) {
        self.basis = create_basis_set(atoms);
    }

    /// Drop the basis set to free memory.
    pub fn clear_basis(&mut self) {
        self.basis = Basis {
            basis_functions: Vec::new(),
            shells: Vec::new(),
            nbas: 0,
        };
    }

    pub fn set_mo_indices(&mut self, n_elec: usize) {
        // get the indices of the occupied and virtual orbitals
        let mut occ_indices: Vec<usize> = Vec::new();
        let mut virt_indices: Vec<usize> = Vec::new();
        (0..self.n_orbs).for_each(|index| {
            if index < (n_elec / 2) {
                occ_indices.push(index)
            } else {
                virt_indices.push(index)
            }
        });
        self.properties.set_occ_indices(occ_indices);
        self.properties.set_virt_indices(virt_indices);
    }
}

/// A distant FMO fragment pair treated with the electrostatic-dimer (ESD)
/// approximation instead of a full SCC.
#[derive(Debug, Clone)]
pub struct XtbESDPair<'a> {
    /// Index of the first monomer
    pub i: usize,
    /// Index of the second monomer
    pub j: usize,
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of atomic orbitals
    pub n_orbs: usize,
    /// Type that holds calculated properties e.g. gamma matrix, overlap matrix and so on.
    pub properties: Properties,
    pub gammafunction: &'a XtbGammaFunction,
}

impl<'a> XtbESDPair<'a> {
    pub fn new(
        i: usize,
        j: usize,
        monomer1: &XtbMonomer,
        monomer2: &XtbMonomer,
        gammafunction: &'a XtbGammaFunction,
    ) -> Self {
        Self {
            i,
            j,
            n_atoms: monomer1.n_real_atoms + monomer2.n_real_atoms,
            n_orbs: monomer1.n_real_orbs + monomer2.n_real_orbs,
            properties: Properties::new(),
            gammafunction,
        }
    }
}

/// Check if the monomers are close to each other or not.
pub fn get_pair_type_xtb(mi_atoms: &[XtbAtom], mj_atoms: &[XtbAtom], vdw_scaling: f64) -> PairType {
    // Check if the shortest distance between two monomers is within the sum of the van-der-Waals
    // radii of the closest atom pair multiplied by a scaling factor. This threshold in terms of
    // DFTB was taken from https://pubs.acs.org/doi/pdf/10.1021/ct500489d (see page 4805).
    // the threshold is generally used in FMO theory and was originally presented in
    // Chem. Phys. Lett. 2002, 351, 475−480
    // For every atom we do a conversion from the u8 type usize. But it was checked and it
    // it does not seem to have a large effect on the performance.
    let mut kind: PairType = PairType::ESD;
    'pair_loop: for atomi in mi_atoms.iter() {
        for atomj in mj_atoms.iter() {
            if (atomi - atomj).norm()
                < vdw_scaling * VDW_SUM[atomi.number as usize][atomj.number as usize]
            {
                kind = PairType::Pair;
                break 'pair_loop;
            }
        }
    }
    kind
}
