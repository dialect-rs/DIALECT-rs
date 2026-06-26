use dialect_state::MolecularSlice;
use dialect_state::Properties;
use crate::initialization::atom::XtbAtom;
use crate::initialization::basis::{create_basis_set, Basis};
use crate::scc::gamma_matrix::XtbGammaFunction;

/// A single FMO fragment (monomer) of the xTB supersystem.
#[derive(Debug, Clone)]
pub struct XtbMonomer<'a> {
    /// Number of atoms (including ghost boundary atoms)
    pub n_atoms: usize,
    /// Number of atomic orbitals (including ghost basis functions)
    pub n_orbs: usize,
    /// Number of shells (including ghost shells)
    pub n_shells: usize,
    /// Index of the monomer in the [SuperSystem]
    pub index: usize,
    /// Different Slices that correspond to this monomer
    pub slice: MolecularSlice,
    /// Type that holds the calculated properties e.g. gamma matrix, overlap matrix and so on.
    pub properties: Properties,
    pub gammafunction: &'a XtbGammaFunction,
    pub basis: Basis,
    /// Ghost boundary hydrogen atoms for covalent fragmentation (HOP).
    /// These are placed at BAA positions to stabilize the electronic structure.
    pub ghost_atoms: Vec<XtbAtom>,
    /// Number of real atoms (without ghost boundary atoms).
    /// Equal to n_atoms when no ghost atoms are present.
    pub n_real_atoms: usize,
    /// Number of real AOs (without ghost basis functions).
    /// Equal to n_orbs when no ghost atoms are present.
    pub n_real_orbs: usize,
    /// Number of real shells (without ghost shells).
    /// Equal to n_shells when no ghost atoms are present.
    pub n_real_shells: usize,
    /// Local atom indices of BDA (Bond Detached Atom) atoms in this fragment.
    /// Used to reduce their reference density by 1 (ZREF approach).
    pub bda_local_indices: Vec<usize>,
}

impl<'a> XtbMonomer<'a> {
    pub fn new(
        n_atoms: usize,
        n_orbs: usize,
        n_shells: usize,
        index: usize,
        slice: MolecularSlice,
        properties: Properties,
        basis: Basis,
        gammafunction: &'a XtbGammaFunction,
    ) -> Self {
        Self {
            n_atoms,
            n_orbs,
            n_shells,
            slice,
            index,
            properties,
            basis,
            gammafunction,
            ghost_atoms: Vec::new(),
            n_real_atoms: n_atoms,
            n_real_orbs: n_orbs,
            n_real_shells: n_shells,
            bda_local_indices: Vec::new(),
        }
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

impl PartialEq for XtbMonomer<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for XtbMonomer<'_> {}
