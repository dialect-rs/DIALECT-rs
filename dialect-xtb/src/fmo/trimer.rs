use dialect_base::constants::VDW_SUM;
use dialect_state::Properties;
use crate::fmo::monomer::XtbMonomer;
use crate::initialization::atom::XtbAtom;
use crate::initialization::basis::{create_basis_set, Basis};
use crate::scc::gamma_matrix::XtbGammaFunction;

/// A close FMO fragment triple treated with a full xTB SCC (the FMO3
/// three-body correction).
#[derive(Debug, Clone)]
pub struct XtbTrimer<'a> {
    /// Index of the first monomer contained in the pair
    pub i: usize,
    /// Index of the second monomer contained in the pair
    pub j: usize,
    /// Index of the third monomer contained in the pair
    pub k: usize,
    /// Number of atoms (including ghost boundary atoms)
    pub n_atoms: usize,
    /// Number of atomic orbitals (including ghost basis functions)
    pub n_orbs: usize,
    pub gammafunction: &'a XtbGammaFunction,
    /// Basis set - only populated during SCC/gradient computation to save memory.
    pub basis_opt: Option<Basis>,
    /// Type that holds the calculated properties e.g. gamma matrix, overlap matrix and so on.
    pub properties: Properties,
    /// Number of real atoms (without ghost boundary atoms).
    pub n_real_atoms: usize,
    /// Number of real AOs (without ghost basis functions).
    pub n_real_orbs: usize,
    /// Number of real shells (without ghost shells).
    pub n_real_shells: usize,
}

impl PartialEq for XtbTrimer<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.i == other.i && self.j == other.j && self.k == other.k
    }
}

impl<'a> XtbTrimer<'a> {
    pub fn new(
        i: usize,
        j: usize,
        k: usize,
        monomer1: &XtbMonomer,
        monomer2: &XtbMonomer,
        monomer3: &XtbMonomer,
        gammafunction: &'a XtbGammaFunction,
    ) -> Self {
        let n_atoms = monomer1.n_real_atoms + monomer2.n_real_atoms + monomer3.n_real_atoms;
        let n_orbs = monomer1.n_real_orbs + monomer2.n_real_orbs + monomer3.n_real_orbs;
        Self {
            i,
            j,
            k,
            n_atoms,
            n_orbs,
            properties: Properties::new(),
            gammafunction,
            basis_opt: None,
            n_real_atoms: n_atoms,
            n_real_orbs: n_orbs,
            n_real_shells: 0,
        }
    }

    /// Returns a reference to the basis set. Panics if basis is not set.
    #[inline]
    pub fn basis(&self) -> &Basis {
        self.basis_opt.as_ref().unwrap()
    }

    /// Build the basis set from atoms. Called before SCC/gradient computation.
    pub fn init_basis(&mut self, atoms: &[XtbAtom]) {
        self.basis_opt = Some(create_basis_set(atoms));
    }

    /// Drop the basis set to free memory.
    pub fn clear_basis(&mut self) {
        self.basis_opt = None;
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

/// Check if the monomers are close to each other or not.
pub fn check_trimer_distances(
    mi_atoms: &[XtbAtom],
    mj_atoms: &[XtbAtom],
    mk_atoms: &[XtbAtom],
    vdw_scaling: f64,
) -> bool {
    // Check if the two shortest distances between the three monomers of the trimer
    // is within the sum of the van-der-Waals radii
    let mut bool_ij: bool = false;
    let mut bool_ik: bool = false;
    let mut bool_jk: bool = false;
    'pair_loop: for atomi in mi_atoms.iter() {
        for atomj in mj_atoms.iter() {
            if (atomi - atomj).norm()
                < vdw_scaling * VDW_SUM[atomi.number as usize][atomj.number as usize]
            {
                bool_ij = true;
                break 'pair_loop;
            }
        }
    }
    'pair_loop: for atomi in mi_atoms.iter() {
        for atomk in mk_atoms.iter() {
            if (atomi - atomk).norm()
                < vdw_scaling * VDW_SUM[atomi.number as usize][atomk.number as usize]
            {
                bool_ik = true;
                break 'pair_loop;
            }
        }
    }
    'pair_loop: for atomj in mj_atoms.iter() {
        for atomk in mk_atoms.iter() {
            if (atomj - atomk).norm()
                < vdw_scaling * VDW_SUM[atomj.number as usize][atomk.number as usize]
            {
                bool_jk = true;
                break 'pair_loop;
            }
        }
    }
    let create_trimer: bool = if bool_ij && bool_jk || bool_ij && bool_ik || bool_ik && bool_jk {
        true
    } else {
        false
    };

    create_trimer
}
