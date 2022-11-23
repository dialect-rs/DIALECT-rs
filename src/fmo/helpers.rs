use crate::fmo::SuperSystem;
use crate::initialization::Atom;
use ndarray::Slice;
use std::ops::{AddAssign, Range};

/// The atoms in the pair are two blocks in the Vec<Atom>, but these two blocks are in general not
/// connected. Since non-contiguous views of data structures are not trivial, the blocks are just
/// sliced individually and then appended afterwards.
pub fn get_pair_slice(atoms: &[Atom], mi_range: Range<usize>, mj_range: Range<usize>) -> Vec<Atom> {
    atoms[mi_range]
        .iter()
        .map(|x| x.clone())
        .chain(atoms[mj_range].iter().map(|x| x.clone()))
        .collect()
}

#[derive(Copy, Clone)]
pub struct MolIndices {
    pub atom: usize,
    pub orbs: usize,
    pub occs: usize,
    pub virts: usize,
}

impl MolIndices {
    pub fn new() -> Self {
        Self {
            atom: 0,
            orbs: 0,
            occs: 0,
            virts: 0,
        }
    }

    pub fn add(&mut self, incr: MolIncrements) {
        self.atom += incr.atom;
        self.orbs += incr.orbs;
        self.occs += incr.occs;
        self.virts += incr.virts;
    }
}

#[derive(Copy, Clone)]
pub struct MolIncrements {
    pub atom: usize,
    pub orbs: usize,
    pub occs: usize,
    pub virts: usize,
}

/// Type that holds different Slices that are frequently used for indexing of molecular subunits
#[derive(Debug, Clone)]
pub struct MolecularSlice {
    /// [Slice](ndarray::prelude::Slice) for the atoms corresponding to the molecular unit
    pub atom: Slice,
    /// Similar to the atom slice, but as an Range. In contrast to the Slice the Range does not
    /// implement the Copy trait
    atom_range: Range<usize>,
    /// Gradient slice, this is the atom slice multiplied by the factor 3
    pub grad: Slice,
    /// [Slice](ndarray::prelude::Slice) for the orbitals corresponding to this molecular unit
    pub orb: Slice,
    /// [Slice](ndarray::prelude::Slice) for occupied orbitals corresponding to this molecular unit
    pub occ_orb: Slice,
    /// [Slice](ndarray::prelude::Slice) for virtual orbitals corresponding to this molecular unit
    pub virt_orb: Slice,
}

impl MolecularSlice {
    pub fn new(indices: MolIndices, incr: MolIncrements) -> Self {
        MolecularSlice {
            atom: Slice::from(indices.atom..(indices.atom + incr.atom)),
            atom_range: indices.atom..(indices.atom + incr.atom),
            grad: Slice::from((indices.atom * 3)..(indices.atom + incr.atom) * 3),
            orb: Slice::from(indices.orbs..(indices.orbs + incr.orbs)),
            occ_orb: Slice::from(indices.occs..(indices.occs + incr.occs)),
            virt_orb: Slice::from(indices.virts..(indices.virts + incr.virts)),
        }
    }

    /// Return the range of the atoms corresponding to this molecular unit
    pub fn atom_as_range(&self) -> Range<usize> {
        // since Range does not implement Copy trait, it need to be cloned every time it gets called
        self.atom_range.clone()
    }
}
