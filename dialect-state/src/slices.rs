//! Index bookkeeping for molecular subunits: per-fragment atom/orbital
//! slice collections used throughout the FMO code.

use ndarray::Slice;
use std::ops::Range;

#[derive(Copy, Clone)]
pub struct MolIndices {
    pub atom: usize,
    pub orbs: usize,
    pub occs: usize,
    pub virts: usize,
    pub shells: usize,
}

impl MolIndices {
    pub fn new() -> Self {
        Self {
            atom: 0,
            orbs: 0,
            occs: 0,
            virts: 0,
            shells: 0,
        }
    }

    pub fn add(&mut self, incr: MolIncrements) {
        self.atom += incr.atom;
        self.orbs += incr.orbs;
        self.occs += incr.occs;
        self.virts += incr.virts;
        self.shells += incr.shells;
    }
}

#[derive(Copy, Clone)]
pub struct MolIncrements {
    pub atom: usize,
    pub orbs: usize,
    pub occs: usize,
    pub virts: usize,
    pub shells: usize,
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
    /// [Slice](ndarray::prelude::Slice) for the shells corresponding to this molecular unit
    pub shell: Slice,
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
            shell: Slice::from(indices.shells..(indices.shells + incr.shells)),
        }
    }

    /// Return the range of the atoms corresponding to this molecular unit
    pub fn atom_as_range(&self) -> Range<usize> {
        // since Range does not implement Copy trait, it need to be cloned every time it gets called
        self.atom_range.clone()
    }
}
