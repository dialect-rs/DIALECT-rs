//! Fragment atom-slice helpers shared by the DFTB and xTB FMO stacks.

use dialect_dftb_core::atom::Atom;
use dialect_xtb_core::initialization::atom::XtbAtom;
use std::ops::Range;

pub fn get_pair_slice(atoms: &[Atom], mi_range: Range<usize>, mj_range: Range<usize>) -> Vec<Atom> {
    atoms[mi_range]
        .iter()
        .cloned()
        .chain(atoms[mj_range].iter().cloned())
        .collect()
}

pub fn get_pair_slice_xtb(
    atoms: &[XtbAtom],
    mi_range: Range<usize>,
    mj_range: Range<usize>,
) -> Vec<XtbAtom> {
    atoms[mi_range]
        .iter()
        .cloned()
        .chain(atoms[mj_range].iter().cloned())
        .collect()
}

pub fn get_trimer_slice_xtb(
    atoms: &[XtbAtom],
    mi_range: Range<usize>,
    mj_range: Range<usize>,
    mk_range: Range<usize>,
) -> Vec<XtbAtom> {
    atoms[mi_range]
        .iter()
        .cloned()
        .chain(atoms[mj_range].iter().cloned())
        .chain(atoms[mk_range].iter().cloned())
        .collect()
}
