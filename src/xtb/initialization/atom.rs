use crate::constants;
use crate::param::elements::Element;
use crate::utils::array_helper::argsort_usize;
use crate::xtb::parameters::REFERENCE_OCCUPATION;
use nalgebra::Vector3;
use ndarray::prelude::*;
use std::cmp::Ordering;
use std::ops::{Neg, Sub};

#[derive(Clone, Debug)]
pub struct XtbAtom {
    /// Name of the chemical element
    pub name: &'static str,
    /// Ordinary number of the element
    pub number: u8,
    /// Element as an enum
    pub kind: Element,
    /// Position of the atom in bohr
    pub xyz: Vector3<f64>,
    /// Number of valence electrons
    pub n_elec: usize,
}

impl From<Element> for XtbAtom {
    /// Create a new [ReducedAtom] from the chemical [Element](crate::initialization::elements::Element).
    /// The parameterization from the parameter files is loaded and the Hubbard parameter
    /// and the valence orbitals are stored in this type.
    fn from(element: Element) -> Self {
        let symbol: &'static str = element.symbol();
        let mut n_elec: usize = 0;
        for idx in 0..3 {
            n_elec += REFERENCE_OCCUPATION[element.number_usize() - 1][idx] as usize;
        }

        XtbAtom {
            name: symbol,
            number: element.number(),
            kind: element,
            xyz: Vector3::<f64>::zeros(),
            n_elec,
        }
    }
}

impl XtbAtom {
    pub fn position_from_slice(&mut self, position: &[f64]) {
        self.xyz = Vector3::from_iterator(position.iter().cloned());
    }

    pub fn position_from_ndarray(&mut self, position: Array1<f64>) {
        let xyz: Vector3<f64> = nalgebra::Matrix::from_vec_generic(
            nalgebra::Const::<3>,
            nalgebra::Const::<1>,
            position.into_raw_vec(),
        );
        self.xyz = xyz;
    }
}

impl From<&str> for XtbAtom {
    /// Create a new [ReducedAtom] from the atomic symbol (case insensitive). The parameterization from the
    /// parameter files is loaded and the Hubbard parameter and the valence orbitals are stored in
    /// this type.
    fn from(symbol: &str) -> Self {
        Self::from(Element::from(symbol))
    }
}

impl From<&XtbAtom> for XtbAtom {
    // Create a new [ReducedAtom] from a reference to an [ReducedAtom].
    fn from(atom: &XtbAtom) -> Self {
        atom.clone()
    }
}

impl From<u8> for XtbAtom {
    /// Create a new [ReducedAtom] from the atomic number. The parameterization from the
    /// parameter files is loaded and the Hubbard parameter and the valence orbitals are stored in
    /// this type.
    fn from(number: u8) -> Self {
        Self::from(Element::from(number))
    }
}
impl PartialEq for XtbAtom {
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
    }
}

impl Sub for XtbAtom {
    type Output = Vector3<f64>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.xyz - rhs.xyz
    }
}

impl Sub for &XtbAtom {
    type Output = Vector3<f64>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.xyz - rhs.xyz
    }
}

impl Eq for XtbAtom {}

impl PartialOrd for XtbAtom {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.number.partial_cmp(&other.number)
    }
}
