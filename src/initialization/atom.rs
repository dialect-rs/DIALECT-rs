use crate::constants;
use crate::initialization::parameters::{PseudoAtom, PseudoAtomSkf, SkfHandler};
use crate::param::elements::Element;
use crate::utils::array_helper::argsort_usize;
use nalgebra::Vector3;
use ndarray::prelude::*;
use std::cmp::Ordering;
use std::ops::{Neg, Sub};

/// `Atom` type that contains basic information about the chemical element as well as the
/// data used for the semi-empirical parameters that are used in the DFTB calculations.
/// The `StructofArray` macro automatically generates code that allows to replace Vec<`Atom`>
/// with a struct of arrays. It will generate the `AtomVec` that looks like:
///
/// ```rust
/// pub struct AtomVec {
///     pub name: Vec<&'static str>,
///     pub number: Vec<u8>,
///     pub kind: Vec<Element>,
///     pub hubbard: Vec<f64>,
///     pub valorbs: Vec<Vec<AtomicOrbital>>,
///     pub n_orbs: Vec<usize>,
///     pub valorbs_occupation: Vec<Vec<f64>>,
///     pub n_elec: Vec<usize>,
/// }
/// ```
/// It will also generate the same functions that a `Vec<Atom>` would have, and a few helper structs:
/// `AtomSlice`, `AtomSliceMut`, `AtomRef` and `AtomRefMut` corresponding respectively
/// to `&[Atom]`, `&mut [Atom]`, `&Atom` and `&mut Atom`.
#[derive(Clone, Debug)]
pub struct Atom {
    /// Name of the chemical element
    pub name: &'static str,
    /// Ordinary number of the element
    pub number: u8,
    /// Element as an enum
    pub kind: Element,
    /// Hubbard parameter
    pub hubbard: f64,
    /// Vector of the valence orbitals for this atom
    pub valorbs: Vec<AtomicOrbital>,
    /// Number of valence orbitals. This is the length of valorbs
    pub n_orbs: usize,
    /// Occupation number for each valence orbitals
    pub valorbs_occupation: Vec<f64>,
    /// Number of valence electrons
    pub n_elec: usize,
    /// Position of the atom in bohr
    pub xyz: Vector3<f64>,
}

impl From<Element> for Atom {
    /// Create a new [Atom] from the chemical [Element](crate::initialization::elements::Element).
    /// The parameterization from the parameter files is loaded and the Hubbard parameter
    /// and the valence orbitals are stored in this type.
    fn from(element: Element) -> Self {
        let symbol: &'static str = element.symbol();
        let confined_atom: PseudoAtom = PseudoAtom::confined_atom(symbol);
        let free_atom: PseudoAtom = PseudoAtom::free_atom(symbol);
        let mut valorbs: Vec<AtomicOrbital> = Vec::new();
        let mut occupation: Vec<f64> = Vec::new();
        let mut n_elec: usize = 0;
        for (i, j) in confined_atom
            .valence_orbitals
            .iter()
            .zip(free_atom.valence_orbitals.iter())
        {
            let n: i8 = confined_atom.nshell[*i as usize];
            let l: i8 = confined_atom.angular_momenta[*i as usize];
            let energy: f64 = free_atom.energies[*j as usize];
            for m in l.neg()..(l + 1) {
                valorbs.push(AtomicOrbital::from(((n - 1, l, m), energy)));
                occupation.push(
                    confined_atom.orbital_occupation[*i as usize] as f64 / (2 * l + 1) as f64,
                );
            }
            n_elec += confined_atom.orbital_occupation[*i as usize] as usize;
        }
        let n_orbs: usize = valorbs.len();

        Atom {
            name: symbol,
            number: element.number(),
            kind: element,
            hubbard: confined_atom.hubbard_u,
            valorbs: valorbs,
            n_orbs: n_orbs,
            valorbs_occupation: occupation,
            n_elec: n_elec,
            xyz: Vector3::<f64>::zeros(),
        }
    }
}

impl From<(Element, &SkfHandler)> for Atom {
    /// Create a new [Atom] from the chemical [Element](crate::initialization::elements::Element) and
    /// the [SkfHandler](crate::initialization::parameters::SkfHandler).
    /// The parameterization from the parameter files is loaded and the Hubbard parameter
    /// and the valence orbitals are stored in this type.
    fn from(tuple: (Element, &SkfHandler)) -> Self {
        let element: Element = tuple.0;
        let symbol: &'static str = element.symbol();
        let pseudo_atom: PseudoAtomSkf = PseudoAtomSkf::from(tuple.1);
        let mut valorbs: Vec<AtomicOrbital> = Vec::new();
        let mut occupation: Vec<f64> = Vec::new();
        let mut n_elec: usize = 0;
        for i in pseudo_atom.valence_orbitals.iter() {
            let n: i8 = pseudo_atom.nshell[*i as usize];
            let l: i8 = pseudo_atom.angular_momenta[*i as usize];
            let energy: f64 = pseudo_atom.energies[*i as usize];
            for m in l.neg()..(l + 1) {
                valorbs.push(AtomicOrbital::from(((n - 1, l, m), energy)));
                occupation
                    .push(pseudo_atom.orbital_occupation[*i as usize] as f64 / (2 * l + 1) as f64);
            }
            n_elec += pseudo_atom.orbital_occupation[*i as usize] as usize;
        }
        let n_orbs: usize = valorbs.len();

        Atom {
            name: symbol,
            number: element.number(),
            kind: element,
            hubbard: pseudo_atom.hubbard_u,
            valorbs: valorbs,
            n_orbs: n_orbs,
            valorbs_occupation: occupation,
            n_elec: n_elec,
            xyz: Vector3::<f64>::zeros(),
        }
    }
}

impl Atom {
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

    pub fn sort_indices_atomic_orbitals(&self) -> Vec<usize> {
        let indices: Array1<usize> = self.valorbs.iter().map(|orb| orb.ord_idx()).collect();
        argsort_usize(indices.view())
    }
}

impl From<&str> for Atom {
    /// Create a new [Atom] from the atomic symbol (case insensitive). The parameterization from the
    /// parameter files is loaded and the Hubbard parameter and the valence orbitals are stored in
    /// this type.
    fn from(symbol: &str) -> Self {
        Self::from(Element::from(symbol))
    }
}

impl From<&Atom> for Atom {
    // Create a new [Atom] from a reference to an [Atom].
    fn from(atom: &Atom) -> Self {
        atom.clone()
    }
}

impl From<u8> for Atom {
    /// Create a new [Atom] from the atomic number. The parameterization from the
    /// parameter files is loaded and the Hubbard parameter and the valence orbitals are stored in
    /// this type.
    fn from(number: u8) -> Self {
        Self::from(Element::from(number))
    }
}
impl PartialEq for Atom {
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
    }
}

impl Sub for Atom {
    type Output = Vector3<f64>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.xyz - rhs.xyz
    }
}

impl Sub for &Atom {
    type Output = Vector3<f64>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.xyz - rhs.xyz
    }
}

impl Eq for Atom {}

impl PartialOrd for Atom {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.number.partial_cmp(&other.number)
    }
}

/// Type that specifies an atomic orbital by its three quantum numbers and holds its energy
#[derive(Copy, Clone, Debug)]
pub struct AtomicOrbital {
    pub n: i8,
    pub l: i8,
    pub m: i8,
    pub energy: f64,
}

impl From<(i8, i8, i8)> for AtomicOrbital {
    fn from(qnumber: (i8, i8, i8)) -> Self {
        Self {
            n: qnumber.0,
            l: qnumber.1,
            m: qnumber.2,
            energy: 0.0,
        }
    }
}

impl From<((i8, i8, i8), f64)> for AtomicOrbital {
    fn from(numbers_energy: ((i8, i8, i8), f64)) -> Self {
        Self {
            n: numbers_energy.0 .0,
            l: numbers_energy.0 .1,
            m: numbers_energy.0 .2,
            energy: numbers_energy.1,
        }
    }
}

impl PartialOrd for AtomicOrbital {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.ord_idx().partial_cmp(&other.ord_idx())
    }
}

impl AtomicOrbital {
    /// Compute an index, that orders the AtomicOrbital in Cartesian order.
    /// 1 s  => 100 |
    /// 2 s  => 200 | 2 px  => 210 | 2 py  => 211 | 2  pz => 212 |
    /// 3 s  => 300 | 3 px  => 310 | 3 py  => 311 | 3  pz => 312 |
    ///             | 3 dz2 => 320 | 3 dzx => 321 | 3 dyz => 322 | 3 dx2y2 => 323 | 3 dxy => 324
    fn ord_idx(&self) -> usize {
        let mut index: usize = (self.n * 100) as usize;
        index += (self.l * 10) as usize;
        index += match self.l {
            0 => 0, // s-orbitals
            1 => {
                // p-orbitals
                match self.m {
                    1 => 0,  // px
                    -1 => 1, // py
                    0 => 2,  // pz
                    _ => 3,
                }
            }
            2 => {
                // d-orbitals
                match self.m {
                    0 => 0,  // dz2
                    1 => 1,  // dzx
                    -1 => 2, // dyz
                    -2 => 3, // dx2y2
                    2 => 4,  // dxy
                    _ => 5,
                }
            }
            _ => 6,
        };
        index
    }
}

impl PartialEq for AtomicOrbital {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.m == other.m && self.l == other.l
    }
}

impl Eq for AtomicOrbital {}
