use crate::excited_states::ProductCache;
use crate::fmo::old_supersystem::OldSupersystem;
use crate::fmo::{PairType, ReducedBasisState};
use crate::initialization::old_system::OldSystem;
use crate::initialization::Atom;
use crate::scc::mixer::{AndersonAccel, BroydenMixer};
use enum_as_inner::EnumAsInner;
use hashbrown::HashMap;
use ndarray::prelude::*;

/// A `Property` is a piece of data that can be associated with an `Molecule` or
/// `ElectronicData`. The idea of this enum is taken from Guillaume Fraux's (@Luthaf) Chemfiles
/// library.
/// The functionality of the `Property` enum is expanded by the use of the `EnumAsInner` macro.
/// This allows to get direct access to the inner values of the enum without doing
/// case matching. As an example the inner fields can be accessed by using the methods `into_$name()`
/// or `as_$name()`. e.g: (see [Documentation of enum-as-inner](https://docs.rs/enum-as-inner/0.3.3/enum_as_inner/)
/// for details).
/// ## Basic example for Bool and Array1
///
///  ```rust
///  let flag: bool = true;
///  let prop1: Property = Property::from(flag);
///  assert_eq!(prop1.as_bool().unwrap(), &true);
///  assert_eq!(prop1.into_bool().unwrap(), true);
///
///  let vector: Array1<f64> = Array1::zeros([4]);
///  let prop2: Property = Property::from(vector);
///  assert_eq!(prop2.as_array1.unwrap(), &Array1::zeros([4]));
///  assert_eq!(prop2.into_array1.unwrap(), Array::zeros([4]));
///  ```
///
///
#[derive(Debug, Clone, EnumAsInner)]
pub enum Property {
    /// Boolean property
    Bool(bool),
    /// Integer property
    Usize(usize),
    /// Floating point property
    Double(f64),
    /// String property
    String(String),
    /// HashMap for types of pairs.
    PairMap(HashMap<(usize, usize), PairType>),
    /// HashMap for indices of pairs.
    PairIndexMap(HashMap<(usize, usize), usize>),
    /// Vector property of u8 type
    VecU8(Vec<u8>),
    /// Vector property of usize type
    VecUsize(Vec<usize>),
    /// Vector property of f64 type
    VecF64(Vec<f64>),
    VecAtom(Vec<Atom>),
    /// Arraybase<f64, Ix1> property
    Array1(Array1<f64>),
    /// Arraybase<f64, Ix2> property
    Array2(Array2<f64>),
    /// Arraybase<f64, Ix3> property
    Array3(Array3<f64>),
    /// Arraybase<bool, Ix2> property
    Array2Bool(Array2<bool>),
    /// SCC Mixer property
    Mixer(BroydenMixer),
    /// SCC Mixer property
    Accel(AndersonAccel),
    /// Excited state product cache
    Cache(ProductCache),
    SuperSystem(OldSupersystem),
    OldSystem(OldSystem),
    // Basis States
    VecBasis(Vec<ReducedBasisState>),
}

impl Default for Property {
    fn default() -> Self {
        Property::Bool(false)
    }
}

impl From<bool> for Property {
    fn from(value: bool) -> Self {
        Property::Bool(value)
    }
}

impl From<usize> for Property {
    fn from(value: usize) -> Self {
        Property::Usize(value)
    }
}

impl From<f64> for Property {
    fn from(value: f64) -> Self {
        Property::Double(value)
    }
}

impl From<String> for Property {
    fn from(value: String) -> Self {
        Property::String(value)
    }
}

impl From<HashMap<(usize, usize), PairType>> for Property {
    fn from(value: HashMap<(usize, usize), PairType>) -> Self {
        Property::PairMap(value)
    }
}

impl From<HashMap<(usize, usize), usize>> for Property {
    fn from(value: HashMap<(usize, usize), usize>) -> Self {
        Property::PairIndexMap(value)
    }
}

impl From<Vec<u8>> for Property {
    fn from(value: Vec<u8>) -> Self {
        Property::VecU8(value)
    }
}
impl From<Vec<usize>> for Property {
    fn from(value: Vec<usize>) -> Self {
        Property::VecUsize(value)
    }
}

impl From<Vec<f64>> for Property {
    fn from(value: Vec<f64>) -> Self {
        Property::VecF64(value)
    }
}

impl From<Vec<ReducedBasisState>> for Property {
    fn from(value: Vec<ReducedBasisState>) -> Self {
        Property::VecBasis(value)
    }
}

impl From<&'_ str> for Property {
    fn from(value: &'_ str) -> Self {
        Property::String(value.into())
    }
}

impl From<Array1<f64>> for Property {
    fn from(value: Array1<f64>) -> Self {
        Property::Array1(value)
    }
}

impl From<ArrayView1<'_, f64>> for Property {
    fn from(value: ArrayView1<'_, f64>) -> Self {
        Property::Array1(value.to_owned())
    }
}

impl From<Array2<f64>> for Property {
    fn from(value: Array2<f64>) -> Self {
        Property::Array2(value)
    }
}

impl From<ArrayView2<'_, f64>> for Property {
    fn from(value: ArrayView2<'_, f64>) -> Self {
        Property::Array2(value.to_owned())
    }
}

impl From<Array3<f64>> for Property {
    fn from(value: Array3<f64>) -> Self {
        Property::Array3(value)
    }
}

impl From<ArrayView3<'_, f64>> for Property {
    fn from(value: ArrayView3<'_, f64>) -> Self {
        Property::Array3(value.to_owned())
    }
}

impl From<Array2<bool>> for Property {
    fn from(value: Array2<bool>) -> Self {
        Property::Array2Bool(value)
    }
}

impl From<ArrayView2<'_, bool>> for Property {
    fn from(value: ArrayView2<'_, bool>) -> Self {
        Property::Array2Bool(value.to_owned())
    }
}

impl From<BroydenMixer> for Property {
    fn from(value: BroydenMixer) -> Self {
        Property::Mixer(value)
    }
}

impl From<AndersonAccel> for Property {
    fn from(value: AndersonAccel) -> Self {
        Property::Accel(value)
    }
}

impl From<ProductCache> for Property {
    fn from(value: ProductCache) -> Self {
        Property::Cache(value)
    }
}

impl From<OldSupersystem> for Property {
    fn from(value: OldSupersystem) -> Self {
        Property::SuperSystem(value)
    }
}

impl From<OldSystem> for Property {
    fn from(value: OldSystem) -> Self {
        Property::OldSystem(value)
    }
}
