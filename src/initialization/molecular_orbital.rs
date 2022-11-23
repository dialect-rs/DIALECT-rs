use ndarray::ArrayView1;

/// Type that defines a Molecular Orbital.
#[derive(Debug, Copy, Clone)]
pub struct MO<'a> {
    /// MO coefficients.
    pub c: ArrayView1<'a, f64>,
    /// MO energy.
    pub e: f64,
    /// Index of the MO.
    pub idx: usize,
    /// MO occupation.
    pub f: Occupation,
}

impl<'a> MO<'a> {
    /// A new `MO` instance is created from the MO coefficients, `c`, its energy, `e`,
    /// its index, `idx` and its occupation number `f`.
    pub fn new(c: ArrayView1<'a, f64>, e: f64, idx: usize, f: f64) -> Self {
        Self {
            c,
            e,
            idx,
            f: Occupation::from(f),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Occupation {
    Occupied,
    Virtual,
    Floating(f64),
}

impl From<f64> for Occupation {
    fn from(occ: f64) -> Self {
        match occ {
            f if (f - 2.0).abs() < f64::EPSILON => Self::Occupied,
            f if f.abs() < f64::EPSILON => Self::Virtual,
            f => Self::Floating(f),
        }
    }
}
