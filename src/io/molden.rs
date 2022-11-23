use crate::initialization::Atom;
use crate::io::basis_set::BasisSet;
use chrono::{DateTime, Utc};
use derive_builder::export::core::fs::File;
use derive_builder::*;
use ndarray::prelude::*;
use std::fmt::{Display, Formatter, Result};
use std::io::Write;
use std::path::Path;

/// Export to Molden format to visualize orbitals.
/// To this end we convert Kohn-Sham atomic orbitals from DFTB calculation into STO-3G basis. Note
/// that the plotted orbitals are only a rough approximation as we replace exact atomic orbitals
/// by a minimal contraction of Gaussians.
/// An instance of the `MoldenExporter` can be created by calling the associated Builder type.
/// An example is shown below:
/// ```
/// MoldenExporterBuilder::default()
///       .atoms(&system.atoms)
///       .orbs(system.properties.orbs().unwrap())
///       .orbe(system.properties.orbe().unwrap())
///       .f(system.properties.occupation().unwrap().to_vec())
///       .build()
///       .unwrap();
/// ```
#[derive(Builder)]
pub struct MoldenExporter<'a> {
    atoms: &'a [Atom],
    #[builder(setter(custom))]
    orbs: Array2<f64>, // MO coefficients, but reorderd in cartesian order.
    orbe: ArrayView1<'a, f64>,
    f: Vec<f64>,
    #[builder(default)]
    basis: BasisSet,
    #[builder(default, setter(strip_option))]
    n_occ: Option<usize>,
    #[builder(default, setter(strip_option))]
    n_virt: Option<usize>,
    #[builder(default = "self.default_title()")]
    title: String,
    // TODO: Add Frequencies, SCF, and so on
}

impl MoldenExporterBuilder<'_> {
    /// Custom implementation for the orbitals, since we only need an ArrayView and have to
    /// order the orbitals. It is necessary, that the atoms are set before the orbitals are set.
    pub fn orbs(&mut self, orbs: ArrayView2<f64>) -> &mut Self {
        let mut new = self;
        let ordered_orbs: Array2<f64> = reorder_orbitals(orbs, new.atoms.as_ref().unwrap());
        new.orbs = Some(ordered_orbs);
        new
    }

    fn default_title(&self) -> String {
        let now: DateTime<Utc> = Utc::now();
        format!("file created at: {}", now.to_rfc2822())
    }
}

impl MoldenExporter<'_> {
    /// Returns a String representation of the atomic coordinates.
    fn repr_atoms(&self) -> String {
        let mut txt: String = "[Atoms] AU\n".to_owned();
        for (i, atom) in self.atoms.iter().enumerate() {
            txt += &format!(
                "{} {} {} {:2.7} {:2.7} {:2.7}\n",
                atom.kind.symbol(),
                i + 1,
                atom.kind.number(),
                atom.xyz.x,
                atom.xyz.y,
                atom.xyz.z
            );
        }
        txt
    }

    /// Return all occupied and virtual MOs and restrict the active space (if requested).
    fn occ_and_virts(&self) -> (Vec<usize>, Vec<usize>) {
        // Indices of the occupied and virtual orbitals.
        let occs: Vec<usize> = self
            .f
            .iter()
            .enumerate()
            .filter_map(|(i, x)| if *x > 0.0 { Some(i) } else { None })
            .collect();
        let virts: Vec<usize> = self
            .f
            .iter()
            .enumerate()
            .filter_map(|(i, x)| if *x > 0.0 { None } else { Some(i) })
            .collect();
        // If requested, the active space is restricted.
        let start: usize = occs.len() - self.n_occ.unwrap_or(occs.len());
        let end: usize = self.n_virt.unwrap_or(virts.len());

        (occs[start..].to_vec(), virts[..end].to_vec())
    }

    fn repr_mos(&self) -> String {
        let mut txt: String = "[5D]\n[MO]\n".to_owned();
        let (occs, virts): (Vec<usize>, Vec<usize>) = self.occ_and_virts();

        for (i, idx) in occs.iter().chain(virts.iter()).enumerate() {
            txt += &format!(" Sym=  {}a\n", i + 1);
            txt += &format!(" Ene=  {:2.7}\n", self.orbe[i]);
            txt += " Spin= Alpha\n";
            txt += &format!(" Occup={:2.7}\n", self.f[i]);
            for (j, aoj) in self.orbs.slice(s![.., i]).iter().enumerate() {
                txt += &format!("    {} {:2.7}\n", j + 1, aoj);
            }
        }
        txt
    }

    fn repr_gtos(&self) -> String {
        let mut txt = "[GTO]\n".to_owned();
        for (i, atom) in self.atoms.iter().enumerate() {
            txt += &format!("{} 0\n", i + 1);
            txt += &self.basis.repr_basis_set(atom.kind);
            txt += "\n";
        }
        txt
    }

    pub fn write_to(&self, path: &Path) -> () {
        let filename: String = path.to_str().unwrap().to_owned();
        let mut f = File::create(path).expect(&*format!("Unable to create file: {}", &filename));
        f.write_all(format!("{}", self).as_bytes())
            .expect(&format!("Unable to write data at: {}", &filename));
    }
}

impl Display for MoldenExporter<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut txt: String = "[Molden Format]\n".to_owned();
        txt += "[Title]\n";
        txt += &format!("{}\n", self.title);
        txt += &self.repr_atoms();
        txt += &self.repr_gtos();
        txt += &self.repr_mos();
        // Write strictly the first element into the supplied output
        // stream: `f`. Returns `fmt::Result` which indicates whether the
        // operation succeeded or failed. Note that `write!` uses syntax which
        // is very similar to `println!`.
        write!(f, "{}", txt)
    }
}

/// Orbitals in one l-shell are ordered by their m-values, -l,...,0,...,l
/// but Molden expects orbitals to be in Cartesian order.
/// The reordering rules are:
///         s                 ->       s
///     py, pz, px            ->   px, py, pz
///     dxy,dyz,dz2,dzx,dx2y2 -> dz2,dzx,dyz,dx2y2,dxy
/// The ordering of the orbitals is primarily done in the impl of `AtomicOrbital`.
fn reorder_orbitals(orbs: ArrayView2<f64>, atoms: &[Atom]) -> Array2<f64> {
    // Empty Vec that holds the indices to sort the MO coefficients.
    let mut indices: Vec<usize> = Vec::with_capacity(orbs.nrows());
    let mut mu: usize = 0;
    for atom in atoms.iter() {
        // Get the indices that sort the orbitals that belong to the current atom.
        let at_indices: Vec<usize> = atom.sort_indices_atomic_orbitals();
        indices.extend(at_indices.iter().map(|x| x + mu));
        mu += atom.n_orbs;
    }
    let mut ordered_orbs: Array2<f64> = Array2::zeros([0, orbs.ncols()]);
    for idx in indices.into_iter() {
        ordered_orbs.push(Axis(0), orbs.row(idx));
    }
    ordered_orbs
}
