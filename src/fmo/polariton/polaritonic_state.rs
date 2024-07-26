use crate::io::settings::PolaritonConfig;
use crate::utils::array_helper::argsort_abs;
use crate::{constants::HARTREE_TO_EV, fmo::BasisState};
use nalgebra::Vector3;
use ndarray::prelude::*;
use ndarray_npy::{write_npy, WriteNpyError};
use num_traits::Zero;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::Write;

/// Structure that contains all necessary information to specify the excited states in
/// the LCMO-FMO framework.
pub struct ExcitonPolaritonStates<'a> {
    /// Total energy of the electronic ground state.
    pub total_energy: f64,
    /// Excitation energies.
    pub energies: Array1<f64>,
    /// Eigenvectors.
    pub coefficients: Array2<f64>,
    /// Exciton basis states.
    pub basis: Vec<BasisState<'a>>,
    /// Oscillator strengths.
    pub f: Array1<f64>,
    /// Transition Dipole moments.
    pub tr_dip: Vec<Vector3<f64>>,
    /// The concatenated MO coefficients of the monomers
    pub orbs: Array2<f64>,
    /// (# occupied orbitals, # virtual orbitals)
    pub dim: (usize, usize),
    pub config: PolaritonConfig,
}

impl<'a> ExcitonPolaritonStates<'a> {
    /// Create a type that contains all necessary information about all LCMO exciton states.
    pub fn new(
        e_tot: f64,
        eig: (Array1<f64>, Array2<f64>),
        basis: Vec<BasisState<'a>>,
        dim: (usize, usize),
        orbs: Array2<f64>,
        config: PolaritonConfig,
    ) -> Self {
        // The transition dipole moments and oscillator strengths need to be computed.
        let mut f: Array1<f64> = Array1::zeros([eig.0.len()]);
        let mut transition_dipoles: Vec<Vector3<f64>> = Vec::with_capacity(eig.0.len());

        // Iterate over all exciton states.
        for (mut fi, (e, vs)) in f.iter_mut().zip(eig.0.iter().zip(eig.1.axis_iter(Axis(1)))) {
            // Initialize the transition dipole moment for the current state.
            let mut tr_dip: Vector3<f64> = Vector3::zero();

            // And all basis states to compute the transition dipole moment. The transition dipole
            // moment of the CT states is assumed to be zero. This is a rather hard approximation
            // and could be easily improved. TODO
            for (idx, v) in vs
                .slice(s![..(vs.len() - config.photon_energy.len())])
                .iter()
                .enumerate()
            {
                match basis.get(idx).unwrap() {
                    BasisState::LE(state) => {
                        tr_dip += state.tr_dipole.scale(*v as f64);
                    }
                    BasisState::PairCT(state) => {
                        tr_dip += state.tr_dipole.scale(*v as f64);
                    }
                    _ => {}
                }
            }
            let tr_dip = tr_dip.map(|val| val as f64);
            let count: usize = vs.len() - config.photon_energy.len();
            for idx in 0..config.photon_energy.len() {
                *fi += 2.0 / 3.0 * e * vs[count + idx].powi(2); // * tr_dip.dot(&tr_dip);
            }
            transition_dipoles.push(tr_dip);
        }

        Self {
            total_energy: e_tot,
            energies: eig.0,
            coefficients: eig.1,
            basis,
            f,
            tr_dip: transition_dipoles,
            orbs: orbs,
            dim: dim,
            config,
        }
    }

    /// Write the excitation energies (in eV) and oscillator strength to a .npy file
    pub fn spectrum_to_npy(&self, filename: &str) -> Result<(), WriteNpyError> {
        // Stack the energies and osc. strengths into a 2D Array (columnwise).
        let mut data: Array2<f64> = Array2::zeros([self.f.len(), 0]);

        let energies_ev: Array1<f64> = HARTREE_TO_EV as f64 * &self.energies;
        // Convert the excitation energy in eV.
        data.push(Axis(1), energies_ev.view());
        data.push(Axis(1), self.f.view());

        // Write the npy file.
        write_npy(filename, &data)
    }

    /// Write the excitation energies (in eV) and the oscillator strength to a text file.
    pub fn spectrum_to_txt(&self, filename: &str) {
        // Two lines of header.
        let mut txt: String = "# Absorption spectrum\n".to_owned();
        txt += "# exc. energy / eV    osc. strength\n";

        // Each energy and oscillator strength is written together to a line.
        for (e, f) in self.energies.iter().zip(self.f.view().iter()) {
            txt += &format!("{:16.14}     {:16.14}\n", e * HARTREE_TO_EV as f64, f);
        }

        // Try to create the output file.
        let mut f =
            File::create(filename).expect(&*format!("Unable to create file: {}", &filename));

        // and write the data.
        f.write_all(txt.as_bytes())
            .expect(&format!("Unable to write data at: {}", &filename));
    }
}

impl Display for ExcitonPolaritonStates<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let threshold = 0.1;
        // Empty line.
        let mut txt: String = format!("{:^80}\n", "");

        // Header for the section.
        txt += &format!("{: ^80}\n", "FMO-LCMO Polariton Excitation Energies");

        // Horizontal rule as delimiter.
        txt += &format!("{:-^75}\n", "");

        // Create the output for each exciton state.
        for (n, (e, v)) in self
            .energies
            .iter()
            .zip(self.coefficients.axis_iter(Axis(1)))
            .enumerate()
        {
            // Absolute energy of each excited state.
            let abs_energy: f64 = self.total_energy + e;

            // Relative excitation energy in eV.
            let rel_energy_ev: f64 = e * HARTREE_TO_EV as f64;

            // The transition dipole moment of the current state.
            let tr_dip: Vector3<f64> = self.tr_dip[n];

            txt += &format!(
                "Excited state {: >5}: Excitation energy = {:>8.6} eV\n",
                n + 1,
                rel_energy_ev
            );
            txt += &format!(
                "Total energy for state {: >5}: {:22.12} Hartree\n",
                n + 1,
                abs_energy
            );
            txt += &format!("  Multiplicity: Singlet\n");
            txt += &format!(
                "  Trans. Mom. (a.u.): {:10.6} X  {:10.6} Y  {:10.6} Z\n",
                tr_dip.x, tr_dip.y, tr_dip.z
            );
            txt += &format!("  Oscillator Strength:  {:10.8}\n", self.f[n]);

            // Sort the indices by coefficients of the current eigenvector.
            let sorted_indices: Vec<usize> = argsort_abs(v.view());

            // Reverse the Iterator to write the largest amplitude first.
            for i in sorted_indices.into_iter().rev() {
                // Amplitude of the current transition.
                let c: f64 = v[i].abs();

                // Only write transition with an amplitude higher than a certain threshold.
                if c > threshold {
                    if i >= (v.len() - self.config.photon_energy.len()) {
                        txt += &format!(
                            "  Photon Mode {: >5} Amplitude: {:6.4} => {:>4.1} %\n",
                            i - (v.len() - self.config.photon_energy.len()),
                            c,
                            c.powi(2) * 1e2
                        );
                    } else {
                        txt += &format!(
                            "  {:28} Amplitude: {:6.4} => {:>4.1} %\n",
                            format!("{}", self.basis.get(i).unwrap()),
                            c,
                            c.powi(2) * 1e2
                        );
                    }
                }
            }

            // Add an empty line after each excited state.
            if n < self.energies.len() - 1 {
                txt += &format!("{: ^80}\n", "");
                // In the last iteration a short horizontal rule is added.
            } else {
                txt += &format!("{:-^62}\n", "");
            }
        }
        // Information at the end about the threshold.
        txt += &format!(
            "All transition with amplitudes > {:10.8} were printed.\n",
            threshold
        );

        // Horizontal rule as delimiter.
        txt += &format!("{:-^75} \n", "");

        write!(f, "{}", txt)
    }
}
