use crate::constants::HARTREE_TO_EV;
use crate::excited_states::ExcitedState;
use ndarray::prelude::*;
use std::fmt::{Display, Formatter};

/// Structure that contains all necessary information
pub struct ExcitedStates {
    /// Total energy of the electronic ground state.
    pub total_energy: f64,
    /// Excitation energies.
    pub energies: Array1<f64>,
    /// Transition density matrices in MO basis. [nocc, nvirt, state]
    pub tdm: Array3<f64>,
    /// Oscillator strengths.
    pub f: Array1<f64>,
    /// Transition Dipole moments.
    pub tr_dip: Array2<f64>,
    /// View on the MO coefficients.
    pub orbs: Array2<f64>,
}

impl ExcitedState for ExcitedStates {
    fn get_lumo(&self) -> usize {
        self.tdm.dim().0
    }

    fn get_mo_coefficients(&self) -> ArrayView2<f64> {
        self.orbs.view()
    }

    fn get_transition_density_matrix(&self, state: usize) -> Array2<f64> {
        self.tdm.slice(s![.., .., state]).to_owned()
    }

    fn get_energies(&self) -> ArrayView1<f64> {
        self.energies.view()
    }

    fn get_oscillator_strengths(&self) -> ArrayView1<f64> {
        self.f.view()
    }

    fn get_num_states(&self) -> usize {
        self.f.len()
    }
}

impl Display for ExcitedStates {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let threshold: f64 = 0.1;
        // Empty line for the new block.
        let mut txt: String = format!("{:^80}\n", "");

        // Header.
        txt += &format!("{: ^80}\n", "Excitation Energies");

        // Horizontal rule.
        txt += &format!("{:-^75}\n", "");

        // Iterate over all excited states.
        for n in 0..self.energies.len() {
            // Absolute energy of the current excited state.
            let abs_energy: f64 = self.total_energy + self.energies[n];

            // Relative excitation energy in eV.
            let rel_energy_ev: f64 = self.energies[n] * HARTREE_TO_EV;

            // Transition dipole moment.
            let tr_dip: ArrayView1<f64> = self.tr_dip.column(n);

            // Transition density matrix in MO basis.
            let tdm: ArrayView2<f64> = self.tdm.slice(s![.., .., n]);

            // Write the index and relative energy of the state.
            txt += &format!(
                "Excited state {: >5}: Excitation energy = {:>8.6} eV\n",
                n + 1,
                rel_energy_ev
            );

            // and the absolute energy.
            txt += &format!(
                "Total energy for state {: >5}: {:22.12} Hartree\n",
                n + 1,
                abs_energy
            );

            // Multiplicity is at the moment always Singlet. This will be advanced in the future.
            txt += &format!("  Multiplicity: Singlet\n");

            // Write the transition dipole moment and the oscillator strength.
            txt += &format!(
                "  Trans. Mom. (a.u.): {:10.6} X  {:10.6} Y  {:10.6} Z\n",
                tr_dip[0], tr_dip[1], tr_dip[2]
            );
            txt += &format!("  Oscillator Strength:  {:12.8}\n", self.f[n]);

            // All orbital transition which coefficients is higher than a threshold are printed.
            for (h, row) in tdm.axis_iter(Axis(0)).rev().enumerate() {
                let occ_label: String =
                    if h == 0 {
                        format!("H")
                    } else {
                        format!("H-{}", h)
                    };
                for (l, value) in row.iter().enumerate() {
                    let virt_label: String = if l == 0 {
                        format!("L")
                    } else {
                        format!("L+{}", l)
                    };
                    if value.abs() > threshold {
                        txt += &format!(
                            "  {: <4} --> {: <4}  Amplitude: {:6.4} => {:>4.1} %\n",
                            occ_label,
                            virt_label,
                            // value.abs(),
                            value,
                            value.powi(2) * 1e2
                        );
                    }
                }
            }

            // If it is not the last state, an empty line is written.
            if n + 1 < self.energies.len() {
                txt += &format!("{: ^80}\n", "");
            }
        }

        // Print the threshold that was used.
        txt += &format!(
            "All transition with amplitudes > {:10.8} were printed.\n",
            threshold
        );
        // and a horizontal rule.
        txt += &format!("{:-^75}\n", "");
        write!(f, "{}", txt)
    }
}
