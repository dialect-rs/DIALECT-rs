pub mod basis;
mod hamiltonian;
mod integrals;
mod lcmo_trans_charges;
use crate::constants::HARTREE_TO_EV;
use crate::excited_states::ntos::get_nto_singular_values;
use crate::excited_states::ExcitedState;
use crate::initialization::Atom;
use crate::utils::array_helper::argsort_abs;
pub use basis::*;
use nalgebra::Vector3;
use ndarray::prelude::*;
use ndarray_npy::write_npy;
use num_traits::Zero;
use rayon::prelude::*;
use std::fmt::{Display, Formatter};
use std::ops::AddAssign;

/// Structure that contains all necessary information to specify the excited states in
/// the LCMO-FMO framework.
pub struct ExcitonStates<'a> {
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
}

impl ExcitedState for ExcitonStates<'_> {
    fn get_lumo(&self) -> usize {
        self.dim.0
    }

    fn get_mo_coefficients(&self) -> ArrayView2<f64> {
        self.orbs.view()
    }

    fn get_transition_density_matrix(&self, state: usize) -> Array2<f64> {
        // Initialization of the transition density matrix.
        let mut tdm: Array2<f64> = Array2::zeros(self.dim);
        let threshold = 1e-4;
        for (state, c) in self
            .basis
            .iter()
            .zip(self.coefficients.column(state).iter())
        {
            if c.abs() > threshold {
                match state {
                    BasisState::LE(state) => {
                        // TDM of monomer * c
                        let occs = state.monomer.slice.occ_orb;
                        let virts = state.monomer.slice.virt_orb;
                        let n_occ: usize = state.monomer.properties.n_occ().unwrap();
                        let n_virt: usize = state.monomer.properties.n_virt().unwrap();
                        // tdm.slice_mut(s![occs, virts]).add_assign(&(*c * &state.tdm.into_shape((n_occ, n_virt)).unwrap()));

                        let state_tdm = state
                            .tdm
                            .as_standard_layout()
                            .to_owned()
                            .into_shape((n_occ, n_virt))
                            .unwrap();
                        tdm.slice_mut(s![occs, virts])
                            .add_assign(&(*c * &state_tdm));
                    }
                    BasisState::PairCT(state) => {
                        let occs = state.occ_orb;
                        let virts = state.virt_orb;

                        tdm.slice_mut(s![occs, virts])
                            .add_assign(&(*c * &state.eigenvectors));
                    }
                }
            }
        }

        tdm
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

impl<'a> ExcitonStates<'a> {
    /// Create a type that contains all necessary information about all LCMO exciton states.
    pub fn new(
        e_tot: f64,
        eig: (Array1<f64>, Array2<f64>),
        basis: Vec<BasisState<'a>>,
        dim: (usize, usize),
        orbs: Array2<f64>,
        _overlap: ArrayView2<f64>,
        _atoms: &[Atom],
    ) -> Self {
        // The transition dipole moments and oscillator strengths need to be computed.
        let mut f: Array1<f64> = Array1::zeros([eig.0.len()]);
        let mut transition_dipoles: Vec<Vector3<f64>> = Vec::with_capacity(eig.0.len());

        // Iterate over all exciton states.
        for (fi, (e, vs)) in f.iter_mut().zip(eig.0.iter().zip(eig.1.axis_iter(Axis(1)))) {
            // Initialize the transition dipole moment for the current state.
            let mut tr_dip: Vector3<f64> = Vector3::zero();

            // And all basis states to compute the transition dipole moment. The transition dipole
            // moment of the CT states is assumed to be zero. This is a rather hard approximation
            // and could be easily improved. TODO
            for (idx, v) in vs.iter().enumerate() {
                match basis.get(idx).unwrap() {
                    BasisState::LE(state) => {
                        tr_dip += state.tr_dipole.scale(*v);
                    }
                    BasisState::PairCT(state) => {
                        tr_dip += state.tr_dipole.scale(*v);
                    }
                }
            }
            *fi = 2.0 / 3.0 * e * tr_dip.dot(&tr_dip);
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
        }
    }

    pub fn print_state_contributions(&self, state: usize) {
        let threshold = 0.05;
        let mut txt: String = format!("{:^80}\n", "");

        // Header for the section.
        txt += &format!("{: ^80}\n", "FMO-LCMO Excitation Energy");
        // Horizontal rule as delimiter.
        txt += &format!("{:-^75}\n", "");

        let n: usize = state;
        let e: f64 = self.energies[state];
        let v = self.coefficients.slice(s![.., state]);

        // Absolute energy of each excited state.
        let abs_energy: f64 = self.total_energy + e;

        // Relative excitation energy in eV.
        let rel_energy_ev: f64 = e * HARTREE_TO_EV;

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
                txt += &format!(
                    "  {:28} Amplitude: {:6.4} => {:>4.1} %\n",
                    format!("{}", self.basis.get(i).unwrap()),
                    c,
                    c.powi(2) * 1e2
                );
            }
        }

        // Information at the end about the threshold.
        txt += &format!(
            "All transition with amplitudes > {:10.8} were printed.\n",
            threshold
        );

        // Horizontal rule as delimiter.
        txt += &format!("{:-^75} \n", "");

        println!("{}", txt);
    }

    pub fn calculate_exciton_parcipitation_numbers(&self) -> Array1<f64> {
        let mut participation_numbers: Array1<f64> = Array1::zeros(self.energies.raw_dim());

        // Create the output for each exciton state.
        for (n, (_e, v)) in self
            .energies
            .iter()
            .zip(self.coefficients.axis_iter(Axis(1)))
            .enumerate()
        {
            // Sort the indices by coefficients of the current eigenvector.
            let sorted_indices: Vec<usize> = argsort_abs(v.view());

            let mut amplitudes: Vec<f64> = Vec::new();
            let mut squared_value: f64 = 0.0;

            // Reverse the Iterator to write the largest amplitude first.
            for i in sorted_indices.into_iter().rev() {
                // Amplitude of the current transition.
                let c: f64 = v[i].abs();

                let state = self.basis.get(i).unwrap();
                match state {
                    BasisState::LE(ref _a) => {
                        amplitudes.push(c);
                        squared_value += c.powi(2);
                    }
                    BasisState::PairCT(ref _a) => {
                        squared_value += c.powi(2);
                    }
                }
            }

            let amplitudes: Array1<f64> = Array::from(amplitudes);
            participation_numbers[n] = amplitudes.sum().powi(2);
        }
        participation_numbers
    }

    pub fn get_transition_densities(&self, states: &Vec<usize>) {
        for state in states {
            let tdm: Array2<f64> = self.get_transition_density_matrix(*state);
            let occ_orbs = self.orbs.slice(s![.., ..self.dim.0]);
            let virt_orbs = self.orbs.slice(s![.., self.dim.0..]);
            let tdm_ao: Array2<f64> = occ_orbs.dot(&tdm.dot(&virt_orbs.t()));

            let mut tmp_string: String = String::from("transition_density_");
            tmp_string.push_str(&state.to_string());
            tmp_string.push_str(".npy");
            write_npy(tmp_string, &tdm_ao);
        }
    }

    pub fn get_particle_hole_densities(&self, states: &Vec<usize>, s: ArrayView2<f64>) {
        for state in states {
            let tdm: Array2<f64> = self.get_transition_density_matrix(*state);
            let occ_orbs = self.orbs.slice(s![.., ..self.dim.0]);
            let virt_orbs = self.orbs.slice(s![.., self.dim.0..]);
            let tdm_ao: Array2<f64> = occ_orbs.dot(&tdm.dot(&virt_orbs.t()));

            let h_mat: Array2<f64> = tdm_ao.dot(&s.dot(&tdm_ao.t()));
            let p_mat: Array2<f64> = tdm_ao.t().dot(&s.dot(&tdm_ao));

            let mut tmp_string: String = String::from("hole_density_");
            tmp_string.push_str(&state.to_string());
            tmp_string.push_str(".npy");
            write_npy(tmp_string, &h_mat);

            let mut tmp_string: String = String::from("particle_density_");
            tmp_string.push_str(&state.to_string());
            tmp_string.push_str(".npy");
            write_npy(tmp_string, &p_mat);
        }
    }

    pub fn nto_participation_numbers(&self, states: &Vec<usize>) -> Array1<f64> {
        let participation_numbers: Vec<f64> = states
            .into_par_iter()
            .map(|n| {
                let lambdas: Array1<f64> =
                    get_nto_singular_values(self.get_transition_density_matrix(*n).view());

                let lambdas_sqr = &lambdas * &lambdas;
                lambdas.sum().powi(2) / lambdas_sqr.sum()
            })
            .collect();
        Array::from(participation_numbers)
    }

    pub fn calculate_ntos_jmol(&self, states: &Vec<usize>, atoms: &[Atom]) {
        states.into_par_iter().for_each(|n| {
            let mut filename: String = String::from("s_");
            filename.push_str(&n.to_string());
            filename.push_str("_ntos.molden");
            self.ntos_to_molden(&atoms, *n, &filename);
        })
    }
}

impl Display for ExcitonStates<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let threshold = 0.1;
        // Empty line.
        let mut txt: String = format!("{:^80}\n", "");

        // Header for the section.
        txt += &format!("{: ^80}\n", "FMO-LCMO Excitation Energies");

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
            let rel_energy_ev: f64 = e * HARTREE_TO_EV;

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
                    txt += &format!(
                        "  {:28} Amplitude: {:6.4} => {:>4.1} %\n",
                        format!("{}", self.basis.get(i).unwrap()),
                        c,
                        c.powi(2) * 1e2
                    );
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
