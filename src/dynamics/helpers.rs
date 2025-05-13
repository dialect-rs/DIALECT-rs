use crate::fmo::{Monomer, ReducedBasisState, SuperSystem};
use crate::initialization::System;
use crate::scc::scc_routine::RestrictedSCC;
use crate::xtb::initialization::system::XtbSystem;
use ndarray::prelude::*;
use ndarray_linalg::{c64, Scalar};
use rayon::prelude::*;
use std::ops::AddAssign;

impl System {
    pub fn calculate_energies_and_gradient(
        &mut self,
        state: usize,
        _state_coupling: bool,
        gs_dynamic: bool,
    ) -> (Array1<f64>, Array1<f64>) {
        let mut gradient: Array1<f64>;
        let mut energies: Array1<f64> = Array1::zeros(self.config.excited.nstates + 1);

        if state == 0 {
            // ground state energy
            self.prepare_scc();
            let gs_energy: f64 = self.run_scc().unwrap();
            energies[0] = gs_energy;

            // if state_coupling && !gs_dynamic {
            if !gs_dynamic {
                // calculate excited states
                self.calculate_excited_states(false);
                let ci_energies: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
                energies
                    .slice_mut(s![1..])
                    .assign(&(gs_energy + &ci_energies));

                // calculate the gradient
                gradient = self.ground_state_gradient(true);
            } else {
                gradient = self.ground_state_gradient(false);
            }
        } else {
            // ground state energy
            let excited_state: usize = state - 1;
            self.prepare_scc();
            let gs_energy: f64 = self.run_scc().unwrap();
            energies[0] = gs_energy;

            // calculate excited states
            self.calculate_excited_states(false);

            let ci_energies: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
            energies
                .slice_mut(s![1..])
                .assign(&(gs_energy + &ci_energies));

            gradient = self.ground_state_gradient(true);
            gradient = gradient + self.calculate_excited_state_gradient(excited_state);
        }

        (energies, gradient)
    }

    pub fn calculate_energies_and_gradient_ehrenfest(
        &mut self,
        state_coefficients: ArrayView1<f64>,
        thresh: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        let mut gradient: Array1<f64>;
        let mut energies: Array1<f64> = Array1::zeros(self.config.excited.nstates + 1);

        // ground state energy
        self.prepare_scc();
        let gs_energy: f64 = self.run_scc().unwrap();
        energies[0] = gs_energy;

        // calculate excited states
        self.calculate_excited_states(false);

        let ci_energies: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
        energies
            .slice_mut(s![1..])
            .assign(&(gs_energy + &ci_energies));
        // .assign(&ci_energies);

        gradient = self.ground_state_gradient(true);

        for (idx, coefficient) in state_coefficients.slice(s![1..]).iter().enumerate() {
            if *coefficient > thresh {
                gradient = gradient + *coefficient * &self.calculate_excited_state_gradient(idx);
            }
        }

        (energies, gradient)
    }

    pub fn calculate_ehrenfest_gradient_nacmes(
        &self,
        energies: ArrayView1<f64>,
        vectors: &Vec<Array1<f64>>,
        state_coefficients: ArrayView1<c64>,
        thresh: f64,
    ) -> Array1<f64> {
        let mut gradient: Array1<f64> = Array1::zeros(self.n_atoms * 3);
        let real_coefficients: Array1<f64> = state_coefficients.map(|val| val.re());

        for (idx, coefficient) in real_coefficients.iter().enumerate() {
            for (idx2, coefficient2) in real_coefficients.iter().enumerate() {
                if idx != idx2 {
                    let mut nac_idx: Option<usize> = None;
                    let mut count: usize = 0;
                    for idx_1 in 0..real_coefficients.len() {
                        for idx_2 in 0..real_coefficients.len() {
                            if idx_1 < idx_2 {
                                if idx == idx_1 && idx2 == idx_2 {
                                    nac_idx = Some(count);
                                } else if idx == idx_2 && idx2 == idx_1 {
                                    nac_idx = Some(count);
                                }
                                count += 1;
                            }
                        }
                    }
                    let vec_idx: usize = nac_idx.unwrap();
                    let coeff_product: f64 = coefficient * coefficient2;
                    if coeff_product > thresh {
                        let vector: Array1<f64> = if idx < idx2 {
                            vectors[vec_idx].clone()
                        } else {
                            -vectors[vec_idx].clone()
                        };
                        let energy_diff: f64 = energies[idx] - energies[idx2];
                        gradient = gradient + vector * coeff_product * energy_diff;
                    }
                }
            }
        }
        gradient
    }

    pub fn calculate_energies_and_gradient_ehrenfest_tab(
        &mut self,
        state_coefficients: ArrayView1<f64>,
        thresh: f64,
    ) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        let mut gradient: Array1<f64>;
        let mut energies: Array1<f64> = Array1::zeros(self.config.excited.nstates + 1);
        let mut gradients: Array2<f64> = Array::zeros([3 * self.n_atoms, state_coefficients.len()]);

        // ground state energy
        self.prepare_scc();
        let gs_energy: f64 = self.run_scc().unwrap();
        energies[0] = gs_energy;

        // calculate excited states
        self.calculate_excited_states(false);

        let ci_energies: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
        energies
            .slice_mut(s![1..])
            .assign(&(gs_energy + &ci_energies));
        // .assign(&ci_energies);

        let gs_grad = self.ground_state_gradient(true);
        gradient = gs_grad.clone();
        gradients.slice_mut(s![.., 0]).assign(&gradient);

        for (idx, coefficient) in state_coefficients.slice(s![1..]).iter().enumerate() {
            let grad: Array1<f64> = self.calculate_excited_state_gradient(idx);
            if *coefficient > thresh {
                gradient = gradient + *coefficient * &grad;
            }
            gradients
                .slice_mut(s![.., idx + 1])
                .assign(&(&grad + &gs_grad));
        }

        (energies, gradient, gradients)
    }
}

impl SuperSystem<'_> {
    pub fn calculate_ehrenfest_gradient(
        &mut self,
        state_coefficients: ArrayView1<f64>,
        thresh: f64,
    ) -> Array1<f64> {
        // ground state energy and gradient
        let gs_gradient = self.ground_state_gradient();

        // mutable gradient
        let mut gradient: Array1<f64> = gs_gradient;

        // get the basis states
        let states = self.properties.basis_states().unwrap().to_vec();

        // prepare LE calculations
        for (idx, state) in states.iter().enumerate() {
            let coefficient = state_coefficients[idx + 1];
            if coefficient > thresh {
                match state {
                    ReducedBasisState::LE(ref a) => {
                        let mol: &mut Monomer = &mut self.monomers[a.monomer_index];
                        mol.prepare_excited_gradient(&self.atoms[mol.slice.atom_as_range()]);
                    }
                    ReducedBasisState::CT(ref _a) => {}
                }
            }
        }

        let vec_array: Vec<Array1<f64>> = states
            .par_iter()
            .enumerate()
            .map(|(idx, state)| {
                let coefficient = state_coefficients[idx + 1];
                if coefficient > thresh {
                    match state {
                        ReducedBasisState::LE(ref a) => {
                            let mol: &Monomer = &self.monomers[a.monomer_index];
                            mol.tda_gradient_lc(a.state_index)
                        }
                        ReducedBasisState::CT(ref a) => self.charge_transfer_pair_gradient(a),
                    }
                } else {
                    Array1::zeros(1)
                }
            })
            .collect();

        // iterate over states
        for ((idx, state), array) in states.iter().enumerate().zip(vec_array.iter()) {
            let coefficient = state_coefficients[idx + 1];
            if coefficient > thresh {
                match state {
                    ReducedBasisState::LE(ref a) => {
                        let mol: &Monomer = &self.monomers[a.monomer_index];
                        gradient
                            .slice_mut(s![mol.slice.grad])
                            .add_assign(&(coefficient * array));
                    }
                    ReducedBasisState::CT(ref a) => {
                        let m_h = &self.monomers[a.m_h];
                        let m_l = &self.monomers[a.m_l];

                        if a.m_h < a.m_l {
                            gradient
                                .slice_mut(s![m_h.slice.grad])
                                .add_assign(&(coefficient * &array.slice(s![..m_h.n_atoms * 3])));
                            gradient
                                .slice_mut(s![m_l.slice.grad])
                                .add_assign(&(coefficient * &array.slice(s![m_h.n_atoms * 3..])));
                        } else {
                            gradient
                                .slice_mut(s![m_l.slice.grad])
                                .add_assign(&(coefficient * &array.slice(s![..m_l.n_atoms * 3])));
                            gradient
                                .slice_mut(s![m_h.slice.grad])
                                .add_assign(&(coefficient * &array.slice(s![m_l.n_atoms * 3..])));
                        }
                    }
                }
            }
        }
        gradient
    }

    pub fn calculate_ehrenfest_gradient_tab(
        &mut self,
        state_coefficients: ArrayView1<f64>,
        tab_grad_threshold: f64,
    ) -> (Array1<f64>, Array2<f64>) {
        // ground state energy and gradient
        let gs_gradient = self.ground_state_gradient();
        let thresh: f64 = tab_grad_threshold;

        // mutable gradient
        let mut gradient: Array1<f64> = gs_gradient;
        let mut grad_array: Array2<f64> =
            Array2::zeros([3 * self.atoms.len(), state_coefficients.len()]);
        for idx in 0..state_coefficients.len() {
            grad_array.slice_mut(s![.., idx]).assign(&gradient);
        }

        // get the basis states
        let states = self.properties.basis_states().unwrap().to_vec();

        // prepare LE calculations
        for (idx, state) in states.iter().enumerate() {
            let coefficient = state_coefficients[idx + 1];
            if coefficient > thresh {
                match state {
                    ReducedBasisState::LE(ref a) => {
                        let mol: &mut Monomer = &mut self.monomers[a.monomer_index];
                        mol.prepare_excited_gradient(&self.atoms[mol.slice.atom_as_range()]);
                    }
                    ReducedBasisState::CT(ref _a) => {}
                }
            }
        }

        let vec_array: Vec<Array1<f64>> = states
            .par_iter()
            .enumerate()
            .map(|(idx, state)| {
                let coefficient = state_coefficients[idx + 1];
                if coefficient > thresh {
                    match state {
                        ReducedBasisState::LE(ref a) => {
                            let mol: &Monomer = &self.monomers[a.monomer_index];
                            mol.tda_gradient_lc(a.state_index)
                            // let mut mol: Monomer = self.monomers[a.monomer_index].clone();
                            // mol.prepare_excited_gradient(&self.atoms[mol.slice.atom_as_range()]);
                            // mol.tda_gradient_lc(a.state_index)
                        }
                        ReducedBasisState::CT(ref a) => self.charge_transfer_pair_gradient(a),
                    }
                } else {
                    Array1::zeros(1)
                }
            })
            .collect();

        // iterate over states
        for ((idx, state), array) in states.iter().enumerate().zip(vec_array.iter()) {
            let coefficient = state_coefficients[idx + 1];
            if coefficient > thresh {
                match state {
                    ReducedBasisState::LE(ref a) => {
                        let mol: &Monomer = &self.monomers[a.monomer_index];
                        gradient
                            .slice_mut(s![mol.slice.grad])
                            .add_assign(&(coefficient * array));
                        grad_array
                            .slice_mut(s![mol.slice.grad, idx + 1])
                            .add_assign(array);
                    }
                    ReducedBasisState::CT(ref a) => {
                        let m_h = &self.monomers[a.m_h];
                        let m_l = &self.monomers[a.m_l];

                        if a.m_h < a.m_l {
                            gradient
                                .slice_mut(s![m_h.slice.grad])
                                .add_assign(&(coefficient * &array.slice(s![..m_h.n_atoms * 3])));
                            gradient
                                .slice_mut(s![m_l.slice.grad])
                                .add_assign(&(coefficient * &array.slice(s![m_h.n_atoms * 3..])));
                            grad_array
                                .slice_mut(s![m_h.slice.grad, idx + 1])
                                .add_assign(&array.slice(s![..m_h.n_atoms * 3]));
                            grad_array
                                .slice_mut(s![m_l.slice.grad, idx + 1])
                                .add_assign(&array.slice(s![m_h.n_atoms * 3..]));
                        } else {
                            gradient
                                .slice_mut(s![m_l.slice.grad])
                                .add_assign(&(coefficient * &array.slice(s![..m_l.n_atoms * 3])));
                            gradient
                                .slice_mut(s![m_h.slice.grad])
                                .add_assign(&(coefficient * &array.slice(s![m_l.n_atoms * 3..])));
                            grad_array
                                .slice_mut(s![m_l.slice.grad, idx + 1])
                                .add_assign(&array.slice(s![..m_l.n_atoms * 3]));
                            grad_array
                                .slice_mut(s![m_h.slice.grad, idx + 1])
                                .add_assign(&array.slice(s![m_l.n_atoms * 3..]));
                        }
                    }
                }
            }
        }

        (gradient, grad_array)
    }
}

impl XtbSystem {
    pub fn calculate_energies_and_gradient(&mut self) -> (Array1<f64>, Array1<f64>) {
        let mut energies: Array1<f64> = Array1::zeros(1);
        // ground state energy
        self.prepare_scc();
        let gs_energy: f64 = self.run_scc().unwrap();
        energies[0] = gs_energy;
        let gradient = self.ground_state_gradient();

        (energies, gradient)
    }
}
