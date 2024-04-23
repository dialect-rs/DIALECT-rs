use crate::fmo::{Monomer, ReducedBasisState, SuperSystem};
use crate::initialization::System;
use crate::scc::scc_routine::RestrictedSCC;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::ops::AddAssign;
use std::time::Instant;

impl System {
    pub fn calculate_energies_and_gradient(
        &mut self,
        state: usize,
        state_coupling: bool,
        gs_dynamic: bool,
    ) -> (Array1<f64>, Array1<f64>) {
        let mut gradient: Array1<f64> = Array::zeros(3 * self.n_atoms);
        let mut energies: Array1<f64> = Array1::zeros(self.config.excited.nstates + 1);

        if state == 0 {
            // ground state energy
            self.prepare_scc();
            let gs_energy: f64 = self.run_scc().unwrap();
            energies[0] = gs_energy;

            gradient = self.ground_state_gradient(false);

            if state_coupling && gs_dynamic == false {
                // calculate excited states
                self.calculate_excited_states(false);
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

        return (energies, gradient);
    }

    pub fn calculate_energies_and_gradient_ehrenfest(
        &mut self,
        state_coefficients: ArrayView1<f64>,
        thresh: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        let mut gradient: Array1<f64> = Array::zeros(3 * self.n_atoms);
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

        return (energies, gradient);
    }
}

impl SuperSystem<'_> {
    pub fn calculate_ehrenfest_gradient(
        &mut self,
        state_coefficients: ArrayView1<f64>,
        thresh: f64,
    ) -> (Array1<f64>) {
        let timer: Instant = Instant::now();
        // ground state energy and gradient
        let gs_gradient = self.ground_state_gradient();

        println!(
            "Time ground state gradient {:.5}",
            timer.elapsed().as_secs_f32()
        );
        drop(timer);
        let timer: Instant = Instant::now();

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
                    ReducedBasisState::CT(ref a) => {}
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

        println!(
            "Time excited state gradient {:.5}",
            timer.elapsed().as_secs_f32()
        );
        drop(timer);
        let timer: Instant = Instant::now();

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
        println!(
            "Time slicing the gradient {:.5}",
            timer.elapsed().as_secs_f32()
        );

        gradient
    }
}
