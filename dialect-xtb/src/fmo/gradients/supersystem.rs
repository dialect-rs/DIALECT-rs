use dialect_utilities::fmo_helpers::get_pair_slice_xtb;
use crate::fmo::monomer::XtbMonomer;
use crate::fmo::supersystem::XtbSuperSystem;
use crate::initialization::atom::XtbAtom;
use crate::scc::hamiltonian::calculate_coordination_numbers;
use ndarray::{s, Array1, ArrayView1};
use rayon::prelude::*;
use std::ops::{AddAssign, SubAssign};
use std::time::Instant;

impl XtbSuperSystem<'_> {
    /// FMO-xTB ground-state gradient: the explicit FMO term (monomer + pair +
    /// embedding + ESD, shell-resolved) plus the CPHF/CN response term. For
    /// covalent fragmentation it delegates to the HOP gradient.
    pub fn ground_state_gradient(&mut self) -> Array1<f64> {
        if self.config.fmo.covalent_fragmentation {
            self.ground_state_gradient_hop()
        } else {
            let (fmo_grad, cn_grad_global) = self.ground_state_gradient_fmo_shell();
            let response_grad = self.response_gradient_fmo_xtb_shell(&cn_grad_global);
            fmo_grad + &response_grad
        }
    }

    /// Compute the full FMO-xTB HOP gradient using the new SCC structs.
    ///
    /// Runs SCC to get all states, then computes explicit + response gradient.
    pub fn ground_state_gradient_hop(&mut self) -> Array1<f64> {
        // Ensure CN is set on properties (needed by gradient code)
        let cn_numbers = calculate_coordination_numbers(&self.atoms);
        self.properties.set_cn(cn_numbers);

        // Run SCC and get all states
        let (_energy, hop_data, mono_states, pair_states, trimer_states) =
            self.run_scc_hop_for_gradient().expect("HOP SCC failed");

        let gammafunction = self.monomers[0].gammafunction.clone();

        // Explicit gradient
        let t1 = Instant::now();
        let (fmo_grad, cn_grad_global) =
            super::hop_gradients::ground_state_gradient_fmo_xtb_hop(
                self,
                &hop_data,
                &mono_states,
                &pair_states,
                &trimer_states,
                &gammafunction,
            );
        let time1 = t1.elapsed().as_secs_f32();

        // Response gradient
        let t2 = Instant::now();
        let frag_atom_ranges: Vec<std::ops::Range<usize>> = self
            .monomers
            .iter()
            .map(|m| m.slice.atom_as_range())
            .collect();
        let pair_scal: Vec<f64> = if self.config.fmo.use_three_body {
            let mut scal = vec![1.0f64; pair_states.len()];
            for ts in trimer_states.iter() {
                for &(a, b) in &[(ts.i, ts.j), (ts.i, ts.k), (ts.j, ts.k)] {
                    if self.properties.type_of_pair(a, b) == dialect_state::PairType::Pair {
                        let idx = self.properties.index_of_pair(a, b);
                        scal[idx] -= 1.0;
                    }
                }
            }
            scal
        } else {
            vec![1.0f64; pair_states.len()]
        };

        let cn_numbers_global: ArrayView1<f64> = self.properties.cn().unwrap();
        let response_grad =
            super::hop_gradients::response::response_gradient_xtb_hop_total(
                &mono_states,
                &pair_states,
                &trimer_states,
                &hop_data,
                &pair_scal,
                &gammafunction,
                cn_numbers_global,
                cn_grad_global.view(),
                &self.atoms,
                self.atoms.len(),
                &frag_atom_ranges,
                self.config.fmo.use_three_body,
            );
        let time2 = t2.elapsed().as_secs_f32();

        // Response enabled
        let total = &fmo_grad + &response_grad;
        total
    }

    pub fn monomer_gradients(&mut self) -> Array1<f64> {
        let mut gradient: Array1<f64> = Array1::zeros([3 * self.atoms.len()]);
        let atoms: &[XtbAtom] = &self.atoms[..];

        // Parallelization
        let gradient_vec: Vec<Array1<f64>> = self
            .monomers
            .par_iter_mut()
            .map(|mol| {
                // let arr: Array1<f64> = mol.ground_state_gradient(&atoms[mol.slice.atom_as_range()]);
                let arr: Array1<f64> = mol.ground_state_gradient_onthefly_serial(
                    &self.config,
                    &atoms[mol.slice.atom_as_range()],
                );
                // let diff = (&arr2 - &arr).map(|val| val * val).sum();
                // println!("Gradient difference: {:.10}", diff);

                arr
            })
            .collect();

        // order the gradient contributions of the monomers
        for (mol, vector) in self.monomers.iter().zip(gradient_vec.iter()) {
            gradient.slice_mut(s![mol.slice.grad]).assign(vector);
        }

        gradient
    }

    pub fn pair_gradients(
        &mut self,
        monomer_gradient: ArrayView1<f64>,
        use_three_body: bool,
    ) -> (Array1<f64>, Option<Vec<Array1<f64>>>) {
        let mut gradient: Array1<f64> = Array1::zeros([3 * self.atoms.len()]);
        let atoms: &[XtbAtom] = &self.atoms[..];
        let monomers: &Vec<XtbMonomer> = &self.monomers;

        // Parallelization
        let gradient_vec: Vec<Array1<f64>> = self
            .pairs
            .par_iter_mut()
            .map(|pair| {
                // get references to the corresponding monomers
                let m_i: &XtbMonomer = &monomers[pair.i];
                let m_j: &XtbMonomer = &monomers[pair.j];

                let pair_atoms: Vec<XtbAtom> =
                    get_pair_slice_xtb(atoms, m_i.slice.atom_as_range(), m_j.slice.atom_as_range());
                // compute the gradient of the pair
                // let arr = pair.ground_state_gradient_low_memory(&pair_atoms[..]);
                let arr = pair.ground_state_gradient_onthefly_serial(&self.config, &pair_atoms);
                // let diff = (&arr_2 - &arr).map(|val| val * val).sum();
                // println!("Gradient difference pair: {:.10}", diff);

                arr
            })
            .collect();

        let mut delta_pair_gradients: Vec<Array1<f64>> = Vec::new();
        for (pair, pair_grad) in self.pairs.iter().zip(gradient_vec.iter()) {
            // get references to the corresponding monomers
            let m_i: &XtbMonomer = &monomers[pair.i];
            let m_j: &XtbMonomer = &monomers[pair.j];

            // subtract the monomer contributions and assemble it into the gradient
            gradient.slice_mut(s![m_i.slice.grad]).add_assign(
                &(&pair_grad.slice(s![0..(3 * m_i.n_atoms)])
                    - &monomer_gradient.slice(s![m_i.slice.grad])),
            );
            gradient.slice_mut(s![m_j.slice.grad]).add_assign(
                &(&pair_grad.slice(s![(3 * m_i.n_atoms)..])
                    - &monomer_gradient.slice(s![m_j.slice.grad])),
            );

            if use_three_body {
                let mut ij_gradient: Array1<f64> = pair_grad.to_owned();
                ij_gradient
                    .slice_mut(s![..3 * m_i.n_atoms])
                    .sub_assign(&monomer_gradient.slice(s![m_i.slice.grad]));
                ij_gradient
                    .slice_mut(s![3 * m_i.n_atoms..])
                    .sub_assign(&monomer_gradient.slice(s![m_j.slice.grad]));
                delta_pair_gradients.push(ij_gradient);
            }
        }

        if use_three_body {
            (gradient, Some(delta_pair_gradients))
        } else {
            (gradient, None)
        }
    }
}
