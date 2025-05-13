use crate::fmo::*;
use crate::initialization::*;
use nalgebra::Vector3;
use ndarray::prelude::*;
use ndarray_npy::write_npy;
use std::ops::AddAssign;

mod ct_state;
mod embedding;
mod es_dimer;
mod le_state;
mod monomer;
mod pair;
mod response;

// mod numerical;
use crate::fmo::gradients::embedding::diag_of_last_dimensions;
use crate::fmo::helpers::get_pair_slice;
use crate::gradients::dispersion::gradient_disp;
use rayon::prelude::*;

pub trait GroundStateGradient {
    fn get_grad_dq(
        &self,
        atoms: &[Atom],
        s: ArrayView2<f64>,
        grad_s: ArrayView3<f64>,
        p: ArrayView2<f64>,
    ) -> Array2<f64>;
    fn scc_gradient(&mut self, atoms: &[Atom]) -> Array1<f64>;
}

impl SuperSystem<'_> {
    pub fn ground_state_gradient(&mut self) -> Array1<f64> {
        let atoms: &[Atom] = &self.atoms[..];
        for mol in self.monomers.iter_mut() {
            let q_vo: Array2<f64> = mol.compute_q_vo(&atoms[mol.slice.atom_as_range()], None);
            mol.properties.set_q_vo(q_vo);
        }
        let mut grad: Array1<f64> = Array1::zeros(3 * atoms.len());

        if self.config.dispersion.use_dispersion {
            grad = grad + gradient_disp(atoms, &self.config.dispersion);
        }
        let monomer_gradient: Array1<f64> = self.monomer_gradients();

        let pair_gradient: Array1<f64> = self.pair_gradients(monomer_gradient.view());

        let embedding_gradient: Array1<f64> = self.embedding_gradient();

        let esd_gradient: Array1<f64> = self.es_dimer_gradient();

        grad = grad + monomer_gradient + pair_gradient + embedding_gradient + esd_gradient;

        // save the gradient
        if self.config.jobtype == "grad" {
            // save the gradient
            write_npy("gs_gradient.npy", &grad).unwrap();
        }
        grad
    }

    fn monomer_gradients(&mut self) -> Array1<f64> {
        let mut gradient: Array1<f64> = Array1::zeros([3 * self.atoms.len()]);
        let atoms: &[Atom] = &self.atoms[..];
        // The derivative of the charge differences is initialized as an array with zeros.
        let mut grad_dq: Array1<f64> = Array1::zeros([3 * self.atoms.len()]);

        // Parallelization
        let gradient_vec: Vec<Array1<f64>> = self
            .monomers
            .par_iter_mut()
            .map(|mol| {
                let arr: Array1<f64> = mol.scc_gradient(&atoms[mol.slice.atom_as_range()]);
                arr
            })
            .collect();

        for (mol, vector) in self.monomers.iter().zip(gradient_vec.iter()) {
            let mol_grad_dq: ArrayView3<f64> = mol
                .properties
                .grad_dq()
                .unwrap()
                .into_shape([3, mol.n_atoms, mol.n_atoms])
                .unwrap();

            grad_dq
                .slice_mut(s![mol.slice.grad])
                .assign(&diag_of_last_dimensions(mol_grad_dq));

            gradient.slice_mut(s![mol.slice.grad]).assign(vector);
        }

        self.properties.set_grad_dq_diag(grad_dq);

        gradient
    }

    fn pair_gradients(&mut self, monomer_gradient: ArrayView1<f64>) -> Array1<f64> {
        let mut gradient: Array1<f64> = Array1::zeros([3 * self.atoms.len()]);
        let atoms: &[Atom] = &self.atoms[..];
        let monomers: &Vec<Monomer> = &self.monomers;

        // Parallelization
        let gradient_vec: Vec<Array1<f64>> = self
            .pairs
            .par_iter_mut()
            .map(|pair| {
                // get references to the corresponding monomers
                let m_i: &Monomer = &monomers[pair.i];
                let m_j: &Monomer = &monomers[pair.j];

                let pair_atoms: Vec<Atom> =
                    get_pair_slice(atoms, m_i.slice.atom_as_range(), m_j.slice.atom_as_range());
                // compute the gradient of the pair
                pair.scc_gradient(&pair_atoms[..])
            })
            .collect();

        for (pair, pair_grad) in self.pairs.iter().zip(gradient_vec.iter()) {
            // get references to the corresponding monomers
            let m_i: &Monomer = &monomers[pair.i];
            let m_j: &Monomer = &monomers[pair.j];

            // subtract the monomer contributions and assemble it into the gradient
            gradient.slice_mut(s![m_i.slice.grad]).add_assign(
                &(&pair_grad.slice(s![0..(3 * m_i.n_atoms)])
                    - &monomer_gradient.slice(s![m_i.slice.grad])),
            );
            gradient.slice_mut(s![m_j.slice.grad]).add_assign(
                &(&pair_grad.slice(s![(3 * m_i.n_atoms)..])
                    - &monomer_gradient.slice(s![m_j.slice.grad])),
            );
        }

        gradient
    }

    pub fn pair_gradients_for_testing(&mut self) -> Array1<f64> {
        let monomer_gradient = self.monomer_gradients();
        let mut gradient: Array1<f64> = Array1::zeros([3 * self.atoms.len()]);
        let atoms: &[Atom] = &self.atoms[..];
        for pair in self.pairs.iter_mut() {
            // get references to the corresponding monomers
            let m_i: &Monomer = &self.monomers[pair.i];
            let m_j: &Monomer = &self.monomers[pair.j];

            let pair_atoms: Vec<Atom> =
                get_pair_slice(atoms, m_i.slice.atom_as_range(), m_j.slice.atom_as_range());
            // compute the gradient of the pair
            let pair_grad: Array1<f64> = pair.scc_gradient(&pair_atoms[..]);
            // subtract the monomer contributions and assemble it into the gradient
            gradient.slice_mut(s![m_i.slice.grad]).add_assign(
                &(&pair_grad.slice(s![0..(3 * m_i.n_atoms)])
                    - &monomer_gradient.slice(s![m_i.slice.grad])),
            );
            gradient.slice_mut(s![m_j.slice.grad]).add_assign(
                &(&pair_grad.slice(s![(3 * m_i.n_atoms)..])
                    - &monomer_gradient.slice(s![m_j.slice.grad])),
            );
        }
        gradient
    }
}

//  Compute the gradient of the repulsive potential
//  Parameters:
//  ===========
//  atomlist: list of tuples (Zi, [xi,yi,zi]) for each atom
//  distances: matrix with distances between atoms, distance[i,j]
//    is the distance between atoms i and j
//  directions: directions[i,j,:] is the unit vector pointing from
//    atom j to atom i
//  VREP: dictionary, VREP[(Zi,Zj)] has to be an instance of RepulsivePotential
//    for the atom pair Zi-Zj
fn gradient_v_rep(atoms: &[Atom], v_rep: &RepulsivePotential) -> Array1<f64> {
    let n_atoms: usize = atoms.len();
    let mut grad: Array1<f64> = Array1::zeros([3 * n_atoms]);
    for (i, atomi) in atoms.iter().enumerate() {
        let mut grad_i: Array1<f64> = Array::zeros([3]);
        for (j, atomj) in atoms.iter().enumerate() {
            if i != j {
                let mut r: Vector3<f64> = atomi - atomj;
                let r_ij: f64 = r.norm();
                r /= r_ij;
                let v_ij_deriv: f64 = v_rep.get(atomi.kind, atomj.kind).spline_deriv(r_ij);
                r *= v_ij_deriv;

                let v = Array1::from_iter(r.iter());
                grad_i = &grad_i + &v;
            }
        }
        grad.slice_mut(s![i * 3..i * 3 + 3]).assign(&grad_i);
    }
    grad
}
