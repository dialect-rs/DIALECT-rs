use crate::excited_states::tda::*;
use crate::excited_states::ExcitedState;
use crate::fmo::{ESDPair, ExcitonStates, Monomer, Pair, PairType, SuperSystem};
use crate::initialization::{Atom, MO};
use crate::io::settings::LcmoConfig;
use crate::properties::Properties;
use crate::utils::Timer;
use crate::{initial_subspace, Davidson};
use nalgebra::{max, Vector3};
use ndarray::prelude::*;
use ndarray::{concatenate, AssignElem, Slice};
use ndarray_linalg::{Eigh, UPLO};
use ndarray_npy::write_npy;
use rayon::prelude::*;
use std::any::Any;
use std::fmt::{Display, Formatter};
use std::sync::Mutex;
use std::time::Instant;

impl SuperSystem<'_> {
    pub fn create_diabatic_basis(&self, n_ct: usize) -> Vec<BasisState> {
        let lcmo_config: LcmoConfig = self.config.fmo_lc_tddftb.clone();
        // Number of LE states per monomer.
        let n_le: usize = lcmo_config.n_le;
        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let threshold_le: f64 = lcmo_config.active_space_threshold_le;
        let threshold_ct: f64 = lcmo_config.active_space_threshold_ct;

        let mut states: Vec<BasisState> = Vec::new();
        // Create all LE states.

        for mol in self.monomers.iter() {
            let homo: usize = mol.properties.homo().unwrap();
            let q_ov: ArrayView2<f64> = mol.properties.q_ov().unwrap();

            let mut le_states: Vec<BasisState> = (0..n_le)
                .into_par_iter()
                .map(|n| {
                    let tdm: ArrayView1<f64> = mol.properties.ci_coefficient(n).unwrap();
                    let tdm_dim2: ArrayView2<f64> = mol.properties.tdm(n).unwrap();

                    // determine the relevant orbital indices
                    let mut occ_indices: Vec<usize> = Vec::new();
                    let mut virt_indices: Vec<usize> = Vec::new();
                    for (idx_i, val_i) in tdm_dim2.outer_iter().enumerate() {
                        for (idx_j, val_j) in val_i.iter().enumerate() {
                            let abs_c_sqr: f64 = val_j.abs().powi(2);
                            if abs_c_sqr > threshold_le {
                                if !occ_indices.contains(&idx_i) {
                                    occ_indices.push(idx_i);
                                }
                                if !virt_indices.contains(&idx_j) {
                                    virt_indices.push(idx_j);
                                }
                            }
                        }
                    }

                    let le_state = LocallyExcited {
                        monomer: mol,
                        n: n,
                        atoms: &atoms[mol.slice.atom_as_range()],
                        q_trans: q_ov.dot(&tdm),
                        occs: mol.properties.orbs_slice(0, Some(homo + 1)).unwrap(),
                        virts: mol.properties.orbs_slice(homo + 1, None).unwrap(),
                        tdm: tdm,
                        tr_dipole: mol.properties.tr_dipole(n).unwrap(),
                        occ_indices,
                        virt_indices,
                    };

                    BasisState::LE(le_state)
                })
                .collect();

            states.append(&mut le_states)
        }

        if n_ct > 0 {
            // Create all CT states.
            let mut ct_basis: Vec<Vec<_>> = self
                .monomers
                .par_iter()
                .enumerate()
                .map(|(idx, m_i)| {
                    let mut ct_basis_temp: Vec<_> = Vec::new();

                    for m_j in self.monomers[idx + 1..].iter() {
                        // get the PairType
                        let type_ij: PairType = self.properties.type_of_pair(m_i.index, m_j.index);

                        // create both CT states
                        let mut state_1 = PairChargeTransfer {
                            m_h: m_i,
                            m_l: m_j,
                            pair_type: type_ij,
                            properties: Properties::new(),
                        };
                        let mut state_2 = PairChargeTransfer {
                            m_h: m_j,
                            m_l: m_i,
                            pair_type: type_ij,
                            properties: Properties::new(),
                        };

                        // prepare the TDA calculation of both states
                        state_1.prepare_ct_tda(
                            self.properties.gamma().unwrap(),
                            self.properties.gamma_lr().unwrap(),
                            self.properties.s().unwrap(),
                            atoms,
                        );
                        state_2.prepare_ct_tda(
                            self.properties.gamma().unwrap(),
                            self.properties.gamma_lr().unwrap(),
                            self.properties.s().unwrap(),
                            atoms,
                        );
                        // do the TDA calculation using the davidson routine
                        state_1.run_ct_tda(
                            atoms,
                            n_ct,
                            self.config.excited.davidson_iterations,
                            1.0e-4,
                            self.config.excited.davidson_subspace_multiplier,
                        );
                        state_2.run_ct_tda(
                            atoms,
                            n_ct,
                            self.config.excited.davidson_iterations,
                            1.0e-4,
                            self.config.excited.davidson_subspace_multiplier,
                        );

                        let q_ov_1: ArrayView2<f64> = state_1.properties.q_ov().unwrap();
                        let q_ov_2: ArrayView2<f64> = state_2.properties.q_ov().unwrap();

                        for n in 0..n_ct {
                            let tdm_1: ArrayView1<f64> =
                                state_1.properties.ci_coefficient(n).unwrap();
                            let tdm_dim2_1: ArrayView2<f64> = state_1.properties.tdm(n).unwrap();

                            // determine the relevant orbital indices
                            let mut occ_indices: Vec<usize> = Vec::new();
                            let mut virt_indices: Vec<usize> = Vec::new();
                            for (idx_i, val_i) in tdm_dim2_1.outer_iter().enumerate() {
                                for (idx_j, val_j) in val_i.iter().enumerate() {
                                    let abs_c_sqr: f64 = val_j.abs().powi(2);
                                    if abs_c_sqr > threshold_ct {
                                        if !occ_indices.contains(&idx_i) {
                                            occ_indices.push(idx_i);
                                        }
                                        if !virt_indices.contains(&idx_j) {
                                            virt_indices.push(idx_j);
                                        }
                                    }
                                }
                            }

                            let ct_1 = ChargeTransferPair {
                                m_h: m_i.index,
                                m_l: m_j.index,
                                state_index: n,
                                state_energy: state_1.properties.ci_eigenvalue(n).unwrap(),
                                eigenvectors: state_1.properties.tdm(n).unwrap().to_owned(),
                                q_tr: q_ov_1.dot(&tdm_1),
                                tr_dipole: state_1.properties.tr_dipole(n).unwrap(),
                                occ_orb: m_i.slice.occ_orb.clone(),
                                virt_orb: m_j.slice.virt_orb.clone(),
                                occ_indices,
                                virt_indices,
                            };

                            let tdm_2: ArrayView1<f64> =
                                state_2.properties.ci_coefficient(n).unwrap();
                            let tdm_dim2_2: ArrayView2<f64> = state_2.properties.tdm(n).unwrap();

                            // determine the relevant orbital indices
                            let mut occ_indices: Vec<usize> = Vec::new();
                            let mut virt_indices: Vec<usize> = Vec::new();
                            for (idx_i, val_i) in tdm_dim2_2.outer_iter().enumerate() {
                                for (idx_j, val_j) in val_i.iter().enumerate() {
                                    let abs_c_sqr: f64 = val_j.abs().powi(2);
                                    if abs_c_sqr > threshold_ct {
                                        if !occ_indices.contains(&idx_i) {
                                            occ_indices.push(idx_i);
                                        }
                                        if !virt_indices.contains(&idx_j) {
                                            virt_indices.push(idx_j);
                                        }
                                    }
                                }
                            }

                            let ct_2 = ChargeTransferPair {
                                m_h: m_j.index,
                                m_l: m_i.index,
                                state_index: n,
                                state_energy: state_2.properties.ci_eigenvalue(n).unwrap(),
                                eigenvectors: state_2.properties.tdm(n).unwrap().to_owned(),
                                q_tr: q_ov_2.dot(&tdm_2),
                                tr_dipole: state_2.properties.tr_dipole(n).unwrap(),
                                occ_orb: m_j.slice.occ_orb.clone(),
                                virt_orb: m_i.slice.virt_orb.clone(),
                                occ_indices,
                                virt_indices,
                            };

                            ct_basis_temp.push(BasisState::PairCT(ct_1));
                            ct_basis_temp.push(BasisState::PairCT(ct_2));
                        }
                    }
                    ct_basis_temp
                })
                .collect();

            for ct_vec in ct_basis.iter_mut() {
                states.append(ct_vec);
            }
        }

        states
    }

    pub fn create_exciton_hamiltonian(&mut self) {
        // Calculate the H' matrix
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let max_iter: usize = 50;
        let tolerance: f64 = 1e-4;
        // Number of LE states per monomer.
        let n_le: usize = self.config.fmo_lc_tddftb.n_le;

        let fock_matrix: ArrayView2<f64> = self.properties.lcmo_fock().unwrap();
        // Calculate the excited states of the monomers
        // Swap the orbital energies of the monomers with the elements of the H' matrix
        self.monomers.par_iter_mut().for_each(|mol| {
            mol.properties.set_orbe(
                fock_matrix
                    .slice(s![mol.slice.orb, mol.slice.orb])
                    .diag()
                    .to_owned(),
            );
            mol.prepare_tda(&atoms[mol.slice.atom_as_range()]);
            mol.run_tda(
                &atoms[mol.slice.atom_as_range()],
                n_le,
                max_iter,
                tolerance,
                self.config.excited.davidson_subspace_multiplier,
            );
        });

        // Construct the basis states.
        let states: Vec<BasisState> = self.create_diabatic_basis(self.config.fmo_lc_tddftb.n_ct);

        let dim: usize = states.len();
        // Initialize the Exciton-Hamiltonian.
        let mut h = vec![0.0; dim * dim];

        // calculate the state couplings
        states
            .par_iter()
            .enumerate()
            .zip(h.par_chunks_exact_mut(dim))
            .for_each(|((i, state_i), h_i)| {
                states
                    .par_iter()
                    .enumerate()
                    .zip(h_i.par_iter_mut())
                    .for_each(|((j, state_j), h_ij)| {
                        if j >= i {
                            *h_ij = self.exciton_coupling(state_i, state_j);
                        }
                    });
            });

        let mut h: Array2<f64> = Array::from(h).into_shape((dim, dim)).unwrap();

        let mut energies: Array1<f64>;
        let mut eigvectors: Array2<f64>;

        // calculate all excited states
        if self.config.fmo_lc_tddftb.calculate_all_states {
            let (energies_tmp, eigvectors_tmp) = h.eigh(UPLO::Lower).unwrap();
            energies = energies_tmp;
            eigvectors = eigvectors_tmp;
        } else {
            // Use the davidson algorithm to obtain a limited number of eigenvalues
            let diag = h.diag();
            h = &h + &h.t() - Array::from_diag(&diag);
            let nroots: usize = self.config.excited.nstates;
            let guess: Array2<f64> = initial_subspace(h.diag(), nroots);
            let davidson: Davidson = Davidson::new(
                &mut h,
                guess,
                nroots,
                1e-4,
                200,
                true,
                self.config.excited.davidson_subspace_multiplier,
            )
            .unwrap();
            energies = davidson.eigenvalues;
            eigvectors = davidson.eigenvectors;
        }

        // get the number of occupied and virtual orbitals
        let n_occ: usize = self
            .monomers
            .iter()
            .map(|m| m.properties.n_occ().unwrap())
            .sum();
        let n_virt: usize = self
            .monomers
            .iter()
            .map(|m| m.properties.n_virt().unwrap())
            .sum();
        let n_orbs: usize = n_occ + n_virt;
        let mut occ_orbs: Array2<f64> = Array2::zeros([n_orbs, n_occ]);
        let mut virt_orbs: Array2<f64> = Array2::zeros([n_orbs, n_virt]);

        // get all occupide and virtual orbitals of the system
        for mol in self.monomers.iter() {
            let mol_orbs: ArrayView2<f64> = mol.properties.orbs().unwrap();
            let lumo: usize = mol.properties.lumo().unwrap();
            occ_orbs
                .slice_mut(s![mol.slice.orb, mol.slice.occ_orb])
                .assign(&mol_orbs.slice(s![.., ..lumo]));
            virt_orbs
                .slice_mut(s![mol.slice.orb, mol.slice.virt_orb])
                .assign(&mol_orbs.slice(s![.., lumo..]));
        }
        let orbs: Array2<f64> = concatenate![Axis(1), occ_orbs, virt_orbs];

        // calculate the oscillator strenghts of the excited states
        let exciton = ExcitonStates::new(
            self.properties.last_energy().unwrap(),
            (energies.clone(), eigvectors.clone()),
            states.clone(),
            (n_occ, n_virt),
            orbs,
            self.properties.s().unwrap(),
            &self.atoms,
        );
        // write the energies and oscillator strenghts to numpy files
        exciton.spectrum_to_npy("lcmo_spec.npy");
        exciton.spectrum_to_txt("lcmo_spec.txt");
        println!("{}", exciton);

        if self.config.fmo_lc_tddftb.calculate_ntos {
            exciton.calculate_ntos_jmol(&self.config.fmo_lc_tddftb.states_to_analyse, &self.atoms);
        }
        if self.config.fmo_lc_tddftb.calculate_transition_densities {
            exciton.get_transition_densities(&self.config.fmo_lc_tddftb.states_to_analyse);
        }
        if self.config.fmo_lc_tddftb.calculate_particle_hole_densities {
            exciton.get_particle_hole_densities(
                &self.config.fmo_lc_tddftb.states_to_analyse,
                self.properties.s().unwrap(),
            );
        }
    }
}

/// Different types of diabatic basis states that are used for the FMO-exciton model.
#[derive(Clone, Debug)]
pub enum BasisState<'a> {
    // Locally excited state that is on one monomer.
    LE(LocallyExcited<'a>),
    // Charge transfer state between two different monomers and two MOs.
    CT(ChargeTransfer<'a>),
    PairCT(ChargeTransferPair),
}

impl Display for BasisState<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BasisState::LE(state) => write!(f, "{}", state),
            BasisState::CT(state) => write!(f, "{}", state),
            BasisState::PairCT(state) => write!(f, "{}", state),
        }
    }
}

/// Type that holds all the relevant data that characterize a locally excited diabatic basis state.
#[derive(Clone, Debug)]
pub struct LocallyExcited<'a> {
    // Reference to the corresponding monomer.
    pub monomer: &'a Monomer<'a>,
    // Number of excited state for the monomer. 1 -> S1, 2 -> S2, ...
    pub n: usize,
    // The atoms corresponding to the monomer of this state.
    pub atoms: &'a [Atom],
    //
    pub q_trans: Array1<f64>,
    //
    pub occs: ArrayView2<'a, f64>,
    //
    pub virts: ArrayView2<'a, f64>,
    //
    pub tdm: ArrayView1<'a, f64>,
    //
    pub tr_dipole: Vector3<f64>,
    //
    pub occ_indices: Vec<usize>,
    pub virt_indices: Vec<usize>,
}

impl Display for LocallyExcited<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LE(S{}) on Frag. {:>4}",
            self.n + 1,
            self.monomer.index + 1
        )
    }
}

impl PartialEq for LocallyExcited<'_> {
    /// Two LE states are considered equal, if it is the same excited state on the same monomer.
    fn eq(&self, other: &Self) -> bool {
        self.monomer.index == other.monomer.index && self.n == other.n
    }
}

#[derive(Clone, Debug)]
pub struct PairChargeTransfer<'a> {
    pub m_h: &'a Monomer<'a>,
    pub m_l: &'a Monomer<'a>,
    pub pair_type: PairType,
    pub properties: Properties,
}

#[derive(Clone, Debug)]
pub struct ChargeTransferPair {
    pub m_h: usize,
    pub m_l: usize,
    pub state_index: usize,
    pub state_energy: f64,
    pub eigenvectors: Array2<f64>,
    pub q_tr: Array1<f64>,
    pub tr_dipole: Vector3<f64>,
    /// [Slice](ndarray::prelude::Slice) for occupied orbitals corresponding to this molecular unit
    pub occ_orb: Slice,
    /// [Slice](ndarray::prelude::Slice) for virtual orbitals corresponding to this molecular unit
    pub virt_orb: Slice,
    pub occ_indices: Vec<usize>,
    pub virt_indices: Vec<usize>,
}

impl PartialEq for ChargeTransferPair {
    fn eq(&self, other: &Self) -> bool {
        self.m_h == other.m_h && self.m_l == other.m_l && self.state_index == other.state_index
    }
}

impl Display for ChargeTransferPair {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CT {}: {} -> {}",
            self.state_index,
            self.m_h + 1,
            self.m_l + 1
        )
    }
}

/// Type that holds all the relevant data that characterize a charge-transfer diabatic basis state.
#[derive(Copy, Clone, Debug)]
pub struct ChargeTransfer<'a> {
    // // Reference to the total system. This is needed to access the complete Gamma matrix.
    // pub system: &'a SuperSystem,
    // The hole of the CT state.
    pub hole: Particle<'a>,
    // The electron of the CT state.
    pub electron: Particle<'a>,
}

impl Display for ChargeTransfer<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "CT: {} -> {}", self.hole, self.electron)
    }
}

impl PartialEq for ChargeTransfer<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.hole == other.hole && self.electron == other.electron
    }
}
#[derive(Copy, Clone, Debug)]
pub struct Particle<'a> {
    /// The index of the corresponding monomer.
    pub idx: usize,
    /// The atoms of the corresponding monomer.
    pub atoms: &'a [Atom],
    /// The corresponding monomer itself.
    pub monomer: &'a Monomer<'a>,
    /// The corresponding molecular orbital.
    pub mo: MO<'a>,
}

impl Display for Particle<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Frag. {: >4}", self.monomer.index + 1)
    }
}

// TODO: this definition could lead to mistakes for degenerate orbitals
impl PartialEq for Particle<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.mo.e == other.mo.e
    }
}
