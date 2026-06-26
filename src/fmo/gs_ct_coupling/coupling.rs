use crate::fmo::helpers::get_pair_slice;
use crate::fmo::{BasisState, ChargeTransferPair, Monomer, Pair, PairType, SuperSystem};
use crate::initialization::Atom;
use ndarray::prelude::*;
use rayon::prelude::*;

impl SuperSystem<'_> {
    pub fn calculate_gs_ct_coupling(&mut self, ct_index: usize) {
        // Calculate the H' matrix
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        // Number of LE states per monomer.
        let n_le: usize = self.config.fmo_lc_tddftb.n_le;
        let n_roots: usize = n_le + 3;

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
            mol.prepare_tda(&atoms[mol.slice.atom_as_range()], &self.config);
            mol.run_tda(
                &atoms[mol.slice.atom_as_range()],
                n_roots,
                self.config.excited.davidson_iterations,
                self.config.excited.davidson_convergence,
                self.config.excited.davidson_subspace_multiplier,
                true,
                &self.config,
            );
        });

        // Construct the basis states.
        let states: Vec<BasisState> = self.create_diabatic_basis(self.config.fmo_lc_tddftb.n_ct);
        let state = &states[ct_index];

        let _coupling: f64 = match state {
            BasisState::PairCT(ref a) => self.gs_ct_coupling(a),
            _ => 0.0,
        };
    }

    pub fn gs_ct_coupling(&self, state: &ChargeTransferPair) -> f64 {
        // get pair type
        let pair_type: PairType = self.properties.type_of_pair(state.m_h, state.m_l);

        let coupling: f64 = if pair_type == PairType::Pair {
            // get pair index
            let pair_index: usize = self.properties.index_of_pair(state.m_h, state.m_l);
            // get correct pair from pairs vector
            let mut pair_ij: Pair = self.pairs[pair_index].clone();
            // get monomers
            let m_i: &Monomer = &self.monomers[pair_ij.i];
            let m_j: &Monomer = &self.monomers[pair_ij.j];

            // check the monomer indices
            assert!(state.m_h == m_i.index);
            assert!(state.m_l == m_j.index);

            // get pair atoms
            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );
            let m_i_atoms: &[Atom] = &pair_atoms[..m_i.n_atoms];
            let m_j_atoms: &[Atom] = &pair_atoms[m_i.n_atoms..];

            // get the pair H0 matrix
            let h0: Array2<f64> = pair_ij.properties.h0().unwrap().to_owned();

            // get the tdm matrix for the CT state
            let tdm: ArrayView2<f64> = state.eigenvectors.view();
            // get the index pairs for which the transitions are above a certain threshold
            let mut indices_vec: Vec<(usize, usize)> = Vec::new();
            let threshold: f64 = 1.0e-3;
            for (idx_i, val_i) in tdm.outer_iter().enumerate() {
                for (idx_j, val_j) in val_i.iter().enumerate() {
                    let abs_c_sqr: f64 = val_j.abs().powi(2);
                    if abs_c_sqr > threshold {
                        indices_vec.push((idx_i, idx_j));
                    }
                }
            }
            // get occ and virt indices from properties
            let occ_indices: &[usize] = pair_ij.properties.occ_indices().unwrap();
            let virt_indices: &[usize] = pair_ij.properties.virt_indices().unwrap();
            let n_occ: usize = occ_indices.len();
            let n_virt: usize = virt_indices.len();

            // create a transformation matrix for the H0 matrix
            pair_ij.prepare_charge_transfer_gradient(&pair_atoms, m_i, m_j, state);
            // get the coefficients
            let coeffs: ArrayView2<f64> = pair_ij.properties.ci_coefficients().unwrap();

            // square all values of the matrix to get the real contributions
            // let coeffs_2d_sqr: Array2<f64> = coeffs
            //     .to_owned()
            //     .slice(s![0, ..])
            //     .to_owned()
            //     .into_shape([n_occ, n_virt])
            //     .unwrap()
            //     .map(|val| val.powi(2));
            let coeffs_2d: Array2<f64> = coeffs
                .to_owned()
                .slice(s![0, ..])
                .to_owned()
                .into_shape([n_occ, n_virt])
                .unwrap();

            // get the H0 matrix which corresponds to the excitation
            // get the off diagonal of the H0 matrix
            let h0_diag: ArrayView2<f64> = h0.slice(s![..n_occ, n_occ..]);
            let h0_new: Array2<f64> = &coeffs_2d * &h0_diag;
            let h0_coupling_val: f64 = h0_new.sum();

            // two electron part of the coupling
            // get the number of occupied and virtual orbitals of the monomers
            let nocc_i: usize = m_i.properties.occ_indices().unwrap().len();
            let nvirt_i: usize = m_i.properties.virt_indices().unwrap().len();

            let nocc_j: usize = m_j.properties.occ_indices().unwrap().len();
            let nvirt_j: usize = m_j.properties.virt_indices().unwrap().len();

            // get the transition charges of the monomers
            // transition charges for monomer I
            let qoo_i: ArrayView2<f64> = m_i.properties.q_oo().unwrap();
            let qov_i: ArrayView2<f64> = m_i.properties.q_ov().unwrap();
            let qvv_i: ArrayView2<f64> = m_i.properties.q_vv().unwrap();
            // let qvo_i: Array2<f64> = qov_i
            //     .to_owned()
            //     .into_shape([m_i.n_atoms, nocc_i, nvirt_i])
            //     .unwrap()
            //     .permuted_axes([0, 2, 1])
            //     .as_standard_layout()
            //     .to_owned()
            //     .into_shape([m_i.n_atoms, nocc_i * nvirt_i])
            //     .unwrap();

            // transition charges for monomer J
            let qoo_j: ArrayView2<f64> = m_j.properties.q_oo().unwrap();
            let qov_j: ArrayView2<f64> = m_j.properties.q_ov().unwrap();
            let qvv_j: ArrayView2<f64> = m_j.properties.q_vv().unwrap();
            // let qvo_j: Array2<f64> = qov_j
            //     .to_owned()
            //     .into_shape([m_j.n_atoms, nocc_j, nvirt_j])
            //     .unwrap()
            //     .permuted_axes([0, 2, 1])
            //     .as_standard_layout()
            //     .to_owned()
            //     .into_shape([m_j.n_atoms, nocc_j * nvirt_j])
            //     .unwrap();

            // get the interfragment transition charges
            let qov_ij: Array2<f64> =
                self.interfragment_q_ov(&pair_ij, m_i, m_j, m_i_atoms, m_j_atoms);
            let qvv_ij: Array2<f64> =
                self.interfragment_q_vv(&pair_ij, m_i, m_j, m_i_atoms, m_j_atoms);
            let qoo_ij: Array2<f64> =
                self.interfragment_q_vv(&pair_ij, m_i, m_j, m_i_atoms, m_j_atoms);

            // get the complete overlap and gamma matrices of the full system
            let g0_full: ArrayView2<f64> = self.properties.gamma().unwrap();
            let g0_lr_full: ArrayView2<f64> = self.properties.gamma_lr().unwrap();
            // let s_full: ArrayView2<f64> = self.properties.s().unwrap();

            // equations
            // sum_{k \in I} (ia|kk) - 1/2 (ik|ka) + \sum_{k \in J} (ia|kk) - 1/2 (ik|ka)
            // get the gamma matrices for the 4 terms
            let mut g_ij_i: Array2<f64> = Array2::zeros([pair_ij.n_atoms, m_i.n_atoms]);
            g_ij_i
                .slice_mut(s![..m_i.n_atoms, ..])
                .assign(&g0_full.slice(s![m_i.slice.atom, m_i.slice.atom]));
            g_ij_i
                .slice_mut(s![m_i.n_atoms.., ..])
                .assign(&g0_full.slice(s![m_j.slice.atom, m_i.slice.atom]));

            let mut g_ij_j: Array2<f64> = Array2::zeros([pair_ij.n_atoms, m_j.n_atoms]);
            g_ij_j
                .slice_mut(s![..m_i.n_atoms, ..])
                .assign(&g0_full.slice(s![m_i.slice.atom, m_j.slice.atom]));
            g_ij_j
                .slice_mut(s![m_i.n_atoms.., ..])
                .assign(&g0_full.slice(s![m_j.slice.atom, m_j.slice.atom]));

            let mut g_lr_i_ij: Array2<f64> = Array2::zeros([m_i.n_atoms, pair_ij.n_atoms]);
            g_lr_i_ij
                .slice_mut(s![.., ..m_i.n_atoms])
                .assign(&g0_lr_full.slice(s![m_i.slice.atom, m_i.slice.atom]));
            g_lr_i_ij
                .slice_mut(s![.., m_i.n_atoms..])
                .assign(&g0_lr_full.slice(s![m_i.slice.atom, m_j.slice.atom]));

            let mut g_lr_ij_j: Array2<f64> = Array2::zeros([pair_ij.n_atoms, m_j.n_atoms]);
            g_lr_ij_j
                .slice_mut(s![..m_i.n_atoms, ..])
                .assign(&g0_lr_full.slice(s![m_i.slice.atom, m_j.slice.atom]));
            g_lr_ij_j
                .slice_mut(s![m_i.n_atoms.., ..])
                .assign(&g0_lr_full.slice(s![m_j.slice.atom, m_j.slice.atom]));

            // get the coupling for the Coulomb integrals, where k is on I
            let coulomb_k_i_oo: Array4<f64> = qov_ij
                .t()
                .dot(&g_ij_i.dot(&qoo_i))
                .into_shape([nocc_i, nvirt_j, nocc_i, nocc_i])
                .unwrap();
            let coulomb_k_i_vv: Array4<f64> = qov_ij
                .t()
                .dot(&g_ij_i.dot(&qvv_i))
                .into_shape([nocc_i, nvirt_j, nvirt_i, nvirt_i])
                .unwrap();
            let mut coulomb_k_i: f64 = 0.0;
            for idx_i in 0..nocc_i {
                for idx_a in 0..nvirt_j {
                    for occ_idx in 0..nocc_i {
                        coulomb_k_i += coeffs_2d[[idx_i, idx_a]]
                            * coulomb_k_i_oo[[idx_i, idx_a, occ_idx, occ_idx]];
                    }
                    for virt_idx in 0..nvirt_i {
                        coulomb_k_i += coeffs_2d[[idx_i, idx_a]]
                            * coulomb_k_i_vv[[idx_i, idx_a, virt_idx, virt_idx]];
                    }
                }
            }

            // get the coupling for the Coulomb integrals, where k is on J
            let coulomb_k_j_oo: Array4<f64> = qov_ij
                .t()
                .dot(&g_ij_j.dot(&qoo_j))
                .into_shape([nocc_i, nvirt_j, nocc_i, nocc_i])
                .unwrap();
            let coulomb_k_j_vv: Array4<f64> = qov_ij
                .t()
                .dot(&g_ij_j.dot(&qvv_j))
                .into_shape([nocc_i, nvirt_j, nvirt_i, nvirt_i])
                .unwrap();
            let mut coulomb_k_j: f64 = 0.0;
            for idx_i in 0..nocc_i {
                for idx_a in 0..nvirt_j {
                    for occ_idx in 0..nocc_j {
                        coulomb_k_j += coeffs_2d[[idx_i, idx_a]]
                            * coulomb_k_j_oo[[idx_i, idx_a, occ_idx, occ_idx]];
                    }
                    for virt_idx in 0..nvirt_j {
                        coulomb_k_j += coeffs_2d[[idx_i, idx_a]]
                            * coulomb_k_j_vv[[idx_i, idx_a, virt_idx, virt_idx]];
                    }
                }
            }

            // get the exchange coupling, where k is on I
            let exchange_k_i_oo_ov: Array4<f64> = qoo_i
                .t()
                .dot(&g_lr_i_ij.dot(&qov_ij))
                .into_shape([nocc_i, nocc_i, nocc_i, nvirt_j])
                .unwrap();
            let exchange_k_i_ov_vv: Array4<f64> = qov_i
                .t()
                .dot(&g_lr_i_ij.dot(&qvv_ij))
                .into_shape([nocc_i, nvirt_i, nvirt_i, nvirt_j])
                .unwrap();
            let mut exchange_k_i: f64 = 0.0;
            for idx_i in 0..nocc_i {
                for idx_a in 0..nvirt_j {
                    for occ_idx in 0..nocc_j {
                        exchange_k_i += coeffs_2d[[idx_i, idx_a]]
                            * exchange_k_i_oo_ov[[idx_i, occ_idx, occ_idx, idx_a]];
                    }
                    for virt_idx in 0..nvirt_j {
                        exchange_k_i += coeffs_2d[[idx_i, idx_a]]
                            * exchange_k_i_ov_vv[[idx_i, virt_idx, virt_idx, idx_a]];
                    }
                }
            }

            // get the exchange coupling, where k is on J
            let exchange_k_j_oo_ov: Array4<f64> = qoo_ij
                .t()
                .dot(&g_lr_ij_j.dot(&qov_j))
                .into_shape([nocc_i, nocc_j, nocc_j, nvirt_j])
                .unwrap();
            let exchange_k_j_ov_vv: Array4<f64> = qov_ij
                .t()
                .dot(&g_lr_ij_j.dot(&qvv_j))
                .into_shape([nocc_i, nvirt_j, nvirt_j, nvirt_j])
                .unwrap();
            let mut exchange_k_j: f64 = 0.0;
            for idx_i in 0..nocc_i {
                for idx_a in 0..nvirt_j {
                    for occ_idx in 0..nocc_j {
                        exchange_k_j += coeffs_2d[[idx_i, idx_a]]
                            * exchange_k_j_oo_ov[[idx_i, occ_idx, occ_idx, idx_a]];
                    }
                    for virt_idx in 0..nvirt_j {
                        exchange_k_j += coeffs_2d[[idx_i, idx_a]]
                            * exchange_k_j_ov_vv[[idx_i, virt_idx, virt_idx, idx_a]];
                    }
                }
            }

            // combine the values
            let two_electron_gs_ct_coupling: f64 =
                coulomb_k_i + coulomb_k_j - 0.5 * exchange_k_i - 0.5 * exchange_k_j;
            let gs_ct_coupling: f64 = h0_coupling_val + two_electron_gs_ct_coupling;
            println!("h0 gs ct coupling: {:.9}", h0_coupling_val);
            println!(
                "two electron integral gs ct coupling: {:.9}",
                two_electron_gs_ct_coupling
            );
            println!("Coupling val: {:.9}", gs_ct_coupling);

            gs_ct_coupling
        } else {
            0.0
        };

        coupling
    }

    fn interfragment_q_ov(
        &self,
        _pair: &Pair,
        m_i: &Monomer,
        m_j: &Monomer,
        atoms_h: &[Atom],
        atoms_l: &[Atom],
    ) -> Array2<f64> {
        // indices of the occupied and virtual orbitals of the CT state
        let occ_indices: &[usize] = m_i.properties.occ_indices().unwrap();
        let virt_indices: &[usize] = m_j.properties.virt_indices().unwrap();

        // get overlap
        let s_full: ArrayView2<f64> = self.properties.s().unwrap();
        let s: ArrayView2<f64> = s_full.slice(s![m_i.slice.orb, m_j.slice.orb]);

        // The index of the HOMO (zero based).
        let homo: usize = occ_indices[occ_indices.len() - 1];
        // The index of the LUMO (zero based).
        let lumo: usize = virt_indices[0];

        let occs = m_i
            .properties
            .orbs_slice(0, Some(homo + 1))
            .unwrap()
            .to_owned();
        let virts = m_j.properties.orbs_slice(lumo, None).unwrap().to_owned();

        // Matrix product of overlap matrix with the orbitals on L.
        let s_c_l: Array2<f64> = s.dot(&virts);
        // Matrix product of overlap matrix with the orbitals on H.
        let s_c_h: Array2<f64> = s.t().dot(&occs);
        // Number of molecular orbitals on monomer I.
        let dim_h: usize = occs.ncols();
        // Number of molecular orbitals on monomer J.
        let dim_l: usize = virts.ncols();
        // get the number of atoms
        let natoms_h: usize = atoms_h.len();
        let natoms_l: usize = atoms_l.len();
        let n_atoms: usize = natoms_h + natoms_l;
        // The transition charges between the two sets of MOs  are initialized.
        let mut q_trans: Array3<f64> = Array3::zeros([n_atoms, dim_h, dim_l]);

        let mut mu: usize = 0;
        for (atom_h, mut q_n) in atoms_h.iter().zip(
            q_trans
                .slice_mut(s![0..natoms_h, .., ..])
                .axis_iter_mut(Axis(0)),
        ) {
            for _ in 0..atom_h.n_orbs {
                for (orb_h, mut q_h) in occs.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                    for (sc, q) in s_c_l.row(mu).iter().zip(q_h.iter_mut()) {
                        *q += orb_h * sc;
                    }
                }
                mu += 1;
            }
        }
        mu = 0;
        for (atom_l, mut q_n) in atoms_l.iter().zip(
            q_trans
                .slice_mut(s![natoms_h.., .., ..])
                .axis_iter_mut(Axis(0)),
        ) {
            for _ in 0..atom_l.n_orbs {
                for (sc, mut q_l) in s_c_h.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                    for (orb_l, q) in virts.row(mu).iter().zip(q_l.iter_mut()) {
                        *q += orb_l * sc;
                    }
                }
                mu += 1;
            }
        }
        q_trans = 0.5 * q_trans;
        q_trans.into_shape([n_atoms, dim_h * dim_l]).unwrap()
    }

    fn interfragment_q_oo(
        &self,
        _pair: &Pair,
        m_i: &Monomer,
        m_j: &Monomer,
        atoms_h: &[Atom],
        atoms_l: &[Atom],
    ) -> Array2<f64> {
        // indices of the occupied and virtual orbitals of the CT state
        let occ_indices: &[usize] = m_i.properties.occ_indices().unwrap();
        let occ_indices2: &[usize] = m_j.properties.occ_indices().unwrap();

        // get overlap
        let s_full: ArrayView2<f64> = self.properties.s().unwrap();
        let s: ArrayView2<f64> = s_full.slice(s![m_i.slice.orb, m_j.slice.orb]);

        // The index of the HOMO (zero based).
        let homo: usize = occ_indices[occ_indices.len() - 1];
        // The index of the LUMO (zero based).
        let homo2: usize = occ_indices2[occ_indices2.len() - 1];

        let occs = m_i
            .properties
            .orbs_slice(0, Some(homo + 1))
            .unwrap()
            .to_owned();
        let occs2 = m_j
            .properties
            .orbs_slice(0, Some(homo2 + 1))
            .unwrap()
            .to_owned();

        // Matrix product of overlap matrix with the orbitals on L.
        let s_c_h2: Array2<f64> = s.dot(&occs2);
        // Matrix product of overlap matrix with the orbitals on H.
        let s_c_h: Array2<f64> = s.t().dot(&occs);
        // Number of molecular orbitals on monomer I.
        let dim_h: usize = occs.ncols();
        // Number of molecular orbitals on monomer J.
        let dim_h2: usize = occs2.ncols();
        // get the number of atoms
        let natoms_h: usize = atoms_h.len();
        let natoms_l: usize = atoms_l.len();
        let n_atoms: usize = natoms_h + natoms_l;
        // The transition charges between the two sets of MOs  are initialized.
        let mut q_trans: Array3<f64> = Array3::zeros([n_atoms, dim_h, dim_h2]);

        let mut mu: usize = 0;
        for (atom_h, mut q_n) in atoms_h.iter().zip(
            q_trans
                .slice_mut(s![0..natoms_h, .., ..])
                .axis_iter_mut(Axis(0)),
        ) {
            for _ in 0..atom_h.n_orbs {
                for (orb_h, mut q_h) in occs.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                    for (sc, q) in s_c_h2.row(mu).iter().zip(q_h.iter_mut()) {
                        *q += orb_h * sc;
                    }
                }
                mu += 1;
            }
        }
        mu = 0;
        for (atom_l, mut q_n) in atoms_l.iter().zip(
            q_trans
                .slice_mut(s![natoms_h.., .., ..])
                .axis_iter_mut(Axis(0)),
        ) {
            for _ in 0..atom_l.n_orbs {
                for (sc, mut q_l) in s_c_h.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                    for (orb_l, q) in occs2.row(mu).iter().zip(q_l.iter_mut()) {
                        *q += orb_l * sc;
                    }
                }
                mu += 1;
            }
        }
        q_trans = 0.5 * q_trans;
        q_trans.into_shape([n_atoms, dim_h * dim_h]).unwrap()
    }

    fn interfragment_q_vv(
        &self,
        _pair: &Pair,
        m_i: &Monomer,
        m_j: &Monomer,
        atoms_h: &[Atom],
        atoms_l: &[Atom],
    ) -> Array2<f64> {
        // indices of the occupied and virtual orbitals of the CT state
        // let virt_indices2: &[usize] = m_i.properties.virt_indices().unwrap();
        let virt_indices: &[usize] = m_j.properties.virt_indices().unwrap();

        // get overlap
        let s_full: ArrayView2<f64> = self.properties.s().unwrap();
        let s: ArrayView2<f64> = s_full.slice(s![m_i.slice.orb, m_j.slice.orb]);

        // The index of the HOMO (zero based).
        // let lumo2: usize = virt_indices2[0];
        // The index of the LUMO (zero based).
        let lumo: usize = virt_indices[0];

        // virtual MO coeffs
        let virts2 = m_i.properties.orbs_slice(lumo, None).unwrap().to_owned();
        let virts = m_j.properties.orbs_slice(lumo, None).unwrap().to_owned();

        // Matrix product of overlap matrix with the orbitals on L.
        let s_c_l: Array2<f64> = s.dot(&virts);
        // Matrix product of overlap matrix with the orbitals on H.
        let s_c_l2: Array2<f64> = s.t().dot(&virts2);
        // Number of molecular orbitals on monomer I.
        let dim_l2: usize = virts2.ncols();
        // Number of molecular orbitals on monomer J.
        let dim_l: usize = virts.ncols();
        // get the number of atoms
        let natoms_h: usize = atoms_h.len();
        let natoms_l: usize = atoms_l.len();
        let n_atoms: usize = natoms_h + natoms_l;
        // The transition charges between the two sets of MOs  are initialized.
        let mut q_trans: Array3<f64> = Array3::zeros([n_atoms, dim_l2, dim_l]);

        let mut mu: usize = 0;
        for (atom_h, mut q_n) in atoms_h.iter().zip(
            q_trans
                .slice_mut(s![0..natoms_h, .., ..])
                .axis_iter_mut(Axis(0)),
        ) {
            for _ in 0..atom_h.n_orbs {
                for (orb_h, mut q_h) in virts2.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                    for (sc, q) in s_c_l.row(mu).iter().zip(q_h.iter_mut()) {
                        *q += orb_h * sc;
                    }
                }
                mu += 1;
            }
        }
        mu = 0;
        for (atom_l, mut q_n) in atoms_l.iter().zip(
            q_trans
                .slice_mut(s![natoms_h.., .., ..])
                .axis_iter_mut(Axis(0)),
        ) {
            for _ in 0..atom_l.n_orbs {
                for (sc, mut q_l) in s_c_l2.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                    for (orb_l, q) in virts.row(mu).iter().zip(q_l.iter_mut()) {
                        *q += orb_l * sc;
                    }
                }
                mu += 1;
            }
        }
        q_trans = 0.5 * q_trans;
        q_trans.into_shape([n_atoms, dim_l2 * dim_l]).unwrap()
    }
}
