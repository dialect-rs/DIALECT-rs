use crate::fmo::lcmo::lcmo_trans_charges::ElecHole;
use crate::fmo::{
    BasisState, ChargeTransferPair, LocallyExcited, Monomer, PairType, SuperSystem, LRC,
};
use ndarray::prelude::*;
use ndarray::Slice;
use ndarray_linalg::Trace;

impl SuperSystem<'_> {
    pub fn exciton_coupling<'a>(&self, lhs: &'a BasisState<'a>, rhs: &'a BasisState<'a>) -> f64 {
        match (lhs, rhs) {
            // Coupling between two LE states.
            (BasisState::LE(ref a), BasisState::LE(ref b)) => {
                if a == b {
                    a.monomer.properties.ci_eigenvalue(a.n).unwrap()
                } else if a.monomer == b.monomer {
                    0.0
                } else {
                    self.le_le(a, b)
                }
            }
            (BasisState::LE(ref a), BasisState::PairCT(ref b)) => self.le_ct(a, b),
            (BasisState::PairCT(ref a), BasisState::LE(ref b)) => self.ct_le(a, b),
            (BasisState::PairCT(ref a), BasisState::PairCT(ref b)) => {
                if a == b {
                    a.state_energy
                } else if a.m_h == b.m_h && a.m_l == b.m_l {
                    0.0
                } else {
                    self.ct_ct(a, b)
                }
            }
        }
    }

    pub fn le_le<'a>(&self, i: &'a LocallyExcited<'a>, j: &'a LocallyExcited<'a>) -> f64 {
        // Check if the ESD approximation is used or not.
        let type_pair: PairType = self
            .properties
            .type_of_pair(i.monomer.index, j.monomer.index);

        // Slices of atoms of I and J.
        let (atoms_i, atoms_j): (Slice, Slice) = (i.monomer.slice.atom, j.monomer.slice.atom);

        // Get the gamma matrix between both sets of atoms.
        let gamma_ab: ArrayView2<f64> = if !self.config.use_shell_resolved_gamma {
            self.properties.gamma_slice(atoms_i, atoms_j).unwrap()
        } else {
            self.properties
                .gamma_ao_slice(i.monomer.slice.orb, j.monomer.slice.orb)
                .unwrap()
        };

        // Compute the Coulomb interaction between both LE states.
        let coulomb: f64 = i.q_trans.dot(&gamma_ab.dot(&j.q_trans));

        // For the exchange energy, the transition charges between both sets of orbitals are needed.
        // In the case that the monomers are far apart approximation is used,
        // the Exchange coupling is zero.
        let mut exchange_val: f64 = 0.0;
        if self.config.lc.long_range_correction {
            let exchange: f64 = match type_pair {
                PairType::ESD => 0.0,
                PairType::Pair => {
                    // Reference to the overlap matrix between both sets of orbitals.
                    let s_ab: ArrayView2<f64> = self
                        .properties
                        .s()
                        .unwrap()
                        .slice_move(s![i.monomer.slice.orb, j.monomer.slice.orb]);

                    // The transition charges between both sets of occupied orbitals are computed. .
                    let q_ij: Array3<f64> = if !self.config.use_shell_resolved_gamma {
                        self.q_lele(i, j, ElecHole::Hole, ElecHole::Hole, s_ab.view())
                    } else {
                        self.q_lele_ao(i, j, ElecHole::Hole, ElecHole::Hole, s_ab.view())
                    };

                    // Transition charges between both sets of virtual orbitals are computed.
                    let q_ab: Array3<f64> = if !self.config.use_shell_resolved_gamma {
                        self.q_lele(i, j, ElecHole::Electron, ElecHole::Electron, s_ab.view())
                    } else {
                        self.q_lele_ao(i, j, ElecHole::Electron, ElecHole::Electron, s_ab.view())
                    };

                    // Reference to the transition density matrix of I in MO basis.
                    let b_ia: Array2<f64> = if self.config.fmo_lc_tddftb.restrict_active_space {
                        let eigenvectors: ArrayView2<f64> = i.monomer.properties.tdm(i.n).unwrap();
                        let dim_occ: usize = i.occ_indices.len();
                        let dim_virt: usize = i.virt_indices.len();
                        let mut arr: Array2<f64> = Array2::zeros((dim_occ, dim_virt));
                        for (en_idx, idx_i) in i.occ_indices.iter().enumerate() {
                            for (en_idx2, idx_a) in i.virt_indices.iter().enumerate() {
                                arr[[en_idx, en_idx2]] = eigenvectors[[*idx_i, *idx_a]]
                            }
                        }
                        arr
                    } else {
                        i.monomer.properties.tdm(i.n).unwrap().to_owned()
                    };

                    // Reference to the transition density matrix of J in MO basis.
                    let b_jb: Array2<f64> = if self.config.fmo_lc_tddftb.restrict_active_space {
                        let eigenvectors: ArrayView2<f64> = j.monomer.properties.tdm(j.n).unwrap();
                        let dim_occ: usize = j.occ_indices.len();
                        let dim_virt: usize = j.virt_indices.len();
                        let mut arr: Array2<f64> = Array2::zeros((dim_occ, dim_virt));
                        for (en_idx, idx_j) in j.occ_indices.iter().enumerate() {
                            for (en_idx2, idx_b) in j.virt_indices.iter().enumerate() {
                                arr[[en_idx, en_idx2]] = eigenvectors[[*idx_j, *idx_b]]
                            }
                        }
                        arr
                    } else {
                        j.monomer.properties.tdm(j.n).unwrap().to_owned()
                    };

                    // Some properties that are used specify the shapes.
                    let n_atoms: usize = if !self.config.use_shell_resolved_gamma {
                        i.monomer.n_atoms + j.monomer.n_atoms
                    } else {
                        i.monomer.n_orbs + j.monomer.n_orbs
                    };
                    // Number of occupied orbitals in both monomers.
                    let (n_i, n_j): (usize, usize) = (q_ij.dim().1, q_ij.dim().2);
                    // Number of virtual orbitals in both monomers.
                    let (n_a, n_b): (usize, usize) = (q_ab.dim().1, q_ab.dim().2);

                    // The lrc-Gamma matrix of the dimer. TODO: WHICH GAMMA IS NEEDED HERE???
                    let gamma_lc_ab: Array2<f64> = if !self.config.use_shell_resolved_gamma {
                        self.gamma_ab_cd(
                            i.monomer.index,
                            j.monomer.index,
                            i.monomer.index,
                            j.monomer.index,
                            LRC::ON,
                        )
                    } else {
                        self.gamma_ab_cd_ao(
                            i.monomer.index,
                            j.monomer.index,
                            i.monomer.index,
                            j.monomer.index,
                            LRC::ON,
                        )
                    };

                    // Contract the product b_ia^I b_jb^J ( i^I j^J | a^I b^J)
                    let bia_ij = q_ij
                        .permuted_axes([0, 2, 1])
                        .as_standard_layout()
                        .into_shape((n_atoms * n_j, n_i))
                        .unwrap()
                        .dot(&b_ia)
                        .into_shape((n_atoms, n_j, n_a))
                        .unwrap()
                        .permuted_axes([0, 2, 1])
                        .as_standard_layout()
                        .into_shape((n_atoms, n_a * n_j))
                        .unwrap()
                        .to_owned();

                    let ab_bjb: Array2<f64> = q_ab
                        .into_shape([n_atoms * n_a, n_b])
                        .unwrap()
                        .dot(&b_jb.t())
                        .into_shape([n_atoms, n_a * n_j])
                        .unwrap();

                    ab_bjb.dot(&bia_ij.t()).dot(&gamma_lc_ab).trace().unwrap()
                }
                PairType::None => 0.0,
            };
            exchange_val = exchange;
        }

        2.0 * coulomb - exchange_val
    }

    pub fn le_ct<'a>(&self, i: &'a LocallyExcited<'a>, j: &ChargeTransferPair) -> f64 {
        self.le_ct_1e(i, j) + self.le_ct_2e(i, j)
    }

    pub fn le_ct_1e<'a>(&self, i: &'a LocallyExcited<'a>, j: &ChargeTransferPair) -> f64 {
        if i.monomer.index == j.m_l {
            // reference to the Monomer of the CT where the hole is placed
            let m_h: &Monomer = &self.monomers[j.m_h];

            // Transition Density Matrix of the LE state in MO basis.
            let tdm_le: ArrayView2<f64> = i.monomer.properties.tdm(i.n).unwrap();

            // Transition Density Matrix of the CT state
            let tdm_ct: &Array2<f64> = &j.eigenvectors;

            // Index of the HOMO of the LE.
            let homo_le: usize = i.monomer.properties.homo().unwrap();

            // Index of the HOMO of the CT
            let occ_indices_ct = m_h.properties.occ_indices().unwrap();
            let homo_ct: usize = occ_indices_ct[occ_indices_ct.len() - 1];

            let f_ij: ArrayView2<f64> = self
                .properties
                .lcmo_fock()
                .unwrap()
                .slice_move(s![i.monomer.slice.orb, m_h.slice.orb])
                .slice_move(s![..=homo_le, ..=homo_ct]);

            let nocc_i: usize = f_ij.dim().0;
            let nocc_j: usize = f_ij.dim().1;

            let t_ij: Array1<f64> = tdm_le
                .dot(&tdm_ct.t())
                .into_shape([nocc_i * nocc_j])
                .unwrap();

            -1.0 * f_ij
                .as_standard_layout()
                .to_owned()
                .into_shape([nocc_i * nocc_j])
                .unwrap()
                .dot(&t_ij)
        } else if i.monomer.index == j.m_h {
            // reference to the Monomer of the CT where the electron is placed
            let m_l: &Monomer = &self.monomers[j.m_l];

            // Transition Density Matrix of the LE state in MO basis.
            let tdm_le: ArrayView2<f64> = i.monomer.properties.tdm(i.n).unwrap();

            // Transition Density Matrix of the CT state
            let tdm_ct: &Array2<f64> = &j.eigenvectors;

            // Index of the LUMO of the LE
            let lumo_le: usize = i.monomer.properties.lumo().unwrap();

            // Index of the LUMO of the CT
            let lumo_ct: usize = m_l.properties.virt_indices().unwrap()[0];

            let f_ab: ArrayView2<f64> = self
                .properties
                .lcmo_fock()
                .unwrap()
                .slice_move(s![i.monomer.slice.orb, m_l.slice.orb])
                .slice_move(s![lumo_le.., lumo_ct..]);

            let nvirt_a: usize = f_ab.dim().0;
            let nvirt_b: usize = f_ab.dim().1;

            let t_ab: Array1<f64> = tdm_le
                .t()
                .dot(tdm_ct)
                .into_shape([nvirt_a * nvirt_b])
                .unwrap();
            f_ab.as_standard_layout()
                .to_owned()
                .into_shape([nvirt_a * nvirt_b])
                .unwrap()
                .dot(&t_ab)
        } else {
            0.0
        }
    }

    pub fn le_ct_2e<'a>(&self, i: &'a LocallyExcited<'a>, j: &ChargeTransferPair) -> f64 {
        // Transition charges of LE state at monomer I.
        let qtrans_le: ArrayView1<f64> = i.q_trans.view();
        let restrict_space: bool = self.config.fmo_lc_tddftb.restrict_active_space;

        // calculate the gamma matrix between the three monomers
        let gamma = if !self.config.use_shell_resolved_gamma {
            self.gamma_ab_c(j.m_h, j.m_l, i.monomer.index, LRC::OFF)
        } else {
            self.gamma_ab_c_ao(j.m_h, j.m_l, i.monomer.index, LRC::OFF)
        };

        // calculate the coulomb interaction between both charge densities
        let coulomb: f64 = qtrans_le.dot(&gamma.t().dot(&j.q_tr));

        let type_le_h: PairType = self.properties.type_of_pair(i.monomer.index, j.m_h);
        let type_le_l: PairType = self.properties.type_of_pair(i.monomer.index, j.m_l);

        let mut exchange_val: f64 = 0.0;
        if self.config.lc.long_range_correction {
            let exchange: f64 = if type_le_h == PairType::Pair
                || type_le_l == PairType::Pair
                || type_le_h == PairType::None
                || type_le_l == PairType::None
            {
                let q_ij = if i.monomer.index == j.m_h {
                    let q_oo = i.monomer.properties.q_oo().unwrap();
                    let nocc: usize = i.occs.ncols();
                    let n_dim: usize = if !self.config.use_shell_resolved_gamma {
                        i.monomer.n_atoms
                    } else {
                        i.monomer.n_orbs
                    };

                    if restrict_space {
                        let q_oo_3d: Array3<f64> =
                            q_oo.into_shape([n_dim, nocc, nocc]).unwrap().to_owned();
                        let dim_occ: usize = i.occ_indices.len();
                        let dim_occ2: usize = j.occ_indices.len();
                        let mut arr: Array3<f64> = Array3::zeros((n_dim, dim_occ, dim_occ2));
                        for (en_idx, idx_o) in i.occ_indices.iter().enumerate() {
                            for (en_idx2, idx_o2) in j.occ_indices.iter().enumerate() {
                                arr.slice_mut(s![.., en_idx, en_idx2])
                                    .assign(&q_oo_3d.slice(s![.., *idx_o, *idx_o2]));
                            }
                        }
                        arr
                    } else {
                        q_oo.into_shape([n_dim, nocc, nocc]).unwrap().to_owned()
                    }
                } else if !self.config.use_shell_resolved_gamma {
                    self.q_lect(i, j, ElecHole::Hole)
                } else {
                    self.q_lect_ao(i, j, ElecHole::Hole)
                };

                let q_ab = if i.monomer.index == j.m_l {
                    let q_vv = i.monomer.properties.q_vv().unwrap();
                    let nvirt: usize = i.virts.ncols();
                    let n_dim: usize = if !self.config.use_shell_resolved_gamma {
                        i.monomer.n_atoms
                    } else {
                        i.monomer.n_orbs
                    };

                    if restrict_space {
                        let q_vv_3d: Array3<f64> =
                            q_vv.into_shape([n_dim, nvirt, nvirt]).unwrap().to_owned();

                        let dim_virt: usize = i.virt_indices.len();
                        let dim_virt2: usize = j.virt_indices.len();
                        let mut arr: Array3<f64> = Array3::zeros((n_dim, dim_virt, dim_virt2));
                        for (en_idx, idx_v) in i.virt_indices.iter().enumerate() {
                            for (en_idx2, idx_v2) in j.virt_indices.iter().enumerate() {
                                arr.slice_mut(s![.., en_idx, en_idx2])
                                    .assign(&q_vv_3d.slice(s![.., *idx_v, *idx_v2]));
                            }
                        }
                        arr
                    } else {
                        q_vv.into_shape([n_dim, nvirt, nvirt]).unwrap().to_owned()
                    }
                } else if !self.config.use_shell_resolved_gamma {
                    self.q_lect(i, j, ElecHole::Electron)
                } else {
                    self.q_lect_ao(i, j, ElecHole::Electron)
                };

                let gamma_lr = if i.monomer.index == j.m_h {
                    let gamma = if !self.config.use_shell_resolved_gamma {
                        self.gamma_ab_c(i.monomer.index, j.m_l, i.monomer.index, LRC::ON)
                    } else {
                        self.gamma_ab_c_ao(i.monomer.index, j.m_l, i.monomer.index, LRC::ON)
                    };
                    gamma.reversed_axes()
                } else if i.monomer.index == j.m_l {
                    if !self.config.use_shell_resolved_gamma {
                        self.gamma_ab_c(i.monomer.index, j.m_h, i.monomer.index, LRC::ON)
                    } else {
                        self.gamma_ab_c_ao(i.monomer.index, j.m_h, i.monomer.index, LRC::ON)
                    }
                } else if !self.config.use_shell_resolved_gamma {
                    self.gamma_ab_cd(i.monomer.index, j.m_h, i.monomer.index, j.m_l, LRC::ON)
                } else {
                    self.gamma_ab_cd_ao(i.monomer.index, j.m_h, i.monomer.index, j.m_l, LRC::ON)
                };

                // Reference to the transition density matrix of I in MO basis.
                let b_ia = if restrict_space {
                    let eigenvectors = i.monomer.properties.tdm(i.n).unwrap();
                    let dim_occ: usize = i.occ_indices.len();
                    let dim_virt: usize = i.virt_indices.len();
                    let mut arr: Array2<f64> = Array2::zeros((dim_occ, dim_virt));
                    for (en_idx, idx_i) in i.occ_indices.iter().enumerate() {
                        for (en_idx2, idx_a) in i.virt_indices.iter().enumerate() {
                            arr[[en_idx, en_idx2]] = eigenvectors[[*idx_i, *idx_a]]
                        }
                    }
                    arr
                } else {
                    i.monomer.properties.tdm(i.n).unwrap().to_owned()
                };

                // Reference to the transition density matrix of J in MO basis.
                let b_jb = if restrict_space {
                    let eigenvectors = j.eigenvectors.view();
                    let dim_occ: usize = j.occ_indices.len();
                    let dim_virt: usize = j.virt_indices.len();
                    let mut arr: Array2<f64> = Array2::zeros((dim_occ, dim_virt));
                    for (en_idx, idx_i) in j.occ_indices.iter().enumerate() {
                        for (en_idx2, idx_a) in j.virt_indices.iter().enumerate() {
                            arr[[en_idx, en_idx2]] = eigenvectors[[*idx_i, *idx_a]]
                        }
                    }
                    arr
                } else {
                    j.eigenvectors.clone()
                };

                // Some properties that are used specify the shapes.
                let n_atoms_ij: usize = q_ij.dim().0;
                let n_atoms_ab: usize = q_ab.dim().0;
                // Number of occupied orbitals in both monomers.
                let (n_i, n_j): (usize, usize) = (q_ij.dim().1, q_ij.dim().2);
                // Number of virtual orbitals in both monomers.
                let (n_a, n_b): (usize, usize) = (q_ab.dim().1, q_ab.dim().2);

                // Contract the product b_ia^I b_jb^J ( i^I j^J | a^I b^J)
                let bia_ij = q_ij
                    .permuted_axes([0, 2, 1])
                    .as_standard_layout()
                    .into_shape((n_atoms_ij * n_j, n_i))
                    .unwrap()
                    .dot(&b_ia)
                    .into_shape((n_atoms_ij, n_j, n_a))
                    .unwrap()
                    .permuted_axes([0, 2, 1])
                    .as_standard_layout()
                    .into_shape((n_atoms_ij, n_a * n_j))
                    .unwrap()
                    .to_owned();

                let ab_bjb: Array2<f64> = q_ab
                    .into_shape([n_atoms_ab * n_a, n_b])
                    .unwrap()
                    .dot(&b_jb.t())
                    .into_shape([n_atoms_ab, n_a * n_j])
                    .unwrap();

                ab_bjb.dot(&bia_ij.t()).dot(&gamma_lr).trace().unwrap()
            } else {
                0.0
            };

            exchange_val = exchange;
        }

        2.0 * coulomb - exchange_val
    }

    pub fn ct_le<'a>(&self, i: &ChargeTransferPair, j: &'a LocallyExcited<'a>) -> f64 {
        self.le_ct(j, i)
    }

    pub fn ct_ct(&self, state_1: &ChargeTransferPair, state_2: &ChargeTransferPair) -> f64 {
        // calculate the gamma matrix between the two pairs
        let gamma_ij_kl: Array2<f64> = if !self.config.use_shell_resolved_gamma {
            self.gamma_ab_cd(state_1.m_h, state_1.m_l, state_2.m_h, state_2.m_l, LRC::OFF)
        } else {
            self.gamma_ab_cd_ao(state_1.m_h, state_1.m_l, state_2.m_h, state_2.m_l, LRC::OFF)
        };

        // calculate the coulomb interaction between both charge densities
        let coulomb: f64 = state_1.q_tr.dot(&gamma_ij_kl.dot(&state_2.q_tr));

        // get all possible pair types of the monomers
        let type_hh: PairType = self.properties.type_of_pair(state_1.m_h, state_2.m_h);
        let type_ll: PairType = self.properties.type_of_pair(state_1.m_l, state_2.m_l);
        // bool for the restriction of the active space for the CT calculation
        let restrict_space: bool = self.config.fmo_lc_tddftb.restrict_active_space;

        let mut exchange_val: f64 = 0.0;
        if self.config.lc.long_range_correction {
            // calculate the exchange like integral in case one of the pair type is a real pair
            let exchange: f64 = if (type_hh == PairType::Pair && type_ll == PairType::Pair)
                || (type_hh == PairType::None && type_ll == PairType::None)
                || (type_hh == PairType::Pair && type_ll == PairType::None)
                || (type_hh == PairType::None && type_ll == PairType::Pair)
            {
                let q_ij = if state_1.m_h == state_2.m_h {
                    let nocc: usize = self.monomers[state_1.m_h]
                        .properties
                        .occ_indices()
                        .unwrap()
                        .len();
                    let q_oo = self.monomers[state_1.m_h].properties.q_oo().unwrap();
                    let n_dim: usize = if !self.config.use_shell_resolved_gamma {
                        self.monomers[state_1.m_h].n_atoms
                    } else {
                        self.monomers[state_1.m_h].n_orbs
                    };

                    if restrict_space {
                        let q_oo_arr: ArrayView3<f64> =
                            q_oo.into_shape([n_dim, nocc, nocc]).unwrap();
                        // let start:usize = nocc-active_ct;
                        let dim_occ: usize = state_1.occ_indices.len();
                        let dim_occ2: usize = state_2.occ_indices.len();
                        let mut arr: Array3<f64> = Array3::zeros((n_dim, dim_occ, dim_occ2));
                        for (en_idx, idx_o) in state_1.occ_indices.iter().enumerate() {
                            for (en_idx2, idx_o2) in state_2.occ_indices.iter().enumerate() {
                                arr.slice_mut(s![.., en_idx, en_idx2])
                                    .assign(&q_oo_arr.slice(s![.., *idx_o, *idx_o2]));
                            }
                        }
                        arr
                    } else {
                        q_oo.into_shape([n_dim, nocc, nocc]).unwrap().to_owned()
                    }
                } else if !self.config.use_shell_resolved_gamma {
                    self.q_ctct(state_1, state_2, ElecHole::Hole)
                } else {
                    self.q_ctct_ao(state_1, state_2, ElecHole::Hole)
                };

                let q_ab: Array3<f64> = if state_1.m_l == state_2.m_l {
                    let nvirt: usize = self.monomers[state_1.m_l]
                        .properties
                        .virt_indices()
                        .unwrap()
                        .len();
                    let q_vv = self.monomers[state_1.m_l].properties.q_vv().unwrap();
                    let n_dim: usize = if !self.config.use_shell_resolved_gamma {
                        self.monomers[state_1.m_l].n_atoms
                    } else {
                        self.monomers[state_1.m_l].n_orbs
                    };

                    if restrict_space {
                        let q_vv_arr: Array3<f64> =
                            q_vv.into_shape([n_dim, nvirt, nvirt]).unwrap().to_owned();
                        let dim_virt: usize = state_1.virt_indices.len();
                        let dim_virt2: usize = state_2.virt_indices.len();
                        let mut arr: Array3<f64> = Array3::zeros((n_dim, dim_virt, dim_virt2));
                        for (en_idx, idx_v) in state_1.virt_indices.iter().enumerate() {
                            for (en_idx2, idx_v2) in state_2.virt_indices.iter().enumerate() {
                                arr.slice_mut(s![.., en_idx, en_idx2])
                                    .assign(&q_vv_arr.slice(s![.., *idx_v, *idx_v2]));
                            }
                        }
                        arr
                    } else {
                        q_vv.into_shape([n_dim, nvirt, nvirt]).unwrap().to_owned()
                    }
                } else if !self.config.use_shell_resolved_gamma {
                    self.q_ctct(state_1, state_2, ElecHole::Electron)
                } else {
                    self.q_ctct_ao(state_1, state_2, ElecHole::Electron)
                };

                let gamma_lr = if state_1.m_h == state_2.m_h {
                    let gamma = if !self.config.use_shell_resolved_gamma {
                        self.gamma_ab_c(state_1.m_l, state_2.m_l, state_1.m_h, LRC::ON)
                    } else {
                        self.gamma_ab_c_ao(state_1.m_l, state_2.m_l, state_1.m_h, LRC::ON)
                    };
                    gamma.reversed_axes()
                } else if state_1.m_l == state_2.m_l {
                    if !self.config.use_shell_resolved_gamma {
                        self.gamma_ab_c(state_1.m_h, state_2.m_h, state_1.m_l, LRC::ON)
                    } else {
                        self.gamma_ab_c_ao(state_1.m_h, state_2.m_h, state_1.m_l, LRC::ON)
                    }
                } else if !self.config.use_shell_resolved_gamma {
                    self.gamma_ab_cd(state_1.m_h, state_2.m_h, state_1.m_l, state_2.m_l, LRC::ON)
                } else {
                    self.gamma_ab_cd_ao(state_1.m_h, state_2.m_h, state_1.m_l, state_2.m_l, LRC::ON)
                };

                // Reference to the transition density matrix of the CT 1
                let b_ia = if restrict_space {
                    let eigenvectors = state_1.eigenvectors.view();
                    let dim_occ: usize = state_1.occ_indices.len();
                    let dim_virt: usize = state_1.virt_indices.len();
                    let mut arr: Array2<f64> = Array2::zeros((dim_occ, dim_virt));
                    for (en_idx, idx_i) in state_1.occ_indices.iter().enumerate() {
                        for (en_idx2, idx_a) in state_1.virt_indices.iter().enumerate() {
                            arr[[en_idx, en_idx2]] = eigenvectors[[*idx_i, *idx_a]]
                        }
                    }
                    arr
                } else {
                    state_1.eigenvectors.clone()
                };
                // Reference to the transition density matrix of the CT 2.
                let b_jb = if restrict_space {
                    let eigenvectors = state_2.eigenvectors.view();
                    let dim_occ: usize = state_2.occ_indices.len();
                    let dim_virt: usize = state_2.virt_indices.len();
                    let mut arr: Array2<f64> = Array2::zeros((dim_occ, dim_virt));
                    for (en_idx, idx_i) in state_2.occ_indices.iter().enumerate() {
                        for (en_idx2, idx_a) in state_2.virt_indices.iter().enumerate() {
                            arr[[en_idx, en_idx2]] = eigenvectors[[*idx_i, *idx_a]]
                        }
                    }
                    arr
                } else {
                    state_2.eigenvectors.clone()
                };

                // Some properties that are used specify the shapes.
                let n_atoms_ij: usize = q_ij.dim().0;
                let n_atoms_ab: usize = q_ab.dim().0;

                // Number of occupied orbitals in both monomers.
                let (n_i, n_j): (usize, usize) = (q_ij.dim().1, q_ij.dim().2);
                // Number of virtual orbitals in both monomers.
                let (n_a, n_b): (usize, usize) = (q_ab.dim().1, q_ab.dim().2);

                // Contract the product b_ia^I b_jb^J ( i^I j^J | a^I b^J)
                let bia_ij = q_ij
                    .permuted_axes([0, 2, 1])
                    .as_standard_layout()
                    .into_shape((n_atoms_ij * n_j, n_i))
                    .unwrap()
                    .dot(&b_ia)
                    .into_shape((n_atoms_ij, n_j, n_a))
                    .unwrap()
                    .permuted_axes([0, 2, 1])
                    .as_standard_layout()
                    .into_shape((n_atoms_ij, n_a * n_j))
                    .unwrap()
                    .to_owned();

                let ab_bjb: Array2<f64> = q_ab
                    .into_shape([n_atoms_ab * n_a, n_b])
                    .unwrap()
                    .dot(&b_jb.t())
                    .into_shape([n_atoms_ab, n_a * n_j])
                    .unwrap();

                ab_bjb.dot(&bia_ij.t()).dot(&gamma_lr).trace().unwrap()
            } else {
                0.0
            };
            exchange_val = exchange;
        }

        2.0 * coulomb - exchange_val
    }
}
