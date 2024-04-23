use crate::fmo::{ChargeTransferPair, LocallyExcited};
use crate::initialization::Atom;
use crate::SuperSystem;
use ndarray::prelude::*;

/// Type to determine the kind of orbitals that are used for the LE-LE transition charges. `Hole`
/// specifies the use of occupied orbitals, while `Electron` means that virtual orbitals are used.
pub enum ElecHole {
    Hole,
    Electron,
}

/// Computes the Mulliken transition charges between two sets of orbitals. These orbitals
/// are part of two LE states, that are either on one monomer or two different ones. The overlap
/// matrix `s`, is the overlap matrix between the basis functions of the corresponding monomers.
pub fn q_lele<'a>(
    a: &'a LocallyExcited<'a>,
    b: &'a LocallyExcited<'a>,
    kind_a: ElecHole,
    kind_b: ElecHole,
    s: ArrayView2<f64>,
) -> Array3<f64> {
    // Number of atoms.
    let n_atoms_i: usize = a.atoms.len();
    let n_atoms_j: usize = b.atoms.len();
    let n_atoms: usize = n_atoms_i + n_atoms_j;
    // Check if the occupied or virtual orbitals of the first LE state are needed.
    let orbs_i: ArrayView2<f64> = match kind_a {
        ElecHole::Hole => a.occs,
        ElecHole::Electron => a.virts,
    };
    // Check if the occupied or virtual orbitals of the second LE state are needed.
    let orbs_j: ArrayView2<f64> = match kind_b {
        ElecHole::Hole => b.occs,
        ElecHole::Electron => b.virts,
    };
    // Number of molecular orbitals on monomer I.
    let dim_i: usize = orbs_i.ncols();
    // Number of molecular orbitals on monomer J.
    let dim_j: usize = orbs_j.ncols();
    // The transition charges between the two sets of MOs  are initialized.
    let mut q_trans: Array3<f64> = Array3::zeros([n_atoms, dim_i, dim_j]);
    // Matrix product of overlap matrix with the orbitals on I.
    let sc_mu_j: Array2<f64> = s.dot(&orbs_j);
    // Matrix product of overlap matrix with the orbitals on J.
    let sc_mu_i: Array2<f64> = s.t().dot(&orbs_i);
    let mut mu: usize = 0;
    // Iteration over all atoms (I).
    for (atom, mut q_n) in a.atoms.iter().zip(
        q_trans
            .slice_mut(s![0..n_atoms_i, .., ..])
            .axis_iter_mut(Axis(0)),
    ) {
        // Iteration over atomic orbitals mu on I.
        for _ in 0..atom.n_orbs {
            // Iteration over orbitals i on monomer I. orb_i -> C_(mu i) (mu on I, i on I)
            for (orb_i, mut q_i) in orbs_i.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                // Iteration over S * C_J on monomer J. sc -> SC_(mu j) (mu on I, j on J)
                for (sc, q) in sc_mu_j.row(mu).iter().zip(q_i.iter_mut()) {
                    // The transition charge is computed.
                    *q += orb_i * sc;
                }
            }
            mu += 1;
        }
    }
    mu = 0;
    // Iteration over all atoms J.
    for (atom, mut q_n) in b.atoms.iter().zip(
        q_trans
            .slice_mut(s![n_atoms_i.., .., ..])
            .axis_iter_mut(Axis(0)),
    ) {
        // Iteration over atomic orbitals mu on J.
        for _ in 0..atom.n_orbs {
            // Iteration over occupied orbital i. sc -> SC_(mu i) (mu on J, i on I)
            for (sc, mut q_i) in sc_mu_i.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                // Iteration over occupied orbital j. C_(mu j) (mu on J, j on J)
                for (orb_j, q) in orbs_j.row(mu).iter().zip(q_i.iter_mut()) {
                    // The transition charge is computed.
                    *q += orb_j * sc;
                }
            }
            mu += 1;
        }
    }
    0.5 * q_trans
}

impl SuperSystem<'_> {
    pub fn q_lect<'a>(
        &self,
        a: &'a LocallyExcited<'a>,
        b: &ChargeTransferPair,
        kind: ElecHole,
    ) -> Array3<f64> {
        match kind {
            ElecHole::Hole => {
                // Number of atoms.
                let n_atoms_le: usize = a.atoms.len();
                let n_atoms_ct_occ: usize = self.monomers[b.m_h].n_atoms;
                let n_atoms_occ: usize = n_atoms_le + n_atoms_ct_occ;

                // get the atoms of the hole
                let atoms_h: &[Atom] = &self.atoms[self.monomers[b.m_h].slice.atom_as_range()];

                // get occupied orbitals of the LE state
                // let occs_le: ArrayView2<f64> = a.occs;
                let occs_le = if self.config.fmo_lc_tddftb.restrict_active_space {
                    let mut arr: Array2<f64> = Array2::zeros((a.occs.dim().0, a.occ_indices.len()));
                    for (en_idx, idx) in a.occ_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&a.occs.slice(s![.., *idx]));
                    }
                    arr
                } else {
                    a.occs.to_owned()
                };

                // get orbitals of the ct state
                let occ_indices: &[usize] = self.monomers[b.m_h].properties.occ_indices().unwrap();
                // The index of the HOMO (zero based).
                let homo: usize = occ_indices[occ_indices.len() - 1];
                // let occs_ct:ArrayView2<f64> = self.monomers[b.m_h].properties.orbs_slice(0, Some(homo + 1)).unwrap();
                let occs_ct = if self.config.fmo_lc_tddftb.restrict_active_space {
                    // self.monomers[b.m_h].properties
                    //     .orbs_slice((homo+1-self.config.lcmo.active_space_ct), Some(homo + 1)).unwrap()
                    let monomer_orbs: ArrayView2<f64> =
                        self.monomers[b.m_h].properties.orbs().unwrap();
                    let mut arr: Array2<f64> =
                        Array2::zeros((monomer_orbs.dim().0, b.occ_indices.len()));
                    for (en_idx, idx) in b.occ_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&monomer_orbs.slice(s![.., *idx]));
                    }
                    arr
                } else {
                    self.monomers[b.m_h]
                        .properties
                        .orbs_slice(0, Some(homo + 1))
                        .unwrap()
                        .to_owned()
                };

                // slice the overlap matrix
                let s_ij: ArrayView2<f64> = self
                    .properties
                    .s_slice(a.monomer.slice.orb, self.monomers[b.m_h].slice.orb)
                    .unwrap();

                // Number of occupied molecular orbitals on I
                let dim_i: usize = occs_le.ncols();
                // Number of molecular orbitals on the hole
                let dim_j: usize = occs_ct.ncols();

                // The transition charges between the two sets of MOs are initialized.
                let mut q_trans_ij: Array3<f64> = Array3::zeros([n_atoms_occ, dim_i, dim_j]);

                // Matrix products between the overlap matrix and the molecular orbitals
                let s_ij_c_ct_occ: Array2<f64> = s_ij.dot(&occs_ct);
                let s_ij_c_le_occ: Array2<f64> = s_ij.t().dot(&occs_le);

                let mut mu: usize = 0;
                for (atom, mut q_n) in a.atoms.iter().zip(
                    q_trans_ij
                        .slice_mut(s![0..n_atoms_le, .., ..])
                        .axis_iter_mut(Axis(0)),
                ) {
                    for _ in 0..atom.n_orbs {
                        for (orb_i, mut q_i) in
                            occs_le.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0)))
                        {
                            for (sc, q) in s_ij_c_ct_occ.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_i * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                mu = 0;

                for (atom, mut q_n) in atoms_h.iter().zip(
                    q_trans_ij
                        .slice_mut(s![n_atoms_le.., .., ..])
                        .axis_iter_mut(Axis(0)),
                ) {
                    for _ in 0..atom.n_orbs {
                        for (sc, mut q_i) in
                            s_ij_c_le_occ.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0)))
                        {
                            for (orb_j, q) in occs_ct.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_j * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                0.5 * q_trans_ij
            }
            ElecHole::Electron => {
                // Number of atoms.
                let n_atoms_le: usize = a.atoms.len();
                let n_atoms_ct_virt: usize = self.monomers[b.m_l].n_atoms;
                let n_atoms_virt: usize = n_atoms_le + n_atoms_ct_virt;

                // get the atoms of the hole
                let atoms_l: &[Atom] = &self.atoms[self.monomers[b.m_l].slice.atom_as_range()];

                // get virtual orbitals of the LE state
                // let virts_le: ArrayView2<f64> = a.virts;
                let virts_le = if self.config.fmo_lc_tddftb.restrict_active_space {
                    let mut arr: Array2<f64> =
                        Array2::zeros((a.virts.dim().0, a.virt_indices.len()));
                    for (en_idx, idx) in a.virt_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&a.virts.slice(s![.., *idx]));
                    }
                    arr
                } else {
                    a.virts.to_owned()
                };

                // get orbitals of the ct state
                let virt_indices: &[usize] =
                    self.monomers[b.m_l].properties.virt_indices().unwrap();
                // The index of the LUMO (zero based).
                let lumo: usize = virt_indices[0];
                // let virts_ct:ArrayView2<f64> = self.monomers[b.m_l].properties.orbs_slice(lumo, None).unwrap();
                let virts_ct = if self.config.fmo_lc_tddftb.restrict_active_space {
                    // self.monomers[b.m_l].properties
                    //     .orbs_slice(lumo, Some(lumo+self.config.lcmo.active_space_ct)).unwrap()
                    let monomer_orbs: ArrayView2<f64> =
                        self.monomers[b.m_l].properties.orbs().unwrap();
                    let mut arr: Array2<f64> =
                        Array2::zeros((monomer_orbs.dim().0, b.virt_indices.len()));
                    for (en_idx, idx) in b.virt_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&monomer_orbs.slice(s![.., lumo + *idx]));
                    }
                    arr
                } else {
                    self.monomers[b.m_l]
                        .properties
                        .orbs_slice(lumo, None)
                        .unwrap()
                        .to_owned()
                };

                // slice the overlap matrix
                let s_ab: ArrayView2<f64> = self
                    .properties
                    .s_slice(a.monomer.slice.orb, self.monomers[b.m_l].slice.orb)
                    .unwrap();

                // Number of virtual molecular orbitals on I
                let dim_a: usize = virts_le.ncols();
                // Number of virtual molecular orbitals on the electron
                let dim_b: usize = virts_ct.ncols();

                // The transition charges between the two sets of MOs are initialized.
                let mut q_trans_ab: Array3<f64> = Array3::zeros([n_atoms_virt, dim_a, dim_b]);

                // Matrix products between the overlap matrix and the molecular orbitals
                let s_ab_c_ct_virt: Array2<f64> = s_ab.dot(&virts_ct);
                let s_ab_c_le_virt: Array2<f64> = s_ab.t().dot(&virts_le);

                let mut mu: usize = 0;
                for (atom, mut q_n) in a.atoms.iter().zip(
                    q_trans_ab
                        .slice_mut(s![0..n_atoms_le, .., ..])
                        .axis_iter_mut(Axis(0)),
                ) {
                    for _ in 0..atom.n_orbs {
                        for (orb_i, mut q_i) in
                            virts_le.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0)))
                        {
                            for (sc, q) in s_ab_c_ct_virt.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_i * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                mu = 0;

                for (atom, mut q_n) in atoms_l.iter().zip(
                    q_trans_ab
                        .slice_mut(s![n_atoms_le.., .., ..])
                        .axis_iter_mut(Axis(0)),
                ) {
                    for _ in 0..atom.n_orbs {
                        for (sc, mut q_i) in s_ab_c_le_virt
                            .row(mu)
                            .iter()
                            .zip(q_n.axis_iter_mut(Axis(0)))
                        {
                            for (orb_j, q) in virts_ct.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_j * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                0.5 * q_trans_ab
            }
        }
    }

    pub fn q_lect_b<'a>(
        &self,
        a: &'a LocallyExcited<'a>,
        b: &ChargeTransferPair,
        kind: ElecHole,
    ) -> Array3<f64> {
        match kind {
            ElecHole::Hole => {
                // Number of atoms.
                let n_atoms_le: usize = a.atoms.len();
                let n_atoms_ct_virt: usize = self.monomers[b.m_l].n_atoms;
                let n_atoms_occ: usize = n_atoms_le + n_atoms_ct_virt;

                // get the atoms of the hole
                let atoms_h: &[Atom] = &self.atoms[self.monomers[b.m_l].slice.atom_as_range()];

                // get occupied orbitals of the LE state
                // let occs_le: ArrayView2<f64> = a.occs;
                let occs_le = if self.config.fmo_lc_tddftb.restrict_active_space {
                    let mut arr: Array2<f64> = Array2::zeros((a.occs.dim().0, a.occ_indices.len()));
                    for (en_idx, idx) in a.occ_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&a.occs.slice(s![.., *idx]));
                    }
                    arr
                } else {
                    a.occs.to_owned()
                };

                // get orbitals of the ct state
                let virt_indices: &[usize] =
                    self.monomers[b.m_l].properties.virt_indices().unwrap();
                // The index of the LUMO (zero based).
                let lumo: usize = virt_indices[0];
                // let virts_ct:ArrayView2<f64> = self.monomers[b.m_l].properties.orbs_slice(lumo, None).unwrap();
                let virts_ct = if self.config.fmo_lc_tddftb.restrict_active_space {
                    // self.monomers[b.m_l].properties
                    //     .orbs_slice(lumo, Some(lumo+self.config.lcmo.active_space_ct)).unwrap()
                    let monomer_orbs: ArrayView2<f64> =
                        self.monomers[b.m_l].properties.orbs().unwrap();
                    let mut arr: Array2<f64> =
                        Array2::zeros((monomer_orbs.dim().0, b.virt_indices.len()));
                    for (en_idx, idx) in b.virt_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&monomer_orbs.slice(s![.., lumo + *idx]));
                    }
                    arr
                } else {
                    self.monomers[b.m_l]
                        .properties
                        .orbs_slice(lumo, None)
                        .unwrap()
                        .to_owned()
                };

                // slice the overlap matrix
                let s_ij: ArrayView2<f64> = self
                    .properties
                    .s_slice(a.monomer.slice.orb, self.monomers[b.m_l].slice.orb)
                    .unwrap();

                // Number of occupied molecular orbitals on I
                let dim_i: usize = occs_le.ncols();
                // Number of molecular orbitals on the hole
                let dim_j: usize = virts_ct.ncols();

                // The transition charges between the two sets of MOs are initialized.
                let mut q_trans_ij: Array3<f64> = Array3::zeros([n_atoms_occ, dim_i, dim_j]);

                // Matrix products between the overlap matrix and the molecular orbitals
                let s_ij_c_ct_virt: Array2<f64> = s_ij.dot(&virts_ct);
                let s_ij_c_le_occ: Array2<f64> = s_ij.t().dot(&occs_le);

                let mut mu: usize = 0;
                for (atom, mut q_n) in a.atoms.iter().zip(
                    q_trans_ij
                        .slice_mut(s![0..n_atoms_le, .., ..])
                        .axis_iter_mut(Axis(0)),
                ) {
                    for _ in 0..atom.n_orbs {
                        for (orb_i, mut q_i) in
                            occs_le.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0)))
                        {
                            for (sc, q) in s_ij_c_ct_virt.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_i * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                mu = 0;

                for (atom, mut q_n) in atoms_h.iter().zip(
                    q_trans_ij
                        .slice_mut(s![n_atoms_le.., .., ..])
                        .axis_iter_mut(Axis(0)),
                ) {
                    for _ in 0..atom.n_orbs {
                        for (sc, mut q_i) in
                            s_ij_c_le_occ.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0)))
                        {
                            for (orb_j, q) in virts_ct.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_j * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                0.5 * q_trans_ij
            }
            ElecHole::Electron => {
                // Number of atoms.
                let n_atoms_le: usize = a.atoms.len();
                let n_atoms_ct_occ: usize = self.monomers[b.m_h].n_atoms;
                let n_atoms_virt: usize = n_atoms_le + n_atoms_ct_occ;

                // get the atoms of the hole
                let atoms_l: &[Atom] = &self.atoms[self.monomers[b.m_h].slice.atom_as_range()];

                // get virtual orbitals of the LE state
                // let virts_le: ArrayView2<f64> = a.virts;
                let virts_le = if self.config.fmo_lc_tddftb.restrict_active_space {
                    let mut arr: Array2<f64> =
                        Array2::zeros((a.virts.dim().0, a.virt_indices.len()));
                    for (en_idx, idx) in a.virt_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&a.virts.slice(s![.., *idx]));
                    }
                    arr
                } else {
                    a.virts.to_owned()
                };

                // get orbitals of the ct state
                let occ_indices: &[usize] = self.monomers[b.m_h].properties.occ_indices().unwrap();
                // The index of the HOMO (zero based).
                let homo: usize = occ_indices[occ_indices.len() - 1];
                // let occs_ct:ArrayView2<f64> = self.monomers[b.m_h].properties.orbs_slice(0, Some(homo + 1)).unwrap();
                let occs_ct = if self.config.fmo_lc_tddftb.restrict_active_space {
                    // self.monomers[b.m_h].properties
                    //     .orbs_slice((homo+1-self.config.lcmo.active_space_ct), Some(homo + 1)).unwrap()
                    let monomer_orbs: ArrayView2<f64> =
                        self.monomers[b.m_h].properties.orbs().unwrap();
                    let mut arr: Array2<f64> =
                        Array2::zeros((monomer_orbs.dim().0, b.occ_indices.len()));
                    for (en_idx, idx) in b.occ_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&monomer_orbs.slice(s![.., *idx]));
                    }
                    arr
                } else {
                    self.monomers[b.m_h]
                        .properties
                        .orbs_slice(0, Some(homo + 1))
                        .unwrap()
                        .to_owned()
                };

                // slice the overlap matrix
                let s_ab: ArrayView2<f64> = self
                    .properties
                    .s_slice(a.monomer.slice.orb, self.monomers[b.m_h].slice.orb)
                    .unwrap();

                // Number of virtual molecular orbitals on I
                let dim_a: usize = virts_le.ncols();
                // Number of virtual molecular orbitals on the electron
                let dim_b: usize = occs_ct.ncols();

                // The transition charges between the two sets of MOs are initialized.
                let mut q_trans_ab: Array3<f64> = Array3::zeros([n_atoms_virt, dim_a, dim_b]);

                // Matrix products between the overlap matrix and the molecular orbitals
                let s_ab_c_ct_virt: Array2<f64> = s_ab.dot(&occs_ct);
                let s_ab_c_le_virt: Array2<f64> = s_ab.t().dot(&virts_le);

                let mut mu: usize = 0;
                for (atom, mut q_n) in a.atoms.iter().zip(
                    q_trans_ab
                        .slice_mut(s![0..n_atoms_le, .., ..])
                        .axis_iter_mut(Axis(0)),
                ) {
                    for _ in 0..atom.n_orbs {
                        for (orb_i, mut q_i) in
                            virts_le.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0)))
                        {
                            for (sc, q) in s_ab_c_ct_virt.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_i * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                mu = 0;

                for (atom, mut q_n) in atoms_l.iter().zip(
                    q_trans_ab
                        .slice_mut(s![n_atoms_le.., .., ..])
                        .axis_iter_mut(Axis(0)),
                ) {
                    for _ in 0..atom.n_orbs {
                        for (sc, mut q_i) in s_ab_c_le_virt
                            .row(mu)
                            .iter()
                            .zip(q_n.axis_iter_mut(Axis(0)))
                        {
                            for (orb_j, q) in occs_ct.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_j * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                0.5 * q_trans_ab
            }
        }
    }

    pub fn q_ctct<'a>(
        &self,
        a: &ChargeTransferPair,
        b: &ChargeTransferPair,
        kind: ElecHole,
    ) -> Array3<f64> {
        match kind {
            ElecHole::Hole => {
                // Number of atoms.
                let n_atoms_ct_a: usize = self.monomers[a.m_h].n_atoms;
                let n_atoms_ct_b: usize = self.monomers[b.m_h].n_atoms;
                let n_atoms_occ: usize = n_atoms_ct_a + n_atoms_ct_b;

                // get the atoms of the hole
                let atoms_a: &[Atom] = &self.atoms[self.monomers[a.m_h].slice.atom_as_range()];
                let atoms_b: &[Atom] = &self.atoms[self.monomers[b.m_h].slice.atom_as_range()];

                // get orbitals of the ct state
                let occ_indices_a: &[usize] =
                    self.monomers[a.m_h].properties.occ_indices().unwrap();
                let occ_indices_b: &[usize] =
                    self.monomers[b.m_h].properties.occ_indices().unwrap();
                // The index of the HOMO (zero based).
                let homo_a: usize = occ_indices_a[occ_indices_a.len() - 1];
                let homo_b: usize = occ_indices_b[occ_indices_b.len() - 1];
                // let occs_ct_a:ArrayView2<f64> = self.monomers[a.m_h].properties.orbs_slice(0, Some(homo_a + 1)).unwrap();
                // let occs_ct_b:ArrayView2<f64> = self.monomers[b.m_h].properties.orbs_slice(0, Some(homo_b + 1)).unwrap();
                let restrict_space: bool = self.config.fmo_lc_tddftb.restrict_active_space;
                // let active_space:usize = self.config.lcmo.active_space_ct;

                let occs_ct_a =
                    if restrict_space {
                        // self.monomers[a.m_h].properties.orbs_slice((homo_a-active_space+1), Some(homo_a + 1)).unwrap()
                        let monomer_orbs: ArrayView2<f64> =
                            self.monomers[a.m_h].properties.orbs().unwrap();
                        let mut arr: Array2<f64> =
                            Array2::zeros((monomer_orbs.dim().0, a.occ_indices.len()));
                        for (en_idx, idx) in a.occ_indices.iter().enumerate() {
                            arr.slice_mut(s![.., en_idx])
                                .assign(&monomer_orbs.slice(s![.., *idx]));
                        }
                        arr
                    } else {
                        self.monomers[a.m_h]
                            .properties
                            .orbs_slice(0, Some(homo_a + 1))
                            .unwrap()
                            .to_owned()
                    };
                let occs_ct_b = if restrict_space {
                    let monomer_orbs: ArrayView2<f64> =
                        self.monomers[b.m_h].properties.orbs().unwrap();
                    let mut arr: Array2<f64> =
                        Array2::zeros((monomer_orbs.dim().0, b.occ_indices.len()));
                    for (en_idx, idx) in b.occ_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&monomer_orbs.slice(s![.., *idx]));
                    }
                    arr
                } else {
                    self.monomers[b.m_h]
                        .properties
                        .orbs_slice(0, Some(homo_b + 1))
                        .unwrap()
                        .to_owned()
                };

                // slice the overlap matrix
                let s_ij: ArrayView2<f64> = self
                    .properties
                    .s_slice(
                        self.monomers[a.m_h].slice.orb,
                        self.monomers[b.m_h].slice.orb,
                    )
                    .unwrap();

                // Number of occupied molecular orbitals on I
                let dim_i: usize = occs_ct_a.ncols();
                // Number of molecular orbitals on the hole
                let dim_j: usize = occs_ct_b.ncols();

                // The transition charges between the two sets of MOs are initialized.
                let mut q_trans_ij: Array3<f64> = Array3::zeros([n_atoms_occ, dim_i, dim_j]);

                // Matrix products between the overlap matrix and the molecular orbitals
                let s_ij_c_ct_b: Array2<f64> = s_ij.dot(&occs_ct_b);
                let s_ij_c_ct_a: Array2<f64> = s_ij.t().dot(&occs_ct_a);

                let mut mu: usize = 0;
                for (atom, mut q_n) in atoms_a.iter().zip(
                    q_trans_ij
                        .slice_mut(s![0..n_atoms_ct_a, .., ..])
                        .axis_iter_mut(Axis(0)),
                ) {
                    for _ in 0..atom.n_orbs {
                        for (orb_i, mut q_i) in
                            occs_ct_a.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0)))
                        {
                            for (sc, q) in s_ij_c_ct_b.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_i * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                mu = 0;

                for (atom, mut q_n) in atoms_b.iter().zip(
                    q_trans_ij
                        .slice_mut(s![n_atoms_ct_a.., .., ..])
                        .axis_iter_mut(Axis(0)),
                ) {
                    for _ in 0..atom.n_orbs {
                        for (sc, mut q_i) in
                            s_ij_c_ct_a.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0)))
                        {
                            for (orb_j, q) in occs_ct_b.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_j * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                0.5 * q_trans_ij
            }
            ElecHole::Electron => {
                // Number of atoms.
                let n_atoms_ct_a: usize = self.monomers[a.m_l].n_atoms;
                let n_atoms_ct_b: usize = self.monomers[b.m_l].n_atoms;
                let n_atoms_virt: usize = n_atoms_ct_a + n_atoms_ct_b;

                // get the atoms of the hole
                let atoms_a: &[Atom] = &self.atoms[self.monomers[a.m_l].slice.atom_as_range()];
                let atoms_b: &[Atom] = &self.atoms[self.monomers[b.m_l].slice.atom_as_range()];

                // get orbitals of the ct state
                let virt_indices_a: &[usize] =
                    self.monomers[a.m_l].properties.virt_indices().unwrap();
                let virt_indices_b: &[usize] =
                    self.monomers[b.m_l].properties.virt_indices().unwrap();
                // The index of the LUMO (zero based).
                let lumo_a: usize = virt_indices_a[0];
                let lumo_b: usize = virt_indices_b[0];
                // let virts_ct_a:ArrayView2<f64> = self.monomers[a.m_l].properties.orbs_slice(lumo_a, None).unwrap();
                // let virts_ct_b:ArrayView2<f64> = self.monomers[b.m_l].properties.orbs_slice(lumo_b, None).unwrap();
                let restrict_space: bool = self.config.fmo_lc_tddftb.restrict_active_space;
                // let active_space:usize = self.config.lcmo.active_space_ct;

                let virts_ct_a = if restrict_space {
                    // self.monomers[a.m_l].properties.orbs_slice(lumo_a, Some(lumo_a + active_space)).unwrap()
                    let monomer_orbs: ArrayView2<f64> =
                        self.monomers[a.m_l].properties.orbs().unwrap();
                    let mut arr: Array2<f64> =
                        Array2::zeros((monomer_orbs.dim().0, a.virt_indices.len()));
                    for (en_idx, idx) in a.virt_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&monomer_orbs.slice(s![.., lumo_a + *idx]));
                    }
                    arr
                } else {
                    self.monomers[a.m_l]
                        .properties
                        .orbs_slice(lumo_a, None)
                        .unwrap()
                        .to_owned()
                };
                let virts_ct_b = if restrict_space {
                    let monomer_orbs: ArrayView2<f64> =
                        self.monomers[b.m_l].properties.orbs().unwrap();
                    let mut arr: Array2<f64> =
                        Array2::zeros((monomer_orbs.dim().0, b.virt_indices.len()));
                    for (en_idx, idx) in b.virt_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&monomer_orbs.slice(s![.., lumo_b + *idx]));
                    }
                    arr
                } else {
                    self.monomers[b.m_l]
                        .properties
                        .orbs_slice(lumo_b, None)
                        .unwrap()
                        .to_owned()
                };

                // slice the overlap matrix
                let s_ab: ArrayView2<f64> = self
                    .properties
                    .s_slice(
                        self.monomers[a.m_l].slice.orb,
                        self.monomers[b.m_l].slice.orb,
                    )
                    .unwrap();

                // Number of virtual molecular orbitals on I
                let dim_a: usize = virts_ct_a.ncols();
                // Number of virtual molecular orbitals on the electron
                let dim_b: usize = virts_ct_b.ncols();

                // The transition charges between the two sets of MOs are initialized.
                let mut q_trans_ab: Array3<f64> = Array3::zeros([n_atoms_virt, dim_a, dim_b]);

                // Matrix products between the overlap matrix and the molecular orbitals
                let s_ab_c_ct_b: Array2<f64> = s_ab.dot(&virts_ct_b);
                let s_ab_c_ct_a: Array2<f64> = s_ab.t().dot(&virts_ct_a);

                let mut mu: usize = 0;
                for (atom, mut q_n) in atoms_a.iter().zip(
                    q_trans_ab
                        .slice_mut(s![0..n_atoms_ct_a, .., ..])
                        .axis_iter_mut(Axis(0)),
                ) {
                    for _ in 0..atom.n_orbs {
                        for (orb_i, mut q_i) in
                            virts_ct_a.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0)))
                        {
                            for (sc, q) in s_ab_c_ct_b.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_i * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                mu = 0;

                for (atom, mut q_n) in atoms_b.iter().zip(
                    q_trans_ab
                        .slice_mut(s![n_atoms_ct_a.., .., ..])
                        .axis_iter_mut(Axis(0)),
                ) {
                    for _ in 0..atom.n_orbs {
                        for (sc, mut q_i) in
                            s_ab_c_ct_a.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0)))
                        {
                            for (orb_j, q) in virts_ct_b.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_j * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                0.5 * q_trans_ab
            }
        }
    }

    pub fn q_ctct_b<'a>(
        &self,
        a: &ChargeTransferPair,
        b: &ChargeTransferPair,
        kind: ElecHole,
    ) -> Array3<f64> {
        match kind {
            ElecHole::Hole => {
                // Number of atoms.
                let n_atoms_ct_a: usize = self.monomers[a.m_h].n_atoms;
                let n_atoms_ct_b: usize = self.monomers[b.m_l].n_atoms;
                let n_atoms_occ: usize = n_atoms_ct_a + n_atoms_ct_b;

                // get the atoms of the hole
                let atoms_a: &[Atom] = &self.atoms[self.monomers[a.m_h].slice.atom_as_range()];
                let atoms_b: &[Atom] = &self.atoms[self.monomers[b.m_l].slice.atom_as_range()];

                // get orbitals of the ct state
                let occ_indices_a: &[usize] =
                    self.monomers[a.m_h].properties.occ_indices().unwrap();
                let virt_indices_b: &[usize] =
                    self.monomers[b.m_l].properties.virt_indices().unwrap();
                // The index of the HOMO (zero based).
                let homo_a: usize = occ_indices_a[occ_indices_a.len() - 1];
                let lumo: usize = virt_indices_b[0];
                // let occs_ct_a:ArrayView2<f64> = self.monomers[a.m_h].properties.orbs_slice(0, Some(homo_a + 1)).unwrap();
                // let occs_ct_b:ArrayView2<f64> = self.monomers[b.m_h].properties.orbs_slice(0, Some(homo_b + 1)).unwrap();
                let restrict_space: bool = self.config.fmo_lc_tddftb.restrict_active_space;
                // let active_space:usize = self.config.lcmo.active_space_ct;

                let occs_ct_a =
                    if restrict_space {
                        // self.monomers[a.m_h].properties.orbs_slice((homo_a-active_space+1), Some(homo_a + 1)).unwrap()
                        let monomer_orbs: ArrayView2<f64> =
                            self.monomers[a.m_h].properties.orbs().unwrap();
                        let mut arr: Array2<f64> =
                            Array2::zeros((monomer_orbs.dim().0, a.occ_indices.len()));
                        for (en_idx, idx) in a.occ_indices.iter().enumerate() {
                            arr.slice_mut(s![.., en_idx])
                                .assign(&monomer_orbs.slice(s![.., *idx]));
                        }
                        arr
                    } else {
                        self.monomers[a.m_h]
                            .properties
                            .orbs_slice(0, Some(homo_a + 1))
                            .unwrap()
                            .to_owned()
                    };
                let virts_ct_b = if restrict_space {
                    let monomer_orbs: ArrayView2<f64> =
                        self.monomers[b.m_l].properties.orbs().unwrap();
                    let mut arr: Array2<f64> =
                        Array2::zeros((monomer_orbs.dim().0, b.virt_indices.len()));
                    for (en_idx, idx) in b.virt_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&monomer_orbs.slice(s![.., lumo + *idx]));
                    }
                    arr
                } else {
                    self.monomers[b.m_l]
                        .properties
                        .orbs_slice(lumo, None)
                        .unwrap()
                        .to_owned()
                };

                // slice the overlap matrix
                let s_ij: ArrayView2<f64> = self
                    .properties
                    .s_slice(
                        self.monomers[a.m_h].slice.orb,
                        self.monomers[b.m_l].slice.orb,
                    )
                    .unwrap();

                // Number of occupied molecular orbitals on I
                let dim_i: usize = occs_ct_a.ncols();
                // Number of molecular orbitals on the hole
                let dim_j: usize = virts_ct_b.ncols();

                // The transition charges between the two sets of MOs are initialized.
                let mut q_trans_ij: Array3<f64> = Array3::zeros([n_atoms_occ, dim_i, dim_j]);

                // Matrix products between the overlap matrix and the molecular orbitals
                let s_ij_c_ct_b: Array2<f64> = s_ij.dot(&virts_ct_b);
                let s_ij_c_ct_a: Array2<f64> = s_ij.t().dot(&occs_ct_a);

                let mut mu: usize = 0;
                for (atom, mut q_n) in atoms_a.iter().zip(
                    q_trans_ij
                        .slice_mut(s![0..n_atoms_ct_a, .., ..])
                        .axis_iter_mut(Axis(0)),
                ) {
                    for _ in 0..atom.n_orbs {
                        for (orb_i, mut q_i) in
                            occs_ct_a.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0)))
                        {
                            for (sc, q) in s_ij_c_ct_b.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_i * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                mu = 0;

                for (atom, mut q_n) in atoms_b.iter().zip(
                    q_trans_ij
                        .slice_mut(s![n_atoms_ct_a.., .., ..])
                        .axis_iter_mut(Axis(0)),
                ) {
                    for _ in 0..atom.n_orbs {
                        for (sc, mut q_i) in
                            s_ij_c_ct_a.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0)))
                        {
                            for (orb_j, q) in virts_ct_b.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_j * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                0.5 * q_trans_ij
            }
            ElecHole::Electron => {
                // Number of atoms.
                let n_atoms_ct_a: usize = self.monomers[a.m_l].n_atoms;
                let n_atoms_ct_b: usize = self.monomers[b.m_h].n_atoms;
                let n_atoms_virt: usize = n_atoms_ct_a + n_atoms_ct_b;

                // get the atoms of the hole
                let atoms_a: &[Atom] = &self.atoms[self.monomers[a.m_l].slice.atom_as_range()];
                let atoms_b: &[Atom] = &self.atoms[self.monomers[b.m_h].slice.atom_as_range()];

                // get orbitals of the ct state
                let virt_indices_a: &[usize] =
                    self.monomers[a.m_l].properties.virt_indices().unwrap();
                let occ_indices_b: &[usize] =
                    self.monomers[b.m_h].properties.occ_indices().unwrap();
                // The index of the LUMO (zero based).
                let lumo_a: usize = virt_indices_a[0];
                let homo_b: usize = occ_indices_b[occ_indices_b.len() - 1];
                // let virts_ct_a:ArrayView2<f64> = self.monomers[a.m_l].properties.orbs_slice(lumo_a, None).unwrap();
                // let virts_ct_b:ArrayView2<f64> = self.monomers[b.m_l].properties.orbs_slice(lumo_b, None).unwrap();
                let restrict_space: bool = self.config.fmo_lc_tddftb.restrict_active_space;
                // let active_space:usize = self.config.lcmo.active_space_ct;

                let virts_ct_a = if restrict_space {
                    // self.monomers[a.m_l].properties.orbs_slice(lumo_a, Some(lumo_a + active_space)).unwrap()
                    let monomer_orbs: ArrayView2<f64> =
                        self.monomers[a.m_l].properties.orbs().unwrap();
                    let mut arr: Array2<f64> =
                        Array2::zeros((monomer_orbs.dim().0, a.virt_indices.len()));
                    for (en_idx, idx) in a.virt_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&monomer_orbs.slice(s![.., lumo_a + *idx]));
                    }
                    arr
                } else {
                    self.monomers[a.m_l]
                        .properties
                        .orbs_slice(lumo_a, None)
                        .unwrap()
                        .to_owned()
                };
                let occs_ct_b = if restrict_space {
                    let monomer_orbs: ArrayView2<f64> =
                        self.monomers[b.m_h].properties.orbs().unwrap();
                    let mut arr: Array2<f64> =
                        Array2::zeros((monomer_orbs.dim().0, b.occ_indices.len()));
                    for (en_idx, idx) in b.occ_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&monomer_orbs.slice(s![.., *idx]));
                    }
                    arr
                } else {
                    self.monomers[b.m_h]
                        .properties
                        .orbs_slice(0, Some(homo_b + 1))
                        .unwrap()
                        .to_owned()
                };

                // slice the overlap matrix
                let s_ab: ArrayView2<f64> = self
                    .properties
                    .s_slice(
                        self.monomers[a.m_l].slice.orb,
                        self.monomers[b.m_l].slice.orb,
                    )
                    .unwrap();

                // Number of virtual molecular orbitals on I
                let dim_a: usize = virts_ct_a.ncols();
                // Number of virtual molecular orbitals on the electron
                let dim_b: usize = occs_ct_b.ncols();

                // The transition charges between the two sets of MOs are initialized.
                let mut q_trans_ab: Array3<f64> = Array3::zeros([n_atoms_virt, dim_a, dim_b]);

                // Matrix products between the overlap matrix and the molecular orbitals
                let s_ab_c_ct_b: Array2<f64> = s_ab.dot(&occs_ct_b);
                let s_ab_c_ct_a: Array2<f64> = s_ab.t().dot(&virts_ct_a);

                let mut mu: usize = 0;
                for (atom, mut q_n) in atoms_a.iter().zip(
                    q_trans_ab
                        .slice_mut(s![0..n_atoms_ct_a, .., ..])
                        .axis_iter_mut(Axis(0)),
                ) {
                    for _ in 0..atom.n_orbs {
                        for (orb_i, mut q_i) in
                            virts_ct_a.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0)))
                        {
                            for (sc, q) in s_ab_c_ct_b.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_i * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                mu = 0;

                for (atom, mut q_n) in atoms_b.iter().zip(
                    q_trans_ab
                        .slice_mut(s![n_atoms_ct_a.., .., ..])
                        .axis_iter_mut(Axis(0)),
                ) {
                    for _ in 0..atom.n_orbs {
                        for (sc, mut q_i) in
                            s_ab_c_ct_a.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0)))
                        {
                            for (orb_j, q) in occs_ct_b.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_j * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                0.5 * q_trans_ab
            }
        }
    }

    pub fn q_lele<'a>(
        &self,
        a: &'a LocallyExcited<'a>,
        b: &'a LocallyExcited<'a>,
        kind_a: ElecHole,
        kind_b: ElecHole,
        s: ArrayView2<f64>,
    ) -> Array3<f64> {
        // Number of atoms.
        let n_atoms_i: usize = a.atoms.len();
        let n_atoms_j: usize = b.atoms.len();
        let n_atoms: usize = n_atoms_i + n_atoms_j;
        // Check if the occupied or virtual orbitals of the first LE state are needed.
        let orbs_i: Array2<f64> = match kind_a {
            ElecHole::Hole => {
                if self.config.fmo_lc_tddftb.restrict_active_space {
                    let mut arr: Array2<f64> = Array2::zeros((a.occs.dim().0, a.occ_indices.len()));
                    for (en_idx, idx) in a.occ_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&a.occs.slice(s![.., *idx]));
                    }
                    arr
                } else {
                    a.occs.to_owned()
                }
            }
            ElecHole::Electron => {
                if self.config.fmo_lc_tddftb.restrict_active_space {
                    let mut arr: Array2<f64> =
                        Array2::zeros((a.virts.dim().0, a.virt_indices.len()));
                    for (en_idx, idx) in a.virt_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&a.virts.slice(s![.., *idx]));
                    }
                    arr
                } else {
                    a.virts.to_owned()
                }
            }
        };

        // Check if the occupied or virtual orbitals of the second LE state are needed.
        let orbs_j: Array2<f64> = match kind_b {
            ElecHole::Hole => {
                if self.config.fmo_lc_tddftb.restrict_active_space {
                    let mut arr: Array2<f64> = Array2::zeros((b.occs.dim().0, b.occ_indices.len()));
                    for (en_idx, idx) in b.occ_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&b.occs.slice(s![.., *idx]));
                    }
                    arr
                } else {
                    b.occs.to_owned()
                }
            }
            ElecHole::Electron => {
                if self.config.fmo_lc_tddftb.restrict_active_space {
                    let mut arr: Array2<f64> =
                        Array2::zeros((b.virts.dim().0, b.virt_indices.len()));
                    for (en_idx, idx) in b.virt_indices.iter().enumerate() {
                        arr.slice_mut(s![.., en_idx])
                            .assign(&b.virts.slice(s![.., *idx]));
                    }
                    arr
                } else {
                    b.virts.to_owned()
                }
            }
        };

        // Number of molecular orbitals on monomer I.
        let dim_i: usize = orbs_i.ncols();
        // Number of molecular orbitals on monomer J.
        let dim_j: usize = orbs_j.ncols();
        // The transition charges between the two sets of MOs  are initialized.
        let mut q_trans: Array3<f64> = Array3::zeros([n_atoms, dim_i, dim_j]);
        // Matrix product of overlap matrix with the orbitals on I.
        let sc_mu_j: Array2<f64> = s.dot(&orbs_j);
        // Matrix product of overlap matrix with the orbitals on J.
        let sc_mu_i: Array2<f64> = s.t().dot(&orbs_i);
        let mut mu: usize = 0;
        // Iteration over all atoms (I).
        for (atom, mut q_n) in a.atoms.iter().zip(
            q_trans
                .slice_mut(s![0..n_atoms_i, .., ..])
                .axis_iter_mut(Axis(0)),
        ) {
            // Iteration over atomic orbitals mu on I.
            for _ in 0..atom.n_orbs {
                // Iteration over orbitals i on monomer I. orb_i -> C_(mu i) (mu on I, i on I)
                for (orb_i, mut q_i) in orbs_i.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                    // Iteration over S * C_J on monomer J. sc -> SC_(mu j) (mu on I, j on J)
                    for (sc, q) in sc_mu_j.row(mu).iter().zip(q_i.iter_mut()) {
                        // The transition charge is computed.
                        *q += orb_i * sc;
                    }
                }
                mu += 1;
            }
        }
        mu = 0;
        // Iteration over all atoms J.
        for (atom, mut q_n) in b.atoms.iter().zip(
            q_trans
                .slice_mut(s![n_atoms_i.., .., ..])
                .axis_iter_mut(Axis(0)),
        ) {
            // Iteration over atomic orbitals mu on J.
            for _ in 0..atom.n_orbs {
                // Iteration over occupied orbital i. sc -> SC_(mu i) (mu on J, i on I)
                for (sc, mut q_i) in sc_mu_i.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                    // Iteration over occupied orbital j. C_(mu j) (mu on J, j on J)
                    for (orb_j, q) in orbs_j.row(mu).iter().zip(q_i.iter_mut()) {
                        // The transition charge is computed.
                        *q += orb_j * sc;
                    }
                }
                mu += 1;
            }
        }
        0.5 * q_trans
    }
}
