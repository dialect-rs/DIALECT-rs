//! HOP data structures and ghost atom creation for FMO-xTB with HOP.
//!
//! Follows the reference convention (fmolib.src:3057-3082):
//! - Each $FMOBND entry: BDA gets ZREF-=1, BAA's fragment gets ghost at BDA position
//! - Ghost is same element as BDA (not H), ZREF(ghost)=1
//! - Ghost placed at BDA coordinates (in BAA's fragment)
//! - NATFRG includes ghost atoms
//! - Supersystem gamma_shell includes ghost atoms

use crate::hop::{
    compute_hop_projector, compute_rotated_sp3, compute_dd_matrix,
    detect_detached_bonds_xtb, get_bda_ao_range,
    DetachedBond, HOP_SHIFT, SP3_COEFF_S, SP3_COEFF_P,
};
use crate::fmo::supersystem::XtbSuperSystem;
use crate::initialization::atom::XtbAtom;
use crate::initialization::basis::{create_basis_set, Basis};
use crate::parameters::{REFERENCE_OCCUPATION, REP_ALPHA_PARAMS, REP_Z_EFF_PARAMS};
use crate::scc::gamma_matrix::{gamma_matrix_shell, XtbGammaFunction};
use crate::scc::hamiltonian::calculate_coordination_numbers;
use nalgebra::Vector3;
use ndarray::prelude::*;
use std::ops::Range;

/// Per-fragment information in the extended (real + ghost) atom ordering.
#[derive(Debug, Clone)]
pub struct XtbHopFragInfo {
    /// Fragment index in the SuperSystem
    pub frag_idx: usize,
    /// Range of this fragment's atoms (real + ghosts) in ext_atoms
    pub ext_range: Range<usize>,
    /// Number of real atoms
    pub n_real_atoms: usize,
    /// Number of ghost atoms added
    pub n_ghost_atoms: usize,
    /// Total orbitals (real + ghost)
    pub n_ext_orbs: usize,
    /// Orbitals from real atoms only
    pub n_real_orbs: usize,
    /// Total shells (real + ghost)
    pub n_ext_shells: usize,
    /// Shells from real atoms only
    pub n_real_shells: usize,
    /// Local indices (within ext_range) of BDA atoms
    pub bda_local_indices: Vec<usize>,
    /// Adjusted electron count (real electrons minus one per BDA bond)
    pub n_elec: usize,
}

/// All HOP-specific state for FMO-xTB.
#[derive(Debug, Clone)]
pub struct XtbHopData {
    /// Deduplicated detached bonds (bda_global < baa_global)
    pub detached_bonds: Vec<DetachedBond>,
    /// Extended atom list: [frag0_real, frag0_ghosts, frag1_real, frag1_ghosts, ...]
    pub ext_atoms: Vec<XtbAtom>,
    /// Total number of extended atoms
    pub n_ext_atoms: usize,
    /// Per-fragment info
    pub frag_info: Vec<XtbHopFragInfo>,
    /// Extended gamma matrix (shell-resolved, over all ext_atoms)
    pub gamma_shell_ext: Array2<f64>,
    /// Response-corrected gamma: ghost-BDA same-position entries atom-averaged.
    /// Used only in response gradient (Lagrangian, SCZV, addlag, inter-frag response).
    pub gamma_shell_ext_response: Array2<f64>,
    /// Extended basis for all ext_atoms
    pub ext_basis: Basis,
    /// Reference charges: sum(REFERENCE_OCCUPATION) for real, 1 for ghosts, BDA -= 1
    pub zref: Array1<f64>,
    /// Unmodified reference charges (QREF): sum(REFERENCE_OCCUPATION)
    pub qref: Array1<f64>,
    /// Extended shell-level charge differences
    pub dq_shell_ext: Array1<f64>,
    /// Extended atom-level charge differences
    pub dq_ext: Array1<f64>,
    /// Monomer indices (atom indices per fragment)
    pub monomer_indices: Vec<Vec<usize>>,
    /// Coordination numbers for extended atoms
    pub cn_ext: Array1<f64>,
}

/// Create a ghost atom at the given position using the BDA atom as template.
///
/// Ghost is the SAME ELEMENT as BDA (not H). ZREF(ghost) = 1.
/// Ghost provides same basis functions and Hubbard parameters as BDA.
pub fn create_xtb_ghost_atom(position: Vector3<f64>, bda_atom: &XtbAtom) -> XtbAtom {
    let mut ghost = XtbAtom::from(bda_atom.number);
    ghost.xyz = position;
    // Ghost has 1 electron (ZREF=1). n_elec is used for initial MO filling.
    ghost.n_elec = 1;
    ghost
}

/// Get QREF for an XtbAtom: sum of REFERENCE_OCCUPATION[0..3]
fn get_qref(atom: &XtbAtom) -> f64 {
    let idx = atom.number as usize - 1;
    REFERENCE_OCCUPATION[idx][0] + REFERENCE_OCCUPATION[idx][1] + REFERENCE_OCCUPATION[idx][2]
}

/// Build all HOP data from the XtbSuperSystem.
///
/// Constructs:
/// 1. Deduplicated detached bonds (one per physical cut)
/// 2. Extended atom lists (real + ghost per fragment)
/// 3. Reference charges (ZREF) with BDA/ghost adjustments
/// 4. Extended shell-resolved gamma matrix
/// 5. Coordination numbers for extended atoms
pub fn build_xtb_hop_data(supersystem: &XtbSuperSystem) -> XtbHopData {
    let atoms = &supersystem.atoms;
    let n_mol = supersystem.n_mol;

    // 1. Extract monomer indices
    let monomer_indices: Vec<Vec<usize>> = supersystem
        .monomers
        .iter()
        .map(|m| m.slice.atom_as_range().collect())
        .collect();

    // 2. Deduplicate detached bonds: keep only bda_global < baa_global
    let detached_bonds: Vec<DetachedBond> = supersystem
        .detached_bonds
        .iter()
        .filter(|b| b.bda_global < b.baa_global)
        .cloned()
        .collect();

    // 3. Build extended atoms and fragment info
    let mut ext_atoms: Vec<XtbAtom> = Vec::new();
    let mut frag_info: Vec<XtbHopFragInfo> = Vec::new();
    let mut zref_vec: Vec<f64> = Vec::new();
    let mut qref_vec: Vec<f64> = Vec::new();

    for frag_idx in 0..n_mol {
        let frag_start = ext_atoms.len();

        // Real atoms for this fragment
        let atom_range = supersystem.monomers[frag_idx].slice.atom_as_range();
        let real_atoms: Vec<XtbAtom> = atoms[atom_range.clone()].to_vec();
        let n_real_atoms = real_atoms.len();

        // Create real basis for counting orbs/shells
        let real_basis = create_basis_set(&real_atoms);
        let n_real_orbs = real_basis.nbas;
        let n_real_shells = real_basis.shells.len();

        // Count BDA bonds where BDA is in this fragment
        let mut bda_bond_count: Vec<usize> = vec![0; n_real_atoms];
        let mut bda_local_indices: Vec<usize> = Vec::new();

        for bond in &detached_bonds {
            if bond.bda_fragment == frag_idx {
                let bda_local = bond.bda_global - atom_range.start;
                bda_bond_count[bda_local] += 1;
                if !bda_local_indices.contains(&bda_local) {
                    bda_local_indices.push(bda_local);
                }
            }
        }

        // Collect ghost atoms: for each bond where BAA is in this fragment,
        // place ghost at BDA's position using BDA's element type
        let mut ghost_atoms: Vec<XtbAtom> = Vec::new();
        for bond in &detached_bonds {
            if bond.baa_fragment == frag_idx {
                let bda_pos = atoms[bond.bda_global].xyz;
                let bda_atom = &atoms[bond.bda_global];
                ghost_atoms.push(create_xtb_ghost_atom(bda_pos, bda_atom));
            }
        }
        let n_ghost_atoms = ghost_atoms.len();

        // Build extended basis to count ghost orbs/shells
        let ext_frag_atoms: Vec<XtbAtom> = real_atoms
            .iter()
            .chain(ghost_atoms.iter())
            .cloned()
            .collect();
        let ext_frag_basis = create_basis_set(&ext_frag_atoms);
        let n_ext_orbs = ext_frag_basis.nbas;
        let n_ext_shells = ext_frag_basis.shells.len();

        // Append real atoms to ext_atoms
        ext_atoms.extend(real_atoms.iter().cloned());

        // ZREF/QREF for real atoms
        for (local_idx, atom) in real_atoms.iter().enumerate() {
            let qr = get_qref(atom);
            let zr = qr - bda_bond_count[local_idx] as f64;
            zref_vec.push(zr);
            qref_vec.push(qr);
        }

        // Append ghost atoms
        ext_atoms.extend(ghost_atoms.iter().cloned());

        // ZREF/QREF for ghosts
        for bond in &detached_bonds {
            if bond.baa_fragment == frag_idx {
                let bda_atom = &atoms[bond.bda_global];
                let bda_qref = get_qref(bda_atom);
                zref_vec.push(1.0);
                qref_vec.push(bda_qref);
            }
        }

        let frag_end = ext_atoms.len();

        // Adjusted electron count
        let n_real_elec: usize = real_atoms.iter().map(|a| a.n_elec).sum();
        let n_bda_bonds: usize = bda_bond_count.iter().sum();
        let n_elec = n_real_elec - n_bda_bonds + n_ghost_atoms;

        frag_info.push(XtbHopFragInfo {
            frag_idx,
            ext_range: frag_start..frag_end,
            n_real_atoms,
            n_ghost_atoms,
            n_ext_orbs,
            n_real_orbs,
            n_ext_shells,
            n_real_shells,
            bda_local_indices,
            n_elec,
        });
    }

    let n_ext_atoms = ext_atoms.len();
    let zref = Array1::from(zref_vec);
    let qref = Array1::from(qref_vec);

    // 4. Build extended basis and gamma_shell
    let ext_basis = create_basis_set(&ext_atoms);
    let gammafunc = supersystem.monomers[0].gammafunction.clone();
    let mut gamma_shell_ext = gamma_matrix_shell(&gammafunc, &ext_atoms, &ext_basis);

    // 4b. Build response-corrected gamma: atom-average ghost-BDA same-position entries.
    // The SCC uses the original gamma_shell_ext (shell-level). The response gradient
    // uses gamma_shell_ext_response (atom-averaged at ghost-BDA positions) to ensure
    // DFTB-like exact cancellation in the inter-fragment coupling.
    let mut gamma_shell_ext_response = gamma_shell_ext.clone();
    {
        let monomer_indices_ref = &supersystem.monomers.iter()
            .map(|m| m.slice.atom_as_range().collect::<Vec<usize>>())
            .collect::<Vec<Vec<usize>>>();
        for bond in &detached_bonds {
            let fi_baa = &frag_info[bond.baa_fragment];
            let fi_bda = &frag_info[bond.bda_fragment];
            // Find ghost ext_atom index
            let mut ghost_ext = None;
            let mut ghost_count = 0;
            for b in &detached_bonds {
                if b.baa_fragment == bond.baa_fragment {
                    if std::ptr::eq(b, bond) {
                        ghost_ext = Some(fi_baa.ext_range.start + fi_baa.n_real_atoms + ghost_count);
                        break;
                    }
                    ghost_count += 1;
                }
            }
            let ghost_ext = ghost_ext.unwrap();
            let bda_local = bond.bda_global - monomer_indices_ref[bond.bda_fragment][0];
            let bda_ext = fi_bda.ext_range.start + bda_local;
            // Find shell indices for ghost and BDA
            let ghost_shells: Vec<usize> = ext_basis.shells.iter().enumerate()
                .filter(|(_, s)| s.atom_index == ghost_ext)
                .map(|(i, _)| i).collect();
            let bda_shells: Vec<usize> = ext_basis.shells.iter().enumerate()
                .filter(|(_, s)| s.atom_index == bda_ext)
                .map(|(i, _)| i).collect();
            if ghost_shells.is_empty() || bda_shells.is_empty() { continue; }
            // Compute atom-averaged gamma over all ghost-BDA shell pairs
            let mut gamma_sum = 0.0;
            let n_pairs = ghost_shells.len() * bda_shells.len();
            for &gs in &ghost_shells {
                for &bs in &bda_shells {
                    gamma_sum += gamma_shell_ext_response[[gs, bs]];
                }
            }
            let gamma_avg = gamma_sum / n_pairs as f64;
            // Replace in response gamma only
            for &gs in &ghost_shells {
                for &bs in &bda_shells {
                    gamma_shell_ext_response[[gs, bs]] = gamma_avg;
                    gamma_shell_ext_response[[bs, gs]] = gamma_avg;
                }
            }
        }
    }

    // 5. Coordination numbers for extended atoms (ghosts get CN=0)
    let cn_real = calculate_coordination_numbers(atoms);
    let mut cn_ext = Array1::zeros(n_ext_atoms);
    let mut real_offset = 0;
    for fi in &frag_info {
        let atom_range_start = fi.ext_range.start;
        cn_ext
            .slice_mut(s![atom_range_start..atom_range_start + fi.n_real_atoms])
            .assign(&cn_real.slice(s![real_offset..real_offset + fi.n_real_atoms]));
        real_offset += fi.n_real_atoms;
        // Ghost atoms remain zero
    }

    // 6. Initialize dq arrays
    let n_ext_shells = ext_basis.shells.len();
    let dq_shell_ext = Array1::zeros(n_ext_shells);
    let dq_ext = Array1::zeros(n_ext_atoms);

    XtbHopData {
        detached_bonds,
        ext_atoms,
        n_ext_atoms,
        frag_info,
        gamma_shell_ext,
        gamma_shell_ext_response,
        ext_basis,
        zref,
        qref,
        dq_shell_ext,
        dq_ext,
        monomer_indices,
        cn_ext,
    }
}

/// Compute the rotated sp3 hybrid orbital coefficients for xTB.
///
/// Uses xTB coefficients (0.5, √3/2) with xTB p-orbital ordering (py, pz, px).
/// Returns [c_s, c_p*b_y, c_p*b_z, c_p*b_x].
pub fn compute_rotated_sp3_xtb(bond_vec: &Vector3<f64>) -> Array1<f64> {
    let norm = bond_vec.norm();
    assert!(norm > 1e-14, "HOP: bond vector has zero length");
    let b_hat = bond_vec / norm;
    // p-orbital ordering: (py, pz, px) matching permuts_2 for l=1
    array![
        SP3_COEFF_S,
        SP3_COEFF_P * b_hat.y,
        SP3_COEFF_P * b_hat.z,
        SP3_COEFF_P * b_hat.x
    ]
}

/// Compute the BDA DD matrix: shift * h * h^T (projects out bond-pointing hybrid).
pub fn compute_bda_dd_xtb(rotated_sp3: ArrayView1<f64>, shift: f64) -> Array2<f64> {
    let n = rotated_sp3.len();
    let mut dd = Array2::<f64>::zeros([n, n]);
    for i in 0..n {
        for j in 0..n {
            dd[[i, j]] = shift * rotated_sp3[i] * rotated_sp3[j];
        }
    }
    dd
}

/// Compute the ghost DD matrix for the 3 non-bond sp3 hybrids.
///
/// For xTB, the sp3 coefficients (0.5, √3/2) ARE orthonormal:
///   c_s^2 + c_p^2 = 0.25 + 0.75 = 1.0
/// So the complement projection is simply: I - h*h^T
/// DD_ghost = shift * (I - h*h^T) where I is the 4x4 identity
/// projected onto the 3 non-bond directions.
///
/// Since xTB sp3 IS orthonormal, Σ_{i=1..4} h_i*h_i^T = I (for s+p block).
/// So: Σ_{i=2,3,4} h_i*h_i^T = I - h_1*h_1^T
/// which gives: DD_ghost = shift * (I - h1*h1^T) for the s+p block.
pub fn compute_ghost_nonbond_dd_xtb(bond_vec: &Vector3<f64>, shift: f64) -> Array2<f64> {
    let norm = bond_vec.norm();
    assert!(norm > 1e-14, "HOP: bond vector has zero length");
    let b_hat = bond_vec / norm;

    // p-orbital ordering: (py, pz, px) matching permuts_2 for l=1
    let hybrid = array![
        SP3_COEFF_S,
        SP3_COEFF_P * b_hat.y,
        SP3_COEFF_P * b_hat.z,
        SP3_COEFF_P * b_hat.x
    ];

    // DD_ghost = shift * (I - h*h^T) for 4x4 s+p block
    let mut dd = Array2::<f64>::zeros([4, 4]);
    for i in 0..4 {
        for j in 0..4 {
            let delta = if i == j { 1.0 } else { 0.0 };
            dd[[i, j]] = shift * (delta - hybrid[i] * hybrid[j]);
        }
    }

    dd
}

/// Compute HOP projector for a monomer fragment (xTB version).
///
/// Two types of HOP projection per cut bond:
/// 1. BDA projection: project out 1 bond-pointing sp3 hybrid on the BDA atom.
/// 2. Ghost projection: project out 3 non-bond sp3 hybrids on the ghost atom,
///    leaving only the bond-pointing hybrid active.
pub fn compute_monomer_hop_projector_xtb(
    detached_bonds: &[DetachedBond],
    frag_idx: usize,
    ext_basis: &Basis,
    s_frag: ArrayView2<f64>,
    atoms: &[XtbAtom],
    frag_atom_range: &Range<usize>,
    n_real_atoms: usize,
) -> Option<Array2<f64>> {
    let bda_bonds: Vec<&DetachedBond> = detached_bonds
        .iter()
        .filter(|b| b.bda_fragment == frag_idx)
        .collect();
    let ghost_bonds: Vec<&DetachedBond> = detached_bonds
        .iter()
        .filter(|b| b.baa_fragment == frag_idx)
        .collect();

    if bda_bonds.is_empty() && ghost_bonds.is_empty() {
        return None;
    }

    let n_ext_orbs = s_frag.nrows();
    let mut p_hop_total = Array2::<f64>::zeros([n_ext_orbs, n_ext_orbs]);

    // 1. BDA projection: project out 1 bond-pointing hybrid on each BDA
    for bond in &bda_bonds {
        let bda_pos = atoms[bond.bda_global].xyz;
        let baa_pos = atoms[bond.baa_global].xyz;
        let bond_vec = baa_pos - bda_pos;

        let rotated_sp3 = compute_rotated_sp3_xtb(&bond_vec);
        let dd = compute_bda_dd_xtb(rotated_sp3.view(), HOP_SHIFT);

        let bda_local = bond.bda_global - frag_atom_range.start;
        let (ao_start, nao) = get_bda_ao_range(ext_basis, bda_local);

        let p_hop = if nao == dd.nrows() {
            compute_hop_projector(s_frag, dd.view(), ao_start, nao)
        } else {
            let mut dd_full = Array2::<f64>::zeros([nao, nao]);
            let dd_size = dd.nrows().min(nao);
            dd_full
                .slice_mut(s![..dd_size, ..dd_size])
                .assign(&dd.slice(s![..dd_size, ..dd_size]));
            compute_hop_projector(s_frag, dd_full.view(), ao_start, nao)
        };

        p_hop_total += &p_hop;
    }

    // 2. Ghost projection: project out 3 non-bond hybrids on each ghost
    for (ghost_idx, bond) in ghost_bonds.iter().enumerate() {
        let bda_pos = atoms[bond.bda_global].xyz;
        let baa_pos = atoms[bond.baa_global].xyz;
        let bond_vec = baa_pos - bda_pos;

        let dd_ghost = compute_ghost_nonbond_dd_xtb(&bond_vec, HOP_SHIFT);

        let ghost_local = n_real_atoms + ghost_idx;
        let (ao_start, nao) = get_bda_ao_range(ext_basis, ghost_local);

        let p_hop = if nao == dd_ghost.nrows() {
            compute_hop_projector(s_frag, dd_ghost.view(), ao_start, nao)
        } else {
            let mut dd_full = Array2::<f64>::zeros([nao, nao]);
            let dd_size = dd_ghost.nrows().min(nao);
            dd_full
                .slice_mut(s![..dd_size, ..dd_size])
                .assign(&dd_ghost.slice(s![..dd_size, ..dd_size]));
            compute_hop_projector(s_frag, dd_full.view(), ao_start, nao)
        };

        p_hop_total += &p_hop;
    }

    Some(p_hop_total)
}

/// Compute repulsive energy with ZREF/QREF scaling for xTB.
///
/// Each pair's repulsive potential is scaled by:
///   erep_ij *= (ZREF[i] / QREF[i]) * (ZREF[j] / QREF[j])
pub fn calculate_repulsive_energy_xtb_scaled(
    atoms: &[XtbAtom],
    zref: ArrayView1<f64>,
    qref: ArrayView1<f64>,
) -> f64 {
    let n_atoms = atoms.len();
    let mut erep: f64 = 0.0;

    for i in 1..n_atoms {
        let ci = if qref[i] > 1e-14 {
            zref[i] / qref[i]
        } else {
            0.0
        };
        let z_eff_i = REP_Z_EFF_PARAMS[atoms[i].number as usize - 1];
        let alpha_i = REP_ALPHA_PARAMS[atoms[i].number as usize - 1];

        for j in 0..i {
            let cj = if qref[j] > 1e-14 {
                zref[j] / qref[j]
            } else {
                0.0
            };
            let z_eff_j = REP_Z_EFF_PARAMS[atoms[j].number as usize - 1];
            let alpha_j = REP_ALPHA_PARAMS[atoms[j].number as usize - 1];

            let diff = &atoms[i] - &atoms[j];
            let distance = (diff.x * diff.x + diff.y * diff.y + diff.z * diff.z).sqrt();
            let energy_val = (-(alpha_i * alpha_j).sqrt() * distance.powf(1.5)).exp()
                * z_eff_i
                * z_eff_j
                / distance;
            erep += energy_val * ci * cj;
        }
    }
    erep
}

/// Get the shell range for a fragment in the extended basis.
pub fn get_frag_shell_range(ext_basis: &Basis, ext_range: &Range<usize>) -> Range<usize> {
    let mut shell_start = usize::MAX;
    let mut shell_end = 0;
    for (idx, shell) in ext_basis.shells.iter().enumerate() {
        if shell.atom_index >= ext_range.start && shell.atom_index < ext_range.end {
            if shell_start == usize::MAX {
                shell_start = idx;
            }
            shell_end = idx + 1;
        }
    }
    if shell_start == usize::MAX {
        0..0
    } else {
        shell_start..shell_end
    }
}
