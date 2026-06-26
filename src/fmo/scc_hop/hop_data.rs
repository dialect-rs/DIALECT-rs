//! HOP data structures and ghost atom creation for FMO-DFTB with HOP.
//!
//! Follows `dftbfo.src` conventions (fmolib.src:3061-3079):
//! - Each $FMOBND entry: BDA gets ZREF-=1, BAA's fragment gets ghost at BDA position
//! - Ghost ZAN=1 (ZREF=1), placed at BDA coordinates
//! - NATFRG includes ghost atoms
//! - Supersystem gamma includes ghost atoms

use crate::fmo::fragmentation::{build_graph, Graph};
use crate::fmo::SuperSystem;
use crate::initialization::atom::Atom;
use crate::initialization::parameters::RepulsivePotential;
use crate::scc::gamma_approximation::gamma_atomwise_hop;
use dialect_xtb::hop::{
    compute_hop_projector, detect_detached_bonds_xtb,
    DetachedBond, HOP_SHIFT,
};
use nalgebra::Vector3;
use ndarray::prelude::*;
use std::ops::Range;

/// DFTB sp3 hybrid coefficient for the s orbital (from $FMOHYB defaults).
/// Standard sp3 uses 0.5; Nishimoto uses 0.562060.
pub const DFTB_SP3_COEFF_S: f64 = 0.562060;

/// DFTB sp3 hybrid coefficient for the p orbital along bond (from $FMOHYB defaults).
/// Standard sp3 uses √3/2 = 0.866; Nishimoto uses 0.827096.
pub const DFTB_SP3_COEFF_P: f64 = 0.827096;

/// Per-fragment information in the extended (real + ghost) atom ordering.
#[derive(Debug, Clone)]
pub struct HopFragInfo {
    /// Fragment index in the SuperSystem
    pub frag_idx: usize,
    /// Range of this fragment's atoms (real + ghosts) in ext_atoms
    pub ext_range: Range<usize>,
    /// Number of real atoms
    pub n_real_atoms: usize,
    /// Number of ghost H atoms added
    pub n_ghost_atoms: usize,
    /// Total orbitals (real + ghost)
    pub n_ext_orbs: usize,
    /// Orbitals from real atoms only
    pub n_real_orbs: usize,
    /// Local indices (within ext_range) of BDA atoms
    pub bda_local_indices: Vec<usize>,
    /// Adjusted electron count (real electrons minus one per BDA bond)
    pub n_elec: usize,
}

/// All HOP-specific state for FMO-DFTB, kept separate from SuperSystem.
#[derive(Debug, Clone)]
pub struct HopData {
    /// All detected detached bonds (two entries per physical cut)
    pub detached_bonds: Vec<DetachedBond>,
    /// Extended atom list: [frag0_real, frag0_ghosts, frag1_real, frag1_ghosts, ...]
    pub ext_atoms: Vec<Atom>,
    /// Total number of extended atoms
    pub n_ext_atoms: usize,
    /// Per-fragment info
    pub frag_info: Vec<HopFragInfo>,
    /// Extended gamma matrix (n_ext_atoms x n_ext_atoms)
    pub gamma_ext: Array2<f64>,
    /// Reference charges: sum(valorbs_occupation) for real, 1 for ghosts, BDA -= 1
    pub zref: Array1<f64>,
    /// Unmodified reference charges (species QREF): sum(original_valorbs_occupation)
    /// Used for ZREF/QREF scaling of repulsive energy (reference convention)
    pub qref: Array1<f64>,
    /// Extended charge differences (Mulliken - ZREF)
    pub dq_ext: Array1<f64>,
    /// Monomer indices (atom indices per fragment, used for bond detection)
    pub monomer_indices: Vec<Vec<usize>>,
}

/// Create a ghost atom at the given position using the BDA atom as template.
///
/// In the reference convention (fmolib.src:3070-3075):
/// - Ghost IAN = BDA's atomic number (e.g. Carbon for C-C bond)
/// - Ghost IZAN/ZREF = 1 (one reference electron in s-orbital)
/// - Ghost has BDA's element type, Hubbard, orbitals, SK parameters
/// - Ghost valorbs_occupation = [1.0, 0.0, 0.0, ...] (only s-orbital occupied)
/// - Position = BDA atom's coordinates (ghost placed AT BDA in BAA's fragment)
pub fn create_ghost_atom(position: Vector3<f64>, bda_atom: &Atom) -> Atom {
    let mut ghost = bda_atom.clone();
    ghost.xyz = position;
    // Ghost has 1 reference electron in s-orbital only
    ghost.n_elec = 1;
    ghost.valorbs_occupation = vec![0.0; ghost.valorbs.len()];
    if !ghost.valorbs_occupation.is_empty() {
        ghost.valorbs_occupation[0] = 1.0; // s-orbital gets 1 electron
    }
    ghost
}


/// Compute the rotated sp3 hybrid orbital coefficients for DFTB.
///
/// Uses $FMOHYB coefficients (0.562060, 0.827096) instead of standard sp3 (0.5, 0.866).
///
/// IMPORTANT: DFTB p orbitals are ordered as real spherical harmonics:
///   AO index 1 = p_y (m=-1), index 2 = p_z (m=0), index 3 = p_x (m=+1)
/// So the hybrid is [c_s, c_p*b_y, c_p*b_z, c_p*b_x].
pub fn compute_rotated_sp3_dftb(bond_vec: &Vector3<f64>) -> Array1<f64> {
    let norm = bond_vec.norm();
    assert!(norm > 1e-14, "HOP: bond vector has zero length");
    let b_hat = bond_vec / norm;
    // p orbital ordering: (py, pz, px) = real spherical harmonics (m=-1, m=0, m=+1)
    array![
        DFTB_SP3_COEFF_S,
        DFTB_SP3_COEFF_P * b_hat.y,
        DFTB_SP3_COEFF_P * b_hat.z,
        DFTB_SP3_COEFF_P * b_hat.x
    ]
}

/// Compute the BDA DD matrix: shift * h * h^T (projects out bond-pointing hybrid).
pub fn compute_bda_dd_matrix(rotated_sp3: ArrayView1<f64>, shift: f64) -> Array2<f64> {
    let n = rotated_sp3.len();
    let mut dd = Array2::<f64>::zeros([n, n]);
    for i in 0..n {
        for j in 0..n {
            dd[[i, j]] = shift * rotated_sp3[i] * rotated_sp3[j];
        }
    }
    dd
}

/// Compute the ghost DD matrix for the 3 non-bond sp3 hybrids (analytical formula).
///
/// The reference code explicitly rotates all 4 FMOHYB hybrids and builds DD as the sum of
/// outer products of the 3 non-bond hybrids (japrjo=1):
///   DD_ghost = B * Σ_{i=2,3,4} h_i * h_i^T
///
/// The FMOHYB sp3 coefficients (0.562060, 0.827096) are NOT orthonormal
/// in coefficient space (h1·h2 ≈ 0.088), so I - h1*h1^T ≠ Σ h_i*h_i^T.
///
/// Using the tetrahedral identity Σ_{i=1..4} d_i * d_i^T = (4/3)*I₃, the analytical
/// formula for the sum of 3 non-bond outer products is:
///   DD[s,s] = B * 3 * c_s²
///   DD[s,p_j] = DD[p_j,s] = B * c_s * c_p * (-b_hat_j)
///   DD[p_j,p_k] = B * c_p² * (4/3 * δ_jk - b_hat_j * b_hat_k)
pub fn compute_ghost_nonbond_dd(bond_vec: &Vector3<f64>, shift: f64) -> Array2<f64> {
    let norm = bond_vec.norm();
    assert!(norm > 1e-14, "HOP: bond vector has zero length");
    // p orbital ordering: (py, pz, px) = real spherical harmonics (m=-1, m=0, m=+1)
    let b = [bond_vec.y / norm, bond_vec.z / norm, bond_vec.x / norm];

    let cs = DFTB_SP3_COEFF_S;
    let cp = DFTB_SP3_COEFF_P;
    let cs2 = cs * cs;
    let cp2 = cp * cp;

    let mut dd = Array2::<f64>::zeros([4, 4]);

    // s-s block: 3 * c_s^2
    dd[[0, 0]] = shift * 3.0 * cs2;

    // s-p and p-s cross terms: c_s * c_p * (-b_hat_j)
    for j in 0..3 {
        let val = shift * cs * cp * (-b[j]);
        dd[[0, j + 1]] = val;
        dd[[j + 1, 0]] = val;
    }

    // p-p block: c_p^2 * (4/3 * δ_jk - b_hat_j * b_hat_k)
    for j in 0..3 {
        for k in 0..3 {
            let delta = if j == k { 1.0 } else { 0.0 };
            dd[[j + 1, k + 1]] = shift * cp2 * (4.0 / 3.0 * delta - b[j] * b[k]);
        }
    }

    dd
}

/// Build all HOP data from the SuperSystem.
///
/// Follows the reference convention (fmolib.src:3057-3082):
/// - For each physical bond cut (one $FMOBND entry):
///   - BDA's fragment: BDA atom ZREF -= 1, n_elec -= 1
///   - BAA's fragment: ghost at BDA position, ZREF(ghost) = 1, n_elec += 1
/// - HOP projector applied only at BDA atoms
///
/// This constructs:
/// 1. The bond graph and unique detached bonds (one per physical cut)
/// 2. Extended atom lists (real + ghost per fragment)
/// 3. Reference charges (ZREF) with BDA/ghost adjustments
/// 4. Extended gamma matrix over all extended atoms
pub fn build_hop_data(supersystem: &SuperSystem) -> HopData {
    let atoms = &supersystem.atoms;
    let n_mol = supersystem.n_mol;

    // 1. Extract monomer indices from SuperSystem
    let monomer_indices: Vec<Vec<usize>> = supersystem
        .monomers
        .iter()
        .map(|m| m.slice.atom_as_range().collect())
        .collect();

    // 2. Build bond graph and detect detached bonds (both directions)
    let graph: Graph = build_graph(atoms.len(), atoms);
    let all_bonds = detect_detached_bonds_xtb(&monomer_indices, &graph);

    // Deduplicate: keep only one direction per physical bond (bda_global < baa_global).
    // In the reference convention, BDA is in the lower-numbered fragment.
    let detached_bonds: Vec<DetachedBond> = all_bonds
        .into_iter()
        .filter(|b| b.bda_global < b.baa_global)
        .collect();

    // 3. Build extended atoms and fragment info
    //
    // Reference convention per bond (ia=BDA, ja=BAA):
    //   - if indat(ja)==ifg: add ghost at BDA position with BDA's element type, zanfrg=1
    //   - if indat(ia)==ifg: zanfrg(BDA) -= 1
    let mut ext_atoms: Vec<Atom> = Vec::new();
    let mut frag_info: Vec<HopFragInfo> = Vec::new();
    let mut zref_vec: Vec<f64> = Vec::new();
    let mut qref_vec: Vec<f64> = Vec::new();

    for frag_idx in 0..n_mol {
        let frag_start = ext_atoms.len();

        // Real atoms for this fragment
        let atom_range = supersystem.monomers[frag_idx].slice.atom_as_range();
        let real_atoms: Vec<Atom> = atoms[atom_range.clone()].to_vec();
        let n_real_atoms = real_atoms.len();
        let n_real_orbs: usize = real_atoms.iter().map(|a| a.n_orbs).sum();

        // Count BDA bonds where BDA is in this fragment (for ZREF adjustment)
        let mut bda_bond_count: Vec<usize> = vec![0; n_real_atoms];
        let mut bda_local_indices: Vec<usize> = Vec::new();

        for bond in &detached_bonds {
            if bond.bda_fragment == frag_idx {
                // BDA is in this fragment → ZREF -= 1
                let bda_local = bond.bda_global - atom_range.start;
                bda_bond_count[bda_local] += 1;
                if !bda_local_indices.contains(&bda_local) {
                    bda_local_indices.push(bda_local);
                }
            }
        }

        // Collect ghost atoms: for each bond where BAA is in this fragment,
        // place a ghost at the BDA position using BDA's element type
        // (ghost IAN = BDA's atomic number, ghost in BAA's fragment at BDA coords)
        let mut ghost_atoms: Vec<Atom> = Vec::new();
        for bond in &detached_bonds {
            if bond.baa_fragment == frag_idx {
                // BAA is in this fragment → add ghost at BDA's position
                let bda_pos = atoms[bond.bda_global].xyz;
                let bda_atom = &atoms[bond.bda_global];
                let ghost = create_ghost_atom(bda_pos, bda_atom);
                ghost_atoms.push(ghost);
            }
        }
        let n_ghost_atoms = ghost_atoms.len();
        let n_ghost_orbs: usize = ghost_atoms.iter().map(|a| a.n_orbs).sum();
        let n_ext_orbs = n_real_orbs + n_ghost_orbs;

        // Append real atoms to ext_atoms
        ext_atoms.extend(real_atoms.iter().cloned());

        // ZREF for real atoms: sum(valorbs_occupation), then subtract 1 per BDA bond
        // QREF for real atoms: unmodified sum(valorbs_occupation) (species reference)
        for (local_idx, atom) in real_atoms.iter().enumerate() {
            let qr: f64 = atom.valorbs_occupation.iter().sum();
            let zr: f64 = qr - bda_bond_count[local_idx] as f64;
            zref_vec.push(zr);
            qref_vec.push(qr);
        }

        // Append ghost atoms to ext_atoms
        ext_atoms.extend(ghost_atoms.iter().cloned());

        // ZREF for ghosts = 1 (IZAN=1, IAN!=IZAN → ZREF=1)
        // QREF for ghosts = original BDA atom's QREF (species reference)
        for bond in &detached_bonds {
            if bond.baa_fragment == frag_idx {
                let bda_atom = &atoms[bond.bda_global];
                let bda_qref: f64 = bda_atom.valorbs_occupation.iter().sum();
                zref_vec.push(1.0);
                qref_vec.push(bda_qref);
            }
        }

        let frag_end = ext_atoms.len();

        // Adjusted electron count:
        // n_elec = real_n_elec - n_bda_bonds_in_frag + n_ghosts_in_frag
        // BDA loses 1 electron, ghost provides 1 electron
        let n_real_elec: usize = real_atoms.iter().map(|a| a.n_elec).sum();
        let n_bda_bonds: usize = bda_bond_count.iter().sum();
        let n_elec = n_real_elec - n_bda_bonds + n_ghost_atoms;

        frag_info.push(HopFragInfo {
            frag_idx,
            ext_range: frag_start..frag_end,
            n_real_atoms,
            n_ghost_atoms,
            n_ext_orbs,
            n_real_orbs,
            bda_local_indices,
            n_elec,
        });
    }

    let n_ext_atoms = ext_atoms.len();
    let zref = Array1::from(zref_vec);
    let qref = Array1::from(qref_vec);

    // 4. Compute extended gamma matrix
    let gf = supersystem.gammafunction.clone();
    let gamma_ext = gamma_atomwise_hop(&gf, &ext_atoms, n_ext_atoms);

    // 5. Initialize dq_ext to zeros
    let dq_ext = Array1::zeros(n_ext_atoms);

    HopData {
        detached_bonds,
        ext_atoms,
        n_ext_atoms,
        frag_info,
        gamma_ext,
        zref,
        qref,
        dq_ext,
        monomer_indices,
    }
}

/// Get the AO range for a DFTB atom at a local index within a fragment's extended atoms.
///
/// Returns (ao_start, nao) where ao_start is the first AO index and nao is the number of AOs.
pub fn get_bda_ao_range_dftb(ext_atoms_slice: &[Atom], local_atom_index: usize) -> (usize, usize) {
    let mut ao_start: usize = 0;
    for (idx, atom) in ext_atoms_slice.iter().enumerate() {
        if idx == local_atom_index {
            return (ao_start, atom.n_orbs);
        }
        ao_start += atom.n_orbs;
    }
    panic!(
        "DFTB HOP: atom index {} not found in ext_atoms slice of len {}",
        local_atom_index,
        ext_atoms_slice.len()
    );
}

/// Compute HOP projector for a monomer fragment.
///
/// Two types of HOP projection per cut bond (matching the reference):
/// 1. **BDA projection** (bond where BDA is in this fragment):
///    Project out 1 bond-pointing sp3 hybrid on the BDA atom.
/// 2. **Ghost projection** (bond where BAA is in this fragment, ghost was placed):
///    Project out 3 non-bond-pointing sp3 hybrids on the ghost atom,
///    leaving only the bond-pointing hybrid active.
///
/// This ensures the ghost effectively contributes only 1 orbital per bond.
pub fn compute_monomer_hop_projector(
    detached_bonds: &[DetachedBond],
    frag_idx: usize,
    ext_atoms_frag: &[Atom],
    s_frag: ArrayView2<f64>,
    atoms: &[Atom],
    frag_atom_range: &Range<usize>,
    n_real_atoms: usize,
) -> Option<Array2<f64>> {
    // Bonds where BDA is in this fragment → BDA projection (1 orbital)
    let bda_bonds: Vec<&DetachedBond> = detached_bonds
        .iter()
        .filter(|b| b.bda_fragment == frag_idx)
        .collect();

    // Bonds where BAA is in this fragment → ghost projection (3 orbitals)
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

        let rotated_sp3 = compute_rotated_sp3_dftb(&bond_vec);
        let dd = compute_bda_dd_matrix(rotated_sp3.view(), HOP_SHIFT);

        let bda_local = bond.bda_global - frag_atom_range.start;
        let (ao_start, nao) = get_bda_ao_range_dftb(ext_atoms_frag, bda_local);

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
    // Ghost atoms are appended after real atoms in the order they were created
    // in build_hop_data (matching the order of ghost_bonds).
    for (ghost_idx, bond) in ghost_bonds.iter().enumerate() {
        // Bond direction: from BDA (= ghost position) toward BAA (= real atom in this fragment)
        let bda_pos = atoms[bond.bda_global].xyz;
        let baa_pos = atoms[bond.baa_global].xyz;
        let bond_vec = baa_pos - bda_pos;

        let dd_ghost = compute_ghost_nonbond_dd(&bond_vec, HOP_SHIFT);

        // Ghost's local index: after real atoms
        let ghost_local = n_real_atoms + ghost_idx;
        let (ao_start, nao) = get_bda_ao_range_dftb(ext_atoms_frag, ghost_local);

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

/// Compute repulsive energy with ZREF/QREF scaling (DFTB_EREP convention).
///
/// Each pair's repulsive potential is scaled by:
///   erep_ij *= (ZREF[i] / QREF[i]) * (ZREF[j] / QREF[j])
///
/// This reduces contributions from BDA atoms (ZREF < QREF) and ghost atoms (ZREF=1, QREF=4 for C).
pub fn get_repulsive_energy_scaled(
    atoms: &[Atom],
    n_atoms: usize,
    v_rep: &RepulsivePotential,
    zref: ArrayView1<f64>,
    qref: ArrayView1<f64>,
) -> f64 {
    let mut e_rep: f64 = 0.0;
    for i in 1..n_atoms {
        let ci = if qref[i] > 1e-14 { zref[i] / qref[i] } else { 0.0 };
        for j in 0..i {
            let cj = if qref[j] > 1e-14 { zref[j] / qref[j] } else { 0.0 };
            let r: f64 = (&atoms[i] - &atoms[j]).norm();
            let val = v_rep.get(atoms[i].kind, atoms[j].kind).spline_eval(r);
            e_rep += val * ci * cj;
        }
    }
    e_rep
}
