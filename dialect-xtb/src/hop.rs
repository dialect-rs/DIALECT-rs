//! Hybrid Orbital Projection (HOP) for FMO-xTB covalent fragmentation.
//!
//! When FMO fragments a molecule by cutting covalent bonds, each fragment has a
//! Bond Detached Atom (BDA) with a dangling bond. HOP adds a large energy penalty
//! to the sp3 bond hybrid orbital on the BDA, forcing electrons away from the bond region.
//!
//! For xTB (s+p valence only), the rotated sp3 hybrid is trivially:
//!   rotated_sp3 = [0.5, (√3/2)·b̂_x, (√3/2)·b̂_y, (√3/2)·b̂_z]
//! where b̂ = unit bond vector (BDA→BAA).

use dialect_utilities::fragmentation::Graph;
use crate::initialization::basis::Basis;
use ndarray::prelude::*;
use nalgebra::Vector3;

// ============================================================================
// Constants
// ============================================================================

/// sp3 hybrid coefficient for the s orbital: 1/2
pub const SP3_COEFF_S: f64 = 0.5;

/// sp3 hybrid coefficient for the p orbital along bond: √3/2
pub const SP3_COEFF_P: f64 = 0.8660254037844386;

/// Energy penalty (Hartree) matching ORSHFT.
pub const HOP_SHIFT: f64 = 1.0e6;

// ============================================================================
// Types
// ============================================================================

/// A covalent bond cut between two FMO fragments.
///
/// Each cut bond produces TWO DetachedBond entries (one per fragment side).
/// The BDA (Bond Detached Atom) is the atom inside the fragment; the BAA (Bond
/// Attached Atom) is on the other side.
#[derive(Debug, Clone)]
pub struct DetachedBond {
    /// Global atom index of the BDA (inside fragment)
    pub bda_global: usize,
    /// Global atom index of the BAA (outside fragment)
    pub baa_global: usize,
    /// Fragment index containing the BDA
    pub bda_fragment: usize,
    /// Fragment index containing the BAA
    pub baa_fragment: usize,
}

// ============================================================================
// Bond detection
// ============================================================================

/// Scan the bond graph for edges crossing fragment boundaries.
///
/// Each cross-fragment edge (i, j) produces TWO DetachedBond entries:
///   - One where atom i is BDA (in its fragment) and j is BAA
///   - One where atom j is BDA (in its fragment) and i is BAA
pub fn detect_detached_bonds_xtb(
    monomer_indices: &[Vec<usize>],
    graph: &Graph,
) -> Vec<DetachedBond> {
    // Build atom → fragment index map
    let max_atom = monomer_indices.iter().flat_map(|v| v.iter()).copied().max().unwrap_or(0);
    let mut atom_to_frag = vec![0usize; max_atom + 1];
    for (frag_idx, indices) in monomer_indices.iter().enumerate() {
        for &atom_idx in indices {
            atom_to_frag[atom_idx] = frag_idx;
        }
    }

    let mut bonds = Vec::new();
    for (a, b, _) in graph.all_edges() {
        let frag_a = atom_to_frag[a];
        let frag_b = atom_to_frag[b];
        if frag_a != frag_b {
            // Two entries: one for each side
            bonds.push(DetachedBond {
                bda_global: a,
                baa_global: b,
                bda_fragment: frag_a,
                baa_fragment: frag_b,
            });
            bonds.push(DetachedBond {
                bda_global: b,
                baa_global: a,
                bda_fragment: frag_b,
                baa_fragment: frag_a,
            });
        }
    }
    bonds
}

// ============================================================================
// Filtering
// ============================================================================

/// Get detached bonds for a specific fragment (where BDA is in that fragment).
pub fn get_detached_bonds_for_fragment(
    all_bonds: &[DetachedBond],
    frag_idx: usize,
) -> Vec<&DetachedBond> {
    all_bonds
        .iter()
        .filter(|b| b.bda_fragment == frag_idx)
        .collect()
}

/// Get detached bonds for a pair calculation.
///
/// Returns bonds where:
/// - BDA is in {frag_i, frag_j} (inside the pair)
/// - BAA is NOT in {frag_i, frag_j} (outside the pair)
///
/// Bonds fully inside the pair (both BDA and BAA in {i,j}) are excluded
/// since the pair calculation treats them as intact bonds.
pub fn get_detached_bonds_for_pair(
    all_bonds: &[DetachedBond],
    frag_i: usize,
    frag_j: usize,
) -> Vec<&DetachedBond> {
    all_bonds
        .iter()
        .filter(|b| {
            let bda_inside = b.bda_fragment == frag_i || b.bda_fragment == frag_j;
            let baa_inside = b.baa_fragment == frag_i || b.baa_fragment == frag_j;
            bda_inside && !baa_inside
        })
        .collect()
}

/// Get detached bonds fully inside a pair (both BDA and BAA in {frag_i, frag_j}).
///
/// These bonds are treated as intact in the pair SCC (no HOP), but the monomer
/// SCC has HOP for them. The FMO delta needs to subtract the monomer HOP contribution.
pub fn get_detached_bonds_fully_inside_pair(
    all_bonds: &[DetachedBond],
    frag_i: usize,
    frag_j: usize,
) -> Vec<&DetachedBond> {
    all_bonds
        .iter()
        .filter(|b| {
            let bda_inside = b.bda_fragment == frag_i || b.bda_fragment == frag_j;
            let baa_inside = b.baa_fragment == frag_i || b.baa_fragment == frag_j;
            bda_inside && baa_inside
        })
        .collect()
}

/// Get detached bonds for a trimer calculation.
///
/// Returns bonds where BDA is in {i,j,k} but BAA is NOT in {i,j,k}.
pub fn get_detached_bonds_for_trimer(
    all_bonds: &[DetachedBond],
    frag_i: usize,
    frag_j: usize,
    frag_k: usize,
) -> Vec<&DetachedBond> {
    all_bonds
        .iter()
        .filter(|b| {
            let bda_inside =
                b.bda_fragment == frag_i || b.bda_fragment == frag_j || b.bda_fragment == frag_k;
            let baa_inside =
                b.baa_fragment == frag_i || b.baa_fragment == frag_j || b.baa_fragment == frag_k;
            bda_inside && !baa_inside
        })
        .collect()
}

// ============================================================================
// sp3 Rotation
// ============================================================================

/// Compute the rotated sp3 hybrid orbital coefficients in the xTB valence basis.
///
/// Given a bond vector (BDA→BAA), returns [c_s, c_px, c_py, c_pz] where:
///   c_s = 0.5, c_p = (√3/2) * b̂
///
/// This is the xTB simplification of the VECROT→TRMAT→ROTCAO chain.
pub fn compute_rotated_sp3(bond_vec: &Vector3<f64>) -> Array1<f64> {
    let norm = bond_vec.norm();
    assert!(norm > 1e-14, "HOP: bond vector has zero length");
    let b_hat = bond_vec / norm;
    array![SP3_COEFF_S, SP3_COEFF_P * b_hat.x, SP3_COEFF_P * b_hat.y, SP3_COEFF_P * b_hat.z]
}

/// Compute the DD matrix: shift * outer(c, c).
///
/// Returns a 4×4 matrix for xTB (s + 3p valence).
pub fn compute_dd_matrix(rotated_sp3: ArrayView1<f64>, shift: f64) -> Array2<f64> {
    let n = rotated_sp3.len();
    let mut dd = Array2::<f64>::zeros([n, n]);
    for i in 0..n {
        for j in 0..n {
            dd[[i, j]] = shift * rotated_sp3[i] * rotated_sp3[j];
        }
    }
    dd
}

/// Compute the HOP projector: P_HOP = S_block^T × DD × S_block
///
/// S_block = S[:, bda_ao_start..bda_ao_start+nao_bda] (L1 × nao_bda)
/// DD is nao_bda × nao_bda
/// Result is L1 × L1 symmetric matrix to add to H.
pub fn compute_hop_projector(
    s: ArrayView2<f64>,
    dd: ArrayView2<f64>,
    bda_ao_start: usize,
    nao_bda: usize,
) -> Array2<f64> {
    let n_orbs = s.nrows();
    // S_block: all rows, BDA columns
    let s_block = s.slice(s![.., bda_ao_start..bda_ao_start + nao_bda]);
    // P_HOP = S_block . DD . S_block^T
    let sd = s_block.dot(&dd);
    sd.dot(&s_block.t())
}

// ============================================================================
// Basis utility
// ============================================================================

/// Get the AO range for a specific atom in a local basis.
///
/// Returns (ao_start, nao) where ao_start is the first AO index and
/// nao is the number of AOs for that atom (typically 4 for C/N/O in xTB: 1s + 3p).
pub fn get_bda_ao_range(basis: &Basis, local_atom_index: usize) -> (usize, usize) {
    let mut ao_start = usize::MAX;
    let mut ao_end = 0;
    for shell in &basis.shells {
        if shell.atom_index == local_atom_index {
            if shell.sph_start < ao_start {
                ao_start = shell.sph_start;
            }
            if shell.sph_end > ao_end {
                ao_end = shell.sph_end;
            }
        }
    }
    assert!(ao_start < ao_end, "HOP: no shells found for atom {}", local_atom_index);
    (ao_start, ao_end - ao_start)
}

/// Convert a global atom index to a local index within a fragment's sorted atom list.
///
/// The fragment atoms are stored in sorted order in the supersystem, so the local
/// index is the position of the global atom within the fragment's atom range.
pub fn global_to_local_atom(
    global_idx: usize,
    atom_range: std::ops::Range<usize>,
) -> Option<usize> {
    if global_idx >= atom_range.start && global_idx < atom_range.end {
        Some(global_idx - atom_range.start)
    } else {
        None
    }
}

/// Count the number of detached bonds in a fragment (for n_elec adjustment).
pub fn count_bonds_in_fragment(all_bonds: &[DetachedBond], frag_idx: usize) -> usize {
    all_bonds
        .iter()
        .filter(|b| b.bda_fragment == frag_idx)
        .count()
}

/// Compute the total HOP projector for a set of detached bonds in a fragment.
///
/// Given a fragment's basis, overlap matrix, and positions, computes the sum of
/// HOP projectors for all cut bonds where BDA is in this fragment.
pub fn compute_total_hop_projector(
    bonds: &[&DetachedBond],
    basis: &Basis,
    s: ArrayView2<f64>,
    positions: &dyn Fn(usize) -> Vector3<f64>,
) -> Option<Array2<f64>> {
    if bonds.is_empty() {
        return None;
    }
    let n_orbs = s.nrows();
    let mut p_hop_total = Array2::<f64>::zeros([n_orbs, n_orbs]);

    for bond in bonds {
        let bda_pos = positions(bond.bda_global);
        let baa_pos = positions(bond.baa_global);
        let bond_vec = baa_pos - bda_pos; // BDA→BAA direction

        let rotated_sp3 = compute_rotated_sp3(&bond_vec);
        let dd = compute_dd_matrix(rotated_sp3.view(), HOP_SHIFT);

        // Find BDA's local atom index in this basis by searching shells
        // The BDA must be in this fragment, so find its local index
        let bda_local = find_bda_local_atom(basis, bond, positions);
        let (ao_start, nao) = get_bda_ao_range(basis, bda_local);

        // Ensure DD size matches nao (it should be 4 for s+p atom)
        // But BDA might have more AOs (e.g., d orbitals). Embed DD in nao×nao.
        let p_hop = if nao == dd.nrows() {
            compute_hop_projector(s, dd.view(), ao_start, nao)
        } else {
            // Embed the 4×4 DD into the larger nao×nao space (s,p block only)
            let mut dd_full = Array2::<f64>::zeros([nao, nao]);
            let dd_size = dd.nrows().min(nao);
            dd_full
                .slice_mut(s![..dd_size, ..dd_size])
                .assign(&dd.slice(s![..dd_size, ..dd_size]));
            compute_hop_projector(s, dd_full.view(), ao_start, nao)
        };

        p_hop_total += &p_hop;
    }

    Some(p_hop_total)
}

/// Find the local atom index of a BDA within a basis set.
///
/// Uses the basis shells' atom_index field. The BDA's global position is used
/// to match against the basis function centers.
fn find_bda_local_atom(
    basis: &Basis,
    bond: &DetachedBond,
    positions: &dyn Fn(usize) -> Vector3<f64>,
) -> usize {
    let bda_pos = positions(bond.bda_global);
    // Find the atom_index in the basis whose center matches BDA position
    for shell in &basis.shells {
        let center = &basis.basis_functions[shell.start].center;
        let cx = Vector3::new(center.0, center.1, center.2);
        let dist = (cx - bda_pos).norm();
        if dist < 1e-6 {
            return shell.atom_index;
        }
    }
    panic!(
        "HOP: could not find BDA atom {} in basis (position: {:?})",
        bond.bda_global, bda_pos
    );
}
