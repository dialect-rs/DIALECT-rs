use crate::initialization::atom::XtbAtom;
use crate::{
    initialization::basis::Basis,
    parameters::{COUL_CHEMICAL_HARDNESS, COUL_SHELL_HARDNESS},
};
use hashbrown::HashMap;
use nalgebra::Vector3;
use ndarray::prelude::*;
use rayon::prelude::*;

/// Wrapper to send a raw pointer across threads.
/// SAFETY: Callers must ensure no two threads write to the same memory location.
#[derive(Copy, Clone)]
struct SendSyncPtr(*mut f64);
unsafe impl Send for SendSyncPtr {}
unsafe impl Sync for SendSyncPtr {}

impl SendSyncPtr {
    fn as_ptr(self) -> *mut f64 {
        self.0
    }
}

// NOTE: declared locally rather than in dialect-utilities::linalg because
// dialect-utilities depends on this crate (cube generation needs the xTB
// basis) -- importing it from there would create a crate cycle.
extern "C" {
    /// BLAS DSYMV: Symmetric matrix-vector multiply
    /// y := alpha * A * x + beta * y, where A is symmetric.
    /// Only reads the upper (or lower) triangle -- halves memory bandwidth.
    fn dsymv_(
        uplo: *const u8,
        n: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        x: *const f64,
        incx: *const i32,
        beta: *const f64,
        y: *mut f64,
        incy: *const i32,
    );
}

/// Initialize shell hardness values for Klopman-Ohno Coulomb (GFN1-xTB style).
/// Stores c = 1/shellHardness, where shellHardness = atomicHardness * (1 + angHardness)
/// The eta value is then: eta = 2/(c_i + c_j) = harmonicAverage(shellHardness_i, shellHardness_j)
pub fn init_avg_hubbard_u(
    atoms: &[XtbAtom],
    basis: &Basis,
    unique_length: usize,
) -> HashMap<(u8, u8), f64> {
    let mut sigmas: HashMap<(u8, u8), f64> = HashMap::with_capacity(unique_length);

    for func in basis.basis_functions.iter() {
        let atom: &XtbAtom = &atoms[func.atom_index];
        let l: usize = func.angular_momentum;
        let z: usize = atom.number as usize - 1;

        if !sigmas.contains_key(&(atom.number, l as u8)) {
            // c = 1/shellHardness = 1 / (atomicHardness * (1 + angHardness))
            sigmas.insert(
                (atom.number, l as u8),
                1.0 / ((1.0 + COUL_SHELL_HARDNESS[z][l]) * COUL_CHEMICAL_HARDNESS[z]),
            );
        }
    }
    sigmas
}

#[derive(Clone, Debug)]
pub struct XtbGammaFunction {
    /// c values: c = 1/shellHardness, indexed by (element, angular_momentum)
    pub c: HashMap<(u8, u8), f64>,
    /// Precomputed eta values: eta = 2/(c_i + c_j) = harmonicAverage(hardness_i, hardness_j)
    pub eta: HashMap<((u8, u8), (u8, u8)), f64>,
}

impl XtbGammaFunction {
    pub fn initialize(&mut self) {
        // Construct eta values from c values
        // eta = 2/(c_i + c_j) = harmonicAverage(1/c_i, 1/c_j) = harmonicAverage(hardness_i, hardness_j)
        for key_r in self.c.keys() {
            for key_s in self.c.keys() {
                let (c_r, c_s) = (self.c[key_r], self.c[key_s]);
                self.eta
                    .insert((*key_r, *key_s), 2.0 / (c_r + c_s));
            }
        }
    }

    /// Evaluate gamma using Klopman-Ohno formula (GFN1-xTB).
    /// γ(r) = 1/sqrt(r² + 1/η²) where η = harmonicAverage(hardness_i, hardness_j)
    /// Equivalent form: 1/(r^2 + 1/g_ij^2)^(1/2) with gExp=2
    pub fn eval(&self, r: f64, z_a: u8, l_a: u8, z_b: u8, l_b: u8) -> f64 {
        let eta = self.eta[&((z_a, l_a), (z_b, l_b))];
        1.0 / (r.powi(2) + 1.0 / eta.powi(2)).sqrt()
    }

    pub fn deriv(&self, r: f64, z_a: u8, l_a: u8, z_b: u8, l_b: u8) -> f64 {
        let eta = self.eta[&((z_a, l_a), (z_b, l_b))];
        -r / (r.powi(2) + 1.0 / eta.powi(2)).powf(1.5)
    }
}

pub fn gamma_matrix_xtb(
    gamma_func: &XtbGammaFunction,
    atoms: &[XtbAtom],
    basis: &Basis,
    n_orbs: usize,
) -> Array2<f64> {
    let mut g0_a0: Array2<f64> = Array2::zeros((n_orbs, n_orbs));

    for (i, funci) in basis.basis_functions.iter().enumerate() {
        let atomi: &XtbAtom = &atoms[funci.atom_index];
        let l_i: usize = funci.angular_momentum;

        for (j, funcj) in basis.basis_functions.iter().enumerate() {
            let atomj: &XtbAtom = &atoms[funcj.atom_index];
            let l_j: usize = funcj.angular_momentum;

            if i <= j {
                let g_val: f64 = gamma_func.eval(
                    (atomi.xyz - atomj.xyz).norm(),
                    atomi.number,
                    l_i as u8,
                    atomj.number,
                    l_j as u8,
                );
                g0_a0[[i, j]] = g_val;
                g0_a0[[j, i]] = g_val;
            }
        }
    }
    g0_a0
}

pub fn gamma_matrix_xtb_new(
    gamma_func: &XtbGammaFunction,
    atoms: &[XtbAtom],
    basis: &Basis,
) -> Array2<f64> {
    let mut g0_a0: Array2<f64> = Array2::zeros((basis.nbas, basis.nbas));

    // Iterate over shell pairs (only upper triangle)
    for (idx_shell_i, shell_i) in basis.shells.iter().enumerate() {
        let atomi: &XtbAtom = &atoms[shell_i.atom_index];
        let l_i: usize = shell_i.angular_momentum;

        for shell_j in basis.shells.iter().skip(idx_shell_i) {
            let atomj: &XtbAtom = &atoms[shell_j.atom_index];
            let l_j: usize = shell_j.angular_momentum;

            // Compute gamma value once per shell pair (distance is same for all orbitals in shell)
            let g_val: f64 = gamma_func.eval(
                (atomi.xyz - atomj.xyz).norm(),
                atomi.number,
                l_i as u8,
                atomj.number,
                l_j as u8,
            );

            // Fill all orbital pairs in this shell pair
            for idx_i in shell_i.sph_start..shell_i.sph_end {
                // For diagonal shell blocks, only fill upper triangle
                let j_start = if shell_i.sph_start == shell_j.sph_start {
                    idx_i
                } else {
                    shell_j.sph_start
                };

                for idx_j in j_start..shell_j.sph_end {
                    g0_a0[[idx_i, idx_j]] = g_val;
                    if idx_i != idx_j {
                        g0_a0[[idx_j, idx_i]] = g_val;
                    }
                }
            }
        }
    }
    g0_a0
}

pub fn gamma_matrix_xtb_par(
    gamma_func: &XtbGammaFunction,
    atoms: &[XtbAtom],
    basis: &Basis,
) -> Array2<f64> {
    let n_shells = basis.shells.len();
    let nbas = basis.nbas;

    // Precompute per-shell data in flat arrays for cache-friendly access
    let mut shell_xyz_vec: Vec<[f64; 3]> = Vec::with_capacity(n_shells);
    let mut shell_type_key: Vec<(u8, u8)> = Vec::with_capacity(n_shells);
    let mut sph_start_vec: Vec<usize> = Vec::with_capacity(n_shells);
    let mut sph_end_vec: Vec<usize> = Vec::with_capacity(n_shells);

    for shell in basis.shells.iter() {
        let atom = &atoms[shell.atom_index];
        shell_xyz_vec.push([atom.xyz.x, atom.xyz.y, atom.xyz.z]);
        shell_type_key.push((atom.number, shell.angular_momentum as u8));
        sph_start_vec.push(shell.sph_start);
        sph_end_vec.push(shell.sph_end);
    }

    // Build compact type index and precompute inv_eta_sq table
    let mut type_map: HashMap<(u8, u8), usize> = HashMap::new();
    let mut n_types = 0usize;
    for &key in shell_type_key.iter() {
        if !type_map.contains_key(&key) {
            type_map.insert(key, n_types);
            n_types += 1;
        }
    }

    let shell_type_vec: Vec<usize> = shell_type_key.iter().map(|k| type_map[k]).collect();

    // inv_eta_sq[type_a * n_types + type_b] = 1 / eta(type_a, type_b)^2
    let mut inv_eta_sq_vec: Vec<f64> = vec![0.0; n_types * n_types];
    for (&(z_a, l_a), &idx_a) in type_map.iter() {
        for (&(z_b, l_b), &idx_b) in type_map.iter() {
            let eta = gamma_func.eta[&((z_a, l_a), (z_b, l_b))];
            inv_eta_sq_vec[idx_a * n_types + idx_b] = 1.0 / (eta * eta);
        }
    }

    // Allocate output and get raw pointer for parallel writes
    let mut g0_a0: Array2<f64> = Array2::zeros((nbas, nbas));
    let ptr = SendSyncPtr(g0_a0.as_mut_ptr());

    // Convert to slices for closure capture (slices are Copy, so closures work with Fn)
    let shell_xyz: &[[f64; 3]] = &shell_xyz_vec;
    let shell_type: &[usize] = &shell_type_vec;
    let sph_start: &[usize] = &sph_start_vec;
    let sph_end: &[usize] = &sph_end_vec;
    let inv_eta_sq: &[f64] = &inv_eta_sq_vec;

    // STEP 1: Fill upper triangle only (parallel over shells).
    // Writes are sequential along rows → cache-friendly. No transpose writes,
    // which would thrash the cache by jumping to random rows for each shell pair.
    // SAFETY: shell orbital ranges are disjoint, so threads writing different
    // idx_i values never touch the same matrix cells.
    (0..n_shells).into_par_iter().for_each(move |idx_i| {
        let [xi, yi, zi] = shell_xyz[idx_i];
        let type_i = shell_type[idx_i];
        let si_start = sph_start[idx_i];
        let si_end = sph_end[idx_i];

        for idx_j in idx_i..n_shells {
            let dx = xi - shell_xyz[idx_j][0];
            let dy = yi - shell_xyz[idx_j][1];
            let dz = zi - shell_xyz[idx_j][2];
            let r_sq = dx * dx + dy * dy + dz * dz;

            let inv_eta_sq_val = inv_eta_sq[type_i * n_types + shell_type[idx_j]];
            let g_val = 1.0 / (r_sq + inv_eta_sq_val).sqrt();

            let sj_start = sph_start[idx_j];
            let sj_end = sph_end[idx_j];

            let raw = ptr.as_ptr();
            for orb_i in si_start..si_end {
                let j_begin = if si_start == sj_start {
                    orb_i
                } else {
                    sj_start
                };
                for orb_j in j_begin..sj_end {
                    unsafe {
                        *raw.add(orb_i * nbas + orb_j) = g_val;
                    }
                }
            }
        }
    });

    // STEP 2: Fill lower triangle from upper triangle using a tiled transpose copy.
    // Tiles ensure both the source and destination blocks fit in L1/L2 cache.
    const TILE: usize = 64; // 64×64×8 = 32 KB per tile, two tiles = 64 KB ≈ L1
    let n_tiles = (nbas + TILE - 1) / TILE;
    let ptr2 = SendSyncPtr(g0_a0.as_mut_ptr());

    // SAFETY: We only write to the strict lower triangle [i,j] where i > j.
    // Each tile-row is handled by one thread and writes to disjoint row ranges.
    (0..n_tiles).into_par_iter().for_each(move |tr| {
        let raw = ptr2.as_ptr();
        let r_start = tr * TILE;
        let r_end = (r_start + TILE).min(nbas);

        // Off-diagonal tiles: copy full rectangular block
        for tc in 0..tr {
            let c_start = tc * TILE;
            let c_end = (c_start + TILE).min(nbas);
            for i in r_start..r_end {
                for j in c_start..c_end {
                    unsafe {
                        *raw.add(i * nbas + j) = *raw.add(j * nbas + i);
                    }
                }
            }
        }

        // Diagonal tile: only the lower triangle within the tile
        for i in r_start..r_end {
            for j in r_start..i {
                unsafe {
                    *raw.add(i * nbas + j) = *raw.add(j * nbas + i);
                }
            }
        }
    });

    g0_a0
}

pub fn gamma_gradient_xtb_new(
    gamma_func: &XtbGammaFunction,
    atoms: &[XtbAtom],
    basis: &Basis,
    n_atoms: usize,
    n_orbs: usize,
) -> Array3<f64> {
    let mut g1: Array3<f64> = Array3::zeros((3 * n_atoms, n_orbs, n_orbs));
    let mut g1_val: Array2<f64> = Array2::zeros((n_orbs, n_orbs));

    for shell_i in basis.shells.iter() {
        let atomi: &XtbAtom = &atoms[shell_i.atom_index];
        let l_i: usize = shell_i.angular_momentum;
        let at_i: usize = shell_i.atom_index;
        // iterate over angular components
        for idx_i in (shell_i.sph_start..shell_i.sph_end) {
            // iterate over the next shells
            for shell_j in basis.shells.iter() {
                let atomj: &XtbAtom = &atoms[shell_j.atom_index];
                let l_j: usize = shell_j.angular_momentum;
                let at_j: usize = shell_j.atom_index;
                // iteratve over angular components
                for idx_j in (shell_j.sph_start..shell_j.sph_end) {
                    // compare the atomic indices
                    if at_i < at_j {
                        let r = atomi.xyz - atomj.xyz;
                        let r_ij: f64 = r.norm();
                        let e_ij: Vector3<f64> = r / r_ij;

                        g1_val[[idx_i, idx_j]] = gamma_func.deriv(
                            r_ij,
                            atomi.number,
                            l_i as u8,
                            atomj.number,
                            l_j as u8,
                        );
                        g1.slice_mut(s![(3 * at_i)..(3 * at_i + 3), idx_i, idx_j])
                            .assign(&Array1::from_iter(
                                (e_ij * g1_val[[idx_i, idx_j]]).iter().cloned(),
                            ));
                        g1.slice_mut(s![(3 * at_i)..(3 * at_i + 3), idx_j, idx_i])
                            .assign(&Array1::from_iter(
                                (e_ij * g1_val[[idx_i, idx_j]]).iter().cloned(),
                            ));
                    } else if at_i > at_j {
                        let r = atomi.xyz - atomj.xyz;
                        let e_ij: Vector3<f64> = r / r.norm();
                        g1.slice_mut(s![(3 * at_i)..(3 * at_i + 3), idx_i, idx_j])
                            .assign(&Array::from_iter(
                                (e_ij * g1_val[[idx_j, idx_i]]).iter().cloned(),
                            ));
                        g1.slice_mut(s![(3 * at_i)..(3 * at_i + 3), idx_j, idx_i])
                            .assign(&Array::from_iter(
                                (e_ij * g1_val[[idx_j, idx_i]]).iter().cloned(),
                            ));
                    }
                }
            }
        }
    }

    g1
}

pub fn gamma_gradient_xtb_atom_specific(
    gamma_func: &XtbGammaFunction,
    atoms: &[XtbAtom],
    basis: &Basis,
    n_orbs: usize,
    atom_idx: usize,
) -> Array3<f64> {
    let mut g1: Array3<f64> = Array3::zeros((3, n_orbs, n_orbs));
    let mut g1_val: Array2<f64> = Array2::zeros((n_orbs, n_orbs));

    for shell_i in basis.shells.iter() {
        if shell_i.atom_index == atom_idx {
            let atomi: &XtbAtom = &atoms[shell_i.atom_index];
            let l_i: usize = shell_i.angular_momentum;
            let at_i: usize = shell_i.atom_index;
            // iterate over angular components
            for idx_i in (shell_i.sph_start..shell_i.sph_end) {
                // iterate over the next shells
                for shell_j in basis.shells.iter() {
                    let atomj: &XtbAtom = &atoms[shell_j.atom_index];
                    let l_j: usize = shell_j.angular_momentum;
                    let at_j: usize = shell_j.atom_index;
                    // iteratve over angular components
                    for idx_j in (shell_j.sph_start..shell_j.sph_end) {
                        // compare the atomic indices
                        if at_i != at_j {
                            let r = atomi.xyz - atomj.xyz;
                            let r_ij: f64 = r.norm();
                            let e_ij: Vector3<f64> = r / r_ij;

                            g1_val[[idx_i, idx_j]] = gamma_func.deriv(
                                r_ij,
                                atomi.number,
                                l_i as u8,
                                atomj.number,
                                l_j as u8,
                            );
                            g1.slice_mut(s![.., idx_i, idx_j])
                                .assign(&Array1::from_iter(
                                    (e_ij * g1_val[[idx_i, idx_j]]).iter().cloned(),
                                ));
                            g1.slice_mut(s![.., idx_j, idx_i])
                                .assign(&Array1::from_iter(
                                    (e_ij * g1_val[[idx_i, idx_j]]).iter().cloned(),
                                ));
                        }
                    }
                }
            }
        }
    }

    g1
}

pub fn gamma_gradient_xtb_contracted(
    gamma_func: &XtbGammaFunction,
    atoms: &[XtbAtom],
    basis: &Basis,
    n_atoms: usize,
    n_orbs: usize,
    vec: ArrayView1<f64>, // The 1D array to contract with [n_orbs]
    contract_last_axis: bool,
) -> Array2<f64> {
    // Result is [3 * n_atoms, n_orbs] instead of [3 * n_atoms, n_orbs, n_orbs]
    let mut g1_contracted: Array2<f64> = Array2::zeros((3 * n_atoms, n_orbs));
    let mut g1_val: Array2<f64> = Array2::zeros((n_orbs, n_orbs));

    for shell_i in basis.shells.iter() {
        let atomi: &XtbAtom = &atoms[shell_i.atom_index];
        let l_i: usize = shell_i.angular_momentum;
        let at_i: usize = shell_i.atom_index;
        // iterate over angular components
        for idx_i in shell_i.sph_start..shell_i.sph_end {
            // iterate over the next shells
            for shell_j in basis.shells.iter() {
                let atomj: &XtbAtom = &atoms[shell_j.atom_index];
                let l_j: usize = shell_j.angular_momentum;
                let at_j: usize = shell_j.atom_index;
                // iterate over angular components
                for idx_j in shell_j.sph_start..shell_j.sph_end {
                    // compare the atomic indices
                    if at_i < at_j {
                        let r = atomi.xyz - atomj.xyz;
                        let r_ij: f64 = r.norm();
                        let e_ij: Vector3<f64> = r / r_ij;

                        g1_val[[idx_i, idx_j]] = gamma_func.deriv(
                            r_ij,
                            atomi.number,
                            l_i as u8,
                            atomj.number,
                            l_j as u8,
                        );

                        let grad_vec: Vector3<f64> = e_ij * g1_val[[idx_i, idx_j]];

                        // Contract: g1[3*at_i:3*at_i+3, idx_i, idx_j] * vec[idx_j]
                        // contributes to g1_contracted[3*at_i:3*at_i+3, idx_i]
                        if contract_last_axis {
                            for k in 0..3 {
                                g1_contracted[[3 * at_i + k, idx_i]] += grad_vec[k] * vec[idx_j];
                                g1_contracted[[3 * at_i + k, idx_j]] += grad_vec[k] * vec[idx_i];
                            }
                        } else {
                            for k in 0..3 {
                                g1_contracted[[3 * at_i + k, idx_j]] += grad_vec[k] * vec[idx_i];
                                g1_contracted[[3 * at_i + k, idx_i]] += grad_vec[k] * vec[idx_j];
                            }
                        }
                    } else if at_i > at_j {
                        let r = atomi.xyz - atomj.xyz;
                        let e_ij: Vector3<f64> = r / r.norm();

                        let grad_vec: Vector3<f64> = e_ij * g1_val[[idx_j, idx_i]];

                        // Contract: g1[3*at_i:3*at_i+3, idx_i, idx_j] * vec[idx_j]
                        // contributes to g1_contracted[3*at_i:3*at_i+3, idx_i]
                        if contract_last_axis {
                            for k in 0..3 {
                                g1_contracted[[3 * at_i + k, idx_i]] += grad_vec[k] * vec[idx_j];
                                g1_contracted[[3 * at_i + k, idx_j]] += grad_vec[k] * vec[idx_i];
                            }
                        } else {
                            for k in 0..3 {
                                g1_contracted[[3 * at_i + k, idx_j]] += grad_vec[k] * vec[idx_i];
                                g1_contracted[[3 * at_i + k, idx_i]] += grad_vec[k] * vec[idx_j];
                            }
                        }
                    }
                }
            }
        }
    }

    g1_contracted
}

/// Compute double-contracted gamma gradient: sum_mu sum_nu dgamma[g, mu, nu] * vec_left[mu] * vec_right[nu]
/// This avoids storing the full 3D array [3*n_atoms, n_orbs, n_orbs] by computing the contraction on-the-fly.
///
/// Parameters:
/// - gamma_func: The gamma function for computing derivatives
/// - atoms: All atoms in the system
/// - basis: The basis set
/// - n_atoms: Number of atoms
/// - vec_left: Left contraction vector [n_orbs]
/// - vec_right: Right contraction vector [n_orbs]
/// - orb_slice_left: (start, end) orbital indices for vec_left (which orbitals to contract)
/// - orb_slice_right: (start, end) orbital indices for vec_right (which orbitals to contract)
/// - atom_slice: (start, end) atom indices for output (which atoms' gradients to compute)
///
/// Returns: Array1<f64> of shape [3 * (atom_slice.1 - atom_slice.0)]
pub fn gamma_gradient_xtb_double_contracted(
    gamma_func: &XtbGammaFunction,
    atoms: &[XtbAtom],
    basis: &Basis,
    vec_left: ArrayView1<f64>,
    vec_right: ArrayView1<f64>,
    orb_slice_left: (usize, usize),
    orb_slice_right: (usize, usize),
    atom_slice: (usize, usize),
) -> Array1<f64> {
    let n_atoms_out = atom_slice.1 - atom_slice.0;
    let mut result: Array1<f64> = Array1::zeros(3 * n_atoms_out);

    for shell_i in basis.shells.iter() {
        let atomi: &XtbAtom = &atoms[shell_i.atom_index];
        let l_i: usize = shell_i.angular_momentum;
        let at_i: usize = shell_i.atom_index;

        // Check if this atom is in our output slice
        let at_i_in_slice = at_i >= atom_slice.0 && at_i < atom_slice.1;

        for idx_i in shell_i.sph_start..shell_i.sph_end {
            for shell_j in basis.shells.iter() {
                let atomj: &XtbAtom = &atoms[shell_j.atom_index];
                let l_j: usize = shell_j.angular_momentum;
                let at_j: usize = shell_j.atom_index;

                for idx_j in shell_j.sph_start..shell_j.sph_end {
                    // We only compute gradient on atoms in atom_slice
                    if !at_i_in_slice {
                        continue;
                    }

                    // Need at_i != at_j for non-zero gradient
                    if at_i == at_j {
                        continue;
                    }

                    let r = atomi.xyz - atomj.xyz;
                    let r_ij: f64 = r.norm();
                    let e_ij: Vector3<f64> = r / r_ij;

                    let g1_val =
                        gamma_func.deriv(r_ij, atomi.number, l_i as u8, atomj.number, l_j as u8);

                    // Contribution from dgamma[at_i, idx_i, idx_j]
                    // Check if idx_i is in left slice and idx_j is in right slice
                    if idx_i >= orb_slice_left.0
                        && idx_i < orb_slice_left.1
                        && idx_j >= orb_slice_right.0
                        && idx_j < orb_slice_right.1
                    {
                        let v_left = vec_left[idx_i - orb_slice_left.0];
                        let v_right = vec_right[idx_j - orb_slice_right.0];
                        let at_i_local = at_i - atom_slice.0;
                        for k in 0..3 {
                            result[3 * at_i_local + k] += e_ij[k] * g1_val * v_left * v_right;
                        }
                    }

                    // Contribution from dgamma[at_i, idx_j, idx_i] (symmetry)
                    // Check if idx_j is in left slice and idx_i is in right slice
                    if idx_j >= orb_slice_left.0
                        && idx_j < orb_slice_left.1
                        && idx_i >= orb_slice_right.0
                        && idx_i < orb_slice_right.1
                    {
                        let v_left = vec_left[idx_j - orb_slice_left.0];
                        let v_right = vec_right[idx_i - orb_slice_right.0];
                        let at_i_local = at_i - atom_slice.0;
                        for k in 0..3 {
                            result[3 * at_i_local + k] += e_ij[k] * g1_val * v_left * v_right;
                        }
                    }
                }
            }
        }
    }

    result
}

/// Compute double-contracted gamma gradient with summation over different orbital slices.
/// This is used for embedding gradient's self-interaction terms where we need:
/// sum over all orbitals: dgamma[g, all_orbs, orb_slice] * dq[all_orbs] * delta_dq[orb_slice]
///
/// Computes: dgamma[atom_slice, :, orb_slice_contract] . dq_full . delta_dq
///
/// Parameters:
/// - gamma_func: The gamma function for computing derivatives
/// - atoms: All atoms in the system
/// - basis: The basis set
/// - dq_i: Charge differences on monomer I orbitals [n_orbs_i]
/// - dq_j: Charge differences on monomer J orbitals [n_orbs_j]
/// - delta_dq: Full delta dq vector [n_orbs_pair]
/// - n_orbs_i: Number of orbitals on monomer I
/// - atom_slice: (start, end) atom indices for output
///
/// Returns: Array1<f64> of shape [3 * (atom_slice.1 - atom_slice.0)]
pub fn gamma_gradient_xtb_double_contracted_sum(
    gamma_func: &XtbGammaFunction,
    atoms: &[XtbAtom],
    basis: &Basis,
    dq_i: ArrayView1<f64>,
    dq_j: ArrayView1<f64>,
    delta_dq: ArrayView1<f64>,
    n_orbs_i: usize,
    atom_slice: (usize, usize),
) -> Array1<f64> {
    let n_atoms_out = atom_slice.1 - atom_slice.0;
    let n_orbs_pair = delta_dq.len();
    let n_orbs_j = n_orbs_pair - n_orbs_i;
    let mut result: Array1<f64> = Array1::zeros(3 * n_atoms_out);

    for shell_i in basis.shells.iter() {
        let atomi: &XtbAtom = &atoms[shell_i.atom_index];
        let l_i: usize = shell_i.angular_momentum;
        let at_i: usize = shell_i.atom_index;

        // Check if this atom is in our output slice
        let at_i_in_slice = at_i >= atom_slice.0 && at_i < atom_slice.1;

        for idx_i in shell_i.sph_start..shell_i.sph_end {
            for shell_j in basis.shells.iter() {
                let atomj: &XtbAtom = &atoms[shell_j.atom_index];
                let l_j: usize = shell_j.angular_momentum;
                let at_j: usize = shell_j.atom_index;

                for idx_j in shell_j.sph_start..shell_j.sph_end {
                    if !at_i_in_slice || at_i == at_j {
                        continue;
                    }

                    let r = atomi.xyz - atomj.xyz;
                    let r_ij: f64 = r.norm();
                    let e_ij: Vector3<f64> = r / r_ij;

                    let g1_val =
                        gamma_func.deriv(r_ij, atomi.number, l_i as u8, atomj.number, l_j as u8);

                    let at_i_local = at_i - atom_slice.0;

                    // Contribution: dgamma[at_i, idx_i, idx_j] * dq[idx_j] * delta_dq[idx_i]
                    // + dgamma[at_i, idx_j, idx_i] * dq[idx_i] * delta_dq[idx_j]
                    // where dq is either dq_i (if idx in [0, n_orbs_i)) or dq_j (if idx in [n_orbs_i, n_orbs_pair))

                    // Get dq values based on which monomer the orbital belongs to
                    let dq_idx_i = if idx_i < n_orbs_i {
                        dq_i[idx_i]
                    } else if idx_i < n_orbs_pair {
                        dq_j[idx_i - n_orbs_i]
                    } else {
                        continue;
                    };

                    let dq_idx_j = if idx_j < n_orbs_i {
                        dq_i[idx_j]
                    } else if idx_j < n_orbs_pair {
                        dq_j[idx_j - n_orbs_i]
                    } else {
                        continue;
                    };

                    // Both idx_i and idx_j must be within pair orbitals
                    if idx_i >= n_orbs_pair || idx_j >= n_orbs_pair {
                        continue;
                    }

                    // dgamma[at_i, idx_i, idx_j] * dq[idx_j] * delta_dq[idx_i]
                    let contrib1 = g1_val * dq_idx_j * delta_dq[idx_i];
                    // dgamma[at_i, idx_j, idx_i] * dq[idx_i] * delta_dq[idx_j]
                    let contrib2 = g1_val * dq_idx_i * delta_dq[idx_j];

                    for k in 0..3 {
                        result[3 * at_i_local + k] += e_ij[k] * (contrib1 + contrib2);
                    }
                }
            }
        }
    }

    result
}

/// Compute double-contracted gamma gradient with summation over different orbital slices for trimers.
/// This is used for trimer embedding gradient's self-interaction terms where we need:
/// sum over all orbitals: dgamma[g, all_orbs, orb_slice] * dq[all_orbs] * delta_dq[orb_slice]
///
/// Parameters:
/// - gamma_func: The gamma function for computing derivatives
/// - atoms: All atoms in the system (trimer atoms)
/// - basis: The basis set
/// - dq_i, dq_j, dq_k: Charge differences on monomers I, J, K
/// - delta_dq: Full delta dq vector [n_orbs_trimer]
/// - n_orbs_i, n_orbs_j: Number of orbitals on monomers I and J (K is inferred)
/// - atom_slice: (start, end) atom indices for output
///
/// Returns: Array1<f64> of shape [3 * (atom_slice.1 - atom_slice.0)]
pub fn gamma_gradient_xtb_double_contracted_sum_trimer(
    gamma_func: &XtbGammaFunction,
    atoms: &[XtbAtom],
    basis: &Basis,
    dq_i: ArrayView1<f64>,
    dq_j: ArrayView1<f64>,
    dq_k: ArrayView1<f64>,
    delta_dq: ArrayView1<f64>,
    n_orbs_i: usize,
    n_orbs_j: usize,
    atom_slice: (usize, usize),
) -> Array1<f64> {
    let n_atoms_out = atom_slice.1 - atom_slice.0;
    let n_orbs_trimer = delta_dq.len();
    let mut result: Array1<f64> = Array1::zeros(3 * n_atoms_out);

    for shell_i in basis.shells.iter() {
        let atomi: &XtbAtom = &atoms[shell_i.atom_index];
        let l_i: usize = shell_i.angular_momentum;
        let at_i: usize = shell_i.atom_index;

        // Check if this atom is in our output slice
        let at_i_in_slice = at_i >= atom_slice.0 && at_i < atom_slice.1;

        for idx_i in shell_i.sph_start..shell_i.sph_end {
            for shell_j in basis.shells.iter() {
                let atomj: &XtbAtom = &atoms[shell_j.atom_index];
                let l_j: usize = shell_j.angular_momentum;
                let at_j: usize = shell_j.atom_index;

                for idx_j in shell_j.sph_start..shell_j.sph_end {
                    if !at_i_in_slice || at_i == at_j {
                        continue;
                    }

                    // Both indices must be within trimer orbitals
                    if idx_i >= n_orbs_trimer || idx_j >= n_orbs_trimer {
                        continue;
                    }

                    let r = atomi.xyz - atomj.xyz;
                    let r_ij: f64 = r.norm();
                    let e_ij: Vector3<f64> = r / r_ij;

                    let g1_val =
                        gamma_func.deriv(r_ij, atomi.number, l_i as u8, atomj.number, l_j as u8);

                    let at_i_local = at_i - atom_slice.0;

                    // Get dq values based on which monomer the orbital belongs to
                    let dq_idx_i = if idx_i < n_orbs_i {
                        dq_i[idx_i]
                    } else if idx_i < n_orbs_i + n_orbs_j {
                        dq_j[idx_i - n_orbs_i]
                    } else {
                        dq_k[idx_i - n_orbs_i - n_orbs_j]
                    };

                    let dq_idx_j = if idx_j < n_orbs_i {
                        dq_i[idx_j]
                    } else if idx_j < n_orbs_i + n_orbs_j {
                        dq_j[idx_j - n_orbs_i]
                    } else {
                        dq_k[idx_j - n_orbs_i - n_orbs_j]
                    };

                    // dgamma[at_i, idx_i, idx_j] * dq[idx_j] * delta_dq[idx_i]
                    let contrib1 = g1_val * dq_idx_j * delta_dq[idx_i];
                    // dgamma[at_i, idx_j, idx_i] * dq[idx_i] * delta_dq[idx_j]
                    let contrib2 = g1_val * dq_idx_i * delta_dq[idx_j];

                    for k in 0..3 {
                        result[3 * at_i_local + k] += e_ij[k] * (contrib1 + contrib2);
                    }
                }
            }
        }
    }

    result
}

/// Compute shell-shell gamma matrix (jmat) for shell-level SCC
/// (used for efficient shell-level mixing)
pub fn gamma_matrix_shell(
    gamma_func: &XtbGammaFunction,
    atoms: &[XtbAtom],
    basis: &Basis,
) -> Array2<f64> {
    let n_shells = basis.shells.len();
    let mut jmat: Array2<f64> = Array2::zeros((n_shells, n_shells));

    for (idx_i, shell_i) in basis.shells.iter().enumerate() {
        let atomi: &XtbAtom = &atoms[shell_i.atom_index];
        let l_i: usize = shell_i.angular_momentum;

        for (idx_j, shell_j) in basis.shells.iter().enumerate().skip(idx_i) {
            let atomj: &XtbAtom = &atoms[shell_j.atom_index];
            let l_j: usize = shell_j.angular_momentum;

            // Compute gamma value for this shell pair
            let g_val: f64 = gamma_func.eval(
                (atomi.xyz - atomj.xyz).norm(),
                atomi.number,
                l_i as u8,
                atomj.number,
                l_j as u8,
            );

            jmat[[idx_i, idx_j]] = g_val;
            if idx_i != idx_j {
                jmat[[idx_j, idx_i]] = g_val;
            }
        }
    }
    jmat
}

/// BLAS dsymv on the full gamma_ao matrix.
/// gamma_ao is symmetric, so dsymv reads only the upper triangle,
/// halving memory bandwidth compared to dgemv (ndarray's .dot()).
///
/// gamma_ao must be in standard row-major (C) order. Since dsymv expects
/// column-major (Fortran) order, we pass 'L' (lower triangle) which
/// corresponds to the upper triangle of the row-major layout.
pub fn gamma_ao_dsymv(gamma_ao: &ArrayView2<f64>, dq_ao: &Array1<f64>) -> Array1<f64> {
    let n = gamma_ao.nrows() as i32;
    let mut y = vec![0.0f64; n as usize];
    let alpha: f64 = 1.0;
    let beta: f64 = 0.0;
    let inc: i32 = 1;

    unsafe {
        dsymv_(
            b"L".as_ptr(), // 'L' in Fortran col-major = upper triangle in row-major
            &n,
            &alpha,
            gamma_ao.as_ptr(),
            &n,
            dq_ao.as_ptr(),
            &inc,
            &beta,
            y.as_mut_ptr(),
            &inc,
        );
    }

    Array1::from_vec(y)
}

/// BLAS dsymv on a shell-level gamma matrix.
/// Same approach as gamma_ao_dsymv but for the smaller shell-level matrix.
/// gamma_shell must be in standard row-major (C) order.
pub fn gamma_shell_dsymv(gamma_shell: &ArrayView2<f64>, dq_shell: &ArrayView1<f64>) -> Array1<f64> {
    let n = gamma_shell.nrows() as i32;
    let mut y = vec![0.0f64; n as usize];
    let alpha: f64 = 1.0;
    let beta: f64 = 0.0;
    let inc: i32 = 1;

    unsafe {
        dsymv_(
            b"L".as_ptr(), // 'L' in Fortran col-major = upper triangle in row-major
            &n,
            &alpha,
            gamma_shell.as_ptr(),
            &n,
            dq_shell.as_ptr(),
            &inc,
            &beta,
            y.as_mut_ptr(),
            &inc,
        );
    }

    Array1::from_vec(y)
}
