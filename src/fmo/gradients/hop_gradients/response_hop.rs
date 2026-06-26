//! FMO-DFTB HOP response gradient implementation.
//!
//! Adapts the non-HOP response gradient for covalent fragmentation:
//! - Extended atom lists (real + ghost boundary atoms)
//! - Q_vo accumulated over ALL extended atoms (matching ZVLAGGET NAT loop)
//! - SHIFTCT from pre-computed embedding + ESPGRAD shifts
//! - Orbital Hessian uses full monomer gamma (n_ext × n_ext)
//! - Z×HOP term: HOP projector derivative evaluated with Z instead of P
//!
//! Reference: `dftbfo.src` HOPDER(1,0,...) with MODE=1 (CPHF mode)

use super::helpers::{
    build_orbital_offsets, build_shift_ao_matrix,
};
use super::hop_projector::{
    hop_overlap_derivative_gradient_dftb,
    hop_coefficient_derivative_gradient_dftb,
};
use crate::defaults::PROXIMITY_CUTOFF;
use crate::fmo::scc_hop::hop_data::{
    compute_bda_dd_matrix, compute_ghost_nonbond_dd, compute_rotated_sp3_dftb,
    get_bda_ao_range_dftb, HopData,
};
use crate::fmo::scc_hop::monomer::MonomerHopScc;
use crate::fmo::Pair;
use crate::initialization::parameters::SlaterKoster;
use crate::initialization::Atom;
use crate::param::slako_transformations::{
    directional_cosines, slako_transformation_gradients_fast, SplineCache,
};
use crate::scc::gamma_approximation::GammaFunction;
use dialect_xtb::hop::{DetachedBond, HOP_SHIFT};
use ndarray::prelude::*;
use rayon::prelude::*;

/// Main entry point for FMO-DFTB HOP response gradient.
///
/// Returns gradient as flat array [3 * n_atoms_total] in global coordinates.
pub fn response_gradient_hop_total(
    hop_data: &HopData,
    mono_states: &[MonomerHopScc],
    _pairs: &[Pair],
    atoms: &[Atom],
    shiftcts: &[Array1<f64>],
    gammafunction: &GammaFunction,
    gammafunction_lc: &Option<GammaFunction>,
    slako: &SlaterKoster,
) -> Array1<f64> {

    // Step 1: Calculate Lagrangian and Q_vo per monomer
    let (lagrangians, qvo_vec) =
        calculate_response_lagrangian_hop(hop_data, mono_states, shiftcts);

    // Step 2: Solve SCZV Z-vector equations
    let z_vectors = solve_sczv_hop(hop_data, mono_states, &lagrangians, &qvo_vec, gammafunction, gammafunction_lc);

    // Step 3: Compute response gradient contributions
    let mut gradient = calculate_response_gradient_contributions(
        hop_data,
        mono_states,
        atoms,
        &z_vectors,
        &qvo_vec,
        gammafunction,
        gammafunction_lc,
        slako,
    );

    // Step 3a: Add response addlag (DFTB_ZVEC_KGRAD)
    // This is: dS/dR * (shift_z * P - 0.5 * P * (shift_z * S) * P)
    // where shift_z = gamma * Q_Z (Z-induced shift from all fragments)
    add_response_addlag_hop(
        hop_data,
        mono_states,
        atoms,
        &z_vectors,
        gammafunction,
        slako,
        &mut gradient,
    );

    // Step 4: Sign convention (MO sign difference)
    -1.0 * gradient
}

// ============================================================================
// Step 1: Lagrangian
// ============================================================================

/// Calculate response Lagrangian L^K for each monomer K.
///
/// L^K_ai = -0.5 * sum_{A in ext} SHIFTCT[A] * Q_vo^K[A, ai]
///
/// Q_vo is accumulated over ALL extended atoms (real + ghost), matching
/// ZVLAGGET which loops over NAT (total extended atoms including ghosts).
/// Ghost atoms have orbitals that participate in the MO coefficients.
///
/// Returns (lagrangians, qvo_vec) where qvo_vec[k] has shape [n_ext_atoms, nvirt*nocc].
fn calculate_response_lagrangian_hop(
    _hop_data: &HopData,
    mono_states: &[MonomerHopScc],
    shiftcts: &[Array1<f64>],
) -> (Vec<Array1<f64>>, Vec<Array2<f64>>) {
    let results: Vec<(Array1<f64>, Array2<f64>)> = mono_states
        .par_iter()
        .enumerate()
        .map(|(frag_idx, mono)| {
            let n_ext_atoms = mono.n_ext_atoms;
            let n_ext_orbs = mono.n_ext_orbs;
            let nocc = mono.n_elec / 2;
            let nvirt = n_ext_orbs - nocc;
            let nnum = nvirt * nocc;

            let orbs = mono.orbs.as_ref().expect("orbs not stored").view();
            let s = mono.s.view();

            // SC = S * C  [n_ext_orbs x n_ext_orbs]
            let sc = s.dot(&orbs);

            // Build Q_vo over ALL extended atoms (real + ghost)
            let ext_atoms = &mono.ext_atoms;
            let orbital_offsets = build_orbital_offsets(ext_atoms);

            let mut qvo = Array2::<f64>::zeros([n_ext_atoms, nnum]);
            for a_idx in 0..n_ext_atoms {
                let mu_start = orbital_offsets[a_idx];
                let mu_end = orbital_offsets[a_idx + 1];

                let c_mu_virt = orbs.slice(s![mu_start..mu_end, nocc..]);
                let sc_mu_occ = sc.slice(s![mu_start..mu_end, ..nocc]);
                let term1 = c_mu_virt.t().dot(&sc_mu_occ);

                let c_mu_occ = orbs.slice(s![mu_start..mu_end, ..nocc]);
                let sc_mu_virt = sc.slice(s![mu_start..mu_end, nocc..]);
                let term2 = c_mu_occ.t().dot(&sc_mu_virt);

                let sum = &term1 + &term2.t();
                let sum_1d = sum.into_shape([nnum]).unwrap();
                qvo.slice_mut(s![a_idx, ..]).assign(&sum_1d);
            }

            let shiftct = &shiftcts[frag_idx];
            let lag = -0.5 * shiftct.dot(&qvo);

            (lag, qvo)
        })
        .collect();

    let (lagrangians, qvo_vec): (Vec<_>, Vec<_>) = results.into_iter().unzip();
    (lagrangians, qvo_vec)
}

// ============================================================================
// Step 2: SCZV solver
// ============================================================================

/// Solve Z-vector equations using SCZV (Self-Consistent Z-Vector) iteration.
///
/// For each monomer I, solves: A^I * Z^I = 4*L^I - sum_{K!=I} (A^{K,I})^T * Z^K
///
/// The orbital Hessian A^I uses real-atom gamma (not extended gamma with ghosts).
/// Inter-fragment coupling uses gamma_ext real-atom subblocks.
///
/// Uses matrix-free preconditioned conjugate gradient instead of building the
/// full A matrix, reducing memory from O(nnum^2) to O(n_atoms * nnum).
fn solve_sczv_hop(
    hop_data: &HopData,
    mono_states: &[MonomerHopScc],
    lagrangians: &[Array1<f64>],
    qvo_vec: &[Array2<f64>],
    _gammafunction: &GammaFunction,
    _gammafunction_lc: &Option<GammaFunction>,
) -> Vec<Array1<f64>> {
    let n_mol = mono_states.len();
    let maxiter = 500;
    let threshold = 1.0e-8;

    // Scale Lagrangian by 4 (SCAL=4.0 for RHF closed-shell)
    let rhs: Vec<Array1<f64>> = lagrangians.iter().map(|l| 4.0 * l).collect();

    // Segment sizes for each fragment
    let seg_sizes: Vec<usize> = mono_states.iter().map(|mono| {
        let nocc = mono.n_elec / 2;
        let nvirt = mono.n_ext_orbs - nocc;
        nvirt * nocc
    }).collect();

    // Full matrix-vector product: computes A·z for the coupled system
    let matvec = |z_vecs: &Vec<Array1<f64>>| -> Vec<Array1<f64>> {
        // Step 1: Compute QINDZ_K = Q_K @ Z_K for all fragments
        let qindz: Vec<Array1<f64>> = (0..n_mol)
            .map(|k| qvo_vec[k].dot(&z_vecs[k]))
            .collect();

        // Step 2: For each fragment, compute A_I·z_I + inter-fragment coupling [parallel]
        (0..n_mol)
            .into_par_iter()
            .map(|idx_i| {
                let mono = &mono_states[idx_i];

                // Intra-fragment: A_I · z_I (diagonal + Coulomb)
                let mut result_i = orbital_hessian_matvec_hop(
                    mono,
                    &qvo_vec[idx_i],
                    &z_vecs[idx_i],
                );

                // LC exchange contribution to intra-fragment A_I · z_I
                if let Some(ref gamma_lr_ao) = mono.gamma_lr_ao {
                    let nocc = mono.n_elec / 2;
                    let nvirt = mono.n_ext_orbs - nocc;
                    let orbs = mono.orbs.as_ref().unwrap().view();
                    let s = mono.s.view();
                    let c_occ = orbs.slice(s![.., ..nocc]);
                    let c_virt = orbs.slice(s![.., nocc..]);
                    let v_mat = z_vecs[idx_i].view().into_shape((nvirt, nocc)).unwrap();

                    // dP = C_virt @ v @ C_occ^T + C_occ @ v^T @ C_virt^T (symmetric)
                    let cv_v: Array2<f64> = c_virt.dot(&v_mat); // [n_orbs, nocc]
                    let dp: Array2<f64> = &cv_v.dot(&c_occ.t()) + &c_occ.dot(&cv_v.t());

                    // lc_exact_exchange formula (element-wise γ⊙):
                    // hx = (γ⊙(S@dP))@S + [(γ⊙(S@dP))@S]^T + γ⊙(S@dP@S) + S@(dP⊙γ)@S
                    let s_dp: Array2<f64> = s.dot(&dp);
                    let g_sdp: Array2<f64> = gamma_lr_ao * &s_dp;
                    let tmp: Array2<f64> = g_sdp.dot(&s);
                    let mut hx: Array2<f64> = &tmp + &tmp.t();
                    hx = &hx + &(gamma_lr_ao * &s_dp.dot(&s));
                    hx = &hx + &s.dot(&(&dp * gamma_lr_ao)).dot(&s);

                    // A_LC @ v = +0.125 * C_v^T @ hx @ C_o
                    let exc_result: Array2<f64> = 0.125 * c_virt.t().dot(&hx.dot(&c_occ));
                    let exc_flat = exc_result.into_shape(nvirt * nocc).unwrap();
                    result_i += &exc_flat;
                }

                // Inter-fragment: sum_{K≠I} (-1) * Q_I^T @ gamma_KI^T @ QINDZ_K
                let fi_i = &hop_data.frag_info[idx_i];
                let mut shift_i = Array1::<f64>::zeros(mono.n_ext_atoms);
                for idx_k in 0..n_mol {
                    if idx_k != idx_i {
                        let fi_k = &hop_data.frag_info[idx_k];
                        let gamma_ik = hop_data.gamma_ext.slice(s![
                            fi_i.ext_range.clone(),
                            fi_k.ext_range.clone()
                        ]);
                        shift_i += &gamma_ik.dot(&qindz[idx_k]);
                    }
                }
                result_i -= &qvo_vec[idx_i].t().dot(&shift_i);

                result_i
            })
            .collect()
    };

    // Jacobi preconditioner: M^{-1} = 1/(eps_i - eps_a)
    let inv_diag: Vec<Array1<f64>> = mono_states
        .iter()
        .map(|mono| {
            let nocc = mono.n_elec / 2;
            let nvirt = mono.n_ext_orbs - nocc;
            let orbe = mono.orbe.as_ref().unwrap();
            let mut inv_d = Array1::<f64>::zeros(nvirt * nocc);
            for a in 0..nvirt {
                for i in 0..nocc {
                    inv_d[a * nocc + i] = 1.0 / (orbe[i] - orbe[nocc + a]);
                }
            }
            inv_d
        })
        .collect();

    // Apply preconditioner: z_i = M^{-1} @ r_i (element-wise)
    let precond = |r_vecs: &Vec<Array1<f64>>| -> Vec<Array1<f64>> {
        r_vecs
            .iter()
            .zip(inv_diag.iter())
            .map(|(ri, inv_d)| ri * inv_d)
            .collect()
    };

    // Dot product over all fragments
    let dot_all = |a: &Vec<Array1<f64>>, b: &Vec<Array1<f64>>| -> f64 {
        a.iter().zip(b.iter()).map(|(ai, bi)| ai.dot(bi)).sum()
    };

    // Vector operations over all fragments
    let vec_sub = |a: &Vec<Array1<f64>>, b: &Vec<Array1<f64>>| -> Vec<Array1<f64>> {
        a.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect()
    };
    let vec_add = |a: &Vec<Array1<f64>>, b: &Vec<Array1<f64>>| -> Vec<Array1<f64>> {
        a.iter().zip(b.iter()).map(|(ai, bi)| ai + bi).collect()
    };
    let vec_scale = |s: f64, a: &Vec<Array1<f64>>| -> Vec<Array1<f64>> {
        a.iter().map(|ai| s * ai).collect()
    };

    // Initialize with zero Z-vectors
    let mut z_vecs: Vec<Array1<f64>> = seg_sizes.iter().map(|&n| Array1::zeros(n)).collect();

    // Preconditioned CG: r = b - A·x, z = M^{-1}·r, p = z
    let az = matvec(&z_vecs);
    let mut r = vec_sub(&rhs, &az);
    let mut z_pre = precond(&r);
    let mut p = z_pre.clone();
    let mut rz = dot_all(&r, &z_pre);

    let rhs_norm = dot_all(&rhs, &rhs).sqrt();
    let abs_threshold = threshold * rhs_norm;

    for iter in 0..maxiter {
        let residual = dot_all(&r, &r).sqrt();
        if residual < abs_threshold {
            break;
        }

        let ap = matvec(&p);
        let pap = dot_all(&p, &ap);
        let alpha = rz / pap;

        z_vecs = vec_add(&z_vecs, &vec_scale(alpha, &p));
        r = vec_sub(&r, &vec_scale(alpha, &ap));

        z_pre = precond(&r);
        let rz_new = dot_all(&r, &z_pre);
        let beta = rz_new / rz;
        rz = rz_new;

        p = vec_add(&z_pre, &vec_scale(beta, &p));

        if iter == maxiter - 1 {
            eprintln!("  WARNING: SCZV-CG did not converge in {} iterations (residual={:.2e})", maxiter, residual);
        }
    }

    z_vecs
}

/// Matrix-free orbital Hessian matvec for a single HOP monomer.
///
/// Computes A_I · v = diag(eps_i - eps_a) · v - Q_I^T · gamma_I · Q_I · v
///
/// Cost: O(n_atoms * nnum) instead of O(nnum^2) for the full matrix.
fn orbital_hessian_matvec_hop(
    mono: &MonomerHopScc,
    qvo: &Array2<f64>,   // [n_ext_atoms, nnum]
    v: &Array1<f64>,      // [nnum]
) -> Array1<f64> {
    let nocc = mono.n_elec / 2;
    let nvirt = mono.n_ext_orbs - nocc;
    let orbe = mono.orbe.as_ref().unwrap();

    // Diagonal (one-electron) term: (eps_i - eps_a) * v_ai
    let mut result = Array1::<f64>::zeros(v.len());
    for a in 0..nvirt {
        for i in 0..nocc {
            let idx = a * nocc + i;
            result[idx] = (orbe[i] - orbe[nocc + a]) * v[idx];
        }
    }

    // Coulomb term: -1 * Q^T @ gamma @ Q @ v
    let qv: Array1<f64> = qvo.dot(v);
    let g_qv: Array1<f64> = mono.gamma.dot(&qv);
    let qt_g_qv: Array1<f64> = qvo.t().dot(&g_qv);
    result = result - &qt_g_qv;

    result
}

// ============================================================================
// Step 3: Response gradient contributions
// ============================================================================

/// Compute all response gradient contributions: Z×H + Z×B + Z×G_intra + Z×G_inter + Z×HOP + LC terms.
fn calculate_response_gradient_contributions(
    hop_data: &HopData,
    mono_states: &[MonomerHopScc],
    atoms: &[Atom],
    z_vectors: &[Array1<f64>],
    qvo_vec: &[Array2<f64>],
    gammafunction: &GammaFunction,
    gammafunction_lc: &Option<GammaFunction>,
    slako: &SlaterKoster,
) -> Array1<f64> {
    let n_atoms_total = atoms.len();
    let n_grad = 3 * n_atoms_total;

    // Per-monomer contributions (parallel, each thread accumulates its own gradient)
    let per_mono_grads: Vec<Array1<f64>> = mono_states
        .par_iter()
        .enumerate()
        .map(|(frag_idx, mono)| {
        let z_i = &z_vectors[frag_idx];
        let mut gradient = Array1::<f64>::zeros(n_grad);
        let n_ext_orbs = mono.n_ext_orbs;
        let nocc = mono.n_elec / 2;
        let nvirt = n_ext_orbs - nocc;
        let ext_atoms = &mono.ext_atoms;

        let orbs = mono.orbs.as_ref().unwrap().view();
        let orbe = mono.orbe.as_ref().unwrap().view();
        let s = mono.s.view();
        let dq = mono.dq.view();

        // Reshape Z to [nvirt, nocc] matrix
        let z_mat = z_i.view().into_shape([nvirt, nocc]).unwrap().to_owned();
        let c_virt = orbs.slice(s![.., nocc..]);
        let c_occ = orbs.slice(s![.., ..nocc]);

        // Z in AO basis: Z_AO = C_virt * Z_mo * C_occ^T
        let z_ao = c_virt.dot(&z_mat.dot(&c_occ.t()));
        // Symmetrized Z_AO for Mulliken charges
        let z_ao_sym = 0.5 * (&z_ao + &z_ao.t());

        // Energy-weighted Z: WZ_AO = C_virt * (Z_mo .* eps_occ) * C_occ^T
        let mut wz_mat = Array2::<f64>::zeros([nvirt, nocc]);
        for i in 0..nocc {
            for a in 0..nvirt {
                wz_mat[[a, i]] = z_mat[[a, i]] * orbe[i];
            }
        }
        let mut wz_ao = c_virt.dot(&wz_mat.dot(&c_occ.t()));

        // LC-DFTB: add D×F_lc(Z)×D correction to energy-weighted density.
        // TRI2 += -0.5 * D * DFTB_LCSHIFT(Z) * D (dftbfo.src:5796-5837)
        if let Some(ref gamma_lr_ao) = mono.gamma_lr_ao {
            let p_ao = mono.p.view();
            let z_sym: Array2<f64> = 0.5 * (&z_ao + &z_ao.t());

            // LCSHIFT formula (DFTB_LCSHIFT2):
            // lwrk = γ⊙(S·Z·S) + [2·γ⊙(S·Z) + S·(γ⊙Z)]·S
            let sz: Array2<f64> = s.dot(&z_sym);
            let szs: Array2<f64> = sz.dot(&s);
            let term_ab: Array2<f64> = gamma_lr_ao * &szs;
            let gz: Array2<f64> = gamma_lr_ao * &z_sym;
            let sgz: Array2<f64> = s.dot(&gz);
            let gsz: Array2<f64> = gamma_lr_ao * &sz;
            let lwrk: Array2<f64> = &term_ab + &(2.0 * &gsz + &sgz).dot(&s);

            // F_lc(Z) = -0.125 * sym(lwrk)
            let f_lc_z: Array2<f64> = -0.125 * (&lwrk + &lwrk.t());
            let dfd: Array2<f64> = p_ao.dot(&f_lc_z).dot(&p_ao);
            wz_ao = wz_ao + (-0.5) * &dfd;
        }

        // LC-DFTB response precomputation: LCGRAD1 and LCGRAD2
        let (f_lc_response, w_lc_zg) = if let Some(ref gamma_lr_ao) = mono.gamma_lr_ao {
            let diff_d: Array2<f64> = &mono.p - &mono.p_ref;

            // LCGRAD1 MODE=1: f_lc = (Z·S·γ)·ΔD^T + (Z·S)·(ΔD·γ)^T + Z·(S·ΔD·γ) + (Z·γ)·(S·ΔD)
            let zs_lr: Array2<f64> = z_ao.dot(&s);
            let sd_lr: Array2<f64> = s.dot(&diff_d);
            let zs_gamma: Array2<f64> = &zs_lr * gamma_lr_ao;
            let d_gamma: Array2<f64> = &diff_d * gamma_lr_ao;
            let sd_gamma: Array2<f64> = &sd_lr * gamma_lr_ao;
            let z_gamma: Array2<f64> = &z_ao * gamma_lr_ao;

            let f_lc_1 = zs_gamma.dot(&diff_d.t());
            let f_lc_2 = zs_lr.dot(&d_gamma.t());
            let f_lc_3 = z_ao.dot(&sd_gamma);
            let f_lc_4 = z_gamma.dot(&sd_lr);
            let f_lc_total: Array2<f64> = &f_lc_1 + &f_lc_2 + &f_lc_3 + &f_lc_4;

            // LCGRAD2: W matrix for gamma_lr derivative
            let ds_lr: Array2<f64> = diff_d.dot(&s);
            let sz_lr: Array2<f64> = s.dot(&z_ao);
            let szs_lr: Array2<f64> = sz_lr.dot(&s);
            let sds_lr: Array2<f64> = sd_lr.dot(&s);
            let w_lc: Array2<f64> =
                &zs_lr * &sd_lr + &ds_lr * &sz_lr + &szs_lr * &diff_d + &sds_lr * &z_ao;

            (Some(f_lc_total), Some(w_lc))
        } else {
            (None, None)
        };

        // Compute ESP_q for this monomer (external ESP from all other fragments)
        let esp_q_ext = super::helpers::compute_esp_q_hop(frag_idx, hop_data);

        // Total shift = gamma_local * dq + esp_q (intra + inter-fragment)
        let intra_shift: Array1<f64> = mono.gamma.dot(&dq);
        let total_shift: Array1<f64> = &intra_shift + &esp_q_ext;

        // Build shift in AO basis
        let shift_ao = build_shift_ao_matrix(total_shift.view(), ext_atoms, n_ext_orbs);

        // WRK_response = shift_ao * Z_AO - WZ_AO
        let wrk_response = &shift_ao * &z_ao - &wz_ao;

        // Build local-to-global mapping for this monomer
        let local_to_global: Vec<usize> = {
            let mut mapping: Vec<usize> = hop_data.monomer_indices[frag_idx].clone();
            for bond in &hop_data.detached_bonds {
                if bond.baa_fragment == frag_idx {
                    mapping.push(bond.bda_global);
                }
            }
            mapping
        };

        // Build orbital offsets for SK loop
        let orbital_offsets = build_orbital_offsets(ext_atoms);
        let n_ext_atoms = mono.n_ext_atoms;

        let mut local_grad = Array1::<f64>::zeros(3 * n_ext_atoms);

        // (a) Z×H + (b) Z×B: loop over atom pairs for SK integrals
        for i_atom in 0..n_ext_atoms {
            let atomi = &ext_atoms[i_atom];
            let mu_start = orbital_offsets[i_atom];

            for j_atom in (i_atom + 1)..n_ext_atoms {
                let atomj = &ext_atoms[j_atom];
                let r_vec = atomi - atomj;
                let dist = r_vec.norm();

                if dist >= PROXIMITY_CUTOFF {
                    continue;
                }

                let nu_start = orbital_offsets[j_atom];

                let (r, x, y, z) = if atomi <= atomj {
                    directional_cosines(&atomi.xyz, &atomj.xyz)
                } else {
                    directional_cosines(&atomj.xyz, &atomi.xyz)
                };

                let skt = slako.get(atomi.kind, atomj.kind);
                let h_cache = SplineCache::new(r, &skt.h_spline);
                let s_cache = SplineCache::new(r, &skt.s_spline);

                let mut mu = mu_start;
                for orbi in atomi.valorbs.iter() {
                    let mut nu = nu_start;
                    for orbj in atomj.valorbs.iter() {
                        let (dh_i, ds_i, dh_j, ds_j) = if atomi <= atomj {
                            let h_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &h_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                            );
                            let s_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &s_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                            );
                            (
                                [-h_grad[0], -h_grad[1], -h_grad[2]],
                                [-s_grad[0], -s_grad[1], -s_grad[2]],
                                h_grad,
                                s_grad,
                            )
                        } else {
                            let h_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &h_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                            );
                            let s_grad = slako_transformation_gradients_fast(
                                r, x, y, z, &s_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                            );
                            (
                                h_grad,
                                s_grad,
                                [-h_grad[0], -h_grad[1], -h_grad[2]],
                                [-s_grad[0], -s_grad[1], -s_grad[2]],
                            )
                        };

                        let z_mn = z_ao[[mu, nu]];
                        let z_nm = z_ao[[nu, mu]];
                        let wrk_mn = wrk_response[[mu, nu]];
                        let wrk_nm = wrk_response[[nu, mu]];

                        for dir in 0..3 {
                            // Z×H: dH/dR * (Z + Z^T)
                            let zh_i = dh_i[dir] * (z_mn + z_nm);
                            let zh_j = dh_j[dir] * (z_mn + z_nm);

                            // Z×B: dS/dR * (wrk + wrk^T)
                            let zb_i = ds_i[dir] * (wrk_mn + wrk_nm);
                            let zb_j = ds_j[dir] * (wrk_mn + wrk_nm);

                            local_grad[3 * i_atom + dir] += zh_i + zb_i;
                            local_grad[3 * j_atom + dir] += zh_j + zb_j;
                        }

                        // LCGRAD1: overlap exchange response
                        if let Some(ref f_lc) = f_lc_response {
                            let f_mn = f_lc[[mu, nu]];
                            let f_nm = f_lc[[nu, mu]];
                            let coeff = -0.125 * (f_mn + f_nm);
                            for dir in 0..3 {
                                local_grad[3 * i_atom + dir] += ds_i[dir] * coeff;
                                local_grad[3 * j_atom + dir] += ds_j[dir] * coeff;
                            }
                        }

                        nu += 1;
                    }
                    mu += 1;
                }
            }
        }

        // (c) Z×G intra: intra-fragment gamma derivative
        // Q^Z (Mulliken charges from Z) over all extended atoms
        let zs = z_ao_sym.dot(&s);
        let mut q_z = Array1::<f64>::zeros(n_ext_atoms);
        for a_idx in 0..n_ext_atoms {
            let mu_start = orbital_offsets[a_idx];
            let mu_end = orbital_offsets[a_idx + 1];
            for mu in mu_start..mu_end {
                q_z[a_idx] += zs[[mu, mu]];
            }
        }

        // Intra-fragment gamma derivative: all ext atom pairs (real + ghost)
        for i_atom in 0..n_ext_atoms {
            let atomi = &ext_atoms[i_atom];
            for j_atom in (i_atom + 1)..n_ext_atoms {
                let atomj = &ext_atoms[j_atom];
                let r_vec = atomi - atomj;
                let dist = r_vec.norm();
                if dist < 1e-10 {
                    continue;
                }

                let dgamma_dr = gammafunction.deriv(dist, atomi.number, atomj.number);
                let factor = (dq[i_atom] * q_z[j_atom] + q_z[i_atom] * dq[j_atom])
                    * dgamma_dr
                    / dist;

                for dir in 0..3 {
                    local_grad[3 * i_atom + dir] += factor * r_vec[dir];
                    local_grad[3 * j_atom + dir] -= factor * r_vec[dir];
                }
            }
        }

        // LCGRAD2: gamma_lr derivative contribution (separate atom-pair loop, no cutoff)
        if let Some(ref w_lc) = w_lc_zg {
            let gamma_lc = gammafunction_lc.as_ref().unwrap();
            let scal: f64 = -0.125;

            for i_atom in 0..n_ext_atoms {
                let atomi = &ext_atoms[i_atom];
                for j_atom in (i_atom + 1)..n_ext_atoms {
                    let atomj = &ext_atoms[j_atom];
                    let r_vec = atomi - atomj;
                    let dist = r_vec.norm();
                    if dist < 1e-10 {
                        continue;
                    }
                    let dgamma_lr_dr = gamma_lc.deriv(dist, atomi.number, atomj.number);

                    // Sum W over orbital block (j, i) — matching calculate_lc_grad2 convention
                    let mu_j = orbital_offsets[j_atom];
                    let nu_i = orbital_offsets[i_atom];
                    let mut tmp = 0.0;
                    for mu_off in 0..ext_atoms[j_atom].n_orbs {
                        for nu_off in 0..ext_atoms[i_atom].n_orbs {
                            tmp += w_lc[[mu_j + mu_off, nu_i + nu_off]];
                        }
                    }

                    let val = tmp * dgamma_lr_dr * scal / dist;
                    for dir in 0..3 {
                        local_grad[3 * i_atom + dir] += r_vec[dir] * val;
                        local_grad[3 * j_atom + dir] -= r_vec[dir] * val;
                    }
                }
            }
        }

        // Scatter local gradient to global
        for (local_idx, &global_idx) in local_to_global.iter().enumerate() {
            if local_idx >= n_ext_atoms {
                break;
            }
            for k in 0..3 {
                gradient[3 * global_idx + k] += local_grad[3 * local_idx + k];
            }
        }

        // (e) Z×HOP: HOP projector derivative with Z_AO instead of P
        compute_z_hop_gradient(
            frag_idx,
            hop_data,
            mono,
            &z_ao_sym,
            atoms,
            slako,
            &mut gradient,
        );

        gradient
    })
    .collect();

    // Sum per-monomer thread-local gradients
    let mut gradient = Array1::<f64>::zeros(n_grad);
    for g in &per_mono_grads {
        gradient += g;
    }

    // (d) Z×G inter: inter-fragment gamma derivative
    add_inter_fragment_response_gradient(
        hop_data,
        mono_states,
        atoms,
        z_vectors,
        qvo_vec,
        gammafunction,
        &mut gradient,
    );

    gradient
}

/// Compute Z×HOP: HOP projector derivative evaluated with Z_AO_sym instead of P.
///
/// This reuses hop_gradient_single_bond_dftb() but passes Z_AO_sym as density.
/// Only monomer contributions are needed — the Z-vector already accounts for
/// pair effects through the Lagrangian.
fn compute_z_hop_gradient(
    frag_idx: usize,
    hop_data: &HopData,
    mono: &MonomerHopScc,
    z_ao_sym: &Array2<f64>,
    atoms: &[Atom],
    slako: &SlaterKoster,
    gradient: &mut Array1<f64>,
) {
    let fi = &hop_data.frag_info[frag_idx];
    let detached_bonds = &hop_data.detached_bonds;
    let ext_atoms = &mono.ext_atoms;
    let n_ext_atoms = mono.n_ext_atoms;
    let s = mono.s.view();
    let frag_atom_start = hop_data.monomer_indices[frag_idx][0];

    // BDA bonds: bond-pointing hybrid
    let bda_bonds: Vec<&DetachedBond> = detached_bonds
        .iter()
        .filter(|b| b.bda_fragment == frag_idx)
        .collect();

    // Ghost bonds: 3 non-bond hybrids
    let ghost_bonds: Vec<&DetachedBond> = detached_bonds
        .iter()
        .filter(|b| b.baa_fragment == frag_idx)
        .collect();

    if bda_bonds.is_empty() && ghost_bonds.is_empty() {
        return;
    }

    let mut local_grad = Array1::<f64>::zeros(3 * n_ext_atoms);

    // BDA projections with Z instead of P
    for bond in &bda_bonds {
        let bda_local = bond.bda_global - frag_atom_start;
        let bda_pos = &atoms[bond.bda_global].xyz;
        let baa_pos = &atoms[bond.baa_global].xyz;
        let bond_vec = *baa_pos - *bda_pos;

        let rotated_sp3 = compute_rotated_sp3_dftb(&bond_vec);
        let dd = compute_bda_dd_matrix(rotated_sp3.view(), HOP_SHIFT);

        let (ao_start, nao) = get_bda_ao_range_dftb(ext_atoms, bda_local);
        let dd_full = if nao == dd.nrows() {
            dd.to_owned()
        } else {
            let mut dd_f = Array2::<f64>::zeros([nao, nao]);
            let sz = dd.nrows().min(nao);
            dd_f.slice_mut(s![..sz, ..sz]).assign(&dd.slice(s![..sz, ..sz]));
            dd_f
        };

        // HOPSDER: overlap derivative contribution
        hop_overlap_derivative_gradient_dftb(
            z_ao_sym.view(), s, dd_full.view(), ao_start, nao, ext_atoms,
            &mut local_grad, slako,
        );

        // HOPCODER: coefficient derivative contribution
        let mut baa_grad_3 = [0.0f64; 3];
        let bda_grad_offset = 3 * bda_local;
        hop_coefficient_derivative_gradient_dftb(
            z_ao_sym.view(), s, rotated_sp3.view(), &bond_vec, ao_start, nao,
            bda_grad_offset, &mut baa_grad_3, &mut local_grad, 1.0,
        );
        for k in 0..3 {
            gradient[3 * bond.baa_global + k] += baa_grad_3[k];
        }
    }

    // Ghost projections with Z instead of P
    for (ghost_idx, bond) in ghost_bonds.iter().enumerate() {
        let ghost_local = fi.n_real_atoms + ghost_idx;
        let bda_pos = &atoms[bond.bda_global].xyz;
        let baa_pos = &atoms[bond.baa_global].xyz;
        let bond_vec = *baa_pos - *bda_pos;

        let dd_ghost = compute_ghost_nonbond_dd(&bond_vec, HOP_SHIFT);
        let rotated_sp3 = compute_rotated_sp3_dftb(&bond_vec);

        let (ao_start, nao) = get_bda_ao_range_dftb(ext_atoms, ghost_local);
        let dd_full = if nao == dd_ghost.nrows() {
            dd_ghost.to_owned()
        } else {
            let mut dd_f = Array2::<f64>::zeros([nao, nao]);
            let sz = dd_ghost.nrows().min(nao);
            dd_f.slice_mut(s![..sz, ..sz]).assign(&dd_ghost.slice(s![..sz, ..sz]));
            dd_f
        };

        // HOPSDER: overlap derivative contribution
        hop_overlap_derivative_gradient_dftb(
            z_ao_sym.view(), s, dd_full.view(), ao_start, nao, ext_atoms,
            &mut local_grad, slako,
        );

        // HOPCODER: coefficient derivative contribution (ghost uses negative sign factor)
        let mut baa_grad_3 = [0.0f64; 3];
        let bda_grad_offset = 3 * ghost_local;
        hop_coefficient_derivative_gradient_dftb(
            z_ao_sym.view(), s, rotated_sp3.view(), &bond_vec, ao_start, nao,
            bda_grad_offset, &mut baa_grad_3, &mut local_grad, -1.0,
        );
        for k in 0..3 {
            gradient[3 * bond.baa_global + k] += baa_grad_3[k];
        }
    }

    // Map local gradient to global (real atoms + ghost → BDA global)
    for (local_idx, &global_idx) in hop_data.monomer_indices[frag_idx].iter().enumerate() {
        for k in 0..3 {
            gradient[3 * global_idx + k] += local_grad[3 * local_idx + k];
        }
    }
    // Ghost atoms → BDA global
    for (ghost_idx, bond) in ghost_bonds.iter().enumerate() {
        let local_idx = fi.n_real_atoms + ghost_idx;
        for k in 0..3 {
            gradient[3 * bond.bda_global + k] += local_grad[3 * local_idx + k];
        }
    }
}

/// Add inter-fragment response gradient contribution.
///
/// For each monomer I, compute Q^Z_I (Mulliken charges from Z, all ext atoms),
/// then for each other monomer J:
///   G^a += Q^Z_I[a] * dq_J[b] * dgamma/dR_{a,b}
///
/// Ghost atoms are clones of BDA (same element, e.g. carbon for C-C bond, with 1 electron).
/// Ghost atom gradients scatter to BDA global positions.
fn add_inter_fragment_response_gradient(
    hop_data: &HopData,
    mono_states: &[MonomerHopScc],
    _atoms: &[Atom],
    z_vectors: &[Array1<f64>],
    _qvo_vec: &[Array2<f64>],
    gammafunction: &GammaFunction,
    gradient: &mut Array1<f64>,
) {
    let n_mol = mono_states.len();

    // Compute Q^Z (Mulliken charges from Z) for each monomer, all ext atoms [parallel]
    let q_z_all: Vec<Array1<f64>> = mono_states
        .par_iter()
        .enumerate()
        .map(|(idx_i, mono)| {
            let n_ext_orbs = mono.n_ext_orbs;
            let nocc = mono.n_elec / 2;
            let nvirt = n_ext_orbs - nocc;
            let ext_atoms = &mono.ext_atoms;

            let orbs = mono.orbs.as_ref().unwrap().view();
            let s = mono.s.view();
            let z_i = &z_vectors[idx_i];

            let z_mat = z_i.view().into_shape([nvirt, nocc]).unwrap().to_owned();
            let c_virt = orbs.slice(s![.., nocc..]);
            let c_occ = orbs.slice(s![.., ..nocc]);
            let z_ao_unsym = c_virt.dot(&z_mat.dot(&c_occ.t()));
            let z_ao_sym = 0.5 * (&z_ao_unsym + &z_ao_unsym.t());

            let zs = z_ao_sym.dot(&s);
            let mut q_z = Array1::<f64>::zeros(mono.n_ext_atoms);
            let orbital_offsets = build_orbital_offsets(ext_atoms);
            for a_idx in 0..mono.n_ext_atoms {
                let mu_start = orbital_offsets[a_idx];
                let mu_end = orbital_offsets[a_idx + 1];
                for mu in mu_start..mu_end {
                    q_z[a_idx] += zs[[mu, mu]];
                }
            }
            q_z
        })
        .collect();

    // Build ext-local-to-global index mapping for each fragment
    let ext_to_global: Vec<Vec<usize>> = (0..n_mol)
        .map(|idx| {
            let mut mapping: Vec<usize> = hop_data.monomer_indices[idx].clone();
            for bond in &hop_data.detached_bonds {
                if bond.baa_fragment == idx {
                    mapping.push(bond.bda_global);
                }
            }
            mapping
        })
        .collect();

    // Inter-fragment gamma derivative: all ext atom pairs between different monomers [parallel]
    let n_grad = gradient.len();
    let inter_grads: Vec<Array1<f64>> = (0..n_mol)
        .into_par_iter()
        .map(|idx_i| {
            let q_z_i = &q_z_all[idx_i];
            let ext_atoms_i = &mono_states[idx_i].ext_atoms;
            let n_ext_i = mono_states[idx_i].n_ext_atoms;
            let mut local_gradient = Array1::<f64>::zeros(n_grad);

            for idx_j in 0..n_mol {
                if idx_i == idx_j {
                    continue;
                }
                let dq_j = &mono_states[idx_j].dq;
                let ext_atoms_j = &mono_states[idx_j].ext_atoms;
                let n_ext_j = mono_states[idx_j].n_ext_atoms;

                for local_i in 0..n_ext_i {
                    let global_i = ext_to_global[idx_i][local_i];
                    let atom_i = &ext_atoms_i[local_i];

                    for local_j in 0..n_ext_j {
                        let global_j = ext_to_global[idx_j][local_j];
                        let atom_j = &ext_atoms_j[local_j];

                        let dx = atom_i.xyz[0] - atom_j.xyz[0];
                        let dy = atom_i.xyz[1] - atom_j.xyz[1];
                        let dz = atom_i.xyz[2] - atom_j.xyz[2];
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                        if dist < 1e-10 {
                            continue;
                        }

                        let dgamma_dr =
                            gammafunction.deriv(dist, atom_i.number, atom_j.number);
                        let factor = q_z_i[local_i] * dq_j[local_j] * dgamma_dr / dist;

                        local_gradient[3 * global_i + 0] += factor * dx;
                        local_gradient[3 * global_i + 1] += factor * dy;
                        local_gradient[3 * global_i + 2] += factor * dz;
                        local_gradient[3 * global_j + 0] -= factor * dx;
                        local_gradient[3 * global_j + 1] -= factor * dy;
                        local_gradient[3 * global_j + 2] -= factor * dz;
                    }
                }
            }
            local_gradient
        })
        .collect();

    for g in &inter_grads {
        *gradient += g;
    }
}

/// Add the "response addlag" contribution to the gradient.
///
/// This corresponds to DFTB_ZVEC_KGRAD: after the SCZV converges, the Z-vector
/// Mulliken charges (Q_Z) create an inter-fragment shift SHIFTZ = gamma * Q_Z.
/// This shift produces a gradient through:
///   gradient += dS/dR * (shift_z_ao * P - 0.5 * P * (shift_z_ao * S) * P)
///
/// This term is separate from Z*B (which uses the SCC shift with Z) and Z*G
/// (which uses dgamma/dR). It uses the gamma VALUE (not derivative) with Q_Z
/// to build a shift, then the density matrix P (not Z) with the overlap derivative.
fn add_response_addlag_hop(
    hop_data: &HopData,
    mono_states: &[MonomerHopScc],
    _atoms: &[Atom],
    z_vectors: &[Array1<f64>],
    gammafunction: &GammaFunction,
    slako: &SlaterKoster,
    gradient: &mut Array1<f64>,
) {
    let n_mol = mono_states.len();

    // Step 1: Compute Q_Z (Mulliken charges from Z-vector) for all fragments [parallel]
    let q_z_all: Vec<Array1<f64>> = mono_states
        .par_iter()
        .enumerate()
        .map(|(idx, mono)| {
            let n_ext_orbs = mono.n_ext_orbs;
            let nocc = mono.n_elec / 2;
            let nvirt = n_ext_orbs - nocc;
            let ext_atoms = &mono.ext_atoms;
            let orbs = mono.orbs.as_ref().unwrap().view();
            let s = mono.s.view();
            let z_i = &z_vectors[idx];

            let z_mat = z_i.view().into_shape([nvirt, nocc]).unwrap().to_owned();
            let c_virt = orbs.slice(s![.., nocc..]);
            let c_occ = orbs.slice(s![.., ..nocc]);
            let z_ao_unsym = c_virt.dot(&z_mat.dot(&c_occ.t()));
            let z_ao_sym = 0.5 * (&z_ao_unsym + &z_ao_unsym.t());

            let zs = z_ao_sym.dot(&s);
            let mut q_z = Array1::<f64>::zeros(mono.n_ext_atoms);
            let orbital_offsets = build_orbital_offsets(ext_atoms);
            for a_idx in 0..mono.n_ext_atoms {
                let mu_start = orbital_offsets[a_idx];
                let mu_end = orbital_offsets[a_idx + 1];
                for mu in mu_start..mu_end {
                    q_z[a_idx] += zs[[mu, mu]];
                }
            }
            q_z
        })
        .collect();

    // Step 2: Build ext-local-to-global mapping for each fragment
    let ext_to_global: Vec<Vec<usize>> = (0..n_mol)
        .map(|idx| {
            let mut mapping: Vec<usize> = hop_data.monomer_indices[idx].clone();
            for bond in &hop_data.detached_bonds {
                if bond.baa_fragment == idx {
                    mapping.push(bond.bda_global);
                }
            }
            mapping
        })
        .collect();

    // Step 3: Compute SHIFTZ for each fragment's extended atoms [parallel]
    // SHIFTZ[a] = sum over ALL atoms b (all fragments): gamma(a,b) * Q_Z[b]
    let shiftz_all: Vec<Array1<f64>> = mono_states
        .par_iter()
        .enumerate()
        .map(|(frag_k, mono_k)| {
            let ext_atoms_k = &mono_k.ext_atoms;

            // Intra: gamma_local.dot(Q_Z_local)
            let mut shiftz = mono_k.gamma.dot(&q_z_all[frag_k]);

            // Inter: sum over atoms in other fragments
            for (frag_j, mono_j) in mono_states.iter().enumerate() {
                if frag_j == frag_k {
                    continue;
                }
                let ext_atoms_j = &mono_j.ext_atoms;
                let q_z_j = &q_z_all[frag_j];

                for a in 0..mono_k.n_ext_atoms {
                    for b in 0..mono_j.n_ext_atoms {
                        let r_vec = &ext_atoms_k[a] - &ext_atoms_j[b];
                        let dist = r_vec.norm();
                        // a ghost atom of fragment k coincides with the BAA
                        // atom it caps in fragment j: use the analytic limit
                        let gamma_ab = if dist < 1.0e-9 {
                            gammafunction.eval_limit0_pair(
                                ext_atoms_k[a].number,
                                ext_atoms_j[b].number,
                            )
                        } else {
                            gammafunction.eval(
                                dist,
                                ext_atoms_k[a].number,
                                ext_atoms_j[b].number,
                            )
                        };
                        shiftz[a] += gamma_ab * q_z_j[b];
                    }
                }
            }
            shiftz
        })
        .collect();

    // Step 4: For each fragment, compute addlag gradient contribution [parallel]
    let n_grad = gradient.len();
    let addlag_grads: Vec<Array1<f64>> = mono_states
        .par_iter()
        .enumerate()
        .map(|(frag_idx, mono)| {
            let ext_atoms = &mono.ext_atoms;
            let n_ext_atoms = mono.n_ext_atoms;
            let n_ext_orbs = mono.n_ext_orbs;
            let p = mono.p.view();
            let s = mono.s.view();
            let shiftz = &shiftz_all[frag_idx];

            let shift_z_ao = build_shift_ao_matrix(shiftz.view(), ext_atoms, n_ext_orbs);

            let shift_s = &shift_z_ao * &s;
            let d_shift_s = p.dot(&shift_s);
            let d_shift_s_d = d_shift_s.dot(&p);

            let mut wrk1_addlag = Array2::<f64>::zeros([n_ext_orbs, n_ext_orbs]);
            for mu in 0..n_ext_orbs {
                for nu in 0..n_ext_orbs {
                    wrk1_addlag[[mu, nu]] =
                        shift_z_ao[[mu, nu]] * p[[mu, nu]] - 0.5 * d_shift_s_d[[mu, nu]];
                }
            }

            let local_to_global = &ext_to_global[frag_idx];
            let orbital_offsets = build_orbital_offsets(ext_atoms);
            let mut local_grad = Array1::<f64>::zeros(3 * n_ext_atoms);

            for i_atom in 0..n_ext_atoms {
                let atomi = &ext_atoms[i_atom];
                let mu_start = orbital_offsets[i_atom];

                for j_atom in (i_atom + 1)..n_ext_atoms {
                    let atomj = &ext_atoms[j_atom];
                    let r_vec = atomi - atomj;
                    let dist = r_vec.norm();

                    if dist >= PROXIMITY_CUTOFF {
                        continue;
                    }

                    let nu_start = orbital_offsets[j_atom];

                    let (r, x, y, z) = if atomi <= atomj {
                        directional_cosines(&atomi.xyz, &atomj.xyz)
                    } else {
                        directional_cosines(&atomj.xyz, &atomi.xyz)
                    };

                    let skt = slako.get(atomi.kind, atomj.kind);
                    let s_cache = SplineCache::new(r, &skt.s_spline);

                    let mut mu = mu_start;
                    for orbi in atomi.valorbs.iter() {
                        let mut nu = nu_start;
                        for orbj in atomj.valorbs.iter() {
                            let (ds_i, ds_j) = if atomi <= atomj {
                                let s_grad = slako_transformation_gradients_fast(
                                    r, x, y, z, &s_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                                );
                                (
                                    [-s_grad[0], -s_grad[1], -s_grad[2]],
                                    s_grad,
                                )
                            } else {
                                let s_grad = slako_transformation_gradients_fast(
                                    r, x, y, z, &s_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                                );
                                (
                                    s_grad,
                                    [-s_grad[0], -s_grad[1], -s_grad[2]],
                                )
                            };

                            let wrk_mn = wrk1_addlag[[mu, nu]];
                            let wrk_nm = wrk1_addlag[[nu, mu]];

                            for dir in 0..3 {
                                local_grad[3 * i_atom + dir] +=
                                    ds_i[dir] * (wrk_mn + wrk_nm);
                                local_grad[3 * j_atom + dir] +=
                                    ds_j[dir] * (wrk_mn + wrk_nm);
                            }

                            nu += 1;
                        }
                        mu += 1;
                    }
                }
            }

            // Scatter to global
            let mut thread_grad = Array1::<f64>::zeros(n_grad);
            for (local_idx, &global_idx) in local_to_global.iter().enumerate() {
                if local_idx >= n_ext_atoms {
                    break;
                }
                for k in 0..3 {
                    thread_grad[3 * global_idx + k] += local_grad[3 * local_idx + k];
                }
            }
            thread_grad
        })
        .collect();

    for g in &addlag_grads {
        *gradient += g;
    }
}
