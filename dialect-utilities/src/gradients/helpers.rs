use dialect_dftb_core::parameters::RepulsivePotential;
use dialect_dftb_core::atom::Atom;
use dialect_dftb_core::gamma_approximation::GammaFunction;
use nalgebra::Vector3;
use ndarray::{s, Array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use rayon::iter::*;

/// Precompute coefficient matrices for on-the-fly LC gradient computation
/// Returns (coeff_s, coeff_g_ao) where:
/// - coeff_s[mu,nu]: coefficient for overlap derivative dS[mu,nu]
/// - coeff_g_ao[mu,beta]: coefficient for gamma_lr_ao derivative dG[mu,beta]
///
/// The LC gradient formula (from f_lr) has 12 terms, 8 involving dS and 4 involving dG.
/// We precompute coefficients so that:
/// gradient_LC = -0.25 * Σ_{mu,nu} dS[mu,nu] * coeff_s[mu,nu]
///             + -0.25 * Σ_{mu,beta} dG[mu,beta] * coeff_g_ao[mu,beta]
pub fn compute_lr_coefficients_onthefly(
    diff_p: ArrayView2<f64>,
    s: ArrayView2<f64>,
    gamma_lr_ao: ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let n_orb = s.nrows();

    // v = diff_p (P - P0)
    let v = diff_p;

    // Precompute auxiliary matrices (same as in f_lr_par)
    let sv: Array2<f64> = s.dot(&v); // S · v
    let v_t = v.t(); // v^T
    let sv_t: Array2<f64> = s.dot(&v_t); // S · v^T
    let gv: Array2<f64> = &gamma_lr_ao * &v; // γ * v (element-wise)
    let t_sv = sv.t(); // (S·v)^T = v^T·S^T
    let svg_t: Array2<f64> = (&sv * &gamma_lr_ao).t().to_owned(); // (S·v * γ)^T
    let sgv_t: Array2<f64> = s.dot(&gv).t().to_owned(); // (S·(γ*v))^T
    let s_t_sv: Array2<f64> = s.dot(&t_sv); // S · (S·v)^T

    // coeff_s[a,k]: coefficient for dS[a,k]
    // This is derived by expanding each of the 8 dS-dependent terms and collecting coefficients.
    //
    // The 8 terms involving dS in f_lr are (before the 0.25 factor):
    // Term 1: g * (dS · (Sv)^T)  -->  g[a,b] * Σ_k dS[a,k] * t_sv[k,b]
    // Term 2: (dS·v^T * g) · S   -->  Σ_c (Σ_k dS[a,k]*v^T[k,c]) * g[a,c] * S[c,b]
    // Term 3: dS · (Sv*g)^T      -->  Σ_k dS[a,k] * svg_t[k,b]
    // Term 4: dS · (S·(g*v))^T   -->  Σ_k dS[a,k] * sgv_t[k,b]
    // Term 5: g * (S · (dS·v)^T) -->  g[a,b] * Σ_c S[b,c] * (dS·v)^T[c,a] = g[a,b]*S[b,c]*(Σ_k dS[c,k]*v[k,a])
    // Term 6: (Sv^T * g) · dS^T  -->  (sv_t*g)[a,c] * dS^T[c,b] = (sv_t*g)[a,c] * dS[b,c]
    // Term 7: S · (dS·v * g)^T   -->  S[a,c] * ((dS·v)*g)^T[c,b] = S[a,c] * (Σ_k dS[b,k]*v[k,c]*g[b,c])
    // Term 8: S · (dS · (g*v))^T -->  S[a,c] * (dS·gv)^T[c,b] = S[a,c] * Σ_k dS[b,k] * gv[k,c]
    //
    // After contraction with diff_p[a,b], the coefficient for dS[mu,nu] is obtained by summing
    // over a,b the contribution from each term where dS[mu,nu] appears.

    let mut coeff_s: Array2<f64> = Array2::zeros((n_orb, n_orb));

    // For efficiency, precompute products with diff_p
    let gv_elem: Array2<f64> = &gamma_lr_ao * &v; // element-wise g*v
    let sv_t_g: Array2<f64> = &sv_t * &gamma_lr_ao;

    // coeff1[a,k] = Σ_b g[a,b] * t_sv[k,b] * v[a,b] = (g*v)[a,:] · t_sv[k,:]^T = (g*v) · sv
    // But we need: Σ_b g[a,b] * v[a,b] * t_sv[k,b]
    // = Σ_b (g*v)[a,b] * sv[b,k]
    let coeff1: Array2<f64> = gv_elem.dot(&sv); // [n_orb, n_orb]: [a,k]

    // coeff2[a,k] = Σ_b Σ_c v^T[k,c] * g[a,c] * S[c,b] * v[a,b]
    // = Σ_c v[c,k] * g[a,c] * (S·v^T)[c,a]
    // = Σ_c v[c,k] * g[a,c] * sv_t[c,a]
    let g_sv_t_elem: Array2<f64> = &gamma_lr_ao * &sv_t.t(); // g[a,c] * sv_t[c,a] transposed: [a,c]*[a,c]
    let coeff2: Array2<f64> = g_sv_t_elem.dot(&v); // [a,k]

    // coeff3[a,k] = Σ_b svg_t[k,b] * v[a,b] = svg_t · v^T  at [k,a], need [a,k]
    let coeff3: Array2<f64> = v.dot(&svg_t.t()); // [a,k]

    // coeff4[a,k] = Σ_b sgv_t[k,b] * v[a,b]
    let coeff4: Array2<f64> = v.dot(&sgv_t.t()); // [a,k]

    // Terms 5-8 involve dS at different positions (like dS[b,k])
    // coeff5: for term 5 - g * (S · (dS·v)^T)
    // Term 5: d_f5[a,b] = g[a,b] * Σ_{c,k} S[a,c] * dS[b,k] * v[k,c]
    // After contracting with v[a,b]:
    // coeff5[b,k] = Σ_{a,c} (g*v)[a,b] * S[a,c] * v[k,c]
    //            = Σ_a (g*v)[a,b] * (S·v^T)[a,k]
    //            = ((g*v)^T · (S·v^T))[b,k]
    let coeff5: Array2<f64> = gv_elem.t().dot(&sv_t); // [b,k]

    // coeff6: (sv_t*g)[a,c] * v[a,b] for dS[b,c]
    // coeff6[b,c] = Σ_a (sv_t*g)[a,c] * v[a,b] = (sv_t*g)^T · v at [c,b], want [b,c]
    let coeff6_cb: Array2<f64> = sv_t_g.t().dot(&v); // [c,b]
    let coeff6: Array2<f64> = coeff6_cb.t().to_owned(); // [b,c]

    // coeff7: S[a,c] * v[k,c] * g[b,c] * v[a,b] for dS[b,k]
    // coeff7[b,k] = Σ_{a,c} S[a,c] * v[a,b] * v[k,c] * g[b,c]
    //             = Σ_c v[k,c] * g[b,c] * Σ_a S[a,c] * v[a,b]
    //             = Σ_c v[k,c] * g[b,c] * (S^T·v)[c,b]
    // Let sv_from_st = S^T · v: [c,b]
    let st_v: Array2<f64> = s.t().dot(&v); // [c,b]
                                           // Then: Σ_c v[k,c] * g[b,c] * st_v[c,b] = Σ_c v[k,c] * (g*st_v^T)[b,c]
                                           // (g * st_v^T)[b,c] where st_v^T[b,c] = st_v[c,b]... this is getting complex
                                           // Let's use: (g .* st_v.T)  [b,c] then dot with v.T[c,k]
    let g_stv_t: Array2<f64> = &gamma_lr_ao * &st_v.t(); // [b,c]
    let coeff7: Array2<f64> = g_stv_t.dot(&v_t); // [b,k]

    // coeff8: S[a,c] * gv[k,c] * v[a,b] for dS[b,k]
    // coeff8[b,k] = Σ_{a,c} S[a,c] * v[a,b] * gv[k,c]
    //             = Σ_a v[a,b] * Σ_c S[a,c] * gv[k,c]
    //             = v^T[b,a] * (S · gv^T)[a,k]
    // s_gv_t = S · gv^T: [a,k]
    let s_gvt: Array2<f64> = s.dot(&gv.t()); // [a,k]
    let coeff8: Array2<f64> = v_t.dot(&s_gvt); // [b,k]

    // Now combine all coefficients
    // Terms 1-4 contribute at [a,k], terms 5-8 at different indices
    // We need to add them at the correct positions
    for a in 0..n_orb {
        for k in 0..n_orb {
            coeff_s[[a, k]] += coeff1[[a, k]] + coeff2[[a, k]] + coeff3[[a, k]] + coeff4[[a, k]];
        }
    }
    // Terms 5-8 are indexed differently, add them
    coeff_s = &coeff_s + &coeff5 + &coeff6 + &coeff7 + &coeff8;

    // ============ Gamma LR gradient coefficients ============
    // Terms 9-12 involve dG (gamma_lr gradient):
    // Term 9:  dG * (S · (Sv)^T)  -->  dG[a,b] * s_t_sv[a,b]
    // Term 10: (Sv^T * dG) · S    -->  (sv_t * dG)[a,c] * S[c,b]
    // Term 11: S · (Sv * dG)^T    -->  S[a,c] * (sv * dG)^T[c,b] = S[a,c] * sv[b,c] * dG[b,c]
    // Term 12: S · (S · (dG*v))^T -->  S[a,c] * (S·(dG*v))^T[c,b] = S[a,c] * (S·(dG*v))[b,c]
    //
    // After contraction with diff_p[a,b]:
    // let mut coeff_g_ao: Array2<f64> = Array2::zeros((n_orb, n_orb));

    // coeff9[a,b] = s_t_sv[a,b] * v[a,b] (element-wise product contracted with v)
    // No, the dG[a,b] multiplies the whole matrix term.
    // Term 9 contribution to gradient = Σ_{a,b} dG[a,b] * s_t_sv[a,b] * v[a,b]
    // So coeff9[a,b] = s_t_sv[a,b] * v[a,b]
    let coeff9: Array2<f64> = &s_t_sv * &v;

    // Term 10: Σ_{a,b,c} (sv_t[a,c] * dG[a,c]) * S[c,b] * v[a,b]
    // For dG at position [a,c]: coeff = Σ_b sv_t[a,c] * S[c,b] * v[a,b]
    //                                 = sv_t[a,c] * (S · v^T)[c,a]
    //                                 = sv_t[a,c] * sv_t[c,a]
    // coeff10[a,c] = sv_t[a,c] * sv_t[c,a]
    let coeff10: Array2<f64> = &sv_t * &sv_t.t();

    // Term 11: Σ_{a,b,c} S[a,c] * sv[b,c] * dG[b,c] * v[a,b]
    // For dG at position [b,c]: coeff = Σ_a S[a,c] * sv[b,c] * v[a,b]
    //                                 = sv[b,c] * Σ_a S[a,c] * v[a,b]
    //                                 = sv[b,c] * (S^T · v)[c,b]
    //                                 = sv[b,c] * st_v[c,b]
    // coeff11[b,c] = sv[b,c] * st_v[c,b] = sv * st_v^T
    let coeff11: Array2<f64> = &sv * &st_v.t();

    // Term 12: Σ_{a,b,c} S[a,c] * (S·(dG*v))[b,c] * v[a,b]
    // (S·(dG*v))[b,c] = Σ_k S[b,k] * dG[k,c] * v[k,c]
    // So for dG at position [k,c]:
    // coeff = Σ_{a,b} S[a,c] * S[b,k] * v[k,c] * v[a,b]
    //       = v[k,c] * Σ_a S[a,c] * Σ_b S[b,k] * v[a,b]
    //       = v[k,c] * Σ_a S[a,c] * (S^T · v^T)[k,a]
    //       = v[k,c] * Σ_a S[a,c] * s_vt[k,a]  where s_vt = S^T · v^T = (v·S)^T
    // Let vs = v · S: [a,k], then s_vt = vs^T: [k,a]
    // coeff12[k,c] = v[k,c] * Σ_a S[a,c] * vs^T[k,a]
    //              = v[k,c] * (vs^T · S)[k,c]
    //              = v[k,c] * (S^T · vs)[c,k]^T ... getting messy
    // Let's compute: vs = v · S, then vs_s = vs · S^T = v · S · S^T
    // coeff12[k,c] = v[k,c] * vs_s^T[k,c] ... no
    // Actually: Σ_a vs^T[k,a] * S[a,c] = (vs^T · S)[k,c] = (S^T · vs)^T [k,c]
    // vs = v · S, so S^T · vs = S^T · v · S
    let stvs: Array2<f64> = s.t().dot(&v).dot(&s); // S^T · v · S: [c,k] after reshape
                                                   // wait, let me redo: S^T [c,a] · v [a,b] · S [b,k] => [c,k]
                                                   // So (S^T · v · S)[c,k] and we need [k,c]
    let coeff12: Array2<f64> = &v * &stvs.t(); // v[k,c] * stvs^T[k,c] where stvs^T[k,c] = stvs[c,k]

    // Combine gamma coefficients
    let coeff_g_ao = &coeff9 + &coeff10 + &coeff11 + &coeff12;

    // IMPORTANT: Symmetrize the coefficient matrices!
    // In the reference computation, grad_s is symmetric (grad_s[nc,mu,nu] = grad_s[nc,nu,mu])
    // and the sum Σ_{mu,nu} grad_s[nc,mu,nu] * coeff[mu,nu] includes both (mu,nu) and (nu,mu).
    // In the on-the-fly computation, each pair (mu,nu) is processed once, so we need to use
    // the symmetrized coefficient: (coeff[mu,nu] + coeff[nu,mu])
    let coeff_s_sym = &coeff_s + &coeff_s.t();
    let coeff_g_ao_sym = &coeff_g_ao + &coeff_g_ao.t();

    (coeff_s_sym, coeff_g_ao_sym)
}

/// Compute dG (gamma gradient) contributions for f_v using atomwise gamma gradients.
///
/// The dgsv term in f_v: dgsv[a] = sum_c dg[nc, a, c] * sv[c]
/// For nc = 3*i + dir, dg[nc, a, c] is non-zero only when orbital a OR c belongs to atom i.
///
/// After full contraction: sum_{a,b} s[a,b] * contract_with[a,b] * (dgsv[a] + dgsv[b])
///
/// This function computes these contributions on-the-fly using gamma_func.deriv()
/// instead of loading the full grad_gamma_ao array.
pub fn compute_fv_dg_contributions(
    gammafunction: &GammaFunction,
    atoms: &[Atom],
    orbital_offsets: &[usize], // orbital_offsets[i] = first orbital of atom i
    s: ArrayView2<f64>,
    v: ArrayView2<f64>,
    contract_with: ArrayView2<f64>,
    n_atoms: usize,
    _n_orbs: usize,
) -> Array1<f64> {
    let mut result = Array1::zeros(3 * n_atoms);

    // Precompute sv and sc (s * contract_with summed appropriately)
    let vp: Array2<f64> = &v + &v.t();
    let sv: Array1<f64> = (&s * &vp).sum_axis(Axis(0)); // sum over rows for each column

    // For dgsv[a] term in f_v:
    // sum_{a,b} 0.25 * s[a,b] * contract_with[a,b] * dgsv[a]
    // = sum_a dgsv[a] * 0.25 * sum_b s[a,b] * contract_with[a,b]
    //
    // dgsv[a] = sum_c dg[nc, a, c] * sv[c]
    // For nc = 3*i+dir, non-zero only when a in atom i or c in atom i
    //
    // Case 1: a belongs to atom i
    //   dgsv[a] = sum_c g1_atomwise[nc, i, atom(c)] * sv[c]
    //   Contribution = sum_{a in atom_i} (sum_c g1[i, atom(c)] * sv[c]) * sc_row[a]
    //               where sc_row[a] = 0.25 * sum_b s[a,b] * c[a,b]
    //
    // Case 2: c belongs to atom i (but a doesn't)
    //   For each c in atom_i: dg[nc, a, c] = g1_atomwise[nc, i, atom(a)]
    //   Contribution to dgsv[a] = sum_{c in atom_i} g1[i, atom(a)] * sv[c]
    //                           = g1[i, atom(a)] * sv_sum[i] where sv_sum[i] = sum_{c in atom_i} sv[c]

    // Similarly for dgsv[b] term

    // Precompute per-orbital and per-atom sums
    let sc: Array2<f64> = &s * &contract_with;
    let sc_row: Array1<f64> = 0.25 * sc.sum_axis(Axis(1)); // [n_orbs], sc_row[a] = 0.25 * sum_b sc[a,b]
    let sc_col: Array1<f64> = 0.25 * sc.sum_axis(Axis(0)); // [n_orbs], sc_col[b] = 0.25 * sum_a sc[a,b]

    // Precompute sv sum per atom
    let mut sv_atom: Array1<f64> = Array1::zeros(n_atoms);
    for j in 0..n_atoms {
        for c in orbital_offsets[j]..orbital_offsets[j + 1] {
            sv_atom[j] += sv[c];
        }
    }

    // Precompute sc_row sum per atom (for dgsv[a] where a in atom i)
    let mut sc_row_atom: Array1<f64> = Array1::zeros(n_atoms);
    for i in 0..n_atoms {
        for a in orbital_offsets[i]..orbital_offsets[i + 1] {
            sc_row_atom[i] += sc_row[a];
        }
    }

    // Precompute sc_col sum per atom (for dgsv[b] term)
    let mut sc_col_atom: Array1<f64> = Array1::zeros(n_atoms);
    for i in 0..n_atoms {
        for b in orbital_offsets[i]..orbital_offsets[i + 1] {
            sc_col_atom[i] += sc_col[b];
        }
    }

    // For each atom i, compute its contribution to the gradient
    for i in 0..n_atoms {
        let atomi = &atoms[i];

        for j in 0..n_atoms {
            if i == j {
                continue;
            }

            let atomj = &atoms[j];
            let diff = atomi.xyz - atomj.xyz;
            let r = diff.norm();
            let e_ij = diff / r; // unit vector from j to i

            // Get gamma derivative value
            let g1_val = gammafunction.deriv(r, atomi.number, atomj.number);

            // Case 1: a belongs to atom i, c belongs to atom j
            // dgsv contribution for nc = 3*i+dir:
            // = sum_{a in atom_i} dgsv[a] * sc_row[a]
            // = sum_{a in atom_i} (g1_val * sv_atom[j]) * sc_row[a]  (since c in atom j)
            // = g1_val * sv_atom[j] * sc_row_atom[i]
            //
            // But we also have c could be in other atoms. For atom i's gradient:
            // - When a is in atom i: dg[3i+dir, a, c] = e_ij[dir] * g1_val for c in atom j
            // - When c is in atom i (and a is not in atom i, say a in atom k):
            //   dg[3i+dir, a, c] = e_ik[dir] * g1(r_ik) for c in atom i, a in atom k

            // Contribution from dgsv[a] where a in atom_i and summing over c in atom_j:
            let contrib_a_in_i = g1_val * sv_atom[j] * sc_row_atom[i];

            // Contribution from dgsv[b] where b in atom_i and summing over c in atom_j:
            let contrib_b_in_i = g1_val * sv_atom[j] * sc_col_atom[i];

            // Contribution from dgsv[a] where c in atom_i (a not in atom_i):
            // dgsv[a] for a in atom_j = g1_val * sv_atom[i]
            // This contributes to atom i's gradient
            let contrib_c_in_i_a = g1_val * sv_atom[i] * sc_row_atom[j];

            // Contribution from dgsv[b] where c in atom_i (b not in atom_i):
            let contrib_c_in_i_b = g1_val * sv_atom[i] * sc_col_atom[j];

            let total_contrib =
                contrib_a_in_i + contrib_b_in_i + contrib_c_in_i_a + contrib_c_in_i_b;

            for dir in 0..3 {
                result[3 * i + dir] += e_ij[dir] * total_contrib;
            }
        }
    }

    result
}

/// Compute dG (gamma_lr gradient) contributions for f_lr (terms 9-12) using atomwise gamma_lr gradients.
///
/// The 4 dG-dependent terms in f_lr are:
/// Term 9:  dG * (S · sv^T)
/// Term 10: (Sv^T * dG) · S
/// Term 11: S · (Sv * dG)^T
/// Term 12: S · (S · dG·v)^T
///
/// This function computes these contributions on-the-fly using gamma_lr_func.deriv()
/// instead of loading the full grad_gamma_lr_ao array.
pub fn compute_flr_dg_contributions(
    gammafunction_lr: &GammaFunction,
    atoms: &[Atom],
    orbital_offsets: &[usize],
    s: ArrayView2<f64>,
    v: ArrayView2<f64>,
    contract_with: ArrayView2<f64>,
    n_atoms: usize,
    _n_orbs: usize,
) -> Array1<f64> {
    let mut result = Array1::zeros(3 * n_atoms);

    // Precompute matrices needed for terms 9-12
    let sv: Array2<f64> = s.dot(&v); // S · v
    let sv_t: Array2<f64> = s.dot(&v.t()); // S · v^T
    let s_t_sv: Array2<f64> = s.dot(&sv.t()); // S · (S·v)^T

    // For each term, we need to extract the coefficient for dG[mu,nu] after contraction.
    // The key insight: for nc = 3*i+dir, dG[nc, mu, nu] is non-zero only when
    // mu OR nu belongs to atom i. The value is g1_atomwise[dir, i, atom(other)].

    // Term 9: dG[a,b] * s_t_sv[a,b] contracted with c[a,b]
    // coeff9[a,b] = s_t_sv[a,b] * c[a,b]
    let coeff9: Array2<f64> = &s_t_sv * &contract_with;

    // Term 10: (sv_t[a,c] * dG[a,c]) · S[c,b] contracted with c[a,b]
    // = sv_t[a,c] * dG[a,c] * sum_b S[c,b] * c[a,b]
    // coeff10[a,c] = sv_t[a,c] * (S · c^T)[c,a]
    let s_ct: Array2<f64> = s.dot(&contract_with.t()); // [c,a]
    let coeff10: Array2<f64> = &sv_t * &s_ct.t(); // [a,c]

    // Term 11: S[a,c] * sv[b,c] * dG[b,c] contracted with c[a,b]
    // = sv[b,c] * dG[b,c] * sum_a S[a,c] * c[a,b]
    // coeff11[b,c] = sv[b,c] * (S^T · c)[c,b]
    let st_c: Array2<f64> = s.t().dot(&contract_with); // [c,b]
    let coeff11: Array2<f64> = &sv * &st_c.t(); // [b,c]

    // Term 12: S[a,c] * (S · dG·v)[b,c] contracted with c[a,b]
    // (S · dG·v)[b,c] = sum_k S[b,k] * dG[k,c] * v[k,c]
    // For dG at [k,c]:
    // coeff = sum_{a,b} S[a,c] * S[b,k] * v[k,c] * c[a,b]
    //       = v[k,c] * (sum_a S[a,c] * sum_b S[b,k] * c[a,b])
    //       = v[k,c] * (S^T)[c,a] · c[a,b] · (S^T)[b,k]
    //       = v[k,c] * (S^T · c · S^T)[c,k]
    // Let's compute S^T · c: [c,b], then · S^T doesn't work dimension-wise
    // Actually: sum_b S[b,k] * c[a,b] = (c · S^T)[a,k]
    // sum_a S[a,c] * (c · S^T)[a,k] = (S^T · (c · S^T))[c,k]
    // coeff12[k,c] = v[k,c] * (S^T · c · S^T)[c,k]
    //             = v[k,c] * ((S^T · c · S^T)^T)[k,c]
    //             = v[k,c] * (S · c^T · S)[k,c]
    let s_ct_s: Array2<f64> = s.dot(&contract_with.t()).dot(&s); // [k,c]
    let coeff12: Array2<f64> = &v * &s_ct_s; // [k,c]

    // Now compute the on-the-fly contributions using atomwise gamma_lr gradients
    // For each atom pair (i, j), compute gamma_lr derivative and accumulate

    // Precompute per-atom sums of coefficients
    // For each coefficient matrix coeff[mu,nu], when dG[nc, mu, nu] is non-zero:
    // - nc = 3*i+dir, mu in atom i, nu in atom j: dG = e_ij * g1_lr
    // - nc = 3*i+dir, mu in atom j, nu in atom i: dG = e_ij * g1_lr (symmetric)

    for i in 0..n_atoms {
        let atomi = &atoms[i];

        for j in 0..n_atoms {
            if i == j {
                continue;
            }

            let atomj = &atoms[j];
            let diff = atomi.xyz - atomj.xyz;
            let r = diff.norm();
            let e_ij = diff / r;

            let g1_lr_val = gammafunction_lr.deriv(r, atomi.number, atomj.number);

            // Sum coefficients for orbital pairs (mu in atom i, nu in atom j)
            let mut coeff_sum = 0.0;

            // Term 9: coeff9[mu, nu] for mu in atom i, nu in atom j
            for mu in orbital_offsets[i]..orbital_offsets[i + 1] {
                for nu in orbital_offsets[j]..orbital_offsets[j + 1] {
                    coeff_sum += coeff9[[mu, nu]];
                }
            }

            // Term 10: coeff10[mu, nu] for mu in atom i, nu in atom j
            for mu in orbital_offsets[i]..orbital_offsets[i + 1] {
                for nu in orbital_offsets[j]..orbital_offsets[j + 1] {
                    coeff_sum += coeff10[[mu, nu]];
                }
            }

            // Term 11: coeff11[mu, nu] for mu in atom i, nu in atom j
            for mu in orbital_offsets[i]..orbital_offsets[i + 1] {
                for nu in orbital_offsets[j]..orbital_offsets[j + 1] {
                    coeff_sum += coeff11[[mu, nu]];
                }
            }

            // Term 12: coeff12[mu, nu] for mu in atom i, nu in atom j
            for mu in orbital_offsets[i]..orbital_offsets[i + 1] {
                for nu in orbital_offsets[j]..orbital_offsets[j + 1] {
                    coeff_sum += coeff12[[mu, nu]];
                }
            }

            // Also add symmetric contributions (nu in atom i, mu in atom j)
            // because dG[nc, nu, mu] is also non-zero
            for nu in orbital_offsets[i]..orbital_offsets[i + 1] {
                for mu in orbital_offsets[j]..orbital_offsets[j + 1] {
                    coeff_sum += coeff9[[mu, nu]];
                    coeff_sum += coeff10[[mu, nu]];
                    coeff_sum += coeff11[[mu, nu]];
                    coeff_sum += coeff12[[mu, nu]];
                }
            }

            // Apply factor 0.25 and accumulate
            for dir in 0..3 {
                result[3 * i + dir] += 0.25 * e_ij[dir] * g1_lr_val * coeff_sum;
            }
        }
    }

    result
}

/// Compute coefficient matrix for f_v dS-dependent terms (on-the-fly computation).
/// The f_v formula is:
///   d_f[a,b] = 0.25 * (ds[a,b] * (gsv[a] + gsv[b]) + s[a,b] * (dgsv[a] + gdsv[a] + dgsv[b] + gdsv[b]))
///
/// This returns the coefficient for the dS contribution after contraction with `contract_with`:
///   coeff_fv_ds[a,b] = contribution to sum_{a,b} dS[a,b] * coeff[a,b] from:
///     - Term 1: ds[a,b] * (gsv[a] + gsv[b]) * contract_with[a,b]
///     - Term 2: s[a,b] * gdsv[a,b] * contract_with[a,b] (gdsv depends on ds)
///
/// The gdsv term: gdsv[a] = g0.dot(&(ds * vp).sum_axis(Axis(0)))[a]
/// After contraction with contract_with:
///   sum_{a,b} s[a,b] * contract_with[a,b] * gdsv[a]
///   = sum_{a,b} s[a,b] * contract_with[a,b] * sum_c g0[a,c] * sum_k ds[c,k] * vp[c,k]
///   = sum_{c,k} ds[c,k] * vp[c,k] * sum_a g0[a,c] * sum_b s[a,b] * contract_with[a,b]
///
/// Returns coefficient matrix where gradient contrib = sum_{mu,nu} dS[mu,nu] * coeff[mu,nu]
pub fn compute_fv_coefficients_onthefly(
    v: ArrayView2<f64>, // diff_p or x_ao
    s: ArrayView2<f64>,
    g0_ao: ArrayView2<f64>,
    contract_with: ArrayView2<f64>,
) -> Array2<f64> {
    let n_orb = s.nrows();

    // vp = v + v^T (symmetrized)
    let vp: Array2<f64> = &v + &v.t();
    // sv = (s * vp).sum(axis=0) = sum over rows for each column
    let sv: Array1<f64> = (&s * &vp).sum_axis(Axis(0));
    // gsv = g0_ao.dot(&sv)
    let gsv: Array1<f64> = g0_ao.dot(&sv);

    let mut coeff_fv: Array2<f64> = Array2::zeros((n_orb, n_orb));

    // Term 1: ds[a,b] * (gsv[a] + gsv[b]) * contract_with[a,b]
    // coeff[a,b] = 0.25 * (gsv[a] + gsv[b]) * contract_with[a,b]
    for a in 0..n_orb {
        for b in 0..n_orb {
            coeff_fv[[a, b]] = 0.25 * (gsv[a] + gsv[b]) * contract_with[[a, b]];
        }
    }

    // Term 2 (gdsv contribution):
    // gdsv[a] = sum_c g0[a,c] * (sum_k ds[c,k] * vp[c,k])
    // After contraction: sum_{a,b} s[a,b] * contract_with[a,b] * (gdsv[a] + gdsv[b])
    //
    // For gdsv[a] term:
    //   sum_{a,b,c,k} s[a,b] * contract_with[a,b] * g0[a,c] * ds[c,k] * vp[c,k]
    //   = sum_{c,k} ds[c,k] * vp[c,k] * sum_a g0[a,c] * sum_b s[a,b] * contract_with[a,b]
    //   = sum_{c,k} ds[c,k] * vp[c,k] * sum_a g0[a,c] * (s*contract)[a] where (s*contract)[a] = sum_b s[a,b]*c[a,b]
    //
    // Let sc = s * contract_with (element-wise), sc_row_sum[a] = sum_b sc[a,b]
    // Weight from gdsv[a]: sum_a g0[a,c] * sc_row_sum[a] = (g0^T · sc_row_sum)[c]
    //
    // For gdsv[b] term:
    //   sum_{a,b,c,k} s[a,b] * contract_with[a,b] * g0[b,c] * ds[c,k] * vp[c,k]
    //   = sum_{c,k} ds[c,k] * vp[c,k] * sum_b g0[b,c] * sum_a s[a,b] * contract_with[a,b]
    //   = sum_{c,k} ds[c,k] * vp[c,k] * (g0^T · sc_col_sum)[c]
    //   where sc_col_sum[b] = sum_a sc[a,b]

    let sc: Array2<f64> = &s * &contract_with;
    let sc_row_sum: Array1<f64> = sc.sum_axis(Axis(1)); // sum_b sc[a,b]
    let sc_col_sum: Array1<f64> = sc.sum_axis(Axis(0)); // sum_a sc[a,b]

    let weight_from_a: Array1<f64> = g0_ao.t().dot(&sc_row_sum); // [c]
    let weight_from_b: Array1<f64> = g0_ao.t().dot(&sc_col_sum); // [c]
    let weight: Array1<f64> = &weight_from_a + &weight_from_b;

    // coeff_gdsv[c,k] = 0.25 * weight[c] * vp[c,k]
    for c in 0..n_orb {
        for k in 0..n_orb {
            coeff_fv[[c, k]] += 0.25 * weight[c] * vp[[c, k]];
        }
    }

    // Symmetrize for on-the-fly computation where we process each pair once
    let coeff_fv_sym = &coeff_fv + &coeff_fv.t();

    coeff_fv_sym
}

/// Optimized Z-vector solver that reuses workspace across iterations
/// Uses direct BLAS DGEMM calls for in-place operations to minimize allocations.
pub fn zvector_lc_optimized(
    a_diag: ArrayView2<f64>,
    r_matrix: ArrayView2<f64>,
    g0: ArrayView2<f64>,
    g0_lr: ArrayView2<f64>,
    qtrans_oo: ArrayView3<f64>,
    qtrans_vv: ArrayView3<f64>,
    qtrans_ov: ArrayView3<f64>,
) -> Array2<f64> {
    let maxiter: usize = 10000;
    // Relaxed convergence threshold (1e-12 vs original 1e-16)
    // Reduces iterations from ~35 to ~24 on large systems while maintaining sufficient accuracy
    let conv: f64 = 1.0e-12;

    // Create workspace with precomputed reshaped arrays and BLAS buffers
    let mut ws = ZVectorWorkspace::new(qtrans_oo, qtrans_vv, qtrans_ov);

    // Set gamma matrices for BLAS operations (done once)
    ws.set_gamma(g0, g0_lr);

    // Initial setup
    let a_inv: Array2<f64> = 1.0 / &a_diag.to_owned();
    let bs: Array2<f64> = &a_inv * &r_matrix;
    let rhs: Array1<f64> = r_matrix.into_shape(ws.kmax).unwrap().to_owned();

    // Initial mult_apb_v_blas call
    let apbv: Array2<f64> = mult_apb_v_blas(a_diag, bs.view(), &mut ws);

    let rkm1 = apbv.into_shape(ws.kmax).unwrap();
    let mut rhs_2 = bs.into_shape(ws.kmax).unwrap();
    let mut rkm1 = rhs - rkm1;
    let mut pkm1 = rkm1.clone();

    for _it in 0..maxiter {
        let apbv: Array2<f64> = mult_apb_v_blas(
            a_diag,
            pkm1.view().into_shape((ws.n_occ, ws.n_virt)).unwrap(),
            &mut ws,
        );
        let apk: Array1<f64> = apbv.into_shape(ws.kmax).unwrap();

        let tmp1: f64 = rkm1.dot(&rkm1);
        let tmp2: f64 = pkm1.dot(&apk);

        rhs_2 = rhs_2 + (tmp1 / tmp2) * &pkm1;
        rkm1 = rkm1 - (tmp1 / tmp2) * &apk;

        let tmp2: f64 = rkm1.dot(&rkm1);

        if tmp2 <= conv {
            break;
        }
        pkm1 = (tmp2 / tmp1) * &pkm1 + &rkm1;
    }

    let out: Array2<f64> = rhs_2.into_shape((ws.n_occ, ws.n_virt)).unwrap();
    out
}

// Workspace for Z-vector CG iterations to avoid repeated allocations
pub struct ZVectorWorkspace {
    // Dimensions
    pub n_occ: usize,
    pub n_virt: usize,
    pub n_at: usize,
    pub kmax: usize,
    // Preallocated reshaped qtrans arrays (computed once)
    pub tmp_q_oo: Array2<f64>,
    pub tmp_q_vv: Array2<f64>,
    pub tmp_q_ov_shape_1: Array2<f64>,
    pub tmp_q_ov_shape_2: Array2<f64>,
    // Precomputed transposed arrays for efficient BLAS (contiguous memory)
    pub tmp_q_oo_t: Array2<f64>, // transpose of tmp_q_oo

    // ========== BLAS workspace buffers for mult_apb_v ==========
    // These are reused across all CG iterations to avoid allocations

    // Buffers for Coulomb (Term 2)
    pub buf_coulomb_tmp: Vec<f64>, // [n_at] - gamma · qtrans_ov · vs
    pub buf_coulomb_out: Vec<f64>, // [n_occ * n_virt] - final coulomb term

    // Buffers for Exchange Term 3
    pub buf_t3_step1: Vec<f64>, // [n_at * n_virt * n_occ] - intermediate result
    pub buf_t3_step2: Vec<f64>, // [n_at * n_virt * n_occ] - gamma_lr · step1 (only used when precomputation disabled)
    pub buf_t3_swapped: Vec<f64>, // [n_at * n_occ * n_virt] - swapped axes buffer
    pub buf_t3_out: Vec<f64>,   // [n_occ * n_virt] - final term3 result

    // Buffers for Exchange Term 4
    pub buf_t4_step1: Vec<f64>, // [n_at * n_virt * n_virt] - intermediate result
    pub buf_t4_step2: Vec<f64>, // [n_at * n_virt * n_virt] - gamma_lr · step1 (only used when precomputation disabled)
    pub buf_t4_swapped: Vec<f64>, // [n_at * n_virt * n_virt] - swapped axes buffer
    pub buf_t4_out: Vec<f64>,   // [n_occ * n_virt] - final term4 result

    // Flat views of input matrices for BLAS
    pub gamma_flat: Vec<f64>,       // [n_at * n_at] - gamma matrix
    pub gamma_lr_flat: Vec<f64>,    // [n_at * n_at] - gamma_lr matrix
    pub qtrans_ov_flat: Vec<f64>,   // [n_at * n_occ * n_virt] - qtrans_ov
    pub qtrans_vv_flat: Vec<f64>,   // [n_at * n_virt * n_virt] - qtrans_vv reshaped
    pub qtrans_ov_1_flat: Vec<f64>, // [n_at * n_virt * n_occ] - tmp_q_ov_shape_1
    pub qtrans_ov_2_flat: Vec<f64>, // [n_occ * n_at * n_virt] - tmp_q_ov_shape_2
    pub qtrans_oo_t_flat: Vec<f64>, // [n_occ * n_at * n_occ] - tmp_q_oo_t

    // ========== Precomputed gamma_lr · qtrans products (optional) ==========
    // These eliminate one DGEMM per term per mult_apb_v call
    // Memory cost: ~1GB for 324 atoms, but saves ~30% of exchange computation
    // Set use_precomputed_gamma=false to disable and save memory
    pub use_precomputed_gamma: bool,
    pub gamma_lr_qvv_flat: Vec<f64>, // [n_at * n_virt * n_virt] - gamma_lr · qtrans_vv
    pub gamma_lr_qov_1_flat: Vec<f64>, // [n_at * n_virt * n_occ] - gamma_lr · qtrans_ov_1
}

impl ZVectorWorkspace {
    /// Create a new workspace with precomputed gamma products enabled (default, faster but ~1GB more memory)
    pub fn new(
        qtrans_oo: ArrayView3<f64>,
        qtrans_vv: ArrayView3<f64>,
        qtrans_ov: ArrayView3<f64>,
    ) -> Self {
        Self::with_options(qtrans_oo, qtrans_vv, qtrans_ov, true)
    }

    /// Create a new workspace with configurable precomputation
    ///
    /// # Arguments
    /// * `use_precomputed_gamma` - If true, precomputes the gamma_lr · qtrans
    ///   products at the cost of additional memory. Set to false for
    ///   memory-constrained systems.
    pub fn with_options(
        qtrans_oo: ArrayView3<f64>,
        qtrans_vv: ArrayView3<f64>,
        qtrans_ov: ArrayView3<f64>,
        use_precomputed_gamma: bool,
    ) -> Self {
        let n_at = qtrans_ov.dim().0;
        let n_occ = qtrans_ov.dim().1;
        let n_virt = qtrans_ov.dim().2;
        let kmax = n_occ * n_virt;

        // Calculate memory usage for precomputation
        // Precompute reshaped qtrans arrays
        let tmp_q_vv: Array2<f64> = qtrans_vv
            .to_owned()
            .into_shape((n_virt * n_at, n_virt))
            .unwrap();
        let tmp_q_oo: Array2<f64> = qtrans_oo
            .to_owned()
            .into_shape((n_at * n_occ, n_occ))
            .unwrap();

        // Precompute contiguous transpose of tmp_q_oo for efficient BLAS in Term 3d
        let tmp_q_oo_t: Array2<f64> = tmp_q_oo.t().to_owned();

        let tmp_q_ov_swapped: Array3<f64> = qtrans_ov
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();
        let tmp_q_ov_shape_1: Array2<f64> =
            tmp_q_ov_swapped.into_shape((n_at * n_virt, n_occ)).unwrap();

        let tmp_q_ov_swapped_2: Array3<f64> = qtrans_ov
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .to_owned();
        let tmp_q_ov_shape_2: Array2<f64> = tmp_q_ov_swapped_2
            .into_shape((n_occ, n_at * n_virt))
            .unwrap();

        // Allocate BLAS workspace buffers
        let buf_coulomb_tmp = vec![0.0; n_at];
        let buf_coulomb_out = vec![0.0; n_occ * n_virt];

        // Term 3 buffers
        let buf_t3_step1 = vec![0.0; n_at * n_virt * n_occ];
        // step2 only needed when precomputation is disabled
        let buf_t3_step2 = if use_precomputed_gamma {
            Vec::new()
        } else {
            vec![0.0; n_at * n_virt * n_occ]
        };
        let buf_t3_swapped = vec![0.0; n_at * n_occ * n_virt];
        let buf_t3_out = vec![0.0; n_occ * n_virt];

        // Term 4 buffers
        let buf_t4_step1 = vec![0.0; n_at * n_virt * n_virt];
        // step2 only needed when precomputation is disabled
        let buf_t4_step2 = if use_precomputed_gamma {
            Vec::new()
        } else {
            vec![0.0; n_at * n_virt * n_virt]
        };
        let buf_t4_swapped = vec![0.0; n_at * n_virt * n_virt];
        let buf_t4_out = vec![0.0; n_occ * n_virt];

        // Flat views of qtrans arrays for BLAS
        // These need to be converted to contiguous row-major layout for BLAS
        let qtrans_ov_flat: Vec<f64> = if let Some(slice) = qtrans_ov.as_slice() {
            slice.to_vec()
        } else {
            qtrans_ov.as_standard_layout().iter().cloned().collect()
        };

        let qtrans_vv_flat: Vec<f64> = if let Some(slice) = tmp_q_vv.as_slice() {
            slice.to_vec()
        } else {
            tmp_q_vv.as_standard_layout().iter().cloned().collect()
        };

        let qtrans_ov_1_flat: Vec<f64> = if let Some(slice) = tmp_q_ov_shape_1.as_slice() {
            slice.to_vec()
        } else {
            tmp_q_ov_shape_1
                .as_standard_layout()
                .iter()
                .cloned()
                .collect()
        };

        let qtrans_ov_2_flat: Vec<f64> = if let Some(slice) = tmp_q_ov_shape_2.as_slice() {
            slice.to_vec()
        } else {
            tmp_q_ov_shape_2
                .as_standard_layout()
                .iter()
                .cloned()
                .collect()
        };

        let qtrans_oo_t_flat: Vec<f64> = if let Some(slice) = tmp_q_oo_t.as_slice() {
            slice.to_vec()
        } else {
            tmp_q_oo_t.as_standard_layout().iter().cloned().collect()
        };

        // Gamma matrices will be set later via set_gamma
        let gamma_flat = vec![0.0; n_at * n_at];
        let gamma_lr_flat = vec![0.0; n_at * n_at];

        // Precomputed gamma_lr · qtrans products (only allocated if enabled)
        // These trade memory for speed by eliminating DGEMM calls per iteration
        let gamma_lr_qvv_flat = if use_precomputed_gamma {
            vec![0.0; n_at * n_virt * n_virt]
        } else {
            Vec::new()
        };
        let gamma_lr_qov_1_flat = if use_precomputed_gamma {
            vec![0.0; n_at * n_virt * n_occ]
        } else {
            Vec::new()
        };

        ZVectorWorkspace {
            n_occ,
            n_virt,
            n_at,
            kmax,
            tmp_q_oo,
            tmp_q_vv,
            tmp_q_ov_shape_1,
            tmp_q_ov_shape_2,
            tmp_q_oo_t,
            buf_coulomb_tmp,
            buf_coulomb_out,
            buf_t3_step1,
            buf_t3_step2,
            buf_t3_swapped,
            buf_t3_out,
            buf_t4_step1,
            buf_t4_step2,
            buf_t4_swapped,
            buf_t4_out,
            gamma_flat,
            gamma_lr_flat,
            qtrans_ov_flat,
            qtrans_vv_flat,
            qtrans_ov_1_flat,
            qtrans_ov_2_flat,
            qtrans_oo_t_flat,
            use_precomputed_gamma,
            gamma_lr_qvv_flat,
            gamma_lr_qov_1_flat,
        }
    }

    /// Set gamma matrices and optionally precompute gamma_lr products for BLAS operations.
    /// Call once before CG iterations. If precomputation is enabled, this precomputes:
    /// - gamma_lr_qvv = gamma_lr · qtrans_vv (eliminates 1 DGEMM per Term 3)
    /// - gamma_lr_qov_1 = gamma_lr · qtrans_ov_1 (eliminates 1 DGEMM per Term 4)
    pub fn set_gamma(&mut self, gamma: ArrayView2<f64>, gamma_lr: ArrayView2<f64>) {
        use crate::linalg::dgemm::dgemm_row_major;

        let n_at = self.n_at;
        let n_virt = self.n_virt;
        let n_occ = self.n_occ;

        // Copy gamma to flat buffer
        if let Some(slice) = gamma.as_slice() {
            self.gamma_flat.copy_from_slice(slice);
        } else {
            let gamma_std = gamma.as_standard_layout();
            self.gamma_flat
                .copy_from_slice(gamma_std.as_slice().unwrap());
        }

        // Copy gamma_lr to flat buffer
        if let Some(slice) = gamma_lr.as_slice() {
            self.gamma_lr_flat.copy_from_slice(slice);
        } else {
            let gamma_lr_std = gamma_lr.as_standard_layout();
            self.gamma_lr_flat
                .copy_from_slice(gamma_lr_std.as_slice().unwrap());
        }

        // Only precompute gamma_lr products if enabled
        if self.use_precomputed_gamma {
            // Precompute gamma_lr · qtrans_vv for Term 3
            // gamma_lr: [n_at, n_at]
            // qtrans_vv_flat: stored as [n_at*n_virt, n_virt], reinterpret as [n_at, n_virt*n_virt]
            // Result: [n_at, n_virt*n_virt]
            // This eliminates one DGEMM per mult_apb_v call in Term 3
            unsafe {
                dgemm_row_major(
                    1.0,
                    &self.gamma_lr_flat,
                    n_at,
                    n_at,
                    &self.qtrans_vv_flat,
                    n_virt * n_virt,
                    0.0,
                    &mut self.gamma_lr_qvv_flat,
                );
            }

            // Precompute gamma_lr · qtrans_ov_1 for Term 4
            // gamma_lr: [n_at, n_at]
            // qtrans_ov_1_flat: stored as [n_at*n_virt, n_occ], reinterpret as [n_at, n_virt*n_occ]
            // Result: [n_at, n_virt*n_occ]
            // This eliminates one DGEMM per mult_apb_v call in Term 4
            unsafe {
                dgemm_row_major(
                    1.0,
                    &self.gamma_lr_flat,
                    n_at,
                    n_at,
                    &self.qtrans_ov_1_flat,
                    n_virt * n_occ,
                    0.0,
                    &mut self.gamma_lr_qov_1_flat,
                );
            }
        }
    }
}

/// Compute coefficient matrix for f_lr dS-dependent terms (8 of 12 terms).
/// This is the on-the-fly version that returns only the dS coefficient matrix.
///
/// The 8 dS-dependent terms in f_lr are (from f_lr_par):
/// Term 1: g * (dS · sv^T)
/// Term 2: (dS·v^T * g) · S
/// Term 3: dS · (Sv*g)^T
/// Term 4: dS · (S·gv)^T
/// Term 5: g * (S · (dS·v)^T)
/// Term 6: (Sv^T * g) · dS^T
/// Term 7: S · (dS·v * g)^T
/// Term 8: S · (dS · gv)^T
///
/// Returns coefficient matrix where gradient contrib = sum_{mu,nu} dS[mu,nu] * coeff[mu,nu]
pub fn compute_flr_s_coefficients(
    v: ArrayView2<f64>, // diff_p or x_ao.t()
    s: ArrayView2<f64>,
    gamma_lr_ao: ArrayView2<f64>,
    contract_with: ArrayView2<f64>,
) -> Array2<f64> {
    let n_orb = s.nrows();

    // Precompute auxiliary matrices (same as in f_lr_par)
    let sv: Array2<f64> = s.dot(&v); // S · v
    let v_t = v.t(); // v^T
    let sv_t: Array2<f64> = s.dot(&v_t); // S · v^T
    let gv: Array2<f64> = &gamma_lr_ao * &v; // γ * v (element-wise)
                                             // let t_sv = sv.t(); // (S·v)^T = v^T·S^T
    let svg_t: Array2<f64> = (&sv * &gamma_lr_ao).t().to_owned(); // (S·v * γ)^T
    let sgv_t: Array2<f64> = s.dot(&gv).t().to_owned(); // (S·(γ*v))^T
                                                        // let s_t_sv: Array2<f64> = s.dot(&t_sv); // S · (S·v)^T

    let mut coeff_s: Array2<f64> = Array2::zeros((n_orb, n_orb));

    // For efficiency, precompute products with contract_with
    // let gv_elem: Array2<f64> = &gamma_lr_ao * &v; // element-wise g*v
    let sv_t_g: Array2<f64> = &sv_t * &gamma_lr_ao;

    // coeff1[a,k] from Term 1: g[a,b] * dS[a,k] * sv^T[k,b] contracted with c[a,b]
    // = dS[a,k] * sum_b g[a,b] * sv^T[k,b] * c[a,b]
    // = dS[a,k] * sum_b (g*c)[a,b] * sv[b,k]
    // coeff1[a,k] = ((g*c) · sv)[a,k]
    let gc: Array2<f64> = &gamma_lr_ao * &contract_with;
    let coeff1: Array2<f64> = gc.dot(&sv);

    // coeff2[a,k] from Term 2: (dS[a,k]*v^T[k,c]*g[a,c])·S[c,b] contracted with c[a,b]
    // = dS[a,k] * sum_c v^T[k,c] * g[a,c] * sum_b S[c,b] * c[a,b]
    // = dS[a,k] * sum_c v[c,k] * g[a,c] * (S·c^T)[c,a]
    // Let Sc_T = S · contract_with^T: [c,a]
    // coeff2[a,k] = sum_c v[c,k] * g[a,c] * Sc_T[c,a]
    //             = sum_c v[c,k] * (g .* Sc_T^T)[a,c]
    // where Sc_T^T[a,c] = Sc_T[c,a]
    let s_ct: Array2<f64> = s.dot(&contract_with.t()); // [n,a]
    let g_sct_t: Array2<f64> = &gamma_lr_ao * &s_ct.t(); // [a,c]
    let coeff2: Array2<f64> = g_sct_t.dot(&v); // [a,k]

    // coeff3[a,k] from Term 3: dS[a,k] * svg_t[k,b] contracted with c[a,b]
    // = dS[a,k] * sum_b svg_t[k,b] * c[a,b]
    // = dS[a,k] * (svg_t · c^T)[k,a]
    // coeff3[a,k] = (c · svg_t^T)[a,k]
    let coeff3: Array2<f64> = contract_with.dot(&svg_t.t());

    // coeff4[a,k] from Term 4: dS[a,k] * sgv_t[k,b] contracted with c[a,b]
    // coeff4[a,k] = (c · sgv_t^T)[a,k]
    let coeff4: Array2<f64> = contract_with.dot(&sgv_t.t());

    // Terms 5-8 involve dS at different positions (like dS[b,k] or dS^T)

    // coeff5 from Term 5: g[a,b] * (S · (dS·v)^T)[a,b] contracted with c[a,b]
    // = g[a,b] * S[a,c] * (dS·v)^T[c,b] * c[a,b]
    // = g[a,b] * S[a,c] * (dS·v)[b,c] * c[a,b]
    // = g[a,b] * S[a,c] * sum_k dS[b,k] * v[k,c] * c[a,b]
    // = sum_k dS[b,k] * sum_c v[k,c] * sum_a S[a,c] * g[a,b] * c[a,b]
    // = sum_k dS[b,k] * sum_c v[k,c] * (S^T · (g*c))[c,b]
    // Let St_gc = S^T · (g*c): [c,b]
    // coeff5[b,k] = sum_c v[k,c] * St_gc[c,b] = (v · St_gc)[k,b]^T = (St_gc^T · v^T)[b,k]
    let st_gc: Array2<f64> = s.t().dot(&gc); // [c,b]
    let coeff5: Array2<f64> = st_gc.t().dot(&v_t); // [b,k]

    // coeff6 from Term 6: (Sv^T[a,c] * g[a,c]) · dS^T[c,b] contracted with c[a,b]
    // = (sv_t * g)[a,c] * dS[b,c] * c[a,b]
    // = sum_{a,b} (sv_t * g)[a,c] * c[a,b] * dS[b,c]
    // coeff6[b,c] = sum_a (sv_t * g)[a,c] * c[a,b] = ((sv_t*g)^T · c^T)[c,b]
    // So coeff6_bc = (c · (sv_t*g))^T
    // let coeff6_cb: Array2<f64> = contract_with.dot(&sv_t_g); // [a,b] · [a,c] doesn't work
    // Actually: coeff6[b,c] = sum_a (sv_t*g)[a,c] * c[a,b]
    //                       = ((sv_t*g)^T · c^T)[c,b]^T = (c · (sv_t*g))^T ... dimension mismatch
    // Let's redo: (sv_t*g)[a,c] * c[a,b]
    // sum_a (sv_t*g)[a,c] * c[a,b] = ((sv_t*g)^T)[c,a] · c[a,b] = ((sv_t*g)^T · c)[c,b]
    let coeff6: Array2<f64> = sv_t_g.t().dot(&contract_with).t().to_owned(); // [c,b]^T = [b,c]

    // coeff7 from Term 7: S[a,c] * (dS·v * g)^T[c,b] contracted with c[a,b]
    // = S[a,c] * (dS·v)[b,c] * g[b,c] * c[a,b]
    // = S[a,c] * sum_k dS[b,k] * v[k,c] * g[b,c] * c[a,b]
    // = sum_k dS[b,k] * sum_c v[k,c] * g[b,c] * sum_a S[a,c] * c[a,b]
    // = sum_k dS[b,k] * sum_c v[k,c] * g[b,c] * (S^T · c)[c,b]
    // Let st_c = S^T · contract_with: [c,b]
    // coeff7[b,k] = sum_c v[k,c] * g[b,c] * st_c[c,b]
    //             = sum_c v[k,c] * (g * st_c^T)[b,c]
    // where (g * st_c^T)[b,c] = g[b,c] * st_c^T[b,c] = g[b,c] * st_c[c,b]
    let st_c: Array2<f64> = s.t().dot(&contract_with); // [c,b]
    let g_stc_t: Array2<f64> = &gamma_lr_ao * &st_c.t(); // [b,c]
    let coeff7: Array2<f64> = g_stc_t.dot(&v_t); // [b,c] · [c,k] = [b,k]

    // coeff8 from Term 8: S[a,c] * (dS · gv)^T[c,b] contracted with c[a,b]
    // = S[a,c] * (dS · gv)[b,c] * c[a,b]
    // = S[a,c] * sum_k dS[b,k] * gv[k,c] * c[a,b]
    // = sum_k dS[b,k] * sum_c gv[k,c] * sum_a S[a,c] * c[a,b]
    // = sum_k dS[b,k] * sum_c gv[k,c] * st_c[c,b]
    // = sum_k dS[b,k] * (gv · st_c)[k,b]
    // coeff8[b,k] = (gv · st_c)^T[b,k] = (st_c^T · gv^T)[b,k]
    let coeff8: Array2<f64> = st_c.t().dot(&gv.t()); // [b,k]

    // Combine all coefficients
    // Terms 1-4 are indexed at [a,k], terms 5-8 at [b,k]
    // Since we need coeff[mu,nu], terms 1-4 go to [a,k] and terms 5-8 go to [b,k]
    // But in our final matrix, all should be combined properly
    coeff_s = &coeff_s + &coeff1 + &coeff2 + &coeff3 + &coeff4;
    coeff_s = &coeff_s + &coeff5 + &coeff6 + &coeff7 + &coeff8;

    // Apply the 0.25 factor
    coeff_s *= 0.25;

    // Symmetrize for on-the-fly computation
    let coeff_s_sym = &coeff_s + &coeff_s.t();

    coeff_s_sym
}

pub fn get_outer_product(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>) -> Array2<f64> {
    let mut matrix: Array2<f64> = Array::zeros((v1.len(), v2.len()));
    for (i, i_value) in v1.outer_iter().enumerate() {
        for (j, j_value) in v2.outer_iter().enumerate() {
            matrix[[i, j]] = (&i_value * &j_value).into_scalar();
        }
    }
    matrix
}

pub fn f_v(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb: usize,
) -> Array3<f64> {
    let vp: Array2<f64> = &v + &(v.t());
    let sv: Array1<f64> = (&s * &vp).sum_axis(Axis(0));
    let gsv: Array1<f64> = g0_ao.dot(&sv);

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb, n_orb));

    for nc in 0..3 * n_atoms {
        let ds: ArrayView2<f64> = grad_s.slice(s![nc, .., ..]);
        let dg: ArrayView2<f64> = g1_ao.slice(s![nc, .., ..]);

        let gdsv: Array1<f64> = g0_ao.dot(&(&ds * &vp).sum_axis(Axis(0)));
        let dgsv: Array1<f64> = dg.dot(&sv);
        let mut d_f: Array2<f64> = Array2::zeros((n_orb, n_orb));

        for b in 0..n_orb {
            for a in 0..n_orb {
                d_f[[a, b]] = ds[[a, b]] * (gsv[a] + gsv[b])
                    + s[[a, b]] * (dgsv[a] + gdsv[a] + dgsv[b] + gdsv[b]);
            }
        }
        d_f *= 0.25;
        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }

    f_return
}

pub fn f_v_par(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb: usize,
) -> Array3<f64> {
    let vp: Array2<f64> = &v + &(v.t());
    let sv: Array1<f64> = (&s * &vp).sum_axis(Axis(0));
    let gsv: Array1<f64> = g0_ao.dot(&sv);

    let mut f_return: Vec<_> = (0..3 * n_atoms)
        .into_par_iter()
        .map(|nc| {
            let ds: ArrayView2<f64> = grad_s.slice(s![nc, .., ..]);
            let dg: ArrayView2<f64> = g1_ao.slice(s![nc, .., ..]);

            let gdsv: Array1<f64> = g0_ao.dot(&(&ds * &vp).sum_axis(Axis(0)));
            let dgsv: Array1<f64> = dg.dot(&sv);

            let mut d_f: Vec<f64> = Vec::new();

            for b in 0..n_orb {
                for a in 0..n_orb {
                    d_f.push(
                        ds[[a, b]] * (gsv[a] + gsv[b])
                            + s[[a, b]] * (dgsv[a] + gdsv[a] + dgsv[b] + gdsv[b]),
                    );
                }
            }
            (Array::from(d_f) * 0.25).to_vec()
        })
        .collect();
    let mut f_result: Vec<f64> = Vec::new();

    for vec in f_return.iter_mut() {
        f_result.append(&mut *vec);
    }
    let f_result_temp: Array1<f64> = Array::from(f_result);
    let f_return: Array3<f64> = f_result_temp
        .into_shape((3 * n_atoms, n_orb, n_orb))
        .unwrap();

    f_return
}

pub fn f_lr(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_lr_a0: ArrayView2<f64>,
    g1_lr_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb: usize,
) -> Array3<f64> {
    let sv: Array2<f64> = s.dot(&v);
    let v_t: ArrayView2<f64> = v.t();
    let sv_t: Array2<f64> = s.dot(&v_t);
    let gv: Array2<f64> = &g0_lr_a0 * &v;

    let t_sv: ArrayView2<f64> = sv.t();
    let svg_t: Array2<f64> = (&sv * &g0_lr_a0).reversed_axes();
    let sgv_t: Array2<f64> = s.dot(&gv).reversed_axes();

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb, n_orb));

    for nc in 0..3 * n_atoms {
        let d_s: ArrayView2<f64> = grad_s.slice(s![nc, .., ..]);
        let d_g: ArrayView2<f64> = g1_lr_ao.slice(s![nc, .., ..]);

        let d_sv_t: Array2<f64> = d_s.dot(&v_t);
        let d_sv: Array2<f64> = d_s.dot(&v);
        let d_gv: Array2<f64> = &d_g * &v;

        let mut d_f: Array2<f64> = Array2::zeros((n_orb, n_orb));
        // 1st term
        d_f = d_f + &g0_lr_a0 * &(d_s.dot(&t_sv));
        // 2nd term
        d_f = d_f + (&d_sv_t * &g0_lr_a0).dot(&s);
        // 3rd term
        d_f = d_f + d_s.dot(&svg_t);
        // 4th term
        d_f = d_f + d_s.dot(&sgv_t);
        // 5th term
        d_f = d_f + &g0_lr_a0 * &(s.dot(&d_sv.t()));
        // 6th term
        d_f = d_f + (&sv_t * &g0_lr_a0).dot(&d_s.t());
        // 7th term
        d_f = d_f + s.dot(&(&d_sv * &g0_lr_a0).t());
        // 8th term
        d_f = d_f + s.dot(&(d_s.dot(&gv)).t());
        // 9th term
        d_f = d_f + &d_g * &(s.dot(&t_sv));
        // 10th term
        d_f = d_f + (&sv_t * &d_g).dot(&s);
        // 11th term
        d_f = d_f + s.dot(&(&sv * &d_g).t());
        // 12th term
        d_f = d_f + s.dot(&(s.dot(&d_gv)).t());
        d_f *= 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }
    f_return
}

pub fn f_lr_atom_specific(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_lr_a0: ArrayView2<f64>,
    g1_lr_ao: ArrayView3<f64>,
    n_orb: usize,
) -> Array3<f64> {
    let sv: Array2<f64> = s.dot(&v);
    let v_t: ArrayView2<f64> = v.t();
    let sv_t: Array2<f64> = s.dot(&v_t);
    let gv: Array2<f64> = &g0_lr_a0 * &v;

    let t_sv: ArrayView2<f64> = sv.t();
    let svg_t: Array2<f64> = (&sv * &g0_lr_a0).reversed_axes();
    let sgv_t: Array2<f64> = s.dot(&gv).reversed_axes();

    let mut f_return: Array3<f64> = Array3::zeros((3, n_orb, n_orb));

    for nc in 0..3 {
        let d_s: ArrayView2<f64> = grad_s.slice(s![nc, .., ..]);
        let d_g: ArrayView2<f64> = g1_lr_ao.slice(s![nc, .., ..]);

        let d_sv_t: Array2<f64> = d_s.dot(&v_t);
        let d_sv: Array2<f64> = d_s.dot(&v);
        let d_gv: Array2<f64> = &d_g * &v;

        let mut d_f: Array2<f64> = Array2::zeros((n_orb, n_orb));
        // 1st term
        d_f = d_f + &g0_lr_a0 * &(d_s.dot(&t_sv));
        // 2nd term
        d_f = d_f + (&d_sv_t * &g0_lr_a0).dot(&s);
        // 3rd term
        d_f = d_f + d_s.dot(&svg_t);
        // 4th term
        d_f = d_f + d_s.dot(&sgv_t);
        // 5th term
        d_f = d_f + &g0_lr_a0 * &(s.dot(&d_sv.t()));
        // 6th term
        d_f = d_f + (&sv_t * &g0_lr_a0).dot(&d_s.t());
        // 7th term
        d_f = d_f + s.dot(&(&d_sv * &g0_lr_a0).t());
        // 8th term
        d_f = d_f + s.dot(&(d_s.dot(&gv)).t());
        // 9th term
        d_f = d_f + &d_g * &(s.dot(&t_sv));
        // 10th term
        d_f = d_f + (&sv_t * &d_g).dot(&s);
        // 11th term
        d_f = d_f + s.dot(&(&sv * &d_g).t());
        // 12th term
        d_f = d_f + s.dot(&(s.dot(&d_gv)).t());
        d_f *= 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }
    f_return
}

pub fn f_lr_par(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_lr_a0: ArrayView2<f64>,
    g1_lr_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb: usize,
) -> Array3<f64> {
    let sv: Array2<f64> = s.dot(&v);
    let v_t: ArrayView2<f64> = v.t();
    let sv_t: Array2<f64> = s.dot(&v_t);
    let gv: Array2<f64> = &g0_lr_a0 * &v;

    let t_sv: ArrayView2<f64> = sv.t();
    let svg_t: Array2<f64> = (&sv * &g0_lr_a0).reversed_axes();
    let sgv_t: Array2<f64> = s.dot(&gv).reversed_axes();

    let mut f_return: Vec<_> = (0..3 * n_atoms)
        .into_par_iter()
        .map(|nc| {
            let d_s: ArrayView2<f64> = grad_s.slice(s![nc, .., ..]);
            let d_g: ArrayView2<f64> = g1_lr_ao.slice(s![nc, .., ..]);

            let d_sv_t: Array2<f64> = d_s.dot(&v_t);
            let d_sv: Array2<f64> = d_s.dot(&v);
            let d_gv: Array2<f64> = &d_g * &v;

            let mut d_f: Array2<f64> = Array2::zeros((n_orb, n_orb));
            // 1st term
            d_f = d_f + &g0_lr_a0 * &(d_s.dot(&t_sv));
            // 2nd term
            d_f = d_f + (&d_sv_t * &g0_lr_a0).dot(&s);
            // 3rd term
            d_f = d_f + d_s.dot(&svg_t);
            // 4th term
            d_f = d_f + d_s.dot(&sgv_t);
            // 5th term
            d_f = d_f + &g0_lr_a0 * &(s.dot(&d_sv.t()));
            // 6th term
            d_f = d_f + (&sv_t * &g0_lr_a0).dot(&d_s.t());
            // 7th term
            d_f = d_f + s.dot(&(&d_sv * &g0_lr_a0).t());
            // 8th term
            d_f = d_f + s.dot(&(d_s.dot(&gv)).t());
            // 9th term
            d_f = d_f + &d_g * &(s.dot(&t_sv));
            // 10th term
            d_f = d_f + (&sv_t * &d_g).dot(&s);
            // 11th term
            d_f = d_f + s.dot(&(&sv * &d_g).t());
            // 12th term
            d_f = d_f + s.dot(&(s.dot(&d_gv)).t());
            d_f *= 0.25;

            d_f.into_shape(n_orb * n_orb).unwrap().to_vec()
        })
        .collect();
    let mut f_result: Vec<f64> = Vec::new();

    for vec in f_return.iter_mut() {
        f_result.append(&mut *vec);
    }

    let f_result_temp: Array1<f64> = Array::from(f_result);
    let f_return: Array3<f64> = f_result_temp
        .into_shape((3 * n_atoms, n_orb, n_orb))
        .unwrap();

    f_return
}

pub fn h_minus(
    g0_lr: ArrayView2<f64>,
    q_ps: ArrayView3<f64>,
    q_qr: ArrayView3<f64>,
    q_pr: ArrayView3<f64>,
    q_qs: ArrayView3<f64>,
    v_rs: ArrayView2<f64>,
) -> Array2<f64> {
    // term 1
    let n_at: usize = q_ps.dim().0;
    let n_virt: usize = q_ps.dim().2;
    let n_occ: usize = q_qr.dim().2;
    let qr_dim_1: usize = q_qr.dim().1;

    // term 1
    let tmp: Array3<f64> = q_qr
        .into_shape((n_at * qr_dim_1, n_occ))
        .unwrap()
        .dot(&v_rs)
        .into_shape((n_at, qr_dim_1, n_virt))
        .unwrap();
    let tmp2: Array3<f64> = g0_lr
        .dot(&(tmp.into_shape((n_at, qr_dim_1 * n_virt)).unwrap()))
        .into_shape((n_at, qr_dim_1, n_virt))
        .unwrap();
    let q_ps_swapped = q_ps
        .permuted_axes([1, 0, 2])
        .as_standard_layout()
        .into_shape((qr_dim_1, n_at * n_virt))
        .unwrap()
        .to_owned();
    let tmp2_swapped = tmp2
        .permuted_axes([0, 2, 1])
        .as_standard_layout()
        .into_shape((n_virt * n_at, qr_dim_1))
        .unwrap()
        .to_owned();
    let mut h_minus_pq: Array2<f64> = q_ps_swapped.dot(&tmp2_swapped);

    // term 2
    let tmp: Array3<f64> = q_qs
        .into_shape((n_at * qr_dim_1, n_virt))
        .unwrap()
        .dot(&v_rs.t())
        .into_shape((n_at, qr_dim_1, n_occ))
        .unwrap();
    let tmp2: Array3<f64> = g0_lr
        .dot(&(tmp.into_shape((n_at, qr_dim_1 * n_occ)).unwrap()))
        .into_shape((n_at, qr_dim_1, n_occ))
        .unwrap();
    let q_pr_swapped = q_pr
        .permuted_axes([1, 0, 2])
        .as_standard_layout()
        .into_shape((qr_dim_1, n_at * n_occ))
        .unwrap()
        .to_owned();
    let tmp2_swapped = tmp2
        .permuted_axes([0, 2, 1])
        .as_standard_layout()
        .into_shape((n_at * n_occ, qr_dim_1))
        .unwrap()
        .to_owned();
    h_minus_pq = h_minus_pq - q_pr_swapped.dot(&tmp2_swapped);
    h_minus_pq
}

pub fn h_plus_no_lr(
    g0: ArrayView2<f64>,
    q_pq: ArrayView3<f64>,
    q_rs: ArrayView3<f64>,
    v_rs: ArrayView2<f64>,
) -> Array2<f64> {
    // term 1
    let n_at: usize = q_pq.dim().0;
    let q_rs_dim_1: usize = q_rs.dim().1;
    let q_rs_dim_2: usize = q_rs.dim().2;
    let q_pq_dim_1: usize = q_pq.dim().1;
    let q_pq_dim_2: usize = q_pq.dim().2;

    let tmp: Array1<f64> = q_rs
        .into_shape((n_at, q_rs_dim_1 * q_rs_dim_2))
        .unwrap()
        .dot(&v_rs.into_shape(q_rs_dim_1 * q_rs_dim_2).unwrap());
    let tmp2: Array1<f64> = g0.dot(&tmp);
    let hplus_pq: Array2<f64> = 4.0
        * tmp2
            .dot(&q_pq.into_shape((n_at, q_pq_dim_1 * q_pq_dim_2)).unwrap())
            .into_shape((q_pq_dim_1, q_pq_dim_2))
            .unwrap();
    hplus_pq
}

pub fn h_a_nolr(
    g0: ArrayView2<f64>,
    q_pq: ArrayView3<f64>,
    q_rs: ArrayView3<f64>,
    v_rs: ArrayView2<f64>,
) -> Array2<f64> {
    // term 1
    let n_at: usize = q_pq.dim().0;
    let q_rs_dim_1: usize = q_rs.dim().1;
    let q_rs_dim_2: usize = q_rs.dim().2;
    let q_pq_dim_1: usize = q_pq.dim().1;
    let q_pq_dim_2: usize = q_pq.dim().2;

    let tmp: Array1<f64> = q_rs
        .into_shape((n_at, q_rs_dim_1 * q_rs_dim_2))
        .unwrap()
        .dot(&v_rs.into_shape(q_rs_dim_1 * q_rs_dim_2).unwrap());
    let tmp2: Array1<f64> = g0.dot(&tmp);
    let hplus_pq: Array2<f64> = 4.0
        * tmp2
            .dot(&q_pq.into_shape((n_at, q_pq_dim_1 * q_pq_dim_2)).unwrap())
            .into_shape((q_pq_dim_1, q_pq_dim_2))
            .unwrap();
    hplus_pq
}

pub struct Hplus<'a> {
    qtrans_ov: ArrayView3<'a, f64>,
    qtrans_vv: ArrayView3<'a, f64>,
    qtrans_oo: ArrayView3<'a, f64>,
    qtrans_vo: ArrayView3<'a, f64>,
    n_occ: usize,
    n_virt: usize,
    n_at: usize,
}

impl Hplus<'_> {
    pub fn new<'a>(
        qtrans_ov: ArrayView3<'a, f64>,
        qtrans_vv: ArrayView3<'a, f64>,
        qtrans_oo: ArrayView3<'a, f64>,
        qtrans_vo: ArrayView3<'a, f64>,
    ) -> Hplus<'a> {
        let n_at: usize = qtrans_ov.dim().0;
        let n_occ: usize = qtrans_ov.dim().1;
        let n_virt: usize = qtrans_ov.dim().2;

        Hplus {
            qtrans_ov,
            qtrans_vv,
            qtrans_oo,
            qtrans_vo,
            n_occ,
            n_virt,
            n_at,
        }
    }

    pub fn compute(
        &self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
        hplus_type: HplusType,
    ) -> Array2<f64> {
        let result: Array2<f64> = match hplus_type {
            HplusType::Tab => self.hplus_tab(g0, g0_lr, v),
            HplusType::Tij => self.hplus_tij(g0, g0_lr, v),
            HplusType::QiaXpy => self.hplus_qia_xpy(g0, g0_lr, v),
            HplusType::QiaTab => self.hplus_qia_tab(g0, g0_lr, v),
            HplusType::QiaTij => self.hplus_qia_tij(g0, g0_lr, v),
            HplusType::Qai => self.hplus_qai_or_wij(g0, g0_lr, v),
            HplusType::Wij => self.hplus_qai_or_wij(g0, g0_lr, v),
        };
        result
    }

    fn hplus_tab(
        &self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ: usize = self.n_occ;
        let n_virt: usize = self.n_virt;
        let n_at: usize = self.n_at;

        // term 1
        let tmp: Array1<f64> = self
            .qtrans_vv
            .into_shape((n_at, n_virt * n_virt))
            .unwrap()
            .dot(&v.into_shape(n_virt * n_virt).unwrap());
        let tmp2: Array1<f64> = g0.dot(&tmp);
        let mut hplus_pq: Array2<f64> = 4.0
            * tmp2
                .dot(&self.qtrans_oo.into_shape((n_at, n_occ * n_occ)).unwrap())
                .into_shape((n_occ, n_occ))
                .unwrap();

        // term 2
        let tmp: Array3<f64> = self
            .qtrans_ov
            .into_shape((n_at * n_occ, n_virt))
            .unwrap()
            .dot(&v.t())
            .into_shape((n_at, n_occ, n_virt))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_occ * n_virt)).unwrap())
            .into_shape((n_at, n_occ, n_virt))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_virt, n_occ))
            .unwrap()
            .to_owned();
        let q_swapped = self
            .qtrans_ov
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape((n_occ, n_at * n_virt))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        // term 3
        let tmp: Array3<f64> = self
            .qtrans_ov
            .into_shape((n_at * n_occ, n_virt))
            .unwrap()
            .dot(&v)
            .into_shape((n_at, n_occ, n_virt))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_occ * n_virt)).unwrap())
            .into_shape((n_at, n_occ, n_virt))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_virt, n_occ))
            .unwrap()
            .to_owned();
        let q_swapped = self
            .qtrans_ov
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape((n_occ, n_at * n_virt))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        hplus_pq
    }

    fn hplus_tij(
        &self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ: usize = self.n_occ;
        let n_at: usize = self.n_at;

        // term 1
        let tmp: Array1<f64> = self
            .qtrans_oo
            .into_shape((n_at, n_occ * n_occ))
            .unwrap()
            .dot(&v.into_shape(n_occ * n_occ).unwrap());
        let tmp2: Array1<f64> = g0.dot(&tmp);
        let mut hplus_pq: Array2<f64> = 4.0
            * tmp2
                .dot(&self.qtrans_oo.into_shape((n_at, n_occ * n_occ)).unwrap())
                .into_shape((n_occ, n_occ))
                .unwrap();

        // term 2
        let tmp: Array3<f64> = self
            .qtrans_oo
            .into_shape((n_at * n_occ, n_occ))
            .unwrap()
            .dot(&v.t())
            .into_shape((n_at, n_occ, n_occ))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_occ * n_occ)).unwrap())
            .into_shape((n_at, n_occ, n_occ))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_occ, n_occ))
            .unwrap()
            .to_owned();
        let q_swapped = self
            .qtrans_oo
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape((n_occ, n_at * n_occ))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        // term 3
        let tmp: Array3<f64> = self
            .qtrans_oo
            .into_shape((n_at * n_occ, n_occ))
            .unwrap()
            .dot(&v)
            .into_shape((n_at, n_occ, n_occ))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_occ * n_occ)).unwrap())
            .into_shape((n_at, n_occ, n_occ))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_occ, n_occ))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        hplus_pq
    }

    fn hplus_qia_xpy(
        &self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ: usize = self.n_occ;
        let n_virt: usize = self.n_virt;
        let n_at: usize = self.n_at;

        // term 1
        let tmp: Array1<f64> = self
            .qtrans_ov
            .into_shape((n_at, n_occ * n_virt))
            .unwrap()
            .dot(&v.into_shape(n_occ * n_virt).unwrap());
        let tmp2: Array1<f64> = g0.dot(&tmp);
        let mut hplus_pq: Array2<f64> = 4.0
            * tmp2
                .dot(&self.qtrans_vv.into_shape((n_at, n_virt * n_virt)).unwrap())
                .into_shape((n_virt, n_virt))
                .unwrap();

        // term 2
        let tmp: Array3<f64> = self
            .qtrans_vv
            .into_shape((n_at * n_virt, n_virt))
            .unwrap()
            .dot(&v.t())
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_virt * n_occ)).unwrap())
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_occ, n_virt))
            .unwrap()
            .to_owned();
        let q_swapped = self
            .qtrans_vo
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape((n_virt, n_at * n_occ))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        // term 3
        let tmp: Array3<f64> = self
            .qtrans_vo
            .into_shape((n_at * n_virt, n_occ))
            .unwrap()
            .dot(&v)
            .into_shape((n_at, n_virt, n_virt))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_virt * n_virt)).unwrap())
            .into_shape((n_at, n_virt, n_virt))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_virt, n_virt))
            .unwrap()
            .to_owned();
        let q_swapped = self
            .qtrans_vv
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape((n_virt, n_at * n_virt))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        hplus_pq
    }

    fn hplus_qia_tab(
        &self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ: usize = self.n_occ;
        let n_virt: usize = self.n_virt;
        let n_at: usize = self.n_at;

        // term 1
        let tmp: Array1<f64> = self
            .qtrans_vv
            .into_shape((n_at, n_virt * n_virt))
            .unwrap()
            .dot(&v.into_shape(n_virt * n_virt).unwrap());
        let tmp2: Array1<f64> = g0.dot(&tmp);
        let mut hplus_pq: Array2<f64> = 4.0
            * tmp2
                .dot(&self.qtrans_ov.into_shape((n_at, n_occ * n_virt)).unwrap())
                .into_shape((n_occ, n_virt))
                .unwrap();

        // term 2
        let tmp: Array3<f64> = self
            .qtrans_vv
            .into_shape((n_at * n_virt, n_virt))
            .unwrap()
            .dot(&v.t())
            .into_shape((n_at, n_virt, n_virt))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_virt * n_virt)).unwrap())
            .into_shape((n_at, n_virt, n_virt))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_virt, n_virt))
            .unwrap()
            .to_owned();
        let q_swapped = self
            .qtrans_ov
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape((n_occ, n_at * n_virt))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        // term 3
        let tmp: Array3<f64> = self
            .qtrans_vv
            .into_shape((n_at * n_virt, n_virt))
            .unwrap()
            .dot(&v)
            .into_shape((n_at, n_virt, n_virt))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_virt * n_virt)).unwrap())
            .into_shape((n_at, n_virt, n_virt))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_virt, n_virt))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        hplus_pq
    }

    fn hplus_qia_tij(
        &self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ: usize = self.n_occ;
        let n_virt: usize = self.n_virt;
        let n_at: usize = self.n_at;

        // term 1
        let tmp: Array1<f64> = self
            .qtrans_oo
            .into_shape((n_at, n_occ * n_occ))
            .unwrap()
            .dot(&v.into_shape(n_occ * n_occ).unwrap());
        let tmp2: Array1<f64> = g0.dot(&tmp);
        let mut hplus_pq: Array2<f64> = 4.0
            * tmp2
                .dot(&self.qtrans_ov.into_shape((n_at, n_occ * n_virt)).unwrap())
                .into_shape((n_occ, n_virt))
                .unwrap();

        // term 2
        let tmp: Array3<f64> = self
            .qtrans_vo
            .into_shape((n_at * n_virt, n_occ))
            .unwrap()
            .dot(&v.t())
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_virt * n_occ)).unwrap())
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_occ, n_virt))
            .unwrap()
            .to_owned();
        let q_swapped = self
            .qtrans_oo
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape((n_occ, n_at * n_occ))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        // term 3
        let tmp: Array3<f64> = self
            .qtrans_vo
            .into_shape((n_at * n_virt, n_occ))
            .unwrap()
            .dot(&v)
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_virt * n_occ)).unwrap())
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_occ, n_virt))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        hplus_pq
    }

    fn hplus_qai_or_wij(
        &self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ: usize = self.n_occ;
        let n_virt: usize = self.n_virt;
        let n_at: usize = self.n_at;

        // term 1
        let tmp: Array1<f64> = self
            .qtrans_ov
            .into_shape((n_at, n_occ * n_virt))
            .unwrap()
            .dot(&v.into_shape(n_occ * n_virt).unwrap());
        let tmp2: Array1<f64> = g0.dot(&tmp);
        let mut hplus_pq: Array2<f64> = 4.0
            * tmp2
                .dot(&self.qtrans_oo.into_shape((n_at, n_occ * n_occ)).unwrap())
                .into_shape((n_occ, n_occ))
                .unwrap();

        // term 2
        let tmp: Array3<f64> = self
            .qtrans_ov
            .into_shape((n_at * n_occ, n_virt))
            .unwrap()
            .dot(&v.t())
            .into_shape((n_at, n_occ, n_occ))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_occ * n_occ)).unwrap())
            .into_shape((n_at, n_occ, n_occ))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_occ, n_occ))
            .unwrap()
            .to_owned();
        let q_swapped = self
            .qtrans_oo
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape((n_occ, n_at * n_occ))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        // term 3
        let tmp: Array3<f64> = self
            .qtrans_oo
            .into_shape((n_at * n_occ, n_occ))
            .unwrap()
            .dot(&v)
            .into_shape((n_at, n_occ, n_virt))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_occ * n_virt)).unwrap())
            .into_shape((n_at, n_occ, n_virt))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_virt, n_occ))
            .unwrap()
            .to_owned();
        let q_swapped = self
            .qtrans_ov
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape((n_occ, n_at * n_virt))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        hplus_pq
    }
}

pub enum HplusType {
    Tab,
    Tij,
    QiaXpy,
    QiaTab,
    QiaTij,
    Qai,
    Wij,
}

pub struct Hav<'a> {
    qtrans_ov: ArrayView3<'a, f64>,
    qtrans_vv: ArrayView3<'a, f64>,
    qtrans_oo: ArrayView3<'a, f64>,
    qtrans_vo: ArrayView3<'a, f64>,
    n_occ: usize,
    n_virt: usize,
    n_at: usize,
}

impl Hav<'_> {
    pub fn new<'a>(
        qtrans_ov: ArrayView3<'a, f64>,
        qtrans_vv: ArrayView3<'a, f64>,
        qtrans_oo: ArrayView3<'a, f64>,
        qtrans_vo: ArrayView3<'a, f64>,
    ) -> Hav<'a> {
        let n_at: usize = qtrans_ov.dim().0;
        let n_occ: usize = qtrans_ov.dim().1;
        let n_virt: usize = qtrans_ov.dim().2;

        Hav {
            qtrans_ov,
            qtrans_vv,
            qtrans_oo,
            qtrans_vo,
            n_occ,
            n_virt,
            n_at,
        }
    }

    pub fn compute(
        &self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
        hplus_type: HplusType,
    ) -> Array2<f64> {
        let result: Array2<f64> = match hplus_type {
            HplusType::Tab => 2.0 * self.hav_tab(g0, g0_lr, v),
            HplusType::Tij => 2.0 * self.hav_tij(g0, g0_lr, v),
            HplusType::QiaXpy => 2.0 * self.hav_qia_x(g0, g0_lr, v),
            HplusType::QiaTab => 2.0 * self.hav_qia_tab(g0, g0_lr, v),
            HplusType::QiaTij => 2.0 * self.hav_qia_tij(g0, g0_lr, v),
            HplusType::Qai => 2.0 * self.hav_qai_or_wij(g0, g0_lr, v),
            HplusType::Wij => 2.0 * self.hav_qai_or_wij(g0, g0_lr, v),
        };
        result
    }

    fn hav_tab(
        &self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ: usize = self.n_occ;
        let n_virt: usize = self.n_virt;
        let n_at: usize = self.n_at;

        // term 1
        let tmp: Array1<f64> = self
            .qtrans_vv
            .into_shape((n_at, n_virt * n_virt))
            .unwrap()
            .dot(&v.into_shape(n_virt * n_virt).unwrap());
        let tmp2: Array1<f64> = g0.dot(&tmp);
        let mut hplus_pq: Array2<f64> = 2.0
            * tmp2
                .dot(&self.qtrans_oo.into_shape((n_at, n_occ * n_occ)).unwrap())
                .into_shape((n_occ, n_occ))
                .unwrap();

        // term 2
        let tmp: Array3<f64> = self
            .qtrans_ov
            .into_shape((n_at * n_occ, n_virt))
            .unwrap()
            .dot(&v.t())
            .into_shape((n_at, n_occ, n_virt))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_occ * n_virt)).unwrap())
            .into_shape((n_at, n_occ, n_virt))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_virt, n_occ))
            .unwrap()
            .to_owned();
        let q_swapped = self
            .qtrans_ov
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape((n_occ, n_at * n_virt))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        hplus_pq
    }

    fn hav_tij(
        &self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ: usize = self.n_occ;
        let n_at: usize = self.n_at;

        // term 1
        let tmp: Array1<f64> = self
            .qtrans_oo
            .into_shape((n_at, n_occ * n_occ))
            .unwrap()
            .dot(&v.into_shape(n_occ * n_occ).unwrap());
        let tmp2: Array1<f64> = g0.dot(&tmp);
        let mut hplus_pq: Array2<f64> = 2.0
            * tmp2
                .dot(&self.qtrans_oo.into_shape((n_at, n_occ * n_occ)).unwrap())
                .into_shape((n_occ, n_occ))
                .unwrap();

        // term 2
        let tmp: Array3<f64> = self
            .qtrans_oo
            .into_shape((n_at * n_occ, n_occ))
            .unwrap()
            .dot(&v.t())
            .into_shape((n_at, n_occ, n_occ))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_occ * n_occ)).unwrap())
            .into_shape((n_at, n_occ, n_occ))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_occ, n_occ))
            .unwrap()
            .to_owned();
        let q_swapped = self
            .qtrans_oo
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape((n_occ, n_at * n_occ))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        hplus_pq
    }

    fn hav_qia_x(
        &self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ: usize = self.n_occ;
        let n_virt: usize = self.n_virt;
        let n_at: usize = self.n_at;

        // term 1
        let tmp: Array1<f64> = self
            .qtrans_ov
            .into_shape((n_at, n_occ * n_virt))
            .unwrap()
            .dot(&v.into_shape(n_occ * n_virt).unwrap());
        let tmp2: Array1<f64> = g0.dot(&tmp);
        let mut hplus_pq: Array2<f64> = 2.0
            * tmp2
                .dot(&self.qtrans_vv.into_shape((n_at, n_virt * n_virt)).unwrap())
                .into_shape((n_virt, n_virt))
                .unwrap();

        // term 2
        let tmp: Array3<f64> = self
            .qtrans_vv
            .into_shape((n_at * n_virt, n_virt))
            .unwrap()
            .dot(&v.t())
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_virt * n_occ)).unwrap())
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_occ, n_virt))
            .unwrap()
            .to_owned();
        let q_swapped = self
            .qtrans_vo
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape((n_virt, n_at * n_occ))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        hplus_pq
    }

    fn hav_qia_tab(
        &self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ: usize = self.n_occ;
        let n_virt: usize = self.n_virt;
        let n_at: usize = self.n_at;

        // term 1
        let tmp: Array1<f64> = self
            .qtrans_vv
            .into_shape((n_at, n_virt * n_virt))
            .unwrap()
            .dot(&v.into_shape(n_virt * n_virt).unwrap());
        let tmp2: Array1<f64> = g0.dot(&tmp);
        let mut hplus_pq: Array2<f64> = 2.0
            * tmp2
                .dot(&self.qtrans_ov.into_shape((n_at, n_occ * n_virt)).unwrap())
                .into_shape((n_occ, n_virt))
                .unwrap();

        // term 2
        let tmp: Array3<f64> = self
            .qtrans_vv
            .into_shape((n_at * n_virt, n_virt))
            .unwrap()
            .dot(&v.t())
            .into_shape((n_at, n_virt, n_virt))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_virt * n_virt)).unwrap())
            .into_shape((n_at, n_virt, n_virt))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_virt, n_virt))
            .unwrap()
            .to_owned();
        let q_swapped = self
            .qtrans_ov
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape((n_occ, n_at * n_virt))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        hplus_pq
    }

    fn hav_qia_tij(
        &self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ: usize = self.n_occ;
        let n_virt: usize = self.n_virt;
        let n_at: usize = self.n_at;

        // term 1
        let tmp: Array1<f64> = self
            .qtrans_oo
            .into_shape((n_at, n_occ * n_occ))
            .unwrap()
            .dot(&v.into_shape(n_occ * n_occ).unwrap());
        let tmp2: Array1<f64> = g0.dot(&tmp);
        let mut hplus_pq: Array2<f64> = 2.0
            * tmp2
                .dot(&self.qtrans_ov.into_shape((n_at, n_occ * n_virt)).unwrap())
                .into_shape((n_occ, n_virt))
                .unwrap();

        // term 2
        let tmp: Array3<f64> = self
            .qtrans_vo
            .into_shape((n_at * n_virt, n_occ))
            .unwrap()
            .dot(&v.t())
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_virt * n_occ)).unwrap())
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_occ, n_virt))
            .unwrap()
            .to_owned();
        let q_swapped = self
            .qtrans_oo
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape((n_occ, n_at * n_occ))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        hplus_pq
    }

    fn hav_qai_or_wij(
        &self,
        g0: ArrayView2<f64>,
        g0_lr: ArrayView2<f64>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_occ: usize = self.n_occ;
        let n_virt: usize = self.n_virt;
        let n_at: usize = self.n_at;

        // term 1
        let tmp: Array1<f64> = self
            .qtrans_ov
            .into_shape((n_at, n_occ * n_virt))
            .unwrap()
            .dot(&v.into_shape(n_occ * n_virt).unwrap());
        let tmp2: Array1<f64> = g0.dot(&tmp);
        let mut hplus_pq: Array2<f64> = 2.0
            * tmp2
                .dot(&self.qtrans_oo.into_shape((n_at, n_occ * n_occ)).unwrap())
                .into_shape((n_occ, n_occ))
                .unwrap();

        // term 2
        let tmp: Array3<f64> = self
            .qtrans_ov
            .into_shape((n_at * n_occ, n_virt))
            .unwrap()
            .dot(&v.t())
            .into_shape((n_at, n_occ, n_occ))
            .unwrap();
        let tmp2: Array3<f64> = g0_lr
            .dot(&tmp.into_shape((n_at, n_occ * n_occ)).unwrap())
            .into_shape((n_at, n_occ, n_occ))
            .unwrap();
        let tmp2_swapped = tmp2
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape((n_at * n_occ, n_occ))
            .unwrap()
            .to_owned();
        let q_swapped = self
            .qtrans_oo
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape((n_occ, n_at * n_occ))
            .unwrap()
            .to_owned();
        hplus_pq = hplus_pq - q_swapped.dot(&tmp2_swapped);

        hplus_pq
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
pub fn gradient_v_rep(atoms: &[Atom], v_rep: &RepulsivePotential) -> Array1<f64> {
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

pub fn zvector_lc(
    a_diag: ArrayView2<f64>,
    r_matrix: ArrayView2<f64>,
    g0: ArrayView2<f64>,
    g0_lr: ArrayView2<f64>,
    qtrans_oo: ArrayView3<f64>,
    qtrans_vv: ArrayView3<f64>,
    qtrans_ov: ArrayView3<f64>,
) -> Array2<f64> {
    // Parameters:
    // ===========
    // A: linear operator, such that A(X) = A.X
    // Adiag: diagonal elements of A-matrix, with dimension (nocc,nvirt)
    // B: right hand side of equation, (nocc,nvirt, k)
    let maxiter: usize = 10000;
    let conv: f64 = 1.0e-16;

    let n_occ: usize = r_matrix.dim().0;
    let n_virt: usize = r_matrix.dim().1;
    let n_at: usize = qtrans_ov.dim().0;
    let kmax: usize = n_occ * n_virt;

    // bs are expansion vectors
    let a_inv: Array2<f64> = 1.0 / &a_diag.to_owned();
    let bs: Array2<f64> = &a_inv * &r_matrix;
    let rhs: Array1<f64> = r_matrix.into_shape(kmax).unwrap().to_owned();

    // create new arrays for transition charges of specific shapes,
    // which are required by the mult_apb_v_routine
    let tmp_q_vv: ArrayView2<f64> = qtrans_vv.into_shape((n_virt * n_at, n_virt)).unwrap();
    let tmp_q_oo: ArrayView2<f64> = qtrans_oo.into_shape((n_at * n_occ, n_occ)).unwrap();
    let tmp_q_oo_t: Array2<f64> = tmp_q_oo.t().to_owned(); // precomputed transpose
    let tmp_q_ov_swapped: ArrayView3<f64> = qtrans_ov.permuted_axes([0, 2, 1]);
    let tmp_q_ov_shape_1: Array2<f64> = tmp_q_ov_swapped
        .as_standard_layout()
        .to_owned()
        .into_shape((n_at * n_virt, n_occ))
        .unwrap();
    let tmp_q_ov_swapped_2: ArrayView3<f64> = qtrans_ov.permuted_axes([1, 0, 2]);
    let tmp_q_ov_shape_2: Array2<f64> = tmp_q_ov_swapped_2
        .as_standard_layout()
        .to_owned()
        .into_shape((n_occ, n_at * n_virt))
        .unwrap();

    let apbv: Array2<f64> = mult_apb_v(
        g0,
        g0_lr,
        qtrans_ov,
        tmp_q_oo.view(),
        tmp_q_oo_t.view(),
        tmp_q_vv.view(),
        tmp_q_ov_shape_1.view(),
        tmp_q_ov_shape_2.view(),
        a_diag,
        bs.view(),
        n_occ,
        n_virt,
    );

    let rkm1 = apbv.into_shape(kmax).unwrap();
    let mut rhs_2 = bs.into_shape(kmax).unwrap();
    let mut rkm1 = rhs - rkm1;
    let mut pkm1 = rkm1.clone();

    for _it in 0..maxiter {
        let apbv: Array2<f64> = mult_apb_v(
            g0,
            g0_lr,
            qtrans_ov,
            tmp_q_oo.view(),
            tmp_q_oo_t.view(),
            tmp_q_vv.view(),
            tmp_q_ov_shape_1.view(),
            tmp_q_ov_shape_2.view(),
            a_diag,
            pkm1.view().into_shape((n_occ, n_virt)).unwrap(),
            n_occ,
            n_virt,
        );
        let apk: Array1<f64> = apbv.into_shape(kmax).unwrap();

        let tmp1: f64 = rkm1.dot(&rkm1);
        let tmp2: f64 = pkm1.dot(&apk);

        rhs_2 = rhs_2 + (tmp1 / tmp2) * &pkm1;
        rkm1 = rkm1 - (tmp1 / tmp2) * &apk;

        let tmp2: f64 = rkm1.dot(&rkm1);

        if tmp2 <= conv {
            break;
        }
        pkm1 = (tmp2 / tmp1) * &pkm1 + &rkm1;
    }

    let out: Array2<f64> = rhs_2.into_shape((n_occ, n_virt)).unwrap();
    out
}

pub fn zvector_no_lc(
    a_diag: ArrayView2<f64>,
    r_matrix: ArrayView2<f64>,
    g0: ArrayView2<f64>,
    qtrans_ov: ArrayView3<f64>,
) -> Array2<f64> {
    let maxiter: usize = 10000;
    let conv: f64 = 1.0e-16;

    let n_occ: usize = r_matrix.dim().0;
    let n_virt: usize = r_matrix.dim().1;
    let kmax: usize = n_occ * n_virt;
    let _n_at: usize = qtrans_ov.dim().0;

    // bs are expansion vectors
    let a_inv: Array2<f64> = 1.0 / &a_diag.to_owned();
    let bs: Array2<f64> = &a_inv * &r_matrix;
    let rhs: Array1<f64> = r_matrix.into_shape(kmax).unwrap().to_owned();
    let apbv: Array2<f64> = mult_apb_v_no_lc(g0, qtrans_ov, a_diag, bs.view(), n_occ, n_virt);

    let rkm1 = apbv.into_shape(kmax).unwrap();
    let mut rhs_2 = bs.into_shape(kmax).unwrap();
    let mut rkm1 = rhs - rkm1;
    let mut pkm1 = rkm1.clone();

    for _it in 0..maxiter {
        let apbv: Array2<f64> = mult_apb_v_no_lc(
            g0,
            qtrans_ov,
            a_diag,
            pkm1.view().into_shape((n_occ, n_virt)).unwrap(),
            n_occ,
            n_virt,
        );
        let apk: Array1<f64> = apbv.into_shape(kmax).unwrap();

        let tmp1: f64 = rkm1.dot(&rkm1);
        let tmp2: f64 = pkm1.dot(&apk);

        rhs_2 = rhs_2 + (tmp1 / tmp2) * &pkm1;
        rkm1 = rkm1 - (tmp1 / tmp2) * &apk;

        let tmp2: f64 = rkm1.dot(&rkm1);

        if tmp2 <= conv {
            break;
        }
        pkm1 = (tmp2 / tmp1) * &pkm1 + &rkm1;
    }

    let out: Array2<f64> = rhs_2.into_shape((n_occ, n_virt)).unwrap();
    out
}

pub fn tda_zvector_no_lc(
    a_diag: ArrayView2<f64>,
    r_matrix: ArrayView2<f64>,
    g0: ArrayView2<f64>,
    qtrans_ov: ArrayView3<f64>,
) -> Array2<f64> {
    let maxiter: usize = 10000;
    let conv: f64 = 1.0e-16;

    let n_occ: usize = r_matrix.dim().0;
    let n_virt: usize = r_matrix.dim().1;
    let kmax: usize = n_occ * n_virt;
    let _n_at: usize = qtrans_ov.dim().0;

    // bs are expansion vectors
    let a_inv: Array2<f64> = 1.0 / &a_diag.to_owned();
    let bs: Array2<f64> = &a_inv * &r_matrix;

    let rhs: Array1<f64> = r_matrix.into_shape(kmax).unwrap().to_owned();
    let apbv: Array2<f64> = mult_av_nolc(g0, qtrans_ov, a_diag, bs.view(), n_occ, n_virt);
    let mut rkm1: Array1<f64> = apbv.into_shape(kmax).unwrap();
    let mut rhs_2: Array1<f64> = bs.into_shape(kmax).unwrap();
    rkm1 = rhs - rkm1;
    let mut pkm1: Array1<f64> = rkm1.clone();

    for _it in 0..maxiter {
        let apbv: Array2<f64> = mult_av_nolc(
            g0,
            qtrans_ov,
            a_diag,
            pkm1.view().into_shape((n_occ, n_virt)).unwrap(),
            n_occ,
            n_virt,
        );
        let apk: Array1<f64> = apbv.into_shape(kmax).unwrap();

        let tmp1: f64 = rkm1.dot(&rkm1);
        let tmp2: f64 = pkm1.dot(&apk);

        rhs_2 = rhs_2 + (tmp1 / tmp2) * &pkm1;
        rkm1 = rkm1 - (tmp1 / tmp2) * &apk;

        let tmp2: f64 = rkm1.dot(&rkm1);

        if tmp2 <= conv {
            break;
        }
        pkm1 = (tmp2 / tmp1) * &pkm1 + &rkm1;
    }

    let out: Array2<f64> = rhs_2.into_shape((n_occ, n_virt)).unwrap();
    out
}

pub fn tda_zvector_lc(
    a_diag: ArrayView2<f64>,
    r_matrix: ArrayView2<f64>,
    g0: ArrayView2<f64>,
    g0_lr: ArrayView2<f64>,
    qtrans_oo: ArrayView3<f64>,
    qtrans_vv: ArrayView3<f64>,
    qtrans_ov: ArrayView3<f64>,
) -> Array2<f64> {
    let maxiter: usize = 10000;
    let conv: f64 = 1.0e-16;

    let n_occ: usize = r_matrix.dim().0;
    let n_virt: usize = r_matrix.dim().1;
    let kmax: usize = n_occ * n_virt;
    let n_at: usize = qtrans_ov.dim().0;

    // bs are expansion vectors
    let a_inv: Array2<f64> = 1.0 / &a_diag.to_owned();
    let bs: Array2<f64> = &a_inv * &r_matrix;
    let rhs: Array1<f64> = r_matrix.into_shape(kmax).unwrap().to_owned();

    // create new arrays for transition charges of specific shapes,
    // which are required by the mult_apb_v_routine
    let tmp_q_vv: ArrayView2<f64> = qtrans_vv.into_shape((n_virt * n_at, n_virt)).unwrap();
    let tmp_q_oo: ArrayView2<f64> = qtrans_oo.into_shape((n_at * n_occ, n_occ)).unwrap();

    let apbv: Array2<f64> = mult_av_lc(
        g0,
        g0_lr,
        qtrans_ov,
        tmp_q_oo,
        tmp_q_vv,
        a_diag,
        bs.view(),
        n_occ,
        n_virt,
    );
    let mut rkm1: Array1<f64> = apbv.into_shape(kmax).unwrap();
    let mut rhs_2: Array1<f64> = bs.into_shape(kmax).unwrap();
    rkm1 = rhs - rkm1;
    let mut pkm1: Array1<f64> = rkm1.clone();

    for _it in 0..maxiter {
        let apbv: Array2<f64> = mult_av_lc(
            g0,
            g0_lr,
            qtrans_ov,
            tmp_q_oo,
            tmp_q_vv,
            a_diag,
            pkm1.view().into_shape((n_occ, n_virt)).unwrap(),
            n_occ,
            n_virt,
        );
        let apk: Array1<f64> = apbv.into_shape(kmax).unwrap();

        let tmp1: f64 = rkm1.dot(&rkm1);
        let tmp2: f64 = pkm1.dot(&apk);

        rhs_2 = rhs_2 + (tmp1 / tmp2) * &pkm1;
        rkm1 = rkm1 - (tmp1 / tmp2) * &apk;

        let tmp2: f64 = rkm1.dot(&rkm1);

        if tmp2 <= conv {
            break;
        }
        pkm1 = (tmp2 / tmp1) * &pkm1 + &rkm1;
    }

    let out: Array2<f64> = rhs_2.into_shape((n_occ, n_virt)).unwrap();
    out
}

fn mult_apb_v(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    qtrans_ov: ArrayView3<f64>,
    _qtrans_oo_reshaped: ArrayView2<f64>,
    qtrans_oo_reshaped_t: ArrayView2<f64>, // precomputed transpose for Term 3d
    qtrans_vv_reshaped: ArrayView2<f64>,
    qtrans_ov_reshaped_1: ArrayView2<f64>,
    qtrans_ov_reshaped_2: ArrayView2<f64>,
    omega: ArrayView2<f64>,
    vs: ArrayView2<f64>,
    n_occ: usize,
    n_virt: usize,
) -> Array2<f64> {
    let n_at: usize = qtrans_ov.dim().0;

    // 1st term - KS orbital energy differences
    let mut u_l: Array2<f64> = &omega * &vs;

    // 2nd term - Coulomb (use gamma directly without copy)
    u_l = u_l
        + 4.0
            * gamma
                .dot(
                    &qtrans_ov
                        .into_shape([n_at, n_occ * n_virt])
                        .unwrap()
                        .dot(&vs.into_shape(n_occ * n_virt).unwrap()),
                )
                .dot(&qtrans_ov.into_shape([n_at, n_occ * n_virt]).unwrap())
                .into_shape([n_occ, n_virt])
                .unwrap();

    // 3rd and 4th terms - Exchange (run in parallel using rayon::join)
    let (tmp33, tmp43) = rayon::join(
        || {
            // Term 3 - Exchange
            let tmp31: Array3<f64> = qtrans_vv_reshaped
                .dot(&vs.t())
                .into_shape((n_at, n_virt, n_occ))
                .unwrap();
            let tmp31_reshaped: Array2<f64> = tmp31.into_shape((n_at, n_virt * n_occ)).unwrap();
            let mut tmp32: Array3<f64> = gamma_lr
                .dot(&tmp31_reshaped)
                .into_shape((n_at, n_virt, n_occ))
                .unwrap();
            tmp32.swap_axes(1, 2);
            let tmp32 = tmp32.as_standard_layout();
            qtrans_oo_reshaped_t.dot(&tmp32.into_shape((n_at * n_occ, n_virt)).unwrap())
        },
        || {
            // Term 4 - Exchange
            let tmp41: Array3<f64> = qtrans_ov_reshaped_1
                .dot(&vs)
                .into_shape((n_at, n_virt, n_virt))
                .unwrap();
            let tmp41_reshaped: Array2<f64> = tmp41.into_shape((n_at, n_virt * n_virt)).unwrap();
            let mut tmp42: Array3<f64> = gamma_lr
                .dot(&tmp41_reshaped)
                .into_shape((n_at, n_virt, n_virt))
                .unwrap();
            tmp42.swap_axes(1, 2);
            let tmp42 = tmp42.as_standard_layout();
            qtrans_ov_reshaped_2.dot(&tmp42.into_shape((n_at * n_virt, n_virt)).unwrap())
        },
    );

    u_l = u_l - tmp33 - tmp43;

    u_l
}

/// Optimized mult_apb_v using direct BLAS DGEMM calls with preallocated workspace.
///
/// This eliminates per-call allocations by reusing workspace buffers.
/// The key optimization is using in-place DGEMM operations instead of ndarray::dot()
/// which allocates a new array for each operation.
///
/// Mathematical operations:
/// - Term 1: u_l = omega * vs (element-wise)
/// - Term 2 (Coulomb): u_l += 4 * (gamma · (qtrans_ov · vs)) · qtrans_ov
/// - Term 3 (Exchange): u_l -= qtrans_oo^T · (gamma_lr · (qtrans_vv · vs^T))_swapped
/// - Term 4 (Exchange): u_l -= qtrans_ov_2 · (gamma_lr · (qtrans_ov_1 · vs))_swapped
fn mult_apb_v_blas(
    omega: ArrayView2<f64>,
    vs: ArrayView2<f64>,
    ws: &mut ZVectorWorkspace,
) -> Array2<f64> {
    use crate::linalg::dgemm::{
        batch_transpose_blocked_unchecked, dgemm_a_bt, dgemm_row_major,
    };

    let n_at = ws.n_at;
    let n_occ = ws.n_occ;
    let n_virt = ws.n_virt;

    // Get vs as a flat slice
    let vs_flat: &[f64] = vs
        .as_slice()
        .unwrap_or_else(|| panic!("vs must be contiguous in memory"));

    // =========== Term 1: u_l = omega * vs (element-wise) ===========
    let mut u_l: Array2<f64> = &omega * &vs;
    let u_l_flat: &mut [f64] = u_l.as_slice_mut().unwrap();

    // =========== Term 2: Coulomb ===========
    // u_l += 4 * gamma · (qtrans_ov · vs) · qtrans_ov
    // Step 2a: tmp = qtrans_ov · vs → [n_at]
    // Step 2b: tmp2 = gamma · tmp → [n_at]
    // Step 2c: u_l += 4 * tmp2 · qtrans_ov → [n_occ * n_virt]
    unsafe {
        // Step 2a: qtrans_ov [n_at, n_occ*n_virt] · vs [n_occ*n_virt] → buf_coulomb_tmp [n_at]
        // This is a matrix-vector product: A[n_at, k] · x[k] = y[n_at]
        // Use DGEMM with n=1: C[n_at, 1] = A[n_at, k] · B[k, 1]
        dgemm_row_major(
            1.0,
            &ws.qtrans_ov_flat,
            n_at,
            n_occ * n_virt, // A: [n_at x (n_occ*n_virt)]
            vs_flat,
            1, // B: [(n_occ*n_virt) x 1]
            0.0,
            &mut ws.buf_coulomb_tmp, // C: [n_at x 1]
        );

        // Step 2b: gamma [n_at, n_at] · buf_coulomb_tmp [n_at] → in-place in buf_coulomb_tmp
        // Actually we need a separate buffer here. Use buf_coulomb_out temporarily.
        // tmp2 = gamma · tmp
        let mut tmp2 = vec![0.0; n_at]; // Small allocation, acceptable
        dgemm_row_major(
            1.0,
            &ws.gamma_flat,
            n_at,
            n_at,
            &ws.buf_coulomb_tmp,
            1,
            0.0,
            &mut tmp2,
        );

        // Step 2c: u_l += 4 * tmp2^T · qtrans_ov
        // This is: [1, n_at] · [n_at, n_occ*n_virt] = [1, n_occ*n_virt]
        // Which is equivalent to: qtrans_ov^T · tmp2 with appropriate reshaping
        // Use DGEMM: C[1, n_occ*n_virt] = A[1, n_at] · B[n_at, n_occ*n_virt]
        dgemm_row_major(
            4.0,
            &tmp2,
            1,
            n_at,
            &ws.qtrans_ov_flat,
            n_occ * n_virt,
            1.0, // beta=1 adds to existing u_l
            u_l_flat,
        );
    }

    // =========== Terms 3 & 4: Exchange (parallel) ===========
    // Take workspace buffers for parallel use (will be restored after)
    let mut buf_t3_step1 = std::mem::take(&mut ws.buf_t3_step1);
    let mut buf_t3_step2 = std::mem::take(&mut ws.buf_t3_step2);
    let mut buf_t3_swapped = std::mem::take(&mut ws.buf_t3_swapped);
    let mut buf_t3_out = std::mem::take(&mut ws.buf_t3_out);
    let mut buf_t4_step1 = std::mem::take(&mut ws.buf_t4_step1);
    let mut buf_t4_step2 = std::mem::take(&mut ws.buf_t4_step2);
    let mut buf_t4_swapped = std::mem::take(&mut ws.buf_t4_swapped);
    let mut buf_t4_out = std::mem::take(&mut ws.buf_t4_out);

    // References to data arrays
    let qtrans_vv_flat = &ws.qtrans_vv_flat;
    let qtrans_ov_1_flat = &ws.qtrans_ov_1_flat;
    let qtrans_ov_2_flat = &ws.qtrans_ov_2_flat;
    let qtrans_oo_t_flat = &ws.qtrans_oo_t_flat;
    let gamma_lr_flat = &ws.gamma_lr_flat;

    if ws.use_precomputed_gamma {
        // ===== FAST PATH: Use precomputed gamma_lr products =====
        let gamma_lr_qvv_flat = &ws.gamma_lr_qvv_flat;
        let gamma_lr_qov_1_flat = &ws.gamma_lr_qov_1_flat;

        rayon::join(
            || {
                // Term 3: Uses precomputed gamma_lr_qvv
                unsafe {
                    dgemm_a_bt(
                        1.0,
                        gamma_lr_qvv_flat,
                        n_at * n_virt,
                        n_virt,
                        vs_flat,
                        n_occ,
                        0.0,
                        &mut buf_t3_step1,
                    );
                    batch_transpose_blocked_unchecked(
                        buf_t3_step1.as_ptr(),
                        buf_t3_swapped.as_mut_ptr(),
                        n_at,
                        n_virt,
                        n_occ,
                    );
                    dgemm_row_major(
                        1.0,
                        qtrans_oo_t_flat,
                        n_occ,
                        n_at * n_occ,
                        &buf_t3_swapped,
                        n_virt,
                        0.0,
                        &mut buf_t3_out,
                    );
                }
            },
            || {
                // Term 4: Uses precomputed gamma_lr_qov_1
                unsafe {
                    dgemm_row_major(
                        1.0,
                        gamma_lr_qov_1_flat,
                        n_at * n_virt,
                        n_occ,
                        vs_flat,
                        n_virt,
                        0.0,
                        &mut buf_t4_step1,
                    );
                    batch_transpose_blocked_unchecked(
                        buf_t4_step1.as_ptr(),
                        buf_t4_swapped.as_mut_ptr(),
                        n_at,
                        n_virt,
                        n_virt,
                    );
                    dgemm_row_major(
                        1.0,
                        qtrans_ov_2_flat,
                        n_occ,
                        n_at * n_virt,
                        &buf_t4_swapped,
                        n_virt,
                        0.0,
                        &mut buf_t4_out,
                    );
                }
            },
        );
    } else {
        // ===== MEMORY-SAVING PATH: Compute gamma_lr products per iteration =====
        rayon::join(
            || {
                // Term 3: qtrans_vv · vs^T, then gamma_lr · result
                unsafe {
                    dgemm_a_bt(
                        1.0,
                        qtrans_vv_flat,
                        n_at * n_virt,
                        n_virt,
                        vs_flat,
                        n_occ,
                        0.0,
                        &mut buf_t3_step1,
                    );
                    dgemm_row_major(
                        1.0,
                        gamma_lr_flat,
                        n_at,
                        n_at,
                        &buf_t3_step1,
                        n_virt * n_occ,
                        0.0,
                        &mut buf_t3_step2,
                    );
                    batch_transpose_blocked_unchecked(
                        buf_t3_step2.as_ptr(),
                        buf_t3_swapped.as_mut_ptr(),
                        n_at,
                        n_virt,
                        n_occ,
                    );
                    dgemm_row_major(
                        1.0,
                        qtrans_oo_t_flat,
                        n_occ,
                        n_at * n_occ,
                        &buf_t3_swapped,
                        n_virt,
                        0.0,
                        &mut buf_t3_out,
                    );
                }
            },
            || {
                // Term 4: qtrans_ov_1 · vs, then gamma_lr · result
                unsafe {
                    dgemm_row_major(
                        1.0,
                        qtrans_ov_1_flat,
                        n_at * n_virt,
                        n_occ,
                        vs_flat,
                        n_virt,
                        0.0,
                        &mut buf_t4_step1,
                    );
                    dgemm_row_major(
                        1.0,
                        gamma_lr_flat,
                        n_at,
                        n_at,
                        &buf_t4_step1,
                        n_virt * n_virt,
                        0.0,
                        &mut buf_t4_step2,
                    );
                    batch_transpose_blocked_unchecked(
                        buf_t4_step2.as_ptr(),
                        buf_t4_swapped.as_mut_ptr(),
                        n_at,
                        n_virt,
                        n_virt,
                    );
                    dgemm_row_major(
                        1.0,
                        qtrans_ov_2_flat,
                        n_occ,
                        n_at * n_virt,
                        &buf_t4_swapped,
                        n_virt,
                        0.0,
                        &mut buf_t4_out,
                    );
                }
            },
        );
    }

    // Restore workspace buffers
    ws.buf_t3_step1 = buf_t3_step1;
    ws.buf_t3_step2 = buf_t3_step2;
    ws.buf_t3_swapped = buf_t3_swapped;
    ws.buf_t3_out = buf_t3_out;
    ws.buf_t4_step1 = buf_t4_step1;
    ws.buf_t4_step2 = buf_t4_step2;
    ws.buf_t4_swapped = buf_t4_swapped;
    ws.buf_t4_out = buf_t4_out;

    // Subtract Term 3 and Term 4 from u_l
    for i in 0..n_occ * n_virt {
        u_l_flat[i] -= ws.buf_t3_out[i] + ws.buf_t4_out[i];
    }

    u_l
}

fn mult_apb_v_no_lc(
    gamma: ArrayView2<f64>,
    qtrans_ov: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    vs: ArrayView2<f64>,
    n_occ: usize,
    n_virt: usize,
) -> Array2<f64> {
    let n_at: usize = qtrans_ov.dim().0;
    let gamma_equiv: Array2<f64> = gamma.to_owned();

    // 1st term - KS orbital energy differences
    let mut u_l: Array2<f64> = &omega * &vs;

    // 2nd term - Coulomb
    u_l = u_l
        + 4.0
            * gamma_equiv
                .dot(
                    &qtrans_ov
                        .into_shape([n_at, n_occ * n_virt])
                        .unwrap()
                        .dot(&vs.into_shape(n_occ * n_virt).unwrap()),
                )
                .dot(&qtrans_ov.into_shape([n_at, n_occ * n_virt]).unwrap())
                .into_shape([n_occ, n_virt])
                .unwrap();

    u_l
}

fn mult_av_nolc(
    gamma: ArrayView2<f64>,
    qtrans_ov: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    vs: ArrayView2<f64>,
    n_occ: usize,
    n_virt: usize,
) -> Array2<f64> {
    let n_at: usize = qtrans_ov.dim().0;
    let gamma_equiv: Array2<f64> = gamma.to_owned();

    // 1st term - KS orbital energy differences
    let mut u_l: Array2<f64> = &omega * &vs;

    // 2nd term - Coulomb
    u_l = u_l
        + 4.0
            * gamma_equiv
                .dot(
                    &qtrans_ov
                        .into_shape([n_at, n_occ * n_virt])
                        .unwrap()
                        .dot(&vs.into_shape(n_occ * n_virt).unwrap()),
                )
                .dot(&qtrans_ov.into_shape([n_at, n_occ * n_virt]).unwrap())
                .into_shape([n_occ, n_virt])
                .unwrap();

    u_l
}

fn mult_av_lc(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    qtrans_ov: ArrayView3<f64>,
    qtrans_oo_reshaped: ArrayView2<f64>,
    qtrans_vv_reshaped: ArrayView2<f64>,
    omega: ArrayView2<f64>,
    vs: ArrayView2<f64>,
    n_occ: usize,
    n_virt: usize,
) -> Array2<f64> {
    let n_at: usize = qtrans_ov.dim().0;
    let gamma_equiv: Array2<f64> = gamma.to_owned();

    // 1st term - KS orbital energy differences
    let mut u_l: Array2<f64> = &omega * &vs;

    // 2nd term - Coulomb
    u_l = u_l
        + 4.0
            * gamma_equiv
                .dot(
                    &qtrans_ov
                        .into_shape([n_at, n_occ * n_virt])
                        .unwrap()
                        .dot(&vs.into_shape(n_occ * n_virt).unwrap()),
                )
                .dot(&qtrans_ov.into_shape([n_at, n_occ * n_virt]).unwrap())
                .into_shape([n_occ, n_virt])
                .unwrap();

    // 3rd term - Exchange
    let tmp31: Array3<f64> = qtrans_vv_reshaped
        .dot(&vs.t())
        .into_shape((n_at, n_virt, n_occ))
        .unwrap();

    let tmp31_reshaped: Array2<f64> = tmp31.into_shape((n_at, n_virt * n_occ)).unwrap();
    let mut tmp32: Array3<f64> = gamma_lr
        .dot(&tmp31_reshaped)
        .into_shape((n_at, n_virt, n_occ))
        .unwrap();
    tmp32.swap_axes(1, 2);
    let tmp32 = tmp32.as_standard_layout();

    let tmp33: Array2<f64> = qtrans_oo_reshaped
        .t()
        .dot(&tmp32.into_shape((n_at * n_occ, n_virt)).unwrap());
    u_l = u_l - tmp33;

    u_l
}
