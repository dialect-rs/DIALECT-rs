//! FMO-xTB HOP gradient assembly.
//!
//! Entry point: `ground_state_gradient_fmo_xtb_hop` orchestrates the full gradient
//! computation using new HOP SCC structs (XtbMonomerHopScc, XtbPairHopScc, XtbTrimerHopScc).

pub mod helpers;
pub mod hop_projector;
pub mod interfragment;
pub mod monomer;
pub mod pair;
pub mod response;
pub mod trimer;

use dialect_utilities::fmo_helpers::get_pair_slice_xtb;
use dialect_state::PairType;
use dialect_utilities::mulliken::shell_to_ao_charges;
use crate::fmo::gradients::fmo_gradient_shell::calculate_coordination_number_gradients_parallel;
use crate::fmo::scc_hop::hop_data::{get_frag_shell_range, XtbHopData};
use crate::fmo::scc_hop::monomer::XtbMonomerHopScc;
use crate::fmo::scc_hop::pair::XtbPairHopScc;
use crate::fmo::scc_hop::trimer::XtbTrimerHopScc;
use crate::fmo::supersystem::XtbSuperSystem;
use crate::gradients::halogen_bonding::gradient_halogen_bonding_xtb;
use crate::gradients::helpers::gradient_disp3_xtb;
use crate::initialization::atom::XtbAtom;
use crate::initialization::basis::create_basis_set;
use crate::scc::gamma_matrix::{
    gamma_gradient_xtb_double_contracted, XtbGammaFunction,
};
use helpers::{
    build_monomer_local_to_global, build_pair_local_to_global, build_trimer_local_to_global,
    compute_ctmul_xtb_hop, compute_esp_q_shell_hop, compute_pref_gradient_fmo_xtb_hop,
    compute_shiftct_espgrad_xtb_hop, get_healed_bonds_for_pair, get_pair_ghost_bda_globals,
    get_partial_bonds_for_pair, get_trimer_ghost_bda_globals, grad_repulsive_energy_xtb_scaled,
    scatter_to_global,
};
use dialect_base::defaults::PROXIMITY_CUTOFF;
use crate::integrals::{calc_overlap_derivative_d_shells, obara_saika_derivatives_all};
use nalgebra::Vector3;
use ndarray::prelude::*;
use rayon::prelude::*;

/// Compute the full FMO-xTB HOP ground-state gradient.
///
/// Reads from new HOP SCC structs instead of `properties`.
/// Returns (explicit_gradient, cn_grad_global) for use by response gradient.
pub fn ground_state_gradient_fmo_xtb_hop(
    supersys: &XtbSuperSystem,
    hop_data: &XtbHopData,
    mono_states: &[XtbMonomerHopScc],
    pair_states: &[XtbPairHopScc],
    trimer_states: &[XtbTrimerHopScc],
    gammafunction: &XtbGammaFunction,
) -> (Array1<f64>, Array2<f64>) {
    let atoms: &[XtbAtom] = &supersys.atoms;
    let n_atoms_total = atoms.len();
    let n_grad = 3 * n_atoms_total;
    let n_frags = mono_states.len();
    let use_three_body = supersys.config.fmo.use_three_body;

    // Build frag_atom_ranges from supersystem monomers
    let frag_atom_ranges: Vec<std::ops::Range<usize>> = supersys
        .monomers
        .iter()
        .map(|m| m.slice.atom_as_range())
        .collect();

    // ========================================================================
    // Step 1: Coordination number gradients
    // ========================================================================
    let cn_numbers_global: ArrayView1<f64> = supersys.properties.cn().unwrap();
    let cn_grad_global: Array2<f64> = calculate_coordination_number_gradients_parallel(atoms);

    // ========================================================================
    // Step 2: Pair SCAL factors for FMO3
    // ========================================================================
    let pair_scal: Vec<f64> = if use_three_body {
        let mut scal = vec![1.0f64; pair_states.len()];
        for ts in trimer_states.iter() {
            for &(a, b) in &[(ts.i, ts.j), (ts.i, ts.k), (ts.j, ts.k)] {
                if supersys.properties.type_of_pair(a, b) == PairType::Pair {
                    let idx = supersys.properties.index_of_pair(a, b);
                    scal[idx] -= 1.0;
                }
            }
        }
        scal
    } else {
        vec![1.0f64; pair_states.len()]
    };

    // ========================================================================
    // Step 3: CTMUL (shell-level embedding Coulomb)
    // ========================================================================
    let ctmul_ext = compute_ctmul_xtb_hop(hop_data, mono_states, pair_states, trimer_states, &pair_scal);

    // ========================================================================
    // Step 4: SHIFTCT + ESPGRAD per monomer (parallel)
    // ========================================================================
    // SHIFTCT = gamma_ext × CTMUL - ESPGRAD. This is invariant to the Step 9
    // ctmul_for_interfrag modification because ESPGRAD subtracts the same terms.
    let shiftcts: Vec<Array1<f64>> = (0..n_frags)
        .into_par_iter()
        .map(|frag_idx| {
            compute_shiftct_espgrad_xtb_hop(
                frag_idx,
                hop_data,
                mono_states,
                pair_states,
                trimer_states,
                ctmul_ext.view(),
                &pair_scal,
            )
        })
        .collect();

    // ========================================================================
    // Step 5: ESP_Q per monomer
    // ========================================================================
    let esp_q_shells: Vec<Array1<f64>> = (0..n_frags)
        .into_par_iter()
        .map(|frag_idx| compute_esp_q_shell_hop(frag_idx, hop_data))
        .collect();

    // ========================================================================
    // Step 6: Monomer gradients (parallel)
    // ========================================================================
    let cn_grad_view = cn_grad_global.view();
    let monomer_results: Vec<(Array1<f64>, Array1<f64>, Array1<f64>)> = mono_states
        .par_iter()
        .enumerate()
        .map(|(frag_idx, mono)| {
            monomer::monomer_gradient_combined_xtb_hop(
                mono,
                hop_data,
                frag_idx,
                shiftcts[frag_idx].view(),
                esp_q_shells[frag_idx].view(),
                cn_numbers_global,
                cn_grad_view,
                gammafunction,
                n_atoms_total,
                frag_atom_ranges[frag_idx].clone(),
            )
        })
        .collect();

    // Scatter monomer gradients to global
    let mut monomer_grad_total = Array1::<f64>::zeros(n_grad);
    let mut addlag_total = Array1::<f64>::zeros(n_grad);
    let mut cn_grad_total = Array1::<f64>::zeros(n_grad);

    for (frag_idx, (mon_grad, add_grad, cn_glob)) in monomer_results.iter().enumerate() {
        let local_to_global =
            build_monomer_local_to_global(frag_atom_ranges[frag_idx].clone(), hop_data, frag_idx);
        let mut frag_mono_global = Array1::<f64>::zeros(n_grad);
        scatter_to_global(
            &mut frag_mono_global,
            mon_grad,
            &local_to_global,
            mono_states[frag_idx].n_ext_atoms,
        );
        let mut frag_add_global = Array1::<f64>::zeros(n_grad);
        scatter_to_global(
            &mut frag_add_global,
            add_grad,
            &local_to_global,
            mono_states[frag_idx].n_ext_atoms,
        );
        monomer_grad_total += &frag_mono_global;
        addlag_total += &frag_add_global;
        cn_grad_total += cn_glob;
    }

    // ========================================================================
    // Step 7: Pair gradients (parallel)
    // ========================================================================
    // Build pair local_to_global and ghost info
    let pair_mappings: Vec<(Vec<usize>, Vec<usize>)> = pair_states
        .iter()
        .map(|ps| {
            let ghost_bda = get_pair_ghost_bda_globals(hop_data, ps.i, ps.j);
            let local_to_global = build_pair_local_to_global(
                frag_atom_ranges[ps.i].clone(),
                frag_atom_ranges[ps.j].clone(),
                &ghost_bda,
            );
            (local_to_global, ghost_bda)
        })
        .collect();

    let pair_results: Vec<(Array1<f64>, Array1<f64>, Array1<f64>)> = pair_states
        .par_iter()
        .zip(pair_mappings.par_iter())
        .map(|(ps, (local_to_global, _ghost_bda))| {
            pair::pair_gradient_combined_xtb_hop(
                ps,
                hop_data,
                mono_states,
                n_atoms_total,
                gammafunction,
                cn_numbers_global,
                cn_grad_view,
                local_to_global,
            )
        })
        .collect();

    // ========================================================================
    // Step 8: Pair delta + CTIJ assembly
    // ========================================================================
    let mut pair_delta_total = Array1::<f64>::zeros(n_grad);
    let mut ctij_total = Array1::<f64>::zeros(n_grad);

    for (pair_idx, (ps, (pair_grad_local, ctij_grad_global, pair_cn_glob))) in
        pair_states.iter().zip(pair_results.iter()).enumerate()
    {
        let (local_to_global, ghost_bda) = &pair_mappings[pair_idx];
        let mon_i_grad = &monomer_results[ps.i].0;
        let mon_j_grad = &monomer_results[ps.j].0;
        let range_i = &frag_atom_ranges[ps.i];
        let range_j = &frag_atom_ranges[ps.j];
        let n_real_i = range_i.len();
        let n_real_j = range_j.len();

        // I's real atoms: pair[local] - mono_I[local]
        for (local_idx, global_idx) in range_i.clone().enumerate() {
            for k in 0..3 {
                pair_delta_total[3 * global_idx + k] +=
                    pair_grad_local[3 * local_idx + k] - mon_i_grad[3 * local_idx + k];
            }
        }
        // J's real atoms
        for (local_idx, global_idx) in range_j.clone().enumerate() {
            for k in 0..3 {
                pair_delta_total[3 * global_idx + k] +=
                    pair_grad_local[3 * (n_real_i + local_idx) + k]
                        - mon_j_grad[3 * local_idx + k];
            }
        }
        // Pair ghost atoms → BDA global
        for (ghost_idx, &bda_global) in ghost_bda.iter().enumerate() {
            let pair_ghost_local = n_real_i + n_real_j + ghost_idx;
            for k in 0..3 {
                pair_delta_total[3 * bda_global + k] +=
                    pair_grad_local[3 * pair_ghost_local + k];
            }
        }

        // Subtract ALL ghost contributions from mono_I and mono_J.
        // Ghost atoms in monomer X are for bonds where baa_fragment == X.
        // Each ghost is at bda_global position.
        for &frag_idx in &[ps.i, ps.j] {
            let mono_grad = &monomer_results[frag_idx].0;
            let n_real = frag_atom_ranges[frag_idx].len();
            let mut ghost_count = 0;
            for db in &hop_data.detached_bonds {
                if db.baa_fragment == frag_idx {
                    let local_idx = n_real + ghost_count;
                    for k in 0..3 {
                        pair_delta_total[3 * db.bda_global + k] -=
                            mono_grad[3 * local_idx + k];
                    }
                    ghost_count += 1;
                }
            }
        }

        let scal = pair_scal[pair_idx];
        ctij_total += &(scal * ctij_grad_global);

        let mon_i_cn = &monomer_results[ps.i].2;
        let mon_j_cn = &monomer_results[ps.j].2;
        cn_grad_total += &(pair_cn_glob - mon_i_cn - mon_j_cn);
    }

    // Note: addlag is monomer-only (pairs don't compute addlag).
    // Ghost addlag contributions are NOT deltad against pairs.

    // ========================================================================
    // Step 9: Inter-fragment gradient
    // ========================================================================
    // For FMO3: subtract trimer ghost CTMUL from ctmul_ext for the interfrag gradient.
    // The trimer's CTIJK gradient (in ctijk_grad_global) only covers real shells of the trimer.
    // But compute_ctmul_xtb_hop adds ghost CTMUL for bonds healed in trimers.
    // These ghost CTMUL values create unbalanced forces in the interfrag gradient.
    let ctmul_for_interfrag = if use_three_body && !trimer_states.is_empty() {
        use helpers::shells_per_atom_in_range;
        let mut ctmul_mod = ctmul_ext.clone();
        for ts in trimer_states.iter() {
            let tri_frags = [ts.i, ts.j, ts.k];
            let tri_fi = [&hop_data.frag_info[ts.i], &hop_data.frag_info[ts.j], &hop_data.frag_info[ts.k]];
            let tri_sr: Vec<std::ops::Range<usize>> = tri_fi.iter()
                .map(|f| get_frag_shell_range(&hop_data.ext_basis, &f.ext_range)).collect();
            let tri_spa: Vec<Vec<usize>> = tri_fi.iter()
                .map(|f| shells_per_atom_in_range(&hop_data.ext_basis, &f.ext_range)).collect();

            for (fp, bond) in hop_data.detached_bonds.iter().enumerate() {
                for frag_pos in 0..3 {
                    if bond.baa_fragment == tri_frags[frag_pos] {
                        let bda_in_tri = bond.bda_fragment == tri_frags[0]
                            || bond.bda_fragment == tri_frags[1]
                            || bond.bda_fragment == tri_frags[2];

                        let n_real = tri_fi[frag_pos].n_real_atoms;
                        // Count ghost index for this frag_pos
                        let mut ghost_count = 0;
                        for b2 in &hop_data.detached_bonds {
                            if b2.baa_fragment == tri_frags[frag_pos] {
                                if std::ptr::eq(b2, bond) { break; }
                                ghost_count += 1;
                            }
                        }
                        let n_ghost_shells = tri_spa[frag_pos][n_real + ghost_count];
                        let ghost_shell_start = tri_sr[frag_pos].start
                            + tri_fi[frag_pos].n_real_shells
                            + (0..ghost_count).map(|g| tri_spa[frag_pos][n_real + g]).sum::<usize>();

                        // Subtract ghost CTMUL that was added by the trimer
                        for s in 0..n_ghost_shells {
                            let gs = ghost_shell_start + s;
                            let mono_dq = mono_states[tri_frags[frag_pos]].dq_shell
                                [tri_fi[frag_pos].n_real_shells
                                + (0..ghost_count).map(|g| tri_spa[frag_pos][n_real + g]).sum::<usize>()
                                + s];
                            let ghost_ctij = if bda_in_tri {
                                -mono_dq // healed in trimer, DQ convention
                            } else {
                                // Partial: pair_dq - mono_dq (complex, skip for now)
                                continue;
                            };
                            ctmul_mod[gs] -= ghost_ctij;
                        }
                        break; // each bond only in one frag_pos
                    }
                }
            }
        }
        ctmul_mod
    } else {
        ctmul_ext.clone()
    };
    let trimer_frags_ifrag: Vec<(usize, usize, usize)> = if use_three_body {
        trimer_states.iter().map(|ts| (ts.i, ts.j, ts.k)).collect()
    } else { Vec::new() };
    let interfrag_grad = interfragment::interfragment_gradient_xtb_hop(
        atoms,
        hop_data,
        mono_states,
        &supersys.esd_pairs,
        ctmul_for_interfrag.view(),
        gammafunction,
        &trimer_frags_ifrag,
    );

    // ========================================================================
    // Step 10: Dispersion + Halogen
    // ========================================================================
    let disp_grad = gradient_disp3_xtb(&supersys.atoms, &supersys.config);
    let halogen_grad = gradient_halogen_bonding_xtb(&supersys.atoms);

    // ========================================================================
    // Step 10b: HOP projector gradient (moved before trimers for FMO3 per-pair HOP delta)
    // ========================================================================
    let trimer_frags_hop: Vec<(usize, usize, usize)> = if use_three_body {
        trimer_states.iter().map(|ts| (ts.i, ts.j, ts.k)).collect()
    } else { Vec::new() };
    let (hop_grad, _hop_mono_grad, _hop_pair_delta_grad, _per_pair_hop_delta) = if !hop_data.detached_bonds.is_empty() {
        let mut hop_total = Array1::<f64>::zeros(n_grad);
        let (hop_mono, hop_pd, ppd) = hop_projector::compute_hop_gradient_fmo_xtb_hop(
            hop_data,
            mono_states,
            pair_states,
            atoms,
            &mut hop_total,
            &frag_atom_ranges,
            &pair_scal,
            &trimer_frags_hop,
            trimer_states,
        );
        (hop_total, hop_mono, hop_pd, ppd)
    } else {
        let empty: Vec<Array1<f64>> = pair_states.iter().map(|_| Array1::zeros(n_grad)).collect();
        (Array1::zeros(n_grad), Array1::zeros(n_grad), Array1::zeros(n_grad), empty)
    };

    // ========================================================================
    // Step 11: FMO3 three-body correction
    // ========================================================================
    let trimer_contribution = if use_three_body && !trimer_states.is_empty() {
        // Build trimer local_to_global and ghost info
        let trimer_mappings: Vec<(Vec<usize>, Vec<usize>)> = trimer_states
            .iter()
            .map(|ts| {
                let ghost_bda = get_trimer_ghost_bda_globals(hop_data, ts.i, ts.j, ts.k);
                let local_to_global = build_trimer_local_to_global(
                    frag_atom_ranges[ts.i].clone(),
                    frag_atom_ranges[ts.j].clone(),
                    frag_atom_ranges[ts.k].clone(),
                    &ghost_bda,
                );
                (local_to_global, ghost_bda)
            })
            .collect();

        let trimer_results: Vec<(Array1<f64>, Array1<f64>, Array1<f64>)> = trimer_states
            .par_iter()
            .zip(trimer_mappings.par_iter())
            .map(|(ts, (local_to_global, _ghost_bda))| {
                trimer::trimer_gradient_combined_xtb_hop(
                    ts,
                    hop_data,
                    mono_states,
                    n_atoms_total,
                    gammafunction,
                    cn_numbers_global,
                    cn_grad_view,
                    local_to_global,
                )
            })
            .collect();

        // ESD pair gradients for trimer subtraction (dgamma/dR term only)
        let esd_pair_grads: Vec<Array1<f64>> = supersys
            .esd_pairs
            .par_iter()
            .map(|esd_pair| {
                // ESD pair gradient: shell-level computation matching the interfragment
                // gradient approach. Using gamma_gradient_xtb_double_contracted with
                // AO-expanded shell charges would overcount by n_orbs_i × n_orbs_j.
                let mono_i = &mono_states[esd_pair.i];
                let mono_j = &mono_states[esd_pair.j];
                let range_i = &frag_atom_ranges[esd_pair.i];
                let range_j = &frag_atom_ranges[esd_pair.j];
                let n_real_i = range_i.len();
                let n_real_j = range_j.len();

                // Shell-level ESD gradient: dq_I × dq_J × dgamma/dR
                let fi_i = &hop_data.frag_info[esd_pair.i];
                let fi_j = &hop_data.frag_info[esd_pair.j];
                let sr_i = get_frag_shell_range(&hop_data.ext_basis, &fi_i.ext_range);
                let sr_j = get_frag_shell_range(&hop_data.ext_basis, &fi_j.ext_range);

                let mut grad_gamma_i = Array1::<f64>::zeros(3 * n_real_i);
                let mut grad_gamma_j = Array1::<f64>::zeros(3 * n_real_j);

                for (si_local, si_global) in (sr_i.start..sr_i.start + fi_i.n_real_shells).enumerate() {
                    let shell_i = &hop_data.ext_basis.shells[si_global];
                    let at_i = shell_i.atom_index - fi_i.ext_range.start;
                    let atomi = &hop_data.ext_atoms[shell_i.atom_index];
                    let dq_si = mono_i.dq_shell[si_local];

                    for (sj_local, sj_global) in (sr_j.start..sr_j.start + fi_j.n_real_shells).enumerate() {
                        let shell_j = &hop_data.ext_basis.shells[sj_global];
                        let at_j = shell_j.atom_index - fi_j.ext_range.start;
                        let atomj = &hop_data.ext_atoms[shell_j.atom_index];
                        let dq_sj = mono_j.dq_shell[sj_local];

                        let dx = atomi.xyz[0] - atomj.xyz[0];
                        let dy = atomi.xyz[1] - atomj.xyz[1];
                        let dz = atomi.xyz[2] - atomj.xyz[2];
                        let dist = (dx*dx + dy*dy + dz*dz).sqrt();
                        if dist < 1e-10 { continue; }
                        let inv = 1.0 / dist;

                        let gd = gammafunction.deriv(
                            dist, atomi.number, shell_i.angular_momentum as u8,
                            atomj.number, shell_j.angular_momentum as u8,
                        );

                        let val = gd * dq_si * dq_sj * inv;
                        grad_gamma_i[3*at_i]   += val * dx;
                        grad_gamma_i[3*at_i+1] += val * dy;
                        grad_gamma_i[3*at_i+2] += val * dz;
                        grad_gamma_j[3*at_j]   -= val * dx;
                        grad_gamma_j[3*at_j+1] -= val * dy;
                        grad_gamma_j[3*at_j+2] -= val * dz;
                    }
                }
                let mut ij_grad =
                    Array1::<f64>::zeros(3 * n_real_i + 3 * range_j.len());
                ij_grad
                    .slice_mut(s![..3 * n_real_i])
                    .assign(&grad_gamma_i);
                ij_grad
                    .slice_mut(s![3 * n_real_i..])
                    .assign(&grad_gamma_j);
                ij_grad
            })
            .collect();

        // Per-pair delta gradients in global coords (needed for trimer subtraction)
        let per_pair_delta_global: Vec<Array1<f64>> = pair_states
            .par_iter()
            .zip(pair_results.par_iter())
            .zip(pair_mappings.par_iter())
            .map(
                |((ps, (pair_grad_local, _ctij, pair_cn_glob)), (_, ghost_bda))| {
                    let mon_i_grad = &monomer_results[ps.i].0;
                    let mon_j_grad = &monomer_results[ps.j].0;
                    let range_i = &frag_atom_ranges[ps.i];
                    let range_j = &frag_atom_ranges[ps.j];
                    let n_real_i = range_i.len();
                    let n_real_j = range_j.len();

                    let mut delta = Array1::<f64>::zeros(n_grad);
                    for (local_idx, global_idx) in range_i.clone().enumerate() {
                        for k in 0..3 {
                            delta[3 * global_idx + k] = pair_grad_local[3 * local_idx + k]
                                - mon_i_grad[3 * local_idx + k];
                        }
                    }
                    for (local_idx, global_idx) in range_j.clone().enumerate() {
                        for k in 0..3 {
                            delta[3 * global_idx + k] =
                                pair_grad_local[3 * (n_real_i + local_idx) + k]
                                    - mon_j_grad[3 * local_idx + k];
                        }
                    }
                    // Ghost contributions
                    for (ghost_idx, &bda_global) in ghost_bda.iter().enumerate() {
                        let pair_ghost_local = n_real_i + n_real_j + ghost_idx;
                        for k in 0..3 {
                            delta[3 * bda_global + k] +=
                                pair_grad_local[3 * pair_ghost_local + k];
                        }
                    }
                    // Subtract ALL ghost contributions from mono_I and mono_J
                    for &fi in &[ps.i, ps.j] {
                        subtract_mono_ghost_grad_all(
                            fi,
                            &monomer_results,
                            &frag_atom_ranges,
                            hop_data,
                            &mut delta,
                        );
                    }

                    let mon_i_cn = &monomer_results[ps.i].2;
                    let mon_j_cn = &monomer_results[ps.j].2;
                    delta += &(pair_cn_glob - mon_i_cn - mon_j_cn);
                    delta
                },
            )
            .collect();

        // Trimer delta + CTIJK assembly
        let mut trimer_delta_total = Array1::<f64>::zeros(n_grad);
        let mut ctijk_total = Array1::<f64>::zeros(n_grad);

        for (tri_idx, (ts, (tri_grad_local, tri_ctijk_glob, tri_cn))) in
            trimer_states.iter().zip(trimer_results.iter()).enumerate()
        {
            let (_, ghost_bda) = &trimer_mappings[tri_idx];
            let mon_i_grad = &monomer_results[ts.i].0;
            let mon_j_grad = &monomer_results[ts.j].0;
            let mon_k_grad = &monomer_results[ts.k].0;
            let range_i = &frag_atom_ranges[ts.i];
            let range_j = &frag_atom_ranges[ts.j];
            let range_k = &frag_atom_ranges[ts.k];
            let n_ri = range_i.len();
            let n_rj = range_j.len();
            let n_rk = range_k.len();

            let mut delta_global = Array1::<f64>::zeros(n_grad);
            for (local_idx, global_idx) in range_i.clone().enumerate() {
                for k in 0..3 {
                    delta_global[3 * global_idx + k] =
                        tri_grad_local[3 * local_idx + k] - mon_i_grad[3 * local_idx + k];
                }
            }
            for (local_idx, global_idx) in range_j.clone().enumerate() {
                for k in 0..3 {
                    delta_global[3 * global_idx + k] =
                        tri_grad_local[3 * (n_ri + local_idx) + k]
                            - mon_j_grad[3 * local_idx + k];
                }
            }
            for (local_idx, global_idx) in range_k.clone().enumerate() {
                for k in 0..3 {
                    delta_global[3 * global_idx + k] =
                        tri_grad_local[3 * (n_ri + n_rj + local_idx) + k]
                            - mon_k_grad[3 * local_idx + k];
                }
            }
            // Ghost contributions
            for (ghost_idx, &bda_global) in ghost_bda.iter().enumerate() {
                let tri_ghost_local = n_ri + n_rj + n_rk + ghost_idx;
                for k in 0..3 {
                    delta_global[3 * bda_global + k] +=
                        tri_grad_local[3 * tri_ghost_local + k];
                }
            }
            // Subtract ALL ghost contributions from mono_I, mono_J, mono_K
            for &fi in &[ts.i, ts.j, ts.k] {
                subtract_mono_ghost_grad_all(
                    fi,
                    &monomer_results,
                    &frag_atom_ranges,
                    hop_data,
                    &mut delta_global,
                );
            }

            // W-correction: compensate for V_HOP eigenvalue effect in monomer W.
            // Skip if V_HOP eigenvalue shifts are negligible (< 1e-6), which is
            // typical since V_HOP only affects virtual orbitals with f=0.
            // For C800H1602 (40 frags), this saves ~4s by avoiding O(n_orbs² × n_shells²)
            // computation that produces corrections of ~1e-10 magnitude.
            for &frag_idx in &[ts.i, ts.j, ts.k] {
                let mono = &mono_states[frag_idx];
                if let Some(ref p_hop) = mono.p_hop {
                    let orbs = mono.orbs.as_ref().unwrap();
                    let n_ext_orbs = mono.n_ext_orbs;

                    // Quick check: compute max |Δε_i| for occupied orbitals
                    let c_vhop = p_hop.dot(orbs);
                    let max_delta_eps: f64 = mono.f.iter().enumerate()
                        .filter(|(_, &fi)| fi > 0.5)
                        .map(|(i, _)| orbs.column(i).dot(&c_vhop.column(i)).abs())
                        .fold(0.0f64, f64::max);

                    if max_delta_eps > 1e-6 {
                        // Full W-correction (only if HOP significantly affects occupied orbitals)
                        let mut delta_eps = Array1::<f64>::zeros(n_ext_orbs);
                        for i in 0..n_ext_orbs {
                            delta_eps[i] = orbs.column(i).dot(&c_vhop.column(i));
                        }
                        let mut delta_w = Array2::<f64>::zeros([n_ext_orbs, n_ext_orbs]);
                        for (i, &fi) in mono.f.iter().enumerate() {
                            if fi > 0.5 && delta_eps[i].abs() > 1e-10 {
                                let de = fi * delta_eps[i];
                                for mu in 0..n_ext_orbs {
                                    for nu in 0..n_ext_orbs {
                                        delta_w[[mu, nu]] += de * orbs[[mu, i]] * orbs[[nu, i]];
                                    }
                                }
                            }
                        }
                        let mono_local = &mono.ext_atoms;
                        let n_atoms_mono = mono.n_ext_atoms;
                        let mut w_corr_local = Array1::<f64>::zeros(3 * n_atoms_mono);
                        for (si_idx, shell_i) in mono.basis.shells.iter().enumerate() {
                            let at_i = shell_i.atom_index;
                            let atomi = &mono_local[at_i];
                            for (sj_idx, shell_j) in mono.basis.shells.iter().enumerate() {
                                let at_j = shell_j.atom_index;
                                if at_i == at_j { continue; }
                                let atomj = &mono_local[at_j];
                                let r_vec: Vector3<f64> = atomi - atomj;
                                let dist = r_vec.norm();
                                if dist >= PROXIMITY_CUTOFF { continue; }
                                let is_same = shell_i.sph_start == shell_j.sph_start
                                    && shell_i.sph_end == shell_j.sph_end;
                                if is_same { continue; }
                                if shell_i.angular_momentum < 2 && shell_j.angular_momentum < 2 {
                                    for idx_i in shell_i.sph_start..shell_i.sph_end {
                                        let il = idx_i - shell_i.sph_start;
                                        for idx_j in shell_j.sph_start..shell_j.sph_end {
                                            let jl = idx_j - shell_j.sph_start;
                                            if idx_i != idx_j {
                                                let w_ij = delta_w[[idx_i, idx_j]];
                                                if w_ij.abs() < 1e-20 { continue; }
                                                let o1 = &mono.basis.basis_functions[shell_i.start + il];
                                                let o2 = &mono.basis.basis_functions[shell_j.start + jl];
                                                let np = o1.contracted_norm * o2.contracted_norm;
                                                let ds = obara_saika_derivatives_all(o1, o2);
                                                for dir in 0..3 {
                                                    w_corr_local[3 * at_i + dir] += ds[dir] * np * w_ij;
                                                    w_corr_local[3 * at_j + dir] -= ds[dir] * np * w_ij;
                                                }
                                            }
                                        }
                                    }
                                }
                                let either_d = shell_i.angular_momentum >= 2 || shell_j.angular_momentum >= 2;
                                if either_d && si_idx < sj_idx {
                                    let ds_d = calc_overlap_derivative_d_shells(&mono.basis, shell_i, shell_j);
                                    let sph_i = shell_i.sph_end - shell_i.sph_start;
                                    let sph_j = shell_j.sph_end - shell_j.sph_start;
                                    for si in 0..sph_i {
                                        let ii = shell_i.sph_start + si;
                                        for sj in 0..sph_j {
                                            let jj = shell_j.sph_start + sj;
                                            let w_ij = delta_w[[ii, jj]];
                                            for dir in 0..3 {
                                                w_corr_local[3 * at_i + dir] += 2.0 * ds_d[[dir, si, sj]] * w_ij;
                                                w_corr_local[3 * at_j + dir] += 2.0 * ds_d[[3 + dir, si, sj]] * w_ij;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        let ltg = build_monomer_local_to_global(
                            frag_atom_ranges[frag_idx].clone(), hop_data, frag_idx,
                        );
                        for local_idx in 0..n_atoms_mono {
                            let global_idx = ltg[local_idx];
                            for k in 0..3 {
                                delta_global[3 * global_idx + k] += w_corr_local[3 * local_idx + k];
                            }
                        }
                    }
                }
            }

            delta_global += &(tri_cn
                - &monomer_results[ts.i].2
                - &monomer_results[ts.j].2
                - &monomer_results[ts.k].2);

            // Subtract pair deltas and ESD pair grads
            for &(a, b) in &[(ts.i, ts.j), (ts.i, ts.k), (ts.j, ts.k)] {
                if supersys.properties.type_of_pair(a, b) == PairType::Pair {
                    let idx = supersys.properties.index_of_pair(a, b);
                    delta_global -= &per_pair_delta_global[idx];
                } else {
                    let idx = supersys.properties.index_of_esd_pair(a, b);
                    let esd_grad_local = &esd_pair_grads[idx];
                    let range_a = &frag_atom_ranges[a];
                    let range_b = &frag_atom_ranges[b];
                    let n_ra = range_a.len();
                    let mut esd_global = Array1::<f64>::zeros(n_grad);
                    for (local_idx, global_idx) in range_a.clone().enumerate() {
                        for k in 0..3 {
                            esd_global[3 * global_idx + k] =
                                esd_grad_local[3 * local_idx + k];
                        }
                    }
                    for (local_idx, global_idx) in range_b.clone().enumerate() {
                        for k in 0..3 {
                            esd_global[3 * global_idx + k] =
                                esd_grad_local[3 * (n_ra + local_idx) + k];
                        }
                    }
                    delta_global -= &esd_global;
                }
            }

            trimer_delta_total += &delta_global;
            ctijk_total += tri_ctijk_glob;
        }

        &trimer_delta_total + &ctijk_total
    } else {
        Array1::zeros(n_grad)
    };

    // ========================================================================
    // Step 12: HOP projector gradient
    // ========================================================================
    // (HOP projector gradient already computed in Step 10b above)

    // ========================================================================
    // Step 12b: P_ref derivative gradient (d(P_ref)/dR through Coulomb)
    // ========================================================================
    let pref_grad = if !hop_data.detached_bonds.is_empty() {
        let g = compute_pref_gradient_fmo_xtb_hop(
            hop_data,
            mono_states,
            pair_states,
            atoms,
            &frag_atom_ranges,
            n_atoms_total,
        );
        g
    } else {
        Array1::zeros(n_grad)
    };

    // ========================================================================
    // Step 13: Assembly
    // ========================================================================
    let total_gradient = &monomer_grad_total
        + &pair_delta_total
        + &ctij_total
        + &interfrag_grad
        + &addlag_total
        + &disp_grad
        + &halogen_grad
        + &cn_grad_total
        + &trimer_contribution
        + &hop_grad
        + &pref_grad;

    (total_gradient, cn_grad_global)
}

/// Subtract ALL ghost gradient contributions from a monomer.
///
/// Ghost atoms in monomer `frag_idx` exist for bonds where `baa_fragment == frag_idx`.
/// Each ghost's gradient is at local index `n_real + ghost_count`, scattered to `bda_global`.
fn subtract_mono_ghost_grad_all(
    frag_idx: usize,
    monomer_results: &[(Array1<f64>, Array1<f64>, Array1<f64>)],
    frag_atom_ranges: &[std::ops::Range<usize>],
    hop_data: &XtbHopData,
    delta: &mut Array1<f64>,
) {
    let mono_grad = &monomer_results[frag_idx].0;
    let n_real = frag_atom_ranges[frag_idx].len();
    let mut ghost_count = 0;
    for db in &hop_data.detached_bonds {
        if db.baa_fragment == frag_idx {
            let local_idx = n_real + ghost_count;
            for k in 0..3 {
                delta[3 * db.bda_global + k] -= mono_grad[3 * local_idx + k];
            }
            ghost_count += 1;
        }
    }
}

/// Get partial bonds for a trimer (BAA inside trimer, BDA outside).
fn get_partial_bonds_for_trimer<'a>(
    hop_data: &'a XtbHopData,
    i: usize,
    j: usize,
    k: usize,
) -> Vec<&'a crate::hop::DetachedBond> {
    hop_data
        .detached_bonds
        .iter()
        .filter(|b| {
            let baa_in =
                b.baa_fragment == i || b.baa_fragment == j || b.baa_fragment == k;
            let bda_in =
                b.bda_fragment == i || b.bda_fragment == j || b.bda_fragment == k;
            baa_in && !bda_in
        })
        .collect()
}
