use dialect_utilities::fmo_helpers::get_pair_slice_xtb;
use dialect_utilities::numerical::{assert_deriv, assert_deriv_5point, assert_deriv_fd};
use dialect_utilities::scc_interface::RestrictedSCC;
use crate::fmo::supersystem::XtbSuperSystem;
use crate::fmo::scc_hop::hop_data::calculate_repulsive_energy_xtb_scaled;
use crate::initialization::atom::XtbAtom;
use crate::scc::scc_helpers::get_dispersion_energy_xtb;
use ndarray::{s, Array1};
use std::time::Instant;

/// Per-component energies from HOP SCC (for numerical gradient decomposition).
#[derive(Debug, Clone)]
struct HopComponentEnergies {
    mono: Vec<f64>,       // per-monomer total (SCC + rep)
    pair: Vec<f64>,       // per-pair total (SCC + rep)
    mono_rep: Vec<f64>,   // per-monomer repulsive
    pair_rep: Vec<f64>,   // per-pair repulsive
    hop_mono: Vec<f64>,   // per-monomer Tr(P * V_HOP)
    hop_pair: Vec<f64>,   // per-pair Tr(P * V_HOP)
    embedding: f64,
    esd: f64,
    disp: f64,
    trimer_delta: f64,    // FMO3 trimer delta energy
    trimer_embedding: f64, // FMO3 trimer embedding correction
}

impl XtbSuperSystem<'_> {
    /// Run HOP SCC and return per-component energies for numerical decomposition.
    fn hop_component_energies(&mut self, geometry: Array1<f64>) -> HopComponentEnergies {
        use crate::fmo::scc_hop::hop_data::build_xtb_hop_data;
        self.properties.reset_full();
        for mol in self.monomers.iter_mut() { mol.properties.reset_full(); }
        for pair in self.pairs.iter_mut() { pair.properties.reset_full(); }
        for trimer in self.trimers.iter_mut() { trimer.properties.reset_full(); }
        self.update_xyz(geometry.view());

        let max_iter = self.config.scf.scf_max_cycles;
        let mut hop_data = build_xtb_hop_data(self);
        let (_mono_e, mono_states) = self.monomer_scc_hop(max_iter, &mut hop_data);
        let (_pair_delta, pair_states) = self.pair_scc_hop(&hop_data, &mono_states);
        let embedding = self.embedding_energy_hop(&hop_data, &mono_states, &pair_states);
        let esd = self.esd_energy_hop(&hop_data);
        let disp = get_dispersion_energy_xtb(&self.atoms, &self.config);

        let mono: Vec<f64> = mono_states.iter().map(|m| m.last_energy).collect();
        let pair: Vec<f64> = pair_states.iter().map(|p| p.last_energy).collect();

        let mono_rep: Vec<f64> = mono_states.iter().enumerate().map(|(fi, ms)| {
            let finfo = &hop_data.frag_info[fi];
            let zref = hop_data.zref.slice(s![finfo.ext_range.clone()]);
            let qref = hop_data.qref.slice(s![finfo.ext_range.clone()]);
            calculate_repulsive_energy_xtb_scaled(&ms.ext_atoms, zref, qref)
        }).collect();
        let pair_rep: Vec<f64> = pair_states.iter().map(|ps| {
            calculate_repulsive_energy_xtb_scaled(&ps.ext_atoms, ps.zref.view(), ps.qref.view())
        }).collect();

        let hop_mono: Vec<f64> = mono_states.iter().map(|m| {
            m.p_hop.as_ref().map_or(0.0, |ph| (&m.p * ph).sum())
        }).collect();
        let hop_pair: Vec<f64> = pair_states.iter().map(|p| {
            p.p_hop.as_ref().map_or(0.0, |ph| (&p.p * ph).sum())
        }).collect();

        // Trimer components (FMO3)
        let (trimer_delta, trimer_embedding) = if self.config.fmo.use_three_body {
            let (td, ts) = self.trimer_scc_hop(&hop_data, &mono_states, &pair_states);
            let te = self.trimer_embedding_energy_hop(&hop_data, &mono_states, &pair_states, &ts);
            (td, te)
        } else {
            (0.0, 0.0)
        };

        HopComponentEnergies { mono, pair, mono_rep, pair_rep, hop_mono, hop_pair, embedding, esd, disp, trimer_delta, trimer_embedding }
    }

    pub fn gs_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset_full();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset_full();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset_full();
        }
        for trimer in self.trimers.iter_mut() {
            trimer.properties.reset_full();
        }
        self.update_xyz(geometry.view());
        if self.config.fmo.covalent_fragmentation {
            self.run_scc_hop().unwrap()
        } else {
            self.prepare_scc();
            self.run_scc().unwrap()
        }
    }

    pub fn gs_gradient_wrapper(&mut self) -> Array1<f64> {
        self.properties.reset_full();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset_full();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset_full();
        }
        for trimer in self.trimers.iter_mut() {
            trimer.properties.reset_full();
        }
        if self.config.fmo.covalent_fragmentation {
            // HOP: ground_state_gradient_hop() runs SCC internally
            let timer = Instant::now();
            let gs_grad = self.ground_state_gradient();
            println!(
                "Time ground-state gradient: {:.5}",
                timer.elapsed().as_secs_f32()
            );
            gs_grad
        } else {
            self.prepare_scc();
            let _gs_energy: f64 = self.run_scc().unwrap();
            let timer = Instant::now();
            let gs_grad = self.ground_state_gradient();
            println!(
                "Time ground-state gradient: {:.5}",
                timer.elapsed().as_secs_f32()
            );
            gs_grad
        }
    }

    pub fn test_gs_gradient(&mut self) {
        self.properties.reset_full();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset_full();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset_full();
        }
        for trimer in self.trimers.iter_mut() {
            trimer.properties.reset_full();
        }

        // Targeted component test: HOP per-component numerical gradients
        if self.config.fmo.covalent_fragmentation {
            let xyz = self.get_xyz();
            let h = 0.01;
            // Test selected coordinates (first BDA atom x,z and some others if they exist)
            let n_coords = 3 * self.atoms.len();
            let mut test_coords: Vec<usize> = vec![0, 2];
            if n_coords > 12 { test_coords.push(12); test_coords.push(14); }
            if n_coords > 21 { test_coords.push(21); test_coords.push(23); }
            if n_coords > 68 { test_coords.push(68); }
            test_coords.retain(|&c| c < n_coords);
            eprintln!("\n=== HOP COMPONENT NUMERICAL GRADIENT TEST ===");
            for &ci in &test_coords {
                let mut xyz_p = xyz.clone();
                let mut xyz_m = xyz.clone();
                xyz_p[ci] += h;
                xyz_m[ci] -= h;
                let comp_p = self.hop_component_energies(xyz_p);
                let comp_m = self.hop_component_energies(xyz_m);
                let dh = 2.0 * h;

                eprintln!("NUMCOMP coord={}", ci);
                // Per-monomer total (SCC + rep)
                for i in 0..comp_p.mono.len() {
                    eprintln!("  mono_{} {:+.10e}", i, (comp_p.mono[i] - comp_m.mono[i]) / dh);
                }
                // Per-pair total (SCC + rep)
                for i in 0..comp_p.pair.len() {
                    eprintln!("  pair_{} {:+.10e}", i, (comp_p.pair[i] - comp_m.pair[i]) / dh);
                }
                // Per-monomer repulsive
                for i in 0..comp_p.mono_rep.len() {
                    eprintln!("  mono_rep_{} {:+.10e}", i, (comp_p.mono_rep[i] - comp_m.mono_rep[i]) / dh);
                }
                // Per-pair repulsive
                for i in 0..comp_p.pair_rep.len() {
                    eprintln!("  pair_rep_{} {:+.10e}", i, (comp_p.pair_rep[i] - comp_m.pair_rep[i]) / dh);
                }
                // Per-monomer HOP energy
                for i in 0..comp_p.hop_mono.len() {
                    eprintln!("  hop_mono_{} {:+.10e}", i, (comp_p.hop_mono[i] - comp_m.hop_mono[i]) / dh);
                }
                // Per-pair HOP energy
                for i in 0..comp_p.hop_pair.len() {
                    eprintln!("  hop_pair_{} {:+.10e}", i, (comp_p.hop_pair[i] - comp_m.hop_pair[i]) / dh);
                }
                // Embedding
                eprintln!("  embedding {:+.10e}", (comp_p.embedding - comp_m.embedding) / dh);
                // ESD
                eprintln!("  esd {:+.10e}", (comp_p.esd - comp_m.esd) / dh);
                // Dispersion
                eprintln!("  disp {:+.10e}", (comp_p.disp - comp_m.disp) / dh);
                // Trimer components (FMO3)
                eprintln!("  trimer_delta {:+.10e}", (comp_p.trimer_delta - comp_m.trimer_delta) / dh);
                eprintln!("  trimer_embedding {:+.10e}", (comp_p.trimer_embedding - comp_m.trimer_embedding) / dh);
                eprintln!("  total_embedding {:+.10e}", ((comp_p.embedding + comp_p.trimer_embedding) - (comp_m.embedding + comp_m.trimer_embedding)) / dh);

                // Derived FMO quantities
                let mono_total_p: f64 = comp_p.mono.iter().sum();
                let mono_total_m: f64 = comp_m.mono.iter().sum();
                eprintln!("  monomer_total {:+.10e}", (mono_total_p - mono_total_m) / dh);

                // Pair delta = sum over pairs (E_pair - E_mono_I - E_mono_J)
                let n_mono = comp_p.mono.len();
                let mut pair_delta_p = 0.0f64;
                let mut pair_delta_m = 0.0f64;
                let mut pair_idx = 0;
                for i in 0..n_mono {
                    for j in (i+1)..n_mono {
                        if pair_idx < comp_p.pair.len() {
                            pair_delta_p += comp_p.pair[pair_idx] - comp_p.mono[i] - comp_p.mono[j];
                            pair_delta_m += comp_m.pair[pair_idx] - comp_m.mono[i] - comp_m.mono[j];
                        }
                        pair_idx += 1;
                    }
                }
                eprintln!("  pair_delta_total {:+.10e}", (pair_delta_p - pair_delta_m) / dh);

                // HOP delta = sum_mono(hop_mono) + sum over pairs (hop_pair - hop_mono_I - hop_mono_J)
                let mut hop_delta_p: f64 = comp_p.hop_mono.iter().sum();
                let mut hop_delta_m: f64 = comp_m.hop_mono.iter().sum();
                pair_idx = 0;
                for i in 0..n_mono {
                    for j in (i+1)..n_mono {
                        if pair_idx < comp_p.hop_pair.len() {
                            hop_delta_p += comp_p.hop_pair[pair_idx] - comp_p.hop_mono[i] - comp_p.hop_mono[j];
                            hop_delta_m += comp_m.hop_pair[pair_idx] - comp_m.hop_mono[i] - comp_m.hop_mono[j];
                        }
                        pair_idx += 1;
                    }
                }
                eprintln!("  hop_fmo_total {:+.10e}", (hop_delta_p - hop_delta_m) / dh);

                // Also compute: mono_scc (= mono_total - mono_rep), pair_scc delta
                let mono_scc_p: f64 = comp_p.mono.iter().zip(comp_p.mono_rep.iter()).map(|(a,b)| a - b).sum();
                let mono_scc_m: f64 = comp_m.mono.iter().zip(comp_m.mono_rep.iter()).map(|(a,b)| a - b).sum();
                eprintln!("  mono_scc_total {:+.10e}", (mono_scc_p - mono_scc_m) / dh);

                let mono_rep_total_p: f64 = comp_p.mono_rep.iter().sum();
                let mono_rep_total_m: f64 = comp_m.mono_rep.iter().sum();
                eprintln!("  mono_rep_total {:+.10e}", (mono_rep_total_p - mono_rep_total_m) / dh);

                let mut pair_rep_delta_p = 0.0f64;
                let mut pair_rep_delta_m = 0.0f64;
                pair_idx = 0;
                for i in 0..n_mono {
                    for j in (i+1)..n_mono {
                        if pair_idx < comp_p.pair_rep.len() {
                            pair_rep_delta_p += comp_p.pair_rep[pair_idx] - comp_p.mono_rep[i] - comp_p.mono_rep[j];
                            pair_rep_delta_m += comp_m.pair_rep[pair_idx] - comp_m.mono_rep[i] - comp_m.mono_rep[j];
                        }
                        pair_idx += 1;
                    }
                }
                eprintln!("  pair_rep_delta {:+.10e}", (pair_rep_delta_p - pair_rep_delta_m) / dh);

                // Per-monomer augmented energy = E_SCC + E_rep + Tr(P·V_HOP)
                // HF grad of this should match numerical derivative (no inter-frag response needed)
                for i in 0..comp_p.mono.len() {
                    let aug_p = comp_p.mono[i] + comp_p.hop_mono[i];
                    let aug_m = comp_m.mono[i] + comp_m.hop_mono[i];
                    eprintln!("  mono_aug_{} {:+.10e}", i, (aug_p - aug_m) / dh);
                }

                // Per-pair augmented energy = E_SCC + E_rep + Tr(P·V_HOP)
                for i in 0..comp_p.pair.len() {
                    let aug_p = comp_p.pair[i] + comp_p.hop_pair[i];
                    let aug_m = comp_m.pair[i] + comp_m.hop_pair[i];
                    eprintln!("  pair_aug_{} {:+.10e}", i, (aug_p - aug_m) / dh);
                }

                // Total = mono + pair_delta + embedding + esd + trimer_delta + trimer_embedding + disp
                let total_p = mono_total_p + pair_delta_p + comp_p.embedding + comp_p.esd
                    + comp_p.trimer_delta + comp_p.trimer_embedding + comp_p.disp;
                let total_m = mono_total_m + pair_delta_m + comp_m.embedding + comp_m.esd
                    + comp_m.trimer_delta + comp_m.trimer_embedding + comp_m.disp;
                eprintln!("  total {:+.10e}", (total_p - total_m) / dh);
            }
            eprintln!("=== END HOP COMPONENT TEST ===\n");
            // Reset
            self.gs_energy_wrapper(xyz);
        }

        // Targeted component test at boundary atoms BEFORE the full test
        // (only for non-HOP; HOP stores energies differently)
        if !self.detached_bonds.is_empty() && !self.config.fmo.covalent_fragmentation {
            let xyz = self.get_xyz();
            let h = 0.01;
            let boundary_atoms = [4usize, 7, 10, 13];
            for &at in &boundary_atoms {
                let idx = 3 * at + 2; // z component
                let mut xyz_p = xyz.clone();
                let mut xyz_m = xyz.clone();
                xyz_p[idx] += h;
                xyz_m[idx] -= h;

                // Compute energies at +h
                let e_p_total = self.gs_energy_wrapper(xyz_p.clone());
                let e_p_mono: f64 = self.monomers.iter().map(|m| m.properties.last_energy().unwrap()).sum();
                let e_p_pair: f64 = self.pairs.iter().map(|p| p.properties.last_energy().unwrap()).sum();
                let e_p_embed = self.embedding_energy_shell();

                // Compute energies at -h
                let e_m_total = self.gs_energy_wrapper(xyz_m.clone());
                let e_m_mono: f64 = self.monomers.iter().map(|m| m.properties.last_energy().unwrap()).sum();
                let e_m_pair: f64 = self.pairs.iter().map(|p| p.properties.last_energy().unwrap()).sum();
                let e_m_embed = self.embedding_energy_shell();

                let num_total = (e_p_total - e_m_total) / (2.0 * h);
                let num_mono = (e_p_mono - e_m_mono) / (2.0 * h);
                let num_pair = (e_p_pair - e_m_pair) / (2.0 * h);
                let num_embed = (e_p_embed - e_m_embed) / (2.0 * h);
                let num_rest = num_total - num_mono - num_pair - num_embed;

                // Also print per-monomer and per-pair energies
                eprintln!("COMPONENT TEST atom {} z:", at);
                eprintln!("  num_total ={:+.10e}", num_total);
                eprintln!("  num_mono  ={:+.10e}", num_mono);
                eprintln!("  num_pair  ={:+.10e}", num_pair);
                eprintln!("  num_embed ={:+.10e}", num_embed);
                eprintln!("  num_rest  ={:+.10e} (disp+halogen)", num_rest);

                // Per-monomer numerical gradient
                self.gs_energy_wrapper(xyz_p.clone());
                let mono_p: Vec<f64> = self.monomers.iter().map(|m| m.properties.last_energy().unwrap()).collect();
                let pair_p: Vec<f64> = self.pairs.iter().map(|p| p.properties.last_energy().unwrap()).collect();
                self.gs_energy_wrapper(xyz_m.clone());
                let mono_m: Vec<f64> = self.monomers.iter().map(|m| m.properties.last_energy().unwrap()).collect();
                let pair_m: Vec<f64> = self.pairs.iter().map(|p| p.properties.last_energy().unwrap()).collect();
                for i in 0..self.monomers.len() {
                    eprintln!("    mon[{}] num_grad_z={:+.10e}", i, (mono_p[i] - mono_m[i]) / (2.0 * h));
                }
                for (i, pair) in self.pairs.iter().enumerate() {
                    eprintln!("    pair({},{}) num_grad_z={:+.10e}", pair.i, pair.j, (pair_p[i] - pair_m[i]) / (2.0 * h));
                }
            }
            // Reset state for the full test
            self.gs_energy_wrapper(xyz);
        }

        assert_deriv_5point(
            self,
            XtbSuperSystem::gs_energy_wrapper,
            XtbSuperSystem::gs_gradient_wrapper,
            self.get_xyz(),
            0.01,
            1e-6,
        );
    }

    pub fn monomer_energies_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();

        let max_iter: usize = self.config.scf.scf_max_cycles;
        // Do the self-consistent monomer calculations
        let (monomer_energies, dq): (f64, Array1<f64>) = self.monomer_scc(max_iter);

        monomer_energies
    }

    pub fn pair_energies_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();

        let max_iter: usize = self.config.scf.scf_max_cycles;
        // Do the self-consistent monomer calculations
        let (monomer_energies, dq): (f64, Array1<f64>) = self.monomer_scc(max_iter);

        // Do the SCC-calculation for each pair individually
        let pair_energies: f64 = self.pair_scc(dq.view());

        pair_energies
    }

    pub fn es_dim_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();

        let max_iter: usize = self.config.scf.scf_max_cycles;
        // Do the self-consistent monomer calculations
        let (monomer_energies, dq): (f64, Array1<f64>) = self.monomer_scc(max_iter);

        // Do the SCC-calculation for each pair individually
        let pair_energies: f64 = self.pair_scc(dq.view());

        // Compute the embedding energy from all pairs
        let embedding: f64 = self.embedding_energy();

        // Compute the energy from pairs that are far apart. The electrostatic dimer approximation
        // is used in this case.
        let esd_pair_energies: f64 = self.esd_pair_energy();

        esd_pair_energies
    }

    pub fn embedding_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();

        let max_iter: usize = self.config.scf.scf_max_cycles;
        // Do the self-consistent monomer calculations
        let (monomer_energies, dq): (f64, Array1<f64>) = self.monomer_scc(max_iter);

        // Do the SCC-calculation for each pair individually
        let pair_energies: f64 = self.pair_scc(dq.view());

        // Compute the embedding energy from all pairs
        let embedding: f64 = self.embedding_energy();

        embedding
    }

    // ============ NEW FMO2-xTB Gradient Component Tests (fmo_gradient.rs) ============

    /// Energy wrapper for dispersion energy only.
    pub fn dispersion_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.update_xyz(geometry.view());
        get_dispersion_energy_xtb(&self.atoms, &self.config)
    }

    /// Energy wrapper for embedding + ESD energy combined.
    pub fn embedding_plus_esd_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();

        let max_iter: usize = self.config.scf.scf_max_cycles;
        let (_, dq): (f64, Array1<f64>) = self.monomer_scc(max_iter);
        let _ = self.pair_scc(dq.view());
        let embedding: f64 = self.embedding_energy();
        let esd: f64 = self.esd_pair_energy();

        embedding + esd
    }

    // ============ FMO3 Test Functions ============

    /// Wrapper for trimer internal energy (E_IJK - E_I - E_J - E_K - ΔE_IJ - ΔE_IK - ΔE_JK)
    pub fn trimer_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for trimer in self.trimers.iter_mut() {
            trimer.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();

        let max_iter: usize = self.config.scf.scf_max_cycles;
        let (_, dq): (f64, Array1<f64>) = self.monomer_scc(max_iter);
        let _ = self.pair_scc(dq.view());
        let trimer_energy = self.trimer_scc();

        trimer_energy
    }

    /// Wrapper for trimer embedding energy
    pub fn trimer_embedding_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for trimer in self.trimers.iter_mut() {
            trimer.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();

        let max_iter: usize = self.config.scf.scf_max_cycles;
        let (_, dq): (f64, Array1<f64>) = self.monomer_scc(max_iter);
        let _ = self.pair_scc(dq.view());
        let _ = self.trimer_scc();
        let trimer_embedding = self.trimer_embedding_energy();

        trimer_embedding
    }

    /// Wrapper for pair embedding energy with FMO3 counter subtraction
    /// This computes: sum_{IJ} (1 - counter_IJ) * E_emb_IJ
    pub fn pair_embedding_with_counter_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        use dialect_state::PairType;

        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for trimer in self.trimers.iter_mut() {
            trimer.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();

        let max_iter: usize = self.config.scf.scf_max_cycles;
        let (_, dq): (f64, Array1<f64>) = self.monomer_scc(max_iter);
        let _ = self.pair_scc(dq.view());

        // Build counter for pairs
        let mut fmo3_pair_counter: Array1<usize> = Array1::zeros(self.pairs.len());
        for trimer in self.trimers.iter() {
            let i = trimer.i;
            let j = trimer.j;
            let k = trimer.k;
            if self.properties.type_of_pair(i, j) == PairType::Pair {
                let index = self.properties.index_of_pair(i, j);
                fmo3_pair_counter[index] += 1;
            }
            if self.properties.type_of_pair(i, k) == PairType::Pair {
                let index = self.properties.index_of_pair(i, k);
                fmo3_pair_counter[index] += 1;
            }
            if self.properties.type_of_pair(j, k) == PairType::Pair {
                let index = self.properties.index_of_pair(j, k);
                fmo3_pair_counter[index] += 1;
            }
        }

        // Compute embedding with counter
        let gamma = self.properties.gamma_ao().unwrap();
        let mut embedding: f64 = 0.0;
        for (pair_idx, pair) in self.pairs.iter().enumerate() {
            let m_i = &self.monomers[pair.i];
            let m_j = &self.monomers[pair.j];
            let dq_i = m_i.properties.dq_ao().unwrap();
            let dq_j = m_j.properties.dq_ao().unwrap();
            let esp_q_i = m_i.properties.esp_q().unwrap();
            let esp_q_j = m_j.properties.esp_q().unwrap();
            let ddq = pair.properties.delta_dq().unwrap();

            let esp_q_i: Array1<f64> =
                esp_q_i.to_owned() - gamma.slice(s![m_i.slice.orb, m_j.slice.orb]).dot(&dq_j);
            let esp_q_j: Array1<f64> =
                esp_q_j.to_owned() - gamma.slice(s![m_j.slice.orb, m_i.slice.orb]).dot(&dq_i);

            let pair_embedding = esp_q_i.dot(&ddq.slice(s![..m_i.n_orbs]))
                + esp_q_j.dot(&ddq.slice(s![m_i.n_orbs..]));

            let counter = fmo3_pair_counter[pair_idx] as f64;
            embedding += (1.0 - counter) * pair_embedding;
        }

        embedding
    }

    /// Wrapper for full FMO3 embedding: pair_emb_with_counter + trimer_embedding
    pub fn fmo3_embedding_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        use dialect_state::PairType;

        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for trimer in self.trimers.iter_mut() {
            trimer.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();

        let max_iter: usize = self.config.scf.scf_max_cycles;
        let (_, dq): (f64, Array1<f64>) = self.monomer_scc(max_iter);
        let _ = self.pair_scc(dq.view());
        let _ = self.trimer_scc();

        // Build counter for pairs
        let mut fmo3_pair_counter: Array1<usize> = Array1::zeros(self.pairs.len());
        for trimer in self.trimers.iter() {
            let i = trimer.i;
            let j = trimer.j;
            let k = trimer.k;
            if self.properties.type_of_pair(i, j) == PairType::Pair {
                let index = self.properties.index_of_pair(i, j);
                fmo3_pair_counter[index] += 1;
            }
            if self.properties.type_of_pair(i, k) == PairType::Pair {
                let index = self.properties.index_of_pair(i, k);
                fmo3_pair_counter[index] += 1;
            }
            if self.properties.type_of_pair(j, k) == PairType::Pair {
                let index = self.properties.index_of_pair(j, k);
                fmo3_pair_counter[index] += 1;
            }
        }

        // Compute pair embedding with counter
        let gamma = self.properties.gamma_ao().unwrap();
        let mut pair_embedding: f64 = 0.0;
        for (pair_idx, pair) in self.pairs.iter().enumerate() {
            let m_i = &self.monomers[pair.i];
            let m_j = &self.monomers[pair.j];
            let dq_i = m_i.properties.dq_ao().unwrap();
            let dq_j = m_j.properties.dq_ao().unwrap();
            let esp_q_i = m_i.properties.esp_q().unwrap();
            let esp_q_j = m_j.properties.esp_q().unwrap();
            let ddq = pair.properties.delta_dq().unwrap();

            let esp_q_i: Array1<f64> =
                esp_q_i.to_owned() - gamma.slice(s![m_i.slice.orb, m_j.slice.orb]).dot(&dq_j);
            let esp_q_j: Array1<f64> =
                esp_q_j.to_owned() - gamma.slice(s![m_j.slice.orb, m_i.slice.orb]).dot(&dq_i);

            let emb = esp_q_i.dot(&ddq.slice(s![..m_i.n_orbs]))
                + esp_q_j.dot(&ddq.slice(s![m_i.n_orbs..]));

            let counter = fmo3_pair_counter[pair_idx] as f64;
            pair_embedding += (1.0 - counter) * emb;
        }

        // Compute trimer embedding
        let trimer_embedding = self.trimer_embedding_energy();

        pair_embedding + trimer_embedding
    }
}

#[cfg(test)]
mod fmo_gradient_tests {
    use crate::fmo::supersystem::{init_fmo_xtb, XtbSuperSystem};
    use dialect_config::Configuration;
    use dialect_utilities::io::read_xyz_frame;
    use dialect_utilities::numerical::assert_deriv_5point;

    /// Finite-difference step (Bohr) for the 5-point numerical gradient.
    const STEP: f64 = 1.0e-2;
    /// Largest tolerated deviation between the analytical and numerical
    /// FMO-xTB ground-state gradient (Hartree/Bohr).
    const TOLERANCE: f64 = 1.0e-6;

    /// FMO-xTB configuration used for the gradient regression. Tight SCC
    /// convergence (1e-13) keeps both the fragment SCC robust under
    /// displacement and the numerical gradient essentially noise-free.
    fn fmo_xtb_config() -> Configuration {
        let mut config: Configuration = toml::from_str("").unwrap();
        config.tight_binding.use_dftb = false;
        config.tight_binding.use_xtb1 = true;
        config.scf.scf_max_cycles = 250;
        config.scf.scf_charge_conv = 1.0e-13;
        config.scf.scf_energy_conv = 1.0e-13;
        config.scf.electronic_temperature = 300.0;
        config.mol.charge = 0;
        config.mol.multiplicity = 1;
        config.fmo.use_fmo = true;
        config.fmo.vdw_scaling = 2.0;
        config.fmo.use_three_body = false;
        config.fmo.trimer_vdw_scaling = 1.5;
        // Broyden mixer (FMO-xTB uses Broyden, not Anderson).
        config.broyden.alpha = 0.4;
        config.broyden.omega0 = 0.01;
        config.broyden.memory = 20;
        config.broyden.safeguard_factor = 1.0;
        config
    }

    /// The analytical FMO-xTB ground-state gradient must agree with a 5-point
    /// numerical gradient for a 20-molecule water cluster (exercises monomer,
    /// pair, embedding and CN-derivative contributions across many fragments).
    #[test]
    fn fmo_xtb_gradient_accuracy_water20() {
        let path: String = format!(
            "{}/../tests/data/water_20/water_20.xyz",
            env!("CARGO_MANIFEST_DIR")
        );
        let frame = read_xyz_frame(&path);
        let (atoms, basis, gammafunc) = init_fmo_xtb(frame);
        let mut system = XtbSuperSystem::from((atoms, basis, &gammafunc, fmo_xtb_config()));

        // Analytical gradient vs the 5-point numerical gradient (asserts that
        // the largest deviation stays below TOLERANCE).
        let origin = system.get_xyz();
        assert_deriv_5point(
            &mut system,
            XtbSuperSystem::gs_energy_wrapper,
            XtbSuperSystem::gs_gradient_wrapper,
            origin,
            STEP,
            TOLERANCE,
        );
    }
}
