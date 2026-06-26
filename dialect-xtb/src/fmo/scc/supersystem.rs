use dialect_utilities::fmo_helpers::{get_pair_slice_xtb, get_trimer_slice_xtb};
use dialect_utilities::scc_helpers::aovec_to_aomat;
use dialect_utilities::fmo_logging as logging;
use dialect_utilities::fmo_logging::fmo_monomer_iteration;
use dialect_state::PairType;
use dialect_utilities::mulliken::shell_to_ao_values;
use dialect_utilities::scc_interface::{RestrictedSCC, SCCError};
use dialect_base::Timer;
use crate::fmo::monomer::XtbMonomer;
use crate::fmo::supersystem::XtbSuperSystem;
use crate::hop::{
    compute_total_hop_projector, get_detached_bonds_for_fragment, get_detached_bonds_for_pair,
    get_detached_bonds_for_trimer,
};
use crate::initialization::atom::XtbAtom;
use crate::initialization::basis::create_basis_set;
use crate::scc::gamma_matrix::{gamma_matrix_shell, gamma_matrix_xtb_new, gamma_shell_dsymv};
use crate::scc::halogen_correction::get_halogen_correction;
use crate::scc::hamiltonian::calculate_coordination_numbers;
use crate::scc::scc_helpers::{calculate_repulsive_energy_xtb, get_dispersion_energy_xtb};
use log::info;
use nalgebra::Vector3;
use ndarray::{concatenate, s, Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::ops::SubAssign;
use std::time::Instant;

impl RestrictedSCC for XtbSuperSystem<'_> {
    /// Set up the FMO-xTB calculation: global coordination numbers and the
    /// per-monomer SCC preparation (overlap, H0, reference charges).
    fn prepare_scc(&mut self) {
        let atoms = &self.atoms;

        // Compute global coordination numbers from ALL atoms in the supersystem
        let cn_numbers: Array1<f64> = calculate_coordination_numbers(atoms);
        self.properties.set_cn(cn_numbers);

        // Pass CN slices to monomers
        let cn: ArrayView1<f64> = self.properties.cn().unwrap();
        let broyden_config = &self.config.broyden;
        self.monomers.par_iter_mut().for_each(|mol| {
            let cn_slice = cn.slice(s![mol.slice.atom_as_range()]);
            mol.prepare_scc(&atoms[mol.slice.atom_as_range()], cn_slice, broyden_config);
        });
        // if self.properties.gamma_ao().is_none() {
        //     let gammafunc = self.monomers[0].gammafunction.clone();
        //     let gamma_ao: Array2<f64> = gamma_matrix_xtb_new(&gammafunc, atoms, &self.basis);
        //     self.properties.set_gamma_ao(gamma_ao);
        // }
        if self.properties.gamma_shell().is_none() {
            let gammafunc = self.monomers[0].gammafunction.clone();
            let gamma_shell: Array2<f64> = gamma_matrix_shell(&gammafunc, atoms, &self.basis);
            self.properties.set_gamma_shell(gamma_shell);
        }

        // Compute and store HOP projectors for monomers with detached bonds.
        // The monomer basis includes ghost boundary atoms, so the HOP projector
        // is computed on the extended basis (with ghost H basis functions).
        if !self.detached_bonds.is_empty() {
            let detached_bonds = &self.detached_bonds;
            let atoms = &self.atoms;
            for mol in self.monomers.iter_mut() {
                let bonds = get_detached_bonds_for_fragment(detached_bonds, mol.index);
                if bonds.is_empty() {
                    continue;
                }
                let s = mol.properties.s().unwrap();
                let positions = |global_idx: usize| -> Vector3<f64> { atoms[global_idx].xyz };
                if let Some(p_hop) = compute_total_hop_projector(&bonds, &mol.basis, s, &positions) {
                    mol.properties.set_hop_projector(p_hop);
                }
            }
        }
    }

    /// Run the full FMO-xTB SCC and assemble the total energy from the monomer,
    /// pair, embedding and ESD terms (plus the FMO3 trimer terms when three-body
    /// is enabled), the repulsive, dispersion and halogen contributions.
    fn run_scc(&mut self) -> Result<f64, SCCError> {
        let timer: Timer = Timer::start();
        // SCC settings from the user input
        let max_iter: usize = self.config.scf.scf_max_cycles;
        logging::fmo_scc_init(max_iter);

        let t_disp = Instant::now();
        let e_disp: f64 = get_dispersion_energy_xtb(&self.atoms, &self.config);
        let e_halogen: f64 = get_halogen_correction(&self.atoms);
        info!(
            "{:>68} {:>8.2} s",
            "D3 dispersion + Halogen correction:",
            t_disp.elapsed().as_secs_f32()
        );

        // Do the self-consistent monomer calculations
        let t_mono = Instant::now();
        let (monomer_energies, dq): (f64, Array1<f64>) = self.monomer_scc(max_iter);
        info!(
            "{:>68} {:>8.2} s",
            "monomer SCC:",
            t_mono.elapsed().as_secs_f32()
        );

        // Do the SCC-calculation for each pair individually
        let t_pair = Instant::now();
        let pair_energies: f64 = self.pair_scc(dq.view());
        info!(
            "{:>68} {:>8.2} s",
            "pair SCC:",
            t_pair.elapsed().as_secs_f32()
        );

        // Compute the embedding energy from all pairs
        let t_emb = Instant::now();
        let embedding: f64 = self.embedding_energy_shell();
        info!(
            "{:>68} {:>8.2} s",
            "pair embedding:",
            t_emb.elapsed().as_secs_f32()
        );

        // Compute the energy from pairs that are far apart. The electrostatic dimer approximation
        // is used in this case.
        let t_esd = Instant::now();
        let esd_pair_energies: f64 = self.esd_pair_energy_shell();
        info!(
            "{:>68} {:>8.2} s",
            "ESD pairs:",
            t_esd.elapsed().as_secs_f32()
        );

        // trimer calculation
        let (trimer_energies, trimer_embedding) = if self.config.fmo.use_three_body {
            let t_tri = Instant::now();
            let trimer_energies = self.trimer_scc_shell();
            info!(
                "{:>68} {:>8.2} s",
                "trimer SCC:",
                t_tri.elapsed().as_secs_f32()
            );

            let t_tri_emb = Instant::now();
            let trimer_embedding = self.trimer_embedding_energy_shell();
            info!(
                "{:>68} {:>8.2} s",
                "trimer embedding:",
                t_tri_emb.elapsed().as_secs_f32()
            );

            (trimer_energies, trimer_embedding)
        } else {
            (0.0, 0.0)
        };

        // Sum up all the individual energies
        let total_energy: f64 = monomer_energies
            + pair_energies
            + embedding
            + esd_pair_energies
            + e_disp
            + e_halogen
            + trimer_energies
            + trimer_embedding;

        // Save the charge differences of all monomers in the SuperSystem
        self.properties.set_dq_ao(dq);

        if self.config.fmo.use_three_body {
            // Print information of the SCC-routine
            logging::fmo_scc_end_trimer(
                timer,
                monomer_energies,
                pair_energies,
                embedding,
                esd_pair_energies,
                e_disp,
                trimer_energies,
                trimer_embedding,
            );
        } else {
            // Print information of the SCC-routine
            logging::fmo_scc_end(
                timer,
                monomer_energies,
                pair_energies,
                embedding,
                esd_pair_energies,
                e_disp,
            );
        }

        self.properties.set_last_energy(total_energy);
        // Return the energy
        Ok(total_energy)
    }
}

impl XtbSuperSystem<'_> {
    /// FMO monomer step: converge every monomer's xTB SCC self-consistently in
    /// the field of the other monomers. Returns the summed monomer energy ΣE_I
    /// and the converged embedding charges used by the pair/trimer steps.
    pub fn monomer_scc(&mut self, max_iter: usize) -> (f64, Array1<f64>) {
        // Vector that holds the information if the scc calculation of the individual monomer
        // is converged or not
        let mut converged: Vec<bool> = vec![false; self.n_mol];
        // charge differences of all atoms. these are needed to compute the electrostatic potential
        // that acts on the monomers. Uses supersystem n_orbs (real orbs only).
        let mut dq_ao: Array1<f64> = if self.properties.dq_ao().is_none() {
            Array1::zeros(self.n_orbs)
        } else {
            self.properties.dq_ao().unwrap().to_owned()
        };

        // atoms
        let atoms = &self.atoms;
        // config
        let config = &self.config;
        // charge consistent loop for the monomers
        'scf_loop: for iter in 0..max_iter {
            // the matrix vector product of the gamma matrix for all atoms and the charge differences
            // yields the electrostatic potential for all atoms. this is then converted into ao basis
            // and given to each monomer scc step
            // let esp_at: Array1<f64> = self.properties.gamma_ao().unwrap().dot(&dq_ao);
            let gamma_shell: ArrayView2<f64> = self.properties.gamma_shell().unwrap();
            // Build dq_shell_full from stored monomer dq_shell's (real shells only)
            let n_shells_full = gamma_shell.nrows();
            let mut dq_shell_full: Array1<f64> = Array1::zeros(n_shells_full);
            for mol in self.monomers.iter() {
                // Extract real-shell charges (first n_real_shells elements)
                let dq_shell_ext = mol.properties.dq_shell().unwrap();
                let dq_shell_real = dq_shell_ext.slice(s![..mol.n_real_shells]);
                dq_shell_full
                    .slice_mut(s![mol.slice.shell])
                    .assign(&dq_shell_real);
            }
            // BLAS dsymv: exploit symmetry of gamma_shell
            let v_shell = gamma_shell_dsymv(&gamma_shell, &dq_shell_full.view());
            let esp_at: Array1<f64> = shell_to_ao_values(&self.basis, self.n_orbs, v_shell.view());

            // Parallelization
            let loop_output: Vec<bool> = self
                .monomers
                .par_iter_mut()
                .map(|mol| {
                    // Build v_esp: real AOs from global ESP, ghost AOs from local gamma shift
                    let v_esp: Array2<f64> = if mol.n_real_orbs < mol.n_orbs {
                        // Ghost AOs need their Coulomb shift from the local gamma matrix.
                        // The global gamma doesn't cover ghost shells, so we compute locally:
                        // shift_ghost = gamma_local[ghost,:] @ dq_local
                        let gamma_local = mol.properties.gamma_shell().unwrap().to_owned();
                        let dq_shell_local = mol.properties.dq_shell().unwrap().to_owned();
                        let local_shift_shell: Array1<f64> = gamma_local.dot(&dq_shell_local);
                        let local_shift_ao: Array1<f64> =
                            shell_to_ao_values(&mol.basis, mol.n_orbs, local_shift_shell.view());

                        let mut esp_ext = Array1::zeros(mol.n_orbs);
                        // Real AOs: from global ESP (includes intra + inter-monomer)
                        esp_ext
                            .slice_mut(s![..mol.n_real_orbs])
                            .assign(&esp_at.slice(s![mol.slice.orb]));
                        // Ghost AOs: from local gamma @ local dq_shell
                        esp_ext
                            .slice_mut(s![mol.n_real_orbs..])
                            .assign(&local_shift_ao.slice(s![mol.n_real_orbs..]));
                        aovec_to_aomat(esp_ext.view(), mol.n_orbs)
                    } else {
                        aovec_to_aomat(esp_at.slice(s![mol.slice.orb]), mol.n_orbs)
                    };
                    mol.scc_step(&atoms[mol.slice.atom_as_range()], v_esp, config)
                })
                .collect();
            for mol in self.monomers.iter() {
                // save the real-orb dq's from the monomer calculation
                let dq_ao_ext = mol.properties.dq_ao().unwrap();
                let dq_ao_real = dq_ao_ext.slice(s![..mol.n_real_orbs]);
                dq_ao
                    .slice_mut(s![mol.slice.orb])
                    .assign(&dq_ao_real);
            }
            converged = loop_output;

            let n_converged: usize = converged.iter().filter(|&n| *n).count();
            fmo_monomer_iteration(iter, n_converged, self.n_mol);
            // the loop ends if all monomers are converged
            if n_converged == self.n_mol {
                break 'scf_loop;
            }
        }
        if converged.contains(&false) {
            panic!("Monomer scc routine did NOT converge!");
        }
        let mut monomer_energies: f64 = 0.0;
        for mol in self.monomers.iter_mut() {
            let scf_energy: f64 = mol.properties.last_energy().unwrap();
            // Repulsive energy from real atoms only (ghosts excluded)
            let vrep: f64 = calculate_repulsive_energy_xtb(&atoms[mol.slice.atom_as_range()]);

            mol.properties.set_last_energy(scf_energy + vrep);
            monomer_energies += scf_energy + vrep;

            // Truncate charges to real-atom dimensions for embedding/pair ESP.
            // Save full versions (with ghost charges) for gradient computation.
            if mol.n_real_orbs < mol.n_orbs {
                let dq_all = mol.properties.dq().unwrap().to_owned();
                mol.properties.set_dq_full(dq_all.clone());
                mol.properties
                    .set_dq(dq_all.slice(s![..mol.n_real_atoms]).to_owned());
                let dq_ao_full = mol.properties.dq_ao().unwrap().to_owned();
                mol.properties
                    .set_dq_ao(dq_ao_full.slice(s![..mol.n_real_orbs]).to_owned());
                let dq_shell_all = mol.properties.dq_shell().unwrap().to_owned();
                mol.properties.set_dq_shell_full(dq_shell_all.clone());
                mol.properties
                    .set_dq_shell(dq_shell_all.slice(s![..mol.n_real_shells]).to_owned());
            }

            // Remove intermediate SCC properties no longer needed
            mol.properties.reset_scc_intermediates();
        }

        (monomer_energies, dq_ao)
    }

    /// FMO pair step: run the xTB SCC for every close ("real") pair in the
    /// field of the surrounding monomers and return the summed pair correction
    /// Σ_{I<J} (E_IJ - E_I - E_J).
    pub fn pair_scc(&mut self, dq: ArrayView1<f64>) -> f64 {
        // Build full dq_shell from monomer dq_shell's
        let gamma_shell: ArrayView2<f64> = self.properties.gamma_shell().unwrap();
        let n_shells = gamma_shell.nrows();
        let mut dq_shell_full: Array1<f64> = Array1::zeros(n_shells);
        for mol in self.monomers.iter() {
            dq_shell_full
                .slice_mut(s![mol.slice.shell])
                .assign(&mol.properties.dq_shell().unwrap());
        }
        // compute the electrostatic potential at shell level and store as shell-level
        for mol in self.monomers.iter_mut() {
            let mol_dq_shell: ArrayView1<f64> = mol.properties.dq_shell().unwrap();
            let mut esp_shell: Array1<f64> = gamma_shell
                .slice(s![mol.slice.shell, ..])
                .dot(&dq_shell_full);
            esp_shell -= &gamma_shell
                .slice(s![mol.slice.shell, mol.slice.shell])
                .dot(&mol_dq_shell);
            mol.properties.set_esp_q(esp_shell);
        }

        let atoms: &[XtbAtom] = &self.atoms[..];
        // Parallelization
        let monomers: &Vec<XtbMonomer> = &self.monomers;
        let config = &self.config;
        let cn_super: ArrayView1<f64> = self.properties.cn().unwrap();
        let detached_bonds = &self.detached_bonds;
        let pair_energy: Vec<f64> = self
            .pairs
            .par_iter_mut()
            .map(|pair| {
                // Get references to the corresponding monomers
                let m_i: &XtbMonomer = &monomers[pair.i];
                let m_j: &XtbMonomer = &monomers[pair.j];

                // The atoms are in general a non-contiguous range of the atoms
                let pair_atoms: Vec<XtbAtom> =
                    get_pair_slice_xtb(atoms, m_i.slice.atom_as_range(), m_j.slice.atom_as_range());

                // ZREF + HOP for partial bonds (BDA in pair, BAA outside).
                // Bonds fully inside the pair are healed (no HOP, no ZREF reduction).
                let partial_bonds = if !detached_bonds.is_empty() {
                    get_detached_bonds_for_pair(detached_bonds, pair.i, pair.j)
                } else {
                    vec![]
                };

                // Add ghost atoms for partial bonds and extend pair
                // Deduplicate: if multiple partial bonds share the same BAA position
                // (e.g., pair(0,2) where both fragments have bonds to fragment 1),
                // only create one ghost atom at that position.
                let ext_pair_atoms: Vec<XtbAtom> = if !partial_bonds.is_empty() {
                    let mut ext = pair_atoms.clone();
                    let mut seen_baa: Vec<usize> = Vec::new();
                    for bond in &partial_bonds {
                        if !seen_baa.contains(&bond.baa_global) {
                            seen_baa.push(bond.baa_global);
                            let mut ghost = XtbAtom::from(1u8);
                            ghost.xyz = atoms[bond.baa_global].xyz;
                            ext.push(ghost);
                        }
                    }
                    // Set adjusted n_elec (BEFORE prepare_scc)
                    let raw_n_elec: usize = pair_atoms.iter().map(|a| a.n_elec).sum();
                    pair.properties
                        .set_n_elec(raw_n_elec - partial_bonds.len());
                    // Update pair dimensions for extended atoms
                    pair.n_real_atoms = pair_atoms.len();
                    pair.n_real_orbs = pair.n_orbs;
                    let ext_basis = create_basis_set(&ext);
                    pair.n_atoms = ext.len();
                    pair.n_orbs = ext_basis.nbas;
                    pair.n_real_shells = m_i.n_real_shells + m_j.n_real_shells;
                    ext
                } else {
                    pair_atoms.clone()
                };

                // Build CN for pair atoms by concatenating slices from global CN
                let cn_i = cn_super.slice(s![m_i.slice.atom_as_range()]);
                let cn_j = cn_super.slice(s![m_j.slice.atom_as_range()]);
                let mut cn_pair = concatenate![Axis(0), cn_i, cn_j];
                // Ghost atoms get CN=0
                if ext_pair_atoms.len() > pair_atoms.len() {
                    let mut ext_cn = Array1::zeros(ext_pair_atoms.len());
                    ext_cn.slice_mut(s![..pair_atoms.len()]).assign(&cn_pair);
                    cn_pair = ext_cn;
                }

                pair.prepare_scc(
                    &ext_pair_atoms[..],
                    m_i,
                    m_j,
                    self.properties.gamma_shell().unwrap(),
                    cn_pair.view(),
                );

                // ZREF p_ref reduction + ghost zeroing + HOP projector for partial bonds
                if !partial_bonds.is_empty() {
                    let mut p_ref = pair.properties.take_p_ref().unwrap();

                    // Count partial bonds per local atom (an atom may have multiple)
                    let n_atoms_i = m_i.n_real_atoms;
                    let mut partial_count = vec![0usize; pair_atoms.len()];
                    for bond in &partial_bonds {
                        let bda_local = if bond.bda_fragment == pair.i {
                            bond.bda_global - m_i.slice.atom_as_range().start
                        } else {
                            (bond.bda_global - m_j.slice.atom_as_range().start) + n_atoms_i
                        };
                        partial_count[bda_local] += 1;
                    }

                    // Reduce BDA p_ref diagonals (ZREF -= count)
                    for (bda_local, &count) in partial_count.iter().enumerate() {
                        if count == 0 {
                            continue;
                        }
                        let bda_n_elec = pair_atoms[bda_local].n_elec as f64;
                        if bda_n_elec > 0.0 {
                            let scale = (bda_n_elec - count as f64) / bda_n_elec;
                            for shell in pair.basis.shells.iter() {
                                if shell.atom_index == bda_local {
                                    for i in shell.sph_start..shell.sph_end {
                                        p_ref[[i, i]] *= scale;
                                    }
                                }
                            }
                        }
                    }

                    // Zero ghost atom reference occupations
                    for shell in pair.basis.shells.iter() {
                        if shell.atom_index >= pair.n_real_atoms {
                            for i in shell.sph_start..shell.sph_end {
                                p_ref[[i, i]] = 0.0;
                            }
                        }
                    }

                    pair.properties.set_p_ref(p_ref.clone());
                    pair.properties.set_p(p_ref);

                    // HOP projector for partial bonds
                    let s = pair.properties.s().unwrap();
                    let positions =
                        |global_idx: usize| -> Vector3<f64> { atoms[global_idx].xyz };
                    if let Some(p_hop) =
                        compute_total_hop_projector(&partial_bonds, &pair.basis, s, &positions)
                    {
                        pair.properties.set_hop_projector(p_hop);
                    }
                }

                // do the SCC iterations
                pair.run_scc(&ext_pair_atoms, config);

                // Truncate delta_dq_shell to real shells for embedding energy
                if !partial_bonds.is_empty() {
                    let ddq_shell_trunc = pair
                        .properties
                        .delta_dq_shell()
                        .unwrap()
                        .slice(s![..pair.n_real_shells])
                        .to_owned();
                    pair.properties.set_delta_dq_shell(ddq_shell_trunc);
                }

                // and compute the SCC energy
                let pair_abs = pair.properties.last_energy().unwrap();
                let mono_i_e = m_i.properties.last_energy().unwrap();
                let mono_j_e = m_j.properties.last_energy().unwrap();
                let pair_energ: f64 = pair_abs - mono_i_e - mono_j_e;

                // Remove intermediate SCC properties no longer needed
                pair.properties.reset_scc_intermediates();

                pair_energ
            })
            .collect();
        let pair_energy: Array1<f64> = Array::from(pair_energy);
        pair_energy.sum()
    }

    /// Shell-resolved electrostatic embedding correction: the interaction of
    /// each pair's density change with the shell-charge field of the monomers
    /// outside the pair.
    pub fn embedding_energy_shell(&self) -> f64 {
        // The embedding energy is initialized to zero.
        let mut embedding: f64 = 0.0;

        // Reference to the shell-level Gamma matrix of the full system.
        let gamma_shell: ArrayView2<f64> = self.properties.gamma_shell().unwrap();
        for pair in self.pairs.iter() {
            let m_i: &XtbMonomer = &self.monomers[pair.i];
            let m_j: &XtbMonomer = &self.monomers[pair.j];
            // Shell-level charge differences (stored in properties)
            let dq_shell_i: ArrayView1<f64> = m_i.properties.dq_shell().unwrap();
            let dq_shell_j: ArrayView1<f64> = m_j.properties.dq_shell().unwrap();
            // ESP without self-interaction (stored as shell-level)
            let esp_q_i: ArrayView1<f64> = m_i.properties.esp_q().unwrap();
            let esp_q_j: ArrayView1<f64> = m_j.properties.esp_q().unwrap();
            // delta_dq at shell level (stored directly)
            let ddq_shell: ArrayView1<f64> = pair.properties.delta_dq_shell().unwrap();
            // Subtract cross-monomer interaction at shell level
            let esp_i: Array1<f64> = &esp_q_i
                - &gamma_shell
                    .slice(s![m_i.slice.shell, m_j.slice.shell])
                    .dot(&dq_shell_j);
            let esp_j: Array1<f64> = &esp_q_j
                - &gamma_shell
                    .slice(s![m_j.slice.shell, m_i.slice.shell])
                    .dot(&dq_shell_i);
            // Embedding energy at shell level
            embedding += esp_i.dot(&ddq_shell.slice(s![..m_i.n_real_shells]));
            embedding += esp_j.dot(&ddq_shell.slice(s![m_i.n_real_shells..]));
        }
        embedding
    }

    /// Shell-resolved electrostatic-dimer (ESD) correction for distant pairs:
    /// their interaction is approximated by the shell-monopole electrostatics
    /// between the two monomers instead of a full pair SCC.
    pub fn esd_pair_energy_shell(&self) -> f64 {
        let gamma_shell: ArrayView2<f64> = self.properties.gamma_shell().unwrap();
        let n_shells = gamma_shell.nrows();

        // Step 1: Construct full dq_shell vector from stored monomer dq_shell's
        let mut dq_shell_full: Array1<f64> = Array1::zeros(n_shells);
        for m in self.monomers.iter() {
            dq_shell_full
                .slice_mut(s![m.slice.shell])
                .assign(&m.properties.dq_shell().unwrap());
        }

        // Step 2: Shell-level dsymv (exploit symmetry)
        let v_shell: Array1<f64> = gamma_shell_dsymv(&gamma_shell, &dq_shell_full.view());

        // Step 3: Total quadratic form = 0.5 * dq_shell^T * G_shell * dq_shell
        let e_total: f64 = 0.5 * dq_shell_full.dot(&v_shell);

        // Step 4: Subtract monomer self-interaction (diagonal blocks)
        let mut e_self: f64 = 0.0;
        for m in self.monomers.iter() {
            let dq_shell_i: ArrayView1<f64> = m.properties.dq_shell().unwrap();
            let g_shell_ii = gamma_shell.slice(s![m.slice.shell, m.slice.shell]);
            e_self += 0.5 * dq_shell_i.dot(&g_shell_ii.dot(&dq_shell_i));
        }

        // Step 5: Subtract close pair contributions
        let mut e_close: f64 = 0.0;
        for pair in self.pairs.iter() {
            let m_i = &self.monomers[pair.i];
            let m_j = &self.monomers[pair.j];
            let dq_shell_i: ArrayView1<f64> = m_i.properties.dq_shell().unwrap();
            let dq_shell_j: ArrayView1<f64> = m_j.properties.dq_shell().unwrap();
            let g_shell_ij = gamma_shell.slice(s![m_i.slice.shell, m_j.slice.shell]);
            e_close += dq_shell_i.dot(&g_shell_ij.dot(&dq_shell_j));
        }

        // E_esd = (all off-diagonal) - (close pairs)
        e_total - e_self - e_close
    }

    /// FMO3 trimer step (shell-resolved): run the SCC for each close triple of
    /// monomers and return the three-body correction
    /// Σ_{I<J<K} (E_IJK - ΔE_IJ - ΔE_IK - ΔE_JK - E_I - E_J - E_K).
    pub fn trimer_scc_shell(&mut self) -> f64 {
        let atoms: &[XtbAtom] = &self.atoms[..];
        // Parallelization
        let monomers: &Vec<XtbMonomer> = &self.monomers;
        let config = &self.config;
        let detached_bonds = &self.detached_bonds;
        // Reference to the shell-level Gamma matrix of the full system.
        let gamma_shell: ArrayView2<f64> = self.properties.gamma_shell().unwrap();
        let cn_super: ArrayView1<f64> = self.properties.cn().unwrap();

        // trimer loop
        let trimer_energies: Vec<f64> = self
            .trimers
            .par_iter_mut()
            .map(|trimer| {
                // Get references to the corresponding monomers
                let m_i: &XtbMonomer = &monomers[trimer.i];
                let m_j: &XtbMonomer = &monomers[trimer.j];
                let m_k: &XtbMonomer = &monomers[trimer.k];
                let trimer_atoms: Vec<XtbAtom> = get_trimer_slice_xtb(
                    &atoms,
                    m_i.slice.atom_as_range(),
                    m_j.slice.atom_as_range(),
                    m_k.slice.atom_as_range(),
                );
                // get monomer energies
                let m_i_energy: f64 = m_i.properties.last_energy().unwrap();
                let m_j_energy: f64 = m_j.properties.last_energy().unwrap();
                let m_k_energy: f64 = m_k.properties.last_energy().unwrap();
                // check pair types and get pair energies (using shell-level gamma for ESDIM)
                let delta_ij: f64 =
                    if self.properties.type_of_pair(trimer.i, trimer.j) == PairType::Pair {
                        let index_ij: usize = self.properties.index_of_pair(trimer.i, trimer.j);
                        let pair_ij_energy: f64 =
                            self.pairs[index_ij].properties.last_energy().unwrap();
                        pair_ij_energy - m_i_energy - m_j_energy
                    } else {
                        let dq_shell_i: ArrayView1<f64> = m_i.properties.dq_shell().unwrap();
                        let dq_shell_j: ArrayView1<f64> = m_j.properties.dq_shell().unwrap();
                        dq_shell_i.dot(
                            &gamma_shell
                                .slice(s![m_i.slice.shell, m_j.slice.shell])
                                .dot(&dq_shell_j),
                        )
                    };
                let delta_ik: f64 =
                    if self.properties.type_of_pair(trimer.i, trimer.k) == PairType::Pair {
                        let index_ik: usize = self.properties.index_of_pair(trimer.i, trimer.k);
                        let pair_ik_energy: f64 =
                            self.pairs[index_ik].properties.last_energy().unwrap();
                        pair_ik_energy - m_i_energy - m_k_energy
                    } else {
                        let dq_shell_i: ArrayView1<f64> = m_i.properties.dq_shell().unwrap();
                        let dq_shell_k: ArrayView1<f64> = m_k.properties.dq_shell().unwrap();
                        dq_shell_i.dot(
                            &gamma_shell
                                .slice(s![m_i.slice.shell, m_k.slice.shell])
                                .dot(&dq_shell_k),
                        )
                    };
                let delta_jk: f64 =
                    if self.properties.type_of_pair(trimer.j, trimer.k) == PairType::Pair {
                        let index_jk: usize = self.properties.index_of_pair(trimer.j, trimer.k);
                        let pair_jk_energy: f64 =
                            self.pairs[index_jk].properties.last_energy().unwrap();
                        pair_jk_energy - m_j_energy - m_k_energy
                    } else {
                        let dq_shell_j: ArrayView1<f64> = m_j.properties.dq_shell().unwrap();
                        let dq_shell_k: ArrayView1<f64> = m_k.properties.dq_shell().unwrap();
                        dq_shell_j.dot(
                            &gamma_shell
                                .slice(s![m_j.slice.shell, m_k.slice.shell])
                                .dot(&dq_shell_k),
                        )
                    };

                // ZREF + HOP for partial bonds (BDA in trimer, BAA outside).
                // Bonds fully inside the trimer are healed (no HOP, no ZREF reduction).
                let partial_bonds = if !detached_bonds.is_empty() {
                    get_detached_bonds_for_trimer(
                        detached_bonds,
                        trimer.i,
                        trimer.j,
                        trimer.k,
                    )
                } else {
                    vec![]
                };

                // Add ghost atoms for partial bonds and extend trimer
                // Deduplicate: if multiple partial bonds share the same BAA position,
                // only create one ghost atom at that position.
                let ext_trimer_atoms: Vec<XtbAtom> = if !partial_bonds.is_empty() {
                    let mut ext = trimer_atoms.clone();
                    let mut seen_baa: Vec<usize> = Vec::new();
                    for bond in &partial_bonds {
                        if !seen_baa.contains(&bond.baa_global) {
                            seen_baa.push(bond.baa_global);
                            let mut ghost = XtbAtom::from(1u8);
                            ghost.xyz = atoms[bond.baa_global].xyz;
                            ext.push(ghost);
                        }
                    }
                    // Set adjusted n_elec (BEFORE prepare_scc)
                    let raw_n_elec: usize = trimer_atoms.iter().map(|a| a.n_elec).sum();
                    trimer
                        .properties
                        .set_n_elec(raw_n_elec - partial_bonds.len());
                    // Update trimer dimensions for extended atoms
                    trimer.n_real_atoms = trimer_atoms.len();
                    trimer.n_real_orbs = trimer.n_orbs;
                    let ext_basis = create_basis_set(&ext);
                    trimer.n_atoms = ext.len();
                    trimer.n_orbs = ext_basis.nbas;
                    trimer.n_real_shells =
                        m_i.n_real_shells + m_j.n_real_shells + m_k.n_real_shells;
                    ext
                } else {
                    trimer_atoms.clone()
                };

                // Build CN for trimer atoms by concatenating slices from global CN
                let cn_i = cn_super.slice(s![m_i.slice.atom_as_range()]);
                let cn_j = cn_super.slice(s![m_j.slice.atom_as_range()]);
                let cn_k = cn_super.slice(s![m_k.slice.atom_as_range()]);
                let mut cn_trimer = concatenate![Axis(0), cn_i, cn_j, cn_k];
                // Ghost atoms get CN=0
                if ext_trimer_atoms.len() > trimer_atoms.len() {
                    let mut ext_cn = Array1::zeros(ext_trimer_atoms.len());
                    ext_cn
                        .slice_mut(s![..trimer_atoms.len()])
                        .assign(&cn_trimer);
                    cn_trimer = ext_cn;
                }

                // Build the basis set on demand (with extended atoms including ghosts)
                trimer.init_basis(&ext_trimer_atoms);

                // prepare scc
                trimer.prepare_scc(
                    &ext_trimer_atoms[..],
                    m_i,
                    m_j,
                    m_k,
                    gamma_shell,
                    cn_trimer.view(),
                );

                // ZREF p_ref reduction + ghost zeroing + HOP projector for partial bonds
                if !partial_bonds.is_empty() {
                    let mut p_ref = trimer.properties.take_p_ref().unwrap();

                    // Count partial bonds per local atom
                    let n_atoms_i = m_i.n_real_atoms;
                    let n_atoms_j = m_j.n_real_atoms;
                    let mut partial_count = vec![0usize; trimer_atoms.len()];
                    for bond in &partial_bonds {
                        let bda_local = if bond.bda_fragment == trimer.i {
                            bond.bda_global - m_i.slice.atom_as_range().start
                        } else if bond.bda_fragment == trimer.j {
                            (bond.bda_global - m_j.slice.atom_as_range().start) + n_atoms_i
                        } else {
                            (bond.bda_global - m_k.slice.atom_as_range().start)
                                + n_atoms_i
                                + n_atoms_j
                        };
                        partial_count[bda_local] += 1;
                    }

                    // Reduce BDA p_ref diagonals (ZREF -= count)
                    for (bda_local, &count) in partial_count.iter().enumerate() {
                        if count == 0 {
                            continue;
                        }
                        let bda_n_elec = trimer_atoms[bda_local].n_elec as f64;
                        if bda_n_elec > 0.0 {
                            let scale = (bda_n_elec - count as f64) / bda_n_elec;
                            for shell in trimer.basis().shells.iter() {
                                if shell.atom_index == bda_local {
                                    for i in shell.sph_start..shell.sph_end {
                                        p_ref[[i, i]] *= scale;
                                    }
                                }
                            }
                        }
                    }

                    // Zero ghost atom reference occupations
                    for shell in trimer.basis().shells.iter() {
                        if shell.atom_index >= trimer.n_real_atoms {
                            for i in shell.sph_start..shell.sph_end {
                                p_ref[[i, i]] = 0.0;
                            }
                        }
                    }

                    trimer.properties.set_p_ref(p_ref.clone());
                    trimer.properties.set_p(p_ref);

                    // HOP projector for partial bonds
                    let s = trimer.properties.s().unwrap();
                    let positions =
                        |global_idx: usize| -> Vector3<f64> { atoms[global_idx].xyz };
                    if let Some(p_hop) = compute_total_hop_projector(
                        &partial_bonds,
                        &trimer.basis(),
                        s,
                        &positions,
                    ) {
                        trimer.properties.set_hop_projector(p_hop);
                    }
                }

                // run scc
                trimer.run_scc(&ext_trimer_atoms, config);

                // Truncate delta_dq_shell to real shells for embedding energy
                if !partial_bonds.is_empty() {
                    let ddq_shell_trunc = trimer
                        .properties
                        .delta_dq_shell()
                        .unwrap()
                        .slice(s![..trimer.n_real_shells])
                        .to_owned();
                    trimer.properties.set_delta_dq_shell(ddq_shell_trunc);
                }
                // and compute the SCC energy
                let trimer_energy: f64 = trimer.properties.last_energy().unwrap()
                    - m_i_energy
                    - m_j_energy
                    - m_k_energy
                    - delta_ij
                    - delta_ik
                    - delta_jk;

                if config.jobtype == "sp" {
                    // For SP: clear heavy properties, only keep delta_dq and last_energy
                    trimer.properties.reset_trimer_sp();
                    trimer.clear_basis();
                } else {
                    // Remove intermediate SCC properties no longer needed
                    trimer.properties.reset_scc_intermediates();
                }

                trimer_energy
            })
            .collect();

        let trimer_energy: Array1<f64> = Array::from(trimer_energies);
        trimer_energy.sum()
    }

    /// Shell-resolved embedding correction for the FMO3 trimers (interaction of
    /// each trimer's density change with the monomers outside the trimer).
    pub fn trimer_embedding_energy_shell(&self) -> f64 {
        // Reference to the shell-level Gamma matrix of the full system.
        let gamma_shell: ArrayView2<f64> = self.properties.gamma_shell().unwrap();
        let embedding_vec: Vec<f64> = self
            .trimers
            .par_iter()
            .map(|trimer| {
                // Reference to Monomers I, J, K.
                let m_i: &XtbMonomer = &self.monomers[trimer.i];
                let m_j: &XtbMonomer = &self.monomers[trimer.j];
                let m_k: &XtbMonomer = &self.monomers[trimer.k];

                // Shell-level charge differences (stored in properties)
                let dq_shell_i: ArrayView1<f64> = m_i.properties.dq_shell().unwrap();
                let dq_shell_j: ArrayView1<f64> = m_j.properties.dq_shell().unwrap();
                let dq_shell_k: ArrayView1<f64> = m_k.properties.dq_shell().unwrap();

                // ESP without self-interaction (stored as shell-level)
                let esp_q_i: ArrayView1<f64> = m_i.properties.esp_q().unwrap();
                let esp_q_j: ArrayView1<f64> = m_j.properties.esp_q().unwrap();
                let esp_q_k: ArrayView1<f64> = m_k.properties.esp_q().unwrap();

                // delta_dq at shell level (stored directly)
                let ddq_shell: ArrayView1<f64> = trimer.properties.delta_dq_shell().unwrap();
                let ddq_shell_i = ddq_shell.slice(s![..m_i.n_real_shells]);
                let ddq_shell_j = ddq_shell.slice(s![m_i.n_real_shells..m_i.n_real_shells + m_j.n_real_shells]);
                let ddq_shell_k = ddq_shell.slice(s![m_i.n_real_shells + m_j.n_real_shells..]);

                // Subtract interaction with other monomers at shell level
                let esp_i: Array1<f64> = &esp_q_i
                    - &gamma_shell
                        .slice(s![m_i.slice.shell, m_j.slice.shell])
                        .dot(&dq_shell_j)
                    - &gamma_shell
                        .slice(s![m_i.slice.shell, m_k.slice.shell])
                        .dot(&dq_shell_k);
                let esp_j: Array1<f64> = &esp_q_j
                    - &gamma_shell
                        .slice(s![m_j.slice.shell, m_i.slice.shell])
                        .dot(&dq_shell_i)
                    - &gamma_shell
                        .slice(s![m_j.slice.shell, m_k.slice.shell])
                        .dot(&dq_shell_k);
                let esp_k: Array1<f64> = &esp_q_k
                    - &gamma_shell
                        .slice(s![m_k.slice.shell, m_i.slice.shell])
                        .dot(&dq_shell_i)
                    - &gamma_shell
                        .slice(s![m_k.slice.shell, m_j.slice.shell])
                        .dot(&dq_shell_j);

                let mut embedding_terms: f64 = 0.0;
                embedding_terms += esp_i.dot(&ddq_shell_i);
                embedding_terms += esp_j.dot(&ddq_shell_j);
                embedding_terms += esp_k.dot(&ddq_shell_k);

                // subtract pair embeddings (all shell-level)
                let embedding_ij: f64 =
                    if self.properties.type_of_pair(trimer.i, trimer.j) == PairType::Pair {
                        let index_ij: usize = self.properties.index_of_pair(trimer.i, trimer.j);
                        let ddq_shell_pair: ArrayView1<f64> =
                            self.pairs[index_ij].properties.delta_dq_shell().unwrap();
                        let esp_q_i: ArrayView1<f64> = m_i.properties.esp_q().unwrap();
                        let esp_q_j: ArrayView1<f64> = m_j.properties.esp_q().unwrap();
                        let corr_i_shell = gamma_shell
                            .slice(s![m_i.slice.shell, m_j.slice.shell])
                            .dot(&dq_shell_j);
                        let corr_j_shell = gamma_shell
                            .slice(s![m_j.slice.shell, m_i.slice.shell])
                            .dot(&dq_shell_i);
                        let esp_i: Array1<f64> = &esp_q_i - &corr_i_shell;
                        let esp_j: Array1<f64> = &esp_q_j - &corr_j_shell;
                        esp_i.dot(&ddq_shell_pair.slice(s![..m_i.n_real_shells]))
                            + esp_j.dot(&ddq_shell_pair.slice(s![m_i.n_real_shells..]))
                    } else {
                        0.0
                    };
                let embedding_ik: f64 =
                    if self.properties.type_of_pair(trimer.i, trimer.k) == PairType::Pair {
                        let index_ik: usize = self.properties.index_of_pair(trimer.i, trimer.k);
                        let ddq_shell_pair: ArrayView1<f64> =
                            self.pairs[index_ik].properties.delta_dq_shell().unwrap();
                        let esp_q_i: ArrayView1<f64> = m_i.properties.esp_q().unwrap();
                        let esp_q_k: ArrayView1<f64> = m_k.properties.esp_q().unwrap();
                        let corr_i_shell = gamma_shell
                            .slice(s![m_i.slice.shell, m_k.slice.shell])
                            .dot(&dq_shell_k);
                        let corr_k_shell = gamma_shell
                            .slice(s![m_k.slice.shell, m_i.slice.shell])
                            .dot(&dq_shell_i);
                        let esp_i: Array1<f64> = &esp_q_i - &corr_i_shell;
                        let esp_k: Array1<f64> = &esp_q_k - &corr_k_shell;
                        esp_i.dot(&ddq_shell_pair.slice(s![..m_i.n_real_shells]))
                            + esp_k.dot(&ddq_shell_pair.slice(s![m_i.n_real_shells..]))
                    } else {
                        0.0
                    };
                let embedding_jk: f64 =
                    if self.properties.type_of_pair(trimer.j, trimer.k) == PairType::Pair {
                        let index_jk: usize = self.properties.index_of_pair(trimer.j, trimer.k);
                        let ddq_shell_pair: ArrayView1<f64> =
                            self.pairs[index_jk].properties.delta_dq_shell().unwrap();
                        let esp_q_j: ArrayView1<f64> = m_j.properties.esp_q().unwrap();
                        let esp_q_k: ArrayView1<f64> = m_k.properties.esp_q().unwrap();
                        let corr_j_shell = gamma_shell
                            .slice(s![m_j.slice.shell, m_k.slice.shell])
                            .dot(&dq_shell_k);
                        let corr_k_shell = gamma_shell
                            .slice(s![m_k.slice.shell, m_j.slice.shell])
                            .dot(&dq_shell_j);
                        let esp_j: Array1<f64> = &esp_q_j - &corr_j_shell;
                        let esp_k: Array1<f64> = &esp_q_k - &corr_k_shell;
                        esp_j.dot(&ddq_shell_pair.slice(s![..m_j.n_real_shells]))
                            + esp_k.dot(&ddq_shell_pair.slice(s![m_j.n_real_shells..]))
                    } else {
                        0.0
                    };
                embedding_terms -= embedding_ij + embedding_ik + embedding_jk;

                embedding_terms
            })
            .collect();
        let embedding: f64 = Array1::from_vec(embedding_vec).sum();

        embedding
    }

    /// Atom-resolved counterpart of [`Self::embedding_energy_shell`].
    pub fn embedding_energy(&self) -> f64 {
        // The embedding energy is initialized to zero.
        let mut embedding: f64 = 0.0;

        // Reference to the Gamma matrix of the full system.
        let gamma: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        for pair in self.pairs.iter() {
            // Reference to Monomer I.
            let m_i: &XtbMonomer = &self.monomers[pair.i];
            // Reference to Monomer J.
            let m_j: &XtbMonomer = &self.monomers[pair.j];
            // Reference to the charge differences of Monomer I.
            let dq_i: ArrayView1<f64> = m_i.properties.dq_ao().unwrap();
            // Reference to the charge differences of Monomer J.
            let dq_j: ArrayView1<f64> = m_j.properties.dq_ao().unwrap();
            // Electrostatic potential that acts on I without the self interaction with I.
            let esp_q_i: ArrayView1<f64> = m_i.properties.esp_q().unwrap();
            // ESP that acts on J without self-interaction.
            let esp_q_j: ArrayView1<f64> = m_j.properties.esp_q().unwrap();
            // Difference between the charge differences of the pair and the corresp. monomers
            let ddq: ArrayView1<f64> = pair.properties.delta_dq().unwrap();
            // The interaction with the other Monomer in the pair is subtracted.
            let esp_q_i: Array1<f64> =
                &esp_q_i - &gamma.slice(s![m_i.slice.orb, m_j.slice.orb]).dot(&dq_j);
            let esp_q_j: Array1<f64> =
                &esp_q_j - &gamma.slice(s![m_j.slice.orb, m_i.slice.orb]).dot(&dq_i);
            // The embedding energy for Monomer I in the pair is computed.
            embedding += esp_q_i.dot(&ddq.slice(s![..m_i.n_real_orbs]));
            // The embedding energy for Monomer J in the pair is computed.
            embedding += esp_q_j.dot(&ddq.slice(s![m_i.n_real_orbs..]));
        }
        embedding
    }

    /// Atom-resolved counterpart of [`Self::esd_pair_energy_shell`].
    pub fn esd_pair_energy(&self) -> f64 {
        let mut esd_energy: f64 = 0.0;
        for esd_pair in self.esd_pairs.iter() {
            let m_i: &XtbMonomer = &self.monomers[esd_pair.i];
            let m_j: &XtbMonomer = &self.monomers[esd_pair.j];
            esd_energy += m_i
                .properties
                .dq_ao()
                .unwrap()
                .dot(
                    &self
                        .properties
                        .gamma_ao()
                        .unwrap()
                        .slice(s![m_i.slice.orb, m_j.slice.orb]),
                )
                .dot(&m_j.properties.dq_ao().unwrap());
        }
        esd_energy
    }

    /// Atom-resolved counterpart of [`Self::trimer_scc_shell`].
    pub fn trimer_scc(&mut self) -> f64 {
        let atoms: &[XtbAtom] = &self.atoms[..];
        // Parallelization
        let monomers: &Vec<XtbMonomer> = &self.monomers;
        let config = &self.config;
        // Reference to the Gamma matrix of the full system.
        let gamma: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let cn_super: ArrayView1<f64> = self.properties.cn().unwrap();

        // trimer loop
        let trimer_energies: Vec<f64> = self
            .trimers
            .par_iter_mut()
            .map(|trimer| {
                // Get references to the corresponding monomers
                let m_i: &XtbMonomer = &monomers[trimer.i];
                let m_j: &XtbMonomer = &monomers[trimer.j];
                let m_k: &XtbMonomer = &monomers[trimer.k];
                let trimer_atoms: Vec<XtbAtom> = get_trimer_slice_xtb(
                    &atoms,
                    m_i.slice.atom_as_range(),
                    m_j.slice.atom_as_range(),
                    m_k.slice.atom_as_range(),
                );
                // get monomer energies
                let m_i_energy: f64 = m_i.properties.last_energy().unwrap();
                let m_j_energy: f64 = m_j.properties.last_energy().unwrap();
                let m_k_energy: f64 = m_k.properties.last_energy().unwrap();
                // check pair types and get pair energies
                let delta_ij: f64 =
                    if self.properties.type_of_pair_reduced(trimer.i, trimer.j) == PairType::Pair {
                        let index_ij: usize = self.properties.index_of_pair(trimer.i, trimer.j);
                        let pair_ij_energy: f64 =
                            self.pairs[index_ij].properties.last_energy().unwrap();
                        pair_ij_energy - m_i_energy - m_j_energy
                    } else {
                        // get the ESDIM energu
                        let esdim: f64 = m_i
                            .properties
                            .dq_ao()
                            .unwrap()
                            .dot(&gamma.slice(s![m_i.slice.orb, m_j.slice.orb]))
                            .dot(&m_j.properties.dq_ao().unwrap());
                        esdim
                    };
                let delta_ik: f64 =
                    if self.properties.type_of_pair_reduced(trimer.i, trimer.k) == PairType::Pair {
                        let index_ik: usize = self.properties.index_of_pair(trimer.i, trimer.k);
                        let pair_ik_energy: f64 =
                            self.pairs[index_ik].properties.last_energy().unwrap();
                        pair_ik_energy - m_i_energy - m_k_energy
                    } else {
                        // get the ESDIM energu
                        let esdim: f64 = m_i
                            .properties
                            .dq_ao()
                            .unwrap()
                            .dot(&gamma.slice(s![m_i.slice.orb, m_k.slice.orb]))
                            .dot(&m_k.properties.dq_ao().unwrap());
                        esdim
                    };
                let delta_jk: f64 =
                    if self.properties.type_of_pair_reduced(trimer.j, trimer.k) == PairType::Pair {
                        let index_jk: usize = self.properties.index_of_pair(trimer.j, trimer.k);
                        let pair_jk_energy: f64 =
                            self.pairs[index_jk].properties.last_energy().unwrap();
                        pair_jk_energy - m_j_energy - m_k_energy
                    } else {
                        // get the ESDIM energu
                        let esdim: f64 = m_j
                            .properties
                            .dq_ao()
                            .unwrap()
                            .dot(&gamma.slice(s![m_j.slice.orb, m_k.slice.orb]))
                            .dot(&m_k.properties.dq_ao().unwrap());
                        esdim
                    };

                // Build CN for trimer atoms by concatenating slices from global CN
                let cn_i = cn_super.slice(s![m_i.slice.atom_as_range()]);
                let cn_j = cn_super.slice(s![m_j.slice.atom_as_range()]);
                let cn_k = cn_super.slice(s![m_k.slice.atom_as_range()]);
                let cn_trimer = concatenate![Axis(0), cn_i, cn_j, cn_k];

                // Build the basis set on demand
                trimer.init_basis(&trimer_atoms);

                // prepare scc
                trimer.prepare_scc(
                    &trimer_atoms[..],
                    m_i,
                    m_j,
                    m_k,
                    self.properties.gamma_ao().unwrap(),
                    cn_trimer.view(),
                );
                // run scc
                trimer.run_scc(&trimer_atoms, config);
                // and compute the SCC energy
                let trimer_energy: f64 = trimer.properties.last_energy().unwrap()
                    - m_i_energy
                    - m_j_energy
                    - m_k_energy
                    - delta_ij
                    - delta_ik
                    - delta_jk;

                if config.jobtype == "sp" {
                    // For SP: clear heavy properties, only keep delta_dq and last_energy
                    trimer.properties.reset_trimer_sp();
                    trimer.clear_basis();
                } else {
                    // Remove intermediate SCC properties no longer needed
                    trimer.properties.reset_scc_intermediates();
                }

                trimer_energy
            })
            .collect();

        let trimer_energy: Array1<f64> = Array::from(trimer_energies);
        trimer_energy.sum()
    }

    /// Atom-resolved counterpart of [`Self::trimer_embedding_energy_shell`].
    pub fn trimer_embedding_energy(&self) -> f64 {
        // Reference to the Gamma matrix of the full system.
        let gamma: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let embedding_vec: Vec<f64> = self
            .trimers
            .par_iter()
            .map(|trimer| {
                // Reference to Monomer I.
                let m_i: &XtbMonomer = &self.monomers[trimer.i];
                // Reference to Monomer J.
                let m_j: &XtbMonomer = &self.monomers[trimer.j];
                // Reference to Monomer K.
                let m_k: &XtbMonomer = &self.monomers[trimer.k];

                // Reference to the charge differences of Monomer I.
                let dq_i: ArrayView1<f64> = m_i.properties.dq_ao().unwrap();
                // Reference to the charge differences of Monomer J.
                let dq_j: ArrayView1<f64> = m_j.properties.dq_ao().unwrap();
                // Reference to the charge differences of Monomer K.
                let dq_k: ArrayView1<f64> = m_k.properties.dq_ao().unwrap();

                // Electrostatic potential that acts on I without the self interaction with I.
                let esp_q_i: ArrayView1<f64> = m_i.properties.esp_q().unwrap();
                // ESP that acts on J without self-interaction.
                let esp_q_j: ArrayView1<f64> = m_j.properties.esp_q().unwrap();
                // ESP that acts on K without self-interaction.
                let esp_q_k: ArrayView1<f64> = m_k.properties.esp_q().unwrap();

                // Difference between the charge differences of the pair and the corresp. monomers
                let ddq: ArrayView1<f64> = trimer.properties.delta_dq().unwrap();

                // The interaction with the other Monomers in the trimer is subtracted.
                let esp_q_i: Array1<f64> = &esp_q_i
                    - &gamma.slice(s![m_i.slice.orb, m_j.slice.orb]).dot(&dq_j)
                    - &gamma.slice(s![m_i.slice.orb, m_k.slice.orb]).dot(&dq_k);
                let esp_q_j: Array1<f64> = &esp_q_j
                    - &gamma.slice(s![m_j.slice.orb, m_i.slice.orb]).dot(&dq_i)
                    - &gamma.slice(s![m_j.slice.orb, m_k.slice.orb]).dot(&dq_k);
                let esp_q_k: Array1<f64> = &esp_q_k
                    - &gamma.slice(s![m_k.slice.orb, m_i.slice.orb]).dot(&dq_i)
                    - &gamma.slice(s![m_k.slice.orb, m_j.slice.orb]).dot(&dq_j);

                let mut embedding_terms: f64 = 0.0;
                // The embedding energy for Monomer I in the trimer is computed.
                embedding_terms += esp_q_i.dot(&ddq.slice(s![..m_i.n_real_orbs]));
                // The embedding energy for Monomer J in the trimer is computed.
                embedding_terms += esp_q_j.dot(&ddq.slice(s![m_i.n_real_orbs..m_i.n_real_orbs + m_j.n_real_orbs]));
                // The embedding energy for Monomer K in the trimer is computed.
                embedding_terms += esp_q_k.dot(&ddq.slice(s![m_i.n_real_orbs + m_j.n_real_orbs..]));

                // substract pair embeddings
                let embedding_ij: f64 =
                    if self.properties.type_of_pair_reduced(trimer.i, trimer.j) == PairType::Pair {
                        let index_ij: usize = self.properties.index_of_pair(trimer.i, trimer.j);
                        let ddq = self.pairs[index_ij].properties.delta_dq().unwrap();
                        // Electrostatic potential that acts on I without the self interaction with I.
                        let esp_q_i: ArrayView1<f64> = m_i.properties.esp_q().unwrap();
                        // ESP that acts on J without self-interaction.
                        let esp_q_j: ArrayView1<f64> = m_j.properties.esp_q().unwrap();
                        // The interaction with the other Monomer in the pair is subtracted.
                        let esp_q_i: Array1<f64> =
                            &esp_q_i - &gamma.slice(s![m_i.slice.orb, m_j.slice.orb]).dot(&dq_j);
                        let esp_q_j: Array1<f64> =
                            &esp_q_j - &gamma.slice(s![m_j.slice.orb, m_i.slice.orb]).dot(&dq_i);
                        // The embedding energy for Monomer I in the pair is computed.
                        esp_q_i.dot(&ddq.slice(s![..m_i.n_real_orbs]))
                            + esp_q_j.dot(&ddq.slice(s![m_i.n_real_orbs..]))
                    } else {
                        0.0
                    };
                let embedding_ik: f64 =
                    if self.properties.type_of_pair_reduced(trimer.i, trimer.k) == PairType::Pair {
                        let index_ik: usize = self.properties.index_of_pair(trimer.i, trimer.k);
                        let ddq = self.pairs[index_ik].properties.delta_dq().unwrap();
                        // Electrostatic potential that acts on I without the self interaction with I.
                        let esp_q_i: ArrayView1<f64> = m_i.properties.esp_q().unwrap();
                        // ESP that acts on J without self-interaction.
                        let esp_q_k: ArrayView1<f64> = m_k.properties.esp_q().unwrap();
                        // The interaction with the other Monomer in the pair is subtracted.
                        let esp_q_i: Array1<f64> =
                            &esp_q_i - &gamma.slice(s![m_i.slice.orb, m_k.slice.orb]).dot(&dq_k);
                        let esp_q_k: Array1<f64> =
                            &esp_q_k - &gamma.slice(s![m_k.slice.orb, m_i.slice.orb]).dot(&dq_i);
                        // The embedding energy for Monomer I in the pair is computed.
                        esp_q_i.dot(&ddq.slice(s![..m_i.n_real_orbs]))
                            + esp_q_k.dot(&ddq.slice(s![m_i.n_real_orbs..]))
                    } else {
                        0.0
                    };
                let embedding_jk: f64 =
                    if self.properties.type_of_pair_reduced(trimer.j, trimer.k) == PairType::Pair {
                        let index_jk: usize = self.properties.index_of_pair(trimer.j, trimer.k);
                        let ddq = self.pairs[index_jk].properties.delta_dq().unwrap();
                        // Electrostatic potential that acts on I without the self interaction with I.
                        let esp_q_j: ArrayView1<f64> = m_j.properties.esp_q().unwrap();
                        // ESP that acts on J without self-interaction.
                        let esp_q_k: ArrayView1<f64> = m_k.properties.esp_q().unwrap();
                        // The interaction with the other Monomer in the pair is subtracted.
                        let esp_q_j: Array1<f64> =
                            &esp_q_j - &gamma.slice(s![m_j.slice.orb, m_k.slice.orb]).dot(&dq_k);
                        let esp_q_k: Array1<f64> =
                            &esp_q_k - &gamma.slice(s![m_k.slice.orb, m_j.slice.orb]).dot(&dq_j);
                        // The embedding energy for Monomer I in the pair is computed.
                        esp_q_j.dot(&ddq.slice(s![..m_j.n_real_orbs]))
                            + esp_q_k.dot(&ddq.slice(s![m_j.n_real_orbs..]))
                    } else {
                        0.0
                    };
                embedding_terms -= (embedding_ij + embedding_ik + embedding_jk);

                embedding_terms
            })
            .collect();
        let embedding: f64 = Array1::from_vec(embedding_vec).sum();

        embedding
    }
}
