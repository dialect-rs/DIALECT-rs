use dialect_base::constants::VDW_SUM;
use dialect_state::{MolIncrements, MolIndices, MolecularSlice};
use dialect_utilities::fmo_helpers::{
    get_pair_slice_xtb, get_trimer_slice_xtb,
};
use dialect_state::PairType;
use dialect_utilities::fragmentation::{
    advanced_manual_fragmentation, build_graph_xtb, fragmentation,
    group_nearest_neighbor_fragments, manual_fragmentation, Graph,
};
use dialect_config::Configuration;
use dialect_utilities::io::xyz_frame_to_coordinates;
use dialect_state::Properties;
use crate::fmo::monomer::XtbMonomer;
use crate::fmo::pair::{get_pair_type_xtb, XtbESDPair, XtbPair};
use crate::fmo::trimer::XtbTrimer;
use crate::hop::{
    detect_detached_bonds_xtb, get_detached_bonds_for_fragment, get_detached_bonds_for_pair,
    get_detached_bonds_for_trimer, DetachedBond,
};
use crate::initialization::atom::XtbAtom;
use crate::initialization::basis::{create_basis_set, Basis};
use crate::initialization::helpers::{get_unique_atoms_xtb, init_gamma_func_xtb};
use crate::scc::gamma_matrix::XtbGammaFunction;
use xyz_parser::XyzFrame;
use hashbrown::HashMap;
use log::info;
use nalgebra::Vector3;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

/// Build the inputs for an FMO-xTB calculation from a geometry frame: the
/// `XtbAtom`s, the AO basis, and the (shell-resolved) xTB gamma function.
pub fn init_fmo_xtb(frame: XyzFrame) -> (Vec<XtbAtom>, Basis, XtbGammaFunction) {
    // get the atomic number and coordinates
    let (numbers, coords) = xyz_frame_to_coordinates(frame);
    // get the unique [Atom]s and the HashMap with the mapping from the numbers to the [Atom]s
    let tmp: (Vec<XtbAtom>, HashMap<u8, XtbAtom>) = get_unique_atoms_xtb(&numbers);
    let unique_atoms = tmp.0;
    let num_to_atom = tmp.1;

    // get all the Atom's from the HashMap
    let mut atoms: Vec<XtbAtom> = Vec::with_capacity(numbers.len());
    numbers
        .iter()
        .for_each(|num| atoms.push((*num_to_atom.get(num).unwrap()).clone()));
    // set the positions for each atom
    coords
        .outer_iter()
        .enumerate()
        .for_each(|(idx, position)| atoms[idx].position_from_slice(position.as_slice().unwrap()));
    // create the basis set
    let basis: Basis = create_basis_set(&atoms);
    // get xtb gamma function
    let gamma: XtbGammaFunction = init_gamma_func_xtb(&atoms, &basis, unique_atoms.len());

    (atoms, basis, gamma)
}

/// FMO-xTB supersystem: the full system partitioned into fragments
/// (monomers), together with the pair / ES-dimer / trimer lists used by the
/// FMO energy and gradient.
#[derive(Debug, Clone)]
pub struct XtbSuperSystem<'a> {
    /// Type that holds all the input settings from the user.
    pub config: Configuration,
    /// Vector with the data and the positions of the individual Atoms
    pub atoms: Vec<XtbAtom>,
    /// Number of fragments in the whole system, this corresponds to self.molecules.len()
    pub n_mol: usize,
    pub n_orbs: usize,
    pub basis: Basis,
    pub properties: Properties,
    pub monomers: Vec<XtbMonomer<'a>>,
    pub pairs: Vec<XtbPair<'a>>,
    pub esd_pairs: Vec<XtbESDPair<'a>>,
    pub trimers: Vec<XtbTrimer<'a>>,
    /// Detached bonds from covalent fragmentation (HOP).
    pub detached_bonds: Vec<DetachedBond>,
}

impl<'a> From<(Vec<XtbAtom>, Basis, &'a XtbGammaFunction, Configuration)> for XtbSuperSystem<'a> {
    fn from(input: (Vec<XtbAtom>, Basis, &'a XtbGammaFunction, Configuration)) -> Self {
        let atoms: Vec<XtbAtom> = input.0;
        // create the basis set
        let mut basis: Basis = input.1;
        let n_orbs: usize = basis.nbas;

        // Get all [Atom]s of the SuperSystem in a sorted order that corresponds to the order of
        // the monomers
        let mut sorted_atoms: Vec<XtbAtom> = Vec::with_capacity(atoms.len());

        // Build fragments either manually or automatically using graph-based fragmentation
        let t_frag = Instant::now();
        let monomer_indices: Vec<Vec<usize>> = if input.3.fmo.advanced_manual_fragmentation {
            advanced_manual_fragmentation(atoms.len(), &input.3.fmo.fragment_index_vector)
        } else if input.3.fmo.manual_fragmentation {
            manual_fragmentation(
                atoms.len(),
                input.3.fmo.fragment_atom_count,
                input.3.fmo.number_of_fragments,
            )
        } else {
            let graph: Graph = build_graph_xtb(atoms.len(), &atoms);
            fragmentation(&graph)
        };

        let monomer_indices = if input.3.fmo.fragments_per_monomer > 1 {
            let positions: Vec<Vector3<f64>> = atoms.iter().map(|a| a.xyz).collect();
            group_nearest_neighbor_fragments(
                monomer_indices,
                &positions,
                input.3.fmo.fragments_per_monomer,
            )
        } else {
            monomer_indices
        };

        info!(
            "{:>68} {:>8.2} s",
            "  fragmentation:",
            t_frag.elapsed().as_secs_f32()
        );

        // Detect detached bonds for covalent fragmentation (HOP).
        // Done AFTER fragments_per_monomer grouping so that bonds between
        // fragments that were fused into the same monomer are excluded.
        let mut detached_bonds: Vec<DetachedBond> = if input.3.fmo.covalent_fragmentation {
            let graph: Graph = build_graph_xtb(atoms.len(), &atoms);
            let bonds = detect_detached_bonds_xtb(&monomer_indices, &graph);
            info!(
                "{:>68} {:>8}",
                "  detached bonds (HOP):",
                bonds.len()
            );
            bonds
        } else {
            vec![]
        };

        // Build original→sorted atom index mapping before monomer_indices is consumed
        let orig_to_sorted: Vec<usize> = if !detached_bonds.is_empty() {
            let mut map = vec![0usize; atoms.len()];
            let mut sorted_pos = 0;
            for indices in monomer_indices.iter() {
                for &orig_idx in indices {
                    map[orig_idx] = sorted_pos;
                    sorted_pos += 1;
                }
            }
            map
        } else {
            vec![]
        };

        // Vec that stores all [Monomer]s
        let mut monomers: Vec<XtbMonomer> = Vec::with_capacity(monomer_indices.len());
        let mut mol_indices: MolIndices = MolIndices::new();
        // Create a new Properties type, which is empty
        let mut properties: Properties = Properties::new();

        for (idx, indices) in monomer_indices.into_iter().enumerate() {
            // Clone the atoms that belong to this monomer, they will be stored in the sorted list
            let mut monomer_atoms: Vec<XtbAtom> =
                indices.iter().map(|&i| atoms[i].clone()).collect();

            // --- Ghost boundary atoms and BDA info for covalent fragmentation (HOP) ---
            let mut ghost_atoms: Vec<XtbAtom> = Vec::new();
            let mut bda_local_indices: Vec<usize> = Vec::new();
            if input.3.fmo.covalent_fragmentation {
                for bond in detached_bonds.iter().filter(|b| b.bda_fragment == idx) {
                    // Ghost H at BAA position: provides basis + nuclear potential, zero electrons
                    let mut ghost = XtbAtom::from(1u8);
                    ghost.xyz = atoms[bond.baa_global].xyz;
                    ghost_atoms.push(ghost);
                    // BDA local index: position of bda_global in this monomer's atom list
                    if let Some(pos) = indices.iter().position(|&i| i == bond.bda_global) {
                        bda_local_indices.push(pos);
                    }
                }
            }

            // Build real basis for supersystem-level indexing (MolIncrements)
            let real_basis: Basis = create_basis_set(&monomer_atoms);
            let real_n_orbs: usize = real_basis.nbas;
            let real_n_shells: usize = real_basis.shells.len();
            let real_n_atoms: usize = monomer_atoms.len();

            // Build extended basis if ghost atoms present
            let (ext_basis, ext_n_orbs, ext_n_shells) = if ghost_atoms.is_empty() {
                (real_basis, real_n_orbs, real_n_shells)
            } else {
                let extended_atoms: Vec<XtbAtom> = monomer_atoms
                    .iter()
                    .chain(ghost_atoms.iter())
                    .cloned()
                    .collect();
                let ext_basis = create_basis_set(&extended_atoms);
                let ext_n_orbs = ext_basis.nbas;
                let ext_n_shells = ext_basis.shells.len();
                (ext_basis, ext_n_orbs, ext_n_shells)
            };

            // Real electron count (from real atoms only, for supersystem-level indexing)
            let real_n_elec: usize = monomer_atoms.iter().map(|atom| atom.n_elec).sum();
            // Number of cut bonds where this fragment has the BDA
            let n_hop_bonds = if input.3.fmo.covalent_fragmentation {
                detached_bonds
                    .iter()
                    .filter(|b| b.bda_fragment == idx)
                    .count()
            } else {
                0
            };
            // ZREF approach: reduce n_elec by 1 per cut bond.
            // The BDA atom's reference density is also reduced by 1 (in prepare_scc).
            // Ghost atoms contribute basis + nuclear potential but zero electrons.
            let ext_n_elec: usize = real_n_elec - n_hop_bonds;

            // MolIncrements use REAL counts (for supersystem-level array indexing)
            let real_n_occ: usize = real_n_elec / 2;
            let real_n_virt: usize = real_n_orbs - real_n_occ;

            // Extended occupied/virtual counts for monomer SCC (includes ghost electrons)
            let n_occ: usize = ext_n_elec / 2;
            let n_virt: usize = ext_n_orbs - n_occ;

            let mut props: Properties = Properties::new();
            props.set_n_occ(n_occ);
            props.set_n_virt(n_virt);
            props.set_n_elec(ext_n_elec);

            let increments: MolIncrements = MolIncrements {
                atom: real_n_atoms,
                orbs: real_n_orbs,
                occs: real_n_occ,
                virts: real_n_virt,
                shells: real_n_shells,
            };

            // Create the slices for the atoms, grads and orbitals
            let m_slice: MolecularSlice = MolecularSlice::new(mol_indices, increments);

            // Create the Monomer object with extended dimensions
            let mut current_monomer = XtbMonomer::new(
                real_n_atoms + ghost_atoms.len(),
                ext_n_orbs,
                ext_n_shells,
                idx,
                m_slice,
                props,
                ext_basis,
                &input.2,
            );

            // Store real counts, ghost atoms, and BDA local indices
            current_monomer.n_real_atoms = real_n_atoms;
            current_monomer.n_real_orbs = real_n_orbs;
            current_monomer.n_real_shells = real_n_shells;
            current_monomer.ghost_atoms = ghost_atoms;
            current_monomer.bda_local_indices = bda_local_indices;

            // Set the indices of the occupied and virtual orbitals (using extended n_elec)
            current_monomer.set_mo_indices(ext_n_elec);

            // Increment the indices..
            mol_indices.add(increments);

            // Save the current Monomer.
            monomers.push(current_monomer);

            // Save only the real Atoms (not ghosts) to the sorted supersystem list
            sorted_atoms.append(&mut monomer_atoms);
        }

        // Remap detached bond indices from original to sorted ordering
        if !detached_bonds.is_empty() {
            for bond in detached_bonds.iter_mut() {
                bond.bda_global = orig_to_sorted[bond.bda_global];
                bond.baa_global = orig_to_sorted[bond.baa_global];
            }
        }

        // Rename the sorted atoms
        let atoms: Vec<XtbAtom> = sorted_atoms;

        // update basis
        basis = create_basis_set(&atoms);

        // Set the number of occupied and virtual orbitals.
        properties.set_n_occ(mol_indices.occs);
        properties.set_n_virt(mol_indices.virts);

        // Precompute monomer proximity matrices for pairs and trimers.
        // Uses squared distances (no sqrt), single pass for both scalings, and rayon parallelism.
        let n_mon = monomers.len();
        let vdw_scaling = input.3.fmo.vdw_scaling;
        let trimer_vdw_scaling = input.3.fmo.trimer_vdw_scaling;
        let vdw_sq = vdw_scaling * vdw_scaling;
        let trimer_vdw_sq = trimer_vdw_scaling * trimer_vdw_scaling;

        // Precompute atom ranges so closures only capture simple slices/references
        let atom_ranges_vec: Vec<std::ops::Range<usize>> =
            monomers.iter().map(|m| m.slice.atom_as_range()).collect();
        let atom_ranges: &[std::ops::Range<usize>] = &atom_ranges_vec;
        let atoms_ref: &[XtbAtom] = &atoms;

        // Compute proximity for all (i, j) pairs in parallel
        let proximity: Vec<(usize, usize, bool, bool)> = (0..n_mon)
            .into_par_iter()
            .flat_map_iter(|i| {
                let atoms_i = &atoms_ref[atom_ranges[i].clone()];
                (i + 1..n_mon).map(move |j| {
                    let atoms_j = &atoms_ref[atom_ranges[j].clone()];
                    let mut close = false;
                    let mut close_trimer = false;
                    'pair_loop: for atomi in atoms_i.iter() {
                        for atomj in atoms_j.iter() {
                            let dx = atomi.xyz.x - atomj.xyz.x;
                            let dy = atomi.xyz.y - atomj.xyz.y;
                            let dz = atomi.xyz.z - atomj.xyz.z;
                            let dist_sq = dx * dx + dy * dy + dz * dz;
                            let vdw_sum = VDW_SUM[atomi.number as usize][atomj.number as usize];
                            let vdw_sum_sq = vdw_sum * vdw_sum;
                            if dist_sq < trimer_vdw_sq * vdw_sum_sq {
                                close = true;
                                close_trimer = true;
                                break 'pair_loop;
                            } else if dist_sq < vdw_sq * vdw_sum_sq {
                                close = true;
                            }
                        }
                    }
                    (i, j, close, close_trimer)
                })
            })
            .collect();

        // Scatter results into symmetric matrices
        let mut is_close: Vec<Vec<bool>> = vec![vec![false; n_mon]; n_mon];
        let mut is_close_trimer: Vec<Vec<bool>> = vec![vec![false; n_mon]; n_mon];
        for &(i, j, close, close_trimer) in &proximity {
            is_close[i][j] = close;
            is_close[j][i] = close;
            is_close_trimer[i][j] = close_trimer;
            is_close_trimer[j][i] = close_trimer;
        }


        // Collect close pair indices and build lookup maps.
        // Only close pairs (170K) are inserted into HashMaps — ESD pairs (25.9M) are
        // identified as the complement (any pair not in pair_types defaults to ESD).
        let mut pair_index_list: Vec<(usize, usize)> = Vec::new();
        let mut pair_indices: HashMap<(usize, usize), usize> = HashMap::new();
        let mut pair_types: HashMap<(usize, usize), PairType> = HashMap::new();

        for i in 0..n_mon {
            for j in (i + 1)..n_mon {
                if is_close[i][j] {
                    pair_types.insert((monomers[i].index, monomers[j].index), PairType::Pair);
                    pair_indices.insert(
                        (monomers[i].index, monomers[j].index),
                        pair_index_list.len(),
                    );
                    pair_index_list.push((i, j));
                }
            }
        }

        // Enumerate trimers using neighbor lists — O(n * d^2) instead of O(n^3).
        // A valid trimer (i,j,k) requires at least 2 of {close_ij, close_ik, close_jk}.
        // Every valid trimer has at least one "apex" vertex connected to both others.
        // We partition into 3 mutually exclusive cases by the smallest apex:
        //   Case 1: apex at i  → close_ij ∧ close_ik
        //   Case 2: apex at j  → close_ij ∧ close_jk ∧ ¬close_ik
        //   Case 3: apex at k  → close_ik ∧ close_jk ∧ ¬close_ij
        let mut trimer_indices: Vec<(usize, usize, usize)> = Vec::new();
        if input.3.fmo.use_three_body {
            // Upper adjacency: for each i, sorted list of j > i where is_close_trimer[i][j]
            let upper_adj: Vec<Vec<usize>> = (0..n_mon)
                .map(|i| (i + 1..n_mon).filter(|&j| is_close_trimer[i][j]).collect())
                .collect();
            // Lower adjacency: for each k, sorted list of i < k where is_close_trimer[k][i]
            let lower_adj: Vec<Vec<usize>> = (0..n_mon)
                .map(|k| (0..k).filter(|&i| is_close_trimer[k][i]).collect())
                .collect();

            // Case 1: apex at i — both j and k are upper neighbors of i
            for i in 0..n_mon {
                for (p, &j) in upper_adj[i].iter().enumerate() {
                    for &k in &upper_adj[i][p + 1..] {
                        trimer_indices.push((i, j, k));
                    }
                }
            }
            // Case 2: apex at j — i in upper_adj means close(i,j), k in upper_adj[j] means close(j,k)
            for i in 0..n_mon {
                for &j in &upper_adj[i] {
                    for &k in &upper_adj[j] {
                        if !is_close_trimer[i][k] {
                            trimer_indices.push((i, j, k));
                        }
                    }
                }
            }
            // Case 3: apex at k — pairs (i,j) from lower neighbors of k, with !close(i,j)
            for k in 0..n_mon {
                for (p, &i) in lower_adj[k].iter().enumerate() {
                    for &j in &lower_adj[k][p + 1..] {
                        if !is_close_trimer[i][j] {
                            trimer_indices.push((i, j, k));
                        }
                    }
                }
            }
        }

        // Build pairs (lightweight — basis is created on demand in prepare_scc)
        let gammafunc_pair_ref = &input.2;
        let pairs: Vec<XtbPair<'a>> = pair_index_list
            .iter()
            .map(|&(i, j)| XtbPair::new(i, j, &monomers[i], &monomers[j], gammafunc_pair_ref))
            .collect();


        // ESD pairs are the complement of close pairs — build directly from is_close
        let is_close_ref: &[Vec<bool>] = &is_close;
        let esd_pair_index_list: Vec<(usize, usize)> = (0..n_mon)
            .flat_map(|i| {
                (i + 1..n_mon)
                    .filter(move |&j| !is_close_ref[i][j])
                    .map(move |j| (i, j))
            })
            .collect();
        let mut esd_pair_indices: HashMap<(usize, usize), usize> = HashMap::new();
        for (idx, &(i, j)) in esd_pair_index_list.iter().enumerate() {
            esd_pair_indices.insert((monomers[i].index, monomers[j].index), idx);
        }
        let esd_pairs: Vec<XtbESDPair<'a>> = esd_pair_index_list
            .iter()
            .map(|&(i, j)| XtbESDPair::new(i, j, &monomers[i], &monomers[j], gammafunc_pair_ref))
            .collect();


        // Build trimers (lightweight — basis is created on demand in prepare_scc)
        let trimers: Vec<XtbTrimer<'a>> = trimer_indices
            .iter()
            .map(|&(i, j, k)| {
                XtbTrimer::new(
                    i,
                    j,
                    k,
                    &monomers[i],
                    &monomers[j],
                    &monomers[k],
                    gammafunc_pair_ref,
                )
            })
            .collect();


        properties.set_pair_types(pair_types);
        properties.set_pair_indices(pair_indices);
        properties.set_esd_pair_indices(esd_pair_indices);

        let mut config = input.3.clone();
        config.dispersion.a1 = 0.63;
        config.dispersion.a2 = 5.0;
        config.dispersion.s6 = 1.0;
        config.dispersion.s8 = 2.4;
        info!("Number of Pairs: {}", pairs.len());
        info!("Number of ESD Pairs: {}", esd_pairs.len());
        info!("Number of Trimers: {}", trimers.len());

        Self {
            config,
            atoms,
            n_mol: monomers.len(),
            n_orbs,
            basis,
            properties,
            monomers,
            pairs,
            esd_pairs,
            trimers,
            detached_bonds,
        }
    }
}


impl<'a> XtbSuperSystem<'a> {
    /// Rebuild the distance-based fragment classification for the current
    /// geometry: close pairs, ESD pairs and (when `fmo.use_three_body` is
    /// set) trimers, plus the pair-type / index maps in `properties`.
    /// Mirrors the classification done at construction.
    ///
    /// Call this after every geometry change (optimization steps,
    /// dynamics) so the pair/trimer lists follow the structure.
    ///
    /// Under HOP covalent fragmentation the *bond* topology
    /// (`detached_bonds`, fragment membership, electron counts) is fixed
    /// and stays untouched; only the distance-based pair/ESD/trimer
    /// classification is re-run -- fragments joined by a detached bond
    /// are always far inside the vdW cutoff and therefore remain real
    /// pairs. The ghost (boundary) atom positions on each monomer are
    /// refreshed from the current BAA coordinates, since the monomer SCC
    /// and the HOP shell gradients read them.
    pub fn update_fragmentation(&mut self) {
        if self.monomers.is_empty() {
            return;
        }

        // HOP: ghost atoms sit at the bond-attached-atom (BAA) positions
        // of the *current* geometry. Iteration order matches the ghost
        // construction order (detached_bonds filtered per BDA fragment).
        if self.config.fmo.covalent_fragmentation && !self.detached_bonds.is_empty() {
            let atoms = &self.atoms;
            let detached_bonds = &self.detached_bonds;
            for monomer in self.monomers.iter_mut() {
                let mut ghost_idx: usize = 0;
                for bond in detached_bonds
                    .iter()
                    .filter(|b| b.bda_fragment == monomer.index)
                {
                    monomer.ghost_atoms[ghost_idx].xyz = atoms[bond.baa_global].xyz;
                    ghost_idx += 1;
                }
            }
        }
        let n_mon = self.monomers.len();
        let vdw_scaling = self.config.fmo.vdw_scaling;
        let trimer_vdw_scaling = self.config.fmo.trimer_vdw_scaling;
        let vdw_sq = vdw_scaling * vdw_scaling;
        let trimer_vdw_sq = trimer_vdw_scaling * trimer_vdw_scaling;

        // Precompute atom ranges so closures only capture simple slices/references
        let atom_ranges_vec: Vec<std::ops::Range<usize>> = self
            .monomers
            .iter()
            .map(|m| m.slice.atom_as_range())
            .collect();
        let atom_ranges: &[std::ops::Range<usize>] = &atom_ranges_vec;
        let atoms_ref: &[XtbAtom] = &self.atoms;

        // Compute proximity for all (i, j) pairs in parallel
        let proximity: Vec<(usize, usize, bool, bool)> = (0..n_mon)
            .into_par_iter()
            .flat_map_iter(|i| {
                let atoms_i = &atoms_ref[atom_ranges[i].clone()];
                (i + 1..n_mon).map(move |j| {
                    let atoms_j = &atoms_ref[atom_ranges[j].clone()];
                    let mut close = false;
                    let mut close_trimer = false;
                    'pair_loop: for atomi in atoms_i.iter() {
                        for atomj in atoms_j.iter() {
                            let dx = atomi.xyz.x - atomj.xyz.x;
                            let dy = atomi.xyz.y - atomj.xyz.y;
                            let dz = atomi.xyz.z - atomj.xyz.z;
                            let dist_sq = dx * dx + dy * dy + dz * dz;
                            let vdw_sum = VDW_SUM[atomi.number as usize][atomj.number as usize];
                            let vdw_sum_sq = vdw_sum * vdw_sum;
                            if dist_sq < trimer_vdw_sq * vdw_sum_sq {
                                close = true;
                                close_trimer = true;
                                break 'pair_loop;
                            } else if dist_sq < vdw_sq * vdw_sum_sq {
                                close = true;
                            }
                        }
                    }
                    (i, j, close, close_trimer)
                })
            })
            .collect();

        // Scatter results into symmetric matrices
        let mut is_close: Vec<Vec<bool>> = vec![vec![false; n_mon]; n_mon];
        let mut is_close_trimer: Vec<Vec<bool>> = vec![vec![false; n_mon]; n_mon];
        for &(i, j, close, close_trimer) in &proximity {
            is_close[i][j] = close;
            is_close[j][i] = close;
            is_close_trimer[i][j] = close_trimer;
            is_close_trimer[j][i] = close_trimer;
        }

        // Close pairs + lookup maps (ESD pairs are the complement).
        let mut pair_index_list: Vec<(usize, usize)> = Vec::new();
        let mut pair_indices: HashMap<(usize, usize), usize> = HashMap::new();
        let mut pair_types: HashMap<(usize, usize), PairType> = HashMap::new();
        for i in 0..n_mon {
            for j in (i + 1)..n_mon {
                if is_close[i][j] {
                    pair_types
                        .insert((self.monomers[i].index, self.monomers[j].index), PairType::Pair);
                    pair_indices.insert(
                        (self.monomers[i].index, self.monomers[j].index),
                        pair_index_list.len(),
                    );
                    pair_index_list.push((i, j));
                }
            }
        }

        // Trimer enumeration: same apex-based three-case partition as the
        // constructor (a valid trimer needs >= 2 close edges).
        let mut trimer_indices: Vec<(usize, usize, usize)> = Vec::new();
        if self.config.fmo.use_three_body {
            let upper_adj: Vec<Vec<usize>> = (0..n_mon)
                .map(|i| (i + 1..n_mon).filter(|&j| is_close_trimer[i][j]).collect())
                .collect();
            let lower_adj: Vec<Vec<usize>> = (0..n_mon)
                .map(|k| (0..k).filter(|&i| is_close_trimer[k][i]).collect())
                .collect();
            for i in 0..n_mon {
                for (p, &j) in upper_adj[i].iter().enumerate() {
                    for &k in &upper_adj[i][p + 1..] {
                        trimer_indices.push((i, j, k));
                    }
                }
            }
            for i in 0..n_mon {
                for &j in &upper_adj[i] {
                    for &k in &upper_adj[j] {
                        if !is_close_trimer[i][k] {
                            trimer_indices.push((i, j, k));
                        }
                    }
                }
            }
            for k in 0..n_mon {
                for (p, &i) in lower_adj[k].iter().enumerate() {
                    for &j in &lower_adj[k][p + 1..] {
                        if !is_close_trimer[i][j] {
                            trimer_indices.push((i, j, k));
                        }
                    }
                }
            }
        }

        // Rebuild the pair / ESD pair / trimer lists (lightweight: the
        // bases are created on demand in prepare_scc).
        let gammafunction: &'a XtbGammaFunction = self.monomers[0].gammafunction;
        let pairs: Vec<XtbPair<'a>> = pair_index_list
            .iter()
            .map(|&(i, j)| {
                XtbPair::new(i, j, &self.monomers[i], &self.monomers[j], gammafunction)
            })
            .collect();
        let esd_pair_index_list: Vec<(usize, usize)> = (0..n_mon)
            .flat_map(|i| {
                let is_close_ref = &is_close;
                (i + 1..n_mon)
                    .filter(move |&j| !is_close_ref[i][j])
                    .map(move |j| (i, j))
            })
            .collect();
        let mut esd_pair_indices: HashMap<(usize, usize), usize> = HashMap::new();
        for (idx, &(i, j)) in esd_pair_index_list.iter().enumerate() {
            esd_pair_indices.insert((self.monomers[i].index, self.monomers[j].index), idx);
        }
        let esd_pairs: Vec<XtbESDPair<'a>> = esd_pair_index_list
            .iter()
            .map(|&(i, j)| {
                XtbESDPair::new(i, j, &self.monomers[i], &self.monomers[j], gammafunction)
            })
            .collect();
        let trimers: Vec<XtbTrimer<'a>> = trimer_indices
            .iter()
            .map(|&(i, j, k)| {
                XtbTrimer::new(
                    i,
                    j,
                    k,
                    &self.monomers[i],
                    &self.monomers[j],
                    &self.monomers[k],
                    gammafunction,
                )
            })
            .collect();

        self.properties.set_pair_types(pair_types);
        self.properties.set_pair_indices(pair_indices);
        self.properties.set_esd_pair_indices(esd_pair_indices);
        self.pairs = pairs;
        self.esd_pairs = esd_pairs;
        self.trimers = trimers;
    }
}

impl XtbSuperSystem<'_> {
    pub fn update_xyz(&mut self, coordinates: ArrayView1<f64>) {
        let coordinates: ArrayView2<f64> = coordinates.into_shape([self.atoms.len(), 3]).unwrap();
        for (atom, xyz) in self.atoms.iter_mut().zip(coordinates.outer_iter()) {
            atom.position_from_ndarray(xyz.to_owned());
        }
        // update the basis centers
        let covalent = !self.detached_bonds.is_empty();
        for mol in self.monomers.iter_mut() {
            let real_atoms = &self.atoms[mol.slice.atom_as_range()];
            if covalent && !mol.ghost_atoms.is_empty() {
                // Update ghost positions from BAA globals and build extended list
                let bonds = get_detached_bonds_for_fragment(&self.detached_bonds, mol.index);
                for (ghost_idx, bond) in bonds.iter().enumerate() {
                    mol.ghost_atoms[ghost_idx].xyz = self.atoms[bond.baa_global].xyz;
                }
                let ext_atoms: Vec<XtbAtom> = real_atoms
                    .iter()
                    .chain(mol.ghost_atoms.iter())
                    .cloned()
                    .collect();
                for func in mol.basis.basis_functions.iter_mut() {
                    let atom = &ext_atoms[func.atom_index];
                    func.center = (atom.xyz.x, atom.xyz.y, atom.xyz.z);
                }
            } else {
                for func in mol.basis.basis_functions.iter_mut() {
                    let atom = &real_atoms[func.atom_index];
                    func.center = (atom.xyz.x, atom.xyz.y, atom.xyz.z);
                }
            }
        }
        for pair in self.pairs.iter_mut() {
            let m_i = &self.monomers[pair.i];
            let m_j = &self.monomers[pair.j];
            let mut pair_atoms = get_pair_slice_xtb(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );
            if covalent {
                let partial_bonds =
                    get_detached_bonds_for_pair(&self.detached_bonds, pair.i, pair.j);
                let mut seen_baa: Vec<usize> = Vec::new();
                for bond in &partial_bonds {
                    if !seen_baa.contains(&bond.baa_global) {
                        seen_baa.push(bond.baa_global);
                        let mut ghost = XtbAtom::from(1u8);
                        ghost.xyz = self.atoms[bond.baa_global].xyz;
                        pair_atoms.push(ghost);
                    }
                }
            }
            for func in pair.basis.basis_functions.iter_mut() {
                let atom = &pair_atoms[func.atom_index];
                func.center = (atom.xyz.x, atom.xyz.y, atom.xyz.z);
            }
        }
        for trimer in self.trimers.iter_mut() {
            let m_i = &self.monomers[trimer.i];
            let m_j = &self.monomers[trimer.j];
            let m_k = &self.monomers[trimer.k];
            let mut trimer_atoms = get_trimer_slice_xtb(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
                m_k.slice.atom_as_range(),
            );
            if covalent {
                let partial_bonds = get_detached_bonds_for_trimer(
                    &self.detached_bonds,
                    trimer.i,
                    trimer.j,
                    trimer.k,
                );
                let mut seen_baa: Vec<usize> = Vec::new();
                for bond in &partial_bonds {
                    if !seen_baa.contains(&bond.baa_global) {
                        seen_baa.push(bond.baa_global);
                        let mut ghost = XtbAtom::from(1u8);
                        ghost.xyz = self.atoms[bond.baa_global].xyz;
                        trimer_atoms.push(ghost);
                    }
                }
            }
            let mut trimer_basis = match trimer.basis_opt.clone() {
                Some(b) => b,
                None => continue, // HOP trimers don't use basis_opt
            };
            for func in trimer_basis.basis_functions.iter_mut() {
                let atom = &trimer_atoms[func.atom_index];
                func.center = (atom.xyz.x, atom.xyz.y, atom.xyz.z);
            }
            trimer.basis_opt = Some(trimer_basis);
        }
        // update complete basis
        for func in self.basis.basis_functions.iter_mut() {
            let atom = &self.atoms[func.atom_index];
            func.center = (atom.xyz.x, atom.xyz.y, atom.xyz.z);
        }
    }

    pub fn get_xyz(&self) -> Array1<f64> {
        let xyz_list: Vec<Vec<f64>> = self
            .atoms
            .iter()
            .map(|atom| atom.xyz.iter().cloned().collect())
            .collect();
        Array1::from_shape_vec(3 * self.atoms.len(), itertools::concat(xyz_list)).unwrap()
    }
}
