use dialect_base::constants::BOND_THRESHOLD;
use dialect_dftb_core::atom::Atom;
use dialect_xtb_core::initialization::atom::XtbAtom;
use hashbrown::HashSet;
use nalgebra::Vector3;
use petgraph::graphmap::GraphMap;
use petgraph::prelude::*;
use petgraph::visit::Bfs;

pub type Graph = GraphMap<usize, (), Undirected>;

/// Construct a graph of a given set of [Atom]s. The edges are determined from the sum of the covalent
/// radii of two atoms scaled by a factor of 1.2.
pub fn build_graph(n_atoms: usize, atoms: &[Atom]) -> Graph {
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(n_atoms);
    for (i, atomi) in (0..n_atoms).zip(atoms.iter()) {
        for (j, atomj) in ((i + 1)..n_atoms).zip(atoms[(i + 1)..].iter()) {
            if (atomi - atomj).norm() < BOND_THRESHOLD[atomi.number as usize][atomj.number as usize]
            {
                edges.push((i, j));
            }
        }
    }
    Graph::from_edges(&edges)
}

pub fn build_graph_xtb(n_atoms: usize, atoms: &[XtbAtom]) -> Graph {
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(n_atoms);
    for (i, atomi) in (0..n_atoms).zip(atoms.iter()) {
        for (j, atomj) in ((i + 1)..n_atoms).zip(atoms[(i + 1)..].iter()) {
            if (atomi - atomj).norm()
                < 0.8 * BOND_THRESHOLD[atomi.number as usize][atomj.number as usize]
            {
                edges.push((i, j));
            }
        }
    }
    Graph::from_edges(&edges)
}

/// Returns all disconnected monomers from the graph. The algorithm works as follows:
/// 1. Create a HashSet containing all atom indices (0 - #atoms)
/// 2. Get one edge (atom) from the graph
/// 3. Search all neighbors of this atom by using Breadth-first search
/// 4. Delete the parent atom and all neighbors from the HashSet
/// 5. If there is no index left in the HashSet -> End
///    Otherwise go back to 1.
pub fn fragmentation(graph: &Graph) -> Vec<Vec<usize>> {
    let mut monomers: Vec<Vec<usize>> = Vec::new();
    let mut indices: HashSet<usize> = (0..graph.node_count()).collect();
    while !indices.is_empty() {
        let mut monomer: Vec<usize> = Vec::new();
        let mut bfs = Bfs::new(&graph, *indices.iter().next().unwrap());
        while let Some(nx) = bfs.next(&graph) {
            monomer.push(nx);
            indices.remove(&nx);
        }
        monomer.sort_unstable();
        monomers.push(monomer);
    }
    monomers.sort_unstable();
    monomers
}

/// Creates fragment indices by splitting atoms into equal-sized successive chunks.
pub fn manual_fragmentation(
    n_atoms: usize,
    fragment_atom_count: usize,
    number_of_fragments: usize,
) -> Vec<Vec<usize>> {
    // Validate parameters
    let expected_atoms = fragment_atom_count * number_of_fragments;
    if expected_atoms != n_atoms {
        panic!(
            "Manual fragmentation error: fragment_atom_count ({}) * number_of_fragments ({}) = {} \
             does not match total atom count ({}).",
            fragment_atom_count, number_of_fragments, expected_atoms, n_atoms
        );
    }
    if fragment_atom_count == 0 || number_of_fragments == 0 {
        panic!("Manual fragmentation error: fragment_atom_count and number_of_fragments cannot be zero.");
    }

    // Create successive chunks
    (0..number_of_fragments)
        .map(|i| {
            let start = i * fragment_atom_count;
            (start..start + fragment_atom_count).collect()
        })
        .collect()
}

/// Creates fragment indices from an explicit list of atom index vectors.
///
/// Each inner vector specifies the atom indices belonging to one fragment.
/// Validates that all atoms are covered exactly once.
pub fn advanced_manual_fragmentation(
    n_atoms: usize,
    fragment_index_vector: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    if fragment_index_vector.is_empty() {
        panic!("Advanced manual fragmentation error: fragment_index_vector is empty.");
    }
    // Validate: every atom index appears exactly once
    let mut seen = vec![false; n_atoms];
    for (frag_idx, frag) in fragment_index_vector.iter().enumerate() {
        for &atom_idx in frag {
            if atom_idx >= n_atoms {
                panic!(
                    "Advanced manual fragmentation error: atom index {} in fragment {} \
                     exceeds total atom count ({}).",
                    atom_idx, frag_idx, n_atoms
                );
            }
            if seen[atom_idx] {
                panic!(
                    "Advanced manual fragmentation error: atom index {} appears in multiple fragments.",
                    atom_idx
                );
            }
            seen[atom_idx] = true;
        }
    }
    for (i, &s) in seen.iter().enumerate() {
        if !s {
            panic!(
                "Advanced manual fragmentation error: atom index {} is not assigned to any fragment.",
                i
            );
        }
    }
    fragment_index_vector.to_vec()
}

/// Groups fragments into larger monomers using greedy nearest-neighbor agglomerative clustering.
///
/// After standard BFS fragmentation identifies individual molecules, this function merges
/// `group_size` nearest-neighbor fragments into a single FMO monomer based on centroid distances.
pub fn group_nearest_neighbor_fragments(
    fragments: Vec<Vec<usize>>,
    positions: &[Vector3<f64>],
    group_size: usize,
) -> Vec<Vec<usize>> {
    assert!(group_size > 0, "fragments_per_monomer must be > 0");
    if group_size == 1 {
        return fragments;
    }
    let n_frags = fragments.len();
    assert_eq!(
        n_frags % group_size,
        0,
        "Number of fragments ({}) is not evenly divisible by fragments_per_monomer ({})",
        n_frags,
        group_size
    );

    // Compute centroid of each fragment
    let centroids: Vec<Vector3<f64>> = fragments
        .iter()
        .map(|frag| {
            let sum: Vector3<f64> = frag.iter().map(|&i| positions[i]).sum();
            sum / frag.len() as f64
        })
        .collect();

    let mut remaining = vec![true; n_frags];
    let mut groups: Vec<Vec<usize>> = Vec::with_capacity(n_frags / group_size);

    for seed in 0..n_frags {
        if !remaining[seed] {
            continue;
        }
        remaining[seed] = false;

        // Start a new group with the seed fragment
        let mut group_atoms: Vec<usize> = fragments[seed].clone();
        let mut group_centroid = centroids[seed];
        let mut group_count: usize = 1;

        // Greedily add nearest neighbors until we reach group_size
        while group_count < group_size {
            // Find nearest ungrouped fragment by centroid distance
            let mut best_idx = usize::MAX;
            let mut best_dist = f64::MAX;
            for j in 0..n_frags {
                if !remaining[j] {
                    continue;
                }
                let dist = (centroids[j] - group_centroid).norm();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = j;
                }
            }

            if best_idx == usize::MAX {
                break; // no more ungrouped fragments
            }

            remaining[best_idx] = false;
            group_atoms.extend_from_slice(&fragments[best_idx]);

            // Update centroid as weighted average
            let old_n = group_count as f64;
            let new_n = (group_count + 1) as f64;
            group_centroid = (group_centroid * old_n + centroids[best_idx]) / new_n;
            group_count += 1;
        }

        group_atoms.sort_unstable();
        groups.push(group_atoms);
    }

    // Sort groups by first atom index for consistency
    groups.sort_unstable_by_key(|g| g[0]);
    groups
}
