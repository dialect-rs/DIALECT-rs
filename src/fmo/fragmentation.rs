use crate::constants::BOND_THRESHOLD;
use crate::initialization::Atom;
use hashbrown::HashSet;
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
                edges.push((i as usize, j as usize));
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
