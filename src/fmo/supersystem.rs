use crate::fmo::fragmentation::{build_graph, fragmentation, Graph};
use crate::fmo::helpers::{MolIncrements, MolIndices, MolecularSlice};
use crate::fmo::{get_pair_type, ESDPair, Monomer, Pair, PairType};
use crate::initialization::parameters::{RepulsivePotential, SlaterKoster};
use crate::initialization::{initialize_gamma_function, Atom};
use crate::io::Configuration;
use crate::properties::Properties;
use crate::scc::gamma_approximation::{gamma_atomwise, GammaFunction};
use crate::scc::h0_and_s::{h0_and_s, s_supersystem};
use crate::utils::Timer;
use chemfiles::Frame;
use hashbrown::HashMap;
use log::info;
use ndarray::prelude::*;
use ndarray::Slice;

#[derive(Debug, Clone)]
pub struct SuperSystem<'a> {
    /// Type that holds all the input settings from the user.
    pub config: Configuration,
    /// Vector with the data and the positions of the individual
    /// atoms that are stored as [Atom](crate::initialization::Atom)
    pub atoms: Vec<Atom>,
    /// Number of fragments in the whole system, this corresponds to self.molecules.len()
    pub n_mol: usize,
    /// List of individuals fragments which are stored as a [Monomer](crate::fmo::Monomer)
    pub monomers: Vec<Monomer<'a>>,
    /// [Vec] that holds the pairs of two fragments if they are close to each other. Each pair is
    /// stored as [Pair](crate::fmo::Pair) that holds all information necessary for scc calculations
    pub pairs: Vec<Pair<'a>>,
    /// [Vec] that holds pairs for which the energy is only calculated within the ESD approximation.
    /// Only a small and minimal amount of information is stored in this [ESDPair] type.
    pub esd_pairs: Vec<ESDPair<'a>>,
    /// Type that can hold calculated properties e.g. gamma matrix for the whole FMO system
    pub properties: Properties,
    /// Type of Gamma function. This can be either `Gaussian` or `Slater` type.
    pub gammafunction: GammaFunction,
    /// Gamma function for the long-range correction. Only used if long-range correction is requested
    pub gammafunction_lc: Option<GammaFunction>,
}

impl<'a>
    From<(
        Frame,
        Configuration,
        &'a SlaterKoster,
        &'a RepulsivePotential,
        Vec<Atom>,
        Vec<Atom>,
    )> for SuperSystem<'a>
{
    /// Creates a new [SuperSystem] from a [Vec](alloc::vec) of atomic numbers, the coordinates as an [Array2](ndarray::Array2) and
    /// the global configuration as [Configuration](crate::io::settings::Configuration).
    fn from(
        input: (
            Frame,
            Configuration,
            &'a SlaterKoster,
            &'a RepulsivePotential,
            Vec<Atom>,
            Vec<Atom>,
        ),
    ) -> Self {
        // Measure the time for the building of the struct
        let timer: Timer = Timer::start();

        let unique_atoms: Vec<Atom> = input.4;
        let atoms: Vec<Atom> = input.5;

        // Get the number of unpaired electrons from the input option
        let _n_unpaired: usize = match input.1.mol.multiplicity {
            1u8 => 0,
            _ => panic!("The specified multiplicity is not implemented"),
        };
        match input.1.mol.charge {
            0 => {}
            _ => {
                panic!("Charged systems are not implemented for the FMO routines.")
            }
        }

        // Create a new Properties type, which is empty
        let mut properties: Properties = Properties::new();

        // Initialize the unscreened Gamma function -> r_lr == 0.00
        let gf: GammaFunction =
            initialize_gamma_function(&unique_atoms, 0.0, input.1.use_gaussian_gamma);

        // Initialize the screened gamma function only if LRC is requested
        let gf_lc: Option<GammaFunction> = if input.1.lc.long_range_correction {
            Some(initialize_gamma_function(
                &unique_atoms,
                input.1.lc.long_range_radius,
                input.1.use_gaussian_gamma,
            ))
        } else {
            None
        };

        // Get all [Atom]s of the SuperSystem in a sorted order that corresponds to the order of
        // the monomers
        let mut sorted_atoms: Vec<Atom> = Vec::with_capacity(atoms.len());

        // Build a connectivity graph to distinguish the individual monomers from each other
        let graph: Graph = build_graph(atoms.len(), &atoms);
        // Here does the fragmentation happens
        let monomer_indices: Vec<Vec<usize>> = fragmentation(&graph);

        // Vec that stores all [Monomer]s
        let mut monomers: Vec<Monomer> = Vec::with_capacity(monomer_indices.len());

        // The [Monomer]s are initialized
        // PARALLEL: this loop should be parallelized
        let mut mol_indices: MolIndices = MolIndices::new();

        for (idx, indices) in monomer_indices.into_iter().enumerate() {
            // Clone the atoms that belong to this monomer, they will be stored in the sorted list
            let mut monomer_atoms: Vec<Atom> =
                indices.into_iter().map(|i| atoms[i].clone()).collect();

            // Count the number of orbitals
            let m_n_orbs: usize = monomer_atoms.iter().fold(0, |n, atom| n + atom.n_orbs);

            // Count the number of electrons.
            let n_elec: usize = monomer_atoms.iter().map(|atom| atom.n_elec).sum();

            // Number of occupied orbitals.
            let n_occ: usize = n_elec / 2;

            // Number of virtual orbitals.
            let n_virt: usize = m_n_orbs - n_occ;

            let mut props: Properties = Properties::new();
            props.set_n_occ(n_occ);
            props.set_n_virt(n_virt);

            let increments: MolIncrements = MolIncrements {
                atom: monomer_atoms.len(),
                orbs: m_n_orbs,
                occs: n_occ,
                virts: n_virt,
            };

            // Create the slices for the atoms, grads and orbitals
            let m_slice: MolecularSlice = MolecularSlice::new(mol_indices.clone(), increments);

            // Create the Monomer object
            let mut current_monomer = Monomer::new(
                monomer_atoms.len(),
                m_n_orbs,
                idx,
                m_slice,
                props,
                input.3,
                input.2,
                gf.clone(),
                gf_lc.clone(),
            );

            // Compute the number of electrons for the monomer and set the indices of the
            // occupied and virtual orbitals.
            current_monomer.set_mo_indices(n_elec);

            // Increment the indices..
            mol_indices.add(increments);

            // Save the current Monomer.
            monomers.push(current_monomer);

            // Save the Atoms from the current Monomer
            sorted_atoms.append(&mut monomer_atoms);
        }
        // Rename the sorted atoms
        let atoms: Vec<Atom> = sorted_atoms;

        // Calculate the number of atomic orbitals for the whole system as the sum of the monomer
        // number of orbitals
        let n_orbs: usize = mol_indices.orbs;
        // Set the number of occupied and virtual orbitals.
        properties.set_n_occ(mol_indices.occs);
        properties.set_n_virt(mol_indices.virts);

        // Compute the Gamma function between all atoms
        properties.set_gamma(gamma_atomwise(&gf, &atoms, atoms.len()));
        // Comupate the Gamma function with long-range correction
        if gf_lc.is_some() {
            properties.set_gamma_lr(gamma_atomwise(&gf_lc.clone().unwrap(), &atoms, atoms.len()));
        }

        // Initialize the close pairs and the ones that are treated within the ES-dimer approx
        let mut pairs: Vec<Pair<'a>> = Vec::new();
        let mut esd_pairs: Vec<ESDPair<'a>> = Vec::new();

        // Create a HashMap that maps the Monomers to the type of Pair. To identify if a pair of
        // monomers are considered a real pair or should be treated with the ESD approx.
        let mut pair_iter: usize = 0;
        let mut esd_iter: usize = 0;
        let mut pair_indices: HashMap<(usize, usize), usize> = HashMap::new();
        let mut esd_pair_indices: HashMap<(usize, usize), usize> = HashMap::new();
        let mut pair_types: HashMap<(usize, usize), PairType> = HashMap::new();

        // The construction of the [Pair]s requires that the [Atom]s in the atoms are ordered after
        // each monomer
        for (i, m_i) in monomers.iter().enumerate() {
            for (j, m_j) in monomers[(i + 1)..].iter().enumerate() {
                match get_pair_type(
                    &atoms[m_i.slice.atom_as_range()],
                    &atoms[m_j.slice.atom_as_range()],
                    input.1.vdw_scaling,
                ) {
                    PairType::Pair => {
                        pairs.push(Pair::new(i, i + j + 1, m_i, m_j, input.2, input.3));
                        pair_types.insert((m_i.index, m_j.index), PairType::Pair);
                        pair_indices.insert((m_i.index, m_j.index), pair_iter);
                        pair_iter += 1;
                    }
                    PairType::ESD => {
                        esd_pairs.push(ESDPair::new(i, i + j + 1, m_i, m_j, input.2, input.3));
                        pair_types.insert((m_i.index, m_j.index), PairType::ESD);
                        esd_pair_indices.insert((m_i.index, m_j.index), esd_iter);
                        esd_iter += 1;
                    }
                    _ => {}
                }
            }
        }
        properties.set_pair_types(pair_types);
        properties.set_pair_indices(pair_indices);
        properties.set_esd_pair_indices(esd_pair_indices);

        info!("{}", timer);

        let s = s_supersystem(n_orbs, &atoms, input.2);
        properties.set_s(s);

        Self {
            config: input.1,
            atoms: atoms,
            n_mol: monomers.len(),
            monomers: monomers,
            properties: properties,
            gammafunction: gf,
            gammafunction_lc: gf_lc,
            pairs,
            esd_pairs,
        }
    }
}

impl SuperSystem<'_> {
    pub fn update_xyz(&mut self, coordinates: ArrayView1<f64>) {
        let coordinates: ArrayView2<f64> = coordinates.into_shape([self.atoms.len(), 3]).unwrap();
        // PARALLEL
        for (atom, xyz) in self.atoms.iter_mut().zip(coordinates.outer_iter()) {
            atom.position_from_ndarray(xyz.to_owned());
        }
        //.for_each(|(atom, xyz)| atom.position_from_ndarray(xyz.to_owned()))
    }

    pub fn get_xyz(&self) -> Array1<f64> {
        let xyz_list: Vec<Vec<f64>> = self
            .atoms
            .iter()
            .map(|atom| atom.xyz.iter().cloned().collect())
            .collect();
        Array1::from_shape_vec(3 * self.atoms.len(), itertools::concat(xyz_list)).unwrap()
    }

    pub fn gamma_a(&self, a: usize, lrc: LRC) -> ArrayView2<f64> {
        self.gamma_a_b(a, a, lrc)
    }

    pub fn gamma_a_b(&self, a: usize, b: usize, lrc: LRC) -> ArrayView2<f64> {
        let atoms_a: Slice = self.monomers[a].slice.atom.clone();
        let atoms_b: Slice = self.monomers[b].slice.atom.clone();
        match lrc {
            LRC::ON => self.properties.gamma_lr_slice(atoms_a, atoms_b).unwrap(),
            LRC::OFF => self.properties.gamma_slice(atoms_a, atoms_b).unwrap(),
        }
    }

    pub fn gamma_ab_c(&self, a: usize, b: usize, c: usize, lrc: LRC) -> Array2<f64> {
        let n_atoms_a: usize = self.monomers[a].n_atoms;
        let mut gamma: Array2<f64> = Array2::zeros([
            n_atoms_a + self.monomers[b].n_atoms,
            self.monomers[c].n_atoms,
        ]);
        gamma
            .slice_mut(s![0..n_atoms_a, ..])
            .assign(&self.gamma_a_b(a, c, lrc));
        gamma
            .slice_mut(s![n_atoms_a.., ..])
            .assign(&self.gamma_a_b(b, c, lrc));
        gamma
    }

    pub fn gamma_ab_cd(&self, a: usize, b: usize, c: usize, d: usize, lrc: LRC) -> Array2<f64> {
        let n_atoms_a: usize = self.monomers[a].n_atoms;
        let n_atoms_c: usize = self.monomers[c].n_atoms;
        let mut gamma: Array2<f64> = Array2::zeros([
            n_atoms_a + self.monomers[b].n_atoms,
            n_atoms_c + self.monomers[d].n_atoms,
        ]);
        gamma
            .slice_mut(s![0..n_atoms_a, ..n_atoms_c])
            .assign(&self.gamma_a_b(a, c, lrc));
        gamma
            .slice_mut(s![n_atoms_a.., ..n_atoms_c])
            .assign(&self.gamma_a_b(b, c, lrc));
        gamma
            .slice_mut(s![0..n_atoms_a, n_atoms_c..])
            .assign(&self.gamma_a_b(a, d, lrc));
        gamma
            .slice_mut(s![n_atoms_a.., n_atoms_c..])
            .assign(&self.gamma_a_b(b, d, lrc));
        gamma
    }

    pub fn update_s(&mut self) {
        let n_orbs: usize = self.properties.n_occ().unwrap() + self.properties.n_virt().unwrap();
        let slako = &self.monomers[0].slako;
        let s = s_supersystem(n_orbs, &self.atoms, slako);
        self.properties.set_s(s);
    }
}

#[derive(Copy, Clone)]
pub enum LRC {
    ON,
    OFF,
}
