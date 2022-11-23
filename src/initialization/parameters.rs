use crate::constants;
use crate::defaults;
use crate::param::Element;
use crate::utils::get_path_prefix;
use ndarray::prelude::*;
use ndarray_npy::write_npy;
use ron::de::from_str;
use rusty_fitpack;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn get_nan_vec() -> Vec<f64> {
    vec![f64::NAN]
}

fn get_nan_value() -> f64 {
    f64::NAN
}

fn init_none() -> Option<(Vec<f64>, Vec<f64>, usize)> {
    None
}

fn get_inf_value() -> f64 {
    f64::INFINITY
}

fn init_hashmap() -> HashMap<u8, (Vec<f64>, Vec<f64>, usize)> {
    HashMap::new()
}

/// A type that contains the atom-wise parameters for the DFTB calculation. The same `PseudoAtom`
/// type is used for the free and the confined atoms. The data will be serialized from the Ron files.
#[derive(Serialize, Deserialize)]
pub struct PseudoAtom {
    z: u8,
    pub hubbard_u: f64,
    n_elec: u8,
    #[serde(default = "get_inf_value")]
    r0: f64,
    pub r: Vec<f64>,
    radial_density: Vec<f64>,
    pub occupation: Vec<(u8, u8, u8)>,
    effective_potential: Vec<f64>,
    orbital_names: Vec<String>,
    pub energies: Vec<f64>,
    pub radial_wavefunctions: Vec<Vec<f64>>,
    pub angular_momenta: Vec<i8>,
    pub valence_orbitals: Vec<u8>,
    pub nshell: Vec<i8>,
    pub orbital_occupation: Vec<i8>,
    #[serde(default = "get_nan_value")]
    pub spin_coupling_constant: f64,
    #[serde(default = "get_nan_value")]
    energy_1s: f64,
    #[serde(default = "get_nan_value")]
    energy_2s: f64,
    #[serde(default = "get_nan_value")]
    energy_3s: f64,
    #[serde(default = "get_nan_value")]
    energy_4s: f64,
    #[serde(default = "get_nan_value")]
    energy_2p: f64,
    #[serde(default = "get_nan_value")]
    energy_3p: f64,
    #[serde(default = "get_nan_value")]
    energy_4p: f64,
    #[serde(default = "get_nan_value")]
    energy_3d: f64,
    #[serde(default = "get_nan_vec")]
    orbital_1s: Vec<f64>,
    #[serde(default = "get_nan_vec")]
    orbital_2s: Vec<f64>,
    #[serde(default = "get_nan_vec")]
    orbital_3s: Vec<f64>,
    #[serde(default = "get_nan_vec")]
    orbital_4s: Vec<f64>,
    #[serde(default = "get_nan_vec")]
    orbital_2p: Vec<f64>,
    #[serde(default = "get_nan_vec")]
    orbital_3p: Vec<f64>,
    #[serde(default = "get_nan_vec")]
    orbital_4p: Vec<f64>,
    #[serde(default = "get_nan_vec")]
    orbital_3d: Vec<f64>,
}

impl PseudoAtom {
    pub fn free_atom(element: &str) -> PseudoAtom {
        let path_prefix: String = get_path_prefix();
        let filename: String = format!(
            "{}/src/param/slaterkoster/free_pseudo_atom/{}.ron",
            path_prefix,
            element.to_lowercase()
        );
        let path: &Path = Path::new(&filename);
        let data: String = fs::read_to_string(path).expect("Unable to read file");
        from_str(&data).expect("RON file was not well-formatted")
    }

    pub fn confined_atom(element: &str) -> PseudoAtom {
        let path_prefix: String = get_path_prefix();
        let filename: String = format!(
            "{}/src/param/slaterkoster/confined_pseudo_atom/{}.ron",
            path_prefix,
            element.to_lowercase()
        );
        let path: &Path = Path::new(&filename);
        let data: String = fs::read_to_string(path).expect("Unable to read file");
        from_str(&data).expect("RON file was not well-formatted")
    }
}

pub struct PseudoAtomMio {
    z: u8,
    pub hubbard_u: f64,
    n_elec: u8,
    pub energies: Vec<f64>,
    pub angular_momenta: Vec<i8>,
    pub valence_orbitals: Vec<u8>,
    pub nshell: Vec<i8>,
    pub orbital_occupation: Vec<i8>,
}

/// Type that holds the mapping between element pairs and their [SlaterKosterTable].
/// This is basically a struct that allows to get the [SlaterKosterTable] without a strict
/// order of the [Element] tuple.
#[derive(Clone, Debug)]
pub struct SlaterKoster {
    pub map: HashMap<(Element, Element), SlaterKosterTable>,
}

impl SlaterKoster {
    /// Create a new [SlaterKoster] type, that maps a tuple of [Element] s to a [SlaterKosterTable].
    pub fn new() -> Self {
        SlaterKoster {
            map: HashMap::new(),
        }
    }

    /// Add a new [SlaterKosterTable] from a tuple of two [Element]s. THe
    pub fn add(&mut self, kind1: Element, kind2: Element) {
        self.map
            .insert((kind1, kind2), SlaterKosterTable::new(kind1, kind2));
    }

    pub fn add_from_handler(
        &mut self,
        kind1: Element,
        kind2: Element,
        handler: SkfHandler,
        optional_table: Option<SlaterKosterTable>,
        order: &str,
    ) {
        self.map.insert(
            (kind1, kind2),
            SlaterKosterTable::from((&handler, optional_table, order)),
        );
    }

    pub fn get(&self, kind1: Element, kind2: Element) -> &SlaterKosterTable {
        self.map
            .get(&(kind1, kind2))
            .unwrap_or_else(|| self.map.get(&(kind2, kind1)).unwrap())
    }
}

/// Type that holds the pairwise atomic parameters for the Slater-Koster matrix elements
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SlaterKosterTable {
    dipole: HashMap<(u8, u8, u8), Vec<f64>>,
    h: HashMap<(u8, u8, u8), Vec<f64>>,
    s: HashMap<(u8, u8, u8), Vec<f64>>,
    /// Atomic number of the first element of the atom pair
    z1: u8,
    /// Atomic number of the second element of the atom pair
    z2: u8,
    /// Grid with the atom-atom distances in bohr for which the H0 and overlap matrix elements
    /// are tabulated
    d: Vec<f64>,
    /// Maximal atom-atom distance of the grid. This is obtained by taking `d.max()`
    /// `dmax` is only checked in
    /// [get_h0_and_s_mu_nu](crate::param::slako_transformations::get_h0_and_s_mu_nu)
    #[serde(default = "get_nan_value")]
    pub dmax: f64,
    index_to_symbol: HashMap<u8, String>,
    #[serde(default = "init_hashmap")]
    /// Spline representation for the overlap matrix elements
    pub s_spline: HashMap<u8, (Vec<f64>, Vec<f64>, usize)>,
    /// Spline representation for the H0 matrix elements
    #[serde(default = "init_hashmap")]
    pub h_spline: HashMap<u8, (Vec<f64>, Vec<f64>, usize)>,
}

impl SlaterKosterTable {
    /// Creates a new [SlaterKosterTable] from two elements and splines the H0 and overlap
    /// matrix elements
    pub fn new(kind1: Element, kind2: Element) -> Self {
        let path_prefix: String = get_path_prefix();
        let (kind1, kind2) = if kind1 > kind2 {
            (kind2, kind1)
        } else {
            (kind1, kind2)
        };
        let filename: String = format!(
            "{}/src/param/slaterkoster/slako_tables/{}_{}.ron",
            path_prefix,
            kind1.symbol().to_lowercase(),
            kind2.symbol().to_lowercase()
        );
        let path: &Path = Path::new(&filename);
        let data: String =
            fs::read_to_string(path).expect(&*format! {"Unable to read file {}", &filename});
        let mut slako_table: SlaterKosterTable =
            from_str(&data).expect("RON file was not well-formatted");
        slako_table.dmax = slako_table.d[slako_table.d.len() - 1];
        slako_table.s_spline = slako_table.spline_overlap();
        slako_table.h_spline = slako_table.spline_hamiltonian();
        slako_table
    }

    pub(crate) fn spline_overlap(&self) -> HashMap<u8, (Vec<f64>, Vec<f64>, usize)> {
        let mut splines: HashMap<u8, (Vec<f64>, Vec<f64>, usize)> = HashMap::new();
        for ((_l1, _l2, i), value) in &self.s {
            let x: Vec<f64> = self.d.clone();
            let y: Vec<f64> = value.clone();
            splines.insert(
                *i,
                rusty_fitpack::splrep(
                    x, y, None, None, None, None, None, None, None, None, None, None,
                ),
            );
        }
        splines
    }

    pub(crate) fn spline_hamiltonian(&self) -> HashMap<u8, (Vec<f64>, Vec<f64>, usize)> {
        let mut splines: HashMap<u8, (Vec<f64>, Vec<f64>, usize)> = HashMap::new();
        for ((_l1, _l2, i), value) in &self.h {
            let x: Vec<f64> = self.d.clone();
            let y: Vec<f64> = value.clone();
            splines.insert(
                *i,
                rusty_fitpack::splrep(
                    x, y, None, None, None, None, None, None, None, None, None, None,
                ),
            );
        }
        splines
    }
}

/// Type that holds the mapping between element pairs and their [RepulsivePotentialTable].
/// This is basically a struct that allows to get the [RepulsivePotentialTable] without a s
/// order of the [Element] tuple.
#[derive(Clone, Debug)]
pub struct RepulsivePotential {
    pub map: HashMap<(Element, Element), RepulsivePotentialTable>,
}

impl RepulsivePotential {
    /// Create a new RepulsivePotential, to map the [Element] pairs to a [RepulsivePotentialTable]
    pub fn new() -> Self {
        RepulsivePotential {
            map: HashMap::new(),
        }
    }

    /// Add a [RepulsivePotentialTable] from a pair of two [Element]s
    pub fn add(&mut self, kind1: Element, kind2: Element) {
        self.map
            .insert((kind1, kind2), RepulsivePotentialTable::new(kind1, kind2));
    }

    /// Return the [RepulsivePotentialTable] for the tuple of two [Element]s. The order of
    /// the tuple does not play a role.
    pub fn get(&self, kind1: Element, kind2: Element) -> &RepulsivePotentialTable {
        self.map
            .get(&(kind1, kind2))
            .unwrap_or_else(|| self.map.get(&(kind2, kind1)).unwrap())
    }
}

/// Type that contains the repulsive potential between a pair of atoms and their derivative as
/// splines.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RepulsivePotentialTable {
    /// Repulsive energy in Hartree on the grid `d`
    pub vrep: Vec<f64>,
    /// Atomic number of first element of the pair
    z1: u8,
    /// Atomic number of the second element of the pair
    z2: u8,
    /// Grid for which the repulsive energies are tabulated in bohr.
    d: Vec<f64>,
    /// Spline representation as a tuple of ticks, coefficients and the degree
    #[serde(default = "init_none")]
    spline_rep: Option<(Vec<f64>, Vec<f64>, usize)>,
    /// Maximal atom-atom distance for which the repulsive energy is tabulated in the parameter file.
    /// The value is set from d.max()
    #[serde(default = "get_nan_value")]
    dmax: f64,
}

impl RepulsivePotentialTable {
    /// Create a new [RepulsivePotentialTable] from two [Elements]. The parameter file will be read
    /// and the repulsive energy will be splined and the spline representation will be stored.
    pub fn new(kind1: Element, kind2: Element) -> Self {
        let path_prefix: String = get_path_prefix();
        let (kind1, kind2) = if kind1 > kind2 {
            (kind2, kind1)
        } else {
            (kind1, kind2)
        };
        let filename: String = format!(
            "{}/src/param/repulsive_potential/reppot_tables/{}_{}.ron",
            path_prefix,
            kind1.symbol().to_lowercase(),
            kind2.symbol().to_lowercase()
        );
        let path: &Path = Path::new(&filename);
        let data: String = fs::read_to_string(path).expect("Unable to read file");
        let mut reppot_table: RepulsivePotentialTable =
            from_str(&data).expect("RON file was not well-formatted");
        reppot_table.spline_rep();
        reppot_table.dmax = reppot_table.d[reppot_table.d.len() - 1];
        return reppot_table;
    }

    /// Create the spline representation by calling the [splrep](rusty_fitpack::splrep) Routine.
    fn spline_rep(&mut self) {
        let spline: (Vec<f64>, Vec<f64>, usize) = rusty_fitpack::splrep(
            self.d.clone(),
            self.vrep.clone(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        self.spline_rep = Some(spline);
    }

    /// Evaluate the spline at the atom-atom distance `x`. The units of x are in bohr.
    /// If `x` > `dmax` then 0 is returned, where `dmax` is the maximal atom-atom distance on the grid
    /// for which the repulsive energy is reported
    pub fn spline_eval(&self, x: f64) -> f64 {
        match &self.spline_rep {
            Some((t, c, k)) => {
                if x <= self.dmax {
                    rusty_fitpack::splev_uniform(t, c, *k, x)
                } else {
                    0.0
                }
            }
            None => panic!("No spline representation available"),
        }
    }
    /// Evaluate the first derivate of the energy w.r.t. to the atomic displacements. The units
    /// of the distance `x` are also in bohr.  If `x` > `dmax` then 0 is returned,
    /// where `dmax` is the maximal atom-atom distance on the grid
    /// for which the repulsive energy is reported
    pub fn spline_deriv(&self, x: f64) -> f64 {
        match &self.spline_rep {
            Some((t, c, k)) => {
                if x <= self.dmax {
                    rusty_fitpack::splder_uniform(t, c, *k, x, 1)
                } else {
                    0.0
                }
            }
            None => panic!("No spline representation available"),
        }
    }
}

impl From<&SkfHandler> for PseudoAtomMio {
    fn from(skf_handler: &SkfHandler) -> Self {
        // split skf data in lines
        let lines: Vec<&str> = skf_handler.data_string.split("\n").collect();

        // read Ed Ep Es SPE Ud Up Us fd fp fs from the second line
        // of the slater koster file
        // Ed Ep Es: one-site energies
        // Ud Up Us: Hubbard Us of the different angular momenta
        // fd fp fs: occupation numbers of the orbitals
        let second_line: Vec<f64> = process_slako_line(lines[1]);
        let energies: Array1<f64> = array![second_line[2], second_line[1], second_line[0]];
        let occupations_numbers: Array1<i8> = array![
            second_line[9] as i8,
            second_line[8] as i8,
            second_line[7] as i8
        ];
        let hubbard_u: Array1<f64> = array![second_line[6], second_line[5], second_line[4]];

        let electron_count: u8 = skf_handler.element_a.number();
        let mut valence_orbitals: Vec<u8> = Vec::new();
        let mut nshell: Vec<i8> = Vec::new();
        // set nshell depending on the electron count of the atom
        if electron_count < 3 {
            nshell.push(1);
        } else if electron_count < 11 {
            nshell.push(2);
            nshell.push(2);
        } else if electron_count < 19 {
            nshell.push(3);
            nshell.push(3);
        }

        // fill angular momenta
        let mut angular_momenta: Vec<i8> = Vec::new();
        for (it, occ) in occupations_numbers.iter().enumerate() {
            if occ > &0 {
                valence_orbitals.push(it as u8);
                if it == 0 {
                    angular_momenta.push(0);
                } else if it == 1 {
                    angular_momenta.push(1);
                } else if it == 2 {
                    angular_momenta.push(2);
                }
            }
        }

        // create PseudoAtom
        let pseudo_atom: PseudoAtomMio = PseudoAtomMio {
            z: skf_handler.element_a.number(),
            hubbard_u: hubbard_u[0],
            energies: energies.to_vec(),
            angular_momenta: angular_momenta,
            valence_orbitals: valence_orbitals,
            nshell: nshell,
            orbital_occupation: occupations_numbers.to_vec(),
            n_elec: skf_handler.element_a.number(),
        };
        pseudo_atom
    }
}

impl From<&SkfHandler> for RepulsivePotentialTable {
    fn from(skf_handler: &SkfHandler) -> Self {
        // split skf data in lines
        let lines: Vec<&str> = skf_handler.data_string.split("\n").collect();

        let mut count: usize = 0;
        // search beginning of repulsive potential in the skf file
        for (it, line) in lines.iter().enumerate() {
            if line.contains("Spline") {
                count = it;
                break;
            }
        }

        // get number of points and the cutoff from the second line
        let second_line: Vec<f64> = process_slako_line(lines[count + 1]);
        let n_int: usize = second_line[0] as usize;
        let cutoff: f64 = second_line[1];

        // Line 3: V(r < r0) = exp(-a1*r+a2) + a3   is r too small to be covered by the spline
        let third_line: Vec<f64> = process_slako_line(lines[count + 2]);
        let a_1: f64 = third_line[0];
        let a_2: f64 = third_line[1];
        let a_3: f64 = third_line[2];

        // read spline values from the skf file
        let mut rs: Array1<f64> = Array1::zeros(n_int);
        let mut cs: Array2<f64> = Array2::zeros((4, n_int));
        let mut last_coeffs: Array1<f64> = Array1::zeros(6);
        // start from the 4th line after "Spline"
        count = count + 3;
        let mut end: f64 = 0.0;
        let mut iteration_count: usize = 0;
        for it in (count..(n_int + count)) {
            let next_line: Vec<f64> = process_slako_line(lines[it]);
            rs[iteration_count] = next_line[0];
            if it == (n_int + count - 1) {
                let array: Array1<f64> = array![
                    next_line[2],
                    next_line[3],
                    next_line[4],
                    next_line[5],
                    next_line[6],
                    next_line[7]
                ];
                last_coeffs = array;
            } else {
                let array: Array1<f64> =
                    array![next_line[2], next_line[3], next_line[4], next_line[5]];
                cs.slice_mut(s![.., iteration_count]).assign(&array);
            }
            end = next_line[1];
            iteration_count += 1;
        }
        assert!((end - cutoff).abs() < f64::EPSILON);

        // Now we evaluate the spline on a equidistant grid
        let npoints: usize = 300;
        let d_arr: Array1<f64> = Array1::linspace(0.0, cutoff, npoints);
        let mut v_rep: Array1<f64> = Array1::zeros(npoints);

        let mut spline_counter: usize = 0;
        for (i, di) in d_arr.iter().enumerate() {
            if di < &rs[0] {
                v_rep[i] = (-&a_1 * di + a_2).exp() + a_3;
            } else {
                // find interval such that r[j] <= di < r[j+1]
                while spline_counter < (n_int - 2) && di >= &rs[spline_counter + 1] {
                    spline_counter += 1;
                }
                if di >= rs.last().unwrap() && di <= &cutoff && spline_counter < (n_int - 1) {
                    spline_counter += 1;
                }
                if spline_counter < (n_int - 1) {
                    assert!(rs[spline_counter] <= *di);
                    assert!(di < &rs[spline_counter + 1]);
                    let c_arr: ArrayView1<f64> = cs.slice(s![.., spline_counter]);
                    let dx = di - rs[spline_counter];
                    let val =
                        c_arr[0] + c_arr[1] * dx + c_arr[2] * dx.powi(2) + c_arr[3] * dx.powi(3);
                    v_rep[i] = val;
                } else if spline_counter == (n_int - 1) {
                    let c_arr: ArrayView1<f64> = last_coeffs.view();
                    let dx = di - rs[spline_counter];

                    v_rep[i] = c_arr[0]
                        + c_arr[1] * dx
                        + c_arr[2] * dx.powi(2)
                        + c_arr[3] * dx.powi(3)
                        + c_arr[4] * dx.powi(4)
                        + c_arr[5] * dx.powi(5);
                } else {
                    v_rep[i] = 0.0;
                }
            }
        }

        let dmax: f64 = d_arr[d_arr.len() - 1];

        let mut rep_table: RepulsivePotentialTable = RepulsivePotentialTable {
            dmax: dmax,
            z1: skf_handler.element_a.number(),
            z2: skf_handler.element_b.number(),
            vrep: v_rep.to_vec(),
            d: d_arr.to_vec(),
            spline_rep: None,
        };
        rep_table.spline_rep();
        rep_table
    }
}

impl From<(&SkfHandler, Option<SlaterKosterTable>, &str)> for SlaterKosterTable {
    fn from(skf: (&SkfHandler, Option<SlaterKosterTable>, &str)) -> Self {
        // split skf data in lines
        let mut lines: Vec<&str> = skf.0.data_string.split("\n").collect();

        // read the first line of the skf file
        // it contains the r0 parameter/the grid distance and
        // the number of grid points
        let first_line: Vec<f64> = process_slako_line(lines[0]);
        let grid_dist: f64 = first_line[0];
        let npoints: usize = first_line[1] as usize;

        // remove first line
        lines.remove(0);
        if skf.0.element_a.number() == skf.0.element_b.number() {
            // remove second line
            lines.remove(0);
        }
        // remove second/third line
        lines.remove(0);

        // create grid
        let d_arr: Array1<f64> =
            Array1::linspace(0.02, grid_dist * ((npoints - 1) as f64), npoints);

        let next_line: Vec<f64> = process_slako_line(lines[0]);
        let length: usize = next_line.len() / 2;
        assert!(length == 10);

        // create vector of tausymbols, which correspond to the orbital combinations
        let tausymbols: Array1<&str> = match (skf.2) {
            ("ab") => (constants::TAUSYMBOLS_AB.iter().cloned().collect()),
            ("ba") => (constants::TAUSYMBOLS_BA.iter().cloned().collect()),
            _ => panic!("Wrong order specified! Only 'ab' or 'ba' is allowed!"),
        };
        let length_tau: usize = tausymbols.len();

        // define hashmaps for h, s and dipole
        let mut h: HashMap<(u8, u8, u8), Vec<f64>> = HashMap::new();
        let mut s: HashMap<(u8, u8, u8), Vec<f64>> = HashMap::new();
        let mut dipole: HashMap<(u8, u8, u8), Vec<f64>> = HashMap::new();

        // Fill hashmaps for h and s with the values of the slako table
        // for the order 'ab' to combine them with 'ba'
        if skf.1.is_some() {
            let slako_table: SlaterKosterTable = skf.1.unwrap();
            h = slako_table.h.clone();
            s = slako_table.s.clone();
            dipole = slako_table.dipole.clone();
        }

        // create Vector of arrays for the spline values of h and s
        // for each of the corresponding tausymbols
        let mut vec_h_arrays: Vec<Array1<f64>> = Vec::new();
        let mut vec_s_arrays: Vec<Array1<f64>> = Vec::new();
        for it in (0..10) {
            vec_s_arrays.push(Array1::zeros(npoints));
            vec_h_arrays.push(Array1::zeros(npoints));
        }
        let temp_vec: Vec<f64> = Array1::zeros(npoints).to_vec();

        // fill all arrays with spline values
        for it in (0..npoints) {
            let next_line: Vec<f64> = process_slako_line(lines[it]);
            for (pos, tausym) in tausymbols.slice(s![-10..]).iter().enumerate() {
                let symbol: (u8, i32, u8, i32) = constants::SYMBOL_2_TAU[*tausym];
                let l1: u8 = symbol.0;
                let l2: u8 = symbol.2;

                let mut orbital_parity: f64 = 0.0;
                if skf.2 == "ba" {
                    orbital_parity = -1.0_f64.powi((l1 + l2) as i32);
                } else {
                    orbital_parity = 1.0;
                }
                vec_h_arrays[pos][it] = orbital_parity * next_line[pos];
                vec_s_arrays[pos][it] = orbital_parity * next_line[length_tau + pos];
            }
        }

        // fill hashmaps with the spline values
        for (pos, tausymbol) in tausymbols.slice(s![-10..]).iter().enumerate() {
            let symbol: (u8, i32, u8, i32) = constants::SYMBOL_2_TAU[*tausymbol];
            let index: u8 = get_tau_2_index(symbol);
            if !h.contains_key(&(symbol.0, symbol.2, index)) {
                h.insert((symbol.0, symbol.2, index), vec_h_arrays[pos].to_vec());
            }
            if !s.contains_key(&(symbol.0, symbol.2, index)) {
                s.insert((symbol.0, symbol.2, index), vec_s_arrays[pos].to_vec());
            }
            dipole.insert((symbol.0, symbol.2, index), temp_vec.clone());
        }

        //create Slako table
        let dmax: f64 = d_arr[d_arr.len() - 1];
        let mut slako: SlaterKosterTable = SlaterKosterTable {
            dipole: dipole,
            s: s,
            h: h,
            d: d_arr.to_vec(),
            dmax: dmax,
            z1: skf.0.element_a.number(),
            z2: skf.0.element_b.number(),
            h_spline: init_hashmap(),
            s_spline: init_hashmap(),
            index_to_symbol: get_index_to_symbol(),
        };
        if skf.2 == "ba" || (skf.0.element_a == skf.0.element_b) {
            slako.s_spline = slako.spline_overlap();
            slako.h_spline = slako.spline_hamiltonian();
        }
        slako
    }
}

#[derive(Clone)]
pub struct SkfHandler {
    pub element_a: Element,
    pub element_b: Element,
    pub data_string: String,
}

impl SkfHandler {
    pub fn new(element_a: Element, element_b: Element, path_prefix: String) -> SkfHandler {
        let element_1: &str = element_a.symbol();
        let element_2: &str = element_b.symbol();
        let filename: String = format!("{}/{}-{}.skf", path_prefix, element_1, element_2);
        let path: &Path = Path::new(&filename);
        let data: String = fs::read_to_string(path).expect("Unable to read file");

        SkfHandler {
            element_a: element_a,
            element_b: element_b,
            data_string: data,
        }
    }
}

pub fn process_slako_line(line: &str) -> Vec<f64> {
    // convert a line into a list of column values respecting the
    // strange format conventions used in DFTB+ Slater-Koster files.
    // In Slater-Koster files used by DFTB+ zero columns
    // are not written: e.g. 4*0.0 has to be replaced
    // by four columns with zeros 0.0 0.0 0.0 0.0.

    let line: String = line.replace(",", " ");
    let new_line: Vec<&str> = line.split(" ").collect();
    // println!("new line {:?}",new_line);
    let mut float_vec: Vec<f64> = Vec::new();
    for string in new_line {
        if string.contains("*") {
            let temp: Vec<&str> = string.split("*").collect();
            let count: usize = temp[0].trim().parse::<usize>().unwrap();
            let value: f64 = temp[1].trim().parse::<f64>().unwrap();
            for it in (0..count) {
                float_vec.push(value);
            }
        } else {
            if string.len() > 0 && string.contains("\t") == false {
                // println!("string {:?}",string);
                let value: f64 = string.trim().parse::<f64>().unwrap();
                float_vec.push(value);
            }
        }
    }
    return float_vec;
}

fn get_tau_2_index(tuple: (u8, i32, u8, i32)) -> u8 {
    let v1: u8 = tuple.0;
    let v2: i32 = tuple.1;
    let v3: u8 = tuple.2;
    let v4: i32 = tuple.3;
    let value: u8 = match (v1, v2, v3, v4) {
        (0, 0, 0, 0) => 0,
        (0, 0, 1, 0) => 2,
        (0, 0, 2, 0) => 3,
        (1, 0, 0, 0) => 4,
        (1, -1, 1, -1) => 5,
        (1, 0, 1, 0) => 6,
        (1, 1, 1, 1) => 5,
        (1, -1, 2, -1) => 7,
        (1, 0, 2, 0) => 8,
        (1, 1, 2, 1) => 7,
        (2, 0, 0, 0) => 9,
        (2, -1, 1, -1) => 10,
        (2, 0, 1, 0) => 11,
        (2, 1, 1, 1) => 10,
        (2, -2, 2, -2) => 12,
        (2, -1, 2, -1) => 13,
        (2, 0, 2, 0) => 14,
        (2, 1, 2, 1) => 13,
        (2, 2, 2, 2) => 12,
        _ => panic!("false combination for tau_2_index!"),
    };
    return value;
}

fn get_index_to_symbol() -> HashMap<u8, String> {
    let mut index_to_symbol: HashMap<u8, String> = HashMap::new();
    index_to_symbol.insert(0, String::from("ss_sigma"));
    index_to_symbol.insert(2, String::from("ss_sigma"));
    index_to_symbol.insert(3, String::from("sp_sigma"));
    index_to_symbol.insert(4, String::from("sd_sigma"));
    index_to_symbol.insert(5, String::from("ps_sigma"));
    index_to_symbol.insert(6, String::from("pp_pi"));
    index_to_symbol.insert(7, String::from("pp_sigma"));
    index_to_symbol.insert(8, String::from("pd_pi"));
    index_to_symbol.insert(9, String::from("pd_sigma"));
    index_to_symbol.insert(10, String::from("ds_sigma"));
    index_to_symbol.insert(11, String::from("dp_pi"));
    index_to_symbol.insert(12, String::from("dp_sigma"));
    index_to_symbol.insert(13, String::from("dd_delta"));
    index_to_symbol.insert(14, String::from("dd_pi"));
    index_to_symbol
}
