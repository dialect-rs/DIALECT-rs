use crate::defaults;
use crate::fmo::SuperSystem;
use crate::initialization::parameters::SlaterKoster;
use crate::param::slako_transformations::{directional_cosines, slako_transformation};
// use crate::scc::gamma_approximation::{
//     gamma_ao_wise_shell_resolved, gamma_ao_wise_shell_resolved_ab, gamma_atomwise,
//     gamma_atomwise_ab, GammaFunction,
// };
use hashbrown::HashMap;
use ndarray::prelude::*;

impl SuperSystem<'_> {
    pub fn create_fragmented_overlap(
        &self,
        skt: &SlaterKoster,
    ) -> (
        HashMap<(usize, usize), Array2<f64>>,
        HashMap<(usize, usize), bool>,
    ) {
        // initialize the hashmap
        let mut s_hash: HashMap<(usize, usize), Array2<f64>> = HashMap::new();
        let mut bool_hash: HashMap<(usize, usize), bool> = HashMap::new();

        for (idx_i, m_i) in self.monomers.iter().enumerate() {
            // get the monomer atoms
            let m_i_atoms = &self.atoms[m_i.slice.atom_as_range()];

            for (idx_j, m_j) in self.monomers.iter().enumerate() {
                // get the monomer atoms
                let m_j_atoms = &self.atoms[m_j.slice.atom_as_range()];

                // only do upper triangle
                if idx_i < idx_j {
                    // check if the distance between the monomers is lower than the threshold
                    let mut calc_s: bool = false;
                    'pair_loop: for atomi in m_i_atoms.iter() {
                        for atomj in m_j_atoms.iter() {
                            if (atomi - atomj).norm() < defaults::PROXIMITY_CUTOFF {
                                calc_s = true;
                                break 'pair_loop;
                            }
                        }
                    }
                    // calculate the overlap between the monomers
                    if calc_s {
                        bool_hash.insert((idx_i, idx_j), true);
                        let mut s: Array2<f64> = Array2::zeros((m_i.n_orbs, m_j.n_orbs));

                        let mut mu: usize = 0;
                        for atomi in m_i_atoms.iter() {
                            // iterate over orbitals on center i
                            for orbi in atomi.valorbs.iter() {
                                // iterate over atoms
                                let mut nu: usize = 0;
                                for atomj in m_j_atoms.iter() {
                                    // iterate over orbitals on center j
                                    for orbj in atomj.valorbs.iter() {
                                        if (atomi - atomj).norm() < defaults::PROXIMITY_CUTOFF {
                                            if atomi <= atomj {
                                                let (r, x, y, z): (f64, f64, f64, f64) =
                                                    directional_cosines(&atomi.xyz, &atomj.xyz);
                                                s[[mu, nu]] = slako_transformation(
                                                    r,
                                                    x,
                                                    y,
                                                    z,
                                                    &skt.get(atomi.kind, atomj.kind).s_spline,
                                                    orbi.l,
                                                    orbi.m,
                                                    orbj.l,
                                                    orbj.m,
                                                );
                                            } else {
                                                let (r, x, y, z): (f64, f64, f64, f64) =
                                                    directional_cosines(&atomj.xyz, &atomi.xyz);
                                                s[[mu, nu]] = slako_transformation(
                                                    r,
                                                    x,
                                                    y,
                                                    z,
                                                    &skt.get(atomj.kind, atomi.kind).s_spline,
                                                    orbj.l,
                                                    orbj.m,
                                                    orbi.l,
                                                    orbi.m,
                                                );
                                            }
                                        }
                                        nu += 1;
                                    }
                                }
                                mu += 1;
                            }
                        }
                        s_hash.insert((idx_i, idx_j), s);
                    } else {
                        bool_hash.insert((idx_i, idx_j), false);
                    }
                } else if idx_i == idx_j {
                    bool_hash.insert((idx_i, idx_j), true);
                    let mut s: Array2<f64> = Array2::zeros((m_i.n_orbs, m_j.n_orbs));

                    let mut mu: usize = 0;
                    for (i, atomi) in m_i_atoms.iter().enumerate() {
                        // iterate over orbitals on center i
                        for orbi in atomi.valorbs.iter() {
                            // iterate over atoms
                            let mut nu: usize = 0;
                            for (j, atomj) in m_j_atoms.iter().enumerate() {
                                // iterate over orbitals on center j
                                for orbj in atomj.valorbs.iter() {
                                    if (atomi - atomj).norm() < defaults::PROXIMITY_CUTOFF {
                                        if mu < nu {
                                            if atomi <= atomj {
                                                if i != j {
                                                    let (r, x, y, z): (f64, f64, f64, f64) =
                                                        directional_cosines(&atomi.xyz, &atomj.xyz);
                                                    s[[mu, nu]] = slako_transformation(
                                                        r,
                                                        x,
                                                        y,
                                                        z,
                                                        &skt.get(atomi.kind, atomj.kind).s_spline,
                                                        orbi.l,
                                                        orbi.m,
                                                        orbj.l,
                                                        orbj.m,
                                                    );
                                                }
                                            } else {
                                                let (r, x, y, z): (f64, f64, f64, f64) =
                                                    directional_cosines(&atomj.xyz, &atomi.xyz);
                                                s[[mu, nu]] = slako_transformation(
                                                    r,
                                                    x,
                                                    y,
                                                    z,
                                                    &skt.get(atomj.kind, atomi.kind).s_spline,
                                                    orbj.l,
                                                    orbj.m,
                                                    orbi.l,
                                                    orbi.m,
                                                );
                                            }
                                        } else if mu == nu {
                                            assert_eq!(atomi.number, atomj.number);
                                            s[[mu, nu]] = 1.0;
                                        } else {
                                            s[[mu, nu]] = s[[nu, mu]];
                                        }
                                    }
                                    nu += 1;
                                }
                            }
                            mu += 1;
                        }
                    }
                    s_hash.insert((idx_i, idx_j), s);
                }
            }
        }
        (s_hash, bool_hash)
    }
}
