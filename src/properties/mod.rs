#![allow(dead_code)]

use hashbrown::HashMap;
pub use property::Property;

mod getter;
pub mod property;
mod setter;
mod taker;

#[derive(Debug, Clone)]
pub struct Properties {
    map: HashMap<&'static str, Property>,
}

impl Properties {
    pub fn new() -> Self {
        Properties {
            map: HashMap::new(),
        }
    }

    pub fn reset_all(&mut self) {
        self.map.clear();
    }

    /// Removes all multi dimensional arrays from the HashMap to free the memory
    pub fn reset(&mut self) {
        let multi_dim_data = [
            "H0",
            "S",
            "X",
            "dq",
            "dq_ao",
            "dq_alpha",
            "dq_beta",
            "delta_dq",
            "diff_density_matrix",
            "mixer",
            "accel",
            //"P",
            "gradH0",
            "gradS",
            "gamma_atom_wise",
            "gamma_ao_wise",
            "gamma_lr_atom_wise",
            "gamma_lr_ao_wise",
            "gamma_atom_wise_gradient",
            "gamma_ao_wise_gradient",
            "gamma_lr_atom_wise_gradient",
            "gamma_lr_ao_wise_gradient",
            "gamma_third_order",
            "q_ov",
            "q_oo",
            "q_vv",
            "xmy",
            "xpy",
            "omega",
            "homo",
            "lumo",
            "cache",
            "ci_eigenvalues",
            "ci_coefficients",
            "q_trans",
            "tr_dipoles",
            "oscillator_strengths",
            "lcmo_fock",
            "coupling_signs",
        ];
        for data_name in multi_dim_data.iter() {
            self.map.remove(*data_name);
        }
    }

    pub fn reset_reduced(&mut self) {
        let mut new_map: HashMap<&'static str, Property> = HashMap::new();
        if let Some(dq) = self.dq() {
            new_map.insert("dq", Property::from(dq));
        }
        if let Some(dq_ao) = self.dq_ao() {
            new_map.insert("dq_ao", Property::from(dq_ao));
        }
        if let Some(n_occ) = self.n_occ() {
            new_map.insert("n_occ", Property::from(n_occ));
        }
        if let Some(n_virt) = self.n_virt() {
            new_map.insert("n_virt", Property::from(n_virt));
        }
        if let Some(occ_indices) = self.occ_indices() {
            new_map.insert("occ_indices", Property::from(occ_indices.to_owned()));
        }
        if let Some(virt_indices) = self.virt_indices() {
            new_map.insert("virt_indices", Property::from(virt_indices.to_owned()));
        }
        if let Some(p) = self.p() {
            new_map.insert("P", Property::from(p));
        }
        if let Some(old_supersystem) = self.old_supersystem() {
            new_map.insert(
                "old_supersystem",
                Property::from(old_supersystem.to_owned()),
            );
        }
        if let Some(ref_supersystem) = self.ref_supersystem() {
            new_map.insert(
                "ref_supersystem",
                Property::from(ref_supersystem.to_owned()),
            );
        }
        if let Some(old_system) = self.old_system() {
            new_map.insert("old_system", Property::from(old_system.to_owned()));
        }
        if let Some(pair_indices) = self.get("pair_indices") {
            new_map.insert(
                "pair_indices",
                Property::from(pair_indices.as_pair_index_map().unwrap().to_owned()),
            );
        }
        if let Some(esd_pair_indices) = self.get("esd_pair_indices") {
            new_map.insert(
                "esd_pair_indices",
                Property::from(esd_pair_indices.as_pair_index_map().unwrap().to_owned()),
            );
        }
        if let Some(pair_types) = self.get("pair_types") {
            new_map.insert(
                "pair_types",
                Property::from(pair_types.as_pair_map().unwrap().to_owned()),
            );
        }
        self.map = new_map;
    }

    pub fn reset_gradient(&mut self) {
        let multi_dim_data = [
            "gradH0",
            "gradS",
            "gamma_ao_wise",
            "gamma_lr_ao_wise",
            "gamma_atom_wise_gradient",
            "gamma_ao_wise_gradient",
            "gamma_lr_atom_wise_gradient",
            "gamma_lr_ao_wise_gradient",
            "cache",
            "omega",
            "ci_eigenvalues",
            "ci_coefficients",
        ];
        for data_name in multi_dim_data.iter() {
            self.map.remove(*data_name);
        }
    }

    pub fn reset_supersystem(&mut self) {
        self.map.remove("old_supersystem");
    }

    pub fn get(&self, name: &'static str) -> Option<&Property> {
        self.map.get(name)
    }

    pub fn get_mut(&mut self, name: &'static str) -> Option<&mut Property> {
        self.map.get_mut(name)
    }

    /// Returns the Property without a reference and removes it from the dict
    pub fn take(&mut self, name: &'static str) -> Option<Property> {
        self.map.remove(name)
    }

    pub fn set(&mut self, name: &'static str, value: Property) {
        self.map.insert(name, value);
    }

    pub fn contains_key(&self, name: &'static str) -> bool {
        self.map.contains_key(name)
    }
}
