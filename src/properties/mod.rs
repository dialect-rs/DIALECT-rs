use hashbrown::HashMap;
pub use property::Property;

pub mod base;
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

    /// Removes all multi dimensional arrays from the HashMap to free the memory
    pub fn reset(&mut self) {
        let multi_dim_data = [
            "H0",
            "S",
            "X",
            "dq",
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
