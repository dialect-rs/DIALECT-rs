use crate::excited_states::ProductCache;
use crate::properties::property::Property;
use crate::properties::Properties;
use crate::scc::mixer::{AndersonAccel, BroydenMixer};
use ndarray::prelude::*;

impl Properties {
    // Takes the scc mixer
    pub fn take_mixer(&mut self) -> Result<BroydenMixer, Property> {
        match self.take("mixer") {
            Some(value) => value.into_mixer(),
            _ => Err(Property::default()),
        }
    }

    // Takes the scc mixer
    pub fn take_accel(&mut self) -> Result<AndersonAccel, Property> {
        match self.take("accel") {
            Some(value) => value.into_accel(),
            _ => Err(Property::default()),
        }
    }

    /// Takes the atomic numbers
    pub fn take_atomic_numbers(&mut self) -> Result<Vec<u8>, Property> {
        match self.take("atomic_numbers") {
            Some(value) => value.into_vec_u8(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the reference density matrix
    pub fn take_p_ref(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("ref_density_matrix") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the H0 matrix in AO basis.
    pub fn take_h0(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("H0") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the overlap matrix in AO basis.
    pub fn take_s(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("S") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the S^-1/2 in AO basis
    pub fn take_x(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("X") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns a reference to the gradient of the H0 matrix in AO basis.
    pub fn take_grad_h0(&mut self) -> Result<Array3<f64>, Property> {
        match self.take("gradH0") {
            Some(value) => value.into_array3(),
            _ => Err(Property::default()),
        }
    }

    /// Returns a reference to the gradient of the overlap matrix in AO basis.
    pub fn take_grad_s(&mut self) -> Result<Array3<f64>, Property> {
        match self.take("gradS") {
            Some(value) => value.into_array3(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the charge differences per atom.
    pub fn take_dq(&mut self) -> Result<Array1<f64>, Property> {
        match self.take("dq") {
            Some(value) => value.into_array1(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the Mulliken charge per orbital.
    pub fn take_q_ao(&mut self) -> Result<Array1<f64>, Property> {
        match self.take("q_ao") {
            Some(value) => value.into_array1(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the charge differences per atom.
    pub fn take_dq_alpha(&mut self) -> Result<Array1<f64>, Property> {
        match self.take("dq_alpha") {
            Some(value) => value.into_array1(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the charge differences per atom.
    pub fn take_dq_beta(&mut self) -> Result<Array1<f64>, Property> {
        match self.take("dq_beta") {
            Some(value) => value.into_array1(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the density matrix in AO basis.
    pub fn take_p(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("P") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the density matrix of the alpha electrons in AO basis.
    pub fn take_p_alpha(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("P_alpha") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the density matrix of the beta electrons in AO basis.
    pub fn take_p_beta(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("P_beta") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the gamma matrix in atomic basis.
    pub fn take_gamma(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("gamma_atom_wise") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the `ProductCache` for the Davidson diagonalization.
    pub fn take_cache(&mut self) -> Result<ProductCache, Property> {
        match self.take("cache") {
            Some(value) => value.into_cache(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the gamma matrix in AO basis.
    pub fn take_gamma_ao(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("gamma_ao_wise") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the long-range corrected gamma matrix in atomic basis.
    pub fn take_gamma_lr(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("gamma_lr_atom_wise") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the long-range corrected gamma matrix in AO basis.
    pub fn take_gamma_lr_ao(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("gamma_lr_ao_wise") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the gradient of the gamma matrix in atomic basis.
    pub fn take_grad_gamma(&mut self) -> Result<Array3<f64>, Property> {
        match self.take("gamma_atom_wise_gradient") {
            Some(value) => value.into_array3(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the gradient of the gamma matrix in AO basis.
    pub fn take_grad_gamma_ao(&mut self) -> Result<Array3<f64>, Property> {
        match self.take("gamma_ao_wise_gradient") {
            Some(value) => value.into_array3(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the gradient of the long-range corrected gamma matrix in atomic basis.
    pub fn take_grad_gamma_lr(&mut self) -> Result<Array3<f64>, Property> {
        match self.take("gamma_lr_atom_wise_gradient") {
            Some(value) => value.into_array3(),
            _ => Err(Property::default()),
        }
    }

    /// Returns a reference to the gradient of the long-range corrected gamma matrix in AO basis.
    pub fn take_grad_gamma_lr_ao(&mut self) -> Result<Array3<f64>, Property> {
        match self.take("gamma_lr_ao_wise_gradient") {
            Some(value) => value.into_array3(),
            _ => Err(Property::default()),
        }
    }

    // pub fn take_old_supersystem(&mut self) -> Result<SuperSystem,Property>{
    //     match self.take("old_supersystem"){
    //         Some(value) =>value.into_super_system(),
    //         _ =>Err(Property::default()),
    //     }
    // }
}
