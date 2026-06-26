//! Regression tests for the DFTB foundations (now in dialect-dftb-core)
//! against reference data; they need the main crate's test molecules.

#[cfg(test)]
mod h0_and_s_tests {
    use crate::scc::h0_and_s::*;
    use ndarray::prelude::*;
    use crate::initialization::System;
    use crate::properties::Properties;
    use crate::utils::tests::{get_molecule, AVAILAIBLE_MOLECULES};

    pub const EPSILON: f64 = 1e-15;

    fn test_h0_and_s(molecule_and_properties: (&str, System, Properties)) {
        let name = molecule_and_properties.0;
        let molecule = molecule_and_properties.1;
        let props = molecule_and_properties.2;
        let (s, h0): (Array2<f64>, Array2<f64>) =
            h0_and_s(molecule.n_orbs, &molecule.atoms, &molecule.slako);
        let s_ref: Array2<f64> = props.get("S").unwrap().as_array2().unwrap().to_owned();
        let h0_ref: Array2<f64> = props.get("H0").unwrap().as_array2().unwrap().to_owned();

        assert!(
            s_ref.abs_diff_eq(&s, EPSILON),
            "Molecule: {}, S (ref): {}  S: {}",
            name,
            s_ref,
            s
        );

        assert!(
            h0_ref.abs_diff_eq(&h0, EPSILON),
            "Molecule: {}, H0 (ref): {}  H0: {}",
            name,
            h0_ref,
            h0
        );
    }

    #[test]
    fn get_h0_and_s() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_h0_and_s(get_molecule(molecule));
        }
    }
}

#[cfg(test)]
mod gamma_approximation_tests {
    use crate::scc::gamma_approximation::*;
    use ndarray::prelude::*;
    use crate::initialization::System;
    use crate::properties::Properties;
    use crate::utils::tests::{get_molecule, AVAILAIBLE_MOLECULES};

    pub const EPSILON: f64 = 1e-12;

    fn test_gamma_atomwise(molecule_and_properties: (&str, System, Properties)) {
        let name = molecule_and_properties.0;
        let molecule = molecule_and_properties.1;
        let props = molecule_and_properties.2;
        let gamma: Array2<f64> =
            gamma_atomwise(&molecule.gammafunction, &molecule.atoms, molecule.n_atoms);

        let gamma_ref: Array2<f64> = props
            .get("gamma_atomwise")
            .unwrap()
            .as_array2()
            .unwrap()
            .to_owned();

        assert!(
            gamma_ref.abs_diff_eq(&gamma, EPSILON),
            "Molecule: {}, Gamma (ref): {}  Gamma: {}",
            name,
            gamma_ref,
            gamma
        );
    }

    fn test_gamma_atomwise_lc(molecule_and_properties: (&str, System, Properties)) {
        let name = molecule_and_properties.0;
        let molecule = molecule_and_properties.1;
        let props = molecule_and_properties.2;
        let gamma: Array2<f64> = gamma_atomwise(
            &molecule.gammafunction_lc.unwrap(),
            &molecule.atoms,
            molecule.n_atoms,
        );
        let gamma_ref: Array2<f64> = props
            .get("gamma_atomwise_lc")
            .unwrap()
            .as_array2()
            .unwrap()
            .to_owned();

        assert!(
            gamma_ref.abs_diff_eq(&gamma, EPSILON),
            "Molecule: {}, Gamma-LC (ref): {}  Gamma-LC: {}",
            name,
            gamma_ref,
            gamma
        );
    }

    fn test_gamma_ao_wise(molecule_and_properties: (&str, System, Properties)) {
        let name = molecule_and_properties.0;
        let molecule = molecule_and_properties.1;
        let props = molecule_and_properties.2;
        let (g0, g0_ao): (Array2<f64>, Array2<f64>) = gamma_ao_wise(
            &molecule.gammafunction,
            &molecule.atoms,
            molecule.n_atoms,
            molecule.n_orbs,
        );
        let g0_ref: Array2<f64> = props
            .get("gamma_atomwise")
            .unwrap()
            .as_array2()
            .unwrap()
            .to_owned();
        let g0_ao_ref: Array2<f64> = props
            .get("gamma_ao_wise")
            .unwrap()
            .as_array2()
            .unwrap()
            .to_owned();
        assert!(
            g0_ref.abs_diff_eq(&g0, EPSILON),
            "Molecule: {}, Gamma-LC (ref): {}  Gamma-LC: {}",
            name,
            g0_ref,
            g0
        );
        assert!(
            g0_ao_ref.abs_diff_eq(&g0_ao, EPSILON),
            "Molecule: {}, Gamma-LC (ao basis) (ref): {}  Gamma-LC (ao basis): {}",
            name,
            g0_ao_ref,
            g0_ao
        );
    }

    fn test_gamma_ao_wise_lc(molecule_and_properties: (&str, System, Properties)) {
        let name = molecule_and_properties.0;
        let molecule = molecule_and_properties.1;
        let props = molecule_and_properties.2;
        let (g0, g0_ao): (Array2<f64>, Array2<f64>) = gamma_ao_wise(
            &molecule.gammafunction_lc.unwrap(),
            &molecule.atoms,
            molecule.n_atoms,
            molecule.n_orbs,
        );
        let g0_ref: Array2<f64> = props
            .get("gamma_atomwise_lc")
            .unwrap()
            .as_array2()
            .unwrap()
            .to_owned();
        let g0_ao_ref: Array2<f64> = props
            .get("gamma_ao_wise_lc")
            .unwrap()
            .as_array2()
            .unwrap()
            .to_owned();
        assert!(
            g0_ref.abs_diff_eq(&g0, EPSILON),
            "Molecule: {}, Gamma-LC (ref): {}  Gamma-LC: {}",
            name,
            g0_ref,
            g0
        );
        assert!(
            g0_ao_ref.abs_diff_eq(&g0_ao, EPSILON),
            "Molecule: {}, Gamma-LC (ao basis) (ref): {}  Gamma-LC (ao basis): {}",
            name,
            g0_ao_ref,
            g0_ao
        );
    }

    #[test]
    fn get_gamma_atomwise() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_gamma_atomwise(get_molecule(molecule));
        }
    }

    #[test]
    fn get_gamma_atomwise_lc() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_gamma_atomwise_lc(get_molecule(molecule));
        }
    }

    #[test]
    fn get_gamma_ao_wise() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_gamma_ao_wise(get_molecule(molecule));
        }
    }

    #[test]
    fn get_gamma_ao_wise_lc() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_gamma_ao_wise_lc(get_molecule(molecule));
        }
    }
}
