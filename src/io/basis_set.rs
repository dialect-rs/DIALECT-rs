use crate::param::Element;
use crate::utils::get_path_prefix;
use serde::{Deserialize, Serialize};
use serde_json::{from_str, Result, Value};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug)]
struct InputData {
    name: String,
    description: String,
    elements: HashMap<usize, InputElement>,
}

#[derive(Serialize, Deserialize, Debug)]
struct InputElement {
    electron_shells: Vec<InputShell>,
}

#[derive(Serialize, Deserialize, Debug)]
struct InputShell {
    angular_momentum: Vec<usize>,
    exponents: Vec<String>,
    coefficients: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct BasisSet {
    name: String,
    description: String,
    basis_functions: HashMap<Element, Vec<BasisFunction>>,
}

impl From<InputData> for BasisSet {
    fn from(data: InputData) -> Self {
        // The HashMap is initialized.
        let mut bfs: HashMap<Element, Vec<BasisFunction>> = HashMap::new();
        for (element, shells) in data.elements.iter() {
            // The corresponding Element is created.
            let el: Element = Element::from(*element as u8);
            // The BasisFunctions are created.
            let mut functions: Vec<BasisFunction> = Vec::new();
            // Iteration over all shells.
            //for shell in shells.electron_shells.iter() {}
            let shell: &InputShell = &shells.electron_shells[shells.electron_shells.len() - 1];
            // The exponents are the same for all angular momenta.
            let exponents: Vec<f64> = shell
                .exponents
                .iter()
                .map(|x| x.parse::<f64>().unwrap())
                .collect();
            // Iteration over all angular momenta.
            for (l, c) in shell.angular_momentum.iter().zip(shell.coefficients.iter()) {
                // The coefficients are converted to floats.
                let coefficients: Vec<f64> = c.iter().map(|x| x.parse::<f64>().unwrap()).collect();
                // Add the new BasisFunction
                functions.push(BasisFunction {
                    l: AngularMomentum::from(*l),
                    exponents: exponents.clone(),
                    coefficients,
                });
            }

            bfs.insert(el, functions);
        }
        Self {
            name: data.name,
            description: data.description,
            basis_functions: bfs,
        }
    }
}

impl Default for BasisSet {
    /// Returns the STO-3G basis set for all Elements up to Xenon (#54).
    fn default() -> Self {
        let path_prefix: String = get_path_prefix();
        let filename: String = format!("{}/src/param/basis_sets/sto-3g.json", path_prefix);
        let path: &Path = Path::new(&filename);
        let data: String =
            fs::read_to_string(path).expect(&*format! {"Unable to read file {}", &filename});
        let data: InputData = from_str(&data).expect("JSON file was not well-formatted");
        Self::from(data)
    }
}

impl BasisSet {
    pub fn repr_basis_set(&self, element: Element) -> String {
        let functions: &[BasisFunction] = self.basis_functions.get(&element).unwrap();
        let mut txt = "".to_owned();
        for function in functions.iter() {
            txt += &format!(
                " {}    {} {:1.2}\n",
                function.l,
                function.exponents.len(),
                1.00
            );
            for (e, c) in function.exponents.iter().zip(function.coefficients.iter()) {
                txt += &format!("{:18.14e} {:18.14e}\n", e, c);
            }
        }
        txt
    }
}

#[derive(Debug, Clone)]
pub struct BasisFunction {
    pub l: AngularMomentum,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
}

#[derive(Debug, Copy, Clone)]
pub enum AngularMomentum {
    S = 0,
    P = 1,
    D = 2,
    F = 3,
    G = 4,
    H = 5,
    I = 6,
}

impl fmt::Display for AngularMomentum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let symbol: &str = match self {
            AngularMomentum::S => "s",
            AngularMomentum::P => "p",
            AngularMomentum::D => "d",
            AngularMomentum::F => "f",
            AngularMomentum::G => "g",
            AngularMomentum::H => "h",
            AngularMomentum::I => "i",
        };
        write!(f, "{}", symbol)
    }
}

impl From<usize> for AngularMomentum {
    fn from(value: usize) -> Self {
        match value {
            0 => Self::S,
            1 => Self::P,
            2 => Self::D,
            3 => Self::F,
            4 => Self::G,
            5 => Self::H,
            6 => Self::I,
            a => {
                panic!("Angular momentum:{} is not implemented", a)
            }
        }
    }
}
