use crate::initialization::parameters::{
    RepulsivePotential, RepulsivePotentialTable, SkfHandler, SlaterKoster, SlaterKosterTable,
};
use crate::initialization::{get_unique_atoms_mio, Atom};
use crate::io::{frame_to_atoms, frame_to_coordinates};
use crate::param::Element;
use crate::Configuration;
use chemfiles::Frame;
use hashbrown::HashMap;
use itertools::Itertools;

pub fn generate_parameters(
    frame: Frame,
    config: Configuration,
) -> (SlaterKoster, RepulsivePotential, Vec<Atom>, Vec<Atom>) {
    // create mutable Vectors
    let mut unique_atoms: Vec<Atom> = Vec::new();
    let mut atoms: Vec<Atom> = Vec::new();
    let mut skf_handlers: Vec<SkfHandler> = Vec::new();

    if config.slater_koster.use_external_skf == true {
        // get the unique [Atom]s and the HashMap with the mapping from the numbers to the [Atom]s
        // if use_mio is true, create a vector of homonuclear SkfHandlers and a vector
        // of heteronuclear SkfHandlers

        let mut num_to_atom: HashMap<u8, Atom> = HashMap::new();
        let (numbers, coords) = frame_to_coordinates(frame);

        let tmp: (Vec<Atom>, HashMap<u8, Atom>, Vec<SkfHandler>) =
            get_unique_atoms_mio(&numbers, &config);
        unique_atoms = tmp.0;
        num_to_atom = tmp.1;
        skf_handlers = tmp.2;

        // get all the Atom's from the HashMap
        numbers
            .iter()
            .for_each(|num| atoms.push((*num_to_atom.get(num).unwrap()).clone()));
        // set the positions for each atom
        coords.outer_iter().enumerate().for_each(|(idx, position)| {
            atoms[idx].position_from_slice(position.as_slice().unwrap())
        });
    } else {
        // get the unique [Atom]s and the HashMap with the mapping from the numbers to the [Atom]s
        let tmp: (Vec<Atom>, Vec<Atom>) = frame_to_atoms(frame);
        atoms = tmp.0;
        unique_atoms = tmp.1;
    }

    // and initialize the SlaterKoster and RepulsivePotential Tables
    let mut slako: SlaterKoster = SlaterKoster::new();
    let mut vrep: RepulsivePotential = RepulsivePotential::new();

    if config.slater_koster.use_external_skf == true {
        for handler in skf_handlers.iter() {
            let repot_table: RepulsivePotentialTable = RepulsivePotentialTable::from(handler);
            let slako_table_ab: SlaterKosterTable = SlaterKosterTable::from((handler, None, "ab"));
            let slako_handler_ba: SkfHandler = SkfHandler::new(
                handler.element_b,
                handler.element_a,
                config.slater_koster.skf_directory.clone(),
            );
            let slako_table: SlaterKosterTable =
                SlaterKosterTable::from((&slako_handler_ba, Some(slako_table_ab), "ba"));

            // insert the tables into the hashmaps
            slako
                .map
                .insert((handler.element_a, handler.element_b), slako_table);
            vrep.map
                .insert((handler.element_a, handler.element_b), repot_table);
        }
    } else {
        let element_iter = unique_atoms.iter().map(|atom| Element::from(atom.number));
        for (kind1, kind2) in element_iter.clone().cartesian_product(element_iter) {
            slako.add(kind1, kind2);
            vrep.add(kind1, kind2);
        }
    }

    return (slako, vrep, atoms, unique_atoms);
}
