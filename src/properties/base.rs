use crate::excited_states::ProductCache;
use crate::fmo::PairType;
use crate::properties::property::Property;
use crate::scc::mixer::BroydenMixer;
use hashbrown::HashMap;
use ndarray::prelude::*;
use ndarray::Slice;
use std::ops::AddAssign;
