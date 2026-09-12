use itertools::Itertools;
use numpy::ndarray::Array1;
use numpy::ndarray::ArrayView2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions;
use pyo3::prelude::*;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, VecDeque};

use crate::structure::bonds::{Bond, BondList, BondType};
use crate::util::check_signals_periodically;

/// Bond orders higher than a triple bond are not assigned.
const MAX_INFERRED_BOND_ORDER: u8 = 3;

/// A possible valence of an atom and the charge that would be associated to it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Valence {
    /// The sum of bond orders of the atom.
    valence: u32,
    /// The formal charge the atom has, if this valence is reached.
    charge: i32,
}

impl Valence {
    const fn new(valence: u32, charge: i32) -> Self {
        Valence { valence, charge }
    }

    /// The number of bond orders the atom can still gain, if it has the given number
    /// of bond partners.
    /// Zero, if the atom already has more bond partners than this valence allows.
    fn unsaturation(&self, degree: u32) -> u32 {
        self.valence.saturating_sub(degree)
    }
}

/// A combination of chosen valence states, one for each atom.
///
/// As the states are taken from a max-heap, the greatest state is visited first.
/// Hence the order prefers, in this precedence:
///
/// 1. the net formal charge closest to the requested total charge, as only states
///    with the requested charge can be accepted at all,
/// 2. the lowest sum of absolute formal charges, i.e. the fewest charged atoms,
/// 3. the highest total valence, i.e. the most multiple bonds,
/// 4. the selected options, merely to make the order deterministic.
#[derive(PartialEq, Eq)]
struct ValenceState {
    charge_deviation: i32,
    abs_charge: i32,
    net_charge: i32,
    valence_sum: u32,
    /// The atom whose valence option was incremented last.
    /// Only atoms after it may be incremented in the successor states, so that each
    /// combination of options is created exactly once.
    last_incremented: usize,
    /// For each atom the index of the selected valence option.
    /// As an atom has only a handful of options, `u8` is sufficient and keeps the
    /// state small, as it is cloned and hashed for each visited state.
    option_indices: Vec<u8>,
}

impl ValenceState {
    /// Create the valence state, in which the valence option at the given index is
    /// selected for each atom.
    fn from_selected_options(
        option_indices: Vec<u8>,
        last_incremented: usize,
        options: &[Vec<Valence>],
        total_charge: i32,
    ) -> Self {
        let mut valence_sum = 0;
        let mut net_charge = 0;
        let mut abs_charge = 0;
        for (atom_i, &option_i) in option_indices.iter().enumerate() {
            let valence = options[atom_i][option_i as usize];
            valence_sum += valence.valence;
            net_charge += valence.charge;
            abs_charge += valence.charge.abs();
        }
        ValenceState {
            charge_deviation: (net_charge - total_charge).abs(),
            abs_charge,
            net_charge,
            valence_sum,
            last_incremented,
            option_indices,
        }
    }
}

impl Ord for ValenceState {
    fn cmp(&self, other: &Self) -> Ordering {
        (
            Reverse(self.charge_deviation),
            Reverse(self.abs_charge),
            self.valence_sum,
            &self.option_indices,
        )
            .cmp(&(
                Reverse(other.charge_deviation),
                Reverse(other.abs_charge),
                other.valence_sum,
                &other.option_indices,
            ))
    }
}

impl PartialOrd for ValenceState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Internal rust function for :func:`biotite.structure.connect.connect_via_residue_names`.
///
/// Parameters
/// ----------
/// res_names
///     The residue name of each atom.
/// atom_names
///     The atom name of each atom.
/// residue_starts
///     The start index of each residue (including exclusive stop index).
/// bond_dict
///     The bond dictionary mapping connected atom name pairs to their bond type if they form a
///     covalent bond.
///
/// Returns
/// -------
/// bonds
///     The bonds between the atoms.
#[pyfunction]
pub fn connect_via_residue_names<'py>(
    _py: Python<'py>,
    res_names: Vec<String>,
    atom_names: Vec<String>,
    residue_starts: PyReadonlyArray1<'py, i64>,
    bond_dict: HashMap<String, Vec<(String, String, BondType)>>,
) -> PyResult<BondList> {
    let mut bond_list = BondList::empty(atom_names.len());
    for (start_i, stop_i) in residue_starts.as_array().iter().tuple_windows() {
        let start_i = *start_i as usize;
        let stop_i = *stop_i as usize;
        let bonds_res = match bond_dict.get(&res_names[start_i]) {
            Some(bonds_res) => bonds_res,
            None => continue,
        };
        let atom_names_in_res = &atom_names[start_i..stop_i];
        for (atom_name1, atom_name2, bond_type) in bonds_res.iter() {
            let atom_indices_1 = find(atom_names_in_res, atom_name1);
            let atom_indices_2 = find(atom_names_in_res, atom_name2);
            // In rare cases the same atom name may appear multiple times
            // (e.g. in altlocs)
            // -> create all possible bond combinations
            for atom_index_i in atom_indices_1.iter() {
                for atom_index_j in atom_indices_2.iter() {
                    unsafe {
                        bond_list.add(Bond::new(
                            (start_i + atom_index_i) as isize,
                            (start_i + atom_index_j) as isize,
                            *bond_type,
                            atom_names.len(),
                        )?)?;
                    }
                }
            }
        }
    }
    bond_list.remove_redundant_bonds();
    Ok(bond_list)
}

/// Internal rust function for :func:`biotite.structure.connect.connect_via_distances`.
///
/// Parameters
/// ----------
/// coord
///     The coordinates of each atom (shape: Nx3, dtype: float32).
/// radii
///     The covalent radius of each atom.
///     NaN for atoms with unknown radius, which consequently never form bonds.
/// residue_starts
///     The start index of each residue (including exclusive stop index).
/// tolerance
///     The tolerance added to the sum of two covalent radii to obtain the maximum
///     bond distance.
/// default_bond_type
///     The bond type to assign to all created bonds.
///
/// Returns
/// -------
/// bonds
///     The bonds between the atoms.
#[pyfunction]
pub fn connect_via_distances<'py>(
    _py: Python<'py>,
    coord: PyReadonlyArray2<'py, f32>,
    radii: PyReadonlyArray1<'py, f32>,
    residue_starts: PyReadonlyArray1<'py, i64>,
    tolerance: f32,
    default_bond_type: BondType,
) -> PyResult<BondList> {
    let coord = coord.as_array();
    let radii = radii.as_array();
    let residue_starts = residue_starts.as_array();
    let n_atoms = radii.len();

    let mut bond_list = BondList::empty(n_atoms);

    for (start, stop) in residue_starts.iter().tuple_windows() {
        let start = *start as usize;
        let stop = *stop as usize;

        for atom_i in start..stop {
            for atom_j in start..atom_i {
                let max_distance = radii[atom_i] + radii[atom_j] + tolerance;
                // Compare squared distances to avoid expensive square roots
                // Both comparands are NaN for atoms with unknown radius or missing
                // coordinates, which makes the comparison false
                if squared_distance(&coord, atom_i, atom_j) <= max_distance * max_distance {
                    unsafe {
                        // atom_j < atom_i is guaranteed by the loop range
                        bond_list.add(Bond {
                            atom1: atom_j,
                            atom2: atom_i,
                            bond_type: default_bond_type,
                        })?;
                    }
                }
            }
        }
    }

    Ok(bond_list)
}

/// Internal rust function for :func:`biotite.structure.connect._connect_inter_residue`.
///
/// Parameters
/// ----------
/// atom_names
///     The atom name of each atom.
/// residue_starts
///     The start index of each residue (including exclusive stop index).
/// link_types
///     The link type of each residue. None if the residue is not found in the CCD.
/// is_disconnected
///     Whether each residue is disconnected from the next residue due to chain or
///     residue discontinuity.
///     Length is ``len(residue_starts) - 2``.
#[pyfunction]
pub fn connect_inter_residue<'py>(
    _py: Python<'py>,
    atom_names: Vec<String>,
    residue_starts: PyReadonlyArray1<'py, i64>,
    link_types: Vec<Option<String>>,
    is_disconnected: PyReadonlyArray1<'py, bool>,
) -> PyResult<BondList> {
    let residue_starts = residue_starts.as_array();
    let is_disconnected = is_disconnected.as_array();
    let n_residues = residue_starts.len() - 1;

    let mut bond_list = BondList::empty(atom_names.len());

    if n_residues < 2 {
        return Ok(bond_list);
    }
    for i in 0..n_residues - 1 {
        // Check if residues are disconnected
        if is_disconnected[i] {
            continue;
        }

        let curr_start = residue_starts[i] as usize;
        let curr_stop = residue_starts[i + 1] as usize;
        let next_start = residue_starts[i + 1] as usize;
        let next_stop = residue_starts[i + 2] as usize;

        // Get link type for both residues
        let (curr_link, next_link) = match (&link_types[i], &link_types[i + 1]) {
            (Some(curr), Some(next)) => (curr.as_str(), next.as_str()),
            _ => continue,
        };

        let (curr_connect_name, next_connect_name) =
            if is_peptide_link(curr_link) && is_peptide_link(next_link) {
                ("C", "N")
            } else if is_nucleic_link(curr_link) && is_nucleic_link(next_link) {
                ("O3'", "P")
            } else {
                // Create no bond if the connection types of consecutive
                // residues are not compatible
                continue;
            };

        // Find first connector atom in each residue
        let curr_connect_idx = atom_names[curr_start..curr_stop]
            .iter()
            .position(|name| name == curr_connect_name);
        let next_connect_idx = atom_names[next_start..next_stop]
            .iter()
            .position(|name| name == next_connect_name);

        if let (Some(curr_idx), Some(next_idx)) = (curr_connect_idx, next_connect_idx) {
            unsafe {
                // curr_start < next_start is guaranteed
                // by iterating consecutive residues
                bond_list.add(Bond {
                    atom1: curr_start + curr_idx,
                    atom2: next_start + next_idx,
                    bond_type: BondType::Single,
                })?;
            }
        }
    }

    Ok(bond_list)
}

/// Internal rust function for :func:`biotite.structure.connect.infer_bond_types`.
///
/// Implements the bond order assignment algorithm from Kim & Kim
/// (https://doi.org/10.1002/bkcs.10334):
/// *Valence states*, i.e. combinations of the allowed valences of all atoms, are
/// enumerated lazily via best-first search.
/// In deviation from the reference, which enumerates purely in decreasing order of
/// total valence, states are ordered by increasing total absolute formal charge first
/// and decreasing total valence second:
/// This finds the chemically preferred state much faster (the state space grows
/// exponentially with the number of multivalent atoms) and prefers resonance
/// structures with fewer formal charges.
/// For each valence state multiple bonds are greedily assigned between adjacent
/// unsaturated atoms, restarting from different atoms if necessary.
/// The first assignment that consumes all unsaturations and whose formal charges sum
/// up to the given total charge is returned.
/// If no such assignment exists (or `max_iterations` is exceeded), the assignment with
/// the maximum total bond order found so far is returned as fallback.
///
/// Parameters
/// ----------
/// elements
///     The element of each atom in upper case.
/// bonds
///     The connectivity between the atoms.
///     The bond types are ignored.
/// total_charge
///     The total formal charge of the atoms, used as constraint.
/// known_charges
///     The formal charge of each atom, if it is already known.
///     Constrains the valence state of each atom to the given charge.
/// max_iterations
///     The maximum number of bond order assignments to try.
///     If `None`, the search is only stopped, when all valence states are tried.
///
/// Returns
/// -------
/// bonds
///     The bonds with assigned bond orders.
/// charges
///     The formal charge of each atom, as implied by the assigned bond orders.
/// converged
///     Whether an assignment satisfying the unsaturation and charge conditions was
///     found.
#[pyfunction]
#[pyo3(signature = (elements, bonds, total_charge, known_charges=None, max_iterations=None))]
pub fn infer_bond_types<'py>(
    py: Python<'py>,
    elements: Vec<String>,
    bonds: &BondList,
    total_charge: i32,
    known_charges: Option<PyReadonlyArray1<'py, i64>>,
    max_iterations: Option<u64>,
) -> PyResult<(BondList, Py<PyAny>, bool)> {
    // No limit on the number of iterations means an exhaustive search
    let max_iterations = max_iterations.unwrap_or(u64::MAX);
    let n_atoms = elements.len();
    if bonds.get_atom_count() != n_atoms {
        return Err(exceptions::PyValueError::new_err(format!(
            "Bond list represents {} atoms, but {} elements are given",
            bonds.get_atom_count(),
            n_atoms
        )));
    }

    let connected: Vec<(usize, usize)> = bonds
        .get_bonds_ref()
        .iter()
        .map(|bond| (bond.atom1, bond.atom2))
        .collect();
    // Adjacent atoms of each atom, given as `(partner_atom, edge_index)`
    let mut adjacency: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_atoms];
    for (edge_i, &(atom1, atom2)) in connected.iter().enumerate() {
        adjacency[atom1].push((atom2, edge_i));
        adjacency[atom2].push((atom1, edge_i));
    }
    let degrees: Vec<u32> = adjacency.iter().map(|adj| adj.len() as u32).collect();

    // Every bond starts as a single bond
    let base_orders: Vec<u8> = vec![1; connected.len()];

    // Allowed valence states of each atom as `(valence, formal charge)`
    // Atoms with unknown elements, no bonds at all or more bonds than any allowed
    // valence are treated as saturated and uncharged
    let mut options: Vec<Vec<Valence>> = (0..n_atoms)
        .map(|i| {
            if degrees[i] == 0 {
                return vec![Valence::new(degrees[i], 0)];
            }
            match allowed_valences(&elements[i], degrees[i]) {
                Some(allowed) => {
                    let valid: Vec<Valence> = allowed
                        .into_iter()
                        .filter(|option| option.valence >= degrees[i])
                        .collect();
                    if valid.is_empty() {
                        vec![Valence::new(degrees[i], 0)]
                    } else {
                        valid
                    }
                }
                None => vec![Valence::new(degrees[i], 0)],
            }
        })
        .collect();

    // If the formal charges are already known, only the valence states with the
    // respective charge are allowed, which constrains the search substantially
    if let Some(known_charges) = known_charges {
        let known_charges = known_charges.as_array();
        if known_charges.len() != n_atoms {
            return Err(exceptions::PyValueError::new_err(format!(
                "{} charges given for {} atoms",
                known_charges.len(),
                n_atoms
            )));
        }
        for i in 0..n_atoms {
            let charge = known_charges[i] as i32;
            let retained: Vec<Valence> = options[i]
                .iter()
                .copied()
                .filter(|option| option.charge == charge)
                .collect();
            options[i] = if retained.is_empty() {
                // The valence model cannot express the given charge for this element
                // -> treat the atom as saturated, but still trust the given charge,
                // as it may stem from a bonding situation that is not covered
                vec![Valence::new(degrees[i], charge)]
            } else {
                retained
            };
        }
    }

    // Iteratively remove valence options whose unsaturation cannot be absorbed by the
    // bond partners, as this may reduce the state space substantially
    loop {
        let mut changed = false;
        for i in 0..n_atoms {
            if options[i].len() < 2 {
                continue;
            }
            // The maximum unsaturation the bond partners can absorb in total
            let capacity: u32 = adjacency[i]
                .iter()
                .map(|&(j, _)| {
                    options[j]
                        .iter()
                        .map(|option| option.unsaturation(degrees[j]))
                        .max()
                        .unwrap_or(0)
                        // A bond can be raised to a triple bond at most
                        .min(MAX_INFERRED_BOND_ORDER as u32 - 1)
                })
                .sum();
            let retained: Vec<Valence> = options[i]
                .iter()
                .copied()
                .filter(|option| option.unsaturation(degrees[i]) <= capacity)
                .collect();
            if retained.len() < options[i].len() {
                // If no option is feasible, keep the least valence
                // The atom is not turned into a saturated one here, as this would
                // silently hide that the molecule cannot be correctly assigned
                options[i] = if retained.is_empty() {
                    vec![*options[i].iter().min().unwrap()]
                } else {
                    retained
                };
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Enumerate the valence states lazily in decreasing order of total valence
    // to avoid materializing the potentially exponential state space
    let mut heap: BinaryHeap<ValenceState> = BinaryHeap::new();
    let initial_indices = vec![0u8; n_atoms];
    heap.push(ValenceState::from_selected_options(
        initial_indices,
        0,
        &options,
        total_charge,
    ));

    let mut iteration: u64 = 0;
    let mut n_visited_states: usize = 0;
    let mut converged = false;
    let mut best_orders: Vec<u8> = base_orders.clone();
    let mut best_charges: Vec<i32> = vec![0; n_atoms];
    let mut best_order_sum: u64 = 0;

    'search: while let Some(state) = heap.pop() {
        n_visited_states += 1;
        check_signals_periodically(py, n_visited_states)?;

        // Enqueue all successor states, i.e. states where a single atom has the next
        // lower allowed valence
        // Only atoms after the last incremented one are considered, as otherwise the
        // same combination of options would be reached via multiple paths
        for i in state.last_incremented..n_atoms {
            if (state.option_indices[i] as usize) < options[i].len() - 1 {
                let mut successor = state.option_indices.clone();
                successor[i] += 1;
                heap.push(ValenceState::from_selected_options(
                    successor,
                    i,
                    &options,
                    total_charge,
                ));
            }
        }

        // Only valence states with the requested total charge can be accepted
        // -> skip the expensive assignment, as soon as a fallback is available
        if state.net_charge != total_charge && best_order_sum > 0 {
            iteration += 1;
            if iteration >= max_iterations {
                break 'search;
            }
            continue;
        }

        // Determine the valence state and unsaturation of each atom
        let charges: Vec<i32> = (0..n_atoms)
            .map(|i| options[i][state.option_indices[i] as usize].charge)
            .collect();
        let initial_unsaturation: Vec<i32> = (0..n_atoms)
            .map(|i| options[i][state.option_indices[i] as usize].unsaturation(degrees[i]) as i32)
            .collect();

        let unsaturated: Vec<usize> = (0..n_atoms)
            .filter(|&i| initial_unsaturation[i] > 0)
            .collect();

        // Quick rejection: If any unsaturated atom has only saturated neighbors,
        // no assignment in this valence state can consume all unsaturations
        if unsaturated.iter().any(|&i| {
            adjacency[i]
                .iter()
                .all(|&(j, _)| initial_unsaturation[j] == 0)
        }) {
            iteration += 1;
            if iteration >= max_iterations {
                break 'search;
            }
            continue;
        }

        // Without unsaturated atoms all bonds simply remain single bonds
        if unsaturated.is_empty() {
            if state.net_charge == total_charge {
                best_orders = base_orders.clone();
                best_charges = charges;
                converged = true;
                break 'search;
            }
            // As all bonds are single bonds, the sum of bond orders is minimal
            // -> this assignment is only kept, if no other one was found yet
            if best_order_sum == 0 {
                best_order_sum = base_orders.len() as u64;
                best_charges = charges;
            }
            iteration += 1;
            if iteration >= max_iterations {
                break 'search;
            }
            continue;
        }

        // The result of the greedy assignment depends on the starting atom
        // -> retry with different starting atoms until the conditions are met
        for &start in &unsaturated {
            iteration += 1;
            let (orders, remaining_unsaturation) =
                assign_bond_orders(&adjacency, &base_orders, &initial_unsaturation, start);
            // If no unsaturation is left, each atom reached the valence of its
            // state, so the formal charges of the state apply
            if remaining_unsaturation == 0 && state.net_charge == total_charge {
                best_orders = orders;
                best_charges = charges;
                converged = true;
                break 'search;
            }
            let total: u64 = orders.iter().map(|&order| order as u64).sum();
            if total > best_order_sum {
                best_order_sum = total;
                best_orders = orders;
                best_charges = charges.clone();
            }
            if iteration >= max_iterations {
                break 'search;
            }
        }
    }

    let mut bond_list = BondList::empty(n_atoms);
    for (i, &(atom1, atom2)) in connected.iter().enumerate() {
        let bond_type = match best_orders[i] {
            1 => BondType::Single,
            2 => BondType::Double,
            3 => BondType::Triple,
            _ => unreachable!("bond orders are capped at `MAX_BOND_ORDER`"),
        };
        unsafe {
            // The input bond list guarantees `atom1 < atom2` and unique bonds
            bond_list.add(Bond {
                atom1,
                atom2,
                bond_type,
            })?;
        }
    }

    let charges: Vec<i64> = best_charges.iter().map(|&charge| charge as i64).collect();

    Ok((
        bond_list,
        Array1::from_vec(charges)
            .into_pyarray(py)
            .into_any()
            .unbind(),
        converged,
    ))
}

/// Allowed valence states of the supported elements as `(valence, formal charge)`
/// (Table 1 in Kim & Kim, 2015, extended by further organic elements).
/// `None` for unsupported elements.
///
/// In addition to the neutral valence, the valences of the isoelectronic anion and
/// cation are allowed, which is required to represent e.g. azides or isocyanides.
/// The neutral state comes first, as it is the starting point of the search.
///
/// In deviation from the reference, the allowed valences of sulfur and selenium are
/// restricted based on the number of bond partners:
/// The formal charge is zero for every neutral sulfur valence, so valence states are
/// ranked by their total valence, i.e. by their unsaturation, alone.
/// Hence, without this restriction the hypervalent state would be chosen for e.g.
/// aromatic sulfur, whose ring neighbors can accept the additional bond orders.
fn allowed_valences(element: &str, degree: u32) -> Option<Vec<Valence>> {
    match element {
        "H" | "F" | "CL" | "BR" | "I" => Some(vec![Valence::new(1, 0)]),
        "B" => Some(vec![Valence::new(3, 0), Valence::new(4, -1)]),
        "C" | "SI" => Some(vec![
            Valence::new(4, 0),
            Valence::new(3, -1),
            Valence::new(3, 1),
        ]),
        "N" => Some(vec![
            Valence::new(3, 0),
            Valence::new(4, 1),
            Valence::new(2, -1),
        ]),
        "O" => Some(vec![
            Valence::new(2, 0),
            Valence::new(1, -1),
            Valence::new(3, 1),
        ]),
        "P" => Some(vec![
            Valence::new(3, 0),
            Valence::new(5, 0),
            Valence::new(4, 1),
            Valence::new(2, -1),
        ]),
        "S" | "SE" => match degree {
            1 => Some(vec![Valence::new(2, 0), Valence::new(1, -1)]),
            2 => Some(vec![Valence::new(2, 0), Valence::new(3, 1)]),
            3 => Some(vec![Valence::new(4, 0), Valence::new(3, 1)]),
            _ => Some(vec![Valence::new(6, 0), Valence::new(4, 0)]),
        },
        _ => None,
    }
}

/// Greedily assign multiple bonds between adjacent unsaturated atoms
/// (Figure 1 in Kim & Kim, 2015):
/// In each step the bond order between the unsaturated atom with the fewest pairable
/// partners and its pairable partner with the fewest pairable partners is increased,
/// until no two adjacent unsaturated atoms are left.
/// `start` forces the atom initiating the first pairing.
/// Returns the assigned bond orders and the total remaining unsaturation.
fn assign_bond_orders(
    adjacency: &[Vec<(usize, usize)>],
    base_orders: &[u8],
    initial_unsaturation: &[i32],
    start: usize,
) -> (Vec<u8>, i32) {
    let mut orders = base_orders.to_vec();
    let mut unsaturation = initial_unsaturation.to_vec();
    let n_atoms = unsaturation.len();

    let mut start = Some(start);
    loop {
        // Number of partners each unsaturated atom can still pair with
        let mut n_partners = vec![0u32; n_atoms];
        for i in 0..n_atoms {
            if unsaturation[i] == 0 {
                continue;
            }
            n_partners[i] = adjacency[i]
                .iter()
                .filter(|&&(j, e)| unsaturation[j] > 0 && orders[e] < MAX_INFERRED_BOND_ORDER)
                .count() as u32;
        }

        let current = match start.take() {
            // In the first iteration the current index is set by the given start
            Some(first) => first,
            // In all subsequent iterations the atom with the fewest pairable
            // partners is chosen, as it has the fewest ways to form a multiple bond
            None => {
                match (0..n_atoms)
                    .filter(|&i| unsaturation[i] > 0 && n_partners[i] > 0)
                    .min_by_key(|&i| n_partners[i])
                {
                    Some(i) => i,
                    // No pairable atoms left
                    None => break,
                }
            }
        };
        let &(partner, edge) = adjacency[current]
            .iter()
            .filter(|&&(j, e)| unsaturation[j] > 0 && orders[e] < MAX_INFERRED_BOND_ORDER)
            .min_by_key(|&&(j, _)| n_partners[j])
            // Guaranteed by `n_partners[current] > 0`
            .unwrap();
        orders[edge] += 1;
        unsaturation[current] -= 1;
        unsaturation[partner] -= 1;
    }

    (
        orders,
        unsaturation.iter().filter(|&&value| value > 0).sum(),
    )
}

/// Get indices to all atoms that are directly or indirectly connected
/// to the root atom indicated by the given index.
///
/// An atom is *connected* to the `root` atom, if that atom is reachable
/// by traversing an arbitrary number of bonds, starting from the
/// `root`.
/// Effectively, this means that all atoms are *connected* to `root`,
/// that are in the same molecule as `root`.
/// Per definition `root` is also *connected* to itself.
///
/// Parameters
/// ----------
/// bond_list : BondList
///     The reference bond list.
/// root : int
///     The index of the root atom.
/// as_mask : bool, optional
///     If true, the connected atom indices are returned as boolean
///     mask.
///     By default, the connected atom indices are returned as integer
///     array.
///
/// Returns
/// -------
/// connected : ndarray, dtype=int or ndarray, dtype=bool
///     Either a boolean mask or an integer array, representing the
///     connected atoms.
///     In case of a boolean mask: ``connected[i] == True``, if the atom
///     with index ``i`` is connected.
///
/// Examples
/// --------
/// Consider a system with 4 atoms, where only the last atom is not
/// bonded with the other ones (``0-1-2 3``):
///
/// >>> bonds = BondList(4)
/// >>> bonds.add_bond(0, 1)
/// >>> bonds.add_bond(1, 2)
/// >>> print(find_connected(bonds, 0))
/// [0 1 2]
/// >>> print(find_connected(bonds, 1))
/// [0 1 2]
/// >>> print(find_connected(bonds, 2))
/// [0 1 2]
/// >>> print(find_connected(bonds, 3))
/// [3]
///
/// The result can be output as a boolean mask:
///
/// >>> print(find_connected(bonds, 0, as_mask=True))
/// [ True  True  True False]
#[pyfunction]
#[pyo3(signature = (bond_list, root, as_mask=false))]
pub fn find_connected(
    py: Python<'_>,
    bond_list: &BondList,
    root: usize,
    as_mask: bool,
) -> PyResult<Py<PyAny>> {
    let atom_count = bond_list.get_atom_count();
    let bonds = bond_list.get_bonds_ref();

    if root >= atom_count {
        return Err(exceptions::PyValueError::new_err(format!(
            "Root atom index {} is out of bounds for bond list representing {} atoms",
            root, atom_count
        )));
    }

    // Build adjacency list
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); atom_count];
    for bond in bonds {
        adj[bond.atom1].push(bond.atom2);
        adj[bond.atom2].push(bond.atom1);
    }

    // Iterative breadth-first search
    let mut visited = vec![false; atom_count];
    let mut queue = VecDeque::new();
    queue.push_back(root);
    visited[root] = true;

    while let Some(current) = queue.pop_front() {
        for &neighbor in &adj[current] {
            if !visited[neighbor] {
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }

    if as_mask {
        let mask = Array1::from_vec(visited);
        Ok(mask.into_pyarray(py).into_any().unbind())
    } else {
        let indices: Vec<i64> = visited
            .iter()
            .enumerate()
            .filter(|(_, &v)| v)
            .map(|(i, _)| i as i64)
            .collect();
        let arr = Array1::from_vec(indices);
        Ok(arr.into_pyarray(py).into_any().unbind())
    }
}

/// Squared Euclidean distance between two atoms in an Nx3 coordinate array.
#[inline(always)]
fn squared_distance(coord: &ArrayView2<f32>, a1: usize, a2: usize) -> f32 {
    let dx = coord[[a1, 0]] - coord[[a2, 0]];
    let dy = coord[[a1, 1]] - coord[[a2, 1]];
    let dz = coord[[a1, 2]] - coord[[a2, 2]];
    dx * dx + dy * dy + dz * dz
}

fn is_peptide_link(link_type: &str) -> bool {
    matches!(
        link_type,
        "PEPTIDE LINKING" | "L-PEPTIDE LINKING" | "D-PEPTIDE LINKING"
    )
}

fn is_nucleic_link(link_type: &str) -> bool {
    matches!(link_type, "RNA LINKING" | "DNA LINKING")
}

/// Find all indices where a value occurs in an array.
fn find<T: PartialEq>(array: &[T], value: &T) -> Vec<usize> {
    array
        .iter()
        .enumerate()
        .filter(|(_, x)| *x == value)
        .map(|(i, _)| i)
        .collect()
}
