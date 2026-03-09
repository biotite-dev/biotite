use itertools::Itertools;
use numpy::ndarray::Array1;
use numpy::ndarray::ArrayView2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions;
use pyo3::prelude::*;
use std::collections::{HashMap, VecDeque};

use crate::structure::bonds::{Bond, BondList, BondType};

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
/// elements
///     The element name of each atom.
/// residue_starts
///     The start index of each residue (including exclusive stop index).
/// dist_ranges
///     The distance ranges mapping element pairs to (min_dist, max_dist).
///     Both element orders must be included.
/// default_bond_type
///     The bond type to assign to all created bonds.
///
/// Returns
/// -------
/// bonds
///     The bonds between the atoms.
#[pyfunction]
#[allow(clippy::needless_range_loop)]
pub fn connect_via_distances<'py>(
    _py: Python<'py>,
    coord: PyReadonlyArray2<'py, f32>,
    elements: Vec<String>,
    residue_starts: PyReadonlyArray1<'py, i64>,
    dist_ranges: HashMap<(String, String), (f32, f32)>,
    default_bond_type: BondType,
) -> PyResult<BondList> {
    let coord = coord.as_array();
    let residue_starts = residue_starts.as_array();
    let n_atoms = elements.len();

    // Create a mapping from element names to integers
    // to allow a more efficient lookup in the hot loop
    let mut element_to_index: HashMap<&str, usize> = HashMap::new();
    for (element1, element2) in dist_ranges.keys() {
        let next = element_to_index.len();
        element_to_index.entry(element1.as_str()).or_insert(next);
        let next = element_to_index.len();
        element_to_index.entry(element2.as_str()).or_insert(next);
    }

    // Map each atom to its element index, None if not in dist_ranges
    let atom_element_index: Vec<Option<usize>> = elements
        .iter()
        .map(|element| element_to_index.get(element.as_str()).copied())
        .collect();

    // Build a flat 2D lookup table for squared distance ranges indexed by element indices
    // to avoid computing expensive square roots in the hot loop
    let n_unique_elements = element_to_index.len();
    let mut distance_range_table: Vec<Option<(f32, f32)>> =
        vec![None; n_unique_elements * n_unique_elements];
    for ((element1, element2), &(min_dist, max_dist)) in &dist_ranges {
        if let (Some(&index1), Some(&index2)) = (
            element_to_index.get(element1.as_str()),
            element_to_index.get(element2.as_str()),
        ) {
            distance_range_table[index1 * n_unique_elements + index2] =
                Some((min_dist * min_dist, max_dist * max_dist));
        }
    }

    let mut bond_list = BondList::empty(n_atoms);

    for (start, stop) in residue_starts.iter().tuple_windows() {
        let start = *start as usize;
        let stop = *stop as usize;

        for atom_i in start..stop {
            let element_i = match atom_element_index[atom_i] {
                Some(index) => index,
                None => continue,
            };
            for atom_j in start..atom_i {
                let element_j = match atom_element_index[atom_j] {
                    Some(index) => index,
                    None => continue,
                };
                let (min_distance, max_distance) =
                    match distance_range_table[element_i * n_unique_elements + element_j] {
                        Some(range) => range,
                        None => continue,
                    };
                let distance = squared_distance(&coord, atom_i, atom_j);
                if distance >= min_distance && distance <= max_distance {
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

/// find_connected(bond_list, root, as_mask=False)
///
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
