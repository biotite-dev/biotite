use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions;
use pyo3::prelude::*;
use std::cmp::max;
use std::convert::TryInto;
use std::ops::{Index, IndexMut};

// Label as a separate module to indicate that this exception comes
// from biotite
mod biotite {
    pyo3::import_exception!(biotite.structure, BadStructureError);
}

/// The desired type of result to be returned by some :class:`CellList` methods.
///
/// - ``MAPPING`` -
///   An array mapping each atom index to the indices of its neighbors.
///   To handle different numbers of neighbors, the array is padded with -1 values.
/// - ``MATRIX`` -
///   A boolean matrix of shape ``(n,n)`` where the value at ``(i,j)`` is true
///   if the atoms at indices ``i`` and ``j`` are neighbors.
/// - ``PAIRS`` -
///   An array of shape ``(n,2)`` where each row contains the indices of two neighboring atoms.
///   This is basically a sparse representation of the ``MATRIX``.
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy)]
#[pyclass]
enum CellListResult {
    MAPPING,
    MATRIX,
    PAIRS,
}


/// Enum for efficient support of both single and multiple radii
/// given in `get_atoms()` and `get_atoms_in_cells()`.
enum Radius<T> {
    Single(T),
    Multiple(Vec<T>),
}

/// A three-dimensional container for storing elements in cells.
struct CellContainer<T> {
    data: Vec<T>,
    offsets: Array3<[usize; 2]>,
    max_cell_count: usize,
}

impl<T: Clone + Default> CellContainer<T> {
    /// Create a new ``CellContainer`` filled with the given elements at the given
    /// positions.
    ///
    /// Parameters
    /// ----------
    /// dimensions : [usize; 3]
    ///     The number of cells in each dimension.
    /// positions : Vec<[usize; 3]>
    ///     The cell to put each element into.
    /// elements : Vec<T>
    ///     The elements to store.
    ///
    pub fn new(dimensions: [usize; 3], positions: Vec<[usize; 3]>, elements: Vec<T>) -> Self {
        if positions.len() != elements.len() {
            panic!("Positions and elements must have the same length");
        }

        // First iteration: Count the number of elements in each cell
        let mut counts: Array3<usize> = Array3::zeros(dimensions);
        for position in positions.iter() {
            counts[*position] += 1;
        }

        // Second iteration: Allocate the elements vector
        // and calculate the offsets to it for each cell
        let mut offsets: Array3<[usize; 2]> = Array3::from_elem(dimensions, [0, 0]);
        let mut max_cell_count = 0;
        let mut current_offset = 0;
        for (count, offset) in counts.iter().zip(offsets.iter_mut()) {
            // The start position is set here
            // The stop position is incremented in third iteration to track the current index
            offset[0] = current_offset;
            offset[1] = current_offset; // Initially start == stop (empty)
            current_offset += count;
            max_cell_count = max(max_cell_count, *count);
        }

        // Pre-allocate the data vector with default values
        let total_count = current_offset;
        let mut data: Vec<T> = vec![T::default(); total_count];

        // Third iteration: Fill the data vector
        for (position, element) in positions.iter().zip(elements.into_iter()) {
            let offset = &mut offsets[*position];
            let current_index = offset[1];
            data[current_index] = element;
            offset[1] += 1;
        }

        CellContainer {
            data,
            offsets,
            max_cell_count,
        }
    }

    /// Check if a given cell position is within the bounds of the ``CellContainer``.
    ///
    /// Parameters
    /// ----------
    /// cell_position
    ///     The cell position to check.
    ///
    /// Returns
    /// -------
    /// True if the cell position is within the bounds of the cell container, false otherwise.
    ///
    /// Notes
    /// -----
    /// If the cell position is outside the bounds of the ``CellContainer``,
    /// ``index()`` will return an empty slice and ``index_mut()`` will panic.
    pub fn cell_exists(&self, cell_position: [isize; 3]) -> bool {
        let shape = self.offsets.shape();
        for i in 0..3 {
            if cell_position[i] < 0 || cell_position[i] >= shape[i] as isize {
                return false;
            }
        }
        true
    }

    /// Get the maximum number of elements in any cell.
    ///
    /// Returns
    /// -------
    /// The maximum number of elements in any cell.
    pub fn max_cell_count(&self) -> usize {
        self.max_cell_count
    }
}

impl<T: Clone + Default> Index<[isize; 3]> for CellContainer<T> {
    type Output = [T];
    fn index(&self, cell_position: [isize; 3]) -> &Self::Output {
        if self.cell_exists(cell_position) {
            let cell_position = as_usize(cell_position).unwrap();
            &self.data[self.offsets[cell_position][0]..self.offsets[cell_position][1]]
        } else {
            &self.data[0..0]
        }
    }
}

impl<T: Clone + Default> IndexMut<[isize; 3]> for CellContainer<T> {
    fn index_mut(&mut self, cell_position: [isize; 3]) -> &mut Self::Output {
        if !self.cell_exists(cell_position) {
            panic!(
                "Cell position ({},{},{}) is out of bounds",
                cell_position[0], cell_position[1], cell_position[2]
            );
        }
        let cell_position = as_usize(cell_position).unwrap();
        &mut self.data[self.offsets[cell_position][0]..self.offsets[cell_position][1]]
    }
}

/// __init__(atom_array, cell_size, periodic=False, box=None, selection=None)
///
/// This class enables the efficient search of atoms in vicinity of a
/// defined location.
///
/// This class stores the indices of an atom array in virtual "cells",
/// each corresponding to a specific coordinate interval.
/// If the atoms in vicinity of a specific location are searched, only
/// the atoms in the relevant cells are checked.
/// Effectively this decreases the operation time for finding atoms
/// with a maximum distance to given coordinates from *O(n)* to *O(1)*,
/// after the :class:`CellList` has been created.
/// Therefore a :class:`CellList` saves calculation time in those
/// cases, where vicinity is checked for multiple locations.
///
/// Parameters
/// ----------
/// atom_array : AtomArray or ndarray, dtype=float, shape=(n,3)
///     The :class:`AtomArray` to create the :class:`CellList` for.
///     Alternatively the atom coordinates are accepted directly.
///     In this case `box` must be set, if `periodic` is true.
/// cell_size : float
///     The coordinate interval each cell has for x, y and z axis.
///     The amount of cells depends on the range of coordinates in the
///     `atom_array` and the `cell_size`.
/// periodic : bool, optional
///     If true, the cell list considers periodic copies of atoms.
///     The periodicity is based on the `box` attribute of `atom_array`.
/// box : ndarray, dtype=float, shape=(3,3), optional
///     If provided, the periodicity is based on this parameter instead
///     of the :attr:`box` attribute of `atom_array`.
///     Only has an effect, if `periodic` is ``True``.
/// selection : ndarray, dtype=bool, shape=(n,), optional
///     If provided, only the atoms masked by this array are stored in
///     the cell list. However, the indices stored in the cell list
///     will still refer to the original unfiltered `atom_array`.
///
/// Examples
/// --------
///
/// >>> cell_list = CellList(atom_array, cell_size=5)
/// >>> near_atoms = atom_array[cell_list.get_atoms(np.array([1,2,3]), radius=7.0)]
#[pyclass]
pub struct CellList {
    coord: Vec<[f32; 3]>,
    // A boolean mask that covers the selected atoms
    selection: Option<Vec<bool>>,
    cells: CellContainer<usize>,
    cell_size: f32,
    // The minimum and maximum coordinates for all atoms
    // Used as origin ('coord_range[0]' is at 'cells[[0,0,0]]')
    // and for bound checks
    coord_range: [[f32; 3]; 2],
    // The box dimensions if periodicity is taken into account
    periodic_box: Option<[[f32; 3]; 3]>,
    // The length of the array before appending periodic copies
    orig_length: usize,
}

#[pymethods]
impl CellList {
    #[new]
    #[pyo3(signature = (atom_array, cell_size, periodic=false, r#box=None, selection=None))]
    fn new<'py>(
        py: Python<'py>,
        atom_array: Bound<'py, PyAny>,
        cell_size: f32,
        periodic: bool,
        r#box: Option<Bound<'py, PyAny>>,
        selection: Option<Vec<bool>>,
    ) -> PyResult<Self> {
        let struc = PyModule::import(py, "biotite.structure")?;

        // Input validation
        if !atom_array.is_instance(&struc.getattr("AtomArray")?)? {
            return Err(exceptions::PyTypeError::new_err(format!(
                "Expected 'AtomArray' but got '{}'",
                atom_array
                    .getattr("__class__")?
                    .getattr("__name__")?
                    .extract::<&str>()?
            )));
        }
        if cell_size <= 0.0 {
            return Err(exceptions::PyValueError::new_err(
                "Cell size must be greater than 0",
            ));
        }

        let orig_length: usize;
        let mut coord_object: Bound<'py, PyAny> = struc.call_method1("coord", (&atom_array,))?;
        let box_object: Bound<'py, PyAny>;
        let periodic_box: Option<[[f32; 3]; 3]>;
        if periodic {
            match r#box {
                Some(wrapped_box) => {
                    box_object = wrapped_box;
                }
                None => {
                    // Use `box` attribute of `AtomArray`
                    match atom_array
                        .getattr("box")?
                        .extract::<Option<Bound<'py, PyAny>>>()?
                    {
                        Some(atoms_box) => {
                            box_object = atoms_box;
                        }
                        None => {
                            return Err(exceptions::PyValueError::new_err(
                                "AtomArray must have a box to enable periodicity",
                            ));
                        }
                    }
                }
            }

            orig_length = coord_object.call_method0("array_length")?.extract()?;
            coord_object = struc
                .getattr("move_inside_box")?
                .call1((&coord_object, &box_object))?;
            coord_object = struc
                .getattr("repeat_box_coord")?
                .call1((&coord_object, &box_object))?
                .extract::<[Bound<'py, PyAny>; 2]>()?
                .first()
                .ok_or_else(|| {
                    exceptions::PyRuntimeError::new_err("Could not get repeated coords")
                })?
                .clone();
            periodic_box = Some(
                extract_coord(box_object.extract::<PyReadonlyArray2<f32>>()?)?
                    .try_into()
                    .map_err(|_| {
                        exceptions::PyTypeError::new_err("Box must be a 3x3 float32 matrix")
                    })?,
            );
        } else {
            periodic_box = None;
            orig_length = coord_object.call_method0("array_length")?.extract()?;
        }
        let coord: Vec<[f32; 3]> = extract_coord(coord_object.extract::<PyReadonlyArray2<f32>>()?)?;
        let coord_range: [[f32; 3]; 2] = calculate_coord_range(&coord)?;

        if let Some(ref sel) = selection {
            if sel.len() != orig_length {
                return Err(exceptions::PyIndexError::new_err(format!(
                    "Atom array has length {}, but selection has length {}",
                    orig_length,
                    sel.len()
                )));
            }
        }

        let mut positions: Vec<[usize; 3]> = Vec::with_capacity(coord.len());
        let mut elements: Vec<usize> = Vec::with_capacity(coord.len());
        for (i, coord) in coord.iter().enumerate() {
            if let Some(ref selection) = selection {
                if !selection[i % orig_length] {
                    continue;
                }
            }
            positions
                .push(as_usize(Self::get_cell_position(*coord, &coord_range, cell_size)).unwrap());
            // Ensure that the atom indices point to the original atom array
            // and not to the periodic copies
            elements.push(i % orig_length);
        }
        // To get the number of cells in each dimension, use the position of the maximum coordinate
        let dimensions: [usize; 3] =
            Self::get_cell_position(coord_range[1], &coord_range, cell_size)
                .iter()
                .map(|x| (x + 1) as usize)
                .collect::<Vec<usize>>()
                .try_into()
                .unwrap();
        let cells = CellContainer::new(dimensions, positions, elements);

        Ok(CellList {
            coord,
            selection,
            cells,
            cell_size,
            coord_range,
            periodic_box,
            orig_length,
        })
    }

    fn create_adjacency_matrix<'py>(
        &self,
        py: Python<'py>,
        threshold_distance: f32,
    ) -> PyResult<Bound<'py, PyArray2<bool>>> {
        if threshold_distance < 0.0 {
            return Err(exceptions::PyValueError::new_err(
                "Threshold distance must be a positive value",
            ));
        }

        match self.selection {
            Some(ref selection) => {
                let coord: Vec<[f32; 3]> = self
                    .coord
                    .iter()
                    .take(self.orig_length)
                    .enumerate()
                    .filter(|(i, _)| selection[*i])
                    .map(|(_, c)| *c)
                    .collect();
                let mut sub_matrix: Array2<bool> =
                    Array2::default((coord.len(), self.orig_length));
                for matrix_index in self.get_atoms_from_slice(&coord, Radius::Single(threshold_distance)) {
                    sub_matrix[matrix_index] = true;
                }
                let mut matrix: Array2<bool> =
                    Array2::default((self.orig_length, self.orig_length));
                let mut sub_row_idx = 0;
                for (i, is_selected) in selection.iter().enumerate().take(self.orig_length) {
                    if *is_selected {
                        matrix.row_mut(i).assign(&sub_matrix.row(sub_row_idx));
                        sub_row_idx += 1;
                    }
                }
                Ok(matrix.into_pyarray(py))
            }
            None => {
                let mut matrix: Array2<bool> =
                    Array2::default((self.orig_length, self.orig_length));
                for matrix_index in self.get_atoms_from_slice(&self.coord, Radius::Single(threshold_distance)) {
                    matrix[matrix_index] = true;
                }
                Ok(matrix.into_pyarray(py))
            }
        }
    }

    #[pyo3(signature = (coord, radius, as_mask=false, result_format=CellListResult::MAPPING))]
    fn get_atoms<'py>(
        &self,
        py: Python<'py>,
        mut coord: &Bound<'py, PyAny>,
        radius: &Bound<'py, PyAny>,
        as_mask: bool,
        mut result_format: CellListResult,
    ) -> PyResult<Bound<'py, PyAny>> {
        let struc = PyModule::import(py, "biotite.structure")?;

        let (converted_coord, is_multi_coord) = prepare_coord_from_python(coord, self.periodic_box)?;
        let converted_radius: Radius<f32> = prepare_radius_from_python(radius)?;
        let pairs = self.get_atoms_from_slice(&converted_coord, converted_radius);
        format_result(py, converted_coord.len(), pairs, result_format, as_mask, is_multi_coord)
    }

    #[pyo3(signature = (coord, cell_radius=1, as_mask=false, result_format=CellListResult::MAPPING))]
    fn get_atoms_in_cells<'py>(
        &self,
        py: Python<'py>,
        mut coord: &Bound<'py, PyAny>,
        cell_radius: &Bound<'py, PyAny>,
        as_mask: bool,
        mut result_format: CellListResult,
    ) -> PyResult<Bound<'py, PyAny>> {
        let struc = PyModule::import(py, "biotite.structure")?;

        let (converted_coord, is_multi_coord) = prepare_coord_from_python(coord, self.periodic_box)?;
        let converted_radius: Radius<i32> = prepare_radius_from_python(cell_radius)?;
        let pairs = self.get_atoms_in_cells_from_slice(&converted_coord, converted_radius);
        format_result(py, converted_coord.len(), pairs, result_format, as_mask, is_multi_coord)
    }


}

impl CellList {
    /// Given a coordinate [x, y, z], compute the corresponding cell position [i, j, k].
    /// Also (negative) cell indices outside the actual initilaized cells can be returned,
    /// if the given coordinate is outside the coordinate range.
    fn get_cell_position(
        coord: [f32; 3],
        coord_range: &[[f32; 3]; 2],
        cellsize: f32,
    ) -> [isize; 3] {
        [
            ((coord[0] - coord_range[0][0]) / cellsize).floor() as isize,
            ((coord[1] - coord_range[0][1]) / cellsize).floor() as isize,
            ((coord[2] - coord_range[0][2]) / cellsize).floor() as isize,
        ]
    }

    fn get_atoms_from_slice(
        &self,
        coord: &[[f32; 3]],
        radius: Radius<f32>,
    ) -> Vec<[usize; 2]> {
        match radius {
            Radius::Single(r) => {
                let sq_r = r.powi(2);
                self.get_atoms_in_cells_from_slice(coord, Radius::Single((r / self.cell_size).ceil() as i32))
                    .iter()
                    .filter(|pair| distance_squared(coord[pair[0]], self.coord[pair[1]]) <= sq_r)
                    .map(|pair| *pair)
                    .collect()
            }
            Radius::Multiple(rs) => {
                let radii = rs.iter().map(|r| (*r / self.cell_size).ceil() as i32).collect();
                self.get_atoms_in_cells_from_slice(coord, Radius::Multiple(radii))
                    .iter()
                    .zip(rs.iter())
                    .filter(|(pair, r)| distance_squared(coord[pair[0]], self.coord[pair[1]]) <= r.powi(2))
                    .map(|(pair, _)| *pair)
                    .collect()
            }
        }
    }


    fn get_atoms_in_cells_from_slice(
        &self,
        coord: &[[f32; 3]],
        cell_radii: Radius<i32>,
    ) -> Vec<[usize; 2]> {
        let mut adjacent_atoms = Vec::new();

        for (coord_idx, c) in coord.iter().enumerate() {
            // Skip non-finite coordinates
            if !c[0].is_finite() || !c[1].is_finite() || !c[2].is_finite() {
                continue;
            }

            let cell_pos = Self::get_cell_position(*c, &self.coord_range, self.cell_size);

            // Iterate over all adjacent cells within the cell_radius
            let cell_radius = match cell_radii {
                Radius::Single(r) => r as isize,
                Radius::Multiple(ref rs) => rs[coord_idx] as isize,
            };
            for adj_i in (cell_pos[0] - cell_radius)..=(cell_pos[0] + cell_radius) {
                for adj_j in (cell_pos[1] - cell_radius)..=(cell_pos[1] + cell_radius) {
                    for adj_k in (cell_pos[2] - cell_radius)..=(cell_pos[2] + cell_radius) {
                        let adj_cell_pos = [adj_i, adj_j, adj_k];
                        // Get all atoms in this cell (returns empty slice if out of bounds)
                        for &atom_idx in &self.cells[adj_cell_pos] {
                            adjacent_atoms.push([coord_idx, atom_idx]);
                        }
                    }
                }
            }
        }
        adjacent_atoms
    }
}

/// Calculate the min and max coordinates from a coordinate list
fn calculate_coord_range(coord: &[[f32; 3]]) -> PyResult<[[f32; 3]; 2]> {
    if coord.is_empty() {
        return Err(exceptions::PyValueError::new_err(
            "Coordinates must not be empty",
        ));
    }

    let mut min_coord = [f32::INFINITY, f32::INFINITY, f32::INFINITY];
    let mut max_coord = [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];

    for c in coord.iter() {
        for i in 0..3 {
            if c[i] < min_coord[i] {
                min_coord[i] = c[i];
            }
            if c[i] > max_coord[i] {
                max_coord[i] = c[i];
            }
        }
    }

    Ok([min_coord, max_coord])
}

/// Extract coordinates from a 2D NumPy array into a Vec of [f32; 3]
fn extract_coord(coord_array: PyReadonlyArray2<f32>) -> PyResult<Vec<[f32; 3]>> {
    let coord_ndarray = coord_array.as_array();
    let shape = coord_ndarray.shape();
    if shape.len() != 2 {
        return Err(exceptions::PyValueError::new_err(
            "Coordinates must have shape (n,3)",
        ));
    }
    if shape[0] == 0 {
        return Err(exceptions::PyValueError::new_err(
            "Coordinates must not be empty",
        ));
    }
    if shape[1] != 3 {
        return Err(exceptions::PyValueError::new_err(
            "Coordinates must have form (x,y,z)",
        ));
    }
    if !coord_array.as_array().is_all_infinite() {
        return Err(exceptions::PyValueError::new_err(
            "Coordinates contain non-finite values",
        ));
    }

    let view = coord_array.as_array();
    let mut result: Vec<[f32; 3]> = Vec::with_capacity(shape[0]);

    for i in 0..shape[0] {
        let x = view[[i, 0]];
        let y = view[[i, 1]];
        let z = view[[i, 2]];
        result.push([x, y, z]);
    }

    Ok(result)
}

fn prepare_coord_from_python(coord: &Bound<'py, PyAny>, periodic_box: Option<[[f32; 3]; 3]>) -> PyResult<(Vec<[f32; 3]>), bool> {
    // Convert into expected f32 array
    coord = coord.call_method("astype", (), Some([("dtype", "float32"), ("copy", false)].into_py_dict(py)?))?;
    if let Some(r#box) = self.periodic_box {
        coord = struc
            .getattr("move_inside_box")?
            .call1((&coord, r#box.into_pyarray(py)))?;
    }
    // Consistently use 2 dimensions
    // -> if only one coordinate is given, convert it to a 2D array
    let converted_coord: Vec<[f32; 3]>;
    if let Ok(c) = coord.cast::<[f32; 3]>() {
        (vec![c], false)
    } else if let Ok(cs) = coord.cast::<PyReadonlyArray2<f32>>() {
        (extract_coord(cs)?, true)
    } else {
        Err(exceptions::PyTypeError::new_err(
            "Coordinates must be a single coordinate or a 2D array of coordinates",
        ));
    }
}

fn prepare_radius_from_python<T>(radius: &Bound<'py, PyAny>) -> PyResult<Radius<T>> {
    if let Ok(r) = radius.cast::<T>() {
        Radius::Single(r);
    } else if let Ok(rs) = radius.cast::<Vec<T>>() {
        Radius::Multiple(rs);
    } else {
        Err(exceptions::PyTypeError::new_err(
            "Radius must be a single float or an array of floats",
        ));
    }
}

fn as_usize(x: [isize; 3]) -> Option<[usize; 3]> {
    for e in &x {
        if *e < 0 {
            return None;
        }
    }
    Some([x[0] as usize, x[1] as usize, x[2] as usize])
}


fn distance_squared(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}


fn format_as_mapping<'py>(py: Python<'py>, n: usize, pairs: Vec<[usize; 2]>, is_multi_coord: bool) -> PyResult<Bound<'py, PyAny>> {
    let mut mapping: Vec<Vec<i64>> = Vec::with_capacity(n);
    for pair in pairs.iter() {
        mapping[pair[0]].push(pair[1] as i64);
    }

    let max_length = mapping.iter().map(|elem| elem.len()).max().unwrap_or(0);
    let mut mapping_matrix = Array2::from_elem((mapping.len(), max_length), -1);
    for (i, row) in mapping.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            mapping_matrix[[i, j]] = *value;
        }
    }

    if is_multi_coord {
        Ok(mapping_matrix.into_pyarray(py))
    } else {
        Ok(mapping_matrix.row(0).into_pyarray(py))
    }
}

fn format_as_matrix<'py>(py: Python<'py>, n: usize, pairs: Vec<[usize; 2]>, is_multi_coord: bool) -> PyResult<Bound<'py, PyAny>> {
    let mut mapping = Array2::from_elem((n, n), false);
    for pair in pairs.iter() {
        mapping[pair] = true;
    }

    if is_multi_coord {
        Ok(mapping.into_pyarray(py))
    } else {
        Ok(mapping.row(0).into_pyarray(py))
    }
}

fn format_as_pairs<'py>(py: Python<'py>, n: usize, pairs: Vec<[usize; 2]>, is_multi_coord: bool) -> PyResult<Bound<'py, PyAny>> {
    if is_multi_coord {
        Ok(pairs.into_pyarray(py))
    } else {
        Ok(pairs.row(1).into_pyarray(py))
    }
}

fn format_result<'py>(py: Python<'py>, n: usize, pairs: Vec<[usize; 2]>, mut result_format: CellListResult, as_mask: bool, is_multi_coord: bool) -> PyResult<Bound<'py, PyAny>> {
    if as_mask {
        result_format = CellListResult::MATRIX;
    }
    match result_format {
        CellListResult::MAPPING => format_as_mapping(py, n, pairs, is_multi_coord),
        CellListResult::MATRIX => format_as_matrix(py, n, pairs, is_multi_coord),
        CellListResult::PAIRS => format_as_pairs(py, n, pairs, is_multi_coord),
    }
}
