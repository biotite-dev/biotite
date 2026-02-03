use num_traits::{Float, FloatConst};
use numpy::ndarray::Axis;
use numpy::ndarray::{
    Array, Array1, Array3, ArrayView1, ArrayView3, ArrayViewD, ArrayViewMut1, Ix3,
};
use numpy::PyArrayMethods;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3,
};
use pyo3::exceptions;
use pyo3::ffi::{c_str, CO_FUTURE_UNICODE_LITERALS};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyEllipsis, PyString, PyType};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::convert::TryInto;
use std::fs;
use std::ops::{Index, IndexMut};
use std::str::FromStr;

use std::cmp::max;
use std::ops::Range;

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
#[pyclass]
enum CellListResult {
    MAPPING,
    MATRIX,
    PAIRS,
}

/// A three-dimensional container for storing elements in cells.
///
/// Attributes
/// ----------
/// max_cell_count : int
///     The maximum number of elements in any cell.
struct CellContainer<T> {
    data: Vec<T>,
    offsets: Array3<[usize; 2]>,
    pub max_cell_count: usize,
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
    fn new(dimensions: [usize; 3], positions: Vec<[usize; 3]>, elements: Vec<T>) -> Self {
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

    fn cell_exists(&self, cell_position: [isize; 3]) -> bool {
        let shape = self.offsets.shape();
        for i in 0..3 {
            if cell_position[i] < 0 || cell_position[i] >= shape[i] as isize {
                return false;
            }
        }
        true
    }
}

impl<T: Clone + Default> Index<[isize; 3]> for CellContainer<T> {
    type Output = [T];
    fn index(&self, cell_position: [isize; 3]) -> &Self::Output {
        if self.cell_exists(cell_position) {
            let cell_position = as_usize(cell_position).unwrap();
            return &self.data[self.offsets[cell_position][0]..self.offsets[cell_position][1]];
        } else {
            return &self.data[0..0];
        }
    }
}

impl<T: Clone + Default> IndexMut<[isize; 3]> for CellContainer<T> {
    fn index_mut(&mut self, cell_position: [isize; 3]) -> &mut Self::Output {
        if self.cell_exists(cell_position) {
            let cell_position = as_usize(cell_position).unwrap();
            return &mut self.data[self.offsets[cell_position][0]..self.offsets[cell_position][1]];
        } else {
            return &mut self.data[0..0];
        }
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
        let mut coord_object: Bound<'py, PyAny> = struc.call_method1("coord", (atom_array,))?;
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
                .call1((coord_object, box_object))?;
            coord_object = struc
                .getattr("repeat_box_coord")?
                .call1((coord_object, periodic_box))?
                .extract::<[Bound<'py, PyAny>; 2]>()?[0];
            periodic_box =
                Some(extract_coord(box_object.extract::<PyReadonlyArray2<f32>>()?)?.into());
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
            if let Some(selection) = selection {
                if !selection[i % orig_length] {
                    continue;
                }
            }
            positions.push(Self::get_cell_position(*coord, &coord_range, cell_size).into());
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
                .into();
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

fn as_usize(x: [isize; 3]) -> Option<[usize; 3]> {
    for i in 0..3 {
        if x[i] < 0 {
            return None;
        }
    }
    Some([x[0] as usize, x[1] as usize, x[2] as usize])
}
