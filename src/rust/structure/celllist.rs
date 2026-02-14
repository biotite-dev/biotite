use crate::structure::util::{distance_squared, extract_coord};
use crate::util::check_signals_periodically;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyTuple};
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
///   An ``(q, k)`` array mapping each query coordinate ``q`` to the indices of its
///   neighbors.
///   Since the coordinates may have different amounts of adjacent
///   atoms, trailing ``-1`` values are used to indicate nonexisting
///   indices.
///   If only a single query coordinate is provided, the return value has shape
///   ``(k,)``.
/// - ``MASK`` -
///   A boolean mask of shape ``(q, n)`` where the value at ``(i, j)`` is ``True``
///   if the query coordinate ``i`` and atom ``j`` are neighbors.
///   If only a single query coordinate is provided, the return value has shape
///   ``(n,)``.
/// - ``PAIRS`` -
///   An array of shape ``(k, 2)`` where each row contains the ``(query_idx, atom_idx)``
///   tuple of a found neighboring atom.
///   This is basically a sparse representation of the ``MASK``.
///   If only a single query coordinate is provided, the return value has shape
///   ``(k,)``, equivalent to ``MAPPING``.
///
/// Examples
/// --------
///
/// >>> # Create a CellList for a small molecule
/// >>> from biotite.structure.info import residue
/// >>> atoms = residue("ALA")
/// >>> cell_list = CellList(atoms, cell_size=2)
/// >>> # Demonstrate results for both, single and multiple query coordinates
/// >>> single_coord = atoms.coord[0]
/// >>> multiple_coords = atoms.coord[:2]
/// >>> # MAPPING: indices of neighboring atoms, -1 indicates padding values to be ignored
/// >>> print(cell_list.get_atoms(single_coord, radius=2, result_format=CellList.Result.MAPPING))
/// [6 1 0 7]
/// >>> print(cell_list.get_atoms(multiple_coords, radius=2, result_format=CellList.Result.MAPPING))
/// [[ 6  1  0  7 -1]
///  [ 2  1  0  4  8]]
/// >>> # MASK: boolean mask indicating neighbors
/// >>> print(cell_list.get_atoms(single_coord, radius=2, result_format=CellList.Result.MASK))
/// [ True  True False False False False  True  True False False False False
///  False]
/// >>> print(cell_list.get_atoms(multiple_coords, radius=2, result_format=CellList.Result.MASK))
/// [[ True  True False False False False  True  True False False False False
///   False]
///  [ True  True  True False  True False False False  True False False False
///   False]]
/// >>> # PAIRS: (query_idx, atom_idx) tuples
/// >>> print(cell_list.get_atoms(single_coord, radius=2, result_format=CellList.Result.PAIRS))
/// [6 1 0 7]
/// >>> print(cell_list.get_atoms(multiple_coords, radius=2, result_format=CellList.Result.PAIRS))
/// [[0 6]
///  [0 1]
///  [0 0]
///  [0 7]
///  [1 2]
///  [1 1]
///  [1 0]
///  [1 4]
///  [1 8]]
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy, PartialEq)]
#[pyclass(module = "biotite.structure")]
pub enum CellListResult {
    MAPPING,
    MASK,
    PAIRS,
}

/// Enum for efficient support of both single and multiple radii
/// given in :meth:`CellList.get_atoms` and :meth:`CellList.get_atoms_in_cells`.
///
/// This avoids the overhead of always allocating a ``Vec`` when a single radius
/// is provided, which is the common case.
#[derive(Clone, Debug)]
pub enum Radius<T> {
    /// A single radius value applied to all coordinates.
    Single(T),
    /// Individual radius values for each coordinate.
    Multiple(Vec<T>),
}

/// A three-dimensional grid of cells, where each cell can store a variable number of elements.
///
/// All elements are stored in a single contiguous ``Vec<T>``, with
/// separate offsets tracking which elements belong to each cell.
/// The offset layout uses interleaved *(start, end)* pairs: for cell *[i, j, k]*,
/// the start offset is at ``offsets[2 * linear_idx]`` and the end offset is at
/// ``offsets[2 * linear_idx + 1]``, where ``linear_idx = i * stride_i + j * stride_j + k``.
struct CellGrid<T> {
    /// All elements stored contiguously. Elements for each cell are grouped together.
    data: Vec<T>,
    /// Flattened array of *(start, end)* offset pairs for each cell.
    /// Layout: for cell *[i, j, k]*, the offset pair is at index
    /// ``2 * (i * stride_i + j * stride_j + k)``.
    offsets: Vec<usize>,
    /// Dimensions of the grid *[dim_i, dim_j, dim_k]*.
    dimensions: [usize; 3],
    /// Precomputed stride for the first dimension: ``dimensions[1] * dimensions[2]``.
    stride_i: usize,
    /// Precomputed stride for the second dimension: ``dimensions[2]``.
    stride_j: usize,
}

impl<T: Clone + Default> CellGrid<T> {
    /// Create a :class:`CellGrid` directly from raw data and offsets.
    ///
    /// This constructor is primarily used for unpickling, where the internal
    /// state has been serialized and needs to be reconstructed without
    /// recomputing the cell assignments.
    ///
    /// Parameters
    /// ----------
    /// data
    ///     The flattened element data.
    /// offsets
    ///     The interleaved *(start, end)* offset pairs for each cell.
    /// dimensions
    ///     The number of cells in each dimension *[dim_i, dim_j, dim_k]*.
    pub fn new(data: Vec<T>, offsets: Vec<usize>, dimensions: [usize; 3]) -> Self {
        let stride_i = dimensions[1] * dimensions[2];
        let stride_j = dimensions[2];
        CellGrid {
            data,
            offsets,
            dimensions,
            stride_i,
            stride_j,
        }
    }

    /// Create a new :class:`CellGrid` by assigning elements to cells based on their positions.
    ///
    /// This is the primary constructor used during :class:`CellList` creation.
    /// It performs three passes over the data:
    ///
    /// 1. Count elements per cell to determine offsets.
    /// 2. Allocate the data vector and compute offset ranges.
    /// 3. Place elements into their respective cells.
    ///
    /// Parameters
    /// ----------
    /// dimensions : [usize; 3]
    ///     The number of cells in each dimension *[dim_i, dim_j, dim_k]*.
    /// positions : Vec<[usize; 3]>
    ///     The cell position *[i, j, k]* for each element.
    /// elements : Vec<T>
    ///     The elements to store.
    ///
    pub fn from_elements(
        dimensions: [usize; 3],
        positions: Vec<[usize; 3]>,
        elements: Vec<T>,
    ) -> Self {
        if positions.len() != elements.len() {
            panic!("Positions and elements must have the same length");
        }

        let n_cells = dimensions[0] * dimensions[1] * dimensions[2];
        let stride_i = dimensions[1] * dimensions[2];
        let stride_j = dimensions[2];

        // First iteration: Count the number of elements in each cell
        let mut counts: Vec<usize> = vec![0; n_cells];
        for position in positions.iter() {
            let linear_idx = position[0] * stride_i + position[1] * stride_j + position[2];
            counts[linear_idx] += 1;
        }

        // Second iteration: Calculate offsets
        // Each cell has a (start, end) pair, stored as offsets[2*idx] and offsets[2*idx + 1]
        let mut offsets: Vec<usize> = vec![0; n_cells * 2];
        let mut current_offset = 0;
        for (idx, count) in counts.iter().enumerate() {
            offsets[idx * 2] = current_offset; // start
            offsets[idx * 2 + 1] = current_offset; // end (will be incremented when filling)
            current_offset += count;
        }

        // Pre-allocate the data vector with default values
        let total_count = current_offset;
        let mut data: Vec<T> = vec![T::default(); total_count];

        // Third iteration: Fill the data vector
        for (position, element) in positions.iter().zip(elements.into_iter()) {
            let linear_idx = position[0] * stride_i + position[1] * stride_j + position[2];
            let end_idx = linear_idx * 2 + 1;
            let current_index = offsets[end_idx];
            data[current_index] = element;
            offsets[end_idx] += 1;
        }

        CellGrid {
            data,
            offsets,
            dimensions,
            stride_i,
            stride_j,
        }
    }

    /// Check if a cell position is within the valid bounds of the grid.
    ///
    /// Returns ``False`` if any index is negative or exceeds the corresponding dimension.
    #[inline(always)]
    fn is_valid_cell_position(&self, cell_position: [isize; 3]) -> bool {
        for (i, &element) in cell_position.iter().enumerate() {
            if element < 0 || element as usize >= self.dimensions[i] {
                return false;
            }
        }
        true
    }

    /// Convert a 3D cell position to a linear index into the offsets array.
    ///
    /// Notes
    /// -----
    /// The caller must ensure that ``cell_position`` is within bounds before
    /// using the returned index to access ``self.offsets``.
    #[inline(always)]
    fn get_linear_index(&self, cell_position: [isize; 3]) -> usize {
        (cell_position[0] as usize) * self.stride_i
            + (cell_position[1] as usize) * self.stride_j
            + (cell_position[2] as usize)
    }
}

/// Immutable indexing into a :class:`CellGrid`.
///
/// Returns a slice of all elements in the cell at the given position.
/// If the position is out of bounds (negative or exceeding dimensions),
/// an empty slice is returned instead of panicking. This allows safe
/// iteration over adjacent cells without explicit bounds checking.
impl<T: Clone + Default> Index<[isize; 3]> for CellGrid<T> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, cell_position: [isize; 3]) -> &Self::Output {
        if !self.is_valid_cell_position(cell_position) {
            return &[];
        }

        let linear_idx = self.get_linear_index(cell_position);
        // SAFETY: bounds are checked above, and offsets was constructed with 2 * n_cells elements
        unsafe {
            let start = *self.offsets.get_unchecked(linear_idx * 2);
            let end = *self.offsets.get_unchecked(linear_idx * 2 + 1);
            self.data.get_unchecked(start..end)
        }
    }
}

/// Mutable indexing into a :class:`CellGrid`.
///
/// Returns a mutable slice of all elements in the cell at the given position.
///
/// Notes
/// -----
/// Panics if the position is out of bounds.
/// Unlike the immutable ``Index``, mutable access requires valid bounds since returning
/// an empty mutable slice would be semantically incorrect.
impl<T: Clone + Default> IndexMut<[isize; 3]> for CellGrid<T> {
    #[inline(always)]
    fn index_mut(&mut self, cell_position: [isize; 3]) -> &mut Self::Output {
        if !self.is_valid_cell_position(cell_position) {
            panic!(
                "Cell position ({},{},{}) is out of bounds",
                cell_position[0], cell_position[1], cell_position[2],
            );
        }

        let linear_idx = self.get_linear_index(cell_position);
        // SAFETY: bounds are checked above
        unsafe {
            let start = *self.offsets.get_unchecked(linear_idx * 2);
            let end = *self.offsets.get_unchecked(linear_idx * 2 + 1);
            self.data.get_unchecked_mut(start..end)
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
#[pyclass(module = "biotite.structure")]
pub struct CellList {
    coord: Vec<[f32; 3]>,
    // A boolean mask that covers the selected atoms
    selection: Option<Vec<bool>>,
    cells: CellGrid<usize>,
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
    fn from_python_objects<'py>(
        py: Python<'py>,
        atom_array: Bound<'py, PyAny>,
        cell_size: f32,
        periodic: bool,
        r#box: Option<Bound<'py, PyAny>>,
        selection: Option<Vec<bool>>,
    ) -> PyResult<Self> {
        let struc = PyModule::import(py, "biotite.structure")?;
        let np = PyModule::import(py, "numpy")?;

        // Input validation
        if !atom_array.is_instance(&struc.getattr("AtomArray")?)?
            && !atom_array.is_instance(&np.getattr("ndarray")?)?
        {
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

            orig_length = coord_object.call_method0("__len__")?.extract()?;
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
            orig_length = coord_object.call_method0("__len__")?.extract()?;
        }
        let coord: Vec<[f32; 3]> = extract_coord(coord_object.extract::<PyReadonlyArray2<f32>>()?)?;
        Self::new(coord, cell_size, periodic_box, selection, orig_length)
    }

    /// Reconstruct a :class:`CellList` from its serialized state.
    ///
    /// This is used for unpickling. It reconstructs the :class:`CellList`
    /// from the internal state that was serialized by :meth:`__reduce__`.
    #[staticmethod]
    #[pyo3(name = "_from_state")]
    #[allow(clippy::too_many_arguments)]
    fn from_state(
        coord: Vec<[f32; 3]>,
        selection: Option<Vec<bool>>,
        cells_data: Vec<usize>,
        cells_offsets: Vec<usize>,
        cells_dimensions: [usize; 3],
        cell_size: f32,
        coord_range: [[f32; 3]; 2],
        periodic_box: Option<[[f32; 3]; 3]>,
        orig_length: usize,
    ) -> PyResult<Self> {
        let cells = CellGrid::new(cells_data, cells_offsets, cells_dimensions);

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

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        // Get the class from the module to ensure pickle can find it
        let struc = PyModule::import(py, "biotite.structure")?;
        let cls = struc.getattr("CellList")?;
        let from_state = cls.getattr("_from_state")?;

        // Build the args tuple
        let args = PyTuple::new(
            py,
            [
                self.coord.clone().into_pyobject(py)?.into_any().unbind(),
                self.selection
                    .clone()
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
                self.cells
                    .data
                    .clone()
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
                self.cells
                    .offsets
                    .clone()
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
                self.cells.dimensions.into_pyobject(py)?.into_any().unbind(),
                self.cell_size.into_pyobject(py)?.into_any().unbind(),
                self.coord_range.into_pyobject(py)?.into_any().unbind(),
                self.periodic_box.into_pyobject(py)?.into_any().unbind(),
                self.orig_length.into_pyobject(py)?.into_any().unbind(),
            ],
        )?;

        // Return (callable, args) tuple
        PyTuple::new(py, [from_state.unbind(), args.into_any().unbind()])
    }

    /// Create an adjacency matrix for the atoms in this cell list.
    ///
    /// An adjacency matrix depicts which atoms *i* and *j* have a distance
    /// lower than a given threshold distance.
    /// The values in the adjacency matrix ``m`` are
    /// ``m[i,j] = True if distance(i,j) <= threshold else False``.
    ///
    /// Parameters
    /// ----------
    /// threshold_distance : float
    ///     The threshold distance.
    ///     All atom pairs that have a distance lower than or equal to this value are
    ///     indicated by ``True`` values in the resulting matrix.
    ///
    /// Returns
    /// -------
    /// matrix : ndarray, dtype=bool, shape=(n,n)
    ///     An *n x n* adjacency matrix.
    ///     If a `selection` was given to the constructor of the
    ///     :class:`CellList`, the rows and columns corresponding to
    ///     atoms that are not masked by the selection have all
    ///     elements set to ``False``.
    ///
    /// Notes
    /// -----
    /// The highest performance is achieved when the `cell_size` is
    /// equal to the `threshold_distance`.
    /// However, this is purely optional: the resulting adjacency matrix is the same for
    /// every `cell_size`.
    ///
    /// Although the adjacency matrix should be symmetric in most cases,
    /// it may occur that ``m[i,j] != m[j,i]`` when ``distance(i,j)``
    /// is very close to the `threshold_distance` due to numerical
    /// errors. The matrix can be symmetrized with ``numpy.maximum(m, m.T)``.
    ///
    /// Examples
    /// --------
    /// Create adjacency matrix for CA atoms in a structure:
    ///
    /// >>> atom_array = atom_array[atom_array.atom_name == "CA"]
    /// >>> cell_list = CellList(atom_array, 5)
    /// >>> matrix = cell_list.create_adjacency_matrix(5)
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
                let mut sub_matrix: Array2<bool> = Array2::default((coord.len(), self.orig_length));
                for matrix_index in
                    self.get_atoms_from_slice(py, &coord, &Radius::Single(threshold_distance))?
                {
                    // Map index to potentially periodic copy to original atom index
                    sub_matrix[[matrix_index[0], matrix_index[1] % self.orig_length]] = true;
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
                for matrix_index in self.get_atoms_from_slice(
                    py,
                    &self.coord[..self.orig_length],
                    &Radius::Single(threshold_distance),
                )? {
                    // Map index to potentially periodic copy to original atom index
                    matrix[[matrix_index[0], matrix_index[1] % self.orig_length]] = true;
                }
                Ok(matrix.into_pyarray(py))
            }
        }
    }

    /// get_atoms(coord, radius, as_mask=False, result_format=CellList.Result.MAPPING)
    ///
    /// Find atoms with a maximum distance from given coordinates.
    ///
    /// Parameters
    /// ----------
    /// coord : ndarray, dtype=float, shape=(3,) or shape=(m,3)
    ///     One or more coordinates around which the atoms are searched.
    /// radius : float or ndarray, shape=(n,), dtype=float
    ///     The radius around `coord`, in which the atoms are searched,
    ///     i.e. all atoms within `radius` distance to `coord` are returned.
    ///     Either a single radius can be given as scalar, or individual
    ///     radii for each position in `coord` can be provided as
    ///     :class:`ndarray`.
    /// as_mask : bool, optional
    ///     **Deprecated:** Use ``result_format=CellList.Result.MASK`` instead.
    ///     If true, the result is returned as boolean mask instead
    ///     of an index array.
    /// result_format : CellList.Result, optional
    ///     The format of the result. See :class:`CellList.Result` for options.
    ///     Default is ``CellList.Result.MAPPING``.
    ///
    /// Returns
    /// -------
    /// result : ndarray
    ///     The result format depends on `result_format`.
    ///     See :class:`CellList.Result` for details.
    ///
    /// See Also
    /// --------
    /// get_atoms_in_cells
    ///
    /// Notes
    /// -----
    /// In case of a :class:`CellList` with `periodic` set to ``True``:
    /// If more than one periodic copy of an atom is within the
    /// threshold radius, the returned indices array may contain the
    /// corresponding index multiple times.
    /// Use ``numpy.unique()`` if this is undesirable.
    ///
    /// Examples
    /// --------
    /// Get adjacent atoms for a single position:
    ///
    /// >>> cell_list = CellList(atom_array, 3)
    /// >>> pos = np.array([1.0, 2.0, 3.0])
    /// >>> indices = cell_list.get_atoms(pos, radius=2.0)
    /// >>> print(indices)
    /// [102 104 112]
    #[pyo3(signature = (coord, radius, as_mask=false, result_format=CellListResult::MAPPING))]
    fn get_atoms<'py>(
        &self,
        py: Python<'py>,
        coord: &Bound<'py, PyAny>,
        radius: &Bound<'py, PyAny>,
        as_mask: bool,
        result_format: CellListResult,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (converted_coord, is_multi_coord) = self.prepare_coord_from_python(py, coord)?;
        let converted_radius = Self::prepare_radius_from_python::<f32>(radius)?;
        let pairs = self.get_atoms_from_slice(py, &converted_coord, &converted_radius)?;
        format_result(
            py,
            converted_coord.len(),
            self.orig_length,
            pairs,
            result_format,
            as_mask,
            is_multi_coord,
        )
    }

    /// get_atoms_in_cells(coord, cell_radius=1, as_mask=False, result_format=CellList.Result.MAPPING)
    ///
    /// Find atoms with a maximum cell distance from given coordinates.
    ///
    /// Instead of using the radius as maximum Euclidean distance to the
    /// given coordinates, the radius is measured as the number of cells:
    /// A radius of ``0`` means that only the atoms in the same cell
    /// as the given coordinates are considered. A radius of ``1`` means
    /// that the atom indices from this cell and the 26 surrounding
    /// cells are returned, and so forth.
    /// This is more efficient than :meth:`get_atoms`.
    ///
    /// Parameters
    /// ----------
    /// coord : ndarray, dtype=float, shape=(3,) or shape=(m,3)
    ///     One or more coordinates around which the atoms are searched.
    /// cell_radius : int or ndarray, shape=(n,), dtype=int, optional
    ///     The radius around `coord` (in number of cells), in which
    ///     the atoms are searched. This does not correspond to the
    ///     Euclidean distance used in :meth:`get_atoms`. In this case, all
    ///     atoms in the cell corresponding to `coord` and in adjacent
    ///     cells are returned.
    ///     Either a single radius can be given as scalar, or individual
    ///     radii for each position in `coord` can be provided as
    ///     :class:`ndarray`.
    ///     By default, atoms are searched in the cell of `coord`
    ///     and directly adjacent cells (``cell_radius=1``).
    /// as_mask : bool, optional
    ///     **Deprecated:** Use ``result_format=CellList.Result.MASK`` instead.
    ///     If true, the result is returned as boolean mask instead
    ///     of an index array.
    /// result_format : CellList.Result, optional
    ///     The format of the result. See :class:`CellList.Result` for options.
    ///     Default is ``CellList.Result.MAPPING``.
    ///
    /// Returns
    /// -------
    /// result : ndarray
    ///     The result format depends on `result_format`.
    ///     See :class:`CellList.Result` for details.
    ///
    /// See Also
    /// --------
    /// get_atoms
    ///
    /// Notes
    /// -----
    /// In case of a :class:`CellList` with `periodic` set to ``True``:
    /// If more than one periodic copy of an atom is within the
    /// threshold radius, the returned indices array may contain the
    /// corresponding index multiple times.
    /// Use ``numpy.unique()`` if this is undesirable.
    #[pyo3(signature = (coord, cell_radius=None, as_mask=false, result_format=CellListResult::MAPPING))]
    fn get_atoms_in_cells<'py>(
        &self,
        py: Python<'py>,
        coord: &Bound<'py, PyAny>,
        cell_radius: Option<&Bound<'py, PyAny>>,
        as_mask: bool,
        result_format: CellListResult,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (converted_coord, is_multi_coord) = self.prepare_coord_from_python(py, coord)?;

        let converted_radius = Self::prepare_radius_from_python::<i32>(
            cell_radius.unwrap_or(&PyInt::new(py, 1).into_any()),
        )?;
        let pairs = self.get_atoms_in_cells_from_slice(py, &converted_coord, &converted_radius)?;
        format_result(
            py,
            converted_coord.len(),
            self.orig_length,
            pairs,
            result_format,
            as_mask,
            is_multi_coord,
        )
    }
}

impl CellList {
    pub fn new(
        coord: Vec<[f32; 3]>,
        cell_size: f32,
        periodic_box: Option<[[f32; 3]; 3]>,
        selection: Option<Vec<bool>>,
        orig_length: usize,
    ) -> PyResult<Self> {
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
            elements.push(i);
        }
        // To get the number of cells in each dimension, use the position of the maximum coordinate
        let dimensions: [usize; 3] =
            Self::get_cell_position(coord_range[1], &coord_range, cell_size)
                .iter()
                .map(|x| (x + 1) as usize)
                .collect::<Vec<usize>>()
                .try_into()
                .unwrap();
        let cells = CellGrid::from_elements(dimensions, positions, elements);

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

    /// Find atoms within a Euclidean distance from given coordinates.
    ///
    /// This method first uses :meth:`get_atoms_in_cells_from_slice` to find candidate
    /// atoms based on cell adjacency, then filters them by actual Euclidean distance.
    ///
    /// Returns
    /// -------
    /// atoms:
    ///     A vector of *[query_idx, atom_idx]* pairs where ``atom_idx`` may refer to
    ///     periodic copies (``indices >= orig_length``). The caller is responsible for
    ///     mapping these back to original indices using ``atom_idx % orig_length``.
    pub fn get_atoms_from_slice(
        &self,
        py: Python<'_>,
        coord: &[[f32; 3]],
        radius: &Radius<f32>,
    ) -> PyResult<Vec<[usize; 2]>> {
        match radius {
            Radius::Single(r) => {
                let sq_r = r.powi(2);
                let pairs = self
                    .get_atoms_in_cells_from_slice(
                        py,
                        coord,
                        &Radius::Single((r / self.cell_size).ceil() as i32),
                    )?
                    .iter()
                    .filter(|pair| distance_squared(coord[pair[0]], self.coord[pair[1]]) <= sq_r)
                    .copied()
                    .collect();
                Ok(pairs)
            }
            Radius::Multiple(rs) => {
                let cell_radii = rs
                    .iter()
                    .map(|r| (*r / self.cell_size).ceil() as i32)
                    .collect();
                let sq_rs: Vec<f32> = rs.iter().map(|r| r.powi(2)).collect();
                let pairs = self
                    .get_atoms_in_cells_from_slice(py, coord, &Radius::Multiple(cell_radii))?
                    .iter()
                    .filter(|pair| {
                        distance_squared(coord[pair[0]], self.coord[pair[1]]) <= sq_rs[pair[0]]
                    })
                    .copied()
                    .collect();
                Ok(pairs)
            }
        }
    }

    /// Find atoms in cells adjacent to given coordinates.
    ///
    /// For each input coordinate, this method identifies the corresponding cell
    /// and iterates over all cells within `cell_radius` distance (in cells, not
    /// Euclidean distance). All atom indices in those cells are collected.
    ///
    /// Parameters
    /// ----------
    /// coord
    ///     The query coordinates.
    /// cell_radii
    ///     Either a single cell radius for all coordinates, or
    ///     individual radii for each coordinate.
    ///
    /// Returns
    /// -------
    /// atoms:
    ///     A vector of *[query_idx, atom_idx]* pairs. The ``atom_idx`` values refer to
    ///     the full coordinate array, which may include periodic copies when
    ///     ``periodic=True``. These indices can exceed ``orig_length`` and should be
    ///     mapped back using ``atom_idx % orig_length`` when needed.
    pub fn get_atoms_in_cells_from_slice(
        &self,
        py: Python<'_>,
        coord: &[[f32; 3]],
        cell_radii: &Radius<i32>,
    ) -> PyResult<Vec<[usize; 2]>> {
        if let Radius::Multiple(ref rs) = cell_radii {
            if rs.len() != coord.len() {
                return Err(exceptions::PyValueError::new_err(format!(
                    "{} coordinates were provided, but {} radii",
                    coord.len(),
                    rs.len()
                )));
            }
        }

        let mut adjacent_atoms = Vec::new();
        for (coord_idx, c) in coord.iter().enumerate() {
            // Skip non-finite coordinates
            if !c[0].is_finite() || !c[1].is_finite() || !c[2].is_finite() {
                continue;
            }

            let cell_pos = Self::get_cell_position(*c, &self.coord_range, self.cell_size);

            // Iterate over all adjacent cells within the cell_radius
            let cell_radius = match cell_radii {
                Radius::Single(r) => *r as isize,
                Radius::Multiple(ref rs) => rs[coord_idx] as isize,
            };
            for adj_i in (cell_pos[0] - cell_radius)..=(cell_pos[0] + cell_radius) {
                for adj_j in (cell_pos[1] - cell_radius)..=(cell_pos[1] + cell_radius) {
                    for adj_k in (cell_pos[2] - cell_radius)..=(cell_pos[2] + cell_radius) {
                        // Get all atoms in this cell (returns empty slice if out of bounds)
                        let cell_atoms = &self.cells[[adj_i, adj_j, adj_k]];
                        adjacent_atoms.reserve(cell_atoms.len());
                        for &atom_idx in cell_atoms {
                            adjacent_atoms.push([coord_idx, atom_idx]);
                        }
                    }
                }
            }
            check_signals_periodically(py, coord_idx)?;
        }
        Ok(adjacent_atoms)
    }

    /// Convert a 3D coordinate to the corresponding cell position in the grid.
    ///
    /// Given a coordinate *[x, y, z]*, this computes the cell indices *[i, j, k]*
    /// based on the coordinate range and cell size.
    ///
    /// Returns
    /// -------
    /// The cell position *[i, j, k]*. Note that negative indices or indices
    /// exceeding the grid dimensions can be returned if the coordinate is
    /// outside the original coordinate range. The :class:`CellGrid` handles
    /// out-of-bounds access gracefully by returning empty slices.
    #[inline(always)]
    fn get_cell_position(
        coord: [f32; 3],
        coord_range: &[[f32; 3]; 2],
        cellsize: f32,
    ) -> [isize; 3] {
        [
            // Conversion to 'isize' automatically floors the result
            ((coord[0] - coord_range[0][0]) / cellsize) as isize,
            ((coord[1] - coord_range[0][1]) / cellsize) as isize,
            ((coord[2] - coord_range[0][2]) / cellsize) as isize,
        ]
    }

    /// Convert Python coordinate input to a uniform Rust representation.
    ///
    /// This method handles the various input formats accepted by the Python API:
    ///
    /// - Single coordinate: ``shape=(3,)`` -> converted to ``Vec`` with one element
    /// - Multiple coordinates: ``shape=(n, 3)`` -> converted to ``Vec`` with n elements
    ///
    /// It also handles:
    ///
    /// - Data type conversion to ``float32`` if necessary
    /// - Periodic boundary conditions (moving coordinates inside the box)
    ///
    /// Returns
    /// -------
    /// coordinates:
    ///     A vector of converted coordinates.
    /// is_multi_coord:
    ///     A boolean indicating whether the input was a 2D array (multiple coordinates)
    ///     or a 1D array (single coordinate).
    fn prepare_coord_from_python<'py>(
        &self,
        py: Python<'py>,
        coord: &Bound<'py, PyAny>,
    ) -> PyResult<(Vec<[f32; 3]>, bool)> {
        // Slow path: need to convert dtype or handle periodicity
        let struc = PyModule::import(py, "biotite.structure")?;

        // Convert into expected f32 array
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", "float32")?;
        kwargs.set_item("copy", false)?;
        let coord = coord.call_method("astype", (), Some(&kwargs))?;

        // Handle periodicity if needed
        let coord = if let Some(box_matrix) = self.periodic_box {
            let box_array = Array2::from_shape_vec((3, 3), box_matrix.concat())
                .map_err(|_| exceptions::PyRuntimeError::new_err("Failed to create box array"))?;
            struc
                .getattr("move_inside_box")?
                .call1((coord, box_array.into_pyarray(py)))?
        } else {
            coord.clone()
        };

        // Consistently use 2 dimensions
        // -> if only one coordinate is given, convert it to a 2D array
        if let Ok(single_coord) = coord.extract::<[f32; 3]>() {
            if !single_coord.iter().all(|c| c.is_finite()) {
                return Err(exceptions::PyValueError::new_err(
                    "Coordinates contain non-finite values",
                ));
            }
            return Ok((vec![single_coord], false));
        }
        if let Ok(array) = coord.extract::<PyReadonlyArray2<f32>>() {
            return Ok((extract_coord(array)?, true));
        }
        Err(exceptions::PyTypeError::new_err(
            "Coordinates must be a single coordinate or a 2D array of coordinates",
        ))
    }

    /// Convert Python radius input to the internal :class:`Radius` enum.
    ///
    /// This method handles two input formats:
    ///
    /// - Single radius: a scalar value -> ``Radius::Single(value)``
    /// - Multiple radii: an array of values -> ``Radius::Multiple(vec)``
    ///
    /// The generic type ``T`` allows this to work for both ``f32`` (Euclidean radius)
    /// and ``i32`` (cell radius).
    fn prepare_radius_from_python<'py, T>(radius: &Bound<'py, PyAny>) -> PyResult<Radius<T>>
    where
        T: for<'a> FromPyObject<'a, 'py>,
        Vec<T>: for<'a> FromPyObject<'a, 'py>,
    {
        if let Ok(r) = radius.extract::<T>() {
            return Ok(Radius::Single(r));
        }
        if let Ok(rs) = radius.extract::<Vec<T>>() {
            return Ok(Radius::Multiple(rs));
        }
        Err(exceptions::PyTypeError::new_err(
            "Radius must be a single value or an array of values",
        ))
    }
}

/// Calculate the bounding box (min and max coordinates) from a coordinate list.
///
/// This is used to determine the spatial extent of the atoms, which in turn
/// determines the origin of the cell grid and the number of cells needed.
///
/// Returns
/// -------
/// coord_range:
///     A vector containing the minimum and maximum coordinates.
///
/// Raises
/// ------
/// PyValueError
///     If the coordinate list is empty.
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

/// Convert an ``[isize; 3]`` to ``[usize; 3]`` if all elements are non-negative.
///
/// Returns ``None`` if any element is negative.
#[inline(always)]
fn as_usize(x: [isize; 3]) -> Option<[usize; 3]> {
    for e in &x {
        if *e < 0 {
            return None;
        }
    }
    Some([x[0] as usize, x[1] as usize, x[2] as usize])
}

/// Format query results as a mapping array (the default format).
///
/// Creates a 2D array where each row corresponds to a query coordinate,
/// and the columns contain the indices of neighboring atoms. Since different
/// queries may have different numbers of neighbors, the array is padded with
/// ``-1`` values to indicate empty slots.
///
/// Parameters
/// ----------
/// n_query
///     Number of query coordinates.
/// n_atoms
///     Number of atoms in the original (non-periodic) structure,
///     used for mapping periodic copy indices back to original indices.
/// pairs
///     Vector of *[query_idx, atom_idx]* pairs.
/// is_multi_coord
///     If ``False``, return a 1D array for the single query.
fn format_as_mapping(
    py: Python<'_>,
    n_query: usize,
    n_atoms: usize,
    pairs: Vec<[usize; 2]>,
    is_multi_coord: bool,
) -> PyResult<Bound<'_, PyAny>> {
    // First pass: count elements per query to find max length
    let mut counts: Vec<usize> = vec![0; n_query];
    for pair in pairs.iter() {
        counts[pair[0]] += 1;
    }
    let max_length = *counts.iter().max().unwrap_or(&0);

    // Allocate final matrix directly
    let mut mapping_matrix = Array2::from_elem((n_query, max_length), -1i64);

    // Reset counts to use as current position tracker
    counts.fill(0);

    // Second pass: fill matrix directly
    for pair in pairs.iter() {
        let row = pair[0];
        let col = counts[row];
        // Map index to potentially periodic copy to original atom index
        mapping_matrix[[row, col]] = (pair[1] % n_atoms) as i64;
        counts[row] += 1;
    }

    if is_multi_coord {
        Ok(mapping_matrix.into_pyarray(py).into_any())
    } else {
        Ok(mapping_matrix
            .row(0)
            .into_owned()
            .into_pyarray(py)
            .into_any())
    }
}

/// Format query results as a boolean mask.
///
/// Creates a 2D boolean array of shape *(n_query, n_atoms)* where
/// ``mask[i, j] = True`` indicates that atom *j* is a neighbor of query *i*.
///
/// Parameters
/// ----------
/// n_query
///     Number of query coordinates.
/// n_atoms
///     Number of atoms in the original (non-periodic) structure.
/// pairs
///     Vector of *[query_idx, atom_idx]* pairs.
/// is_multi_coord
///     If ``False``, return a 1D boolean array for the single query.
fn format_as_mask(
    py: Python<'_>,
    n_query: usize,
    n_atoms: usize,
    pairs: Vec<[usize; 2]>,
    is_multi_coord: bool,
) -> PyResult<Bound<'_, PyAny>> {
    let mut mask = Array2::from_elem((n_query, n_atoms), false);
    for pair in pairs.iter() {
        // Map index to potentially periodic copy to original atom index
        mask[[pair[0], pair[1] % n_atoms]] = true;
    }

    if is_multi_coord {
        Ok(mask.into_pyarray(py).into_any())
    } else {
        Ok(mask.row(0).into_owned().into_pyarray(py).into_any())
    }
}

/// Format query results as an array of index pairs.
///
/// Creates a 2D array of shape *(p, 2)* where each row contains
/// *[query_idx, atom_idx]* for a neighboring pair.
///
/// Parameters
/// ----------
/// n_atoms
///     Number of atoms in the original (non-periodic) structure,
///     used for mapping periodic copy indices back to original indices.
/// pairs
///     Vector of *[query_idx, atom_idx]* pairs.
/// is_multi_coord
///     If ``False``, return just the atom indices as a 1D array.
fn format_as_pairs(
    py: Python<'_>,
    n_atoms: usize,
    pairs: Vec<[usize; 2]>,
    is_multi_coord: bool,
) -> PyResult<Bound<'_, PyAny>> {
    let n_pairs = pairs.len();

    if is_multi_coord {
        // Create a flat vector of i64 directly
        let mut flat_data: Vec<i64> = Vec::with_capacity(n_pairs * 2);
        for pair in pairs.iter() {
            flat_data.push(pair[0] as i64);
            // Map index to potentially periodic copy to original atom index
            flat_data.push((pair[1] % n_atoms) as i64);
        }
        let pairs_array = Array2::from_shape_vec((n_pairs, 2), flat_data).expect("Shape mismatch");
        Ok(pairs_array.into_pyarray(py).into_any())
    } else {
        // For single coord, return just the second column (atom indices)
        let atoms: Vec<i64> = pairs
            .iter()
            .map(|pair| (pair[1] % n_atoms) as i64)
            .collect();
        Ok(atoms.into_pyarray(py).into_any())
    }
}

/// Format query results according to the requested output format.
///
/// This is the main dispatch function that routes to the appropriate
/// formatting function based on the `result_format` parameter.
///
/// Parameters
/// ----------
/// n_query
///     Number of query coordinates.
/// n_atoms
///     Number of atoms in the original (non-periodic) structure.
/// pairs
///     Vector of *[query_idx, atom_idx]* pairs.
/// result_format
///     The desired output format.
/// as_mask
///     If ``True``, overrides `result_format` to ``MASK``
///     and emits a deprecation warning.
/// is_multi_coord
///     Whether multiple query coordinates were provided.
fn format_result(
    py: Python<'_>,
    n_query: usize,
    n_atoms: usize,
    pairs: Vec<[usize; 2]>,
    mut result_format: CellListResult,
    as_mask: bool,
    is_multi_coord: bool,
) -> PyResult<Bound<'_, PyAny>> {
    if as_mask {
        // Raise DeprecationWarning when `as_mask`` is used
        let warnings = py.import("warnings")?;
        let _ = warnings.call_method1(
            "warn",
            (
                "The 'as_mask' parameter is deprecated, use 'result_format' instead.",
                py.import("builtins")?.getattr("DeprecationWarning")?,
            ),
        );
        result_format = CellListResult::MASK;
    }
    match result_format {
        CellListResult::MAPPING => format_as_mapping(py, n_query, n_atoms, pairs, is_multi_coord),
        CellListResult::MASK => format_as_mask(py, n_query, n_atoms, pairs, is_multi_coord),
        CellListResult::PAIRS => format_as_pairs(py, n_atoms, pairs, is_multi_coord),
    }
}
