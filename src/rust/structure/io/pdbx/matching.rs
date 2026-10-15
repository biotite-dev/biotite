use crate::util::check_signals_periodically;
use numpy::{IntoPyArray, PyArray1, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions;
use pyo3::prelude::*;
use smallvec::SmallVec;
use std::collections::HashMap;

// Label as a separate module to indicate that this exception comes
// from biotite
mod biotite {
    pyo3::import_exception!(biotite, InvalidFileError);
}

/// A view on the raw bytes of a 1-D NumPy array.
///
/// As the matching only requires equality checks, the values are never interpreted:
/// each element is simply the `itemsize` bytes it occupies.
/// Hence any dtype with a fixed item size is supported, including fixed-size strings.
struct ByteColumn {
    data: *const u8,
    stride: isize,
    itemsize: usize,
    length: usize,
}

impl ByteColumn {
    /// Get the bytes of the element at the given index.
    ///
    /// Safety
    /// ------
    /// The caller must ensure that `index < self.length` and that the underlying array
    /// is still alive.
    #[inline]
    unsafe fn element(&self, index: usize) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.offset(self.stride * index as isize),
                self.itemsize,
            )
        }
    }
}

/// Create a byte view on the given 1-D NumPy array.
fn as_byte_column(array: &Bound<'_, PyUntypedArray>) -> PyResult<ByteColumn> {
    if array.ndim() != 1 {
        return Err(exceptions::PyValueError::new_err(
            "Only 1-D arrays are supported",
        ));
    }
    let dtype = array.dtype();
    if dtype.has_object() {
        // Elements would be pointers to Python objects, i.e. equal values may have
        // different bytes
        return Err(exceptions::PyValueError::new_err(
            "Arrays with 'object' dtype are not supported",
        ));
    }
    Ok(ByteColumn {
        data: unsafe { (*array.as_array_ptr()).data } as *const u8,
        stride: array.strides()[0],
        itemsize: dtype.itemsize(),
        length: array.len(),
    })
}

/// Write the concatenated bytes of the given row into `key`.
#[inline]
fn write_key(columns: &[ByteColumn], index: usize, key: &mut [u8]) {
    let mut offset = 0;
    for column in columns {
        let itemsize = column.itemsize;
        key[offset..offset + itemsize].copy_from_slice(unsafe { column.element(index) });
        offset += itemsize;
    }
}

/// Check that all given arrays have the same length and return it.
fn common_length(columns: &[ByteColumn]) -> PyResult<usize> {
    let length = columns[0].length;
    if columns.iter().any(|column| column.length != length) {
        return Err(exceptions::PyValueError::new_err(
            "All arrays must have the same length",
        ));
    }
    Ok(length)
}

/// For each index in the `query_arrays` find the index in the `reference_arrays`, where
/// all query values match the reference counterpart.
///
/// If no match is found for a query, the corresponding index is ``-1``.
///
/// Parameters
/// ----------
/// query_arrays, reference_arrays : list of ndarray
///     The columns to match.
///     Corresponding columns must have equivalent dtypes.
///     Any dtype with a fixed item size is supported, as values are compared bytewise.
///
/// Returns
/// -------
/// match_indices : ndarray, dtype=int64
///     For each query the index of the matching reference or ``-1``.
///
/// Raises
/// ------
/// InvalidFileError
///     If a query matches more than one reference, i.e. the corresponding bond in
///     ``struct_conn`` cannot be unambiguously assigned to an atom in ``atom_site``.
///
/// Notes
/// -----
/// This function uses a query-based hash map approach to find matches to the reference
/// arrays.
/// The values of a row are concatenated into a single key comprising the raw bytes of
/// each column.
/// As only equality is required, the values themselves are never interpreted, which
/// makes the matching agnostic to the dtype of the given arrays.
#[pyfunction]
pub fn find_matches<'py>(
    py: Python<'py>,
    query_arrays: Vec<Bound<'py, PyUntypedArray>>,
    reference_arrays: Vec<Bound<'py, PyUntypedArray>>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if query_arrays.is_empty() || query_arrays.len() != reference_arrays.len() {
        return Err(exceptions::PyValueError::new_err(
            "The same non-zero number of query and reference arrays is required",
        ));
    }
    for (query, reference) in query_arrays.iter().zip(reference_arrays.iter()) {
        if !query.dtype().is_equiv_to(&reference.dtype()) {
            return Err(exceptions::PyValueError::new_err(format!(
                "Query dtype '{}' does not match reference dtype '{}'",
                query.dtype(),
                reference.dtype()
            )));
        }
    }

    let query_columns = query_arrays
        .iter()
        .map(as_byte_column)
        .collect::<PyResult<Vec<_>>>()?;
    let reference_columns = reference_arrays
        .iter()
        .map(as_byte_column)
        .collect::<PyResult<Vec<_>>>()?;
    let query_count = common_length(&query_columns)?;
    let reference_count = common_length(&reference_columns)?;
    let key_size: usize = query_columns.iter().map(|column| column.itemsize).sum();

    // Map the comparatively few queries instead of the references
    // -> this keeps the map small
    let mut query_keys = vec![0u8; query_count * key_size];
    for index in 0..query_count {
        write_key(
            &query_columns,
            index,
            &mut query_keys[index * key_size..(index + 1) * key_size],
        );
    }
    // Multiple queries may point to the same atom, hence a key can have multiple
    // indices, but still the most common case is a single index per key
    let mut query_indices: HashMap<&[u8], SmallVec<[usize; 1]>> =
        HashMap::with_capacity(query_count);
    for index in 0..query_count {
        query_indices
            .entry(&query_keys[index * key_size..(index + 1) * key_size])
            .or_default()
            .push(index);
    }

    // -1 indicates that no match was found in the reference
    let mut match_indices = vec![-1i64; query_count];
    let mut key = vec![0u8; key_size];
    for reference_index in 0..reference_count {
        check_signals_periodically(py, reference_index)?;
        write_key(&reference_columns, reference_index, &mut key);
        let Some(indices) = query_indices.get(key.as_slice()) else {
            // No query matches this position in the reference
            continue;
        };
        for &query_index in indices {
            if match_indices[query_index] == -1 {
                match_indices[query_index] = reference_index as i64;
            } else {
                // A second match means the query cannot be uniquely assigned
                return Err(biotite::InvalidFileError::new_err(format!(
                    "The covalent bond in the 'struct_conn' category at index \
                    {query_index} cannot be unambiguously assigned to atoms in \
                    the 'atom_site' category"
                )));
            }
        }
    }

    Ok(match_indices.into_pyarray(py))
}
