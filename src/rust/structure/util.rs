use numpy::PyReadonlyArray2;
use pyo3::exceptions;
use pyo3::prelude::*;

/// Extract coordinates from a 2D NumPy array into a ``Vec<[f32; 3]>``.
///
/// Raises
/// ------
/// PyValueError
///     If the array does not have shape *(n, 3)* or contains non-finite values (NaN or Inf).
pub fn extract_coord(coord_array: PyReadonlyArray2<f32>) -> PyResult<Vec<[f32; 3]>> {
    let coord_ndarray = coord_array.as_array();
    let shape = coord_ndarray.shape();
    if shape.len() != 2 {
        return Err(exceptions::PyValueError::new_err(
            "Coordinates must have shape (n,3)",
        ));
    }
    if shape[1] != 3 {
        return Err(exceptions::PyValueError::new_err(
            "Coordinates must have form (x,y,z)",
        ));
    }
    if !coord_ndarray.is_all_infinite() {
        return Err(exceptions::PyValueError::new_err(
            "Coordinates contain non-finite values",
        ));
    }

    // Try to access as contiguous slice for maximum efficiency
    if let Some(slice) = coord_ndarray.as_slice() {
        // Data is contiguous in memory, we can reinterpret directly
        let n_coords = shape[0];
        let mut result: Vec<[f32; 3]> = Vec::with_capacity(n_coords);
        for i in 0..n_coords {
            let base = i * 3;
            result.push([slice[base], slice[base + 1], slice[base + 2]]);
        }
        Ok(result)
    } else {
        // Fallback for non-contiguous arrays (e.g., transposed or sliced)
        let mut result: Vec<[f32; 3]> = Vec::with_capacity(shape[0]);
        for i in 0..shape[0] {
            result.push([
                coord_ndarray[[i, 0]],
                coord_ndarray[[i, 1]],
                coord_ndarray[[i, 2]],
            ]);
        }
        Ok(result)
    }
}

/// Compute the squared Euclidean distance between two 3D points.
///
/// Using squared distance avoids the expensive square root operation,
/// which is sufficient for distance comparisons.
#[inline(always)]
pub fn distance_squared(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}
