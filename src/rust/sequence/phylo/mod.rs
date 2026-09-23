use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub mod nj;
pub mod upgma;

/// The edges of a tree as returned to Python:
/// The parent and child node of each edge and the distance between them.
pub type EdgeArrays<'py> = (
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<f32>>,
);

/// Edges of a rooted tree in insertion order, as returned to Python.
///
/// Each edge points from the parent to the child node and carries the distance
/// between them.
/// The Python side converts this into a :class:`networkx.DiGraph`.
pub struct TreeEdges {
    pub parents: Vec<u32>,
    pub children: Vec<u32>,
    pub distances: Vec<f32>,
}

impl TreeEdges {
    pub fn with_capacity(capacity: usize) -> Self {
        TreeEdges {
            parents: Vec::with_capacity(capacity),
            children: Vec::with_capacity(capacity),
            distances: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, parent: u32, child: u32, distance: f32) {
        self.parents.push(parent);
        self.children.push(child);
        self.distances.push(distance);
    }
}

/// Check that the given distance matrix is square, symmetric and contains only
/// finite, non-negative values and return the number of nodes and a mutable
/// flat (row-major) copy of the matrix.
///
/// The symmetry check uses the same tolerances as :func:`numpy.allclose()`.
pub fn check_distance_matrix(distances: &PyReadonlyArray2<f32>) -> PyResult<(usize, Vec<f32>)> {
    let distances = distances.as_array();
    let n = distances.nrows();
    if distances.ncols() != n {
        return Err(PyValueError::new_err("Distance matrix must be symmetric"));
    }
    if n == 0 {
        return Err(PyValueError::new_err("Distance matrix is empty"));
    }
    let mut flat = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let dist = distances[[i, j]];
            let dist_t = distances[[j, i]];
            if dist.is_nan() {
                return Err(PyValueError::new_err("Distance matrix contains NaN values"));
            }
            if dist >= f32::MAX {
                return Err(PyValueError::new_err("Distance matrix contains infinity"));
            }
            if dist < 0.0 {
                return Err(PyValueError::new_err("Distances must be positive"));
            }
            if (dist - dist_t).abs() > 1e-8 + 1e-5 * dist_t.abs() {
                return Err(PyValueError::new_err("Distance matrix must be symmetric"));
            }
            flat.push(dist);
        }
    }
    Ok((n, flat))
}

pub fn module<'py>(parent_module: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyModule>> {
    let module = PyModule::new(parent_module.py(), "phylo")?;
    module.add_function(wrap_pyfunction!(upgma::upgma, &module)?)?;
    module.add_function(wrap_pyfunction!(nj::neighbor_joining, &module)?)?;
    Ok(module)
}
