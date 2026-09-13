use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::sequence::phylo::{check_distance_matrix, EdgeArrays, TreeEdges};
use crate::util::check_signals_periodically;

/// neighbor_joining(distances)
///
/// Perform hierarchical clustering using the *neighbor joining* algorithm.
///
/// Parameters
/// ----------
/// distances : ndarray, dtype=float32, shape=(n, n)
///     Pairwise distance matrix.
///
/// Returns
/// -------
/// parents, children : ndarray, dtype=uint32, shape=(2n-3,)
///     The parent and child node of each edge in the resulting tree.
///     The leaf nodes are ``0`` to ``n-1``, i.e. the indices of `distances`.
///     Intermediate nodes continue the numbering in the order of their
///     creation, hence the root is the node ``2n-3``.
///     The root has three children, all other intermediate nodes have two.
/// distances : ndarray, dtype=float32, shape=(2n-3,)
///     The distance between the parent and child node of each edge.
#[pyfunction]
pub fn neighbor_joining<'py>(
    py: Python<'py>,
    distances: PyReadonlyArray2<'py, f32>,
) -> PyResult<EdgeArrays<'py>> {
    let (n, mut distances) = check_distance_matrix(&distances)?;
    if n < 4 {
        return Err(PyValueError::new_err("At least 4 nodes are required"));
    }

    // The node ID at each position of the distance matrix
    // Initially these are the leaf nodes, after clustering a position
    // refers to the created intermediate node
    let mut node_ids: Vec<u32> = (0..n as u32).collect();
    let mut next_node_id = n as u32;
    // Indicates whether an index in the distance matrix has already been
    // clustered and the respective rows and columns can be ignored
    let mut is_clustered = vec![false; n];
    let mut n_rem_nodes = n;
    // The divergence of a 'taxon' describes the relative evolution rate
    let mut divergence = vec![0.0f32; n];
    // Lower triangle matrix storing the divergence corrected distances
    let mut corr_distances = vec![0.0f32; n * n];
    let mut edges = TreeEdges::with_capacity(2 * n);

    for iteration in 0.. {
        check_signals_periodically(py, iteration)?;

        // Calculate divergence
        for i in 0..n {
            if is_clustered[i] {
                continue;
            }
            let mut dist_sum = 0.0f32;
            for k in 0..n {
                if is_clustered[k] {
                    continue;
                }
                dist_sum += distances[i * n + k];
            }
            divergence[i] = dist_sum;
        }

        // Calculate corrected distance matrix
        for i in 0..n {
            if is_clustered[i] {
                continue;
            }
            for j in 0..i {
                if is_clustered[j] {
                    continue;
                }
                corr_distances[i * n + j] =
                    (n_rem_nodes - 2) as f32 * distances[i * n + j] - divergence[i] - divergence[j];
            }
        }

        // Find minimum corrected distance
        let mut dist_min = f32::MAX;
        let mut i_min = usize::MAX;
        let mut j_min = usize::MAX;
        for i in 0..n {
            if is_clustered[i] {
                continue;
            }
            for j in 0..i {
                if is_clustered[j] {
                    continue;
                }
                let dist = corr_distances[i * n + j];
                if dist < dist_min {
                    dist_min = dist;
                    i_min = i;
                    j_min = j;
                }
            }
        }
        if i_min == usize::MAX {
            // Cannot happen, as the loop returns when three nodes remain
            break;
        }

        // Cluster the nodes with minimum distance
        // replacing the node at position `i_min`
        // and leaving the node at position `j_min` empty
        // The evaluation order of the expressions is chosen to
        // reproduce the single precision arithmetic of the original
        // implementation
        let dist_ij = distances[i_min * n + j_min];
        let divergence_diff = divergence[i_min] - divergence[j_min];
        let inv_rem = 1.0 / (n_rem_nodes - 2) as f64;
        let node_dist_i = (0.5 * (dist_ij as f64 + inv_rem * divergence_diff as f64)) as f32;
        let node_dist_j = (0.5 * (dist_ij as f64 + inv_rem * -divergence_diff as f64)) as f32;
        if n_rem_nodes > 3 {
            // Clustering is not finished
            // -> Create a node with two children
            edges.push(next_node_id, node_ids[i_min], node_dist_i);
            edges.push(next_node_id, node_ids[j_min], node_dist_j);
            node_ids[i_min] = next_node_id;
            next_node_id += 1;
            is_clustered[j_min] = true;
        } else {
            // Clustering is finished
            // -> Combine the last three nodes into the root node
            is_clustered[i_min] = true;
            is_clustered[j_min] = true;
            // The index of the remaining one (other than i_min and j_min)
            let k = is_clustered.iter().position(|&c| !c).unwrap();
            let node_dist_k = 0.5 * (distances[i_min * n + k] + distances[j_min * n + k] - dist_ij);
            edges.push(next_node_id, node_ids[i_min], node_dist_i);
            edges.push(next_node_id, node_ids[j_min], node_dist_j);
            edges.push(next_node_id, node_ids[k], node_dist_k);
            break;
        }

        // Update distance matrix
        // Calculate distances of the new node to all other nodes
        for k in 0..n {
            if !is_clustered[k] && k != i_min {
                let dist = 0.5 * (distances[i_min * n + k] + distances[j_min * n + k] - dist_ij);
                distances[i_min * n + k] = dist;
                distances[k * n + i_min] = dist;
            }
        }
        n_rem_nodes -= 1;
    }

    Ok((
        edges.parents.into_pyarray(py),
        edges.children.into_pyarray(py),
        edges.distances.into_pyarray(py),
    ))
}
