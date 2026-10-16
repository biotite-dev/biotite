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
    if n < 3 {
        return Err(PyValueError::new_err("At least 3 nodes are required"));
    }

    // The node ID at each position of the distance matrix
    // Initially these are the leaf nodes, after clustering a position
    // refers to the created intermediate node
    let mut node_ids: Vec<u32> = (0..n as u32).collect();
    let mut next_node_id = n as u32;
    // The indices in the distance matrix that are not clustered yet,
    // in ascending order
    // Iterating over this compact list instead of skipping clustered
    // indices keeps the hot loops free of branches
    let mut active: Vec<usize> = (0..n).collect();
    // The divergence of a 'taxon' describes the relative evolution rate
    // It is the sum of distances to all other unclustered nodes
    // It is computed once and afterwards updated incrementally
    let mut divergence = vec![0.0f32; n];
    for &i in &active {
        let row = &distances[i * n..(i + 1) * n];
        divergence[i] = active.iter().map(|&k| row[k]).sum();
    }
    let mut edges = TreeEdges::with_capacity(2 * n);

    // Cluster until three nodes remain, which are joined into the root
    // Note that at this point the corrected distances of all three
    // pairs are mathematically equal, so a minimum search would decide
    // by rounding noise, which of the nodes becomes the root
    for iteration in 0.. {
        let n_rem_nodes = active.len();
        if n_rem_nodes == 3 {
            break;
        }
        check_signals_periodically(py, iteration)?;

        // Find minimum divergence corrected distance
        // The corrected distances are computed on the fly,
        // as they are not needed afterwards
        let factor = (n_rem_nodes - 2) as f32;
        let mut dist_min = f32::MAX;
        let mut i_min = 0;
        let mut j_min = 0;
        for (pos_i, &i) in active.iter().enumerate() {
            let row = &distances[i * n..(i + 1) * n];
            let divergence_i = divergence[i];
            for &j in &active[..pos_i] {
                let dist = factor * row[j] - divergence_i - divergence[j];
                if dist < dist_min {
                    dist_min = dist;
                    i_min = i;
                    j_min = j;
                }
            }
        }

        // Cluster the nodes with minimum distance into a new node
        // replacing the node at position `i_min`
        // and removing the node at position `j_min`
        let dist_ij = distances[i_min * n + j_min];
        let divergence_diff = divergence[i_min] - divergence[j_min];
        let inv_rem = 1.0 / (n_rem_nodes - 2) as f32;
        let node_dist_i = 0.5 * (dist_ij + inv_rem * divergence_diff);
        let node_dist_j = 0.5 * (dist_ij - inv_rem * divergence_diff);
        edges.push(next_node_id, node_ids[i_min], node_dist_i);
        edges.push(next_node_id, node_ids[j_min], node_dist_j);
        node_ids[i_min] = next_node_id;
        next_node_id += 1;
        let pos_j = active.binary_search(&j_min).unwrap();
        active.remove(pos_j);

        // Update distance matrix
        // Calculate distances of the new node to all other nodes
        // and update the divergences accordingly:
        // The distances to the two merged nodes are replaced by the
        // distance to the new node
        let mut divergence_new = 0.0f32;
        for &k in &active {
            if k != i_min {
                let dist_ik = distances[i_min * n + k];
                let dist_jk = distances[j_min * n + k];
                let dist = 0.5 * (dist_ik + dist_jk - dist_ij);
                distances[i_min * n + k] = dist;
                distances[k * n + i_min] = dist;
                divergence[k] += dist - dist_ik - dist_jk;
                divergence_new += dist;
            }
        }
        divergence[i_min] = divergence_new;
    }

    // Combine the last three nodes into the root node
    // The distance of each node to the root follows from the three-point
    // condition
    let (i, j, k) = (active[0], active[1], active[2]);
    let dist_ij = distances[i * n + j];
    let dist_ik = distances[i * n + k];
    let dist_jk = distances[j * n + k];
    edges.push(
        next_node_id,
        node_ids[i],
        0.5 * (dist_ij + dist_ik - dist_jk),
    );
    edges.push(
        next_node_id,
        node_ids[j],
        0.5 * (dist_ij + dist_jk - dist_ik),
    );
    edges.push(
        next_node_id,
        node_ids[k],
        0.5 * (dist_ik + dist_jk - dist_ij),
    );

    Ok((
        edges.parents.into_pyarray(py),
        edges.children.into_pyarray(py),
        edges.distances.into_pyarray(py),
    ))
}
