use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::sequence::phylo::{check_distance_matrix, EdgeArrays, TreeEdges};
use crate::util::check_signals_periodically;

/// upgma(distances)
///
/// Perform hierarchical clustering using the *unweighted pair group method
/// with arithmetic mean* (UPGMA).
///
/// Parameters
/// ----------
/// distances : ndarray, dtype=float32, shape=(n, n)
///     Pairwise distance matrix.
///
/// Returns
/// -------
/// parents, children : ndarray, dtype=uint32, shape=(2n-2,)
///     The parent and child node of each edge in the resulting tree.
///     The leaf nodes are ``0`` to ``n-1``, i.e. the indices of `distances`.
///     Intermediate nodes continue the numbering in the order of their
///     creation, hence the root is the node ``2n-2``.
/// distances : ndarray, dtype=float32, shape=(2n-2,)
///     The distance between the parent and child node of each edge.
#[pyfunction]
pub fn upgma<'py>(
    py: Python<'py>,
    distances: PyReadonlyArray2<'py, f32>,
) -> PyResult<EdgeArrays<'py>> {
    let (n, mut distances) = check_distance_matrix(&distances)?;

    // The node ID at each position of the distance matrix
    // Initially these are the leaf nodes, after clustering a position
    // refers to the created intermediate node
    let mut node_ids: Vec<u32> = (0..n as u32).collect();
    let mut next_node_id = n as u32;
    // Indicates whether an index in the distance matrix has already been
    // clustered and the respective rows and columns can be ignored
    let mut is_clustered = vec![false; n];
    // Number of leaves in the cluster at each position (cardinality),
    // required for proportional averaging
    let mut cluster_size = vec![1u32; n];
    // Distance of each node from its leaf nodes,
    // used for calculation of the distance to its child nodes
    let mut node_heights = vec![0.0f32; n];
    // The minimum distance in each row of the lower triangle matrix
    // and the column where it is found
    // Tracking the row minima avoids scanning the entire matrix
    // in each iteration
    let mut row_min_dist = vec![f32::MAX; n];
    let mut row_min_col = vec![usize::MAX; n];
    for i in 0..n {
        (row_min_dist[i], row_min_col[i]) = scan_row(&distances, &is_clustered, n, i);
    }
    let mut edges = TreeEdges::with_capacity(2 * n);

    for iteration in 0.. {
        check_signals_periodically(py, iteration)?;

        // Find minimum distance
        let mut dist_min = f32::MAX;
        let mut i_min = usize::MAX;
        for i in 0..n {
            if !is_clustered[i] && row_min_dist[i] < dist_min {
                dist_min = row_min_dist[i];
                i_min = i;
            }
        }
        if i_min == usize::MAX {
            // No distance found -> all leaf nodes are clustered
            break;
        }
        let j_min = row_min_col[i_min];

        // Cluster the nodes with minimum distance
        // replacing the node at position `i_min`
        // and leaving the node at position `j_min` empty
        let height = dist_min / 2.0;
        edges.push(next_node_id, node_ids[i_min], height - node_heights[i_min]);
        edges.push(next_node_id, node_ids[j_min], height - node_heights[j_min]);
        node_ids[i_min] = next_node_id;
        next_node_id += 1;
        node_heights[i_min] = height;
        is_clustered[j_min] = true;
        // Calculate the arithmetic mean distances of the child nodes
        // as distances for the new node and update the matrix
        let size_i = cluster_size[i_min] as f32;
        let size_j = cluster_size[j_min] as f32;
        for k in 0..n {
            if !is_clustered[k] && k != i_min {
                let mean = (distances[i_min * n + k] * size_i + distances[j_min * n + k] * size_j)
                    / (size_i + size_j);
                distances[i_min * n + k] = mean;
                distances[k * n + i_min] = mean;
            }
        }
        cluster_size[i_min] += cluster_size[j_min];

        // Update the row minima affected by the changed row and column
        for k in 0..n {
            if is_clustered[k] {
                continue;
            }
            if k == i_min || row_min_col[k] == i_min || row_min_col[k] == j_min {
                // The entire row changed or the previous minimum is invalid
                (row_min_dist[k], row_min_col[k]) = scan_row(&distances, &is_clustered, n, k);
            } else if i_min < k {
                // Only the entry in the changed column may be the new minimum
                // Ties are broken by the smaller column,
                // consistent with `scan_row()`
                let dist = distances[k * n + i_min];
                if dist < row_min_dist[k] || (dist == row_min_dist[k] && i_min < row_min_col[k]) {
                    row_min_dist[k] = dist;
                    row_min_col[k] = i_min;
                }
            }
        }
    }

    Ok((
        edges.parents.into_pyarray(py),
        edges.children.into_pyarray(py),
        edges.distances.into_pyarray(py),
    ))
}

/// Find the minimum distance in row `i` over all unclustered columns `j < i`.
///
/// Returns the distance and the column.
/// If no such column exists, `f32::MAX` and `usize::MAX` are returned.
fn scan_row(distances: &[f32], is_clustered: &[bool], n: usize, i: usize) -> (f32, usize) {
    let mut dist_min = f32::MAX;
    let mut j_min = usize::MAX;
    for j in 0..i {
        if is_clustered[j] {
            continue;
        }
        let dist = distances[i * n + j];
        if dist < dist_min {
            dist_min = dist;
            j_min = j;
        }
    }
    (dist_min, j_min)
}
