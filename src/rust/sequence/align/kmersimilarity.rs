//! Hot path for [`ScoreThresholdRule`](biotite).
//!
//! [`similar_kmers`] enumerates, for a given *k-mer*, every *k-mer* whose
//! summed substitution score against it reaches a threshold. It uses the
//! *branch-and-bound* traversal of :footcite:`Hauser2013`: positions are filled
//! left to right and a branch is pruned as soon as the partial score can no
//! longer reach the threshold even with the best possible remaining symbols.
//! The traversal is inherently sequential and stateful, so it stays in Rust
//! while `ScoreThresholdRule` itself lives in Python.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// similar_kmers(matrix, max_scores, split_kmer, alphabet_length, threshold)
///
/// Find all *k-mers* whose substitution score against `split_kmer` is greater
/// than or equal to `threshold`, returned as split symbol codes.
///
/// Parameters
/// ----------
/// matrix : ndarray, dtype=int32, shape=(n, n)
///     The substitution score matrix, indexed by base-alphabet symbol codes.
/// max_scores : ndarray, dtype=int32, shape=(n,)
///     The maximum score in each row of `matrix`.
/// split_kmer : ndarray, dtype=int64, shape=(k,)
///     The reference *k-mer* as base-alphabet symbol codes.
/// alphabet_length : int
///     The number of symbols in the base alphabet.
/// threshold : int
///     The minimum similarity score a *k-mer* must reach to be reported.
///
/// Returns
/// -------
/// similar_split_kmers : ndarray, dtype=int64, shape=(m, k)
///     The similar *k-mers* as split symbol codes. Includes `split_kmer`
///     itself and contains no duplicates.
#[pyfunction]
pub fn similar_kmers<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, i32>,
    max_scores: PyReadonlyArray1<'py, i32>,
    split_kmer: PyReadonlyArray1<'py, i64>,
    alphabet_length: usize,
    threshold: i32,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    // Index the matrix through a flat C-contiguous slice (`row * n_cols + col`)
    // instead of `ndarray`'s per-element 2D indexing, which dominates run time otherwise
    let n_cols = matrix.as_array().ncols();
    let matrix = matrix.as_slice()?;
    let max_scores = max_scores.as_slice()?;
    let split_kmer = split_kmer.as_slice()?;
    let k = split_kmer.len();

    // Calculate the minimum score for each k-mer position that is necessary to
    // reach a total higher than or equal to the threshold score
    let mut positional_thresholds = vec![0i32; k];
    let mut total_max_score: i32 = 0;
    for i in (0..k).rev() {
        positional_thresholds[i] = threshold - total_max_score;
        total_max_score += max_scores[split_kmer[i] as usize];
    }

    // This array holds the current k-mer to be tested
    let mut current_split_kmer = vec![0i64; k];
    // This array stores the accepted k-mers, i.e. k-mers that reach the threshold
    // score (stored flat, k symbols each)
    let mut similar_split_kmers: Vec<i64> = Vec::new();

    // `pos` is the current position within the k-mer where symbols are
    // substituted; it is -1 after all symbol codes at position 0 are traversed
    let mut pos: isize = 0;
    while pos != -1 {
        let p = pos as usize;
        if current_split_kmer[p] >= alphabet_length as i64 {
            // All symbol codes were traversed at this position
            // -> jump one k-mer position back and proceed with the next symbol
            pos -= 1;
            if pos != -1 {
                current_split_kmer[pos as usize] += 1;
            }
            continue;
        }
        // Get the total similarity score between the input k-mer and the generated
        // k-mer up to the point of the current position
        let mut score: i32 = 0;
        for i in 0..=p {
            score += matrix[split_kmer[i] as usize * n_cols + current_split_kmer[i] as usize];
        }
        // Check the score threshold condition
        if score >= positional_thresholds[p] {
            // Threshold condition is fulfilled: either go deeper in the same
            // branch (jump one position forward) ...
            if p < k - 1 {
                pos += 1;
                current_split_kmer[pos as usize] = 0;
            } else {
                // ... or store the similar k-mer, if already at maximum depth
                // (the last k-mer position), then proceed with the next symbol
                similar_split_kmers.extend_from_slice(&current_split_kmer);
                current_split_kmer[p] += 1;
            }
        } else {
            // The threshold score is not reached -> this branch ends and we
            // proceed with the next symbol at this position
            current_split_kmer[p] += 1;
        }
    }

    let n_similar = similar_split_kmers.len() / k;
    let array = Array2::from_shape_vec((n_similar, k), similar_split_kmers)
        .expect("flat buffer length is a multiple of k");
    Ok(array.into_pyarray(py))
}
