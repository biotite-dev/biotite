//! Hot path for the *k-mer* subset selectors ([`MinimizerSelector`](biotite)
//! and friends).
//!
//! [`minimize`] reports, for every overlapping window of *k-mers*, the position
//! of the minimum *k-mer* (the *minimizer*). It uses the linear-time sliding
//! window minimum of Marcel van Herk :footcite:`VanHerk1992`: a forward and a
//! reverse chunk-wise *argument of the cumulative minimum* are precomputed, and
//! each window minimum is then the smaller of the two overlapping chunk minima.
//! Ties resolve to the leftmost position. The Python selector classes call this
//! after turning sequences into *k-mer* / ordering arrays.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// minimize(kmers, ordering, window, include_duplicates)
///
/// Select the minimizer of each overlapping window of `window` *k-mers*.
///
/// Parameters
/// ----------
/// kmers : ndarray, dtype=int64
///     The *k-mer* codes; the returned minimizers are taken from this array.
/// ordering : ndarray, dtype=int64
///     The sort key the minimum is taken over (same length as `kmers`). Equal
///     to `kmers` for the standard order, or a permuted order otherwise.
/// window : int
///     The number of *k-mers* per window.
/// include_duplicates : bool
///     If false, a minimizer shared by consecutive windows is reported only
///     once. If true, every window contributes an entry.
///
/// Returns
/// -------
/// minimizer_indices : ndarray, dtype=uint32
///     The indices into `kmers` where the selected minimizers start.
/// minimizers : ndarray, dtype=int64
///     The corresponding minimizer *k-mer* codes.
#[pyfunction]
pub fn minimize<'py>(
    py: Python<'py>,
    kmers: PyReadonlyArray1<'py, i64>,
    ordering: PyReadonlyArray1<'py, i64>,
    window: u32,
    include_duplicates: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let kmers = kmers.as_slice()?;
    let ordering = ordering.as_slice()?;
    let window = window as usize;

    let n = kmers.len();
    let n_windows = n - (window - 1);
    // The cumulative minima at all positions
    let forward_argcummins = chunked_argcummin::<Forward>(ordering, window);
    let reverse_argcummins = chunked_argcummin::<Reverse>(ordering, window);

    // Pessimistic array allocation -> Expect that every window has a new minimizer
    let mut minimizer_pos: Vec<u32> = Vec::with_capacity(n_windows);
    let mut minimizers: Vec<i64> = Vec::with_capacity(n_windows);
    // A position that can never occur, as there is no previous value at the start
    let mut prev_argcummin: usize = n;

    for seq_i in 0..n_windows {
        let forward_argcummin = forward_argcummins[seq_i + window - 1];
        let reverse_argcummin = reverse_argcummins[seq_i];
        // At ties the leftmost position is taken, which stems from the reverse pass
        let combined_argcummin =
            if ordering[forward_argcummin as usize] < ordering[reverse_argcummin as usize] {
                forward_argcummin
            } else {
                reverse_argcummin
            };

        if include_duplicates || combined_argcummin as usize != prev_argcummin {
            minimizer_pos.push(combined_argcummin);
            minimizers.push(kmers[combined_argcummin as usize]);
            prev_argcummin = combined_argcummin as usize;
        }
    }

    PyTuple::new(
        py,
        [
            minimizer_pos.into_pyarray(py).into_any(),
            minimizers.into_pyarray(py).into_any(),
        ],
    )
}

/// The sweep direction of [`chunked_argcummin`], abstracting the only three
/// differences between the forward and reverse passes.
trait Direction {
    /// The order in which positions are visited.
    fn order(n: usize) -> impl Iterator<Item = usize>;
    /// Whether `seq_i` is the first position of a chunk, where the running
    /// minimum is reset.
    fn is_chunk_start(seq_i: usize, chunk_size: usize) -> bool;
    /// Whether `value` replaces the current minimum. The forward pass uses `<`
    /// and the reverse pass `<=`, so that on ties the leftmost position wins.
    fn is_new_min(value: i64, current_min: i64) -> bool;
}

struct Forward;
struct Reverse;

impl Direction for Forward {
    fn order(n: usize) -> impl Iterator<Item = usize> {
        0..n
    }
    fn is_chunk_start(seq_i: usize, chunk_size: usize) -> bool {
        seq_i.is_multiple_of(chunk_size)
    }
    fn is_new_min(value: i64, current_min: i64) -> bool {
        value < current_min
    }
}

impl Direction for Reverse {
    fn order(n: usize) -> impl Iterator<Item = usize> {
        (0..n).rev()
    }
    fn is_chunk_start(seq_i: usize, chunk_size: usize) -> bool {
        // The chunk border sits on the left here, so the reverse chunks stay
        // aligned with the forward pass instead of reversing the input
        seq_i % chunk_size == chunk_size - 1
    }
    fn is_new_min(value: i64, current_min: i64) -> bool {
        value <= current_min
    }
}

/// The argument of the cumulative minimum, restarted at the beginning of each
/// chunk of `chunk_size` values and swept in the given `D`irection.
fn chunked_argcummin<D: Direction>(values: &[i64], chunk_size: usize) -> Vec<u32> {
    // Check once that the window is non-zero, so the compiler can rely on the
    // per-iteration modulo to never divide by zero
    assert!(chunk_size != 0);
    let mut min_pos = vec![0u32; values.len()];
    let mut current_min_i: u32 = 0;
    let mut current_min = i64::MAX;
    for seq_i in D::order(values.len()) {
        if D::is_chunk_start(seq_i, chunk_size) {
            current_min = i64::MAX;
        }
        let value = values[seq_i];
        if D::is_new_min(value, current_min) {
            current_min_i = seq_i as u32;
            current_min = value;
        }
        min_pos[seq_i] = current_min_i;
    }
    min_pos
}
