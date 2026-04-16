//! Traceback over a filled [`StridedTable`] to reconstruct alignments.

use super::cell::{Cell, TraceDirection};
use super::table::{StridedTable, TableIndex, TableOffset};
use numpy::ndarray::Array2;
use std::collections::HashSet;

/// Walk the trace table from a start index, collecting one or more optimal
/// traces (each a path of table indices, in end-to-start order).
///
/// `offset` maps a [`TraceDirection`] to the [`TableOffset`] step to take in the
/// table. For an ordinary rectangular table the diagonal moves by
/// `index_offset((-1, -1))`, etc.; banded alignments supply a different
/// mapping. This keeps the diagonal-to-rectangle index logic out of
/// `follow_trace`.
///
/// `count` is the shared branch counter and `max_count` the branch limit
/// (`max_number`): once `count` reaches `max_count`, no further branches are
/// spawned.
#[allow(clippy::too_many_arguments)]
pub fn follow_trace<C, Off>(
    table: &StridedTable<C>,
    start: TableIndex,
    start_state: C::StateInfo,
    offset: &Off,
    count: &mut usize,
    max_count: usize,
    out: &mut Vec<Vec<TableIndex>>,
) where
    C: Cell,
    Off: Fn(TraceDirection) -> TableOffset,
{
    follow_trace_inner(
        table,
        start,
        start_state,
        offset,
        count,
        max_count,
        out,
        Vec::new(),
    );
}

#[allow(clippy::too_many_arguments)]
fn follow_trace_inner<C, Off>(
    table: &StridedTable<C>,
    start: TableIndex,
    start_state: C::StateInfo,
    offset: &Off,
    count: &mut usize,
    max_count: usize,
    out: &mut Vec<Vec<TableIndex>>,
    mut path: Vec<TableIndex>,
) where
    C: Cell,
    Off: Fn(TraceDirection) -> TableOffset,
{
    let mut index = start;
    let mut state = start_state;
    loop {
        let cell = table[index];
        let steps = cell.traceback(state);
        if steps.is_empty() {
            // No further direction -> the trace ends here (this cell is not
            // part of the alignment).
            break;
        }
        path.push(index);
        // Spawn a branch for every additional direction (a tie).
        for &(dir, next_state) in steps.iter().skip(1) {
            if *count < max_count {
                *count += 1;
                follow_trace_inner(
                    table,
                    index + offset(dir),
                    next_state,
                    offset,
                    count,
                    max_count,
                    out,
                    path.clone(),
                );
            }
        }
        // Continue the main trace with the first (highest-priority) direction.
        let (dir, next_state) = steps[0];
        index += offset(dir);
        state = next_state;
    }
    out.push(path);
}

/// Convert a trace path (table indices, end-to-start order) into an alignment
/// trace array of shape `(length, 2)` with `-1` marking gaps.
///
/// `to_seq` maps a table index to the `(seq1_index, seq2_index)` it represents
/// (before gap filtering). For non-banded alignment this is `(i - 1, j - 1)`.
///
/// After reversing the path to start-to-end order, the first occurrence of each
/// sequence index is kept and any repeats become `-1` (a gap): a run of equal
/// indices in one sequence corresponds to a stretch of gaps in that sequence.
pub fn path_to_trace<F>(path: &[TableIndex], to_seq: F) -> Array2<i64>
where
    F: Fn(TableIndex) -> (i64, i64),
{
    let len = path.len();
    let mut col0: Vec<i64> = Vec::with_capacity(len);
    let mut col1: Vec<i64> = Vec::with_capacity(len);
    // The path is in end-to-start order; reverse it to start-to-end.
    for &index in path.iter().rev() {
        let (s0, s1) = to_seq(index);
        col0.push(s0);
        col1.push(s1);
    }
    gap_filter(&mut col0);
    gap_filter(&mut col1);

    // Drop rows that became a gap in both sequences (defensive; should not
    // normally happen, as every step advances at least one sequence).
    let mut rows: Vec<[i64; 2]> = Vec::with_capacity(len);
    for k in 0..len {
        if col0[k] == -1 && col1[k] == -1 {
            continue;
        }
        rows.push([col0[k], col1[k]]);
    }

    Array2::from_shape_fn((rows.len(), 2), |(r, c)| rows[r][c])
}

/// Replace every repeated value in a column with `-1`, keeping only the first
/// occurrence of each value.
fn gap_filter(column: &mut [i64]) {
    let mut seen: HashSet<i64> = HashSet::new();
    for value in column.iter_mut() {
        // `insert` returns false if the value was already present.
        if !seen.insert(*value) {
            *value = -1;
        }
    }
}
