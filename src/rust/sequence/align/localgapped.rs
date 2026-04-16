//! Local *X-Drop* gapped alignment of a single region (`align_region`).
//!
//! This module aligns one region extending from the *start* of the two given
//! (sub)sequences, using the X-Drop heuristic :footcite:`Zhang2000`: the
//! dynamic programming table is filled antidiagonal by antidiagonal and a cell
//! is only kept while its score stays within `threshold` of the best score seen
//! so far. The [`StridedTable`] grows on demand (via
//! [`StridedTable::from_data`]), so memory is proportional to the aligned region
//! rather than the full sequences.
//!
//! The upstream/downstream split around the seed, the sequence reversal for the
//! upstream region and the trace combination are handled by the Python layer;
//! this module only aligns a single forward region.
//!
//! The X-Drop [`FillRule`] differs from the rectangular one in
//! [`pairwise`](super::pairwise): a score of `0` marks an unreached (invalid)
//! cell, the diagonal predecessor only contributes if it is itself valid, the
//! `(0, 0)` anchor is seeded with `threshold + 1` (subtracted from the final
//! score), and [`fill`](FillRule::fill) prunes a cell against the running
//! maximum instead of always producing a value. Only the max-selection
//! primitive ([`get_trace_linear`]/[`get_trace_affine`]) and the local
//! traceback-start search are shared with the other algorithms.

use super::cell::{
    get_trace_affine, get_trace_linear, AffineCell, AffineDirection, AffineDirectionStore,
    AffineScore, Cell, LinearCell, LinearDirection, LinearDirectionStore, TraceDirection,
    TraceState,
};
use super::scoring::{GapPenalty, Score, Scoring, ScoringScheme};
use super::symbol::Symbol;
use super::table::{StridedTable, TableIndex};
use super::trace::{follow_trace, path_to_trace};
use crate::dispatch_dtype;
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyMemoryError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::cmp::Ordering;
use std::marker::PhantomData;

/// The initial side length of the dynamic programming table.
const INIT_SIZE: usize = 100;

/// Signals that a table growth would exceed the user-provided maximum size.
struct TableSizeExceeded;

/// Align a single region extending from the start of `code1`/`code2`.
///
/// Parameters
/// ----------
/// code1, code2 : ndarray
///     The region sequence codes, sharing the same unsigned integer dtype.
/// scoring : ndarray, dtype=int32 or tuple(int, int)
///     A substitution matrix or a `(match, mismatch)` score pair.
/// gap_penalty : int or tuple(int, int)
///     A linear or affine gap penalty.
/// threshold : int
///     The X-Drop threshold.
/// max_number : int
///     The maximum number of (co-optimal) alignments to return.
/// score_only : bool
///     If true, only the score is computed and the trace list is empty.
/// max_table_size : int
///     Raise `MemoryError` if the dynamic programming table would exceed this
///     number of cells.
///
/// Returns
/// -------
/// traces : list of ndarray
///     The alignment traces, each of shape `(length, 2)`. Empty if `score_only`.
/// score : int
///     The local alignment score of the region.
#[pyfunction]
#[pyo3(signature = (code1, code2, scoring, gap_penalty, threshold, max_number, score_only, max_table_size))]
#[allow(clippy::too_many_arguments)]
pub fn align_region<'py>(
    py: Python<'py>,
    code1: &Bound<'py, PyUntypedArray>,
    code2: &Bound<'py, PyUntypedArray>,
    scoring: Scoring,
    gap_penalty: GapPenalty,
    threshold: Score,
    max_number: usize,
    score_only: bool,
    max_table_size: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let dtype = code1.dtype();
    dispatch_dtype!(
        py,
        &dtype,
        [u8, u16, u32, u64],
        align_region_dtype(
            py,
            code1,
            code2,
            scoring,
            gap_penalty,
            threshold,
            max_number,
            score_only,
            max_table_size
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn align_region_dtype<'py, S: Symbol + numpy::Element>(
    py: Python<'py>,
    code1: &Bound<'py, PyUntypedArray>,
    code2: &Bound<'py, PyUntypedArray>,
    scoring: Scoring,
    gap_penalty: GapPenalty,
    threshold: Score,
    max_number: usize,
    score_only: bool,
    max_table_size: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    let code1_array = code1.cast::<PyArray1<S>>()?.readonly();
    let code2_array = code2.cast::<PyArray1<S>>()?.readonly();
    let code1 = code1_array.as_slice()?;
    let code2 = code2_array.as_slice()?;

    // Instantiate the fill rule for the runtime scoring scheme, gap penalty and
    // `score_only` flag, then run the generic core. The direction-storage type
    // (`$dir` for a full alignment, `()` for score-only) selects the cell size.
    macro_rules! run {
        ($rule:ident, $dir:ty, $scheme:expr, ($($penalty:expr),+)) => {
            align_region_generic(
                code1, code2,
                <$rule<_, _, $dir>>::new($scheme, $($penalty),+, threshold),
                max_number, max_table_size,
            )
        };
    }
    macro_rules! by_consts {
        ($rule:ident, $dir:ty, $scheme:expr, ($($penalty:expr),+)) => {
            if score_only {
                run!($rule, (), $scheme, ($($penalty),+))
            } else {
                run!($rule, $dir, $scheme, ($($penalty),+))
            }
        };
    }
    macro_rules! by_penalty {
        ($scheme:expr) => {
            match gap_penalty {
                GapPenalty::Linear(gap) => {
                    by_consts!(LinearGapPenalty, LinearDirection, $scheme, (gap))
                }
                GapPenalty::Affine(open, extend) => {
                    by_consts!(AffineGapPenalty, AffineDirection, $scheme, (open, extend))
                }
            }
        };
    }
    let result = match scoring {
        Scoring::Matrix(scheme) => by_penalty!(scheme),
        Scoring::Match(scheme) => by_penalty!(scheme),
    };
    let (traces, score) =
        result.map_err(|_| PyMemoryError::new_err("Maximum table size exceeded"))?;

    let list = PyList::empty(py);
    for trace in traces {
        list.append(trace.into_pyarray(py))?;
    }
    PyTuple::new(py, [list.into_any(), score.into_pyobject(py)?.into_any()])
}

/// Core local X-Drop alignment of a single region, generic over the symbol type
/// and fill rule.
fn align_region_generic<S, F>(
    code1: &[S],
    code2: &[S],
    rule: F,
    max_number: usize,
    max_table_size: usize,
) -> Result<(Vec<Array2<i64>>, Score), TableSizeExceeded>
where
    S: Symbol,
    F: FillRule<S>,
{
    let (table, max_score) = fill_xdrop(code1, code2, &rule, max_table_size)?;
    // The fill already tracks the maximum score; only the traceback needs the
    // start cells, so `score_only` avoids that extra table scan.
    let traces = if F::SCORE_ONLY {
        Vec::new()
    } else {
        let (_, starts) = rule.find_traceback_starts(&table);
        traceback(&table, starts, max_number)
    };
    Ok((traces, max_score - rule.init_score()))
}

/// Fill the X-Drop table antidiagonal by antidiagonal.
///
/// The table grows on demand and the fill stops once an antidiagonal has no
/// surviving cell. Returns the filled table and the maximum score reached.
fn fill_xdrop<S, F>(
    code1: &[S],
    code2: &[S],
    rule: &F,
    max_table_size: usize,
) -> Result<(StridedTable<F::Cell>, Score), TableSizeExceeded>
where
    S: Symbol,
    F: FillRule<S>,
{
    let n = code1.len();
    let m = code2.len();
    let mut table = StridedTable::<F::Cell>::new(((n + 1).min(INIT_SIZE), (m + 1).min(INIT_SIZE)));
    let origin = table.index((0, 0));
    table[origin] = rule.fill_origin();

    let mut max_score = rule.init_score();

    // The valid `i` range of the current (`k0`) and two preceding antidiagonals (`k1` and `k2`)
    // The `k2` pair is always assigned (from `k1`) before it is read.
    let (mut i_min_k0, mut i_max_k0) = (0usize, 0usize);
    let (mut i_min_k1, mut i_max_k1) = (0usize, 0usize);
    let (mut i_min_k2, mut i_max_k2): (usize, usize);

    // Instead of iteration over rows and columns, iterate over antidiagonals and diagonals to
    // achieve symmetric treatment of both sequences
    for k in 1..=(n + m) {
        // Move up the antidiagonal chain
        i_min_k2 = i_min_k1;
        i_max_k2 = i_max_k1;
        i_min_k1 = i_min_k0;
        i_max_k1 = i_max_k0;
        // Reset to the most restrictive range; `k` is the 'unset' sentinel for
        // `i_min`, since every valid `i` in this antidiagonal is `< k`.
        i_min_k0 = k;
        i_max_k0 = 0;

        // Prune the `i` range to where valid cells can exist,
        // then clamp it to the sequence bounds
        let i_min = i_min_k1.min(i_min_k2 + 1).max(k.saturating_sub(m));
        let i_max = (i_max_k1 + 1).max(i_max_k2 + 1).min(n);
        if i_min > i_max {
            // The calculated antidiagonal has no range of valid cells -> the alignment has finished
            break;
        }
        let j_max = k - i_min;

        // Grow the table if the current antidiagonal would exceed it.
        let (rows, cols) = table.shape();
        if i_max >= rows {
            let new_rows = rows * 2;
            if new_rows
                .checked_mul(cols)
                .is_none_or(|size| size > max_table_size)
            {
                return Err(TableSizeExceeded);
            }
            table = StridedTable::from_data((new_rows, cols), table);
        }
        let (rows, cols) = table.shape();
        if j_max >= cols {
            let new_cols = cols * 2;
            if rows
                .checked_mul(new_cols)
                .is_none_or(|size| size > max_table_size)
            {
                return Err(TableSizeExceeded);
            }
            table = StridedTable::from_data((rows, new_cols), table);
        }

        // Constant offsets to the three predecessors. Consecutive cells of the
        // antidiagonal move by `diag_step` (down one row, left one column), so
        // the index is walked instead of recomputed per cell. The offsets are
        // determined here, after any table growth, where the stride is final.
        let diag_offset = table.index_offset((-1, -1));
        let gap1_offset = table.index_offset((0, -1));
        let gap2_offset = table.index_offset((-1, 0));
        let diag_step = table.index_offset((1, -1));
        let mut current = table.index((i_min, k - i_min));

        for i in i_min..=i_max {
            let j = k - i;
            // The diagonal predecessor (and its symbols) only exist away from
            // the top row and left column.
            let diagonal = if i > 0 && j > 0 {
                Some((code1[i - 1], code2[j - 1], table[current + diag_offset]))
            } else {
                None
            };
            let from_gap1 = if j > 0 {
                table[current + gap1_offset]
            } else {
                F::Cell::default()
            };
            let from_gap2 = if i > 0 {
                table[current + gap2_offset]
            } else {
                F::Cell::default()
            };
            if let Some(cell) = rule.fill(diagonal, &from_gap1, &from_gap2, &mut max_score) {
                if i_min_k0 == k {
                    i_min_k0 = i;
                }
                i_max_k0 = i;
                table[current] = cell;
            }
            current += diag_step;
        }
    }

    Ok((table, max_score))
}

/// Run the traceback over a filled table and convert the paths to trace arrays.
fn traceback<C: Cell>(
    table: &StridedTable<C>,
    starts: Vec<(TableIndex, <C as Cell>::StateInfo)>,
    max_number: usize,
) -> Vec<Array2<i64>> {
    let match_offset = table.index_offset((-1, -1));
    let gap1_offset = table.index_offset((0, -1));
    let gap2_offset = table.index_offset((-1, 0));
    let trace_offset = |direction| match direction {
        TraceDirection::Diag => match_offset,
        TraceDirection::To1 => gap1_offset,
        TraceDirection::To2 => gap2_offset,
    };
    let mut paths = Vec::new();
    for (start, state) in starts {
        let mut count: usize = 1;
        follow_trace(
            table,
            start,
            state,
            &trace_offset,
            &mut count,
            max_number,
            &mut paths,
        );
    }
    paths.truncate(max_number);
    paths
        .into_iter()
        .map(|path| {
            path_to_trace(&path, |index| {
                let (i, j) = table.unindex(index);
                (i as i64 - 1, j as i64 - 1)
            })
        })
        .collect()
}

/// Analogous to the [`FillRule`](super::pairwise::FillRule) in `pairwise.rs`.
pub trait FillRule<S: Symbol> {
    type Cell: Cell;

    const SCORE_ONLY: bool;

    fn fill_origin(&self) -> Self::Cell;

    fn init_score(&self) -> Score;

    fn fill(
        &self,
        diagonal: Option<(S, S, Self::Cell)>,
        from_gap1: &Self::Cell,
        from_gap2: &Self::Cell,
        max_observed_score: &mut Score,
    ) -> Option<Self::Cell>;

    #[allow(clippy::type_complexity)]
    fn find_traceback_starts(
        &self,
        table: &StridedTable<Self::Cell>,
    ) -> (Score, Vec<(TableIndex, <Self::Cell as Cell>::StateInfo)>);
}

pub struct LinearGapPenalty<S, V, D>
where
    S: Symbol,
    V: ScoringScheme<S>,
    D: LinearDirectionStore,
{
    scoring_scheme: V,
    gap_penalty: Score,
    threshold: Score,
    _phantom: PhantomData<(S, D)>,
}

impl<S, V, D> LinearGapPenalty<S, V, D>
where
    S: Symbol,
    V: ScoringScheme<S>,
    D: LinearDirectionStore,
{
    pub fn new(scoring_scheme: V, gap_penalty: Score, threshold: Score) -> Self {
        LinearGapPenalty {
            scoring_scheme,
            gap_penalty,
            threshold,
            _phantom: PhantomData,
        }
    }
}

impl<S, V, D> FillRule<S> for LinearGapPenalty<S, V, D>
where
    S: Symbol,
    V: ScoringScheme<S>,
    D: LinearDirectionStore,
{
    type Cell = LinearCell<D>;

    const SCORE_ONLY: bool = D::SCORE_ONLY;

    #[inline]
    fn fill_origin(&self) -> LinearCell<D> {
        LinearCell {
            score: self.init_score(),
            direction: D::default(),
        }
    }

    #[inline]
    fn init_score(&self) -> Score {
        // One above the threshold, so that a score of `0` can mark an invalid
        // cell while the anchor itself stays valid.
        self.threshold + 1
    }

    #[inline(always)]
    fn fill(
        &self,
        diagonal: Option<(S, S, LinearCell<D>)>,
        from_gap1: &LinearCell<D>,
        from_gap2: &LinearCell<D>,
        max_observed_score: &mut Score,
    ) -> Option<LinearCell<D>> {
        // The diagonal only contributes if its predecessor is itself valid.
        let from_match = match diagonal {
            Some((symbol1, symbol2, predecessor)) if predecessor.score != 0 => {
                predecessor.score + self.scoring_scheme.score(symbol1, symbol2)
            }
            _ => 0,
        };
        let from_gap1 = from_gap1.score + self.gap_penalty;
        let from_gap2 = from_gap2.score + self.gap_penalty;
        let (score, direction) = if D::SCORE_ONLY {
            (from_match.max(from_gap1).max(from_gap2), D::default())
        } else {
            let (score, direction) = get_trace_linear(from_match, from_gap1, from_gap2);
            (score, D::wrap(direction))
        };
        if score >= *max_observed_score - self.threshold {
            if score > *max_observed_score {
                *max_observed_score = score;
            }
            Some(LinearCell { score, direction })
        } else {
            None
        }
    }

    fn find_traceback_starts(
        &self,
        table: &StridedTable<LinearCell<D>>,
    ) -> (Score, Vec<(TableIndex, ())>) {
        let mut max = Score::MIN;
        let mut starts = Vec::new();
        for (index, cell) in table.indexed_iter() {
            match cell.score.cmp(&max) {
                Ordering::Greater => {
                    max = cell.score;
                    starts.clear();
                    starts.push((index, ()));
                }
                Ordering::Equal => starts.push((index, ())),
                Ordering::Less => {}
            }
        }
        (max, starts)
    }
}

pub struct AffineGapPenalty<S, V, D>
where
    S: Symbol,
    V: ScoringScheme<S>,
    D: AffineDirectionStore,
{
    scoring_scheme: V,
    gap_open: Score,
    gap_extend: Score,
    threshold: Score,
    _phantom: PhantomData<(S, D)>,
}

impl<S, V, D> AffineGapPenalty<S, V, D>
where
    S: Symbol,
    V: ScoringScheme<S>,
    D: AffineDirectionStore,
{
    pub fn new(scoring_scheme: V, gap_open: Score, gap_extend: Score, threshold: Score) -> Self {
        AffineGapPenalty {
            scoring_scheme,
            gap_open,
            gap_extend,
            threshold,
            _phantom: PhantomData,
        }
    }
}

impl<S, V, D> FillRule<S> for AffineGapPenalty<S, V, D>
where
    S: Symbol,
    V: ScoringScheme<S>,
    D: AffineDirectionStore,
{
    type Cell = AffineCell<D>;

    const SCORE_ONLY: bool = D::SCORE_ONLY;

    #[inline]
    fn fill_origin(&self) -> AffineCell<D> {
        AffineCell {
            score: AffineScore {
                m: self.init_score(),
                g1: 0,
                g2: 0,
            },
            direction: D::default(),
        }
    }

    #[inline]
    fn init_score(&self) -> Score {
        self.threshold + 1
    }

    #[inline(always)]
    fn fill(
        &self,
        diagonal: Option<(S, S, AffineCell<D>)>,
        from_gap1: &AffineCell<D>,
        from_gap2: &AffineCell<D>,
        max_observed_score: &mut Score,
    ) -> Option<AffineCell<D>> {
        // Transitions into the match table from the diagonal predecessor; each
        // contributes only if the originating table was valid.
        let (mm, g1m, g2m) = match diagonal {
            Some((symbol1, symbol2, predecessor)) => {
                let similarity = self.scoring_scheme.score(symbol1, symbol2);
                let gated = |score: Score| if score != 0 { score + similarity } else { 0 };
                (
                    gated(predecessor.score.m),
                    gated(predecessor.score.g1),
                    gated(predecessor.score.g2),
                )
            }
            None => (0, 0, 0),
        };
        // Transitions into the gap-in-1 (left) and gap-in-2 (top) tables.
        let mg1 = from_gap1.score.m + self.gap_open;
        let g1g1 = from_gap1.score.g1 + self.gap_extend;
        let mg2 = from_gap2.score.m + self.gap_open;
        let g2g2 = from_gap2.score.g2 + self.gap_extend;

        let (m_score, g1_score, g2_score, direction) = if D::SCORE_ONLY {
            (
                mm.max(g1m).max(g2m),
                mg1.max(g1g1),
                mg2.max(g2g2),
                D::default(),
            )
        } else {
            let (m_score, g1_score, g2_score, direction) =
                get_trace_affine(mm, g1m, g2m, mg1, g1g1, mg2, g2g2);
            (m_score, g1_score, g2_score, D::wrap(direction))
        };

        // Each table is independently kept if it reaches the threshold; the cell
        // is valid (and the full trace is stored) if any of them is.
        let mut score = AffineScore::default();
        let mut valid = false;
        let mut req_score = *max_observed_score - self.threshold;
        if m_score >= req_score {
            score.m = m_score;
            valid = true;
            if m_score > *max_observed_score {
                *max_observed_score = m_score;
                req_score = *max_observed_score - self.threshold;
            }
        }
        if g1_score >= req_score {
            score.g1 = g1_score;
            valid = true;
            if g1_score > *max_observed_score {
                *max_observed_score = g1_score;
                req_score = *max_observed_score - self.threshold;
            }
        }
        if g2_score >= req_score {
            score.g2 = g2_score;
            valid = true;
            if g2_score > *max_observed_score {
                *max_observed_score = g2_score;
            }
        }
        if valid {
            Some(AffineCell { score, direction })
        } else {
            None
        }
    }

    fn find_traceback_starts(
        &self,
        table: &StridedTable<AffineCell<D>>,
    ) -> (Score, Vec<(TableIndex, TraceState)>) {
        // Only the match table is considered: a local alignment cannot start or
        // end with a gap.
        let mut max = Score::MIN;
        let mut starts = Vec::new();
        for (index, cell) in table.indexed_iter() {
            match cell.score.m.cmp(&max) {
                Ordering::Greater => {
                    max = cell.score.m;
                    starts.clear();
                    starts.push((index, TraceState::Match));
                }
                Ordering::Equal => starts.push((index, TraceState::Match)),
                Ordering::Less => {}
            }
        }
        (max, starts)
    }
}
