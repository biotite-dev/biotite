//! Optimal pairwise alignment (`align_optimal`).
//!
//! The Python layer widens both sequence codes to a common unsigned integer
//! type and passes the raw code arrays plus the scoring scheme; this module
//! Fills the dynamic programming table, runs the traceback and returns the
//! resulting trace arrays and score.
//!
//! The [`FillRule`] for the full Needleman-Wunsch/Smith-Waterman/Gotoh table is
//! defined here, specialized at compile time via const generics for
//! local/global alignment, terminal gap penalty handling and score-only mode.

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
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::cmp::Ordering;
use std::marker::PhantomData;

/// Perform an optimal global or local pairwise alignment.
///
/// Parameters
/// ----------
/// code1, code2 : ndarray
///     The sequence codes. Both must share the same unsigned integer dtype.
/// scoring : ndarray, dtype=int32 or tuple(int, int)
///     A substitution matrix or a `(match, mismatch)` score pair.
/// gap_penalty : int or tuple(int, int)
///     A linear or affine gap penalty.
/// terminal_penalty : bool
///     Whether terminal gaps are penalized.
/// local : bool
///     Whether to perform a local (instead of global) alignment.
/// score_only : bool
///     If true, only the score is computed and the trace list is empty.
/// max_number : int
///     The maximum number of (co-optimal) alignments to return.
///
/// Returns
/// -------
/// traces : list of ndarray
///     The alignment traces, each of shape `(length, 2)`. Empty if `score_only`.
/// score : int
///     The optimal alignment score.
#[pyfunction]
#[pyo3(signature = (code1, code2, scoring, gap_penalty, terminal_penalty, local, score_only, max_number))]
#[allow(clippy::too_many_arguments)]
pub fn align_optimal<'py>(
    py: Python<'py>,
    code1: &Bound<'py, PyUntypedArray>,
    code2: &Bound<'py, PyUntypedArray>,
    scoring: Scoring,
    gap_penalty: GapPenalty,
    terminal_penalty: bool,
    local: bool,
    score_only: bool,
    max_number: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let dtype = code1.dtype();
    dispatch_dtype!(
        py,
        &dtype,
        [u8, u16, u32, u64],
        align_optimal_dtype(
            py,
            code1,
            code2,
            scoring,
            gap_penalty,
            terminal_penalty,
            local,
            score_only,
            max_number
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn align_optimal_dtype<'py, S: Symbol + numpy::Element>(
    py: Python<'py>,
    code1: &Bound<'py, PyUntypedArray>,
    code2: &Bound<'py, PyUntypedArray>,
    scoring: Scoring,
    gap_penalty: GapPenalty,
    terminal_penalty: bool,
    local: bool,
    score_only: bool,
    max_number: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    let code1_array = code1.cast::<PyArray1<S>>()?.readonly();
    let code2_array = code2.cast::<PyArray1<S>>()?.readonly();
    let code1 = code1_array.as_slice()?;
    let code2 = code2_array.as_slice()?;

    // Instantiate the fill rule for the runtime scoring scheme, gap penalty and
    // `local`/`terminal_penalty`/`score_only` flags, then run the generic core.
    // The direction-storage type (`$dir` for a full alignment, `()` for
    // score-only) selects the cell size.
    macro_rules! run {
        ($rule:ident, $dir:ty, $local:literal, $terminal:literal, $scheme:expr, ($($penalty:expr),+)) => {
            align_optimal_generic(
                code1, code2,
                <$rule<_, _, $dir, $local, $terminal>>::new($scheme, $($penalty),+),
                max_number,
            )
        };
    }
    macro_rules! by_consts {
        ($rule:ident, $dir:ty, $scheme:expr, ($($penalty:expr),+)) => {
            match (local, terminal_penalty, score_only) {
                (true, true, true) => run!($rule, (), true, true, $scheme, ($($penalty),+)),
                (true, true, false) => run!($rule, $dir, true, true, $scheme, ($($penalty),+)),
                (true, false, true) => run!($rule, (), true, false, $scheme, ($($penalty),+)),
                (true, false, false) => run!($rule, $dir, true, false, $scheme, ($($penalty),+)),
                (false, true, true) => run!($rule, (), false, true, $scheme, ($($penalty),+)),
                (false, true, false) => run!($rule, $dir, false, true, $scheme, ($($penalty),+)),
                (false, false, true) => run!($rule, (), false, false, $scheme, ($($penalty),+)),
                (false, false, false) => run!($rule, $dir, false, false, $scheme, ($($penalty),+)),
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
    let (traces, score) = match scoring {
        Scoring::Matrix(scheme) => by_penalty!(scheme),
        Scoring::Match(scheme) => by_penalty!(scheme),
    };

    let list = PyList::empty(py);
    for trace in traces {
        list.append(trace.into_pyarray(py))?;
    }
    PyTuple::new(py, [list.into_any(), score.into_pyobject(py)?.into_any()])
}

/// Core optimal pairwise alignment, generic over the symbol type and fill rule.
///
/// Returns the list of optimal traces and their shared score.
pub(crate) fn align_optimal_generic<S, F>(
    code1: &[S],
    code2: &[S],
    rule: F,
    max_number: usize,
) -> (Vec<Array2<i64>>, Score)
where
    S: Symbol,
    F: FillRule<S>,
{
    let n = code1.len();
    let m = code2.len();

    // The table is transposed relative to the usual visualization: the first
    // sequence runs down the rows, the second along the columns
    let mut table = StridedTable::<F::Cell>::new((n + 1, m + 1));

    // The flat index is affine in `(row, column)`, so the offset from a cell to
    // each of its predecessors is constant
    // -> Compute these offsets once instead of building an index per cell
    let match_offset = table.index_offset((-1, -1));
    let gap1_offset = table.index_offset((0, -1));
    let gap2_offset = table.index_offset((-1, 0));
    // Steps to the next cell along a row and down a column
    let column_step = table.index_offset((0, 1));
    let row_step = table.index_offset((1, 0));

    let origin = table.index((0, 0));
    table[origin] = rule.fill_origin();
    // First column: walk downward, each cell filled from the one above it.
    let mut current = origin;
    for _ in 0..n {
        let predecessor = table[current];
        current += row_step;
        table[current] = rule.fill_start1(&predecessor);
    }
    // First row: walk rightward, each cell filled from the one to its left.
    let mut current = origin;
    for _ in 0..m {
        let predecessor = table[current];
        current += column_step;
        table[current] = rule.fill_start2(&predecessor);
    }

    for (i_offset, &s1) in code1.iter().enumerate() {
        let i = i_offset + 1;
        let last_row = i == n;
        // Index of the first cell in this row
        // Consecutive cells along the row are reached by adding `column_step`
        let mut current = table.index((i, 1));
        for (j_offset, &s2) in code2.iter().enumerate() {
            let last_col = j_offset + 1 == m;
            let from_match = table[current + match_offset];
            let from_gap1 = table[current + gap1_offset];
            let from_gap2 = table[current + gap2_offset];
            table[current] = if last_row && last_col {
                rule.fill_corner(s1, s2, &from_match, &from_gap1, &from_gap2)
            } else if last_row {
                rule.fill_end2(s1, s2, &from_match, &from_gap1, &from_gap2)
            } else if last_col {
                rule.fill_end1(s1, s2, &from_match, &from_gap1, &from_gap2)
            } else {
                rule.fill(s1, s2, &from_match, &from_gap1, &from_gap2)
            };
            current += column_step;
        }
    }

    let (score, starts) = rule.find_traceback_starts(&table);

    if F::SCORE_ONLY {
        return (Vec::new(), score);
    }

    let trace_offset = |direction| match direction {
        TraceDirection::Diag => match_offset,
        TraceDirection::To1 => gap1_offset,
        TraceDirection::To2 => gap2_offset,
    };
    let mut paths = Vec::new();
    for (start, state) in starts {
        // The branch counter is reset per start
        // The total number of traces is capped afterwards
        let mut count: usize = 1;
        follow_trace(
            &table,
            start,
            state,
            &trace_offset,
            &mut count,
            max_number,
            &mut paths,
        );
    }
    paths.truncate(max_number);

    let traces = paths
        .into_iter()
        .map(|path| {
            path_to_trace(&path, |index| {
                let (i, j) = table.unindex(index);
                (i as i64 - 1, j as i64 - 1)
            })
        })
        .collect();
    (traces, score)
}

/// Encapsulates how a single DP cell of the full (rectangular) table is computed
/// from its predecessors.
///
/// Implementors use const generic parameters to specialize for local/global
/// alignment, terminal gap penalty handling and score-only mode, so these
/// branches are resolved at compile time rather than per cell.
pub trait FillRule<S: Symbol> {
    /// The cell type this rule produces.
    type Cell: Cell;

    /// Whether only the score is required, so the trace can be omitted.
    const SCORE_ONLY: bool;

    /// Fill the origin cell `(0, 0)`.
    fn fill_origin(&self) -> Self::Cell;

    /// Fill a cell in the first column from the cell above it (a leading gap in
    /// the second sequence).
    fn fill_start1(&self, from_gap2: &Self::Cell) -> Self::Cell;

    /// Fill a cell in the first row from the cell to its left (a leading gap in
    /// the first sequence).
    fn fill_start2(&self, from_gap1: &Self::Cell) -> Self::Cell;

    /// Fill an interior cell from its diagonal, left and top predecessors.
    fn fill(
        &self,
        symbol1: S,
        symbol2: S,
        from_match: &Self::Cell,
        from_gap1: &Self::Cell,
        from_gap2: &Self::Cell,
    ) -> Self::Cell;

    /// Fill a cell in the last column, where a trailing gap in the second
    /// sequence is exempt from penalty (unless terminal gaps are penalized).
    fn fill_end1(
        &self,
        symbol1: S,
        symbol2: S,
        from_match: &Self::Cell,
        from_gap1: &Self::Cell,
        from_gap2: &Self::Cell,
    ) -> Self::Cell;

    /// Fill a cell in the last row, where a trailing gap in the first sequence
    /// is exempt from penalty (unless terminal gaps are penalized).
    fn fill_end2(
        &self,
        symbol1: S,
        symbol2: S,
        from_match: &Self::Cell,
        from_gap1: &Self::Cell,
        from_gap2: &Self::Cell,
    ) -> Self::Cell;

    /// Fill the bottom-right corner cell, where trailing gaps in both sequences
    /// are exempt from penalty (unless terminal gaps are penalized).
    fn fill_corner(
        &self,
        symbol1: S,
        symbol2: S,
        from_match: &Self::Cell,
        from_gap1: &Self::Cell,
        from_gap2: &Self::Cell,
    ) -> Self::Cell;

    /// Find the traceback start position(s) and the optimal score.
    ///
    /// For global alignment this is the bottom-right corner; for local
    /// alignment it is the cell(s) with the maximum score.
    #[allow(clippy::type_complexity)]
    fn find_traceback_starts(
        &self,
        table: &StridedTable<Self::Cell>,
    ) -> (Score, Vec<(TableIndex, <Self::Cell as Cell>::StateInfo)>);
}

/// A [`FillRule`] using a linear gap penalty.
pub struct LinearGapPenalty<S, V, D, const LOCAL: bool, const TERMINAL_PENALTY: bool>
where
    S: Symbol,
    V: ScoringScheme<S>,
    D: LinearDirectionStore,
{
    scoring_scheme: V,
    gap_penalty: Score,
    _phantom: PhantomData<(S, D)>,
}

impl<S, V, D, const LOCAL: bool, const TERMINAL_PENALTY: bool>
    LinearGapPenalty<S, V, D, LOCAL, TERMINAL_PENALTY>
where
    S: Symbol,
    V: ScoringScheme<S>,
    D: LinearDirectionStore,
{
    pub fn new(scoring_scheme: V, gap_penalty: Score) -> Self {
        LinearGapPenalty {
            scoring_scheme,
            gap_penalty,
            _phantom: PhantomData,
        }
    }

    /// The shared recurrence. `WAIVE_GAP1`/`WAIVE_GAP2` drop the gap penalty for
    /// a trailing gap in the first/second sequence (used in the last row/column
    /// when terminal gaps are not penalized).
    #[inline(always)]
    fn recurrence<const WAIVE_GAP1: bool, const WAIVE_GAP2: bool>(
        &self,
        symbol1: S,
        symbol2: S,
        from_match: &LinearCell<D>,
        from_gap1: &LinearCell<D>,
        from_gap2: &LinearCell<D>,
    ) -> LinearCell<D> {
        let similarity = self.scoring_scheme.score(symbol1, symbol2);
        let score_from_match = from_match.score + similarity;
        let score_from_gap1 = if WAIVE_GAP1 {
            from_gap1.score
        } else {
            from_gap1.score + self.gap_penalty
        };
        let score_from_gap2 = if WAIVE_GAP2 {
            from_gap2.score
        } else {
            from_gap2.score + self.gap_penalty
        };
        // Computing the trace directions is only worthwhile if they are needed.
        let (mut score, mut direction) = if D::SCORE_ONLY {
            (
                score_from_match.max(score_from_gap1).max(score_from_gap2),
                D::default(),
            )
        } else {
            let (score, direction) =
                get_trace_linear(score_from_match, score_from_gap1, score_from_gap2);
            (score, D::wrap(direction))
        };
        // Local alignment: a non-positive cell ends the trace and resets to 0.
        if LOCAL && score <= 0 {
            score = 0;
            direction = D::default();
        }
        LinearCell { score, direction }
    }
}

impl<S, V, D, const LOCAL: bool, const TERMINAL_PENALTY: bool> FillRule<S>
    for LinearGapPenalty<S, V, D, LOCAL, TERMINAL_PENALTY>
where
    S: Symbol,
    V: ScoringScheme<S>,
    D: LinearDirectionStore,
{
    type Cell = LinearCell<D>;

    const SCORE_ONLY: bool = D::SCORE_ONLY;

    #[inline]
    fn fill_origin(&self) -> LinearCell<D> {
        LinearCell::default()
    }

    #[inline]
    fn fill_start1(&self, from_gap2: &LinearCell<D>) -> LinearCell<D> {
        // Local alignments must not extend into the leading gap region.
        if LOCAL {
            LinearCell::default()
        } else {
            let score = if TERMINAL_PENALTY {
                from_gap2.score + self.gap_penalty
            } else {
                0
            };
            LinearCell {
                score,
                direction: D::wrap(LinearDirection::GAP2),
            }
        }
    }

    #[inline]
    fn fill_start2(&self, from_gap1: &LinearCell<D>) -> LinearCell<D> {
        if LOCAL {
            LinearCell::default()
        } else {
            let score = if TERMINAL_PENALTY {
                from_gap1.score + self.gap_penalty
            } else {
                0
            };
            LinearCell {
                score,
                direction: D::wrap(LinearDirection::GAP1),
            }
        }
    }

    #[inline(always)]
    fn fill(
        &self,
        symbol1: S,
        symbol2: S,
        from_match: &LinearCell<D>,
        from_gap1: &LinearCell<D>,
        from_gap2: &LinearCell<D>,
    ) -> LinearCell<D> {
        self.recurrence::<false, false>(symbol1, symbol2, from_match, from_gap1, from_gap2)
    }

    #[inline(always)]
    fn fill_end1(
        &self,
        symbol1: S,
        symbol2: S,
        from_match: &LinearCell<D>,
        from_gap1: &LinearCell<D>,
        from_gap2: &LinearCell<D>,
    ) -> LinearCell<D> {
        // A terminal-penalty or local alignment fills this like an ordinary
        // interior cell; only a global alignment without terminal penalty
        // exempts the trailing gap.
        if TERMINAL_PENALTY || LOCAL {
            self.recurrence::<false, false>(symbol1, symbol2, from_match, from_gap1, from_gap2)
        } else {
            self.recurrence::<false, true>(symbol1, symbol2, from_match, from_gap1, from_gap2)
        }
    }

    #[inline(always)]
    fn fill_end2(
        &self,
        symbol1: S,
        symbol2: S,
        from_match: &LinearCell<D>,
        from_gap1: &LinearCell<D>,
        from_gap2: &LinearCell<D>,
    ) -> LinearCell<D> {
        if TERMINAL_PENALTY || LOCAL {
            self.recurrence::<false, false>(symbol1, symbol2, from_match, from_gap1, from_gap2)
        } else {
            self.recurrence::<true, false>(symbol1, symbol2, from_match, from_gap1, from_gap2)
        }
    }

    #[inline(always)]
    fn fill_corner(
        &self,
        symbol1: S,
        symbol2: S,
        from_match: &LinearCell<D>,
        from_gap1: &LinearCell<D>,
        from_gap2: &LinearCell<D>,
    ) -> LinearCell<D> {
        if TERMINAL_PENALTY || LOCAL {
            self.recurrence::<false, false>(symbol1, symbol2, from_match, from_gap1, from_gap2)
        } else {
            self.recurrence::<true, true>(symbol1, symbol2, from_match, from_gap1, from_gap2)
        }
    }

    fn find_traceback_starts(
        &self,
        table: &StridedTable<LinearCell<D>>,
    ) -> (Score, Vec<(TableIndex, ())>) {
        if LOCAL {
            // The start is the highest-scoring cell.
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
        } else {
            // The start is the bottom-right corner
            let shape = table.shape();
            let index = table.index((shape.0 - 1, shape.1 - 1));
            let score = table[index].score;
            (score, vec![(index, ())])
        }
    }
}

/// A [`FillRule`] using an affine gap penalty.
pub struct AffineGapPenalty<S, V, D, const LOCAL: bool, const TERMINAL_PENALTY: bool>
where
    S: Symbol,
    V: ScoringScheme<S>,
    D: AffineDirectionStore,
{
    scoring_scheme: V,
    gap_open: Score,
    gap_extend: Score,
    /// Value representing negative infinity, chosen so that adding a gap penalty
    /// or similarity score during the recurrence can never underflow.
    neg_inf: Score,
    _phantom: PhantomData<(S, D)>,
}

impl<S, V, D, const LOCAL: bool, const TERMINAL_PENALTY: bool>
    AffineGapPenalty<S, V, D, LOCAL, TERMINAL_PENALTY>
where
    S: Symbol,
    V: ScoringScheme<S>,
    D: AffineDirectionStore,
{
    pub fn new(scoring_scheme: V, gap_open: Score, gap_extend: Score) -> Self {
        // The sentinel is offset away from `Score::MIN` by the magnitudes of
        // both gap penalties and the minimum similarity, so that any single
        // addition in the recurrence stays representable. The penalties are
        // guaranteed non-positive by the Python layer.
        let mut neg_inf = Score::MIN - gap_open - gap_extend;
        let min_score = scoring_scheme.min();
        if min_score < 0 {
            neg_inf -= min_score;
        }
        AffineGapPenalty {
            scoring_scheme,
            gap_open,
            gap_extend,
            neg_inf,
            _phantom: PhantomData,
        }
    }

    /// The shared recurrence. `WAIVE_GAP1`/`WAIVE_GAP2` drop the gap penalty for
    /// a trailing gap in the first/second sequence (used in the last row/column
    /// when terminal gaps are not penalized).
    #[inline(always)]
    fn recurrence<const WAIVE_GAP1: bool, const WAIVE_GAP2: bool>(
        &self,
        symbol1: S,
        symbol2: S,
        from_match: &AffineCell<D>,
        from_gap1: &AffineCell<D>,
        from_gap2: &AffineCell<D>,
    ) -> AffineCell<D> {
        let similarity = self.scoring_scheme.score(symbol1, symbol2);
        // Transitions into the match table come from the diagonal predecessor
        let mm = from_match.score.m + similarity;
        let g1m = from_match.score.g1 + similarity;
        let g2m = from_match.score.g2 + similarity;
        // Transitions into the gap-in-1 table come from the left predecessor
        let (mg1, g1g1) = if WAIVE_GAP1 {
            (from_gap1.score.m, from_gap1.score.g1)
        } else {
            (
                from_gap1.score.m + self.gap_open,
                from_gap1.score.g1 + self.gap_extend,
            )
        };
        // Transitions into the gap-in-2 table come from the top predecessor
        let (mg2, g2g2) = if WAIVE_GAP2 {
            (from_gap2.score.m, from_gap2.score.g2)
        } else {
            (
                from_gap2.score.m + self.gap_open,
                from_gap2.score.g2 + self.gap_extend,
            )
        };

        // `direction` stays a raw `AffineDirection` (computed only when traced);
        // it is wrapped into the cell's `D` at the end, so the local masking
        // below is plain bit arithmetic that the score-only build discards.
        let (m_score, g1_score, g2_score, mut direction) = if D::SCORE_ONLY {
            (
                mm.max(g1m).max(g2m),
                mg1.max(g1g1),
                mg2.max(g2g2),
                AffineDirection::empty(),
            )
        } else {
            get_trace_affine(mm, g1m, g2m, mg1, g1g1, mg2, g2g2)
        };
        let mut score = AffineScore {
            m: m_score,
            g1: g1_score,
            g2: g2_score,
        };

        if LOCAL {
            // A non-positive cell ends the trace in the respective table
            if m_score <= 0 {
                direction &= !(AffineDirection::MATCH_TO_MATCH
                    | AffineDirection::GAP1_TO_MATCH
                    | AffineDirection::GAP2_TO_MATCH);
                score.m = 0;
            }
            if g1_score <= 0 {
                direction &= !(AffineDirection::MATCH_TO_GAP1 | AffineDirection::GAP1_TO_GAP1);
                score.g1 = self.neg_inf;
            }
            if g2_score <= 0 {
                direction &= !(AffineDirection::MATCH_TO_GAP2 | AffineDirection::GAP2_TO_GAP2);
                score.g2 = self.neg_inf;
            }
        }

        AffineCell {
            score,
            direction: D::wrap(direction),
        }
    }
}

impl<S, V, D, const LOCAL: bool, const TERMINAL_PENALTY: bool> FillRule<S>
    for AffineGapPenalty<S, V, D, LOCAL, TERMINAL_PENALTY>
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
                m: 0,
                g1: self.neg_inf,
                g2: self.neg_inf,
            },
            direction: D::default(),
        }
    }

    #[inline]
    fn fill_start1(&self, from_gap2: &AffineCell<D>) -> AffineCell<D> {
        if LOCAL {
            // Fill with negative infinity values to prevent that an
            // alignment trace starts with a gap extension
            // instead of a gap opening
            return AffineCell {
                score: AffineScore {
                    m: self.neg_inf,
                    g1: self.neg_inf,
                    g2: 0,
                },
                direction: D::default(),
            };
        }
        // Open the gap from the predecessor's match table, or extend its
        // gap-in-2 table; the larger one wins and dictates the direction
        let (open, extend) = if TERMINAL_PENALTY {
            (
                from_gap2.score.m + self.gap_open,
                from_gap2.score.g2 + self.gap_extend,
            )
        } else {
            (from_gap2.score.m, from_gap2.score.g2)
        };
        let (g2, direction) = if open >= extend {
            (open, AffineDirection::MATCH_TO_GAP2)
        } else {
            (extend, AffineDirection::GAP2_TO_GAP2)
        };
        AffineCell {
            score: AffineScore {
                m: self.neg_inf,
                g1: self.neg_inf,
                g2,
            },
            direction: D::wrap(direction),
        }
    }

    #[inline]
    fn fill_start2(&self, from_gap1: &AffineCell<D>) -> AffineCell<D> {
        if LOCAL {
            return AffineCell {
                score: AffineScore {
                    m: self.neg_inf,
                    g1: 0,
                    g2: self.neg_inf,
                },
                direction: D::default(),
            };
        }
        let (open, extend) = if TERMINAL_PENALTY {
            (
                from_gap1.score.m + self.gap_open,
                from_gap1.score.g1 + self.gap_extend,
            )
        } else {
            (from_gap1.score.m, from_gap1.score.g1)
        };
        let (g1, direction) = if open >= extend {
            (open, AffineDirection::MATCH_TO_GAP1)
        } else {
            (extend, AffineDirection::GAP1_TO_GAP1)
        };
        AffineCell {
            score: AffineScore {
                m: self.neg_inf,
                g1,
                g2: self.neg_inf,
            },
            direction: D::wrap(direction),
        }
    }

    #[inline(always)]
    fn fill(
        &self,
        symbol1: S,
        symbol2: S,
        from_match: &AffineCell<D>,
        from_gap1: &AffineCell<D>,
        from_gap2: &AffineCell<D>,
    ) -> AffineCell<D> {
        self.recurrence::<false, false>(symbol1, symbol2, from_match, from_gap1, from_gap2)
    }

    #[inline(always)]
    fn fill_end1(
        &self,
        symbol1: S,
        symbol2: S,
        from_match: &AffineCell<D>,
        from_gap1: &AffineCell<D>,
        from_gap2: &AffineCell<D>,
    ) -> AffineCell<D> {
        // A terminal-penalty or local alignment fills this like an ordinary
        // interior cell; only a global alignment without terminal penalty
        // exempts the trailing gap
        if TERMINAL_PENALTY || LOCAL {
            self.recurrence::<false, false>(symbol1, symbol2, from_match, from_gap1, from_gap2)
        } else {
            self.recurrence::<false, true>(symbol1, symbol2, from_match, from_gap1, from_gap2)
        }
    }

    #[inline(always)]
    fn fill_end2(
        &self,
        symbol1: S,
        symbol2: S,
        from_match: &AffineCell<D>,
        from_gap1: &AffineCell<D>,
        from_gap2: &AffineCell<D>,
    ) -> AffineCell<D> {
        if TERMINAL_PENALTY || LOCAL {
            self.recurrence::<false, false>(symbol1, symbol2, from_match, from_gap1, from_gap2)
        } else {
            self.recurrence::<true, false>(symbol1, symbol2, from_match, from_gap1, from_gap2)
        }
    }

    #[inline(always)]
    fn fill_corner(
        &self,
        symbol1: S,
        symbol2: S,
        from_match: &AffineCell<D>,
        from_gap1: &AffineCell<D>,
        from_gap2: &AffineCell<D>,
    ) -> AffineCell<D> {
        if TERMINAL_PENALTY || LOCAL {
            self.recurrence::<false, false>(symbol1, symbol2, from_match, from_gap1, from_gap2)
        } else {
            self.recurrence::<true, true>(symbol1, symbol2, from_match, from_gap1, from_gap2)
        }
    }

    fn find_traceback_starts(
        &self,
        table: &StridedTable<AffineCell<D>>,
    ) -> (Score, Vec<(TableIndex, TraceState)>) {
        if LOCAL {
            // Only the match table is considered:
            // a local alignment cannot start or end with a gap
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
        } else {
            // The start is the bottom-right corner; the trace may start in any
            // of the three tables that reaches the maximum score.
            let shape = table.shape();
            let index = table.index((shape.0 - 1, shape.1 - 1));
            let cell = table[index];
            let max = cell.score.m.max(cell.score.g1).max(cell.score.g2);
            let mut starts = Vec::new();
            for state in [TraceState::Match, TraceState::Gap1, TraceState::Gap2] {
                if table[index].score.get(state) == max {
                    starts.push((index, state));
                }
            }
            (max, starts)
        }
    }
}
