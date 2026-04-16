//! Banded pairwise alignment (`align_banded`).
//!
//! Only the cells within a diagonal band `lower_diag..=upper_diag` (with
//! `diag = j - i`) are filled. The band is 'straightened' into a table of shape
//! `(seq1_len + 1, band_width + 2)`, where the two extra columns are
//! out-of-band sentinels.
//!
//! Banded alignment is always either local or semi-global (terminal gaps are
//! never penalized) and fills every in-band cell with the same interior
//! recurrence, so its [`FillRule`] is simpler than the rectangular one in
//! [`pairwise`](super::pairwise): no first-row/column or terminal-edge methods,
//! and only the `LOCAL` and `SCORE_ONLY` const generics.
//!
//! The Python layer guarantees `seq1` is the shorter sequence and supplies the
//! cropped band.

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

/// Perform a local or semi-global banded pairwise alignment.
///
/// Parameters
/// ----------
/// code1, code2 : ndarray
///     The sequence codes, sharing the same unsigned integer dtype. `code1`
///     must be the shorter sequence.
/// scoring : ndarray, dtype=int32 or tuple(int, int)
///     A substitution matrix or a `(match, mismatch)` score pair.
/// gap_penalty : int or tuple(int, int)
///     A linear or affine gap penalty.
/// lower_diag, upper_diag : int
///     The (already cropped) lower and upper band diagonals.
/// local : bool
///     Whether to perform a local (instead of semi-global) alignment.
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
///     The optimal alignment score within the band.
#[pyfunction]
#[pyo3(signature = (code1, code2, scoring, gap_penalty, lower_diag, upper_diag, local, score_only, max_number))]
#[allow(clippy::too_many_arguments)]
pub fn align_banded<'py>(
    py: Python<'py>,
    code1: &Bound<'py, PyUntypedArray>,
    code2: &Bound<'py, PyUntypedArray>,
    scoring: Scoring,
    gap_penalty: GapPenalty,
    lower_diag: isize,
    upper_diag: isize,
    local: bool,
    score_only: bool,
    max_number: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let dtype = code1.dtype();
    dispatch_dtype!(
        py,
        &dtype,
        [u8, u16, u32, u64],
        align_banded_dtype(
            py,
            code1,
            code2,
            scoring,
            gap_penalty,
            lower_diag,
            upper_diag,
            local,
            score_only,
            max_number
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn align_banded_dtype<'py, S: Symbol + numpy::Element>(
    py: Python<'py>,
    code1: &Bound<'py, PyUntypedArray>,
    code2: &Bound<'py, PyUntypedArray>,
    scoring: Scoring,
    gap_penalty: GapPenalty,
    lower_diag: isize,
    upper_diag: isize,
    local: bool,
    score_only: bool,
    max_number: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    let code1_array = code1.cast::<PyArray1<S>>()?.readonly();
    let code2_array = code2.cast::<PyArray1<S>>()?.readonly();
    let code1 = code1_array.as_slice()?;
    let code2 = code2_array.as_slice()?;

    // Instantiate the fill rule for the runtime scoring scheme, gap penalty,
    // `local` and `score_only` flags, then run the generic core. The
    // direction-storage type (`$dir` for a full alignment, `()` for score-only)
    // selects the cell size.
    macro_rules! run {
        ($rule:ident, $dir:ty, $local:literal, $scheme:expr, ($($penalty:expr),+)) => {
            align_banded_generic(
                code1, code2,
                <$rule<_, _, $dir, $local>>::new($scheme, $($penalty),+),
                lower_diag, upper_diag, max_number,
            )
        };
    }
    macro_rules! by_consts {
        ($rule:ident, $dir:ty, $scheme:expr, ($($penalty:expr),+)) => {
            match (local, score_only) {
                (true, true) => run!($rule, (), true, $scheme, ($($penalty),+)),
                (true, false) => run!($rule, $dir, true, $scheme, ($($penalty),+)),
                (false, true) => run!($rule, (), false, $scheme, ($($penalty),+)),
                (false, false) => run!($rule, $dir, false, $scheme, ($($penalty),+)),
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

/// Core banded alignment, generic over the symbol type and fill rule.
pub(crate) fn align_banded_generic<S, F>(
    code1: &[S],
    code2: &[S],
    rule: F,
    lower_diag: isize,
    upper_diag: isize,
    max_number: usize,
) -> (Vec<Array2<i64>>, Score)
where
    S: Symbol,
    F: FillRule<S>,
{
    let n = code1.len();
    let m = code2.len();
    let band_width = (upper_diag - lower_diag + 1) as usize;

    // The 'straightened' band: rows are the first sequence, the `band_width`
    // interior columns are the band, flanked by two out-of-band sentinel
    // columns (0 and `band_width + 1`)
    // Initialize the interior with the edge value, then overwrite the two sentinel columns
    let mut table = StridedTable::filled((n + 1, band_width + 2), rule.fill_edges());
    for i in 0..=n {
        let left = table.index((i, 0));
        let right = table.index((i, band_width + 1));
        table[left] = rule.fill_sentinel();
        table[right] = rule.fill_sentinel();
    }

    // Constant offsets to the three predecessors in band coordinates:
    // Due to the straightening, the diagonal/left/top neighbors become these:
    let match_offset = table.index_offset((-1, 0));
    let gap1_offset = table.index_offset((0, -1));
    let gap2_offset = table.index_offset((-1, 1));
    let column_step = table.index_offset((0, 1));

    for (seq_i, &s1) in code1.iter().enumerate() {
        let i = seq_i + 1;
        let seq_j_start = (seq_i as isize + lower_diag).max(0) as usize;
        let seq_j_end = ((seq_i as isize + upper_diag + 1).min(m as isize)).max(0) as usize;
        // Consecutive `seq_j` map to consecutive band columns, so walk the index
        // instead of recomputing it per cell
        let j_start = seq_j_start.wrapping_add_signed(1 - seq_i as isize - lower_diag);
        let mut current = table.index((i, j_start));
        for &s2 in code2.iter().take(seq_j_end).skip(seq_j_start) {
            let from_match = table[current + match_offset];
            let from_gap1 = table[current + gap1_offset];
            let from_gap2 = table[current + gap2_offset];
            table[current] = rule.fill(s1, s2, &from_match, &from_gap1, &from_gap2);
            current += column_step;
        }
    }

    let (score, starts) = rule.find_traceback_starts(&table, n, m, lower_diag, upper_diag);

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
                // 'Unstraighthen' the table indices to sequence indices
                let seq_i = i as i64 - 1;
                let seq_j = j as i64 + seq_i + lower_diag as i64 - 1;
                (seq_i, seq_j)
            })
        })
        .collect();
    (traces, score)
}

/// Analogous to the [`FillRule`](super::pairwise::FillRule) in `pairwise.rs`
pub trait FillRule<S: Symbol> {
    type Cell: Cell;

    const SCORE_ONLY: bool;

    /// The initial value of the in-band cells.
    ///
    /// Most of the cells are overwritten by [`fill`]
    /// But the out-of-band triangles are read from this state.
    fn fill_edges(&self) -> Self::Cell;

    fn fill_sentinel(&self) -> Self::Cell;

    fn fill(
        &self,
        symbol1: S,
        symbol2: S,
        from_match: &Self::Cell,
        from_gap1: &Self::Cell,
        from_gap2: &Self::Cell,
    ) -> Self::Cell;

    #[allow(clippy::type_complexity)]
    fn find_traceback_starts(
        &self,
        table: &StridedTable<Self::Cell>,
        seq1_len: usize,
        seq2_len: usize,
        lower_diag: isize,
        upper_diag: isize,
    ) -> (Score, Vec<(TableIndex, <Self::Cell as Cell>::StateInfo)>);
}

pub struct LinearGapPenalty<S, V, D, const LOCAL: bool>
where
    S: Symbol,
    V: ScoringScheme<S>,
    D: LinearDirectionStore,
{
    scoring_scheme: V,
    gap_penalty: Score,
    neg_inf: Score,
    _phantom: PhantomData<(S, D)>,
}

impl<S, V, D, const LOCAL: bool> LinearGapPenalty<S, V, D, LOCAL>
where
    S: Symbol,
    V: ScoringScheme<S>,
    D: LinearDirectionStore,
{
    pub fn new(scoring_scheme: V, gap_penalty: Score) -> Self {
        let mut neg_inf = Score::MIN - gap_penalty;
        let min_score = scoring_scheme.min();
        if min_score < 0 {
            neg_inf -= min_score;
        }
        LinearGapPenalty {
            scoring_scheme,
            gap_penalty,
            neg_inf,
            _phantom: PhantomData,
        }
    }
}

impl<S, V, D, const LOCAL: bool> FillRule<S> for LinearGapPenalty<S, V, D, LOCAL>
where
    S: Symbol,
    V: ScoringScheme<S>,
    D: LinearDirectionStore,
{
    type Cell = LinearCell<D>;

    const SCORE_ONLY: bool = D::SCORE_ONLY;

    #[inline]
    fn fill_edges(&self) -> LinearCell<D> {
        LinearCell::default()
    }

    #[inline]
    fn fill_sentinel(&self) -> LinearCell<D> {
        LinearCell {
            score: self.neg_inf,
            direction: D::default(),
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
        let similarity = self.scoring_scheme.score(symbol1, symbol2);
        let score_from_match = from_match.score + similarity;
        let score_from_gap1 = from_gap1.score + self.gap_penalty;
        let score_from_gap2 = from_gap2.score + self.gap_penalty;
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
        if LOCAL && score <= 0 {
            score = 0;
            direction = D::default();
        }
        LinearCell { score, direction }
    }

    fn find_traceback_starts(
        &self,
        table: &StridedTable<LinearCell<D>>,
        seq1_len: usize,
        seq2_len: usize,
        lower_diag: isize,
        upper_diag: isize,
    ) -> (Score, Vec<(TableIndex, ())>) {
        if LOCAL {
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
            let potential_starts = get_potential_starts(seq1_len, seq2_len, lower_diag, upper_diag);
            let mut max = Score::MIN;
            for &(i, j) in &potential_starts {
                max = max.max(table[table.index((i, j))].score);
            }
            let starts = potential_starts
                .iter()
                .map(|&(i, j)| table.index((i, j)))
                .filter(|&index| table[index].score == max)
                .map(|index| (index, ()))
                .collect();
            (max, starts)
        }
    }
}

pub struct AffineGapPenalty<S, V, D, const LOCAL: bool>
where
    S: Symbol,
    V: ScoringScheme<S>,
    D: AffineDirectionStore,
{
    scoring_scheme: V,
    gap_open: Score,
    gap_extend: Score,
    neg_inf: Score,
    _phantom: PhantomData<(S, D)>,
}

impl<S, V, D, const LOCAL: bool> AffineGapPenalty<S, V, D, LOCAL>
where
    S: Symbol,
    V: ScoringScheme<S>,
    D: AffineDirectionStore,
{
    pub fn new(scoring_scheme: V, gap_open: Score, gap_extend: Score) -> Self {
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
}

impl<S, V, D, const LOCAL: bool> FillRule<S> for AffineGapPenalty<S, V, D, LOCAL>
where
    S: Symbol,
    V: ScoringScheme<S>,
    D: AffineDirectionStore,
{
    type Cell = AffineCell<D>;

    const SCORE_ONLY: bool = D::SCORE_ONLY;

    #[inline]
    fn fill_edges(&self) -> AffineCell<D> {
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
    fn fill_sentinel(&self) -> AffineCell<D> {
        AffineCell {
            score: AffineScore {
                m: self.neg_inf,
                g1: self.neg_inf,
                g2: self.neg_inf,
            },
            direction: D::default(),
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
        let similarity = self.scoring_scheme.score(symbol1, symbol2);
        let mm = from_match.score.m + similarity;
        let g1m = from_match.score.g1 + similarity;
        let g2m = from_match.score.g2 + similarity;
        let mg1 = from_gap1.score.m + self.gap_open;
        let g1g1 = from_gap1.score.g1 + self.gap_extend;
        let mg2 = from_gap2.score.m + self.gap_open;
        let g2g2 = from_gap2.score.g2 + self.gap_extend;

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
            // Local alignment specialty:
            // If score is less than or equal to 0, then the score of the cell remains 0
            // and the trace ends here
            if m_score <= 0 {
                // End trace by filtering out the respective bits
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

    fn find_traceback_starts(
        &self,
        table: &StridedTable<AffineCell<D>>,
        seq1_len: usize,
        seq2_len: usize,
        lower_diag: isize,
        upper_diag: isize,
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
            let potential_starts = get_potential_starts(seq1_len, seq2_len, lower_diag, upper_diag);
            let mut max = Score::MIN;
            for &(i, j) in &potential_starts {
                let cell = table[table.index((i, j))];
                max = max.max(cell.score.m).max(cell.score.g1).max(cell.score.g2);
            }
            let mut starts = Vec::new();
            for state in [TraceState::Match, TraceState::Gap1, TraceState::Gap2] {
                for &(i, j) in &potential_starts {
                    let index = table.index((i, j));
                    if table[index].score.get(state) == max {
                        starts.push((index, state));
                    }
                }
            }
            (max, starts)
        }
    }
}

/// Potential starting points for a semi-global banded alignment.
/// These are cells that touch the end of either sequence.
fn get_potential_starts(
    seq1_len: usize,
    seq2_len: usize,
    lower_diag: isize,
    upper_diag: isize,
) -> Vec<(usize, usize)> {
    let band_width = (upper_diag - lower_diag + 1) as usize;
    let mut starts = Vec::with_capacity(band_width);
    // Start from the end from the first (shorter) sequence,
    // if the table cell is in bounds of the second (longer) sequence,
    // otherwise start from the end of the second sequence
    for j in 1..=band_width as isize {
        let seq_j = j + (seq1_len as isize - 1) + lower_diag - 1;
        let i = if seq_j < seq2_len as isize {
            // The cell touches the end of the first (shorter) sequence.
            seq1_len as isize
        } else {
            // The cell touches the end of the second sequence; solve
            // `seq2_len - 1 = j + (i - 1) + lower_diag - 1` for `i`.
            //
            // Take:
            //
            // seq_j = j + (seq1_len-1) + lower_diag - 1
            //
            // Replace seq_j with last sequence position of second sequence
            // and last sequence position of first sequence with seq_i:
            //
            // (seq2_len-1) = j + seq_i + lower_diag - 1
            //
            // Replace seq_i with corresponding i in table:
            //
            // (seq2_len-1) = j + (i - 1) + lower_diag - 1
            //
            // Solve for i:
            //
            seq2_len as isize - j - lower_diag + 1
        };
        starts.push((i as usize, j as usize));
    }
    starts
}
