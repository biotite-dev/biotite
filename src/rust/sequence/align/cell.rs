//! Dynamic programming cell types shared by all alignment algorithms.
//!
//! A [`Cell`] is a single element of a Needleman-Wunsch-like DP table, holding
//! the score(s) and the trace information. The per-cell recurrence (the
//! `FillRule`) lives in each alignment module, since each algorithm has
//! different requirements; the only computation shared here is the
//! max-selection primitive ([`get_trace_linear`] / [`get_trace_affine`]).

use super::scoring::Score;
use bitflags::bitflags;
use smallvec::SmallVec;

/// Traceback steps returned by [`Cell::traceback`], parameterized by the cell's
/// [`StateInfo`](Cell::StateInfo).
///
/// At most three co-optimal directions are produced per cell (match / gap-in-1 /
/// gap-in-2; the affine recurrence likewise yields at most three transitions for
/// any single state), so they are kept inline without heap allocation.
pub type TraceVec<S> = SmallVec<[(TraceDirection, S); 3]>;

/// A direction the traceback moves from a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceDirection {
    /// Diagonal: alignment of two symbols.
    Diag,
    /// Gap in the first sequence (move along a row).
    To1,
    /// Gap in the second sequence (move along a column).
    To2,
}

/// The score table the traceback is currently in.
/// Required for affine gap penalty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceState {
    /// The match table.
    Match,
    /// The table for gaps in the first sequence.
    Gap1,
    /// The table for gaps in the second sequence.
    Gap2,
}

bitflags! {
    /// Traceback directions for linear gap penalty (one score table).
    ///
    /// A set bit means the cell can be reached from that direction.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct LinearDirection: u8 {
        /// Diagonal: alignment of two symbols.
        const MATCH = 1;
        /// Gap in the first sequence.
        const GAP1 = 2;
        /// Gap in the second sequence.
        const GAP2 = 4;
    }
}

bitflags! {
    /// Traceback transitions for affine gap penalty (three score tables).
    ///
    /// Each bit encodes a transition between two tables/states.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct AffineDirection: u8 {
        /// Match -> match transition.
        const MATCH_TO_MATCH = 1;
        /// Gap in sequence 1 -> match transition.
        const GAP1_TO_MATCH = 2;
        /// Gap in sequence 2 -> match transition.
        const GAP2_TO_MATCH = 4;
        /// Match -> gap in sequence 1 transition.
        const MATCH_TO_GAP1 = 8;
        /// Gap in sequence 1 -> gap in sequence 1 transition (extension).
        const GAP1_TO_GAP1 = 16;
        /// Match -> gap in sequence 2 transition.
        const MATCH_TO_GAP2 = 32;
        /// Gap in sequence 2 -> gap in sequence 2 transition (extension).
        const GAP2_TO_GAP2 = 64;
    }
}

/// A single element of a dynamic programming table.
///
/// It tracks both the score(s) needed to fill the table and the trace
/// information needed to reconstruct the alignment.
pub trait Cell: Copy + Default {
    /// Extra information required to resume the traceback from this cell.
    type StateInfo: Copy;

    /// The predecessor steps to follow during traceback, given the state the
    /// traceback arrived in.
    ///
    /// Each step is the direction to move and the state to continue in.
    /// An empty result terminates the (branch of the) traceback.
    /// The order matters: the first step is the one followed by the main trace,
    /// further steps spawn additional alignment branches.
    fn traceback(&self, state_info: Self::StateInfo) -> TraceVec<Self::StateInfo>;
}

/// The trace-direction storage of a [`LinearCell`].
///
/// This is what lets a score-only table cell shrink to just its score: a full
/// alignment stores the real [`LinearDirection`] bitflags, while a score-only
/// alignment stores the zero-sized `()` placeholder, halving the cell size.
pub trait LinearDirectionStore: Copy + Default {
    /// Whether the cell is score-only, storing no trace directions (and thus
    /// not computing them).
    const SCORE_ONLY: bool;

    /// Wrap a freshly computed direction, discarding it in score-only mode.
    fn wrap(direction: LinearDirection) -> Self;

    /// The traceback steps encoded by the stored direction.
    fn traceback(&self) -> TraceVec<()>;
}

impl LinearDirectionStore for LinearDirection {
    const SCORE_ONLY: bool = false;

    #[inline(always)]
    fn wrap(direction: LinearDirection) -> Self {
        direction
    }

    #[inline]
    fn traceback(&self) -> TraceVec<()> {
        let mut steps = TraceVec::new();
        if self.contains(LinearDirection::MATCH) {
            steps.push((TraceDirection::Diag, ()));
        }
        if self.contains(LinearDirection::GAP1) {
            steps.push((TraceDirection::To1, ()));
        }
        if self.contains(LinearDirection::GAP2) {
            steps.push((TraceDirection::To2, ()));
        }
        steps
    }
}

impl LinearDirectionStore for () {
    const SCORE_ONLY: bool = true;

    #[inline(always)]
    fn wrap(_direction: LinearDirection) -> Self {}

    fn traceback(&self) -> TraceVec<()> {
        unreachable!("score-only cells are never traced back")
    }
}

/// A DP cell for linear gap penalty: a single score plus trace directions.
///
/// `D` is the [`LinearDirectionStore`]: [`LinearDirection`] for a full
/// alignment, or `()` in score-only mode (shrinking the cell to its score).
#[derive(Debug, Clone, Copy, Default)]
pub struct LinearCell<D = LinearDirection> {
    pub score: Score,
    pub direction: D,
}

impl<D: LinearDirectionStore> Cell for LinearCell<D> {
    type StateInfo = ();

    #[inline]
    fn traceback(&self, _state_info: Self::StateInfo) -> TraceVec<Self::StateInfo> {
        self.direction.traceback()
    }
}

/// The three scores of an affine [`AffineCell`], one per score table.
#[derive(Debug, Clone, Copy, Default)]
pub struct AffineScore {
    /// Score in the match table.
    pub m: Score,
    /// Score in the gap-in-sequence-1 table.
    pub g1: Score,
    /// Score in the gap-in-sequence-2 table.
    pub g2: Score,
}

impl AffineScore {
    /// The score of the table corresponding to the given traceback state.
    #[inline(always)]
    pub fn get(&self, state: TraceState) -> Score {
        match state {
            TraceState::Match => self.m,
            TraceState::Gap1 => self.g1,
            TraceState::Gap2 => self.g2,
        }
    }
}

/// The trace-transition storage of an [`AffineCell`].
///
/// The affine analogue of [`LinearDirectionStore`]: [`AffineDirection`] for a
/// full alignment, or `()` in score-only mode.
pub trait AffineDirectionStore: Copy + Default {
    /// Whether the cell is score-only, storing no trace transitions (and thus
    /// not computing them).
    const SCORE_ONLY: bool;

    /// Wrap a freshly computed transition, discarding it in score-only mode.
    fn wrap(direction: AffineDirection) -> Self;

    /// The traceback steps encoded by the stored transition for `state`.
    fn traceback(&self, state: TraceState) -> TraceVec<TraceState>;
}

impl AffineDirectionStore for AffineDirection {
    const SCORE_ONLY: bool = false;

    #[inline(always)]
    fn wrap(direction: AffineDirection) -> Self {
        direction
    }

    #[inline]
    fn traceback(&self, state: TraceState) -> TraceVec<TraceState> {
        let d = *self;
        let mut steps = TraceVec::new();
        // Only the transitions valid for the current state are followed; within
        // a state, every step moves in the same direction.
        match state {
            TraceState::Match => {
                if d.contains(AffineDirection::MATCH_TO_MATCH) {
                    steps.push((TraceDirection::Diag, TraceState::Match));
                }
                if d.contains(AffineDirection::GAP1_TO_MATCH) {
                    steps.push((TraceDirection::Diag, TraceState::Gap1));
                }
                if d.contains(AffineDirection::GAP2_TO_MATCH) {
                    steps.push((TraceDirection::Diag, TraceState::Gap2));
                }
            }
            TraceState::Gap1 => {
                if d.contains(AffineDirection::MATCH_TO_GAP1) {
                    steps.push((TraceDirection::To1, TraceState::Match));
                }
                if d.contains(AffineDirection::GAP1_TO_GAP1) {
                    steps.push((TraceDirection::To1, TraceState::Gap1));
                }
            }
            TraceState::Gap2 => {
                if d.contains(AffineDirection::MATCH_TO_GAP2) {
                    steps.push((TraceDirection::To2, TraceState::Match));
                }
                if d.contains(AffineDirection::GAP2_TO_GAP2) {
                    steps.push((TraceDirection::To2, TraceState::Gap2));
                }
            }
        }
        steps
    }
}

impl AffineDirectionStore for () {
    const SCORE_ONLY: bool = true;

    #[inline(always)]
    fn wrap(_direction: AffineDirection) -> Self {}

    fn traceback(&self, _state: TraceState) -> TraceVec<TraceState> {
        unreachable!("score-only cells are never traced back")
    }
}

/// A DP cell for affine gap penalty: three scores (match, gap-in-1, gap-in-2)
/// plus trace transitions.
///
/// `D` is the [`AffineDirectionStore`]: [`AffineDirection`] for a full
/// alignment, or `()` in score-only mode (shrinking the cell to its scores).
#[derive(Debug, Clone, Copy, Default)]
pub struct AffineCell<D = AffineDirection> {
    pub score: AffineScore,
    pub direction: D,
}

impl<D: AffineDirectionStore> Cell for AffineCell<D> {
    type StateInfo = TraceState;

    #[inline]
    fn traceback(&self, state_info: Self::StateInfo) -> TraceVec<Self::StateInfo> {
        self.direction.traceback(state_info)
    }
}

/// Pick the maximum of three scores for linear gap penalty and return the
/// corresponding set of trace directions (ties yield multiple directions).
#[inline(always)]
pub(crate) fn get_trace_linear(
    match_score: Score,
    gap1_score: Score,
    gap2_score: Score,
) -> (Score, LinearDirection) {
    use LinearDirection as D;
    // NOTE: `if`/`else` is deliberately used instead of `match score.cmp(...)`:
    // Matching on `Ordering` materializes a three-valued result and reorders the
    // comparisons, which the optimizer turns into measurably slower code (rust-lang/rust#86511)
    if match_score > gap1_score {
        if match_score > gap2_score {
            (match_score, D::MATCH)
        } else if match_score == gap2_score {
            (match_score, D::MATCH | D::GAP2)
        } else {
            (gap2_score, D::GAP2)
        }
    } else if match_score == gap1_score {
        if match_score > gap2_score {
            (match_score, D::MATCH | D::GAP1)
        } else if match_score == gap2_score {
            (match_score, D::MATCH | D::GAP1 | D::GAP2)
        } else {
            (gap2_score, D::GAP2)
        }
    } else if gap1_score > gap2_score {
        (gap1_score, D::GAP1)
    } else if gap1_score == gap2_score {
        (gap1_score, D::GAP1 | D::GAP2)
    } else {
        (gap2_score, D::GAP2)
    }
}

/// Pick the maximum scores for the three affine tables and return the
/// corresponding trace transitions.
///
/// Returns `(match_score, gap1_score, gap2_score, directions)`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_trace_affine(
    mm: Score,
    g1m: Score,
    g2m: Score,
    mg1: Score,
    g1g1: Score,
    mg2: Score,
    g2g2: Score,
) -> (Score, Score, Score, AffineDirection) {
    use AffineDirection as D;
    // NOTE: `if`/`else` is deliberately used instead of `match score.cmp(...)`:
    // Matching on `Ordering` materializes a three-valued result and reorders the
    // comparisons, which the optimizer turns into measurably slower code (rust-lang/rust#86511)
    // Match table
    let (m_score, mut trace) = if mm > g1m {
        if mm > g2m {
            (mm, D::MATCH_TO_MATCH)
        } else if mm == g2m {
            (mm, D::MATCH_TO_MATCH | D::GAP2_TO_MATCH)
        } else {
            (g2m, D::GAP2_TO_MATCH)
        }
    } else if mm == g1m {
        if mm > g2m {
            (mm, D::MATCH_TO_MATCH | D::GAP1_TO_MATCH)
        } else if mm == g2m {
            (mm, D::MATCH_TO_MATCH | D::GAP1_TO_MATCH | D::GAP2_TO_MATCH)
        } else {
            (g2m, D::GAP2_TO_MATCH)
        }
    } else if g1m > g2m {
        (g1m, D::GAP1_TO_MATCH)
    } else if g1m == g2m {
        (g1m, D::GAP1_TO_MATCH | D::GAP2_TO_MATCH)
    } else {
        (g2m, D::GAP2_TO_MATCH)
    };

    // 'Gap left' table
    let g1_score = if mg1 > g1g1 {
        trace |= D::MATCH_TO_GAP1;
        mg1
    } else if mg1 < g1g1 {
        trace |= D::GAP1_TO_GAP1;
        g1g1
    } else {
        trace |= D::MATCH_TO_GAP1 | D::GAP1_TO_GAP1;
        mg1
    };

    // 'Gap top' table
    let g2_score = if mg2 > g2g2 {
        trace |= D::MATCH_TO_GAP2;
        mg2
    } else if mg2 < g2g2 {
        trace |= D::GAP2_TO_GAP2;
        g2g2
    } else {
        trace |= D::MATCH_TO_GAP2 | D::GAP2_TO_GAP2;
        mg2
    };

    (m_score, g1_score, g2_score, trace)
}
