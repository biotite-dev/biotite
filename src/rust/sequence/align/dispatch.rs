//! Runtime dispatch from the dynamic Python arguments to the statically
//! monomorphized alignment core.
//!
//! The Python layer passes a scoring scheme and a gap penalty whose concrete
//! types (and the local/terminal/score-only flags) are only known at run time,
//! whereas the alignment core is generic over the [`ScoringScheme`], the
//! [`FillRule`] and the const flags. The macros and [`dispatch_scoring`] here
//! bridge the two by matching each runtime value and calling the core with the
//! corresponding concrete types.

use super::cell::{AffineGapPenalty, LinearGapPenalty};
use super::pairwise::align_optimal_generic;
use super::scoring::{Match, Score, SubstitutionMatrix};
use super::symbol::Symbol;
use numpy::ndarray::Array2;
use pyo3::prelude::*;

/// The gap penalty configuration passed from the Python layer.
#[derive(FromPyObject)]
pub enum GapPenalty {
    /// A single linear gap penalty.
    Linear(Score),
    /// An affine gap penalty as `(gap_open, gap_extend)`.
    Affine(Score, Score),
}

/// A scoring scheme accepted from the Python layer: either a substitution
/// matrix (a 2D array) or a match/mismatch tuple.
#[derive(FromPyObject)]
pub enum Scoring {
    Matrix(SubstitutionMatrix),
    Match(Match),
}

/// Instantiate the fill rule for each `(local, terminal, score_only)` const
/// combination and call the alignment core `$align`.
macro_rules! dispatch_consts {
    (
        $align:path, $rule:ident, $scheme:expr, ($($penalty:expr),+),
        $local:expr, $terminal:expr, $score_only:expr,
        $code1:expr, $code2:expr, $max:expr
    ) => {
        match ($local, $terminal, $score_only) {
            (true, true, true) => $align(
                $code1, $code2, <$rule<_, _, true, true, true>>::new($scheme, $($penalty),+), $max,
            ),
            (true, true, false) => $align(
                $code1, $code2, <$rule<_, _, true, true, false>>::new($scheme, $($penalty),+), $max,
            ),
            (true, false, true) => $align(
                $code1, $code2, <$rule<_, _, true, false, true>>::new($scheme, $($penalty),+), $max,
            ),
            (true, false, false) => $align(
                $code1, $code2, <$rule<_, _, true, false, false>>::new($scheme, $($penalty),+), $max,
            ),
            (false, true, true) => $align(
                $code1, $code2, <$rule<_, _, false, true, true>>::new($scheme, $($penalty),+), $max,
            ),
            (false, true, false) => $align(
                $code1, $code2, <$rule<_, _, false, true, false>>::new($scheme, $($penalty),+), $max,
            ),
            (false, false, true) => $align(
                $code1, $code2, <$rule<_, _, false, false, true>>::new($scheme, $($penalty),+), $max,
            ),
            (false, false, false) => $align(
                $code1, $code2, <$rule<_, _, false, false, false>>::new($scheme, $($penalty),+), $max,
            ),
        }
    };
}

/// Select the concrete fill rule from the gap penalty, then dispatch the const
/// flags via [`dispatch_consts`].
macro_rules! dispatch_penalty {
    (
        $align:path, $scheme:expr, $gap:expr,
        $local:expr, $terminal:expr, $score_only:expr,
        $code1:expr, $code2:expr, $max:expr
    ) => {
        match $gap {
            GapPenalty::Linear(gap) => dispatch_consts!(
                $align,
                LinearGapPenalty,
                $scheme,
                (gap),
                $local,
                $terminal,
                $score_only,
                $code1,
                $code2,
                $max
            ),
            GapPenalty::Affine(open, extend) => dispatch_consts!(
                $align,
                AffineGapPenalty,
                $scheme,
                (open, extend),
                $local,
                $terminal,
                $score_only,
                $code1,
                $code2,
                $max
            ),
        }
    };
}

/// Dispatch over the scoring scheme and run [`align_optimal_generic`] for the
/// resulting concrete scheme, gap penalty and const flags.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_scoring<S: Symbol>(
    scoring: Scoring,
    gap_penalty: GapPenalty,
    local: bool,
    terminal_penalty: bool,
    score_only: bool,
    code1: &[S],
    code2: &[S],
    max_number: usize,
) -> (Vec<Array2<i64>>, Score) {
    match scoring {
        Scoring::Matrix(scheme) => dispatch_penalty!(
            align_optimal_generic,
            scheme,
            gap_penalty,
            local,
            terminal_penalty,
            score_only,
            code1,
            code2,
            max_number
        ),
        Scoring::Match(scheme) => dispatch_penalty!(
            align_optimal_generic,
            scheme,
            gap_penalty,
            local,
            terminal_penalty,
            score_only,
            code1,
            code2,
            max_number
        ),
    }
}
