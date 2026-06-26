//! Scoring schemes for sequence alignment.
//!
//! A [`ScoringScheme`] assigns a similarity score to a pair of symbols.
//! Two implementations are provided: [`SubstitutionMatrix`] for general
//! matrix-based scoring and [`Match`] for simple binary match/mismatch scoring.

use super::symbol::Symbol;
use numpy::ndarray::ArrayView2;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::Borrowed;

/// The universal type used to represent alignment scores.
pub type Score = i32;

/// A scheme that assigns a similarity score to a pair of symbols.
pub trait ScoringScheme<S: Symbol> {
    /// The similarity score for aligning `symbol1` with `symbol2`.
    fn score(&self, symbol1: S, symbol2: S) -> Score;

    /// The minimum score this scheme can return.
    ///
    /// This is used to compute a safe 'negative-infinity' sentinel score
    /// without risking integer underflow.
    fn min(&self) -> Score;
}

/// Scoring by lookup in a substitution matrix.
///
/// The matrix is stored in row-major (strided) format for efficient
/// lookup.
pub struct SubstitutionMatrix {
    /// Flat row-major score buffer of length `n_rows * n_cols`.
    data: Vec<Score>,
    /// Number of columns, i.e. the size of the second alphabet.
    n_cols: usize,
    /// Precomputed minimum entry, returned by [`ScoringScheme::min`].
    min: Score,
}

impl SubstitutionMatrix {
    /// Create a matrix from a 2D score array.
    ///
    /// The number of rows is the size of the first sequence's alphabet, the
    /// number of columns the size of the second.
    pub fn new(matrix: ArrayView2<Score>) -> Self {
        let n_rows = matrix.shape()[0];
        let n_cols = matrix.shape()[1];
        let mut data = Vec::with_capacity(n_rows * n_cols);
        for row in 0..n_rows {
            for col in 0..n_cols {
                data.push(matrix[[row, col]]);
            }
        }
        let min = data.iter().copied().min().unwrap_or(0);
        SubstitutionMatrix { data, n_cols, min }
    }
}

impl<S: Symbol> ScoringScheme<S> for SubstitutionMatrix {
    #[inline(always)]
    fn score(&self, symbol1: S, symbol2: S) -> Score {
        self.data[symbol1.index() * self.n_cols + symbol2.index()]
    }

    #[inline(always)]
    fn min(&self) -> Score {
        self.min
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for SubstitutionMatrix {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Accept either a raw 2D score array (as produced by the Python layer)
        // or a `SubstitutionMatrix` object exposing `score_matrix()`
        if let Ok(matrix) = obj.extract::<PyReadonlyArray2<Score>>() {
            return Ok(SubstitutionMatrix::new(matrix.as_array()));
        }
        let matrix: PyReadonlyArray2<Score> = obj.getattr("score_matrix")?.call0()?.extract()?;
        Ok(SubstitutionMatrix::new(matrix.as_array()))
    }
}

/// A simple binary match/mismatch scoring scheme.
pub struct Match {
    match_score: Score,
    mismatch_score: Score,
}

impl Match {
    pub fn new(match_score: Score, mismatch_score: Score) -> Self {
        Match {
            match_score,
            mismatch_score,
        }
    }
}

impl<S: Symbol> ScoringScheme<S> for Match {
    #[inline(always)]
    fn score(&self, symbol1: S, symbol2: S) -> Score {
        if symbol1 == symbol2 {
            self.match_score
        } else {
            self.mismatch_score
        }
    }

    #[inline(always)]
    fn min(&self) -> Score {
        self.match_score.min(self.mismatch_score)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Match {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let (match_score, mismatch_score): (Score, Score) = obj.extract()?;
        Ok(Match::new(match_score, mismatch_score))
    }
}

/// The gap penalty configuration passed from the Python layer.
///
/// The concrete variant is only known at run time, so each alignment module
/// matches on it to dispatch to its own statically monomorphized fill rule.
#[derive(FromPyObject)]
pub enum GapPenalty {
    /// A single linear gap penalty.
    Linear(Score),
    /// An affine gap penalty as `(gap_open, gap_extend)`.
    Affine(Score, Score),
}

/// A scoring scheme accepted from the Python layer: either a [`SubstitutionMatrix`]
/// (a 2D array) or a [`Match`]/mismatch tuple.
///
/// Like [`GapPenalty`], the variant is resolved at run time and drives the
/// dispatch to a monomorphized fill rule.
#[derive(FromPyObject)]
pub enum Scoring {
    Matrix(SubstitutionMatrix),
    Match(Match),
}
