//! Local ungapped seed extension ([`SeedExtension`]).
//!
//! Extends an ungapped alignment outward from a seed position in both
//! directions, accumulating the substitution score symbol by symbol and
//! stopping in each direction once the running score falls more than
//! `threshold` below the best score seen so far (*X-Drop* :footcite:`Zhang2000`).
//!
//! The scoring scheme, threshold and direction are preprocessed once when the
//! [`SeedExtension`] is constructed (in particular the substitution matrix is
//! extracted only once), so repeated [`align`](SeedExtension::align) calls — the
//! typical seeding workload — avoid that per-call overhead. Each `align` takes
//! the two [`Sequence`](biotite) objects directly, widens their codes to a
//! common dtype, walks the upstream/downstream regions via zero-copy slice
//! iterators and returns the resulting [`Alignment`](biotite) (or just the
//! score). As the alignment is ungapped (a single diagonal), no dynamic
//! programming table or [`Cell`](super::cell::Cell) is required.

use super::scoring::{Score, Scoring, ScoringScheme};
use super::symbol::Symbol;
use crate::dispatch_dtype;
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

/// A reusable local ungapped seed extension (*X-Drop*).
///
/// The scoring scheme, threshold and direction are preprocessed once on
/// construction, so the same :class:`SeedExtension` can run :meth:`align` on
/// multiple sequence pairs.
///
/// Parameters
/// ----------
/// matrix : SubstitutionMatrix or tuple(int, int)
///     Either a substitution matrix or a ``(match, mismatch)`` pair of scores.
/// threshold : int
///     If the current score falls this value below the maximum score
///     found, the alignment terminates.
/// direction : {'both', 'upstream', 'downstream'}, optional
///     Controls in which direction the alignment extends starting from the
///     seed.
///     If ``'upstream'``, the alignment ends at the seed; if ``'downstream'``,
///     it starts at the seed; if ``'both'`` (default) it extends in both
///     directions.
///     The seed position itself is always included in the alignment.
///
/// See Also
/// --------
/// align_local_gapped
///     For gapped local alignments with the same *X-Drop* technique.
///
/// Examples
/// --------
///
/// >>> seq1 = ProteinSequence("BIQTITE")
/// >>> seq2 = ProteinSequence("PYRRHQTITE")
/// >>> matrix = SubstitutionMatrix.std_protein_matrix()
/// >>> extension = SeedExtension(matrix, threshold=10)
/// >>> print(extension.align(seq1, seq2, seed=(4, 7)))
/// QTITE
/// QTITE
/// >>> print(extension.align(seq1, seq2, seed=(4, 7), score_only=True))
/// 24
#[pyclass]
pub struct SeedExtension {
    scoring: Scoring,
    threshold: Score,
    upstream: bool,
    downstream: bool,
}

#[pymethods]
impl SeedExtension {
    #[new]
    #[pyo3(signature = (matrix, threshold, direction = "both"))]
    fn new(matrix: Scoring, threshold: Score, direction: &str) -> PyResult<Self> {
        if threshold < 0 {
            return Err(PyValueError::new_err(
                "The threshold value must be a non-negative integer",
            ));
        }
        let (upstream, downstream) = match direction {
            "both" => (true, true),
            "upstream" => (true, false),
            "downstream" => (false, true),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Direction '{direction}' is invalid"
                )));
            }
        };
        Ok(SeedExtension {
            scoring: matrix,
            threshold,
            upstream,
            downstream,
        })
    }

    /// Extend the alignment from `seed` between `seq1` and `seq2`.
    ///
    /// Parameters
    /// ----------
    /// seq1, seq2 : Sequence
    ///     The sequences to be aligned.
    /// seed : tuple(int, int)
    ///     The indices in `seq1` and `seq2` where the alignment starts.
    ///     The indices must be non-negative and in bounds.
    /// score_only : bool, optional
    ///     If set to ``True``, only the similarity score is returned instead
    ///     of the :class:`Alignment`.
    ///
    /// Returns
    /// -------
    /// alignment : Alignment
    ///     The resulting ungapped alignment.
    ///     Only returned, if `score_only` is ``False``.
    /// score : int
    ///     The alignment similarity score.
    ///     Only returned, if `score_only` is ``True``.
    #[pyo3(signature = (seq1, seq2, seed, score_only = false))]
    fn align<'py>(
        &self,
        py: Python<'py>,
        seq1: &Bound<'py, PyAny>,
        seq2: &Bound<'py, PyAny>,
        seed: &Bound<'py, PyAny>,
        score_only: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Accept any indexable seed (e.g. a tuple or a NumPy array)
        let i: i64 = seed.get_item(0)?.extract()?;
        let j: i64 = seed.get_item(1)?.extract()?;
        if i < 0 || j < 0 {
            return Err(PyIndexError::new_err("Seed must contain positive indices"));
        }
        let (i, j) = (i as usize, j as usize);
        let len1 = seq1.len()?;
        let len2 = seq2.len()?;
        if i >= len1 || j >= len2 {
            return Err(PyIndexError::new_err(format!(
                "Seed {:?} is out of bounds for the sequences of length {len1} and {len2}",
                (i, j)
            )));
        }

        // The two sequences may use different code dtypes
        // -> widen both to a common type so the kernel is dispatched once
        let np = py.import("numpy")?;
        let code1_obj = seq1.getattr("code")?;
        let code2_obj = seq2.getattr("code")?;
        let common_dtype = np.call_method1(
            "promote_types",
            (code1_obj.getattr("dtype")?, code2_obj.getattr("dtype")?),
        )?;
        let code1 = np.call_method1("ascontiguousarray", (code1_obj, &common_dtype))?;
        let code2 = np.call_method1("ascontiguousarray", (code2_obj, &common_dtype))?;
        let code1 = code1.cast::<PyUntypedArray>()?;
        let code2 = code2.cast::<PyUntypedArray>()?;

        let dtype = code1.dtype();
        dispatch_dtype!(
            py,
            &dtype,
            [u8, u16, u32, u64],
            align_dtype(py, self, seq1, seq2, code1, code2, (i, j), score_only)
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn align_dtype<'py, S: Symbol + numpy::Element>(
    py: Python<'py>,
    extension: &SeedExtension,
    seq1: &Bound<'py, PyAny>,
    seq2: &Bound<'py, PyAny>,
    code1: &Bound<'py, PyUntypedArray>,
    code2: &Bound<'py, PyUntypedArray>,
    seed: (usize, usize),
    score_only: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let code1_array = code1.cast::<PyArray1<S>>()?.readonly();
    let code2_array = code2.cast::<PyArray1<S>>()?.readonly();
    let code1_slice = code1_array.as_slice()?;
    let code2_slice = code2_array.as_slice()?;

    let (total_score, start_offset, stop_offset) = match &extension.scoring {
        Scoring::Matrix(scheme) => seed_extend(extension, code1_slice, code2_slice, scheme, seed),
        Scoring::Match(scheme) => seed_extend(extension, code1_slice, code2_slice, scheme, seed),
    };

    if score_only {
        return Ok(total_score.into_pyobject(py)?.into_any());
    }

    // The alignment is ungapped, so the trace is a contiguous diagonal
    let (i, j) = seed;
    let length = (stop_offset - start_offset) as usize;
    let mut trace = Array2::<i64>::zeros((length, 2));
    for k in 0..length {
        let offset = start_offset + k as i64;
        trace[[k, 0]] = i as i64 + offset;
        trace[[k, 1]] = j as i64 + offset;
    }
    let trace = trace.into_pyarray(py);

    let sequences = PyList::new(py, [seq1, seq2])?;
    let alignment = py
        .import("biotite.sequence.align.alignment")?
        .getattr("Alignment")?
        .call1((sequences, trace, total_score))?;
    Ok(alignment)
}

/// Core seed extension, generic over the symbol type and scoring scheme.
///
/// Returns the total score (including the seed) and the seed-relative
/// `start_offset`/`stop_offset` of the aligned diagonal.
fn seed_extend<S, V>(
    extension: &SeedExtension,
    code1: &[S],
    code2: &[S],
    scoring: &V,
    seed: (usize, usize),
) -> (Score, i64, i64)
where
    S: Symbol,
    V: ScoringScheme<S>,
{
    let (i, j) = seed;
    // The seed position itself is always part of the alignment
    let mut total_score = scoring.score(code1[i], code2[j]);
    let mut start_offset: i64 = 0;
    let mut stop_offset: i64 = 1;

    if extension.upstream {
        // Walk the prefixes before the seed in reverse (zero-copy)
        let pairs = code1[..i]
            .iter()
            .rev()
            .zip(code2[..j].iter().rev())
            .map(|(&s1, &s2)| (s1, s2));
        let (score, length) = extend(scoring, extension.threshold, pairs);
        total_score += score;
        start_offset = -length;
    }
    if extension.downstream {
        // Walk the suffixes after the seed
        let pairs = code1[i + 1..]
            .iter()
            .zip(code2[j + 1..].iter())
            .map(|(&s1, &s2)| (s1, s2));
        let (score, length) = extend(scoring, extension.threshold, pairs);
        total_score += score;
        stop_offset = 1 + length;
    }

    (total_score, start_offset, stop_offset)
}

/// Run the X-Drop extension over a stream of aligned symbol pairs.
///
/// Returns the maximum score reached and the number of pairs up to (and
/// including) that maximum.
#[inline]
fn extend<S, V, I>(scoring: &V, threshold: Score, pairs: I) -> (Score, i64)
where
    S: Symbol,
    V: ScoringScheme<S>,
    I: Iterator<Item = (S, S)>,
{
    let mut total_score: Score = 0;
    let mut max_score: Score = 0;
    let mut length: i64 = 0;
    for (offset, (s1, s2)) in pairs.enumerate() {
        total_score += scoring.score(s1, s2);
        if total_score >= max_score {
            // A new maximum (ties extend the alignment over zero-score columns)
            max_score = total_score;
            length = (offset + 1) as i64;
        } else if max_score - total_score > threshold {
            // Score dropped too far below the maximum -> terminate
            break;
        }
    }
    (max_score, length)
}
