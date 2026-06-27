//! Hot paths for [`KmerAlphabet`](biotite).
//!
//! [`create_kmers`] decomposes a sequence code into the codes of all its
//! overlapping *k-mers* (continuous or spaced), and [`split_kmers`] is the
//! inverse of the vectorized *fuse* operation: it splits *k-mer* codes back into
//! the base-alphabet symbol codes. Both are tight integer loops that do not
//! vectorize well in NumPy, so they live in Rust while the surrounding
//! `KmerAlphabet` class stays in Python.

use crate::dispatch_dtype;
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray1,
    PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// Label as a separate module to indicate that this exception comes
// from biotite
mod biotite {
    pyo3::import_exception!(biotite.sequence, AlphabetError);
}

/// Build a Python `AlphabetError` with the given message.
#[cold]
#[inline(never)]
fn alphabet_error(message: String) -> PyErr {
    biotite::AlphabetError::new_err(message)
}

/// Check that every symbol code in `seq_code` is within the base alphabet,
/// returning an `AlphabetError` for the first one that is not.
fn validate_codes<T>(seq_code: &[T], alphabet_length: usize) -> PyResult<()>
where
    T: Copy,
    u64: From<T>,
{
    for &code in seq_code {
        let code = u64::from(code);
        if code >= alphabet_length as u64 {
            return Err(alphabet_error(format!(
                "Symbol code {code} is out of range"
            )));
        }
    }
    Ok(())
}

/// The radix multipliers ``[n^(k-1), n^(k-2), ..., n^0]`` of a `KmerAlphabet`,
/// where ``n`` is the base alphabet length. Position `i` within a *k-mer* is
/// weighted by ``radix[i]``.
fn radix_multipliers(k: usize, alphabet_length: usize) -> Vec<i64> {
    let mut radix = vec![0i64; k];
    let mut value: i64 = 1;
    for slot in radix.iter_mut().rev() {
        *slot = value;
        value *= alphabet_length as i64;
    }
    radix
}

/// create_kmers(seq_code, k, alphabet_length, spacing=None)
///
/// Create the *k-mer* codes for all overlapping *k-mers* in `seq_code`.
///
/// Parameters
/// ----------
/// seq_code : ndarray, dtype={uint8, uint16, uint32, uint64}
///     The sequence code to decompose.
/// k : int
///     The number of informative positions per *k-mer*.
/// alphabet_length : int
///     The number of symbols in the base alphabet.
/// spacing : ndarray, dtype=int64, optional
///     The informative positions of a spaced *k-mer* model, sorted ascending.
///     If omitted, continuous *k-mers* are created.
///
/// Returns
/// -------
/// kmers : ndarray, dtype=int64
///     The *k-mer* codes.
#[pyfunction]
#[pyo3(signature = (seq_code, k, alphabet_length, spacing=None))]
pub fn create_kmers<'py>(
    py: Python<'py>,
    seq_code: &Bound<'py, PyUntypedArray>,
    k: usize,
    alphabet_length: usize,
    spacing: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<Bound<'py, PyAny>> {
    let spacing = spacing.as_ref().map(|s| s.as_slice()).transpose()?;
    let dtype = seq_code.dtype();
    dispatch_dtype!(
        py,
        &dtype,
        [u8, u16, u32, u64],
        create_kmers_dtype(py, seq_code, k, alphabet_length, spacing)
    )
}

fn create_kmers_dtype<'py, T>(
    py: Python<'py>,
    seq_code: &Bound<'py, PyUntypedArray>,
    k: usize,
    alphabet_length: usize,
    spacing: Option<&[i64]>,
) -> PyResult<Bound<'py, PyArray1<i64>>>
where
    T: numpy::Element + Copy,
    u64: From<T>,
{
    let array = seq_code.cast::<PyArray1<T>>()?.readonly();
    let seq_code = array.as_slice()?;
    let kmers = match spacing {
        None => create_continuous_kmers(seq_code, k, alphabet_length)?,
        Some(spacing) => create_spaced_kmers(seq_code, k, alphabet_length, spacing)?,
    };
    Ok(kmers.into_pyarray(py))
}

/// Decompose `seq_code` into continuous *k-mers*.
///
/// Each *k-mer* is computed from the previous one by dropping the leftmost
/// symbol, shifting and appending the next symbol, so the loop runs only over
/// the sequence length.
fn create_continuous_kmers<T>(
    seq_code: &[T],
    k: usize,
    alphabet_length: usize,
) -> PyResult<Vec<i64>>
where
    T: Copy,
    u64: From<T>,
{
    if seq_code.len() < k {
        return Err(PyValueError::new_err(
            "The length of the sequence code is shorter than k",
        ));
    }
    let radix = radix_multipliers(k, alphabet_length);
    let end_radix = radix[0];
    let n_kmers = seq_code.len() - k + 1;
    // Reserve without zero-initializing; every element is written via `push`
    let mut kmers = Vec::with_capacity(n_kmers);

    // Validate inline rather than in a separate pass: this loop is O(len), so a
    // second full pass over `seq_code` would roughly double its runtime
    let checked_code = |code: T| -> PyResult<i64> {
        let code = u64::from(code);
        if code >= alphabet_length as u64 {
            Err(alphabet_error(format!(
                "Symbol code {code} is out of range"
            )))
        } else {
            Ok(code as i64)
        }
    };

    // First k-mer via the naive approach
    let mut kmer: i64 = 0;
    for (&r, &code) in radix.iter().zip(&seq_code[..k]) {
        kmer += r * checked_code(code)?;
    }
    kmers.push(kmer);

    // All following k-mers from the previous one: each k-mer drops the symbol
    // `k` positions back and appends the next one.
    // Only the appended symbol needs checking, as the dropped one was validated when it was
    // appended earlier.
    // Zipping the two offset slices avoids the per-index bounds checks and visits
    // each symbol exactly once.
    let dropped_codes = &seq_code[..n_kmers - 1];
    let new_codes = &seq_code[k..];
    for (&dropped_code, &new_code) in dropped_codes.iter().zip(new_codes) {
        let dropped = u64::from(dropped_code) as i64;
        let new = checked_code(new_code)?;
        kmer =
            // Remove first symbol
            (kmer - dropped * end_radix)
            // Shift k-mer to left
            * (alphabet_length as i64)
            // Add new symbol
            + new;
        kmers.push(kmer);
    }
    Ok(kmers)
}

/// Decompose `seq_code` into spaced *k-mers* using the informative positions in
/// `spacing`.
fn create_spaced_kmers<T>(
    seq_code: &[T],
    k: usize,
    alphabet_length: usize,
    spacing: &[i64],
) -> PyResult<Vec<i64>>
where
    T: Copy,
    u64: From<T>,
{
    // The last (largest) informative position defines the total k-mer span
    let max_offset = spacing[spacing.len() - 1] as usize + 1;
    if seq_code.len() < max_offset {
        return Err(PyValueError::new_err(
            "The length of the sequence code is shorter than the k-mer span",
        ));
    }
    // Validate all symbol codes once up front (every position is read by some
    // k-mer), so the hot loop below is a branchless multiply-accumulate
    validate_codes(seq_code, alphabet_length)?;
    let radix = radix_multipliers(k, alphabet_length);
    let n_kmers = seq_code.len() - max_offset + 1;
    // Reserve without zero-initializing; every element is written via `push`
    let mut kmers = Vec::with_capacity(n_kmers);

    for i in 0..n_kmers {
        let mut kmer: i64 = 0;
        // Zipping `spacing` and `radix` elides their bounds checks in the loop
        for (&offset, &r) in spacing.iter().zip(&radix) {
            let code = u64::from(seq_code[i + offset as usize]);
            kmer += r * code as i64;
        }
        kmers.push(kmer);
    }
    Ok(kmers)
}

/// split_kmers(codes, k, alphabet_length)
///
/// Split *k-mer* codes into the base-alphabet symbol codes, i.e. the inverse of
/// the *fuse* operation.
///
/// The caller must ensure that all `codes` are valid for the alphabet; no
/// bounds checking is performed.
///
/// Parameters
/// ----------
/// codes : ndarray, dtype=int64
///     The *k-mer* codes to split.
/// k : int
///     The number of symbols per *k-mer*.
/// alphabet_length : int
///     The number of symbols in the base alphabet.
///
/// Returns
/// -------
/// split_codes : ndarray, dtype=uint64, shape=(n, k)
///     The base-alphabet symbol codes for each *k-mer*.
#[pyfunction]
pub fn split_kmers<'py>(
    py: Python<'py>,
    codes: PyReadonlyArray1<'py, i64>,
    k: usize,
    alphabet_length: usize,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    let codes = codes.as_slice()?;
    let total = codes.len() * k;
    let mut split: Vec<u64> = vec![0; total];
    for (row, &code) in codes.iter().enumerate() {
        // Extract the base-alphabet symbols least-significant by repeatedly
        // dividing by the alphabet length
        let mut remainder = code as usize;
        let base = row * k;
        for n in (0..k).rev() {
            let symbol = remainder % alphabet_length;
            remainder /= alphabet_length;
            split[base + n] = symbol as u64;
        }
    }
    let split =
        Array2::from_shape_vec((codes.len(), k), split).expect("flat buffer length is n * k");
    Ok(split.into_pyarray(py))
}
