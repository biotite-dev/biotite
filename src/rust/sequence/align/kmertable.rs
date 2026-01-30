//! Containers for efficient retrieval of k-mer matches.

use super::nested::NestedArray;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::iter::zip;

// Label as a separate module to indicate that this exception comes
// from biotite
mod biotite {
    pyo3::import_exception!(biotite.sequence, AlphabetError);
}

#[derive(Default)]
struct KmerTableElement {
    ref_id: u32,
    position: u32,
}

/// A thin wrapper around the Python `KmerAlphabet` class.
/// It re-exposes the Python methods to Rust.
struct KmerAlphabet {
    wrapped: Py<PyAny>,
}
impl KmerAlphabet {
    fn new(
        py: Python<'_>,
        alphabet: Bound<'_, PyAny>,
        k: usize,
        spacing: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let module = PyModule::import(py, "biotite.sequence.align.kmeralphabet")?;
        let kmer_alphabet = module
            .getattr("KmerAlphabet")?
            .call1((alphabet, k, spacing))?;
        Ok(Self {
            wrapped: kmer_alphabet.unbind(),
        })
    }

    fn len(&self, py: Python<'_>) -> PyResult<usize> {
        let wrapped = self.wrapped.bind(py);
        let len = wrapped.call_method0("__len__")?.extract()?;
        Ok(len)
    }

    fn create_kmers<'py>(
        &self,
        py: Python<'py>,
        sequence: &Bound<'py, PyAny>,
    ) -> PyResult<PyReadonlyArray1<'py, i64>> {
        let wrapped = self.wrapped.bind(py);
        let sequence_code = sequence.getattr("code")?;
        let py_array: Bound<'py, PyArray1<i64>> = wrapped
            .call_method1("create_kmers", (sequence_code,))?
            .extract()?;
        Ok(py_array.readonly())
    }

    fn kmer_array_length(&self, py: Python<'_>, seq_length: usize) -> PyResult<usize> {
        let wrapped = self.wrapped.bind(py);
        let length: usize = wrapped
            .call_method1("kmer_array_length", (seq_length,))?
            .extract()?;
        Ok(length)
    }

    fn k(&self, py: Python<'_>) -> PyResult<usize> {
        let wrapped = self.wrapped.bind(py);
        wrapped.getattr("k")?.extract()
    }

    fn spacing<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<i64>>> {
        let wrapped = self.wrapped.bind(py);
        let spacing = wrapped.getattr("spacing")?;
        if spacing.is_none() {
            Ok(None)
        } else {
            let spacing_array: Bound<'py, PyArray1<i64>> = spacing.extract()?;
            Ok(Some(spacing_array.to_vec()?))
        }
    }
}

/// This class represents a *k-mer* index table.
/// It maps *k-mers* (subsequences with length *k*) to the sequence
/// positions, where the *k-mer* appears.
/// It is primarily used to find *k-mer* matches between two sequences.
/// A match is defined as a *k-mer* that appears in both sequences.
/// Instances of this class are immutable.
///
/// There are multiple ways to create a :class:`KmerTable`:
///
///     - :meth:`from_sequences()` iterates through all overlapping
///         *k-mers* in a sequence and stores the sequence position of
///         each *kmer* in the table.
///     - :meth:`from_kmers()` is similar to :meth:`from_sequences()`
///         but directly accepts *k-mers* as input instead of sequences.
///     - :meth:`from_kmer_selection()` takes a combination of *k-mers*
///         and their positions in a sequence, which can be used to
///         apply subset selectors, such as :class:`MinimizerSelector`.
///     - :meth:`from_tables()` merges the entries from multiple
///         :class:`KmerTable` objects into a new table.
///     - :meth:`from_positions()` let's the user provide manual
///         *k-mer* positions, which can be useful for loading a
///         :class:`KmerTable` from file.
///
/// Each indexed *k-mer* position is represented by a tuple of
///
///     1. a unique reference ID that identifies to which sequence a
///         position refers to and
///     2. the zero-based sequence position of the first symbol in the
///         *k-mer*.
///
/// The :meth:`match()` method iterates through all overlapping *k-mers*
/// in another sequence and, for each *k-mer*, looks up the reference
/// IDs and positions of this *k-mer* in the table.
/// For each matching position, it adds the *k-mer* position in this
/// sequence, the matching reference ID and the matching sequence
/// position to the array of matches.
/// Finally these matches are returned to the user.
/// Optionally, a :class:`SimilarityRule` can be supplied, to find
/// also matches for similar *k-mers*.
/// This is especially useful for protein sequences to match two
/// *k-mers* with a high substitution probability.
///
/// The positions for a given *k-mer* code can be obtained via indexing.
/// Iteration over a :class:`KmerTable` yields the *k-mers* that have at
/// least one associated position.
/// The *k-mer* code for a *k-mer* can be calculated with
/// ``table.kmer_alphabet.encode()`` (see :class:`KmerAlphabet`).
///
/// Attributes
/// ----------
/// kmer_alphabet : KmerAlphabet
///     The internal :class:`KmerAlphabet`, that is used to
///     encode all overlapping *k-mers* of an input sequence.
/// alphabet : Alphabet
///     The base alphabet, from which this :class:`KmerTable` was
///     created.
/// k : int
///     The length of the *k-mers*.
///
/// See Also
/// --------
/// BucketKmerTable
///
/// Notes
/// -----
///
/// The design of the :class:`KmerTable` is inspired by the *MMseqs2*
/// software :footcite:`Steinegger2017`.
///
/// *Memory consumption*
///
/// For efficient mapping, a :class:`KmerTable` contains two large arrays:
///
/// 1. An array that holds all *k-mer* positions.
/// 2. An accompanying array that maps each *k-mer* to the index range in the first
/// array, where the corresponding *k-mer* positions are stored.
///
/// The required memory space :math:`S` in byte is
///
/// .. math::
///
///     S = 8 n^k + 8L,
///
/// where :math:`n` is the number of symbols in the alphabet and
/// :math:`L` is the summed length of all sequences added to the table.
///
/// *Multiprocessing*
///
/// :class:`KmerTable` objects can be used in multi-processed setups:
/// Adding a large database of sequences to a table can be sped up by
/// splitting the database into smaller chunks and create a separate
/// table for each chunk in separate processes.
/// Eventually, the tables can be merged to one large table using
/// :meth:`from_tables()`.
///
/// Since :class:`KmerTable` supports the *pickle* protocol,
/// the matching step can also be divided into multiple processes, if
/// multiple sequences need to be matched.
///
/// *Storage on hard drive*
///
/// The most time efficient way to read/write a :class:`KmerTable` is
/// the *pickle* format.
/// If a custom format is desired, the user needs to extract the
/// reference IDs and position for each *k-mer*.
/// To restrict this task to all *k-mers* that have at least one match
/// :meth:`get_kmers()` can be used.
/// Conversely, the reference IDs and positions can be restored via
/// :meth:`from_positions()`.
///
/// References
/// ----------
///
/// .. footbibliography::
///
/// Examples
/// --------
///
/// Create a *2-mer* index table for some nucleotide sequences:
///
/// >>> table = KmerTable.from_sequences(
/// ...     k = 2,
/// ...     sequences = [NucleotideSequence("TTATA"), NucleotideSequence("CTAG")],
/// ...     ref_ids = [0, 1]
/// ... )
///
/// Display the contents of the table as
/// (reference ID, sequence position) tuples:
///
/// >>> print(table)
/// AG: (1, 2)
/// AT: (0, 2)
/// CT: (1, 0)
/// TA: (0, 1), (0, 3), (1, 1)
/// TT: (0, 0)
///
/// Find matches of the table with a sequence:
///
/// >>> query = NucleotideSequence("TAG")
/// >>> matches = table.match(query)
/// >>> for query_pos, table_ref_id, table_pos in matches:
/// ...     print("Query sequence position:", query_pos)
/// ...     print("Table reference ID:  ", table_ref_id)
/// ...     print("Table sequence position:", table_pos)
/// ...     print()
/// Query sequence position: 0
/// Table reference ID: 0
/// Table sequence position: 1
/// <BLANKLINE>
/// Query sequence position: 0
/// Table reference ID: 0
/// Table sequence position: 3
/// <BLANKLINE>
/// Query sequence position: 0
/// Table reference ID: 1
/// Table sequence position: 1
/// <BLANKLINE>
/// Query sequence position: 1
/// Table reference ID: 1
/// Table sequence position: 2
/// <BLANKLINE>
///
/// Get all reference IDs and positions for a given *k-mer*:
///
/// >>> kmer_code = table.kmer_alphabet.encode("TA")
/// >>> print(table[kmer_code])
/// [[0 1]
///  [0 3]
///  [1 1]]
#[pyclass]
pub struct KmerTable {
    k: usize,
    kmer_alphabet: KmerAlphabet,
    table: NestedArray<KmerTableElement>,
}

#[pymethods]
impl KmerTable {
    /// from_sequences(k, sequences, ref_ids=None, ignore_masks=None, alphabet=None, spacing=None)
    ///
    /// Create a :class:`KmerTable` by storing the positions of all
    /// overlapping *k-mers* from the input `sequences`.
    ///
    /// Parameters
    /// ----------
    /// k : int
    ///     The length of the *k-mers*.
    /// sequences : sized iterable object of Sequence, length=m
    ///     The sequences to get the *k-mer* positions from.
    ///     These sequences must have equal alphabets, or one of these
    ///     sequences must have an alphabet that extends the alphabets
    ///     of all other sequences.
    /// ref_ids : sized iterable object of int, length=m, optional
    ///     The reference IDs for the given sequences.
    ///     These are used to identify the corresponding sequence for a
    ///     *k-mer* match.
    ///     By default the IDs are counted from *0* to *m*.
    /// ignore_masks : sized iterable object of (ndarray, dtype=bool), length=m, optional
    ///     Sequence positions to ignore.
    ///     *k-mers* that involve these sequence positions are not added
    ///     to the table.
    ///     This is used e.g. to skip repeat regions.
    ///     If provided, the list must contain one boolean mask
    ///     (or ``None``) for each sequence, and each bolean mask must
    ///     have the same length as the sequence.
    ///     By default, no sequence position is ignored.
    /// alphabet : Alphabet, optional
    ///     The alphabet to use for this table.
    ///     It must extend the alphabets of the input `sequences`.
    ///     By default, an appropriate alphabet is inferred from the
    ///     input `sequences`.
    ///     This option is usually used for compatibility with another
    ///     sequence/table in the matching step.
    /// spacing : None or str or list or ndarray, dtype=int, shape=(k,)
    ///     If provided, spaced *k-mers* are used instead of continuous
    ///     ones.
    ///     The value contains the *informative* positions relative to
    ///     the start of the *k-mer*, also called the *model*.
    ///     The number of *informative* positions must equal *k*.
    ///     Refer to :class:`KmerAlphabet` for more details.
    ///
    /// See Also
    /// --------
    /// from_kmers : The same functionality based on already created *k-mers*
    ///
    /// Returns
    /// -------
    /// table : KmerTable
    ///     The newly created table.
    ///
    /// Examples
    /// --------
    ///
    /// >>> sequences = [NucleotideSequence("TTATA"), NucleotideSequence("CTAG")]
    /// >>> table = KmerTable.from_sequences(
    /// ...     2, sequences, ref_ids=[100, 101]
    /// ... )
    /// >>> print(table)
    /// AG: (101, 2)
    /// AT: (100, 2)
    /// CT: (101, 0)
    /// TA: (100, 1), (100, 3), (101, 1)
    /// TT: (100, 0)
    ///
    /// Give an explicit compatible alphabet:
    ///
    /// >>> table = KmerTable.from_sequences(
    /// ...     2, sequences, ref_ids=[100, 101],
    /// ...     alphabet=NucleotideSequence.ambiguous_alphabet()
    /// ... )
    ///
    /// Ignore all ``N`` in a sequence:
    ///
    /// >>> sequence = NucleotideSequence("ACCNTANNG")
    /// >>> table = KmerTable.from_sequences(
    /// ...     2, [sequence], ignore_masks=[sequence.symbols == "N"]
    /// ... )
    /// >>> print(table)
    /// AC: (0, 0)
    /// CC: (0, 1)
    /// TA: (0, 4)
    #[staticmethod]
    #[pyo3(signature = (k, sequences, ref_ids=None, ignore_masks=None, alphabet=None, spacing=None))]
    fn from_sequences(
        py: Python<'_>,
        k: usize,
        sequences: Vec<Bound<'_, PyAny>>,
        ref_ids: Option<Vec<u32>>,
        ignore_masks: Option<Vec<Option<Bound<'_, PyAny>>>>,
        alphabet: Option<Bound<'_, PyAny>>,
        spacing: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let n_sequences = sequences.len();
        let ref_ids: Vec<u32> = compute_ref_ids(ref_ids, n_sequences)?;
        let ignore_masks: Vec<Option<Bound<'_, PyAny>>> = compute_masks(ignore_masks, n_sequences)?;
        let alphabet = get_alphabet(py, alphabet, &sequences)?;
        let kmer_alphabet = KmerAlphabet::new(py, alphabet, k, spacing)?;

        // Calculate k-mers and masks for all sequences
        let mut kmers_list: Vec<PyReadonlyArray1<'_, i64>> = Vec::with_capacity(n_sequences);
        let mut masks_list: Vec<Vec<bool>> = Vec::with_capacity(n_sequences);
        for (sequence, ignore_mask) in sequences.iter().zip(ignore_masks.iter()) {
            let seq_length: usize = sequence.call_method0("__len__")?.extract()?;
            let kmers = kmer_alphabet.create_kmers(py, sequence)?;
            let mask = convert_ignore_into_include_mask(
                py,
                &kmer_alphabet,
                ignore_mask.as_ref(),
                seq_length,
            )?;
            kmers_list.push(kmers);
            masks_list.push(mask);
        }

        // Count the number of appearances of each k-mer
        let mut counts: Vec<usize> = init_counts(py, &kmer_alphabet)?;
        for (kmers, mask) in kmers_list.iter().zip(masks_list.iter()) {
            for (&kmer, &is_included) in zip(kmers.as_slice()?.iter(), mask.iter()) {
                if is_included {
                    counts[kmer as usize] += 1;
                }
            }
        }

        // Create NestedArray with appropriate sizes
        let mut table: NestedArray<KmerTableElement> = NestedArray::new(counts);
        // Fill the table with k-mer positions
        // Track current write position for each k-mer
        let n_kmers = kmer_alphabet.len(py)?;
        let mut write_indices: Vec<usize> = vec![0; n_kmers];
        for ((kmers, mask), ref_id) in kmers_list.iter().zip(masks_list.iter()).zip(ref_ids.iter())
        {
            for (seq_pos, (&kmer, &is_included)) in
                zip(kmers.as_slice()?.iter(), mask.iter()).enumerate()
            {
                if is_included {
                    let kmer_idx = kmer as usize;
                    let write_idx = write_indices[kmer_idx];
                    table[kmer_idx][write_idx] = KmerTableElement {
                        ref_id: *ref_id,
                        position: seq_pos as u32,
                    };
                    write_indices[kmer_idx] += 1;
                }
            }
        }

        Ok(Self {
            k,
            kmer_alphabet,
            table,
        })
    }
}

/// Initialize a vector of counts for each k-mer.
fn init_counts(py: Python<'_>, kmer_alphabet: &KmerAlphabet) -> PyResult<Vec<usize>> {
    let n_kmers = kmer_alphabet.len(py)?;
    // Reserve with one extra element to make `NestedArray::new()` more efficient
    let mut counts = Vec::with_capacity(n_kmers + 1);
    // Initialize each k-mer count with 0
    counts.resize(n_kmers, 0usize);
    Ok(counts)
}

/// Compute reference IDs: if None, create sequential IDs from 0 to n_sequences-1.
fn compute_ref_ids(ref_ids: Option<Vec<u32>>, n_sequences: usize) -> PyResult<Vec<u32>> {
    match ref_ids {
        Some(ids) => {
            if ids.len() != n_sequences {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "{} reference IDs were given, but there are {} sequences",
                    ids.len(),
                    n_sequences
                )));
            }
            Ok(ids)
        }
        None => Ok((0..n_sequences as u32).collect()),
    }
}

/// Compute ignore masks: if None, create a list of None values.
fn compute_masks(
    masks: Option<Vec<Option<Bound<'_, PyAny>>>>,
    n_sequences: usize,
) -> PyResult<Vec<Option<Bound<'_, PyAny>>>> {
    match masks {
        Some(m) => {
            if m.len() != n_sequences {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "{} masks were given, but there are {} sequences",
                    m.len(),
                    n_sequences
                )));
            }
            Ok(m)
        }
        None => Ok(vec![None; n_sequences]),
    }
}

/// Convert an ignore mask into a positive mask for k-mers.
/// If ignore_mask is None, all positions are included.
/// Otherwise, a position is included if none of the k-mer's positions are masked.
fn convert_ignore_into_include_mask(
    py: Python<'_>,
    kmer_alphabet: &KmerAlphabet,
    ignore_mask: Option<&Bound<'_, PyAny>>,
    seq_length: usize,
) -> PyResult<Vec<bool>> {
    let kmer_array_length = kmer_alphabet.kmer_array_length(py, seq_length)?;

    match ignore_mask {
        None => {
            // No mask: include all k-mers
            Ok(vec![true; kmer_array_length])
        }
        Some(mask_array) => {
            // Extract the boolean mask from Python
            let py_array: &Bound<'_, PyArray1<bool>> = mask_array.cast()?;
            let mask = py_array.readonly();
            let mask_slice = mask.as_slice()?;

            if mask_slice.len() != seq_length {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "ignore mask has length {}, but the length of the sequence is {}",
                    mask_slice.len(),
                    seq_length
                )));
            }

            // Get k and spacing from kmer_alphabet
            let k = kmer_alphabet.k(py)?;
            let spacing = kmer_alphabet.spacing(py)?;

            let mut kmer_mask = vec![true; kmer_array_length];

            match spacing {
                None => {
                    // Continuous k-mers: a k-mer at position i covers positions i to i+k-1
                    // If any of those positions is True in ignore_mask, exclude the k-mer
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..kmer_array_length {
                        for j in i..(i + k) {
                            if mask_slice[j] {
                                kmer_mask[i] = false;
                                break;
                            }
                        }
                    }
                }
                Some(spacing_array) => {
                    // Spaced k-mers: use the spacing array
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..kmer_array_length {
                        for &offset in spacing_array.iter() {
                            let pos = i + offset as usize;
                            if mask_slice[pos] {
                                kmer_mask[i] = false;
                                break;
                            }
                        }
                    }
                }
            }

            Ok(kmer_mask)
        }
    }
}

/// If `given_alphabet` is None, find a common alphabet among
/// `sequence_alphabets` and raise an exception if this is not possible.
/// Otherwise just check compatibility of alphabets.
fn get_alphabet<'py>(
    py: Python<'py>,
    given_alphabet: Option<Bound<'py, PyAny>>,
    sequences: &[Bound<'py, PyAny>],
) -> PyResult<Bound<'py, PyAny>> {
    let module = PyModule::import(py, "biotite.sequence.alphabet")?;

    // Extract alphabets from sequences
    let sequence_alphabets: Vec<Bound<'py, PyAny>> = sequences
        .iter()
        .map(|seq| seq.getattr("alphabet"))
        .collect::<PyResult<Vec<_>>>()?;

    match given_alphabet {
        Some(alphabet) => {
            // Check if the alphabet extends all sequence alphabets
            for sequence_alphabet in &sequence_alphabets {
                if !alphabet
                    .call_method1("extends", (sequence_alphabet,))?
                    .extract::<bool>()?
                {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "The given alphabet does not extend at least one \
                         alphabet of the given sequences",
                    ));
                }
            }
            Ok(alphabet)
        }
        None => {
            // Find the common alphabet that extends all sequence alphabets
            let common_alphabet_fn = module.getattr("common_alphabet")?;
            let alphabet = common_alphabet_fn.call1((sequence_alphabets,))?;
            if alphabet.is_none() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "There is no common alphabet that extends all alphabets",
                ));
            }
            Ok(alphabet)
        }
    }
}
