//! Containers for efficient retrieval of k-mer matches.
//!
//! `KmerTable` and `BucketKmerTable` map *k-mers* to the sequence positions where they appear.
//! Both share the same algorithms, which therefore live on a single generic core
//! (`GenericKmerTable`) parameterized by a `TableKind`. The kind selects
//!
//! 1. the entry type stored per position (`KmerEntry`) — a plain
//!    `(ref_id, position)` for `KmerTable`, or a `(kmer, ref_id, position)`
//!    for the bucketed variant, and
//! 2. how a *k-mer* maps to a table slot — the identity for `KmerTable` (one
//!    slot per *k-mer*), or `kmer % n_buckets` for the bucketed variant.
//!
//! The backing store is a single `NestedArray`: one contiguous data buffer
//! plus an offsets array, so there is no per-*k-mer* allocation.

use super::nested::{NestedArray, NestedArrayBuilder};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyModule, PyString, PyTuple};
use pyo3::Borrowed;
use std::mem::ManuallyDrop;

// Label as a separate module to indicate that this exception comes
// from biotite
mod biotite {
    pyo3::import_exception!(biotite.sequence, AlphabetError);
}

/// A *k-mer* code.
pub type Kmer = i64;

/// A thin wrapper around the Python `KmerAlphabet` class.
/// It re-exposes the Python methods to Rust.
struct KmerAlphabet {
    wrapped: Py<PyAny>,
}

impl KmerAlphabet {
    /// Construct a new `KmerAlphabet` from a base `alphabet`, `k` and `spacing`.
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

    /// A clone of the wrapped Python object.
    fn as_bound<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.wrapped.bind(py).clone()
    }

    /// A new reference to the wrapped Python object.
    fn clone_ref(&self, py: Python<'_>) -> Py<PyAny> {
        self.wrapped.clone_ref(py)
    }

    /// Whether this *k-mer* alphabet equals `other`.
    fn equals(&self, py: Python<'_>, other: &KmerAlphabet) -> PyResult<bool> {
        self.wrapped.bind(py).eq(other.wrapped.bind(py))
    }

    /// The number of possible *k-mers*.
    fn len(&self, py: Python<'_>) -> PyResult<usize> {
        self.wrapped.bind(py).len()
    }

    /// The base alphabet the *k-mers* are composed of.
    fn base_alphabet<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.wrapped.bind(py).getattr("base_alphabet")
    }

    /// Whether the base alphabet extends `alphabet`.
    fn base_alphabet_extends(&self, py: Python<'_>, alphabet: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.base_alphabet(py)?
            .call_method1("extends", (alphabet,))?
            .extract()
    }

    /// Decode a *k-mer* code into its symbols.
    fn decode<'py>(&self, py: Python<'py>, kmer: Kmer) -> PyResult<Bound<'py, PyAny>> {
        self.wrapped.bind(py).call_method1("decode", (kmer,))
    }

    /// The overlapping *k-mer* codes of `sequence`.
    fn create_kmers<'py>(
        &self,
        py: Python<'py>,
        sequence: &Bound<'py, PyAny>,
    ) -> PyResult<PyReadonlyArray1<'py, i64>> {
        let sequence_code = sequence.getattr("code")?;
        let py_array: Bound<'py, PyArray1<i64>> = self
            .wrapped
            .bind(py)
            .call_method1("create_kmers", (sequence_code,))?
            .extract()?;
        Ok(py_array.readonly())
    }

    /// The number of overlapping *k-mers* in a sequence of `seq_length`.
    fn kmer_array_length(&self, py: Python<'_>, seq_length: usize) -> PyResult<usize> {
        self.wrapped
            .bind(py)
            .call_method1("kmer_array_length", (seq_length,))?
            .extract()
    }

    /// The length of the *k-mers*.
    fn k(&self, py: Python<'_>) -> PyResult<usize> {
        self.wrapped.bind(py).getattr("k")?.extract()
    }

    /// The spacing model, if spaced *k-mers* are used.
    fn spacing(&self, py: Python<'_>) -> PyResult<Option<Vec<i64>>> {
        let spacing = self.wrapped.bind(py).getattr("spacing")?;
        if spacing.is_none() {
            Ok(None)
        } else {
            let spacing_array: Bound<'_, PyArray1<i64>> = spacing.extract()?;
            Ok(Some(spacing_array.to_vec()?))
        }
    }
}

/// A single *k-mer* match: one row of `N` `int64` columns of a match result.
struct Match<const N: usize>([i64; N]);

impl<const N: usize> Match<N> {
    /// Convert collected matches into the expected `(n, N)` `int64` array.
    fn into_array(matches: Vec<Self>, py: Python<'_>) -> Bound<'_, PyArray2<i64>> {
        let n = matches.len();
        // `Vec<Match<N>>` is `Vec<[i64; N]>`, i.e. `n * N` contiguous `i64`.
        // Reinterpret it as a flat `Vec<i64>` without copying: same allocation,
        // same size and alignment.
        let mut matches = ManuallyDrop::new(matches);
        let flat = unsafe {
            Vec::from_raw_parts(
                matches.as_mut_ptr() as *mut i64,
                n * N,
                matches.capacity() * N,
            )
        };
        Array2::from_shape_vec((n, N), flat)
            .unwrap()
            .into_pyarray(py)
    }
}

/// A position entry stored in a `GenericKmerTable`.
///
/// Implementations carry exactly the data required by their `TableKind`:
/// `KmerTable` does not need to store the *k-mer* (it is the slot index),
/// while the bucketed variant does.
trait KmerEntry: Copy + Default + PartialEq {
    /// Create an entry for the given `kmer` at `position` of sequence `ref_id`.
    fn new(kmer: Kmer, ref_id: u32, position: u32) -> Self;
    /// The reference ID of the indexed sequence.
    fn ref_id(&self) -> u32;
    /// The position within the indexed sequence.
    fn position(&self) -> u32;
    /// Whether this entry actually belongs to the given `kmer`.
    ///
    /// Always true for `KmerTable` (the slot already identifies the *k-mer*),
    /// but the bucketed variant must compare the stored *k-mer*.
    fn matches(&self, kmer: Kmer) -> bool;
}

/// An entry of a `KmerTable`: just the position of a *k-mer*.
///
/// The *k-mer* itself is not stored, as it is the slot index. `repr(C)` gives it
/// a stable layout for serialization (see `GenericKmerTable::get_state`).
#[derive(Copy, Clone, Default, PartialEq)]
#[repr(C)]
struct PlainEntry {
    ref_id: u32,
    position: u32,
}

impl KmerEntry for PlainEntry {
    #[inline(always)]
    fn new(_kmer: Kmer, ref_id: u32, position: u32) -> Self {
        PlainEntry { ref_id, position }
    }
    #[inline(always)]
    fn ref_id(&self) -> u32 {
        self.ref_id
    }
    #[inline(always)]
    fn position(&self) -> u32 {
        self.position
    }
    #[inline(always)]
    fn matches(&self, _kmer: Kmer) -> bool {
        true
    }
}

/// An entry of a `BucketKmerTable`: the *k-mer* alongside its position.
///
/// As several distinct *k-mers* can share a bucket, the *k-mer* must be stored
/// to tell them apart on lookup. `repr(C)` gives it a stable layout for
/// serialization (see `GenericKmerTable::get_state`).
#[derive(Copy, Clone, Default, PartialEq)]
#[repr(C)]
struct BucketEntry {
    kmer: Kmer,
    ref_id: u32,
    position: u32,
}

impl KmerEntry for BucketEntry {
    #[inline(always)]
    fn new(kmer: Kmer, ref_id: u32, position: u32) -> Self {
        BucketEntry {
            kmer,
            ref_id,
            position,
        }
    }
    #[inline(always)]
    fn ref_id(&self) -> u32 {
        self.ref_id
    }
    #[inline(always)]
    fn position(&self) -> u32 {
        self.position
    }
    #[inline(always)]
    fn matches(&self, kmer: Kmer) -> bool {
        self.kmer == kmer
    }
}

/// Selects the entry type and slot mapping of a `GenericKmerTable`.
trait TableKind: Sized {
    /// The position entry stored per slot.
    type Entry: KmerEntry;

    /// The slot index a `kmer` is stored in, given the number of slots.
    fn slot_of(kmer: Kmer, n_slots: usize) -> usize;

    /// The *k-mer* represented by `entry` located in `slot`.
    fn kmer_of_entry(slot: usize, entry: &Self::Entry) -> Kmer;

    /// The number of slots to allocate.
    ///
    /// `n_kmers` is the size of the *k-mer* alphabet, `total_kmers` the total
    /// number of *k-mers* that will be stored (used to pick a default bucket
    /// number) and `n_buckets` an explicitly requested bucket number.
    fn resolve_n_slots(
        py: Python<'_>,
        n_kmers: usize,
        total_kmers: usize,
        n_buckets: Option<usize>,
    ) -> PyResult<usize>;

    /// The number of entries in `slot` that belong to `kmer`.
    ///
    /// For `KmerTable` this is simply the slice length (the slot already
    /// identifies the *k-mer*); the bucketed variant must filter by *k-mer*.
    fn count_in_slot(slot: &[Self::Entry], kmer: Kmer) -> usize;

    /// All stored *k-mer* codes in ascending order.
    fn collect_kmers(table: &GenericKmerTable<Self>) -> Vec<Kmer>;
}

/// The `TableKind` of `KmerTable`: one slot per *k-mer*.
struct Plain;

impl TableKind for Plain {
    type Entry = PlainEntry;

    #[inline(always)]
    fn slot_of(kmer: Kmer, _n_slots: usize) -> usize {
        kmer as usize
    }

    #[inline(always)]
    fn kmer_of_entry(slot: usize, _entry: &PlainEntry) -> Kmer {
        slot as Kmer
    }

    fn resolve_n_slots(
        _py: Python<'_>,
        n_kmers: usize,
        _total_kmers: usize,
        _n_buckets: Option<usize>,
    ) -> PyResult<usize> {
        Ok(n_kmers)
    }

    #[inline(always)]
    fn count_in_slot(slot: &[PlainEntry], _kmer: Kmer) -> usize {
        // Every entry in the slot belongs to the slot's k-mer
        slot.len()
    }

    fn collect_kmers(table: &GenericKmerTable<Self>) -> Vec<Kmer> {
        // Each non-empty slot's index *is* a k-mer, and slots are ascending
        (0..table.n_slots)
            .filter(|&slot| !table.table[slot].is_empty())
            .map(|slot| slot as Kmer)
            .collect()
    }
}

/// The `TableKind` of `BucketKmerTable`: *k-mers* are pooled into buckets.
struct Bucketed;

impl TableKind for Bucketed {
    type Entry = BucketEntry;

    #[inline(always)]
    fn slot_of(kmer: Kmer, n_slots: usize) -> usize {
        kmer as usize % n_slots
    }

    #[inline(always)]
    fn kmer_of_entry(_slot: usize, entry: &BucketEntry) -> Kmer {
        entry.kmer
    }

    fn resolve_n_slots(
        py: Python<'_>,
        n_kmers: usize,
        total_kmers: usize,
        n_buckets: Option<usize>,
    ) -> PyResult<usize> {
        let n_buckets = match n_buckets {
            Some(n_buckets) => n_buckets,
            None => {
                // `bucket_number()` returns a NumPy float, so round it to an int
                let n_buckets = PyModule::import(py, "biotite.sequence.align.buckets")?
                    .getattr("bucket_number")?
                    .call1((total_kmers,))?;
                PyModule::import(py, "builtins")?
                    .getattr("int")?
                    .call1((n_buckets,))?
                    .extract()?
            }
        };
        // Never use more buckets than possible k-mers
        Ok(n_buckets.min(n_kmers))
    }

    #[inline(always)]
    fn count_in_slot(slot: &[BucketEntry], kmer: Kmer) -> usize {
        slot.iter().filter(|entry| entry.matches(kmer)).count()
    }

    fn collect_kmers(table: &GenericKmerTable<Self>) -> Vec<Kmer> {
        // Distinct k-mers are scattered across buckets, so gather and sort them
        let mut kmers: Vec<Kmer> = Vec::new();
        for slot in 0..table.n_slots {
            for entry in &table.table[slot] {
                kmers.push(entry.kmer);
            }
        }
        kmers.sort_unstable();
        kmers.dedup();
        kmers
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for KmerAlphabet {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let kmer_alphabet_class =
            PyModule::import(obj.py(), "biotite.sequence.align.kmeralphabet")?
                .getattr("KmerAlphabet")?;
        if !obj.is_instance(&kmer_alphabet_class)? {
            return Err(PyTypeError::new_err(format!(
                "Got {}, but KmerAlphabet was expected",
                obj.get_type().name()?
            )));
        }
        Ok(KmerAlphabet {
            wrapped: obj.to_owned().unbind(),
        })
    }
}

/// A source of `(kmer, ref_id, position)` triples for table construction.
///
/// `KmerSupplier::for_each` is generic over the sink, so the
/// counting and filling passes in `GenericKmerTable::build` are each
/// monomorphized — there is no per-element dynamic dispatch.
trait KmerSupplier {
    /// Feed every `(kmer, ref_id, position)` triple into `sink`.
    ///
    /// Called twice during construction and must yield identical triples both
    /// times. `py` lets a supplier produce its triples lazily.
    fn for_each<F: FnMut(Kmer, u32, u32)>(&self, py: Python<'_>, sink: F) -> PyResult<()>;
}

/// Supplies a triple per *k-mer* of each input sequence (`from_sequences`).
///
/// The *k-mers* are created lazily inside `for_each`, so only one *k-mer* array
/// is alive at any moment instead of one per sequence.
struct SequenceSupplier<'a, 'py> {
    sequences: &'a [Bound<'py, PyAny>],
    kmer_alphabet: KmerAlphabet,
    ignore_masks: &'a [Option<Bound<'py, PyAny>>],
    ref_ids: &'a [u32],
}

impl KmerSupplier for SequenceSupplier<'_, '_> {
    fn for_each<F: FnMut(Kmer, u32, u32)>(&self, py: Python<'_>, mut sink: F) -> PyResult<()> {
        for ((sequence, ignore_mask), &ref_id) in self
            .sequences
            .iter()
            .zip(self.ignore_masks)
            .zip(self.ref_ids)
        {
            let kmers = self.kmer_alphabet.create_kmers(py, sequence)?;
            let kmers = kmers.as_slice()?;
            match ignore_mask {
                // No mask: every k-mer is included, no include mask is built
                None => {
                    for (i, &kmer) in kmers.iter().enumerate() {
                        sink(kmer, ref_id, i as u32);
                    }
                }
                Some(_) => {
                    let mask = convert_ignore_into_include_mask(
                        py,
                        &self.kmer_alphabet,
                        ignore_mask.as_ref(),
                        sequence.len()?,
                    )?;
                    for (i, (&kmer, &included)) in kmers.iter().zip(&mask).enumerate() {
                        if included {
                            sink(kmer, ref_id, i as u32);
                        }
                    }
                }
            }
            // The k-mer array is dropped here before the next sequence
        }
        Ok(())
    }
}

/// Supplies a triple per given *k-mer*, using its index in the array as the
/// position, optionally filtered by a boolean include mask (`from_kmers`).
struct IndexedSupplier<'a> {
    kmers: &'a [&'a [Kmer]],
    masks: &'a [Vec<bool>],
    ref_ids: &'a [u32],
}

impl KmerSupplier for IndexedSupplier<'_> {
    #[inline]
    fn for_each<F: FnMut(Kmer, u32, u32)>(&self, _py: Python<'_>, mut sink: F) -> PyResult<()> {
        for ((kmers, mask), &ref_id) in self.kmers.iter().zip(self.masks).zip(self.ref_ids) {
            for (i, (&kmer, &included)) in kmers.iter().zip(mask).enumerate() {
                if included {
                    sink(kmer, ref_id, i as u32);
                }
            }
        }
        Ok(())
    }
}

/// Supplies triples with explicit positions and no mask (`from_kmer_selection`).
struct SelectedSupplier<'a> {
    kmers: &'a [&'a [Kmer]],
    positions: &'a [&'a [u32]],
    ref_ids: &'a [u32],
}

impl KmerSupplier for SelectedSupplier<'_> {
    #[inline]
    fn for_each<F: FnMut(Kmer, u32, u32)>(&self, _py: Python<'_>, mut sink: F) -> PyResult<()> {
        for ((kmers, positions), &ref_id) in self.kmers.iter().zip(self.positions).zip(self.ref_ids)
        {
            for (&kmer, &position) in kmers.iter().zip(positions.iter()) {
                sink(kmer, ref_id, position);
            }
        }
        Ok(())
    }
}

/// Supplies the triples from a `from_positions` mapping, iterating the original
/// `(ref_id, position)` arrays directly (no intermediate copy).
struct PositionSupplier<'a, 'py> {
    entries: &'a [(Kmer, PyReadonlyArray2<'py, u32>)],
}

impl KmerSupplier for PositionSupplier<'_, '_> {
    #[inline]
    fn for_each<F: FnMut(Kmer, u32, u32)>(&self, _py: Python<'_>, mut sink: F) -> PyResult<()> {
        for (kmer, array) in self.entries {
            for row in array.as_array().rows() {
                sink(*kmer, row[0], row[1]);
            }
        }
        Ok(())
    }
}

/// Supplies the triples taken from the entries of existing tables (`from_tables`).
struct MergedSupplier<'a, K: TableKind> {
    tables: &'a [&'a GenericKmerTable<K>],
}

impl<K: TableKind> KmerSupplier for MergedSupplier<'_, K> {
    #[inline]
    fn for_each<F: FnMut(Kmer, u32, u32)>(&self, _py: Python<'_>, mut sink: F) -> PyResult<()> {
        for table in self.tables {
            for slot in 0..table.n_slots {
                for entry in &table.table[slot] {
                    sink(
                        K::kmer_of_entry(slot, entry),
                        entry.ref_id(),
                        entry.position(),
                    );
                }
            }
        }
        Ok(())
    }
}

/// The generic core shared by `KmerTable` and the bucketed variant.
///
/// `K` selects the entry type and slot mapping (see `TableKind`).
struct GenericKmerTable<K: TableKind> {
    /// The length of the *k-mers*.
    k: usize,
    /// The size of the *k-mer* alphabet (number of possible *k-mers*).
    n_kmers: usize,
    /// The number of table slots (equal to `n_kmers` for `KmerTable`).
    n_slots: usize,
    /// The wrapped Python `KmerAlphabet`.
    kmer_alphabet: KmerAlphabet,
    /// One inner array of position entries per slot.
    table: NestedArray<K::Entry>,
}

impl<K: TableKind> GenericKmerTable<K> {
    /// Build a table by emitting every `(kmer, ref_id, position)` triple
    /// twice: once to count occurrences per slot and once to scatter them into a
    /// `NestedArrayBuilder` via counting sort.
    fn build<C: KmerSupplier>(
        py: Python<'_>,
        kmer_alphabet: KmerAlphabet,
        k: usize,
        n_kmers: usize,
        n_slots: usize,
        supplier: C,
    ) -> PyResult<Self> {
        // Count the number of entries per slot
        // Reserve one extra element so `NestedArrayBuilder::new()` can append the
        // final offset without reallocating
        let mut counts: Vec<usize> = Vec::with_capacity(n_slots + 1);
        counts.resize(n_slots, 0);
        supplier.for_each(py, |kmer, _ref_id, _position| {
            counts[K::slot_of(kmer, n_slots)] += 1;
        })?;

        // Scatter the entries into their slots via counting sort
        let mut builder = NestedArrayBuilder::new(counts);
        supplier.for_each(py, |kmer, ref_id, position| {
            // SAFETY: the count pass above counted exactly one entry per
            // triple in slot `slot_of(kmer)`, and the same triples
            // are replayed here, so no slot receives more entries than its
            // length.
            unsafe {
                builder.push(
                    K::slot_of(kmer, n_slots),
                    <K::Entry as KmerEntry>::new(kmer, ref_id, position),
                );
            }
        })?;

        Ok(Self {
            k,
            n_kmers,
            n_slots,
            kmer_alphabet,
            table: builder.build(),
        })
    }

    /// The entries stored for `kmer` (filtered by the *k-mer* for buckets).
    #[inline]
    fn lookup(&self, kmer: Kmer) -> impl Iterator<Item = &K::Entry> {
        self.table[K::slot_of(kmer, self.n_slots)]
            .iter()
            .filter(move |entry| entry.matches(kmer))
    }

    /// The reference IDs and positions stored for `kmer`, in storage order.
    fn positions(&self, kmer: Kmer) -> Vec<(u32, u32)> {
        self.lookup(kmer)
            .map(|entry| (entry.ref_id(), entry.position()))
            .collect()
    }

    /// The number of stored positions for `kmer`.
    #[inline]
    fn count_kmer(&self, kmer: Kmer) -> usize {
        K::count_in_slot(&self.table[K::slot_of(kmer, self.n_slots)], kmer)
    }

    /// The *k-mer* codes similar to `kmer` according to a `SimilarityRule`.
    fn similar_kmers(
        &self,
        py: Python<'_>,
        rule: &Bound<'_, PyAny>,
        kmer: Kmer,
    ) -> PyResult<Vec<Kmer>> {
        let array: Bound<'_, PyArray1<i64>> = rule
            .call_method1("similar_kmers", (self.kmer_alphabet.as_bound(py), kmer))?
            .extract()?;
        Ok(array.readonly().as_slice()?.to_vec())
    }

    /// All *k-mer* codes with at least one stored position, in ascending order.
    fn collect_kmers(&self) -> Vec<Kmer> {
        K::collect_kmers(self)
    }

    /// `from_sequences()` constructor, see the Python docstring.
    #[allow(clippy::too_many_arguments)]
    fn from_sequences(
        py: Python<'_>,
        k: usize,
        sequences: Vec<Bound<'_, PyAny>>,
        ref_ids: Option<Vec<u32>>,
        ignore_masks: Option<Vec<Option<Bound<'_, PyAny>>>>,
        alphabet: Option<Bound<'_, PyAny>>,
        spacing: Option<Bound<'_, PyAny>>,
        n_buckets: Option<usize>,
    ) -> PyResult<Self> {
        let n_sequences = sequences.len();
        let ref_ids = compute_ref_ids(ref_ids, n_sequences)?;
        let ignore_masks = compute_masks(ignore_masks, n_sequences)?;
        let alphabet = get_alphabet(py, alphabet, &sequences)?;
        let kmer_alphabet = KmerAlphabet::new(py, alphabet, k, spacing)?;
        let n_kmers = kmer_alphabet.len(py)?;

        // The total (unmasked) number of k-mers, used to pick a bucket number.
        // Computed without creating the k-mers, so no array is materialized here.
        let total_kmers = sequences
            .iter()
            .map(|sequence| kmer_alphabet.kmer_array_length(py, sequence.len()?))
            .sum::<PyResult<usize>>()?;
        let n_slots = K::resolve_n_slots(py, n_kmers, total_kmers, n_buckets)?;

        // The k-mers are created lazily during construction, so only a single
        // k-mer array is alive at a time. The supplier holds its own reference
        // to the alphabet, leaving `kmer_alphabet` free to move into the table.
        let supplier = SequenceSupplier {
            sequences: &sequences,
            kmer_alphabet: KmerAlphabet {
                wrapped: kmer_alphabet.clone_ref(py),
            },
            ignore_masks: &ignore_masks,
            ref_ids: &ref_ids,
        };
        Self::build(py, kmer_alphabet, k, n_kmers, n_slots, supplier)
    }

    /// `from_kmers()` constructor, see the Python docstring.
    fn from_kmers(
        py: Python<'_>,
        kmer_alphabet: KmerAlphabet,
        kmers: Vec<Bound<'_, PyAny>>,
        ref_ids: Option<Vec<u32>>,
        masks: Option<Vec<Option<Bound<'_, PyAny>>>>,
        n_buckets: Option<usize>,
    ) -> PyResult<Self> {
        let k = kmer_alphabet.k(py)?;
        let n_arrays = kmers.len();
        let ref_ids = compute_ref_ids(ref_ids, n_arrays)?;
        let masks = compute_masks(masks, n_arrays)?;

        let n_kmers = kmer_alphabet.len(py)?;
        let kmers_list: Vec<PyReadonlyArray1<'_, i64>> = kmers
            .iter()
            .map(|array| to_kmer_array(py, array))
            .collect::<PyResult<_>>()?;
        let kmers_slices = as_slices(&kmers_list)?;
        for kmers in &kmers_slices {
            check_kmer_bounds(kmers, n_kmers)?;
        }

        // Resolve each mask into an explicit boolean include mask
        let masks_list: Vec<Vec<bool>> = masks
            .iter()
            .zip(kmers_slices.iter())
            .map(|(mask, kmers)| to_include_mask(mask.as_ref(), kmers.len()))
            .collect::<PyResult<_>>()?;

        let total_kmers: usize = kmers_slices.iter().map(|s| s.len()).sum();
        let n_slots = K::resolve_n_slots(py, n_kmers, total_kmers, n_buckets)?;

        let supplier = IndexedSupplier {
            kmers: &kmers_slices,
            masks: &masks_list,
            ref_ids: &ref_ids,
        };
        Self::build(py, kmer_alphabet, k, n_kmers, n_slots, supplier)
    }

    /// `from_kmer_selection()` constructor, see the Python docstring.
    fn from_kmer_selection(
        py: Python<'_>,
        kmer_alphabet: KmerAlphabet,
        positions: Vec<Bound<'_, PyAny>>,
        kmers: Vec<Bound<'_, PyAny>>,
        ref_ids: Option<Vec<u32>>,
        n_buckets: Option<usize>,
    ) -> PyResult<Self> {
        let k = kmer_alphabet.k(py)?;
        let n_kmers = kmer_alphabet.len(py)?;

        check_position_shape(&positions, &kmers)?;
        let ref_ids = compute_ref_ids(ref_ids, kmers.len())?;

        let kmers_list: Vec<PyReadonlyArray1<'_, i64>> = kmers
            .iter()
            .map(|array| to_kmer_array(py, array))
            .collect::<PyResult<_>>()?;
        let kmers_slices = as_slices(&kmers_list)?;
        for kmers in &kmers_slices {
            check_kmer_bounds(kmers, n_kmers)?;
        }
        let positions_list: Vec<PyReadonlyArray1<'_, u32>> = positions
            .iter()
            .map(|array| to_u32_array(py, array))
            .collect::<PyResult<_>>()?;
        let positions_slices = as_slices(&positions_list)?;

        let total_kmers: usize = kmers_slices.iter().map(|s| s.len()).sum();
        let n_slots = K::resolve_n_slots(py, n_kmers, total_kmers, n_buckets)?;

        let supplier = SelectedSupplier {
            kmers: &kmers_slices,
            positions: &positions_slices,
            ref_ids: &ref_ids,
        };
        Self::build(py, kmer_alphabet, k, n_kmers, n_slots, supplier)
    }

    /// `from_positions()` constructor, see the Python docstring.
    fn from_positions(
        py: Python<'_>,
        kmer_alphabet: KmerAlphabet,
        kmer_positions: &Bound<'_, PyDict>,
        n_buckets: Option<usize>,
    ) -> PyResult<Self> {
        let k = kmer_alphabet.k(py)?;
        let n_kmers = kmer_alphabet.len(py)?;

        // Collect and validate the position arrays, but iterate them directly
        // during construction rather than materializing every triple
        let mut entries: Vec<(Kmer, PyReadonlyArray2<'_, u32>)> = Vec::new();
        for (key, value) in kmer_positions.iter() {
            let kmer: Kmer = key.extract()?;
            if kmer < 0 || kmer as usize >= n_kmers {
                return Err(biotite::AlphabetError::new_err(format!(
                    "k-mer code {kmer} does not represent a valid k-mer"
                )));
            }
            let array = to_u32_array2(py, &value)?;
            let (n_rows, n_cols) = array.as_array().dim();
            if n_rows == 0 {
                continue;
            }
            if n_cols != 2 {
                return Err(PyIndexError::new_err(format!(
                    "Each entry in position array has {n_cols} values, but 2 were expected"
                )));
            }
            entries.push((kmer, array));
        }

        let total_kmers: usize = entries
            .iter()
            .map(|(_, array)| array.as_array().dim().0)
            .sum();
        let n_slots = K::resolve_n_slots(py, n_kmers, total_kmers, n_buckets)?;
        let supplier = PositionSupplier { entries: &entries };
        Self::build(py, kmer_alphabet, k, n_kmers, n_slots, supplier)
    }

    /// `from_tables()` constructor, see the Python docstring.
    fn from_tables(py: Python<'_>, tables: &[&GenericKmerTable<K>]) -> PyResult<Self> {
        if tables.is_empty() {
            return Err(PyValueError::new_err("At least one table must be given"));
        }
        let first = tables[0];
        for table in &tables[1..] {
            if !first.kmer_alphabet.equals(py, &table.kmer_alphabet)? {
                return Err(PyValueError::new_err(
                    "The *k-mer* alphabets of the tables are not equal to each other",
                ));
            }
            if first.n_slots != table.n_slots {
                return Err(PyValueError::new_err(
                    "The number of buckets of the tables are not equal to each other",
                ));
            }
        }

        let kmer_alphabet = KmerAlphabet {
            wrapped: first.kmer_alphabet.clone_ref(py),
        };
        let supplier = MergedSupplier { tables };
        Self::build(
            py,
            kmer_alphabet,
            first.k,
            first.n_kmers,
            first.n_slots,
            supplier,
        )
    }

    /// `match_table()` method, see the Python docstring.
    fn match_table<'py>(
        &self,
        py: Python<'py>,
        other: &GenericKmerTable<K>,
        similarity_rule: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        if !self.kmer_alphabet.equals(py, &other.kmer_alphabet)? {
            return Err(PyValueError::new_err(
                "The *k-mer* alphabets of the tables are not equal to each other",
            ));
        }
        if self.n_slots != other.n_slots {
            return Err(PyValueError::new_err(
                "The number of buckets of the tables are not equal to each other",
            ));
        }

        let mut matches: Vec<Match<4>> = Vec::new();
        for slot in 0..other.n_slots {
            for other_entry in &other.table[slot] {
                let kmer = K::kmer_of_entry(slot, other_entry);
                let other_ref = other_entry.ref_id() as i64;
                let other_pos = other_entry.position() as i64;
                if let Some(rule) = similarity_rule {
                    for sim_kmer in self.similar_kmers(py, rule, kmer)? {
                        for self_entry in self.lookup(sim_kmer) {
                            matches.push(Match([
                                other_ref,
                                other_pos,
                                self_entry.ref_id() as i64,
                                self_entry.position() as i64,
                            ]));
                        }
                    }
                } else {
                    for self_entry in self.lookup(kmer) {
                        matches.push(Match([
                            other_ref,
                            other_pos,
                            self_entry.ref_id() as i64,
                            self_entry.position() as i64,
                        ]));
                    }
                }
            }
        }
        Ok(Match::into_array(matches, py))
    }

    /// `match()` method, see the Python docstring.
    fn match_sequence<'py>(
        &self,
        py: Python<'py>,
        sequence: &Bound<'py, PyAny>,
        similarity_rule: Option<&Bound<'py, PyAny>>,
        ignore_mask: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let seq_length = sequence.len()?;
        if seq_length < self.k {
            return Err(PyValueError::new_err("Sequence code is shorter than k"));
        }
        if !self
            .kmer_alphabet
            .base_alphabet_extends(py, &sequence.getattr("alphabet")?)?
        {
            return Err(PyValueError::new_err(
                "The alphabet used for the k-mer index table is not equal to \
                 the alphabet of the sequence",
            ));
        }

        let kmers = self.kmer_alphabet.create_kmers(py, sequence)?;
        let kmers = kmers.as_slice()?;
        let mask =
            convert_ignore_into_include_mask(py, &self.kmer_alphabet, ignore_mask, seq_length)?;

        let mut matches: Vec<Match<3>> = Vec::new();
        for (i, &kmer) in kmers.iter().enumerate() {
            if !mask[i] {
                continue;
            }
            let query_pos = i as i64;
            if let Some(rule) = similarity_rule {
                for sim_kmer in self.similar_kmers(py, rule, kmer)? {
                    for entry in self.lookup(sim_kmer) {
                        matches.push(Match([
                            query_pos,
                            entry.ref_id() as i64,
                            entry.position() as i64,
                        ]));
                    }
                }
            } else {
                for entry in self.lookup(kmer) {
                    matches.push(Match([
                        query_pos,
                        entry.ref_id() as i64,
                        entry.position() as i64,
                    ]));
                }
            }
        }
        Ok(Match::into_array(matches, py))
    }

    /// `match_kmer_selection()` method, see the Python docstring.
    fn match_kmer_selection<'py>(
        &self,
        py: Python<'py>,
        positions: &Bound<'py, PyAny>,
        kmers: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let kmers = to_kmer_array(py, kmers)?;
        let kmers = kmers.as_slice()?;
        let positions = to_u32_array(py, positions)?;
        let positions = positions.as_slice()?;
        check_kmer_bounds(kmers, self.n_kmers)?;
        if positions.len() != kmers.len() {
            return Err(PyIndexError::new_err(format!(
                "{} positions were given for {} k-mers",
                positions.len(),
                kmers.len()
            )));
        }

        let mut matches: Vec<Match<3>> = Vec::new();
        for (&kmer, &seq_pos) in kmers.iter().zip(positions.iter()) {
            for entry in self.lookup(kmer) {
                matches.push(Match([
                    seq_pos as i64,
                    entry.ref_id() as i64,
                    entry.position() as i64,
                ]));
            }
        }
        Ok(Match::into_array(matches, py))
    }

    /// `count()` for the given `kmers`.
    fn count<'py>(
        &self,
        py: Python<'py>,
        kmers: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let kmers = to_kmer_array(py, kmers)?;
        let kmers = kmers.as_slice()?;
        check_kmer_bounds(kmers, self.n_kmers)?;
        let counts: Vec<i64> = kmers
            .iter()
            .map(|&kmer| self.count_kmer(kmer) as i64)
            .collect();
        Ok(counts.into_pyarray(py))
    }

    /// `count()` for all *k-mers* in ascending order.
    fn count_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let counts: Vec<i64> = (0..self.n_kmers as Kmer)
            .map(|kmer| self.count_kmer(kmer) as i64)
            .collect();
        counts.into_pyarray(py)
    }

    /// `get_kmers()` method.
    fn get_kmers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.collect_kmers().into_pyarray(py)
    }

    /// `__getitem__()` method.
    fn get_item<'py>(&self, py: Python<'py>, kmer: Kmer) -> PyResult<Bound<'py, PyArray2<u32>>> {
        if kmer < 0 || kmer as usize >= self.n_kmers {
            return Err(biotite::AlphabetError::new_err(format!(
                "k-mer code {kmer} is out of bounds for the given KmerAlphabet"
            )));
        }
        let positions = self.positions(kmer);
        let mut flat: Vec<u32> = Vec::with_capacity(positions.len() * 2);
        for (ref_id, position) in positions {
            flat.push(ref_id);
            flat.push(position);
        }
        let n = flat.len() / 2;
        Ok(Array2::from_shape_vec((n, 2), flat)
            .unwrap()
            .into_pyarray(py))
    }

    /// `__contains__()` method.
    fn contains(&self, kmer: Kmer) -> bool {
        kmer >= 0 && (kmer as usize) < self.n_kmers && self.lookup(kmer).next().is_some()
    }

    /// `__eq__()` method.
    fn equals(&self, py: Python<'_>, other: &GenericKmerTable<K>) -> PyResult<bool> {
        // Equal k-mer alphabets imply equal `k` and (for `KmerTable`) `n_slots`;
        // `n_slots` is still checked for the bucketed variant
        if self.n_slots != other.n_slots {
            return Ok(false);
        }
        if !self.kmer_alphabet.equals(py, &other.kmer_alphabet)? {
            return Ok(false);
        }
        for slot in 0..self.n_slots {
            if self.table[slot] != other.table[slot] {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// `__str__()` method.
    fn to_str(&self, py: Python<'_>) -> PyResult<String> {
        let base_alphabet = self.kmer_alphabet.base_alphabet(py)?;
        let is_letter = is_letter_alphabet(py, &base_alphabet)?;
        let builtins = PyModule::import(py, "builtins")?;

        let mut lines = Vec::new();
        for kmer in self.collect_kmers() {
            let symbols = self.kmer_alphabet.decode(py, kmer)?;
            let symbols = if is_letter {
                PyString::new(py, "")
                    .call_method1("join", (symbols,))?
                    .extract::<String>()?
            } else {
                let tuple = builtins.getattr("tuple")?.call1((symbols,))?;
                builtins
                    .getattr("str")?
                    .call1((tuple,))?
                    .extract::<String>()?
            };
            let positions = self
                .positions(kmer)
                .iter()
                .map(|(ref_id, position)| format!("({ref_id}, {position})"))
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("{symbols}: {positions}"));
        }
        Ok(lines.join("\n"))
    }

    /// The picklable state: the offsets array (as `int64`) and the raw entry
    /// buffer (see `from_state` for the inverse).
    fn pickle_state<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyBytes>) {
        let (data, offsets) = self.table.raw_parts();
        let offsets: Vec<i64> = offsets.iter().map(|&o| o as i64).collect();
        // SAFETY: `data` is a live, initialized slice of `size_of_val(data)`
        // bytes; the byte view borrows it and is only read (copied into the
        // `PyBytes`) before `data` is released. `K::Entry` is `repr(C)` plain
        // data, so its bytes are well-defined.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        (offsets.into_pyarray(py), PyBytes::new(py, bytes))
    }

    /// Reconstruct a table from its pickled state (see `pickle_state`).
    ///
    /// This builds the table directly, so unpickling does not first allocate an
    /// empty `n_slots`-sized table only to discard it.
    fn from_state(
        py: Python<'_>,
        kmer_alphabet: KmerAlphabet,
        offsets: &[i64],
        bytes: &[u8],
    ) -> PyResult<Self> {
        if offsets.is_empty() {
            return Err(PyValueError::new_err("Invalid state: empty offsets array"));
        }
        let k = kmer_alphabet.k(py)?;
        let n_kmers = kmer_alphabet.len(py)?;
        let n_slots = offsets.len() - 1;
        let offsets: Vec<usize> = offsets.iter().map(|&o| o as usize).collect();

        // Copy the entry bytes into an uninitialized buffer (no zero-init)
        let entry_size = std::mem::size_of::<K::Entry>();
        let total = bytes.len() / entry_size;
        let mut data: Vec<K::Entry> = Vec::with_capacity(total);
        // SAFETY: All bit patterns are valid KmerEntry objects,
        // so even from untrusted sources no invalid entries can be created
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                data.as_mut_ptr() as *mut u8,
                total * entry_size,
            );
            data.set_len(total);
        }

        // SAFETY: in the worst case indexing the NestedArray will panic if the offsets are invalid
        let table = unsafe { NestedArray::from_raw_parts(data, offsets) };
        Ok(Self {
            k,
            n_kmers,
            n_slots,
            kmer_alphabet,
            table,
        })
    }
}

/// Whether the base alphabet is a `LetterAlphabet`.
fn is_letter_alphabet(py: Python<'_>, base_alphabet: &Bound<'_, PyAny>) -> PyResult<bool> {
    let letter_alphabet =
        PyModule::import(py, "biotite.sequence.alphabet")?.getattr("LetterAlphabet")?;
    base_alphabet.is_instance(&letter_alphabet)
}

/// Extract the contiguous slices of a list of read-only NumPy arrays.
fn as_slices<'a, T: numpy::Element>(
    arrays: &'a [PyReadonlyArray1<'_, T>],
) -> PyResult<Vec<&'a [T]>> {
    arrays
        .iter()
        .map(|array| array.as_slice().map_err(Into::into))
        .collect()
}

/// Initialize the reference IDs, defaulting to `0..n` if not given.
fn compute_ref_ids(ref_ids: Option<Vec<u32>>, n: usize) -> PyResult<Vec<u32>> {
    match ref_ids {
        Some(ids) => {
            if ids.len() != n {
                return Err(PyIndexError::new_err(format!(
                    "{} reference IDs were given, but there are {} sequences",
                    ids.len(),
                    n
                )));
            }
            Ok(ids)
        }
        None => Ok((0..n as u32).collect()),
    }
}

/// Initialize the masks, defaulting to a list of `None` if not given.
fn compute_masks(
    masks: Option<Vec<Option<Bound<'_, PyAny>>>>,
    n: usize,
) -> PyResult<Vec<Option<Bound<'_, PyAny>>>> {
    match masks {
        Some(masks) => {
            if masks.len() != n {
                return Err(PyIndexError::new_err(format!(
                    "{} masks were given, but there are {} sequences",
                    masks.len(),
                    n
                )));
            }
            Ok(masks)
        }
        None => Ok((0..n).map(|_| None).collect()),
    }
}

/// Convert an ignore mask (over sequence positions) into a *k-mer* include mask.
///
/// A *k-mer* is included if none of its informative positions is ignored.
fn convert_ignore_into_include_mask(
    py: Python<'_>,
    kmer_alphabet: &KmerAlphabet,
    ignore_mask: Option<&Bound<'_, PyAny>>,
    seq_length: usize,
) -> PyResult<Vec<bool>> {
    let kmer_array_length = kmer_alphabet.kmer_array_length(py, seq_length)?;
    let Some(mask_array) = ignore_mask else {
        return Ok(vec![true; kmer_array_length]);
    };

    let py_array: Bound<'_, PyArray1<bool>> = mask_array.extract()?;
    let mask = py_array.readonly();
    let mask = mask.as_slice()?;
    if mask.len() != seq_length {
        return Err(PyIndexError::new_err(format!(
            "ignore mask has length {}, but the length of the sequence is {}",
            mask.len(),
            seq_length
        )));
    }

    let k = kmer_alphabet.k(py)?;
    let spacing = kmer_alphabet.spacing(py)?;
    let mut include = vec![true; kmer_array_length];
    match spacing {
        None => {
            // Continuous k-mers cover positions i..i+k
            for (i, included) in include.iter_mut().enumerate() {
                if mask[i..i + k].iter().any(|&ignored| ignored) {
                    *included = false;
                }
            }
        }
        Some(spacing) => {
            // Spaced k-mers cover positions i + offset for each model offset
            for (i, included) in include.iter_mut().enumerate() {
                if spacing.iter().any(|&offset| mask[i + offset as usize]) {
                    *included = false;
                }
            }
        }
    }
    Ok(include)
}

/// Resolve a single mask into an explicit boolean include mask of `length`.
fn to_include_mask(mask: Option<&Bound<'_, PyAny>>, length: usize) -> PyResult<Vec<bool>> {
    match mask {
        None => Ok(vec![true; length]),
        Some(mask) => {
            let array: Bound<'_, PyArray1<bool>> = mask.extract()?;
            Ok(array.readonly().as_slice()?.to_vec())
        }
    }
}

/// Find or check the common alphabet of the input `sequences`.
fn get_alphabet<'py>(
    py: Python<'py>,
    given_alphabet: Option<Bound<'py, PyAny>>,
    sequences: &[Bound<'py, PyAny>],
) -> PyResult<Bound<'py, PyAny>> {
    let module = PyModule::import(py, "biotite.sequence.alphabet")?;
    let sequence_alphabets: Vec<Bound<'py, PyAny>> = sequences
        .iter()
        .map(|seq| seq.getattr("alphabet"))
        .collect::<PyResult<_>>()?;

    match given_alphabet {
        Some(alphabet) => {
            for sequence_alphabet in &sequence_alphabets {
                if !alphabet
                    .call_method1("extends", (sequence_alphabet,))?
                    .extract::<bool>()?
                {
                    return Err(PyValueError::new_err(
                        "The given alphabet is incompatible with a least one \
                         alphabet of the given sequences",
                    ));
                }
            }
            Ok(alphabet)
        }
        None => {
            let alphabet = module
                .getattr("common_alphabet")?
                .call1((sequence_alphabets,))?;
            if alphabet.is_none() {
                return Err(PyValueError::new_err(
                    "There is no common alphabet that extends all alphabets",
                ));
            }
            Ok(alphabet)
        }
    }
}

/// Coerce a Python object into a contiguous `int64` k-mer array (no copy if
/// already `int64`).
fn to_kmer_array<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, i64>> {
    let np = PyModule::import(py, "numpy")?;
    let array: Bound<'py, PyArray1<i64>> = np
        .call_method1("ascontiguousarray", (array, np.getattr("int64")?))?
        .extract()?;
    Ok(array.readonly())
}

/// Coerce a Python object into a contiguous `uint32` array (no copy if already
/// `uint32`).
fn to_u32_array<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, u32>> {
    let np = PyModule::import(py, "numpy")?;
    let array: Bound<'py, PyArray1<u32>> = np
        .call_method1("ascontiguousarray", (array, np.getattr("uint32")?))?
        .extract()?;
    Ok(array.readonly())
}

/// Coerce a Python object into a contiguous `uint32` 2D array.
fn to_u32_array2<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray2<'py, u32>> {
    let np = PyModule::import(py, "numpy")?;
    let array: Bound<'py, PyArray2<u32>> = np
        .call_method1("ascontiguousarray", (array, np.getattr("uint32")?))?
        .extract()?;
    Ok(array.readonly())
}

/// Raise an `AlphabetError` if any *k-mer* code is out of bounds.
fn check_kmer_bounds(kmers: &[Kmer], n_kmers: usize) -> PyResult<()> {
    // Out-of-bounds k-mers are not expected, so mark the error path `#[cold]`:
    // this steers the branch predictor and code layout towards the common
    // in-bounds case.
    #[cold]
    #[inline(never)]
    fn out_of_bounds() -> PyErr {
        biotite::AlphabetError::new_err("Given k-mer codes do not represent valid k-mers")
    }
    for &kmer in kmers {
        if kmer < 0 || kmer as usize >= n_kmers {
            return Err(out_of_bounds());
        }
    }
    Ok(())
}

/// Check that the number of position arrays and k-mer arrays match elementwise.
fn check_position_shape(
    positions: &[Bound<'_, PyAny>],
    kmers: &[Bound<'_, PyAny>],
) -> PyResult<()> {
    if positions.len() != kmers.len() {
        return Err(PyIndexError::new_err(format!(
            "{} position arrays for {} k-mer arrays were given",
            positions.len(),
            kmers.len()
        )));
    }
    for (i, (positions, kmers)) in positions.iter().zip(kmers.iter()).enumerate() {
        if positions.len()? != kmers.len()? {
            return Err(PyIndexError::new_err(format!(
                "{} positions for {} k-mers were given at index {}",
                positions.len()?,
                kmers.len()?,
                i
            )));
        }
    }
    Ok(())
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
///       *k-mers* in a sequence and stores the sequence position of
///       each *kmer* in the table.
///     - :meth:`from_kmers()` is similar to :meth:`from_sequences()`
///       but directly accepts *k-mers* as input instead of sequences.
///     - :meth:`from_kmer_selection()` takes a combination of *k-mers*
///       and their positions in a sequence, which can be used to
///       apply subset selectors, such as :class:`MinimizerSelector`.
///     - :meth:`from_tables()` merges the entries from multiple
///       :class:`KmerTable` objects into a new table.
///     - :meth:`from_positions()` let's the user provide manual
///       *k-mer* positions, which can be useful for loading a
///       :class:`KmerTable` from file.
///
/// Each indexed *k-mer* position is represented by a tuple of
///
///     1. a unique reference ID that identifies to which sequence a
///        position refers to and
///     2. the zero-based sequence position of the first symbol in the
///        *k-mer*.
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
/// 2. An accompanying array that maps each *k-mer* to the index range in the
///    first array, where the corresponding *k-mer* positions are stored.
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
#[pyclass(module = "biotite.rust.sequence.align")]
pub struct KmerTable(GenericKmerTable<Plain>);

#[pymethods]
impl KmerTable {
    /// Reconstruct a table from its pickled state (called by `__reduce__`).
    #[staticmethod]
    fn _from_state(
        py: Python<'_>,
        kmer_alphabet: KmerAlphabet,
        offsets: PyReadonlyArray1<'_, i64>,
        data: Bound<'_, PyBytes>,
    ) -> PyResult<Self> {
        Ok(KmerTable(GenericKmerTable::from_state(
            py,
            kmer_alphabet,
            offsets.as_slice()?,
            data.as_bytes(),
        )?))
    }

    #[getter]
    fn kmer_alphabet(&self, py: Python<'_>) -> Py<PyAny> {
        self.0.kmer_alphabet.clone_ref(py)
    }

    #[getter]
    fn alphabet<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.kmer_alphabet.base_alphabet(py)
    }

    #[getter]
    fn k(&self) -> usize {
        self.0.k
    }

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
        Ok(KmerTable(GenericKmerTable::from_sequences(
            py,
            k,
            sequences,
            ref_ids,
            ignore_masks,
            alphabet,
            spacing,
            None,
        )?))
    }

    /// from_kmers(kmer_alphabet, kmers, ref_ids=None, masks=None)
    ///
    /// Create a :class:`KmerTable` by storing the positions of all
    /// input *k-mers*.
    ///
    /// Parameters
    /// ----------
    /// kmer_alphabet : KmerAlphabet
    ///     The :class:`KmerAlphabet` to use for the new table.
    ///     Should be the same alphabet that was used to calculate the
    ///     input *kmers*.
    /// kmers : sized iterable object of (ndarray, dtype=np.int64), length=m
    ///     List where each array contains the *k-mer* codes from a
    ///     sequence.
    ///     For each array the index of the *k-mer* code in the array
    ///     is stored in the table as sequence position.
    /// ref_ids : sized iterable object of int, length=m, optional
    ///     The reference IDs for the sequences.
    ///     These are used to identify the corresponding sequence for a
    ///     *k-mer* match.
    ///     By default the IDs are counted from *0* to *m*.
    /// masks : sized iterable object of (ndarray, dtype=bool), length=m, optional
    ///     A *k-mer* code at a position, where the corresponding mask
    ///     is false, is not added to the table.
    ///     By default, all positions are added.
    ///
    /// See Also
    /// --------
    /// from_sequences : The same functionality based on undecomposed sequences
    ///
    /// Returns
    /// -------
    /// table : KmerTable
    ///     The newly created table.
    ///
    /// Examples
    /// --------
    ///
    /// >>> sequences = [ProteinSequence("BIQTITE"), ProteinSequence("NIQBITE")]
    /// >>> kmer_alphabet = KmerAlphabet(ProteinSequence.alphabet, 3)
    /// >>> kmer_codes = [kmer_alphabet.create_kmers(s.code) for s in sequences]
    /// >>> for code in kmer_codes:
    /// ...     print(code)
    /// [11701  4360  7879  9400  4419]
    /// [ 6517  4364  7975 11704  4419]
    /// >>> table = KmerTable.from_kmers(
    /// ...     kmer_alphabet, kmer_codes
    /// ... )
    /// >>> print(table)
    /// IQT: (0, 1)
    /// IQB: (1, 1)
    /// ITE: (0, 4), (1, 4)
    /// NIQ: (1, 0)
    /// QTI: (0, 2)
    /// QBI: (1, 2)
    /// TIT: (0, 3)
    /// BIQ: (0, 0)
    /// BIT: (1, 3)
    #[staticmethod]
    #[pyo3(signature = (kmer_alphabet, kmers, ref_ids=None, masks=None))]
    fn from_kmers(
        py: Python<'_>,
        kmer_alphabet: KmerAlphabet,
        kmers: Vec<Bound<'_, PyAny>>,
        ref_ids: Option<Vec<u32>>,
        masks: Option<Vec<Option<Bound<'_, PyAny>>>>,
    ) -> PyResult<Self> {
        Ok(KmerTable(GenericKmerTable::from_kmers(
            py,
            kmer_alphabet,
            kmers,
            ref_ids,
            masks,
            None,
        )?))
    }

    /// from_kmer_selection(kmer_alphabet, positions, kmers, ref_ids=None)
    ///
    /// Create a :class:`KmerTable` by storing the positions of a
    /// filtered subset of input *k-mers*.
    ///
    /// This can be used to reduce the number of stored *k-mers* using
    /// a *k-mer* subset selector such as :class:`MinimizerSelector`.
    ///
    /// Parameters
    /// ----------
    /// kmer_alphabet : KmerAlphabet
    ///     The :class:`KmerAlphabet` to use for the new table.
    ///     Should be the same alphabet that was used to calculate the
    ///     input *kmers*.
    /// positions : sized iterable object of (ndarray, shape=(n,), dtype=uint32), length=m
    ///     List where each array contains the sequence positions of
    ///     the filtered subset of *k-mers* given in `kmers`.
    ///     The list may contain multiple elements for multiple
    ///     sequences.
    /// kmers : sized iterable object of (ndarray, shape=(n,), dtype=np.int64), length=m
    ///     List where each array contains the filtered subset of
    ///     *k-mer* codes from a sequence.
    ///     For each array the index of the *k-mer* code in the array,
    ///     is stored in the table as sequence position.
    ///     The list may contain multiple elements for multiple
    ///     sequences.
    /// ref_ids : sized iterable object of int, length=m, optional
    ///     The reference IDs for the sequences.
    ///     These are used to identify the corresponding sequence for a
    ///     *k-mer* match.
    ///     By default the IDs are counted from *0* to *m*.
    ///
    /// Returns
    /// -------
    /// table : KmerTable
    ///     The newly created table.
    ///
    /// Examples
    /// --------
    ///
    /// Reduce the size of sequence data in the table using minimizers:
    ///
    /// >>> sequence1 = ProteinSequence("THIS*IS*A*SEQVENCE")
    /// >>> kmer_alph = KmerAlphabet(sequence1.alphabet, k=3)
    /// >>> minimizer = MinimizerSelector(kmer_alph, window=4)
    /// >>> minimizer_pos, minimizers = minimizer.select(sequence1)
    /// >>> kmer_table = KmerTable.from_kmer_selection(
    /// ...     kmer_alph, [minimizer_pos], [minimizers]
    /// ... )
    ///
    /// Use the same :class:`MinimizerSelector` to select the minimizers
    /// from the query sequence and match them against the table.
    /// Although the amount of *k-mers* is reduced, matching is still
    /// guanrateed to work, if the two sequences share identity in the
    /// given window:
    ///
    /// >>> sequence2 = ProteinSequence("ANQTHER*SEQVENCE")
    /// >>> minimizer_pos, minimizers = minimizer.select(sequence2)
    /// >>> matches = kmer_table.match_kmer_selection(minimizer_pos, minimizers)
    /// >>> print(matches)
    /// [[ 9  0 11]
    ///  [12  0 14]]
    /// >>> for query_pos, _, db_pos in matches:
    /// ...     print(sequence1)
    /// ...     print(" " * (db_pos-1) + "^" * kmer_table.k)
    /// ...     print(sequence2)
    /// ...     print(" " * (query_pos-1) + "^" * kmer_table.k)
    /// ...     print()
    /// THIS*IS*A*SEQVENCE
    ///   ^^^
    /// ANQTHER*SEQVENCE
    ///         ^^^
    /// <BLANKLINE>
    /// THIS*IS*A*SEQVENCE
    ///             ^^^
    /// ANQTHER*SEQVENCE
    ///         ^^^
    /// <BLANKLINE>
    #[staticmethod]
    #[pyo3(signature = (kmer_alphabet, positions, kmers, ref_ids=None))]
    fn from_kmer_selection(
        py: Python<'_>,
        kmer_alphabet: KmerAlphabet,
        positions: Vec<Bound<'_, PyAny>>,
        kmers: Vec<Bound<'_, PyAny>>,
        ref_ids: Option<Vec<u32>>,
    ) -> PyResult<Self> {
        Ok(KmerTable(GenericKmerTable::from_kmer_selection(
            py,
            kmer_alphabet,
            positions,
            kmers,
            ref_ids,
            None,
        )?))
    }

    /// from_tables(tables)
    ///
    /// Create a :class:`KmerTable` by merging the *k-mer* positions
    /// from existing `tables`.
    ///
    /// Parameters
    /// ----------
    /// tables : iterable object of KmerTable
    ///     The tables to be merged.
    ///     All tables must have equal :class:`KmerAlphabet` objects,
    ///     i.e. the same *k* and equal base alphabets.
    ///
    /// Returns
    /// -------
    /// table : KmerTable
    ///     The newly created table.
    ///
    /// Examples
    /// --------
    ///
    /// >>> table1 = KmerTable.from_sequences(
    /// ...     2, [NucleotideSequence("TTATA")], ref_ids=[100]
    /// ... )
    /// >>> table2 = KmerTable.from_sequences(
    /// ...     2, [NucleotideSequence("CTAG")], ref_ids=[101]
    /// ... )
    /// >>> merged_table = KmerTable.from_tables([table1, table2])
    /// >>> print(merged_table)
    /// AG: (101, 2)
    /// AT: (100, 2)
    /// CT: (101, 0)
    /// TA: (100, 1), (100, 3), (101, 1)
    /// TT: (100, 0)
    #[staticmethod]
    fn from_tables(py: Python<'_>, tables: Vec<PyRef<'_, KmerTable>>) -> PyResult<Self> {
        let refs: Vec<&GenericKmerTable<Plain>> = tables.iter().map(|table| &table.0).collect();
        Ok(KmerTable(GenericKmerTable::from_tables(py, &refs)?))
    }

    /// from_positions(kmer_alphabet, kmer_positions)
    ///
    /// Create a :class:`KmerTable` from *k-mer* reference IDs and
    /// positions.
    /// This constructor is especially useful for restoring a table
    /// from previously serialized data.
    ///
    /// Parameters
    /// ----------
    /// kmer_alphabet : KmerAlphabet
    ///     The :class:`KmerAlphabet` to use for the new table
    /// kmer_positions : dict of (int -> ndarray, shape=(n,2), dtype=int)
    ///     A dictionary representing the *k-mer* reference IDs and
    ///     positions to be stored in the newly created table.
    ///     It maps a *k-mer* code to a :class:`ndarray`.
    ///     To achieve a high performance the data type ``uint32``
    ///     is preferred for the arrays.
    ///
    /// Returns
    /// -------
    /// table : KmerTable
    ///     The newly created table.
    ///
    /// Examples
    /// --------
    ///
    /// >>> sequence = ProteinSequence("BIQTITE")
    /// >>> table = KmerTable.from_sequences(3, [sequence], ref_ids=[100])
    /// >>> print(table)
    /// IQT: (100, 1)
    /// ITE: (100, 4)
    /// QTI: (100, 2)
    /// TIT: (100, 3)
    /// BIQ: (100, 0)
    /// >>> data = {kmer: table[kmer] for kmer in table}
    /// >>> print(data)
    /// {4360: array([[100,   1]], dtype=uint32), 4419: array([[100,   4]], dtype=uint32), 7879: array([[100,   2]], dtype=uint32), 9400: array([[100,   3]], dtype=uint32), 11701: array([[100,   0]], dtype=uint32)}
    /// >>> restored_table = KmerTable.from_positions(table.kmer_alphabet, data)
    /// >>> print(restored_table)
    /// IQT: (100, 1)
    /// ITE: (100, 4)
    /// QTI: (100, 2)
    /// TIT: (100, 3)
    /// BIQ: (100, 0)
    #[staticmethod]
    fn from_positions(
        py: Python<'_>,
        kmer_alphabet: KmerAlphabet,
        kmer_positions: Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        Ok(KmerTable(GenericKmerTable::from_positions(
            py,
            kmer_alphabet,
            &kmer_positions,
            None,
        )?))
    }

    /// match_table(table, similarity_rule=None)
    ///
    /// Find matches between the *k-mers* in this table with the
    /// *k-mers* in another `table`.
    ///
    /// This means that for each *k-mer* the cartesian product between
    /// the positions in both tables is added to the matches.
    ///
    /// Parameters
    /// ----------
    /// table : KmerTable
    ///     The table to be matched.
    ///     Both tables must have equal :class:`KmerAlphabet` objects,
    ///     i.e. the same *k* and equal base alphabets.
    /// similarity_rule : SimilarityRule, optional
    ///     If this parameter is given, not only exact *k-mer* matches
    ///     are considered, but also similar ones according to the given
    ///     :class:`SimilarityRule`.
    ///
    /// Returns
    /// -------
    /// matches : ndarray, shape=(n,4), dtype=np.int64
    ///     The *k-mer* matches.
    ///     Each row contains one match. Each match has the following
    ///     columns:
    ///
    ///         0. The reference ID of the matched sequence in the other
    ///            table
    ///         1. The sequence position of the matched sequence in the
    ///            other table
    ///         2. The reference ID of the matched sequence in this
    ///            table
    ///         3. The sequence position of the matched sequence in this
    ///            table
    ///
    /// Notes
    /// -----
    ///
    /// There is no guaranteed order of the reference IDs or
    /// sequence positions in the returned matches.
    ///
    /// Examples
    /// --------
    ///
    /// >>> sequence1 = ProteinSequence("BIQTITE")
    /// >>> table1 = KmerTable.from_sequences(3, [sequence1], ref_ids=[100])
    /// >>> print(table1)
    /// IQT: (100, 1)
    /// ITE: (100, 4)
    /// QTI: (100, 2)
    /// TIT: (100, 3)
    /// BIQ: (100, 0)
    /// >>> sequence2 = ProteinSequence("TITANITE")
    /// >>> table2 = KmerTable.from_sequences(3, [sequence2], ref_ids=[101])
    /// >>> print(table2)
    /// ANI: (101, 3)
    /// ITA: (101, 1)
    /// ITE: (101, 5)
    /// NIT: (101, 4)
    /// TAN: (101, 2)
    /// TIT: (101, 0)
    /// >>> print(table1.match_table(table2))
    /// [[101   5 100   4]
    ///  [101   0 100   3]]
    #[pyo3(signature = (table, similarity_rule=None))]
    fn match_table<'py>(
        &self,
        py: Python<'py>,
        table: PyRef<'py, KmerTable>,
        similarity_rule: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.0.match_table(py, &table.0, similarity_rule.as_ref())
    }

    /// match(sequence, similarity_rule=None, ignore_mask=None)
    ///
    /// Find matches between the *k-mers* in this table with all
    /// overlapping *k-mers* in the given `sequence`.
    /// *k* is determined by the table.
    ///
    /// Parameters
    /// ----------
    /// sequence : Sequence
    ///     The sequence to be matched.
    ///     The table's base alphabet must extend the alphabet of the
    ///     sequence.
    /// similarity_rule : SimilarityRule, optional
    ///     If this parameter is given, not only exact *k-mer* matches
    ///     are considered, but also similar ones according to the given
    ///     :class:`SimilarityRule`.
    /// ignore_mask : ndarray, dtype=bool, optional
    ///     Boolean mask of sequence positions to ignore.
    ///     *k-mers* that involve these sequence positions are not added
    ///     to the table.
    ///     This is used e.g. to skip repeat regions.
    ///     By default, no sequence position is ignored.
    ///
    /// Returns
    /// -------
    /// matches : ndarray, shape=(n,3), dtype=np.int64
    ///     The *k-mer* matches.
    ///     Each row contains one match. Each match has the following
    ///     columns:
    ///
    ///         0. The sequence position in the input sequence
    ///         1. The reference ID of the matched sequence in the table
    ///         2. The sequence position of the matched sequence in the
    ///            table
    ///
    /// Notes
    /// -----
    ///
    /// The matches are ordered by the first column.
    ///
    /// Examples
    /// --------
    ///
    /// >>> sequence1 = ProteinSequence("BIQTITE")
    /// >>> table = KmerTable.from_sequences(3, [sequence1], ref_ids=[100])
    /// >>> print(table)
    /// IQT: (100, 1)
    /// ITE: (100, 4)
    /// QTI: (100, 2)
    /// TIT: (100, 3)
    /// BIQ: (100, 0)
    /// >>> sequence2 = ProteinSequence("TITANITE")
    /// >>> print(table.match(sequence2))
    /// [[  0 100   3]
    ///  [  5 100   4]]
    #[pyo3(name = "match", signature = (sequence, similarity_rule=None, ignore_mask=None))]
    fn match_<'py>(
        &self,
        py: Python<'py>,
        sequence: Bound<'py, PyAny>,
        similarity_rule: Option<Bound<'py, PyAny>>,
        ignore_mask: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.0.match_sequence(
            py,
            &sequence,
            similarity_rule.as_ref(),
            ignore_mask.as_ref(),
        )
    }

    /// match_kmer_selection(positions, kmers)
    ///
    /// Find matches between the *k-mers* in this table with the given
    /// *k-mer* selection.
    ///
    /// It is intended to use this method to find matches in a table
    /// that was created using :meth:`from_kmer_selection()`.
    ///
    /// Parameters
    /// ----------
    /// positions : ndarray, shape=(n,), dtype=uint32
    ///     Sequence positions of the filtered subset of *k-mers* given
    ///     in `kmers`.
    /// kmers : ndarray, shape=(n,), dtype=np.int64
    ///     Filtered subset of *k-mer* codes to match against.
    ///
    /// Returns
    /// -------
    /// matches : ndarray, shape=(n,3), dtype=np.int64
    ///     The *k-mer* matches.
    ///     Each row contains one *k-mer* match.
    ///     Each match has the following columns:
    ///
    ///         0. The sequence position of the input *k-mer*, taken
    ///            from `positions`
    ///         1. The reference ID of the matched sequence in the table
    ///         2. The sequence position of the matched *k-mer* in the
    ///            table
    fn match_kmer_selection<'py>(
        &self,
        py: Python<'py>,
        positions: Bound<'py, PyAny>,
        kmers: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.0.match_kmer_selection(py, &positions, &kmers)
    }

    /// count(kmers=None)
    ///
    /// Count the number of occurences for each *k-mer* in the table.
    ///
    /// Parameters
    /// ----------
    /// kmers : ndarray, dtype=np.int64, optional
    ///     The count is returned for these *k-mer* codes.
    ///     By default all *k-mers* are counted in ascending order, i.e.
    ///     ``count_for_kmer = counts[kmer]``.
    ///
    /// Returns
    /// -------
    /// counts : ndarray, dtype=np.int64
    ///     The counts for each given *k-mer*.
    ///
    /// Examples
    /// --------
    /// >>> table = KmerTable.from_sequences(
    /// ...     k = 2,
    /// ...     sequences = [NucleotideSequence("TTATA"), NucleotideSequence("CTAG")],
    /// ...     ref_ids = [0, 1]
    /// ... )
    /// >>> print(table.count(table.kmer_alphabet.encode_multiple(["TA", "AG"])))
    /// [3 1]
    /// >>> counts = table.count()
    /// >>> print(counts)
    /// [0 0 1 1 0 0 0 1 0 0 0 0 3 0 0 1]
    #[pyo3(signature = (kmers=None))]
    fn count<'py>(
        &self,
        py: Python<'py>,
        kmers: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        match kmers {
            Some(kmers) => self.0.count(py, &kmers),
            None => Ok(self.0.count_all(py)),
        }
    }

    /// Get the *k-mer* codes for all *k-mers* that have at least one
    /// position in the table.
    ///
    /// Returns
    /// -------
    /// kmers : ndarray, shape=(n,), dtype=np.int64
    ///     The *k-mer* codes in ascending order.
    fn get_kmers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.0.get_kmers(py)
    }

    fn __getitem__<'py>(&self, py: Python<'py>, kmer: Kmer) -> PyResult<Bound<'py, PyArray2<u32>>> {
        self.0.get_item(py, kmer)
    }

    fn __len__(&self) -> usize {
        self.0.n_kmers
    }

    fn __contains__(&self, kmer: Kmer) -> bool {
        self.0.contains(kmer)
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kmers = PyList::new(py, self.0.collect_kmers())?;
        Ok(kmers.try_iter()?.into_any())
    }

    fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut kmers = self.0.collect_kmers();
        kmers.reverse();
        let kmers = PyList::new(py, kmers)?;
        Ok(kmers.try_iter()?.into_any())
    }

    fn __eq__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        match other.extract::<PyRef<'_, KmerTable>>() {
            Ok(other) => self.0.equals(py, &other.0),
            Err(_) => Ok(false),
        }
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        self.0.to_str(py)
    }

    /// Pickle via a direct reconstructor, so unpickling does not build an empty
    /// `n_slots`-sized table first.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let reconstructor = py.get_type::<KmerTable>().getattr("_from_state")?;
        let (offsets, data) = self.0.pickle_state(py);
        let args = PyTuple::new(
            py,
            [
                self.0.kmer_alphabet.clone_ref(py).into_bound(py),
                offsets.into_any(),
                data.into_any(),
            ],
        )?;
        Ok((reconstructor, args))
    }
}

/// This class represents a *k-mer* index table.
/// In contrast to :class:`KmerTable`, which gives every possible *k-mer* its
/// own slot, a :class:`BucketKmerTable` pools the *k-mers* into a limited
/// number of buckets.
/// Hence, different *k-mers* may be stored in the same bucket, like in a
/// hash table.
/// This approach makes *k-mer* indices with large *k-mer* alphabets
/// fit into memory.
///
/// Otherwise, the API for creating a :class:`BucketKmerTable` and
/// matching to it is analogous to :class:`KmerTable`.
///
/// Attributes
/// ----------
/// kmer_alphabet : KmerAlphabet
///     The internal :class:`KmerAlphabet`, that is used to
///     encode all overlapping *k-mers* of an input sequence.
/// alphabet : Alphabet
///     The base alphabet, from which this :class:`BucketKmerTable` was
///     created.
/// k : int
///     The length of the *k-mers*.
/// n_buckets : int
///     The number of buckets, the *k-mers* are divided into.
///
/// See Also
/// --------
/// KmerTable
///
/// Notes
/// -----
///
/// *Memory consumption*
///
/// For efficient mapping, a :class:`BucketKmerTable` contains two large arrays:
///
/// 1. An array that holds all *k-mers* with their positions.
/// 2. An accompanying array that maps each bucket to the index range in the
///    first array, where the corresponding entries are stored.
///
/// As buckets are used, the memory requirements are limited to the number
/// of buckets instead of scaling with the :class:`KmerAlphabet` size.
/// If each bucket is used, the required memory space :math:`S` in byte
/// is
///
/// .. math::
///
///     S = 8B + 16L
///
/// where :math:`B` is the number of buckets and :math:`L` is the summed
/// length of all sequences added to the table.
///
/// *Buckets*
///
/// The ratio :math:`L/B` is called *load_factor*.
/// By default :class:`BucketKmerTable` uses a load factor of
/// approximately 0.8 to ensure efficient *k-mer* matching.
/// The number fo buckets can be adjusted by setting the
/// `n_buckets` parameters on :class:`BucketKmerTable` creation.
/// It is recommended to use :func:`bucket_number()` to compute an
/// appropriate number of buckets.
///
/// *Multiprocessing*
///
/// :class:`BucketKmerTable` objects can be used in multi-processed
/// setups:
/// Adding a large database of sequences to a table can be sped up by
/// splitting the database into smaller chunks and create a separate
/// table for each chunk in separate processes.
/// Eventually, the tables can be merged to one large table using
/// :meth:`from_tables()`.
///
/// Since :class:`BucketKmerTable` supports the *pickle* protocol,
/// the matching step can also be divided into multiple processes, if
/// multiple sequences need to be matched.
///
/// *Storage on hard drive*
///
/// The most time efficient way to read/write a :class:`BucketKmerTable`
/// is the *pickle* format.
///
/// *Indexing and iteration*
///
/// Due to the higher complexity in the *k-mer* lookup compared to
/// :class:`KmerTable`, this class is still indexable but not iterable.
///
/// Examples
/// --------
///
/// Create a *2-mer* index table for some nucleotide sequences:
///
/// >>> table = BucketKmerTable.from_sequences(
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
#[pyclass(module = "biotite.rust.sequence.align")]
pub struct BucketKmerTable(GenericKmerTable<Bucketed>);

#[pymethods]
impl BucketKmerTable {
    /// Reconstruct a table from its pickled state (called by `__reduce__`).
    #[staticmethod]
    fn _from_state(
        py: Python<'_>,
        kmer_alphabet: KmerAlphabet,
        offsets: PyReadonlyArray1<'_, i64>,
        data: Bound<'_, PyBytes>,
    ) -> PyResult<Self> {
        Ok(BucketKmerTable(GenericKmerTable::from_state(
            py,
            kmer_alphabet,
            offsets.as_slice()?,
            data.as_bytes(),
        )?))
    }

    #[getter]
    fn kmer_alphabet(&self, py: Python<'_>) -> Py<PyAny> {
        self.0.kmer_alphabet.clone_ref(py)
    }

    #[getter]
    fn alphabet<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.kmer_alphabet.base_alphabet(py)
    }

    #[getter]
    fn k(&self) -> usize {
        self.0.k
    }

    #[getter]
    fn n_buckets(&self) -> usize {
        self.0.n_slots
    }

    /// from_sequences(k, sequences, ref_ids=None, ignore_masks=None, alphabet=None, spacing=None, n_buckets=None)
    ///
    /// Create a :class:`BucketKmerTable` by storing the positions of
    /// all overlapping *k-mers* from the input `sequences`.
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
    ///     By default the IDs are counted from *0* to *m*.
    /// ignore_masks : sized iterable object of (ndarray, dtype=bool), length=m, optional
    ///     Sequence positions to ignore.
    ///     *k-mers* that involve these sequence positions are not added
    ///     to the table.
    ///     By default, no sequence position is ignored.
    /// alphabet : Alphabet, optional
    ///     The alphabet to use for this table.
    ///     By default, an appropriate alphabet is inferred from the
    ///     input `sequences`.
    /// spacing : None or str or list or ndarray, dtype=int, shape=(k,)
    ///     If provided, spaced *k-mers* are used instead of continuous
    ///     ones.
    /// n_buckets : int, optional
    ///     Set the number of buckets in the table, e.g. to use a
    ///     different load factor.
    ///     It is recommended to use :func:`bucket_number()` for this
    ///     purpose.
    ///     By default, a load factor of approximately 0.8 is used.
    ///
    /// See Also
    /// --------
    /// from_kmers : The same functionality based on already created *k-mers*
    ///
    /// Returns
    /// -------
    /// table : BucketKmerTable
    ///     The newly created table.
    ///
    /// Examples
    /// --------
    ///
    /// >>> sequences = [NucleotideSequence("TTATA"), NucleotideSequence("CTAG")]
    /// >>> table = BucketKmerTable.from_sequences(
    /// ...     2, sequences, ref_ids=[100, 101]
    /// ... )
    /// >>> print(table)
    /// AG: (101, 2)
    /// AT: (100, 2)
    /// CT: (101, 0)
    /// TA: (100, 1), (100, 3), (101, 1)
    /// TT: (100, 0)
    #[staticmethod]
    #[pyo3(signature = (k, sequences, ref_ids=None, ignore_masks=None, alphabet=None, spacing=None, n_buckets=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_sequences(
        py: Python<'_>,
        k: usize,
        sequences: Vec<Bound<'_, PyAny>>,
        ref_ids: Option<Vec<u32>>,
        ignore_masks: Option<Vec<Option<Bound<'_, PyAny>>>>,
        alphabet: Option<Bound<'_, PyAny>>,
        spacing: Option<Bound<'_, PyAny>>,
        n_buckets: Option<usize>,
    ) -> PyResult<Self> {
        Ok(BucketKmerTable(GenericKmerTable::from_sequences(
            py,
            k,
            sequences,
            ref_ids,
            ignore_masks,
            alphabet,
            spacing,
            n_buckets,
        )?))
    }

    /// from_kmers(kmer_alphabet, kmers, ref_ids=None, masks=None, n_buckets=None)
    ///
    /// Create a :class:`BucketKmerTable` by storing the positions of
    /// all input *k-mers*.
    ///
    /// Parameters
    /// ----------
    /// kmer_alphabet : KmerAlphabet
    ///     The :class:`KmerAlphabet` to use for the new table.
    /// kmers : sized iterable object of (ndarray, dtype=np.int64), length=m
    ///     List where each array contains the *k-mer* codes from a
    ///     sequence.
    /// ref_ids : sized iterable object of int, length=m, optional
    ///     The reference IDs for the sequences.
    ///     By default the IDs are counted from *0* to *m*.
    /// masks : sized iterable object of (ndarray, dtype=bool), length=m, optional
    ///     A *k-mer* code at a position, where the corresponding mask
    ///     is false, is not added to the table.
    ///     By default, all positions are added.
    /// n_buckets : int, optional
    ///     Set the number of buckets in the table.
    ///     By default, a load factor of approximately 0.8 is used.
    ///
    /// See Also
    /// --------
    /// from_sequences : The same functionality based on undecomposed sequences
    ///
    /// Returns
    /// -------
    /// table : BucketKmerTable
    ///     The newly created table.
    ///
    /// Examples
    /// --------
    ///
    /// >>> sequences = [ProteinSequence("BIQTITE"), ProteinSequence("NIQBITE")]
    /// >>> kmer_alphabet = KmerAlphabet(ProteinSequence.alphabet, 3)
    /// >>> kmer_codes = [kmer_alphabet.create_kmers(s.code) for s in sequences]
    /// >>> table = BucketKmerTable.from_kmers(kmer_alphabet, kmer_codes)
    /// >>> print(table)
    /// IQT: (0, 1)
    /// IQB: (1, 1)
    /// ITE: (0, 4), (1, 4)
    /// NIQ: (1, 0)
    /// QTI: (0, 2)
    /// QBI: (1, 2)
    /// TIT: (0, 3)
    /// BIQ: (0, 0)
    /// BIT: (1, 3)
    #[staticmethod]
    #[pyo3(signature = (kmer_alphabet, kmers, ref_ids=None, masks=None, n_buckets=None))]
    fn from_kmers(
        py: Python<'_>,
        kmer_alphabet: KmerAlphabet,
        kmers: Vec<Bound<'_, PyAny>>,
        ref_ids: Option<Vec<u32>>,
        masks: Option<Vec<Option<Bound<'_, PyAny>>>>,
        n_buckets: Option<usize>,
    ) -> PyResult<Self> {
        Ok(BucketKmerTable(GenericKmerTable::from_kmers(
            py,
            kmer_alphabet,
            kmers,
            ref_ids,
            masks,
            n_buckets,
        )?))
    }

    /// from_kmer_selection(kmer_alphabet, positions, kmers, ref_ids=None, n_buckets=None)
    ///
    /// Create a :class:`BucketKmerTable` by storing the positions of a
    /// filtered subset of input *k-mers*.
    ///
    /// This can be used to reduce the number of stored *k-mers* using
    /// a *k-mer* subset selector such as :class:`MinimizerSelector`.
    ///
    /// Parameters
    /// ----------
    /// kmer_alphabet : KmerAlphabet
    ///     The :class:`KmerAlphabet` to use for the new table.
    /// positions : sized iterable object of (ndarray, shape=(n,), dtype=uint32), length=m
    ///     List where each array contains the sequence positions of
    ///     the filtered subset of *k-mers* given in `kmers`.
    /// kmers : sized iterable object of (ndarray, shape=(n,), dtype=np.int64), length=m
    ///     List where each array contains the filtered subset of
    ///     *k-mer* codes from a sequence.
    /// ref_ids : sized iterable object of int, length=m, optional
    ///     The reference IDs for the sequences.
    ///     By default the IDs are counted from *0* to *m*.
    /// n_buckets : int, optional
    ///     Set the number of buckets in the table.
    ///     By default, a load factor of approximately 0.8 is used.
    ///
    /// Returns
    /// -------
    /// table : BucketKmerTable
    ///     The newly created table.
    #[staticmethod]
    #[pyo3(signature = (kmer_alphabet, positions, kmers, ref_ids=None, n_buckets=None))]
    fn from_kmer_selection(
        py: Python<'_>,
        kmer_alphabet: KmerAlphabet,
        positions: Vec<Bound<'_, PyAny>>,
        kmers: Vec<Bound<'_, PyAny>>,
        ref_ids: Option<Vec<u32>>,
        n_buckets: Option<usize>,
    ) -> PyResult<Self> {
        Ok(BucketKmerTable(GenericKmerTable::from_kmer_selection(
            py,
            kmer_alphabet,
            positions,
            kmers,
            ref_ids,
            n_buckets,
        )?))
    }

    /// from_tables(tables)
    ///
    /// Create a :class:`BucketKmerTable` by merging the *k-mer*
    /// positions from existing `tables`.
    ///
    /// Parameters
    /// ----------
    /// tables : iterable object of BucketKmerTable
    ///     The tables to be merged.
    ///     All tables must have equal number of buckets and equal
    ///     :class:`KmerAlphabet` objects, i.e. the same *k* and equal
    ///     base alphabets.
    ///
    /// Returns
    /// -------
    /// table : BucketKmerTable
    ///     The newly created table.
    #[staticmethod]
    fn from_tables(py: Python<'_>, tables: Vec<PyRef<'_, BucketKmerTable>>) -> PyResult<Self> {
        let refs: Vec<&GenericKmerTable<Bucketed>> = tables.iter().map(|table| &table.0).collect();
        Ok(BucketKmerTable(GenericKmerTable::from_tables(py, &refs)?))
    }

    /// match_table(table, similarity_rule=None)
    ///
    /// Find matches between the *k-mers* in this table with the
    /// *k-mers* in another `table`.
    ///
    /// Parameters
    /// ----------
    /// table : BucketKmerTable
    ///     The table to be matched.
    ///     Both tables must have equal number of buckets and equal
    ///     :class:`KmerAlphabet` objects.
    /// similarity_rule : SimilarityRule, optional
    ///     If this parameter is given, not only exact *k-mer* matches
    ///     are considered, but also similar ones according to the given
    ///     :class:`SimilarityRule`.
    ///
    /// Returns
    /// -------
    /// matches : ndarray, shape=(n,4), dtype=np.int64
    ///     The *k-mer* matches.
    ///
    /// Notes
    /// -----
    ///
    /// There is no guaranteed order of the reference IDs or
    /// sequence positions in the returned matches.
    #[pyo3(signature = (table, similarity_rule=None))]
    fn match_table<'py>(
        &self,
        py: Python<'py>,
        table: PyRef<'py, BucketKmerTable>,
        similarity_rule: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.0.match_table(py, &table.0, similarity_rule.as_ref())
    }

    /// match(sequence, similarity_rule=None, ignore_mask=None)
    ///
    /// Find matches between the *k-mers* in this table with all
    /// overlapping *k-mers* in the given `sequence`.
    ///
    /// Parameters
    /// ----------
    /// sequence : Sequence
    ///     The sequence to be matched.
    /// similarity_rule : SimilarityRule, optional
    ///     If this parameter is given, not only exact *k-mer* matches
    ///     are considered, but also similar ones according to the given
    ///     :class:`SimilarityRule`.
    /// ignore_mask : ndarray, dtype=bool, optional
    ///     Boolean mask of sequence positions to ignore.
    ///     By default, no sequence position is ignored.
    ///
    /// Returns
    /// -------
    /// matches : ndarray, shape=(n,3), dtype=np.int64
    ///     The *k-mer* matches.
    ///
    /// Notes
    /// -----
    ///
    /// The matches are ordered by the first column.
    #[pyo3(name = "match", signature = (sequence, similarity_rule=None, ignore_mask=None))]
    fn match_<'py>(
        &self,
        py: Python<'py>,
        sequence: Bound<'py, PyAny>,
        similarity_rule: Option<Bound<'py, PyAny>>,
        ignore_mask: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.0.match_sequence(
            py,
            &sequence,
            similarity_rule.as_ref(),
            ignore_mask.as_ref(),
        )
    }

    /// match_kmer_selection(positions, kmers)
    ///
    /// Find matches between the *k-mers* in this table with the given
    /// *k-mer* selection.
    ///
    /// Parameters
    /// ----------
    /// positions : ndarray, shape=(n,), dtype=uint32
    ///     Sequence positions of the filtered subset of *k-mers* given
    ///     in `kmers`.
    /// kmers : ndarray, shape=(n,), dtype=np.int64
    ///     Filtered subset of *k-mer* codes to match against.
    ///
    /// Returns
    /// -------
    /// matches : ndarray, shape=(n,3), dtype=np.int64
    ///     The *k-mer* matches.
    fn match_kmer_selection<'py>(
        &self,
        py: Python<'py>,
        positions: Bound<'py, PyAny>,
        kmers: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.0.match_kmer_selection(py, &positions, &kmers)
    }

    /// count(kmers)
    ///
    /// Count the number of occurences for each given *k-mer* in the
    /// table.
    ///
    /// Parameters
    /// ----------
    /// kmers : ndarray, dtype=np.int64
    ///     The count is returned for these *k-mer* codes.
    ///
    /// Returns
    /// -------
    /// counts : ndarray, dtype=np.int64
    ///     The counts for each given *k-mer*.
    ///
    /// Notes
    /// -----
    /// As each bucket need to be inspected for the actual *k-mer*
    /// entries, this method requires far more computation time than its
    /// :class:`KmerTable` equivalent.
    fn count<'py>(
        &self,
        py: Python<'py>,
        kmers: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        self.0.count(py, &kmers)
    }

    /// Get the *k-mer* codes for all *k-mers* that have at least one
    /// position in the table.
    ///
    /// Returns
    /// -------
    /// kmers : ndarray, shape=(n,), dtype=np.int64
    ///     The *k-mer* codes in ascending order.
    ///
    /// Notes
    /// -----
    /// As each bucket need to be inspected for the actual *k-mer*
    /// entries, this method requires far more computation time than its
    /// :class:`KmerTable` equivalent.
    fn get_kmers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.0.get_kmers(py)
    }

    fn __getitem__<'py>(&self, py: Python<'py>, kmer: Kmer) -> PyResult<Bound<'py, PyArray2<u32>>> {
        self.0.get_item(py, kmer)
    }

    fn __len__(&self) -> usize {
        self.0.n_kmers
    }

    fn __eq__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        match other.extract::<PyRef<'_, BucketKmerTable>>() {
            Ok(other) => self.0.equals(py, &other.0),
            Err(_) => Ok(false),
        }
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        self.0.to_str(py)
    }

    /// Pickle via a direct reconstructor, so unpickling does not build an empty
    /// `n_slots`-sized table first.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let reconstructor = py.get_type::<BucketKmerTable>().getattr("_from_state")?;
        let (offsets, data) = self.0.pickle_state(py);
        let args = PyTuple::new(
            py,
            [
                self.0.kmer_alphabet.clone_ref(py).into_bound(py),
                offsets.into_any(),
                data.into_any(),
            ],
        )?;
        Ok((reconstructor, args))
    }
}
