//! Containers for efficient retrieval of k-mer matches.

use super::nested::NestedArray;
use pyo3::prelude::*;
use pyo3::types::PyModule;

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
    kmer_alphabet: Py<PyAny>,
}
impl KmerAlphabet {
    fn new(
        py: Python<'_>,
        alphabet: Bound<'_, PyAny>,
        k: usize,
        spacing: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let align = PyModule::import(py, "biotite.sequence.align")?;
        let kmer_alphabet = align
            .getattr("KmerAlphabet")?
            .call1((alphabet, k, spacing))?;
        Ok(Self {
            kmer_alphabet: kmer_alphabet.unbind(),
        })
    }

    fn len(&self, py: Python<'_>) -> PyResult<usize> {
        let wrapped = self.kmer_alphabet.bind(py);
        let len = wrapped.call_method0("__len__")?.extract::<usize>()?;
        Ok(len)
    }
}

#[pyclass]
pub struct KmerTable {
    k: usize,
    kmer_alphabet: KmerAlphabet,
    table: NestedArray<KmerTableElement>,
}

#[pymethods]
impl KmerTable {
    #[staticmethod]
    #[pyo3(signature = (k, sequences, ref_ids=None, ignore_masks=None, alphabet=None, spacing=None))]
    fn from_sequences(
        py: Python<'_>,
        k: usize,
        sequences: Vec<Bound<'_, PyAny>>,
        ref_ids: Option<Vec<u32>>,
        ignore_masks: Option<Vec<Bound<'_, PyAny>>>,
        alphabet: Option<Bound<'_, PyAny>>,
        spacing: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let kmer_alphabet = KmerAlphabet::new(py, alphabet, k, spacing)?;
        let table = NestedArray::new(vec![0; kmer_alphabet.len(py)?]);
        Ok(Self {
            k,
            kmer_alphabet,
            table,
        })
    }
}
