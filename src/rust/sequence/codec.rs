use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// A codec for encoding/decoding between ASCII symbols and integer codes,
/// based on a given alphabet of allowed symbols.
#[pyclass(module = "biotite.rust.sequence")]
pub struct AlphabetCodec {
    /// Maps ASCII byte value -> symbol code. `illegal_code` for unmapped symbols.
    symbol_to_code: [u8; 256],
    /// The alphabet symbols (as bytes), indexed by code.
    code_to_symbol: Vec<u8>,
    /// The sentinel code that marks an illegal/unmapped symbol.
    illegal_code: u8,
}

#[pymethods]
impl AlphabetCodec {
    /// Create a new codec from the alphabet's symbols.
    ///
    /// This class can only be used if the symbols are ASCII characters.
    ///
    /// Parameters
    /// ----------
    /// symbols : ndarray, dtype=uint8
    ///     The ASCII characters as bytes.
    ///     The index of each symbol becomes its code.
    #[new]
    fn new(symbols: PyReadonlyArray1<u8>) -> PyResult<Self> {
        let symbols = symbols.as_slice()?;
        if symbols.len() > 255 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Alphabet must have at most 255 symbols",
            ));
        }
        // As the symbol code `symbols.len()` is always illegal,
        // it can be used later to check for invalid input symbols
        let illegal_code = symbols.len() as u8;
        // An array based map that maps from symbol to code
        // Since the maximum value of a char is 256
        // the size of the map is known at compile time
        // Initially fill the map with the illegal symbol
        // Consequently, the map will later return the illegal symbol
        // when indexed with a character that is not part of the alphabet
        let mut symbol_to_code = [illegal_code; 256];
        for (i, &symbol) in symbols.iter().enumerate() {
            symbol_to_code[symbol as usize] = i as u8;
        }
        Ok(AlphabetCodec {
            symbol_to_code,
            code_to_symbol: symbols.to_vec(),
            illegal_code,
        })
    }

    /// Encode an array of ASCII symbols into an array of symbol codes.
    ///
    /// Parameters
    /// ----------
    /// symbols : ndarray, dtype=uint8
    ///     The symbols (as bytes) to encode.
    ///
    /// Returns
    /// -------
    /// code : ndarray, dtype=uint8
    ///     The encoded symbol codes.
    ///
    /// Raises
    /// ------
    /// AlphabetError
    ///     If any symbol is not in the alphabet.
    fn encode<'py>(
        &self,
        py: Python<'py>,
        symbols: PyReadonlyArray1<u8>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let symbols = symbols.as_slice()?;
        let mut code = Array1::<u8>::uninit(symbols.len());

        for (&sym, out_code) in symbols.iter().zip(
            code.as_slice_mut()
                .expect("Array not contiguous")
                .iter_mut(),
        ) {
            let c = self.symbol_to_code[sym as usize];
            if c == self.illegal_code {
                let alphabet_error = py
                    .import("biotite.sequence.alphabet")?
                    .getattr("AlphabetError")?;
                return Err(PyErr::from_value(alphabet_error.call1((format!(
                    "Symbol {} is not in the alphabet",
                    repr_char(sym)
                ),))?));
            }
            out_code.write(c);
        }
        // SAFETY: All n elements have been written above
        let code = unsafe { code.assume_init() };
        Ok(code.into_pyarray(py))
    }

    /// Decode an array of symbol codes into an array of ASCII symbols.
    ///
    /// Parameters
    /// ----------
    /// code : ndarray, dtype=uint8
    ///     The symbol codes to decode.
    ///
    /// Returns
    /// -------
    /// symbols : ndarray, dtype=uint8
    ///     The decoded symbols as bytes.
    ///
    /// Raises
    /// ------
    /// AlphabetError
    ///     If any code is not valid in the alphabet.
    fn decode<'py>(
        &self,
        py: Python<'py>,
        code: PyReadonlyArray1<u8>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let code = code.as_slice()?;
        let mut symbols = Array1::<u8>::uninit(code.len());

        for (&c, out_symbol) in code.iter().zip(
            symbols
                .as_slice_mut()
                .expect("Array not contiguous")
                .iter_mut(),
        ) {
            if (c as usize) >= self.code_to_symbol.len() {
                let alphabet_error = py
                    .import("biotite.sequence.alphabet")?
                    .getattr("AlphabetError")?;
                return Err(PyErr::from_value(
                    alphabet_error.call1((format!("'{}' is not a valid code", c),))?,
                ));
            }
            out_symbol.write(self.code_to_symbol[c as usize]);
        }
        // SAFETY: All n elements have been written above
        let symbols = unsafe { symbols.assume_init() };
        Ok(symbols.into_pyarray(py))
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let cls = py
            .import("biotite.rust.sequence")?
            .getattr("AlphabetCodec")?;
        let code_to_symbol = self.code_to_symbol.clone().into_pyarray(py);
        let args = PyTuple::new(py, [code_to_symbol.into_any()])?;
        PyTuple::new(py, [cls.unbind(), args.into_any().unbind()])
    }
}

/// Format a byte value as a Python-style repr for error messages.
fn repr_char(byte: u8) -> String {
    let c = byte as char;
    if c.is_ascii_graphic() || c == ' ' {
        format!("'{}'", c)
    } else {
        format!("'\\x{:02x}'", byte)
    }
}
