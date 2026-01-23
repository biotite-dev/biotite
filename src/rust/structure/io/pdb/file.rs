//! Low-level PDB file parsing and writing.

use ndarray::Axis;
use num_traits::{Float, FloatConst};
use numpy::ndarray::{
    Array, Array1, Array2, ArrayView1, ArrayView3, ArrayViewD, ArrayViewMut1, Ix3,
};
use numpy::PyArrayMethods;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyEllipsis, PyString, PyType};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::convert::TryInto;
use std::fs;
use std::str::FromStr;

use crate::structure::io::pdb::hybrid36::{decode_hybrid36, encode_hybrid36};
use std::ops::Range;

// Column ranges for ATOM/HETATM records
const RECORD: Range<usize> = 0..6;
const ATOM_ID: Range<usize> = 6..11;
const ATOM_NAME: Range<usize> = 12..16;
const ALT_LOC: Range<usize> = 16..17;
const RES_NAME: Range<usize> = 17..20;
const CHAIN_ID: Range<usize> = 21..22;
const RES_ID: Range<usize> = 22..26;
const INS_CODE: Range<usize> = 26..27;
const COORD_X: Range<usize> = 30..38;
const COORD_Y: Range<usize> = 38..46;
const COORD_Z: Range<usize> = 46..54;
const OCCUPANCY: Range<usize> = 54..60;
const TEMP_F: Range<usize> = 60..66;
const ELEMENT: Range<usize> = 76..78;
const CHARGE: Range<usize> = 78..80;
// Column ranges for CRYST1 record
const CRYST_A: Range<usize> = 6..15;
const CRYST_B: Range<usize> = 15..24;
const CRYST_C: Range<usize> = 24..33;
const CRYST_ALPHA: Range<usize> = 33..40;
const CRYST_BETA: Range<usize> = 40..47;
const CRYST_GAMMA: Range<usize> = 47..54;

// Label as a separate module to indicate that this exception comes
// from biotite
mod biotite {
    pyo3::import_exception!(biotite, InvalidFileError);
    pyo3::import_exception!(biotite.structure, BadStructureError);
}

/// Private base class for :class:`PDBFile`.
///
/// It contains efficient Rust implementation of the methods that would otherwise
/// become major bottlenecks
#[pyclass(subclass)]
pub struct PDBFile {
    /// Lines of text from the PDB file.
    #[pyo3(get)]
    lines: Vec<String>,
    model_start_i: Vec<usize>,
    atom_line_i: Vec<usize>,
}

#[pymethods]
impl PDBFile {
    #[new]
    #[pyo3(signature = (lines=None))]
    fn new(lines: Option<Vec<String>>) -> Self {
        let lines = lines.unwrap_or_default();
        let mut pdb_file = PDBFile {
            lines,
            model_start_i: Vec::new(),
            atom_line_i: Vec::new(),
        };
        pdb_file._index_models_and_atoms();
        pdb_file
    }

    #[getter]
    fn _model_start_i<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(PyArray1::from_iter(
            py,
            self.model_start_i.iter().map(|x| *x as i64),
        ))
    }

    #[getter]
    fn _atom_line_i<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(PyArray1::from_iter(py, self.atom_line_i.iter().map(|x| *x as i64)).to_owned())
    }

    /// read(file)
    ///
    /// Read a PDB file.
    ///
    /// Parameters
    /// ----------
    /// file : file-like object or str
    ///     The file to be read.
    ///     Alternatively a file path can be supplied.
    ///
    /// Returns
    /// -------
    /// file_object : PDBFile
    ///     The parsed file.
    #[classmethod]
    fn read<'py>(cls: &Bound<'py, PyType>, file: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let contents;
        if let Ok(file_path) = file.downcast::<PyString>() {
            // String path
            contents = fs::read_to_string(file_path.extract::<String>()?).map_err(|_| {
                exceptions::PyOSError::new_err(format!("'{}' cannot be read", file_path))
            })?;
        } else if file.hasattr("__fspath__")? {
            // Path-like object (e.g. pathlib.Path)
            let file_path: String = file.call_method0("__fspath__")?.extract()?;
            contents = fs::read_to_string(&file_path).map_err(|_| {
                exceptions::PyOSError::new_err(format!("'{}' cannot be read", file_path))
            })?;
        } else {
            // File-like object
            contents = file.call_method0("read")?.extract::<String>()?;
        }
        let lines: Vec<String> = contents
            .lines()
            // Pad lines with whitespace if lines are shorter than the required 80 characters
            .map(|line| format!("{:<80}", line))
            .collect();
        // Instantiate via the class (cls) to support subclasses
        cls.call1((lines,))
    }

    fn write(&self, file: Bound<'_, PyAny>) -> PyResult<()> {
        let content = self.lines.join("\n");
        if let Ok(file_path) = file.downcast::<PyString>() {
            // String path
            fs::write(file_path.extract::<String>()?, content).map_err(|_| {
                exceptions::PyOSError::new_err(format!("'{}' cannot be written", file_path))
            })?;
        } else if file.hasattr("__fspath__")? {
            // Path-like object (e.g. pathlib.Path)
            let file_path: String = file.call_method0("__fspath__")?.extract()?;
            fs::write(&file_path, content).map_err(|_| {
                exceptions::PyOSError::new_err(format!("'{}' cannot be written", file_path))
            })?;
        } else {
            // File-like object
            file.call_method1("write", (content,))?;
        }
        Ok(())
    }

    /// get_remark(self, number)
    ///
    /// Get the lines containing the *REMARK* records with the given
    /// `number`.
    ///
    /// Parameters
    /// ----------
    /// number : int
    ///     The *REMARK* number, i.e. the `XXX` in ``REMARK XXX``.
    ///
    /// Returns
    /// -------
    /// remark_lines : None or list of str
    ///     The content of the selected *REMARK* lines.
    ///     Each line is an element of this list.
    ///     The ``REMARK XXX `` part of each line is omitted.
    ///     Furthermore, the first line, which always must be empty, is
    ///     not included.
    ///     ``None`` is returned, if the selected *REMARK* records do not
    ///     exist in the file.
    ///
    /// Examples
    /// --------
    ///
    /// >>> import os.path
    /// >>> file = PDBFile.read(os.path.join(path_to_structures, "1l2y.pdb"))
    /// >>> remarks = file.get_remark(900)
    /// >>> print("\n".join(remarks))
    /// RELATED ENTRIES
    /// RELATED ID: 5292   RELATED DB: BMRB
    /// BMRB 5292 IS CHEMICAL SHIFTS FOR TC5B IN BUFFER AND BUFFER
    /// CONTAINING 30 VOL-% TFE.
    /// RELATED ID: 1JRJ   RELATED DB: PDB
    /// 1JRJ IS AN ANALAGOUS C-TERMINAL STRUCTURE.
    /// >>> nonexistent_remark = file.get_remark(999)
    /// >>> print(nonexistent_remark)
    /// None
    fn get_remark(&self, number: i64) -> PyResult<Option<Vec<String>>> {
        const CONTENT_START_COLUMN: usize = 11;

        if !(0..=999).contains(&number) {
            return Err(exceptions::PyValueError::new_err(
                "The number must be in range 0-999",
            ));
        }
        let remark_string = format!("REMARK {:>3}", number);
        let mut remark_lines: Vec<String> = self
            .lines
            .iter()
            .filter(|line| line.starts_with(&remark_string))
            .map(|line| line[CONTENT_START_COLUMN..].to_owned())
            .collect();
        match remark_lines.len() {
            0 => Ok(None),
            // Remove first empty line
            _ => {
                remark_lines.remove(0);
                Ok(Some(remark_lines))
            }
        }
    }

    /// get_model_count(self)
    ///
    /// Get the number of models contained in the PDB file.
    ///
    /// Returns
    /// -------
    /// model_count : int
    ///     The number of models.
    fn get_model_count(&self) -> usize {
        self.model_start_i.len()
    }

    /// get_coord(self, model=None)
    ///
    /// Get only the coordinates from the PDB file.
    ///
    /// Parameters
    /// ----------
    /// model : int, optional
    ///     If this parameter is given, the function will return a
    ///     2D coordinate array from the atoms corresponding to the
    ///     given model number (starting at 1).
    ///     Negative values are used to index models starting from the
    ///     last model instead of the first model.
    ///     If this parameter is omitted, an 3D coordinate array
    ///     containing all models will be returned, even if
    ///     the structure contains only one model.
    ///
    /// Returns
    /// -------
    /// coord : ndarray, shape=(m,n,3) or shape=(n,3), dtype=float
    ///     The coordinates read from the ATOM and HETATM records of the
    ///     file.
    ///
    /// Notes
    /// -----
    /// Note that :func:`get_coord()` may output more coordinates than
    /// the atom array (stack) from the corresponding
    /// :func:`get_structure()` call has.
    /// The reason for this is, that :func:`get_structure()` filters
    /// *altloc* IDs, while `get_coord()` does not.
    ///
    /// Examples
    /// --------
    /// Read an :class:`AtomArrayStack` from multiple PDB files, where
    /// each PDB file contains the same atoms but different positions.
    /// This is an efficient approach when a trajectory is spread into
    /// multiple PDB files, as done e.g. by the *Rosetta* modeling
    /// software.
    ///
    /// For the purpose of this example, the PDB files are created from
    /// an existing :class:`AtomArrayStack`.
    ///
    /// >>> import os.path
    /// >>> from tempfile import gettempdir
    /// >>> file_names = []
    /// >>> for i in range(atom_array_stack.stack_depth()):
    /// ...     pdb_file = PDBFile()
    /// ...     pdb_file.set_structure(atom_array_stack[i])
    /// ...     file_name = os.path.join(gettempdir(), f"model_{i+1}.pdb")
    /// ...     pdb_file.write(file_name)
    /// ...     file_names.append(file_name)
    /// >>> print(file_names)
    /// ['...model_1.pdb', '...model_2.pdb', ..., '...model_38.pdb']
    ///
    /// Now the PDB files are used to create an :class:`AtomArrayStack`,
    /// where each model represents a different model.
    ///
    /// Construct a new :class:`AtomArrayStack` with annotations taken
    /// from one of the created files used as template and coordinates
    /// from all of the PDB files.
    ///
    /// >>> template_file = PDBFile.read(file_names[0])
    /// >>> template = template_file.get_structure()
    /// >>> coord = []
    /// >>> for i, file_name in enumerate(file_names):
    /// ...     pdb_file = PDBFile.read(file_name)
    /// ...     coord.append(pdb_file.get_coord(model=1))
    /// >>> new_stack = from_template(template, np.array(coord))
    ///
    /// The newly created :class:`AtomArrayStack` should now be equal to
    /// the :class:`AtomArrayStack` the PDB files were created from.
    ///
    /// >>> print(np.allclose(new_stack.coord, atom_array_stack.coord))
    /// True
    #[pyo3(signature = (model=None))]
    fn get_coord<'py>(&self, py: Python<'py>, model: Option<isize>) -> PyResult<Bound<'py, PyAny>> {
        match model {
            Some(model_number) => {
                let atom_line_i = self.get_atom_indices(model_number)?;
                let mut coord = Array::zeros((atom_line_i.len(), 3));
                for (atom_i, line_i) in atom_line_i.iter().enumerate() {
                    let line = &self.lines[*line_i];
                    if line.len() < 80 {
                        return Err(biotite::InvalidFileError::new_err("Line is too short"));
                    }
                    coord[[atom_i, 0]] = parse_float_from_string(line, &COORD_X)?;
                    coord[[atom_i, 1]] = parse_float_from_string(line, &COORD_Y)?;
                    coord[[atom_i, 2]] = parse_float_from_string(line, &COORD_Z)?;
                }
                Ok(coord.into_pyarray(py).into_any())
            }

            None => {
                let length = self.get_model_length()?;
                let mut coord = Array::zeros((self.atom_line_i.len(), 3));
                for (atom_i, line_i) in self.atom_line_i.iter().enumerate() {
                    let line = &self.lines[*line_i];
                    if line.len() < 80 {
                        return Err(biotite::InvalidFileError::new_err("Line is too short"));
                    }
                    coord[[atom_i, 0]] = parse_float_from_string(line, &COORD_X)?;
                    coord[[atom_i, 1]] = parse_float_from_string(line, &COORD_Y)?;
                    coord[[atom_i, 2]] = parse_float_from_string(line, &COORD_Z)?;
                }
                Ok(coord
                    .into_shape_with_order((self.model_start_i.len(), length, 3))
                    .expect("Model length is invalid")
                    .into_pyarray(py)
                    .into_any())
            }
        }
    }

    /// get_b_factor(self, model=None)
    ///
    /// Get only the B-factors from the PDB file.
    ///
    /// Parameters
    /// ----------
    /// model : int, optional
    ///     If this parameter is given, the function will return a
    ///     1D B-factor array from the atoms corresponding to the
    ///     given model number (starting at 1).
    ///     Negative values are used to index models starting from the
    ///     last model instead of the first model.
    ///     If this parameter is omitted, an 2D B-factor array
    ///     containing all models will be returned, even if
    ///     the structure contains only one model.
    ///
    /// Returns
    /// -------
    /// b_factor : ndarray, shape=(m,n) or shape=(n,), dtype=float
    ///     The B-factors read from the ATOM and HETATM records of the
    ///     file.
    ///
    /// Notes
    /// -----
    /// Note that :func:`get_b_factor()` may output more B-factors
    /// than the atom array (stack) from the corresponding
    /// :func:`get_structure()` call has atoms.
    /// The reason for this is, that :func:`get_structure()` filters
    /// *altloc* IDs, while `get_b_factor()` does not.
    #[pyo3(signature = (model=None))]
    fn get_b_factor<'py>(
        &self,
        py: Python<'py>,
        model: Option<isize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match model {
            Some(model_number) => {
                let atom_line_i = self.get_atom_indices(model_number)?;
                let mut b_factor = Array::zeros(atom_line_i.len());
                for (atom_i, line_i) in atom_line_i.iter().enumerate() {
                    let line = &self.lines[*line_i];
                    if line.len() < 80 {
                        return Err(biotite::InvalidFileError::new_err("Line is too short"));
                    }
                    b_factor[atom_i] = parse_float_from_string(line, &TEMP_F)?;
                }
                Ok(b_factor.into_pyarray(py).into_any())
            }

            None => {
                let length = self.get_model_length()?;
                let mut b_factor = Array::zeros(self.atom_line_i.len());
                for (atom_i, line_i) in self.atom_line_i.iter().enumerate() {
                    let line = &self.lines[*line_i];
                    if line.len() < 80 {
                        return Err(biotite::InvalidFileError::new_err("Line is too short"));
                    }
                    b_factor[atom_i] = parse_float_from_string(line, &TEMP_F)?;
                }
                Ok(b_factor
                    .into_shape_with_order((self.model_start_i.len(), length))
                    .expect("Model length is invalid")
                    .into_pyarray(py)
                    .into_any())
            }
        }
    }

    /// get_structure(self, model=None, altloc="first", extra_fields=[], include_bonds=False)
    ///
    /// Get an :class:`AtomArray` or :class:`AtomArrayStack` from the PDB file.
    ///
    /// This function parses standard base-10 PDB files as well as
    /// hybrid-36 PDB.
    ///
    /// Parameters
    /// ----------
    /// model : int, optional
    ///     If this parameter is given, the function will return an
    ///     :class:`AtomArray` from the atoms corresponding to the given
    ///     model number (starting at 1).
    ///     Negative values are used to index models starting from the
    ///     last model instead of the first model.
    ///     If this parameter is omitted, an :class:`AtomArrayStack`
    ///     containing all models will be returned, even if the
    ///     structure contains only one model.
    /// altloc : {'first', 'occupancy', 'all'}
    ///     This parameter defines how *altloc* IDs are handled:
    ///         - ``'first'`` - Use atoms that have the first
    ///           *altloc* ID appearing in a residue.
    ///         - ``'occupancy'`` - Use atoms that have the *altloc* ID
    ///           with the highest occupancy for a residue.
    ///         - ``'all'`` - Use all atoms.
    ///           Note that this leads to duplicate atoms.
    ///           When this option is chosen, the ``altloc_id``
    ///           annotation array is added to the returned structure.
    /// extra_fields : list of str, optional
    ///     The strings in the list are optional annotation categories
    ///     that should be stored in the output array or stack.
    ///     These are valid values:
    ///     ``'atom_id'``, ``'b_factor'``, ``'occupancy'`` and
    ///     ``'charge'``.
    /// include_bonds : bool, optional
    ///     If set to true, a :class:`BondList` will be created for the
    ///     resulting :class:`AtomArray` containing the bond information
    ///     from the file.
    ///     Bonds, whose order could not be determined from the
    ///     *Chemical Component Dictionary*
    ///     (e.g. especially inter-residue bonds),
    ///     have :attr:`BondType.ANY`, since the PDB format itself does
    ///     not support bond orders.
    ///
    /// Returns
    /// -------
    /// array : AtomArray or AtomArrayStack
    ///     The return type depends on the `model` parameter.
    #[pyo3(signature = (model=None, altloc="first", extra_fields=None, include_bonds=false))]
    fn get_structure<'py>(
        &self,
        py: Python<'py>,
        model: Option<isize>,
        altloc: &str,
        extra_fields: Option<Vec<String>>,
        include_bonds: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let struc = PyModule::import(py, "biotite.structure")?;

        let atom_line_i;
        let length;
        let atoms = match model {
            Some(model_number) => {
                atom_line_i = self.get_atom_indices(model_number)?;
                length = atom_line_i.len();
                struc.getattr("AtomArray")?.call1((length,))?
            }
            None => {
                atom_line_i = self.get_atom_indices(1)?;
                length = self.get_model_length()?;
                let depth = self.model_start_i.len();
                struc.getattr("AtomArrayStack")?.call1((depth, length))?
            }
        };

        atoms.setattr("coord", self.get_coord(py, model)?)?;

        let mut include_atom_id = false;
        let mut include_b_factor = false;
        let mut include_occupancy = false;
        let mut include_charge = false;
        if let Some(extra_fields) = extra_fields {
            for field in extra_fields {
                match field.as_str() {
                    "atom_id" => include_atom_id = true,
                    "b_factor" => include_b_factor = true,
                    "occupancy" => include_occupancy = true,
                    "charge" => include_charge = true,
                    _ => {
                        return Err(exceptions::PyValueError::new_err(format!(
                            "Unknown extra field: {}",
                            field
                        )))
                    }
                }
            }
        }
        if altloc == "occupancy" {
            include_occupancy = true;
        }

        let mut chain_id: Array2<u32> = Array::zeros((atom_line_i.len(), 4));
        let mut res_id: Array1<i64> = Array::zeros(atom_line_i.len());
        let mut ins_code: Array2<u32> = Array::zeros((atom_line_i.len(), 1));
        let mut res_name: Array2<u32> = Array::zeros((atom_line_i.len(), 5));
        let mut hetero: Array1<bool> = Array::default(atom_line_i.len());
        let mut atom_name: Array2<u32> = Array::zeros((atom_line_i.len(), 6));
        let mut element: Array2<u32> = Array::zeros((atom_line_i.len(), 2));
        let mut altloc_id: Array2<u32> = Array::zeros((atom_line_i.len(), 1));

        let mut atom_id: Array1<i64>;
        if include_atom_id || include_bonds {
            atom_id = Array::zeros(atom_line_i.len());
        } else {
            // Array will not be used
            atom_id = Array::zeros(0);
        }
        let mut b_factor: Array1<f64>;
        if include_b_factor {
            b_factor = Array::zeros(atom_line_i.len());
        } else {
            b_factor = Array::zeros(0);
        }
        let mut occupancy: Array1<f64>;
        if include_occupancy {
            occupancy = Array::zeros(atom_line_i.len());
        } else {
            occupancy = Array::zeros(0);
        }
        let mut charge: Array1<i64>;
        if include_charge {
            charge = Array::zeros(atom_line_i.len());
        } else {
            charge = Array::zeros(0);
        }

        // Iterate over ATOM and HETATM records to write annotation arrays
        for (atom_i, line_i) in atom_line_i.iter().enumerate() {
            let line = &self.lines[*line_i];
            if line.len() < 80 {
                return Err(biotite::InvalidFileError::new_err("Line is too short"));
            }
            write_string_to_array(&mut chain_id.row_mut(atom_i), line[CHAIN_ID.clone()].trim());
            res_id[atom_i] = decode_hybrid36(&line[RES_ID.clone()])?;
            write_string_to_array(&mut ins_code.row_mut(atom_i), line[INS_CODE.clone()].trim());
            write_string_to_array(&mut res_name.row_mut(atom_i), line[RES_NAME.clone()].trim());
            hetero[atom_i] = &line[RECORD.clone()] != "ATOM  ";
            write_string_to_array(
                &mut atom_name.row_mut(atom_i),
                line[ATOM_NAME.clone()].trim(),
            );
            write_string_to_array(&mut element.row_mut(atom_i), line[ELEMENT.clone()].trim());
            write_string_to_array(&mut altloc_id.row_mut(atom_i), &line[ALT_LOC.clone()]);

            // Set optional annotation arrays
            if include_atom_id || include_bonds {
                atom_id[atom_i] = decode_hybrid36(&line[ATOM_ID.clone()])?;
            }
            if include_b_factor {
                b_factor[atom_i] = parse_number(&line[TEMP_F.clone()])?;
            }
            if include_occupancy {
                occupancy[atom_i] = parse_number(&line[OCCUPANCY.clone()])?;
            }
            if include_charge {
                let charge_raw = line[CHARGE.clone()].as_bytes();
                // Allow both "X+" and "+X"
                charge[atom_i] = match charge_raw[0] {
                    b' ' => 0,
                    b'+' => parse_digit(&charge_raw[1])?,
                    b'-' => -parse_digit(&charge_raw[1])?,
                    _ => match charge_raw[1] {
                        b'+' => parse_digit(&charge_raw[0])?,
                        b'-' => -parse_digit(&charge_raw[0])?,
                        _ => Err(biotite::InvalidFileError::new_err(format!(
                            "'{}' is not a valid charge identifier",
                            &line[CHARGE.clone()]
                        )))?,
                    },
                }
            }
        }

        // Set annotations in the structure
        // String annotations need to be converted from uint32 arrays to unicode arrays
        let convert_uint32_to_unicode = PyModule::import(py, "biotite.structure.io.util")?
            .getattr("convert_uint32_to_unicode")?;
        atoms.call_method1(
            "set_annotation",
            (
                "chain_id",
                convert_uint32_to_unicode.call1((chain_id.into_pyarray(py),))?,
            ),
        )?;
        atoms.call_method1("set_annotation", ("res_id", res_id.into_pyarray(py)))?;
        atoms.call_method1(
            "set_annotation",
            (
                "ins_code",
                convert_uint32_to_unicode.call1((ins_code.into_pyarray(py),))?,
            ),
        )?;
        atoms.call_method1(
            "set_annotation",
            (
                "res_name",
                convert_uint32_to_unicode.call1((res_name.into_pyarray(py),))?,
            ),
        )?;
        atoms.call_method1("set_annotation", ("hetero", hetero.into_pyarray(py)))?;
        atoms.call_method1(
            "set_annotation",
            (
                "atom_name",
                convert_uint32_to_unicode.call1((atom_name.into_pyarray(py),))?,
            ),
        )?;
        atoms.call_method1(
            "set_annotation",
            (
                "element",
                convert_uint32_to_unicode.call1((element.into_pyarray(py),))?,
            ),
        )?;
        atoms.call_method1(
            "set_annotation",
            (
                "altloc_id",
                convert_uint32_to_unicode.call1((altloc_id.into_pyarray(py),))?,
            ),
        )?;
        // Added for convenience in altloc filtering, removed later if not in 'extra_fields'
        if include_atom_id || include_bonds {
            atoms.call_method1("set_annotation", ("atom_id", atom_id.into_pyarray(py)))?;
        }
        if include_b_factor {
            atoms.call_method1("set_annotation", ("b_factor", b_factor.into_pyarray(py)))?;
        }
        if include_occupancy {
            atoms.call_method1("set_annotation", ("occupancy", occupancy.into_pyarray(py)))?;
        }
        if include_charge {
            atoms.call_method1("set_annotation", ("charge", charge.into_pyarray(py)))?;
        }

        match self.parse_box(py) {
            Ok(Some(box_vectors)) => {
                if atoms.is_instance(&struc.getattr("AtomArrayStack")?)? {
                    // For AtomArrayStack, repeat box for each model
                    let np = PyModule::import(py, "numpy")?;
                    let stack_depth: i64 = atoms.call_method0("stack_depth")?.extract()?;
                    // box[np.newaxis, ...]
                    let box_expanded =
                        box_vectors.get_item((np.getattr("newaxis")?, py.Ellipsis()))?;
                    // np.repeat(box[np.newaxis, ...], stack_depth, axis=0)
                    let box_repeated = np.call_method(
                        "repeat",
                        (box_expanded, stack_depth),
                        Some(&[("axis", 0)].into_py_dict(py)?),
                    )?;
                    atoms.setattr("box", box_repeated)?
                } else {
                    atoms.setattr("box", box_vectors)?
                }
            }
            Ok(None) => {}
            Err(_) => {
                PyErr::warn(
                    py,
                    &py.get_type::<pyo3::exceptions::PyUserWarning>(),
                    c_str!("File contains invalid 'CRYST1' record, box is ignored"),
                    0,
                )?;
            }
        };

        let atoms = filter_altloc(py, atoms, altloc)?;

        if include_bonds {
            let atom_id: Bound<'py, PyArray1<i64>> = atoms.getattr("atom_id")?.extract()?;
            let bond_array = self.parse_bonds(py, &atom_id.readonly().as_array())?;
            let bond_list = struc
                .getattr("BondList")?
                .call1((atom_id.len()?, bond_array))?;
            atoms.setattr("bonds", bond_list)?;
        }

        if !include_atom_id {
            atoms.call_method1("del_annotation", ("atom_id",))?;
        }
        if altloc != "all" {
            atoms.call_method1("del_annotation", ("altloc_id",))?;
        }
        Ok(atoms)
    }

    /// set_structure(self, atoms, hybrid36=False)
    ///
    /// Set the :class:`AtomArray` or :class:`AtomArrayStack` for the
    /// file.
    ///
    /// This makes also use of the optional annotation arrays
    /// ``'atom_id'``, ``'b_factor'``, ``'occupancy'`` and ``'charge'``.
    /// If the atom array (stack) contains the annotation ``'atom_id'``,
    /// these values will be used for atom numbering instead of
    /// continuous numbering.
    ///
    /// Parameters
    /// ----------
    /// array : AtomArray or AtomArrayStack
    ///     The array or stack to be saved into this file. If a stack
    ///     is given, each array in the stack is saved as separate
    ///     model.
    /// hybrid36 : bool, optional
    ///     Defines wether the file should be written in hybrid-36
    ///     format.
    ///
    /// Notes
    /// -----
    /// If `array` has an associated :class:`BondList`, ``CONECT``
    /// records are also written for all non-water hetero residues
    /// and all inter-residue connections.
    fn set_structure(
        &mut self,
        py: Python<'_>,
        atoms: Bound<'_, PyAny>,
        hybrid36: bool,
    ) -> PyResult<()> {
        // Write the CRYST1 record if box is present
        let box_vectors = atoms.getattr("box")?;
        if !box_vectors.is_none() {
            // For AtomArrayStack, the box is 3D - use first model's box
            let box_ndim: usize = box_vectors.getattr("ndim")?.extract()?;
            let box_to_write = match box_ndim {
                3 => box_vectors.get_item(0)?,
                2 => box_vectors,
                _ => Err(biotite::BadStructureError::new_err(
                    "Invalid box dimensions",
                ))?,
            };
            self.write_box(py, box_to_write)?;
        }

        // Create a binding first to avoid borrowing a dropped array
        let coord_array = atoms
            .getattr("coord")?
            .extract::<Bound<'_, PyArrayDyn<f32>>>()?
            .readonly();
        let coord: ArrayViewD<f32> = coord_array.as_array();
        let coord: ArrayView3<f32> = match coord.ndim() {
            2 => coord.insert_axis(Axis(0)),
            3 => coord,
            _ => Err(biotite::BadStructureError::new_err(
                "Invalid dimensions of coordinates",
            ))?,
        }
        .into_dimensionality::<Ix3>()
        .unwrap();
        let is_multi_model = coord.shape()[0] > 1;

        let chain_id = get_annotation_as_strings(&atoms, "chain_id")?;
        let res_id_numpy = get_annotation_as_type(&atoms, "res_id", "int64")?.readonly();
        let res_id: ArrayView1<i64> = res_id_numpy.as_array();
        let ins_code = get_annotation_as_strings(&atoms, "ins_code")?;
        let res_name = get_annotation_as_strings(&atoms, "res_name")?;
        let hetero_numpy = get_annotation_as_type(&atoms, "hetero", "bool")?.readonly();
        let hetero: ArrayView1<bool> = hetero_numpy.as_array();
        let atom_name = get_annotation_as_strings(&atoms, "atom_name")?;
        let element = get_annotation_as_strings(&atoms, "element")?;

        let atom_id_numpy: Option<PyReadonlyArray1<i64>> = if atoms.hasattr("atom_id")? {
            Some(get_annotation_as_type(&atoms, "atom_id", "int64")?.readonly())
        } else {
            None
        };
        let atom_id = atom_id_numpy.as_ref().map(|arr| arr.as_array());
        let b_factor_numpy: Option<PyReadonlyArray1<f64>> = if atoms.hasattr("b_factor")? {
            Some(get_annotation_as_type(&atoms, "b_factor", "float64")?.readonly())
        } else {
            None
        };
        let b_factor = b_factor_numpy.as_ref().map(|arr| arr.as_array());
        let occupancy_numpy: Option<PyReadonlyArray1<f64>> = if atoms.hasattr("occupancy")? {
            Some(get_annotation_as_type(&atoms, "occupancy", "float64")?.readonly())
        } else {
            None
        };
        let occupancy = occupancy_numpy.as_ref().map(|arr| arr.as_array());
        let charge_numpy: Option<PyReadonlyArray1<i64>> = if atoms.hasattr("charge")? {
            Some(get_annotation_as_type(&atoms, "charge", "int64")?.readonly())
        } else {
            None
        };
        let charge = charge_numpy.as_ref().map(|arr| arr.as_array());

        // These will contain the ATOM records for each atom
        // These are reused in every model by adding the coordinates to the string
        // This procedure aims to increase the performance is repetitive formatting is omitted
        let mut prefix: Vec<String> = Vec::new();
        let mut suffix: Vec<String> = Vec::new();

        for i in 0..coord.shape()[1] {
            let element_i = &element[i];
            let atom_name_i = &atom_name[i];

            let raw_atom_id_i = atom_id.as_ref().map_or((i + 1) as i64, |arr| arr[i]);
            let atom_id_i = if hybrid36 {
                encode_hybrid36(raw_atom_id_i, 5)?
            } else {
                truncate_id(raw_atom_id_i, 99999).to_string()
            };

            let res_id_i = if hybrid36 {
                encode_hybrid36(res_id[i], 4)?
            } else {
                truncate_id(res_id[i], 9999).to_string()
            };

            prefix.push(format!(
                "{:6}{:>5} {:4} {:>3} {:1}{:>4}{:1}   ",
                if hetero[i] { "HETATM" } else { "ATOM" },
                atom_id_i,
                if element_i.len() == 1 && atom_name_i.len() < 4 {
                    format!(" {}", atom_name_i)
                } else {
                    atom_name_i.to_string()
                },
                res_name[i],
                chain_id[i],
                res_id_i,
                ins_code[i],
            ));

            suffix.push(format!(
                "{:>6.2}{:>6.2}          {:>2}{}",
                occupancy.as_ref().map_or(1f64, |arr| arr[i]),
                b_factor.as_ref().map_or(0f64, |arr| arr[i]),
                element_i,
                charge.as_ref().map_or(String::from("  "), |arr| {
                    let c = arr[i];
                    match c.cmp(&0) {
                        Ordering::Greater => format!("{:1}+", c),
                        Ordering::Less => format!("{:1}-", -c),
                        Ordering::Equal => String::from("  "),
                    }
                }),
            ));
        }

        for model_i in 0..coord.shape()[0] {
            if is_multi_model {
                self.lines.push(format!("MODEL {:>8}", model_i + 1));
            }
            for atom_i in 0..coord.shape()[1] {
                let coord_string = format!(
                    "{:>8.3}{:>8.3}{:>8.3}",
                    coord[[model_i, atom_i, 0]],
                    coord[[model_i, atom_i, 1]],
                    coord[[model_i, atom_i, 2]],
                );
                self.lines
                    .push(prefix[atom_i].clone() + &coord_string + &suffix[atom_i]);
            }
            if is_multi_model {
                self.lines.push(String::from("ENDMDL"));
            }
        }

        // Write CONECT records if bonds are present
        let bonds = atoms.getattr("bonds")?;
        if !bonds.is_none() {
            // Bond type is unused since PDB does not support bond orders
            let (bonds_array, _bond_types): (PyReadonlyArray2<'_, i32>, PyReadonlyArray2<'_, i8>) =
                bonds.call_method0("get_all_bonds")?.extract()?;
            let atom_id_for_bonds: Bound<'_, PyArray1<i64>> = match &atom_id_numpy {
                Some(arr) => arr.as_array().to_owned().into_pyarray(py),
                // If atom_id annotation is not present, generate sequential IDs
                None => PyArray1::from_iter(py, (1..=coord.shape()[1] as i64).collect::<Vec<_>>()),
            };
            self.write_bonds(bonds_array, atom_id_for_bonds.readonly())?;
        }

        self._index_models_and_atoms();
        Ok(())
    }

    /// Index lines in the file that correspond to starts of new models and to
    /// `ATOM` or `HETATM` records.
    /// Must be called after the content of the file has been changed.
    fn _index_models_and_atoms(&mut self) {
        self.atom_line_i = self
            .lines
            .iter()
            .enumerate()
            .filter(|(_i, line)| line.starts_with("ATOM") || line.starts_with("HETATM"))
            .map(|(i, _line)| i)
            .collect();
        self.model_start_i = self
            .lines
            .iter()
            .enumerate()
            .filter(|(_i, line)| line.starts_with("MODEL"))
            .map(|(i, _line)| i)
            .collect();
        // It could be an empty file or a file with a single model,
        // where the 'MODEL' line is missing
        if self.model_start_i.is_empty() && !self.atom_line_i.is_empty() {
            self.model_start_i = vec![0]
        }
    }

    /// Set the line at the given index to the given string.
    fn _set_line(&mut self, line_index: usize, line: String) {
        self.lines[line_index] = line;
    }
}

impl PDBFile {
    /// Get indices to `ATOM` and `HETATM` records within the given model number.
    fn get_atom_indices(&self, model: isize) -> PyResult<Vec<usize>> {
        // Find the model index corresponding to the given model number
        let model_i: isize;
        match model.cmp(&0) {
            Ordering::Greater => model_i = model - 1,
            Ordering::Less => model_i = self.model_start_i.len() as isize + model,
            Ordering::Equal => {
                return Err(exceptions::PyValueError::new_err(
                    "Model index must not be 0",
                ))
            }
        };
        if model_i >= self.model_start_i.len() as isize || model_i < 0 {
            return Err(exceptions::PyValueError::new_err(format!(
                "The file has {} models, the given model {} does not exist",
                self.model_start_i.len(),
                model
            )));
        }

        // Get the start and stop line index for this model index
        let (model_start, model_stop) = match model_i.cmp(&(self.model_start_i.len() as isize - 1))
        {
            Ordering::Less => (
                self.model_start_i[model_i as usize],
                self.model_start_i[(model_i + 1) as usize],
            ),
            // Last model -> Model reaches to end of file
            Ordering::Equal => (self.model_start_i[model_i as usize], self.lines.len()),
            // This case was excluded above
            _ => panic!("This branch should not be reached"),
        };

        // Get the atom records within these line boundaries
        Ok(self
            .atom_line_i
            .iter()
            .copied()
            .filter(|i| *i >= model_start && *i < model_stop)
            .collect())
    }

    /// Get the number of atoms in each model of the PDB file.
    /// A `PyErr` is returned if the number of atoms per model differ from each other.
    fn get_model_length(&self) -> PyResult<usize> {
        let n_models = self.model_start_i.len();
        let mut length: Option<usize> = None;
        for model_i in 0..n_models {
            let model_start: usize = self.model_start_i[model_i];
            let model_stop: usize = if model_i + 1 < n_models {
                self.model_start_i[model_i + 1]
            } else {
                self.lines.len()
            };
            let model_length = self
                .atom_line_i
                .iter()
                .filter(|&line_i| *line_i >= model_start && *line_i < model_stop)
                .count();
            match length {
                None => length = Some(model_length),
                Some(l) => {
                    if model_length != l {
                        return Err(biotite::InvalidFileError::new_err(
                            "Inconsistent number of models",
                        ));
                    }
                }
            };
        }

        match length {
            None => panic!("Length cannot be 'None'"),
            Some(l) => Ok(l),
        }
    }

    /// Parse the `CRYST1` record of the PDB file to obtain the box dimensions.
    fn parse_box<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
        let struc = PyModule::import(py, "biotite.structure")?;

        for line in self.lines.iter() {
            if line.starts_with("CRYST1") {
                if line.len() < 80 {
                    return Err(biotite::InvalidFileError::new_err("Line is too short"));
                }
                let len_a: f64 = parse_number(&line[CRYST_A.clone()])?;
                let len_b: f64 = parse_number(&line[CRYST_B.clone()])?;
                let len_c: f64 = parse_number(&line[CRYST_C.clone()])?;
                let alpha: f64 = parse_number(&line[CRYST_ALPHA.clone()])?;
                let beta: f64 = parse_number(&line[CRYST_BETA.clone()])?;
                let gamma: f64 = parse_number(&line[CRYST_GAMMA.clone()])?;
                return Ok(Some(
                    struc
                        .getattr("vectors_from_unitcell")?
                        .call1((
                            len_a,
                            len_b,
                            len_c,
                            deg2rad(alpha),
                            deg2rad(beta),
                            deg2rad(gamma),
                        ))?
                        .extract()?,
                ));
            }
        }
        // File has no 'CRYST1' record
        Ok(None)
    }

    /// Parse the `CONECT` records of the PDB file to obtain a 2D *NumPy* array
    /// of indices.
    /// Each index points to the respective atom in the `AtomArray`.
    /// The `atom_id` annotation array is required to map the atom IDs in `CONECT` records
    /// to atom indices.
    fn parse_bonds<'py>(
        &self,
        py: Python<'py>,
        atom_id: &ArrayView1<i64>,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        // Mapping from atom ids to indices in an AtomArray
        let mut atom_id_to_index: HashMap<i64, u32> = HashMap::new();
        for (i, id) in atom_id.iter().enumerate() {
            atom_id_to_index.insert(*id, i as u32);
        }

        // Cannot preemptively determine number of bonds
        // -> Memory allocation for all bonds is not possible
        // -> No benefit in finding 'CONECT' record lines prior to iteration
        let mut bonds: Vec<(u32, u32)> = Vec::new();
        for line in self.lines.iter() {
            if line.starts_with("CONECT") {
                if line.len() < 80 {
                    return Err(biotite::InvalidFileError::new_err("Line is too short"));
                }
                // Extract ID of center atom
                let center_index = atom_id_to_index.get(&decode_hybrid36(&line[ATOM_ID.clone()])?);
                match center_index {
                    None => continue, // Atom ID is not in the AtomArray (probably removed altloc)
                    Some(center_index) => {
                        // Iterate over atom IDs bonded to center atom
                        for i in (11..31).step_by(5) {
                            match &decode_hybrid36(&line[i..i + 5]) {
                                Ok(bonded_id) => {
                                    let bonded_index = atom_id_to_index.get(bonded_id);
                                    match bonded_index {
                                        None => continue, // Atom ID is not in the AtomArray
                                        Some(bonded_index) => {
                                            bonds.push((*center_index, *bonded_index))
                                        }
                                    }
                                }
                                // String is empty -> no further IDs
                                Err(_) => break,
                            };
                        }
                    }
                }
            }
        }

        let mut bond_array: Array2<u32> = Array::zeros((bonds.len(), 2));
        for (i, (center_id, bonded_id)) in bonds.iter().enumerate() {
            bond_array[[i, 0]] = *center_id;
            bond_array[[i, 1]] = *bonded_id;
        }
        Ok(bond_array.into_pyarray(py))
    }

    /// Write the `CRYST1` record to this [`PDBFile`] based on the given box vectors.
    fn write_box(&mut self, py: Python<'_>, box_vectors: Bound<'_, PyAny>) -> PyResult<()> {
        let struc = PyModule::import(py, "biotite.structure")?;
        let (len_a, len_b, len_c, alpha, beta, gamma): (f64, f64, f64, f64, f64, f64) = struc
            .getattr("unitcell_from_vectors")?
            .call1((box_vectors,))?
            .extract()?;
        self.lines.push(format!(
            "CRYST1{:>9.3}{:>9.3}{:>9.3}{:>7.2}{:>7.2}{:>7.2} P 1           1          ",
            len_a,
            len_b,
            len_c,
            rad2deg(alpha),
            rad2deg(beta),
            rad2deg(gamma)
        ));
        Ok(())
    }

    /// Write `CONECT` records to this [`PDBFile`] based on the given `bonds`
    /// array containing indices pointing to bonded atoms in the `AtomArray`.
    /// The `atom_id` annotation array is required to map the atom IDs in `CONECT` records
    /// to atom indices.
    fn write_bonds<'py>(
        &mut self,
        bonds: PyReadonlyArray2<'py, i32>,
        atom_id: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<()> {
        let bonds = bonds.as_array();
        let atom_id = atom_id.as_array();

        for (center_i, bonded_indices) in bonds.outer_iter().enumerate() {
            let mut n_added: usize = 0;
            let mut line: String = String::new();
            for bonded_i in bonded_indices.iter() {
                if *bonded_i == -1 {
                    // Reached padding values
                    break;
                }
                if n_added == 0 {
                    // Add new record
                    line.push_str(&format!(
                        "CONECT{:>5}",
                        encode_hybrid36(atom_id[center_i], 5)?
                    ));
                }
                line.push_str(&format!(
                    "{:>5}",
                    encode_hybrid36(atom_id[*bonded_i as usize], 5)?
                ));
                n_added += 1;
                if n_added == 4 {
                    // Only a maximum of 4 bond partners can be put
                    // into a single line
                    // If there are more, use an extra record
                    n_added = 0;
                    // Pad the line to 80 characters
                    self.lines.push(format!("{:<80}", line));
                    line = String::new();
                }
            }
            if n_added > 0 {
                self.lines.push(format!("{:<80}", line));
            }
        }
        Ok(())
    }
}

// Get an `AtomArray` annotation as a NumPy array with the given dtype
fn get_annotation_as_type<'py, T: numpy::Element>(
    atoms: &Bound<'py, PyAny>,
    annotation: &str,
    dtype: &str,
) -> PyResult<Bound<'py, PyArray1<T>>> {
    let kwargs = PyDict::new(atoms.py());
    kwargs.set_item("dtype", dtype.to_string())?;
    kwargs.set_item("copy", false)?;
    atoms
        .getattr(annotation)?
        .call_method("astype", (), Some(&kwargs))?
        .extract()
}

// Get an `AtomArray` annotation as a `Vec` of strings.
// The annotation must be of some Unicode dtype.
fn get_annotation_as_strings<'py>(
    atoms: &Bound<'py, PyAny>,
    annotation: &str,
) -> PyResult<Vec<String>> {
    let convert_unicode_to_uint32 = PyModule::import(atoms.py(), "biotite.structure.io.util")?
        .getattr("convert_unicode_to_uint32")?;

    let annot_array = atoms.getattr(annotation)?;
    let string_array: Bound<'py, PyArray2<u32>> = convert_unicode_to_uint32
        .call1((annot_array,))?
        .extract::<Bound<'py, PyArray2<u32>>>()?;
    string_array
        .readonly()
        .as_array()
        .rows()
        .into_iter()
        .map(|x| parse_string_from_array(&x))
        .collect()
}

/// Filter the given ``AtomArray`` according to the given *altloc* identifier.
fn filter_altloc<'py>(
    py: Python<'py>,
    atoms: Bound<'py, PyAny>,
    altloc: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let struc = PyModule::import(py, "biotite.structure")?;

    match altloc {
        "first" => {
            let altloc_id = atoms.getattr("altloc_id")?;
            let filter = struc
                .getattr("filter_first_altloc")?
                .call1((atoms.clone(), altloc_id))?;
            // Ellipsis in the first dimension in case of an ``AtomArrayStack``
            let index = (PyEllipsis::get(py), filter);
            Ok(atoms.call_method1("__getitem__", (index,))?)
        }
        "occupancy" => {
            let altloc_id = atoms.getattr("altloc_id")?;
            let occupancy = atoms.getattr("occupancy")?;
            let filter = struc.getattr("filter_highest_occupancy_altloc")?.call1((
                atoms.clone(),
                altloc_id,
                occupancy,
            ))?;
            let index = (PyEllipsis::get(py), filter);
            Ok(atoms.call_method1("__getitem__", (index,))?)
        }
        "all" => Ok(atoms),
        _ => Err(biotite::InvalidFileError::new_err(format!(
            "'{}' is not a valid 'altloc' option",
            altloc
        ))),
    }
}

/// Write the characters of a ``String`` into a UTF-32 formatted array.
/// This function is the reverse of [`parse_string_from_array()`].
#[inline(always)]
fn write_string_to_array(array: &mut ArrayViewMut1<u32>, string: &str) {
    array
        .iter_mut()
        .zip(string.chars())
        .for_each(|(element, char)| {
            *element = char as u32;
        });
}

/// Create a ``String`` from a UTF-32 formatted  array.
/// Such an array would be for example an element of a NumPy array with *Unicode* dtype.
/// This function is the reverse of [`write_string_to_array()`].
#[inline(always)]
fn parse_string_from_array(array: &ArrayView1<u32>) -> PyResult<String> {
    let maybe_string: Result<String, _> = array
        .iter()
        .take_while(|&utf32_val| *utf32_val != 0)
        .map(|&utf32_val| TryInto::<u8>::try_into(utf32_val).map(|x| x as char))
        .collect();
    maybe_string.map_err(|_| {
        biotite::InvalidFileError::new_err("Unicode array contains invalid characters".to_string())
    })
}

/// Parse a string into a number.
/// Returns a ``PyErr`` if the parsing fails.
#[inline(always)]
fn parse_number<T: FromStr>(string: &str) -> PyResult<T> {
    string.trim().parse().map_err(|_| {
        biotite::InvalidFileError::new_err(format!(
            "'{}' cannot be parsed into a number",
            string.trim()
        ))
    })
}

/// Parse a byte into a digit.
/// Returns a ``PyErr`` if the parsing fails.
#[inline(always)]
fn parse_digit(digit: &u8) -> PyResult<i64> {
    (*digit as char).to_digit(10).map_or_else(
        || {
            Err(biotite::InvalidFileError::new_err(format!(
                "'{}' cannot be parsed into a number",
                digit
            )))
        },
        |v| Ok(v as i64),
    )
}

/// Parse a float from a string slice.
/// Parse a float from a string slice using a Range.
fn parse_float_from_string(line: &str, range: &Range<usize>) -> PyResult<f32> {
    line[range.clone()].trim().parse().map_err(|_| {
        biotite::InvalidFileError::new_err(format!(
            "'{}' cannot be parsed into a float",
            line[range.clone()].trim()
        ))
    })
}

/// If a given `id` exceeds `max_id` the returned ID restarts counting at 1.
/// Otherwise, `id` is returned.
/// This function is necessary, because there is a maximum number for atom and residue IDs in
/// PDB files.
#[inline(always)]
fn truncate_id(id: i64, max_id: i64) -> i64 {
    if id < 0 {
        return id;
    }
    ((id - 1) % max_id) + 1
}

/// Convert degrees to radians.
fn deg2rad<T: Float + FloatConst>(deg: T) -> T {
    deg * T::PI() / T::from(180.0).unwrap()
}

/// Convert radians to degrees.
fn rad2deg<T: Float + FloatConst>(rad: T) -> T {
    rad * T::from(180.0).unwrap() / T::PI()
}
