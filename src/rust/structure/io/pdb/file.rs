//! Low-level PDB file parsing and writing.

use numpy::ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions;
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::convert::TryInto;
use std::fs;
use std::str::FromStr;
use pyo3::types::{PyString, PyEllipsis, PyDict};
use numpy::PyArrayMethods;
use num_traits::{Float, FloatConst};
use pyo3::ffi::c_str;

pyo3::import_exception!(biotite, InvalidFileError);
// Label as a separate module to indicate that this exception comes
// from biotite
mod biotite {
    pyo3::import_exception!(biotite, InvalidFileError);
}

/// This is a low-level abstraction of a PDB file.
/// While the actual file input and output is managed in Python, this struct is able to parse
/// coordinates, models, bonds etc. from lines of text and vice versa.
#[pyclass]
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
    fn new(lines: Vec<String>) -> Self {
        //let ljust_lines = lines.iter().map(|line| format!("{:<80}", line)).collect();
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


    #[staticmethod]
    fn read(file: Bound<'_, PyAny>) -> PyResult<Self> {
        let contents;
        if let Ok(file_path) = file.downcast::<PyString>() {
            // Path-like object
            contents = fs::read_to_string(file_path.extract::<String>()?).map_err(|_| {
                exceptions::PyOSError::new_err(format!("'{}' cannot be read", file_path))
            })?;
        } else {
            // File-like object
            contents = file.call_method0("read")?.extract::<String>()?;
        }
        let lines = contents
            .lines()
            .map(|line| format!("{:<80}", line))
            .collect();
        let mut pdb_file = PDBFile {
            lines,
            model_start_i: Vec::new(),
            atom_line_i: Vec::new(),
        };
        pdb_file._index_models_and_atoms();
        Ok(pdb_file)
    }

    fn write(&self, file: Bound<'_, PyAny>) -> PyResult<()> {
        let content = self.lines.join("\n");
        if let Ok(file_path) = file.downcast::<PyString>() {
            // Path-like object
            fs::write(file_path.extract::<String>()?, content).map_err(|_| {
                exceptions::PyOSError::new_err(format!("'{}' cannot be written", file_path))
            })?;
        } else {
            // File-like object
            file.call_method1("write", (content,))?;
        }
        Ok(())
    }

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

    fn get_model_count(&self) -> usize {
        self.model_start_i.len()
    }

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
                    coord[[atom_i, 0]] = parse_float_from_string(line, 30, 38)?;
                    coord[[atom_i, 1]] = parse_float_from_string(line, 38, 46)?;
                    coord[[atom_i, 2]] = parse_float_from_string(line, 46, 54)?;
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
                    coord[[atom_i, 0]] = parse_float_from_string(line, 30, 38)?;
                    coord[[atom_i, 1]] = parse_float_from_string(line, 38, 46)?;
                    coord[[atom_i, 2]] = parse_float_from_string(line, 46, 54)?;
                }
                Ok(
                    coord
                        .into_shape_with_order((self.model_start_i.len(), length, 3))
                        .expect("Model length is invalid")
                        .into_pyarray(py)
                        .into_any()
                )
            }
        }
    }

    #[pyo3(signature = (model=None, altloc="first", extra_fields=None, include_bonds=false))]
    fn get_structure<'py>(
        &self, py: Python<'py>, model: Option<isize>, altloc: &str, extra_fields: Option<Vec<String>>, include_bonds: bool
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

        let include_atom_id;
        let include_b_factor;
        let mut include_occupancy;
        let include_charge;
        match extra_fields {
            Some(extra_fields) => {
                include_atom_id = extra_fields.contains(&"atom_id".to_string());
                include_b_factor = extra_fields.contains(&"b_factor".to_string());
                include_occupancy = extra_fields.contains(&"occupancy".to_string());
                include_charge = extra_fields.contains(&"charge".to_string());
            },
            None => {
                include_atom_id = false;
                include_b_factor = false;
                include_occupancy = false;
                include_charge = false;
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
            write_string_to_array(&mut chain_id.column_mut(atom_i), line[21..22].trim());
            res_id[atom_i] = parse_number(&line[22..26])?;
            write_string_to_array(&mut ins_code.column_mut(atom_i), line[26..27].trim());
            write_string_to_array(&mut res_name.column_mut(atom_i), line[17..20].trim());
            hetero[atom_i] = &line[0..4] != "ATOM";
            write_string_to_array(&mut atom_name.column_mut(atom_i), line[12..16].trim());
            write_string_to_array(&mut element.column_mut(atom_i), line[76..78].trim());
            write_string_to_array(&mut altloc_id.column_mut(atom_i), &line[16..17]);

            // Set optional annotation arrays
            if include_atom_id || include_bonds {
                atom_id[atom_i] = parse_number(&line[6..11])?;
            }
            if include_b_factor {
                b_factor[atom_i] = parse_number(&line[60..66])?;
            }
            if include_occupancy {
                occupancy[atom_i] = parse_number(&line[54..60])?;
            }
            if include_charge {
                let charge_raw = line[78..80].as_bytes();
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
                            &line[78..80]
                        )))?,
                    },
                }
            }
        }

        // Set annotations in the structure
        atoms.call_method1("set_annotation", ("chain_id", chain_id.into_pyarray(py)))?;
        atoms.call_method1("set_annotation", ("res_id", res_id.into_pyarray(py)))?;
        atoms.call_method1("set_annotation", ("ins_code", ins_code.into_pyarray(py)))?;
        atoms.call_method1("set_annotation", ("res_name", res_name.into_pyarray(py)))?;
        atoms.call_method1("set_annotation", ("hetero", hetero.into_pyarray(py)))?;
        atoms.call_method1("set_annotation", ("atom_name", atom_name.into_pyarray(py)))?;
        atoms.call_method1("set_annotation", ("element", element.into_pyarray(py)))?;
        atoms.call_method1("set_annotation", ("altloc_id", altloc_id.into_pyarray(py)))?;
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
            Ok(box_vectors) => atoms.setattr("box", box_vectors)?,
            Err(_) => {
                PyErr::warn(
                    py,
                    &py.get_type::<pyo3::exceptions::PyUserWarning>(),
                    c_str!("File contains invalid 'CRYST1' record, box is ignored"),
                    0
                )?;
            }
        };

        let atoms = filter_altloc(py, atoms, altloc)?;

        if include_bonds {
            let atom_id: Bound<'py, PyArray1<i64>> = atoms.getattr("atom_id")?.extract()?;
            let bond_array = self.parse_bonds(
                py, &atom_id.readonly().as_array()
            )?;
            let bond_list = struc.getattr("BondList")?.call1((length, bond_array))?;
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

    fn set_structure(
        &mut self,
        atoms: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let coord: ArrayView2<f32> = atoms
            .getattr("coord")?
            .extract::<Bound<'_, PyArray2<f32>>>()?
            .readonly()
            .as_array();
        let chain_id = get_annotation_as_type(atoms, "chain_id", "U")?.readonly().as_array();
        let res_id = get_annotation_as_type(atoms, "res_id", "int64")?.readonly().as_array();
        let ins_code = get_annotation_as_type(atoms, "ins_code", "U")?.readonly().as_array();
        let res_name = get_annotation_as_type(atoms, "res_name", "U")?.readonly().as_array();
        let hetero = get_annotation_as_type(atoms, "hetero", "bool")?.readonly().as_array();
        let atom_name = get_annotation_as_type(atoms, "atom_name", "U")?.readonly().as_array();
        let element = get_annotation_as_type(atoms, "element", "U")?.readonly().as_array();

        let atom_id = atom_id.as_ref().map(|arr| arr.as_array());
        let b_factor = b_factor.as_ref().map(|arr| arr.as_array());
        let occupancy = occupancy.as_ref().map(|arr| arr.as_array());
        let charge = charge.as_ref().map(|arr| arr.as_array());

        let is_multi_model = coord.shape()[0] > 1;

        // These will contain the ATOM records for each atom
        // These are reused in every model by adding the coordinates to the string
        // This procedure aims to increase the performance is repetitive formatting is omitted
        let mut prefix: Vec<String> = Vec::new();
        let mut suffix: Vec<String> = Vec::new();

        for i in 0..coord.shape()[1] {
            let element_i = parse_string_from_array(&element.column(i))?;
            let atom_name_i = parse_string_from_array(&atom_name.column(i))?;

            prefix.push(format!(
                "{:6}{:>5} {:4} {:>3} {:1}{:>4}{:1}   ",
                if hetero[i] { "HETATM" } else { "ATOM" },
                atom_id
                    .as_ref()
                    .map_or((i + 1) as i64, |arr| truncate_id(arr[i], 99999)),
                if element_i.len() == 1 && atom_name_i.len() < 4 {
                    format!(" {}", atom_name_i)
                } else {
                    atom_name_i
                },
                parse_string_from_array(&res_name.column(i))?,
                parse_string_from_array(&chain_id.column(i))?,
                truncate_id(res_id[i], 9999),
                parse_string_from_array(&ins_code.column(i))?,
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
    fn parse_box<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<u32>>>> {
        let struc = PyModule::import(py, "biotite.structure")?;

        for line in self.lines.iter() {
            if line.starts_with("CRYST1") {
                if line.len() < 80 {
                    return Err(biotite::InvalidFileError::new_err("Line is too short"));
                }
                let len_a: f64 = parse_number(&line[6..15])?;
                let len_b: f64 = parse_number(&line[15..24])?;
                let len_c: f64 = parse_number(&line[24..33])?;
                let alpha: f64 = parse_number(&line[33..40])?;
                let beta: f64 = parse_number(&line[40..47])?;
                let gamma: f64 = parse_number(&line[47..54])?;
                return Ok(Some(
                    struc.getattr("vectors_from_unitcell")?.call1(
                        (len_a, len_b, len_c, deg2rad(alpha), deg2rad(beta), deg2rad(gamma))
                    )?.extract()?
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
                // Extract ID of center atom
                let center_index = atom_id_to_index.get(&parse_number(&line[6..11])?);
                match center_index {
                    None => continue, // Atom ID is not in the AtomArray (probably removed altloc)
                    Some(center_index) => {
                        // Iterate over atom IDs bonded to center atom
                        for i in (11..31).step_by(5) {
                            match &parse_number(&line[i..i + 5]) {
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

    /// Write the `CRYST1` record to this [`PDBFile`] based on the given unit cell parameters.
    fn write_box(&mut self, len_a: f32, len_b: f32, len_c: f32, alpha: f32, beta: f32, gamma: f32) {
        self.lines.push(format!(
            "CRYST1{:>9.3}{:>9.3}{:>9.3}{:>7.2}{:>7.2}{:>7.2} P 1           1          ",
            len_a, len_b, len_c, alpha, beta, gamma
        ));
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
                    line.push_str(&format!("CONECT{:>5}", atom_id[center_i]));
                }
                line.push_str(&format!("{:>5}", atom_id[*bonded_i as usize]));
                n_added += 1;
                if n_added == 4 {
                    // Only a maximum of 4 bond partners can be put
                    // into a single line
                    // If there are more, use an extra record
                    n_added = 0;
                    self.lines.push(line);
                    line = String::new();
                }
            }
            if n_added > 0 {
                self.lines.push(line);
            }
        }
        Ok(())
    }
}


fn get_annotation_as_type<'py, T: numpy::Element>(
    atoms: Bound<'py, PyAny>,
    annotation: &str,
    dtype: &str,
) -> PyResult<Bound<'py, PyArray1<T>>> {
    let kwargs = PyDict::new(atoms.py());
    kwargs.set_item("dtype", dtype.to_string())?;
    kwargs.set_item("copy", false)?;
    Ok(atoms
        .getattr(annotation)?
        .call_method("astype", (), Some(&kwargs))?
        .extract()?
    )
}

/// Filter the given ``AtomArray`` according to the given *altloc* identifier.
fn filter_altloc<'py>(
    py: Python<'py>, atoms: Bound<'py, PyAny>, altloc: &str
) -> PyResult<Bound<'py, PyAny>> {
    let struc = PyModule::import(py, "biotite.structure")?;

    match altloc {
        "first" => {
            let altloc_id = atoms.getattr("altloc_id")?;
            let filter = struc.getattr("filter_first_altloc")?.call1((atoms.clone(), altloc_id))?;
            // Ellipsis in the first dimension in case of an ``AtomArrayStack``
            let index = (PyEllipsis::get(py), filter);
            Ok(atoms.call_method1("__getitem__", (index,))?)
        },
        "occupancy" => {
            let altloc_id = atoms.getattr("altloc_id")?;
            let occupancy = atoms.getattr("occupancy")?;
            let filter = struc.getattr("filter_highest_occupancy_altloc")?.call1(
                (atoms.clone(), altloc_id, occupancy)
            )?;
            let index = (PyEllipsis::get(py), filter);
            Ok(atoms.call_method1("__getitem__", (index,))?)
        },
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
    array.iter_mut().zip(string.chars()).for_each(|(element, char)| {
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
        biotite::InvalidFileError::new_err(format!(
            "Unicode array contains invalid characters"
        ))
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
fn parse_float_from_string(line: &str, start: usize, stop: usize) -> PyResult<f32> {
    line[start..stop].trim().parse().map_err(|_| {
        biotite::InvalidFileError::new_err(format!(
            "'{}' cannot be parsed into a float",
            line[start..stop].trim()
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