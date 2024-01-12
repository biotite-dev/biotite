//! Low-level PDB file parsing and writing.

use std::fs;
use std::str::FromStr;
use std::convert::TryInto;
use std::collections::HashMap;
use std::cmp::Ordering;
use ndarray::{Array, Ix1, Ix2, Ix3};
use pyo3::prelude::*;
use pyo3::exceptions;
use numpy::PyArray;


pyo3::import_exception!(biotite, InvalidFileError);
// Label as a separate module to indicate that this exception comes
// from biotite
mod biotite {
    pyo3::import_exception!(biotite, InvalidFileError);
}


/// Used to allow a function to process both, `AtomArray` and `AtomArrayStack` objects,
/// since they differ only in the shape of the objects.
#[derive(Debug)]
enum CoordArray {
    Single(Array<f32, Ix2>),
    Multi(Array<f32, Ix3>)
}


/// This is a low-level abstraction of a PDB file.
/// While the actual file input and output is managed in Python, this struct is able to parse
/// coordinates, models, bonds etc. from lines of text and vice versa.
#[pyclass]
struct PDBFile {
    /// Lines of text from the PDB file.
    #[pyo3(get)]
    lines: Vec<String>,
    model_start_i: Vec<usize>,
    atom_line_i: Vec<usize>
}


#[pymethods]
impl PDBFile {

    /// Create an new [`PDBFile`].
    /// The lines of text are given to `lines`.
    /// An empty `Vec` represents and empty PDB file.
    #[new]
    fn new(lines: Vec<String>) -> Self {
        //let ljust_lines = lines.iter().map(|line| format!("{:<80}", line)).collect();
        let mut pdb_file = PDBFile {
            lines: lines,
            model_start_i: Vec::new(),
            atom_line_i: Vec::new()
        };
        pdb_file.index_models_and_atoms();
        pdb_file
    }


    /// Read a [`PDBFile`] from a file.
    /// The file is indicated by its file path as `String`.
    #[staticmethod]
    fn read(file_path: &str) -> PyResult<Self> {
        let contents = fs::read_to_string(file_path).map_err(
            |_| exceptions::PyOSError::new_err(format!("'{}' cannot be read", file_path))
        )?;
        let lines = contents.lines().map(
            |line| format!("{:<80}", line)
        ).collect();
        let mut pdb_file = PDBFile {
            lines: lines,
            model_start_i: Vec::new(),
            atom_line_i: Vec::new()
        };
        pdb_file.index_models_and_atoms();
        Ok(pdb_file)
    }


    #[getter]
    fn model_start_i(&self) -> PyResult<Py<PyArray<i64, Ix1>>> {
        Python::with_gil(|py| {
            Ok(PyArray::from_iter(
                py, self.model_start_i.iter().map(|x| *x as i64)
            ).to_owned())
        })
    }


    #[getter]
    fn atom_line_i(&self) -> PyResult<Py<PyArray<i64, Ix1>>> {
        Python::with_gil(|py| {
            Ok(PyArray::from_iter(
                py, self.atom_line_i.iter().map(|x| *x as i64)
            ).to_owned())
        })
    }


    /// Get the number of models contained in the file.
    fn get_model_count(&self) -> usize {
        self.model_start_i.len()
    }


    /// Parse the given `REMARK` record of the PDB file to obtain its content as strings
    fn parse_remark(&self, number: i64) -> PyResult<Option<Vec<String>>> {
        const CONTENT_START_COLUMN: usize = 11;

        if number < 0 || number > 999 {
            return Err(exceptions::PyValueError::new_err("The number must be in range 0-999"));
        }
        let remark_string = format!("REMARK {:>3}", number);
        let mut remark_lines: Vec<String> = self.lines.iter()
                           .filter(|line| line.starts_with(&remark_string))
                           .map(|line| line[CONTENT_START_COLUMN..].to_owned())
                           .collect();
        match remark_lines.len() {
            0 => Ok(None),
            // Remove first empty line
            _ => { remark_lines.remove(0); Ok(Some(remark_lines)) }
        }
    }


    /// Parse the `CRYST1` record of the PDB file to obtain the unit cell lengths
    /// and angles (in degrees).
    fn parse_box(&self) -> PyResult<Option<(f32, f32, f32, f32, f32, f32)>> {
        for line in self.lines.iter() {
            if line.starts_with("CRYST1") {
                if line.len() < 80 {
                    return Err(biotite::InvalidFileError::new_err("Line is too short"))
                }
                let len_a = parse_number(&line[ 6..15])?;
                let len_b = parse_number(&line[15..24])?;
                let len_c = parse_number(&line[24..33])?;
                let alpha = parse_number(&line[33..40])?;
                let beta  = parse_number(&line[40..47])?;
                let gamma = parse_number(&line[47..54])?;
                return Ok(Some((len_a, len_b, len_c, alpha, beta, gamma)));
            }
        }
        // File has no 'CRYST1' record
        Ok(None)
    }


    /// Get a 2D coordinate *NumPy* array for a single model in the file.
    /// `model` is the 1-based number of the requested model.
    fn parse_coord_single_model(&self, model: isize) -> PyResult<Py<PyArray<f32, Ix2>>> {
        let array = self.parse_coord(Some(model))?;
        Python::with_gil(|py| {
            match array {
                CoordArray::Single(array) => Ok(PyArray::from_array(py, &array).to_owned()),
                CoordArray::Multi(_) => panic!("No multi-model coordinates should be returned"),
            }
        })
    }


    /// Get a 3D coordinate *NumPy* array for all models in the file.
    /// A `PyErr` is returned if the number of atoms per model differ from each other.
    fn parse_coord_multi_model(&self) -> PyResult<Py<PyArray<f32, Ix3>>> {
        let array = self.parse_coord(None)?;
        Python::with_gil(|py| {
            match array {
                CoordArray::Single(_) => panic!("No single-model coordinates should be returned"),
                CoordArray::Multi(array) => Ok(PyArray::from_array(py, &array).to_owned()),
            }
        })
    }


    /// Parse the annotation arrays from a given `model` (1-based) in the file.
    /// The optional annotation categories for an `AtomArray` or `AtomArrayStack` are only parsed
    /// if the respective `include_<annotation>` parameter is set to true.
    /// Unicode *NumPy* arrays are represented by 2D *NumPy* arrays with
    /// `uint32` *dtype*.
    /// The returned tuple contains the following annotations in the given order:
    ///
    /// - `chain_id`
    /// - `res_id`
    /// - `ins_code`
    /// - `res_name`
    /// - `hetero`
    /// - `atom_name`
    /// - `element`
    /// - `altloc_id`
    /// - `atom_id`
    /// - `b_factor`
    /// - `occupancy`
    /// - `charge`
    fn parse_annotations(&self,
                        model: isize,
                        include_atom_id: bool,
                        include_b_factor: bool,
                        include_occupancy: bool,
                        include_charge: bool) -> PyResult<(Py<PyArray<u32,  Ix2>>,
                                                           Py<PyArray<i64,  Ix1>>,
                                                           Py<PyArray<u32,  Ix2>>,
                                                           Py<PyArray<u32,  Ix2>>,
                                                           Py<PyArray<bool, Ix1>>,
                                                           Py<PyArray<u32,  Ix2>>,
                                                           Py<PyArray<u32,  Ix2>>,
                                                           Py<PyArray<u32,  Ix2>>,
                                                           Option<Py<PyArray<i64,  Ix1>>>,
                                                           Option<Py<PyArray<f64,  Ix1>>>,
                                                           Option<Py<PyArray<f64,  Ix1>>>,
                                                           Option<Py<PyArray<i64,  Ix1>>>)> {
        let atom_line_i: Vec<usize> = self.get_atom_indices(model)?;

        let mut chain_id:  Array<u32,  Ix2> = Array::zeros((atom_line_i.len(), 4));
        let mut res_id:    Array<i64,  Ix1> = Array::zeros(atom_line_i.len());
        let mut ins_code:  Array<u32,  Ix2> = Array::zeros((atom_line_i.len(), 1));
        let mut res_name:  Array<u32,  Ix2> = Array::zeros((atom_line_i.len(), 5));
        let mut hetero:    Array<bool, Ix1> = Array::default(atom_line_i.len());
        let mut atom_name: Array<u32,  Ix2> = Array::zeros((atom_line_i.len(), 6));
        let mut element:   Array<u32,  Ix2> = Array::zeros((atom_line_i.len(), 2));
        let mut altloc_id: Array<u32,  Ix2> = Array::zeros((atom_line_i.len(), 1));

        let mut atom_id: Array<i64, Ix1>;
        if include_atom_id {
            atom_id = Array::zeros(atom_line_i.len());
        }
        else {
            // Array will not be used
            atom_id = Array::zeros(0);
        }
        let mut b_factor: Array<f64, Ix1>;
        if include_b_factor {
            b_factor = Array::zeros(atom_line_i.len());
        }
        else {
            b_factor = Array::zeros(0);
        }
        let mut occupancy: Array<f64, Ix1>;
        if include_occupancy {
            occupancy = Array::zeros(atom_line_i.len());
        }
        else {
            occupancy = Array::zeros(0);
        }
        let mut charge: Array<i64, Ix1>;
        if include_charge {
            charge = Array::zeros(atom_line_i.len());
        }
        else {
            charge = Array::zeros(0);
        }

        // Iterate over ATOM and HETATM records to write annotation arrays
        for (atom_i, line_i) in atom_line_i.iter().enumerate() {
            let line = &self.lines[*line_i];
            if line.len() < 80 {
                return Err(biotite::InvalidFileError::new_err("Line is too short"))
            }
            write_string_to_array(&mut chain_id,  atom_i, line[21..22].trim());
            res_id[atom_i] = parse_number(&line[22..26])?;
            write_string_to_array(&mut ins_code,  atom_i, line[26..27].trim());
            write_string_to_array(&mut res_name,  atom_i, line[17..20].trim());
            hetero[atom_i] = !(&line[0..4] == "ATOM");
            write_string_to_array(&mut atom_name, atom_i, line[12..16].trim());
            write_string_to_array(&mut element,   atom_i, line[76..78].trim());
            write_string_to_array(&mut altloc_id, atom_i, &line[16..17]);

            // Set optional annotation arrays
            if include_atom_id {
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
                            "'{}' is not a valid charge identifier", &line[78..80]
                        )))?,
                    }
                }
            }
        }

        Python::with_gil(|py| {
            Ok((
                PyArray::from_array(py, &chain_id ).to_owned(),
                PyArray::from_array(py, &res_id   ).to_owned(),
                PyArray::from_array(py, &ins_code ).to_owned(),
                PyArray::from_array(py, &res_name ).to_owned(),
                PyArray::from_array(py, &hetero   ).to_owned(),
                PyArray::from_array(py, &atom_name).to_owned(),
                PyArray::from_array(py, &element  ).to_owned(),
                PyArray::from_array(py, &altloc_id).to_owned(),
                if include_atom_id   { Some(PyArray::from_array(py, &atom_id).to_owned())   } else {None},
                if include_b_factor  { Some(PyArray::from_array(py, &b_factor).to_owned())  } else {None},
                if include_occupancy { Some(PyArray::from_array(py, &occupancy).to_owned()) } else {None},
                if include_charge    { Some(PyArray::from_array(py, &charge).to_owned())    } else {None},
            ))
        })
    }


    /// Parse the `CONECT` records of the PDB file to obtain a 2D *NumPy* array
    /// of indices.
    /// Each index points to the respective atom in the `AtomArray`.
    /// The `atom_id` annotation array is required to map the atom IDs in `CONECT` records
    /// to atom indices.
    fn parse_bonds(&self, atom_id: Py<PyArray<i64, Ix1>>) -> PyResult<Py<PyArray<u32, Ix2>>> {
        // Mapping from atom ids to indices in an AtomArray
        let mut atom_id_to_index: HashMap<i64, u32> = HashMap::new();
        Python::with_gil(|py| {
            for (i, id) in atom_id.as_ref(py).to_owned_array().iter().enumerate() {
                atom_id_to_index.insert(*id, i as u32);
            }
        });

        // Cannot preemptively determine number of bonds
        // -> Memory allocation for all bonds is not possible
        // -> No benefit in finding 'CONECT' record lines prior to iteration
        let mut bonds: Vec<(u32, u32)> = Vec::new();
        for line in self.lines.iter() {
            if line.starts_with("CONECT") {
                // Extract ID of center atom
                let center_index = *atom_id_to_index.get(&parse_number(&line[6..11])?).unwrap();
                // Iterate over atom IDs bonded to center atom
                for i in (11..31).step_by(5) {
                    match &parse_number(&line[i..i+5]) {
                        Ok(bonded_id) => bonds.push(
                            (center_index, *atom_id_to_index.get(bonded_id).unwrap())
                        ),
                        // String is empty -> no further IDs
                        Err(_) => break
                    };
                }
            }
        }

        let mut bond_array: Array<u32, Ix2> = Array::zeros((bonds.len(), 2));
        for (i, (center_id, bonded_id)) in bonds.iter().enumerate() {
            bond_array[[i, 0]] = *center_id;
            bond_array[[i, 1]] = *bonded_id;
        }
        Python::with_gil(|py| {
            Ok(PyArray::from_array(py, &bond_array).to_owned())
        })
    }




    /// Write the `CRYST1` record to this [`PDBFile`] based on the given unit cell parameters.
    fn write_box(&mut self,
                 len_a: f32, len_b: f32, len_c: f32,
                 alpha: f32, beta: f32, gamma: f32) {
        self.lines.push(
            format!(
                "CRYST1{:>9.3}{:>9.3}{:>9.3}{:>7.2}{:>7.2}{:>7.2} P 1           1          ",
                len_a, len_b, len_c, alpha, beta, gamma
            )
        );
    }


    /// Write models to this [`PDBFile`] based on the given coordinates and annotation arrays.
    fn write_models(&mut self,
        coord:     Py<PyArray<f32,  Ix3>>,
        chain_id:  Py<PyArray<u32,  Ix2>>,
        res_id:    Py<PyArray<i64,  Ix1>>,
        ins_code:  Py<PyArray<u32,  Ix2>>,
        res_name:  Py<PyArray<u32,  Ix2>>,
        hetero:    Py<PyArray<bool, Ix1>>,
        atom_name: Py<PyArray<u32,  Ix2>>,
        element:   Py<PyArray<u32,  Ix2>>,
        atom_id:   Option<Py<PyArray<i64,  Ix1>>>,
        b_factor:  Option<Py<PyArray<f64,  Ix1>>>,
        occupancy: Option<Py<PyArray<f64,  Ix1>>>,
        charge:    Option<Py<PyArray<i64,  Ix1>>>) -> PyResult<()> {
        Python::with_gil(|py| {
            let coord     = coord.as_ref(py).to_owned_array();
            let chain_id  = chain_id.as_ref(py).to_owned_array();
            let res_id    = res_id.as_ref(py).to_owned_array();
            let ins_code  = ins_code.as_ref(py).to_owned_array();
            let res_name  = res_name.as_ref(py).to_owned_array();
            let hetero    = hetero.as_ref(py).to_owned_array();
            let atom_name = atom_name.as_ref(py).to_owned_array();
            let element   = element.as_ref(py).to_owned_array();
            let atom_id   =   atom_id.map(|arr| arr.as_ref(py).to_owned_array());
            let b_factor  =  b_factor.map(|arr| arr.as_ref(py).to_owned_array());
            let occupancy = occupancy.map(|arr| arr.as_ref(py).to_owned_array());
            let charge    =    charge.map(|arr| arr.as_ref(py).to_owned_array());

            let is_multi_model = coord.shape()[0] > 1;

            // These will contain the ATOM records for each atom
            // These are reused in every model by adding the coordinates to the string
            // This procedure aims to increase the performance is repetitive formatting is omitted
            let mut prefix: Vec<String> = Vec::new();
            let mut suffix: Vec<String> = Vec::new();

            for i in 0..coord.shape()[1] {
                let element_i = parse_string_from_array(&element, i)?;
                let atom_name_i = parse_string_from_array(&atom_name, i)?;

                prefix.push(format!(
                    "{:6}{:>5} {:4} {:>3} {:1}{:>4}{:1}   ",
                    if hetero[i] { "HETATM" } else { "ATOM" },
                    atom_id.as_ref().map_or((i+1) as i64, |arr| truncate_id(arr[i], 99999)),
                    if element_i.len() == 1 && atom_name_i.len() < 4 {
                        format!(" {}", atom_name_i)
                    } else {
                        atom_name_i
                    },
                    parse_string_from_array(&res_name, i)?,
                    parse_string_from_array(&chain_id, i)?,
                    truncate_id(res_id[i], 9999),
                    parse_string_from_array(&ins_code, i)?,
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
                            Ordering::Equal => String::from("  ")
                        }
                    }),
                ));
            }

            for model_i in 0..coord.shape()[0] {
                if is_multi_model {
                    self.lines.push(format!("MODEL {:>8}", model_i+1));
                }
                for atom_i in 0..coord.shape()[1] {
                    let coord_string = format!(
                        "{:>8.3}{:>8.3}{:>8.3}",
                        coord[[model_i, atom_i, 0]],
                        coord[[model_i, atom_i, 1]],
                        coord[[model_i, atom_i, 2]],
                    );
                    self.lines.push(prefix[atom_i].clone() + &coord_string + &suffix[atom_i]);
                }
                if is_multi_model {
                    self.lines.push(String::from("ENDMDL"));
                }
            }
            Ok(())
        })
    }


    /// Write `CONECT` records to this [`PDBFile`] based on the given `bonds`
    /// array containing indices pointing to bonded atoms in the `AtomArray`.
    /// The `atom_id` annotation array is required to map the atom IDs in `CONECT` records
    /// to atom indices.
    fn write_bonds(&mut self,
                   bonds: Py<PyArray<i32, Ix2>>,
                   atom_id: Py<PyArray<i64, Ix1>>) -> PyResult<()> {
        Python::with_gil(|py| {
            let bonds = bonds.as_ref(py).to_owned_array();
            let atom_id = atom_id.as_ref(py).to_owned_array();

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
        })
    }

    /// Index lines in the file that correspond to starts of new models and to
    /// `ATOM` or `HETATM` records.
    /// Must be called after the content of the file has been changed.
    fn index_models_and_atoms(&mut self) {
        self.atom_line_i = self.lines.iter().enumerate()
            .filter(|(_i, line)| line.starts_with("ATOM") || line.starts_with("HETATM"))
            .map(|(i, _line)| i)
            .collect();
        self.model_start_i= self.lines.iter().enumerate()
            .filter(|(_i, line)| line.starts_with("MODEL"))
            .map(|(i, _line)| i)
            .collect();
        // It could be an empty file or a file with a single model,
        // where the 'MODEL' line is missing
        if self.model_start_i.is_empty() && !self.atom_line_i.is_empty(){
            self.model_start_i = vec![0]
        }
    }
}


impl PDBFile {

    /// Get a *NumPy* array containing coordinates for a single model (2D) or
    /// multiple models (3D) in the file.
    /// The number of returned dimensions is based on whether a `model` is
    /// given.
    fn parse_coord(&self, model: Option<isize>) -> PyResult<CoordArray> {
        match model {
            Some(model_number) => {
                let atom_line_i = self.get_atom_indices(model_number)?;
                let mut coord = Array::zeros((atom_line_i.len(), 3));
                for (atom_i, line_i) in atom_line_i.iter().enumerate() {
                    let line = &self.lines[*line_i];
                    if line.len() < 80 {
                        return Err(biotite::InvalidFileError::new_err("Line is too short"))
                    }
                    coord[[atom_i, 0]] = parse_float_from_string(line, 30, 38)?;
                    coord[[atom_i, 1]] = parse_float_from_string(line, 38, 46)?;
                    coord[[atom_i, 2]] = parse_float_from_string(line, 46, 54)?;
                }
                Ok(CoordArray::Single(coord))
            },

            None => {
                let length = self.get_model_length()?;
                let mut coord = Array::zeros((self.atom_line_i.len(), 3));
                for (atom_i, line_i) in self.atom_line_i.iter().enumerate() {
                    let line = &self.lines[*line_i];
                    if line.len() < 80 {
                        return Err(biotite::InvalidFileError::new_err("Line is too short"))
                    }
                    coord[[atom_i, 0]] = parse_float_from_string(line, 30, 38)?;
                    coord[[atom_i, 1]] = parse_float_from_string(line, 38, 46)?;
                    coord[[atom_i, 2]] = parse_float_from_string(line, 46, 54)?;
                }
                Ok(
                    CoordArray::Multi(coord.into_shape((self.model_start_i.len(), length, 3))
                    .expect("Model length is invalid"))
                )
            }
        }
    }


    /// Get indices to `ATOM` and `HETATM` records within the given model number.
    fn get_atom_indices(&self, model: isize) -> PyResult<Vec<usize>> {
        // Find the model index corresponding to the given model number
        let model_i: isize;
        match model.cmp(&0) {
            Ordering::Greater => model_i = model - 1,
            Ordering::Less => model_i = self.model_start_i.len() as isize + model,
            Ordering::Equal => return Err(exceptions::PyValueError::new_err(
                "Model index must not be 0"
            )),
        };
        if model_i >= self.model_start_i.len() as isize || model_i < 0 {
            return Err(exceptions::PyValueError::new_err(format!(
                "The file has {} models, the given model {} does not exist",
                self.model_start_i.len(), model
            )));
        }

        // Get the start and stop line index for this model index
        let (model_start, model_stop) = match model_i.cmp(&(self.model_start_i.len() as isize - 1)){
            Ordering::Less => (
                self.model_start_i[model_i as usize],
                self.model_start_i[(model_i+1) as usize]
            ),
            // Last model -> Model reaches to end of file
            Ordering::Equal => (
                self.model_start_i[model_i as usize],
                self.lines.len()
            ),
            // This case was excluded above
            _ => panic!("This branch should not be reached")
        };

        // Get the atom records within these line boundaries
        Ok(
            self.atom_line_i.iter().copied()
                .filter(|i| *i >= model_start && *i < model_stop)
                .collect()
        )
    }


    /// Get the number of atoms in each model of the PDB file.
    /// A `PyErr` is returned if the number of atoms per model differ from each other.
    fn get_model_length(&self) -> PyResult<usize> {
        let n_models = self.model_start_i.len();
        let mut length: Option<usize> = None;
        for model_i in 0..n_models {
            let model_start: usize = self.model_start_i[model_i];
            let model_stop: usize = if model_i+1 < n_models { self.model_start_i[model_i+1] }
                                    else { self.lines.len() };
            let model_length = self.atom_line_i.iter()
                .filter(|&line_i| *line_i >= model_start && *line_i < model_stop)
                .count();
            match length {
                None => length = Some(model_length),
                Some(l) => if model_length != l { return Err(biotite::InvalidFileError::new_err(
                    "Inconsistent number of models"
                )); }
            };
        }

        match length {
            None => panic!("Length cannot be 'None'"),
            Some(l) => Ok(l)
        }
    }
}


/// Write the characters of a `String` into the 2nd dimension of a 2D *Numpy* array.
/// This function is the reverse of [`parse_string_from_array()`].
#[inline(always)]
fn write_string_to_array(array: &mut Array<u32, Ix2>, index: usize, string: &str) {
    for (i, char) in string.chars().enumerate() {
        array[[index, i]] = char as u32
    }
}


/// Create a `String` from a row in a 2D *Numpy* array.
/// Each `uint32` value in the array is interpreted a an UTF-32 character.
/// This function is the reverse of [`write_string_to_array()`].
#[inline(always)]
fn parse_string_from_array(array: &Array<u32, Ix2>, index: usize) -> PyResult<String> {
    let mut out_string: String = String::new();
    for i in 0..array.shape()[1] {
        let utf_32_val = array[[index, i]];
        if utf_32_val == 0 {
            // End of string
            break
        }
        let ascii_val: u8 = utf_32_val.try_into()?;
        out_string.push(ascii_val as char);
    }
    Ok(out_string)
}


/// Parse a string into a number.
/// Returns a ``PyErr`` if the parsing fails.
#[inline(always)]
fn parse_number<T: FromStr>(string: &str) -> PyResult<T> {
    string.trim().parse().map_err(|_|
        biotite::InvalidFileError::new_err(format!(
            "'{}' cannot be parsed into a number", string.trim()
        ))
    )
}

/// Parse a byte into a digit.
/// Returns a ``PyErr`` if the parsing fails.
#[inline(always)]
fn parse_digit(digit: &u8) -> PyResult<i64> {
    (*digit as char).to_digit(10).map_or_else(
        || Err(biotite::InvalidFileError::new_err(format!(
            "'{}' cannot be parsed into a number", digit)
        )),
        |v| Ok(v as i64)
    )
}

//
fn parse_float_from_string(line: &str, start: usize, stop: usize) -> PyResult<f32>{
    line[start..stop].trim().parse().map_err(|_|
        biotite::InvalidFileError::new_err(format!(
            "'{}' cannot be parsed into a float", line[start..stop].trim()
        ))
    )
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




#[pymodule]
fn fastpdb(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PDBFile>()?;
    Ok(())
}
