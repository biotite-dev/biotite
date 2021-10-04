use std::str::FromStr;
use std::collections::HashMap;
use ndarray::{Array, Ix1, Ix2, Ix3};
use pyo3::prelude::*;
use pyo3::exceptions;
use pyo3::create_exception;
use numpy::PyArray;


create_exception!(fastpdb, InvalidFileError, exceptions::PyException);
create_exception!(fastpdb, BadStructureError, exceptions::PyException);


#[derive(Debug)]
enum CoordArray {
    Single(Array<f32, Ix2>),
    Multi(Array<f32, Ix3>),
}


#[pyclass]
struct PDBFile {
    lines: Vec<String>
}


#[pymethods]
impl PDBFile {
    
    #[new]
    fn new(lines: Vec<String>) -> Self {
        PDBFile { lines }
    }


    fn get_model_count(&self) -> usize {
        return self.get_model_start_indices().len()
    }


    fn parse_box(&self) -> PyResult<Option<(f32, f32, f32, f32, f32, f32)>> {
        for line in self.lines.iter() {
            if line.starts_with("CRYST1") {
                if line.len() < 80 {
                    return Err(BadStructureError::new_err("Line is too short"))
                }
                let len_a = parse_number(&line[ 6..15])?;
                let len_b = parse_number(&line[15..24])?;
                let len_c = parse_number(&line[24..33])?;
                // Angles given in degrees, needs to be converted into rad later on
                let alpha = parse_number(&line[33..40])?;
                let beta  = parse_number(&line[40..47])?;
                let gamma = parse_number(&line[47..54])?;
                return Ok(Some((len_a, len_b, len_c, alpha, beta, gamma)));
            }
        }
        // File has no 'CRYST1' record
        Ok(None)
    }


    fn parse_coord_single_model(&self, model: isize) -> PyResult<Py<PyArray<f32, Ix2>>> {
        let array = self.parse_coord(Some(model))?;
        Python::with_gil(|py| {
            match array{
                CoordArray::Single(array) => Ok(PyArray::from_array(py, &array).to_owned()),
                CoordArray::Multi(_) => panic!("No multi-model coordinates should be returned"),
            }
        })
    }


    fn parse_coord_multi_model(&self) -> PyResult<Py<PyArray<f32, Ix3>>> {
        let array = self.parse_coord(None)?;
        Python::with_gil(|py| {
            match array{
                CoordArray::Single(_) => panic!("No single-model coordinates should be returned"),
                CoordArray::Multi(array) => Ok(PyArray::from_array(py, &array).to_owned()),
            }
        })
    }


    fn parse_annotations(&self,
                        model: isize,
                        include_atom_id: bool,
                        include_b_factor: bool,
                        include_occupancy: bool,
                        include_charge: bool) -> PyResult<(Py<PyArray<u32,  Ix2>>,
                                                            Py<PyArray<i32,  Ix1>>,
                                                            Py<PyArray<u32,  Ix2>>,
                                                            Py<PyArray<u32,  Ix2>>,
                                                            Py<PyArray<bool, Ix1>>,
                                                            Py<PyArray<u32,  Ix2>>,
                                                            Py<PyArray<u32,  Ix2>>,
                                                            Py<PyArray<u32,  Ix2>>,
                                                            Option<Py<PyArray<i32,  Ix1>>>,
                                                            Option<Py<PyArray<f32,  Ix1>>>,
                                                            Option<Py<PyArray<f32,  Ix1>>>,
                                                            Option<Py<PyArray<i32,  Ix1>>>)> {
        let model_start_i = self.get_model_start_indices();
        let (model_start, model_stop) = self.get_model_boundaries(model, &model_start_i)?;

        let atom_line_i: Vec<usize> = self.get_atom_indices(model_start, model_stop);
        
        let mut chain_id:  Array<u32,  Ix2> = Array::zeros((atom_line_i.len(), 4));
        let mut res_id:    Array<i32,  Ix1> = Array::zeros(atom_line_i.len());
        let mut ins_code:  Array<u32,  Ix2> = Array::zeros((atom_line_i.len(), 1));
        let mut res_name:  Array<u32,  Ix2> = Array::zeros((atom_line_i.len(), 3));
        let mut hetero:    Array<bool, Ix1> = Array::default(atom_line_i.len());
        let mut atom_name: Array<u32,  Ix2> = Array::zeros((atom_line_i.len(), 6));
        let mut element:   Array<u32,  Ix2> = Array::zeros((atom_line_i.len(), 2));
        let mut altloc_id: Array<u32,  Ix2> = Array::zeros((atom_line_i.len(), 1));
        
        let mut atom_id: Array<i32, Ix1>;
        if include_atom_id {
            atom_id = Array::zeros(atom_line_i.len());
        }
        else {
            // Array will not be used
            atom_id = Array::zeros(0);
        }
        let mut b_factor: Array<f32, Ix1>;
        if include_b_factor {
            b_factor = Array::zeros(atom_line_i.len());
        }
        else {
            b_factor = Array::zeros(0);
        }
        let mut occupancy: Array<f32, Ix1>;
        if include_occupancy {
            occupancy = Array::zeros(atom_line_i.len());
        }
        else {
            occupancy = Array::zeros(0);
        }
        let mut charge: Array<i32, Ix1>;
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
                return Err(BadStructureError::new_err("Line is too short"))
            }
            write_string_to_array(&mut chain_id,  atom_i, line[21..22].trim());
            res_id[atom_i] = parse_number(&line[22..26])?;
            write_string_to_array(&mut ins_code,  atom_i, line[26..27].trim());
            write_string_to_array(&mut res_name,  atom_i, line[17..20].trim());
            hetero[atom_i] = if &line[0..4] == "ATOM" { false } else { true };
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
                let mut charge_raw = line[78..80].chars();
                let raw_number = charge_raw.next().unwrap();
                let raw_sign = charge_raw.next().unwrap();
                let number;
                if raw_number == ' ' {
                    number = 0;
                }
                else {
                    number = raw_number.to_digit(10).ok_or(
                        BadStructureError::new_err(format!(
                            "'{}' cannot be parsed into a number", raw_number
                        ))
                    )?;
                }
                let sign = if raw_sign == '-' { -1 } else { 1 };
                charge[atom_i] = number as i32 * sign;
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


    fn parse_bonds(&self, atom_ids: Py<PyArray<i32, Ix1>>) -> PyResult<Py<PyArray<u32, Ix2>>> {
        // Mapping from atom ids to indices in an AtomArray
        let mut atom_id_to_index: HashMap<i32, u32> = HashMap::new();
        Python::with_gil(|py| {
            for (i, id) in atom_ids.as_ref(py).to_owned_array().iter().enumerate() {
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

        Python::with_gil(|py| {
            let bond_array: &PyArray<u32, Ix2> = PyArray::zeros(py, (bonds.len(), 2), false);
            unsafe {
                // There are only unsafe ways to get a mutable version
                // of a PyArray
                let mut array_view = bond_array.as_array_mut();
                for (i, (center_id, bonded_id)) in bonds.iter().enumerate() {
                    array_view[[i, 0]] = *center_id;
                    array_view[[i, 1]] = *bonded_id;
                }
                Ok(bond_array.to_owned())
            }
        })
    }
}


impl PDBFile {

    fn parse_coord(&self, model: Option<isize>) -> PyResult<CoordArray> {
        let model_start_i = self.get_model_start_indices();
        
        let (model_start, model_stop) = match model {
            Some(model_number) => self.get_model_boundaries(model_number, &model_start_i)?,
            None => (0, self.lines.len())
        };
    
        let atom_line_i: Vec<usize> = self.get_atom_indices(model_start, model_stop);
        
        let mut coord = Array::zeros((atom_line_i.len(), 3));
        for (atom_i, line_i) in atom_line_i.iter().enumerate() {
            let line = &self.lines[*line_i];
            if line.len() < 80 {
                return Err(BadStructureError::new_err("Line is too short"))
            }
            coord[[atom_i, 0]] = line[30..38].trim().parse().or_else(|_|
                Err(BadStructureError::new_err(format!(
                    "'{}' cannot be parsed into a float", line[30..38].trim()
                )))
            )?;
            coord[[atom_i, 1]] = line[38..46].trim().parse().or_else(|_|
                Err(BadStructureError::new_err(format!(
                    "'{}' cannot be parsed into a float", line[38..46].trim()
                )))
            )?;
            coord[[atom_i, 2]] = line[46..54].trim().parse().or_else(|_|
                Err(BadStructureError::new_err(format!(
                    "'{}' cannot be parsed into a float", line[46..54].trim()
                )))
            )?;
        }
        
        match model {
            Some(_) => Ok(CoordArray::Single(coord)),
            None => {
                // Check whether all models have the same length
                let length = self.get_model_length(&model_start_i, &atom_line_i)?;
                Ok(
                    CoordArray::Multi(coord.into_shape((model_start_i.len(), length, 3))
                    .expect("Model length is invalid"))
                )
            }
        }
    }


    fn get_atom_indices(&self, model_start: usize, model_stop: usize) -> Vec<usize> {
        self.lines[model_start..model_stop].iter().enumerate()
            .filter(|(_i, line)| line.starts_with("ATOM") || line.starts_with("HETATM"))
            .map(|(i, _line)| i + model_start)
            .collect()
    }

    
    fn get_model_start_indices(&self) -> Vec<usize> {
        let mut model_start_i: Vec<usize> = self.lines.iter().enumerate()
            .filter(|(_i, line)| line.starts_with("MODEL"))
            .map(|(i, _line)| i)
            .collect();
        // Structures containing only one model may omit MODEL record
        // In these cases model starting index is set to 0
        if model_start_i.len() == 0 {
            model_start_i = vec![0]
        }
        model_start_i
    }
    
    fn get_model_boundaries(&self,
                            model: isize,
                            model_start_i: &Vec<usize>) -> PyResult<(usize, usize)> {
        let model_i: isize;
        if model > 0 {
            model_i = model - 1;
        }
        else if model < 0 {
            model_i = model_start_i.len() as isize + model;
        }
        else {
            return Err(exceptions::PyValueError::new_err("Model index must not be 0"));
        }
    
        if model_i >= model_start_i.len() as isize || model_i < 0 {
            return Err(exceptions::PyValueError::new_err(format!(
                "The file has {} models, the given model {} does not exist",
                model_start_i.len(), model
            )));
        }
    
        if model_i == model_start_i.len() as isize - 1 {
            // Last model -> Model reaches to end of file
            Ok((model_start_i[model_i as usize], self.lines.len()))
        }
        else {
            Ok((model_start_i[model_i as usize], model_start_i[(model_i+1) as usize]))
        }
    }
    
    fn get_model_length(&self,
                        model_start_i: &Vec<usize>,
                        atom_line_i: &Vec<usize>) -> PyResult<usize> {
        let n_models = model_start_i.len();
        let mut length: Option<usize> = None;
        for model_i in 0..n_models {
            let model_start: usize = model_start_i[model_i];
            let model_stop: usize = if model_i+1 < n_models { model_start_i[model_i+1] }
                                    else { self.lines.len() };
            let model_length = atom_line_i.iter()
                .filter(|&line_i| *line_i >= model_start && *line_i < model_stop)
                .count();
            match length {
                None => length = Some(model_length),
                Some(l) => if model_length != l { return Err(BadStructureError::new_err(
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

#[inline(always)]
fn write_string_to_array(array: &mut Array<u32, Ix2>, index: usize, string: &str) {
    for (i, char) in string.chars().enumerate() {
        array[[index, i]] = char as u32
    }
}


#[inline(always)]
fn parse_number<T: FromStr>(string: &str) -> PyResult<T> {
    string.trim().parse().or_else(|_|
        Err(BadStructureError::new_err(format!(
            "'{}' cannot be parsed into a number", string.trim()
        )))
    )
}


#[pymodule]
fn fastpdb(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PDBFile>()?;
    Ok(())
}