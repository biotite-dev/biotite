use ndarray::{Array, Ix1, Ix2, Ix3};
use pyo3::prelude::*;
use pyo3::exceptions;

use numpy::PyArray;


#[derive(Debug)]
enum CoordArray {
    Single(Array<f32, Ix2>),
    Multi(Array<f32, Ix3>),
}


#[pyfunction]
fn parse_coord_single_model(lines: Vec<String>, model: isize) -> PyResult<Py<PyArray<f32, Ix2>>> {
    let array = parse_coord(&lines, Some(model));
    Python::with_gil(|py| {
        match array{
            CoordArray::Single(array) => Ok(PyArray::from_array(py, &array).to_owned()),
            CoordArray::Multi(_) => Err(exceptions::PyTypeError::new_err(""))
        }
    })
}

#[pyfunction]
fn parse_coord_multi_model(lines: Vec<String>) -> PyResult<Py<PyArray<f32, Ix3>>> {
    let array = parse_coord(&lines, None);
    Python::with_gil(|py| {
        match array{
            CoordArray::Single(_) => Err(exceptions::PyTypeError::new_err("")),
            CoordArray::Multi(array) => Ok(PyArray::from_array(py, &array).to_owned()),
        }
    })
}


fn parse_coord(lines: &Vec<String>, model: Option<isize>) -> CoordArray {
    let model_start_i = get_model_start_indices(lines);
    
    let (model_start, model_stop) = match model {
        Some(model_number) => get_model_boundaries(lines, model_number, &model_start_i),
        None => (0, lines.len())
    };

    let atom_line_i: Vec<usize> = get_atom_indices(lines, model_start, model_stop);
    
    let mut coord = Array::zeros((atom_line_i.len(), 3));
    for (atom_i, line_i) in atom_line_i.iter().enumerate() {
        let line = &lines[*line_i];
        coord[[atom_i, 0]] = line[30..38].trim().parse().unwrap();
        coord[[atom_i, 1]] = line[38..46].trim().parse().unwrap();
        coord[[atom_i, 2]] = line[46..54].trim().parse().unwrap();
    }
    
    match model {
        Some(_) => CoordArray::Single(coord),
        None => {
            // Check whether all models have the same length
            let length = get_model_length(lines, &model_start_i, &atom_line_i);
            CoordArray::Multi(coord.into_shape((model_start_i.len(), length, 3)).unwrap())
        }
    }
}


#[pyfunction]
fn parse_annotations(lines: Vec<String>,
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
    let model_start_i = get_model_start_indices(&lines);
    let (model_start, model_stop) = get_model_boundaries(&lines, model, &model_start_i);

    let atom_line_i: Vec<usize> = get_atom_indices(&lines, model_start, model_stop);
    
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
        let line = &lines[*line_i];
        write_string_to_array(&mut chain_id,  atom_i, line[21..22].trim());
        res_id[atom_i] = line[22..26].trim().parse().unwrap();
        write_string_to_array(&mut ins_code,  atom_i, line[26..27].trim());
        write_string_to_array(&mut res_name,  atom_i, line[17..20].trim());
        hetero[atom_i] = if &line[0..4] == "ATOM" { false } else { true };
        write_string_to_array(&mut atom_name, atom_i, line[12..16].trim());
        write_string_to_array(&mut element,   atom_i, line[76..78].trim());
        write_string_to_array(&mut altloc_id, atom_i, line[16..17].trim());

        // Set optional annotation arrays
        if include_atom_id {
            atom_id[atom_i] = line[6..11].trim().parse().unwrap();
        }
        if include_b_factor {
            b_factor[atom_i] = line[60..66].trim().parse().unwrap();
        }
        if include_occupancy {
            occupancy[atom_i] = line[54..60].trim().parse().unwrap();
        }
        if include_charge {
            let mut charge_raw = line[78..80].chars();
            let raw_number = charge_raw.next().unwrap();
            let raw_sign = charge_raw.next().unwrap();
            let number = if raw_number == ' ' { 0 } else { raw_number.to_digit(10).unwrap() };
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


fn get_atom_indices(lines: &Vec<String>, model_start: usize, model_stop: usize) -> Vec<usize> {
    lines[model_start..model_stop].iter().enumerate()
        .filter(|(_i, line)| line.starts_with("ATOM") || line.starts_with("HETATM"))
        .map(|(i, _line)| i + model_start)
        .collect()
}

fn get_model_start_indices(lines: &Vec<String>) -> Vec<usize> {
    let mut model_start_i: Vec<usize> = lines.iter().enumerate()
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

fn get_model_boundaries(lines: &Vec<String>,
                        model: isize,
                        model_start_i: &Vec<usize>) -> (usize, usize) {
    let model_i: isize;
    if model > 0 {
        model_i = model - 1;
    }
    else if model < 0 {
        model_i = model_start_i.len() as isize + model;
    }
    else {
        panic!("The model index must not be 0");
    }

    if model_i >= model_start_i.len() as isize || model_i < 0 {
        panic!("Out of bounds model");
    }

    if model_i == model_start_i.len() as isize - 1 {
        // Last model -> Model reaches to end of file
        (model_start_i[model_i as usize], lines.len())
    }
    else {
        (model_start_i[model_i as usize], model_start_i[(model_i+1) as usize])
    }
}

fn get_model_length(lines: &Vec<String>, model_start_i: &Vec<usize>, atom_line_i: &Vec<usize>) -> usize {
    let n_models = model_start_i.len();
    let mut length: Option<usize> = None;
    for model_i in 0..n_models {
        let model_start: usize = model_start_i[model_i];
        let model_stop: usize = if model_i+1 < n_models { model_start_i[model_i+1] }
                                else { lines.len() };
        let model_length = atom_line_i.iter()
            .filter(|&line_i| *line_i >= model_start && *line_i < model_stop)
            .count();
        match length {
            None => length = Some(model_length),
            Some(l) => if model_length != l { panic!("Inconsistent number of models"); }
        };
    }

    match length {
        None => panic!("File contains no model"),
        Some(l) => l
    }
}

#[inline(always)]
fn write_string_to_array(array: &mut Array<u32, Ix2>, index: usize, string: &str) {
    for (i, char) in string.chars().enumerate() {
        array[[index, i]] = char as u32
    }
}



#[pymodule]
fn fastpdb(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_coord_single_model, m)?)?;
    m.add_function(wrap_pyfunction!(parse_coord_multi_model, m)?)?;
    m.add_function(wrap_pyfunction!(parse_annotations, m)?)?;
    Ok(())
}