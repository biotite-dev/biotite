use ndarray::prelude::*;
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
    let mut model_start_i = get_model_start_indices(lines);
    // Structures containing only one model may omit MODEL record
    // In these cases model starting index is set to 0
    if model_start_i.len() == 0 {
        model_start_i = vec![0]
    }
    
    let (model_start, model_stop) = match model {
        Some(model_number) => {
            let model_i: isize;
            if model_number > 0 {
                model_i = model_number - 1;
            }
            else if model_number < 0 {
                model_i = model_start_i.len() as isize + model_number;
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
        },
        None => (0, lines.len())
    };

    let atom_line_i: Vec<usize> = get_atom_indices(lines, model_start, model_stop);
    
    let mut coord = Array::zeros((atom_line_i.len(), 3));
    for (atom_i, line_i) in atom_line_i.iter().enumerate() {
        coord[[atom_i, 0]] = lines[*line_i][30..38].trim().parse().unwrap();
        coord[[atom_i, 1]] = lines[*line_i][38..46].trim().parse().unwrap();
        coord[[atom_i, 2]] = lines[*line_i][46..54].trim().parse().unwrap();
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


fn get_atom_indices(lines: &Vec<String>, model_start: usize, model_stop: usize) -> Vec<usize> {
    lines[model_start..model_stop].iter().enumerate()
        .filter(|(_i, line)| line.starts_with("ATOM") || line.starts_with("HETATM"))
        .map(|(i, _line)| i + model_start)
        .collect()
}

fn get_model_start_indices(lines: &Vec<String>) -> Vec<usize> {
    lines.iter().enumerate()
        .filter(|(_i, line)| line.starts_with("MODEL"))
        .map(|(i, _line)| i)
        .collect()
}

fn get_model_length(lines: &Vec<String>, model_start_i: &Vec<usize>, atom_line_i: &Vec<usize>) -> usize {
    let n_models = model_start_i.len();
    let mut length: isize = -1;
    for model_i in 0..n_models {
        let model_start: usize = model_start_i[model_i];
        let model_stop: usize = if model_i+1 < n_models { model_start_i[model_i+1] }
                                else { lines.len() };
        let model_length = atom_line_i.iter()
            .filter(|&line_i| *line_i >= model_start && *line_i < model_stop)
            .count();
        if length == -1 {
            length = model_length as isize;
        }
        else if model_length != length as usize {
            panic!("Inconsistent number of models");
        }
    }

    if length == -1 {
        panic!("File contains no model");
    }
    length as usize
}


#[pymodule]
fn fastpdb(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_coord_single_model, m)?)?;
    m.add_function(wrap_pyfunction!(parse_coord_multi_model, m)?)?;
    Ok(())
}




//fn read(file_path: &str) -> Vec<String> {
//    let lines = fs::read_to_string(file_path).unwrap().lines()
//        .map(|element| {String::from(element)}).collect();
//    lines
//}
//
//
//fn main() {
//    let lines = read("1aki.pdb");
//    let coord = parse_coord(&lines, None);
//    //println!("{:?}", &coord.shape());
//}