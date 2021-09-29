use std::fs;
use ndarray::prelude::*;
use ndarray::{Array, Ix1, Ix2, Ix3};
//use pyo3::prelude::*;
//use numpy::PyArray1;


#[derive(Debug)]
enum CoordArray {
    Single(Array<f32, Ix2>),
    Multi(Array<f32, Ix3>),
}


struct PDBFile {
    pub lines: Vec<String>
}


impl PDBFile {

    pub fn read(file_path: &str) -> PDBFile {
        let lines = fs::read_to_string(file_path).unwrap().lines()
            .map(|element| {String::from(element)}).collect();
        PDBFile { lines }
    }

    pub fn get_coord(&self, model: Option<isize>) -> CoordArray {
        let mut model_start_i = self.get_model_start_indices();
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
                    panic!("Out of bound model");
                }

                if model_i == model_start_i.len() as isize - 1 {
                    // Last model -> Model reaches to end of file
                    (model_start_i[model_i as usize], self.lines.len())
                }
                else {
                    (model_start_i[model_i as usize], model_start_i[(model_i+1) as usize])
                }
            },
            None => (0, self.lines.len())
        };

        let atom_line_i: Vec<usize> = self.get_atom_indices(model_start, model_stop);
        
        let mut coord = Array::zeros((atom_line_i.len(), 3));
        for (atom_i, line_i) in atom_line_i.iter().enumerate() {
            coord[[atom_i, 0]] = self.lines[*line_i][30..38].trim().parse().unwrap();
            coord[[atom_i, 1]] = self.lines[*line_i][38..46].trim().parse().unwrap();
            coord[[atom_i, 2]] = self.lines[*line_i][46..54].trim().parse().unwrap();
        }
        
        match model {
            Some(_) => CoordArray::Single(coord),
            None => {
                let depth = coord.shape()[0] / model_start_i.len();
                CoordArray::Multi(coord.into_shape((depth, model_start_i.len(), 3)).unwrap())
            }
        }
    }


    fn get_atom_indices(&self, model_start: usize, model_stop: usize) -> Vec<usize> {
        self.lines[model_start..model_stop].iter().enumerate()
            .filter(|(_i, line)| line.starts_with("ATOM") || line.starts_with("HETATM"))
            .map(|(i, _line)| i)
            .collect()
    }


    fn get_model_start_indices(&self) -> Vec<usize> {
        self.lines.iter().enumerate()
            .filter(|(_i, line)| line.starts_with("MODEL"))
            .map(|(i, _line)| i)
            .collect()
    }
}


fn main() {
    let pdb_file = PDBFile::read("1aki.pdb");
    let coord = pdb_file.get_coord(None);
    println!("{:?}", &coord);
}
