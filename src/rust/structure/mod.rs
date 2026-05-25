use crate::add_subpackage;
use pyo3::prelude::*;

pub mod bonds;
pub mod celllist;
pub mod charges;
pub mod connect;
pub mod io;
pub mod sasa;
pub mod util;

pub fn module<'py>(parent_module: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyModule>> {
    let module = PyModule::new(parent_module.py(), "structure")?;
    module.add_class::<bonds::BondList>()?;
    module.add_class::<celllist::CellList>()?;
    module.add_class::<celllist::CellListResult>()?;
    module.add_function(wrap_pyfunction!(bonds::bond_type_members, &module)?)?;
    module.add_function(wrap_pyfunction!(charges::partial_charges, &module)?)?;
    module.add_function(wrap_pyfunction!(
        connect::connect_via_residue_names,
        &module
    )?)?;
    module.add_function(wrap_pyfunction!(connect::connect_inter_residue, &module)?)?;
    module.add_function(wrap_pyfunction!(connect::connect_via_distances, &module)?)?;
    module.add_function(wrap_pyfunction!(connect::find_connected, &module)?)?;
    module.add_function(wrap_pyfunction!(sasa::sasa, &module)?)?;
    add_subpackage(&module, &io::module(&module)?, "biotite.rust.structure.io")?;
    Ok(module)
}
