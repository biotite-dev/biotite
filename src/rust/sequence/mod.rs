use crate::add_subpackage;
use pyo3::prelude::*;

pub mod align;
pub mod codec;
pub mod phylo;

pub fn module<'py>(parent_module: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyModule>> {
    let module = PyModule::new(parent_module.py(), "sequence")?;
    module.add_class::<codec::AlphabetCodec>()?;
    add_subpackage(
        &module,
        &align::module(&module)?,
        "biotite.rust.sequence.align",
    )?;
    add_subpackage(
        &module,
        &phylo::module(&module)?,
        "biotite.rust.sequence.phylo",
    )?;
    Ok(module)
}
