use pyo3::prelude::*;

pub mod banded;
pub mod cell;
pub mod kmeralphabet;
pub mod kmersimilarity;
pub mod kmertable;
pub mod localgapped;
pub mod localungapped;
pub mod nested;
pub mod pairwise;
pub mod scoring;
pub mod selector;
pub mod symbol;
pub mod table;
pub mod trace;

pub fn module<'py>(parent_module: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyModule>> {
    let module = PyModule::new(parent_module.py(), "align")?;
    module.add_class::<localungapped::SeedExtension>()?;
    module.add_class::<kmertable::KmerTable>()?;
    module.add_class::<kmertable::BucketKmerTable>()?;
    module.add_function(wrap_pyfunction!(pairwise::align_optimal, &module)?)?;
    module.add_function(wrap_pyfunction!(banded::align_banded, &module)?)?;
    module.add_function(wrap_pyfunction!(localgapped::align_region, &module)?)?;
    module.add_function(wrap_pyfunction!(kmeralphabet::create_kmers, &module)?)?;
    module.add_function(wrap_pyfunction!(kmeralphabet::split_kmers, &module)?)?;
    module.add_function(wrap_pyfunction!(kmersimilarity::similar_kmers, &module)?)?;
    module.add_function(wrap_pyfunction!(selector::minimize, &module)?)?;
    Ok(module)
}
