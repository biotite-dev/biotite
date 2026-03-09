use crate::dispatch_dtype;
use numpy::{IntoPyArray, PyArray1, PyArrayDescr, PyReadonlyArray1};
use numpy::{PyArrayDescrMethods, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions;
use pyo3::prelude::*;
use std::convert::{TryFrom, TryInto};

/// Convenience wrapper around ``dispatch_dtype!``` for the six BinaryCIF integer types.
macro_rules! dispatch_integer_dtype {
    ($py:expr, $dtype:expr, $func:ident $args:tt) => {
        dispatch_dtype!($py, $dtype, [i8, i16, i32, u8, u16, u32], $func $args)
    };
}

/// Encode an integer array using run-length encoding.
#[pyfunction]
pub fn run_length_encode<'py>(
    py: Python<'py>,
    data: &Bound<'py, numpy::PyUntypedArray>,
) -> PyResult<Bound<'py, pyo3::PyAny>> {
    let dt = data.dtype();
    dispatch_integer_dtype!(py, &dt, run_length_encode_inner(py, data))
}

fn run_length_encode_inner<'py, T: numpy::Element + Copy + TryInto<i32>>(
    py: Python<'py>,
    data: &Bound<'py, numpy::PyUntypedArray>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let typed = data.cast::<PyArray1<T>>()?;
    let data = unsafe { typed.as_slice()? };

    if data.is_empty() {
        return Ok(Vec::<i32>::new().into_pyarray(py));
    }

    let mut output: Vec<i32> = Vec::with_capacity(data.len() * 2);
    let mut val = data[0];
    let mut run_length: i32 = 1;
    for &curr_val in &data[1..] {
        if to_i32(curr_val)? == to_i32(val)? {
            run_length += 1;
        } else {
            output.push(to_i32(val)?);
            output.push(run_length);
            val = curr_val;
            run_length = 1;
        }
    }
    output.push(to_i32(val)?);
    output.push(run_length);
    Ok(output.into_pyarray(py))
}

/// Decode a run-length encoded integer array into the given output dtype.
#[pyfunction]
pub fn run_length_decode<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, i32>,
    src_size: Option<usize>,
    out_dtype: &Bound<'py, PyArrayDescr>,
) -> PyResult<Bound<'py, pyo3::PyAny>> {
    let data = data.as_slice()?;
    if data.len() % 2 != 0 {
        return Err(exceptions::PyValueError::new_err(
            "Invalid run-length encoded data",
        ));
    }

    let length = match src_size {
        Some(s) => s,
        None => (1..data.len()).step_by(2).map(|i| data[i] as usize).sum(),
    };

    dispatch_integer_dtype!(py, out_dtype, run_length_decode_inner(py, data, length))
}

fn run_length_decode_inner<'py, T: numpy::Element + Copy + Default + TryFrom<i32>>(
    py: Python<'py>,
    data: &[i32],
    length: usize,
) -> PyResult<Bound<'py, PyArray1<T>>> {
    let mut output: Vec<T> = vec![T::default(); length];
    let mut j: usize = 0;
    for i in (0..data.len()).step_by(2) {
        let value = from_i32::<T>(data[i])?;
        let repeat = data[i + 1] as usize;
        let end = j + repeat;
        if end > length {
            return Err(exceptions::PyValueError::new_err(
                "Run-length data exceeds expected output size",
            ));
        }
        for slot in &mut output[j..end] {
            *slot = value;
        }
        j = end;
    }
    Ok(output.into_pyarray(py))
}

/// Encode an integer array using integer packing into the given packed dtype.
#[pyfunction]
pub fn integer_packing_encode<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, i32>,
    byte_count: usize,
    is_unsigned: bool,
    out_dtype: &Bound<'py, PyArrayDescr>,
) -> PyResult<Bound<'py, pyo3::PyAny>> {
    let data = data.as_slice()?;
    let (min_val, max_val) = get_bounds(byte_count, is_unsigned)?;

    dispatch_integer_dtype!(
        py,
        out_dtype,
        integer_packing_encode_inner(py, data, min_val as i32, max_val as i32)
    )
}

fn integer_packing_encode_inner<'py, T: numpy::Element + Default + Clone + TryFrom<i32>>(
    py: Python<'py>,
    data: &[i32],
    min_val: i32,
    max_val: i32,
) -> PyResult<Bound<'py, PyArray1<T>>> {
    // Compute output length
    let mut length: usize = 0;
    for &number in data {
        match number.cmp(&0) {
            std::cmp::Ordering::Less => {
                if min_val == 0 {
                    return Err(exceptions::PyValueError::new_err(
                        "Cannot pack negative numbers into unsigned type",
                    ));
                }
                length += (number / min_val) as usize + 1;
            }
            std::cmp::Ordering::Greater => {
                length += (number / max_val) as usize + 1;
            }
            std::cmp::Ordering::Equal => {
                length += 1;
            }
        }
    }

    let mut output: Vec<T> = vec![T::default(); length];
    let mut i: usize = 0;
    for &number in data {
        let mut remainder = number;
        match remainder.cmp(&0) {
            std::cmp::Ordering::Less => {
                while remainder <= min_val {
                    remainder -= min_val;
                    output[i] = from_i32::<T>(min_val)?;
                    i += 1;
                }
            }
            std::cmp::Ordering::Greater => {
                while remainder >= max_val {
                    remainder -= max_val;
                    output[i] = from_i32::<T>(max_val)?;
                    i += 1;
                }
            }
            std::cmp::Ordering::Equal => {}
        }
        output[i] = from_i32::<T>(remainder)?;
        i += 1;
    }
    Ok(output.into_pyarray(py))
}

/// Decode an integer-packed array.
#[pyfunction]
pub fn integer_packing_decode<'py>(
    py: Python<'py>,
    data: &Bound<'py, numpy::PyUntypedArray>,
    byte_count: usize,
    is_unsigned: bool,
    src_size: usize,
) -> PyResult<Bound<'py, pyo3::PyAny>> {
    let dt = data.dtype();
    dispatch_integer_dtype!(
        py,
        &dt,
        integer_packing_decode_inner(py, data, byte_count, is_unsigned, src_size)
    )
}

fn integer_packing_decode_inner<'py, T: numpy::Element + Copy + TryInto<i32>>(
    py: Python<'py>,
    data: &Bound<'py, numpy::PyUntypedArray>,
    byte_count: usize,
    is_unsigned: bool,
    src_size: usize,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let typed = data.cast::<PyArray1<T>>()?;
    let data = unsafe { typed.as_slice()? };

    let (min_val, max_val) = get_bounds(byte_count, is_unsigned)?;
    // For unsigned integers, do not check lower bound
    // -> Set lower bound to value that is never reached
    let min_val = if min_val == 0 { -1 } else { min_val };

    let mut output: Vec<i32> = vec![0; src_size];
    let mut i: usize = 0;
    let mut unpacked_val: i32 = 0;
    for &packed in data {
        let packed_val = to_i32(packed)?;
        unpacked_val += packed_val;
        if packed_val as isize != max_val && packed_val as isize != min_val {
            // Unpacking of this value has ended -> Proceed to next value
            if i >= src_size {
                return Err(exceptions::PyValueError::new_err(
                    "Decoded data exceeds expected output size",
                ));
            }
            output[i] = unpacked_val;
            unpacked_val = 0;
            i += 1;
        }
    }
    Ok(output.into_pyarray(py))
}

/// Return the `(min, max)` bounds for a packed integer type given its byte count and signedness.
fn get_bounds(byte_count: usize, is_unsigned: bool) -> PyResult<(isize, isize)> {
    match (byte_count, is_unsigned) {
        (1, false) => Ok((i8::MIN as isize, i8::MAX as isize)),
        (1, true) => Ok((0, u8::MAX as isize)),
        (2, false) => Ok((i16::MIN as isize, i16::MAX as isize)),
        (2, true) => Ok((0, u16::MAX as isize)),
        _ => Err(exceptions::PyValueError::new_err("Unsupported byte count")),
    }
}

/// Convert an `i32` to `T`, returning a `PyErr` on overflow.
#[inline(always)]
fn from_i32<T: TryFrom<i32>>(val: i32) -> PyResult<T> {
    T::try_from(val).map_err(|_| {
        exceptions::PyOverflowError::new_err(format!("Value {} does not fit into target type", val))
    })
}

/// Convert a `T` to `i32`, returning a `PyErr` on overflow.
#[inline(always)]
fn to_i32<T: TryInto<i32>>(val: T) -> PyResult<i32> {
    val.try_into()
        .map_err(|_| exceptions::PyOverflowError::new_err("Value does not fit into i32"))
}
