use pyo3::exceptions;
use pyo3::prelude::*;
use std::cmp::Ordering;

const ASCII_FIRST_NUMBER: u8 = 48;
const ASCII_FIRST_LETTER_UPPER: u8 = 65;
const ASCII_FIRST_LETTER_LOWER: u8 = 97;
const ASCII_LAST_NUMBER: u8 = 57;
const ASCII_LAST_LETTER_UPPER: u8 = 90;
const ASCII_LAST_LETTER_LOWER: u8 = 122;

/// Encode an integer value into a hybrid-36 string representation.
///
/// Parameters
/// ----------
/// number
///     A positive integer to be converted into a string.
/// length
///     The desired length of the string representation.
///     The resulting hybrid-36 string depends on the length the string
///     should have.
///
/// Returns
/// -------
/// hybrid36 : str
///     The hybrid-36 string representation.
#[pyfunction]
pub fn encode_hybrid36(mut number: i64, length: u32) -> PyResult<String> {
    if number < 0 {
        return Err(exceptions::PyValueError::new_err(
            "Only positive integers can be converted into hybrid-36 notation",
        ));
    }
    if length < 1 {
        return Err(exceptions::PyValueError::new_err(
            "String length must be at least 1",
        ));
    }

    if number < 10i64.pow(length) {
        // Normal decimal representation
        return Ok(number.to_string());
    }
    // Subtract the amount of values
    // represented by decimal representation
    number -= 10i64.pow(length);
    if number < 26 * 36i64.pow(length - 1) {
        // Upper case hybrid-36 representation
        // Transform value into base-36 value
        // Ex.: number = 10000, length = 4
        //      10**4 have been subtracted previously -> number = 0
        //      But first hybrid-36 string is 'A000'
        //      'A000' in base-36 is 10 * 36**3
        //      Hence 10 * 36**3 is added to the number
        //      to transform 10000 to 'A000'
        number += 10 * 36i64.pow(length - 1);
        return Ok(encode_base36(number, length, ASCII_FIRST_LETTER_UPPER));
    }
    // Subtract the amount of values represented by upper case hybrid-36 representation
    number -= 26 * 36i64.pow(length - 1);
    if number < 26 * 36i64.pow(length - 1) {
        // Lower case hybrid-36 representation
        number += 10 * 36i64.pow(length - 1);
        return Ok(encode_base36(number, length, ASCII_FIRST_LETTER_LOWER));
    }
    Err(exceptions::PyValueError::new_err(format!(
        "Value {} is too large for hybrid-36 encoding at a string length of {}",
        number, length
    )))
}

/// Convert a hybrid-36 string into a integer value.
///
/// Parameters
/// ----------
/// string : str
///     A hybrid-36 string representing a positive integer.
///
/// Returns
/// -------
/// number : int
///     The integer value represented by the hybrid-36 string.
#[pyfunction]
pub fn decode_hybrid36(hybrid36: &str) -> PyResult<i64> {
    let hybrid36 = hybrid36.trim();

    // Check if string is already base-10
    if let Ok(number) = hybrid36.parse::<i64>() {
        return Ok(number);
    }

    // String is not parsable -> expect base-36 string
    let ascii_array = hybrid36.as_bytes();
    if ascii_array.is_empty() {
        return Err(exceptions::PyValueError::new_err(
            "Cannot parse empty string into integer",
        ));
    }
    let length = ascii_array.len() as u32;

    match ascii_array[0] {
        // String uses upper case letters
        ASCII_FIRST_LETTER_UPPER..=ASCII_LAST_LETTER_UPPER => {
            let base_value = decode_base36(hybrid36, ASCII_FIRST_LETTER_UPPER);
            // Transform e.g. base-36 'A000' into 10000 instead of 1
            // (For more information see 'encode_hybrid36()')
            Ok(base_value - 10 * 36i64.pow(length - 1) + 10i64.pow(length))
        }
        // String uses lower case letters
        ASCII_FIRST_LETTER_LOWER..=ASCII_LAST_LETTER_LOWER => {
            let base_value = decode_base36(hybrid36, ASCII_FIRST_LETTER_LOWER);
            Ok(base_value + (26 - 10) * 36i64.pow(length - 1) + 10i64.pow(length))
        }
        _ => Err(exceptions::PyValueError::new_err(format!(
            "Illegal hybrid-36 string '{}'",
            hybrid36
        ))),
    }
}

/// max_hybrid36_number(length)
/// --
///
/// Give the maximum integer value that can be represented by a
/// hybrid-36 string of the given length.
///
/// Parameters
/// ----------
/// length : int
///     The length of a hybrid-36 string.
///
/// Returns
/// -------
/// max_number : int
///     The maximum integer value that can be represented by a hybrid-36
///     string of the given `length`.
#[pyfunction]
pub fn max_hybrid36_number(length: u32) -> i64 {
    //     |-- Decimal -|     |--- lo + up base-36 ---|
    10i64.pow(length) - 1 + 2 * (26 * 36i64.pow(length - 1))
}

/// Encode an integer value into a base-36 string representation.
///
/// Parameters
/// ----------
/// number
///     A positive integer to be converted into a string.
/// length
///     The desired length of the string representation.
/// ascii_letter_offset
///     The ASCII value of the first letter.
///     This parameter determines whether upper or lower case letters
///     should be used.
///
/// Returns
/// -------
/// The base-36 string representation.
fn encode_base36(mut number: i64, length: u32, ascii_letter_offset: u8) -> String {
    let mut ascii_array = vec![0u8; length as usize];
    // Iterate in reverse through the symbol array,
    // i.e. start at least significant digit
    for symbol in ascii_array.iter_mut().rev() {
        // Remove the last base-36 'digit' from the value using integer division...
        let remaining = number / 36;
        // ...and obtain the removed base-36 'digit'
        let last = (number - remaining * 36) as u8;
        // Convert the last base-36 'digit' into ASCII value
        // of corresponding base-36 symbol
        *symbol = match last.cmp(&10) {
            // 'Digit' gets numeric symbol if representable as decimal
            Ordering::Less => last + ASCII_FIRST_NUMBER,
            // Else 'digit' is converted into a letter
            Ordering::Equal | Ordering::Greater => last + ascii_letter_offset - 10,
        };
        // The new number is the original number without last 'digit'
        number = remaining;
    }
    String::from_utf8(ascii_array).expect("Invalid UTF-8 sequence")
}

/// Convert a base-36 string into a integer value.
///
/// Parameters
/// ----------
/// hybrid36
///     A hybrid-36 string representing a positive integer.
/// ascii_letter_offset
///     The ASCII value of the first letter.
///     This parameter determines whether the string uses upper or
///     lower case letters.
///
/// Returns
/// -------
/// The integer value represented by the base-36 string.
fn decode_base36(hybrid36: &str, ascii_letter_offset: u8) -> i64 {
    let ascii_array = hybrid36.as_bytes();
    let mut number = 0i64;
    for symbol in ascii_array {
        // Multiply existing value by base -> Shifting number one 'digit' to the left
        number *= 36;
        // Get the numerical value of the 'digit' and add to number
        match symbol.cmp(&ASCII_LAST_NUMBER) {
            Ordering::Less | Ordering::Equal => {
                number += (*symbol - ASCII_FIRST_NUMBER) as i64;
            }
            Ordering::Greater => {
                number += (*symbol - ascii_letter_offset + 10) as i64;
            }
        }
    }
    number
}
