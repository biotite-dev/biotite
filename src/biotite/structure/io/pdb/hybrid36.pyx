# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["encode_hybrid36", "decode_hybrid36"]

cimport cython


cdef int _ASCII_FIRST_NUMBER = 48
cdef int _ASCII_FIRST_LETTER_UPPER = 65
cdef int _ASCII_FIRST_LETTER_LOWER = 97
cdef int _ASCII_LAST_NUMBER = 57
cdef int _ASCII_LAST_LETTER_UPPER = 90
cdef int _ASCII_LAST_LETTER_LOWER = 122


def encode_hybrid36(int number, int length):
    if number < 0:
        raise ValueError(
            "Only positive integers can be converted into hybrid-36 notation"
        )
    if length < 1:
        raise ValueError(
            "String length must be at least 1"
        )
    
    cdef int num = number
    if num < 10**length:
        return str(num)
    num -= 10**length
    if num < 26 * 36**(length-1):
        num += 10 * 36**(length-1)
        return _encode_base36(num, length, _ASCII_FIRST_LETTER_UPPER)
    num -= 26 * 36**(length-1)
    if num < 26 * 36**(length-1):
        num += 10 * 36**(length-1)
        return _encode_base36(num, length, _ASCII_FIRST_LETTER_LOWER)
    raise ValueError(
        f"Value {number} is too large for hybrid-36 encoding "
        f"at a string length of {length}"
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef str _encode_base36(int number, int length, int ascii_letter_offset):
    cdef unsigned char ascii_char
    cdef int remaining
    cdef int last
    cdef bytearray char_array = bytearray(length)
    cdef unsigned char[:] char_array_v = char_array
    # Set start index to end of character array and iterate in reverse
    cdef int i = char_array_v.shape[0] - 1
    while i >= 0:
        remaining = number // 36
        last = number - remaining * 36
        if last < 10:
            char_array_v[i] = last + _ASCII_FIRST_NUMBER
        else:
            char_array_v[i] = last + ascii_letter_offset - 10
        number = remaining
        i -= 1
    return char_array.decode("ascii")


def decode_hybrid36(str string):
    cdef int base_value
    cdef int length
    
    try:
        return int(string)
    except ValueError:
        pass
    
    # String is not parseable -> expect base36 string
    cdef bytearray char_array = bytearray(string.strip().encode("ascii"))
    cdef unsigned char[:] char_array_v = char_array
    length = char_array_v.shape[0]
    if length == 0:
        raise ValueError("Cannot parse empty string into integer")
    if      char_array_v[0] >= _ASCII_FIRST_LETTER_UPPER \
        and char_array_v[0] <= _ASCII_LAST_LETTER_UPPER:
            # String uses upper case letters
            base_value = _decode_base36(
                char_array_v, _ASCII_FIRST_LETTER_UPPER
            )
            return base_value - 10 * 36**(length-1) + 10**length
    elif    char_array_v[0] >= _ASCII_FIRST_LETTER_LOWER \
        and char_array_v[0] <= _ASCII_LAST_LETTER_LOWER:
            # String uses lower case letters
            base_value = _decode_base36(
                char_array_v, _ASCII_FIRST_LETTER_LOWER
            )
            return base_value + (26-10) * 36**(length-1) + 10**length
    else:
        raise ValueError(
            f"Illegal hybrid-36 string '{string.strip()}'"
        )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _decode_base36(unsigned char[:] char_array_v,
                        int ascii_letter_offset):
    cdef int i
    cdef int number = 0
    cdef unsigned char ascii_code
    for i in range(char_array_v.shape[0]):
        number *= 36
        ascii_code = char_array_v[i]
        if ascii_code <= _ASCII_LAST_NUMBER:
            number += ascii_code - _ASCII_FIRST_NUMBER
        else:
            number += ascii_code - ascii_letter_offset + 10
    return number