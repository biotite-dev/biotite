# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.pdb"
__author__ = "Patrick Kunzmann"
__all__ = ["encode_hybrid36", "decode_hybrid36", "max_hybrid36_number"]

cimport cython


cdef int _ASCII_FIRST_NUMBER = 48
cdef int _ASCII_FIRST_LETTER_UPPER = 65
cdef int _ASCII_FIRST_LETTER_LOWER = 97
cdef int _ASCII_LAST_NUMBER = 57
cdef int _ASCII_LAST_LETTER_UPPER = 90
cdef int _ASCII_LAST_LETTER_LOWER = 122


@cython.cpow(True)
def encode_hybrid36(int number, unsigned int length):
    """
    Encode an integer value into a hyrbid-36 string representation.
    
    Parameters
    ----------
    number : int
        A positive integer to be converted into a string.
    length : int
        The desired length of the string representation.
        The resulting hybrid-36 string depends on the length the string
        should have.
    
    Returns
    -------
    hybrid36 : str
        The hybrid-36 string representation.
    """
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
        # Normal decimal representation
        return str(num)
    # Subtract the amount of values
    # represented by decimal representation
    num -= 10**length
    if num < 26 * 36**(length-1):
        # Upper case hybrid-36 representation
        # Transform value into base-36 value
        # Ex.: number = 10000, length = 4
        #      10**4 have been suubtracted previously -> num = 0
        #      But first hybrid-36 string is 'A000'
        #      'A000' in base-36 is 10 * 36**3
        #      Hence 10 * 36**3 is added to the num
        #      to transform 10000 to 'A000'
        num += 10 * 36**(length-1)
        return _encode_base36(num, length, _ASCII_FIRST_LETTER_UPPER)
    # Subtract the amount of values
    # represented by upper case hybrid-36 representation
    num -= 26 * 36**(length-1)
    if num < 26 * 36**(length-1):
        # Lower case hybrid-36 representation
        num += 10 * 36**(length-1)
        return _encode_base36(num, length, _ASCII_FIRST_LETTER_LOWER)
    raise ValueError(
        f"Value {number} is too large for hybrid-36 encoding "
        f"at a string length of {length}"
    )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cpow(True)
cdef str _encode_base36(int number, unsigned int length,
                        unsigned int ascii_letter_offset):
    """
    Encode an integer value into a base-36 string representation.
    
    Parameters
    ----------
    number : int
        A positive integer to be converted into a string.
    length : int
        The desired length of the string representation.
    ascii_letter_offset : int
        The ASCII value of the first letter.
        This parameter determines whether upper or lower case letters
        should be used.
    
    Returns
    -------
    hybrid36 : str
        The base-36 string representation.
    """
    cdef unsigned char ascii_char
    cdef int remaining
    cdef int last
    cdef bytearray char_array = bytearray(length)
    cdef unsigned char[:] char_array_v = char_array
    # Set start index to end of character array and iterate in reverse
    cdef int i = char_array_v.shape[0] - 1
    while i >= 0:
        # Remove the last base-36 'digit' from the value...
        remaining = number // 36
        # ...and obtain the removed base-36 'digit'
        last = number - remaining * 36
        # Convert the last base-36 'digit' into ASCII value
        # of corresponding base-36 character
        if last < 10:
            # 'Digit' gets numeric character if representable as decimal
            char_array_v[i] = last + _ASCII_FIRST_NUMBER
        else:
            # Else 'digit' is converted into a letter
            char_array_v[i] = last + ascii_letter_offset - 10
        # The new number is the original number without last 'digit'
        number = remaining
        i -= 1
        # Repeat until all digits are removed
    return char_array.decode("ascii")


@cython.cpow(True)
def decode_hybrid36(str string):
    """
    Convert a hybrid-36 string into a integer value.
    
    Parameters
    ----------
    string : str
        A hybrid-36 string representing a positive integer.
    
    Returns
    -------
    number : int
        The integer value represented by the hybrid-36 string.
    """
    cdef int base_value
    cdef unsigned int length
    
    try:
        return int(string)
    except ValueError:
        pass
    
    # String is not parseable -> expect base36 string
    cdef bytes char_array = string.strip().encode("ascii")
    cdef const unsigned char[:] char_array_v = char_array
    length = char_array_v.shape[0]
    if length == 0:
        raise ValueError("Cannot parse empty string into integer")
    if      char_array_v[0] >= _ASCII_FIRST_LETTER_UPPER \
        and char_array_v[0] <= _ASCII_LAST_LETTER_UPPER:
            # String uses upper case letters
            base_value = _decode_base36(
                char_array_v, _ASCII_FIRST_LETTER_UPPER
            )
            # Transform e.g. base-36 'A000' into 10000
            # (For more information see 'encode_hybrid36()')
            return base_value - 10 * 36**(length-1) + 10**length
    elif    char_array_v[0] >= _ASCII_FIRST_LETTER_LOWER \
        and char_array_v[0] <= _ASCII_LAST_LETTER_LOWER:
            # String uses lower case letters
            base_value = _decode_base36(
                char_array_v, _ASCII_FIRST_LETTER_LOWER
            )
            # Transform and add the value range represented
            # by upper case hybrid-36:
            # 
            #   |----- e.g. A000 to 10000 -----| |---- upper hy36 ---|
            # - 10 * 36**(length-1) + 10**length + 26 * 36**(length-1)
            #
            # The following formula results from factoring out
            return base_value + (26-10) * 36**(length-1) + 10**length
    else:
        raise ValueError(
            f"Illegal hybrid-36 string '{string.strip()}'"
        )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _decode_base36(const unsigned char[:] char_array_v,
                        unsigned int ascii_letter_offset):
    """
    Convert a base-36 string into a integer value.
    
    Parameters
    ----------
    char_array_v : bytes
        A hybrid-36 string representing a positive integer.
    ascii_letter_offset : int
        The ASCII value of the first letter.
        This parameter determines whether teh string uses upper or
        lower case letters.
    
    Returns
    -------
    number : int
        The integer value represented by the base-36 string.
    """
    cdef int i
    cdef int number = 0
    cdef unsigned char ascii_code
    for i in range(char_array_v.shape[0]):
        # Multiply existing value by base
        # -> Shifting number one 'digit' to the left
        number *= 36
        # Get ASCII get of next base-36 'digit'
        ascii_code = char_array_v[i]
        # Get the numerical value of the 'digit' and add to number 
        if ascii_code <= _ASCII_LAST_NUMBER:
            number += ascii_code - _ASCII_FIRST_NUMBER
        else:
            number += ascii_code - ascii_letter_offset + 10
    return number

def max_hybrid36_number(length):
    """
    Give the maximum integer value that can be represented by a
    hybrid-36 string of the given length.
    
    Parameters
    ----------
    length : int
        The length of a hybrid-36 string.
    
    Returns
    -------
    max_number : int
        The maximum integer value that can be represented by a hybrid-36
        string of the given `length`.
    """
    #      |-- Decimal -|     |--- lo + up base-36 ---|
    return 10**length - 1  +  2 * (26 * 36**(length-1))