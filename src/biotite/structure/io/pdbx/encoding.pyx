# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module contains data encodings for BinaryCIF files.
"""

__name__ = "biotite.structure.io.pdbx"
__author__ = "Patrick Kunzmann"
__all__ = ["ByteArrayEncoding", "FixedPointEncoding",
           "IntervalQuantizationEncoding", "RunLengthEncoding",
           "DeltaEncoding", "IntegerPackingEncoding", "StringArrayEncoding",
           "TypeCode"]

cimport cython
cimport numpy as np

from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
from numbers import Integral
from enum import IntEnum
import re
import numpy as np
from .component import _Component
from ....file import InvalidFileError

ctypedef np.int8_t int8
ctypedef np.int16_t int16
ctypedef np.int32_t int32
ctypedef np.uint8_t uint8
ctypedef np.uint16_t uint16
ctypedef np.uint32_t uint32
ctypedef np.float32_t float32
ctypedef np.float64_t float64

ctypedef fused Integer:
    uint8
    uint16
    uint32
    int8
    int16
    int32

# Used to create cartesian product of type combinations
# in run-length encoding
ctypedef fused OutputInteger:
    uint8
    uint16
    uint32
    int8
    int16
    int32

ctypedef fused Float:
    float32
    float64


CAMEL_CASE_PATTERN = re.compile(r"(?<!^)(?=[A-Z])")


class TypeCode(IntEnum):
    """
    This enum type represents integers that represent data types in
    *BinaryCIF*.
    """
    INT8 = 1
    INT16 = 2
    INT32 = 3
    UINT8 = 4
    UINT16 = 5
    UINT32 = 6
    FLOAT32 = 32
    FLOAT64 = 33

    @staticmethod
    def from_dtype(dtype):
        """
        Convert a *NumPy* dtype to a *BinaryCIF* type code.

        Parameters
        ----------
        dtype : dtype or int or TypeCode
            The data type to be converted.
            If already a type code, it is simply returned.

        Returns
        -------
        type_code : TypeCode
            The corresponding type code.
        """
        if isinstance(dtype, Integral):
            # Already a type code
            return TypeCode(dtype)
        else:
            dtype = np.dtype(dtype)
            # Find the closest dtype supported by the format
            if np.issubdtype(dtype, np.integer):
                # int64 is not supported by format
                if dtype == np.int64:
                    supported_dtype = np.int32
                elif dtype == np.uint64:
                    supported_dtype = np.uint32
                else:
                    supported_dtype = dtype
            elif np.issubdtype(dtype, np.floating):
                if dtype == np.float16:
                    supported_dtype = np.float32
                # float128 is not available on all architectures
                elif hasattr(np, "float128") and dtype == np.float128:
                    supported_dtype = np.float64
                else:
                    supported_dtype = dtype
            else:
                raise ValueError(
                    f"dtype '{dtype}' is not supported by BinaryCIF"
                )
            return _DTYPE_TO_TYPE_CODE[
                np.dtype(supported_dtype).newbyteorder("<").str
            ]

    def to_dtype(self):
        """
        Convert this type code to a *NumPy* dtype.

        Returns
        -------
        dtype : dtype
            The corresponding data type.
        """
        return _TYPE_CODE_TO_DTYPE[self]

# Converts BCIF integers representing the type to an actual NumPy dtype
_TYPE_CODE_TO_DTYPE = {
    # All data types are little-endian
    TypeCode.INT8: "|i1",
    TypeCode.INT16: "<i2",
    TypeCode.INT32: "<i4",
    TypeCode.UINT8: "|u1",
    TypeCode.UINT16: "<u2",
    TypeCode.UINT32: "<u4",
    TypeCode.FLOAT32: "<f4",
    TypeCode.FLOAT64: "<f8"
}
_DTYPE_TO_TYPE_CODE = {val: key for key, val in _TYPE_CODE_TO_DTYPE.items()}


class Encoding(_Component, metaclass=ABCMeta):
    """
    Abstract base class for *BinaryCIF* data encodings.

    Notes
    -----
    The encoding classes do not omit bound checks for decoding,
    since the file content may be invalid/malicious.
    """

    @classmethod
    def deserialize(cls, content):
        params = {
            _camel_to_snake_case(param): value
            for param, value in content.items()
        }
        # 'kind' is no parameter, but indicates the class itself
        params.pop("kind")
        try:
            encoding = cls(**params)
        except TypeError as e:
            raise InvalidFileError(
                f"Invalid encoding parameters for {cls.__name__}"
            )
        except ValueError:
            raise InvalidFileError(
                f"Missing encoding parameters for {cls.__name__}"
            )
        return encoding

    def serialize(self):
        for param in self.__annotations__:
            if getattr(self, param) is None:
                raise ValueError(
                    f"'{param}' must be explicitly given or needs to be "
                    "determined from first encoding pass, before it is "
                    "serialized"
                )

        serialized = {
            _snake_to_camel_case(param): getattr(self, param)
            for param in self.__annotations__
        }
        serialized.update({
            "kind": _encoding_classes_kinds[type(self).__name__]
        })
        return serialized

    @abstractmethod
    def encode(self, data):
        """
        Apply this encoding to the given data.

        Parameters
        ----------
        data : ndarray
            The data to be encoded.

        Returns
        -------
        encoded_data : ndarray or bytes
            The encoded data.
        """
        raise NotImplementedError()

    @abstractmethod
    def decode(self, data):
        """
        Apply the inverse of this encoding to the given data.

        Parameters
        ----------
        data : ndarray or bytes
            The data to be decoded.

        Returns
        -------
        decoded_data : ndarray
            The decoded data.
        """
        # Important: Do not omit bound checks for decoding,
        # since the file content may be invalid/malicious.
        raise NotImplementedError()

    def __str__(self):
        # Restore original behavior, as `__str__()` implementation of `_Component`
        # may require serialization, which is not possible for some encodings prior
        # to the first encoding pass
        return object.__str__(self)


@dataclass
class ByteArrayEncoding(Encoding):
    r"""
    Encoding that encodes an array into bytes.

    Parameters
    ----------
    type : dytpe or TypeCode, optional
        The data type of the array to be encoded.
        Either a NumPy dtype or a *BinaryCIF* type code is accepted.
        If omitted, the data type is taken from the data the
        first time :meth:`encode()` is called.

    Attributes
    ----------
    type : TypeCode

    Examples
    --------

    >>> data = np.arange(3)
    >>> print(data)
    [0 1 2]
    >>> print(ByteArrayEncoding().encode(data))
    b'\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00'
    """
    type: ... = None

    def __post_init__(self):
        if self.type is not None:
            self.type = TypeCode.from_dtype(self.type)

    def encode(self, data):
        if self.type is None:
            self.type = TypeCode.from_dtype(data.dtype)
        return _safe_cast(data, self.type.to_dtype()).tobytes()

    def decode(self, data):
        # Data is raw bytes in this case
        return np.frombuffer(data, dtype=self.type.to_dtype())


@dataclass
class FixedPointEncoding(Encoding):
    """
    Lossy encoding that multiplies floating point values with a given
    factor and subsequently rounds them to the nearest integer.

    Parameters
    ----------
    factor : float
        The factor by which the data is multiplied before rounding.
    src_type : dtype or TypeCode, optional
        The data type of the array to be encoded.
        Either a NumPy dtype or a *BinaryCIF* type code is accepted.
        The dtype must be a float type.
        If omitted, the data type is taken from the data the
        first time :meth:`encode()` is called.

    Attributes
    ----------
    factor : float
    src_type : TypeCode

    Examples
    --------

    >>> data = np.array([9.87, 6.543])
    >>> print(data)
    [9.870 6.543]
    >>> print(FixedPointEncoding(factor=100).encode(data))
    [987 654]
    """
    factor: ...
    src_type: ... = None

    def __post_init__(self):
        if self.src_type is not None:
            self.src_type = TypeCode.from_dtype(self.src_type)
            if self.src_type not in (TypeCode.FLOAT32, TypeCode.FLOAT64):
                raise ValueError(
                    "Only floating point types are supported"
                )

    def encode(self, data):
        # If not given in constructor, it is determined from the data
        if self.src_type is None:
            self.src_type = TypeCode.from_dtype(data.dtype)
            if self.src_type not in (TypeCode.FLOAT32, TypeCode.FLOAT64):
                raise ValueError(
                    "Only floating point types are supported"
                )

        # Round to avoid wrong values due to floating point inaccuracies
        scaled_data = np.round(data * self.factor)
        return _safe_cast(scaled_data, np.int32, allow_decimal_loss=True)

    def decode(self, data):
        return (data / self.factor).astype(
            dtype=self.src_type.to_dtype(), copy=False
        )


@dataclass
class IntervalQuantizationEncoding(Encoding):
    """
    Lossy encoding that sorts floating point values into bins.
    Each bin is represented by an integer

    Parameters
    ----------
    min, max : float
        The minimum and maximum value the bins comprise.
    num_steps : int
        The number of bins.
    src_type : dtype or TypeCode, optional
        The data type of the array to be encoded.
        Either a NumPy dtype or a *BinaryCIF* type code is accepted.
        The dtype must be a float type.
        If omitted, the data type is taken from the data the
        first time :meth:`encode()` is called.

    Attributes
    ----------
    min, max : float
    num_steps : int
    src_type : TypeCode

    Examples
    --------

    >>> data = np.linspace(11, 12, 6)
    >>> print(data)
    [11.0 11.2 11.4 11.6 11.8 12.0]
    >>> # Use 0.5 as step size
    >>> encoding = IntervalQuantizationEncoding(min=10, max=20, num_steps=21)
    >>> # The encoding is lossy, as different values are mapped to the same bin
    >>> encoded = encoding.encode(data)
    >>> print(encoded)
    [2 3 3 4 4 4]
    >>> decoded = encoding.decode(encoded)
    >>> print(decoded)
    [11.0 11.5 11.5 12.0 12.0 12.0]
    """
    min: ...
    max: ...
    num_steps: ...
    src_type: ... = None

    def __post_init__(self):
        if self.src_type is not None:
            self.src_type = TypeCode.from_dtype(self.src_type)

    def encode(self, data):
        # If not given in constructor, it is determined from the data
        if self.src_type is None:
            self.src_type = TypeCode.from_dtype(data.dtype)

        steps = np.linspace(
            self.min, self.max, self.num_steps, dtype=data.dtype
        )
        indices = np.searchsorted(steps, data, side="left")
        return _safe_cast(indices, np.int32)

    def decode(self, data):
        output = data * (self.max - self.min) / (self.num_steps - 1)
        output = output.astype(self.src_type.to_dtype(), copy=False)
        output += self.min
        return output


@dataclass
class RunLengthEncoding(Encoding):
    """
    Encoding that compresses runs of equal values into pairs of
    (value, run length).

    Parameters
    ----------
    src_size : int, optional
        The size of the array to be encoded.
        If omitted, the size is determined from the data the
        first time :meth:`encode()` is called.
    src_type : dtype or TypeCode, optional
        The data type of the array to be encoded.
        Either a NumPy dtype or a *BinaryCIF* type code is accepted.
        The dtype must be a integer type.
        If omitted, the data type is taken from the data the
        first time :meth:`encode()` is called.

    Attributes
    ----------
    src_size : int
    src_type : TypeCode

    Examples
    --------

    >>> data = np.array([1, 1, 1, 5, 3, 3])
    >>> print(data)
    [1 1 1 5 3 3]
    >>> encoded = RunLengthEncoding().encode(data)
    >>> print(encoded)
    [1 3 5 1 3 2]
    >>> # Emphasize the the pairs
    >>> print(encoded.reshape(-1, 2))
    [[1 3]
     [5 1]
     [3 2]]
    """
    src_size: ... = None
    src_type: ... = None

    def __post_init__(self):
        if self.src_type is not None:
            self.src_type = TypeCode.from_dtype(self.src_type)

    def encode(self, data):
        # If not given in constructor, it is determined from the data
        if self.src_type is None:
            self.src_type = TypeCode.from_dtype(data.dtype)
        if self.src_size is None:
            self.src_size = data.shape[0]
        elif self.src_size != data.shape[0]:
            raise IndexError(
                "Given source size does not match actual data size"
            )
        return self._encode(_safe_cast(data, self.src_type.to_dtype()))

    def decode(self, data):
        return self._decode(
            data, np.empty(0, dtype=self.src_type.to_dtype())
        )

    def _encode(self, const Integer[:] data):
        # Pessimistic allocation of output array
        # -> Run length is 1 for every element
        cdef int32[:] output = np.zeros(data.shape[0] * 2, dtype=np.int32)
        cdef int i=0, j=0
        cdef int val = data[0]
        cdef int run_length = 0
        cdef int curr_val
        for i in range(data.shape[0]):
            curr_val = data[i]
            if curr_val == val:
                run_length += 1
            else:
                # New element -> Write element with run-length
                output[j] = val
                output[j+1] = run_length
                j += 2
                val = curr_val
                run_length = 1
        # Write last element
        output[j] = val
        output[j+1] = run_length
        j += 2
        # Trim to correct size
        return np.asarray(output)[:j]

    def _decode(self, const Integer[:] data, OutputInteger[:] output_type):
        """
        `output_type` is merely a typed placeholder to allow for static
        typing of output.
        """
        if data.shape[0] % 2 != 0:
            raise ValueError("Invalid run-length encoded data")

        cdef int length = 0
        cdef int i, j
        cdef int value, repeat

        if self.src_size is None:
            # Determine length of output array by summing run lengths
            for i in range(1, data.shape[0], 2):
                length += data[i]
        else:
            length = self.src_size

        cdef OutputInteger[:] output = np.zeros(
            length, dtype=np.asarray(output_type).dtype
        )
        # Fill output array
        j = 0
        for i in range(0, data.shape[0], 2):
            value = data[i]
            repeat = data[i+1]
            output[j : j+repeat] = value
            j += repeat
        return np.asarray(output)


@dataclass
class DeltaEncoding(Encoding):
    """
    Encoding that encodes an array of integers into an array of
    consecutive differences.

    Parameters
    ----------
    src_type : dtype or TypeCode, optional
        The data type of the array to be encoded.
        Either a NumPy dtype or a *BinaryCIF* type code is accepted.
        The dtype must be a integer type.
        If omitted, the data type is taken from the data the
        first time :meth:`encode()` is called.
    origin : int, optional
        The starting value from which the differences are calculated.
        If omitted, the value is taken from the first array element the
        first time :meth:`encode()` is called.

    Attributes
    ----------
    src_type : TypeCode
    origin : int

    Examples
    --------

    >>> data = np.array([1, 1, 2, 3, 5, 8])
    >>> encoding = DeltaEncoding()
    >>> print(encoding.encode(data))
    [0 0 1 1 2 3]
    >>> print(encoding.origin)
    1
    """
    src_type: ... = None
    origin: ... = None

    def __post_init__(self):
        if self.src_type is not None:
            self.src_type = TypeCode.from_dtype(self.src_type)

    def encode(self, data):
        # If not given in constructor, it is determined from the data
        if self.src_type is None:
            self.src_type = TypeCode.from_dtype(data.dtype)
        if self.origin is None:
            self.origin = data[0]

        # Differences (including `np.diff`) return an array with the same dtype as the
        # input array
        # As the input dtype may be unsigned, the output dtype could underflow,
        # if the difference is negative
        # -> cast to int64 to avoid this
        data = data.astype(np.int64, copy=False)
        data = data - self.origin
        return _safe_cast(np.diff(data, prepend=0), np.int32)

    def decode(self, data):
        output = np.cumsum(data, dtype=self.src_type.to_dtype())
        output += self.origin
        return output


@dataclass
class IntegerPackingEncoding(Encoding):
    """
    Encoding that compresses an array of 32-bit integers into an array
    of smaller sized integers.

    If a value does not fit into smaller integer type,
    the integer is represented by a sum of consecutive elements
    in the compressed array.

    Parameters
    ----------
    byte_count : int
        The number of bytes the packed integers should occupy.
        Supported values are 1 and 2 for 8-bit and 16-bit integers,
        respectively.
    src_size : int, optional
        The size of the array to be encoded.
        If omitted, the size is determined from the data the
        first time :meth:`encode()` is called.
    is_unsigned : bool, optional
        Whether the values should be packed into signed or unsigned
        integers.
        If omitted, first time :meth:`encode()` is called, determines whether
        the values fit into unsigned integers.

    Attributes
    ----------
    byte_count : int
    src_size : int
    is_unsigned : bool

    Examples
    --------

    >>> data = np.array([1, 2, -3, 128])
    >>> print(data)
    [  1   2  -3 128]
    >>> print(IntegerPackingEncoding(byte_count=1).encode(data))
    [  1   2  -3 127   1]
    """
    byte_count: ...
    src_size: ... = None
    is_unsigned: ... = None

    def encode(self, data):
        if self.src_size is None:
            self.src_size = len(data)
        elif self.src_size != len(data):
            raise IndexError(
                "Given source size does not match actual data size"
            )
        if self.is_unsigned is None:
            # Only positive values -> use unsigned integers
            self.is_unsigned = data.min().item() >= 0

        data = _safe_cast(data, np.int32)
        return self._encode(
            data, np.empty(0, dtype=self._determine_packed_dtype())
        )

    def decode(self, const Integer[:] data):
        cdef int i, j
        cdef int min_val, max_val
        cdef int packed_val, unpacked_val
        bounds = self._get_bounds(data)
        min_val = bounds[0]
        max_val = bounds[1]
        # For signed integers, do not check lower bound (is always 0)
        # -> Set lower bound to value that is never reached
        if min_val == 0:
            min_val = -1

        cdef int32[:] output = np.zeros(self.src_size, dtype=np.int32)
        j = 0
        unpacked_val = 0
        for i in range(data.shape[0]):
            packed_val = data[i]
            if packed_val == max_val or packed_val == min_val:
                unpacked_val += packed_val
            else:
                unpacked_val += packed_val
                output[j] = unpacked_val
                unpacked_val = 0
                j += 1
        # Trim to correct size and return
        return np.asarray(output)

    def _determine_packed_dtype(self):
        if self.byte_count == 1:
            if self.is_unsigned:
                return np.uint8
            else:
                return np.int8
        elif self.byte_count == 2:
            if self.is_unsigned:
                return np.uint16
            else:
                return np.int16
        else:
            raise ValueError("Unsupported byte count")

    @cython.cdivision(True)
    def _encode(self, const Integer[:] data, OutputInteger[:] output_type):
        """
        `output_type` is merely a typed placeholder to allow for static
        typing of output.
        """
        cdef int i=0, j=0

        packed_type = np.asarray(output_type).dtype
        cdef int min_val = np.iinfo(packed_type).min
        cdef int max_val = np.iinfo(packed_type).max

        # Get length of output array
        # by summing up required length of each element
        cdef int number
        cdef long length = 0
        for i in range(data.shape[0]):
            number = data[i]
            if number < 0:
                if min_val == 0:
                    raise ValueError(
                        "Cannot pack negative numbers into unsigned type"
                    )
                # The required packed length for an element is the
                # number of times min_val/max_val need to be repeated
                length += number // min_val + 1
            elif number > 0:
                length += number // max_val + 1
            else:
                # number = 0
                length += 1

        # Fill output
        cdef OutputInteger[:] output = np.zeros(length, dtype=packed_type)
        cdef int remainder
        j = 0
        for i in range(data.shape[0]):
            remainder = data[i]
            if remainder < 0:
                if min_val == 0:
                    raise ValueError(
                        "Cannot pack negative numbers into unsigned type"
                    )
                while remainder <= min_val:
                    remainder -= min_val
                    output[j] = min_val
                    j += 1
            elif remainder > 0:
                while remainder >= max_val:
                    remainder -= max_val
                    output[j] = max_val
                    j += 1
            output[j] = remainder
            j += 1
        return np.asarray(output)

    @staticmethod
    def _get_bounds(const Integer[:] data):
        if Integer is int8:
            info = np.iinfo(np.int8)
        elif Integer is int16:
            info = np.iinfo(np.int16)
        elif Integer is int32:
            info = np.iinfo(np.int32)
        elif Integer is uint8:
            info = np.iinfo(np.uint8)
        elif Integer is uint16:
            info = np.iinfo(np.uint16)
        elif Integer is uint32:
            info = np.iinfo(np.uint32)
        else:
            raise ValueError("Unsupported integer type")
        return info.min, info.max


@dataclass
class StringArrayEncoding(Encoding):
    """
    Encoding that compresses an array of strings into an array of
    indices that point to the unique strings in that array.

    The unique strings themselves are stored as part of the
    :class:`StringArrayEncoding` as concatenated string.
    The start index of each unique string in the concatenated string
    is stored in an *offset* array.

    Parameters
    ----------
    strings : ndarray, optional
        The unique strings that are used for encoding.
        If omitted, the unique strings are determined from the data the
        first time :meth:`encode()` is called.
    data_encoding : list of Encoding, optional
        The encodings that are applied to the index array.
        If omitted, the array is directly encoded into bytes without
        further compression.
    offset_encoding : list of Encoding, optional
        The encodings that are applied to the offset array.
        If omitted, the array is directly encoded into bytes without
        further compression.

    Attributes
    ----------
    strings : ndarray
    data_encoding : list of Encoding
    offset_encoding : list of Encoding

    Examples
    --------

    >>> data = np.array(["apple", "banana", "cherry", "apple", "banana", "apple"])
    >>> print(data)
    ['apple' 'banana' 'cherry' 'apple' 'banana' 'apple']
    >>> # By default the indices would directly be encoded into bytes
    >>> # However, the indices should be printed here -> data_encoding=[]
    >>> encoding = StringArrayEncoding(data_encoding=[])
    >>> encoded = encoding.encode(data)
    >>> print(encoding.strings)
    ['apple' 'banana' 'cherry']
    >>> print(encoded)
    [0 1 2 0 1 0]
    """

    strings: ... = None
    data_encoding: ... = None
    offset_encoding: ... = None

    def __init__(self, strings=None, data_encoding=None, offset_encoding=None):
        self.strings = strings
        if data_encoding is None:
            data_encoding = [ByteArrayEncoding(TypeCode.INT32)]
        self.data_encoding = data_encoding
        if offset_encoding is None:
            offset_encoding = [ByteArrayEncoding(TypeCode.INT32)]
        self.offset_encoding = offset_encoding

    @staticmethod
    def deserialize(content):
        data_encoding = [
            deserialize_encoding(e) for e in content["dataEncoding"]
        ]
        offset_encoding = [
            deserialize_encoding(e) for e in content["offsetEncoding"]
        ]
        cdef str concatenated_strings = content["stringData"]
        cdef np.ndarray offsets = decode_stepwise(
            content["offsets"], offset_encoding
        )

        strings = np.array([
            concatenated_strings[offsets[i]:offsets[i+1]]
            # The final offset is the exclusive stop index
            for i in range(len(offsets)-1)
        ], dtype="U")

        return StringArrayEncoding(strings, data_encoding, offset_encoding)

    def serialize(self):
        if self.strings is None:
            raise ValueError(
                "'strings' must be explicitly given or needs to be "
                "determined from first encoding pass, before it is serialized"
            )

        string_data = "".join(self.strings)
        offsets = np.cumsum([0] + [len(s) for s in self.strings])

        return {
            "kind": "StringArray",
            "dataEncoding": [e.serialize() for e in self.data_encoding],
            "stringData": string_data,
            "offsets": encode_stepwise(offsets, self.offset_encoding),
            "offsetEncoding": [e.serialize() for e in self.offset_encoding],
        }

    def encode(self, data):
        if not np.issubdtype(data.dtype, np.str_):
            raise TypeError("Data must be of string type")

        if self.strings is None:
            # 'unique()' already sorts the strings, but this is not necessarily
            # desired, as this makes efficient encoding of the indices more difficult
            # -> Bring into the original order
            _, unique_indices = np.unique(data, return_index=True)
            self.strings = data[np.sort(unique_indices)]
            check_present = False
        else:
            check_present = True

        string_order = _safe_cast(np.argsort(self.strings), np.int32)
        sorted_strings = self.strings[string_order]
        sorted_indices = np.searchsorted(sorted_strings, data)
        indices = string_order[sorted_indices]
        if check_present and not np.all(self.strings[indices] == data):
            raise ValueError("Data contains strings not present in 'strings'")
        return encode_stepwise(indices, self.data_encoding)

    def decode(self, data):
        indices = decode_stepwise(data, self.data_encoding)
        return self.strings[indices]

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if not np.array_equal(self.strings, other.strings):
            return False
        if self.data_encoding != other.data_encoding:
            return False
        if self.offset_encoding != other.offset_encoding:
            return False
        return True


_encoding_classes = {
    "ByteArray": ByteArrayEncoding,
    "FixedPoint": FixedPointEncoding,
    "IntervalQuantization": IntervalQuantizationEncoding,
    "RunLength": RunLengthEncoding,
    "Delta": DeltaEncoding,
    "IntegerPacking": IntegerPackingEncoding,
    "StringArray": StringArrayEncoding,
}
_encoding_classes_kinds = {
    "ByteArrayEncoding": "ByteArray",
    "FixedPointEncoding": "FixedPoint",
    "IntervalQuantizationEncoding": "IntervalQuantization",
    "RunLengthEncoding": "RunLength",
    "DeltaEncoding": "Delta",
    "IntegerPackingEncoding": "IntegerPacking",
    "StringArrayEncoding": "StringArray",
}


def deserialize_encoding(content):
    """
    Create a :class:`Encoding` by deserializing the given *BinaryCIF* content.

    Parameters
    ----------
    content : dict
        The encoding represenet as *BinaryCIF* dictionary.

    Returns
    -------
    encoding : Encoding
        The deserialized encoding.
    """
    try:
        encoding_class = _encoding_classes[content["kind"]]
    except KeyError:
        raise ValueError(
            f"Unknown encoding kind '{content['kind']}'"
        )
    return encoding_class.deserialize(content)


def create_uncompressed_encoding(array):
    """
    Create a simple encoding for the given array that does not compress the data.

    Parameters
    ----------
    array : ndarray
        The array to to create the encoding for.

    Returns
    -------
    encoding : list of Encoding
        The encoding for the data.
    """
    if np.issubdtype(array.dtype, np.str_):
        return [StringArrayEncoding()]
    else:
        return [ByteArrayEncoding()]


def encode_stepwise(data, encoding):
    """
    Apply a list of encodings stepwise to the given data.

    Parameters
    ----------
    data : ndarray
        The data to be encoded.
    encoding : list of Encoding
        The encodings to be applied.

    Returns
    -------
    encoded_data : ndarray or bytes
        The encoded data.
    """
    for encoding in encoding:
        data = encoding.encode(data)
    return data


def decode_stepwise(data, encoding):
    """
    Apply a list of encodings stepwise to the given data.

    Parameters
    ----------
    data : ndarray or bytes
        The data to be decoded.
    encoding : list of Encoding
        The encodings to be applied.

    Returns
    -------
    decoded_data : ndarray
        The decoded data.
    """
    for enc in reversed(encoding):
        data = enc.decode(data)
    return data


def _camel_to_snake_case(attribute_name):
    return CAMEL_CASE_PATTERN.sub("_", attribute_name).lower()


def _snake_to_camel_case(attribute_name):
    attribute_name = "".join(
        word.capitalize() for word in attribute_name.split("_")
    )
    return attribute_name[0].lower() + attribute_name[1:]


def _safe_cast(array, dtype, allow_decimal_loss=False):
    source_dtype = array.dtype
    target_dtype = np.dtype(dtype)

    if target_dtype == source_dtype:
        return array

    if np.issubdtype(target_dtype, np.integer):
        if np.issubdtype(source_dtype, np.floating):
            if not allow_decimal_loss:
                raise ValueError("Cannot cast floating point to integer")
            if not np.isfinite(array).all():
                raise ValueError("Data contains non-finite values")
        elif not np.issubdtype(source_dtype, np.integer):
            # Neither float, nor integer -> cannot cast
            raise ValueError(f"Cannot cast '{source_dtype}' to integer")
        dtype_info = np.iinfo(target_dtype)
        # Check if an integer underflow/overflow would occur during conversion
        if np.max(array) > dtype_info.max or np.min(array) < dtype_info.min:
            raise ValueError("Values do not fit into the given dtype")

    return array.astype(target_dtype)