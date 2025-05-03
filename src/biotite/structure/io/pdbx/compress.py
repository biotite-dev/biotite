__all__ = ["compress"]
__name__ = "biotite.structure.io.pdbx"
__author__ = "Patrick Kunzmann"

import itertools
import warnings
import msgpack
import numpy as np
import biotite.structure.io.pdbx.bcif as bcif
from biotite.structure.io.pdbx.bcif import _encode_numpy as encode_numpy
from biotite.structure.io.pdbx.encoding import (
    ByteArrayEncoding,
    DeltaEncoding,
    FixedPointEncoding,
    IntegerPackingEncoding,
    RunLengthEncoding,
    StringArrayEncoding,
)


def compress(data, float_tolerance=None, rtol=1e-6, atol=1e-4):
    """
    Try to reduce the size of a *BinaryCIF* file (or block, category, etc.) by testing
    different data encodings for each data array and selecting the one, which results in
    the smallest size.

    Parameters
    ----------
    data : BinaryCIFFile or BinaryCIFBlock or BinaryCIFCategory or BinaryCIFColumn or BinaryCIFData
        The data to compress.
    float_tolerance : float, optional
        The relative error that is accepted when compressing floating point numbers.
        DEPRECATED: Use `rtol` instead.
    rtol, atol : float, optional
        The compression factor of floating point numbers is chosen such that
        either the relative (`rtol`) or absolute (`atol`) tolerance is fulfilled
        for each value, i.e. the difference between the compressed and uncompressed
        value is smaller than the tolerance.

    Returns
    -------
    compressed_file : BinaryCIFFile or BinaryCIFBlock or BinaryCIFCategory or BinaryCIFColumn or BinaryCIFData
        The compressed data with the same type as the input data.
        If no improved compression is found for a :class:`BinaryCIFData` array,
        the input data is kept.
        Hence, the return value is no deep copy of the input data.

    Examples
    --------

    >>> from io import BytesIO
    >>> pdbx_file = BinaryCIFFile()
    >>> set_structure(pdbx_file, atom_array_stack)
    >>> # Write uncompressed file
    >>> uncompressed_file = BytesIO()
    >>> pdbx_file.write(uncompressed_file)
    >>> _ = uncompressed_file.seek(0)
    >>> print(f"{len(uncompressed_file.read()) // 1000} KB")
    927 KB
    >>> # Write compressed file
    >>> pdbx_file = compress(pdbx_file)
    >>> compressed_file = BytesIO()
    >>> pdbx_file.write(compressed_file)
    >>> _ = compressed_file.seek(0)
    >>> print(f"{len(compressed_file.read()) // 1000} KB")
    111 KB
    """
    if float_tolerance is not None:
        warnings.warn(
            "The 'float_tolerance' parameter is deprecated, use 'rtol' instead",
            DeprecationWarning,
        )

    match type(data):
        case bcif.BinaryCIFFile:
            return _compress_file(data, rtol, atol)
        case bcif.BinaryCIFBlock:
            return _compress_block(data, rtol, atol)
        case bcif.BinaryCIFCategory:
            return _compress_category(data, rtol, atol)
        case bcif.BinaryCIFColumn:
            return _compress_column(data, rtol, atol)
        case bcif.BinaryCIFData:
            return _compress_data(data, rtol, atol)
        case _:
            raise TypeError(f"Unsupported type {type(data).__name__}")


def _compress_file(bcif_file, rtol, atol):
    compressed_file = bcif.BinaryCIFFile()
    for block_name, bcif_block in bcif_file.items():
        try:
            compressed_block = _compress_block(bcif_block, rtol, atol)
        except Exception:
            raise ValueError(f"Failed to compress block '{block_name}'")
        compressed_file[block_name] = compressed_block
    return compressed_file


def _compress_block(bcif_block, rtol, atol):
    compressed_block = bcif.BinaryCIFBlock()
    for category_name, bcif_category in bcif_block.items():
        try:
            compressed_category = _compress_category(bcif_category, rtol, atol)
        except Exception:
            raise ValueError(f"Failed to compress category '{category_name}'")
        compressed_block[category_name] = compressed_category
    return compressed_block


def _compress_category(bcif_category, rtol, atol):
    compressed_category = bcif.BinaryCIFCategory()
    for column_name, bcif_column in bcif_category.items():
        try:
            compressed_column = _compress_column(bcif_column, rtol, atol)
        except Exception:
            raise ValueError(f"Failed to compress column '{column_name}'")
        compressed_category[column_name] = compressed_column
    return compressed_category


def _compress_column(bcif_column, rtol, atol):
    data = _compress_data(bcif_column.data, rtol, atol)
    if bcif_column.mask is not None:
        mask = _compress_data(bcif_column.mask, rtol, atol)
    else:
        mask = None
    return bcif.BinaryCIFColumn(data, mask)


def _compress_data(bcif_data, rtol, atol):
    array = bcif_data.array
    if len(array) == 1:
        # No need to compress a single value -> Use default uncompressed encoding
        return bcif.BinaryCIFData(array)

    if np.issubdtype(array.dtype, np.str_):
        # Leave encoding empty for now, as it is explicitly set later
        encoding = StringArrayEncoding(data_encoding=[], offset_encoding=[])
        # Run encode to initialize the data and offset arrays
        indices = encoding.encode(array)
        offsets = np.cumsum([0] + [len(s) for s in encoding.strings])
        encoding.data_encoding, _ = _find_best_integer_compression(indices)
        encoding.offset_encoding, _ = _find_best_integer_compression(offsets)
        return bcif.BinaryCIFData(array, [encoding])

    elif np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            # NaN/inf values cannot be represented by integers
            # -> do not use integer encoding
            return bcif.BinaryCIFData(array, [ByteArrayEncoding()])
        to_integer_encoding = FixedPointEncoding(
            10 ** _get_decimal_places(array, rtol, atol)
        )
        try:
            integer_array = to_integer_encoding.encode(array)
        except ValueError:
            # With the given tolerances integer underflow/overflow would occur
            # -> do not use integer encoding
            return bcif.BinaryCIFData(array, [ByteArrayEncoding()])
        else:
            best_encoding, size_compressed = _find_best_integer_compression(
                integer_array
            )
            if size_compressed < _data_size_in_file(bcif.BinaryCIFData(array)):
                return bcif.BinaryCIFData(array, [to_integer_encoding] + best_encoding)
            else:
                # The float array is smaller -> encode it directly as bytes
                return bcif.BinaryCIFData(array, [ByteArrayEncoding()])

    elif np.issubdtype(array.dtype, np.integer):
        array = _to_smallest_integer_type(array)
        encodings, _ = _find_best_integer_compression(array)
        return bcif.BinaryCIFData(array, encodings)

    else:
        raise TypeError(f"Unsupported data type {array.dtype}")


def _find_best_integer_compression(array):
    """
    Try different data encodings on an integer array and return the one that results in
    the smallest size.
    """
    best_encoding_sequence = None
    smallest_size = np.inf

    for use_delta in [False, True]:
        if use_delta:
            encoding = DeltaEncoding()
            array_after_delta = encoding.encode(array)
            encodings_after_delta = [encoding]
        else:
            encodings_after_delta = []
            array_after_delta = array
        for use_run_length in [False, True]:
            # Use encoded data from previous step to save time
            if use_run_length:
                encoding = RunLengthEncoding()
                array_after_rle = encoding.encode(array_after_delta)
                encodings_after_rle = encodings_after_delta + [encoding]
            else:
                encodings_after_rle = encodings_after_delta
                array_after_rle = array_after_delta
            for packed_byte_count in [None, 1, 2]:
                if packed_byte_count is not None:
                    # Quickly check this heuristic
                    # to avoid computing an exploding packed data array
                    if (
                        _estimate_packed_length(array_after_rle, packed_byte_count)
                        >= array_after_rle.nbytes
                    ):
                        # Packing would not reduce the size
                        continue
                    encoding = IntegerPackingEncoding(packed_byte_count)
                    array_after_packing = encoding.encode(array_after_rle)
                    encodings_after_packing = encodings_after_rle + [encoding]
                else:
                    encodings_after_packing = encodings_after_rle
                    array_after_packing = array_after_rle
                encoding = ByteArrayEncoding()
                encoded_array = encoding.encode(array_after_packing)
                encodings = encodings_after_packing + [encoding]
                # Pack data directly instead of using the BinaryCIFData class
                # to avoid the unnecessary re-encoding of the array,
                # as it is already available in 'encoded_array'
                serialized_encoding = [enc.serialize() for enc in encodings]
                serialized_data = {
                    "data": encoded_array,
                    "encoding": serialized_encoding,
                }
                size = _data_size_in_file(serialized_data)
                if size < smallest_size:
                    best_encoding_sequence = encodings
                    smallest_size = size
    return best_encoding_sequence, smallest_size


def _estimate_packed_length(array, packed_byte_count):
    """
    Estimate the length of an integer array after packing it with a given number of
    bytes.

    Parameters
    ----------
    array : numpy.ndarray
        The array to pack.
    packed_byte_count : int
        The number of bytes used for packing.

    Returns
    -------
    length : int
        The estimated length of the packed array.
    """
    # Use int64 to avoid integer overflow in the following line
    max_val_per_element = np.int64(2 ** (8 * packed_byte_count))
    n_bytes_per_element = packed_byte_count * (np.abs(array // max_val_per_element) + 1)
    return np.sum(n_bytes_per_element, dtype=np.int64)


def _to_smallest_integer_type(array):
    """
    Convert an integer array to the smallest possible integer type, that is still able
    to represent all values in the array.

    Parameters
    ----------
    array : numpy.ndarray
        The array to convert.

    Returns
    -------
    array : numpy.ndarray
        The converted array.
    """
    if array.min() >= 0:
        for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            if np.all(array <= np.iinfo(dtype).max):
                return array.astype(dtype)
    for dtype in [np.int8, np.int16, np.int32, np.int64]:
        if np.all(array >= np.iinfo(dtype).min) and np.all(
            array <= np.iinfo(dtype).max
        ):
            return array.astype(dtype)
    raise ValueError("Array is out of bounds for all integer types")


def _data_size_in_file(data):
    """
    Get the size of the data, it would have when written into a *BinaryCIF* file.

    Parameters
    ----------
    data : BinaryCIFData or dict
        The data array whose size is measured.
        Can be either a :class:`BinaryCIFData` object or already serialized data.

    Returns
    -------
    size : int
        The size of the data array in the file in bytes.
    """
    if isinstance(data, bcif.BinaryCIFData):
        data = data.serialize()
    bytes_in_file = msgpack.packb(data, use_bin_type=True, default=encode_numpy)
    return len(bytes_in_file)


def _get_decimal_places(array, rtol, atol):
    """
    Get the number of decimal places in a floating point array.

    Parameters
    ----------
    array : numpy.ndarray
        The array to analyze.
    rtol, atol : float, optional
        The relative and absolute tolerance allowed when the values are cut off after
        the returned number of decimal places.

    Returns
    -------
    decimals : int
        The number of decimal places.
    """
    if rtol <= 0 and atol <= 0:
        raise ValueError("At least one of 'rtol' and 'atol' must be greater than 0")
    # 0 would give NaN when rounding on decimals
    array = array[array != 0]
    for decimals in itertools.count(start=min(0, -_order_magnitude(array))):
        error = np.abs(np.round(array, decimals) - array)
        if decimals == 100:
            raise
        if np.all((error < rtol * np.abs(array)) | (error < atol)):
            return decimals


def _order_magnitude(array):
    """
    Get the order of magnitude of floating point values.

    Parameters
    ----------
    array : ndarray, dtype=float
        The value to analyze.

    Returns
    -------
    magnitude : int
        The order of magnitude, i.e. the maximum exponent a number in the array would
        have in scientific notation, if only one digit is left of the decimal point.
    """
    array = array[array != 0]
    if len(array) == 0:
        # No non-zero values -> define order of magnitude as 0
        return 0
    return int(np.max(np.floor(np.log10(np.abs(array)))).item())
