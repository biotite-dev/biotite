# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["apply_segment_wise", "spread_segment_wise", "get_segment_masks",
           "get_segment_starts_for", "get_segment_positions", "segment_iter"]

import numpy as np


def apply_segment_wise(starts, data, function, axis):
    """
    Generalized version of :func:`apply_residue_wise()` for
    residues and chains.

    Parameters
    ----------
    starts : ndarray, dtype=int
        The sorted start indices of segments.
        Includes exclusive stop, i.e. the length of the corresponding
        atom array.
    """
    # The result array
    processed_data = None
    for i in range(len(starts)-1):
        segment = data[starts[i]:starts[i+1]]
        if axis == None:
            value = function(segment)
        else:
            value = function(segment, axis=axis)
        value = function(segment, axis=axis)
        # Identify the shape of the resulting array by evaluation
        # of the function return value for the first segment
        if processed_data is None:
            if isinstance(value, np.ndarray):
                # Maximum length of the processed data
                # is length of segment of size 1 -> length of all IDs
                # (equal to atom array length)
                processed_data = np.zeros(
                    (len(starts)-1,) + value.shape, dtype=value.dtype
                )
            else:
                # Scalar value -> one dimensional result array
                processed_data = np.zeros(
                    len(starts)-1, dtype=type(value)
                )
        # Write values into result arrays
        processed_data[i] = value
    return processed_data


def spread_segment_wise(starts, input_data):
    """
    Generalized version of :func:`spread_residue_wise()`
    for residues and chains.

    Parameters
    ----------
    starts : ndarray, dtype=int
        The sorted start indices of segments.
        Includes exclusive stop, i.e. the length of the corresponding
        atom array.
    """
    output_data = np.zeros(starts[-1], dtype=input_data.dtype)
    for i in range(len(starts)-1):
        start = starts[i]
        stop = starts[i + 1]
        output_data[start:stop] = input_data[i]
    return output_data


def get_segment_masks(starts, indices):
    """
    Generalized version of :func:`get_residue_masks()`
    for residues and chains.

    Parameters
    ----------
    starts : ndarray, dtype=int
        The sorted start indices of segments.
        Includes exclusive stop, i.e. the length of the corresponding
        atom array.
    """
    indices = np.asarray(indices)
    length = starts[-1]
    masks = np.zeros((len(indices), length), dtype=bool)

    if (indices < 0).any():
        raise ValueError("This function does not support negative indices")
    if (indices >= length).any():
        index = np.min(np.where(indices >= length)[0])
        raise ValueError(
            f"Index {index} is out of range for "
            f"an atom array with length {length}"
        )
    
    insertion_points = np.searchsorted(starts, indices, side="right") - 1
    for i, point in enumerate(insertion_points):
        masks[i, starts[point] : starts[point+1]] = True
    
    return masks


def get_segment_starts_for(starts, indices):
    """
    Generalized version of :func:`get_residue_starts_for()`
    for residues and chains.

    Parameters
    ----------
    starts : ndarray, dtype=int
        The sorted start indices of segments.
        Includes exclusive stop, i.e. the length of the corresponding
        atom array.
    """
    indices = np.asarray(indices)
    length = starts[-1]
    # Remove exclusive stop
    starts = starts[:-1]

    if (indices < 0).any():
        raise ValueError("This function does not support negative indices")
    if (indices >= length).any():
        index = np.min(np.where(indices >= length)[0])
        raise ValueError(
            f"Index {index} is out of range for "
            f"an atom array with length {length}"
        )
    
    insertion_points = np.searchsorted(starts, indices, side="right") - 1
    return starts[insertion_points]


def get_segment_positions(starts, indices):
    """
    Generalized version of :func:`get_residue_positions()`
    for residues and chains.

    Parameters
    ----------
    starts : ndarray, dtype=int
        The sorted start indices of segments.
        Includes exclusive stop, i.e. the length of the corresponding
        atom array.
    """
    indices = np.asarray(indices)
    length = starts[-1]
    # Remove exclusive stop
    starts = starts[:-1]

    if (indices < 0).any():
        raise ValueError("This function does not support negative indices")
    if (indices >= length).any():
        index = np.min(np.where(indices >= length)[0])
        raise ValueError(
            f"Index {index} is out of range for "
            f"an atom array with length {length}"
        )
    
    return np.searchsorted(starts, indices, side="right") - 1


def segment_iter(array, starts):
    """
    Generalized version of :func:`residue_iter()`
    for residues and chains.

    Parameters
    ----------
    starts : ndarray, dtype=int
        The sorted start indices of segments.
        Includes exclusive stop, i.e. the length of the corresponding
        atom array.
    """
    for i in range(len(starts)-1):
        yield array[..., starts[i] : starts[i+1]]
