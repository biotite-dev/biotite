# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = [
    "get_segment_starts",
    "apply_segment_wise",
    "spread_segment_wise",
    "get_segment_masks",
    "get_segment_starts_for",
    "get_segment_positions",
    "segment_iter",
]

import numpy as np


def get_segment_starts(
    array, add_exclusive_stop, continuous_categories=(), equal_categories=()
):
    """
    Generalized version of :func:`get_residue_starts()` for residues and chains.

    The starts are determined from value changes in the given annotations.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack) to get the segment starts from.
    add_exclusive_stop : bool, optional
        If true, the exclusive stop of the input atom array,
        i.e. ``array.array_length()``, is added to the returned array of start indices
        as last element.
    continuous_categories : tuple of str, optional
        Annotation categories that are expected to be continuously increasing within a
        segment.
        This means if the value of such an annotation decreases from one atom to
        another, a new segment is started.
    equal_categories : tuple of str, optional
        Annotation categories that are expected to be equal within a segment.
        This means if the value of such an annotation changes from one atom to
        another, a new segment is started.

    Returns
    -------
    starts : ndarray, dtype=int
        The start indices of segments in `array`.
    """
    if array.array_length() == 0:
        return np.array([], dtype=int)

    segment_start_mask = np.zeros(array.array_length() - 1, dtype=bool)
    for annot_name in continuous_categories:
        annotation = array.get_annotation(annot_name)
        segment_start_mask |= np.diff(annotation) < 0
    for annot_name in equal_categories:
        annotation = array.get_annotation(annot_name)
        segment_start_mask |= annotation[1:] != annotation[:-1]

    # Convert mask to indices
    # Add 1, to shift the indices from the end of a segment
    # to the start of a new segment
    chain_starts = np.where(segment_start_mask)[0] + 1

    # The first chain is not included yet -> Insert '[0]'
    if add_exclusive_stop:
        return np.concatenate(([0], chain_starts, [array.array_length()]))
    else:
        return np.concatenate(([0], chain_starts))


def apply_segment_wise(starts, data, function, axis=None):
    """
    Generalized version of :func:`apply_residue_wise()` for
    residues and chains.

    Parameters
    ----------
    starts : ndarray, dtype=int
        The sorted start indices of segments.
        Includes exclusive stop, i.e. the length of the corresponding
        atom array.
    data : ndarray
        The data, whose intervals are the parameter for `function`.
        Must have same length as `array`.
    function : function
        The `function` must have either the form *f(data)* or
        *f(data, axis)* in case `axis` is given. Every `function` call
        must return a value with the same shape and data type.
    axis : int, optional
        This value is given to the `axis` parameter of `function`.

    Returns
    -------
    processed_data : ndarray
        Segment-wise evaluation of `data` by `function`.
        The size of the first dimension of this array is equal to the amount of
        residues.
    """
    # The result array
    processed_data = None
    for i in range(len(starts) - 1):
        segment = data[starts[i] : starts[i + 1]]
        if axis is None:
            value = function(segment)
        else:
            value = function(segment, axis=axis)
        # Identify the shape of the resulting array by evaluation
        # of the function return value for the first segment
        if processed_data is None:
            if isinstance(value, np.ndarray):
                # Maximum length of the processed data
                # is length of segment of size 1 -> length of all IDs
                # (equal to atom array length)
                processed_data = np.zeros(
                    (len(starts) - 1,) + value.shape, dtype=value.dtype
                )
            else:
                # Scalar value -> one dimensional result array
                processed_data = np.zeros(len(starts) - 1, dtype=type(value))
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
    input_data : ndarray
        The data to be spread.
        The length of the 0-th axis must be equal to the amount of different residue IDs
        in `array`.

    Returns
    -------
    output_data : ndarray
        Segment-wise spread `input_data`.
        Length is the same as `array_length()` of `array`.
    """
    seg_lens = starts[1:] - starts[:-1]
    return np.repeat(input_data, seg_lens, axis=0)


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
    indices : ndarray, dtype=int, shape=(k,)
        These indices indicate the atoms to get the corresponding
        segments for.
        Negative indices are not allowed.

    Returns
    -------
    residues_masks : ndarray, dtype=bool, shape=(k,n)
        Multiple boolean masks, one for each given index in `indices`.
        Each array masks the atoms that belong to the same segment as
        the atom at the given index.
    """
    indices = np.asarray(indices)
    length = starts[-1]
    masks = np.zeros((len(indices), length), dtype=bool)

    if (indices < 0).any():
        raise ValueError("This function does not support negative indices")
    if (indices >= length).any():
        index = np.min(np.where(indices >= length)[0])
        raise ValueError(
            f"Index {index} is out of range for an atom array with length {length}"
        )

    insertion_points = np.searchsorted(starts, indices, side="right") - 1
    for i, point in enumerate(insertion_points):
        masks[i, starts[point] : starts[point + 1]] = True

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
    indices : ndarray, dtype=int, shape=(k,)
        These indices point to the atoms to get the corresponding
        segment starts for.
        Negative indices are not allowed.

    Returns
    -------
    start_indices : ndarray, dtype=int, shape=(k,)
        The indices that point to the segment starts for the input
        `indices`.
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
            f"Index {index} is out of range for an atom array with length {length}"
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
    indices : ndarray, shape=(k,)
        These indices point to the atoms to get the corresponding
        residue positions for.
        Negative indices are not allowed.

    Returns
    -------
    segment_indices : ndarray, shape=(k,)
        The indices that point to the position of the segments.
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
            f"Index {index} is out of range for an atom array with length {length}"
        )

    return np.searchsorted(starts, indices, side="right") - 1


def segment_iter(array, starts):
    """
    Generalized version of :func:`residue_iter()`
    for residues and chains.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The structure to iterate over.
    starts : ndarray, dtype=int
        The sorted start indices of segments.
        Includes exclusive stop, i.e. the length of the corresponding
        atom array.

    Yields
    ------
    segment : AtomArray or AtomArrayStack
       Each residue or chain of the structure.
    """
    for i in range(len(starts) - 1):
        yield array[..., starts[i] : starts[i + 1]]
