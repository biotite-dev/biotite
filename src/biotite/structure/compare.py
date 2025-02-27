# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for calculation of characteristic values when
comparing multiple structures with each other.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["rmsd", "rmspd", "rmsf", "average", "lddt"]

import collections.abc
import warnings
import numpy as np
from biotite.structure.atoms import AtomArray, AtomArrayStack, coord
from biotite.structure.celllist import CellList
from biotite.structure.chains import get_chain_count, get_chain_positions
from biotite.structure.geometry import index_distance
from biotite.structure.residues import get_residue_count, get_residue_positions
from biotite.structure.util import vector_dot


def rmsd(reference, subject):
    r"""
    Calculate the RMSD between two structures.

    The *root mean square deviation* (RMSD) indicates the overall
    deviation of each model of a structure to a reference structure.
    It is defined as:

    .. math:: RMSD = \sqrt{ \frac{1}{n} \sum\limits_{i=1}^n (x_i - x_{ref,i})^2}

    Parameters
    ----------
    reference : AtomArray or ndarray, dtype=float, shape=(n,3)
        The reference structure.
        Alternatively, coordinates can be provided directly as
        :class:`ndarray`.
    subject : AtomArray or AtomArrayStack or ndarray, dtype=float, shape=(n,3) or shape=(m,n,3)
        Structure(s) to be compared with `reference`.
        Alternatively, coordinates can be provided directly as
        :class:`ndarray`.

    Returns
    -------
    rmsd : float or ndarray, dtype=float, shape=(m,)
        RMSD between subject and reference.
        If subject is an :class:`AtomArray` a float is returned.
        If subject is an :class:`AtomArrayStack` a :class:`ndarray`
        containing the RMSD for each model is returned.

    See Also
    --------
    rmsf : The *root mean square fluctuation*.

    Notes
    -----
    This function does not superimpose the subject to its reference.
    In most cases :func:`superimpose()` should be called prior to this
    function.

    Examples
    --------

    Calculate the RMSD of all models to the first model:

    >>> superimposed, _ = superimpose(atom_array, atom_array_stack)
    >>> rms = rmsd(atom_array, superimposed)
    >>> print(np.around(rms, decimals=3))
    [0.000 1.928 2.103 2.209 1.806 2.172 2.704 1.360 2.337 1.818 1.879 2.471
     1.939 2.035 2.167 1.789 1.653 2.348 2.247 2.529 1.583 2.115 2.131 2.050
     2.512 2.666 2.206 2.397 2.328 1.868 2.316 1.984 2.124 1.761 2.642 1.721
     2.571 2.579]
    """
    return np.sqrt(np.mean(_sq_euclidian(reference, subject), axis=-1))


def rmspd(reference, subject, periodic=False, box=None):
    r"""
    Calculate the RMSD of atom pair distances for given structures
    relative to those found in a reference structure.

    Unlike the standard RMSD, the *root-mean-square-pairwise-deviation*
    (RMSPD) is a fit-free method to determine deviations between
    a structure and a preset reference.

    .. math:: RMSPD = \sqrt{ \frac{1}{n^2} \sum\limits_{i=1}^n \sum\limits_{j \neq i}^n (d_{ij} - d_{ref,ij})^2}

    Parameters
    ----------
    reference : AtomArray or ndarray, dtype=float, shape=(n,3)
        The reference structure.
        Alternatively, coordinates can be provided directly as
        :class:`ndarray`.
    subject : AtomArray or AtomArrayStack or ndarray, dtype=float, shape=(n,3) or shape=(m,n,3)
        Structure(s) to be compared with `reference`.
        Alternatively, coordinates can be provided directly as
        :class:`ndarray`.
    periodic : bool, optional
        If set to true, periodic boundary conditions are taken into
        account (minimum-image convention).
        The `box` attribute of the `atoms` parameter is used for
        calculation.
        An alternative box can be provided via the `box` parameter.
        By default, periodicity is ignored.
    box : ndarray, shape=(3,3) or shape=(m,3,3), optional
        If this parameter is set, the given box is used instead of the
        `box` attribute of `atoms`.

    Returns
    -------
    rmspd : float or ndarray, dtype=float, shape=(m,)
        Atom pair distance RMSD between subject and reference.
        If subject is an :class:`AtomArray` a float is returned.
        If subject is an :class:`AtomArrayStack` a :class:`ndarray`
        containing the RMSD for each model is returned.

    Warnings
    --------
    Internally, this function uses :func:`index_distance()`.
    For non-orthorombic boxes (at least one angle deviates from
    90 degrees), periodic boundary conditions should be corrected
    prior to the computation of RMSPDs with `periodic` set to false
    to ensure correct results.
    (e.g. with :func:`remove_pbc()`).

    See Also
    --------
    rmsd : The *root mean square fluctuation*.
    """
    # Compute index pairs in reference structure -> pair_ij for j < i
    reflen = reference.array_length()
    index_i = np.repeat(np.arange(reflen), reflen)
    index_j = np.tile(np.arange(reflen), reflen)
    pairs = np.stack([index_i, index_j]).T
    refdist = index_distance(reference, pairs, periodic=periodic, box=box)
    subjdist = index_distance(subject, pairs, periodic=periodic, box=box)

    rmspd = np.sqrt(np.sum((subjdist - refdist) ** 2, axis=-1)) / reflen
    return rmspd


def rmsf(reference, subject):
    r"""
    Calculate the RMSF between two structures.

    The *root-mean-square-fluctuation* (RMSF) indicates the positional
    deviation of a structure to a reference structure, averaged over all
    models.
    Usually the reference structure, is the average over all models.
    The RMSF is defined as:

    .. math:: RMSF(i) = \sqrt{ \frac{1}{T} \sum\limits_{t=1}^T (x_i(t) - x_{ref,i}(t))^2}

    Parameters
    ----------
    reference : AtomArray or ndarray, dtype=float, shape=(n,3)
        The reference structure.
        Alternatively, coordinates can be provided directly as
        :class:`ndarray`.
    subject : AtomArrayStack or ndarray, dtype=float, shape=(m,n,3)
        Structures to be compared with `reference`.
        The time *t* is represented by the models in the
        :class:`AtomArrayStack`.
        Alternatively, coordinates can be provided directly as
        :class:`ndarray`.

    Returns
    -------
    rmsf : ndarray, dtype=float, shape=(n,)
        RMSF between subject and reference structure.
        Each element gives the RMSF for the atom at the respective
        index.

    See Also
    --------
    rmsd : The *root mean square deviation*.
    average : Average the structure over the models to be used as reference in this function.

    Notes
    -----
    This function does not superimpose the subject to its reference.
    In most cases :func:`superimpose()` should be called prior to this
    function.

    Examples
    --------

    Calculate the :math:`C_\alpha` RMSF of all models to the average
    model:

    >>> ca = atom_array_stack[:, atom_array_stack.atom_name == "CA"]
    >>> ca_average = average(ca)
    >>> ca, _ = superimpose(ca_average, ca)
    >>> print(rmsf(ca_average, ca))
    [1.372 0.360 0.265 0.261 0.288 0.204 0.196 0.306 0.353 0.238 0.266 0.317
     0.358 0.448 0.586 0.369 0.332 0.396 0.410 0.968]
    """
    return np.sqrt(np.mean(_sq_euclidian(reference, subject), axis=-2))


def average(atoms):
    """
    Calculate an average structure.

    The average structure has the average coordinates
    of the input models.

    Parameters
    ----------
    atoms : AtomArrayStack or ndarray, dtype=float, shape=(m,n,3)
        The structure models to be averaged.
        Alternatively, coordinates can be provided directly as
        :class:`ndarray`.

    Returns
    -------
    average : AtomArray or ndarray, dtype=float, shape=(n,3)
        Structure with averaged atom coordinates.
        If `atoms` is a :class:`ndarray` and :class:`ndarray` is also
        returned.

    Notes
    -----
    The calculated average structure is not suitable for visualization
    or geometric calculations, since bond lengths and angles will
    deviate from meaningful values.
    This method is rather useful to provide a reference structure for
    calculation of e.g. the RMSD or RMSF.
    """
    coords = coord(atoms)
    if coords.ndim != 3:
        raise TypeError("Expected an AtomArrayStack or an ndarray with shape (m,n,3)")
    mean_coords = np.mean(coords, axis=0)
    if isinstance(atoms, AtomArrayStack):
        mean_array = atoms[0].copy()
        mean_array.coord = mean_coords
        return mean_array
    else:
        return mean_coords


def lddt(
    reference,
    subject,
    aggregation="all",
    atom_mask=None,
    partner_mask=None,
    inclusion_radius=15,
    distance_bins=(0.5, 1.0, 2.0, 4.0),
    exclude_same_residue=True,
    exclude_same_chain=False,
    filter_function=None,
    symmetric=False,
):
    """
    Calculate the *local Distance Difference Test* (lDDT) score of a structure with
    respect to its reference.
    :footcite:`Mariani2013`

    Parameters
    ----------
    reference : AtomArray
        The reference structure.
    subject : AtomArray or AtomArrayStack or ndarray, dtype=float, shape=(n,3) or shape=(m,n,3)
        The structure(s) to evaluate with respect to `reference`.
        The number of atoms must be the same as in `reference`.
        Alternatively, coordinates can be provided directly as
        :class:`ndarray`.
    aggregation : {'all', 'chain', 'residue', 'atom'} or ndarray, shape=(n,), dtype=int, optional
        Defines on which scale the lDDT score is calculated.

        - `'all'`: The score is computed over all contacts.
        - `'chain'`: The score is calculated for each chain separately.
        - `'residue'`: The score is calculated for each residue separately.
        - `'atom'`: The score is calculated for each atom separately.

        Alternatively, an array of aggregation bins can be provided, i.e. each contact
        is assigned to the corresponding bin.
    atom_mask : ndarray, shape=(n,), dtype=bool, optional
        If given, the contacts are only computed for the masked atoms.
        Atoms excluded by the mask do not have any contacts and their *lDDT* would
        be NaN in case of ``aggregation="atom"``.
        Providing this mask can significantly speed up the computation, if
        only for certain chains/residues/atoms the *lDDT* is of interest.
    partner_mask : ndarray, shape=(n,), dtype=bool, optional
        If given, only contacts **to** the masked atoms are considered.
        While `atom_mask` does not alter the *lDDT* for the masked atoms,
        `partner_mask` does, as for each atom only the masked atoms are considered
        as potential contact partners.
    inclusion_radius : float, optional
        Pairwise atom distances are considered within this radius in `reference`.
    distance_bins : list of float, optional
        The distance bins for the score calculation, i.e if a distance deviation is
        within the first bin, the score is 1, if it is outside all bins, the score is 0.
    exclude_same_residue : bool, optional
        If true, only atom distances between different residues are considered.
        Otherwise, also atom distances within the same residue are included.
    exclude_same_chain : bool, optional
        If true, only atom distances between different chains are considered.
        Otherwise, also atom distances within the same chain are included.
    filter_function : Callable(ndarray, shape=(n,2), dtype=int -> ndarray, shape=(n,), dtype=bool), optional
        Used for custom contact filtering, if the other parameters are not sufficient.
        A function that takes an array of contact atom indices and returns a mask that
        is ``True`` for all contacts that should be retained.
        All other contacts are not considered for lDDT computation.
    symmetric : bool, optional
        If set to true, the *lDDT* score is computed symmetrically.
        This means both contacts found in the `reference` and `subject` structure are
        considered.
        Hence the score is independent of which structure is given as `reference` and
        `subject`.
        Note that in this case `subject` must be an :class:`AtomArray` as well.
        By default, only contacts in the `reference` are considered.

    Returns
    -------
    lddt : float or ndarray, dtype=float
        The lDDT score for each model and aggregation bin.
        The shape depends on `subject` and `aggregation`:
        If `subject` is an :class:`AtomArrayStack` (or equivalent coordinate
        :class:`ndarray`), a dimension depicting each model is added.
        if `aggregation` is not ``'all'``, a second dimension with the length equal to
        the number of aggregation bins is added (i.e. number of chains, residues, etc.).
        If both, an :class:`AtomArray` as `subject` and ``aggregation='all'`` is passed,
        a float is returned.

    Notes
    -----
    The lDDT score measures how well the pairwise atom distances in a model match the
    corresponding distances in a reference.
    Hence, like :func:`rmspd()` it works superimposition-free, but instead of capturing
    the global deviation, only the local environment within the `inclusion_radius` is
    considered.

    Note that by default, also hydrogen atoms are considered in the distance
    calculation.
    If this is undesired, the hydrogen atoms can be removed prior to the calculation.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    Calculate the global lDDT of all models to the first model:

    >>> reference = atom_array_stack[0]
    >>> subject = atom_array_stack[1:]
    >>> print(lddt(reference, subject))
    [0.799 0.769 0.792 0.836 0.799 0.752 0.860 0.769 0.825 0.777 0.760 0.787
     0.790 0.783 0.804 0.842 0.769 0.797 0.757 0.852 0.811 0.786 0.805 0.755
     0.734 0.794 0.771 0.778 0.842 0.772 0.815 0.789 0.828 0.750 0.826 0.739
     0.760]

    Calculate the residue-wise lDDT for a single model:

    >>> subject = atom_array_stack[1]
    >>> print(lddt(reference, subject, aggregation="residue"))
    [0.599 0.692 0.870 0.780 0.830 0.881 0.872 0.658 0.782 0.901 0.888 0.885
     0.856 0.795 0.847 0.603 0.895 0.878 0.871 0.789]

    As example for custom aggregation, calculate the lDDT for each chemical element:

    >>> unique_elements = np.unique(reference.element)
    >>> element_bins = np.array(
    ...     [np.where(unique_elements == element)[0][0] for element in reference.element]
    ... )
    >>> element_lddt = lddt(reference, subject, aggregation=element_bins)
    >>> for element, lddt_for_element in zip(unique_elements, element_lddt):
    ...     print(f"{element}: {lddt_for_element:.3f}")
    C: 0.837
    H: 0.770
    N: 0.811
    O: 0.808

    If the reference structure has more atoms resolved than the subject structure,
    the missing atoms can be indicated with *NaN* values:

    >>> reference = atom_array_stack[0]
    >>> subject = atom_array_stack[1].copy()
    >>> # Simulate the situation where the first residue is missing in the subject
    >>> subject.coord[subject.res_id == 1] = np.nan
    >>> global_lddt = lddt(reference, subject)
    >>> print(f"{global_lddt:.3f}")
    0.751
    """
    reference_coord = coord(reference)
    subject_coord = coord(subject)
    if subject_coord.shape[-2] != reference_coord.shape[-2]:
        raise IndexError(
            f"The given reference has {reference_coord.shape[-2]} atoms, but the "
            f"subject has {subject_coord.shape[-2]} atoms"
        )

    contacts = _find_contacts(
        reference,
        atom_mask,
        partner_mask,
        inclusion_radius,
        exclude_same_residue,
        exclude_same_chain,
        filter_function,
    )
    if symmetric:
        if not isinstance(subject, AtomArray):
            raise TypeError(
                "Expected 'AtomArray' as subject, as symmetric lDDT is enabled, "
                f"but got '{type(subject).__name__}'"
            )
        subject_contacts = _find_contacts(
            subject,
            atom_mask,
            partner_mask,
            inclusion_radius,
            exclude_same_residue,
            exclude_same_chain,
            filter_function,
        )
        contacts = np.concatenate((contacts, subject_contacts), axis=0)
        # Adding additional contacts may introduce duplicates between the existing and
        # new ones -> filter them out
        contacts = np.unique(contacts, axis=0)
    if (
        isinstance(aggregation, str)
        and aggregation == "all"
        and atom_mask is None
        and partner_mask is None
    ):
        # Remove duplicate pairs as each pair appears twice
        # (if i is in threshold distance to j, j is also in threshold distance to i)
        # keep only the pair where i < j
        # This improves performance due to less distances that need to be computed
        # The assumption also only works when no atoms are masked
        contacts = contacts[contacts[:, 0] < contacts[:, 1]]

    reference_distances = index_distance(reference_coord, contacts)
    subject_distances = index_distance(subject_coord, contacts)
    deviations = np.abs(subject_distances - reference_distances)
    distance_bins = np.asarray(distance_bins)
    fraction_preserved_bins = np.count_nonzero(
        deviations[..., np.newaxis] <= distance_bins[np.newaxis, :], axis=-1
    ) / len(distance_bins)

    # Aggregate the fractions over the desired level
    if isinstance(aggregation, str) and aggregation == "all":
        # Average over all contacts
        return np.mean(fraction_preserved_bins, axis=-1)
    else:
        # A string is also a 'Sequence'
        # -> distinguish between string and array, list, etc.
        if isinstance(
            aggregation, (np.ndarray, collections.abc.Sequence)
        ) and not isinstance(aggregation, str):
            return _average_over_indices(
                fraction_preserved_bins,
                bins=np.asarray(aggregation)[contacts[:, 0]],
            )
        elif aggregation == "chain":
            return _average_over_indices(
                fraction_preserved_bins,
                bins=get_chain_positions(reference, contacts[:, 0]),
                n_bins=get_chain_count(reference),
            )
        elif aggregation == "residue":
            return _average_over_indices(
                fraction_preserved_bins,
                bins=get_residue_positions(reference, contacts[:, 0]),
                n_bins=get_residue_count(reference),
            )
        elif aggregation == "atom":
            return _average_over_indices(
                fraction_preserved_bins, contacts[:, 0], reference.array_length()
            )
        else:
            raise ValueError(f"Invalid aggregation level '{aggregation}'")


def _sq_euclidian(reference, subject):
    """
    Calculate squared euclidian distance between atoms in two
    structures.

    Parameters
    ----------
    reference : AtomArray or ndarray, dtype=float, shape=(n,3)
        Reference structure.
    subject : AtomArray or AtomArrayStack or ndarray, dtype=float, shape=(n,3) or shape=(m,n,3)
        Structure(s) whose atoms squared euclidian distance to
        `reference` is measured.

    Returns
    -------
    ndarray, dtype=float, shape=(n,) or shape=(m,n)
        Squared euclidian distance between subject and reference.
        If subject is an :class:`AtomArray` a 1-D array is returned.
        If subject is an :class:`AtomArrayStack` a 2-D array is
        returned.
        In this case the first dimension indexes the AtomArray.
    """
    reference_coord = coord(reference)
    subject_coord = coord(subject)
    if reference_coord.ndim != 2:
        raise TypeError(
            "Expected an AtomArray or an ndarray with shape (n,3) as reference"
        )
    dif = subject_coord - reference_coord
    return vector_dot(dif, dif)


def _to_sparse_indices(all_contacts):
    """
    Create tuples of contact indices from the :meth:`CellList.get_atoms()` return value.

    In other words, they would mark the non-zero elements in a dense contact matrix.

    Parameters
    ----------
    all_contacts : ndarray, dtype=int, shape=(m,n)
        The contact indices as returned by :meth:`CellList.get_atoms()`.
        Padded with -1, in the second dimension.
        Dimension *m* marks the query atoms, dimension *n* marks the contact atoms.

    Returns
    -------
    combined_indices : ndarray, dtype=int, shape=(l,2)
        The contact indices.
        Each column contains the query and contact atom index.
    """
    # Find rows where a query atom has at least one contact
    non_empty_indices = np.where(np.any(all_contacts != -1, axis=1))[0]
    # Take those rows and flatten them
    contact_indices = all_contacts[non_empty_indices].flatten()
    # For each row the corresponding query atom is the same
    # Hence in the flattened form the query atom index is simply repeated
    query_indices = np.repeat(non_empty_indices, all_contacts.shape[1])
    combined_indices = np.stack([query_indices, contact_indices], axis=1)
    # Remove the padding values
    return combined_indices[contact_indices != -1]


def _find_contacts(
    atoms=None,
    atom_mask=None,
    partner_mask=None,
    inclusion_radius=15,
    exclude_same_residue=False,
    exclude_same_chain=True,
    filter_function=None,
):
    """
    Find contacts between the atoms in the given structure.

    Parameters
    ----------
    atoms : AtomArray
        The structure to find the contacts for.
    atom_mask : ndarray, shape=(n,), dtype=bool, optional
        If given, the contacts are only computed for the masked atoms.
        Atoms excluded by the mask do not have any contacts and their *lDDT* would
        be NaN in case of ``aggregation="atom"``.
        Providing this mask can significantly speed up the computation, if
        only for certain chains/residues/atoms the *lDDT* is of interest.
    partner_mask : ndarray, shape=(n,), dtype=bool, optional
        If given, only contacts **to** the masked atoms are considered.
        While `atom_mask` does not alter the *lDDT* for the masked atoms,
        `partner_mask` does, as for each atom only the masked atoms are considered
        as potential contact partners.
    inclusion_radius : float, optional
        Pairwise atom distances are considered within this radius.
    exclude_same_residue : bool, optional
        If true, only atom distances between different residues are considered.
        Otherwise, also atom distances within the same residue are included.
    exclude_same_chain : bool, optional
        If true, only atom distances between different chains are considered.
        Otherwise, also atom distances within the same chain are included.
    filter_function : Callable(ndarray, shape=(n,2), dtype=int -> ndarray, shape=(n,), dtype=bool), optional
        Used for custom contact filtering, if the other parameters are not sufficient.
        A function that takes an array of contact atom indices and returns a mask that
        is ``True`` for all contacts that should be retained.
        All other contacts are not considered for lDDT computation.

    Returns
    -------
    contacts : ndarray, shape=(n,2), dtype=int
        The array of contacts.
        Each element represents a pair of atom indices that are in contact.
    """
    coords = coord(atoms)
    selection = ~np.isnan(coords).any(axis=-1)
    if partner_mask is not None:
        selection &= partner_mask
    # Use a cell list to find atoms within inclusion radius in O(n) time complexity
    cell_list = CellList(coords, inclusion_radius, selection=selection)
    # Pairs of indices for atoms within the inclusion radius
    if atom_mask is None:
        all_contacts = cell_list.get_atoms(coords, inclusion_radius)
    else:
        filtered_contacts = cell_list.get_atoms(coords[atom_mask], inclusion_radius)
        # Map the contacts for the masked atoms to the original coordinates
        # Rows that were filtered out by the mask are fully padded with -1
        # consistent with the padding of `get_atoms()`
        all_contacts = np.full(
            (coords.shape[0], filtered_contacts.shape[-1]),
            -1,
            dtype=filtered_contacts.dtype,
        )
        all_contacts[atom_mask] = filtered_contacts
    # Convert into pairs of indices
    contacts = _to_sparse_indices(all_contacts)

    if exclude_same_chain:
        # Do the same for the chain level
        chain_indices = get_chain_positions(atoms, contacts.flatten()).reshape(
            contacts.shape
        )
        contacts = contacts[chain_indices[:, 0] != chain_indices[:, 1]]
    elif exclude_same_residue:
        # Find the index of the residue for each atom
        residue_indices = get_residue_positions(atoms, contacts.flatten()).reshape(
            contacts.shape
        )
        # Remove contacts between atoms of the same residue
        contacts = contacts[residue_indices[:, 0] != residue_indices[:, 1]]
    else:
        # In any case self-contacts should not be considered
        contacts = contacts[contacts[:, 0] != contacts[:, 1]]
    if filter_function is not None:
        mask = filter_function(contacts)
        if mask.shape != (contacts.shape[0],):
            raise IndexError(
                f"Mask returned from filter function has shape {mask.shape}, "
                f"but expected ({contacts.shape[0]},)"
            )
        contacts = contacts[mask, :]
    return contacts


def _average_over_indices(values, bins, n_bins=None):
    """
    For each unique index in `bins`, average the corresponding values in `values`.

    Based on
    https://stackoverflow.com/questions/79140661/how-to-sum-values-based-on-a-second-index-array-in-a-vectorized-manner

    Parameters
    ----------
    values : ndarray, shape=(..., n)
        The values to average.
    bins : ndarray, shape=(n,) dtype=int
        Associates each value from `values` with a bin.
    n_bins : int
        The total number of bins.
        This is necessary as the some bin in `bins`may be empty.
        By default the number of bins is determined from `bins`.

    Returns
    -------
    averaged : ndarray, shape=(..., k)
        The averaged values.
        *k* is the maximum value in `bins` + 1.
    """
    if n_bins is None:
        n_elements_per_bin = np.bincount(bins)
        n_bins = len(n_elements_per_bin)
    else:
        n_elements_per_bin = np.bincount(bins, minlength=n_bins)
    # The last dimension is replaced by the number of bins
    # Broadcasting in 'np.add.at()' requires the replaced dimension to be the first
    aggregated = np.zeros((n_bins, *values.shape[:-1]), dtype=values.dtype)
    np.add.at(aggregated, bins, np.swapaxes(values, 0, -1))
    # If an atom has no contacts, the corresponding value is NaN
    # This result is expected, hence the warning is ignored
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Bring the bin dimension into the last dimension again
        return np.swapaxes(aggregated, 0, -1) / n_elements_per_bin
