# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functionality for pseudoknot detection.
"""

__name__ = "biotite.structure"
__author__ = "Tom David Müller"
__all__ = ["pseudoknots"]

import numpy as np
import networkx as nx
from itertools import chain, product

def pseudoknots(base_pairs, scores=None, max_pseudoknot_order=None):
    """
    Identify the pseudoknot order for each base pair in a given set of
    base pairs.

    By default the algorithm removes base pairs until the remaining
    base pairs are completely nested i.e. no pseudoknots appear.
    The pseudoknot order of the removed base pairs is incremented and
    the procedure is repeated with these base pairs.
    Base pairs are removed in a way that maximizes the number of
    remaining base pairs.
    However, an optional score for each individual base pair can be
    provided.

    Parameters
    ----------
    base_pairs : ndarray, dtype=int, shape=(n,2)
        The base pairs to determine the pseudoknot order of. Each row
        represents indices form two paired bases. The structure of
        the :class:`ndarray` is equal to the structure of the output of
        :func:`base_pairs()`, where the indices represent the
        beginning of the residues.
    scores : ndarray, dtype=int, shape=(n,), optional
        The score for each base pair.
        By default, the score of each base pair is ``1``.
    max_pseudoknot_order : int, optional
        The maximum pseudoknot order to be found. If a base pair would
        be of a higher order, its order is specified as ``-1``.
        By default, the algorithm is run until all base pairs
        have an assigned pseudoknot order.

    Returns
    -------
    pseudoknot_order : ndarray, dtype=int, shape=(m,n)
        The pseudoknot order of the input `base_pairs`.
        Multiple solutions that maximize the number of basepairs or
        the given score, respectively, may be possible.
        Therefore all *m* individual solutions are returned.

    Notes
    -----
    The dynamic programming approach by Smit *et al*
    :footcite:`Smit2008` is applied to detect pseudoknots.
    The algorithm was originally developed to remove pseudoknots from a
    structure.
    However, if it is run iteratively on removed knotted pairs it can be
    used to identify the pseudoknot order.

    The pseudoknot order is defined as the minimum number of base pair
    set decompositions resulting in a nested structure
    :footcite:`Antczak2018`.
    Therefore, there are no pseudoknots between base pairs with the same
    pseudoknot order.

    Examples
    --------
    Remove the pseudoknotted base pair for the sequence *ABCbac*, where
    the corresponding big and small letters each represent a base pair:

    Define the base pairs as :class:`ndarray`:

    >>> basepairs = np.array([[0, 4],
    ...                       [1, 3],
    ...                       [2, 5]])

    Find the unknotted base pairs, optimizing for the maximum number of
    base pairs:

    >>> print(pseudoknots(basepairs, max_pseudoknot_order=0))
    [[ 0  0 -1]]

    This indicates that the base pair *Cc* is a pseudoknot.

    Given the length of the sequence (6 bases), we can also represent
    the unknotted structure in dot bracket notation:

    >>> print(dot_bracket(basepairs, 6, max_pseudoknot_order=0)[0])
    ((.)).

    If the maximum pseudoknot order is not restricted, the order of the
    knotted pairs is determined and can be represented using dot bracket
    letter notation:

    >>> print(pseudoknots(basepairs))
    [[0 0 1]]
    >>> print(dot_bracket(basepairs, 6)[0])
    (([))]

    See Also
    --------
    base_pairs
    dot_bracket

    References
    ----------

    .. footbibliography::

    """
    # List containing the results
    results = [np.full(len(base_pairs), -1, dtype='int32')]

    # if no score array is given, each base pairs' score is one
    if scores is None:
        scores = np.ones(len(base_pairs))

    # Make sure `base_pairs` has the same length as the score array
    if len(base_pairs) != len(scores):
        raise ValueError(
            "'base_pair' and 'scores' must have the same shape"
        )

    # Split the base pairs in regions
    regions = _find_regions(base_pairs, scores)

    # Compute results
    results = _get_results(regions, results, max_pseudoknot_order)

    return np.vstack(results)


class _Region():
    """
    This class represents a paired region.

    A region is a set of base pairs. This class provides methods to
    access the minimum and maximum index of the bases that are part of
    the region, handles score calculation, and backtracing to the
    original base pair array.

    Parameters
    ----------
    base_pairs: ndarray, shape=(n,2), dtype=int
        All base pairs of the structure the region is a subset for.
    region_pairs: ndarray, dtype=int
        The indices of the base pairs in ``base_pairs`` that are part of
        the region.
    scores : ndarray, dtype=int, shape=(n,) (default: None)
        The score for each base pair.
    """

    def __init__ (self, base_pairs, region_pairs, scores):
        # The Start and Stop indices for each Region
        self.start = np.min(base_pairs[region_pairs])
        self.stop = np.max(base_pairs[region_pairs])

        self.region_pairs = region_pairs
        self.score = np.sum(scores[region_pairs])

    def get_index_array(self):
        """
        Return an index array with the positions of the region`s bases
        in the original base pair array.

        Returns
        -------
        region_pairs : ndarray
            The indices of the bases in the original base pair array.
        """
        return self.region_pairs

    def __lt__(self, other):
        """
        This comparison operator is required for :func:`np.unique()`. As
        only the difference between the regions is relevant and not any
        particular order, a distinction is made by the objects unique
        ids.

        Parameters
        ----------
        other : _region
            The other region.

        Returns
        -------
        comparision : bool
            The evaluated comparison.
        """
        return id(self) < id(other)


def _find_regions(base_pairs, scores):
    """
    Find regions in a base pair array. A region is defined as a set of
    consecutively nested base pairs.

    Parameters
    ----------
    base_pairs : ndarray, dtype=int, shape=(n, 2)
        Each row is equivalent to one base pair and contains the first
        indices of the residues corresponding to each base.
    scores : ndarray, dtype=int, shape=(n,) (default: None)
        The score for each base pair.

    Returns
    -------
    regions : Graph
        The ``_Region`` objects as graph, where the edges represent
        conflicts.
    """
    # Make sure the lower residue is on the left for each row
    sorted_base_pairs = np.sort(base_pairs, axis=1)

    # Sort the first column in ascending order
    original_indices = np.argsort(sorted_base_pairs[:, 0])
    sorted_base_pairs = sorted_base_pairs[original_indices]

    # Rank each base
    # E.g.: [[3, 5]  --> [[0, 1]
    #        [9, 7]]      [3, 2]]
    order = np.argsort(sorted_base_pairs.flatten())
    rank = np.argsort(order).reshape(base_pairs.shape)

    # The base pairs belonging to the current region
    region_pairs = []
    # The individual regions
    regions = set()

    # Find separate regions
    for i in range(len(sorted_base_pairs)):
        # if a new region is to be started append the current base pair
        if len(region_pairs) == 0:
            region_pairs.append(original_indices[i])
            continue

        # Check if the current base pair belongs to the region that is
        # currently being defined
        previous_upstream_rank = rank[i-1, 0]
        this_upstream_rank = rank[i, 0]
        previous_downstream_rank = rank[i-1, 1]
        this_downstream_rank = rank[i, 1]

        # if the current base pair belongs to a new region, save the
        # current region and start a new region
        if ((previous_downstream_rank - this_downstream_rank) != 1 or
            (this_upstream_rank - previous_upstream_rank) != 1):
                regions.add(
                    _Region(base_pairs, np.array(region_pairs), scores)
                )
                region_pairs = []

        # Append the current base pair to the region
        region_pairs.append(original_indices[i])

    # The last region has no endpoint defined by the beginning of a
    # new region.
    regions.add(_Region(base_pairs, np.array(region_pairs), scores))

    # Return the graphical representation of the conflicting regions
    return _generate_graphical_representation(regions)


def _generate_graphical_representation(regions):
    """
    Find the conflicting regions and represent them graphically using
    the ``Graph`` class from ``Networkx``.

    Parameters
    ----------
    regions : set {_region, ...}
        The regions representing the consecutively nested base pairs.

    Returns
    -------
    regions : Graph
        The ``_Region`` objects as graph, where the edges represent
        conflicts.
    """

    # Create a graph
    region_graph = nx.Graph()

    # Add the regions to the graph as nodes
    region_graph.add_nodes_from(regions)

    # Get the region array and a boolean array, where the start of each
    # region is ``True``.
    region_array, (start_stops,) = _get_region_array_for(
        regions, content=[lambda a : [True, False]], dtype=['bool']
    )

    # Check each region for conflicts with other regions
    for start, region in enumerate(region_array):
        # Check each region only once
        if not start_stops[start]:
            continue

        # Find the index of the stopping of the region in the region
        # array
        stop = _get_first_occurrence_for(region_array[start+1:], region)
        stop += (start + 1)

        # Store regions the current region conflicts with
        conflicts = set()

        # Iterate over the regions between the starting and stopping
        # point of the current region
        for other_region in region_array[start+1:stop]:
            # If the other region is not already a conflict, add it to
            # the conflict set
            if other_region not in conflicts:
                conflicts.add(other_region)
            # If the other region is twice between the starting and
            # stopping point of the current region, its starting and
            # stopping point lie between the current region and it is
            # thus non-conflicting
            else:
                conflicts.remove(other_region)

        # Conflicts between regions are represented as graph edges
        edges = []

        # Convert the edges in a ``NetworkX`` compatible format
        for conflict in conflicts:
            edges.append((region, conflict))

        # Add the edges to the graph
        region_graph.add_edges_from(edges)
    return region_graph


def _get_first_occurrence_for(iterable, wanted_object):
    """
    Get the first occurrence of an object in an iterable.

    Parameters
    ----------
    iterable : iterable
        The iterable containing the object.
    wanted_object : object
        The object to be found.

    Returns
    -------
    index : int
        The index of the first occurrence of the object.
    """
    for i, value in enumerate(iterable):
        if value is wanted_object:
            return i


def _get_region_array_for(regions, content=[], dtype=[]):
    """
    Get a :class:`ndarray` of region objects. Each object occurs twice,
    representing its start and end point. The regions positions in the
    array reflect their relative positions.

    Furthermore, a list of functions can be provided enabling custom
    outputs for each objects` start and end point.

    Parameters
    ----------
    regions : set {_region, ...}
        The regions to be considered
    content : list [function, ...] (default: [])
        The functions to be considered for custom outputs. For a given
        region they must return a tuple of which the first value is
        placed at the start position and the second value at the end
        position of the region relative to the other regions.
    dtype : list [str, ...] (default: [])
        The data type of the output of the custom functions.

    Returns
    -------
    region_array : ndarray, dtype=object
        The array of ordered region objects.
    custom_content : list [ndarray, ...]
        The custom output.
    """
    # region_array and index array
    region_array = np.empty(len(regions)*2, dtype=_Region)
    index_array = np.empty(len(regions)*2, dtype='int32')

    # Content array for custom return arrays
    content_list = [None]*len(content)
    for i in range(len(content)):
        content_list[i] = np.empty(len(regions)*2, dtype=dtype[i])

    # Fill the arrays
    for i, reg in enumerate(regions):
        indices = [2*i, 2*i+1]
        region_array[indices] = reg
        for c in range(len(content_list)):
            content_list[c][indices] = content[c](reg)
        index_array[indices] = [reg.start, reg.stop]

    # Order the arrays by the base indices
    sort_mask = np.argsort(index_array)
    region_array = region_array[sort_mask]

    # if no custom array content is given only return the ordered array
    # containing the regions
    if content == []:
        return region_array

    # if custom content is given also return the ordered content
    for i in range(len(content_list)):
        content_list[i] = content_list[i][sort_mask]
    return region_array, content_list


def _remove_pseudoknots(regions):
    """
    Get the optimal solutions according to the algorithm referenced in
    :func:`pseudoknots()`.

    The algorithm uses a dynamic programming matrix in order to find
    the optimal solutions with the highest combined region scores.

    Parameters
    ----------
    regions : set {_region, ...}
        The conflicting regions for whích optimal solutions are to be
        found.
    scores : ndarray
        The score array.

    Returns
    -------
    solutions : ndarray, dtype=object
        The optimal solutions. Each solution in the ``ndarray`` is
        represented as ``set`` of unknotted regions.
    """
    # Create dynamic programming matrix
    dp_matrix_shape = len(regions)*2, len(regions)*2
    dp_matrix = np.empty(dp_matrix_shape, dtype='object')
    dp_matrix_solutions_starts = np.zeros_like(dp_matrix)
    dp_matrix_solutions_stops = np.zeros_like(dp_matrix)

    # Each index corresponds to the position in the dp matrix.
    # ``region_array`` contains the region objects and ``start_stops``
    # contains the lowest and highest positions of the regions
    region_array, (start_stops,) = _get_region_array_for(
        regions,
        [lambda a : (a.start, a.stop)],
        ['int32']
    )
    # Initialise the matrix diagonal with ndarrays of empty frozensets
    for i in range(len(dp_matrix)):
        dp_matrix[i, i] = np.array([frozenset()])

    # Iterate through the top right half of the dynamic programming
    # matrix
    for j in range(len(regions)*2):
        for i in range(j-1, -1, -1):
            solution_candidates = set()
            left = dp_matrix[i, j-1]
            bottom = dp_matrix[i+1, j]

            # Add all solutions of the cell to the left
            for solution in left:
                solution_candidates.add(solution)

            # Add all solutions of the cell to the bottom
            for solution in bottom:
               solution_candidates.add(solution)

            # Check if i and j are start/end-points of the same region
            if region_array[i] is region_array[j]:

                # Add all solutions from the cell to the bottom left
                # plus this region
                bottom_left = dp_matrix[i+1, j-1]
                for solution in bottom_left:
                    solution_candidates.add(solution | set([region_array[i]]))

            # Perform additional tests if solution in the left cell and
            # bottom cell both differ from an empty solution
            if (np.any(left != [frozenset()]) and
                np.any(bottom != [frozenset()])):

                left_highest = dp_matrix_solutions_stops[i, j-1]
                bottom_lowest = dp_matrix_solutions_starts[i+1, j]

                # For each pair of solutions check if solutions are
                # disjoint
                for solution1, highest in zip(left, left_highest):
                    for solution2, lowest in zip(bottom, bottom_lowest):
                        if highest < lowest:
                            # Both solutions are disjoint
                            solution_candidates.add(solution1 | solution2)
                        else:
                            # Both solutions are not disjoint
                            # Add subsolutions
                            for k in range(
                                np.where(start_stops==lowest)[0][0]-1,
                                np.where(start_stops==highest)[0][0]+1
                            ):
                                cell1 = dp_matrix[i, k]
                                cell2 = dp_matrix[k+1, j]
                                for subsolution1 in cell1:
                                    for subsolution2 in cell2:
                                        solution_candidates.add(
                                            subsolution1 | subsolution2
                                        )

            # Make solution candidates ``ndarray`` array of sets
            solution_candidates = np.array(list(solution_candidates))

            # Calculate the scores for each solution
            solution_scores = np.zeros(len(solution_candidates))
            for s, solution in enumerate(solution_candidates):
                score = 0
                for reg in solution:
                    score += reg.score
                solution_scores[s] = score
            # Get the indices where the score is at a maximum
            highest_scores = np.argwhere(
                solution_scores == np.amax(solution_scores)
            ).flatten()

            # Get the solutions with the highest score
            solution_candidates = solution_candidates[highest_scores]

            # Add the solutions to the dynamic programming matrix
            dp_matrix[i, j] = solution_candidates

            solution_starts = np.zeros_like(solution_candidates, dtype='int32')
            solution_stops = np.zeros_like(solution_candidates, dtype='int32')

            for s, solution in enumerate(solution_candidates):
                solution_starts[s] = min(
                    [reg.start for reg in solution], default=-1
                )
                solution_stops[s] = max(
                    [reg.stop for reg in solution], default=-1
                )

            dp_matrix_solutions_starts[i, j] = solution_starts
            dp_matrix_solutions_stops[i, j] = solution_stops

    # The top right corner contains the optimal solutions
    return dp_matrix[0, -1]


def _get_results(regions, results, max_pseudoknot_order, order=0):
    """
    Use the dynamic programming algorithm to get the pseudoknot order
    of a given set of regions. If there are remaining conflicts their
    results are recursively calculated and merged with the current
    results.

    Parameters
    ----------
    regions : set {_region, ...}
        The regions for whích optimal solutions are to be found.
    results : list [ndarray, ...]
        The results
    max_pseudoknot_order : int
        The maximum pseudoknot order to be found. If a base pair would
        be of a higher order, its order is specified as -1. If ``None``
        is given, all base pairs are evaluated.
    order : int (default: 0)
        The order that is currently evaluated.

    Returns
    -------
    results : list [ndarray, ...]
        The results
    """

    # Remove non-conflicting regions
    non_conflicting = [isolate for isolate in nx.isolates(regions)]
    regions.remove_nodes_from(non_conflicting)

    # Non-conflicting regions are of the current order:
    index_list_non_conflicting = list(
            chain(
                *[region.get_index_array() for region in non_conflicting]
            )
        )
    for result in results:
        result[index_list_non_conflicting] = order


    # If no conflicts remain, the results are complete
    if len(regions) == 0:
        return results

    # Get the optimal solutions for given regions. Evaluate each clique
    # of mutually conflicting regions seperately
    cliques = [component for component in nx.connected_components(regions)]
    solutions = [set(chain(*e)) for e in product(
        *[_remove_pseudoknots(clique) for clique in cliques]
    )]

    # Get a copy of the current results for each optimal solution
    results_list = [
        [result.copy() for result in results] for _ in range(len(solutions))
    ]

    # Evaluate each optimal solution
    for i, solution in enumerate(solutions):

        # Get the pseudoknotted regions
        pseudoknotted_regions = regions.copy()
        pseudoknotted_regions.remove_nodes_from(solution)

        # Get an index list of the unknotted base pairs
        index_list_unknotted = list(
            chain(
                *[region.get_index_array() for region in solution]
            )
        )

        # Write results for current solution
        for j, result in enumerate(results_list[i]):
            result[index_list_unknotted] = order

        # If this order is the specified maximum order, stop evaluation
        if max_pseudoknot_order == order:
            continue

        # Evaluate the pseudoknotted region
        results_list[i] = _get_results(
            pseudoknotted_regions, results_list[i],
            max_pseudoknot_order, order=order+1
        )

    # Flatten the results
    return list(chain(*results_list))
