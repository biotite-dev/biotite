# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for pseudoknot detection.
"""

__name__ = "biotite.structure"
__author__ = "Tom David Müller"
__all__ = ["pseudoknots"]

import numpy as np
from copy import deepcopy


class _region():
    """
    A representation for a region.

    A region is a set of basepairs. This class provides function to
    access the minimum and maximum index of the bases that are part of
    the region, handles score calculation and backtracing to the
    original basepair array.
    """

    def __init__ (self, base_pairs, region_pairs):
        # The Start and Stop indices for each Region
        self.start = np.min(base_pairs[region_pairs])
        self.stop = np.max(base_pairs[region_pairs])

        self.base_pairs = base_pairs
        self.region_pairs = region_pairs
        self.score = None

    def get_index_mask(self):
        """
        Return an index mask with the positions of the bases in the region
        in the original basepair array.

        Returns
        -------
        region_pairs : ndarray
            The indices of the bases in the original basepair array.
        """
        return self.region_pairs

    def get_score(self, scoring):
        """
        Return the score of the region according to a scoring array. It is
        calculated once on demand and then stored in memory.

        Parameters
        ----------
        scoring : ndarray
            The scoring array.

        Returns
        -------
        score : int
            The regions score.
        """
        if self.score is None:
            self.score = np.sum(scoring[self.get_index_mask()])
        return self.score

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

def pseudoknots(base_pairs, scoring=None):

    # Result array
    results = [np.zeros(len(base_pairs), dtype='int32')]

    # if no scoring function is given, each basepairs score is one
    if scoring is None:
        scoring = np.ones(len(base_pairs))

    # Make sure base_pairs has the same length as the scoring function
    if len(base_pairs) != len(scoring):
        raise ValueError(
        "Each Value of the scoring vector must correspond to a basepair."
    )

    # Split the basepairs in regions
    regions = _find_regions(base_pairs)

    # Only retain conflicting regions
    cleaned_regions = _remove_non_conflicting_regions(regions)

    # Group mutually conflicting regions
    conflict_cliques = _conflict_cliques(cleaned_regions)

    # For each clique calculate all optimal solutions
    for clique in conflict_cliques:
        results = _get_result_diff(clique, scoring)

    return np.vstack(results)


def _find_regions(base_pairs):
    """
    Find regions in a base pair arrray. A region is defined as a set of
    consecutively nested basepairs.

    Parameters
    ----------
    base_pairs : ndarray, dtype=int, shape=(n, 2)
        Each row is equivalent to one basepair and contains the first
        indices of the residues corresponding to each base.

    Returns
    -------
    regions : set {_region, ...}
        The regions representing the consecutively nested basepairs.
    """

    # Make sure the lower residue is on the left for each row
    sorted_base_pairs = np.sort(base_pairs, axis=1)

    # Sort the first column in ascending order
    original_indices = np.argsort(sorted_base_pairs[:, 0])
    sorted_base_pairs = sorted_base_pairs[original_indices]

    # Rank the right side in ascending order
    downstream_order = np.argsort(sorted_base_pairs[:,1])
    downstream_rank = np.argsort(downstream_order)

    # The basepairs belonging to the current region
    region_pairs = []
    # The individual regions
    regions = set()

    # Find seperate regions
    for i, base_pair in enumerate(sorted_base_pairs):
        # if a new region is to be started append the current basepair
        if len(region_pairs) == 0:
            region_pairs.append(original_indices[i])
            continue

        # Check if the current basepair belongs to the region that is
        # currently being defined
        previous_rank = downstream_rank[i-1]
        this_rank = downstream_rank[i]
        # if the current basepair belongs to a new region, save the
        # current region and start a new region
        if (previous_rank - this_rank) != 1:
            regions.add(_region(base_pairs, np.array(region_pairs)))
            region_pairs = []

        # Append the current basepair to the region
        region_pairs.append(original_indices[i])

    # The last regions has no endpoint defined by the beginning of a
    # new region.
    regions.add(_region(base_pairs, np.array(region_pairs)))

    return regions


def _remove_non_conflicting_regions(regions):
    """
    Remove regions that are not conflicting.

    Parameters
    ----------
    regions : set {_region, ...}
        Regions including non-conflicting regions.

    Returns
    -------
    regions : set {_region, ...}
        The regions without non-conflicting regions.
    """
    # Get the region array and a boolean array, where the start of each
    # region ``True``.
    region_array, start_stops = _get_region_array_for(
        regions, content=[lambda a : [True, False]], dtype=['bool']
    )
    starts = np.nonzero(start_stops[0])[0]

    # Regions that are not conflicting
    to_remove = []
    for start_index in starts:
        # Get the current regions start and stop indices in the region
        # array
        stop_index = _get_first_occurrence_for(
            region_array[start_index+1:], region_array[start_index]
        )
        stop_index = start_index + 1 + stop_index

        # Count the occurrence of each individual region between the
        # start and stop indices of the regions
        _, counts = np.unique(
            region_array[start_index+1:stop_index], return_counts=True
        )
        # if no regions are between the start and stop indices the
        # region is non-conflicting
        if len(counts) == 0:
            to_remove.append(region_array[start_index])
        # if every region between the start and stop indices has its
        # start and stop between the current region's start and stop
        # indices the current region is not conflicting
        elif np.amin(counts) == 2:
            to_remove.append(region_array[start_index])

    # Remove all non conflicting regions and return the set of
    # conflicting regions
    region_array = region_array[~ np.isin(region_array, to_remove)]
    return set(region_array)


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
    Get a ``ndarray`` of region objects. Each object occurs twice
    representing its start and end point. Their position in the array
    reflects the regions relative positions.

    Furthermore, a list of functions can be provided enabling custom
    outputs for each objects` start and end point.

    Parameters
    ----------
    regions : set {_region, ...}
        The regions to be considered
    content : list [function, ...] (default: [])
        The functions to be considered for custom outputs.
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
    region_array = np.empty(len(regions)*2, dtype=_region)
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


def _conflict_cliques(regions):
    # Get a region array and an array where each region start is +1 and
    # each stop is -1
    region_array, start_stops = _get_region_array_for(
        regions, content=[lambda a : [1, -1]], dtype=['int32']
    )
    start_stops = start_stops[0]

    # Iterate through the array and add up the values of the
    # corresponding ``start_stops`` array. Separate conflicts are marked
    # by zero sums.
    total = 0
    start = 0
    cliques = []
    for i in range(len(start_stops)):
        total += start_stops[i]
        if total == 0:
            cliques.append(set(region_array[start:i+1]))
            start = i+1
    if len(region_array[start:]) > 0:
        cliques.append(set(region_array[start:]))

    # Return the conflict cliques as list of lists
    return cliques


def _get_optimal_solutions(clique, scoring):

    # Create dynamic programming matrix
    dp_matrix_shape = len(clique)*2, len(clique)*2
    dp_matrix = np.empty(dp_matrix_shape, dtype='object')
    dp_matrix_solutions_starts = np.empty_like(dp_matrix)
    dp_matrix_solutions_stops = np.empty_like(dp_matrix)

    # Each index corresponds to the position in the dp matrix.
    # ``region_array`` contains the region objects and ``start_stops``
    # contains the lowest and highest positions of the regions
    region_array, (start_stops,) = _get_region_array_for(
        clique,
        [lambda a : (a.start, a.stop)],
        ['int32']
    )
    # Initialise the matrix diagonal with ``ndarray``s of empty
    # ``frozenset``s
    for i in range(len(dp_matrix)):
        dp_matrix[i, i] = np.array([frozenset()])

    # Iterate through the top right of the dynamic programming matrix
    for j in range(len(clique)*2):
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
                starts = np.empty(
                    (2, max(len(left), len(bottom))), dtype='int32'
                )

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
            scores = np.zeros(len(solution_candidates))
            for s, solution in enumerate(solution_candidates):
                score = 0
                for reg in solution:
                    score += reg.get_score(scoring)
                scores[s] = score
            # Get the indices where the score is at a maximum
            highest_scores = np.argwhere(scores == np.amax(scores)).flatten()

            # Get the solutions with the highest score
            solution_candidates = solution_candidates[highest_scores]

            # Add the solutions to the dynamic programming matrix
            dp_matrix[i, j] = solution_candidates

            solution_starts = np.empty_like(solution_candidates, dtype='int32')
            solution_stops = np.empty_like(solution_candidates, dtype='int32')

            for s, solution in enumerate(solution_candidates):
                minimum = -1
                maximum = -1
                for reg in solution:
                    if minimum == -1 or maximum == -1:
                        minimum = reg.start
                        maximum = reg.stop
                        continue
                    if minimum > reg.start:
                        minimum = reg.start
                    if maximum < reg.stop:
                        maximum = reg.stop
                solution_starts[s] = minimum
                solution_stops[s] = maximum

            dp_matrix_solutions_starts[i, j] = solution_starts
            dp_matrix_solutions_stops[i, j] = solution_stops

    # The top right corner contains the optimal solutions
    return dp_matrix[0, -1]


def _get_result_diff(clique, scoring):

    # Get the optimal solutions for the given clique
    optimal_solutions = _get_optimal_solutions(clique, scoring)

    # Each optimal solution gets their own list of solutions
    results_diff = np.empty(len(optimal_solutions), dtype=list)

    # Get the results for each optimal solution
    for o, optimal_solution in enumerate(optimal_solutions):

        # The results for the ṕarticular solution
        results = np.zeros(len(scoring), dtype='int32')

        # The removed regions in the optimal solution
        next_clique = clique - optimal_solution

        # Increment the order of the basepairs in the removed regions
        for reg in next_clique:
            results[reg.get_index_mask()] += 1

        # Remove non-conflicting-regions for evaluation of the next
        # clique
        next_clique = _remove_non_conflicting_regions(next_clique)

        # The next clique is only evaluated if it contains conflicting
        # regions
        if len(next_clique) > 0:
            next_result_diff = _get_result_diff(next_clique, scoring)

            # Merge results of this clique with the next clique
            results_diff[o] = []
            for i, diff in enumerate(next_result_diff):
                results_diff[o].append(deepcopy(results))
                results_diff[o][-1] = (results_diff[o][-1] + diff)

        else:
            results_diff[o] = [results]

    # Return flattened results
    return [result for results in results_diff for result in results]
