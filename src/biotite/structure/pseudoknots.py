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

class region():

    def __init__ (self, base_pairs, region_pairs):
        # The Start and Stop indices for each Region
        self.start = np.min(base_pairs[region_pairs])
        self.stop = np.max(base_pairs[region_pairs])
        # The base pair array
        self.base_pairs = base_pairs
        self.region_pairs = region_pairs
        self.score = None


    # Gets a boolean mask from the original basepair array
    def get_index_mask(self):
        return self.region_pairs

    # Get the score of a given region. Only calculate when score is
    # needed.
    def get_score(self, scoring):
        if self.score is None:
            self.score = np.sum(scoring[self.get_index_mask()])
        return self.score

    # Required for numpy unique support
    def __lt__(self, other):
        return False


def pseudoknots(base_pairs, scoring=None):

    # Result array
    results = [np.zeros(len(base_pairs), dtype='int32')]

    # if no scoring function is given, each basepairs score is one
    if scoring is None:
        scoring = np.ones(len(base_pairs))

    # Make sure base_pairs has the same length as the scoring function
    # TODO: Throw Error
    assert len(base_pairs) == len(scoring)

    # Split the basepairs in regions
    regions = _find_regions(base_pairs)

    # Only retain conflicting regions
    cleaned_regions = _remove_non_conflicting_regions(regions)

    # Group mutually conflicting regions
    conflict_clusters = _cluster_conflicts(cleaned_regions)

    # For each cluster calculate all optimal solutions
    for cluster in conflict_clusters:
        results = _get_result_diff(cluster, scoring, results)

    return np.vstack(results)

def _get_result_diff(cluster, scoring, order=0):

    # Get the optimal solutions for the given cluster
    optimal_solutions = _get_optimal_solutions(cluster, scoring)

    results_diff = np.empty(len(optimal_solutions), dtype=list)

    # Get the results for each optimal solution
    for o, optimal_solution in enumerate(optimal_solutions):

        results = np.zeros(len(scoring), dtype='int32')

        next_cluster = cluster - optimal_solution

        for reg in next_cluster:
            results[reg.get_index_mask()] += 1

        next_cluster = _remove_non_conflicting_regions(next_cluster)

        if len(next_cluster) > 0:
            next_result_diff = _get_result_diff(next_cluster, scoring)

            this_result_diff = []
            for i, diff in enumerate(next_result_diff):
                print('start')
                print(this_result_diff)
                print(diff)
                print('stop')

                this_result_diff.append(deepcopy(results))
                this_result_diff[-1] = (this_result_diff[-1] + diff)

        else:
            this_result_diff = [results]

        results_diff[o] = this_result_diff


        """
        if len(next_cluster) > 0:
            next_results = _get_results(next_cluster)
        # The regions retained in the optimal solutions are of the
        # current order
        for reg in optimal_solution:
            results_copy[o][r][reg.get_index_mask()] = order

        # Get the regions not retained in the optimal solution
        new_cluster = cluster - optimal_solution


        # remove non-conflicting regions in new cluster
        new_cluster_cleaned =_remove_non_conflicting_regions(new_cluster)
        for reg in (new_cluster - new_cluster_cleaned):
            print('a')
            print(results_copy[o][r])
            results_copy[o][r][reg.get_index_mask()] = order + 1
            print(results_copy[o][r])
            print(reg.get_index_mask())
            print(order)
        new_cluster = new_cluster_cleaned
        # if there is more than one region remaining, increase the
        # order and solve for the remaining regions
        if len(new_cluster) > 1:
            new_results = _get_results(
                new_cluster, scoring, results_copy[o][r], order=(order+1)
            )
        # if there is one or less region remaining it is of the next
        # highest order
        else:
            for reg in new_cluster:
                results_copy[o][r][reg.get_index_mask()] = order + 1
        """

    # Return flattened results
    return [result for results in results_diff for result in results]

def _apply_results():
    pass

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
    basepair_candidates : set {region, ...]
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
            regions.add(region(base_pairs, np.array(region_pairs)))
            region_pairs = []

        # Append the current basepair to the region
        region_pairs.append(original_indices[i])

    # The last regions has no endpoint defined by the beginning of a
    # new region.
    regions.add(region(base_pairs, np.array(region_pairs)))

    return regions


def _remove_non_conflicting_regions(regions):
    """
    Remove regions that are not conflicting
    TODO: Some Regions cant be removed
        Right now only non-conflicting-regions of type ABB'A' can be
        removed but ABCB'C'A' -> BCB'C' cannot be performed!
    """
    region_array, start_stops = _get_region_array_for(
        regions, content=[lambda a : [True, False]], dtype=['bool']
    )
    #print(len(start_stops[0]))
    #print(region_array)
    starts = np.nonzero(start_stops[0])[0]
    #print(starts)
    to_remove = []
    #print(starts)
    for start_index in starts:
        #print(region_array)
        #print(start_index)
        stop_index = _get_first_occurrence_for(
            region_array[start_index+1:], region_array[start_index]
        )
        stop_index = start_index + 1 + stop_index
        _, counts = np.unique(region_array[start_index+1:stop_index], return_counts=True)
        #print(counts)
        #print(region_array[start_index])
        #print(region_array[stop_index])
        if len(counts) == 0:
            to_remove.append(region_array[start_index])
        elif np.amin(counts) == 2:
            to_remove.append(region_array[start_index])

    region_array = region_array[~ np.isin(region_array, to_remove)]
    return set(region_array)


    region_array = _get_region_array_for(regions)
    to_remove = [None]
    while to_remove != []:
        to_remove = []
        for i in range(len(region_array)-1):
            if region_array[i] is region_array[i+1]:
                to_remove.append(region_array[i])
        region_array = region_array[~ np.isin(region_array, to_remove)]
    return set(region_array)


def _get_first_occurrence_for(array, wanted_value):
    """
    Returns the first occurrences of a numpy array
    """
    for i, value in enumerate(array):
        if value is wanted_value:
            return i


def _get_region_array_for(regions, content=[], dtype=[]):
    # region_array and index array
    region_array = np.empty(len(regions)*2, dtype=region)
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

def _cluster_conflicts(regions):
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
    clusters = []
    for i in range(len(start_stops)):
        total += start_stops[i]
        if total == 0:
            clusters.append(set(region_array[start:i+1]))
            start = i+1
    if len(region_array[start:]) > 0:
        clusters.append(set(region_array[start:]))

    # Return the conflict clusters as list of lists
    return clusters

def _get_optimal_solutions(cluster, scoring):

    # Create dynamic programming matrix
    dp_matrix_shape = len(cluster)*2, len(cluster)*2
    dp_matrix = np.empty(dp_matrix_shape, dtype='object')

    # Each index corresponds to the position in the dp matrix.
    # ``region_array`` contains the region objects and ``start_stops``
    # contains the lowest and highest positions of the regions
    region_array, (start_stops,) = _get_region_array_for(
        cluster,
        [lambda a : (a.start, a.stop)],
        ['int32']
    )
    #print(region_array)
    # Initialise the matrix diagonal with ``ndarray``s of empty
    # ``frozenset``s
    for i in range(len(dp_matrix)):
        dp_matrix[i, i] = np.array([frozenset()])

    # Iterate through the top right of the dynamic programming matrix
    for j in range(len(cluster)*2):
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
            #print(left)
            #print(bottom)
            if (np.any(left != [frozenset()])) and (np.any(bottom != [frozenset()])):
                #print(dp_matrix)
                #print(i)
                #print(j)
                starts = np.empty(
                    (2, max(len(left), len(bottom))), dtype='int32'
                )
                stops = np.empty_like(starts)

                # Precalculate the minimum and maximum base position of
                # each solution
                for c, cell in enumerate([left, bottom]):
                    for s, solution in enumerate(cell):
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
                        starts[c, s] = minimum
                        stops[c, s] = maximum

                # For each pair of solutions check if solutions are
                # disjoint
                for l, solution1 in enumerate(left):
                    for b, solution2 in enumerate(bottom):
                        # for each pair of solutions get the lowest and
                        # highest value
                        lowest = starts[1][b]
                        highest = stops[0][l]
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

            # Make solution candidates ``ndarray`` array of sets to
            # allow fancy indexing
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

    # The top right corner contains the optimal solutions
    return dp_matrix[0, -1]







