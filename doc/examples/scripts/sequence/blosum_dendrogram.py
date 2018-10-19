"""
Dendrogram of the BLOSUM62 matrix
=================================

"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.phylo as phylo
import biotite.sequence.graphics as graphics

matrix = align.SubstitutionMatrix.std_protein_matrix()
print(matrix)

########################################################################
#
#
#

# Matrix should not contain ambiguous symbols or stop signal
matrix = align.SubstitutionMatrix(
    seq.Alphabet(matrix.get_alphabet1().get_symbols()[:-4]),
    seq.Alphabet(matrix.get_alphabet2().get_symbols()[:-4]),
    matrix.score_matrix()[:-4, :-4]
)
similarities = matrix.score_matrix()
print(matrix)

########################################################################
#
#
#

def get_distance(similarities, i, j):
    """
    s_max = (similarities[i,i] + similarities[j,j]) / 2
    s_rand = (np.mean(similarities[i,:]) + np.mean(similarities[j,:])) / 2
    print(similarities[i,j] - s_rand)
    return -np.log((similarities[i,j] - s_rand) / (s_max - s_rand))
    """
    s_max = (similarities[i,i] + similarities[j,j]) / 2
    return s_max - similarities[i,j]

distances = np.zeros(similarities.shape)
for i in range(distances.shape[0]):
    for j in range(distances.shape[1]):
        distances[i,j] = get_distance(similarities, i, j)

tree = phylo.upgma(distances)

fig = plt.figure(figsize=(8.0, 5.0))
ax = fig.add_subplot(111)
graphics.plot_dendrogram(
    ax, tree, orientation="top", labels=matrix.get_alphabet1().get_symbols()
)
ax.set_ylabel("Distance")
# Add grid for clearer distance perception
ax.yaxis.grid(color="lightgray")
plt.show()