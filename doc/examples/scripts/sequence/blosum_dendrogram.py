"""
Dendrogram of the BLOSUM62 matrix
=================================

In this example a dendrogram is created, that displays the similarity
of amino acids in the *BLOSUM62* substitution matrix.
The amino acids are clustered with the UPGMA method.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.phylo as phylo
import biotite.sequence.graphics as graphics

# Obtain BLOSUM62
matrix = align.SubstitutionMatrix.std_protein_matrix()
print(matrix)

########################################################################
# The original *BLOSUM62* contains symbols for ambiguous amino acids and
# the stop signal.
# As these are not actual amino acids, a new substitution matrix is
# created, where these symbols are are removed.

# Matrix should not contain ambiguous symbols or stop signal
matrix = align.SubstitutionMatrix(
    seq.Alphabet(matrix.get_alphabet1().get_symbols()[:-4]),
    seq.Alphabet(matrix.get_alphabet2().get_symbols()[:-4]),
    matrix.score_matrix()[:-4, :-4]
)
similarities = matrix.score_matrix()
print(matrix)

########################################################################
# Now a function must be defined, that converts the similarity depicted
# by a substitution matrix into a distance required by the UPGMA method.
# In this case, the distance is defined as the difference between the
# similarity of the two symbols and the average maximum similarity of
# the symbols to themselves.
#
# Finally the obtained (phylogenetic) tree is plotted as dendrogram.
def get_distance(similarities, i, j):
    s_max = (similarities[i,i] + similarities[j,j]) / 2
    return s_max - similarities[i,j]

distances = np.zeros(similarities.shape)
for i in range(distances.shape[0]):
    for j in range(distances.shape[1]):
        distances[i,j] = get_distance(similarities, i, j)

tree = phylo.upgma(distances)

fig = plt.figure(figsize=(8.0, 5.0))
ax = fig.add_subplot(111)
# Use the 3-letter amino acid code aa label
labels = [seq.ProteinSequence.convert_letter_1to3(letter).capitalize()
          for letter in matrix.get_alphabet1()]
graphics.plot_dendrogram(
    ax, tree, orientation="top", labels=labels
)
ax.set_ylabel("Distance")
# Add grid for clearer distance perception
ax.yaxis.grid(color="lightgray")
plt.show()