"""
Structural formula of a small molecule in xkcd style
====================================================

This script draws a structural formula of a molecule from an
:class:`AtomArray`, by using the bond, element and charge information.
The layout of the diagram is computed via *NetworkX*' *Kamada–Kawai*
implementation.
The *Matplotlib* *xkcd* style distracts from the fact that used graph
layout algorithm is poorly suitable for structural formulas, so the
application of this example is probably in rather non-serious contexts.
"""


# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import biotite.structure as struc
import biotite.structure.info as info


SEPARATION = 0.01
NODE_SIZE = 20
COLORS = {
    "N": "royalblue",
    "O": "firebrick",
    "S": "gold",
}


BOND_ORDER = {
    struc.BondType.SINGLE: 1,
    struc.BondType.DOUBLE: 2,
    struc.BondType.TRIPLE: 3,
    struc.BondType.QUADRUPLE: 4,
    struc.BondType.AROMATIC_SINGLE: 1,
    struc.BondType.AROMATIC_DOUBLE: 2,
}


def split_bond(i_pos, j_pos, separation, number):
    """
    Get line start and end positions for double, triple, etc. bonds.
    """
    if number == 1:
        return [(i_pos, j_pos)]
    
    diff = np.asarray(j_pos) - np.asarray(i_pos)
    orth = [[0,-1], [1,0]] @ diff
    sep = orth / np.linalg.norm(orth) * separation
    return [
        (i_pos + p*sep , j_pos + p*sep )
        for p in np.linspace(-number/2, number/2, number)
    ]

# Cyanidin, partly responsible for the red color in roses
molecule = info.residue("HWB")

# Combine an heteroatom + hydrogen in the label
labels = molecule.element.copy()
for i, j, _ in molecule.bonds.as_array():
    if molecule.element[i] == "H":
        labels[j] += "H"
    if molecule.element[j] == "H":
        labels[i] += "H"
# Add charge to heteroatom labels
for i in range(molecule.array_length()):
    if molecule.charge[i] > 0:
        labels[i] += "+"
    if molecule.charge[i] < 0:
        labels[i] += "-"

# Do not show hydrogen atoms
hydrogen_mask = (molecule.element == "H")
molecule = molecule[~hydrogen_mask]
labels = labels[~hydrogen_mask]

# Arrange atoms in 2D space
graph = nx.Graph()
graph.add_edges_from(
    [(i, j, {"bond_type": int(bond_type)})
     for i, j, bond_type in molecule.bonds.as_array()]
)
pos = nx.kamada_kawai_layout(graph)
pos = {i: np.flip(p) for i, p in pos.items()}


# Create the diagram
with plt.xkcd(scale=1, length=100, randomness=2):
    fig, ax = plt.subplots(figsize=(6.0, 8.0))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "ROSES ARE RED,\n"
        "VIOLETS ARE BLUE,\n"
        "AND ANTHOCYANINS\n"
        "GIVE THEM THEIR HUE",
        fontsize=30
    )
    
    # Plot the bonds
    for i, j in graph.edges():
        i_pos = pos[i]
        j_pos = pos[j]
        order = BOND_ORDER[graph.edges[i,j]["bond_type"]]
        for i_pos, j_pos in split_bond(i_pos, j_pos, SEPARATION, order):
            x_data = (i_pos[0], j_pos[0])
            y_data = (i_pos[1], j_pos[1])
            ax.plot(x_data, y_data, color="black")

    # Plot the heteroatom labels
    for i in graph.nodes():
        if molecule.element[i] != "C":
            x = pos[i][0]
            y = pos[i][1]
            t = ax.text(
                x, y, labels[i],
                fontsize=NODE_SIZE, ha="center", va="center",
                color=COLORS.get(molecule.element[i], "black")
            )
            # Remove clashes of the label with bond lines
            # by setting a white background behind the label
            t.set_bbox({'pad': 0, 'color': 'white'})
    
    fig.tight_layout()

plt.show()