from __future__ import annotations

__name__ = "biotite.sequence.graphics"
__author__ = "Patrick Kunzmann"
__all__ = ["plot_dendrogram"]

from typing import Any, Literal
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from biotite.sequence.phylo.tree import get_leaves, get_root
from biotite.typing import MplColor


def plot_dendrogram(
    axes: Axes,
    tree: nx.DiGraph,
    orientation: Literal["left", "right", "bottom", "top"] = "left",
    use_distances: bool = True,
    labels: list[str] | None = None,
    label_size: float | None = None,
    color: MplColor = "black",
    show_distance: bool = True,
    **kwargs: Any,
) -> None:
    """
    Plot a dendrogram from a (phylogenetic) tree.

    Parameters
    ----------
    axes : Axes
        A *Matplotlib* axes, that is used as plotting area.
    tree : DiGraph
        The tree to be visualized.
    orientation : {'left', 'right', 'bottom', 'top'}, optional
        The position of the root node in the plot
    use_distances : bool, optional
        If true, the ``"distance"`` attribute of the edges is used as
        distance measure.
        Otherwise the topological distance is used.
    labels : list of str, optional
        The leaf node labels.
        The label of a leaf node is the entry at the position of the
        leaf node.
    label_size : float, optional
        The font size of the labels.
    color : tuple or str, optional
        A *Matplotlib* compatible color, that is used to draw the lines
        of the dendrogram.
    show_distance : bool, optional
        If true, the distance from the root is shown on the
        corresponding axis.
    **kwargs
        Additional parameters that are used to draw the dendrogram
        lines.
    """
    if orientation not in ("left", "right", "bottom", "top"):
        raise ValueError(f"'{orientation}' is not a valid orientation")

    root = get_root(tree)
    # The leaves are placed on the 'label' axis in depth-first order
    leaves = get_leaves(tree, root)
    leaf_positions = {leaf: i for i, leaf in enumerate(leaves)}

    # The term 'distance'
    # refers to positions along the 'distance' axis
    # the term 'pos'
    # refers to positions along the other axis
    node_distances: dict[Any, float] = {root: 0.0}
    for parent, child in nx.dfs_edges(tree, root):
        if use_distances:
            edge_distance = tree.edges[parent, child]["distance"]
        else:
            # Use topologic distance of the child to its parent,
            # which is always 1
            edge_distance = 1
        node_distances[child] = node_distances[parent] + edge_distance
    # Required for setting the plot limits
    max_distance = max(node_distances.values())

    # Position of each node on the 'label' axis,
    # which is the center of its children
    node_positions: dict[Any, float] = {}
    for node in nx.dfs_postorder_nodes(tree, root):
        children = list(tree.successors(node))
        if len(children) == 0:
            # No children -> no line can be drawn
            node_positions[node] = leaf_positions[node]
            continue
        distance = node_distances[node]
        child_distances = [node_distances[child] for child in children]
        child_pos = [node_positions[child] for child in children]
        node_positions[node] = sum(child_pos) / len(child_pos)
        if orientation in ["left", "right"]:
            # Line connecting the children
            axes.plot(
                [distance, distance],
                [child_pos[0], child_pos[-1]],
                color=color,
                marker="None",
                **kwargs,
            )
            # Lines depicting the distances of the children
            for child_dist, pos in zip(child_distances, child_pos):
                axes.plot(
                    [distance, child_dist],
                    [pos, pos],
                    color=color,
                    marker="None",
                    **kwargs,
                )
        else:
            # Line connecting the children
            axes.plot(
                [child_pos[0], child_pos[-1]],
                [distance, distance],
                color=color,
                marker="None",
                **kwargs,
            )
            # Lines depicting the distances of the children
            for child_dist, pos in zip(child_distances, child_pos):
                axes.plot(
                    [pos, pos],
                    [distance, child_dist],
                    color=color,
                    marker="None",
                    **kwargs,
                )

    sorted_labels: list[str]
    if labels is not None:
        # Sort labels using the order of the leaves in the tree
        sorted_labels = [labels[leaf] for leaf in leaves]
    else:
        sorted_labels = [str(leaf) for leaf in leaves]
    # The distance axis does not start at 0,
    # since the root line would not properly rendered
    # Hence the limit is set a to small fraction of the entire axis
    # beyond 0
    zero_limit = -0.01 * max_distance
    if orientation == "left":
        axes.set_xlim(zero_limit, max_distance)
        axes.set_ylim(-1, len(leaves))
        axes.set_yticks(np.arange(0, len(leaves)))
        axes.set_yticklabels(sorted_labels)
        axes.yaxis.set_tick_params(
            left=False,
            right=False,
            labelleft=False,
            labelright=True,
            labelsize=label_size,
        )
        axes.xaxis.set_tick_params(
            bottom=True,
            top=False,
            labelbottom=show_distance,
            labeltop=False,
            labelsize=label_size,
        )
    elif orientation == "right":
        axes.set_xlim(max_distance, zero_limit)
        axes.set_ylim(-1, len(leaves))
        axes.set_yticks(np.arange(0, len(leaves)))
        axes.set_yticklabels(sorted_labels)
        axes.yaxis.set_tick_params(
            left=False,
            right=False,
            labelleft=True,
            labelright=False,
            labelsize=label_size,
        )
        axes.xaxis.set_tick_params(
            bottom=True,
            top=False,
            labelbottom=show_distance,
            labeltop=False,
            labelsize=label_size,
        )
    elif orientation == "bottom":
        axes.set_ylim(zero_limit, max_distance)
        axes.set_xlim(-1, len(leaves))
        axes.set_xticks(np.arange(0, len(leaves)))
        axes.set_xticklabels(sorted_labels)
        axes.xaxis.set_tick_params(
            bottom=False,
            top=False,
            labelbottom=False,
            labeltop=True,
            labelsize=label_size,
        )
        axes.yaxis.set_tick_params(
            left=True,
            right=False,
            labelleft=show_distance,
            labelright=False,
            labelsize=label_size,
        )
    else:
        axes.set_ylim(max_distance, zero_limit)
        axes.set_xlim(-1, len(leaves))
        axes.set_xticks(np.arange(0, len(leaves)))
        axes.set_xticklabels(sorted_labels)
        axes.xaxis.set_tick_params(
            bottom=False,
            top=False,
            labelbottom=True,
            labeltop=False,
            labelsize=label_size,
        )
        axes.yaxis.set_tick_params(
            left=True,
            right=False,
            labelleft=show_distance,
            labelright=False,
            labelsize=label_size,
        )
    axes.set_frame_on(False)
