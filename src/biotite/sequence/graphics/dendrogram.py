# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.sequence.graphics"
__author__ = "Patrick Kunzmann"
__all__ = ["plot_dendrogram"]

from typing import Any, Literal
import numpy as np
from matplotlib.axes import Axes
from biotite.sequence.phylo.tree import Tree, TreeNode
from biotite.typing import MplColor


def plot_dendrogram(
    axes: Axes,
    tree: Tree,
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
    tree : Tree
        The tree to be visualized.
    orientation : {'left', 'right', 'bottom', 'top'}, optional
        The position of the root node in the plot
    use_distances : bool, optional
        If true, the `distance` attribute of the :class:`TreeNode`
        objects are used as distance measure.
        Otherwise the topological distance is used.
    labels : list of str, optional
        The leaf node labels.
        The label of a leaf node is the entry at the position of its
        `index` attribute.
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

    indices = tree.root.get_indices()
    leaf_dict: dict[int, int] = {int(indices[i]): int(i) for i in indices}

    # Required for setting the plot limits
    max_distance = 0

    def _plot_node(node: TreeNode, distance: float) -> float:
        """
        Draw the lines from the given node to its children.

        Parameters
        ----------
        dist : float
            the distance of the node from root

        Returns
        -------
        pos : float
            the position of the node on the 'label' axis
        """
        # The term 'distance'
        # refers to positions along the 'distance' axis
        # the term 'pos'
        # refers to positions along the other axis
        nonlocal max_distance
        if max_distance < distance:
            max_distance = distance
        if node.is_leaf():
            # No children -> no line can be drawn
            if node.index is None:
                raise ValueError("Leaf node has no index")
            return leaf_dict[node.index]
        else:
            children = node.children
            if use_distances:
                child_distances = [distance + c.distance for c in children]
            else:
                # Use topologic distance of children to this node,
                # which is always 1
                child_distances = [distance + 1 for c in children]
            child_pos = [
                _plot_node(child, child_distance)
                for child, child_distance in zip(children, child_distances)
            ]
            # Position of this node is in the center of the child nodes
            center_pos = sum(child_pos) / len(child_pos)
            if orientation in ["left", "right"]:
                # Line connecting the childs
                axes.plot(
                    [distance, distance],
                    [child_pos[0], child_pos[-1]],
                    color=color,
                    marker="None",
                    **kwargs,
                )
                # Lines depicting the distances of the childs
                for child_dist, pos in zip(child_distances, child_pos):
                    axes.plot(
                        [distance, child_dist],
                        [pos, pos],
                        color=color,
                        marker="None",
                        **kwargs,
                    )
            elif orientation in ["bottom", "top"]:
                # Line connecting the childs
                axes.plot(
                    [child_pos[0], child_pos[-1]],
                    [distance, distance],
                    color=color,
                    marker="None",
                    **kwargs,
                )
                # Lines depicting the distances of the childs
                for child_dist, pos in zip(child_distances, child_pos):
                    axes.plot(
                        [pos, pos],
                        [distance, child_dist],
                        color=color,
                        marker="None",
                        **kwargs,
                    )
            else:
                raise ValueError(f"'{orientation}' is not a valid orientation")
            return center_pos

    _plot_node(tree.root, 0)

    sorted_labels: list[str]
    if labels is not None:
        # Sort labels using the order of indices in the tree
        # A list cannot be directly indexed with a list,
        # hence the conversion to a ndarray
        sorted_labels = np.array(labels)[indices].tolist()
    else:
        sorted_labels = [str(i) for i in indices]
    # The distance axis does not start at 0,
    # since the root line would not properly rendered
    # Hence the limit is set a to small fraction of the entire axis
    # beyond 0
    zero_limit = -0.01 * max_distance
    if orientation == "left":
        axes.set_xlim(zero_limit, max_distance)
        axes.set_ylim(-1, len(indices))
        axes.set_yticks(np.arange(0, len(indices)))
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
        axes.set_ylim(-1, len(indices))
        axes.set_yticks(np.arange(0, len(indices)))
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
        axes.set_xlim(-1, len(indices))
        axes.set_xticks(np.arange(0, len(indices)))
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
    elif orientation == "top":
        axes.set_ylim(max_distance, zero_limit)
        axes.set_xlim(-1, len(indices))
        axes.set_xticks(np.arange(0, len(indices)))
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
    else:
        raise ValueError(f"'{orientation}' is not a valid orientation")
    axes.set_frame_on(False)
