"""
Test
====
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import biotite
import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb as rcsb


PDB_ID = "1GYA"

SHEET_DISTANCE = 2.0
ARROW_TAIL_WITH = 0.4
ARROW_HEAD_WITH = 0.7
ARROW_HEAD_LENGTH = 0.25
ARROW_LINE_WIDTH = 1
ARROW_COLORS = [
    biotite.colors["darkgreen"],
    biotite.colors["lightgreen"],
    biotite.colors["dimorange"],
    biotite.colors["brightorange"],
]
CONNECTION_COLOR = "black"
CONNECTION_LINE_WIDTH = 1.5
CONNECTION_HEIGHT = 0.2
CONNECTION_SEPARATION = 0.1
RES_ID_HEIGHT = -0.2
RES_ID_FONT_SIZE = 8


pdbx_file = pdbx.PDBxFile.read(rcsb.fetch(PDB_ID, "pdbx", "."))
sheet_order_dict = pdbx_file["struct_sheet_order"]
sheet_range_dict = pdbx_file["struct_sheet_range"]

unique_sheet_ids = np.unique(sheet_order_dict["sheet_id"]).tolist()
sheet_indices = np.array(
    [unique_sheet_ids.index(sheet) for sheet in sheet_order_dict["sheet_id"]]
)
print(sheet_indices)
print()

adjacent_strands = np.array([
    (strand_i, strand_j) for strand_i, strand_j in zip(
        sheet_order_dict["range_id_1"],
        sheet_order_dict["range_id_2"]
    )
])
print(adjacent_strands)
print()



strand_chain_ids = sheet_range_dict["beg_auth_asym_id"]
strand_res_id_begs = sheet_range_dict["beg_auth_seq_id"].astype(int)
strand_res_id_ends = sheet_range_dict["end_auth_seq_id"].astype(int)

# Secondarily sort by residue ID
order = np.argsort(strand_res_id_begs)
# Primarily sort by chain ID
order = order[np.argsort(strand_chain_ids[order])]

sorted_strand_ids = sheet_range_dict["id"][order]
sorted_sheet_indices = np.array(
    [unique_sheet_ids.index(sheet) for sheet in sheet_range_dict["sheet_id"]]
)[order]
sorted_chain_ids = strand_chain_ids[order]

connections = []
for i in range(len(sorted_strand_ids) -1):
    if sorted_chain_ids[i] == sorted_chain_ids[i+1]:
        connections.append((
            (sorted_sheet_indices[i],   sorted_strand_ids[i]  ),
            (sorted_sheet_indices[i+1], sorted_strand_ids[i+1])
        ))
for strand_i, strand_j in connections:
    print(f"{strand_i} -> {strand_j}")
print()

ranges = {
    (sheet_i, strand_id): (begin, end)
    for sheet_i, strand_id, begin, end
    in zip(
        sorted_sheet_indices, sorted_strand_ids,
        strand_res_id_begs[order], strand_res_id_ends[order]
    )
}
print(ranges)
print()



sheet_graphs = []
for sheet_index in np.unique(sheet_indices):
    sheet_mask = sheet_indices == sheet_index
    sheet_graphs.append(nx.Graph([
        (strand_i, strand_j, {"is_parallel": is_parallel})
        for (strand_i, strand_j), is_parallel in zip(
            adjacent_strands[sheet_mask],
            sheet_order_dict["sense"][sheet_mask] == "parallel"
        )
    ]))

for graph in sheet_graphs:
    initial_strand = adjacent_strands[0,0]
    graph.nodes[initial_strand]["is_upwards"] = True
    for strand in graph.nodes:
        if strand == initial_strand:
            continue
        this_strand_is_upwards = []
        for adj_strand in graph.neighbors(strand):
            is_upwards = graph.nodes[adj_strand].get("is_upwards")
            if is_upwards is None:
                # The arrow direction for this adjacent strand is not
                # yet determined
                continue
            is_parallel = graph.edges[(strand, adj_strand)]["is_parallel"]
            this_strand_is_upwards.append(
                is_upwards ^ ~is_parallel
            )
        if len(this_strand_is_upwards) == 0:
            raise ValueError(
                "Cannot determine arrow direction from adjacent strands"
            )
        elif all(this_strand_is_upwards):
            graph.nodes[strand]["is_upwards"] = True
        elif not any(this_strand_is_upwards):
            graph.nodes[strand]["is_upwards"] = False
        else:
            raise ValueError(
                "Conflicting arrow directions from adjacent strands"
            )



fig, ax = plt.subplots(figsize=(8.0, 4.0))

coord_dict = {}
current_position = 0
for sheet_index, graph in enumerate(sheet_graphs):
    positions = nx.kamada_kawai_layout(graph, dim=1)
    strand_ids = np.array(list(positions.keys()))
    positions = np.array(list(positions.values()))
    # Each position has only one dimension
    # -> Remove the last dimension
    positions = positions[:, 0]
    # Transform positions to achieve a spacing of at least 1.0
    dist_matrix = np.abs(positions[:, np.newaxis] - positions[np.newaxis, :])
    positions /= np.min(dist_matrix[dist_matrix != 0])
    # Transform positions, so that they start at 'current_position'
    positions -= np.min(positions)
    positions += np.min(current_position)
    current_position = np.max(positions) + SHEET_DISTANCE

    for strand_id, pos in zip(strand_ids, positions):
        if graph.nodes[strand_id]["is_upwards"]:
            y = -1
            dy = 2
        else:
            y = 1
            dy = -2
        ax.add_patch(
            FancyArrow(
                x=pos, y=y, dx=0, dy=dy,
                length_includes_head=True,
                width = ARROW_TAIL_WITH,
                head_width = ARROW_HEAD_WITH,
                head_length = ARROW_HEAD_LENGTH,
                facecolor = ARROW_COLORS[sheet_index % len(ARROW_COLORS)],
                edgecolor = CONNECTION_COLOR,
                linewidth = ARROW_LINE_WIDTH,
            )
        )
        # Start and end coordinates of the respective arrow
        coord_dict[sheet_index, strand_id] = ((pos, y), (pos, y + dy))

for i, (strand_i, strand_j) in enumerate(connections):
    horizontal_line_height = 1 + CONNECTION_HEIGHT + i * CONNECTION_SEPARATION
    coord_i_beg, coord_i_end = coord_dict[strand_i]
    coord_j_beg, coord_j_end = coord_dict[strand_j]
    
    if coord_i_end[1] == coord_j_beg[1]:
        x = (
            coord_i_end[0],
            coord_i_end[0],
            coord_j_beg[0],
            coord_j_beg[0]
        )
        y = (
            coord_i_end[1],
            np.sign(coord_i_end[1]) * horizontal_line_height,
            np.sign(coord_j_beg[1]) * horizontal_line_height,
            coord_j_beg[1]
        )
        # Start and end are on the same side of the arrows
    else:
        # Start and end are on different sides
        x = (
            coord_i_end[0],
            coord_i_end[0],
            coord_i_end[0] + 0.5,
            coord_i_end[0] + 0.5,
            coord_j_beg[0],
            coord_j_beg[0]
        )
        y = (
            coord_i_end[1],
            np.sign(coord_i_end[1]) * horizontal_line_height,
            np.sign(coord_i_end[1]) * horizontal_line_height,
            np.sign(coord_j_beg[1]) * horizontal_line_height,
            np.sign(coord_j_beg[1]) * horizontal_line_height,
            coord_j_beg[1]
        )
    ax.plot(
        x, y,
        color = CONNECTION_COLOR,
        linewidth = CONNECTION_LINE_WIDTH,
        # Avoid intersecticoord_i_beg, coord_i_endon of the line's end with the arrow
        solid_capstyle = "butt"
    )

for strand, (res_id_beg, res_id_end) in ranges.items():
    coord_beg, coord_end = coord_dict[strand]
    for coord, res_id in zip((coord_beg, coord_end), (res_id_beg, res_id_end)):
        ax.text(
            coord[0],
            np.sign(coord[1]) * (np.abs(coord[1]) + RES_ID_HEIGHT),
            str(res_id),
            ha="center", va="center", fontsize=RES_ID_FONT_SIZE, weight="bold"
        )

ax.set_xlim(-1, current_position - SHEET_DISTANCE + 1)
ax.set_ylim(-2, 2)
fig.tight_layout()
plt.show()