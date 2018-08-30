r"""
Four ways to get the secondary structure of a protein
=====================================================

In this example, we will obtain the secondary structure of the
transketolase crystal structure (PDB: 1QGD) in four different ways and
visualize it using a customized feature map.

At first, we will write draw functions for visualization of helices and
sheets in feature maps.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import biotite
import biotite.structure as struc
import biotite.structure.io.mmtf as mmtf
import biotite.sequence as seq
import biotite.sequence.graphics as graphics
import biotite.sequence.io.genbank as gb
import biotite.database.rcsb as rcsb
import biotite.database.entrez as entrez
import biotite.application.dssp as dssp

# All 'FeatureMap' draw functions have the signature
# draw(feature, x, y, width, height, figure, loc_index, style_dict)
def draw_secondary_strucure(feature, x, y, width, height,
                            figure, loc_index, style_dict):
    if feature.qual["sec_str_type"] == "helix":
        _draw_helix(
            feature, x, y, width, height, figure, loc_index, style_dict
        )
    if feature.qual["sec_str_type"] == "sheet":
        _draw_sheet(
            feature, x, y, width, height, figure, loc_index, style_dict
        )

def _draw_helix(feature, x, y, width, height, figure, loc_index, style_dict):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    loc = feature.locs[loc_index]
    # Approx. 1 turn per 3.6 residues to resemble natural helix
    n_turns = np.ceil((loc.last - loc.first + 1) / 3.6)
    x_val = np.linspace(0, n_turns * 2*np.pi, 100)
    # Curve ranges from 0.3 to 0.7
    y_val = (-0.4*np.sin(x_val) + 1) / 2
    
    # Transform values for correct location in feature map
    x_val *= width / (n_turns * 2*np.pi)
    x_val += x
    y_val *= height
    y_val += y
    
    # Draw white background to overlay the guiding line
    background = Rectangle((x,y), width, height, color="white", linewidth=0)
    figure.patches.append(background)
    helix = Line2D(
        x_val, y_val, linewidth=2, color=biotite.colors["dimgreen"]
    )
    figure.lines.append(helix)

def _draw_sheet(feature, x, y, width, height, figure, loc_index, style_dict):
    from matplotlib.patches import FancyArrow

    head_height = 0.8*height
    tail_height = 0.5*height
    head_width = 0.4*height

    tail_x = x
    arrow_y = y + height/2
    dx = width
    dy = 0
    if head_width > width:
        # If fteaure is to short, draw only narrowed head
        head_width = width
    loc = feature.locs[loc_index]
    if loc.defect & seq.Location.Defect.MISS_RIGHT:
            head_width = 0
            head_height = tail_height

    arrow = FancyArrow(tail_x, arrow_y, dx, dy,
                       width=tail_height, head_width=head_height,
                       head_length=head_width, length_includes_head=True,
                       color=biotite.colors["orange"], linewidth=0)
    figure.patches.append(arrow)

# Test our drawing functions with example annotation
annotation = seq.Annotation([
    seq.Feature("SecStr", [seq.Location(10, 40)], {"sec_str_type" : "helix"}),
    seq.Feature("SecStr", [seq.Location(60, 90)], {"sec_str_type" : "sheet"}),
])
feature_map = graphics.FeatureMap(
    annotation, loc_range=(1,100), multi_line=False
)
feature_map.drawfunc["SecStr"] = draw_secondary_strucure
figure = feature_map.generate()

########################################################################
# Now let us do some serious application.
# We want to visualize the secondary structure of one monomer of the
# homodimeric transketolase (PDB: 1QGD).
# The simplest way to do that, is to fetch the corresponding GenBank
# file, extract an `Annotation` object from the file and draw the
# annotation.

# Fetch GenBank files of the TK's first chain and extract annotatation
file_name = entrez.fetch("1QGD_A", biotite.temp_dir(), "gb", "protein", "gb")
gb_file = gb.GenBankFile()
gb_file.read(file_name)
annotation = gb_file.get_annotation(include_only=["SecStr"])
# Length of the sequence
length = int(gb_file.get_locus()["length"])
# 'loc_range' takes exclusive stop -> length+1 is required
feature_map = graphics.FeatureMap(
    annotation, line_length=150, loc_range=(1,length+1)
)
feature_map.add_location_numbers(size=50)
feature_map.drawfunc["SecStr"] = draw_secondary_strucure
figure = feature_map.generate()

########################################################################
# Another (more complicated) approach is the creation of an `Annotation`
# containing the secondary structure from a structure file.
# All file formats distributed by the *RCSB PDB* contain this
# information, but it is most easily extracted from the
# ``'secStructList'`` field in MMTF files.
# Since the two sources use different means of secondary structure
# calculation, the results will differ from each other.

# Dictionary to convert 'secStructList' codes to DSSP values
# https://github.com/rcsb/mmtf/blob/master/spec.md#secstructlist
sec_struct_codes = {0 : "I",
                    1 : "S",
                    2 : "H",
                    3 : "E",
                    4 : "G",
                    5 : "B",
                    6 : "T",
                    7 : "C"}
# Converter for the DSSP secondary structure elements
# to the classical ones
dssp_to_abc = {"I" : "c",
               "S" : "c",
               "H" : "a",
               "E" : "b",
               "G" : "c",
               "B" : "b",
               "T" : "c",
               "C" : "c"}


# Fetch and load structure
file_name = rcsb.fetch("1QGD", "mmtf", biotite.temp_dir())
mmtf_file = mmtf.MMTFFile()
mmtf_file.read(file_name)
array = mmtf.get_structure(mmtf_file, model=1)
# Transketolase homodimer
tk_dimer = array[struc.filter_amino_acids(array)]
# Transketolase monomer
tk_mono = tk_dimer[tk_dimer.chain_id == "A"]

# The chain ID corresponding to each residue
chain_id_per_res = array.chain_id[struc.get_residue_starts(tk_dimer)]
sse = mmtf_file["secStructList"]
sse = sse[sse != -1]
sse = sse[chain_id_per_res == "A"]
sse = np.array([sec_struct_codes[code] for code in sse if code != -1],
               dtype="U1")
sse = np.array([dssp_to_abc[e] for e in sse], dtype="U1")

# Helper function to convert secondary structure array to annotation
# and visualize it
def visualize_secondary_structure(sse, first_id):
    
    def _add_sec_str(annotation, first, last, str_type):
        if str_type == "a":
            str_type = "helix"
        elif str_type == "b":
            str_type = "sheet"
        else:
            # coil
            return
        feature = seq.Feature(
            "SecStr", [seq.Location(first, last)], {"sec_str_type" : str_type}
        )
        annotation.add_feature(feature)
    
    # Find the intervals for each secondary structure element
    # and add to annotation
    annotation = seq.Annotation()
    curr_sse = None
    curr_start = None
    for i in range(len(sse)):
        if curr_start is None:
            curr_start = i
            curr_sse = sse[i]
        else:
            if sse[i] != sse[i-1]:
                _add_sec_str(
                    annotation, curr_start+first_id, i-1+first_id, curr_sse
                )
                curr_start = i
                curr_sse = sse[i]
    # Add last secondary structure element to annotation
    _add_sec_str(annotation, curr_start+first_id, i-1+first_id, curr_sse)
    
    feature_map = graphics.FeatureMap(
        annotation, line_length=150, loc_range=(1,length+1)
    )
    feature_map.add_location_numbers(size=50)
    feature_map.drawfunc["SecStr"] = draw_secondary_strucure
    return feature_map.generate()

# Visualize seconday structure array
# Sine the residues may not start at 1,
# provide the actual first residue ID
visualize_secondary_structure(sse, tk_mono.res_id[0])

########################################################################
# Almost the same result can be achieved, when we calculate the
# secondary structure ourselves using the DSSP software,
# as the content in ``'secStructList'`` is also calculated by the RCSB.

sse = dssp.DsspApp.annotate_sse(tk_mono)
sse = np.array([dssp_to_abc[e] for e in sse], dtype="U1")
visualize_secondary_structure(sse, 2)
# sphinx_gallery_thumbnail_number = 4

########################################################################
# The one and only difference is that the second helix is slightly
# shorter.
# This is probably caused by different versions of DSSP.
# 
# Last but not least we calculate the secondary structure using
# *Biotite*'s built-in method, based on the P-SEA algorithm.

sse = struc.annotate_sse(array, chain_id="A")
visualize_secondary_structure(sse, 2)

plt.show()