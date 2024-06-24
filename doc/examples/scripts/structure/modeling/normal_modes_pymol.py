from pymol import cmd
from os.path import join, isdir


INPUT_STRUCTURE = "normal_modes.pdb"
OUTPUT_DIR = "normal_modes"


# Load structure and remove water
cmd.load(INPUT_STRUCTURE)

# Add secondary structure
cmd.dss()

# Define colors
cmd.set_color("biotite_lightgreen",    [111/255, 222/255,  76/255])

# Set overall colors
cmd.color("biotite_lightgreen", "chain A")

# Set view
cmd.set_view((
     0.605540633,    0.363677770,   -0.707855821,
    -0.416691631,    0.902691007,    0.107316799,
     0.678002179,    0.229972601,    0.698157668,
     0.000000000,    0.000000000, -115.912551880,
    32.098876953,   31.005725861,   78.377349854,
    89.280677795,  142.544403076,  -20.000000000
))

# Prepare output video frames
cmd.mset()
cmd.set("ray_shadows", 0)
if not isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
cmd.mpng(join(OUTPUT_DIR, "img_"), mode=2, width=600, height=600)


# Render animated GIF
# convert -delay 3 -loop 0 -dispose 2 normal_modes/*.png normal_modes.gif
