from pymol import cmd
from os.path import join, isdir


INPUT_STRUCTURE = "glycosylase_oscillation.pdb"
OUTPUT_DIR = "glycosylase_oscillation"


# Load structure and remove water
cmd.load(INPUT_STRUCTURE)

# Add secondary structure
cmd.dss()

# Define colors
cmd.set_color("biotite_lightgreen",    [111/255, 222/255,  76/255])

# Set overall colors
cmd.color("biotite_lightgreen", "chain A")

# Set view
#cmd.set_view((
#    -0.044524662,    0.767611504,    0.639355302,
#     0.998693943,    0.018437184,    0.047413416,
#     0.024606399,    0.640637815,   -0.767439663,
#     0.000000000,    0.000000000, -115.614288330,
#    56.031833649,   23.317802429,    3.761308193,
#    73.517341614,  157.711288452,  -20.000000000 
#))

# Prepare output video frames
cmd.mset()
cmd.set("ray_shadows", 0)
if not isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
cmd.mpng(join(OUTPUT_DIR, "img_"), mode=2, width=1000, height=1000)