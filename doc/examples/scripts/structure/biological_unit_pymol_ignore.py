from pymol import cmd


# Load structure and remove hydrogen
cmd.load("biological_unit.pdb")

# Define colors
cmd.set_color("biotite_brightorange", [255.0/255, 181.0/255, 105.0/255])
cmd.set_color("biotite_dimorange",    [220.0/255, 112.0/255,   0.0/255])
cmd.set_color("biotite_lightgreen",   [111.0/255, 222.0/255,  76.0/255])
cmd.set_color("biotite_darkgreen",    [ 56.0/255, 154.0/255,  26.0/255])

cmd.show("spheres")
cmd.color("biotite_lightgreen", "chain A")
cmd.color("biotite_brightorange", "chain B")
cmd.color("gray", "chain C")


# Save image
cmd.ray(1000, 1000)
cmd.png("biological_unit.png")