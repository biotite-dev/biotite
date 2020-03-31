from pymol import cmd


# Load structure and remove hydrogen
cmd.load("leaflets.pdb")
cmd.remove("elem H")

# Define colors
cmd.set_color("biotite_brightorange", [255.0/255, 181.0/255, 105.0/255])
cmd.set_color("biotite_dimorange",    [220.0/255, 112.0/255,   0.0/255])
cmd.set_color("biotite_lightgreen",   [111.0/255, 222.0/255,  76.0/255])
cmd.set_color("biotite_darkgreen",    [ 56.0/255, 154.0/255,  26.0/255])

# Configure lipid tails
cmd.set("transparency", 0.3, "resn TIP")
cmd.color("gray", "resn TIP")
cmd.color("biotite_lightgreen", "chain A")
cmd.color("biotite_brightorange", "chain B")
cmd.show("sticks", "chain A or chain B")

# Configure lipid heads
cmd.color("biotite_darkgreen", "chain A and name P")
cmd.color("biotite_dimorange", "chain B and name P")
cmd.set("sphere_scale", 1.5)
cmd.show("spheres", "(chain A or chain B) and name P")

# Save image
cmd.ray(1000, 800)
cmd.png("leaflet.png")