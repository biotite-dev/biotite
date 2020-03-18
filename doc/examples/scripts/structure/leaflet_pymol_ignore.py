from pymol import cmd


# Load structure and remove water
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

# Set view
#cmd.set_view((
#    -0.044524662,    0.767611504,    0.639355302,
#     0.998693943,    0.018437184,    0.047413416,
#     0.024606399,    0.640637815,   -0.767439663,
#     0.000000000,    0.000000000, -115.614288330,
#    56.031833649,   23.317802429,    3.761308193,
#    73.517341614,  157.711288452,  -20.000000000 
#))

# Save image
cmd.ray(1000, 500)
cmd.png("leaflet.png")