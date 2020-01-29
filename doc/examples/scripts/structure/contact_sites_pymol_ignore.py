from pymol import cmd


# Load structure and remove water
cmd.fetch("2or1")
cmd.remove("resn HOH")

# Define colors
cmd.set_color("biotite_brightorange", [255/255, 181/255, 105/255])
cmd.set_color("biotite_dimorange",    [220/255, 112/255,   0/255])
cmd.set_color("biotite_lightgreen",   [111/255, 222/255,  76/255])
cmd.set_color("biotite_darkgreen",    [ 56/255, 154/255,  26/255])

# Set overall colors
cmd.color("gray", "chain A or chain B")
cmd.color("biotite_brightorange", "chain L")
cmd.color("biotite_lightgreen", "chain R")

# Set view
cmd.set_view((
    -0.044524662,    0.767611504,    0.639355302,
     0.998693943,    0.018437184,    0.047413416,
     0.024606399,    0.640637815,   -0.767439663,
     0.000000000,    0.000000000, -115.614288330,
    56.031833649,   23.317802429,    3.761308193,
    73.517341614,  157.711288452,  -20.000000000 
))

# Highlight contacts
for res_id in [16, 17, 28, 29, 32, 33, 36, 38, 39, 40, 41, 42, 43, 44]:
    cmd.show("sticks", f"(chain L or chain R) and resi {res_id}")
    cmd.color(
        "biotite_dimorange", f"chain L and resi {res_id} and (not name CA)"
    )
    cmd.color(
        "biotite_darkgreen", f"chain R and resi {res_id} and (not name CA)"
    )

# Save image
cmd.ray(1000, 550)
cmd.png("contact_sites.png")