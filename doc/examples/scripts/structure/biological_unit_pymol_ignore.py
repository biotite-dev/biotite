from pymol import cmd


cmd.load("biological_unit.pdb")

cmd.show("spheres")
cmd.set("sphere_scale", 1.5)
cmd.set_color("biotite_brightorange", [255.0/255, 181.0/255, 105.0/255])
cmd.set_color("biotite_lightgreen",   [111.0/255, 222.0/255,  76.0/255])
cmd.color("biotite_lightgreen", "chain A")
cmd.color("biotite_brightorange", "chain B")
cmd.color("gray", "chain C")

# Due to a bug, the first call does not produce a valid PNG file
cmd.png("biological_unit.png", 1000, 1000)
cmd.png("biological_unit.png")