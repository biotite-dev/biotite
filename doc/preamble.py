import os
import numpy as np
from matplotlib.colors import to_rgb

# Colors are based on Lab color space at L = 50
RED = to_rgb("#db3a35")
GREEN = to_rgb("#088b05")
BLUE = to_rgb("#1772f0")
VIOLET = to_rgb("#cb38aa")
GRAY = to_rgb("#767676")


def setup_script(*args, **kwargs):
    """
    Prepare API keys, formatting, etc. for running a tutorial or example script.
    """
    # Import inside function as Biotite may not be known
    # at the time of function definition
    import biotite
    import biotite.application.blast as blast
    import biotite.database.entrez as entrez
    import biotite.interface.pymol as pymol_interface

    # Improve readability of large arrays
    np.set_printoptions(precision=2)

    # Use API key to increase request limit
    ncbi_api_key = os.environ.get("NCBI_API_KEY")
    if ncbi_api_key is not None and ncbi_api_key != "":
        entrez.set_api_key(ncbi_api_key)

    # Reset PyMOL canvas
    pymol_interface.reset()
    # Set style of PyMOL images
    pymol_interface.cmd.bg_color("white")
    pymol_interface.cmd.set("depth_cue", 0)
    pymol_interface.cmd.set("ray_shadows", 0)
    pymol_interface.cmd.set("spec_reflect", 0)
    pymol_interface.cmd.set("ray_trace_mode", 1)
    pymol_interface.cmd.set("ray_trace_disco_factor", 1)
    pymol_interface.cmd.set("cartoon_side_chain_helper", 1)
    pymol_interface.cmd.set("valence", 0)
    pymol_interface.cmd.set("cartoon_oval_length", 1.0)
    pymol_interface.cmd.set("label_color", "black")
    pymol_interface.cmd.set("label_size", 30)
    pymol_interface.cmd.set("dash_gap", 0.3)
    pymol_interface.cmd.set("dash_width", 2.0)
    pymol_interface.cmd.set_color("red", RED)
    pymol_interface.cmd.set_color("green", GREEN)
    pymol_interface.cmd.set_color("blue", BLUE)
    pymol_interface.cmd.set_color("violet", VIOLET)
    pymol_interface.cmd.set_color("gray", GRAY)
    pymol_interface.cmd.set_color("carbon", GRAY)
    pymol_interface.cmd.set_color("oxygen", RED)
    pymol_interface.cmd.set_color("nitrogen", BLUE)
    # Expose the Biotite colors with the 'biotite_' prefix
    for color_name, color_value in biotite.colors.items():
        pymol_interface.cmd.set_color("biotite_" + color_name, to_rgb(color_value))

    # Mock the BlastWebApp class
    # to allow subsequent BLAST calls when building the tutorial
    class MockedBlastApp(blast.BlastWebApp):
        def __init__(self, *args, **kwargs):
            kwargs["obey_rules"] = False
            super().__init__(*args, **kwargs)

    blast.BlastWebApp = MockedBlastApp
