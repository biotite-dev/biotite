import os
import numpy as np


def setup_script():
    """
    Prepare API keys, formatting, etc. for running a tutorial or example script.
    """
    # Import inside function as Biotite may not be known
    # at the time of function definition
    import biotite.application.blast as blast
    import biotite.database.entrez as entrez

    # Improve readability of large arrays
    np.set_printoptions(precision=2)

    # Use API key to increase request limit
    ncbi_api_key = os.environ.get("NCBI_API_KEY")
    if ncbi_api_key is not None and ncbi_api_key != "":
        entrez.set_api_key(ncbi_api_key)

    # Mock the BlastWebApp class
    # to allow subsequent BLAST calls when building the tutorial
    class MockedBlastApp(blast.BlastWebApp):
        def __init__(self, *args, **kwargs):
            kwargs["obey_rules"] = False
            super().__init__(*args, **kwargs)

    blast.BlastWebApp = MockedBlastApp
