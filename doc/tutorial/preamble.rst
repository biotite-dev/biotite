.. jupyter-execute::
    :hide-code:

    import os
    import numpy as np
    import biotite.database.entrez as entrez
    import biotite.application.blast as blast

    # Improve readability of large arrays
    np.set_printoptions(precision=2)

    # Use API key to increase request limit
    entrez.set_api_key(os.environ.get("NCBI_API_KEY"))

    # Mock the BlastWebApp class
    # to allow subsequent BLAST calls when building the tutorial
    class MockedBlastApp(blast.BlastWebApp):
        def __init__(self, *args, **kwargs):
            kwargs["obey_rules"] = False
            super().__init__(*args, **kwargs)
    blast.BlastWebApp = MockedBlastApp