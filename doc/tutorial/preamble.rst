.. jupyter-execute::
    :hide-code:

    import os
    import biotite.database.entrez as entrez

    entrez.set_api_key(os.environ.get("NCBI_API_KEY"))