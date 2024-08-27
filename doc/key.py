import os


def set_ncbi_api_key_from_env(*args, **kwargs):
    # Import inside function as Biotite may not be known
    # at the time of function definition
    import biotite.database.entrez as entrez

    ncbi_api_key = os.environ.get("NCBI_API_KEY")
    if ncbi_api_key is not None and ncbi_api_key != "":
        entrez.set_api_key(ncbi_api_key)
