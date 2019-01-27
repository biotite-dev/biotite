def rdf(center, atoms, selection=None, range=(0,10), bins=100, box=None,
        periodic=False):
    r"""
    Compute the radial distribution function g(r) for a given point and a selection.

    Parameters
    ----------
    center : ndArray or Atom or AtomArray or AtomArrayStack
        Coordinates or Atoms(s) to use as origin for rdf calculation
    atoms : AtomArray or AtomArrayStack
        Simulation cell to use for rdf calculation. Please not that atoms must
        have an associated box.
    selection : ndarray or None, optional
        Boolean mask for atoms to limit the RDF calculation on specific atoms
        (Default: None).
    range : tuple or None, optional
        The range for the RDF in Angstroem (Default: (0, 10)).
    bins : int or sequence of scalars or str, optional
        Bins for the RDF. If bins is an int, it defines the number of bins for
        given range. If bins is a sequence, it defines the bin edges. If bins is
        a string, it defines the function used to calculate the bins
        (Default: 100).
    box : ndarray, shape=(3,3) or shape=(m,3,3), optional
        If this parameter is set, the given box is used instead of the
        `box` attribute of `atoms`.
    periodic : bool, optional
        Defines if periodic boundary conditions are taken into account
        (Default: False).

    Returns
    -------
    bins : ndarray, dtype=float, shape=n
        The n bin coordinates of the RDF where n is defined by bins
    rdf : ndarry, dtype=float, shape=n
        RDF values for every bin

    Notes
    -----
    Since the RDF depends on the average particle density of the system, this
    function strictly requires an box.

    Examples
    --------
    TODO

    """

    if box is None and atoms.box is None:
        raise ValueError("Please supply a box.")