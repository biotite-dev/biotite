__name__ = "biotite.interface.pymol"
__author__ = "Patrick Kunzmann"
__all__ = [
    "get_and_set_pymol_instance",
    "launch_pymol",
    "launch_interactive_pymol",
    "reset",
    "setup_parameters",
    "DuplicatePyMOLError",
]


_pymol = None


def get_and_set_pymol_instance(pymol_instance=None):
    """
    Get the global *PyMOL* instance.

    This function is intended for internal purposes and should only be
    used for advanced usages.

    Parameters
    ----------
    pymol_instance : module or PyMOL, optional
        If a ``PyMOL`` instance is given here, the global instance is set
        to this instance.
        If *PyMOL* is already running and both instances are not the
        same, an exception is raised.
        By default *PyMOL* is started in library mode, if no *PyMOL*
        instance is currently running.

    Returns
    -------
    pymol_instance : module or PyMOL
        The global ``pymol`` instance.
    """
    global _pymol
    if pymol_instance is None:
        if not is_launched():
            _pymol = launch_pymol()
        return _pymol
    elif _pymol is None:
        if not hasattr(pymol_instance, "cmd"):
            raise TypeError("Given object is not a PyMOL instance")
        _pymol = pymol_instance
    elif _pymol is not pymol_instance:
        # Both the global pymol instance and the given instance are not
        # the same -> duplicate PyMOL instances
        raise DuplicatePyMOLError("A PyMOL instance is already running")
    return _pymol


def is_launched():
    """
    Check whether a *PyMOL* session is already running.

    Returns
    -------
    running : bool
        True, if a *PyMOL* instance is already running, false otherwise.
    """
    return _pymol is not None


def launch_pymol():
    """
    Launch *PyMOL* in object-oriented library mode.

    This is the recommended way to launch *PyMOL* if no GUI is
    required.
    This function simply creates a :class:`PymMOL` object,
    calls its :func:`start()` method and sets up necessary parameters using
    :func:`setup_parameters()`.

    Returns
    -------
    pymol : PyMOL
        The started *PyMOL* instance.
        *PyMOL* commands can be invoked by using its :attr:`cmd` attribute.
    """
    from pymol2 import PyMOL

    global _pymol

    if is_launched():
        raise DuplicatePyMOLError("A PyMOL instance is already running")
    else:
        _pymol = PyMOL()
        _pymol.start()
        setup_parameters(_pymol)
    return _pymol


def launch_interactive_pymol(*args):
    """
    Launch a *PyMOL* GUI with the given command line arguments.

    It starts *PyMOL* by calling :func:`pymol.finish_launching()`,
    reinitializes *PyMOL* to clear the workspace and sets up necessary
    parameters using :func:`setup_parameters()`.

    Parameters
    ----------
    *args : str
        The command line options given to *PyMOL*.

    Returns
    -------
    pymol : module
        The :mod:`pymol` module.
        *PyMOL* commands can be invoked by using its :attr:`cmd`
        attribute.
    """
    import pymol

    global _pymol

    if is_launched():
        if _pymol is not pymol:
            raise DuplicatePyMOLError("PyMOL is already running in library mode")
        else:
            raise DuplicatePyMOLError("A PyMOL instance is already running")
    else:
        pymol.finish_launching(["pymol"] + list(args))
        _pymol = pymol
        pymol.cmd.reinitialize()
        setup_parameters(_pymol)
    return pymol


def reset():
    """
    Delete all objects in the PyMOL workspace and reset parameters to
    defaults.

    If *PyMOL* is not yet running, launch *PyMOL* in object-oriented
    library mode.
    """
    global _pymol

    if not is_launched():
        _pymol = launch_pymol()
    _pymol.cmd.reinitialize()
    setup_parameters(_pymol)


def setup_parameters(pymol_instance):
    """
    Sets *PyMOL* parameters that are necessary for *Biotite* to interact
    properly with *PyMOL*.

    Parameters
    ----------
    pymol_instance : module or SingletonPyMOL or PyMOL, optional
        If *PyMOL* is used in library mode, the :class:`PyMOL`
        or :class:`SingletonPyMOL` object is given here.
        If otherwise *PyMOL* is used in GUI mode, the :mod:`pymol`
        module is given.
    """
    # The selections only work properly,
    # if the order stays the same after adding a model to PyMOL
    pymol_instance.cmd.set("retain_order", 1)
    # Mute the PyMOL messages when rendering movies
    pymol_instance.cmd.feedback("disable", "movie", "everything")


class DuplicatePyMOLError(Exception):
    pass
