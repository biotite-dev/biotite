# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage that provides interfaces for external software in case
*Biotite*’s integrated functionality is not sufficient for your tasks.
These interfaces range from locally installed software
(e.g. MSA software) to web services (e.g. BLAST).
The interfaces are seamless:
Writing input files and reading output files is handled internally.
The user only needs to provide objects like a :class:`Sequence`
and will receive objects like an :class:`Alignment`.

Note that in order to use an interface in :mod:`biotite.application`
the corresponding software must be installed or the web server must be
reachable, respectively.
These programs are not shipped with the *Biotite* package.

Each application is represented by its respective :class:`Application`
class.
Each :class:`Application` instance has a life cycle, starting with its
creation and ending with the result extraction.
Each state in this life cycle is described by the value of the
*enum* :class:`AppState`, that each :class:`Application` contains:
Directly after its instantiation the app is in the ``CREATED`` state.
In this state further parameters can be set for the application run.
After the user calls the :func:`Application.start()` method, the app
state is set to ``RUNNING`` and the app performs the calculations.
When the application finishes the AppState
changes to ``FINISHED``.
The user can now call the :func:`Application.join()` method, concluding
the application in the ``JOINED`` state and making the results of the
application accessible.
Furthermore, this may trigger cleanup actions in some applications.
:func:`Application.join()` can even be called in the ``RUNNING`` state:
This will constantly check if the application has finished and will
directly go into the ``JOINED`` state as soon as the application reaches
the ``FINISHED`` state.
Calling the :func:`Application.cancel()` method while the application is
``RUNNING`` or ``FINISHED`` leaves the application in the ``CANCELLED``
state.
This triggers cleanup, too, but there are no accessible results.
If a method is called in an unsuitable app state, an
:class:`AppStateError` is called.
At each state in the life cycle, :class:`Application` type specific
methods are called, as shown in the following diagram.

.. figure:: /static/assets/figures/app_lifecycle.png
    :alt: Application life cycle
    :scale: 50%

    Taken from
    `Kunzmann & Hamacher 2018 <https://doi.org/10.1186/s12859-018-2367-z>`_
    licensed under `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_.

The execution of an :class:`Application` can run in parallel:
The time between starting the run and collecting the results can be
used to run other code, similar to the *Python* :class:`Thread` or
:class:`Process` classes.
"""

__name__ = "biotite.application"
__author__ = "Patrick Kunzmann"

from .application import *
from .localapp import *
from .msaapp import *
from .webapp import *
