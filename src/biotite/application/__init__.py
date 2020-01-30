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

Each application is represented by its respective :class:`Application`
class.
:class:`Application` objects are created, started and after the run has
finished, the results are collected.
The current state of the the execution is indicated by an
:class:`AppState` object, which restricts which method calls are
allowed:
For example, the parameters can only be set, when the
:class:`Application` has not been started yet and the results can only
be collected after :class:`Application` has finished.

The execution of an :class:`Application` can run in parallel:
In the time between starting the run and collecting the results can be
used to run other code, similar to the *Python* :class:`Thread` or
class:`Process` classes.
"""

__name__ = "biotite.application"
__author__ = "Patrick Kunzmann"

from .application import *
from .localapp import *
from .webapp import *
from .msaapp import *