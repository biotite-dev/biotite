# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.application_v2.mmseqs"
__author__ = "Patrick Kunzmann"
__all__ = [
    "Database",
    "SequenceDatabase",
    "AlignmentDatabase",
    "MSADatabase",
    "ProfileDatabase",
    "ClusterDatabase",
]

import copy
import shutil
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Generic, Self, TypeGuard, TypeVar

if TYPE_CHECKING:
    from biotite.application_v2.mmseqs.app import MMseqsLikeApp

T = TypeVar("T", bound="MMseqsLikeApp")


class Database(Generic[T]):
    """
    A database produced by *MMseqs2* or *Foldseek*.

    A database is represented by a directory containing exactly one database
    prefix. If no path is given, the directory is temporary and is removed when
    this object is discarded. Use :meth:`copy()` or :meth:`move()` to make it
    persistent.

    Parameters
    ----------
    app : LocalApp
        The application the database is compatible with.
    path : path-like, optional
        The database directory. By default, a temporary directory is created.
    suffix : str, optional
        A suffix appended to the database prefix.

    Attributes
    ----------
    path : Path
        The database directory.
    name : Path
        The complete database prefix passed to the command line program.
    """

    def __init__(
        self,
        app: T,
        path: PathLike[str] | str | None = None,
        suffix: str = "",
    ) -> None:
        self._app_type = type(app)
        self._suffix = suffix
        self._parent: Database[T] | None = None
        if path is None:
            self._temporary_directory = TemporaryDirectory(prefix=f"{app.path.name}_")
            self._path = Path(self._temporary_directory.name)
        else:
            self._temporary_directory = None
            self._path = Path(path)
            self._path.mkdir(exist_ok=True, parents=True)

    @property
    def path(self) -> Path:
        """
        The database directory.

        Returns
        -------
        path : Path
            The database directory.
        """
        return self._path

    @property
    def name(self) -> Path:
        """
        The complete database prefix passed to the command line program.

        Returns
        -------
        name : Path
            The complete database prefix.
        """
        return self._path / f"database{self._suffix}"

    def set_parent(self, parent: Database[T]) -> None:
        """
        Keep the owner of this database's directory alive.

        Some databases are views of files in another database's directory.
        If that directory is temporary, retaining its owner prevents premature
        deletion while the view is still in use.

        Parameters
        ----------
        parent : Database
            The database that owns the shared directory.
        """
        self._parent = parent

    def copy(self, destination: PathLike[str] | str, force: bool = False) -> Self:
        """
        Copy the database into a persistent directory.

        Parameters
        ----------
        destination : path-like
            The destination directory.
        force : bool, optional
            If true, replace a non-empty destination.

        Returns
        -------
        database : Database
            A persistent copy with the same concrete type and metadata.
        """
        destination = Path(destination)
        _prepare_destination(destination, force)
        shutil.copytree(self._path, destination)
        clone = copy.copy(self)
        clone._path = destination
        clone._temporary_directory = None
        return clone

    def move(self, destination: PathLike[str] | str, force: bool = False) -> None:
        """
        Move the database into a persistent directory.

        Parameters
        ----------
        destination : path-like
            The destination directory.
        force : bool, optional
            If true, replace a non-empty destination.
        """
        destination = Path(destination)
        _prepare_destination(destination, force)
        source = self._path
        source.rename(destination)
        if self._temporary_directory is not None:
            # The directory was moved, so cleanup only detaches the finalizer.
            self._temporary_directory.cleanup()
        self._temporary_directory = None
        self._path = destination

    def is_compatible_with(self, app: MMseqsLikeApp) -> TypeGuard[T]:
        """
        Check whether the database can be used by the given application.

        Parameters
        ----------
        app : MMseqsApp or FoldseekApp
            The application to check.

        Returns
        -------
        compatible : bool
            True if the database and application have the same concrete type.
        """
        return self._app_type is type(app)


class SequenceDatabase(Database[T]):
    """
    An *MMseqs2* or *Foldseek* sequence database.
    """


class AlignmentDatabase(Database[T]):
    """
    A database of pairwise alignment results.

    Parameters
    ----------
    app : LocalApp
        The application the database is compatible with.
    query_db : SequenceDatabase or ProfileDatabase
        The query database used to create the alignments.
    target_db : SequenceDatabase
        The target database used to create the alignments.
    path : path-like, optional
        The database directory. By default, a temporary directory is created.
    suffix : str, optional
        A suffix appended to the database prefix.
    """

    def __init__(
        self,
        app: T,
        query_db: SequenceDatabase[T] | ProfileDatabase[T],
        target_db: SequenceDatabase[T],
        path: PathLike[str] | str | None = None,
        suffix: str = "",
    ) -> None:
        super().__init__(app, path, suffix)
        self._query_db = query_db
        self._target_db = target_db

    @property
    def query_db(self) -> SequenceDatabase[T] | ProfileDatabase[T]:
        """
        The query database used to create the alignments.

        Returns
        -------
        query_db : SequenceDatabase or ProfileDatabase
            The query database.
        """
        return self._query_db

    @property
    def target_db(self) -> SequenceDatabase[T]:
        """
        The target database used to create the alignments.

        Returns
        -------
        target_db : SequenceDatabase
            The target database.
        """
        return self._target_db


class MSADatabase(Database[T]):
    """
    A database of multiple sequence alignments.
    """


class ProfileDatabase(Database[T]):
    """
    A database of sequence profiles.

    A profile database can be used in place of a :class:`SequenceDatabase` as
    query in :meth:`MMseqsApp.search()`, which turns the search into a more
    sensitive profile-to-sequence search.
    """


class ClusterDatabase(Database[T]):
    """
    A database of sequence clusters.

    Each entry is keyed by the representative sequence of a cluster and lists
    its members.

    Parameters
    ----------
    app : LocalApp
        The application the database is compatible with.
    sequence_db : SequenceDatabase
        The sequence database that was clustered.
    path : path-like, optional
        The database directory. By default, a temporary directory is created.
    suffix : str, optional
        A suffix appended to the database prefix.
    """

    def __init__(
        self,
        app: T,
        sequence_db: SequenceDatabase[T],
        path: PathLike[str] | str | None = None,
        suffix: str = "",
    ) -> None:
        super().__init__(app, path, suffix)
        self._sequence_db = sequence_db

    @property
    def sequence_db(self) -> SequenceDatabase[T]:
        """
        The sequence database that was clustered.

        Returns
        -------
        sequence_db : SequenceDatabase
            The clustered sequence database.
        """
        return self._sequence_db


def _prepare_destination(destination: Path, force: bool) -> None:
    """
    Ensure that `destination` can be populated.
    """
    if destination.exists():
        is_empty_directory = destination.is_dir() and not any(destination.iterdir())
        if not force and not is_empty_directory:
            raise FileExistsError(f"'{destination}' already exists")
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(exist_ok=True, parents=True)
