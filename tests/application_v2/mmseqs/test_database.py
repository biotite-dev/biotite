# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pytest
from biotite.application_v2.mmseqs import (
    AlignmentDatabase,
    FoldseekApp,
    MMseqsApp,
    MSADatabase,
    ProfileDatabase,
    SequenceDatabase,
)

DATABASE_TYPES = [SequenceDatabase, AlignmentDatabase, MSADatabase, ProfileDatabase]


def _random_suffix(rng):
    length = int(rng.integers(0, 9))
    return "".join(rng.choice(list("_abcdefghijklmnopqrstuvwxyz"), size=length))


def _make_database(database_type, app, suffix):
    if database_type is AlignmentDatabase:
        query_db = SequenceDatabase(app)
        target_db = SequenceDatabase(app)
        return AlignmentDatabase(app, query_db, target_db, suffix=suffix)
    return database_type(app, suffix=suffix)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("database_type", DATABASE_TYPES)
def test_copy_preserves_database_properties(database_type, seed):
    """
    Copying any database type preserves its type, content, compatibility and metadata.
    """
    rng = np.random.default_rng(seed)
    suffix = _random_suffix(rng)
    content = rng.bytes(int(rng.integers(0, 257)))
    app = MMseqsApp("true")
    database = _make_database(database_type, app, suffix)
    database.name.write_bytes(content)

    persistent_directory = TemporaryDirectory()
    destination = Path(persistent_directory.name) / "copied"
    copied = database.copy(destination)

    assert type(copied) is database_type
    assert copied.name.read_bytes() == content
    assert copied.name.name == database.name.name
    assert copied.is_compatible_with(app)
    assert not copied.is_compatible_with(FoldseekApp("true"))
    if isinstance(database, AlignmentDatabase):
        assert copied.query_db is database.query_db
        assert copied.target_db is database.target_db


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("database_type", DATABASE_TYPES)
def test_move_preserves_database_properties(database_type, seed):
    """
    Moving any database type changes only its storage location and persistence.
    """
    rng = np.random.default_rng(seed)
    suffix = _random_suffix(rng)
    content = rng.bytes(int(rng.integers(0, 257)))
    app = MMseqsApp("true")
    database = _make_database(database_type, app, suffix)
    old_path = database.path
    database.name.write_bytes(content)
    persistent_directory = TemporaryDirectory()
    destination = Path(persistent_directory.name) / "moved"

    database.move(destination)

    assert database.path == destination
    assert database.name.read_bytes() == content
    assert not old_path.exists()
    assert database.is_compatible_with(app)
    if isinstance(database, AlignmentDatabase):
        assert isinstance(database.query_db, SequenceDatabase)
        assert isinstance(database.target_db, SequenceDatabase)

    del database
    assert destination.exists()


@pytest.mark.parametrize("force", [False, True])
def test_existing_destination(force):
    """
    A populated destination is replaced exactly when `force` is true.
    """
    app = MMseqsApp("true")
    database = SequenceDatabase(app)
    database.name.touch()
    persistent_directory = TemporaryDirectory()
    destination = Path(persistent_directory.name) / "destination"
    destination.mkdir()
    (destination / "existing").touch()

    if force:
        copied = database.copy(destination, force=True)
        assert copied.name.exists()
        assert not (destination / "existing").exists()
    else:
        with pytest.raises(FileExistsError):
            database.copy(destination)


def test_temporary_database_cleanup():
    """
    A temporary database directory follows the lifetime of its result object.
    """
    database = SequenceDatabase(MMseqsApp("true"))
    path = database.path
    assert path.exists()
    del database
    assert not path.exists()


def test_parent_keeps_temporary_database_alive():
    """
    A database view keeps the owner of its shared temporary directory alive.
    """
    app = MMseqsApp("true")
    parent = SequenceDatabase(app)
    path = parent.path
    child = SequenceDatabase(app, path, suffix="_child")
    child.set_parent(parent)

    del parent
    assert path.exists()

    del child
    assert not path.exists()
