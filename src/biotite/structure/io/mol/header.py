# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.mol"
__author__ = "Patrick Kunzmann"
__all__ = ["Header"]

import datetime
import warnings
from dataclasses import dataclass

_DATE_FORMAT = "%m%d%y%H%M"


@dataclass
class Header:
    """
    The header for connection tables.

    Parameters
    ----------
    mol_name : str, optional
        The name of the molecule.
    initials : str, optional
        The author's initials. Maximum length is 2.
    program : str, optional
        The program name. Maximum length is 8.
    time : datetime or date, optional
        The time of file creation.
    dimensions : str, optional
        Dimensional codes. Maximum length is 2.
    scaling_factors : str, optional
        Scaling factors. Maximum length is 12.
    energy : str, optional
        Energy from modeling program. Maximum length is 12.
    registry_number : str, optional
        MDL registry number. Maximum length is 6.
    comments : str, optional
        Additional comments.

    Attributes
    ----------
    mol_name, initials, program, time, dimensions, scaling_factors, energy, registry_number, comments
        Same as the parameters.
    """

    mol_name: ... = ""
    initials: ... = ""
    program: ... = ""
    time: ... = None
    dimensions: ... = ""
    scaling_factors: ... = ""
    energy: ... = ""
    registry_number: ... = ""
    comments: ... = ""

    @staticmethod
    def deserialize(text):
        lines = text.splitlines()

        mol_name = lines[0].strip()
        initials = lines[1][0:2].strip()
        program = lines[1][2:10].strip()
        time_string = lines[1][10:20]
        if time_string.strip() == "":
            time = None
        else:
            try:
                time = datetime.datetime.strptime(time_string, _DATE_FORMAT)
            except ValueError:
                warnings.warn(f"Invalid time format '{time_string}' in file header")
                time = None
        dimensions = lines[1][20:22].strip()
        scaling_factors = lines[1][22:34].strip()
        energy = lines[1][34:46].strip()
        registry_number = lines[1][46:52].strip()

        comments = lines[2].strip()

        return Header(
            mol_name,
            initials,
            program,
            time,
            dimensions,
            scaling_factors,
            energy,
            registry_number,
            comments,
        )

    def serialize(self):
        text = ""

        if self.time is None:
            time_str = ""
        else:
            time_str = self.time.strftime(_DATE_FORMAT)

        if len(self.mol_name) > 80:
            raise ValueError("Molecule name must not exceed 80 characters")
        text += str(self.mol_name) + "\n"
        # Fixed columns -> minimum and maximum length is the same
        # Shorter values are padded, longer values are truncated
        text += (
            f"{self.initials:>2.2}"
            f"{self.program:>8.8}"
            f"{time_str:>10.10}"
            f"{self.dimensions:>2.2}"
            f"{self.scaling_factors:>12.12}"
            f"{self.energy:>12.12}"
            f"{self.registry_number:>6.6}"
            "\n"
        )
        text += str(self.comments) + "\n"
        return text

    def __str__(self):
        return self.serialize()
