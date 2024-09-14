# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

import abc
import enum
import functools
import struct
import typing

import numpy
import numpy.ma

from . import _unkerasify
from .layers import Layer, CentroidLayer, Model
from .utils import normalize, last

try:
    from importlib.resources import files as resource_files
except ImportError:
    from importlib_resources import files as resource_files

T = typing.TypeVar("T")
if typing.TYPE_CHECKING:
    from Bio.PDB import Chain
    from .utils import ArrayN, ArrayNx2, ArrayNx3, ArrayNx10, ArrayNxM

    try:
        from typing import Literal
    except ImportError:
        from typing_extensions import Literal


DISTANCE_ALPHA_BETA = 1.5336
ALPHABET = numpy.array(list("ACDEFGHIKLMNPQRSTVWYX"))


class _BaseEncoder(abc.ABC, typing.Generic[T]):
    @abc.abstractmethod
    def encode_atoms(
        self,
        ca: ArrayNx3[numpy.floating],
        cb: ArrayNx3[numpy.floating],
        n: ArrayNx3[numpy.floating],
        c: ArrayNx3[numpy.floating],
    ) -> T:
        """Encode the given atom coordinates to a different representation.

        Arguments:
            ca (`numpy.ndarray` of shape :math:`(N, 3)`): The coordinates of
                the *Cα* atom for each residue.
            cb (`numpy.ndarray` of shape :math:`(N, 3)`): The coordinates of
                the *Cβ* atom for each residue.
            n (`numpy.ndarray` of shape :math:`(N, 3)`): The coordinates of
                the *N* atom for each residue.
            c (`numpy.ndarray` of shape :math:`(N, 3)`): The coordinates of
                the *C* atom for each residue.

        """
        raise NotImplementedError

    def encode_chain(
        self,
        chain: Chain,
        ca_residue: bool = True,
        disordered_atom: Literal["best", "last"] = "best",
    ) -> T:
        """Encode the given chain to a different representation.

        Arguments:
            chain (`Bio.PDB.Chain`): A single chain object parsed from a
                PDB structure.
            ca_residue (`bool`, *optional*): Only extract coordinates of
                residues which have a *CA* atom. Set to `False` to use every
                residue returned by the `~Bio.PDB.Chain.Chain.get_residues`
                method.
            disordered_atom (`str`): How to handle disordered atoms in the
                source chain. The default (`"best"`) will retain the atom
                with the best occupancy. Setting this to `"last"` will
                use the last atom instead, in order to produce the same
                results as Foldseek.

        .. versionadded:: 0.2.0
           The ``disordered_atom`` argument.

        """
        # extract residues
        if ca_residue:
            residues = [residue for residue in chain.get_residues() if "CA" in residue]
        else:
            residues = list(chain.get_residues())
        # extract atom coordinates
        r = len(residues)
        ca = numpy.array(numpy.nan, dtype=numpy.float32).repeat(3 * r).reshape(r, 3)
        cb = ca.copy()
        n = ca.copy()
        c = ca.copy()
        for i, residue in enumerate(residues):
            for atom in residue.get_atoms():
                if atom.is_disordered() and disordered_atom == "last":
                    atom = last(atom)
                if atom.get_name() == "CA":
                    ca[i, :] = atom.coord
                elif atom.get_name() == "N":
                    n[i, :] = atom.coord
                elif atom.get_name() == "C":
                    c[i, :] = atom.coord
                elif atom.get_name() == "CB" or atom.get_name() == "CB A":
                    cb[i, :] = atom.coord
        # encode coordinates
        return self.encode_atoms(ca, cb, n, c)


class VirtualCenterEncoder(_BaseEncoder["ArrayNx3[numpy.float32]"]):
    """An encoder for converting a protein structure to a virtual center.

    For each residue, the coordinates of the virtual center are computed
    from the coordinates of the *Cα*, *Cβ* and *N* atoms. The virtual center
    *V* is defined by the angle θ (V-Cα-Cβ), the dihedral angle τ
    (V-Cα-Cβ-N) and the length l (∣V − Cα∣). The default parameters used
    in ``foldseek`` were selected after optimization on a validation set.

    """

    def __init__(
        self,
        *,
        distance_alpha_beta = DISTANCE_ALPHA_BETA,
        distance_alpha_v: float = 2.0,
        theta: float = 270.0,
        tau: float = 0.0,
    ) -> None:
        """Create a new encoder.

        Arguments:
            distance_alpha_beta (`float`): The default distance between the
                *Cα* and *Cβ* atoms to use when reconstructing missing *Cβ*
                coordinates.
            distance_alpha_v (`float`): The distance between the virtual
                center *V* and the *Cα* atom, used to compute the virtual
                center coordinates.
            theta (`float`): The angle θ between the virtual center V, the
                *Cα* and *Cβ* atoms, used to compute the virtual center
                coordinates.
            tau (`float`): The dihedral angle τ between the virtual center V,
                and the *Cα*, *Cβ* and *N* atoms, used to compute the virtual
                center coordinates.

        """
        self.theta = theta
        self.tau = tau
        self.distance_alpha_v = distance_alpha_v
        self.distance_alpha_beta = distance_alpha_beta

    @property
    def theta(self) -> float:
        return numpy.rad2deg(self._theta)

    @theta.setter
    def theta(self, theta: float) -> None:
        self._theta = numpy.deg2rad(theta)
        self._cos_theta = numpy.cos(self._theta)
        self._sin_theta = numpy.sin(self._theta)

    @property
    def tau(self) -> float:
        return numpy.rad2deg(self._tau)

    @tau.setter
    def tau(self, tau: float) -> None:
        self._tau = numpy.deg2rad(tau)
        self._cos_tau = numpy.cos(self._tau)
        self._sin_tau = numpy.sin(self._tau)

    def _compute_virtual_center(
        self,
        ca: ArrayNx3[numpy.floating],
        cb: ArrayNx3[numpy.floating],
        n: ArrayNx3[numpy.floating],
    ) -> ArrayNx3[numpy.floating]:
        assert ca.shape == n.shape
        assert ca.shape == cb.shape
        v = cb - ca
        a = cb - ca
        b = n - ca
        # normal angle
        k = normalize(numpy.cross(a, b, axis=-1), inplace=True)
        v = (
            v * self._cos_theta
            + numpy.cross(k, v) * self._sin_theta
            + k * (k * v).sum(axis=-1).reshape(-1, 1) * (1 - self._cos_theta)
        )
        # dihedral angle
        k = normalize(n - ca, inplace=True)
        v = (
            v * self._cos_tau
            + numpy.cross(k, v) * self._sin_tau
            + k * (k * v).sum(axis=-1).reshape(-1, 1) * (1 - self._cos_tau)
        )
        # apply final vector to Cα
        v *= self.distance_alpha_v
        v += ca
        return v

    def _approximate_cb_position(
        self,
        ca: ArrayNx3[numpy.floating],
        n: ArrayNx3[numpy.floating],
        c: ArrayNx3[numpy.floating],
    ) -> ArrayNx3[numpy.floating]:
        """Approximate the position of the Cβ from the backbone atoms."""
        assert ca.shape == n.shape
        assert ca.shape == c.shape
        v1 = normalize(c - ca, inplace=True)
        v2 = normalize(n - ca, inplace=True)
        v3 = v1 / 3.0

        b1 = numpy.add(v2, v3, out=v2)
        b2 = numpy.cross(v1, b1, axis=-1)
        u1 = normalize(b1, inplace=True)
        u2 = normalize(b2, inplace=True)

        out = (numpy.sqrt(8) / 3.0) * ((-u1 / 2.0) - (u2 * numpy.sqrt(3) / 2.0)) - v3
        out *= self.distance_alpha_beta
        out += ca
        return out

    def _create_nan_mask(
        self,
        ca: ArrayNx3[numpy.floating],
        n: ArrayNx3[numpy.floating],
        c: ArrayNx3[numpy.floating],
    ) -> ArrayNx3[numpy.bool_]:
        """Mask any column which contains at least one NaN value.
        """
        mask_ca = numpy.isnan(ca).max(axis=1)
        mask_n = numpy.isnan(n).max(axis=1)
        mask_c = numpy.isnan(n).max(axis=1)
        return (mask_ca | mask_n | mask_c).repeat(3).reshape(-1, 3)

    def encode_atoms(
        self,
        ca: ArrayNx3[numpy.floating],
        cb: ArrayNx3[numpy.floating],
        n: ArrayNx3[numpy.floating],
        c: ArrayNx3[numpy.floating],
    ) -> ArrayNx3[numpy.float32]:
        ca = numpy.asarray(ca)
        cb = numpy.asarray(cb)
        n = numpy.asarray(n)
        c = numpy.asarray(c)

        assert ca.shape == cb.shape
        assert ca.shape == c.shape
        assert ca.shape == n.shape

        # fix CB positions if needed
        nan_indices = numpy.isnan(cb)
        if numpy.any(nan_indices):
            cb_approx = self._approximate_cb_position(ca, n, c)
            # avoid writing to CB directly since it should be callee-save
            cb_approx[~nan_indices] = cb[~nan_indices]
            cb = cb_approx
        # compute virtual center
        vc = self._compute_virtual_center(ca, cb, n)
        # mask residues without coordinates
        return numpy.ma.masked_array(  # type: ignore
            vc,
            mask=self._create_nan_mask(ca, n, c),
            fill_value=numpy.nan,
        )


class PartnerIndexEncoder(_BaseEncoder["ArrayN[numpy.int64]"]):
    """An encoder for converting a protein structure to partner indices.

    For each residue, the coordinates of the virtual center are computed
    from the coordinates of the *Cα*, *Cβ* and *N* atoms. A pairwise
    distance matrix is then created, and the index of the closest partner
    residue is extracted for each position.

    """

    def __init__(self) -> None:
        self.vc_encoder = VirtualCenterEncoder()

    def _find_residue_partners(
        self,
        x: ArrayNx3[numpy.floating],
    ) -> ArrayN[numpy.int64]:
        # compute pairwise squared distance matrix
        r = numpy.sum(x * x, axis=-1).reshape(-1, 1)
        r[0] = r[-1] = numpy.nan
        D = r - 2 * numpy.ma.dot(x, x.T) + r.T
        # avoid selecting residue itself as the best
        D[numpy.diag_indices_from(D)] = numpy.inf
        # get the closest non-masked residue
        return numpy.nan_to_num(D, copy=False, nan=numpy.inf).argmin(axis=1)

    def encode_atoms(
        self,
        ca: ArrayNx3[numpy.floating],
        cb: ArrayNx3[numpy.floating],
        n: ArrayNx3[numpy.floating],
        c: ArrayNx3[numpy.floating],
    ) -> ArrayN[numpy.int64]:
        # encode backbone atoms to virtual center
        vc = self.vc_encoder.encode_atoms(ca, cb, n, c)
        # find closest neighbor for each residue
        return self._find_residue_partners(vc)


class FeatureEncoder(_BaseEncoder["ArrayN[numpy.float32]"]):
    """An encoder for converting a protein structure to structural descriptors.
    """

    def __init__(self) -> None:
        self.partner_index_encoder = PartnerIndexEncoder()
        self.vc_encoder = self.partner_index_encoder.vc_encoder

    def _calc_conformation_descriptors(
        self,
        ca: ArrayNx3[numpy.floating],
        partner_index: ArrayN[numpy.int64],
        dtype: typing.Type[numpy.floating] = numpy.float32,
    ) -> ArrayNx10[numpy.floating]:
        # build arrays of indices to use for vectorized angles
        n = ca.shape[0]
        I = numpy.arange(1, ca.shape[-2] - 1)
        J = partner_index[I]
        # compute conformational descriptors
        u1 = normalize(ca[..., I, :] - ca[..., I - 1, :], inplace=True)
        u2 = normalize(ca[..., I + 1, :] - ca[..., I, :], inplace=True)
        u3 = normalize(ca[..., J, :] - ca[..., J - 1, :], inplace=True)
        u4 = normalize(ca[..., J + 1, :] - ca[..., J, :], inplace=True)
        u5 = normalize(ca[..., J, :] - ca[..., I, :], inplace=True)
        desc = numpy.zeros((ca.shape[0], 10), dtype=dtype)
        desc[I, 0] = numpy.sum(u1 * u2, axis=-1)
        desc[I, 1] = numpy.sum(u3 * u4, axis=-1)
        desc[I, 2] = numpy.sum(u1 * u5, axis=-1)
        desc[I, 3] = numpy.sum(u3 * u5, axis=-1)
        desc[I, 4] = numpy.sum(u1 * u4, axis=-1)
        desc[I, 5] = numpy.sum(u2 * u3, axis=-1)
        desc[I, 6] = numpy.sum(u1 * u3, axis=-1)
        desc[I, 7] = numpy.linalg.norm(ca[I] - ca[J], axis=-1)
        desc[I, 8] = numpy.clip(J - I, -4, 4)
        desc[I, 9] = numpy.copysign(numpy.log(numpy.abs(J - I) + 1), J - I)
        return desc

    def _create_descriptor_mask(
        self,
        mask: ArrayN[numpy.bool_],
        partner_index: ArrayN[numpy.int64],
    ) -> ArrayNx10[numpy.bool_]:
        I = numpy.arange(1, mask.shape[0] - 1)
        J = partner_index[I]
        out = numpy.zeros((mask.shape[0], 10), dtype=numpy.bool_)
        out[1:-1, :] |= (
            mask[I - 1] | mask[I] | mask[I + 1] | mask[J - 1] | mask[J] | mask[J + 1]
        ).reshape(mask.shape[0]-2, 1)
        out[0] = out[-1] = True
        return out

    def encode_atoms(
        self,
        ca: ArrayNx3[numpy.floating],
        cb: ArrayNx3[numpy.floating],
        n: ArrayNx3[numpy.floating],
        c: ArrayNx3[numpy.floating],
    ) -> ArrayN[numpy.uint8]:
        # encode backbone atoms to virtual center
        vc = self.vc_encoder.encode_atoms(ca, cb, n, c)
        # find closest neighbor for each residue
        partner_index = self.partner_index_encoder._find_residue_partners(vc)
        # build position features from residue angles
        descriptors = self._calc_conformation_descriptors(ca, partner_index)
        # create mask
        mask = self._create_descriptor_mask(vc.mask[:, 0], partner_index)
        return numpy.ma.masked_array(  # type: ignore
            descriptors,
            mask=mask,
            fill_value=numpy.nan,
        )


class Encoder(_BaseEncoder["ArrayN[numpy.uint8]"]):
    """An encoder for converting a protein structure to 3di states.
    """

    _INVALID_STATE = 2
    _CENTROIDS: ArrayNx2[numpy.float32] = numpy.array(
        [
            [-1.0729, -0.3600],
            [-0.1356, -1.8914],
            [0.4948, -0.4205],
            [-0.9874, 0.8128],
            [-1.6621, -0.4259],
            [2.1394, 0.0486],
            [1.5558, -0.1503],
            [2.9179, 1.1437],
            [-2.8814, 0.9956],
            [-1.1400, -2.0068],
            [3.2025, 1.7356],
            [1.7769, -1.3037],
            [0.6901, -1.2554],
            [-1.1061, -1.3397],
            [2.1495, -0.8030],
            [2.3060, -1.4988],
            [2.5522, 0.6046],
            [0.7786, -2.1660],
            [-2.3030, 0.3813],
            [1.0290, 0.8772],
        ]
    )

    def __init__(self) -> None:
        self.feature_encoder = FeatureEncoder()
        with resource_files(__package__).joinpath("encoder_weights_3di.kerasify").open("rb") as f:
            layers = _unkerasify.load(f)
            layers.append(CentroidLayer(self._CENTROIDS))
        self.vae_encoder = Model(layers)

    def encode_atoms(
        self,
        ca: ArrayNx3[numpy.floating],
        cb: ArrayNx3[numpy.floating],
        n: ArrayNx3[numpy.floating],
        c: ArrayNx3[numpy.floating],
    ) -> ArrayN[numpy.uint8]:
        descriptors = self.feature_encoder.encode_atoms(ca, cb, n, c)
        states = self.vae_encoder(descriptors.data)
        return numpy.ma.masked_array(
            states,
            mask=descriptors.mask[:, 0],
            fill_value=self._INVALID_STATE,
        )

    def build_sequence(self, states: ArrayN[numpy.uint8]) -> str:
        return "".join( ALPHABET[states.filled()] )

