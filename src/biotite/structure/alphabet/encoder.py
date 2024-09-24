# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Implementation of the encoder neural network adapted from ``foldseek``.
"""

__name__ = "biotite.structure.alphabet"
__author__ = "Martin Larralde"
__all__ = ["Encoder", "VirtualCenterEncoder", "PartnerIndexEncoder", "FeatureEncoder"]

import abc
from importlib.resources import files as resource_files
import numpy
import numpy.ma
from biotite.structure.alphabet.layers import CentroidLayer, Model
from biotite.structure.alphabet.unkerasify import load_kerasify


class _BaseEncoder(abc.ABC):
    @abc.abstractmethod
    def encode(self, ca, cb, n, c):
        """
        Encode the given atom coordinates to a different representation.

        Parameters
        ----------
        ca, cb, n, c : ndarray, shape=(n, 3), dtype=float
            The coordinates of the ``CA``, ``CB``, ``N`` and ``C`` atoms for each
            residue.
            *NaN* if missing, e.g. ``CB`` for glycine.

        Returns
        -------
        encoded : MaskedArray, shape=(n, m), dtype=float
            The encoded representation.
        """
        raise NotImplementedError


class VirtualCenterEncoder(_BaseEncoder):
    r"""
    An encoder for converting a protein structure to a virtual center.

    For each residue, the coordinates of the virtual center are computed
    from the coordinates of the ``CA``, ``CB`` and ``N`` atoms. The virtual center
    :math:`V` is defined by the angle :math:`\theta = \angle V C_{\alpha} C_{\beta}`,
    the dihedral angle :math:`\tau = \angle V C_{\alpha} C_{\beta} N` and the length
    :math:`l = |V - C_{\alpha}|`. The default parameters used
    in ``foldseek`` were selected after optimization on a validation set.

    Parameters
    ----------
    distance_alpha_beta : float
        The default distance between the ``CA`` and ``CB`` atoms to use when
        reconstructing missing *Cβ* coordinates.
    distance_alpha_v : float
        The distance between the virtual center *V* and the ``CA`` atom, used to compute
        the virtual center coordinates.
    theta : float
        The angle θ between the virtual center *V*, the ``CA`` and ``CB`` atoms, used to
        compute the virtual center coordinates.
    tau : float
        The dihedral angle τ between the virtual center *V* and the ``CA``, ``CB``
        and ``N`` atoms, used to compute the virtual center coordinates.
    """

    _DISTANCE_ALPHA_BETA = 1.5336

    def __init__(
        self,
        *,
        distance_alpha_beta=_DISTANCE_ALPHA_BETA,
        distance_alpha_v=2.0,
        theta=270.0,
        tau=0.0,
    ):
        self.theta = theta
        self.tau = tau
        self.distance_alpha_v = distance_alpha_v
        self.distance_alpha_beta = distance_alpha_beta

    @property
    def theta(self):
        return numpy.rad2deg(self._theta)

    @theta.setter
    def theta(self, theta):
        self._theta = numpy.deg2rad(theta)
        self._cos_theta = numpy.cos(self._theta)
        self._sin_theta = numpy.sin(self._theta)

    @property
    def tau(self):
        return numpy.rad2deg(self._tau)

    @tau.setter
    def tau(self, tau):
        self._tau = numpy.deg2rad(tau)
        self._cos_tau = numpy.cos(self._tau)
        self._sin_tau = numpy.sin(self._tau)

    def _compute_virtual_center(self, ca, cb, n):
        assert ca.shape == n.shape
        assert ca.shape == cb.shape
        v = cb - ca
        a = cb - ca
        b = n - ca
        # normal angle
        k = _normalize(numpy.cross(a, b, axis=-1), inplace=True)
        v = (
            v * self._cos_theta
            + numpy.cross(k, v) * self._sin_theta
            + k * (k * v).sum(axis=-1).reshape(-1, 1) * (1 - self._cos_theta)
        )
        # dihedral angle
        k = _normalize(n - ca, inplace=True)
        v = (
            v * self._cos_tau
            + numpy.cross(k, v) * self._sin_tau
            + k * (k * v).sum(axis=-1).reshape(-1, 1) * (1 - self._cos_tau)
        )
        # apply final vector to Cα
        v *= self.distance_alpha_v
        v += ca
        return v

    def _approximate_cb_position(self, ca, n, c):
        """
        Approximate the position of ``CB`` from the backbone atoms.
        """
        assert ca.shape == n.shape
        assert ca.shape == c.shape
        v1 = _normalize(c - ca, inplace=True)
        v2 = _normalize(n - ca, inplace=True)
        v3 = v1 / 3.0

        b1 = numpy.add(v2, v3, out=v2)
        b2 = numpy.cross(v1, b1, axis=-1)
        u1 = _normalize(b1, inplace=True)
        u2 = _normalize(b2, inplace=True)

        out = (numpy.sqrt(8) / 3.0) * ((-u1 / 2.0) - (u2 * numpy.sqrt(3) / 2.0)) - v3
        out *= self.distance_alpha_beta
        out += ca
        return out

    def _create_nan_mask(self, ca, n, c):
        """
        Mask any column which contains at least one *NaN* value.
        """
        mask_ca = numpy.isnan(ca).max(axis=1)
        mask_n = numpy.isnan(n).max(axis=1)
        mask_c = numpy.isnan(c).max(axis=1)
        return (mask_ca | mask_n | mask_c).repeat(3).reshape(-1, 3)

    def encode(self, ca, cb, n, c):
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
        return numpy.ma.masked_array(
            vc,
            mask=self._create_nan_mask(ca, n, c),
            fill_value=numpy.nan,
        )


class PartnerIndexEncoder(_BaseEncoder):
    """
    An encoder for converting a protein structure to partner indices.

    For each residue, the coordinates of the virtual center are computed from the
    coordinates of the ``CA``, ``CB`` and ``N`` atoms.
    A pairwise distance matrix is then created, and the index of the closest partner
    residue is extracted for each position.
    """

    def __init__(self):
        self.vc_encoder = VirtualCenterEncoder()

    def _find_residue_partners(
        self,
        x,
    ):
        # compute pairwise squared distance matrix
        r = numpy.sum(x * x, axis=-1).reshape(-1, 1)
        r[0] = r[-1] = numpy.nan
        D = r - 2 * numpy.ma.dot(x, x.T) + r.T
        # avoid selecting residue itself as the best
        D[numpy.diag_indices_from(D)] = numpy.inf
        # get the closest non-masked residue
        return numpy.nan_to_num(D, copy=False, nan=numpy.inf).argmin(axis=1)

    def encode(self, ca, cb, n, c):
        # encode backbone atoms to virtual center
        vc = self.vc_encoder.encode(ca, cb, n, c)
        # find closest neighbor for each residue
        return self._find_residue_partners(vc)


class FeatureEncoder(_BaseEncoder):
    """
    An encoder for converting a protein structure to structural descriptors.
    """

    def __init__(self):
        self.partner_index_encoder = PartnerIndexEncoder()
        self.vc_encoder = self.partner_index_encoder.vc_encoder

    def _calc_conformation_descriptors(self, ca, partner_index, dtype=numpy.float32):
        # build arrays of indices to use for vectorized angles
        i = numpy.arange(1, ca.shape[-2] - 1)
        j = partner_index[i]
        # compute conformational descriptors
        u1 = _normalize(ca[..., i, :] - ca[..., i - 1, :], inplace=True)
        u2 = _normalize(ca[..., i + 1, :] - ca[..., i, :], inplace=True)
        u3 = _normalize(ca[..., j, :] - ca[..., j - 1, :], inplace=True)
        u4 = _normalize(ca[..., j + 1, :] - ca[..., j, :], inplace=True)
        u5 = _normalize(ca[..., j, :] - ca[..., i, :], inplace=True)
        desc = numpy.zeros((ca.shape[0], 10), dtype=dtype)
        desc[i, 0] = numpy.sum(u1 * u2, axis=-1)
        desc[i, 1] = numpy.sum(u3 * u4, axis=-1)
        desc[i, 2] = numpy.sum(u1 * u5, axis=-1)
        desc[i, 3] = numpy.sum(u3 * u5, axis=-1)
        desc[i, 4] = numpy.sum(u1 * u4, axis=-1)
        desc[i, 5] = numpy.sum(u2 * u3, axis=-1)
        desc[i, 6] = numpy.sum(u1 * u3, axis=-1)
        desc[i, 7] = numpy.linalg.norm(ca[i] - ca[j], axis=-1)
        desc[i, 8] = numpy.clip(j - i, -4, 4)
        desc[i, 9] = numpy.copysign(numpy.log(numpy.abs(j - i) + 1), j - i)
        return desc

    def _create_descriptor_mask(self, mask, partner_index):
        i = numpy.arange(1, mask.shape[0] - 1)
        j = partner_index[i]
        out = numpy.zeros((mask.shape[0], 10), dtype=numpy.bool_)
        out[1:-1, :] |= (
            mask[i - 1] | mask[i] | mask[i + 1] | mask[j - 1] | mask[j] | mask[j + 1]
        ).reshape(mask.shape[0] - 2, 1)
        out[0] = out[-1] = True
        return out

    def encode(self, ca, cb, n, c):
        # encode backbone atoms to virtual center
        vc = self.vc_encoder.encode(ca, cb, n, c)
        # find closest neighbor for each residue
        partner_index = self.partner_index_encoder._find_residue_partners(vc)
        # build position features from residue angles
        descriptors = self._calc_conformation_descriptors(ca, partner_index)
        # create mask
        mask = self._create_descriptor_mask(vc.mask[:, 0], partner_index)
        return numpy.ma.masked_array(
            descriptors,
            mask=mask,
            fill_value=numpy.nan,
        )


class Encoder(_BaseEncoder):
    """
    An encoder for converting a protein structure to 3di states.
    """

    _INVALID_STATE = 2
    _CENTROIDS = numpy.array(
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

    def __init__(self):
        self.feature_encoder = FeatureEncoder()
        layers = load_kerasify(
            resource_files(__package__).joinpath("encoder_weights_3di.kerasify")
        )
        self.vae_encoder = Model(layers + (CentroidLayer(self._CENTROIDS),))

    def encode(
        self,
        ca,
        cb,
        n,
        c,
    ):
        descriptors = self.feature_encoder.encode(ca, cb, n, c)
        states = self.vae_encoder(descriptors.data)
        return numpy.ma.masked_array(
            states,
            mask=descriptors.mask[:, 0],
            fill_value=self._INVALID_STATE,
        )


def _normalize(x, *, inplace=False):
    norm = numpy.linalg.norm(x, axis=-1).reshape(*x.shape[:-1], 1)
    return numpy.divide(x, norm, out=x if inplace else None, where=norm != 0)
