import healpy as hp
import numpy as np

from .sh import l0s, ls, l_max, n_coeffs, sh
from .grad import Gradient


_n_sides = 2**3
_x, _y, _z = hp.pix2vec(_n_sides, np.arange(12 * _n_sides**2))
_vertices = np.vstack((_x, _y, _z)).T
_thetas = np.arccos(_vertices[:, 2])
_phis = np.arctan2(_vertices[:, 1], _vertices[:, 0]) + np.pi
isft = np.zeros((len(_vertices), n_coeffs))
for l in range(0, l_max + 1, 2):
    for m in range(-l, l + 1):
        isft[:, int(0.5 * l * (l + 1) + m)] = sh(l, m, _thetas, _phis)
sft = np.linalg.inv(isft.T @ isft) @ isft.T
_rf_btens_lte = Gradient(np.ones(len(_vertices)), _vertices, "linear").btens
_rf_btens_pte = Gradient(np.ones(len(_vertices)), _vertices, "planar").btens


def compartment_model_simulation(gradient, fs, ads, rds, odf_sh):
    """Generate simulated signals.

    Parameters
    ----------
    gradient : micromodelsim.grad.Gradient
        Object containing gradient information.
    fs : array_like
        Compartment signal fractions.
    ads : array_like
        Axial diffusivities.
    rds : array_like
        Radial diffusivities.
    odf_sh : array_like
        Spherical harmonic coefficients of the ODF.

    Returns
    -------
    signals : numpy.ndarray
    """
    n_compartments = len(fs)
    Ds = np.zeros((n_compartments, 3, 3))
    Ds[:, 2, 2] = ads
    Ds[:, 1, 1] = rds
    Ds[:, 0, 0] = rds
    signals = np.zeros(len(gradient.bvals))
    for i, idx in enumerate(gradient.shell_idx_list):
        if gradient.bten_shape == "linear":
            response = np.sum(
                fs[:, np.newaxis]
                * np.exp(
                    -np.sum(
                        gradient.bs[i] * _rf_btens_lte[np.newaxis] * Ds[:, np.newaxis],
                        axis=(2, 3),
                    )
                ),
                axis=0,
            )
        elif gradient.bten_shape == "planar":
            response = np.sum(
                fs[:, np.newaxis]
                * np.exp(
                    -np.sum(
                        gradient.bs[i] * _rf_btens_lte[np.newaxis] * Ds[:, np.newaxis],
                        axis=(2, 3),
                    )
                ),
                axis=0,
            )
        response_sh = sft @ response
        convolution_sh = np.sqrt(4 * np.pi / (2 * ls + 1)) * odf_sh * response_sh[l0s]
        signals[idx] = gradient._bvecs_isft_list[i] @ convolution_sh
    return signals


def add_noise(signals, SNR):
    r"""Add Rician noise to signals.

    Parameters
    ----------
    signals : array_like
        Signals to which noise is added.
    SNR : int
        Signal-to-noise ratio.

    Returns
    -------
    noisy_signals : np.ndarray
        `signals` with noise added to it.

    Notes
    -----
    Noisy signals are

    .. math:: S_\text{noisy} = \sqrt{(S + X)^2 + Y^2},

    where :math:`X` and :math:`Y` are sampled from a normal distribution with
    zero mean and standard deviation of :math:`1/\text{SNR}`.
    """
    return abs(
        signals * np.random.normal(0, 1 / SNR, signals.shape)
        + 1j * np.random.normal(0, 1 / SNR, signals.shape)
    )
