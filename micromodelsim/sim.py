import numpy as np

from .sh import sh
from .grad import Gradient
from .vertices import vertices_768 as _vertices


def add_noise(signals, snr):
    r"""Add Rician noise to signals.

    Parameters
    ----------
    signals : numpy.ndarray
        Signals to which noise is added.
    snr : int
        Signal-to-noise ratio.

    Returns
    -------
    noisy_signals : numpy.ndarray
        `signals` with added noise.

    Notes
    -----
    Noise is added according to the following equation:

    .. math:: S_\text{noisy} = \sqrt{(S + X)^2 + Y^2},

    where :math:`X` and :math:`Y` are sampled from a normal distribution with
    zero mean and standard deviation of 1/`snr`.
    """
    return abs(
        signals
        + np.random.normal(0, 1 / snr, signals.shape)
        + 1j * np.random.normal(0, 1 / snr, signals.shape)
    )


def compartment_model_simulation(gradient, fs, ads, rds, odf_sh, l_max=16):
    """Generate simulated signals.

    Parameters
    ----------
    gradient : micromodelsim.grad.Gradient
        Object containing gradient information.
    fs : numpy.ndarray
        Compartment signal fractions in an array with shape (n of simulations,
        n of compartments).
    ads : numpy.ndarray
        Axial diffusivities in an array with shape (n of simulations, n of
        compartments).
    rds : numpy.ndarray
        Radial diffusivities in an array with shape (n of simulations, n of
        compartments).
    odf_sh : numpy.ndarray
        Spherical harmonic coefficients of the ODF in an array with shape (n of
        coefficients,). The ODF should be normalized to 1.
    l_max : int, optional
        Highest degree of the spherical harmonics included in the response
        function expansion.

    Returns
    -------
    numpy.ndarray
        Simulated signals.
    """
    if fs.ndim == 1:
        fs = fs[np.newaxis]

    if ads.ndim == 1:
        ads = ads[np.newaxis]

    if rds.ndim == 1:
        rds = rds[np.newaxis]

    n_simulations = fs.shape[0]
    n_compartments = fs.shape[1]

    Ds = np.zeros((n_simulations, n_compartments, 3, 3))
    Ds[:, :, 2, 2] = ads
    Ds[:, :, 1, 1] = rds
    Ds[:, :, 0, 0] = rds
    signals = np.zeros((n_simulations, len(gradient.bvals)))
    for i, idx in enumerate(gradient.shell_idx_list):
        if gradient.bten_shape == "linear":
            response = np.sum(
                fs[..., np.newaxis]
                * np.exp(
                    -np.sum(
                        gradient.bs[i]
                        * _rf_btens_lte[np.newaxis, np.newaxis]
                        * Ds[:, :, np.newaxis],
                        axis=(-2, -1),
                    )
                ),
                axis=1,
            )
        elif gradient.bten_shape == "planar":
            response = np.sum(
                fs[..., np.newaxis]
                * np.exp(
                    -np.sum(
                        gradient.bs[i]
                        * _rf_btens_pte[np.newaxis, np.newaxis]
                        * Ds[:, :, np.newaxis],
                        axis=(-2, -1),
                    )
                ),
                axis=1,
            )
        elif gradient.bten_shape == "spherical":
            response = np.sum(
                fs[..., np.newaxis]
                * np.exp(
                    -np.sum(
                        gradient.bs[i]
                        * _rf_btens_ste[np.newaxis, np.newaxis]
                        * Ds[:, :, np.newaxis],
                        axis=(-2, -1),
                    )
                ),
                axis=1,
            )
        response_sh = (sft[np.newaxis] @ response[:, :, np.newaxis])[..., 0]
        convolution_sh = (
            np.sqrt(4 * np.pi / (2 * ls + 1)) * odf_sh[np.newaxis] * response_sh[:, l0s]
        )
        signals[:, idx] = (
            gradient._bvecs_isft_list[i] @ convolution_sh[:, :, np.newaxis]
        )[..., 0]

    signals = np.squeeze(signals)
    return signals


def dtd_simulation(gradient, dtd, p=None):
    r"""Simulate signals from a diffusion tensor distribution.

    Parameters
    ----------
    gradient : micromodelsim.grad.Gradient
        Object containing gradient information.
    dtd : numpy.ndarray
        Diffusion tensor distribution in an array with shape (number of
        tensors, 3, 3).
    f : numpy.ndarray, optional
        1D array whose length equal to the number of tensors in `dtd`
        containing the signal fractions of each tensor. The sum of the signal
        fractions must be equal to 1. If not given, each tensor has the same
        signal fraction.

    Returns
    -------
    signals : numpy.ndarray
        Simulated signals.

    Notes
    -----
    Signals are generated according to the following equation:

    .. math:: S = \sum_{i=1}^N f_i \exp \left( -\mathbf{b}:\mathbf{D}_i
              \right),

    where :math:`N` is the number of diffusion tensors, :math:`f_i` is a signal
    fraction, :math:`\mathbf{b}` is the b-tensor, :math:`\mathbf{D}_i` is a
    diffusion tensor, and :math:`:` denotes the generalized scalar product:
    :math:`\mathbf{b}:\mathbf{D}=\sum_{i=1}^3 \sum_{j=1}^3b_{ij}D_{ij}`.
    """
    if not isinstance(gradient, Gradient):
        raise TypeError("Incorrect type for `gradient`")
    if not isinstance(dtd, np.ndarray):
        raise TypeError("Incorrect type for `dtd`")
    if dtd.ndim != 3 or dtd.shape[1:3] != (3, 3):
        raise ValueError("Incorrect shape for `dtd`")
    if p is None:
        p = np.ones(dtd.shape[0]) / dtd.shape[0]
    else:
        if not isinstance(p, np.ndarray):
            raise TypeError("Incorrect type for `p`")
        if p.ndim != 1 or p.shape[0] != dtd.shape[0]:
            raise ValueError("Incorrect shape for `dtd`")
        if p.sum() != 1:
            raise ValueError("`f` is not normalized")
    signals = np.sum(
        p[:, np.newaxis]
        * np.exp(
            -np.sum(gradient.btens[np.newaxis] * dtd[:, np.newaxis], axis=(-2, -1))
        ),
        axis=0,
    )
    return signals
