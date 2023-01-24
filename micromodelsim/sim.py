import numpy as np

from .sh import l0s, ls, sft, sh
from .grad import Gradient
from .vertices import vertices_3072


rf_btens_lte = Gradient(np.ones(3072), vertices_3072).btens
rf_btens_pte = Gradient(np.ones(3072), vertices_3072, "planar").btens
rf_btens_ste = Gradient(np.ones(3072), vertices_3072, "spherical").btens


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


def compartment_model_simulation(gradient, fs, ads, rds, odfs_sh):
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
    odfs_sh : numpy.ndarray
        Spherical harmonic coefficients of the ODF in an array with shape (n of
        simulations, n of coefficients).

    Returns
    -------
    numpy.ndarray
        Simulated signals.
    """
    if not isinstance(gradient, Gradient):
        raise TypeError("Incorrect type for `gradient`")
    if not isinstance(fs, np.ndarray):
        raise TypeError("Incorrect type for `fs`")
    if fs.ndim > 2:
        raise ValueError("Incorrect shape for `fs`")
    if not isinstance(ads, np.ndarray):
        raise TypeError("Incorrect type for `ads`")
    if ads.ndim > 2:
        raise ValueError("Incorrect shape for `ads`")
    if not isinstance(rds, np.ndarray):
        raise TypeError("Incorrect type for `ads`")
    if rds.ndim > 2:
        raise ValueError("Incorrect shape for `ads`")
    if not (fs.shape == ads.shape == rds.shape):
        raise ValueError("`fs`, `ads`, and `rds` must have the same shape")
    if not isinstance(odfs_sh, np.ndarray):
        raise TypeError("Incorrect type for `odfs_sh`")
    n_coeffs = odfs_sh.shape[1]
    l_max = 0.5 * (np.sqrt(8 * n_coeffs + 1) - 3)
    if len(odfs_sh) != len(fs) or (int(l_max) - l_max) > 1e-10:
        raise ValueError("Incorrect shape for `odfs_sh`")

    n_simulations = fs.shape[0]
    n_compartments = fs.shape[1]

    odfs_sh = odfs_sh / odfs_sh[:, 0][:, np.newaxis] / np.sqrt(4 * np.pi)

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
                        * rf_btens_lte[np.newaxis, np.newaxis]
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
                        * rf_btens_pte[np.newaxis, np.newaxis]
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
                        * np.eye(3)[np.newaxis, np.newaxis]
                        * Ds[:, :, np.newaxis],
                        axis=(-2, -1),
                    )
                ),
                axis=1,
            )
        response_sh = (sft[np.newaxis, 0:n_coeffs] @ response[:, :, np.newaxis])[..., 0]
        convolution_sh = (
            np.sqrt(4 * np.pi / (2 * ls[0:n_coeffs] + 1))
            * odfs_sh[np.newaxis]
            * response_sh[:, l0s[0:n_coeffs]]
        )
        signals[:, idx] = (
            gradient._bvecs_isft_list[i][:, 0:n_coeffs]
            @ convolution_sh[..., np.newaxis]
        )[..., 0]
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
