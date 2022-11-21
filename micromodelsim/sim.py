import healpy as hp
import numpy as np

from .sh import l0s, ls, l_max, n_coeffs, sh
from .grad import Gradient


_n_sides = 2**2
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
_rf_btens_ste = Gradient(np.ones(len(_vertices)), _vertices, "spherical").btens


def compartment_model_simulation(gradient, fs, ads, rds, odf_sh):
    """Generate simulated signals.

    Parameters
    ----------
    gradient : micromodelsim.grad.Gradient
        Object containing gradient information.
    fs : array_like
        Compartment signal fractions in an array with shape (n of simulations,
        n of compartments).
    ads : array_like
        Axial diffusivities in an array with shape (n of simulations, n of
        compartments).
    rds : array_like
        Radial diffusivities in an array with shape (n of simulations, n of
        compartments).
    odf_sh : array_like
        Spherical harmonic coefficients of the ODF in an array with shape (n of
        coefficients,).

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



def dtd_simulation(gradient, dtd, P = None):
    """Generate simulated signals.

    Parameters
    ----------
    gradient : micromodelsim.grad.Gradient
        Object containing gradient information.
    dtd : array_like
        Diffusion tensor distribution, [# compartments, 3, 3].
    P : array_like
        Weight of each tensor in distribution. If 'None' then tensors are evenly weighted

    Returns
    -------
    signals : numpy.ndarray
    
    Notes
    -----
    Signals are generated using:
    
    .. math:: S = S_0 \int P(\mathbf{D}_\mu) \exp(-\mathbf{b:D})\,d\mathbf{D}_\mu
    
    """
    if P is None:
        P = np.ones(dtd.shape[0]) / dtd.shape[0]
    
    signals = np.sum(P[:, np.newaxis]*
                     np.exp(
                         -np.sum(gradient.btens[np.newaxis]*dtd[:,np.newaxis], axis=(-2, -1))
                         ), 
                     axis=0)
    
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
