import numpy as np
from scipy.special import sph_harm


l_max = 8
n_coeffs = int(0.5 * (l_max + 1) * (l_max + 2))

ls = np.zeros(n_coeffs).astype(int)
l0s = np.zeros(n_coeffs).astype(int)
for l in range(0, l_max + 1, 2):
    for m in range(-l, l + 1):
        ls[int(0.5 * l * (l + 1) + m)] = l
        l0s[int(0.5 * l * (l + 1) + m)] = int(0.5 * l * (l + 1))


def sh(l, m, thetas, phis):
    r"""Real and symmetric spherical harmonic basis function.

    Parameters
    ----------
    l : int
        Degree of the harmonic.
    m : int
        Order of the harmonic.
    thetas : array_like
        Polar coordinate.
    phis : array_like
        Azimuthal coordinate.

    Returns
    -------
    float or numpy.ndarray
        The harmonic sampled at `phi` and `theta`.

    Notes
    -----
    The basis function is defined as

    .. math:: S_l^m = \left\{\begin{matrix}
              0 & \text{if } l \text{ is odd} \\
              \sqrt{2} \ \Im \left( Y_l^{-m} \right) & \text{if} \ m < 0 \\ 
              Y_l^0 & \text{if} \ m = 0 \\
              \sqrt{2} \ \Re \left( Y_l^{m} \right) & \text{if} \ m > 0 
              \end{matrix}\right. ,

    where 

    .. math:: Y_l^m(\theta, \phi) \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}
              P_l^m(\cos \theta) e^{im\phi},
    
    where :math:`P_l^m` is the associated Legendre function.
    """
    if l % 2 == 1:
        return np.zeros(len(phis))
    if m < 0:
        return np.sqrt(2) * sph_harm(-m, l, phis, thetas).imag
    if m == 0:
        return sph_harm(m, l, phis, thetas).real
    if m > 0:
        return np.sqrt(2) * sph_harm(m, l, phis, thetas).real
