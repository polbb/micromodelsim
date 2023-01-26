import numpy as np

from .sh import _l_max, n_coeffs, sh


def _vec2vec_rotmat(v, k):
    """Return a rotation matrix defining a rotation that aligns `v` with `k`.

    Parameters
    -----------
    v : numpy.ndarray
        1D array with length 3.
    k : numpy.ndarray
        1D array with length 3.

    Returns
    ---------
    numpy.ndarray
        Rotation matrix.
    """
    v = v / np.linalg.norm(v)
    k = k / np.linalg.norm(k)
    axis = np.cross(v, k)
    if np.linalg.norm(axis) < 1e-10:  # the vectors are aligned
        if np.linalg.norm(v - k) > np.linalg.norm(v):
            return -np.eye(3)
        else:
            return np.eye(3)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.dot(v, k))
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    R = (
        np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.matmul(K, K)
    )  # Rodrigues' rotation formula
    return R


class Gradient(object):
    """Class for storing gradient information.

    Parameters
    ----------
    bvals : numpy.ndarray
        b-values in a 1D array whose length is equal to the number of
        measurements.
    bvecs : numpy.ndarray
        b-vectors in a 2D array with shape (number of measurements, 3).
    bten_shape : {"linear", "planar", "spherical"}, optional
        b-tensor shape. The default linear.

    Attributes
    ----------
    bvals : numpy.ndarray
        b-values.
    bvecs : numpy.ndarray
        b-vectors.
    bs : numpy.ndarray
        Unique b-values.
    bten_shape : numpy.ndarray
        b-tensor shape.
    btens : numpy.ndarray
        b-tensors.
    shell_idx_list : list
        Indices of `bvals` and `bvecs` corresponding to different shells.
    """

    def __init__(self, bvals, bvecs, bten_shape="linear"):

        if not isinstance(bvals, np.ndarray):
            raise TypeError("Incorrect type for `bvals`")
        if bvals.ndim != 1:
            raise ValueError("Incorrect shape for `bvals`")
        if not isinstance(bvecs, np.ndarray):
            raise TypeError("Incorrect type for `bvecs`")
        if bvecs.ndim != 2 or bvecs.shape[1] != 3:
            raise ValueError("Incorrect shape for `bvecs`")
        if len(bvals) != len(bvecs):
            raise ValueError("`bvals` and `bvecs` must have the same length")

        self.bvals = bvals
        self.bvecs = bvecs
        self.bten_shape = bten_shape
        self.btens = np.zeros((len(self.bvals), 3, 3))
        for i, (bval, bvec) in enumerate(zip(self.bvals, self.bvecs)):
            if bval == 0 and np.all(bvec == np.zeros(3)):
                self.btens[i] = np.zeros((3, 3))
            else:
                R = _vec2vec_rotmat(np.array([1, 0, 0]), bvec)
                if self.bten_shape == "linear":
                    self.btens[i] = (
                        R @ np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]) @ R.T * bval
                    )
                elif self.bten_shape == "planar":
                    self.btens[i] = (
                        R @ np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]) / 2 @ R.T * bval
                    )
                elif self.bten_shape == "spherical":
                    self.btens[i] = bval * np.eye(3) / 3
        self.bs = np.unique(bvals)
        self.shell_idx_list = [np.where(self.bvals == b)[0] for b in self.bs]

        self._bvecs_isft_list = []
        for bvecs in [bvecs[self.bvals == b] for b in self.bs]:
            thetas = np.arccos(bvecs[:, 2])
            phis = np.arctan2(bvecs[:, 1], bvecs[:, 0]) + np.pi
            bvecs_isft = np.zeros((len(bvecs), n_coeffs), dtype=float)
            for l in range(0, _l_max + 1, 2):
                for m in range(-l, l + 1):
                    bvecs_isft[:, int(0.5 * l * (l + 1) + m)] = sh(l, m, thetas, phis)
            self._bvecs_isft_list.append(bvecs_isft)
