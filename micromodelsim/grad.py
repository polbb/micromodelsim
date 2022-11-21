try: 
    import JAX.numpy as np
except ImportError:
    print("Unable to import JAX, numpy is imported instead")
    import numpy as np


from .sh import sh, l_max, n_coeffs


def _vec2vec_rotmat(v, k):
    """Return a rotation matrix defining a rotation that aligns `v` with `k`.

    Parameters
    -----------
    v : array_like
        1D array with length 3.
    k : array_like
        1D array with length 3.

    Returns
    ---------
    R : numpy.ndarray
        3 by 3 rotation matrix.
    """
    v = v / np.linalg.norm(v)
    k = k / np.linalg.norm(k)
    axis = np.cross(v, k)
    if np.linalg.norm(axis) < 1e-10:
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
    bvals : array_like
        b-values in an array with shape (number of measurements,).
    bvecs : array_like
        b-vectors in an array with shape (number of measurements, 3).
    bten_shape : {"linear", "planar", "spherical"}, optional
        b-tensor shape.

    Attributes
    ----------
    bvals : numpy.ndarray
        b-values.
    bvecs : numpy.ndarray
        b-vectors.
    bten_shape : numpy.ndarray
        b-tensor shape.
    btens : numpy.ndarray
        b-tensors.
    bs : int
        Unique b-values.
    shell_idx_list : list
        Indices of `bvals` and `bvecs` corresponding to different shells.
    """

    def __init__(self, bvals, bvecs, bten_shape="linear"):

        if bvals.ndim != 1:
            raise ValueError("Incorrect value for `bvals`")
        if bvecs.ndim != 2 or bvecs.shape[1] != 3:
            raise ValueError(f"Incorrect value for `bvecs`")
        if len(bvals) != len(bvecs):
            raise ValueError("`bvals` and `bvecs` should be the same length.")

        self.bvals = bvals
        self.bvecs = bvecs
        self.bten_shape = bten_shape
        self.btens = np.zeros((len(self.bvals), 3, 3))
        for i, (bval, bvec) in enumerate(zip(self.bvals, self.bvecs)):
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
        for idx in self.shell_idx_list:
            thetas = np.arccos(self.bvecs[idx, 2])
            phis = np.arctan2(self.bvecs[idx, 1], self.bvecs[idx, 0]) + np.pi
            bvecs_isft = np.zeros((len(self.bvecs[idx]), n_coeffs))
            for l in range(0, l_max + 1, 2):
                for m in range(-l, l + 1):
                    bvecs_isft[:, int(0.5 * l * (l + 1) + m)] = sh(l, m, thetas, phis)
            self._bvecs_isft_list.append(bvecs_isft)
