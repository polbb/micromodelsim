import numpy as np
import numpy.testing as npt

import micromodelsim as mmsim


def test__vec2vec_rotmat():
    np.random.seed(123)
    for _ in range(int(1e4)):
        v = np.random.random(3) - 0.5
        k = np.random.random(3) - 0.5
        R = mmsim._vec2vec_rotmat(v, k)
        aligned_v = R @ v
        assert (np.linalg.norm(v) - np.linalg.norm(aligned_v)) < 1e-10
        assert np.all(
            (k / np.linalg.norm(k) - aligned_v / np.linalg.norm(aligned_v)) < 1e-10
        )


def test_Gradient():
    npt.assert_raises(TypeError, mmsim.Gradient, bvals=[1, 2], bvecs=np.zeros((2, 3)))
    npt.assert_raises(
        ValueError, mmsim.Gradient, bvals=np.zeros((2, 3)), bvecs=np.zeros((2, 3))
    )
    npt.assert_raises(TypeError, mmsim.Gradient, bvals=np.zeros(2), bvecs=[1, 2])
    npt.assert_raises(
        ValueError, mmsim.Gradient, bvals=np.zeros(2), bvecs=np.zeros((3, 3))
    )
    npt.assert_raises(
        ValueError, mmsim.Gradient, bvals=np.zeros(3), bvecs=np.zeros((2, 3))
    )
    bvals = np.array([1, 2, 2])
    bvecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    gradient = mmsim.Gradient(bvals, bvecs)
    assert np.all(gradient.bvals == bvals)
    assert np.all(gradient.bvecs == bvecs)
    assert np.all(gradient.bs == np.unique(bvals))
    assert gradient.bten_shape == "linear"
    assert np.all(
        (
            gradient.btens
            - np.array(
                [
                    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 2]],
                ]
            )
        )
        < 1e-10,
    )
    assert np.all(gradient.shell_idx_list[0] == np.array([0]))
    assert np.all(gradient.shell_idx_list[1] == np.array([1, 2]))
