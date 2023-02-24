import matplotlib.pyplot as plt
import numpy as np


def show_tensor(D):
    """Visualize a diffusion tensor as an ellipsoid.

    Parameters
    ----------
    D : numpy.ndarray
        Array with a shape (3, 3).

    Returns
    -------
    None
    """
    evals, evecs = np.linalg.eig(D)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = evals[0] * np.outer(np.cos(u), np.sin(v))
    y = evals[1] * np.outer(np.sin(u), np.sin(v))
    z = evals[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot(evecs, [x[i, j], y[i, j], z[i, j]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-np.max(evals), np.max(evals))
    ax.set_ylim(-np.max(evals), np.max(evals))
    ax.set_zlim(-np.max(evals), np.max(evals))
    plt.show()
