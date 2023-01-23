import os

import numpy as np

import micromodelsim

vertices_12 = np.loadtxt(
    os.path.join(os.path.dirname(micromodelsim.__file__), "healpix", "vertices_12.txt")
)
vertices_48 = np.loadtxt(
    os.path.join(os.path.dirname(micromodelsim.__file__), "healpix", "vertices_48.txt")
)
vertices_192 = np.loadtxt(
    os.path.join(os.path.dirname(micromodelsim.__file__), "healpix", "vertices_192.txt")
)
vertices_768 = np.loadtxt(
    os.path.join(os.path.dirname(micromodelsim.__file__), "healpix", "vertices_768.txt")
)
vertices_3072 = np.loadtxt(
    os.path.join(
        os.path.dirname(micromodelsim.__file__), "healpix", "vertices_3072.txt"
    )
)
