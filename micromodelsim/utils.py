import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np 

def plot_ellipsoid(center, orientation, semiaxes): 
    """
    Plot an ellipsoid using matplotlib.

    Parameters
    ----------
    center : array 
        The x, y, z coordinates of the center of the ellipsoid.
    orientation : array 
        The orientation of the ellipsoid.
    semiaxes : array 
        The length of the semiaxes of the ellipsoid.

    Returns
    -------
    None 
    """
    # Generate data for ellipsoid 
    u = np.linspace(0.0, 2.0 * np.pi, 100) 
    v = np.linspace(0.0, np.pi, 100) 
    x = semiaxes[0] * np.outer(np.cos(u), np.sin(v)) 
    y = semiaxes[1] * np.outer(np.sin(u), np.sin(v)) 
    z = semiaxes[2] * np.outer(np.ones_like(u), np.cos(v)) 
    
    # Rotate data 
    for i in range(len(x)): 
        for j in range(len(x)): 
            [x[i, j], y[i, j], z[i, j]] = np.dot(orientation, [x[i, j], y[i, j], z[i, j]]) + center 
    
    # Plot the surface 
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d') 
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b') 
    plt.show() 


# Driver code
center = [0, 0, 0] 
orientation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] 
semiaxes = [2, 3, 4] 

plot_ellipsoid(center, orientation, semiaxes)