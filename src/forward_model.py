# Forward models for Figre 8 coil 
import numpy as np


def vector_potential_loop(X, Y, Z, center_x, R=0.035, Nseg=300, current_sign=1):
    """
    Approximate magnetic vector potential A from a circular current loop.
    Constants are ignored because we only need normalized field shape.
    """

    # Coil loop parameter
    theta = np.linspace(0, 2*np.pi, Nseg, endpoint=False)
    dtheta = theta[1] - theta[0]

    Ax = np.zeros_like(X)
    Ay = np.zeros_like(Y)
    Az = np.zeros_like(Z)

    for t in theta:
        # source point on loop
        xs = center_x + R * np.cos(t)
        ys = R * np.sin(t)
        zs = 0.0

        # current element direction
        dlx = -R * np.sin(t) * dtheta
        dly =  R * np.cos(t) * dtheta
        dlz = 0.0

        rx = X - xs
        ry = Y - ys
        rz = Z - zs
        r = np.sqrt(rx**2 + ry**2 + rz**2)

        Ax += current_sign * dlx / r
        Ay += current_sign * dly / r
        Az += current_sign * dlz / r

    return Ax, Ay, Az
