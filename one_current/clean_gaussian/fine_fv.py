"""Real-valued finite-volume concentration loss for derivative-free fits.

Each reported spatial bin averages two uniform finite-volume cells. Time is
advanced through every valid observation frame using the benchmark's midpoint
scheme. Masked frames may have padded timestamps or concentrations.
"""

import numpy as np
from numba import njit

from .physics_fv import face_terms, update


@njit(cache=True, nogil=True, error_model="numpy")
def objective(theta, currents, tau, mask, y):
    """Mean squared dimensionless concentration error across valid bin-frames."""
    bins = y.shape[2]
    if bins == 0 or theta.size != 8 or not np.all(np.isfinite(theta)):
        return np.inf
    if np.any(theta < 0.) or np.any(theta > 1.):
        return np.inf
    total = 0.
    frames = 0
    for j in range(len(currents)):
        c = np.ones(2 * bins)
        previous = 0.
        for t in range(tau.shape[1]):
            if not mask[j, t]:
                continue
            span = tau[j, t] - previous
            if not np.isfinite(span) or span < -1e-11:
                return np.inf
            while span > 1e-11:
                dt = min(span, min(0.008, 0.00025 + previous * 0.18))
                a, b = face_terms(c, theta, currents[j])
                mid = update(c, a, b, currents[j], dt / 2., 1.)
                a, b = face_terms(mid, theta, currents[j])
                c = update(c, a, b, currents[j], dt, 0.5)
                previous += dt
                span -= dt
            for x in range(bins):
                err = 0.5 * (c[2 * x] + c[2 * x + 1]) - y[j, t, x]
                total += err * err
            if not np.isfinite(total):
                return np.inf
            frames += 1
    if frames == 0:
        return np.inf
    return total / (frames * bins)
