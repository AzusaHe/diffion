"""Differentiable-solver refinement (CPU): Junbo's 580-cell finite-volume scheme (physics_fv face_terms/update,
two cells per 290-grid bin = fine_fv.py), complex-step exact gradient, L-BFGS-B with Junbo's physics_runs.py settings."""
import os, sys, json, time, numpy as np
from numba import njit
REPO = os.environ.get('DIFFION_REPO', os.path.expanduser('~/diffion'))
sys.path.insert(0, os.path.join(REPO, 'one_current', 'clean_gaussian'))
from physics_fv import face_terms, update
from scipy.optimize import minimize

@njit(cache=True, nogil=True)
def objective(theta, currents, tau, mask, y):
    bins = y.shape[2]; total = theta[0]*0; frames = 0
    for j in range(len(currents)):
        c = np.ones(2*bins, dtype=theta.dtype); previous = 0.
        for t in range(tau.shape[1]):
            if not mask[j, t]: continue
            span = tau[j, t] - previous
            while span > 1e-11:
                dt = min(span, min(0.008, 0.00025 + previous*0.18))
                a, b = face_terms(c, theta, currents[j]); mid = update(c, a, b, currents[j], dt/2., 1.)
                a, b = face_terms(mid, theta, currents[j]); c = update(c, a, b, currents[j], dt, 0.5)
                previous += dt; span -= dt
            for x in range(bins):
                err = 0.5*(c[2*x] + c[2*x+1]) - y[j, t, x]; total += err*err
            frames += 1
    return total/(frames*bins)

@njit(cache=True, nogil=True)
def value_grad(theta, currents, tau, mask, y):
    v = objective(theta.astype(np.complex128), currents, tau, mask, y).real
    g = np.zeros(8); z = theta.astype(np.complex128)
    for k in range(8):
        z[k] += 1e-22j; g[k] = objective(z, currents, tau, mask, y).imag*1e22; z[k] = theta[k]
    return v, g

def fit(task):
    """task = (row, data_file, start_file, out_dir). One JSON per material, so the run resumes."""
    row, data_file, start_file, out_dir = task
    dest = os.path.join(out_dir, f'{row:03d}.json')
    if os.path.exists(dest): return
    z = np.load(data_file); s = np.load(start_file)
    m = z['mask'][row]; y = np.nan_to_num(z['c'][row].astype(np.float64)); tau = z['tau'][row]; io = z['iota'][row].astype(float)
    th0 = np.clip(s['theta'][row].astype(float), 1e-6, 1-1e-6)
    best = [np.inf, th0.copy()]
    def loss(th):
        try: v, g = value_grad(th, io, tau, m, y)
        except (OverflowError, ZeroDivisionError): return 1e8, np.zeros(8)
        if not (np.isfinite(v) and np.isfinite(g).all()): return 1e8, np.zeros(8)
        if v < best[0]: best[:] = [v, th.copy()]
        return v*1e8, g*1e8
    t = time.perf_counter()
    r = minimize(loss, th0, method='L-BFGS-B', jac=True, bounds=[(0., 1.)]*8,
                 options=dict(maxiter=150, maxfun=220, ftol=1e-11, gtol=1e-5, maxls=20))
    el = time.perf_counter() - t
    os.makedirs(out_dir, exist_ok=True)
    json.dump(dict(row=row, id=int(z['ids'][row]), theta=best[1].tolist(), c_rmse=float(np.sqrt(best[0])),
                   evals=int(r.nfev), elapsed_s=el, status=str(r.message)), open(dest + '.tmp', 'w'))
    os.replace(dest + '.tmp', dest)
