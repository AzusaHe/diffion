"""Derivative-free baseline optimizers: Nelder-Mead (NM), CMA-ES and Bayesian optimization (BO).

Every method searches the normalized box z in [0, 1]^8, starts from the same point and calls the
same loss. A Recorder wraps the loss, logs every call, and ends the run at the first of:
  * best loss <= target          -> status "reached_target"
  * evaluations == budget        -> status "used_budget"
  * wall time >= max_wall_s      -> status "used_walltime"
A run that ends inside the optimizer's own stopping rule has status "optimizer_converged".
"""
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

METHODS = ("NM", "CMAES", "BO")
LABELS = {"NM": "NM", "CMAES": "CMA-ES", "BO": "BO"}
CSV_COLUMNS = ["eval", "wall_s", "solver_s", "rmse_M", "best_rmse_M"] + [f"theta{k}" for k in range(8)]


class _EndRun(Exception):
    pass


class Recorder:
    """Callable loss wrapper that logs each evaluation and enforces the stopping rules."""

    def __init__(self, loss, target, budget, max_wall_s, to_theta):
        self.loss, self.target, self.budget, self.max_wall_s = loss, target, budget, max_wall_s
        self.to_theta = to_theta
        self.rows = []
        self.best = np.inf
        self.solver_s = 0.0
        self.hit = None
        self.ended = None
        self.t0 = time.perf_counter()

    def __call__(self, z):
        if len(self.rows) >= self.budget:
            self.ended = "used_budget"
            raise _EndRun
        if time.perf_counter() - self.t0 >= self.max_wall_s:
            self.ended = "used_walltime"
            raise _EndRun
        z = np.clip(np.asarray(z, float), 0.0, 1.0)
        t = time.perf_counter()
        f = float(self.loss(z))
        self.solver_s += time.perf_counter() - t
        self.best = min(self.best, f)
        wall = time.perf_counter() - self.t0
        n = len(self.rows) + 1
        self.rows.append([n, wall, self.solver_s, f, self.best, *np.asarray(self.to_theta(z)).tolist()])
        if self.best <= self.target:
            self.hit = dict(evals=n, wall_s=wall, solver_s=self.solver_s)
            self.ended = "reached_target"
            raise _EndRun
        return f

    def status(self):
        if self.ended:
            return self.ended
        return "used_budget" if len(self.rows) >= self.budget else "optimizer_converged"


def nelder_mead(f, z0, seed, budget):
    step = np.where(z0 + 0.1 <= 1.0, 0.1, -0.1)
    simplex = np.vstack([z0] + [z0 + step[i] * np.eye(len(z0))[i] for i in range(len(z0))])
    minimize(f, z0, method="Nelder-Mead", bounds=[(0, 1)] * len(z0),
             options=dict(initial_simplex=simplex, adaptive=True, maxfev=budget, xatol=1e-14, fatol=1e-14))


def cma_es(f, z0, seed, budget):
    import cma
    es = cma.CMAEvolutionStrategy(list(z0), 0.25, dict(bounds=[0, 1], seed=seed + 1, maxfevals=budget,
                                                         tolfun=1e-15, tolx=1e-15, verbose=-9))
    while not es.stop():
        X = es.ask()
        es.tell(X, [f(x) for x in X])


def bayes_opt(f, z0, seed, budget):
    from skopt import gp_minimize
    gp_minimize(lambda x: f(np.array(x)), [(0.0, 1.0)] * len(z0), x0=[list(z0)],
                n_initial_points=16, n_calls=budget, acq_func="EI", random_state=seed)


OPTIMIZERS = {"NM": nelder_mead, "CMAES": cma_es, "BO": bayes_opt}


def sobol_starts(n, dim=8, seed=0):
    """Shared starting points: the first n of 8 scrambled Sobol points in [0, 1]^dim."""
    return qmc.Sobol(d=dim, scramble=True, seed=seed).random(8)[:n]


def run(method, loss, z0, seed, target, budget, max_wall_s, to_theta):
    """Run one optimizer from z0. Returns (summary dict, per-evaluation rows as an array)."""
    rec = Recorder(loss, target, budget, max_wall_s, to_theta)
    try:
        OPTIMIZERS[method](rec, np.array(z0, float), seed, budget)
    except _EndRun:
        pass
    rows = np.array(rec.rows)
    best = int(np.argmin(rows[:, 3]))
    hit = rec.hit or {}
    summary = dict(
        method=method, seed=seed, status=rec.status(), reached_target=rec.hit is not None,
        target_rmse_M=target, budget_evals=budget, max_wall_s=max_wall_s,
        evals_to_target=hit.get("evals"), wall_s_to_target=hit.get("wall_s"),
        total_evals=len(rows), total_wall_s=float(rows[-1, 1]), total_solver_s=float(rows[-1, 2]),
        best_rmse_M=float(rows[best, 3]), theta_best=rows[best, 5:].tolist(), z0=list(map(float, z0)),
    )
    return summary, rows
