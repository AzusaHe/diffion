"""Two-stage PINN seeded from the FNO result (multi-stage training after Wang & Lai, J. Comput. Phys. 2024).

    python -m pinn.multistage --out results/pinn_multistage --w-data 1e-2,1

Stage 0 (no PINN): take theta0 = the FNO+TTT estimate (input A), run the solver once at theta0 on its native grid
  (40 cells x 104 times, solver/dataset_fine.py), and fit the c and v networks of Build 1 to that field (Adam, then
  L-BFGS; v from velocity_on_grid of the solver c).  These networks satisfy the physics at theta0 almost exactly,
  so their misfit to the measured data is mostly the theta error.
Stage 1 (correction): freeze them and add one correction network each,
      c = 1 + s [N_c(z) + eps_c N_c1(kappa_c * z)],   v = v_scale x [N_v(z) + eps_v N_v1(kappa_v * z)],
  z = (2x - 1, 2s - 1, i / 5).  eps is the RMS size of stage 0's misfit to the measured data (c bins, v from c),
  so the correction targets an O(1) function; kappa (per input) comes from the zero crossings of that misfit
  along x and along s (kappa = pi n / range of the input); the correction nets have a sin first layer and a zero
  last layer, so stage 1 starts exactly at stage 0.  Then the usual data-first joint loss on the measured
  benchmark data (13 frames x 20 bins): w_data (c and v data) + pde + vol + bc, correction nets and theta trained
  (Adam, then L-BFGS), stage-0 nets and eps, kappa fixed.
Results: <out>/wd<w>/A_fno_ttt/cv/m<k>.json (format of torch_pinn, plus eps, kappa and the stage-0 fit), and
<out>/table.md next to the data-first joint run from the same start (results/pinn_prefit/wd1e-2).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")        # the solver runs on the CPU; torch keeps the GPU

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.func import vmap  # noqa: E402


def solver_field(theta, iotas, K):
    """c (n, 3, 104, 40) from the solver at theta on its native grid; NaN past each material's last frame."""
    import jax.numpy as jnp
    from solver.dataset_fine import SUB, T_FINE, simulate_batch
    c, _ = simulate_batch(jnp.asarray(theta), jnp.asarray(iotas))
    c = np.asarray(c, dtype=float)
    keep = np.arange(len(T_FINE))[None, :] < (SUB * np.asarray(K))[:, None]
    return np.where(keep[:, None, :, None], c, np.nan), np.asarray(T_FINE)


def zero_crossings(r, axis):
    """Mean number of sign changes of r along `axis` (NaN-free slices only)."""
    sgn = np.sign(r)
    ch = (sgn[..., 1:] * sgn[..., :-1] < 0) if axis == -1 else (np.swapaxes(sgn, axis, -1)[..., 1:] *
                                                               np.swapaxes(sgn, axis, -1)[..., :-1] < 0)
    return float(np.mean(ch.sum(-1)))


def correction_scales(base, ys, iotas, device, dtype, kmax=20.0):
    """eps and kappa of the correction stage from stage 0's misfit to the measured data (one dict per fit)."""
    from pinn.torch_pinn import Losses, _leaves, _rebuild, _stack, problem_stats
    LS = Losses("cv", dtype, device)
    out = []
    for p0, y, io in zip(base, ys, iotas):
        S0 = problem_stats(y, io)
        S = {k: torch.as_tensor(np.asarray(S0[k]), device=device, dtype=dtype) for k in ("iotas", "tmax", "vscale")}
        p = _rebuild(p0, iter([z.to(device=device, dtype=dtype) for z in _leaves(p0)]))
        with torch.no_grad():
            pred = vmap(lambda i: vmap(lambda t: LS.bin_average(p, t, i, S["tmax"]))(LS.t_obs))(S["iotas"]).cpu().numpy()
            from pinn.torch_pinn import v_point
            vpred = vmap(lambda i: vmap(lambda t: vmap(lambda x: v_point(p, x, t, i, S["tmax"], S["vscale"]))(LS.XC))(
                LS.t_obs))(S["iotas"]).cpu().numpy()
        m = S0["mask"]
        rc = (y[:, m] - pred[:, m])                                      # (3, K, 20)
        rv = (S0["v_obs"][:, m] - vpred[:, m])
        s = np.sqrt(LS.t_obs.cpu().numpy()[m] / S0["tmax"])             # frames in s
        x = LS.XC.cpu().numpy()
        eps_c = float(np.sqrt(np.mean(rc ** 2)) / np.sqrt(np.mean(s ** 2)))
        eps_v = float(np.sqrt(np.mean(rv ** 2)) / (S0["vscale"] * np.sqrt(np.mean(x ** 2))))
        rng_s = 2.0 * (1.0 - S0["s_lo"])

        def kappa(r):
            nx, ns = zero_crossings(r, -1), zero_crossings(r, 1)
            return [float(np.clip(np.pi * nx / 2.0, 1.0, kmax)), float(np.clip(np.pi * ns / rng_s, 1.0, kmax)), 1.0]
        out.append(dict(eps_c=eps_c, eps_v=eps_v, k_c=kappa(rc), k_v=kappa(rv),
                        stage0_c_rel_rmse=float(np.sqrt(np.mean(rc ** 2)) / S0["mean_c"]),
                        stage0_v_rel_rmse=float(np.sqrt(np.mean(rv ** 2)) / S0["vscale"])))
    return out


def main():
    from inverse.lbfgs import load_test
    from pinn.jax_pinn import load_starts, summarize
    from pinn.torch_pinn import ObsGrid, _stack, init_mlp, train_batch
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("results/pinn_multistage"))
    p.add_argument("--start", default="fno_ttt")
    p.add_argument("--materials", default="0:8")
    p.add_argument("--w-data", default="1e-2,1", help="data weights of stage 1 (one run each)")
    p.add_argument("--fit-steps", type=int, default=30000, help="stage 0: Adam steps on the solver field")
    p.add_argument("--fit-lbfgs-iters", type=int, default=3000)
    p.add_argument("--steps", type=int, default=5000, help="stage 1: Adam steps (correction nets and theta)")
    p.add_argument("--lr", type=float, default=1e-3, help="stage 1 network learning rate")
    p.add_argument("--phi-lr-mult", type=float, default=1.0)
    p.add_argument("--lbfgs-iters", type=int, default=300)
    p.add_argument("--corr-width", type=int, default=32)
    p.add_argument("--corr-depth", type=int, default=3)
    p.add_argument("--n-res", type=int, default=1024)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    dev, dt = a.device, torch.float64
    T = load_test()
    lo, hi = map(int, a.materials.split(":"))
    ks = list(range(lo, hi))
    th0 = np.clip(load_starts(a.start, T)[ks], 1e-4, 1 - 1e-4)
    ys = [T["y"][k] for k in ks]
    iotas = [T["iotas"][k] for k in ks]
    K = [int(np.isfinite(y[0, :, 0]).sum()) for y in ys]
    a.out.mkdir(parents=True, exist_ok=True)
    pre = "A_" if a.start != "center" else "B_"

    # ---- stage 0: networks fitted to the solver field at theta0
    c_sol, t_fine = solver_field(th0, np.asarray(iotas), K)
    from solver.dataset import W_BINS
    from solver.dataset_fine import SUB
    frames = c_sol[:, :, SUB - 1::SUB, :] @ W_BINS.T                  # the solver at theta0 on the data grid
    sol_mis = [float(np.sqrt(np.nanmean((f - y) ** 2)) / np.nanmean(y)) for f, y in zip(frames, ys)]
    print("solver at theta0 vs measured data, rel rmse:", " ".join(f"{m:.2e}" for m in sol_mis), flush=True)
    cache, tag = a.out / "cache", f"_solver_{a.start}"
    print(f"stage 0: fitting {len(ks)} materials to the solver field at the {a.start} theta", flush=True)
    R0 = train_batch(list(c_sol), iotas, list(th0), "cv", ks, steps=0, lbfgs_iters=0, fit_steps=a.fit_steps,
                     fit_lbfgs_iters=a.fit_lbfgs_iters, fit_v=True, prefit_cache=cache, cache_tag=tag,
                     obs=ObsGrid.fine(t_fine), device=dev, dtype=dt, n_res=a.n_res,
                     progress=lambda i, w, r, s: print(f"  {s:8s} step {i:6d} {w:7.0f} s  median rel {np.median(r):.2e}",
                                                       flush=True))
    s0 = [dict(solver_fit_rel_rmse=r["fit_end_rel_rmse"], solver_fit_v_rel_rmse=r["fit_end_v_rel_rmse"],
               solver_vs_data_rel_rmse=m) for r, m in zip(R0, sol_mis)]
    ftag = f"cv_v1_f{a.fit_steps}_lr0.001_l{a.fit_lbfgs_iters}_w64x4_g{len(t_fine)}x40{tag}"
    base = [{k: v for k, v in torch.load(cache / f"{ftag}_s{k_}.pt", map_location="cpu").items() if k != "phi"}
            for k_ in ks]
    sc = correction_scales(base, ys, iotas, dev, dt)
    for k_, a0, b0 in zip(ks, s0, sc):
        print(f"  m{k_:03d} solver fit {a0['solver_fit_rel_rmse']:.1e}  data misfit at theta0 c {b0['stage0_c_rel_rmse']:.2e}"
              f" v {b0['stage0_v_rel_rmse']:.2e}  eps_c {b0['eps_c']:.2e} eps_v {b0['eps_v']:.2e}"
              f"  k_c {b0['k_c'][:2]} k_v {b0['k_v'][:2]}", flush=True)

    # ---- stage 1: frozen stage-0 nets + correction nets, theta free, on the measured data
    sizes = [3] + [a.corr_width] * a.corr_depth + [1]
    for wd in a.w_data.split(","):
        d = a.out / f"wd{wd}" / f"{pre}{a.start}" / "cv"
        d.mkdir(parents=True, exist_ok=True)
        todo = [i for i, k_ in enumerate(ks) if not (d / f"m{k_:03d}.json").exists()]
        if not todo:
            continue
        init = []
        for i in todo:
            g = torch.Generator().manual_seed(1000 + ks[i])
            c1, v1 = init_mlp(g, sizes, dt), init_mlp(g, sizes, dt)
            for net in (c1, v1):                                        # zero last layer: start at stage 0
                net[-1] = (torch.zeros_like(net[-1][0]), torch.zeros_like(net[-1][1]))
            ms = {kk: torch.tensor(sc[i][kk], dtype=dt) for kk in ("eps_c", "eps_v", "k_c", "k_v")}
            init.append(dict(c=base[i]["c"], v=base[i]["v"], ms=ms, c1=c1, v1=v1))
        print(f"stage 1, w_data {wd}: {len(todo)} fits", flush=True)
        R = train_batch([ys[i] for i in todo], [iotas[i] for i in todo], [th0[i] for i in todo], "cv",
                        [ks[i] for i in todo], steps=a.steps, lr=a.lr, phi_lr_mult=a.phi_lr_mult,
                        w=dict(data=float(wd), pde=1.0, vol=1.0, bc=1.0), fit_v=True, lbfgs_iters=a.lbfgs_iters,
                        freeze=("c", "v", "ms"), init_params=init, device=dev, dtype=dt, n_res=a.n_res, log_every=100,
                        progress=lambda i, w, r, s: print(f"  {s:8s} step {i:6d} {w:7.0f} s  median rel {np.median(r):.2e}",
                                                          flush=True))
        for i, r in zip(todo, R):
            r.update(k=ks[i], index=int(T["index"][ks[i]]), start=a.start, theta0=th0[i].tolist(), data="benchmark",
                     multistage=dict(**s0[i], **sc[i], corr_sizes=sizes))
            (d / f"m{ks[i]:03d}.json").write_text(json.dumps(r))

    # ---- table
    rows = [("data-first joint w_data 1e-2 (earlier)", Path("results/pinn_prefit/wd1e-2") / f"{pre}{a.start}" / "cv")]
    rows += [(f"two-stage from solver field, w_data {wd}", a.out / f"wd{wd}" / f"{pre}{a.start}" / "cv")
             for wd in a.w_data.split(",")]
    lines = ["| run | D err % (start -> fit) | better | t+ rmse (start -> fit) | c misfit % |", "|---|---|---|---|---|"]
    for name, d in rows:
        if not list(d.glob("m*.json")):
            continue
        s = summarize(d)
        lines.append(f"| {name} | {s['median_start']['D_relL2_pct']:.2f} -> {s['median_fit']['D_relL2_pct']:.2f} | "
                     f"{s['D_err_improved']}/{s['n']} | {s['median_start']['tplus_rmse']:.3f} -> "
                     f"{s['median_fit']['tplus_rmse']:.3f} | {100 * s['median_rel_rmse']:.3f} |")
    lines += ["", f"stage 0 (solver field at the {a.start} theta), medians: solver vs measured data "
              f"{100 * np.median(sol_mis):.3f} %; network fit to the solver field "
              f"{np.median([x['solver_fit_rel_rmse'] for x in s0]):.1e}; misfit to the measured data "
              f"c {100 * np.median([x['stage0_c_rel_rmse'] for x in sc]):.3f} %, "
              f"v {100 * np.median([x['stage0_v_rel_rmse'] for x in sc]):.3f} %"]
    txt = "\n".join(lines)
    (a.out / "table.md").write_text(txt + "\n")
    print(txt, flush=True)


if __name__ == "__main__":
    main()
