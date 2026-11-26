"""Does the theta bias come from the v data target?  (test V1)

diag_truth scored the physics loss of the stage-1 networks, whose v was fitted to v_obs, the velocity rebuilt
from the 13 binned frames (surrogate.data.velocity_from_c).  That target is off by 2-4 % of v_scale (RMS) from the
v of the native-grid solver, yet the stage-1 fit holds it to ~0.03 %.  Here the c network stays at its cached
stage-1 fit and theta is held fixed, and only the v network is retrained on the physics alone (no c data, no v
data): 3000 Adam + 300 L-BFGS from the stage-1 v.  This is done at each theta of diag_truth (the truth, starts A
and B, and the thetas the theta-only fit frozen_cv_wd1e-2 and the data-first joint fit wd1e-2 ended at), and the physics loss pde + vol + bc (weights 1)
is then scored on fresh residual points.  If the truth now scores lowest, v_obs was the source of the bias.

    python -m pinn.diag_vfree --out results/pinn_prefit/diag_vfree
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.func import vmap


def main():
    from analysis.metrics import score_all
    from inverse.lbfgs import load_test
    from pinn.jax_pinn import load_starts
    from pinn.torch_pinn import Losses, _leaves, _rebuild, _stack, problem_stats, train_batch
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("results/pinn_prefit/diag_vfree"))
    p.add_argument("--prefit", type=Path, default=Path("results/pinn_prefit"))
    p.add_argument("--materials", default="0:8")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lbfgs-iters", type=int, default=300)
    p.add_argument("--n-eval", type=int, default=4096, help="fresh residual points per current for the final score")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    dev, dt = a.device, torch.float64
    T = load_test()
    lo, hi = map(int, a.materials.split(":"))
    ks = list(range(lo, hi))
    B = len(ks)
    truth = np.asarray(T["theta"])[ks]

    def from_run(run, start):
        return np.array([json.loads((a.prefit / run / start / "cv" / f"m{k:03d}.json").read_text())["theta"] for k in ks])
    thetas = {"truth": truth, "start A (FNO+TTT)": load_starts("fno_ttt", T)[ks], "start B (center)": load_starts("center", T)[ks],
              "theta-only fit, from A": from_run("frozen_cv_wd1e-2", "A_fno_ttt"),
              "theta-only fit, from B": from_run("frozen_cv_wd1e-2", "B_center"),
              "joint fit w_data 1e-2, from A": from_run("wd1e-2", "A_fno_ttt"),
              "joint fit w_data 1e-2, from B": from_run("wd1e-2", "B_center")}

    st = [problem_stats(T["y"][k], T["iotas"][k]) for k in ks]
    S = {k: torch.tensor(np.stack([s[k] for s in st]), device=dev, dtype=dt)
         for k in ("y", "mask", "v_obs", "vmask", "iotas", "tmax", "vscale", "st", "sigma", "s_lo")}
    S["mask"], S["vmask"] = S["mask"].bool(), S["vmask"].bool()
    LS = Losses("cv", dt, dev, n_res=a.n_eval, w=dict(data=0.0, vdata=0.0, pde=1.0, vol=1.0, bc=1.0))
    LS.use_vdata = True
    gen = torch.Generator(device=dev).manual_seed(123)
    x = torch.rand(B, 3, a.n_eval, generator=gen, device=dev, dtype=dt)
    u = torch.rand(B, 3, a.n_eval, generator=gen, device=dev, dtype=dt)
    lo_s = S["s_lo"][:, None, None]
    t = S["tmax"][:, None, None] * (lo_s + (1 - lo_s) * u) ** 2
    tag = "cv_v1_f30000_lr0.001_l3000_w64x4"
    nets0 = _stack([torch.load(a.prefit / "cache" / f"{tag}_s{k}.pt", map_location="cpu") for k in ks])
    nets0 = _rebuild(nets0, iter([z.to(device=dev, dtype=dt) for z in _leaves(nets0)]))

    def score(params, th):
        pp = {kk: v for kk, v in params.items() if kk != "phi"}
        thc = np.clip(th, 1e-4, 1 - 1e-4)
        pp["phi"] = torch.tensor(np.log(thc / (1 - thc)), device=dev, dtype=dt)
        _, parts = vmap(LS.loss)(pp, S, x, t)
        parts = {kk: v.detach().cpu().numpy() for kk, v in parts.items()}
        return {kk: parts[kk] for kk in ("pde", "vol", "bc", "vdata")}

    out = {}
    for name, th in thetas.items():
        print(f"== {name}", flush=True)
        before = score(nets0, th)
        R, P = train_batch([T["y"][k] for k in ks], [T["iotas"][k] for k in ks], list(np.clip(th, 1e-4, 1 - 1e-4)), "cv",
                           ks, steps=a.steps, lr=a.lr, phi_lr_mult=0.0, w=dict(data=0.0, vdata=0.0, pde=1.0, vol=1.0, bc=1.0),
                           fit_v=True, fit_steps=30000, fit_lbfgs_iters=3000, prefit_cache=a.prefit / "cache",
                           freeze=("c", "phi"), lbfgs_iters=a.lbfgs_iters, device=dev, dtype=dt, log_every=500,
                           return_params=True, progress=lambda i, w, r, s: print(f"  {s:6s} step {i:5d} {w:6.0f} s", flush=True))
        after = score(P, th)
        moved = float(np.max(np.abs(np.array([r["theta"] for r in R]) - np.clip(th, 1e-4, 1 - 1e-4))))
        sc = score_all(th, truth, np.asarray(T["window"])[ks])
        out[name] = dict(before={kk: v.tolist() for kk, v in before.items()}, after={kk: v.tolist() for kk, v in after.items()},
                         D_err=np.asarray(sc["D_relL2_pct"]).tolist(), tplus_rmse=np.asarray(sc["tplus_rmse"]).tolist(),
                         theta_max_change=moved, v_rel_rmse_end=[r["v_rel_rmse"] for r in R])
        phys = lambda d: d["pde"] + d["vol"] + d["bc"]  # noqa: E731
        print(f"  physics before {np.median(phys(before)):.3e}  after {np.median(phys(after)):.3e}  "
              f"(theta moved by {moved:.1e})", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "diag_vfree.json").write_text(json.dumps(out, indent=1))

    phys = lambda d: np.array(d["pde"]) + np.array(d["vol"]) + np.array(d["bc"])  # noqa: E731
    trA, trB = phys(out["truth"]["after"]), phys(out["truth"]["before"])
    lines = [f"### Physics loss (pde + vol + bc) at each theta, median over {B} materials", "",
             "before = the stage-1 networks (v fitted to v_obs, as in diag_truth); after = v retrained on the physics "
             "alone at that theta (c and theta fixed).  Ratio and count are against the truth row of the same column.", "",
             "| theta | D err % | t+ rmse | before: loss | before: / truth | before: truth lower | after: loss | after: pde | "
             "after: vol | after: bc | after: / truth | after: truth lower |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in out.items():
        b, f = phys(r["before"]), phys(r["after"])
        lines.append(f"| {name} | {np.median(r['D_err']):.1f} | {np.median(r['tplus_rmse']):.3f} | {np.median(b):.3e} | "
                     f"{np.median(b / trB):.3f} | {int((trB < b).sum())}/{B} | {np.median(f):.3e} | "
                     f"{np.median(r['after']['pde']):.2e} | {np.median(r['after']['vol']):.2e} | {np.median(r['after']['bc']):.2e} | "
                     f"{np.median(f / trA):.3f} | {int((trA < f).sum())}/{B} |")
    lines += ["", "Per material, physics loss after the v retraining (pde + vol + bc):", "",
              "| theta | " + " | ".join(f"m{k:03d}" for k in ks) + " |", "|---|" + "---|" * B]
    for name, r in out.items():
        lines.append(f"| {name} | " + " | ".join(f"{z:.2e}" for z in phys(r["after"])) + " |")
    lowest = [min(out, key=lambda n: phys(out[n]["after"])[i]) for i in range(B)]
    lines += ["", "Lowest after the v retraining, per material: " + ", ".join(f"m{k:03d} {n}" for k, n in zip(ks, lowest))]
    txt = "\n".join(lines)
    (a.out / "diag_vfree.md").write_text(txt + "\n")
    print(txt, flush=True)


if __name__ == "__main__":
    main()
