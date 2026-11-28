"""Is the PINN's physics loss lowest at the true theta?  (tuning test 1)

Takes the stage-1 networks (c and v fitted to the data alone, results/pinn_prefit/cache) and evaluates the
physics loss L_pde + L_vol + L_bc (weights 1, one large fixed set of residual points) at several thetas:
the truth, the two starts (A = FNO+TTT, B = center) and the thetas the earlier runs ended at.  With the networks
frozen this is exactly the objective the theta-only fit (frozen_cv_wd1e-2) minimized, so if a fitted theta scores
lower than the truth, the objective itself prefers the wrong D and t+ and no optimizer tuning can fix it.
Both residual forms are evaluated: strong (pointwise) and fv (integral form, test 2).

    python -m pinn.diag_truth --out results/pinn_prefit/diag_truth
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
    from pinn.torch_pinn import Losses, _rebuild, _leaves, _stack, problem_stats
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("results/pinn_prefit/diag_truth"))
    p.add_argument("--prefit", type=Path, default=Path("results/pinn_prefit"))
    p.add_argument("--cache", type=Path, default=None, help="stage-1 network cache (default <prefit>/cache)")
    p.add_argument("--tag", default="cv_v1_f30000_lr0.001_l3000_w64x4")
    p.add_argument("--materials", default="0:8")
    p.add_argument("--n-res", type=int, default=4096)
    p.add_argument("--n-fv-times", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    dev, dt = a.device, torch.float64
    T = load_test()
    lo, hi = map(int, a.materials.split(":"))
    ks = list(range(lo, hi))
    B = len(ks)
    st = [problem_stats(T["y"][k], T["iotas"][k]) for k in ks]
    S = {k: torch.tensor(np.stack([s[k] for s in st]), device=dev, dtype=dt)
         for k in ("y", "mask", "v_obs", "vmask", "iotas", "tmax", "vscale", "st", "sigma", "s_lo")}
    S["mask"], S["vmask"] = S["mask"].bool(), S["vmask"].bool()
    nets = _stack([torch.load((a.cache or a.prefit / "cache") / f"{a.tag}_s{k}.pt", map_location="cpu") for k in ks])
    nets = _rebuild(nets, iter([z.to(device=dev, dtype=dt) for z in _leaves(nets)]))

    def from_runs(run, start):
        th = []
        for k in ks:
            f = a.prefit / run / start / "cv" / f"m{k:03d}.json"
            th.append(json.loads(f.read_text())["theta"] if f.exists() else [np.nan] * 8)
        return np.array(th)
    thetas = {"truth": np.asarray(T["theta"])[ks], "start A (FNO+TTT)": load_starts("fno_ttt", T)[ks],
              "start B (center)": load_starts("center", T)[ks]}
    for run, label in (("frozen_cv_wd1e-2", "theta-only fit"), ("wd1e-2", "joint fit w_data 1e-2")):
        for start in ("A_fno_ttt", "B_center"):
            thetas[f"{label}, from {start[0]}"] = from_runs(run, start)

    gen = torch.Generator(device=dev).manual_seed(123)
    x = torch.rand(B, 3, a.n_res, generator=gen, device=dev, dtype=dt)
    u = torch.rand(B, 3, a.n_res, generator=gen, device=dev, dtype=dt)
    lo_s = S["s_lo"][:, None, None]
    t = S["tmax"][:, None, None] * (lo_s + (1 - lo_s) * u) ** 2
    out = {}
    for residual in ("strong", "fv"):
        LS = Losses("cv", dt, dev, n_res=a.n_res, w=dict(data=0.0, pde=1.0, vol=1.0, bc=1.0), residual=residual,
                    n_fv_times=a.n_fv_times)
        LS.use_vdata = False
        out[residual] = {}
        for name, th in thetas.items():
            if np.isnan(th).any():
                continue
            pp = dict(nets)
            pp["phi"] = torch.tensor(np.log(np.clip(th, 1e-4, 1 - 1e-4) / (1 - np.clip(th, 1e-4, 1 - 1e-4))),
                                     device=dev, dtype=dt)
            with torch.no_grad():
                _, parts = vmap(LS.loss)(pp, S, x, t)
            parts = {k: v.cpu().numpy() for k, v in parts.items() if k != "data"}
            phys = parts["pde"] + parts["vol"] + parts["bc"]
            sc = score_all(th, np.asarray(T["theta"])[ks], np.asarray(T["window"])[ks])
            out[residual][name] = dict(phys=phys.tolist(), **{k: v.tolist() for k, v in parts.items()},
                                       D_err=sc["D_relL2_pct"], tplus_rmse=sc["tplus_rmse"])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "diag_truth.json").write_text(json.dumps(out, indent=1))
    lines = []
    for residual, R in out.items():
        tr = np.array(R["truth"]["phys"])
        lines += [f"\n### {residual} residual: physics loss (pde + vol + bc) on the stage-1 networks, median over "
                  f"{B} materials", "",
                  "| theta | D err % | t+ rmse | physics loss | loss / loss at truth | materials where truth is lower |",
                  "|---|---|---|---|---|---|"]
        for name, r in R.items():
            ph = np.array(r["phys"])
            lines.append(f"| {name} | {np.median(r['D_err']):.1f} | {np.median(r['tplus_rmse']):.3f} | "
                         f"{np.median(ph):.3e} | {np.median(ph / tr):.3f} | {int((tr < ph).sum())}/{B} |")
    txt = "\n".join(lines)
    (a.out / "diag_truth.md").write_text(txt + "\n")
    print(txt, flush=True)


if __name__ == "__main__":
    main()
