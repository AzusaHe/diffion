"""Is the PINN loss lowest at the truth when the networks are trained for it?  (fair loss-at-truth test)

Scores the final networks of several runs (saved with torch_pinn --save-params) on ONE fresh set of residual points
(same seed for every run), under two losses of the data-first joint stage, w_data 1e-2 and physics weights 1:
    loss-cur  w_data (L_c + L_v) + L_pde + L_vol + L_bc      (w_vdata 1, every run so far)
    loss-new  w_data L_c + L_pde + L_vol + L_bc               (w_vdata 0, V2)
truthfixed_* runs start at the true theta and hold it fixed while the networks train as usual, so they give the
loss at the truth with networks fitted for it; the others are the joint runs from the truth, A and B.

    python -m pinn.loss_at_truth --out results/pinn_prefit/loss_at_truth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.func import vmap

RUNS = {"loss-cur": [("truth fixed, nets trained (T-cur)", "truthfixed_wd1e-2/A_truth"),
                     ("joint from truth", "rerun_wd1e-2/A_truth"),
                     ("joint from A", "rerun_wd1e-2/A_fno_ttt"),
                     ("joint from B", "rerun_wd1e-2/B_center")],
        "loss-new": [("truth fixed, nets trained (T-new)", "truthfixed_novdata_wd1e-2/A_truth"),
                     ("joint from A, no v data (V2)", "novdata_wd1e-2/A_fno_ttt"),
                     ("joint from B, no v data (V2)", "novdata_wd1e-2/B_center")]}


def main():
    from analysis.metrics import score_all
    from inverse.lbfgs import load_test
    from pinn.torch_pinn import Losses, _leaves, _rebuild, _stack, problem_stats
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("results/pinn_prefit/loss_at_truth"))
    p.add_argument("--prefit", type=Path, default=Path("results/pinn_prefit"))
    p.add_argument("--materials", default="0:8")
    p.add_argument("--w-data", type=float, default=1e-2)
    p.add_argument("--n-eval", type=int, default=4096)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    dev, dt = a.device, torch.float64
    T = load_test()
    lo, hi = map(int, a.materials.split(":"))
    ks = list(range(lo, hi))
    B = len(ks)
    truth = np.asarray(T["theta"])[ks]
    st = [problem_stats(T["y"][k], T["iotas"][k]) for k in ks]
    S = {k: torch.tensor(np.stack([s[k] for s in st]), device=dev, dtype=dt)
         for k in ("y", "mask", "v_obs", "vmask", "iotas", "tmax", "vscale", "st", "sigma", "s_lo")}
    S["mask"], S["vmask"] = S["mask"].bool(), S["vmask"].bool()
    LS = Losses("cv", dt, dev, n_res=a.n_eval, w=dict(data=1.0, vdata=1.0, pde=1.0, vol=1.0, bc=1.0))
    LS.use_vdata = True
    gen = torch.Generator(device=dev).manual_seed(123)
    x = torch.rand(B, 3, a.n_eval, generator=gen, device=dev, dtype=dt)
    u = torch.rand(B, 3, a.n_eval, generator=gen, device=dev, dtype=dt)
    lo_s = S["s_lo"][:, None, None]
    t = S["tmax"][:, None, None] * (lo_s + (1 - lo_s) * u) ** 2

    out = {}
    for loss_name, runs in RUNS.items():
        for name, rel in runs:
            d = a.prefit / rel / "cv"
            files = [d / f"m{k:03d}_params.pt" for k in ks]
            if not all(f.exists() for f in files):
                print(f"skip {name}: missing {d}", flush=True)
                continue
            P = _stack([torch.load(f, map_location="cpu") for f in files])
            P = _rebuild(P, iter([z.to(device=dev, dtype=dt) for z in _leaves(P)]))
            _, parts = vmap(LS.loss)(P, S, x, t)
            parts = {k: v.detach().cpu().numpy() for k, v in parts.items()}
            phys = parts["pde"] + parts["vol"] + parts["bc"]
            th = torch.sigmoid(P["phi"]).cpu().numpy()
            sc = score_all(th, truth, np.asarray(T["window"])[ks])
            out[name] = dict(loss=loss_name, phys=phys.tolist(), c_data=(a.w_data * parts["data"]).tolist(),
                             v_data=(a.w_data * parts["vdata"]).tolist(),
                             total_cur=(phys + a.w_data * (parts["data"] + parts["vdata"])).tolist(),
                             total_new=(phys + a.w_data * parts["data"]).tolist(),
                             D_err=np.asarray(sc["D_relL2_pct"]).tolist(), tplus_rmse=np.asarray(sc["tplus_rmse"]).tolist())
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "loss_at_truth.json").write_text(json.dumps(out, indent=1))

    lines = []
    for loss_name, key, ref in (("loss-cur (w_vdata 1)", "total_cur", "truth fixed, nets trained (T-cur)"),
                                ("loss-new (w_vdata 0)", "total_new", "truth fixed, nets trained (T-new)")):
        rows = [n for n, r in out.items() if r["loss"] == ("loss-cur" if key == "total_cur" else "loss-new")]
        if ref not in out:
            continue
        tr = np.array(out[ref][key])
        lines += [f"\n### {loss_name}: total loss on one fresh residual set ({a.n_eval} points per current, seed 123), "
                  f"median over {B} materials", "",
                  "| run | D err % | t+ rmse | total | physics | c data | v data | total / truth-fixed | truth-fixed lower |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for n in rows:
            r = out[n]
            tot = np.array(r[key])
            vd = np.median(r["v_data"]) if key == "total_cur" else float("nan")
            lines.append(f"| {n} | {np.median(r['D_err']):.2f} | {np.median(r['tplus_rmse']):.3f} | {np.median(tot):.3e} | "
                         f"{np.median(r['phys']):.3e} | {np.median(r['c_data']):.3e} | {vd:.3e} | "
                         f"{np.median(tot / tr):.3f} | {int((tr < tot).sum())}/{B} |")
        lines += ["", "Per material, total loss:", "", "| run | " + " | ".join(f"m{k:03d}" for k in ks) + " |",
                  "|---|" + "---|" * B]
        for n in rows:
            lines.append(f"| {n} | " + " | ".join(f"{z:.2e}" for z in out[n][key]) + " |")
        low = [min(rows, key=lambda n: out[n][key][i]) for i in range(B)]
        lines += ["", "Lowest per material: " + ", ".join(f"m{k:03d} {n}" for k, n in zip(ks, low))]
    txt = "\n".join(lines)
    (a.out / "loss_at_truth.md").write_text(txt + "\n")
    print(txt, flush=True)


if __name__ == "__main__":
    main()
