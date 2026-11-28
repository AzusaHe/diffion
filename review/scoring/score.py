import json, glob, os, math, numpy as np
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
R = str(ROOT / "results")
T = np.load(str(ROOT / "review/scoring/truth.npz")); TH, WIN = T["theta"], T["window"]
LDm, LDM, DREF, TM, TX = math.log(1e-12), math.log(2e-11), 1e-11, -0.25, 1.0
def B(u): return np.stack([(1-u)**3, 3*u*(1-u)**2, 3*u**2*(1-u), u**3], -1)
def curves(th, c):
    u = np.clip(c/2, 0, 1); b = B(u); th = np.asarray(th, float)
    return np.exp(LDm + b@th[:4]*(LDM-LDm))/DREF, TM + b@th[4:]*(TX-TM)
def score(th, k):
    c = np.linspace(WIN[k][0], WIN[k][1], 301); d, t = curves(th, c); dt, tt = curves(TH[k], c)
    return float(np.linalg.norm(d-dt)/np.linalg.norm(dt)*100), float(np.sqrt(np.mean((t-tt)**2)))
med = lambda v: float(np.median(v)) if len(v) else None

def folder(path, batch_div=False):
    out = {}
    for f in glob.glob(path + "/m*.json"):
        r = json.load(open(f)); k = r["k"]
        De, te = score(r["theta"], k); Ds, ts = score(r["theta0"], k)
        mis = r.get("rel_rmse", r.get("rmse", np.nan) / r.get("mean_c", 1.0) if "rmse" in r else None)
        wall = r.get("wall_s"); 
        if batch_div and wall is not None: wall = wall / r.get("batch", 16 if "batch" in r or True else 1)
        out[k] = dict(De=De, te=te, Ds=Ds, mis=mis, wall=wall, evals=r.get("evals"), steps=r.get("steps"))
    return out

def summarize(per, ks):
    rows = [per[k] for k in ks if k in per]
    if not rows: return None
    g = lambda key: [x[key] for x in rows if x.get(key) is not None and np.isfinite(x[key])]
    return dict(n=len(rows), Ds=med(g("Ds")), De=med(g("De")), te=med(g("te")),
                better=sum(x["De"] < x["Ds"] for x in rows if x.get("Ds") is not None) if rows[0].get("Ds") is not None else None,
                mis=med(g("mis")), wall=med(g("wall")), evals=med(g("evals")), steps=med(g("steps")))

runs = []
def add(group, name, inp, per, note=""):
    runs.append(dict(group=group, name=name, input=inp, per=per, note=note))

# learned nets: seed 0 (the seed whose FNO+TTT prediction is input A)
for arch in ("mlp", "fno"):
    sup = json.load(open(f"{R}/vanilla pinn/{arch}_s0/result.json"))
    ttt = json.load(open(f"{R}/vanilla pinn/ttt/{arch}_s0.json"))
    ps, pt = {}, {}
    for k in range(300):
        De, te = score(sup["test"]["theta"][k], k)
        ps[k] = dict(De=De, te=te, Ds=None, mis=None, wall=sup["test"]["infer_s_per_material"], evals=None)
        pt[k] = dict(De=ttt["scores"]["150"]["D_relL2_pct"][k], te=ttt["scores"]["150"]["tplus_rmse"][k], Ds=De, mis=None,
                     wall=ttt["wall_s_per_material"], evals=150)
    A = arch.upper()
    add("Learned (amortized)", f"{A}, supervised", "data", ps)
    add("Learned (amortized)", f"{A} + TTT", "data", pt)
for st, lab in (("A_mlp", "A · MLP"), ("A_mlp_ttt", "A · MLP+TTT"), ("A_fno", "A · FNO"), ("A_fno_ttt", "A · FNO+TTT"), ("B_center", "B · center")):
    add("Gradient (differentiable solver)", "L-BFGS-B", lab, folder(f"{R}/lbfgs/{st}"))
for m, nm in (("NM", "Nelder–Mead"), ("CMAES", "CMA-ES"), ("BO", "BO, 150 evals")):
    for st, lab in (("A_fno_ttt", "A · FNO+TTT"), ("B_center", "B · center")):
        add("Gradient-free (solver)", nm, lab, folder(f"{R}/gradfree/{st}/{m}"))
P = [("PINN total loss, JAX CPU", "pinn", ""), ("PINN total loss, GPU", "pinn_gpu", ""),
     ("PINN staged, w=100", "pinn_staged/w100", ""), ("PINN staged, w=1", "pinn_staged/w1", ""),
     ("PINN data-first, w_data=1e-2", "pinn_prefit/wd1e-2", ""), ("PINN data-first, w_data=1e-3", "pinn_prefit/wd1e-3", ""),
     ("PINN data-first, w_data=1e-4", "pinn_prefit/wd1e-4", ""),
     ("PINN frozen c,v (θ only)", "pinn_prefit/frozen_cv_wd1e-2", ""), ("PINN frozen c (v, θ trained), w_data=1e-2", "pinn_prefit/frozen_c_wd1e-2", "")]
for nm, d, _ in P:
    for b in ("cv", "c"):
        for st, lab in (("A_fno_ttt", "A · FNO+TTT"), ("B_center", "B · center")):
            p = f"{R}/{d}/{st}/{b}"
            if os.path.isdir(p):
                add("PINN (per-material network fit)", nm + (" · Build 1" if b == "cv" else " · Build 2"), lab, folder(p, batch_div=d != "pinn"))
out = []
for r in runs:
    allk = sorted(r["per"])
    out.append(dict(group=r["group"], name=r["name"], input=r["input"], s8=summarize(r["per"], range(8)), sall=summarize(r["per"], allk)))
json.dump(out, open(str(ROOT / "review/scoring/compare.json"), "w"), indent=1)
for o in out:
    a, b = o["s8"], o["sall"]
    f = lambda x: "-" if x is None else (f"{x:.3g}")
    print(f'{o["name"][:44]:44s} {o["input"]:12s} | 8: {f(a["Ds"])}->{f(a["De"])} t+{f(a["te"])} b{a["better"]} mis{f(a["mis"])} w{f(a["wall"])} | all n{b["n"]}: {f(b["Ds"])}->{f(b["De"])} t+{f(b["te"])} b{b["better"]} mis{f(b["mis"])} w{f(b["wall"])} ev{f(b["evals"])}')
