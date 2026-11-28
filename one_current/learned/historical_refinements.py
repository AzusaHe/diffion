"""Rescore the archived three-current inverse fits on exactly the paired test materials.

Run from diffion-ms with python one_current/learned/historical_refinements.py.
Output names match the one-current scoring script; the pilot warm start uses theta0.
"""
from pathlib import Path
import json
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from one_current.score_results import curves, save_csv

RESULTS = REPO / "results"
JUNBO = REPO.parent / "diffion-junbo" / "results"
OUT = REPO / "one_current" / "random" / "report_data"


def cohorts():
    for start in ("center", "mlp", "fno", "mlp_ttt", "fno_ttt"):
        code = "B_center" if start == "center" else "A_" + start
        yield "lbfgs_"+start, RESULTS/"lbfgs"/code, 300, "theta"
    yield "fno_ttt_pilot", RESULTS/"pinn_gpu"/"A_fno_ttt"/"cv", 8, "theta0"
    for start, code in (("center","B_center"),("fno_ttt","A_fno_ttt")):
        for opt in ("NM","CMAES","BO"):
            yield f"{opt}_{start}", RESULTS/"gradfree"/code/opt, 30 if opt=="BO" else 300, "theta"
        for build in ("c","cv"):
            yield f"pinn_standard_{build}_{start}", RESULTS/"pinn_gpu"/code/build, 8, "theta"
            for weight in ("w1","w100"):
                yield f"pinn_staged_{build}_{weight}_{start}", RESULTS/"pinn_staged"/weight/code/build, 8, "theta"
        yield "pinn_data_first_"+start, RESULTS/"pinn_prefit"/"wd1e-2"/code/"cv", 8, "theta"
        yield "pinn_fine_"+start, RESULTS/"pinn_fine"/"wd1e-2"/code/"cv", 8, "theta"
    yield "pinn_data_correction_fno_ttt", JUNBO/"pinn_prefit"/"ms2_wd1e-2"/"A_fno_ttt"/"cv", 8, "theta"
    for label,weight in (("solver_correction","wd1e-2"),("solver_correction_weight1","wd1")):
        yield f"pinn_{label}_fno_ttt", RESULTS/"pinn_multistage"/weight/"A_fno_ttt"/"cv", 8, "theta"


def main():
    original = REPO/"data"
    selected = np.flatnonzero(np.load(original/"train_split.npy") == "id_test")[:300]
    with np.load(original/"train.npz") as d:
        ids, theta, window = (d[key][selected] for key in ("index","theta","window"))
    with np.load(REPO/"one_current"/"random"/"data"/"train.npz") as d:
        np.testing.assert_array_equal(ids,d["index"][selected])
        np.testing.assert_array_equal(theta,d["theta"][selected])
    rows=[]
    for name,folder,n,field in cohorts():
        files = sorted(folder.glob("m[0-9][0-9][0-9].json"))
        if len(files)!=n:
            raise ValueError(f"{name}: expected {n} material records, found {len(files)}")
        records = [json.loads(p.read_text()) for p in files]
        by_id = {r["index"]:r for r in records}
        if len(by_id)!=n or set(by_id)!=set(ids[:n]):
            raise ValueError(f"{name}: wrong material IDs")
        joined=[by_id[mid] for mid in ids[:n]]
        np.testing.assert_array_equal([r["k"] for r in joined],np.arange(n))
        estimate=np.asarray([r[field] for r in joined])
        d,t=curves(estimate,theta[:n],window[:n])
        observed = [float(np.median(d)),float(np.median(t))]
        summary=json.loads((folder/"summary.json").read_text())
        archive=summary["median_fit" if field=="theta" else "median_start"]
        np.testing.assert_allclose(observed,[archive["D_relL2_pct"],archive["tplus_rmse"]],rtol=1e-8,atol=1e-12)
        rows.append(dict(method=name,n=n,D_common_pct=observed[0],t_common_rmse=observed[1]))
    OUT.mkdir(parents=True,exist_ok=True)
    save_csv(OUT/"historical_refinements.csv",rows)
    print(f"Verified {len(rows)} source-ID matched groups against archived medians.")


if __name__ == "__main__":
    main()
