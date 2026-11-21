"""Every number quoted in the progress report, from results/stage2.

    python -m analysis.summary            -> results/stage2/summary.txt and summary.json
"""
from __future__ import annotations

import json

import numpy as np

from .figures import REDUCTION, error_reduction
from .load import ARCHS, NAME, RESULTS, SEEDS, Results

METRICS = ("mse_lnD", "mse_tplus", "D_relL2_pct", "group_rmse")


def paired_fno_vs_mlp(R, ttt, metric):
    """Same materials, errors averaged over seeds: share of materials where the FNO is lower, and the
    median FNO / MLP error ratio."""
    mlp = np.mean([R.per_material("mlp", ttt, s, metric) for s in SEEDS], 0)
    fno = np.mean([R.per_material("fno", ttt, s, metric) for s in SEEDS], 0)
    return dict(fno_lower_share=float(np.mean(fno < mlp)), median_ratio=float(np.exp(np.median(np.log(fno / mlp)))))


def main():
    R = Results()
    out = dict(n_test_materials=len(R.per_material("fno", False, 0, "mse_lnD")), seeds=list(SEEDS),
               pde_loss_true_coefficients=R.truth_pde_loss, models={}, ttt={}, fno_vs_mlp={}, cost={})
    for a in ARCHS:
        for ttt in (False, True):
            out["models"][f"{a}_{'ttt' if ttt else 'sup'}"] = {m: R.seed_stat(a, ttt, m) for m in METRICS}
        b, t = R.pooled(a, False, "mse_lnD"), R.pooled(a, True, "mse_lnD")
        out["ttt"][a] = dict(
            lr=R.select[a]["lr"], steps=R.steps,
            error_reduction={m: float(np.mean(error_reduction(R, a, m, f))) for m, _, f in REDUCTION},
            improved_share=float(np.mean(t < b)), halved_share=float(np.mean(t < b / 2)),
            trajectory={m: {k: R.seed_stat(a, True, m, k)[0] for k in R.checkpoints} for m in ("mse_lnD", "pde_loss")})
        out["cost"][a] = dict(n_params=R.timing[a]["n_params"], train_minutes=R.train_minutes(a),
                              train_flops=R.sup[(a, 0)]["flops_train"],
                              best_val_coef_mse=[R.sup[(a, s)]["best_val_coef_mse"] for s in SEEDS],
                              ttt_s_per_material=float(np.mean([R.ttt[(a, s)]["wall_s_per_material"] for s in SEEDS])))
    for ttt in (False, True):
        out["fno_vs_mlp"]["ttt" if ttt else "sup"] = {m: paired_fno_vs_mlp(R, ttt, m) for m in ("mse_lnD", "mse_tplus")}

    L = [f"Stage 2: {out['n_test_materials']} test materials x {len(SEEDS)} seeds.  Median over materials per seed; "
         "mean of the seed medians [min, max].", ""]
    for m in METRICS:
        L.append(m)
        for key, v in out["models"].items():
            L.append(f"  {key:8s} {v[m][0]:.4g} [{v[m][1]:.3g}, {v[m][2]:.3g}]")
    L.append("")
    for a in ARCHS:
        t = out["ttt"][a]
        L.append(f"{NAME[a]} TTT: lr {t['lr']:g}, {t['steps']} steps; error reduction " +
                 ", ".join(f"{k} {v:.2f}x" for k, v in t["error_reduction"].items()) +
                 f"; improved {100 * t['improved_share']:.0f} %, halved {100 * t['halved_share']:.0f} %")
        for m, traj in t["trajectory"].items():
            L.append(f"   {m} vs step: " + "  ".join(f"{k}:{v:.4g}" for k, v in traj.items()))
    L.append(f"PDE loss at the true coefficients: {out['pde_loss_true_coefficients']:.4g}")
    L.append("")
    for stage, d in out["fno_vs_mlp"].items():
        for m, v in d.items():
            L.append(f"FNO vs MLP ({stage}, {m}): FNO lower on {100 * v['fno_lower_share']:.0f} % of materials, "
                     f"median ratio {v['median_ratio']:.2f}")
    L.append("")
    for a, c in out["cost"].items():
        L.append(f"{NAME[a]}: {c['n_params']:,} parameters, training {c['train_minutes']:.1f} min ({c['train_flops']:.2g} FLOPs), "
                 f"best val coef MSE {min(c['best_val_coef_mse']):.4f}-{max(c['best_val_coef_mse']):.4f}, "
                 f"TTT {c['ttt_s_per_material']:.2f} s/material")
    (RESULTS / "summary.txt").write_text("\n".join(L) + "\n")
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
