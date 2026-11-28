"""Summarize completed one-current solver records without rerunning a fit."""
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from .run import DATA, OUT, atomic_json, outfile, task_list


def median(values):
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(finite)) if finite else None


def main():
    groups = {}
    for task in task_list():
        groups.setdefault(task[:2], []).append(task)
    rows, stratified = [], []
    if (DATA / "current_rank.npy").exists():
        test_rows = np.flatnonzero(np.load(DATA / "train_split.npy") == "id_test")[:300]
        current_rank = np.load(DATA / "current_rank.npy")[test_rows]
    else:
        current_rank = None
    for (method, start), tasks in groups.items():
        existing = [outfile(t) for t in tasks if outfile(t).is_file()]
        records = [json.loads(path.read_text()) for path in existing]
        row = dict(method=method, start=start, expected=len(tasks), completed=len(records))
        if records and method == "compact":
            row.update(reached=sum(r["reached_target"] for r in records),
                       median_evals=median([r["total_evals"] for r in records]),
                       median_wall_s=median([r["total_wall_s"] for r in records]),
                       median_rel_rmse_c=median([r["best_rmse_M"]/(r["target_rmse_M"]/r["target_rel"]) for r in records]))
        elif records:
            scores = [r["scores"] for r in records]
            by_target = {}
            for rel in (1e-2,1e-3,1e-4):
                hits = [r["to_target"][str(rel)] for r in records]
                by_target[str(rel)] = dict(reached=sum(h[0] is not None for h in hits),
                                           median_evals=median([h[0] for h in hits]),median_wall_s=median([h[1] for h in hits]))
            row.update(reached=by_target["0.0001"]["reached"],median_evals=median([r["evals"] for r in records]),
                       median_wall_s=median([r["wall_s"] for r in records]),
                       median_rel_rmse_c=median([r["rmse"]/r["mean_c"] for r in records]),
                       median_D_relL2_pct=median([s["D_relL2_pct"] for s in scores]),
                       median_tplus_rmse=median([s["tplus_rmse"] for s in scores]),
                       median_group_rmse=median([s["group_rmse"] for s in scores]),
                       p95_D_relL2_pct=float(np.percentile([s["D_relL2_pct"] for s in scores],95)),
                       max_D_relL2_pct=max(s["D_relL2_pct"] for s in scores))
            atomic_json(existing[0].parent/"summary.json",dict(**row, targets=by_target,
                        status=dict(Counter(r["status"] for r in records))))
            if current_rank is not None:
                for rank, label in enumerate(("low","middle","high")):
                    arm = [r for r in records if current_rank[r["k"]] == rank]
                    if arm:
                        stratified.append(dict(method=method,start=start,stratum=label,n=len(arm),
                            median_D_relL2_pct=median([r["scores"]["D_relL2_pct"] for r in arm]),
                            median_tplus_rmse=median([r["scores"]["tplus_rmse"] for r in arm]),
                            median_rel_rmse_c=median([r["rmse"]/r["mean_c"] for r in arm]),
                            median_evals=median([r["evals"] for r in arm]),
                            target_reached=sum(r["to_target"]["0.0001"][0] is not None for r in arm)))
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    for dest, table in ((OUT/"summary.csv", rows), (OUT/"by_current_stratum.csv", stratified)):
        if not table:
            continue
        temp = dest.with_suffix(".csv.tmp")
        keys = list(dict.fromkeys(key for row in table for key in row))
        with temp.open("w", newline="") as f:
            writer = csv.DictWriter(f,keys)
            writer.writeheader()
            writer.writerows(table)
        temp.replace(dest)
    print(f"{sum(r['completed'] for r in rows)}/{sum(r['expected'] for r in rows)} fits")


if __name__ == "__main__":
    main()
