"""Rescore true and inferred transport on identical frozen data-first PINN fields."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pinn_benchmark import CACHE, HERE, engine, statistics


def check_fields(domain, currents, weight, data, rows, losses, points_per_current):
    ids = data["ids"][rows]
    cache = CACHE / ("derivative_fields" if weight else "derivative_control_fields")
    directory = HERE / ("derivative_results" if weight else "derivative_controls")
    stem = f"{domain}__{currents}"
    checkpoint = torch.load(cache / f"{stem}__data__0.pt", map_location="cpu")
    field = checkpoint["params"]
    with np.load(directory / f"{stem}__pinn_data_first_fno_ttt.npz") as estimates:
        np.testing.assert_array_equal(estimates["ids"], ids)
        np.testing.assert_allclose(torch.sigmoid(field["phi"]).numpy(), estimates["theta"], atol=1e-6)

    results = []
    observed = data[domain][rows, :currents]
    times = data["tau"][rows, :currents]
    iota = data["iota"][rows, :currents]
    for index, mid in enumerate(ids):
        stats = statistics(observed[index], iota[index], times[index])
        state = {k: torch.as_tensor(v, dtype=torch.bool if k in ("mask", "vmask") else torch.float32)
                 for k, v in stats.items()}
        points = torch.stack([
            torch.quasirandom.SobolEngine(2, scramble=True, seed=27641 + 10 * int(mid) + j).draw(points_per_current)
            for j in range(currents)
        ])
        x = points[..., 0]
        t = state["end_times"][:, None] * (state["s_lo"] + (1 - state["s_lo"]) * points[..., 1]).square()
        params = engine._rebuild(field, iter(a[index] for a in engine._leaves(field)))
        truth = torch.as_tensor(data["theta"][rows[index]], dtype=torch.float32)
        record = dict(domain=domain, currents=currents, derivative_weight=weight, material_id=int(mid))
        for label, logits in (("fitted", params["phi"]), ("truth", torch.logit(truth))):
            with torch.no_grad():
                _, terms = losses.loss_cv(params | {"phi": logits}, state, x, t)
            components = {key: float(terms[key]) for key in ("pde", "vol", "bc")}
            record[label + "_components"] = components
            record[label] = sum(components.values())
        results.append(record)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=int, choices=(0, 1), nargs="+", default=[0, 1])
    parser.add_argument("--output", type=Path, default=HERE / "derivative_results" / "fixed_field_physics.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    points = 2048
    grid = engine.ObsGrid(np.arange(13), 20, 2)
    losses = engine.Losses("cv", torch.float32, "cpu", n_res=points, n_times=6, n_cells=12, obs=grid)
    scores = []
    with np.load(CACHE / "binned.npz") as data:
        rows = np.flatnonzero(data["split"] == "test")[:8]
        for domain in ("clean", "gaussian_noisy"):
            for currents in (1, 3):
                for weight in args.weights:
                    scores.extend(check_fields(domain, currents, weight, data, rows, losses, points))
    output = dict(method="pinn_data_first_fno_ttt", interior_points_per_current=points,
                  boundary_times_per_current=256, metric="pde + vol + bc", rows=scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
