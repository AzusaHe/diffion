"""Build the offline case viewer from the verified one/three-current pairs."""
import csv
import json
import math
from pathlib import Path

if __package__:
    from .field_view import fragment
else:
    from field_view import fragment

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "random" / "visuals"
INPUT = OUT / "paired_cases.csv"


def describe_method(key):
    learned = {
        "mlp_supervised": "MLP", "fno_supervised": "FNO",
        "mlp_ttt": "MLP + TTT", "fno_ttt": "FNO + TTT",
    }
    if key in learned:
        return "Pretrained estimators", learned[key]
    prefix, _, suffix = key.partition("_")
    starts = {"center": "center", "mlp": "MLP", "fno": "FNO",
              "mlp_ttt": "MLP + TTT", "fno_ttt": "FNO + TTT"}
    if prefix != "pinn":
        method = {"lbfgs": "L-BFGS-B", "CMAES": "CMA-ES", "BO": "Bayesian optimization", "NM": "Nelder–Mead"}[prefix]
        return "Numerical inversion", f"{method} · {starts[suffix]} start"
    start = "fno_ttt" if suffix.endswith("_fno_ttt") else "center"
    variant = suffix.removesuffix("_" + start)
    names = {"standard_c": "Joint c only", "standard_cv": "Joint c and v", "data_first": "Data-first joint",
             "data_correction": "Data-first + correction", "fine": "Native-grid data-first",
             "solver_correction": "Solver-seeded correction, w = 0.01",
             "solver_correction_weight1": "Solver-seeded correction, w = 1"}
    if variant.startswith("staged_"):
        _, build, weight = variant.split("_")
        name = f"Staged {'c only' if build == 'c' else 'c and v'}, w = {weight[1:]}"
    else:
        name = names[variant]
    return "PINN refinements", f"{name} · {starts[start]} start"


def number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite score or input")
    return float(f"{value:.10g}")


def main():
    material_fields = ["iota", "iota_low", "iota_middle", "iota_high"]
    window_fields = ["one_window_min", "one_window_max", "three_window_min", "three_window_max"]
    score_fields = ["D_one_observed_pct", "D_three_observed_pct", "t_one_observed_rmse", "t_three_observed_rmse"]
    materials, methods, labels = {}, {}, {}
    windows = {"binned": {}, "native": {}}
    method_grids = {}
    seen = set()
    with INPUT.open(encoding="utf-8-sig", newline="") as source:
        for record in csv.DictReader(source):
            key = record["method"]
            seed = int(record["seed"]) if record["seed"] else -1
            mid = int(record["material_id"])
            rank = int(record["rank"])
            material = [rank] + [number(record[field]) for field in material_fields]
            if materials.setdefault(mid, material) != material or material[1] != material[rank+2]:
                raise ValueError(f"Inconsistent current for material {mid}")
            grid = record["observation_grid"]
            if grid not in windows or method_grids.setdefault(key,grid) != grid:
                raise ValueError(f"Inconsistent observation grid for method {key}")
            bounds = [number(record[field]) for field in window_fields]
            if windows[grid].setdefault(mid,bounds) != bounds or bounds[0]>=bounds[1] or bounds[2]>=bounds[3]:
                raise ValueError(f"Inconsistent {grid} observed concentration range for material {mid}")
            if (key, mid, seed) in seen:
                raise ValueError(f"Duplicate fit {key}, {mid}, {seed}")
            seen.add((key, mid, seed))
            methods.setdefault(key, []).append([mid, seed] + [number(record[f]) for f in score_fields])
            labels[key] = describe_method(key)
    if len(materials) != 300 or len(methods) != 34 or len(seen) != 6512:
        raise ValueError(f"Incomplete source: {len(materials)} materials / {len(methods)} methods / {len(seen)} fits")
    curves = json.loads((OUT / "curve_coefficients.json").read_text(encoding="utf-8"))
    for key, mid, seed in seen:
        pair = curves["fits"][key][str(seed) if seed >= 0 else "fixed"][str(mid)]
        if len(curves["truth"][str(mid)]) != 8 or len(pair) != 2 or len(pair[0]) != 8:
            raise ValueError(f"Missing property coefficients for {key}, {mid}, {seed}")
        if pair[1] is None and (key not in {"fno_ttt", "mlp_ttt"} or seed not in {1, 2}):
            raise ValueError(f"Unexpected missing historical coefficients for {key}, {mid}, {seed}")
    payload = {"materials": materials, "methods": methods, "labels": labels, "curves": curves,
               "windows": windows, "methodGrids": method_grids}
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    template = (HERE / "explorer_template.html").read_text(encoding="utf-8")
    destination = OUT / "explorer.html"
    document = template.replace("/*__CASE_DATA__*/", data).replace("<!--FIELD_VIEW-->", fragment())
    destination.write_text(document, encoding="utf-8")
    print(f"{destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
