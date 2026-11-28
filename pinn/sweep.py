"""Run many total-loss PINN fits (pinn/jax_pinn.py) as parallel processes: the tuning sweep or the test runs.

    python -m pinn.sweep --grid tune --materials 0:1 --steps 10000 --jobs 20 --out results/pinn_sweep
    python -m pinn.sweep --grid test --materials 0:8 --steps 5000  --jobs 32 --out results/pinn

tune: build (cv, c) x phi learning-rate multiplier (1, 0.1) x physics weight (1, 100, on every physics term)
      x residual time window (early, observed), all from the input-A start (fno_ttt), plus 'truth' runs that
      hold theta at the true value (phi rate 0) to show how small the physics loss gets at the truth.
test: both builds from input A (fno_ttt) and input B (center) with the setting the sweep picked
      (phi rate 0.1 x the network rate, physics weight 100, residual points from the first frame on).
The materials of each variant are split over --jobs / (number of variants) processes.  Each variant writes
<out>/<dir>/m<k>.json and summary.json, and a table is printed at the end.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path


def variants(grid):
    """(dir, build, start, phi_lr_mult, physics weight, residual window)."""
    if grid == "tune":
        for b, m, w, win in itertools.product(["cv", "c"], [1.0, 0.1], [1.0, 100.0], ["early", "observed"]):
            yield f"{b}_A_m{m:g}_w{w:g}_{win}", b, "fno_ttt", m, w, win
        for b, w in itertools.product(["cv", "c"], [1.0, 100.0]):
            yield f"{b}_truth_w{w:g}_observed", b, "truth", 0.0, w, "observed"
    else:
        for st, b in itertools.product(["A_fno_ttt", "B_center"], ["cv", "c"]):
            yield f"{st}/{b}", b, st[2:], 0.1, 100.0, "observed"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", choices=["tune", "test"], default="tune")
    p.add_argument("--materials", default="0:1")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--jobs", type=int, default=20)
    p.add_argument("--max-wall-s", type=float, default=4 * 3600.0, help="wall-time cap per fit (s)")
    p.add_argument("--out", type=Path, default=Path("results/pinn_sweep"))
    a = p.parse_args()
    V = list(variants(a.grid))
    shards = max(1, a.jobs // len(V))
    todo = [(v, i) for v in V for i in range(shards)]
    running, t0 = [], time.time()
    while todo or running:
        while todo and len(running) < a.jobs:
            (tag, b, st, m, w, win), i = todo.pop(0)
            out = a.out / tag
            out.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "pinn.jax_pinn", "--build", b, "--start", st, "--out", str(out),
                   "--materials", a.materials, "--shard", f"{i}/{shards}", "--steps", str(a.steps),
                   "--phi-lr-mult", str(m), "--w-pde", str(w), "--w-vol", str(w), "--w-bc", str(w),
                   "--res-window", win, "--max-wall-s", str(a.max_wall_s)]
            log = open(out / f"log{i}.txt", "w")
            running.append((f"{tag} shard {i}", subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        for r in list(running):
            if r[1].poll() is not None:
                r[2].close()
                running.remove(r)
                print(f"{time.time() - t0:7.0f} s  done {r[0]} (exit {r[1].returncode})", flush=True)
        time.sleep(5)
    from pinn.jax_pinn import summarize
    print(f"{'variant':34s} {'n':>3s} {'D% start':>8s} {'D% fit':>8s} {'t+ start':>8s} {'t+ fit':>7s} "
          f"{'D better':>8s} {'rel rmse':>9s} {'steps':>6s} {'wall s':>7s}")
    for tag, *_ in V:
        s = summarize(a.out / tag)
        if s:
            steps = json.loads(sorted((a.out / tag).glob("m*.json"))[0].read_text())["steps"]
            print(f"{tag:34s} {s['n']:3d} {s['median_start']['D_relL2_pct']:8.2f} {s['median_fit']['D_relL2_pct']:8.2f} "
                  f"{s['median_start']['tplus_rmse']:8.4f} {s['median_fit']['tplus_rmse']:7.4f} "
                  f"{s['D_err_improved']:4d}/{s['n']:<3d} {s['median_rel_rmse']:9.2e} {steps:6d} {s['median_wall_s']:7.0f}")


if __name__ == "__main__":
    main()
