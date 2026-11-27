"""Run and resume the paired random-current PINN protocols on the eight original pilot cases."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(os.environ.get("DIFFION_ONE_CURRENT_ROOT",
                           str(Path(__file__).resolve().parents[1] / "random"))).resolve()
DATA = ROOT / "data"
HERE = ROOT / "pinn_runs"
RESULT = HERE / "results"


def wait_for(path):
    while not path.exists():
        time.sleep(3)


def wait_for_solver_prefit():
    # A completed native fit is shared across the two correction loss weights. The
    # first joint progress entry is emitted only after every file is written and closed.
    coarse_log = HERE / 'solver_correction_fno_ttt.log'
    finished = RESULT / 'solver_correction/fno_ttt/summary.json'
    while not finished.exists():
        if coarse_log.exists() and '\njoint ' in coarse_log.read_text(encoding='utf-8'):
            return
        time.sleep(3)


def launch(method, start):
    summary = RESULT / method / start / "summary.json"
    if summary.exists() or (start == "paired" and all(
            (RESULT / method / s / "summary.json").exists() for s in ("center", "fno_ttt"))):
        return
    if start in ("fno_ttt", "paired"):
        wait_for(ROOT / "learned/fno_s0/ttt_first8.npz")
    if method == "fine":
        wait_for(DATA / "fine_test.npz")
    env = os.environ.copy()
    env["DIFFION_PINN_GRAPHS"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    path = HERE / f"{method}_{start}.log"
    args = [sys.executable, "-B", "-u", "-m", "one_current.pinn_runs.run",
            "--method", method, "--start", start]
    print(f"Starting {method}/{start}",flush=True)
    with path.open("a", encoding="utf-8") as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            if result.returncode not in (3221225477,-1073741819) and "out of memory" not in path.read_text().lower():
                result.check_returncode()
            print(f"Retrying {method}/{start} in batches of four after GPU process exit {result.returncode}",flush=True)
            subprocess.run(args + ["--batch-size","4"], check=True, stdout=log,stderr=subprocess.STDOUT,env=env)
    print(f"Completed {method}/{start}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=["refinements", "standard", "staged", "staged1", "solver-weight1"], required=True)
    lane = parser.parse_args().lane
    HERE.mkdir(exist_ok=True, parents=True)
    wait_for(DATA / "train.npz")
    protocol = json.loads((DATA / "protocol.json").read_text())
    if protocol["protocol"] != "random" or protocol["number_of_currents"] != 1:
        raise ValueError(f"Unexpected campaign data in {DATA}")
    if lane == "refinements":
        jobs = [("data_first", "center"), ("data_correction", "fno_ttt"),
                ("solver_correction", "fno_ttt"), ("fine", "center"), ("fine", "fno_ttt"),
                ("solver_correction_weight1", "fno_ttt")]
    elif lane == "standard":
        jobs = [(method, start) for start in ("center", "fno_ttt") for method in ("standard_cv", "standard_c")]
    elif lane == "staged":
        wait_for(RESULT / "standard_c/fno_ttt/summary.json")
        jobs = [(f"staged_{build}_w100", "paired") for build in ("cv","c")]
    elif lane == "staged1":
        wait_for(RESULT / "staged_c_w100/fno_ttt/summary.json")
        jobs = [(f"staged_{build}_w1", "paired") for build in ("cv","c")]
    else:
        wait_for_solver_prefit()
        jobs = [("solver_correction_weight1", "fno_ttt")]
    for method, start in jobs:
        launch(method, start)


if __name__ == "__main__":
    main()
