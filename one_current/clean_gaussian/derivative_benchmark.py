"""Matched observed spatial-slope tests on the eight held-out PINN materials.

The original benchmark defines the data, initializations, optimizers and budgets.
Outputs are isolated by objective; result JSON files mark completed fits for resume.
"""
import argparse
import os

import torch

from pinn_benchmark import fit


CORRECTIONS = ("data_correction", "solver_correction", "solver_correction_weight1")
METHODS = ("data_first", *CORRECTIONS, "frozen", "standard_cv", "staged_cv_w1", "staged_cv_w100", "fine")
DOMAINS = ("clean", "gaussian_noisy")
STARTS = ("fno_ttt", "center")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=DOMAINS)
    parser.add_argument("--currents", nargs="+", type=int, choices=(1, 3), default=(1, 3))
    parser.add_argument("--starts", nargs="+", choices=STARTS, default=STARTS)
    parser.add_argument("--arm", nargs="+", choices=("slopes", "control"), default=("slopes", "control"))
    parser.add_argument("--force", action="store_true", help="Repeat completed configurations selected by the filters")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    os.environ.setdefault("DIFFION_PINN_GRAPHS", "1")
    # The data-seeded correction reuses its own arm's completed field fit.
    for method in args.methods:
        for arm in args.arm:
            for start in args.starts:
                if start == "center" and method in CORRECTIONS:
                    continue
                for domain in args.domains:
                    for ncur in args.currents:
                        done = fit(
                            domain, ncur, method, start, args.device,
                            derivative_weight=float(arm == "slopes"), force=args.force,
                        )
                        if not done:
                            raise FileNotFoundError(f"Missing benchmark input for {domain}, {ncur}, {method}, {start}, {arm}")


if __name__ == "__main__":
    main()
