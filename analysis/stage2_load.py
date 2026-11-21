"""Read the stage-2 results (results/stage2) and summarise them per model.

Four models: MLP or FNO, supervised only ('sup') or after PDE test-time training ('ttt').
Every model is scored on the same 300 test materials, for 3 seeds.  Summary of a metric: the median over
materials for each seed; reported as the mean of the three medians and their range.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "stage2"
ARCHS = ("mlp", "fno")
SEEDS = (0, 1, 2)
NAME = {"mlp": "MLP", "fno": "FNO"}


class Results:
    def __init__(self, root=RESULTS):
        root = Path(root)
        self.sup = {(a, s): json.loads((root / f"{a}_s{s}" / "result.json").read_text()) for a in ARCHS for s in SEEDS}
        self.ttt = {(a, s): json.loads((root / "ttt" / f"{a}_s{s}.json").read_text()) for a in ARCHS for s in SEEDS}
        self.select = {a: json.loads((root / "ttt" / f"select_{a}.json").read_text()) for a in ARCHS}
        self.timing = {a: json.loads((root / f"timing_{a}.json").read_text()) for a in ARCHS}
        self.checkpoints = self.ttt[("fno", 0)]["checkpoints"]
        self.steps = self.ttt[("fno", 0)]["steps"]
        self.truth_pde_loss = float(self.sup[("fno", 0)]["test"]["truth_pde_loss"])

    def per_material(self, arch, ttt, seed, metric, step=None):
        """Per-material values (300,) of a metric; ttt=True gives the value after `step` TTT steps (default 150)."""
        if not ttt:
            return np.asarray(self.sup[(arch, seed)]["test"][metric], float)
        return np.asarray(self.ttt[(arch, seed)]["scores"][str(self.steps if step is None else step)][metric], float)

    def seed_stat(self, arch, ttt, metric, step=None, f=lambda v: v):
        """(mean, min, max) over seeds of the median over materials of f(metric)."""
        meds = [np.median(f(self.per_material(arch, ttt, s, metric, step))) for s in SEEDS]
        return float(np.mean(meds)), float(np.min(meds)), float(np.max(meds))

    def pooled(self, arch, ttt, metric):
        """All seeds x materials (900,)."""
        return np.concatenate([self.per_material(arch, ttt, s, metric) for s in SEEDS])

    def train_minutes(self, arch):
        """Standalone single-thread time per step x number of steps."""
        return self.timing[arch]["step_s_median"] * self.sup[(arch, 0)]["steps"] / 60
