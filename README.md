# DiffIon

CBE 236 project: physics-informed inference of electrolyte transport properties. Differentiable solvers and inverse methods for ion transport recover the diffusion coefficient $D(c)$ and the transference number $t_+(c)$ from measurements of the salt concentration $c(x,t)$ in a cell under constant current.

$$
\frac{\partial c}{\partial t} + \frac{\partial (q + c v)}{\partial x} = 0, \qquad
q = -D(c)\,\alpha(c)\,\frac{\partial c}{\partial x} + t_+(c)\,\iota, \qquad
\frac{\partial v}{\partial x} = -V(c)\,\frac{\partial q}{\partial x}
$$

The boundary conditions are $q + cv = \iota$ at both walls and $v = 0$ at $x = 0$. $D$ and $t_+$ are cubic Bernstein curves in $c$ with 8 coefficients $\boldsymbol\theta \in [0,1]^8$. $\alpha(c)$ and $V(c)$ come from a known density. This full model is behind the networks and test-time training. The optimizer baselines use a compact version (`solver/transport_1d.py`) with $\alpha = 1$, a constant $\bar V = 0.1$, its own coefficient ranges and currents $\iota = 0.3, 0.6, 0.9$.

```text
diffion/
├── solver/      # physics and forward simulation (JAX)
│   ├── transport_1d.py  compact model for the optimizer baselines: FV, backward Euler, RMSE loss
│   ├── properties.py    stage-2 material model: D(c), t+(c), density -> alpha(c), V(c)
│   ├── fv.py            conservative finite-volume fluxes
│   ├── structured.py    exact O(N) Newton solve for one time step
│   ├── integrate.py     Crank–Nicolson + Newton, implicit-function-theorem gradients
│   └── dataset.py       synthetic materials: sample, simulate, split
├── inverse/     # parameter estimation and optimizer comparisons
│   ├── baselines.py     Nelder–Mead, CMA-ES, BO with per-evaluation logging
│   └── run_baselines.py command-line runner
├── surrogate/   # neural networks (PyTorch)
│   ├── data.py          splits, electrolyte velocity v(x,t) from c alone
│   ├── models.py        MLP and FNO (about 65k parameters each)
│   └── train.py         supervised training on the coefficient loss
├── pinn/        # physics-informed refinement
│   ├── material.py      D(c), t+(c) in torch
│   ├── pde_loss.py      PDE loss = squared residual of salt conservation on the data grid
│   └── ttt.py           test-time training on the PDE loss
├── analysis/    # scoring, summaries, figures
│   ├── benchmark.py     optimizer-baseline summaries and figures
│   ├── make_figures.py  rebuilds every baseline summary.csv and figure
│   ├── metrics.py       per-material errors of D(c), t+(c)
│   ├── stage2_load.py   reads results/stage2
│   ├── stage2_summary.py every number in the progress report
│   └── stage2_figures.py figures A1–A6
├── results/
│   ├── baselines/       target_*/ (one JSON + CSV per run, summary.csv), figures/
│   └── stage2/          saved models, per-material scores, summary, figures/
├── tests/       # solver checks: Newton convergence, mass conservation, gradient
├── agent/       # future experiment automation
└── pyproject.toml
```

## Setup

From the repository root:

```bash
pip install -e ".[dev]"
pytest
```

## Optimizer baselines

The baselines fit the 8 Bernstein coefficients of $D(c)$ and $t_+(c)$ in the compact model (`solver/transport_1d.py`) to noise-free synthetic data of one material. Nelder–Mead (NM), CMA-ES and BO each start from the same 5 points and run until the concentration RMSE reaches a target, given as a fraction of mean $c$.

| Results folder | Target RMSE | Budget per run |
| --- | --- | --- |
| `results/baselines/target_0.01pct` | 0.1 mM | 2000 evaluations or 30 min, all methods |
| `results/baselines/target_0.1pct` | 1 mM | 2000 evaluations (NM, CMA-ES), 400 (BO) |
| `results/baselines/target_1pct` | 10 mM | 2000 evaluations (NM, CMA-ES), 400 (BO) |

To reproduce the 0.01% target, run one method at a time, then rebuild the figures:

```bash
python -m inverse.run_baselines --target-rel 1e-4 --budget 2000 --max-wall-s 1800 --methods NM    --out results/baselines/target_0.01pct
python -m inverse.run_baselines --target-rel 1e-4 --budget 2000 --max-wall-s 1800 --methods CMAES --out results/baselines/target_0.01pct
python -m inverse.run_baselines --target-rel 1e-4 --budget 2000 --max-wall-s 1800 --methods BO    --out results/baselines/target_0.01pct
python -m analysis.make_figures
```

Wall times depend on the machine. One forward solve took about 33 ms for the 0.1% runs and about 52 ms for the 0.01% and 1% runs, so compare wall times only within one figure.

| Target | Runs reaching the target (NM / CMA-ES / BO) |
| --- | --- |
| 10 mM | 5/5 / 5/5 / 5/5 |
| 1 mM | 5/5 / 5/5 / 0/5 |
| 0.1 mM | 4/5 / 4/5 / 0/5 |

## Networks and test-time training (stage 2)

Run every command from the repository root.

```bash
python -m solver.dataset                       # data/train.npz (16,000 materials, ~20 min, not committed)
python -m surrogate.train --arch mlp --seed 0  # repeat for --arch mlp|fno, --seed 0 1 2
python -m surrogate.train --arch fno --timing  # time per training step
python -m pinn.ttt --arch fno --seed 0         # seed 0 first: it chooses the TTT learning rate
python -m analysis.stage2_summary              # results/stage2/summary.txt
python -m analysis.stage2_figures              # results/stage2/figures/A1–A6
```

### Setup of the reported results

- **Data:** 14,995 simulated materials, each run at **three currents**. Each is observed at 13 frames × 20 positions with noise-free $c$. The random split gives 10,336 for training, 593 for validation, and 300 test materials, the same 300 for every model.
- **Input:** $c - 1$ and $v$ for each current, a frame mask, $t$, $x$ and $\iota$. $v$ is computed from $c$ by conservation and the volume constraint; nothing is fitted.
- **Networks:**
  - MLP: 1576 → 40 → 40 → 8, 65,048 parameters.
  - FNO: width 11, 5 × 6 Fourier modes, 4 layers, 65,807 parameters.
- **Supervised training:** AdamW, learning rate $10^{-3}$ with cosine decay, batch 64, 100 epochs, 3 seeds. The loss is the MSE of the 8 coefficients.
- **Test-time training (TTT):** for each test material, all network weights take 150 AdamW steps on the PDE loss alone, with learning rate $10^{-4}$ chosen on validation. The weights reset before each material.

### Results (300 test materials, mean of 3 seeds)

| Median over materials | MLP | MLP + TTT | FNO | FNO + TTT |
|---|---|---|---|---|
| MSE of $\ln D$ | 0.0091 | 0.0047 | 0.0082 | 0.0035 |
| MSE of $t_+$ | 0.0027 | 0.0014 | 0.0021 | 0.0010 |
| $D$ relative error | 9.1 % | 6.6 % | 8.4 % | 5.5 % |
| RMS error of $\ln[(1-t_+)/D]$ | 0.072 | 0.013 | 0.056 | 0.013 |

- **TTT reduces the RMS error** of $\ln D$ by 1.4× (MLP) and 1.5× (FNO), of $t_+$ by 1.4× for both, and of $\ln[(1-t_+)/D]$ by 5.4× and 4.5×. It improves 78 % (MLP) and 84 % (FNO) of materials.
- **FNO vs MLP:** after TTT the FNO has the lower error on 75–76 % of materials, with a median FNO/MLP error ratio of 0.76–0.78.
- **Training cost** on one CPU thread: 0.4 min for the MLP and 8.9 min for the FNO.
- **PDE loss:** at the true coefficients it is 0.0052, which is the discretization error of the 20-point grid.

`results/stage2/summary.txt` has every number and `results/stage2/figures/` has figures A1–A6.

### Reproducibility

The code here regenerates the stored data bit for bit. Loading the saved models (`results/stage2/*/model.pt`) reproduces the saved predictions, scores, test-time training trajectories and figures exactly. Retraining the MLP reproduces it exactly. The FNO draws its initial weights in a different order than the run that produced the saved model, so retraining it gives an equivalent network with slightly different numbers.
