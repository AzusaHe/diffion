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
├── pinn/        # physics-informed refinement and PINNs
│   ├── material.py      D(c), t+(c), alpha(c), V(c) in torch
│   ├── pde_loss.py      PDE loss = squared residual of salt conservation on the data grid
│   ├── ttt.py           test-time training on the PDE loss
│   ├── jax_pinn.py      per-material PINNs, Builds 1 and 2, total loss + Adam (JAX, CPU)
│   ├── torch_pinn.py    the same PINNs in PyTorch, many fits batched on the GPU
│   └── sweep.py         runs the PINN tuning sweep and test runs as parallel processes
├── analysis/    # scoring, summaries, figures
│   ├── benchmark.py     optimizer-baseline summaries and figures
│   ├── make_figures.py  rebuilds every baseline summary.csv and figure
│   ├── metrics.py       per-material errors of D(c), t+(c)
│   ├── stage2_load.py   reads results/stage2
│   ├── stage2_summary.py every number in the progress report
│   ├── stage2_figures.py figures A1–A6
│   ├── lbfgs_figures.py L-BFGS convergence by start
│   └── pinn_figures.py  PINN results with L-BFGS and NM on the same materials
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

## Gradient-based fit: differentiable solver + L-BFGS

`inverse/lbfgs.py` fits the 8 coefficients of each of the 300 test materials (the ones the networks are scored on) with the full model in `solver/integrate.py`: cell-centred finite volumes, Crank–Nicolson with a Rannacher start and a Newton solve per step. The gradient comes from reverse mode through the solver, which applies the implicit-function theorem at every step, so one evaluation is one forward and one adjoint solve per current. L-BFGS-B searches the box $\boldsymbol\theta \in [0,1]^8$.

The start sets the input option: `center` ($\boldsymbol\theta = 0.5$, input B, no learned start) or `fno`, `mlp`, `fno_ttt`, `mlp_ttt` (input A, the network's prediction, optionally after TTT). It needs `data/train.npz`; build it first with `python -m solver.dataset`.

```bash
python -m inverse.lbfgs --start center --out "results/lbfgs/B_center"
python -m inverse.lbfgs --start fno    --out "results/lbfgs/A_fno"
python -m inverse.lbfgs --start center --out "results/lbfgs/B_center" --shard 0/4   # one of 4 parallel processes
```

Each material writes `m<k>.json` (fitted and starting $\boldsymbol\theta$, RMSE, one log row per evaluation), and `summary.json` gives median errors, evaluations and wall time, and evaluations to reach an RMSE of 1 %, 0.1 % and 0.01 % of mean $c$.

Results on the 300 test materials (noise-free data, one fit per material, 4 parallel processes on a 4-core cloud machine; one evaluation is about 40 ms):

| Start | D error at start | Evaluations to 1 % / 0.1 % / 0.01 % RMSE (median) | Reached 0.01 % | Evaluations to converge (median) | Wall time per material (median) |
| --- | --- | --- | --- | --- | --- |
| B · generic start (θ = 0.5) | 38.9 % | 6 / 18 / 45 | 298/300 | 82.5 | 3.4 s |
| A · MLP | 9.0 % | 1 / 7 / 40 | 299/300 | 80 | 3.5 s |
| A · MLP + TTT | 6.5 % | 1 / 4 / 35 | 299/300 | 78 | 3.4 s |
| A · FNO | 8.5 % | 1 / 7 / 38.5 | 300/300 | 79 | 3.2 s |
| A · FNO + TTT | 5.5 % | 1 / 3 / 32 | 299/300 | 77 | 3.1 s |

Every fit converges, and the median final RMSE is about 5e-12 in units of $c$, so $D(c)$ and $t_+(c)$ are recovered to about 1e-7 % (the data come from the same solver and grid, so an exact fit exists). A learned start cuts the evaluations to the 0.01 % target from 45 to 32–40, and to the 0.1 % target from 18 to 3–7. One or two materials per start stop early above the 0.01 % target (`to_target` is null in their `m<k>.json`).

## Gradient-free fits on the full model: inputs A and B

`inverse/gradfree.py` runs NM, CMA-ES and BO, with the same settings as the compact-model baselines below, on the same 300 test materials, full model and starts as the L-BFGS fit. Each run stops at an RMSE of 0.01 % of mean $c$, the evaluation budget, or the wall-time cap.

```bash
python -m inverse.gradfree --start fno_ttt --methods NM CMAES --out "results/gradfree/A_fno_ttt"   # input A
python -m inverse.gradfree --start center  --methods NM CMAES --out "results/gradfree/B_center"    # input B
python -m inverse.gradfree --start fno_ttt --methods BO --materials 0:30 --budget 300 --out "results/gradfree/A_fno_ttt"
```

Results (run on one PC, 8 processes in parallel; one evaluation is about 31 ms). Medians over the materials that reached each target:

| Method | Input | Reached 0.01 % | Evaluations to 1 % / 0.1 % / 0.01 % (median) | Time to 0.01 % (median) | D error at stop (median) |
| --- | --- | --- | --- | --- | --- |
| NM | A · FNO + TTT | 299/300 | 1 / 84 / 588 | 10.2 s | 3.4 % |
| NM | B · generic | 278/300 | 45.5 / 272.5 / 814 | 12.4 s | 3.6 % |
| CMA-ES | A · FNO + TTT | 205/300 | 27 / 569.5 / 1450 | 22.6 s | 4.4 % |
| CMA-ES | B · generic | 174/300 | 65 / 742.5 / 1524 | 22.9 s | 4.1 % |
| BO (30 materials, 150 evaluations) | A · FNO + TTT | 0/30 | 1 / 1 / – | – | 5.5 % |
| BO (30 materials, 150 evaluations) | B · generic | 0/30 | 63 / – / – | – | 29.5 % |

From the learned start NM needs fewer evaluations than from the generic start on 74 % of materials (median ratio 0.73); CMA-ES gains little (0.94), since its initial step σ₀ = 0.25 spreads the search far from the start. Stopping at 0.01 % RMSE in $c$ still leaves $D$ off by 3–4 %, while L-BFGS, which runs far below that, recovers it exactly.

The L-BFGS convergence and cost by start are plotted in `results/lbfgs/figures/lbfgs_start_comparison.png` (`python -m analysis.lbfgs_figures`).

## Physics-informed neural networks (PINNs): total loss, no staging

Two per-material PINNs fit the 8 coefficients together with networks for the fields, on the same test materials and starts as the fits above. Input A starts $\boldsymbol\theta$ at the FNO + TTT prediction and input B at $\boldsymbol\theta = 0.5$. The networks take $(x, s, \iota/5)$ with $s = \sqrt{t/t_{\max}}$; $c = 1 + s\,N_c$ makes $c(x,0) = 1$ exact, and $\boldsymbol\theta = \mathrm{sigmoid}(\boldsymbol\phi)$.

- **Build 1** (`--build cv`): networks for $c$ and $v$ ($v = v_{\text{scale}}\,x\,N_v$, so $v(0) = 0$ exactly). It penalizes conservation $r_c = c_t + J_x$, the volume constraint $r_v = v_x + V q_x$, and $J = \iota$ at both walls.
- **Build 2** (`--build c`): a network for $c$ only. $J = \iota - \int_0^x c_t\,dx'$ (3-point Gauss on 32 cells), so conservation and $J(0) = \iota$ hold exactly, and $v = (J - q)/c$. It penalizes $r_v$, $q(0) = \iota$ and $J(1) = \iota$.

**Loss** (one total loss, every term on from the first step):

$$
L = \frac{\overline{(\bar c_{\text{bin}} - y)^2}}{\sigma^2} + w\left(\frac{\overline{r_c^2}}{s_t^2} + \frac{\overline{r_v^2}}{(v_{\text{scale}}\,s_t)^2} + \frac{\overline{r_{\text{wall}}^2}}{\overline{\iota^2}}\right), \qquad w = 100,
$$

with $\sigma = 10^{-3}\,\bar c$, $s_t$ the RMS of $\partial_t c$ in the data, $v_{\text{scale}} = \max|v|$ of the velocity reconstructed from the data, and the $r_c$ term in Build 1 only. Residual points are redrawn every step: 1024 per current for Build 1 and 32 times per current for Build 2, from the first frame to the last.

**Optimizer:** one Adam over the networks (4 × 64, tanh) and $\boldsymbol\phi$, learning rate $10^{-3}$ with cosine decay to $10^{-5}$ over 5000 steps, and $\boldsymbol\phi$ at 0.1 × that rate. There is no staged training and no L-BFGS polish. A 20-variant sweep on one material (`results/pinn_sweep`) picked this setting: every setting moved $D(c)$ away from the truth from input A, and θ at 0.1 × the rate with $w = 100$ moved it least. With θ held at the truth, the loss was no lower than at the wrong θ.

```bash
python -m pinn.sweep --grid test --materials 0:8 --steps 5000 --jobs 32 --out results/pinn   # JAX, CPU, one fit per process
python -m pinn.torch_pinn --build cv --starts fno_ttt,center --materials 0:8 --out results/pinn_gpu   # PyTorch, CUDA, 16 fits per batch
python -m pinn.torch_pinn --build c  --starts fno_ttt,center --materials 0:8 --out results/pinn_gpu
python -m analysis.pinn_figures --root results/pinn_gpu   # figure + table with L-BFGS and NM on the same materials
```

`pinn/jax_pinn.py` and `pinn/torch_pinn.py` are the same method; `tests/test_torch_pinn.py` checks that every loss term and the θ gradient agree to 1e-9 in float64. `pinn/material.py` holds the PyTorch copy of $\alpha(c)$ and $V(c)$.

Results on test materials 0–7 (medians; one PC; GPU = RTX 4090, float64):

| Method | Input | D error at start | D error at end | t+ RMSE at end | D better than start | Data RMSE / mean c | Wall time per fit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PINN Build 1, JAX (CPU) | A · FNO + TTT | 5.7 % | 9.4 % | 0.032 | 0/8 | 5.1e-3 | 6003 s ¹ |
| PINN Build 2, JAX (CPU) | A · FNO + TTT | 5.7 % | 9.4 % | 0.028 | 0/8 | 5.7e-3 | 5856 s ¹ |
| PINN Build 1, JAX (CPU) | B · generic | 31.8 % | 29.0 % | 0.184 | 5/8 | 5.6e-3 | 6005 s ¹ |
| PINN Build 2, JAX (CPU) | B · generic | 31.8 % | 29.2 % | 0.187 | 5/8 | 7.6e-3 | 5861 s ¹ |
| PINN Build 1, PyTorch (GPU) | A · FNO + TTT | 5.7 % | 8.7 % | 0.036 | 0/8 | 4.9e-3 | 50 s ² |
| PINN Build 2, PyTorch (GPU) | A · FNO + TTT | 5.7 % | 10.6 % | 0.031 | 0/8 | 5.7e-3 | 33 s ² |
| PINN Build 1, PyTorch (GPU) | B · generic | 31.8 % | 29.4 % | 0.184 | 4/8 | 5.1e-3 | 50 s ² |
| PINN Build 2, PyTorch (GPU) | B · generic | 31.8 % | 28.3 % | 0.186 | 5/8 | 7.6e-3 | 33 s ² |
| L-BFGS | A · FNO + TTT | 5.7 % | 0.0 % | 0.000 | 8/8 | 2.2e-12 | 3.2 s |
| L-BFGS | B · generic | 31.8 % | 0.0 % | 0.000 | 8/8 | 4.6e-12 | 3.2 s |
| NM | A · FNO + TTT | 5.7 % | 3.9 % | 0.017 | 7/8 | 9.9e-5 | 13.4 s |
| NM | B · generic | 31.8 % | 3.6 % | 0.016 | 8/8 | 1.0e-4 | 12.3 s |

¹ 32 fits in parallel on the CPU, one thread each. ² Batch wall time ÷ 16 fits per batch.

With the total loss, neither build recovers the coefficients. From input A, $D(c)$ ends worse than the start on all 8 materials; from input B it improves slightly on 4–5 of 8. The data misfit stalls at 0.5–0.8 % of mean $c$, far above the 0.01 % that the solver-based fits reach, so the physics residuals are evaluated on a $c$ field whose derivatives are not accurate enough to single out θ. Build 2 also diverges on some of the materials whose concentration falls lowest (minimum $c$ 0.04–0.15: materials 3, 5 and 7), where $v = (J - q)/c$ divides by a small network $c$; the medians hide these fits. The JAX and PyTorch runs agree within the spread of their different random residual points. Figures: `results/pinn/figures/pinn_results.png` (JAX) and `results/pinn_gpu/figures/pinn_results.png` (PyTorch).

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
