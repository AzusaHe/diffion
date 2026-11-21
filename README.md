# DiffIon

CBE 236 project: physics-informed inference of electrolyte transport properties. A differentiable solver for ion transport recovers the diffusion coefficient $D(c)$ and the transference number $t_+(c)$ from measurements of the salt concentration $c(x,t)$ in a cell under constant current.

$$
\frac{\partial c}{\partial t} + \frac{\partial (q + c v)}{\partial x} = 0, \qquad
q = -D(c)\,\alpha(c)\,\frac{\partial c}{\partial x} + t_+(c)\,\iota, \qquad
\frac{\partial v}{\partial x} = -V(c)\,\frac{\partial q}{\partial x}
$$

The boundary conditions are $q + cv = \iota$ at both walls and $v = 0$ at $x = 0$. $D$ and $t_+$ are cubic Bernstein curves in $c$ with 8 coefficients $\boldsymbol\theta \in [0,1]^8$. $\alpha(c)$ and $V(c)$ come from a known density.

```text
diffion/
├── solver/      # physics and forward simulation (JAX)
│   ├── properties.py   material model: D(c), t+(c), density -> alpha(c), V(c)
│   ├── fv.py           conservative finite-volume fluxes
│   ├── structured.py   exact O(N) Newton solve for one time step
│   ├── integrate.py    Crank–Nicolson + Newton, implicit-function-theorem gradients
│   └── dataset.py      synthetic materials: sample, simulate, split
├── surrogate/   # neural networks (PyTorch)
│   ├── data.py         splits, electrolyte velocity v(x,t) from c alone
│   ├── models.py       MLP and FNO (about 65k parameters each)
│   └── train.py        supervised training on the coefficient loss
├── pinn/        # physics-informed refinement
│   ├── material.py     D(c), t+(c) in torch
│   ├── pde_loss.py     PDE loss = squared residual of salt conservation on the data grid
│   └── ttt.py          test-time training on the PDE loss
├── inverse/     # next: numerical baselines (BO, Nelder–Mead) and differentiable-solver fits
├── analysis/    # scoring, report numbers, figures
│   ├── metrics.py      per-material errors of D(c), t+(c)
│   ├── load.py         reads results/stage2
│   ├── summary.py      every number in the progress report
│   └── figures.py      figures A1–A6
├── results/stage2/  # reported results: models, per-material scores, figures, summary
└── agent/       # future experiment automation
```

## Pipeline

Run every command from the repository root.

```bash
pip install -r requirements.txt
python -m solver.dataset                      # data/train.npz (16,000 materials, ~20 min, not committed)
python -m surrogate.train --arch mlp --seed 0 # repeat for --arch mlp|fno, --seed 0 1 2
python -m surrogate.train --arch fno --timing # time per training step
python -m pinn.ttt --arch fno --seed 0        # seed 0 first: it chooses the TTT learning rate
python -m analysis.summary                    # results/stage2/summary.txt
python -m analysis.figures                    # results/stage2/figures/A1–A6
```

## Setup of the reported results (stage 2)

- **Data:** 14,995 simulated materials, each run at **three currents**. Each is observed at 13 frames × 20 positions with noise-free $c$. The random split gives 10,336 for training, 593 for validation, and 300 test materials, the same 300 for every model.
- **Input:** $c - 1$ and $v$ for each current, a frame mask, $t$, $x$ and $\iota$. $v$ is computed from $c$ by conservation and the volume constraint; nothing is fitted.
- **Networks:**
  - MLP: 1576 → 40 → 40 → 8, 65,048 parameters.
  - FNO: width 11, 5 × 6 Fourier modes, 4 layers, 65,807 parameters.
- **Supervised training:** AdamW, learning rate $10^{-3}$ with cosine decay, batch 64, 100 epochs, 3 seeds. The loss is the MSE of the 8 coefficients.
- **Test-time training (TTT):** for each test material, all network weights take 150 AdamW steps on the PDE loss alone, with learning rate $10^{-4}$ chosen on validation. The weights reset before each material.

## Results (300 test materials, mean of 3 seeds)

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

## Reproducibility

The code here regenerates the stored data bit for bit. Loading the saved models (`results/stage2/*/model.pt`) reproduces the saved predictions, scores, test-time training trajectories and figures exactly. Retraining the MLP reproduces it exactly. The FNO draws its initial weights in a different order than the run that produced the saved model, so retraining it gives an equivalent network with slightly different numbers.
