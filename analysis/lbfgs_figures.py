"""Figure: L-BFGS convergence and cost from the generic start (input B) and the MLP/FNO starts with and without TTT (input A).

    python -m analysis.lbfgs_figures        -> results/lbfgs/figures/lbfgs_start_comparison.png
"""
import json, glob, numpy as np, matplotlib
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BLUE, ORANGE, GRAY, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#8a8a85', '#1f1f1e', '#6b6a64', '#e6e5e0'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
S = [('B_center', 'Generic start', GRAY, ':'), ('A_mlp', 'MLP', BLUE, '--'), ('A_mlp_ttt', 'MLP + TTT', BLUE, '-'),
     ('A_fno', 'FNO', ORANGE, '--'), ('A_fno_ttt', 'FNO + TTT', ORANGE, '-')]
TTT_S = {'A_mlp_ttt': 0.25, 'A_fno_ttt': 0.69}            # TTT seconds per material, results/vanilla pinn/summary.txt
R = {d: sorted([json.load(open(p)) for p in glob.glob(str(REPO / 'results' / 'lbfgs' / d / 'm*.json'))], key=lambda r: r['k']) for d, *_ in S}
N = 90
def curve(rows):
    M = np.full((len(rows), N + 1), np.nan)
    for i, r in enumerate(rows):
        b = np.array([l[2] for l in r['log']]) / r['mean_c']
        M[i, 1:len(b) + 1] = b[:N]; M[i, len(b) + 1:] = b[-1]
        M[i, 0] = r['log'][0][1] / r['mean_c']
    return M[:, 1:]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
a = ax[0]
for t in (1e-2, 1e-3, 1e-4):
    a.axhline(t, color=GRID, lw=1, zorder=0); a.text(N, t * 1.15, f'{t*100:g} %', color=MUTED, fontsize=8.5, ha='right')
for d, lab, c, ls in S:
    m = np.nanmedian(curve(R[d]), 0)
    a.plot(np.arange(1, N + 1), m, color=c, ls=ls, lw=2, label=lab)
a.set_yscale('log'); a.set_ylim(1e-11, 0.2); a.set_xlim(1, N)
a.set_xlabel('L-BFGS evaluations'); a.set_ylabel('best RMSE / mean c  (median over 300)')
a.set_title('Convergence', loc='left', fontsize=11, color=INK, fontweight='bold')
a.legend(frameon=False, fontsize=9, loc='lower left')
# panel 2: median evaluations to targets
b = ax[1]; x = np.arange(len(S)); w = 0.38
for j, (rel, alpha, lab) in enumerate([('0.001', 0.45, '0.1 % target'), ('0.0001', 1.0, '0.01 % target')]):
    vals = [np.nanmedian([r['to_target'][rel][0] or np.nan for r in R[d]]) for d, *_ in S]
    bars = b.bar(x + (j - 0.5) * w, vals, w - 0.04, color=[s[2] for s in S], alpha=alpha, edgecolor='#fcfcfb', lw=1.5)
    for xi, v in zip(x + (j - 0.5) * w, vals): b.text(xi, v + 0.8, f'{v:g}', ha='center', fontsize=8.5, color=INK)
b.set_xticks(x, [s[1].replace(' + ', '\n+ ') for s in S], fontsize=8.5)
b.set_ylabel('median evaluations'); b.set_title('Evaluations to reach target', loc='left', fontsize=11, color=INK, fontweight='bold')
b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.45), plt.Rectangle((0, 0), 1, 1, color=MUTED)],
         labels=['0.1 % target', '0.01 % target'], frameon=False, fontsize=9, loc='upper right')
b.yaxis.grid(True, color=GRID); b.set_axisbelow(True)
# panel 3: time to 0.01 % incl. TTT
cdx = ax[2]
lb = [np.nanmedian([r['to_target']['0.0001'][1] or np.nan for r in R[d]]) for d, *_ in S]
tt = [TTT_S.get(d, 0.0) for d, *_ in S]
cdx.bar(x, tt, 0.6, color=[s[2] for s in S], alpha=0.35, edgecolor='#fcfcfb', lw=1.5)
cdx.bar(x, lb, 0.6, bottom=tt, color=[s[2] for s in S], edgecolor='#fcfcfb', lw=1.5)
for xi, t0, l0 in zip(x, tt, lb): cdx.text(xi, t0 + l0 + 0.04, f'{t0 + l0:.2f}', ha='center', fontsize=8.5, color=INK)
cdx.set_xticks(x, [s[1].replace(' + ', '\n+ ') for s in S], fontsize=8.5)
cdx.set_ylabel('seconds per material (median)'); cdx.set_title('Time to 0.01 %, incl. TTT', loc='left', fontsize=11, color=INK, fontweight='bold')
cdx.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.35), plt.Rectangle((0, 0), 1, 1, color=MUTED)],
           labels=['TTT (150 steps)', 'L-BFGS'], frameon=False, fontsize=9, loc='upper left')
cdx.yaxis.grid(True, color=GRID); cdx.set_axisbelow(True); cdx.set_ylim(0, 2.4)
fig.tight_layout()
out = REPO / 'results' / 'lbfgs' / 'figures'
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'lbfgs_start_comparison.png', dpi=160)
