# Results review

`index.html` is the results and architecture review page (open it in a browser; math renders with MathJax from a CDN). It holds one comparison of every recorded inverse run, rescored from the recorded θ, plus diagrams of the MLP, FNO, TTT and PINN builds. Figures are copies from `results/*/figures`.

Rebuild the comparison numbers from the repository root:

```bash
python review/scoring/truth.py . review/scoring/truth.npz   # true θ and observed c window of the 300 test materials (~11 s)
python review/scoring/score.py                              # writes review/scoring/compare.json
```
