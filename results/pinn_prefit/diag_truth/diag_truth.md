
### strong residual: physics loss (pde + vol + bc) on the stage-1 networks, median over 8 materials

| theta | D err % | t+ rmse | physics loss | loss / loss at truth | materials where truth is lower |
|---|---|---|---|---|---|
| truth | 0.0 | 0.000 | 5.754e-01 | 1.000 | 0/8 |
| start A (FNO+TTT) | 5.7 | 0.033 | 5.822e-01 | 1.027 | 6/8 |
| start B (center) | 31.8 | 0.189 | 1.849e+00 | 3.877 | 8/8 |
| theta-only fit, from A | 16.0 | 0.155 | 3.318e-01 | 0.663 | 0/8 |
| theta-only fit, from B | 16.7 | 0.150 | 3.312e-01 | 0.664 | 0/8 |
| joint fit w_data 1e-2, from A | 7.7 | 0.027 | 5.870e-01 | 1.000 | 4/8 |
| joint fit w_data 1e-2, from B | 11.6 | 0.061 | 6.457e-01 | 1.122 | 7/8 |

### fv residual: physics loss (pde + vol + bc) on the stage-1 networks, median over 8 materials

| theta | D err % | t+ rmse | physics loss | loss / loss at truth | materials where truth is lower |
|---|---|---|---|---|---|
| truth | 0.0 | 0.000 | 4.391e-01 | 1.000 | 0/8 |
| start A (FNO+TTT) | 5.7 | 0.033 | 4.743e-01 | 1.029 | 6/8 |
| start B (center) | 31.8 | 0.189 | 1.530e+00 | 4.023 | 8/8 |
| theta-only fit, from A | 16.0 | 0.155 | 2.476e-01 | 0.705 | 0/8 |
| theta-only fit, from B | 16.7 | 0.150 | 2.478e-01 | 0.706 | 0/8 |
| joint fit w_data 1e-2, from A | 7.7 | 0.027 | 4.386e-01 | 1.000 | 4/8 |
| joint fit w_data 1e-2, from B | 11.6 | 0.061 | 5.598e-01 | 1.171 | 7/8 |
