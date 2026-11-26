### Physics loss (pde + vol + bc) at each theta, median over 8 materials

before = the stage-1 networks (v fitted to v_obs, as in diag_truth); after = v retrained on the physics alone at that theta (c and theta fixed).  Ratio and count are against the truth row of the same column.

| theta | D err % | t+ rmse | before: loss | before: / truth | before: truth lower | after: loss | after: pde | after: vol | after: bc | after: / truth | after: truth lower |
|---|---|---|---|---|---|---|---|---|---|---|---|
| truth | 0.0 | 0.000 | 5.754e-01 | 1.000 | 0/8 | 1.505e-01 | 1.08e-01 | 4.67e-02 | 3.22e-03 | 1.000 | 0/8 |
| start A (FNO+TTT) | 5.7 | 0.033 | 5.822e-01 | 1.027 | 6/8 | 1.544e-01 | 1.11e-01 | 4.95e-02 | 3.81e-03 | 1.026 | 6/8 |
| start B (center) | 31.8 | 0.189 | 1.849e+00 | 3.877 | 8/8 | 5.675e-01 | 3.86e-01 | 5.96e-02 | 7.38e-02 | 3.816 | 7/8 |
| theta-only fit, from A | 16.0 | 0.155 | 3.318e-01 | 0.663 | 0/8 | 9.481e-02 | 6.35e-02 | 2.43e-02 | 2.44e-03 | 0.710 | 0/8 |
| theta-only fit, from B | 16.7 | 0.150 | 3.312e-01 | 0.664 | 0/8 | 9.462e-02 | 6.23e-02 | 2.39e-02 | 2.37e-03 | 0.703 | 1/8 |
| joint fit w_data 1e-2, from A | 7.7 | 0.027 | 5.870e-01 | 1.000 | 4/8 | 1.572e-01 | 1.12e-01 | 4.46e-02 | 2.93e-03 | 1.015 | 5/8 |
| joint fit w_data 1e-2, from B | 11.6 | 0.061 | 6.457e-01 | 1.122 | 7/8 | 1.740e-01 | 1.25e-01 | 4.78e-02 | 5.35e-03 | 1.094 | 6/8 |

Per material, physics loss after the v retraining (pde + vol + bc):

| theta | m000 | m001 | m002 | m003 | m004 | m005 | m006 | m007 |
|---|---|---|---|---|---|---|---|---|
| truth | 1.56e-01 | 1.21e-01 | 2.04e-01 | 8.00e-01 | 1.45e-01 | 2.60e-01 | 1.38e-01 | 5.47e-02 |
| start A (FNO+TTT) | 1.61e-01 | 1.24e-01 | 2.35e-01 | 5.33e-01 | 1.45e-01 | 2.61e-01 | 1.48e-01 | 6.03e-02 |
| start B (center) | 5.96e-01 | 1.02e+00 | 7.74e-01 | 5.39e-01 | 3.42e-01 | 1.31e+00 | 1.83e-01 | 4.11e-01 |
| theta-only fit, from A | 7.24e-02 | 1.08e-01 | 1.22e-01 | 5.61e-02 | 1.45e-01 | 2.26e-01 | 8.14e-02 | 4.50e-02 |
| theta-only fit, from B | 7.13e-02 | 1.08e-01 | 1.18e-01 | 5.50e-02 | 1.47e-01 | 2.26e-01 | 8.16e-02 | 4.45e-02 |
| joint fit w_data 1e-2, from A | 1.64e-01 | 1.26e-01 | 2.07e-01 | 5.68e-01 | 1.50e-01 | 2.60e-01 | 1.15e-01 | 5.55e-02 |
| joint fit w_data 1e-2, from B | 1.95e-01 | 1.33e-01 | 2.23e-01 | 6.33e-01 | 1.53e-01 | 3.07e-01 | 1.26e-01 | 1.23e-01 |

Lowest after the v retraining, per material: m000 theta-only fit, from B, m001 theta-only fit, from B, m002 theta-only fit, from B, m003 theta-only fit, from B, m004 theta-only fit, from A, m005 theta-only fit, from B, m006 theta-only fit, from A, m007 theta-only fit, from B
