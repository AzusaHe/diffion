
### loss-cur (w_vdata 1): total loss on one fresh residual set (4096 points per current, seed 123), median over 8 materials

| run | D err % | t+ rmse | total | physics | c data | v data | total / truth-fixed | truth-fixed lower |
|---|---|---|---|---|---|---|---|---|
| truth fixed, nets trained (T-cur) | 0.00 | 0.000 | 2.677e-02 | 2.012e-02 | 4.945e-03 | 1.137e-03 | 1.000 | 0/8 |
| joint from truth | 3.50 | 0.014 | 2.342e-02 | 1.742e-02 | 5.083e-03 | 1.081e-03 | 1.002 | 5/8 |
| joint from A | 7.67 | 0.027 | 2.444e-02 | 1.761e-02 | 5.415e-03 | 1.062e-03 | 0.993 | 4/8 |
| joint from B | 11.41 | 0.060 | 3.305e-02 | 2.248e-02 | 7.876e-03 | 1.150e-03 | 1.219 | 7/8 |

Per material, total loss:

| run | m000 | m001 | m002 | m003 | m004 | m005 | m006 | m007 |
|---|---|---|---|---|---|---|---|---|
| truth fixed, nets trained (T-cur) | 2.55e-02 | 2.80e-02 | 2.22e-02 | 2.11e-02 | 2.94e-02 | 3.89e-02 | 2.26e-02 | 2.87e-02 |
| joint from truth | 2.42e-02 | 2.86e-02 | 2.23e-02 | 2.11e-02 | 3.02e-02 | 4.17e-02 | 1.90e-02 | 2.27e-02 |
| joint from A | 2.45e-02 | 2.74e-02 | 2.29e-02 | 2.12e-02 | 3.02e-02 | 4.19e-02 | 1.89e-02 | 2.44e-02 |
| joint from B | 3.06e-02 | 3.88e-02 | 2.76e-02 | 4.98e-02 | 3.32e-02 | 8.71e-02 | 2.08e-02 | 3.29e-02 |

Lowest per material: m000 joint from truth, m001 joint from A, m002 truth fixed, nets trained (T-cur), m003 truth fixed, nets trained (T-cur), m004 truth fixed, nets trained (T-cur), m005 truth fixed, nets trained (T-cur), m006 joint from A, m007 joint from truth

### loss-new (w_vdata 0): total loss on one fresh residual set (4096 points per current, seed 123), median over 8 materials

| run | D err % | t+ rmse | total | physics | c data | v data | total / truth-fixed | truth-fixed lower |
|---|---|---|---|---|---|---|---|---|
| truth fixed, nets trained (T-new) | 0.00 | 0.000 | 1.198e-02 | 9.205e-03 | 2.396e-03 | nan | 1.000 | 0/8 |
| joint from A, no v data (V2) | 6.58 | 0.024 | 1.137e-02 | 8.286e-03 | 2.615e-03 | nan | 1.022 | 5/8 |
| joint from B, no v data (V2) | 11.64 | 0.047 | 1.707e-02 | 1.215e-02 | 4.343e-03 | nan | 1.333 | 7/8 |

Per material, total loss:

| run | m000 | m001 | m002 | m003 | m004 | m005 | m006 | m007 |
|---|---|---|---|---|---|---|---|---|
| truth fixed, nets trained (T-new) | 1.53e-02 | 1.22e-02 | 7.16e-03 | 8.56e-03 | 1.64e-02 | 3.05e-02 | 9.63e-03 | 1.18e-02 |
| joint from A, no v data (V2) | 1.47e-02 | 1.26e-02 | 8.11e-03 | 8.68e-03 | 1.69e-02 | 3.21e-02 | 6.85e-03 | 1.01e-02 |
| joint from B, no v data (V2) | 1.99e-02 | 1.56e-02 | 1.07e-02 | 2.73e-02 | 1.81e-02 | 7.38e-02 | 9.26e-03 | 1.61e-02 |

Lowest per material: m000 joint from A, no v data (V2), m001 truth fixed, nets trained (T-new), m002 truth fixed, nets trained (T-new), m003 truth fixed, nets trained (T-new), m004 truth fixed, nets trained (T-new), m005 truth fixed, nets trained (T-new), m006 joint from A, no v data (V2), m007 joint from A, no v data (V2)
