| run | input | D err % (start -> fit) | better | t+ rmse | c misfit % | v misfit % | c / v misfit % after stage 1 |
|---|---|---|---|---|---|---|---|
| staged w100 (earlier) | A | 5.7 -> 13.3 | 0/8 | 0.141 | 0.265 | nan | nan / nan |
| staged w100 (earlier) | B | 31.8 -> 17.3 | 5/8 | 0.172 | 0.273 | nan | nan / nan |
| staged w1 (earlier) | A | 5.7 -> 27.9 | 0/8 | 0.185 | 0.266 | nan | nan / nan |
| staged w1 (earlier) | B | 31.8 -> 35.7 | 2/8 | 0.227 | 0.260 | nan | nan / nan |
| data-first joint w_data=1e-2 | A | 5.7 -> 7.7 | 3/8 | 0.027 | 0.070 | 0.03 | 0.0199 / 0.032 |
| data-first joint w_data=1e-2 | B | 31.8 -> 11.6 | 7/8 | 0.061 | 0.089 | 0.03 | 0.0199 / 0.032 |
| data-first joint w_data=1e-3 | A | 5.7 -> 7.3 | 3/8 | 0.030 | 0.137 | 0.06 | 0.0199 / 0.032 |
| data-first joint w_data=1e-3 | B | 31.8 -> 13.9 | 7/8 | 0.048 | 0.188 | 0.08 | 0.0199 / 0.032 |
| data-first joint w_data=1e-4 | A | 5.7 -> 8.9 | 2/8 | 0.034 | 0.238 | 0.20 | 0.0199 / 0.032 |
| data-first joint w_data=1e-4 | B | 31.8 -> 17.2 | 7/8 | 0.061 | 0.335 | 0.23 | 0.0199 / 0.032 |
| data-first, c and v frozen (theta only) | A | 5.7 -> 16.0 | 0/8 | 0.155 | 0.020 | 0.03 | 0.0199 / 0.032 |
| data-first, c and v frozen (theta only) | B | 31.8 -> 16.7 | 6/8 | 0.150 | 0.020 | 0.03 | 0.0199 / 0.032 |
| data-first, c frozen (v, theta) w_data=1e-2 | A | 5.7 -> 13.2 | 0/8 | 0.095 | 0.020 | 0.10 | 0.0199 / 0.032 |
| data-first, c frozen (v, theta) w_data=1e-2 | B | 31.8 -> 15.8 | 8/8 | 0.101 | 0.020 | 0.13 | 0.0199 / 0.032 |
| fv_wd1e-2 | A | 5.7 -> 7.2 | 3/8 | 0.027 | 0.069 | 0.03 | 0.0199 / 0.032 |
| fv_wd1e-2 | B | 31.8 -> 11.6 | 7/8 | 0.060 | 0.083 | 0.03 | 0.0199 / 0.032 |
| hardbc_wd1e-2 | A | 5.7 -> 6.0 | 1/8 | 0.034 | 0.092 | 0.05 | 0.0334 / 0.035 |
| hardbc_wd1e-2 | B | 31.8 -> 16.4 | 6/8 | 0.068 | 0.134 | 0.04 | 0.0308 / 0.031 |
