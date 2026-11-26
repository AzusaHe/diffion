| run | D err % (start -> fit) | better | t+ rmse (start -> fit) | c misfit % |
|---|---|---|---|---|
| data-first joint w_data 1e-2 (earlier) | 5.69 -> 7.68 | 3/8 | 0.033 -> 0.027 | 0.070 |
| two-stage from solver field, w_data 1e-2 | 5.69 -> 6.50 | 2/8 | 0.033 -> 0.031 | 0.058 |
| two-stage from solver field, w_data 1 | 5.69 -> 7.74 | 1/8 | 0.033 -> 0.032 | 0.029 |

stage 0 (solver field at the fno_ttt theta), medians: solver vs measured data 0.166 %; network fit to the solver field 4.6e-04; misfit to the measured data c 0.170 %, v 2.556 %
