#!/bin/bash
# The PINN tuning tests (data-first schedule, w_data 1e-2, physics weights 1, materials 0:8).
#   bash results/pinn_tune/run_all.sh <name> [<name> ...]   runs those tests one after another
# Two queues at a time fit on a 24 GB GPU; four at once overflow it and stall.
cd "$(dirname "$0")/../.."
O=results/pinn_tune
COMMON="--build cv --materials 0:8 --fit-v --fit-steps 30000 --fit-lbfgs-iters 3000 --steps 5000 --lr 1e-4 --phi-lr-mult 10 --lbfgs-iters 300 --w-data 1e-2 --w-pde 1 --w-vol 1 --w-bc 1"
C="--prefit-cache results/pinn_prefit/cache"
declare -A ARGS=(
  [truth_frozen_cv]="--starts truth --freeze c,v $C"
  [truth_joint]="--starts truth $C"
  [smooth_w1e-2]="--starts fno_ttt,center --w-smooth 1e-2 $C"
  [prior_w0.1]="--starts fno_ttt --w-prior 0.1 $C"
  [prior_w1]="--starts fno_ttt --w-prior 1 $C"
  [prior_w10]="--starts fno_ttt --w-prior 10 $C"
  [hardbc]="--starts fno_ttt,center --hard-bc --prefit-cache $O/cache_hardbc"
)
for name in "$@"; do
  python -m pinn.torch_pinn $COMMON --out $O/$name ${ARGS[$name]} > $O/run_$name.txt 2>&1
done
