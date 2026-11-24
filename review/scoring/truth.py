import json, glob, sys, numpy as np
sys.path.insert(0, sys.argv[1])
from solver.dataset import generate
recs = sorted((json.load(open(f))["k"], json.load(open(f))["index"]) for f in glob.glob(sys.argv[1] + "/results/lbfgs/B_center/m*.json"))
ks = [k for k, _ in recs]; assert ks == list(range(300)), ks[:5]
idx = [i for _, i in recs]
d = generate(idx)
assert list(d["index"]) == idx, "a test material was dropped"
np.savez(sys.argv[2], theta=d["theta"], window=d["window"], index=d["index"])
print("ok", d["theta"].shape)
