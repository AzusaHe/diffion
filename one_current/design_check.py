"""Local observability on the same eight materials as the PINN comparison."""
import csv
from pathlib import Path
import numpy as np
import jax
from inverse.lbfgs import _forward
from solver.dataset import DATA


def main():
    root=Path(__file__).resolve().parent/"random"
    test=np.flatnonzero(np.load(root/"data/train_split.npy")=="id_test")[:8]
    rank=np.load(root/"data/current_rank.npy")[test]
    with np.load(root/"data/train.npz") as a, np.load(DATA/"train.npz") as b:
        ids,theta,single_k=a["index"][test],a["theta"][test],a["K"][test]
        currents,old_k=b["iotas"][test],b["K"][test]
        np.testing.assert_array_equal(a["iotas"][test,0],currents[np.arange(8),rank])
    evaluate=jax.jit(lambda t, io: _forward(t,io)[0])
    h=0.0001
    rows=[]
    for k in range(8):
        cols=[np.asarray(evaluate(theta[k]+h*e,currents[k])-evaluate(theta[k]-h*e,currents[k]))/(2*h)
              for e in np.eye(8)]
        jac=np.stack(cols,axis=-1)
        sel=rank[k]
        def cond(j):
            s=np.linalg.svd(j.reshape(-1,8),compute_uv=False)
            return float(s[0]/s[-1])
        rows.append(dict(material_id=int(ids[k]),rank=int(sel),iota=float(currents[k,sel]),
                         single_K=int(single_k[k]),original_K=int(old_k[k]),
                         one_available=cond(jac[sel,:single_k[k]]),one_same_frames=cond(jac[sel,:old_k[k]]),
                         three_same_frames=cond(jac[:,:old_k[k]])))
    path=root/"report_data"/"local_sensitivity.csv"
    path.parent.mkdir(exist_ok=True)
    with path.open("w",newline="") as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    print({key:np.median([r[key] for r in rows]) for key in ("one_available","one_same_frames","three_same_frames")})


if __name__=="__main__":
    main()
