"""Equal-material scoring of the historical CNNs on the same held-out IDs.

Each material has one input in the random-current run. Historical triplet-level
squared errors are averaged within material before the test population average.
"""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/"random"
HIST=ROOT.parents[2]/"latest_synthetic_ml"/"reference_runs"


def main():
    with np.load(ROOT/"legacy/data/index.npz") as data:
        ids=data["ids"][int(data["train_n"]):]
    u=np.linspace(0,1,201)
    basis=np.stack([(1-u)**3,3*u*(1-u)**2,3*u*u*(1-u),u**3])
    output={}
    for domain in ("clean","xray_noisy"):
        errors={int(i): [] for i in ids}
        with (HIST/("cnn_"+domain)/"test_predictions.jsonl").open() as file:
            for line in file:
                r=json.loads(line)
                truth=np.array([r["D_target_coefficients_01"],r["t_plus_target_coefficients_01"]])
                pred=np.array([r["D_mean_coefficients_01"],r["t_plus_mean_coefficients_01"]])
                errors[r["material_id"]].append(np.mean(((pred-truth)@basis)**2,axis=1))
        assert all(len(row) for row in errors.values())
        arr=np.array([np.mean(vals,axis=0) for vals in errors.values()])
        d,t=np.sqrt(np.mean(arr,axis=0))
        output[domain]=dict(n=len(arr),n_triplets=sum(map(len,errors.values())),D_normalized_rmse=d,t_normalized_rmse=t,
                            combined_normalized_rmse=float(np.sqrt(np.mean(arr))))
    (ROOT/"report_data").mkdir(exist_ok=True)
    (ROOT/"report_data"/"legacy_three_current_matched.json").write_text(json.dumps(output,indent=2))
    print(output)


if __name__=="__main__":
    main()
