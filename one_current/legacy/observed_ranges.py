"""Rescore the clean/noisy archive on each input's own measured concentration interval."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "one_current/random"
SOURCE = REPO.parent / "latest_synthetic_ml"
HIST = SOURCE / "reference_runs"
OUT = RUN / "report_data/legacy_observed_ranges.json"
sys.path.insert(0, str(SOURCE / "src"))
from three_current_cnn import change_aware_temporal_compression, USABLE_STATUSES

FIELDS = {"clean":"concentration_nd","xray_noisy":"concentration_noisy_nd"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def limits(field):
    return [float(field.min()),float(field.max())]


def input_ranges(path):
    """Recover the bounds of each individual CNN input, including its actual keyframe selection."""
    mid=int(path.stem.removeprefix("material_"))
    result={}
    with np.load(path,allow_pickle=False) as raw:
        tau, valid, currents=raw["tau"],raw["valid_time_mask"],raw["iota"]
        eligible=np.flatnonzero(np.isin(raw["status"].astype(str),list(USABLE_STATUSES)) & (valid.sum(1)>=2))
        for domain,field in FIELDS.items():
            observations=raw[field]
            compressed,full={},{}
            for j in eligible:
                values=observations[j,valid[j]].astype(np.float32)
                assert np.isfinite(values).all()
                full[j]=limits(values)
                if len(values)<=48 or limits(values[[0,-1]])==full[j]:
                    # Both endpoints are always selected in a long trajectory.
                    compressed[j]=full[j]
                else:
                    keyframes,_,mask=change_aware_temporal_compression(observations[j],tau,valid[j],n_keyframes=48)
                    compressed[j]=limits(keyframes[mask])
            result[domain]=(compressed,full)
        labels=np.r_[raw["D_coefficients_01"],raw["t_plus_coefficients_01"]]
    return mid,(currents,labels,result,valid.sum(1))


def interval(bounds,indices):
    # c_nd = c_phys/3000; the property curves are calibrated on 0..6000 mol/m³.
    # Cut the observed interval at the property domain limits if a noisy measurement crosses them.
    left=min(bounds[int(j)][0] for j in indices)/2
    right=max(bounds[int(j)][1] for j in indices)/2
    lo,hi=max(0.0,left),min(1.0,right)
    assert lo<hi
    return lo,hi,(lo!=left or hi!=right)


def basis(u):
    return np.asarray([(1-u)**3,3*u*(1-u)**2,3*u*u*(1-u),u**3])


def mse(fit_minus_truth,lo,hi):
    error=np.asarray(fit_minus_truth).reshape(2,4)@basis(np.linspace(lo,hi,201))
    return np.mean(error*error,axis=1)


def summarize(squared):
    d,t=np.mean(squared,axis=0)
    return dict(D_normalized_rmse=float(np.sqrt(d)),t_normalized_rmse=float(np.sqrt(t)),
                combined_normalized_rmse=float(np.sqrt((d+t)/2)))


def cnn_scores(domain,inputs,chosen,cache_rows):
    folder=RUN/f"legacy/cnn_{domain}"
    with np.load(folder/"predictions.npz") as z:
        ids,pred,truth=z["ids"],z["theta"],z["truth"]
    records={int(i):[] for i in ids}
    history=read(HIST/f"cnn_{domain}/history.json")
    assert history["run_configuration"]["keyframes"]==48
    assert set(history["split_material_ids"]["test"])==set(records)
    cached=RUN/f"legacy/data/{domain}.npy"
    actual=np.load(cached,mmap_mode="r") if cached.exists() else None
    with np.load(RUN/"legacy/data/index.npz") as z:
        masks=z[f"mask_{domain}"]
    widths_one=[]; widths_three=[]; clipped_one=0;clipped_three=0
    one=[]
    for k,mid in enumerate(ids):
        j=chosen[mid]; bounds=inputs[mid][2][domain][0]
        if actual is not None:
            np.testing.assert_array_equal(bounds[j],limits(actual[cache_rows[mid],masks[cache_rows[mid]]]))
        np.testing.assert_allclose(inputs[mid][1],truth[k],atol=1e-7)
        lo,hi,clipped=interval(bounds,[j])
        widths_one.append(hi-lo);clipped_one+=clipped
        one.append(mse(pred[k].astype(float)-truth[k],lo,hi))
    seen=set()
    with (HIST/f"cnn_{domain}/test_predictions.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r=json.loads(line); mid=r["material_id"]; indices=r["current_indices"]
            key=(mid,*sorted(indices))
            assert key not in seen
            seen.add(key)
            np.testing.assert_allclose(inputs[mid][0][indices],r["currents_iota"],rtol=1e-6)
            label=np.array([r["D_target_coefficients_01"],r["t_plus_target_coefficients_01"]])
            np.testing.assert_allclose(label.ravel(),inputs[mid][1],atol=1e-7)
            est=np.array([r["D_mean_coefficients_01"],r["t_plus_mean_coefficients_01"]])
            lo,hi,clipped=interval(inputs[mid][2][domain][0],indices)
            widths_three.append(hi-lo);clipped_three+=clipped
            records[mid].append(mse(est-label,lo,hi))
    assert len(seen)==18934 and len(ids)==389 and all(records.values())
    three=np.array([np.mean(records[mid],axis=0) for mid in ids])
    return dict(n_materials=len(ids),n_triplets=len(seen),one=summarize(one),three=summarize(three),
                n_intervals_clipped_to_property_domain=dict(one=clipped_one,three=clipped_three),
                median_input_width_u=dict(one=float(np.median(widths_one)),three=float(np.median(widths_three))),
                material_ids=ids.tolist(),one_combined_by_material=np.sqrt(np.mean(one,axis=1)).tolist(),
                three_combined_by_material=np.sqrt(np.mean(three,axis=1)).tolist())


def pinn_scores(domain,inputs):
    new=RUN/f"legacy/pinn_{domain}"
    old=HIST/("pinn_clean_accepted" if domain=="clean" else "pinn_xray_noisy_rejected")
    accepted=domain=="clean"
    archived=read((old if accepted else old/"rejected_final_checkpoint_evaluation")/"pinn_evaluation_summary.json")
    single=read(new/"one_current_result.json")
    checkpoint=torch.load(new/single["checkpoint"],map_location="cpu",weights_only=False)
    three=np.array([archived["D_coefficients_01"],archived["t_plus_coefficients_01"]])
    one=np.array([checkpoint["effective_D_coefficients_01"],checkpoint["effective_t_plus_coefficients_01"]])
    result=dict(material_id=2,truth=inputs[2][1].reshape(2,4).tolist())
    for name,folder,coeff in (("one",new,one),("three",old,three)):
        config=read(folder/"run_config.json")
        phys=config["physics"]
        assert (phys["concentration_ref_mol_m3"],phys["property_concentration_min_mol_m3"],phys["property_concentration_max_mol_m3"])==(3000,0,6000)
        obs=config["observations"];indices=obs["current_indices"]
        assert indices==([3] if name=="one" else [2,3,7]) and obs["material_id"]==2
        assert obs["observation_time_counts"]==obs["valid_time_counts"]==inputs[2][3][indices].tolist()
        assert obs["concentration_field"]==FIELDS[domain]
        lo,hi,clipped=interval(inputs[2][2][domain][1],indices)
        truth=inputs[2][1].reshape(2,4)
        full_error=np.sqrt(mse(coeff-truth,0,1))
        if name=="one":
            np.testing.assert_allclose(full_error,[single["full_curve_scores"][key+"_curve_rmse_01"] for key in ("D","t_plus")],atol=2e-7)
        result[name]=dict(**summarize([mse(coeff-truth,lo,hi)]),status="accepted" if accepted else "rejected",
                          u_min=lo,u_max=hi,clipped_to_property_domain=clipped,coefficients=coeff.tolist())
    return result


def main():
    with np.load(RUN/"legacy/data/index.npz") as z:
        ids=z["ids"]; selected=set(ids[int(z["train_n"]):]);chosen=dict(zip(ids,z["chosen"]))
        rows=dict(zip(ids,range(len(ids))))
    paths=[p for p in (SOURCE/"data/xray_noisy").glob("*/samples/material_*.npz") if int(p.stem[9:]) in selected]
    with ThreadPoolExecutor(max_workers=5) as pool:
        inputs=dict(pool.map(input_ranges,paths))
    assert len(paths)==len(inputs)==389 and set(inputs)==selected
    report=dict(scoring="201 evenly spaced points per observed interval; u=c_nd/2, intersected with [0,1]. "
                       "CNN bounds use retained valid input keyframes, PINN bounds use all observed frames. "
                       "CNN squared errors are averaged within each material over its triplets, then across materials before taking roots.")
    for dom in FIELDS:
        report[dom]=dict(cnn=cnn_scores(dom,inputs,chosen,rows),pinn=pinn_scores(dom,inputs))
        print(dom,[(k,report[dom]["cnn"][k]["combined_normalized_rmse"]) for k in ("one","three")],flush=True)
    OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
