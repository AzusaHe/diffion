"""Single-current rerun of the separate clean and X-ray CNN campaign."""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SOURCE = Path(__file__).resolve().parent
HERE = SOURCE.parent / "random" / "legacy"
LEGACY = SOURCE.parents[2] / "latest_synthetic_ml"
sys.path.insert(0, str(LEGACY / "src"))
import three_current_cnn as cnn
import train_three_current_cnn as training

DOMAINS = {"clean": "concentration_nd", "xray_noisy": "concentration_noisy_nd"}
CACHE = HERE / "data"
cnn.N_EXPERIMENTS = 1


def prepare():
    CACHE.mkdir(exist_ok=True,parents=True)
    hist = json.loads((LEGACY / "reference_runs/cnn_clean/history.json").read_text())
    split = hist["split_material_ids"]
    ids = np.asarray(split["train"] + split["test"])
    paths = {int(p.stem[9:]): p for p in (LEGACY / "data/xray_noisy").glob("*/samples/material_*.npz")}
    dims = (len(ids),48,1160)
    fields = {key: np.lib.format.open_memmap(CACHE / f"{key}.npy", mode="w+", dtype="float32", shape=dims)
              for key in DOMAINS}
    times = {key: np.empty(dims[:2],np.float32) for key in DOMAINS}
    masks = {key: np.empty(dims[:2],bool) for key in DOMAINS}
    io = np.empty(len(ids)); chosen = np.empty(len(ids),int); labels = np.empty((len(ids),8))
    def process(row):
        with np.load(paths[int(ids[row])],allow_pickle=False) as art:
            currents, status, frames = art["iota"], art["status"].astype(str), art["valid_time_mask"]
            eligible = np.flatnonzero(np.isin(status,list(cnn.USABLE_STATUSES)) & (frames.sum(1)>=2))
            assert len(eligible)>=3  # retain the original comparison cohort
            index = int(np.random.default_rng([236,int(ids[row])]).choice(eligible))
            chosen[row],io[row]=index,currents[index]
            tau = art["tau"]
            labels[row] = np.r_[art["D_coefficients_01"],art["t_plus_coefficients_01"]]
            for key, source in DOMAINS.items():
                compact, t, m = cnn.change_aware_temporal_compression(art[source][index],tau,frames[index],n_keyframes=48)
                fields[key][row],times[key][row],masks[key][row]=compact,t,m
    with ThreadPoolExecutor(max_workers=4) as pool:
        for k,_ in enumerate(pool.map(process,range(len(ids))),1):
            if k % 250 == 0: print("processed",k,flush=True)
    for matrix in fields.values(): matrix.flush()
    with np.load(paths[int(ids[0])]) as a:
        x, horizon = a["x_cells_nd"], float(a["tau"][-1])
    np.savez(CACHE / "index.tmp.npz", ids=ids,train_n=len(split["train"]),iota=io,chosen=chosen,target=labels,x=x,horizon=horizon,
             **{f"time_{k}":v for k,v in times.items()},**{f"mask_{k}":v for k,v in masks.items()})
    os.replace(CACHE / "index.tmp.npz",CACHE / "index.npz")
    print("prepared",len(ids),"materials",flush=True)


class Profiles(Dataset):
    def __init__(self, domain, is_train):
        self.field=np.load(CACHE / f"{domain}.npy",mmap_mode="r")
        with np.load(CACHE / "index.npz") as d:
            self.t,self.m,self.x,self.horizon=d[f"time_{domain}"],d[f"mask_{domain}"],d["x"],float(d["horizon"])
            self.io,self.ids,self.target=d["iota"],d["ids"],d["target"].astype(np.float32)
            split=int(d["train_n"])
        self.rows=np.arange(split) if is_train else np.arange(split,len(self.ids))
    def __len__(self): return len(self.rows)
    def __getitem__(self,k):
        i=self.rows[k]
        profile=cnn.build_profile_channels(self.field[i],self.t[i],self.m[i],self.x,global_horizon_s=self.horizon)
        return {"profiles":torch.from_numpy(profile[None]),"current_metadata":torch.tensor([[self.io[i]/5]],dtype=torch.float32),
                "target":torch.from_numpy(self.target[i]),"material_id":int(self.ids[i])}


def train(domain, device_name):
    out=HERE / ("cnn_"+domain); out.mkdir(exist_ok=True)
    training.seed_everything(242)
    torch.set_num_threads(1)
    device=torch.device(device_name)
    model=cnn.ThreeCurrentSetCNN().to(device)
    loss=cnn.CoefficientCurveLoss(curve_points=64).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,50,eta_min=1e-6)
    amp=device.type=="cuda"
    scaler=torch.amp.GradScaler("cuda",enabled=amp,init_scale=1024,growth_interval=1000)
    tr,te=Profiles(domain,True),Profiles(domain,False)
    loader=DataLoader(tr,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(343))
    path=out/"last.pt"; first=0
    if path.exists():
        last=torch.load(path,map_location=device,weights_only=False)
        model.load_state_dict(last["weights"]);optimizer.load_state_dict(last["opt"]);scheduler.load_state_dict(last["sched"])
        scaler.load_state_dict(last["scale"]); loader.generator.set_state(last["loader"].cpu());torch.set_rng_state(last["cpu_rng"].cpu())
        if amp: torch.cuda.set_rng_state(last["gpu_rng"].cpu(),device)
        first=last["epoch"]
    for epoch in range(first+1,51):
        metrics=training.train_one_epoch(model,loader,loss,optimizer,device=device,amp_enabled=amp,scaler=scaler,gradient_clip=1.)
        scheduler.step()
        snapshot=dict(epoch=epoch,weights=model.state_dict(),opt=optimizer.state_dict(),sched=scheduler.state_dict(),scale=scaler.state_dict(),
                      loader=loader.generator.get_state(),cpu_rng=torch.get_rng_state(),gpu_rng=torch.cuda.get_rng_state(device) if amp else None)
        torch.save(snapshot,out/"last.tmp");os.replace(out/"last.tmp",path)
        print(domain,epoch,metrics["loss_total"],flush=True)
    model.eval()
    predictions=[]
    with torch.inference_mode():
        for batch in DataLoader(te,batch_size=16):
            predictions.extend(model(batch["profiles"].to(device),batch["current_metadata"].to(device))["coefficients_01"].cpu().numpy())
    predictions=np.asarray(predictions)
    truth=te.target[te.rows]
    u=np.linspace(0,1,201)
    basis=np.stack([(1-u)**3,3*u*(1-u)**2,3*u*u*(1-u),u**3],axis=0)
    errors=(predictions-truth).reshape(-1,2,4)@basis
    rmse=np.sqrt(np.mean(errors**2,axis=(0,2)))
    score=dict(n=len(predictions),D_normalized_rmse=float(rmse[0]),t_normalized_rmse=float(rmse[1]),combined_normalized_rmse=float(np.sqrt(np.mean(rmse**2))))
    np.savez(out/"predictions.npz",ids=te.ids[te.rows],theta=predictions,truth=truth)
    config=dict(protocol="one_random_usable_current_per_material_seed_236",experiments_per_item=1,input_channels=[DOMAINS[domain]+"_minus_1","tau_over_campaign_horizon","x_nd","valid_time_mask"],dataset_scan={"tau_horizon":te.horizon})
    torch.save(dict(model_configuration=model.configuration_dict(), model_state_dict=model.state_dict(),run_configuration={"preprocessing_contract":config}),out/"final_checkpoint.pt")
    (out/"summary.json").write_text(json.dumps(score,indent=2)); print(score,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job",choices=["prepare",*DOMAINS])
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    if args.job=="prepare": prepare()
    else: train(args.job,args.device)
