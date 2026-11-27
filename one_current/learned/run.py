"""Fit the original MLP and FNO architectures to a genuine one-current input.

Run from any directory with the project's interpreter. Defaults to the randomly
assigned current protocol in one_current/random/data and writes its results in
one_current/random/learned. Seed 0 selects TTT's rate on 100 validation cases.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "one_current" / "random" / "learned"
sys.path.insert(0, str(REPO))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call, vmap

from analysis.metrics import score_all
from pinn.pde_loss import observed_fields, residual_scale, pde_loss, total_flux, H
from surrogate.data import velocity_from_c
from surrogate.models import SpectralConv2d, masked_pool, n_params

DATA = ROOT.parent / "data"
RATES = (1e-5, 1e-4, 1e-3)
STEPS = 150
EPOCHS = 100
BATCH = 64


def write_json(file, obj):
    temp = file.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(file)


def write_predictions(file, ids, predictions):
    if len(ids) != len(predictions) or not np.isfinite(predictions).all():
        raise ValueError("Invalid predictions")
    temp = file.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez(stream, index=ids, theta=predictions)
    temp.replace(file)


class Spectral(SpectralConv2d):
    """The project's two spectral blocks; construct the result on the input device.

    Concatenating the positive and negative blocks also permits vectorizing over
    independent parameter copies at test time.
    """
    def forward(self, x):
        xf = torch.fft.rfft2(x)
        lo = torch.einsum("bixy,ioxy->boxy", xf[:, :, :self.m1, :self.m2], self.w1)
        hi = torch.einsum("bixy,ioxy->boxy", xf[:, :, -self.m1:, :self.m2], self.w2)
        center = lo.new_zeros(lo.shape[0], lo.shape[1], xf.shape[2]-2*self.m1, self.m2)
        spectrum = F.pad(torch.cat((lo, center, hi), 2), (0, xf.shape[3]-self.m2))
        return torch.fft.irfft2(spectrum, s=x.shape[-2:])


class FNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.lift = nn.Conv2d(6, 11, 1)
        self.spec = nn.ModuleList(Spectral(11, 11, 5, 6) for _ in range(4))
        self.pw = nn.ModuleList(nn.Conv2d(11, 11, 1) for _ in range(4))
        self.proj = nn.Conv2d(11, 22, 1)
        self.head = nn.Sequential(nn.Linear(44,128), nn.GELU(), nn.Linear(128,8))

    def forward(self, features):
        mask = features[:, 2:3]
        x = F.pad(self.lift(features), (0,4,0,3))
        for k, (spec, point) in enumerate(zip(self.spec, self.pw)):
            x = spec(x) + point(x)
            if k < 3:
                x = F.gelu(x)
        x = F.gelu(self.proj(x[:, :, :features.shape[-2], :features.shape[-1]]))
        return self.head(masked_pool(x, mask))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2*13*20+13+1,40), nn.GELU(),
                                 nn.Linear(40,40), nn.GELU(), nn.Linear(40,8))

    def forward(self, features):
        return self.net(torch.cat([features[:, :2].flatten(1),features[:,2,:,0],features[:,5,0,:1]],1))


def feature_tensor(yy, currents, scales, device):
    """Six one-current channels, with RMS taken only from the training set."""
    vv = np.concatenate([velocity_from_c(yy[s:s+128],currents[s:s+128]) for s in range(0,len(yy),128)])
    cc = np.nan_to_num(yy-1.0)
    vv = np.nan_to_num(vv)
    mask = np.broadcast_to(np.isfinite(yy[:,:,:,0,None]),yy.shape)
    if scales is None:
        count = mask.sum()
        scales = (np.sqrt(np.sum(cc**2)/count),np.sqrt(np.sum(vv**2)/count))
    tt = np.broadcast_to(np.linspace(0,1,13)[None,None,:,None],yy.shape)
    xx = np.broadcast_to(np.linspace(0,1,20)[None,None,None,:],yy.shape)
    ii = np.broadcast_to(currents[:,:,None,None]/3, yy.shape)
    feat = torch.as_tensor(np.concatenate([cc/scales[0],vv/scales[1],mask,tt,xx,ii],1),dtype=torch.float32,device=device)
    return feat, scales


def load_data(device):
    split = np.load(DATA / "train_split.npy")
    with np.load(DATA / "train.npz") as d:
        y, io, truth, ids, window = (d[k] for k in ("y","iotas","theta","index","window"))
    if y.shape[1:] != (1,13,20) or io.shape != (len(y),1):
        raise ValueError("Expected one current per material")
    lookup = {}
    scales = None
    for name in ("train","val","id_test"):
        select = np.flatnonzero(split==name)
        if name=="id_test":
            select = select[:300]
        yy = y[select]
        feat, scales = feature_tensor(yy,io[select],scales,device)
        fields = observed_fields(yy,io[select])
        if name=="train":
            scale = residual_scale(fields)
            fields={}
        else:
            fields={k:v.to(device) for k,v in fields.items()}
        lookup[name]=dict(x=feat,target=torch.tensor(truth[select],dtype=torch.float32,device=device),
                          truth=truth[select],index=ids[select],window=window[select],fields=fields)
    return lookup, scale, scales


@torch.no_grad()
def predict(net, x):
    return torch.cat([net(batch).sigmoid() for batch in x.split(256)])


def train(net, sets, folder, seed, device):
    tr,va=sets["train"],sets["val"]
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3)
    batches=math.ceil(len(tr["x"])/BATCH)
    sched=torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 0.5*(1+math.cos(math.pi*min(step,EPOCHS*batches)/(EPOCHS*batches))))
    perms=torch.Generator().manual_seed(seed+1)
    best=np.inf
    history=[]
    start=time.perf_counter()
    for ep in range(1,EPOCHS+1):
        order=torch.randperm(len(tr["x"]),generator=perms).to(device)
        for sl in order.split(BATCH):
            err=(net(tr["x"][sl]).sigmoid()-tr["target"][sl]).square().mean()
            opt.zero_grad(set_to_none=True)
            err.backward()
            opt.step()
            sched.step()
        mse=float((predict(net,va["x"])-va["target"]).square().mean().item())
        history.append(mse)
        if mse<best:
            best=mse
            torch.save(net.state_dict(),folder/"best.pt")
        if ep%5==0 or ep==1:
            print(f"{folder.name} epoch {ep}/100; val MSE {mse:.5g}; {time.perf_counter()-start:.1f}s",flush=True)
    return dict(history=history,wall_s=time.perf_counter()-start,best_val_coef_mse=best)


def per_case_loss(theta, fields, scale):
    idx=torch.arange(len(theta),device=theta.device)
    J=total_flux(theta,fields,idx)
    div=J.diff(dim=-1)/H
    r=(fields["ct"]+0.5*(div[:,:,1:]+div[:,:,:-1]))/scale
    m=fields["mmid"][:,None,:,None].expand_as(r)
    dims=(1,2,3)
    return (r.square()*m).sum(dims)/m.sum(dims)


def adapt_batch(net, x, fields, scale, lr, steps=STEPS):
    # Each slice of a stacked parameter has its own AdamW moment estimates and its
    # own loss; summing case losses preserves each single-material gradient.
    weights={k:v.detach().unsqueeze(0).repeat((len(x),)+(1,)*v.ndim).requires_grad_() for k,v in net.named_parameters()}
    opt=torch.optim.AdamW(weights.values(),lr=lr)
    def one(params, features):
        return functional_call(net,params,(features.unsqueeze(0),)).squeeze(0).sigmoid()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        theta=vmap(one)(weights,x)
        loss=per_case_loss(theta,fields,scale).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return vmap(one)(weights,x).clone().cpu().numpy()


def subset_fields(fields, s):
    return {k:v[s] for k,v in fields.items()}


def validate_independence(net,S,scale):
    sl=slice(0,2)
    fields=subset_fields(S["fields"],sl)
    ensemble=adapt_batch(net,S["x"][sl],fields,scale,1e-4,steps=5)
    refs=[]
    for i in range(2):
        individual=copy.deepcopy(net)
        opt=torch.optim.AdamW(individual.parameters(),lr=1e-4)
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            th=individual(S["x"][i:i+1]).sigmoid()
            loss=pde_loss(th,fields,torch.tensor([i],device=th.device),scale)
            loss.backward()
            opt.step()
        with torch.no_grad():
            refs.append(individual(S["x"][i:i+1]).sigmoid().cpu().numpy()[0])
    difference=np.max(np.abs(ensemble-np.asarray(refs)))
    if difference>2e-5:
        raise ValueError(f"Ensemble failed serial AdamW comparison: {difference}")
    return float(difference)


def adapt_set(net, S, scale, lr, a, b, chunk=32):
    preds=[]
    for i in range(a,b,chunk):
        sl=slice(i,min(i+chunk,b))
        preds.append(adapt_batch(net,S["x"][sl],subset_fields(S["fields"],sl),scale,lr))
    return np.concatenate(preds)


def run(arch,seed,sets,scale,scales,device):
    folder=ROOT/f"{arch}_s{seed}"
    folder.mkdir(exist_ok=True)
    tr,te,va=sets["train"],sets["id_test"],sets["val"]
    torch.manual_seed(seed)
    net=(FNO() if arch=="fno" else MLP()).to(device)
    if not (folder/"training.json").exists():
        history=train(net,sets,folder,seed,device)
        write_json(folder/"training.json",dict(**history,seed=seed,epochs=EPOCHS,batch=BATCH,n_params=n_params(net),
                                               lr=1e-3,n_train=len(tr["index"]),n_val=len(va["index"]),scales=list(scales),device=str(device)))
    net.load_state_dict(torch.load(folder/"best.pt",map_location=device,weights_only=True))
    sup=predict(net,te["x"]).cpu().numpy()
    write_predictions(folder/"supervised.npz",te["index"],sup)
    if (folder/"summary.json").exists():
        return
    independence=validate_independence(net,va,scale)
    selection=ROOT/f"select_{arch}.json"
    if not selection.exists():
        if seed!=0:
            raise ValueError("Run seed zero before the other seeds")
        grid={}
        for lr in RATES:
            th=adapt_set(net,va,scale,lr,0,100)
            scores=score_all(th,va["truth"][:100],va["window"][:100])
            grid[str(lr)]=float(np.median(scores["mse_lnD"]))
            print(arch,"validation TTT",lr,grid[str(lr)],flush=True)
        lr=min(RATES,key=lambda r:grid[str(r)])
        write_json(selection,dict(lr=lr,grid=grid,metric="median validation MSE of log D",n=100,steps=STEPS))
    lr=json.loads(selection.read_text())["lr"]
    start=time.perf_counter()
    th8=adapt_set(net,te,scale,lr,0,8)
    write_predictions(folder/"ttt_first8.npz",te["index"][:8],th8)
    print(folder.name,"released first 8 TTT",flush=True)
    ttt=np.concatenate([th8,adapt_set(net,te,scale,lr,8,len(te["x"]))])
    write_predictions(folder/"ttt.npz",te["index"],ttt)
    wall=time.perf_counter()-start
    def med(p):
        return {k:float(np.median(v)) for k,v in score_all(p,te["truth"],te["window"]).items()}
    write_json(folder/"summary.json",dict(seed=seed,arch=arch,n=len(ttt),supervised=med(sup),ttt=med(ttt),
                                        ttt_steps=STEPS,ttt_lr=lr,ttt_wall_s=wall,serial_equivalence_max_abs=independence,
                                        residual_scale=scale))
    print(folder.name,"done",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,default=ROOT.parent,
                        help="Protocol directory containing data/; writes into its learned/ directory")
    parser.add_argument("--arch",nargs="+",choices=["mlp","fno"],default=["fno","mlp"])
    parser.add_argument("--seed",nargs="+",type=int,default=[0,1,2])
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    a=parser.parse_args()
    ROOT=a.run_root.resolve() / "learned"
    DATA=a.run_root.resolve() / "data"
    ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    print("Loading one-current training records",flush=True)
    sets,scale,scales=load_data(a.device)
    for seed in a.seed:
        for arch in a.arch:
            run(arch,seed,sets,scale,scales,a.device)
