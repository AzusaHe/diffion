"""Train and test the inverse MLP/FNO on paired clean and Gaussian observations.

Run after the shared binned.npz exists. Each network has one supervised
checkpoint. TTT updates a separate copy of all its weights per test material.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call, vmap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from pinn.material import alpha_V, bernstein_basis
from surrogate.models import SpectralConv2d, masked_pool, n_params

WIDTH = 11
EPOCHS = 100
STEPS = 150
RATES = (1e-5, 1e-4, 1e-3)
BATCH = 64
RESULTS = HERE / 'results'
CHECKPOINTS = HERE / 'learned_checkpoints'


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def save_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def spectra(x, w1, w2, batched):
    fourier = torch.fft.rfft2(x)
    mode_t, mode_x = w1.shape[-2:]
    equation = 'biuv,biouv->bouv' if batched else 'biuv,iouv->bouv'
    low = torch.einsum(equation, fourier[:, :, :mode_t, :mode_x], w1)
    high = torch.einsum(equation, fourier[:, :, -mode_t:, :mode_x], w2)
    mid = low.new_zeros(x.shape[0], low.shape[1], x.shape[-2] - 2*mode_t, mode_x)
    spectrum = F.pad(torch.cat((low, mid, high), dim=2), (0, fourier.shape[-1] - mode_x))
    return torch.fft.irfft2(spectrum, s=x.shape[-2:])


class Spectral(SpectralConv2d):
    def forward(self, x):
        return spectra(x, self.w1, self.w2, False)


class FNO(nn.Module):
    def __init__(self, currents):
        super().__init__()
        self.currents = currents
        self.lift = nn.Conv2d(5*currents + 1, WIDTH, 1)
        self.spec = nn.ModuleList(Spectral(WIDTH, WIDTH, 5, 6) for _ in range(4))
        self.pw = nn.ModuleList(nn.Conv2d(WIDTH, WIDTH, 1) for _ in range(4))
        self.proj = nn.Conv2d(WIDTH, 2*WIDTH, 1)
        self.head = nn.Sequential(nn.Linear(4*WIDTH, 128), nn.GELU(), nn.Linear(128, 8))

    def mask(self, features):
        return features[:, 2*self.currents:3*self.currents].amax(1, keepdim=True)

    def forward(self, features):
        x = F.pad(self.lift(features), (0, 4, 0, 3))
        for k, (spectral, point) in enumerate(zip(self.spec, self.pw)):
            x = spectral(x) + point(x)
            if k < 3:
                x = F.gelu(x)
        x = F.gelu(self.proj(x[:, :, :13, :20]))
        return self.head(masked_pool(x, self.mask(features)))


class MLP(nn.Module):
    def __init__(self, currents):
        super().__init__()
        self.currents = currents
        self.net = nn.Sequential(nn.Linear(currents*(2*13*20 + 2*13 + 1), 40), nn.GELU(),
                                 nn.Linear(40, 40), nn.GELU(), nn.Linear(40, 8))

    def forward(self, features):
        k = self.currents
        return self.net(torch.cat([features[:, :2*k].flatten(1),
                                   features[:, 2*k:4*k, :, 0].flatten(1),
                                   features[:, 4*k:5*k, 0, 0]], 1))


def prepare(z, domain, currents, device):
    """Compute v from measured c using per-current valid intervals and physical times."""
    def tensor(name, dtype=torch.float32):
        return torch.as_tensor(z[name][:, :currents], dtype=dtype, device=device)
    raw = tensor(domain)
    valid = tensor('mask', torch.bool)
    times = tensor('tau')
    iota = tensor('iota')
    assert torch.equal(torch.isfinite(raw).all(-1), valid)
    c = torch.nan_to_num(raw, nan=1.0)
    times = torch.where(valid, times, 0.0)
    midpoint = valid[:, :, 1:] & valid[:, :, :-1]
    delta = torch.where(midpoint, times.diff(dim=2), 1.0)
    assert (delta > 0).all()
    ct = c.diff(dim=2) / delta[..., None] * midpoint[..., None]
    z4 = torch.zeros_like(c[:, :, :1])
    dcdt = (torch.cat([ct, z4], 2) + torch.cat([z4, ct], 2))
    div = F.pad(midpoint.float(), (0, 1)) + F.pad(midpoint.float(), (1, 0))
    dcdt = dcdt / div[..., None].clamp_min(1)
    cf = (c[..., 1:] + c[..., :-1])/2
    a, _ = alpha_V(cf)
    _, volume = alpha_V(c)
    faces = torch.cat([c[..., :1], cf, c[..., -1:]], dim=-1)
    flux = iota[:, :, None, None] - F.pad(dcdt.cumsum(-1)/20, (1, 0))
    v = [torch.zeros_like(cf[..., 0])]
    q = flux[..., 0]
    for j in range(20):
        v.append((v[-1] - volume[..., j]*(flux[..., j+1] - q)) /
                 (1 - volume[..., j]*faces[..., j+1]))
        q = flux[..., j+1] - faces[..., j+1]*v[-1]
    velocity = torch.stack([(left+right)/2 for left,right in zip(v[:-1], v[1:])], -1)
    splits = {key: torch.as_tensor(np.flatnonzero(z['split'] == key), device=device)
              for key in ('train', 'val', 'test')}
    tr = splits['train']
    mask = valid[..., None].expand_as(c)
    scale = torch.stack([(channel[tr].square()*mask[tr]).sum((0,2,3)) /
                         mask[tr].sum((0,2,3)) for channel in (c-1, velocity)]).sqrt().clamp_min(1e-8)
    feat = torch.cat([(c-1)/scale[0, :, None, None]*mask,
                      velocity/scale[1, :, None, None]*mask, mask,
                      times[..., None].expand_as(c)/(36000/33640),
                      iota[:, :, None, None].expand_as(c)/5,
                      torch.linspace(0, 1, 20, device=device)[None,None,None,:].expand(len(c),1,13,20)],1)
    if not torch.isfinite(feat).all():
        raise ValueError('Nonfinite input features')
    ct_scale = float((ct[tr].square().sum()/midpoint[tr].sum()/20).sqrt())
    fields = dict(basis=bernstein_basis((cf/2).clamp(0,1)), a_cx=a*c.diff(dim=-1)*20,
                  volume=volume[..., :-1], cf=cf, ct=ct, valid=midpoint,
                  iota=iota[:, :, None, None])
    bounds = np.stack([np.nanmin(z[domain][:,:currents], axis=(1,2,3)),
                       np.nanmax(z[domain][:,:currents], axis=(1,2,3))], axis=1)
    return dict(feat=feat, fields=fields, scale=ct_scale, scales=scale.cpu().tolist(),
                truth=torch.as_tensor(z['theta'], dtype=torch.float32, device=device), splits=splits,
                ids=z['ids'], bounds=bounds)


def residual(theta, p, scale):
    basis = p['basis']
    d01 = (basis * theta[:, None,None,None,:4]).sum(-1)
    t01 = (basis * theta[:, None,None,None,4:]).sum(-1)
    salt = -0.1*(math.log(20)*d01).exp()*p['a_cx'] + (-0.25+1.25*t01)*p['iota']
    wall = p['iota'].expand(*salt.shape[:-1],1)
    vel = (-p['volume']*torch.cat([wall,salt],-1).diff(dim=-1)).cumsum(-1)
    flux = torch.cat([wall,salt+p['cf']*vel,wall],-1)
    div = 20*flux.diff(dim=-1)
    r = (p['ct'] + 0.5*(div[:,:,1:]+div[:,:,:-1]))/scale
    return (r.square()*p['valid'][...,None]).sum((1,2,3))/(20*p['valid'].sum((1,2)))


def fit(model, data, folder, epochs, seed):
    tr, va = data['splits']['train'], data['splits']['val']
    feat, truth = data['feat'], data['truth']
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    batches = math.ceil(len(tr)/BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=batches*epochs)
    rng = torch.Generator().manual_seed(seed + 1)
    best = np.inf
    hist = []
    best_state = None
    sync()
    start = perf_counter()
    for epoch in range(epochs):
        order = tr[torch.randperm(len(tr), generator=rng).to(tr.device)]
        for batch in order.split(BATCH):
            opt.zero_grad(set_to_none=True)
            loss = (model(feat[batch]).sigmoid()-truth[batch]).square().mean()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            val = float((model(feat[va]).sigmoid()-truth[va]).square().mean())
            hist.append(val)
            if val < best:
                best = val
                best_state = copy.deepcopy(model.state_dict())
        if (epoch+1)%25 == 0:
            print(folder.name, epoch+1, round(val, 6), flush=True)
    sync()
    elapsed = perf_counter()-start
    if best_state is None:
        raise ValueError('No finite validation loss')
    torch.save(best_state, folder/'best.pt')
    info = dict(train_s=elapsed, best_val_coef_mse=best, epoch_val_mse=hist,
                epochs=epochs, seed=seed, input_rms=data['scales'], n_train=len(tr), n_val=len(va),
                loss='MSE of eight normalized coefficients', params=n_params(model), batch_size=BATCH)
    save_json(folder/'training.json', info)
    return info


def independent(model, state, features):
    # One weight matrix/tensor per material. The batched spectral contraction
    # avoids transforming FFT gradients inside vmap on older CUDA releases.
    if isinstance(model, MLP):
        return vmap(lambda p,x: functional_call(model, p, (x[None],))[0])(state, features)

    def conv(x, name):
        w = state[name+'.weight'][...,0,0]
        return torch.bmm(w, x.flatten(2)).reshape(x.shape[0], w.shape[1], *x.shape[-2:]) + state[name+'.bias'][:,:,None,None]

    def linear(x, name):
        return torch.bmm(state[name+'.weight'], x[...,None]).squeeze(-1)+state[name+'.bias']

    x = F.pad(conv(features, 'lift'), (0,4,0,3))
    for j in range(4):
        x = spectra(x, state[f'spec.{j}.w1'], state[f'spec.{j}.w2'], True) + conv(x, f'pw.{j}')
        if j < 3:
            x = F.gelu(x)
    x = F.gelu(conv(x[:,:,:13,:20], 'proj'))
    x = masked_pool(x, model.mask(features))
    return linear(F.gelu(linear(x, 'head.0')), 'head.2')


def adapt(model, data, idx, rate, steps):
    p = {k:v[idx] for k,v in data['fields'].items()}
    x = data['feat'][idx]
    weights = {k:w.detach()[None].repeat((len(x),)+(1,)*w.ndim).requires_grad_() for k,w in model.named_parameters()}
    opt = torch.optim.AdamW(weights.values(), lr=rate, foreach=True)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        estimate = independent(model, weights, x).sigmoid()
        residual(estimate, p, data['scale']).sum().backward()
        opt.step()
    with torch.no_grad():
        return independent(model, weights, x).sigmoid().cpu().numpy()


def score(pred, data, idx):
    bounds = data['bounds'][idx.cpu().numpy()].clip(0,2)/2
    u = bounds[:,:1] + np.diff(bounds)*np.linspace(0,1,201)
    basis = np.stack([(1-u)**3, 3*u*(1-u)**2, 3*u**2*(1-u), u**3], axis=-1)
    target = data['truth'][idx].cpu().numpy()
    err = (basis*(pred-target)[:,None,:4]).sum(-1)*math.log(20)
    return float(np.median((err**2).mean(1)))


def save_pred(path, data, idx, predictions):
    if predictions.shape != (len(idx),8) or not np.isfinite(predictions).all():
        raise ValueError('Incomplete predictions')
    sel = idx.cpu().numpy()
    bounds = data['bounds'][sel]
    tmp = path.with_suffix('.partial')
    with tmp.open('wb') as stream:
        np.savez(stream, ids=data['ids'][sel], theta=predictions, lo=bounds[:,0], hi=bounds[:,1])
    os.replace(tmp, path)


def run(z, domain, currents, arch, args):
    prefix = f'{domain}__{currents}__{arch}'
    folder = CHECKPOINTS / prefix
    folder.mkdir(parents=True, exist_ok=True)
    model = (FNO if arch=='fno' else MLP)(currents)
    model = model.to(args.device)
    data = prepare(z, domain, currents, args.device)
    idx = data['splits']['test']
    if (folder/'training.json').exists():
        info = json.loads((folder/'training.json').read_text())
    else:
        if args.stage == 'adapt':
            raise FileNotFoundError(f'Pretrain {prefix} and seed {args.seed} first')
        info = fit(model, data, folder, EPOCHS, args.seed)
    model.load_state_dict(torch.load(folder/'best.pt', map_location=args.device, weights_only=True))
    raw = RESULTS / prefix
    if not raw.with_suffix('.json').exists():
        with torch.no_grad():
            # Warm up then measure a synchronized batched pass. Average repeated
            # passes to improve timer resolution for the MLP.
            x = data['feat'][idx]
            model(x[:32])
            sync()
            start = perf_counter()
            for _ in range(10):
                pred = torch.cat([model(b).sigmoid() for b in x.split(128)])
            sync()
            raw_s = (perf_counter()-start)/10
        save_pred(raw.with_suffix('.npz'), data, idx, pred.cpu().numpy())
        details = dict(train_s=info['train_s'], test_s=raw_s, test_batch=128, test_repeats=10,
                       seed=args.seed, n_train=info['n_train'], n_val=info['n_val'], n_test=len(idx),
                       epochs=info['epochs'], parameters=info['params'], device=torch.cuda.get_device_name(0) if args.device=='cuda' else 'CPU',
                       description='Supervised coefficient regression using binned concentration, reconstructed velocity, per-current time, mask and current. Best validation epoch.')
        save_json(raw.with_suffix('.json'), details)
    if args.stage=='train':
        return
    output = RESULTS / (prefix+'_ttt')
    if output.with_suffix('.json').exists():
        return
    va = data['splits']['val'][:100]
    grid_file = HERE/'learned_checkpoints'/prefix/'rate.json'
    if grid_file.exists():
        rates = json.loads(grid_file.read_text())
    else:
        if args.seed != 0:
            raise ValueError('Run seed zero to select the TTT rate before fitting the other seeds.')
        sync()
        start = perf_counter()
        grid = {}
        for rate in RATES:
            predictions = np.concatenate([adapt(model, data, b, rate, STEPS) for b in va.split(args.chunk)])
            grid[rate] = score(predictions, data, va)
            print(prefix, 'validation', rate, grid[rate], flush=True)
        rates = dict(lr=min(grid, key=grid.get), score=grid, n_val=len(va), tune_s=perf_counter()-start)
        save_json(grid_file, rates)
    sync()
    start = perf_counter()
    predictions = []
    for j,b in enumerate(idx.split(args.chunk)):
        predictions.append(adapt(model, data, b, rates['lr'], STEPS))
        if j==0 and arch=='fno':
            save_pred(RESULTS/(prefix+'_ttt_first8.npz'), data, idx[:8], predictions[0][:8])
    sync()
    ttt_s = perf_counter()-start
    pred = np.concatenate(predictions)
    save_pred(output.with_suffix('.npz'), data, idx, pred)
    base = json.loads(raw.with_suffix('.json').read_text())
    details = dict(**base)
    details.update(test_s=ttt_s, test_batch=args.chunk, test_repeats=1, ttt_steps=STEPS, ttt_lr=rates['lr'],
                   tune_s=rates['tune_s'] if args.seed == 0 else 0.0, shared_train=True, ttt_s=ttt_s,
                   description='Same supervised checkpoint; 150 AdamW updates to an independent network copy per material, minimizing the concentration conservation residual at its observed times. Learning rate chosen on 100 validation materials.')
    save_json(output.with_suffix('.json'), details)
    print(prefix, 'TTT test completed', round(ttt_s,2), flush=True)


def main():
    global CHECKPOINTS, RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', nargs='+', choices=['clean','gaussian_noisy'], default=['clean','gaussian_noisy'])
    parser.add_argument('--currents', nargs='+', type=int, choices=[1,3], default=[1,3])
    parser.add_argument('--arch', nargs='+', choices=['mlp','fno'], default=['mlp','fno'])
    parser.add_argument('--stage', choices=['train','adapt','all'], default='all')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--chunk', default=64, type=int)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    if args.seed != 0:
        seed_root = HERE/'learned_seeds'/f'seed_{args.seed}'
        CHECKPOINTS = seed_root/'checkpoints'
        RESULTS = HERE/'results_seeds'/str(args.seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with np.load(HERE/'cache/binned.npz', allow_pickle=False) as z:
        for domain in args.domain:
            for k in args.currents:
                for arch in args.arch:
                    torch.manual_seed(args.seed)
                    run(z, domain, k, arch, args)


if __name__=='__main__':
    main()
