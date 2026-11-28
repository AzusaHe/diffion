# ---- TTT of Junbo's clean 290x151 one-current FNO (same code path as learn.py adapt()) ----
import sys, json, math, time, numpy as np, torch
from pathlib import Path

def setup(repo):
    cg = Path(repo)/'one_current'/'clean_gaussian'
    for p in (str(repo), str(cg), str(cg/'trained_models')): 
        if p not in sys.path: sys.path.insert(0, p)
    import learn, inference
    from pinn.material import alpha_V, bernstein_basis
    return cg, learn, inference, alpha_V, bernstein_basis

def load_fno(cg, learn, seed, device):
    leaf = cg/'trained_models'/'grid_290x151'/'clean'/'1_current'/f'seed_{seed}'/'fno'
    meta = json.loads((leaf/'manifest.json').read_text())
    model = learn.FNO(1).to(device)
    model.load_state_dict(torch.load(leaf/'best.pt', map_location=device, weights_only=True))
    rms = torch.tensor([meta['inputs']['concentration_deviation_rms'], meta['inputs']['velocity_rms']], dtype=torch.float32)
    return model.eval(), rms

def build(z, rms, learn, inference, alpha_V, bernstein_basis, device, ct_scale=None):
    """Features exactly as in training (saved per-current RMS) + the PDE-residual fields of learn.prepare()."""
    c = torch.as_tensor(z['c'], dtype=torch.float32, device=device)
    valid = torch.as_tensor(z['mask'], dtype=torch.bool, device=device)
    tau = torch.as_tensor(z['tau'], dtype=torch.float32, device=device)
    iota = torch.as_tensor(z['iota'], dtype=torch.float32, device=device)
    feat = inference.features_from_observations(c, tau, iota, valid, rms.to(device))
    cc, velocity, times, midpoint, ct, cf, volume = learn.reconstruct_observations(c, valid, tau, iota)
    a, _ = alpha_V(cf)
    n_space = c.shape[-1]
    fields = dict(basis=bernstein_basis((cf/2).clamp(0, 1)), a_cx=a*cc.diff(dim=-1)*n_space,
                  volume=volume[..., :-1], cf=cf, ct=ct, valid=midpoint, iota=iota[:, :, None, None])
    if ct_scale is None:
        ct_scale = float((ct.square().sum()/midpoint.sum()/n_space).sqrt())
    bounds = np.stack([np.nanmin(z['c'], axis=(1, 2, 3)), np.nanmax(z['c'], axis=(1, 2, 3))], 1)
    return dict(feat=feat, fields=fields, scale=ct_scale, bounds=bounds,
                truth=torch.as_tensor(z['theta'], dtype=torch.float32, device=device), ids=z['ids'])

def adapt(model, data, idx, rate, steps, learn):
    """= learn.adapt: independent weight copy per material, AdamW on the concentration-conservation residual."""
    p = {k: v[idx] for k, v in data['fields'].items()}
    x = data['feat'][idx]
    w = {k: v.detach()[None].repeat((len(x),)+(1,)*v.ndim).requires_grad_() for k, v in model.named_parameters()}
    opt = torch.optim.AdamW(w.values(), lr=rate, foreach=True)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        learn.residual(learn.independent(model, w, x).sigmoid(), p, data['scale']).sum().backward()
        opt.step()
    with torch.no_grad():
        return learn.independent(model, w, x).sigmoid().cpu().numpy()

def errors(est, truth, lo, hi, n=301):
    u = np.linspace(np.maximum(lo, 0)/2, np.minimum(hi, 2)/2, n, axis=1)
    b = np.stack(((1-u)**3, 3*u*(1-u)**2, 3*u**2*(1-u), u**3), axis=2)
    cur = lambda th: (np.exp(np.log(20)*np.einsum('ntk,nk->nt', b, th[:, :4])), -.25+1.25*np.einsum('ntk,nk->nt', b, th[:, 4:]))
    (d, t), (dt, tt) = cur(est), cur(truth)
    return 100*np.linalg.norm(d-dt, axis=1)/np.linalg.norm(dt, axis=1), np.sqrt(np.mean((t-tt)**2, axis=1))

def junbo_score(pred, truth, bounds):
    """= learn.score: median over materials of the mean squared ln-D error (201 points in the observed c range). Used to pick the TTT rate."""
    b = bounds.clip(0, 2)/2
    u = b[:, :1] + np.diff(b)*np.linspace(0, 1, 201)
    basis = np.stack([(1-u)**3, 3*u*(1-u)**2, 3*u**2*(1-u), u**3], axis=-1)
    err = (basis*(pred-truth)[:, None, :4]).sum(-1)*math.log(20)
    return float(np.median((err**2).mean(1)))
