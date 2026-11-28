"""Supervised retraining on resolution-specific space-time observation grids."""
import argparse
import json
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from learn import (BATCH, EPOCHS, FNO, HERE, fit, n_params, prepare, reconstruct_observations,
                   save_json, save_pred, sync)

FRAMES = {40:26, 80:52, 160:104, 290:151}
COARSENING_PASSES = {40:0, 80:1, 160:2, 290:2}


class ConvMLP(nn.Module):
    def __init__(self, currents, n_space=40):
        super().__init__()
        self.currents = currents
        self.n_space = n_space
        self.conv = nn.Conv2d(4, 3, kernel_size=3, stride=(2, 3), padding=1)
        self.net = nn.Sequential(nn.Linear(currents * (3 * 13 * 14 + 1), 40), nn.GELU(),
                                 nn.Linear(40, 40), nn.GELU(), nn.Linear(40, 8))

    def forward(self, features):
        batch, _, n_time, n_space = features.shape
        k = self.currents
        # Stack each current separately in the batch to share the convolution.
        fields = features[:, :4*k].reshape(batch, 4, k, n_time, n_space).transpose(1, 2)
        fields = fields.reshape(batch*k, 4, n_time, n_space)
        fields = fields[:, [0, 1, 3, 2]]
        valid = fields[:, 3:4, :, 0]
        for _ in range(COARSENING_PASSES[self.n_space]):
            # Reuse the same kernel and bias at both resolutions. Transport the
            # validity mask alongside the three first-stage learned channels.
            first = F.gelu(F.conv2d(fields, self.conv.weight, self.conv.bias, stride=2, padding=1))
            valid = F.max_pool1d(valid, kernel_size=3, stride=2, padding=1)
            mask = valid.unsqueeze(-1).expand(-1, 1, -1, first.shape[-1])
            fields = torch.cat((first * mask, mask), dim=1)
        if self.n_space == 290:
            # 38 x 73 -> 26 x 40. Average the learned channels over only
            # valid positions, carrying any valid support to the next pass.
            pooled = F.adaptive_avg_pool2d(fields, (26, 40))
            coverage = pooled[:, 3:4]
            mask = (coverage > 0).to(fields.dtype)
            means = pooled[:, :3] / torch.where(coverage > 0, coverage, 1.0)
            fields = torch.cat((means * mask, mask), dim=1)
            valid = mask[:, :, :, 0]
        encoded = F.gelu(self.conv(fields))
        # Padding and bias can generate activations in unobserved time windows.
        valid = F.max_pool1d(valid, kernel_size=3, stride=2, padding=1)
        encoded = (encoded * valid.unsqueeze(-1)).reshape(batch, k, -1)
        current = features[:, 4*k:5*k, 0, 0].unsqueeze(-1)
        return self.net(torch.cat((encoded, current), dim=2).flatten(1))


def prepare_supervised(observations, common, currents, device, chunk=32):
    """Assemble features in GPU chunks, computing only the fields used in supervised fits."""
    n_materials, _, n_time, n_space = observations.shape
    k = currents
    feat = torch.empty(n_materials, 5*k+1, n_time, n_space, device=device)
    scale_sum = torch.zeros(2, k, dtype=torch.float64, device=device)
    scale_count = torch.zeros(k, dtype=torch.int64, device=device)
    bounds = np.empty((n_materials, 2), dtype=observations.dtype)
    x = torch.linspace(0, 1, n_space, device=device)
    for start in range(0, n_materials, chunk):
        rows = slice(start, start+chunk)
        raw = observations[rows, :k]
        bounds[rows,0] = np.nanmin(raw, axis=(1,2,3))
        bounds[rows,1] = np.nanmax(raw, axis=(1,2,3))
        valid = torch.as_tensor(common['mask'][rows,:k], dtype=torch.bool, device=device)
        times = torch.as_tensor(common['tau'][rows,:k], dtype=torch.float32, device=device)
        iota = torch.as_tensor(common['iota'][rows,:k], dtype=torch.float32, device=device)
        c, velocity, times, *_ = reconstruct_observations(
            torch.tensor(raw, dtype=torch.float32, device=device), valid, times, iota)
        feat[rows,:k] = (c - 1) * valid[..., None]
        feat[rows,k:2*k] = velocity * valid[..., None]
        feat[rows,2*k:3*k] = valid[..., None]
        feat[rows,3*k:4*k] = times[..., None] / (36000/33640)
        feat[rows,4*k:5*k] = iota[:,:,None,None] / 5
        feat[rows,-1] = x
        train = torch.as_tensor(common['split'][rows] == 'train', device=device)
        squared = feat[rows, :2*k][train].square().sum((0,2,3)).reshape(2,k)
        scale_sum += squared
        scale_count += valid[train].sum((0,2)) * n_space
        if not torch.isfinite(feat[rows]).all():
            raise ValueError(f'Nonfinite features at material slice {rows}')
    scale = (scale_sum / scale_count).sqrt().float().clamp_min(1e-8)
    for start in range(0, n_materials, chunk):
        feat[start:start+chunk, :2*k].div_(scale.reshape(1,2*k,1,1))
    splits = {key: torch.as_tensor(np.flatnonzero(common['split'] == key), device=device)
              for key in ('train','val','test')}
    return dict(feat=feat, scales=scale.cpu().tolist(), splits=splits, ids=common['ids'], bounds=bounds,
                truth=torch.as_tensor(common['theta'], dtype=torch.float32, device=device))


def run(data, domain, currents, architecture, seed, device, n_space, n_time):
    stem = f'{domain}__{currents}__{architecture}'
    folder = HERE / f'grid{n_space}_results' / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    result = folder / stem
    checkpoint = HERE / 'cache' / f'grid{n_space}_checkpoints' / f'seed{seed}' / stem
    checkpoint.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = (FNO(currents) if architecture == 'fno' else ConvMLP(currents, n_space)).to(device)
    expected = ({1:65741, 3:65851} if architecture == 'fno' else {1:23999, 3:67759})[currents]
    if n_params(model) != expected:
        raise ValueError(f'{architecture} has {n_params(model)} real parameters, expected {expected}')
    idx = data['splits']['test']
    if result.with_suffix('.json').is_file() and result.with_suffix('.npz').is_file():
        existing = json.loads(result.with_suffix('.json').read_text())
        with np.load(result.with_suffix('.npz')) as preds:
            if (existing['parameters'] == expected and existing['epochs'] == EPOCHS
                    and preds['theta'].shape == (len(idx), 8)
                    and np.isfinite(preds['theta']).all()
                    and np.array_equal(preds['ids'], data['ids'][idx.cpu().numpy()])):
                return
        raise ValueError(f'Incomplete saved result for {result}')
    if (checkpoint / 'training.json').is_file() and (checkpoint / 'best.pt').is_file():
        info = json.loads((checkpoint / 'training.json').read_text())
    else:
        print('Training', n_space, seed, stem, flush=True)
        info = fit(model, data, checkpoint, EPOCHS, seed)
    model.load_state_dict(torch.load(checkpoint / 'best.pt', weights_only=True, map_location=device))
    model.eval()
    with torch.no_grad():
        x = data['feat'][idx]
        model(x[:32])
        sync()
        start = perf_counter()
        for _ in range(10):
            pred = torch.cat([model(b).sigmoid() for b in x.split(128)])
        sync()
        elapsed = (perf_counter() - start) / 10
    save_pred(result.with_suffix('.npz'), data, idx, pred.cpu().numpy())
    details = dict(train_s=info['train_s'], test_s=elapsed, test_repeats=10, test_batch=128,
                   architecture=architecture, currents=currents, domain=domain, seed=seed,
                   n_train=info['n_train'], n_val=info['n_val'], n_test=len(idx),
                   parameters=expected, params=expected, epochs=EPOCHS, batch_size=BATCH,
                   best_epoch=int(np.argmin(info['epoch_val_mse'])) + 1,
                   best_val_coef_mse=info['best_val_coef_mse'], loss=info['loss'],
                   input_rms=info['input_rms'], optimizer='AdamW', learning_rate=0.001,
                   weight_decay=0.01, scheduler='CosineAnnealingLR, per optimizer step',
                   n_space=n_space, n_time=n_time,
                   device=torch.cuda.get_device_name(device) if device.startswith('cuda') else 'CPU')
    if architecture == 'conv_mlp':
        details.update(conv_kernel=[3,3], conv_out_channels=3, conv_shared_across_currents=True,
                       conv_strides_time_space=[[2,2]]*COARSENING_PASSES[n_space]+[[2,3]],
                       conv_shared_across_layers=(n_space > 40))
        if n_space == 290:
            details['adaptive_pool'] = 'Mask-weighted adaptive average from 38 x 73 to 26 x 40 after pass two'
    save_json(result.with_suffix('.json'), details)
    print('Completed', seed, stem, f"{info['train_s']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', nargs='+', choices=['clean', 'gaussian_noisy'],
                        default=['clean', 'gaussian_noisy'])
    parser.add_argument('--currents', nargs='+', type=int, choices=[1, 3], default=[1, 3])
    parser.add_argument('--arch', nargs='+', choices=['fno', 'conv_mlp'], default=['conv_mlp', 'fno'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--grid', type=int, choices=sorted(FRAMES), default=40)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    n_space = args.grid
    n_time = FRAMES[n_space]
    archive_path = HERE / 'cache' / f'binned_{n_space}x{n_time}.npz'
    with np.load(archive_path, allow_pickle=False) as archive:
        common = {key: archive[key] for key in ('ids','split','theta','iota','tau','mask')}
        for domain in args.domain:
            if n_space >= 160:
                from grid40_prepare import mmap_grid_field
                observations = mmap_grid_field(archive_path, domain)
            else:
                observations = {**common, domain: archive[domain]}
            for currents in args.currents:
                data = (prepare_supervised(observations, common, currents, args.device) if n_space >= 160
                        else prepare(observations, domain, currents, args.device))
                if data['feat'].shape[-2:] != (n_time,n_space):
                    raise ValueError(f'Expected {n_time} times and {n_space} spatial bins per current')
                for seed in args.seeds:
                    for arch in args.arch:
                        run(data, domain, currents, arch, seed, args.device, n_space, n_time)
                del data
            del observations


if __name__ == '__main__':
    main()
