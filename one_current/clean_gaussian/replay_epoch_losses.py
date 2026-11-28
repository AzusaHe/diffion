"""Collect end-of-epoch losses and test replay fidelity against the archived fit."""
import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from grid40_learn import FRAMES, ConvMLP, prepare_supervised
from grid40_prepare import mmap_grid_field
from learn import EPOCHS, FNO, HERE, MLP, fit, prepare, save_json


def original_files(grid, seed, stem):
    if grid == 20:
        checkpoint = (HERE / 'learned_checkpoints' if seed == 0
                      else HERE / 'learned_seeds' / f'seed_{seed}' / 'checkpoints')
        result = HERE / 'results' if seed == 0 else HERE / 'results_seeds' / str(seed)
    else:
        checkpoint = HERE / 'cache' / f'grid{grid}_checkpoints' / f'seed{seed}'
        result = HERE / f'grid{grid}_results' / f'seed{seed}'
    return checkpoint / stem, result / (stem + '.npz')


def original_reference(folder, package):
    if (folder / 'training.json').is_file() and (folder / 'best.pt').is_file():
        return json.loads((folder / 'training.json').read_text()), folder / 'training.json', folder / 'best.pt'
    manifest_path = package / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    inputs = manifest['inputs']
    info = dict(input_rms=[inputs['concentration_deviation_rms'], inputs['velocity_rms']],
                epoch_val_mse=manifest['training']['original_epoch_validation_mse'])
    return info, manifest_path, package / manifest['checkpoint']['file']


def collect(grid, frames, domain, k, arch, seed, data, output_root):
    stem = f'{domain}__{k}__{arch}'
    original, prediction_file = original_files(grid, seed, stem)
    relative = Path(f'grid_{grid}x{frames}') / domain / f'{k}_current' / f'seed_{seed}' / arch
    package = HERE / 'trained_models' / relative
    output = output_root / relative
    output.mkdir(parents=True, exist_ok=True)
    provenance = output / 'loss_provenance.json'
    csv_path = output / 'epochs.csv'
    if (provenance.exists() and csv_path.exists()
            and 'original_best_train_mse' in json.loads(provenance.read_text())):
        with csv_path.open() as stream:
            records = list(csv.DictReader(stream))
        if len(records) == EPOCHS and all(int(row['epoch']) == i for i, row in enumerate(records, 1)):
            return
        raise ValueError(f'Invalid epoch CSV: {csv_path}')
    original_info, reference_file, checkpoint_file = original_reference(original, package)
    original_vals = np.asarray(original_info['epoch_val_mse'])
    if len(original_vals) != EPOCHS or not np.array_equal(original_info['input_rms'], data['scales']):
        raise ValueError(f'Inconsistent archived training normalization or schedule: {original}')
    torch.manual_seed(seed)
    device = data['feat'].device
    model = (FNO(k) if arch == 'fno' else MLP(k) if arch == 'mlp' else ConvMLP(k,grid)).to(device)
    states = []
    def snapshot(current):
        states.append({name:value.detach().cpu().clone() for name,value in current.state_dict().items()})

    with tempfile.TemporaryDirectory(dir=HERE / 'cache', prefix='loss_replay_') as tmp:
        print('Replay', grid, stem, seed, flush=True)
        info = fit(model, data, Path(tmp), EPOCHS, seed, on_epoch_end=snapshot)
        vals = np.array(info['epoch_val_mse'])
        best = int(np.argmin(vals))
        original_state = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
        best_state = states[best]
        if set(original_state) != set(best_state):
            raise ValueError('Inconsistent checkpoint parameter names')
        weights_equal = all(torch.equal(x, best_state[name]) for name,x in original_state.items())
        max_weight_change = max(float((x-best_state[name]).abs().max()) for name,x in original_state.items())
        if not weights_equal:
            chk = output / 'replay_best.pt'
            shutil.copyfile(Path(tmp) / 'best.pt', chk.with_suffix('.tmp'))
            chk.with_suffix('.tmp').replace(chk)

    model.eval()
    train, test = data['splits']['train'], data['splits']['test']
    # Pre-select evaluation batches once. Inference uses raw normalized features
    # and no augmentation. These models have no batch-dependent running state.
    chunks = {name:[data['feat'][b] for b in idx.split(128)] for name,idx in [('train',train),('test',test)]}
    labels = {name:data['truth'][idx] for name,idx in [('train',train),('test',test)]}
    rows = []
    with torch.no_grad():
        model.load_state_dict(original_state)
        original_losses = {}
        for name,batches in chunks.items():
            pred = torch.cat([model(x).sigmoid() for x in batches])
            original_losses[name] = float((pred - labels[name]).square().mean())
            if name == 'test':
                published = pred.cpu().numpy()
        validation = data['splits']['val']
        original_losses['validation'] = float(
            (model(data['feat'][validation]).sigmoid() - data['truth'][validation]).square().mean())
        if prediction_file.is_file():
            with np.load(prediction_file) as ref:
                np.testing.assert_array_equal(ref['ids'], data['ids'][test.cpu().numpy()])
                original_prediction_check = float(np.max(np.abs(published - ref['theta'])))
                published = ref['theta']
        else:
            original_prediction_check = None
        for i,state in enumerate(states):
            model.load_state_dict(state)
            losses = {}
            for name,batches in chunks.items():
                pred = torch.cat([model(x).sigmoid() for x in batches])
                losses[name] = float((pred - labels[name]).square().mean())
                if name == 'test' and i == best:
                    replay_pred = pred.cpu().numpy()
            rows.append((i+1,losses['train'],vals[i],losses['test'],original_vals[i]))
    rows = np.asarray(rows)
    if rows.shape != (EPOCHS,5) or not np.isfinite(rows).all():
        raise ValueError(f'Invalid epoch measurements: {stem}')
    val_equal = np.array_equal(original_vals, vals)
    pred_equal = np.array_equal(replay_pred, published)
    exact = bool(weights_equal and val_equal and pred_equal)
    status = 'exact_archived_checks' if exact else 'independent_replay'
    tmp_csv = csv_path.with_suffix('.tmp')
    with tmp_csv.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch','train_mse','validation_mse','test_mse','original_validation_mse'])
        for epoch, tr, va, te, old_va in rows:
            writer.writerow([int(epoch),tr,va,te,old_va])
    tmp_csv.replace(csv_path)
    replay_checkpoint = 'replay_best.pt'
    if weights_equal:
        reference = package / 'best.pt'
        if not reference.exists():
            reference = checkpoint_file
        try:
            replay_checkpoint = Path(os.path.relpath(reference, output)).as_posix()
        except ValueError:  # Windows volumes do not share a relative path.
            replay_checkpoint = reference.resolve().as_posix()
    save_json(provenance, dict(
        status=status, grid=[grid,frames], domain=domain, currents=k, seed=seed, architecture=arch,
        metric='Mean squared error across all materials and eight normalized Bernstein coefficients, after sigmoid.',
        train_mse='Fixed end-of-epoch weights scored on all 1402 training materials.',
        validation_mse='Replay validation loss computed by the original trainer after each epoch, 156 validation materials.',
        test_mse='Post-training replay diagnostic scored on all 389 held-out materials at each saved epoch.',
        original_validation_mse='Original per-epoch validation losses from the archived training.json.',
        checkpoint_selection='Minimum validation MSE. Test data were first scored after training finished.',
        provenance='Train/test epoch histories were absent from the original artifacts. Independently replayed the original initialization, shuffle and AdamW schedule, then evaluated saved epoch states.',
        epochs=EPOCHS, optimizer_updates=EPOCHS*math.ceil(len(train)/64), effective_batch_size=64,
        train_eval_batch_size=128, test_eval_batch_size=128, replay_best_epoch=best+1,
        original_best_epoch=int(np.argmin(original_vals))+1,
        original_best_train_mse=original_losses['train'],
        original_best_validation_mse=original_losses['validation'],
        original_best_test_mse=original_losses['test'],
        original_best_archived_validation_mse=float(original_vals.min()),
        validation_trace_exact=bool(val_equal), best_weights_exact=bool(weights_equal),
        best_test_predictions_exact=bool(pred_equal),
        max_abs_validation_difference=float(np.max(np.abs(vals-original_vals))),
        max_abs_best_weight_difference=max_weight_change,
        max_abs_best_prediction_difference=float(np.max(np.abs(replay_pred-published))),
        replay_best_checkpoint=replay_checkpoint,
        original_training_file=reference_file.relative_to(HERE).as_posix(),
        original_predictions=(prediction_file.relative_to(HERE).as_posix() if prediction_file.is_file()
                              else 'Recomputed from original packaged checkpoint'),
        max_abs_original_checkpoint_vs_published_predictions=original_prediction_check,
        device=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        torch_version=str(torch.__version__),
    ))
    print('Published', grid, stem, seed, status, flush=True)


def main():
    grids = {20:13, **FRAMES}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=int, nargs='+', choices=sorted(grids), default=list(grids))
    parser.add_argument('--output', type=Path, default=HERE / 'trained_models',
                        help='Destination for recorded replay histories; packaged original references stay in trained_models.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for grid in args.grid:
        nt = grids[grid]
        path = HERE / 'cache' / ('binned.npz' if grid == 20 else f'binned_{grid}x{nt}.npz')
        with np.load(path) as archive:
            common = {key:archive[key] for key in ('ids','split','theta','iota','tau','mask')}
            for domain in ('clean','gaussian_noisy'):
                observations = mmap_grid_field(path,domain) if grid >= 160 else {**common,domain:archive[domain]}
                for k in (1,3):
                    data = (prepare_supervised(observations,common,k,'cuda') if grid >= 160
                            else prepare(observations,domain,k,'cuda'))
                    cpu_data = prepare(observations,domain,k,'cpu') if grid == 20 else None
                    for seed in range(3):
                        for arch in (('mlp' if grid == 20 else 'conv_mlp'),'fno'):
                            collect(grid,nt,domain,k,arch,seed,cpu_data if arch == 'mlp' else data,
                                    args.output.resolve())
                    del data, cpu_data
                del observations


if __name__ == '__main__':
    main()
