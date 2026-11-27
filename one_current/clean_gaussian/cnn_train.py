"""Fit the native set CNN on a fixed one- or three-current input per material."""
import argparse
import json
import os
import time

import numpy as np
import torch

from cnn_prepare import DOMAINS, HERE, cnn
import train_three_current_cnn as training


class NativeObservations:
    def __init__(self, domain, count, device):
        cache = HERE / "cache"
        with np.load(cache / "index.npz") as index, np.load(cache / f"cnn_{domain}.npz") as meta:
            np.testing.assert_array_equal(index["ids"], meta["ids"])
            np.testing.assert_array_equal(index["chosen"], meta["chosen"])
            self.ids = index["ids"]
            self.train = np.flatnonzero(index["split"] == "train")
            self.test = np.flatnonzero(index["split"] == "test")
            self.lo = meta["lo"][:, :count].min(axis=1)
            self.hi = meta["hi"][:, :count].max(axis=1)
            self.horizon = float(meta["horizon"])
            self.targets = torch.as_tensor(index["theta"], dtype=torch.float32, device=device)
            self.x = torch.as_tensor(meta["x"].astype(np.float32), device=device)
            self.t = torch.as_tensor(meta["time"][:, :count] / self.horizon, device=device)
            self.mask = torch.as_tensor(meta["mask"][:, :count], device=device)
            self.current = torch.as_tensor(index["iota"][:, :count, None] / 5., dtype=torch.float32, device=device)
        stored = np.load(cache / f"cnn_{domain}_fields.npy", mmap_mode="r")
        self.c = torch.as_tensor(np.array(stored[:, :count], copy=True), device=device)
        self.device = device

    def batch(self, rows, dtype):
        c = self.c[rows]
        m = self.mask[rows, :, :, None]
        channels = torch.stack(((c-1) * m, (self.t[rows, :, :, None] * m).expand_as(c),
                                self.x * m, m.expand_as(c)), dim=2)
        # Place channels last in each individual CNN profile.
        shape = channels.shape
        channels = channels.flatten(0, 1).to(dtype=dtype, memory_format=torch.channels_last).reshape(shape)
        return channels, self.current[rows], self.targets[rows]


def train(domain, count, epochs, micro, device_name, resume):
    cnn.N_EXPERIMENTS = count
    training.seed_everything(242)
    torch.set_num_threads(1)
    device = torch.device(device_name)
    gpu = device.type == "cuda"
    amp = gpu
    dtype = torch.float16 if amp else torch.float32
    start_all = time.perf_counter()
    data = NativeObservations(domain, count, device)
    model = cnn.ThreeCurrentSetCNN().to(device, memory_format=torch.channels_last)
    criterion = cnn.CoefficientCurveLoss(curve_points=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=1024, growth_interval=1000)
    permutation_rng = torch.Generator().manual_seed(343)
    work = HERE / "cache" / f"cnn_{domain}_{count}"
    work.mkdir(parents=True, exist_ok=True)
    check = work / "progress.pt"
    completed = 0
    elapsed = 0.
    if resume and check.is_file():
        prev = torch.load(check, map_location=device, weights_only=False)
        if prev["epochs"] != epochs:
            raise ValueError("Resume requires the same number of planned training epochs")
        model.load_state_dict(prev["weights"])
        optimizer.load_state_dict(prev["optimizer"])
        scheduler.load_state_dict(prev["scheduler"])
        scaler.load_state_dict(prev["scaler"])
        permutation_rng.set_state(prev["permutation"].cpu())
        torch.set_rng_state(prev["rng"].cpu())
        if gpu:
            torch.cuda.set_rng_state(prev["device_rng"].cpu(), device)
        completed, elapsed = prev["completed"], prev["elapsed"]
    if gpu: torch.cuda.synchronize()
    setup_s = time.perf_counter() - start_all
    learn_start = time.perf_counter()
    model.train()
    for epoch in range(completed+1, epochs+1):
        rows = data.train[torch.randperm(len(data.train), generator=permutation_rng).numpy()]
        losses = []
        for start in range(0, len(rows), 64):
            batch = rows[start:start+64]
            optimizer.zero_grad(set_to_none=True)
            for j in range(0, len(batch), micro):
                chunk = batch[j:j+micro]
                profile, current, target = data.batch(chunk, dtype)
                with torch.autocast(device.type, torch.float16, enabled=amp):
                    outputs = model(profile, current)
                    loss, _ = criterion(outputs, target)
                scaler.scale(loss * (len(chunk)/len(batch))).backward()
                losses.append(loss.detach())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        avg = float(torch.stack(losses).mean().item())
        if not np.isfinite(avg):
            raise FloatingPointError(f"Nonfinite training loss in epoch {epoch}")
        latest_s = elapsed + time.perf_counter() - learn_start
        if epoch % 5 == 0 or epoch == epochs:
            state = dict(weights=model.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                         permutation=permutation_rng.get_state(), completed=epoch, epochs=epochs,
                         rng=torch.get_rng_state(), device_rng=torch.cuda.get_rng_state(device) if gpu else None,
                         elapsed=latest_s)
            torch.save(state, work / "progress.tmp")
            os.replace(work / "progress.tmp", check)
        print(domain, count, "epoch", epoch, "loss", round(avg, 6), "wall s", round(latest_s, 1), flush=True)
    if gpu: torch.cuda.synchronize()
    train_s = elapsed + time.perf_counter() - learn_start
    model.eval()
    estimates = []
    stds = []
    test_start = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(data.test), micro):
            chunk = data.test[start:start+micro]
            profile, current, _ = data.batch(chunk, torch.float32)
            outputs = model(profile, current)
            estimates.append(outputs["coefficients_01"].cpu().numpy())
            stds.append(outputs["coefficient_std_01"].cpu().numpy())
    if gpu: torch.cuda.synchronize()
    test_s = time.perf_counter() - test_start
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    dest = results / f"{domain}__{count}__cnn"
    theta = np.concatenate(estimates)
    assert theta.shape == (len(data.test),8) and np.isfinite(theta).all()
    np.savez(dest.with_suffix(".npz"), ids=data.ids[data.test], theta=theta, std=np.concatenate(stds),
             lo=data.lo[data.test], hi=data.hi[data.test])
    desc = f"Native set CNN, {epochs} fixed epochs of AdamW, effective batch 64, {count} fixed current(s) per material, 48 change-aware keyframes, 1,160 cells."
    metadata = dict(train_s=train_s, test_s=test_s, n_train=len(data.train), n_test=len(data.test),
                    setup_s=setup_s, description=desc, epochs=epochs, microbatch=micro,
                    device=torch.cuda.get_device_name(device) if gpu else "CPU",
                    pytorch=str(torch.__version__), parameters=sum(p.numel() for p in model.parameters()))
    dest.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    preprocess = dict(input_channels=["concentration_nd_minus_1", "tau_over_campaign_horizon", "x_nd", "valid_time_mask"],
                      dataset_scan={"tau_horizon":data.horizon})
    torch.save(dict(model_configuration=model.configuration_dict(), model_state_dict=model.state_dict(),
                    run_configuration={"preprocessing_contract":preprocess}), work / "final.pt")
    print("completed", domain, count, train_s, test_s, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=DOMAINS)
    parser.add_argument("count", type=int, choices=(1,3))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--micro", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    a = parser.parse_args()
    train(a.domain, a.count, a.epochs, a.micro, a.device, a.resume)
