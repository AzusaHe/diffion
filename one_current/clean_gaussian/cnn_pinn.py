"""Run the archived gated PINN schedule on held-out material 2 with a freshly trained CNN prior."""
import argparse
import json
import time

import numpy as np
import torch

from cnn_prepare import DOMAINS, HERE, SOURCE, cnn
import three_current_pinn as pinn


def fit_material(domain, count, device, material_id, out, schedule=None):
    cnn.N_EXPERIMENTS = pinn.N_EXPERIMENTS = count
    cfg = json.loads((SOURCE / "reference_runs/pinn_clean_accepted/run_config.json").read_text())
    trainer_config = pinn.PINNTrainerConfig(**{**cfg["trainer"], **(schedule or {}), "device": device})
    network = pinn.FieldNetworkConfig(**cfg["field_network"])
    physics = pinn.PINNPhysicsConfig(**cfg["physics"])
    weights = pinn.PINNLossWeights(**cfg["loss_weights"])
    composite = pinn.SteadyCompositeConfig(**cfg["steady_composite"])
    np.random.seed(trainer_config.seed)
    torch.manual_seed(trainer_config.seed)
    torch.set_num_threads(1)
    if device == "cuda":
        torch.cuda.manual_seed_all(trainer_config.seed)
    with np.load(HERE / "cache/index.npz") as idx:
        row = np.flatnonzero(idx["ids"] == material_id).item()
        assert idx["split"][row] == "test"
        selected = idx["chosen"][row, :count]
    source = next((SOURCE / "data" / domain).glob(f"*/samples/material_{material_id:06d}.npz"))
    observations = pinn.load_three_current_observations(source, selected, concentration_field="concentration_nd")
    values = observations.concentration_nd[observations.observation_time_mask]
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = HERE / "cache" / f"cnn_{domain}_{count}" / "final.pt"
    init_start = time.perf_counter()
    init = pinn.infer_cnn_initialization(checkpoint, observations, device=device, n_keyframes=48)
    if device == "cuda": torch.cuda.synchronize()
    init_s = time.perf_counter() - init_start
    fit_start = time.perf_counter()
    model = pinn.ThreeCurrentPINN(physics=physics, field_config=network, initialization=init)
    trainer = pinn.PINNTrainer(model=model, observations=observations, config=trainer_config,
                               loss_weights=weights, steady_composite_config=composite, output_dir=out)
    status = "accepted"
    chosen = out / "final_checkpoint.pt"
    try:
        trainer.fit()
    except RuntimeError as err:
        if (str(err) != "No joint checkpoint passed the deterministic selection eligibility gates"
                and not str(err).startswith("Final model failed deterministic selection eligibility gates:")):
            raise
        status = "rejected"
        chosen = out / "latest_checkpoint.pt"
    if device == "cuda": torch.cuda.synchronize()
    fit_s = time.perf_counter() - fit_start
    payload = torch.load(chosen, weights_only=False, map_location="cpu")
    if schedule and payload["global_step"] != trainer.global_step:
        raise RuntimeError("Expected the reported checkpoint to account for the complete optimizer budget")
    theta = np.r_[payload["effective_D_coefficients_01"], payload["effective_t_plus_coefficients_01"]].reshape(1,8)
    if not np.isfinite(theta).all():
        raise FloatingPointError("A gated PINN checkpoint contains nonfinite coefficients")
    performed = {phase: max(r["phase_step"] for r in trainer.history if r["phase"] == phase)
                 for phase in ("field_fit", "physics_warmup", "joint")}
    return dict(id=int(material_id), theta=theta[0].tolist(), lo=float(values.min()), hi=float(values.max()),
                init_s=init_s, fit_s=fit_s, device=torch.cuda.get_device_name() if device=="cuda" else "CPU",
                status=status, checkpoint=chosen.name, steps=trainer.global_step, phases=performed,
                current_indices=selected.tolist(), currents_iota=observations.iota.tolist(),
                failed_gates=list((trainer.last_selection_eligibility or {}).get("failed_gates", [])))


def fit(domain, count, device):
    row = fit_material(domain, count, device, 2, HERE / "cache" / f"pinn_{domain}_{count}")
    dest = HERE / "results" / f"{domain}__{count}__cnn_pinn"
    np.savez(dest.with_suffix(".npz"), ids=[row["id"]], theta=[row["theta"]], lo=[row["lo"]], hi=[row["hi"]])
    description = ("Staged CNN-initialized PINN, full valid observations, material 2 only. "
                   "1,000 field-only, 2,000 physics warmup, 4,000 joint AdamW steps and up to 100 optional L-BFGS iterations. "
                   "Rejected cases use the finite terminal joint checkpoint for diagnosis.")
    dest.with_suffix(".json").write_text(json.dumps(dict(train_s=0., test_s=row["fit_s"], n_train=0, n_test=1,
        warm_start="cnn", description=description, device=row["device"],
        gate_status=row["status"], gate_accepted=(row["status"]=="accepted"), checkpoint=row["checkpoint"],
        cnn_init_s=row["init_s"], current_indices=row["current_indices"], currents_iota=row["currents_iota"], field_steps=1000,
        warmup_steps=2000, joint_steps=4000, max_lbfgs_steps=100), indent=2))
    print(domain, count, row["status"], "seconds", round(row["fit_s"], 1), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("domain", choices=DOMAINS)
    p.add_argument("count", type=int, choices=(1,3))
    p.add_argument("--device", default="cuda", choices=("cuda","cpu"))
    a = p.parse_args()
    fit(a.domain, a.count, a.device)
