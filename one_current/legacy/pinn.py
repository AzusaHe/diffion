"""Repeat the recorded secondary-campaign PINN schedule on one actual experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cnn import HERE, LEGACY, DOMAINS
import three_current_pinn as physics

physics.N_EXPERIMENTS = 1


def main(domain, device):
    out = HERE / ("pinn_" + domain)
    out.mkdir(exist_ok=True)
    reference = "pinn_clean_accepted" if domain == "clean" else "pinn_xray_noisy_rejected"
    reference_config = json.loads((LEGACY / "reference_runs" / reference / "run_config.json").read_text())
    cfg = physics.PINNTrainerConfig(**{**reference_config["trainer"], "device":device})
    field = physics.FieldNetworkConfig(**reference_config["field_network"])
    weights = physics.PINNLossWeights(**reference_config["loss_weights"])
    model_physics = physics.PINNPhysicsConfig(**reference_config["physics"])
    composite = physics.SteadyCompositeConfig(**reference_config["steady_composite"])
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)
    if device == "cuda": torch.cuda.manual_seed_all(cfg.seed)
    artifact = next((LEGACY / "data/xray_noisy").glob("*/samples/material_000002.npz"))
    with np.load(HERE / "data/index.npz") as d:
        row = np.flatnonzero(d["ids"]==2).item()
        index = int(d["chosen"][row])
    obs = physics.load_three_current_observations(artifact,current_indices=[index],concentration_field=DOMAINS[domain])
    assert len(obs.iota)==1 and obs.steady_eligible[0]
    checkpoint = HERE / ("cnn_"+domain) / "final_checkpoint.pt"
    init = physics.infer_cnn_initialization(checkpoint,obs,device=device,n_keyframes=48)
    model = physics.ThreeCurrentPINN(physics=model_physics,field_config=field,initialization=init)
    trainer = physics.PINNTrainer(model=model,observations=obs,config=cfg,loss_weights=weights,
                                   steady_composite_config=composite,output_dir=out)
    try:
        trainer.fit()
        outcome="accepted"
        chosen=out/"final_checkpoint.pt"
    except RuntimeError as error:
        if str(error) != "No joint checkpoint passed the deterministic selection eligibility gates":
            raise
        outcome="rejected"
        chosen=out/"latest_checkpoint.pt"
    fitted,_=physics.load_pinn_checkpoint(chosen,device="cpu")
    truth=physics.read_synthetic_truth_for_evaluation(artifact)
    evaluation=physics.compare_with_synthetic_truth(fitted,truth)
    result=dict(material=2,domain=domain,source_current_index=index,iota=float(obs.iota[0]),checkpoint=chosen.name,status=outcome,
                full_curve_scores=evaluation)
    physics._atomic_write_json(out/"one_current_result.json",result)
    print(outcome,domain,evaluation["D_curve_rmse_01"],evaluation["t_plus_curve_rmse_01"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain",choices=list(DOMAINS))
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    main(args.domain,args.device)
