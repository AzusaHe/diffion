"""Retrain supervised models on the corrected observations supplied for one grid."""
import argparse

import numpy as np
import torch

from corrected_grid_inputs import GRIDS, OUTPUT, REVISION, pack
from grid40_learn import prepare_supervised, run
from grid40_prepare import mmap_grid_field
from learn import HERE, prepare


def train(space):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    domain = "gaussian_noisy"
    nt = GRIDS[space]
    archive, info = pack(space)
    result = OUTPUT / f"grid{space}"
    provenance = {k:info[k] for k in ("dataset_revision","source_grid","source_folder","input_sha256","source_sha256")}
    provenance.update(training_precision="float32_tf32_off", cpu_threads=torch.get_num_threads())
    assert provenance["dataset_revision"] == REVISION
    weights = archive.parent / "checkpoints" / info["input_sha256"][:16] / "same_fit_fp32"
    provenance["checkpoint_root"] = weights.relative_to(HERE).as_posix()
    with np.load(archive,allow_pickle=False) as z:
        common = {k:z[k] for k in ("ids","split","theta","iota","tau","mask")}
        field = mmap_grid_field(archive, domain)
        direct = {**common,domain:field}
        for currents in (1,3):
            families = ("mlp","fno") if space == 20 else ("conv_mlp","fno")
            prev_device = None
            for arch in families:
                # The original dense 20 x 13 MLP was fitted on CPU. Both finer-grid
                # Conv-MLPs and FNOs used the GPU at every seed and current count.
                device = "cpu" if arch == "mlp" else "cuda"
                if device != prev_device:
                    if prev_device is not None:
                        del data
                    if space >= 160:
                        data = prepare_supervised(field, common, currents, device)
                    else:
                        data = prepare(direct, domain, currents, device)
                    if data["feat"].shape != (len(common["ids"]), 5*currents+1, nt, space):
                        raise ValueError("Wrong feature shape for supplied corrected grid")
                    prev_device = device
                for seed in (0,1,2):
                    run(data,domain,currents,arch,seed,device,space,nt,
                        result_root=result,checkpoint_root=weights,provenance=provenance,epoch_metrics=True)
            del data
    print(f"Completed all 12 corrected supervised fits at {space} x {nt}.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid",type=int,choices=GRIDS,required=True)
    train(parser.parse_args().grid)
