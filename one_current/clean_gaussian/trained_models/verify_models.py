"""Reproduce held-out material 000002 predictions on CPU from packaged files."""
import argparse
from itertools import product
from pathlib import Path
import sys

import numpy as np
import torch

if __package__:
    from .inference import FRAMES, load_checkpoint
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inference import FRAMES, load_checkpoint


PACKAGE = Path(__file__).resolve().parent


@torch.inference_mode()
def verify(root=PACKAGE, checkpoint=None):
    """Check one leaf or all 120 checkpoints against independent archived predictions."""
    root = Path(root)
    expected_runs = []
    for (width, frames), domain, count, seed in product(
            FRAMES.items(), ("clean", "gaussian_noisy"), (1, 3), range(3)):
        for arch in ("fno", "mlp" if width == 20 else "conv_mlp"):
            leaf = root / f"grid_{width}x{frames}" / domain / f"{count}_current" / f"seed_{seed}" / arch
            if checkpoint is None or leaf.resolve() == Path(checkpoint).resolve():
                expected_runs.append((leaf, width, domain, count, seed, arch))
    if not expected_runs:
        raise ValueError("The checkpoint must be a leaf directory inside this package.")
    maximum = -1.0
    worst_run = None
    with np.load(root / "example_material_000002.npz", allow_pickle=False) as example:
        assert example["material_id"].item() == 2
        for leaf, width, domain, count, seed, arch in expected_runs:
            model = load_checkpoint(leaf)
            predicted = model(example[f"{domain}_{width}"][:count], example[f"tau_{width}"][:count],
                              example["iota"][:count], example[f"mask_{width}"][:count])[0].numpy()
            reference = example[f"expected_{width}_{domain}_{count}_{seed}_{arch}"]
            difference = float(np.max(np.abs(predicted - reference)))
            np.testing.assert_allclose(predicted, reference, atol=5e-5, rtol=0, err_msg=str(leaf))
            if len(expected_runs) == 1:
                print("Predicted [D_0 .. D_3, t_plus_0 .. t_plus_3]:", predicted)
                print("Archived prediction:", reference)
            if difference > maximum:
                maximum, worst_run = difference, leaf.relative_to(root)
    print(f"Verified {len(expected_runs)} checkpoints on held-out material 000002. "
          f"Largest absolute coefficient difference: {maximum:.8g} ({worst_run}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Run one checkpoint directory and print its prediction.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    verify(checkpoint=args.checkpoint)
