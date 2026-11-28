"""Build a cross-model table from validated, per-checkpoint epoch histories."""

import csv
import json
import math
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
MODEL_KEYS = ("grid_space", "grid_time", "domain", "currents", "seed", "architecture")
LOSS_KEYS = ("train_mse", "validation_mse", "test_mse", "original_validation_mse")


def write_table(name, rows):
    destination = PACKAGE / name
    temp = destination.with_suffix(".partial")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(destination)


def main():
    with (PACKAGE / "inventory.csv").open(newline="", encoding="utf-8") as stream:
        models = list(csv.DictReader(stream))
    if len(models) != 120 or len({model["model_dir"] for model in models}) != len(models):
        raise ValueError("Expected 120 unique models in inventory.csv")

    rows, best_checkpoints = [], []
    for model in models:
        folder = PACKAGE / model["model_dir"]
        provenance = json.loads((folder / "loss_provenance.json").read_text(encoding="utf-8"))
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        history = manifest["training"]["original_epoch_validation_mse"]
        with (folder / "epochs.csv").open(newline="", encoding="utf-8") as stream:
            epochs = list(csv.DictReader(stream))
        if len(epochs) != len(history) or len(history) != 100:
            raise ValueError(f"Expected 100 epochs in {folder}")
        labels = {key: model[key] for key in MODEL_KEYS}
        best_epoch = min(range(len(history)), key=history.__getitem__) + 1
        if (provenance["original_best_epoch"] != best_epoch
                or provenance["original_best_archived_validation_mse"] != history[best_epoch-1]):
            raise ValueError(f"Inconsistent original best checkpoint in {folder}")
        best_checkpoints.append({
            **labels, "original_best_epoch": best_epoch,
            "train_mse": provenance["original_best_train_mse"],
            "validation_mse": provenance["original_best_validation_mse"],
            "test_mse": provenance["original_best_test_mse"],
            "original_recorded_validation_mse": history[best_epoch-1],
            "replay_status": provenance["status"], "model_dir": model["model_dir"],
        })
        for epoch, (metrics, archived) in enumerate(zip(epochs, history), 1):
            losses = {key: float(metrics[key]) for key in LOSS_KEYS}
            if (int(metrics["epoch"]) != epoch or archived != losses["original_validation_mse"]
                    or not all(math.isfinite(x) and x >= 0 for x in losses.values())):
                raise ValueError(f"Inconsistent epoch {epoch} in {folder}")
            rows.append({**labels, "epoch": epoch, "replay_status": provenance["status"],
                         "replay_train_mse": losses["train_mse"], "replay_validation_mse": losses["validation_mse"],
                         "replay_test_mse": losses["test_mse"], "original_validation_mse": archived})

    write_table("epoch_losses.csv", rows)
    write_table("original_checkpoint_scores.csv", best_checkpoints)
    print(f"{len(rows):,} verified epochs across {len(models)} models.")


if __name__ == "__main__":
    main()
