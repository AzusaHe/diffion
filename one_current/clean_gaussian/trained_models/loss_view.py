"""Generate the offline model selector from the per-model manifests and CSVs."""
import csv
import json
from pathlib import Path


def render(package):
    with (package/"inventory.csv").open(newline="", encoding="utf-8") as stream:
        models = list(csv.DictReader(stream))
    data = []
    for row in models:
        folder = package/row["model_dir"]
        provenance = json.loads((folder/"loss_provenance.json").read_text())
        with (folder/"epochs.csv").open(newline="") as stream:
            history = list(csv.DictReader(stream))
        data.append(dict(grid=f"{row['grid_space']} × {row['grid_time']}", domain=row["domain"], k=row["currents"],
                         seed=row["seed"], arch=row["architecture"], folder=row["model_dir"], status=provenance["status"],
                         epoch=int(row["best_epoch_one_based"]),
                         best=[provenance[f"original_best_{split}_mse"] for split in ("train","validation","test")],
                         curves=[[float(entry[key]) for entry in history]
                                 for key in ("train_mse","validation_mse","test_mse","original_validation_mse")]))
    template = Path(__file__).with_name("loss_view_template.html").read_text(encoding="utf-8")
    content = template.replace("__MODEL_DATA__", json.dumps(data,separators=(",",":"),allow_nan=False))
    (package/"loss_curves.html").write_text(content,encoding="utf-8")


if __name__=="__main__":
    render(Path(__file__).resolve().parent)
