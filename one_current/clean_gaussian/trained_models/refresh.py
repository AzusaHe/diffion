"""Stage corrected same-fit checkpoints and verify their held-out predictions on CPU.

Run from a checkout with the archived clean package and corrected training artifacts.
--available stages any finished corrected fits. --publish requires all sixty.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import torch

if __package__:
    from . import combine_losses
    from .collect import BENCHMARK, CACHE, PACKAGE, collect_model, csv_write, relative, sha256
    from .inference import load_checkpoint
    from .loss_view import render
    from .verify_models import verify
else:
    import combine_losses
    from collect import BENCHMARK, CACHE, PACKAGE, collect_model, csv_write, relative, sha256
    from inference import load_checkpoint
    from loss_view import render
    from verify_models import verify

from corrected_grid_inputs import GRIDS, REVISION, digest_file
from grid40_prepare import mmap_grid_field

STAGE = CACHE / "tm_stage"
QA = CACHE / "tm_stage_qa"
DOMAIN = "gaussian_noisy"
INDEX = BENCHMARK / "corrected_noisy_results" / "material_index.npz"
JSON_KEYS = ("source_folder", "source_grid", "input_sha256", "source_sha256", "dataset_revision")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def table(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def portable_epoch_hashes(path):
    """The canonical CSV checksum survives Git's LF/CRLF checkout conversion."""
    raw = path.read_bytes()
    return {
        "packaged_epoch_history_sha256": hashlib.sha256(raw).hexdigest(),
        "packaged_epoch_history_lf_sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        "packaged_epoch_history_lf_normalization": "Replace each CRLF byte pair with LF, then SHA-256 the resulting bytes.",
    }


def check_split(common):
    split = read(PACKAGE / "split_indices.json")
    materials = table(PACKAGE / "materials.csv")
    assert len(materials) == 1947 and len(set(common["ids"])) == 1947
    for label, key, n in (("train","train",1402), ("val","validation",156), ("test","test",389)):
        rows = np.flatnonzero(common["split"] == label)
        assert len(rows) == n
        np.testing.assert_array_equal(rows, split[key]["benchmark_rows_zero_based"])
        np.testing.assert_array_equal(common["ids"][rows], split[key]["material_ids"])
    for i, row in enumerate(materials):
        assert (int(row["benchmark_row_zero_based"]), int(row["material_id"]), row["split"]) == (
            i, common["ids"][i], common["split"][i])
        slots = [int(row[f"current_{j}_source_index_zero_based"]) for j in range(1,4)]
        currents = [float(row[f"current_{j}_iota"]) for j in range(1,4)]
        np.testing.assert_array_equal(slots, common["chosen"][i])
        np.testing.assert_array_equal(currents, common["iota"][i])


def property_scores(theta, truth, lo, hi):
    u = np.linspace(np.maximum(lo,0), np.minimum(hi,2), 301).T / 2
    basis = np.stack([(1-u)**3, 3*(1-u)**2*u, 3*(1-u)*u**2, u**3],axis=2)
    def fields(p):
        z = np.einsum("ijk,ik->ij", basis, p[:,:4])
        return 20**z, 1.25*np.einsum("ijk,ik->ij", basis, p[:,4:])-.25
    d,t = fields(theta)
    true_d,true_t = fields(truth)
    d_error = 100*np.linalg.norm(d-true_d,axis=1)/np.linalg.norm(true_d,axis=1)
    t_error = np.sqrt(np.mean((t-true_t)**2,axis=1))
    return {"median_d_relative_l2_percent":float(np.median(d_error)),
            "median_t_absolute_rmse":float(np.median(t_error))}


@torch.inference_mode()
def check_weights(leaf, field, common, k, pred, best):
    """Reconstruct raw observations independently and check every archived test output."""
    model = load_checkpoint(leaf)
    diagnostics = {}
    for name, expected in (("test", best["test_mse"]),("val",best["validation_mse"])):
        rows = np.flatnonzero(common["split"] == name)
        batches = []
        for i in range(0,len(rows),8):
            j = rows[i:i+8]
            batches.append(model(field[j,:k], common["tau"][j,:k],common["iota"][j,:k],
                                 common["mask"][j,:k]).numpy())
        output = np.concatenate(batches)
        mse = np.mean((output - common["theta"][rows])**2)
        np.testing.assert_allclose(mse, expected, rtol=3e-4, atol=1e-6, err_msg=str(leaf))
        diagnostics[f"cpu_{name}_mse"] = float(mse)
        if name == "test":
            diff = np.abs(output-pred["theta"]).max()
            np.testing.assert_allclose(output, pred["theta"], rtol=0, atol=5e-5, err_msg=str(leaf))
            diagnostics["cpu_max_abs_test_coefficient_difference"] = float(diff)
    return diagnostics


def stage_model(nx, nt, k, seed, arch, common, field, bounds, prep, result, source):
    measurements = read(result.with_suffix(".json"))
    assert measurements["epoch_metrics"] == "same_fit" and prep["dataset_revision"] == REVISION
    assert (measurements["training_precision"], measurements["cpu_threads"]) == ("float32_tf32_off",1)
    assert all(measurements[key] == prep[key] for key in JSON_KEYS)
    assert (measurements["seed"],measurements["architecture"],measurements["domain"],
            measurements["currents"],measurements["n_space"],measurements["n_time"]) == (seed,arch,DOMAIN,k,nx,nt)
    assert (measurements["optimizer"],measurements["learning_rate"],measurements["weight_decay"]) == ("AdamW",.001,.01)
    assert measurements["scheduler"] == "CosineAnnealingLR, per optimizer step"
    assert measurements["checkpoint_root"] == relative(source.parent.parent)
    files = dict(weights=source/"best.pt", history=source/"training.json", metrics=source/"epoch_metrics.csv",
                 predictions=result.with_suffix(".npz"), settings=result.with_suffix(".json"))
    hashes = {key: sha256(path) for key,path in files.items()}
    qa_path = QA / f"{nx}_{k}_{seed}_{arch}.json"
    model_path = Path(f"grid_{nx}x{nt}",DOMAIN,f"{k}_current",f"seed_{seed}",arch)
    leaf = STAGE / model_path
    if qa_path.is_file():
        cached = read(qa_path)
        if (cached["hashes"] == hashes and all((leaf/name).is_file() and sha256(leaf/name)==digest
                for name,digest in cached["staged_hashes"].items()) and
                read(leaf/"manifest.json")["files"].get("packaged_epoch_history_sha256")==cached["staged_hashes"]["epochs.csv"]):
            manifest = read(leaf/"manifest.json")
            portable = portable_epoch_hashes(leaf/"epochs.csv")
            if any(manifest["files"].get(key) != value for key,value in portable.items()):
                manifest["files"].update(portable)
                save(leaf/"manifest.json",manifest)
                cached["staged_hashes"]["manifest.json"] = sha256(leaf/"manifest.json")
                save(qa_path,cached)
            return cached["inventory"]
    test = np.flatnonzero(common["split"]=="test")
    row = collect_model(nx,nt,DOMAIN,k,seed,arch,common["ids"][test],package=STAGE,sources=(source,result))
    manifest = read(leaf/"manifest.json")
    with np.load(files["predictions"],allow_pickle=False) as z:
        pred = dict(z)
    np.testing.assert_array_equal(pred["lo"],bounds[k][0])
    np.testing.assert_array_equal(pred["hi"],bounds[k][1])
    assert np.all((pred["theta"]>=0)&(pred["theta"]<=1))
    history = manifest["training"]["original_epoch_validation_mse"]
    epochs = table(files["metrics"])
    assert len(epochs)==100 and set(epochs[0])=={"epoch","train_mse","validation_mse","test_mse"}
    for i,epoch in enumerate(epochs,1):
        assert int(epoch["epoch"])==i
        for key in ("train_mse","validation_mse","test_mse"):
            epoch[key] = float(epoch[key])
            assert np.isfinite(epoch[key]) and epoch[key] >= 0
        epoch["original_validation_mse"] = history[i-1]
        assert epoch["validation_mse"] == history[i-1]
    best = epochs[row["best_epoch_one_based"]-1]
    np.testing.assert_allclose(np.mean((pred["theta"]-common["theta"][test])**2),best["test_mse"],rtol=1e-5)
    diagnostics = check_weights(leaf,field,common,k,pred,best)
    manifest["dataset"] = {key: prep[key] for key in (*JSON_KEYS,"index_sha256","selection")}
    manifest["training"].update(training_precision=measurements["training_precision"],cpu_threads=1)
    manifest["files"].update(epoch_metrics_sha256=hashes["metrics"], predictions_sha256=hashes["predictions"],
                             original_epoch_metrics_from_benchmark_root=relative(files["metrics"]))
    manifest["timing"]["post_training_epoch_evaluation_seconds"] = measurements["epoch_evaluation_s"]
    csv_write(leaf/"epochs.csv",epochs)
    manifest["files"].update(portable_epoch_hashes(leaf/"epochs.csv"))
    save(leaf/"manifest.json",manifest)
    scores = property_scores(pred["theta"],common["theta"][test],pred["lo"],pred["hi"])
    provenance = dict(status="same_fit", epochs=100, optimizer_updates=2200, domain=DOMAIN,
                      original_best_epoch=row["best_epoch_one_based"],
                      original_best_archived_validation_mse=row["best_validation_mse"],
                      checkpoint_selection="First minimum of validation coefficient MSE; held-out scoring after selection.",
                      metric="Mean squared error over all materials in each split and eight normalized coefficients.",
                      provenance="Train and test scored on the exact frozen end-of-epoch weights after fitting. Validation recorded in the same fit.",
                      original_best_train_mse=best["train_mse"],original_best_validation_mse=best["validation_mse"],
                      original_best_test_mse=best["test_mse"],original_files_sha256=hashes,
                      cpu_test_materials=389,cpu_validation_materials=156,**diagnostics,**scores)
    save(leaf/"loss_provenance.json",provenance)
    published = {name:sha256(leaf/name) for name in ("best.pt","epochs.csv","manifest.json","loss_provenance.json")}
    save(qa_path,dict(hashes=hashes,staged_hashes=published,inventory=row,scores=scores,
                      source_sha256=prep["source_sha256"],input_sha256=prep["input_sha256"],**diagnostics))
    print(f"Verified {model_path}: max CPU test coefficient difference "
          f"{diagnostics['cpu_max_abs_test_coefficient_difference']:.3g}",flush=True)
    return row


def grid_models(nx, nt, index, sample):
    folder = BENCHMARK/"corrected_noisy_results"/f"grid{nx}"
    if not (folder/"preparation.json").is_file():
        return []
    ready = []
    for k,seed,arch in product((1,3),range(3),("fno","mlp" if nx==20 else "conv_mlp")):
        stem = f"{DOMAIN}__{k}__{arch}"
        result = folder/f"seed{seed}"/stem
        if result.with_suffix(".json").is_file():
            metadata=read(result.with_suffix(".json"))
            if (metadata.get("epoch_metrics"),metadata.get("training_precision"),metadata.get("cpu_threads"))==(
                    "same_fit","float32_tf32_off",1):
                ready.append((k,seed,arch,result))
    if not ready:
        return []
    prep = read(folder/"preparation.json")
    assert prep["index_sha256"] == sha256(INDEX) and prep["source_grid"] == [nx,nt]
    assert prep["source_folder"].startswith("Corrected_noisy_data/") and prep["eligible"]==len(index["ids"])
    root = CACHE/"corrected_noisy_supervised"/f"{nx}x{nt}"
    archive = root/"inputs.npz"
    assert prep["input_sha256"] == digest_file(archive)
    with np.load(archive,allow_pickle=False) as z:
        common = {name:z[name] for name in (*index,"tau","mask")}
    for key,a in index.items():
        np.testing.assert_array_equal(a,common[key])
    field = mmap_grid_field(archive,DOMAIN)
    i = int(np.flatnonzero(common["ids"]==sample["material_id"].item())[0])
    np.testing.assert_array_equal(sample["current_indices"],common["chosen"][i])
    np.testing.assert_array_equal(sample["iota"],common["iota"][i])
    np.testing.assert_array_equal(sample["true_coefficients"],common["theta"][i])
    for name,a in ((f"{DOMAIN}_{nx}",field[i]),(f"{DOMAIN}_tau_{nx}",common["tau"][i]),
                   (f"{DOMAIN}_mask_{nx}",common["mask"][i])):
        sample[name]=np.array(a)
    test = np.flatnonzero(common["split"]=="test")
    bounds = {}
    for k in (1,3):
        lo,hi = [],[]
        for j in np.array_split(test,25):
            c = field[j,:k]
            lo.extend(np.nanmin(c,axis=(1,2,3)))
            hi.extend(np.nanmax(c,axis=(1,2,3)))
        bounds[k] = np.asarray(lo),np.asarray(hi)
    raw_bounds = CACHE / f"corrected_{nx}_source_bounds_qa.npz"
    if raw_bounds.exists():
        with np.load(raw_bounds,allow_pickle=False) as audit:
            for name in ("ids","split","chosen"):
                np.testing.assert_array_equal(audit[name],common[name])
            for col,k in enumerate((1,3)):
                np.testing.assert_array_equal(audit["lo"][test,col],bounds[k][0])
                np.testing.assert_array_equal(audit["hi"][test,col],bounds[k][1])
    rows = []
    for k,seed,arch,result in ready:
        source = root/"checkpoints"/prep["input_sha256"][:16]/"same_fit_fp32"/f"seed{seed}"/result.name
        rows.append(stage_model(nx,nt,k,seed,arch,common,field,bounds,prep,result,source))
        with np.load(result.with_suffix(".npz")) as z:
            sample[f"expected_{nx}_{DOMAIN}_{k}_{seed}_{arch}"] = z["theta"][z["ids"]==common["ids"][i]][0]
    return rows


def publish(models,sample):
    clean = [row for row in table(PACKAGE/"inventory.csv") if row["domain"]=="clean"]
    if len(clean) != 60 or len({row["model_dir"] for row in clean}) != 60:
        raise ValueError("Expected 60 distinct clean model folders in the existing package.")
    if len(models) != 60 or len({row["model_dir"] for row in models}) != 60:
        counts = {nx: sum(int(row["grid_space"]) == nx for row in models) for nx in GRIDS}
        raise ValueError(f"Publication requires 12 verified corrected fits per grid. Available by grid: {counts}")
    for row in clean:
        name = row["model_dir"]
        assert sha256(PACKAGE/name/"best.pt")==row["weight_sha256"]
        shutil.copytree(PACKAGE/name, STAGE/name,dirs_exist_ok=True)
    for row in clean+models:
        leaf = STAGE/row["model_dir"]
        dataset = read(leaf/"manifest.json").get("dataset",{})
        row.update(dataset_revision=dataset.get("dataset_revision","original_clean"),
                   source_folder=dataset.get("source_folder","clean"),
                   epoch_history_status=read(leaf/"loss_provenance.json")["status"],
                   corrected_input_sha256=dataset.get("input_sha256",""))
    csv_write(STAGE/"inventory.csv",sorted(clean+models,key=lambda r:(int(r["grid_space"]),r["domain"],
                                                int(r["currents"]),int(r["seed"]),r["architecture"] != "fno")))
    np.savez_compressed(STAGE/"example_material_000002.npz",**sample)
    combine_losses.main(STAGE)
    render(STAGE)
    verify(STAGE)
    paths = [Path(n) for n in ("inventory.csv","epoch_losses.csv","original_checkpoint_scores.csv",
                               "example_material_000002.npz","loss_curves.html")]
    old_replays = []
    backup = CACHE/"tm_before_corrected"
    for row in models:
        name = Path(row["model_dir"])
        paths.extend(name / n for n in ("best.pt","epochs.csv","loss_provenance.json","manifest.json"))
        replay = PACKAGE/name/"replay_best.pt"
        if replay.exists():
            old = read(PACKAGE/name/"loss_provenance.json")
            if old.get("replay_best_checkpoint") != "replay_best.pt" and (backup/name/"loss_provenance.json").exists():
                old = read(backup/name/"loss_provenance.json")
            if old.get("replay_best_checkpoint") != "replay_best.pt":
                raise ValueError(f"Unrecognized existing checkpoint: {replay}")
            old_replays.append(replay)
    for name in paths:
        dest = PACKAGE/name
        if dest.is_file() and sha256(dest)==sha256(STAGE/name):
            continue
        if dest.is_file() and not (backup/name).exists():
            (backup/name).parent.mkdir(exist_ok=True,parents=True)
            shutil.copy2(dest,backup/name)
        tmp = dest.with_suffix(".publishing")
        shutil.copyfile(STAGE/name,tmp)
        tmp.replace(dest)
    for old in old_replays:
        target = backup/old.relative_to(PACKAGE)
        if target.is_file() and sha256(target)!=sha256(old):
            raise ValueError(f"Backup collision at {target}")
        target.parent.mkdir(parents=True,exist_ok=True)
        old.replace(target)
    for name in paths:
        assert sha256(STAGE/name)==sha256(PACKAGE/name)
    print("Published 60 corrected same-fit models alongside 60 verified clean models.",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--available",action="store_true",help="Stage and score completed corrected models.")
    mode.add_argument("--publish",action="store_true",help="Verify all 120 models and promote the staged bundle.")
    parser.add_argument("--grid",type=int,nargs="+",choices=list(GRIDS))
    args=parser.parse_args()
    if args.publish and args.grid:
        parser.error("Publication requires every grid.")
    torch.set_num_threads(3)
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    QA.mkdir(parents=True,exist_ok=True)
    STAGE.mkdir(parents=True,exist_ok=True)
    with np.load(INDEX,allow_pickle=False) as z:
        index = dict(z)
    check_split(index)
    with np.load(PACKAGE/"example_material_000002.npz",allow_pickle=False) as z:
        sample={k:z[k] for k in z.files if DOMAIN not in k}
    models=[]
    for nx in args.grid or GRIDS:
        models.extend(grid_models(nx,GRIDS[nx],index,sample))
    print(f"Verified {len(models)} corrected models in requested grids.",flush=True)
    if args.publish:
        publish(models,sample)


if __name__=="__main__":
    main()
