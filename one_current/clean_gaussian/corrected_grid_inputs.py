"""Pack the supplied corrected observation grid using the original benchmark index."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import re

import numpy as np

from prepare import CACHE, DATA, HERE


GRIDS = {20: 13, 40: 26, 80: 52, 160: 104, 290: 151}
SOURCE = DATA / "Corrected_noisy_data"
OUTPUT = HERE / "corrected_noisy_results"
REVISION = "corrected_grid_native_v1"


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def grid_folder(space):
    time = GRIDS[space]
    matches = [p for p in SOURCE.iterdir() if p.is_dir() and
               re.search(rf"(?<!\d){space}[\s_x×-]+{time}(?!\d)", p.name, re.I)]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one supplied {space} x {time} folder below {SOURCE}; found {matches}")
    return matches[0]


def pack(space):
    nt = GRIDS[space]
    source = grid_folder(space)
    files = tuple(sorted(source.rglob("material_*.npz")))
    by_id = {int(p.stem.split("_")[-1]): p for p in files}
    if len(by_id) != len(files):
        raise ValueError(f"Repeated material ID in {source}")
    index = OUTPUT / "material_index.npz"
    with np.load(index) as z:
        common = dict(z)
    ids, chosen = common["ids"], common["chosen"]
    missing = set(ids) - by_id.keys()
    if missing:
        raise FileNotFoundError(f"{source.name} still needs {len(missing)} indexed material files")
    folder = CACHE / "corrected_noisy_supervised" / f"{space}x{nt}"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "inputs.npz"
    report = OUTPUT / f"grid{space}" / "preparation.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    field = np.empty((len(ids),3,nt,space), dtype=np.float32)
    mask = np.empty(field.shape[:-1], dtype=bool)
    tau = np.empty(mask.shape, dtype=np.float64)
    with np.load(by_id[int(ids[0])]) as first:
        centers, faces = first["x_cells_nd"], first["x_faces_nd"]
    np.testing.assert_allclose(faces,np.linspace(0.,1.,space+1),atol=1e-7)
    if centers.shape != (space,) or not np.all((centers>faces[:-1]) & (centers<faces[1:])):
        raise ValueError(f"Invalid supplied cell centers for {source.name}")

    def read_row(row):
        mid = int(ids[row])
        raw = by_id[mid].read_bytes()
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            if int(z["material_sample_id"]) != mid or len(set(chosen[row])) != 3:
                raise ValueError(f"Material or current mismatch for {mid}")
            np.testing.assert_array_equal(common["theta"][row],np.r_[z["D_coefficients_01"],z["t_plus_coefficients_01"]])
            j = chosen[row]
            np.testing.assert_array_equal(common["iota"][row], z["iota"][j])
            np.testing.assert_array_equal(z["x_cells_nd"], centers)
            np.testing.assert_array_equal(z["x_faces_nd"], faces)
            c = z["concentration_nd"][j]
            valid = z["valid_time_mask"][j]
            if c.shape != (3,nt,space) or c.dtype != np.float32 or valid.shape != (3,nt):
                raise ValueError(f"Material {mid} supplies {c.shape} ({c.dtype}) and mask {valid.shape}, expected 3 x {nt} x {space}")
            t = z["tau"]
            times = np.broadcast_to(t,valid.shape) if t.ndim==1 else t[j]
            if np.any(valid.sum(1)<1) or np.any(valid[:,1:] & ~valid[:,:-1]):
                raise ValueError(f"Invalid observed-frame mask for material {mid}")
            if not np.isfinite(times[valid]).all() or np.any((np.diff(times,axis=1)<=0) & valid[:,1:]):
                raise ValueError(f"Invalid observation times for material {mid}")
            if not np.isfinite(c[valid]).all() or np.isfinite(c[~valid]).any():
                raise ValueError(f"Inconsistent concentration support for material {mid}")
            field[row] = c
            mask[row] = valid
            tau[row] = np.where(valid,times,0.)
            return hashlib.sha256(raw).digest()

    with ThreadPoolExecutor(max_workers=6) as pool:
        source_hash = hashlib.sha256(b"".join(pool.map(read_row,range(len(ids))))).hexdigest()
    # The stored grid is copied verbatim. Invalid padding is masked by the original trainer.
    index_hash = digest_file(index)
    previous = json.loads(report.read_text()) if report.exists() else None
    if previous and (previous["source_sha256"] != source_hash or previous["index_sha256"] != index_hash):
        raise ValueError(f"Corrected inputs changed since training at {space} x {nt}; previous results belong to another revision")
    if previous and archive.exists():
        if previous["input_sha256"] != digest_file(archive):
            raise ValueError(f"Cached input differs from the published fingerprint at {space} x {nt}")
        return archive, previous
    temp = archive.with_suffix(".building.npz")
    np.savez(temp, **common, mask=mask, tau=tau, gaussian_noisy=field)
    input_hash = digest_file(temp)
    if previous and previous["input_sha256"] != input_hash:
        raise ValueError(f"Rebuilt input differs from the published fingerprint at {space} x {nt}")
    temp.replace(archive)
    counts = mask.sum(-1)
    info = dict(dataset_revision=REVISION, source_grid=[space,nt], space_bins=space, max_selected_frames=nt,
                source_folder=str(source.relative_to(DATA).as_posix()), source_sha256=source_hash,
                index_sha256=index_hash, input_sha256=input_hash,
                eligible=len(ids), source_materials=len(files),
                split={s:int((common["split"]==s).sum()) for s in ("train","val","test")},
                selected_frames_per_current=[int(counts.min()),int(counts.max())],
                single_frame_trajectories=int(np.sum(counts==1)),
                valid_frames_total=int(counts.sum()),
                currents="Original index column 0 for one current, columns 0 to 2 for three currents.",
                selection="All supplied grid cells and all valid supplied frames, copied directly.",
                concentration_ranges={"gaussian_noisy":[float(np.nanmin(field)),float(np.nanmax(field))]})
    tmp = report.with_suffix(".tmp")
    tmp.write_text(json.dumps(info,indent=2)+"\n")
    tmp.replace(report)
    print(f"Verified and packed {len(ids)} corrected {space} x {nt} materials: {archive}",flush=True)
    return archive, info


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid",type=int,choices=GRIDS,required=True)
    pack(parser.parse_args().grid)
