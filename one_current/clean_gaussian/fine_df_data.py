"""Prepare native-time, four-cell-averaged observations for the held-out solver study."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import time

import numpy as np

from .prepare import CACHE, DOMAINS, HERE, paths

COHORT = 300
STARTS = ("center", "fno", "mlp", "fno_ttt", "mlp_ttt")
COUNTS = (1, 3)
STORE = CACHE / "fine_df_v1"
RESULTS = HERE / "fine_df_results"
ARCHIVE = HERE / "results"


def write_npz(dest, **arrays):
    tmp = dest.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    replace_when_readers_close(tmp, dest)


def write_json(dest, value):
    tmp = dest.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    replace_when_readers_close(tmp, dest)


def replace_when_readers_close(tmp, dest):
    # On Windows a progress reader can briefly hold an existing checkpoint open.
    for attempt in range(8):
        try:
            tmp.replace(dest)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.01 * 2**attempt)


@lru_cache(maxsize=1)
def cohort():
    with np.load(CACHE / "index.npz") as z:
        take = np.flatnonzero(z["split"] == "test")[:COHORT]
        d = {k: z[k][take] for k in ("ids", "chosen", "theta", "iota")}
    if len(set(d["ids"])) != COHORT or any(len(set(js)) != 3 for js in d["chosen"]):
        raise ValueError("Expected 300 unique materials and three distinct currents each")
    return d


def sample_path(mid):
    return STORE / "samples" / f"{mid:06d}.npz"


def get_starts():
    ids = cohort()["ids"]
    starts = {}
    hashes = {}
    for domain in DOMAINS:
        for count in COUNTS:
            starts[domain, count, "center"] = np.full((len(ids), 8), .5)
            for label in STARTS[1:]:
                path = ARCHIVE / f"{domain}__{count}__{label}.npz"
                with np.load(path) as z:
                    archive_ids, theta = z["ids"], z["theta"].astype(float)
                if len(set(archive_ids)) != len(archive_ids):
                    raise ValueError(f"Duplicate IDs in {path}")
                by_id = dict(zip(archive_ids, theta))
                values = np.array([by_id[mid] for mid in ids])
                if values.shape != (len(ids), 8) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid start coefficients in {path}")
                starts[domain, count, label] = np.clip(values, 1.e-6, 1.-1.e-6)
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return starts, hashes


def prepare():
    clock = time.perf_counter()
    (STORE / "samples").mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    ix = cohort()
    sources = {d: paths(d) for d in DOMAINS}

    def make(row):
        mid = int(ix["ids"][row])
        chosen = ix["chosen"][row]
        dest = sample_path(mid)
        if not dest.exists():
            with (np.load(sources["clean"][mid]) as clean,
                  np.load(sources["gaussian_noisy"][mid]) as noisy):
                for key in ("material_sample_id", "valid_time_mask", "tau", "iota", "current_density_a_m2",
                            "x_cells_nd", "D_coefficients_01", "t_plus_coefficients_01", "run_fingerprint"):
                    np.testing.assert_array_equal(clean[key], noisy[key])
                assert clean["material_sample_id"].item() == mid
                assert "gaussian" in str(noisy["noise_model"]).lower() and noisy["noise_sigma_ref_mM"].item() == 2.2
                np.testing.assert_array_equal(clean["iota"][chosen], ix["iota"][row])
                np.testing.assert_array_equal(np.r_[clean["D_coefficients_01"], clean["t_plus_coefficients_01"]], ix["theta"][row])
                mask = clean["valid_time_mask"][chosen]
                tau = clean["tau"]
                if mask.shape != (3, 151) or np.any(mask[:, 1:] & ~mask[:, :-1]) or np.any(mask.sum(1) < 2):
                    raise ValueError(f"Unexpected data coverage for material {mid}")
                if tau.shape != (151,) or tau[0] != 0 or not np.all(np.diff(tau) > 0):
                    raise ValueError(f"Unexpected native time grid for material {mid}")
                kwargs = dict(id=mid, chosen=chosen, iota=clean["iota"][chosen], density=clean["current_density_a_m2"][chosen],
                              truth=ix["theta"][row], tau=np.tile(tau, (3, 1)), mask=mask, fingerprint=clean["run_fingerprint"])
                for domain, z in zip(DOMAINS, (clean, noisy)):
                    raw = z["concentration_nd"][chosen]
                    if raw.shape != (3, 151, 1160) or not np.isfinite(raw[mask]).all():
                        raise ValueError(f"Invalid observations for {mid} in {domain}")
                    binned = np.full((3, 151, 290), np.nan)
                    binned[mask] = raw[mask].reshape(-1, 290, 4).mean(-1, dtype=np.float64)
                    kwargs[domain] = binned
                write_npz(dest, **kwargs)
        with np.load(dest) as z:
            assert z["id"].item() == mid
            np.testing.assert_array_equal(z["chosen"], chosen)
            mask = z["mask"]
            fields = [z[d][mask] for d in DOMAINS]
            delta = fields[1] - fields[0]
            return z["density"], mask.sum(1), z["fingerprint"].item(), float(np.sum(delta**2)), delta.size

    with ThreadPoolExecutor(5) as pool:
        records = list(pool.map(make, range(COHORT)))
    densities, valid, fingerprints, ss, n = map(np.asarray, zip(*records))
    write_npz(RESULTS / "cohort.npz", ids=ix["ids"], theta_true=ix["theta"], current_indices=ix["chosen"],
              iota=ix["iota"], current_density_a_m2=densities, valid_frames=valid, fingerprint=fingerprints)
    _, hashes = get_starts()
    info = dict(cohort=COHORT, cohort_source="First 300 test materials in the existing cache/index.npz ordering",
                domains=list(DOMAINS), current_counts=list(COUNTS), starts=list(STARTS),
                observation_source="../latest_synthetic_ml/data/{clean,gaussian_noisy}/<batch>/samples/material_<id>.npz",
                noise="Source Gaussian-noisy concentration_nd, noise_sigma_ref_mM=2.2; measurement noise depends on concentration.",
                observed_noise_rms_after_binning=float(np.sqrt(ss.sum() / n.sum())),
                observations="1160 native cells averaged in consecutive groups of four in float64. All valid native timestamps up to 151.",
                observation_grid=[290,151], solver_cells=580, theta0_clip=[1.e-6,1.-1.e-6], start_archives_sha256=hashes,
                budget_evals={"nm":500,"cmaes":500,"bo":35},target_concentration_rmse=1.e-4, max_wall_s=300.,
                objective="Square root of summed squared dimensionless concentration residuals divided by 290 times total valid current-time observations.",
                evaluation_policy="First objective call includes the saved start or geometric center, counts toward budget and is eligible as best for every optimizer.",
                timing="Elapsed wall seconds of each fit including initial fit evaluation and optimizer overhead, excluding source loading, JIT compilation and prior inference.",
                archived_initializers="Domain- and current-specific archived predicted coefficient vectors, with archived TTT where indicated. No new training or TTT.",
                stopping="First target, budget, wall cap, or optimizer convergence; seed is material ID (CMA random seed ID+1).",
                optimizers="Adaptive bounded Nelder-Mead; bounded CMA-ES sigma=0.25; skopt GP expected improvement with 16 random points after the seed.",
                observation_bounds="Each material/domain/current case uses the min and max valid fine-grid observed concentrations for its property-error interval.")
    write_json(RESULTS / "protocol.json", info)
    print(f"Prepared paired observations and starts for {COHORT} materials in {time.perf_counter()-clock:.1f}s.", flush=True)


if __name__ == "__main__":
    prepare()
