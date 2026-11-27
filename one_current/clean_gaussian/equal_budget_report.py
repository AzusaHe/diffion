"""Validate equal-budget fits and assemble the Accuracy tab's rows and CSV."""
import csv
import io
import json

import numpy as np

STAGES = ("ttt", "field_prefit", "physics_warmup", "joint", "material_only", "correction")


def build_equal_budget(root, materials, bank, baselines, pilot, paired, errors, label, preview=False):
    directory = root / "equal_budget_results"
    methods = list(dict.fromkeys([*pilot, *paired, "cnn_pinn"]))
    rows, summaries = [], []
    expected = [m["id"] for m in materials[:8]]
    by_id = {m["id"]:m for m in materials}
    shared_ids = expected[:6]
    assert shared_ids == [2, 3, 11, 21, 22, 32] and expected[6:] == [37, 45]
    for r in bank.values():
        if r["slug"] in baselines:
            obs = np.asarray(r["observations"])
            summaries.append(summary(r, obs, {}, [], shared_ids))
    for domain in ("clean", "gaussian_noisy"):
        for count in (1, 3):
            for slug in methods:
                arms = ("control", "slopes") if slug in paired else ("control",)
                reference = None
                for arm in arms:
                    stem = directory / f"{domain}__{count}__{slug}__{arm}"
                    if preview and not stem.with_suffix(".json").exists():
                        continue
                    metadata = json.loads(stem.with_suffix(".json").read_text())
                    stages = metadata["stages"]
                    assert set(stages) <= set(STAGES), (stem, stages)
                    assert all(isinstance(n, int) and n >= 0 for n in stages.values()), stem
                    assert sum(stages.values()) == metadata["budget"] == 1000, stem
                    assert metadata.get("lbfgs_steps", metadata.get("lbfgs_iters")) == 0, stem
                    source = None
                    if slug.endswith("fno_ttt"):
                        source = bank[f"{domain}__{count}__fno_ttt"]
                        assert stages["ttt"] == source["meta"]["ttt_steps"] == 150, stem
                    else:
                        assert not stages.get("ttt"), stem
                    with np.load(stem.with_suffix(".npz")) as z:
                        ids, theta, lo, hi = (z[k] for k in ("ids", "theta", "lo", "hi"))
                        n = len(ids)
                        expected_ids = shared_ids if slug == "cnn_pinn" else expected
                        assert ids.tolist() == expected_ids and theta.shape == (n, 8), stem
                        currents = [by_id[mid]["chosen"][:count] for mid in ids]
                        assert np.isfinite(theta).all() and np.all((theta >= 0) & (theta <= 1)), stem
                        assert lo.shape == hi.shape == (n,) and np.isfinite(lo).all() and np.isfinite(hi).all(), stem
                        counts = z["steps"] if slug == "cnn_pinn" else z["total_updates"]
                        np.testing.assert_array_equal(counts, np.full(n, 1000), err_msg=stem.name)
                        if slug == "cnn_pinn":
                            ineligible = metadata["ineligible"]
                            assert sorted(m["id"] for m in ineligible) == expected[6:], stem
                            for m in ineligible:
                                assert m["reason"] and m["current_indices"] == by_id[m["id"]]["chosen"][:count], stem
                            gate = z["gate_status"].tolist()
                            assert len(gate) == n and set(gate) <= {"accepted", "rejected"}
                            assert gate == metadata["gate_status"], stem
                            source = bank[f"{domain}__{count}__cnn"]
                            init_s = metadata["cnn_init_s"] / n
                            np.testing.assert_allclose(z["elapsed_s"].sum(), metadata["test_s"])
                            np.testing.assert_allclose(z["init_s"].sum(), metadata["cnn_init_s"])
                        else:
                            gate = []
                            ineligible = []
                            np.testing.assert_array_equal(z["ttt_updates"], np.full(n, stages.get("ttt", 0)))
                            np.testing.assert_array_equal(z["pinn_updates"] + z["ttt_updates"], counts)
                            init_s = source["per_s"] if source else 0.0
                        if "current_indices" in z:
                            np.testing.assert_array_equal(z["current_indices"], currents, err_msg=stem.name)
                        np.testing.assert_array_equal(metadata["current_indices"], currents, err_msg=stem.name)
                        if reference is not None:
                            base_lo, base_hi, base_stages = reference
                            np.testing.assert_array_equal(lo, base_lo)
                            np.testing.assert_array_equal(hi, base_hi)
                            assert stages == base_stages, stem
                        reference = lo, hi, stages
                        old_slug = slug.replace("pinn_frozen_", "pinn_data_first_")
                        archived = bank.get(f"{domain}__{count}__{old_slug}")
                        if archived:
                            archived_obs = np.array(archived["observations"])
                            np.testing.assert_array_equal(lo[:len(archived_obs)], archived_obs[:, 3])
                            np.testing.assert_array_equal(hi[:len(archived_obs)], archived_obs[:, 4])
                        d, t = errors(theta, [by_id[mid]["truth"] for mid in ids], lo, hi)
                        obs = [[int(mid), float(de), float(te), float(l), float(h)] for mid, de, te, l, h in zip(ids, d, t, lo, hi)]
                    stage_s = metadata["test_s"] / n
                    assert np.isfinite([init_s,stage_s]).all() and init_s >= 0 and stage_s > 0, stem
                    r = dict(slug=slug, name=label(slug), arm=arm, stages=stages, family="PINN refinement",
                             domain=domain, count=count, n=n, observations=obs, gates=gate, ineligible=ineligible, budget=1000,
                             per_s=init_s+stage_s, init_s=init_s, stage_s=stage_s,
                             training=source["training"] if source else 0., tuning=source["tuning"] if source else 0.,
                             device=metadata.get("gpu") or metadata["device"],
                             forward_s=metadata.get("forward_s", 0.0)/n)
                    rows.append(r)
                    summaries.append(summary(r, np.asarray(obs), stages, gate, shared_ids))
    if not preview:
        assert len(rows) == 148 and len(summaries) == 212
        for slug in methods:
            for arm in ("control","slopes") if slug in paired else ("control",):
                matches=[r for r in rows if r["slug"] == slug and r["arm"] == arm]
                assert len(matches)==4 and all(r["stages"]==matches[0]["stages"] for r in matches)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
    writer.writeheader()
    writer.writerows(summaries)
    data = stream.getvalue()
    if not preview:
        (root / "equal_budget_accuracy_cost.csv").write_bytes(data.encode("utf-8"))
    return dict(rows=rows, csv=data, shared_ids=shared_ids)


def summary(r, obs, stages, gate, shared_ids):
    matched=obs[np.isin(obs[:,0], shared_ids)]
    np.testing.assert_array_equal(matched[:,0], shared_ids)
    span=np.minimum(2,obs[:,4])-np.maximum(0,obs[:,3])
    matched_span=np.minimum(2,matched[:,4])-np.maximum(0,matched[:,3])
    return dict(domain=r["domain"], currents=r["count"], method=r["slug"], family=r["family"],
                arm=r.get("arm",""), n_test=r["n"], budget=r.get("budget",""),
                **{k+"_updates":stages.get(k,0) if stages else "" for k in STAGES},
                D_median_pct=float(np.median(obs[:,1])), t_median_rmse=float(np.median(obs[:,2])),
                median_observed_c_span=float(np.median(span)), matched6_D_median_pct=float(np.median(matched[:,1])),
                matched6_t_median_rmse=float(np.median(matched[:,2])),
                matched6_median_observed_c_span=float(np.median(matched_span)),
                pipeline_s_per_material=r["per_s"], initialization_s_per_material=r["init_s"],
                fit_s_per_material=r["stage_s"], shared_training_s=r["training"], validation_search_s=r["tuning"],
                included_solver_seed_s_per_material=r.get("forward_s",""),
                device=r["device"] if "device" in r else r["meta"]["device"],
                gate_accepted=gate.count("accepted") if gate else "", gate_rejected=gate.count("rejected") if gate else "",
                ineligible_ids=" ".join(str(m["id"]) for m in r.get("ineligible",[])))
