"""Reference fine-grid forward RMSE at the known synthetic coefficients."""

import numpy as np

from .fine_df_data import COUNTS, DOMAINS, RESULTS, cohort, sample_path, write_json
from .fine_fv import objective


def describe(values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite reference RMSE")
    return dict(median=float(np.median(values)), p95=float(np.quantile(values, .95)),
                p99=float(np.quantile(values, .99)), min=float(np.min(values)),
                max=float(np.max(values)), at_or_below_target_1e_4=int(np.sum(values <= 1.e-4)))


def run():
    ids = cohort()["ids"]
    losses = {(domain, count): [] for domain in DOMAINS for count in COUNTS}
    noise = {count: [] for count in COUNTS}
    noise_ss = {count: 0. for count in COUNTS}
    valid_bins = {count: 0 for count in COUNTS}
    for mid in ids:
        with np.load(sample_path(mid)) as sample:
            currents, tau, mask, truth = [sample[k] for k in ("iota", "tau", "mask", "truth")]
            y = {domain: sample[domain] for domain in DOMAINS}
        for count in COUNTS:
            valid = mask[:count]
            for domain in DOMAINS:
                mse = objective(truth, currents[:count], tau[:count], valid, y[domain][:count])
                losses[domain, count].append(np.sqrt(mse))
            delta = y["gaussian_noisy"][:count][valid] - y["clean"][:count][valid]
            ss = np.sum(delta * delta)
            noise[count].append(np.sqrt(ss / delta.size))
            noise_ss[count] += ss
            valid_bins[count] += delta.size
    result = dict(n_materials=len(ids), units="Dimensionless concentration",
                  description="RMSE of the 580-cell real forward model evaluated at the generating coefficients. "
                              "Each material contributes one value per condition. The paired observation noise "
                              "uses differences between the 290-bin Gaussian and clean measurements.",
                  cases=[dict(domain=d, currents=c, **describe(v)) for (d, c), v in losses.items()],
                  paired_observation_noise=[dict(currents=c, pooled_rmse=float(np.sqrt(noise_ss[c]/valid_bins[c])),
                                                 **describe(noise[c])) for c in COUNTS])
    write_json(RESULTS / "forward_validation.json", result)


if __name__ == "__main__":
    run()
