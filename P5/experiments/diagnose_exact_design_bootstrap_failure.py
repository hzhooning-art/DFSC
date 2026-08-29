"""Replay the single failed bootstrap calibration from the power probe."""

from __future__ import annotations

import json

import numpy as np
import torch

from probe_memory_rank import DEVICE, DTYPE, fit_rank
from probe_refusal_calibration import RESULTS
from probe_multiwindow_external_calibration import NOISE_STD, response
from probe_sampling_process_stress import split_indices
from probe_cluster_geometry_conditional_null import make_variable_clustered_times
from probe_exact_design_conditional_bootstrap import BOOTSTRAP_REPLICATES, fit_observations


CASE = "oscillation_decay_016"
REPEAT = 3
SEED = 311303


def main() -> None:
    times = make_variable_clustered_times(SEED)
    clean = response(CASE, times)
    rng = np.random.default_rng(SEED + 17)
    observations = clean + NOISE_STD * torch.tensor(
        rng.standard_normal(clean.shape), dtype=DTYPE, device=DEVICE
    )
    train_np, diagnostic_np, windows_np = split_indices(times, SEED + 29)
    outer, fitted_null = fit_observations(
        times, observations, train_np, diagnostic_np, windows_np, SEED
    )
    bootstrap = []
    retry_diagnostics = []
    for bootstrap_index in range(BOOTSTRAP_REPLICATES):
        bootstrap_seed = SEED * 1000 + 100 + bootstrap_index
        bootstrap_rng = np.random.default_rng(bootstrap_seed)
        bootstrap_observations = fitted_null + NOISE_STD * torch.tensor(
            bootstrap_rng.standard_normal(fitted_null.shape), dtype=DTYPE, device=DEVICE
        )
        fitted, _ = fit_observations(
            times,
            bootstrap_observations,
            train_np,
            diagnostic_np,
            windows_np,
            bootstrap_seed,
        )
        fitted["bootstrap_index"] = bootstrap_index
        fitted["seed"] = bootstrap_seed
        bootstrap.append(fitted)
        if not fitted["fit_quality_pass"]:
            train_idx = torch.tensor(train_np, dtype=torch.long, device=DEVICE)
            diagnostic_idx = torch.tensor(diagnostic_np, dtype=torch.long, device=DEVICE)
            retries = [
                fit_rank(
                    times,
                    bootstrap_observations,
                    train_idx,
                    diagnostic_idx,
                    rank=1,
                    seed=bootstrap_seed * 100 + start,
                    adam_steps=165,
                    lbfgs_steps=48,
                )
                for start in range(2, 6)
            ]
            retry_diagnostics.append({
                "bootstrap_index": bootstrap_index,
                "seed": bootstrap_seed,
                "additional_starts": [
                    {
                        "start": start,
                        "bic": candidate.bic,
                        "validation_rmse": candidate.val_rmse,
                        "jacobian_condition": candidate.jacobian_condition,
                    }
                    for start, candidate in zip(range(2, 6), retries)
                ],
            })
    payload = {
        "experiment": "diagnose_exact_design_bootstrap_failure",
        "source_case": CASE,
        "source_repeat": REPEAT,
        "source_seed": SEED,
        "outer_fit": outer,
        "bootstrap_records": bootstrap,
        "invalid_records": [record for record in bootstrap if not record["fit_quality_pass"]],
        "retry_diagnostics": retry_diagnostics,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "exact_design_bootstrap_failure_diagnostic.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "outer_fit": outer,
        "invalid_records": payload["invalid_records"],
        "retry_diagnostics": retry_diagnostics,
    }, indent=2))


if __name__ == "__main__":
    main()
