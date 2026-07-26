"""Utilities for provenance-aware experimental single-particle trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SPTTrajectoryDataset:
    """A collection of experimental 2D trajectories and its provenance."""

    trajectories: tuple[np.ndarray, ...]
    source: str
    citation: str
    condition: str
    coordinate_unit: str = "source coordinate unit"
    time_unit: str = "frame"

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories)

    @property
    def num_localizations(self) -> int:
        return sum(track.shape[0] for track in self.trajectories)


@dataclass(frozen=True)
class SPTObservables:
    """Ensemble MSD and directional intermediate scattering estimates."""

    lags: np.ndarray
    msd: np.ndarray
    scattering: np.ndarray
    sample_counts: np.ndarray
    wave_number: float


def load_anomdiffdb_mat(
    path: str | Path,
    *,
    variable: str = "tr",
    condition: str = "750 nm H-actin mesh",
) -> SPTTrajectoryDataset:
    """Load the public AnomDiffDB ``x, y, trajectory_id`` MATLAB format."""

    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("SciPy is required for MATLAB benchmark files; install dfsc[benchmark]") from exc

    payload = loadmat(Path(path))
    if variable not in payload:
        raise ValueError(f"MATLAB file does not contain variable {variable!r}")
    rows = np.asarray(payload[variable], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError("AnomDiffDB rows must have columns x, y, trajectory_id")
    if not np.isfinite(rows).all():
        raise ValueError("trajectory rows must be finite")
    ids = rows[:, 2].astype(np.int64)
    trajectories = tuple(np.ascontiguousarray(rows[ids == track_id, :2]) for track_id in np.unique(ids))
    if not trajectories or min(track.shape[0] for track in trajectories) < 2:
        raise ValueError("dataset must contain nonempty trajectories with at least two positions")
    return SPTTrajectoryDataset(
        trajectories=trajectories,
        source="https://github.com/AnomDiffDB/DB",
        citation="Granik et al., Biophysical Journal 117, 185-192 (2019), doi:10.1016/j.bpj.2019.06.015",
        condition=condition,
    )


def split_trajectories(
    trajectories: Sequence[np.ndarray],
    *,
    train_fraction: float = 0.7,
    seed: int = 0,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Split at trajectory level so displacement samples cannot leak."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")
    if len(trajectories) < 2:
        raise ValueError("at least two trajectories are required")
    permutation = np.random.default_rng(seed).permutation(len(trajectories))
    cut = min(max(1, int(train_fraction * len(trajectories))), len(trajectories) - 1)
    train_ids, test_ids = permutation[:cut], permutation[cut:]
    return (
        tuple(trajectories[index] for index in train_ids),
        tuple(trajectories[index] for index in test_ids),
    )


def estimate_wave_number(trajectories: Sequence[np.ndarray], *, phase_at_median_step: float = 0.5) -> float:
    """Choose a reproducible wave number from the pooled one-frame step scale."""

    steps = np.concatenate(
        [np.linalg.norm(np.diff(track, axis=0), axis=1) for track in trajectories if track.shape[0] > 1]
    )
    median_step = float(np.median(steps))
    if not np.isfinite(median_step) or median_step <= 0.0:
        raise ValueError("median displacement must be positive")
    return phase_at_median_step / median_step


def empirical_spt_observables(
    trajectories: Sequence[np.ndarray],
    lags: Sequence[int] | np.ndarray,
    *,
    wave_number: float,
) -> SPTObservables:
    """Compute ensemble MSD and an isotropic directional scattering estimate.

    The scattering estimate averages ``cos(q dx)`` and ``cos(q dy)``. This
    avoids assuming calibrated physical units while retaining a scalar modal
    relaxation curve suitable for model comparison.
    """

    lag_values = np.asarray(lags, dtype=np.int64)
    if lag_values.ndim != 1 or lag_values.size == 0 or np.any(lag_values <= 0):
        raise ValueError("lags must be a nonempty one-dimensional positive sequence")
    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    msd_values: list[float] = []
    scattering_values: list[float] = []
    counts: list[int] = []
    for lag in lag_values:
        squared_sum = 0.0
        scattering_sum = 0.0
        count = 0
        for track in trajectories:
            if track.shape[0] <= lag:
                continue
            displacement = track[lag:] - track[:-lag]
            squared_sum += float(np.sum(displacement * displacement))
            scattering_sum += float(
                np.sum((np.cos(wave_number * displacement[:, 0]) + np.cos(wave_number * displacement[:, 1])) / 2.0)
            )
            count += displacement.shape[0]
        if count == 0:
            raise ValueError(f"no trajectory supports lag {lag}")
        msd_values.append(squared_sum / count)
        scattering_values.append(scattering_sum / count)
        counts.append(count)
    return SPTObservables(
        lags=lag_values,
        msd=np.asarray(msd_values),
        scattering=np.asarray(scattering_values),
        sample_counts=np.asarray(counts),
        wave_number=float(wave_number),
    )
