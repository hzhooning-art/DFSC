"""Family-level, project-clustered metrics for the frozen P4 benchmark.

The module makes the two independence boundaries executable: variants collapse
to one defect family, and uncertainty resamples whole projects.  LOPO training
objects carry explicit provenance so held-out labels, observations, and
case-specific thresholds cannot silently enter a fitted strategy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .benchmark import BenchmarkCase, FrozenManifest, leave_one_project_out


class MetricsError(ValueError):
    """Raised when an evaluation violates its statistical or transfer contract."""


@dataclass(frozen=True)
class LopoSplit:
    holdout: str
    training: tuple[str, ...]


@dataclass(frozen=True)
class ProjectSummary:
    project: str
    families: int
    detected_families: int
    detection_rate: float
    fixed_false_rejections: int
    non_evaluable_families: int


@dataclass(frozen=True)
class BenchmarkSummary:
    families: int
    projects: int
    detected_families: int
    detection_rate: float
    fixed_false_rejection_rate: float
    non_evaluable_families: int
    per_project: tuple[ProjectSummary, ...]


@dataclass(frozen=True)
class ClusterBootstrap:
    cluster: str
    seed: int
    replicates: int
    clusters: tuple[str, ...]
    estimate: float
    lower: float
    upper: float
    resample_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TrainedStrategy:
    """A fitted LOPO strategy with auditable sources for learned artifacts."""

    training_projects: tuple[str, ...]
    training_case_ids: tuple[str, ...]
    training_defect_families: tuple[str, ...]
    case_thresholds: Mapping[str, float]
    buggy_observations: Mapping[str, Any]
    evaluator: Callable[[BenchmarkCase], Mapping[str, Any]]

    def __post_init__(self) -> None:
        if not callable(self.evaluator):
            raise MetricsError("trained strategy evaluator must be callable")
        object.__setattr__(self, "case_thresholds", MappingProxyType(dict(self.case_thresholds)))
        object.__setattr__(self, "buggy_observations", MappingProxyType(dict(self.buggy_observations)))


@dataclass(frozen=True)
class ProjectEvaluation:
    held_out_project: str
    training_projects: tuple[str, ...]
    held_out_families: int
    detected_families: int
    detection_rate: float
    fixed_false_rejections: int
    non_evaluable_families: int


@dataclass(frozen=True)
class _FamilyRow:
    family: str
    project: str
    detected: bool
    fixed_rejected: bool
    non_evaluable: bool


def _required_text(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value
    raise MetricsError(f"row requires a non-empty {' or '.join(names)}")


def _required_bool(row: Mapping[str, Any], name: str, *, default: bool | None = None) -> bool:
    if name not in row and default is not None:
        return default
    value = row.get(name)
    if not isinstance(value, bool):
        raise MetricsError(f"{name} must be boolean")
    return value


def _collapse_families(rows: Sequence[Mapping[str, Any]]) -> tuple[_FamilyRow, ...]:
    if not rows:
        raise MetricsError("at least one result row is required")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise MetricsError("each result row must be a mapping")
        family = _required_text(row, "family", "defect_family")
        grouped.setdefault(family, []).append(row)

    collapsed: list[_FamilyRow] = []
    for family in sorted(grouped):
        variants = grouped[family]
        projects = {_required_text(row, "project") for row in variants}
        if len(projects) != 1:
            raise MetricsError(f"defect family {family!r} appears in multiple projects")
        detected = any(_required_bool(row, "detected") for row in variants)
        fixed_rejected = any(_required_bool(row, "fixed_rejected", default=False) for row in variants)
        statuses = {str(row.get("status", "")) for row in variants}
        non_evaluable = bool(statuses) and statuses == {"non_evaluable"}
        collapsed.append(
            _FamilyRow(family, next(iter(projects)), detected, fixed_rejected, non_evaluable)
        )
    return tuple(collapsed)


def family_detection_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return detections per unique defect family, never per variant."""

    families = _collapse_families(rows)
    return sum(item.detected for item in families) / len(families)


def lopo_splits(projects: Sequence[str]) -> tuple[LopoSplit, ...]:
    """Return deterministic leave-one-project-out project-name splits."""

    if not projects or any(not isinstance(item, str) or not item.strip() for item in projects):
        raise MetricsError("projects must contain non-empty names")
    unique = tuple(sorted(set(projects)))
    if len(unique) < 2:
        raise MetricsError("LOPO evaluation requires at least two projects")
    return tuple(LopoSplit(project, tuple(item for item in unique if item != project)) for project in unique)


def _project_summary(project: str, rows: Sequence[_FamilyRow]) -> ProjectSummary:
    selected = tuple(item for item in rows if item.project == project)
    count = len(selected)
    detected = sum(item.detected for item in selected)
    return ProjectSummary(
        project=project,
        families=count,
        detected_families=detected,
        detection_rate=detected / count,
        fixed_false_rejections=sum(item.fixed_rejected for item in selected),
        non_evaluable_families=sum(item.non_evaluable for item in selected),
    )


def summarize_results(rows: Sequence[Mapping[str, Any]]) -> BenchmarkSummary:
    """Summarize result rows at the independent defect-family unit."""

    families = _collapse_families(rows)
    projects = tuple(sorted({item.project for item in families}))
    detected = sum(item.detected for item in families)
    rejected = sum(item.fixed_rejected for item in families)
    return BenchmarkSummary(
        families=len(families),
        projects=len(projects),
        detected_families=detected,
        detection_rate=detected / len(families),
        fixed_false_rejection_rate=rejected / len(families),
        non_evaluable_families=sum(item.non_evaluable for item in families),
        per_project=tuple(_project_summary(project, families) for project in projects),
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    cluster: str = "project",
    seed: int = 20260916,
    replicates: int = 10_000,
) -> ClusterBootstrap:
    """Bootstrap detection by sampling entire projects with replacement."""

    if cluster != "project":
        raise MetricsError("only project-clustered bootstrap is supported")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise MetricsError("replicates must be a positive integer")
    families = _collapse_families(rows)
    clusters = tuple(sorted({item.project for item in families}))
    grouped = {name: tuple(item for item in families if item.project == name) for name in clusters}
    generator = random.Random(seed)
    indices: list[tuple[int, ...]] = []
    estimates: list[float] = []
    for _ in range(replicates):
        draw = tuple(generator.randrange(len(clusters)) for _ in clusters)
        indices.append(draw)
        sampled = tuple(item for index in draw for item in grouped[clusters[index]])
        estimates.append(sum(item.detected for item in sampled) / len(sampled))
    estimate = sum(item.detected for item in families) / len(families)
    return ClusterBootstrap(
        cluster=cluster,
        seed=seed,
        replicates=replicates,
        clusters=clusters,
        estimate=estimate,
        lower=_percentile(estimates, 0.025),
        upper=_percentile(estimates, 0.975),
        resample_indices=tuple(indices),
    )


def _audit_training(strategy: TrainedStrategy, training: tuple[BenchmarkCase, ...], held_out: tuple[BenchmarkCase, ...]) -> None:
    expected_projects = tuple(sorted({case.project for case in training}))
    expected_ids = tuple(sorted(case.case_id for case in training))
    expected_families = tuple(sorted({case.defect_family for case in training}))
    if tuple(sorted(strategy.training_projects)) != expected_projects:
        raise MetricsError("training project provenance does not match the LOPO split")
    if tuple(sorted(strategy.training_case_ids)) != expected_ids:
        raise MetricsError("training case provenance does not match the LOPO split")
    if tuple(sorted(strategy.training_defect_families)) != expected_families:
        raise MetricsError("held-out defect label leaked into training artifacts")

    held_out_ids = {case.case_id for case in held_out}
    held_out_families = {case.defect_family for case in held_out}
    if held_out_ids.intersection(strategy.case_thresholds):
        raise MetricsError("held-out case-specific threshold leaked into training artifacts")
    if held_out_ids.intersection(strategy.buggy_observations):
        raise MetricsError("held-out buggy observation leaked into training artifacts")
    if held_out_families.intersection(strategy.training_defect_families):
        raise MetricsError("held-out defect label leaked into training artifacts")
    unknown_thresholds = set(strategy.case_thresholds) - set(expected_ids)
    unknown_observations = set(strategy.buggy_observations) - set(expected_ids)
    if unknown_thresholds or unknown_observations:
        raise MetricsError("training artifacts contain sources outside the training split")


def evaluate_lopo(
    manifest: FrozenManifest,
    strategy_factory: Callable[[tuple[BenchmarkCase, ...]], TrainedStrategy],
) -> tuple[ProjectEvaluation, ...]:
    """Fit on all other projects and evaluate each held-out project once."""

    if not isinstance(manifest, FrozenManifest):
        raise MetricsError("LOPO evaluation requires a frozen manifest")
    evaluations: list[ProjectEvaluation] = []
    for split in leave_one_project_out(manifest):
        strategy = strategy_factory(split.training)
        if not isinstance(strategy, TrainedStrategy):
            raise MetricsError("strategy_factory must return TrainedStrategy")
        _audit_training(strategy, split.training, split.held_out)
        rows: list[Mapping[str, Any]] = []
        for case in split.held_out:
            result = strategy.evaluator(case)
            if not isinstance(result, Mapping):
                raise MetricsError("LOPO evaluator must return a mapping")
            rows.append({"family": case.defect_family, "project": case.project, **result})
        summary = summarize_results(rows)
        evaluations.append(
            ProjectEvaluation(
                held_out_project=split.held_out_project,
                training_projects=tuple(sorted(strategy.training_projects)),
                held_out_families=summary.families,
                detected_families=summary.detected_families,
                detection_rate=summary.detection_rate,
                fixed_false_rejections=sum(item.fixed_false_rejections for item in summary.per_project),
                non_evaluable_families=summary.non_evaluable_families,
            )
        )
    return tuple(evaluations)
