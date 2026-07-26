"""Dataset contracts for dfsc benchmark integration.

dfsc does not bundle third-party physical datasets. This module defines a small
manifest format so future public benchmarks can be loaded with explicit
provenance instead of becoming undocumented experiment folders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class BenchmarkSpec:
    """Metadata required for a dfsc benchmark dataset."""

    name: str
    domain: str
    task: str
    source: str
    license: str
    citation: str
    status: str = "external-required"

    def to_dict(self) -> dict[str, str]:
        return self.__dict__.copy()


PUBLIC_BENCHMARK_TARGETS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        name="porous-media-anomalous-diffusion",
        domain="anomalous transport",
        task="forward prediction or inverse diffusivity/order recovery",
        source="public experimental or benchmark data required",
        license="to be recorded before use",
        citation="to be recorded before use",
    ),
    BenchmarkSpec(
        name="single-particle-anomalous-diffusion-trajectories",
        domain="anomalous diffusion trajectory analysis",
        task="trajectory-level order or model-parameter inference",
        source="https://github.com/AnomDiffDB/DB/blob/master/750nm_mesh_size.mat",
        license="not stated on source page; local analysis only pending redistribution permission",
        citation="Granik et al., Biophysical Journal 117, 185-192 (2019), doi:10.1016/j.bpj.2019.06.015",
        status="integrated-local-license-review",
    ),
    BenchmarkSpec(
        name="battery-degradation-or-impedance",
        domain="electrochemical diffusion and degradation",
        task="fractional-order response prediction or inverse parameter recovery",
        source="public battery aging or impedance dataset required",
        license="to be recorded before use",
        citation="to be recorded before use",
    ),
)


def benchmark_targets() -> list[dict[str, str]]:
    """Return planned and locally integrated public benchmark targets."""

    return [spec.to_dict() for spec in PUBLIC_BENCHMARK_TARGETS]


def validate_dataset_manifest(manifest: dict[str, object]) -> tuple[bool, tuple[str, ...]]:
    """Validate the minimum metadata needed for a dfsc benchmark."""

    required = {
        "name",
        "domain",
        "task",
        "source",
        "license",
        "citation",
        "splits",
        "tensors",
    }
    missing = sorted(key for key in required if key not in manifest)
    return not missing, tuple(missing)


def load_tensor_dataset(dataset_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load a manifest-backed tensor dataset.

    The directory must contain `manifest.json` and tensor files listed under the
    manifest `tensors` object. Tensor files are loaded with `torch.load`, so this
    helper is intended for trusted local benchmark assets.
    """

    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ok, missing = validate_dataset_manifest(manifest)
    if not ok:
        raise ValueError(f"dataset manifest is missing required fields: {', '.join(missing)}")

    tensor_map = manifest["tensors"]
    if not isinstance(tensor_map, dict):
        raise ValueError("manifest field 'tensors' must be an object")
    tensors = {}
    for name, rel_path in tensor_map.items():
        if not isinstance(rel_path, str):
            raise ValueError("tensor paths must be strings")
        tensors[name] = torch.load(root / rel_path, map_location="cpu")
    return tensors
