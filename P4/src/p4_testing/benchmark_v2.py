"""Strict holdout and benchmark-gate helpers for the EMSE route."""

from __future__ import annotations

from typing import Any, Mapping

from .benchmark import FrozenManifest


def validate_holdout(split: Mapping[str, Any], manifest: FrozenManifest) -> bool:
    held_out = split.get("held_out_project")
    training = tuple(split.get("training_projects", ()))
    projects = {case.project for case in manifest.cases}
    if not isinstance(held_out, str) or held_out not in projects:
        return False
    if held_out in training or set(training) != projects - {held_out}:
        return False
    return True
