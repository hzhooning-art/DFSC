"""Auditable intake of upstream differentiable-computing defects.

Only an upstream report plus replay on both a buggy and a fixed revision may
count as a completed historical-defect pair. Current-version replay alone is
recorded as partial evidence and cannot satisfy the JSS project gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import warnings

import torch


def replay_torch_xlogy() -> dict:
    x = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    two = torch.tensor(2.0, dtype=torch.float64)
    gradient = torch.autograd.grad(torch.xlogy(x, two), x)[0]
    expected = torch.log(two)
    error = float(torch.max(torch.abs(gradient - expected)))
    return {
        "executed": True,
        "passes_fixed_expectation": error <= 1.0e-12,
        "observed_gradient": float(gradient.item()),
        "expected_gradient": float(expected.item()),
        "maximum_absolute_error": error,
        "device": "cpu",
    }


def replay_torch_noncontiguous_einsum() -> dict:
    if not torch.cuda.is_available():
        return {
            "executed": False,
            "passes_fixed_expectation": None,
            "reason": "CUDA unavailable; upstream reproduction is CUDA-specific.",
        }
    torch.manual_seed(42)
    device = torch.device("cuda")
    base = torch.rand(1, 2, device=device, dtype=torch.float64)
    expanded = base.expand(2, -1).requires_grad_()
    weight = torch.rand(2, 2, device=device, dtype=torch.float64)
    grad_output = torch.eye(2, device=device, dtype=torch.float64)
    matmul = expanded @ weight
    einsum = torch.einsum("...x,xy->...y", expanded, weight)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")
        reference = torch.autograd.grad(matmul, expanded, grad_output, retain_graph=True)[0]
        candidate = torch.autograd.grad(einsum, expanded, grad_output)[0]
    error = float(torch.max(torch.abs(reference - candidate)).cpu())
    return {
        "executed": True,
        "passes_fixed_expectation": error <= 1.0e-10,
        "maximum_absolute_gradient_disagreement": error,
        "input_is_contiguous": expanded.is_contiguous(),
        "device": str(device),
    }


def run() -> dict:
    cases = [
        {
            "case_id": "pytorch_80770_xlogy_zero_gradient",
            "project": "PyTorch",
            "upstream_issue": "https://github.com/pytorch/pytorch/issues/80770",
            "reported_version": "1.11.0",
            "upstream_state": "closed",
            "defect_class": "silent_wrong_gradient",
            "reported_environment": "CPU float64",
            "current_fixed_replay": replay_torch_xlogy(),
            "buggy_revision_replay": {"executed": False, "reason": "PyTorch 1.11.0 is not installed."},
        },
        {
            "case_id": "pytorch_30303_noncontiguous_einsum_cuda_gradient",
            "project": "PyTorch",
            "upstream_issue": "https://github.com/pytorch/pytorch/issues/30303",
            "reported_version": "1.3.1",
            "upstream_state": "closed",
            "defect_class": "silent_wrong_gradient",
            "reported_environment": "CUDA, non-contiguous expanded input",
            "current_fixed_replay": replay_torch_noncontiguous_einsum(),
            "buggy_revision_replay": {"executed": False, "reason": "PyTorch 1.3.1 is not installed."},
        },
        {
            "case_id": "jax_15400_complex_conjugate_jit_vjp",
            "project": "JAX",
            "upstream_issue": "https://github.com/jax-ml/jax/issues/15400",
            "reported_version": "0.4.1",
            "upstream_state": "closed",
            "defect_class": "backward_exception",
            "reported_environment": "CPU complex128 with jit/value_and_grad",
            "current_fixed_replay": {
                "executed": False,
                "reason": "JAX is not installed in the frozen environment.",
                "dependency_present": importlib.util.find_spec("jax") is not None,
            },
            "buggy_revision_replay": {"executed": False, "reason": "JAX 0.4.1 is not installed."},
        },
        {
            "case_id": "pybnf_538_species_condition_gradient_route",
            "project": "PyBNF",
            "upstream_issue": "https://github.com/lanl/PyBNF/issues/538",
            "reported_version": "upstream issue snapshot 2026-08-03",
            "upstream_state": "closed by pull request 540",
            "defect_class": "missing_gradient_route",
            "reported_environment": "project integration test",
            "current_fixed_replay": {
                "executed": False,
                "reason": "PyBNF and its simulator dependencies are not installed.",
                "dependency_present": importlib.util.find_spec("pybnf") is not None,
            },
            "buggy_revision_replay": {"executed": False, "reason": "Upstream buggy revision is not checked out."},
        },
    ]
    complete_pairs = [
        row for row in cases
        if row["current_fixed_replay"]["executed"]
        and row["current_fixed_replay"].get("passes_fixed_expectation") is True
        and row["buggy_revision_replay"]["executed"]
    ]
    fixed_only = [
        row for row in cases
        if row["current_fixed_replay"]["executed"]
        and row["current_fixed_replay"].get("passes_fixed_expectation") is True
        and not row["buggy_revision_replay"]["executed"]
    ]
    verified_projects = sorted({row["project"] for row in complete_pairs})
    fixed_only_projects = sorted({row["project"] for row in fixed_only})
    return {
        "schema": "P4-Historical-Defect-Intake-v1",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "admission_rule": (
            "A completed historical defect requires an upstream issue or pull request, an executable "
            "failure on the reported buggy revision, and an executable pass on the fixed revision."
        ),
        "cases": cases,
        "summary": {
            "candidate_cases": len(cases),
            "current_fixed_replays_passed": len(fixed_only) + len(complete_pairs),
            "complete_buggy_fixed_pairs": len(complete_pairs),
            "projects_with_complete_pairs": verified_projects,
            "projects_with_fixed_only_replay": fixed_only_projects,
        },
        "jss_readiness_gate": {
            "required_projects_with_complete_pairs": 3,
            "observed_projects_with_complete_pairs": len(verified_projects),
            "passes": len(verified_projects) >= 3,
        },
        "claim_boundary": (
            "Upstream provenance and a current fixed-version pass do not reconstruct the historical failure. "
            "No case is counted until the buggy revision is executed."
        ),
        "record_storage": "Complete case-level intake records included.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout-summary", action="store_true")
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
