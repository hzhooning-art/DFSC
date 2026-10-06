"""Minimal cross-version runner for two upstream PyTorch gradient defects.

This file intentionally depends only on the Python standard library and torch
so it can be copied unchanged into legacy environments. JSON is printed to
stdout; the orchestrating study, not this runner, decides whether a buggy/fixed
pair is complete.
"""

from __future__ import print_function

import argparse
import json
import math
import sys
import warnings

import torch


def xlogy_zero_gradient():
    x = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    two = torch.tensor(2.0, dtype=torch.float64)
    gradient = torch.autograd.grad(torch.xlogy(x, two), x)[0]
    observed = float(gradient.item())
    expected = math.log(2.0)
    error = abs(observed - expected)
    return {
        "executed": True,
        "observed_gradient": observed,
        "expected_gradient": expected,
        "maximum_absolute_error": error,
        "matches_fixed_expectation": error <= 1.0e-12,
        "matches_reported_bug": abs(observed) <= 1.0e-15,
        "device": "cpu",
    }


def noncontiguous_einsum_cuda_gradient():
    if not torch.cuda.is_available():
        return {
            "executed": False,
            "reason": "CUDA unavailable; upstream reproduction is CUDA-specific.",
            "matches_fixed_expectation": None,
            "matches_reported_bug": None,
        }
    torch.manual_seed(42)
    device = torch.device("cuda")
    base = torch.rand(1, 2, device=device)
    expanded = base.expand(2, -1).requires_grad_()
    weight = torch.rand(2, 2, device=device)
    grad_output = torch.eye(2, device=device)
    matmul = expanded @ weight
    einsum = torch.einsum("...x,xy->...y", [expanded, weight])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")
        reference = torch.autograd.grad(matmul, expanded, grad_output, retain_graph=True)[0]
        candidate = torch.autograd.grad(einsum, expanded, grad_output)[0]
    error = float(torch.max(torch.abs(reference - candidate)).cpu().item())
    return {
        "executed": True,
        "maximum_absolute_gradient_disagreement": error,
        "matches_fixed_expectation": error <= 1.0e-6,
        "matches_reported_bug": error > 1.0e-4,
        "input_is_contiguous": bool(expanded.is_contiguous()),
        "device": str(device),
    }


CASES = {
    "pytorch_80770": {
        "upstream_issue": "https://github.com/pytorch/pytorch/issues/80770",
        "reported_buggy_version": "1.11.0",
        "runner": xlogy_zero_gradient,
    },
    "pytorch_30303": {
        "upstream_issue": "https://github.com/pytorch/pytorch/issues/30303",
        "reported_buggy_version": "1.3.1",
        "runner": noncontiguous_einsum_cuda_gradient,
    },
}


def run_case(case_id, expected_role):
    definition = CASES[case_id]
    observation = definition["runner"]()
    if not observation["executed"]:
        role_confirmed = False
    elif expected_role == "fixed":
        role_confirmed = observation["matches_fixed_expectation"] is True
    else:
        role_confirmed = observation["matches_reported_bug"] is True
    return {
        "schema": "P4-Historical-Pair-Runner-v1",
        "case_id": case_id,
        "expected_role": expected_role,
        "role_confirmed": role_confirmed,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "upstream_issue": definition["upstream_issue"],
        "reported_buggy_version": definition["reported_buggy_version"],
        "observation": observation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--expected-role", choices=("buggy", "fixed"), required=True)
    args = parser.parse_args()
    print(json.dumps(run_case(args.case, args.expected_role), indent=2))


if __name__ == "__main__":
    main()
