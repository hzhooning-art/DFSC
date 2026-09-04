"""External-subject pilot for the JSS-oriented P4 empirical study.

The systems under test are three differentiable PyTorch component interfaces;
SciPy supplies independent value references.  This is deliberately labelled a
single-vendor pilot, not a completed cross-project benchmark.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy import linalg, special


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

FAULTS = (
    "parameter_scaling",
    "output_quantization",
    "late_batch_corruption",
    "detached_gradient",
    "batch_crosstalk",
    "order_sensitive_batch",
    "nondeterministic_replay",
    "silent_dtype_downgrade",
)
SEEDS = tuple(range(10))
VALUE_TOLERANCE = 1.0e-5
GRADIENT_TOLERANCE = 5.0e-4
PROPERTY_TOLERANCE = 1.0e-10


class MatrixExponentialAction:
    name = "torch_matrix_exp_action"

    def __call__(self, inputs: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        matrix = parameters.reshape(2, 2)
        exponent = torch.matrix_exp(inputs[:, :1, None] * matrix)
        return torch.bmm(exponent, inputs[:, 1:, None]).squeeze(-1)

    @staticmethod
    def reference(inputs: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        matrix = parameters.reshape(2, 2)
        return np.stack([linalg.expm(row[0] * matrix) @ row[1:] for row in inputs])


class LinearSolveAction:
    name = "torch_linalg_solve"

    def __call__(self, inputs: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        matrix = parameters.reshape(2, 2)
        return torch.linalg.solve(matrix, inputs.T).T

    @staticmethod
    def reference(inputs: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        return np.linalg.solve(parameters.reshape(2, 2), inputs.T).T


class LogGammaShift:
    name = "torch_lgamma_shift"

    def __call__(self, inputs: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        return torch.lgamma(inputs + parameters[0])

    @staticmethod
    def reference(inputs: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        return special.gammaln(inputs + parameters[0])


SUBJECTS = (MatrixExponentialAction, LinearSolveAction, LogGammaShift)


class FaultWrapper:
    def __init__(self, backend, fault: str | None = None):
        self.backend = backend
        self.fault = fault
        self.calls = 0
        self.observed_compute_dtype = None

    def __call__(self, inputs: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        local_parameters = parameters * 0.90 if self.fault == "parameter_scaling" else parameters
        if self.fault == "silent_dtype_downgrade":
            self.observed_compute_dtype = torch.float32
            output = self.backend(inputs.float(), local_parameters.float()).to(inputs.dtype)
        else:
            self.observed_compute_dtype = inputs.dtype
            output = self.backend(inputs, local_parameters)

        if self.fault == "output_quantization":
            output = torch.round(output * 100.0) / 100.0
        elif self.fault == "late_batch_corruption":
            mask = torch.zeros_like(output)
            mask[-1] = 2.0e-2
            output = output + mask
        elif self.fault == "detached_gradient":
            output = output.detach() + 0.0 * parameters.sum()
        elif self.fault == "batch_crosstalk":
            output = output + 2.0e-7 * output.mean(dim=0, keepdim=True)
        elif self.fault == "order_sensitive_batch":
            shape = (len(output),) + (1,) * (output.ndim - 1)
            index = torch.arange(len(output), dtype=output.dtype, device=output.device).reshape(shape)
            output = output + 2.0e-7 * index
        elif self.fault == "nondeterministic_replay":
            output = output + (2.0e-7 if self.calls % 2 else -2.0e-7)
        return output


def make_subject(subject_type, seed: int) -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(71000 + 101 * seed + 1009 * SUBJECTS.index(subject_type))
    if subject_type is MatrixExponentialAction:
        times = rng.uniform(0.05, 1.5, size=(12, 1))
        vectors = rng.normal(size=(12, 2))
        inputs = np.concatenate((times, vectors), axis=1)
        parameters = np.array([-0.75, 0.20, -0.10, -0.45])
    elif subject_type is LinearSolveAction:
        inputs = rng.normal(size=(12, 2))
        parameters = np.array([2.0, 0.20, 0.10, 1.5])
    else:
        inputs = rng.uniform(0.25, 4.0, size=(12, 1))
        parameters = np.array([0.8])
    reference = subject_type.reference(inputs, parameters)
    return (
        subject_type(),
        torch.tensor(inputs, dtype=torch.float64),
        torch.tensor(parameters, dtype=torch.float64),
        torch.tensor(reference, dtype=torch.float64),
    )


def audit(subject_type, seed: int, fault: str | None) -> dict:
    backend, inputs, parameters, reference = make_subject(subject_type, seed)
    wrapped = FaultWrapper(backend, fault)

    values = wrapped(inputs, parameters)
    point_error = float(torch.abs(values.reshape(-1)[0] - reference.reshape(-1)[0]))
    full_error = float(torch.max(torch.abs(values - reference)))

    differentiable_parameters = parameters.clone().requires_grad_(True)
    gradient_values = wrapped(inputs, differentiable_parameters)
    (gradient,) = torch.autograd.grad(gradient_values.sum(), differentiable_parameters)
    direction = torch.linspace(0.4, 0.9, len(parameters), dtype=parameters.dtype)
    epsilon = 1.0e-5
    plus = wrapped(inputs, parameters + epsilon * direction).sum()
    minus = wrapped(inputs, parameters - epsilon * direction).sum()
    finite_difference = (plus - minus) / (2.0 * epsilon)
    automatic = torch.dot(gradient, direction)
    gradient_error = float(torch.abs(finite_difference - automatic) / (torch.abs(finite_difference) + 1.0e-12))

    batched = wrapped(inputs, parameters)
    separate = torch.cat([wrapped(inputs[index : index + 1], parameters) for index in range(len(inputs))])
    batch_independence_error = float(torch.max(torch.abs(batched - separate)))

    permutation = torch.arange(len(inputs) - 1, -1, -1)
    permuted = wrapped(inputs[permutation], parameters)
    restored = permuted[permutation]
    permutation_error = float(torch.max(torch.abs(batched - restored)))

    replay_a = wrapped(inputs, parameters)
    replay_b = wrapped(inputs, parameters)
    repeatability_error = float(torch.max(torch.abs(replay_a - replay_b)))
    checks = {
        "point_value": point_error <= VALUE_TOLERANCE,
        "full_batch_value": full_error <= VALUE_TOLERANCE,
        "directional_gradient": gradient_error <= GRADIENT_TOLERANCE,
        "batch_independence": batch_independence_error <= PROPERTY_TOLERANCE,
        "permutation_equivariance": permutation_error <= PROPERTY_TOLERANCE,
        "repeatability": repeatability_error <= PROPERTY_TOLERANCE,
        "dtype_conformance": wrapped.observed_compute_dtype == torch.float64,
        "finite": bool(torch.isfinite(values).all() and torch.isfinite(gradient).all()),
    }
    return {
        "subject": backend.name,
        "seed": seed,
        "fault": fault or "clean",
        "checks": checks,
        "metrics": {
            "point_value_error": point_error,
            "full_batch_value_error": full_error,
            "directional_gradient_relative_error": gradient_error,
            "batch_independence_error": batch_independence_error,
            "permutation_equivariance_error": permutation_error,
            "repeatability_error": repeatability_error,
        },
    }


STRATEGIES = {
    "single_point_unit_test": ("point_value", "finite"),
    "full_batch_value_test": ("point_value", "full_batch_value", "finite"),
    "value_gradient_test": ("point_value", "full_batch_value", "directional_gradient", "finite"),
    "numerical_property_suite": (
        "point_value",
        "full_batch_value",
        "directional_gradient",
        "batch_independence",
        "permutation_equivariance",
        "repeatability",
        "finite",
    ),
    "execution_evidence_suite": (
        "point_value",
        "full_batch_value",
        "directional_gradient",
        "batch_independence",
        "permutation_equivariance",
        "repeatability",
        "finite",
        "dtype_conformance",
    ),
}


def accepted(record: dict, strategy: str) -> bool:
    return all(record["checks"][name] for name in STRATEGIES[strategy])


def summarize(records: list[dict]) -> dict:
    clean = [row for row in records if row["fault"] == "clean"]
    injected = [row for row in records if row["fault"] != "clean"]
    output = {}
    for strategy in STRATEGIES:
        detections = [not accepted(row, strategy) for row in injected]
        false_rejections = [not accepted(row, strategy) for row in clean]
        combinations = defaultdict(list)
        for row, detected in zip(injected, detections):
            combinations[(row["subject"], row["fault"])].append(detected)
        output[strategy] = {
            "declared_checks": list(STRATEGIES[strategy]),
            "injected_trials": len(injected),
            "detected_injected_trials": int(sum(detections)),
            "instance_detection_rate": float(np.mean(detections)),
            "subject_fault_combinations": len(combinations),
            "fully_detected_subject_fault_combinations": int(sum(all(values) for values in combinations.values())),
            "clean_trials": len(clean),
            "false_rejections": int(sum(false_rejections)),
            "false_rejection_rate": float(np.mean(false_rejections)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout-summary", action="store_true")
    args = parser.parse_args()
    records = [
        audit(subject, seed, fault)
        for subject in SUBJECTS
        for seed in SEEDS
        for fault in (None, *FAULTS)
    ]
    aggregate = {
        "schema": "P4-External-Subject-Pilot-v1",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "scipy": scipy.__version__,
            "device": "cpu",
        },
        "design": {
            "sut_vendor_codebases": ["PyTorch"],
            "independent_reference_codebases": ["SciPy"],
            "subjects": [subject.name for subject in SUBJECTS],
            "faults": list(FAULTS),
            "seeds_per_subject_fault": len(SEEDS),
            "primary_unit": "subject-fault combination",
        },
        "summary": summarize(records),
        "claim_boundary": (
            "This pilot establishes executable adapters and independent value references for three "
            "external component interfaces from one vendor codebase. It is not a multi-project or "
            "field-defect study, and injected trials within one subject-fault combination are not "
            "independent fault families."
        ),
    }
    if args.stdout_summary:
        aggregate["record_storage"] = "Aggregate-only artifact; the script deterministically regenerates trial records."
        print(json.dumps(aggregate, indent=2))
        return
    payload = {**aggregate, "records": records, "record_storage": "Complete individual records included."}
    output = RESULTS / "p4_external_subject_pilot.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": aggregate["summary"]}, indent=2))


if __name__ == "__main__":
    main()
