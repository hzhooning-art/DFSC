"""Stage 7 controlled property benchmark across NumPy, SciPy, and PyTorch."""

from __future__ import annotations

import argparse
import json
import platform
from collections import defaultdict

import numpy as np
import scipy
import torch
from scipy import linalg, special


SEEDS = tuple(range(12))
FAULTS = (
    "parameter_scaling",
    "output_quantization",
    "late_batch_corruption",
    "batch_crosstalk",
    "order_sensitive_batch",
    "nondeterministic_replay",
    "silent_dtype_downgrade",
)
VALUE_TOLERANCE = 1.0e-6
PROPERTY_TOLERANCE = 1.0e-10


class NumPyLinearSolve:
    project = "numpy/numpy"
    name = "numpy_linalg_solve"

    def __call__(self, inputs: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        return np.linalg.solve(parameters.reshape(2, 2), inputs.T).T

    @staticmethod
    def reference(inputs: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        a, b, c, d = parameters
        inverse = np.array([[d, -b], [-c, a]]) / (a * d - b * c)
        return inputs @ inverse.T


class SciPyMatrixExponential:
    project = "scipy/scipy"
    name = "scipy_linalg_expm_action"

    def __call__(self, inputs: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        matrix = parameters.reshape(2, 2)
        return np.stack([linalg.expm(row[0] * matrix) @ row[1:] for row in inputs])

    @staticmethod
    def reference(inputs: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        time = torch.as_tensor(inputs[:, :1], dtype=torch.float64)
        vector = torch.as_tensor(inputs[:, 1:], dtype=torch.float64)
        matrix = torch.as_tensor(parameters.reshape(2, 2), dtype=torch.float64)
        exponential = torch.matrix_exp(time[:, :, None] * matrix)
        return torch.bmm(exponential, vector[:, :, None]).squeeze(-1).numpy()


class PyTorchLogGamma:
    project = "pytorch/pytorch"
    name = "torch_lgamma"

    def __call__(self, inputs: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        dtype = torch.float32 if inputs.dtype == np.float32 else torch.float64
        values = torch.as_tensor(inputs, dtype=dtype)
        shift = torch.as_tensor(parameters, dtype=dtype)
        return torch.lgamma(values + shift[0]).numpy()

    @staticmethod
    def reference(inputs: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        return special.gammaln(inputs + parameters[0])


SUBJECTS = (NumPyLinearSolve, SciPyMatrixExponential, PyTorchLogGamma)


class FaultWrapper:
    def __init__(self, backend, fault: str | None):
        self.backend = backend
        self.fault = fault
        self.calls = 0
        self.observed_compute_dtype = None

    def __call__(self, inputs: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        self.calls += 1
        local_parameters = parameters * 0.90 if self.fault == "parameter_scaling" else parameters
        if self.fault == "silent_dtype_downgrade":
            local_inputs = inputs.astype(np.float32)
            local_parameters = local_parameters.astype(np.float32)
            self.observed_compute_dtype = "float32"
        else:
            local_inputs = inputs
            self.observed_compute_dtype = str(inputs.dtype)
        output = np.asarray(self.backend(local_inputs, local_parameters), dtype=float)
        if self.fault == "output_quantization":
            output = np.round(output * 100.0) / 100.0
        elif self.fault == "late_batch_corruption":
            output = output.copy()
            output[-1] += 2.0e-2
        elif self.fault == "batch_crosstalk":
            output = output + 2.0e-7 * output.mean(axis=0, keepdims=True)
        elif self.fault == "order_sensitive_batch":
            shape = (len(output),) + (1,) * (output.ndim - 1)
            output = output + 2.0e-7 * np.arange(len(output)).reshape(shape)
        elif self.fault == "nondeterministic_replay":
            output = output + (2.0e-7 if self.calls % 2 else -2.0e-7)
        return output


def make_subject(subject_type, seed: int):
    rng = np.random.default_rng(83000 + 101 * seed + 1009 * SUBJECTS.index(subject_type))
    if subject_type is NumPyLinearSolve:
        inputs = rng.normal(size=(12, 2))
        parameters = np.array([2.0, 0.20, 0.10, 1.5])
    elif subject_type is SciPyMatrixExponential:
        inputs = np.column_stack((rng.uniform(0.05, 1.5, 12), rng.normal(size=(12, 2))))
        parameters = np.array([-0.75, 0.20, -0.10, -0.45])
    else:
        inputs = rng.uniform(0.25, 4.0, size=(12, 1))
        parameters = np.array([0.8])
    backend = subject_type()
    inputs = inputs.astype(np.float64)
    return backend, inputs, parameters, subject_type.reference(inputs, parameters)


def audit(subject_type, seed: int, fault: str | None) -> dict:
    backend, inputs, parameters, reference = make_subject(subject_type, seed)
    wrapped = FaultWrapper(backend, fault)
    values = wrapped(inputs, parameters)
    point_error = float(np.abs(values.reshape(-1)[0] - reference.reshape(-1)[0]))
    full_error = float(np.max(np.abs(values - reference)))
    batched = wrapped(inputs, parameters)
    separate = np.concatenate([wrapped(inputs[index:index + 1], parameters) for index in range(len(inputs))])
    batch_error = float(np.max(np.abs(batched - separate)))
    permutation = np.arange(len(inputs) - 1, -1, -1)
    restored = wrapped(inputs[permutation], parameters)[permutation]
    permutation_error = float(np.max(np.abs(batched - restored)))
    replay_a = wrapped(inputs, parameters)
    replay_b = wrapped(inputs, parameters)
    repeatability_error = float(np.max(np.abs(replay_a - replay_b)))
    checks = {
        "point_value": point_error <= VALUE_TOLERANCE,
        "full_batch_value": full_error <= VALUE_TOLERANCE,
        "batch_independence": batch_error <= PROPERTY_TOLERANCE,
        "permutation_equivariance": permutation_error <= PROPERTY_TOLERANCE,
        "repeatability": repeatability_error <= PROPERTY_TOLERANCE,
        "dtype_conformance": wrapped.observed_compute_dtype == "float64",
        "finite": bool(np.isfinite(values).all()),
    }
    return {
        "project": subject_type.project,
        "subject": subject_type.name,
        "seed": seed,
        "fault": fault or "clean",
        "checks": checks,
        "metrics": {
            "point_value_error": point_error,
            "full_batch_value_error": full_error,
            "batch_independence_error": batch_error,
            "permutation_equivariance_error": permutation_error,
            "repeatability_error": repeatability_error,
        },
    }


STRATEGIES = {
    "single_point_unit_test": ("point_value", "finite"),
    "full_batch_value_test": ("point_value", "full_batch_value", "finite"),
    "numerical_property_suite": (
        "point_value", "full_batch_value", "batch_independence",
        "permutation_equivariance", "repeatability", "finite",
    ),
    "execution_evidence_suite": (
        "point_value", "full_batch_value", "batch_independence",
        "permutation_equivariance", "repeatability", "dtype_conformance", "finite",
    ),
}


def accepted(record: dict, strategy: str) -> bool:
    return all(record["checks"][key] for key in STRATEGIES[strategy])


def run() -> dict:
    records = [
        audit(subject, seed, fault)
        for subject in SUBJECTS for seed in SEEDS for fault in (None, *FAULTS)
    ]
    clean = [row for row in records if row["fault"] == "clean"]
    injected = [row for row in records if row["fault"] != "clean"]
    summary = {}
    for strategy in STRATEGIES:
        detections = [not accepted(row, strategy) for row in injected]
        clusters = defaultdict(list)
        for row, detected in zip(injected, detections):
            clusters[(row["project"], row["subject"], row["fault"])].append(detected)
        summary[strategy] = {
            "injected_trials": len(injected),
            "detected_injected_trials": int(sum(detections)),
            "instance_detection_rate": float(np.mean(detections)),
            "project_subject_fault_clusters": len(clusters),
            "fully_detected_clusters": int(sum(all(values) for values in clusters.values())),
            "clean_trials": len(clean),
            "false_rejections": int(sum(not accepted(row, strategy) for row in clean)),
        }
    return {
        "schema": "P4-Cross-Project-External-Subjects-v1",
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__,
        },
        "design": {
            "independent_sut_projects": sorted({subject.project for subject in SUBJECTS}),
            "subjects": [subject.name for subject in SUBJECTS],
            "fault_families": list(FAULTS),
            "seeds_per_subject_fault": len(SEEDS),
            "primary_unit": "project-subject-fault cluster",
        },
        "summary": summary,
        "records": records,
        "claim_boundary": (
            "This controlled benchmark covers one interface from each of NumPy, SciPy, and PyTorch. "
            "Injected wrappers are not historical field defects, repeated seeds are not independent "
            "projects, and the non-PyTorch subjects do not evaluate automatic differentiation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout-summary", action="store_true")
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
